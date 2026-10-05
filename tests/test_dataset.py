"""Relative-chunk training dataset on tiny synthetic demonstration files (CPU only)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")
pytest.importorskip("robomimic")

from robomimic.config import config_factory  # noqa: E402
from robomimic.utils import obs_utils  # noqa: E402

from con_dec_imp.lie import matrix_to_rot6d, so3_exp  # noqa: E402
from con_dec_imp.policy.actions import absolute_to_relative  # noqa: E402
from con_dec_imp.policy.dataset import RelativeChunkDataset, RelativeChunkMetaDataset, make_dataset  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "policy" / "single_task.json"
HORIZON = 4
LENGTHS = {"demo_0": 7, "demo_1": 3, "demo_2": 6}  # demo_1 is shorter than the horizon
SPLITS = {"train": ["demo_0", "demo_1"], "valid": ["demo_2"]}


def _poses(rng, n, drift):
    position = np.cumsum(rng.normal(scale=0.02, size=(n, 3)), axis=0) + drift
    rotation = so3_exp(torch.from_numpy(np.cumsum(rng.normal(scale=0.2, size=(n, 3)), axis=0)).float())
    return np.concatenate([position, matrix_to_rot6d(rotation).numpy()], axis=-1).astype(np.float32)


def _write(path, seed, drift):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "test", "type": 1, "env_kwargs": {}})
        for demo, n in LENGTHS.items():
            g = data.create_group(demo)
            g.attrs["num_samples"] = n
            gripper = (np.arange(n) >= n // 2).astype(np.float32)[:, None]
            g["actions"] = np.concatenate([_poses(rng, n, drift), gripper], axis=-1)
            g["obs/eef_pose"] = _poses(rng, n, drift)
            g["obs/gripper"] = 1.0 - gripper
            g["obs/wrist_cam"] = rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8)
            g["rewards"] = np.zeros(n)
            g["dones"] = np.zeros(n, dtype=bool)
        for split, demos in SPLITS.items():
            f[f"mask/{split}"] = np.array(demos, dtype="S")


@pytest.fixture()
def files(tmp_path):
    paths = [tmp_path / "a.hdf5", tmp_path / "b.hdf5"]
    _write(paths[0], seed=0, drift=0.0)
    _write(paths[1], seed=1, drift=0.5)
    return [str(p) for p in paths]


def _config(paths):
    external = json.loads(CONFIG.read_text())
    external["experiment"].pop("validation_every_n_epochs")
    config = config_factory(external["algo_name"])
    with config.unlocked():
        config.update(external)
        config.train.data = [{"path": p, "weight": 1.0} for p in paths]
        config.train.seq_length = HORIZON
        config.train.normalize_weights_by_ds_size = len(paths) > 1
    obs_utils.initialize_obs_utils_with_config(config)
    return config


def _relative_rows(path, demos):
    """Every padded relative chunk of the given demos, computed directly."""
    rows = {}
    with h5py.File(path, "r") as f:
        for demo in demos:
            actions, anchors = f[f"data/{demo}/actions"][()], f[f"data/{demo}/obs/eef_pose"][()]
            n = len(actions)
            for t in range(n):
                chunk = actions[np.minimum(np.arange(t, t + HORIZON), n - 1)]
                rows[(demo, t)] = absolute_to_relative(torch.from_numpy(chunk), torch.from_numpy(anchors[t])).numpy()
    return rows


def _expected_stats(rows):
    stacked = np.concatenate([r for r in rows.values()])
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    scale = (hi - lo) / 1.999998
    offset = lo + 0.999999 * scale
    scale[3:9], offset[3:9] = 1.0, 0.0
    return scale, offset


def test_samples_are_relative_chunks_anchored_at_t(files):
    config = _config(files[:1])
    dataset = make_dataset(config, config.all_obs_keys, "train")
    assert isinstance(dataset, RelativeChunkDataset)
    rows = _relative_rows(files[0], SPLITS["train"])
    assert len(dataset) == len(rows) == sum(LENGTHS[d] for d in SPLITS["train"])
    stats = dataset.get_action_normalization_stats()["actions"]
    with h5py.File(files[0], "r") as f:
        obs = {(d, k): f[f"data/{d}/obs/{k}"][()] for d in SPLITS["train"] for k in ("wrist_cam", "eef_pose")}
    for index in range(len(dataset)):
        sample = dataset[index]
        demo = dataset._index_to_demo_id[index]
        t = index - dataset._demo_id_to_start_indices[demo]
        actions = sample["actions"] * stats["scale"] + stats["offset"]
        np.testing.assert_allclose(actions, rows[(demo, t)], atol=1e-6)
        assert sample["actions"].shape == (HORIZON, 10)
        # observation at t only (one-step history)
        assert sample["obs"]["wrist_cam"].shape == (1, 8, 8, 3)
        np.testing.assert_array_equal(sample["obs"]["wrist_cam"][0], obs[(demo, "wrist_cam")][t])
        np.testing.assert_array_equal(sample["obs"]["eef_pose"][0], obs[(demo, "eef_pose")][t])


def test_statistics_cover_padded_chunks_with_identity_rotation(files):
    config = _config(files[:1])
    stats = make_dataset(config, config.all_obs_keys, "train").get_action_normalization_stats()["actions"]
    scale, offset = _expected_stats(_relative_rows(files[0], SPLITS["train"]))
    np.testing.assert_allclose(stats["scale"][0], scale, rtol=1e-6)
    np.testing.assert_allclose(stats["offset"][0], offset, atol=1e-6)
    assert np.all(stats["scale"][0, 3:9] == 1.0) and np.all(stats["offset"][0, 3:9] == 0.0)
    assert stats["offset"][0, 9] == pytest.approx(0.5) and stats["scale"][0, 9] == pytest.approx(0.5, rel=1e-5)


def test_normalized_actions_stay_in_range(files):
    config = _config(files[:1])
    dataset = make_dataset(config, config.all_obs_keys, "train")
    actions = np.stack([dataset[i]["actions"] for i in range(len(dataset))])
    assert np.abs(actions[..., [0, 1, 2, 9]]).max() <= 1.0
    assert np.abs(actions[..., 3:9]).max() <= 1.0 + 1e-6


def test_validation_split_uses_its_own_statistics(files):
    config = _config(files[:1])
    valid = make_dataset(config, config.all_obs_keys, "valid")
    assert len(valid) == LENGTHS["demo_2"]
    scale, offset = _expected_stats(_relative_rows(files[0], SPLITS["valid"]))
    np.testing.assert_allclose(valid.get_action_normalization_stats()["actions"]["scale"][0], scale, rtol=1e-6)


def test_multi_task_pools_files_with_shared_statistics(files):
    config = _config(files)
    dataset = make_dataset(config, config.all_obs_keys, "train")
    assert isinstance(dataset, RelativeChunkMetaDataset)
    rows = {**{("a",) + k: v for k, v in _relative_rows(files[0], SPLITS["train"]).items()},
            **{("b",) + k: v for k, v in _relative_rows(files[1], SPLITS["train"]).items()}}
    assert len(dataset) == len(rows)
    scale, offset = _expected_stats(rows)
    stats = dataset.get_action_normalization_stats()["actions"]
    np.testing.assert_allclose(stats["scale"][0], scale, rtol=1e-6)
    np.testing.assert_allclose(stats["offset"][0], offset, atol=1e-6)
    for member in dataset.datasets:
        assert member.get_action_normalization_stats() is dataset.get_action_normalization_stats()
    # equal sampling mass per file
    weights = np.asarray(dataset.ds_weights) * np.array([len(d) for d in dataset.datasets])
    np.testing.assert_allclose(weights, weights[0])
