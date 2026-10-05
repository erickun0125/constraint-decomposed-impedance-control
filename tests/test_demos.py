"""Demonstration files: layout written by DemoWriter and loading by the training dataset (CPU only)."""

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
from con_dec_imp.policy.dataset import RelativeChunkDataset, make_dataset  # noqa: E402
from con_dec_imp_sim.demos import Demonstration, DemoWriter  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "policy" / "single_task.json"
LENGTHS = (7, 9, 6, 8, 5, 10, 7, 6, 9, 8)
ENV_NAME = "ConDecImp-FlyingGripper-Revolute-Collect-v0"


def _demo(rng, n):
    rot6d = matrix_to_rot6d(so3_exp(torch.from_numpy(rng.normal(scale=0.3, size=(n, 3))).float())).numpy()
    position = rng.normal(scale=0.1, size=(n, 3)).astype(np.float32)
    closed = (np.arange(n) >= n // 2).astype(np.float32)[:, None]
    return Demonstration(
        eef_pose=np.concatenate([position, rot6d], axis=-1),
        gripper=1.0 - closed,
        wrist_cam=rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8),
        actions=np.concatenate([position + 0.01, rot6d, closed], axis=-1),
        rewards=np.zeros(n, dtype=np.float32),
    )


@pytest.fixture()
def demo_file(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "demos.hdf5"
    writer = DemoWriter(path, ENV_NAME)
    for n in LENGTHS:
        writer.write(_demo(rng, n))
    assert not path.exists()  # written under a temporary name until closed
    assert writer.close(train_fraction=0.9) == path
    assert not path.with_name(path.name + ".partial").exists()
    return path


def test_layout(demo_file):
    with h5py.File(demo_file, "r") as f:
        assert set(f.keys()) == {"data", "mask"}
        assert dict(f.attrs) == {"action_frame": "absolute"}
        assert json.loads(f["data"].attrs["env_args"]) == {"env_name": ENV_NAME, "env_kwargs": {}, "type": 1}
        assert json.loads(f["data"].attrs["obs_meta"]) == {
            "state_obs_keys": ["eef_pose", "gripper"], "image_obs_keys": ["wrist_cam"],
        }
        assert list(f["data"].keys()) == sorted(f"demo_{i}" for i in range(len(LENGTHS)))
        for i, n in enumerate(LENGTHS):
            demo = f[f"data/demo_{i}"]
            assert set(demo.keys()) == {"obs", "actions", "rewards", "dones"}
            assert set(demo["obs"].keys()) == {"eef_pose", "gripper", "wrist_cam"}
            assert demo.attrs["num_samples"] == n and demo.attrs["success"]
            shapes = {key: (demo[key].shape, demo[key].dtype) for key in
                      ("obs/eef_pose", "obs/gripper", "obs/wrist_cam", "actions", "rewards", "dones")}
            assert shapes == {
                "obs/eef_pose": ((n, 9), np.float32),
                "obs/gripper": ((n, 1), np.float32),
                "obs/wrist_cam": ((n, 8, 8, 3), np.uint8),
                "actions": ((n, 10), np.float32),
                "rewards": ((n,), np.float64),
                "dones": ((n,), np.bool_),
            }
            assert demo["obs/wrist_cam"].compression == "gzip"
            assert demo["dones"][()].tolist() == [False] * (n - 1) + [True]


def test_masks_split_in_order(demo_file):
    with h5py.File(demo_file, "r") as f:
        train = [name.decode() for name in f["mask/train"][()]]
        valid = [name.decode() for name in f["mask/valid"][()]]
    assert train == [f"demo_{i}" for i in range(9)] and valid == ["demo_9"]


def test_from_steps_stacks_records():
    demo = _demo(np.random.default_rng(1), 4)
    steps = [{name: getattr(demo, name)[t] for name in vars(demo)} for t in range(4)]
    stacked = Demonstration.from_steps(steps)
    assert len(stacked) == 4
    for name, value in vars(demo).items():
        np.testing.assert_array_equal(getattr(stacked, name), value)


def test_discard_and_empty_file(tmp_path):
    path = tmp_path / "demos.hdf5"
    writer = DemoWriter(path, ENV_NAME)
    writer.write(_demo(np.random.default_rng(2), 5))
    writer.discard()
    assert not path.exists() and not path.with_name(path.name + ".partial").exists()
    writer = DemoWriter(path, ENV_NAME)
    with pytest.raises(RuntimeError):
        writer.close(train_fraction=0.9)
    writer.discard()


def test_training_dataset_loads_both_splits(demo_file):
    external = json.loads(CONFIG.read_text())
    external["experiment"].pop("validation_every_n_epochs")
    config = config_factory(external["algo_name"])
    with config.unlocked():
        config.update(external)
        config.train.data = [{"path": str(demo_file), "weight": 1.0}]
    obs_utils.initialize_obs_utils_with_config(config)
    horizon = config.train.seq_length
    for split, demos in (("train", range(9)), ("valid", [9])):
        dataset = make_dataset(config, config.all_obs_keys, split)
        assert isinstance(dataset, RelativeChunkDataset)
        assert len(dataset) == sum(LENGTHS[i] for i in demos)
        sample = dataset[0]
        assert sample["actions"].shape == (horizon, 10)
        assert sample["obs"]["eef_pose"].shape == (1, 9)
        assert sample["obs"]["wrist_cam"].shape == (1, 8, 8, 3)
        assert np.abs(sample["actions"]).max() <= 1.0 + 1e-6
