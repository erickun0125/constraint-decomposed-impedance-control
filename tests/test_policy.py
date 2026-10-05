"""Diffusion Policy wrapper: release export, input handling and ensemble sampling (CPU)."""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("robomimic")
pytest.importorskip("diffusers")

from robomimic.algo.diffusion_policy import DiffusionPolicyUNet  # noqa: E402
from robomimic.config import config_factory  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402

from con_dec_imp.lie import matrix_to_rot6d, rot6d_to_matrix, so3_exp  # noqa: E402
from con_dec_imp.policy.actions import absolute_to_relative  # noqa: E402
from con_dec_imp.policy.diffusion_policy import DiffusionPolicy, release_checkpoint  # noqa: E402

HORIZON, IMAGE, STEPS = 8, 64, 2
SHAPES = {"eef_pose": [9], "gripper": [1], "wrist_cam": [3, IMAGE, IMAGE]}
SCALE = [0.3, 0.2, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
OFFSET = [0.1, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
TRAIN_ONLY = "datasets/demos_train.hdf5"


def _small_config() -> dict:
    """Policy config of the paper with a small image and horizon."""
    config = config_factory("diffusion_policy")
    with config.values_unlocked():
        config.update({
            "experiment": {"name": "small_run"},
            "train": {"data": [{"path": TRAIN_ONLY}], "output_dir": "outputs/small_run"},
            "algo": {"horizon": {"observation_horizon": 1, "action_horizon": 4, "prediction_horizon": HORIZON}},
            "observation": {
                "modalities": {"obs": {"low_dim": ["eef_pose", "gripper"], "rgb": ["wrist_cam"]}},
                "encoder": {"rgb": {
                    "core_class": "VisualCore",
                    "core_kwargs": {
                        "feature_dimension": 8,
                        "backbone_class": "ResNet18Conv",
                        "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
                        "pool_class": "SpatialSoftmax",
                        "pool_kwargs": {"num_kp": 4, "learnable_temperature": False,
                                        "temperature": 1.0, "noise_std": 0.0},
                    },
                    "obs_randomizer_class": "CropRandomizer",
                    "obs_randomizer_kwargs": {"crop_height": IMAGE - 8, "crop_width": IMAGE - 8,
                                              "num_crops": 1, "pos_enc": False},
                }},
            },
        })
    return json.loads(config.dump())


@pytest.fixture(scope="module")
def training_checkpoint() -> dict:
    """A robomimic-style training checkpoint with random (EMA) weights."""
    config_dict = _small_config()
    config = config_factory("diffusion_policy")
    with config.values_unlocked():
        config.update(config_dict)
        config.algo.ema.enabled = False
    with config.unlocked():
        config.algo.optim_params = {}
    ObsUtils.initialize_obs_utils_with_config(config)
    torch.manual_seed(0)
    algo = DiffusionPolicyUNet(
        algo_config=config.algo, obs_config=config.observation, global_config=config,
        obs_key_shapes=SHAPES, ac_dim=10, device=torch.device("cpu"),
    )
    ema = algo.nets.state_dict()
    shape_metadata = {"ac_dim": 10, "all_shapes": SHAPES, "all_obs_keys": list(SHAPES), "use_images": True}
    return {
        "algo_name": "diffusion_policy",
        "config": json.dumps(config_dict),
        "shape_metadata": [shape_metadata, dict(shape_metadata)],
        "action_normalization_stats": {"actions": {"scale": [SCALE], "offset": [OFFSET]}},
        "variable_state": {"epoch": 3, "best_valid_loss": 0.1},
        "env_metadata": {"env_name": "small_env"},
        "model": {
            "nets": {k: torch.zeros_like(v) for k, v in ema.items()},
            "ema": ema,
            "optimizers": {"policy": {}},
            "lr_schedulers": {"policy": None},
        },
    }


@pytest.fixture(scope="module")
def release_file(training_checkpoint, tmp_path_factory):
    path = tmp_path_factory.mktemp("release") / "small.pth"
    torch.save(release_checkpoint(training_checkpoint), path)
    return path


@pytest.fixture(scope="module")
def policy(release_file) -> DiffusionPolicy:
    return DiffusionPolicy.load(release_file, device="cpu")


def _observation(num_envs: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    rotation = so3_exp(torch.randn(num_envs, 3, generator=g))
    return {
        "eef_pose": torch.cat((torch.randn(num_envs, 3, generator=g) * 0.2, matrix_to_rot6d(rotation)), dim=-1),
        "gripper": torch.rand(num_envs, 1, generator=g),
        "wrist_cam": torch.randint(0, 256, (num_envs, IMAGE, IMAGE, 3), generator=g, dtype=torch.uint8),
    }


def _generators(seeds) -> list[torch.Generator]:
    return [torch.Generator().manual_seed(s) for s in seeds]


def _sample(policy, obs, seeds, **kwargs) -> torch.Tensor:
    return policy.sample_ensemble(obs, _generators(seeds), num_samples=3, num_steps=STEPS, **kwargs)


def test_release_contents(release_file, training_checkpoint):
    release = torch.load(release_file, map_location="cpu", weights_only=True)
    assert set(release) == {"algo_name", "config", "shape_metadata", "action_frame",
                            "action_normalization_stats", "model"}
    assert set(release["config"]) == {"algo", "observation"}
    assert set(release["config"]["algo"]) == {"horizon", "unet", "ddpm"}
    assert release["shape_metadata"] == {"ac_dim": 10, "all_shapes": SHAPES}
    assert release["action_frame"] == "relative"
    text = json.dumps({k: v for k, v in release.items() if k not in ("model", "action_normalization_stats")})
    for training_only in ("small_run", TRAIN_ONLY, "epoch", "optim", "small_env"):
        assert training_only not in text
    ema = training_checkpoint["model"]["ema"]
    assert release["model"].keys() == ema.keys()
    assert all(torch.equal(release["model"][k], ema[k]) for k in ema)
    stats = release["action_normalization_stats"]["actions"]
    assert torch.equal(stats["scale"], torch.tensor(SCALE)) and torch.equal(stats["offset"], torch.tensor(OFFSET))


def test_inconsistent_shape_metadata_is_rejected(training_checkpoint):
    checkpoint = dict(training_checkpoint)
    first = training_checkpoint["shape_metadata"][0]
    checkpoint["shape_metadata"] = [first, {**first, "ac_dim": 7}]
    with pytest.raises(ValueError):
        release_checkpoint(checkpoint)


def test_file_and_memory_releases_agree(policy, training_checkpoint):
    in_memory = DiffusionPolicy(release_checkpoint(training_checkpoint), device="cpu")
    obs = _observation(2)
    assert torch.equal(policy.predict(obs, num_steps=STEPS), in_memory.predict(obs, num_steps=STEPS))
    assert torch.equal(_sample(policy, obs, [1, 2]), _sample(in_memory, obs, [1, 2]))


def test_predict_is_the_zero_noise_chunk_relative_to_the_anchor(policy):
    obs = _observation(2)
    chunk = policy.predict(obs, num_steps=STEPS)
    assert chunk.shape == (2, HORIZON, 10)
    assert torch.equal(chunk, policy.predict(obs, num_steps=STEPS))
    with torch.no_grad():
        relative = policy._denoise(torch.zeros(2, HORIZON, 10), policy.encode(obs), STEPS)
    recovered = absolute_to_relative(chunk, obs["eef_pose"])
    torch.testing.assert_close(recovered[..., :3], relative[..., :3], atol=1e-5, rtol=0)
    torch.testing.assert_close(recovered[..., 9], relative[..., 9], atol=1e-6, rtol=0)
    torch.testing.assert_close(  # the network's 6-D output is not orthonormal; compare rotations
        rot6d_to_matrix(recovered[..., 3:9]), rot6d_to_matrix(relative[..., 3:9]), atol=1e-5, rtol=0
    )


def test_ensemble_shape_and_generator_determinism(policy):
    obs = _observation(2)
    samples = _sample(policy, obs, [1, 2])
    assert samples.shape == (2, 3, HORIZON, 10)
    assert torch.equal(samples, _sample(policy, obs, [1, 2]))
    assert not torch.allclose(samples[0], _sample(policy, obs, [5, 2])[0])
    assert (samples[:, 1:] - samples[:, :1]).abs().amax() > 0  # samples differ within an ensemble


def test_ensemble_independent_of_chunking_and_of_other_envs(policy):
    obs = _observation(3)
    reference = _sample(policy, obs, [1, 2, 3])
    for max_rows in (1, 2, 4, 9):
        torch.testing.assert_close(_sample(policy, obs, [1, 2, 3], max_rows=max_rows), reference, atol=1e-5, rtol=0)
    alone = _sample(policy, {k: v[1:2] for k, v in obs.items()}, [2])
    torch.testing.assert_close(alone[0], reference[1], atol=1e-5, rtol=0)
    order = [2, 0, 1]
    permuted = _sample(policy, {k: v[order] for k, v in obs.items()}, [3, 1, 2])
    torch.testing.assert_close(permuted, reference[order], atol=1e-5, rtol=0)


def test_inputs_are_validated(policy):
    obs = _observation(2)
    with pytest.raises(ValueError):
        policy.predict({**obs, "wrist_cam": obs["wrist_cam"].float() / 255.0}, num_steps=STEPS)
    with pytest.raises(ValueError):
        policy.sample_ensemble(obs, _generators([1]), num_samples=3, num_steps=STEPS)
