"""Diffusion Policy inference (paper, Section 5.1 and Appendix H).

The policy is robomimic's UNet Diffusion Policy. It is conditioned on one observation
(wrist RGB image, end-effector pose as position + 6-D rotation, gripper opening) and
predicts a relative chunk of H poses (:mod:`con_dec_imp.policy.actions`). Two samplers
are used, both DDIM on the training noise schedule:

* :meth:`DiffusionPolicy.predict` gives the executed reference, the deterministic chunk
  denoised from zero initial noise.
* :meth:`DiffusionPolicy.sample_ensemble` gives the trajectory ensemble of Algorithm 1
  (line 2): N chunks per environment from Gaussian initial noise with eta = 0, denoised
  in one batched pass. The evaluation samples each environment separately (B = 1):
  GPU kernel selection depends on the batch shape, so batching several environments
  changes the samples slightly.

Both convert the network's anchor-relative output to absolute chunks ``(..., H, 10)`` in
the frame of the observed end-effector pose ``eef_pose`` (the environment frame).

Release checkpoints are weight-only dictionaries (see :func:`release_checkpoint`) that
load with ``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.config import config_factory

from con_dec_imp.policy.actions import relative_to_absolute
from con_dec_imp.settings import ESTIMATION

ANCHOR_KEY = "eef_pose"
"""Observation key of the end-effector pose ``(B, 9)`` that anchors the relative chunk."""

MAX_ROWS = 512
"""Upper bound on the rows (environments x samples) denoised in one network call."""

_CONFIG_ALGO_SECTIONS = ("horizon", "unet", "ddpm")


def release_checkpoint(training_checkpoint: Mapping) -> dict:
    """Weight-only release checkpoint from a robomimic training checkpoint.

    Keeps the EMA weights (the weights robomimic uses for inference), the robomimic
    config sections that define the network and the training noise schedule
    (``algo.horizon``, ``algo.unet``, ``algo.ddpm`` and ``observation``), the observation
    shapes and the action normalization. Dataset paths, run names, optimizer and
    scheduler states, and training progress are dropped. The result contains only
    tensors and plain Python containers, so it loads with ``weights_only=True``.
    """
    config = training_checkpoint["config"]
    config = json.loads(config if isinstance(config, str) else json.dumps(config))
    algo = config["algo"]
    model = training_checkpoint["model"]
    if not algo["ema"]["enabled"]:
        raise ValueError("the training checkpoint has no EMA weights (algo.ema.enabled is false)")
    weights = model["ema"]

    shape_metadata = training_checkpoint["shape_metadata"]
    if isinstance(shape_metadata, list):
        # Multi-dataset training stores one entry per dataset; they describe one network.
        if any(entry != shape_metadata[0] for entry in shape_metadata[1:]):
            raise ValueError("shape_metadata entries of the training datasets differ")
        shape_metadata = shape_metadata[0]
    ac_dim = int(shape_metadata["ac_dim"])

    stats = training_checkpoint["action_normalization_stats"]["actions"]

    def as_vector(value) -> torch.Tensor:
        vector = torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(-1).copy())
        if vector.numel() != ac_dim:
            raise ValueError(f"action normalization has {vector.numel()} entries, expected {ac_dim}")
        return vector

    return {
        "algo_name": "diffusion_policy",
        "config": {
            "algo": {section: algo[section] for section in _CONFIG_ALGO_SECTIONS},
            "observation": config["observation"],
        },
        "shape_metadata": {
            "ac_dim": ac_dim,
            "all_shapes": {key: [int(d) for d in shape] for key, shape in shape_metadata["all_shapes"].items()},
        },
        "action_frame": "relative",
        "action_normalization_stats": {
            "actions": {"scale": as_vector(stats["scale"]), "offset": as_vector(stats["offset"])},
        },
        "model": {name: tensor.detach().clone() for name, tensor in weights.items()},
    }


class DiffusionPolicy:
    """Diffusion Policy restored from a release checkpoint, for inference only."""

    def __init__(self, release: Mapping, device: str | torch.device = "cuda"):
        if release["algo_name"] != "diffusion_policy" or release["action_frame"] != "relative":
            raise ValueError("expected a Diffusion Policy checkpoint with relative action chunks")
        self.device = torch.device(device)

        # Network: robomimic's DiffusionPolicyUNet built from the stored config sections.
        # The stored weights are already the EMA weights, so no EMA copy is created, and
        # no optimizer state is needed.
        config = config_factory("diffusion_policy")
        with config.values_unlocked():
            config.algo.update(release["config"]["algo"])
            config.observation.update(release["config"]["observation"])
            config.algo.ema.enabled = False
        with config.unlocked():
            config.algo.optim_params = {}
        config.lock()
        if config.algo.horizon.observation_horizon != 1:
            raise ValueError("the policy is conditioned on a single observation")
        ObsUtils.initialize_obs_utils_with_config(config)

        shape_metadata = release["shape_metadata"]
        algo = DiffusionPolicyUNet(
            algo_config=config.algo,
            obs_config=config.observation,
            global_config=config,
            obs_key_shapes=OrderedDict((k, list(v)) for k, v in shape_metadata["all_shapes"].items()),
            ac_dim=shape_metadata["ac_dim"],
            device=self.device,
        )
        algo.nets.load_state_dict(release["model"])
        algo.set_eval()
        self._encoder = algo.nets["policy"]["obs_encoder"]
        self._unet = algo.nets["policy"]["noise_pred_net"]

        self.obs_keys: tuple[str, ...] = tuple(algo.obs_shapes)
        """Observation keys the policy reads, in the order of the encoder features."""
        self.horizon: int = config.algo.horizon.prediction_horizon
        """Chunk length H."""
        self.action_dim: int = algo.ac_dim
        self._image_keys = frozenset(config.observation.modalities.obs.rgb)

        stats = release["action_normalization_stats"]["actions"]
        self._action_scale = stats["scale"].to(self.device, torch.float32)
        self._action_offset = stats["offset"].to(self.device, torch.float32)

        # DDIM on the forward process the network was trained with.
        ddpm = config.algo.ddpm
        self._scheduler = DDIMScheduler(
            num_train_timesteps=ddpm.num_train_timesteps,
            beta_schedule=ddpm.beta_schedule,
            clip_sample=ddpm.clip_sample,
            prediction_type=ddpm.prediction_type,
            set_alpha_to_one=True,
            steps_offset=0,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cuda") -> DiffusionPolicy:
        """Load a release checkpoint file."""
        return cls(torch.load(path, map_location="cpu", weights_only=True), device)

    @torch.no_grad()
    def predict(
        self,
        observation: Mapping[str, torch.Tensor],
        num_steps: int = ESTIMATION.denoising_steps,
    ) -> torch.Tensor:
        """Executed reference: the DDIM chunk denoised from zero initial noise.

        Args:
            observation: batched observation, see :meth:`encode`.
            num_steps: DDIM denoising steps.

        Returns:
            Absolute chunk ``(B, H, 10)``: position, 6-D rotation, gripper command.
        """
        cond = self.encode(observation)
        noise = torch.zeros(cond.shape[0], self.horizon, self.action_dim, device=self.device)
        return relative_to_absolute(self._denoise(noise, cond, num_steps), self._anchor(observation))

    @torch.no_grad()
    def sample_ensemble(
        self,
        observation: Mapping[str, torch.Tensor],
        generators: Sequence[torch.Generator],
        num_samples: int = ESTIMATION.num_samples,
        num_steps: int = ESTIMATION.denoising_steps,
        max_rows: int = MAX_ROWS,
    ) -> torch.Tensor:
        """Trajectory ensemble of Algorithm 1 (line 2), DDIM with eta = 0.

        The observation is encoded once per environment; the ``B * N`` rows are then
        denoised together, split into calls of at most ``max_rows`` rows only to bound
        memory. The initial noise of environment ``b`` is drawn from ``generators[b]``
        alone, so it does not depend on which other environments are sampled in the
        same call. The denoised samples do depend slightly on the batch shape, because
        GPU kernel selection does; the evaluation therefore samples each environment
        separately (B = 1).

        Args:
            observation: batched observation of ``B`` environments, see :meth:`encode`.
            generators: one ``torch.Generator`` per environment (any device).
            num_samples: ensemble size N.
            num_steps: DDIM denoising steps.
            max_rows: rows per network call.

        Returns:
            Absolute chunks ``(B, N, H, 10)``.
        """
        cond = self.encode(observation)
        num_envs = cond.shape[0]
        if len(generators) != num_envs:
            raise ValueError(f"got {len(generators)} generators for {num_envs} environments")
        shape = (num_samples, self.horizon, self.action_dim)
        noise = torch.stack(
            [torch.randn(shape, generator=g, device=g.device, dtype=torch.float32) for g in generators]
        ).to(self.device)
        noise = noise.reshape(num_envs * num_samples, *shape[1:])
        cond = cond.repeat_interleave(num_samples, dim=0)
        relative = torch.cat(
            [
                self._denoise(noise[i : i + max_rows], cond[i : i + max_rows], num_steps)
                for i in range(0, noise.shape[0], max_rows)
            ]
        ).reshape(num_envs, *shape)
        anchor = self._anchor(observation)[:, None].expand(num_envs, num_samples, -1)
        return relative_to_absolute(relative, anchor)

    @torch.no_grad()
    def encode(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Observation features ``(B, D)`` that condition the denoiser.

        ``observation`` maps every key of :attr:`obs_keys` to a batched tensor: images
        ``(B, H, W, 3)`` uint8 RGB, the others ``(B, d)`` (``eef_pose``: position +
        6-D rotation in the environment frame; ``gripper``: finger opening in [0, 1]).
        """
        inputs = {}
        for key in self.obs_keys:
            x = observation[key].to(self.device)
            if key in self._image_keys:
                if x.dtype != torch.uint8 or x.dim() != 4 or x.shape[-1] != 3:
                    raise ValueError(f"'{key}' must be a (B, H, W, 3) uint8 image batch")
                # (B, 3, H, W) in [0, 1]. The permuted (channels-last) memory layout is kept
                # rather than made contiguous: the layout selects the convolution kernels.
                x = x.permute(0, 3, 1, 2).float() / 255.0
            else:
                x = x.float()
            inputs[key] = x
        return self._encoder(obs=inputs)

    def _anchor(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return observation[ANCHOR_KEY].to(self.device, torch.float32)

    def _denoise(self, noise: torch.Tensor, cond: torch.Tensor, num_steps: int) -> torch.Tensor:
        """DDIM (eta = 0) from ``noise`` ``(R, H, A)``; returns the unnormalized relative chunk."""
        self._scheduler.set_timesteps(num_steps)
        timesteps = self._scheduler.timesteps
        sample = noise
        for t, t_device in zip(timesteps, timesteps.to(self.device)):
            eps = self._unet(sample=sample, timestep=t_device, global_cond=cond)
            sample = self._scheduler.step(model_output=eps, timestep=t, sample=sample).prev_sample
        return sample * self._action_scale + self._action_offset
