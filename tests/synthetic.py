"""Synthetic trajectory ensembles of rigidly grasped articulated objects (test helpers)."""

from __future__ import annotations

import math

import torch

from con_dec_imp.lie import se3_exp


def screw(axis, point=None) -> torch.Tensor:
    """Body twist ``(omega, v)`` of a unit rotation about ``axis`` through ``point``, or a
    unit translation along ``axis`` when ``point`` is None."""
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    if point is None:
        return torch.cat((torch.zeros(3, dtype=torch.float64), a))
    q = torch.tensor(point, dtype=torch.float64)
    return torch.cat((a, torch.linalg.cross(q, a)))


# Feasible body-twist bases of configuration-independent joints (columns), dimensions 1, 2, 3.
JOINT_BASES = {
    "revolute": torch.stack([screw((0, 0, 1), (0.2, 0.1, 0.0))], dim=1),
    "cylindrical": torch.stack([screw((0, 0, 1), (0.0, -0.15, 0.0)), screw((0, 0, 1))], dim=1),
    "planar": torch.stack([screw((1, 0, 0), (0.0, 0.0, 0.0)), screw((0, 1, 0)), screw((0, 0, 1))], dim=1),
}


def make_ensemble(
    basis: torch.Tensor,
    num_samples: int = 64,
    horizon: int = 24,
    dt: float = 0.1,
    noise: float = 1e-4,
    seed: int = 0,
) -> torch.Tensor:
    """Pose trajectories ``(N, H, 4, 4)`` (float64) whose body twists lie in ``range(basis)``.

    Each trajectory follows ``T_{t+1} = T_t exp(dt * basis @ c(t))`` with per-trajectory
    random speeds ``c(t)`` (0.15-0.6 per unit screw, slowly modulated), starting at a common
    pose, plus independent pose noise ``T_t exp(noise * xi)``.
    """
    gen = torch.Generator().manual_seed(seed)
    m = basis.shape[1]
    T0 = se3_exp(torch.tensor([0.3, -0.2, 0.5, 0.4, 0.1, 0.6], dtype=torch.float64))
    speed = (0.15 + 0.45 * torch.rand(num_samples, 1, m, generator=gen, dtype=torch.float64))
    speed = speed * torch.where(torch.rand(num_samples, 1, m, generator=gen) < 0.5, -1.0, 1.0)
    phase = 2 * math.pi * torch.rand(num_samples, 1, m, generator=gen, dtype=torch.float64)
    t = torch.arange(horizon, dtype=torch.float64)[None, :, None]
    coeff = speed * (1.0 + 0.3 * torch.sin(0.3 * t + phase))  # (N, H, m)
    steps = se3_exp(dt * coeff @ basis.T)  # (N, H, 4, 4)
    poses = [T0.expand(num_samples, 4, 4)]
    for k in range(horizon - 1):
        poses.append(poses[-1] @ steps[:, k])
    poses = torch.stack(poses, dim=1)
    xi = noise * torch.randn(num_samples, horizon, 6, generator=gen, dtype=torch.float64)
    return poses @ se3_exp(xi)
