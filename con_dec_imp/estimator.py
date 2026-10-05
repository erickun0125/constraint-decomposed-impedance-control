"""Algorithm 1: decomposed impedance gains from a trajectory ensemble.

    poses (..., N, H, 4, 4)  --body_twists-->  W (..., 6, N H)  --estimate_subspace-->
    SubspaceEstimate  --decomposed_gains-->  K_v, D_v (..., 6, 6)

Leading batch dimensions (several ensembles) are processed together; each ensemble gets
its own characteristic length, directions and dimension. A mask of grasped waypoints
(``grasped``) applies to one ensemble at a time.
"""

from __future__ import annotations

import torch

from con_dec_imp.gains import decomposed_gains
from con_dec_imp.settings import ESTIMATION, GAINS, EstimationSettings, GainSettings
from con_dec_imp.subspace import SubspaceEstimate, estimate_subspace
from con_dec_imp.twists import body_twists, selected_body_twists


def twist_matrix(
    poses: torch.Tensor, dt: float = ESTIMATION.dt, grasped: torch.Tensor | None = None
) -> torch.Tensor:
    """Twist matrix ``W = [V^(1)_1 ... V^(1)_H  V^(2)_1 ... V^(N)_H]`` of an ensemble.

    Args:
        poses: Ensemble of pose trajectories ``(..., N, H, 4, 4)``.
        dt: Time between consecutive waypoints [s].
        grasped: Optional mask ``(N, H)`` of the waypoints at which the sampled trajectory
            commands a closed gripper. Only these waypoints are differentiated, so motion
            that a sample still plans before closing the gripper does not enter ``W``; a
            sample with fewer than two such waypoints is left out. Needs an unbatched
            ensemble.

    Returns:
        ``W`` with shape ``(..., 6, N H)``, or ``(6, M)`` with ``M`` the number of selected
        waypoints when ``grasped`` is given.
    """
    if poses.dim() < 4:
        raise ValueError(f"poses must have shape (..., N, H, 4, 4); got {tuple(poses.shape)}")
    if grasped is None:
        return body_twists(poses, dt).flatten(-3, -2).mT
    if poses.dim() != 4:
        raise ValueError("a waypoint mask needs an unbatched ensemble (N, H, 4, 4)")
    twists, valid = selected_body_twists(poses, grasped, dt)
    return twists[valid].mT


def estimate_feasible_subspace(
    poses: torch.Tensor,
    settings: EstimationSettings = ESTIMATION,
    grasped: torch.Tensor | None = None,
) -> SubspaceEstimate:
    """Feasible-subspace estimate of an ensemble ``(..., N, H, 4, 4)`` (Algorithm 1, lines 3-9)."""
    return estimate_subspace(twist_matrix(poses, settings.dt, grasped), settings)


def gains_from_estimate(
    estimate: SubspaceEstimate, metric: torch.Tensor, settings: GainSettings = GAINS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decomposed gains ``K_v, D_v`` of a (batched) estimate (Algorithm 1, lines 10-11).

    Each ensemble uses its own selected dimension; ``m = 0`` gives the isotropic gains.

    Returns:
        ``(K_v, D_v)``, each ``(..., 6, 6)`` with the batch shape of the estimate.
    """
    batch = estimate.alpha.shape
    gains = [decomposed_gains(metric, e.basis(), settings) for e in estimate.unbind()]
    K = torch.stack([k for k, _ in gains]).reshape(batch + (6, 6))
    D = torch.stack([d for _, d in gains]).reshape(batch + (6, 6))
    return K, D


def ensemble_gains(
    poses: torch.Tensor,
    metric: torch.Tensor,
    estimation: EstimationSettings = ESTIMATION,
    gain_settings: GainSettings = GAINS,
    grasped: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SubspaceEstimate]:
    """Algorithm 1 from a sampled ensemble (lines 3-11): ``(K_v, D_v, estimate)`` from poses ``(..., N, H, 4, 4)``.

    ``grasped`` optionally restricts the twists to the waypoints at which each sample
    commands a closed gripper (see :func:`twist_matrix`).
    """
    estimate = estimate_feasible_subspace(poses, estimation, grasped)
    K, D = gains_from_estimate(estimate, metric, gain_settings)
    return K, D, estimate
