"""Feasible-subspace estimation from a twist matrix (Algorithm 1, lines 6-9).

Given the twist matrix ``W = [V_1 ... V_M]`` (``6 x M`` with ``M >= 6``, twists ``(omega, v)``):

1. Characteristic length (Section 4.1, Appendix D.2): with ``u = (u_omega, u_v)`` the
   first left singular vector of ``W``, ``alpha = clip(|u_v| / |u_omega|, alpha_min, alpha_max)``.
2. Weighted PCA in the metric ``G = blkdiag(alpha^2 I3, I3)`` (Appendix D.1): with
   ``L_alpha = blkdiag(alpha I3, I3)``, ``L_alpha W = U~ S~ V~^T``, ``U = L_alpha^{-1} U~``
   and ``lambda_k = sigma~_k^2 / M`` (uncentered), so ``sqrt(lambda_k)`` is the RMS
   speed along ``u_k``.
3. Dimension selection (Section 4.1, Appendix D.1):
   ``m = #{k : sqrt(lambda_k) > max(kappa_rel sqrt(lambda_6), v_abs)}``.

All functions accept a single twist matrix ``(6, M)`` or a batch ``(..., 6, M)`` and
compute in the dtype of ``W``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from con_dec_imp.settings import ESTIMATION, EstimationSettings


@dataclass(frozen=True)
class SubspaceEstimate:
    """Result of the weighted PCA and the dimension rule.

    Attributes:
        directions: ``(..., 6, 6)`` principal directions ``U = [u_1 ... u_6]`` in twist
            coordinates, orthonormal in ``G = blkdiag(alpha^2 I3, I3)``, by decreasing
            ``lambda_k``.
        rms_speeds: ``(..., 6)`` RMS speeds ``sqrt(lambda_k)`` along the directions.
        alpha: ``(...)`` characteristic length of the PCA metric [m].
        dim: ``(...)`` selected feasible dimension ``m`` (int64).
    """

    directions: torch.Tensor
    rms_speeds: torch.Tensor
    alpha: torch.Tensor
    dim: torch.Tensor

    def basis(self) -> torch.Tensor:
        """Feasible basis ``U_par = [u_1 ... u_m]``, shape ``(6, m)``.

        Defined for an unbatched estimate; split a batch with :meth:`unbind`.
        """
        if self.alpha.dim() != 0:
            raise ValueError("basis() needs an unbatched estimate; split the batch with unbind()")
        return self.directions[:, : int(self.dim)]

    def unbind(self) -> list[SubspaceEstimate]:
        """Split a batched estimate into unbatched ones (row-major over the batch shape)."""
        return [
            SubspaceEstimate(U, s, a, m)
            for U, s, a, m in zip(
                self.directions.reshape(-1, 6, 6),
                self.rms_speeds.reshape(-1, 6),
                self.alpha.reshape(-1),
                self.dim.reshape(-1),
            )
        ]


def _check_twist_matrix(W: torch.Tensor) -> None:
    if W.dim() < 2 or W.shape[-2] != 6:
        raise ValueError(f"W must have shape (..., 6, M); got {tuple(W.shape)}")
    if W.shape[-1] < 6:
        raise ValueError(f"W needs at least 6 twists (columns) for six principal directions; got M={W.shape[-1]}")


def characteristic_length(
    W: torch.Tensor,
    alpha_min: float = ESTIMATION.alpha_min,
    alpha_max: float = ESTIMATION.alpha_max,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Characteristic length ``alpha = clip(|u_v| / |u_omega|, alpha_min, alpha_max)``.

    ``u`` is the first left singular vector of the unweighted twist matrix ``W``. Both
    norms are offset by ``eps``, so a pure translation (``u_omega = 0``) clips to
    ``alpha_max`` and a pure rotation about the control point to ``alpha_min``.

    Args:
        W: Twist matrix ``(..., 6, M)``.

    Returns:
        ``alpha`` with shape ``(...)``.
    """
    _check_twist_matrix(W)
    # Left singular vectors of W are the right singular vectors of W^T.
    _, _, Vh = torch.linalg.svd(W.mT, full_matrices=False)
    u = Vh[..., 0, :]
    ratio = (torch.linalg.norm(u[..., 3:], dim=-1) + eps) / (torch.linalg.norm(u[..., :3], dim=-1) + eps)
    return ratio.clamp(alpha_min, alpha_max)


def weighted_pca(W: torch.Tensor, alpha: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    """Uncentered PCA of the twists in the metric ``G = blkdiag(alpha^2 I3, I3)``.

    Args:
        W: Twist matrix ``(..., 6, M)``.
        alpha: Characteristic length, scalar or shape ``(...)``.

    Returns:
        ``(U, rms_speeds)``: the ``G``-orthonormal directions ``U = L_alpha^{-1} U~``
        ``(..., 6, 6)`` and ``sqrt(lambda_k)`` with ``lambda_k = sigma~_k^2 / M``
        ``(..., 6)``, both by decreasing ``lambda_k``.
    """
    _check_twist_matrix(W)
    alpha = torch.as_tensor(alpha, dtype=W.dtype, device=W.device)
    scale = torch.ones(W.shape[:-2] + (6,), dtype=W.dtype, device=W.device)
    scale[..., :3] = alpha[..., None]
    _, sigma, Vh = torch.linalg.svd((W * scale[..., :, None]).mT, full_matrices=False)
    directions = torch.linalg.solve_triangular(torch.diag_embed(scale), Vh.mT, upper=False)
    rms_speeds = torch.sqrt(sigma.square() / W.shape[-1])
    return directions, rms_speeds


def select_dimension(
    rms_speeds: torch.Tensor,
    v_abs: float = ESTIMATION.v_abs,
    kappa_rel: float = ESTIMATION.kappa_rel,
) -> torch.Tensor:
    """Feasible dimension ``m = #{k : sqrt(lambda_k) > max(kappa_rel sqrt(lambda_6), v_abs)}``.

    The count runs over all six directions. With the speeds sorted in decreasing
    order, the directions that pass are exactly the leading ``m``.

    Args:
        rms_speeds: ``(..., 6)`` RMS speeds ``sqrt(lambda_k)``, decreasing.
        v_abs: Absolute threshold [m/s].
        kappa_rel: Relative threshold (ratio to ``sqrt(lambda_6)``).

    Returns:
        ``m`` as an int64 tensor of shape ``(...)``.
    """
    threshold = torch.clamp(kappa_rel * rms_speeds[..., -1], min=v_abs)
    return (rms_speeds > threshold[..., None]).sum(dim=-1)


def estimate_subspace(W: torch.Tensor, settings: EstimationSettings = ESTIMATION) -> SubspaceEstimate:
    """Characteristic length, weighted PCA and dimension rule for a twist matrix ``(..., 6, M)``."""
    alpha = characteristic_length(W, settings.alpha_min, settings.alpha_max)
    directions, rms_speeds = weighted_pca(W, alpha)
    dim = select_dimension(rms_speeds, settings.v_abs, settings.kappa_rel)
    return SubspaceEstimate(directions=directions, rms_speeds=rms_speeds, alpha=alpha, dim=dim)
