"""Inertia metric, metric-orthogonal projectors and impedance gains (Section 4.2).

With a basis ``U`` of the estimated feasible subspace and the inertia metric
``Lam`` (block-isotropic, Appendix A.2), the feedback gains are

    P_par  = U (U^T Lam U)^{-1} U^T Lam,      P_perp = I - P_par,
    K_v    = Lam (k_par P_par + k_perp P_perp),
    D_v    = Lam (d_par P_par + d_perp P_perp).

Iso, Ours and Oracle (Appendix G.2) all use this construction: Iso sets the
reduction ratio r = 1 (so K_v = k0 Lam), Ours uses the estimated subspace and
Oracle the analytic reference subspace, both with r < 1.
"""

from __future__ import annotations

import math

import torch

from con_dec_imp.settings import GAINS, GainSettings


def inertia_metric(
    rotational_inertia: float = GAINS.rotational_inertia,
    mass: float = GAINS.mass,
    *,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Block-isotropic inertia metric ``blkdiag(I_bar I3, m_bar I3)``.

    With ``normalize=True`` the metric is divided by ``m_bar``, giving
    ``blkdiag(l^2 I3, I3)`` with ``l = sqrt(I_bar / m_bar)``. Scaling the metric
    leaves the projectors unchanged and only rescales the gains, so the scalar
    gains of :class:`~con_dec_imp.settings.GainSettings` refer to this normalized form.
    """
    rot, lin = (rotational_inertia / mass, 1.0) if normalize else (rotational_inertia, mass)
    diag = torch.tensor([rot] * 3 + [lin] * 3, dtype=dtype, device=device)
    return torch.diag(diag)


def metric_projectors(basis: torch.Tensor, metric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Metric-orthogonal projectors onto ``range(basis)`` and its complement.

    Args:
        basis: ``(..., 6, m)`` basis of the feasible subspace (need not be orthonormal).
        metric: ``(6, 6)`` symmetric positive-definite metric.

    Returns:
        ``(P_par, P_perp)``, each ``(..., 6, 6)``. For ``m = 0``, ``P_par = 0``.
    """
    eye = torch.eye(6, dtype=basis.dtype, device=basis.device).expand(basis.shape[:-2] + (6, 6))
    if basis.shape[-1] == 0:
        return torch.zeros_like(eye), eye.clone()
    metric = metric.to(dtype=basis.dtype, device=basis.device)
    UtM = basis.transpose(-1, -2) @ metric
    P_par = basis @ torch.linalg.solve(UtM @ basis, UtM)
    return P_par, eye - P_par


def modal_gains(k0: float, reduction_ratio: float, zeta: float) -> tuple[float, float, float, float]:
    """Scalar gains ``(k_par, k_perp, d_par, d_perp)`` of the second-order gain rule.

    ``k_par = k0``, ``k_perp = r k0`` and ``d = 2 zeta sqrt(k)`` for both, so the
    complement keeps the damping ratio (``d_perp = sqrt(r) d_par``).
    """
    k_par, k_perp = k0, reduction_ratio * k0
    return k_par, k_perp, 2.0 * zeta * math.sqrt(k_par), 2.0 * zeta * math.sqrt(k_perp)


def isotropic_gains(
    metric: torch.Tensor, settings: GainSettings = GAINS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iso gains ``K_v = k0 Lam``, ``D_v = d0 Lam``."""
    k0, _, d0, _ = modal_gains(settings.k0, 1.0, settings.zeta)
    return k0 * metric, d0 * metric


def decomposed_gains(
    metric: torch.Tensor,
    basis: torch.Tensor,
    settings: GainSettings = GAINS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decomposed gains ``K_v, D_v`` for a feasible-subspace basis ``(..., 6, m)``.

    A zero-dimensional basis carries no feasible direction to keep stiff, and the
    isotropic gains are returned instead.
    """
    if basis.shape[-1] == 0:
        K, D = isotropic_gains(metric.to(basis), settings)
        shape = basis.shape[:-2] + (6, 6)
        return K.expand(shape).clone(), D.expand(shape).clone()
    k_par, k_perp, d_par, d_perp = modal_gains(settings.k0, settings.reduction_ratio, settings.zeta)
    P_par, P_perp = metric_projectors(basis, metric)
    metric = metric.to(dtype=basis.dtype, device=basis.device)
    MP_par, MP_perp = metric @ P_par, metric @ P_perp
    return k_par * MP_par + k_perp * MP_perp, d_par * MP_par + d_perp * MP_perp
