"""Evaluation metrics (Appendix H, "Evaluation metrics"), numpy only.

Wrenches are ordered ``(moment, force)`` and twists ``(omega, v)``, both in the
end-effector body frame. The reference feasible subspace ``S_ref`` with basis
``U_ref = [U_omega; U_v]`` (``6 x k``) is scored blockwise: with ``Q_omega, Q_v``
orthonormal bases of the column spaces of ``U_omega`` and ``U_v``,

    P_{s,par} = Q_s Q_s^T,   P_{s,perp} = I3 - P_{s,par},   s in {omega, v}.

Force is projected with the linear block and moment with the angular block. Over the
evaluation window, sampled at interval ``dt``,

    ICF = sum_t |f_perp| dt,   PCF = Q95(|f_perp|),   ICM, PCM likewise for the moment,
    lambda_f = J_f,perp / (J_f,par + J_f,perp)      (load fraction; lambda_m analogous),
    beta_f = J^fb_f,par / (J^fb_f,par + J^fb_f,perp) (feedback feasible fraction on the
                                                     feedback wrench; beta_m analogous),

with ``J_par = sum_t |f_par| dt``. ``Q95`` is the 95th percentile with linear
interpolation. The fractions add ``1e-8`` to the denominator. The subspace error (SE) is
the largest principal angle between the full estimated and reference subspaces in the PCA
metric ``G = blkdiag(alpha^2 I3, I3)`` with a fixed evaluation length ``alpha`` per object;
subspaces of different dimension score 90 degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRACTION_EPS = 1e-8
"""Denominator offset of the load and feedback fractions."""


@dataclass(frozen=True)
class ReferenceProjectors:
    """Blockwise feasible/complement projectors ``(3, 3)`` of a reference subspace."""

    moment_par: np.ndarray
    moment_perp: np.ndarray
    force_par: np.ndarray
    force_perp: np.ndarray


def column_space(block: np.ndarray, rank_tol: float = 1e-6, abs_tol: float = 1e-12) -> np.ndarray:
    """Orthonormal basis ``(3, r)`` of the column space of ``block``.

    The rank counts singular values above ``max(rank_tol * sigma_max, abs_tol)``, so a
    zero block has rank 0.
    """
    block = np.asarray(block, dtype=np.float64)
    if block.shape[1] == 0:
        return np.zeros((block.shape[0], 0))
    u, s, _ = np.linalg.svd(block, full_matrices=False)
    rank = int((s > max(rank_tol * s[0], abs_tol)).sum())
    return u[:, :rank]


def reference_projectors(U_ref: np.ndarray, rank_tol: float = 1e-6) -> ReferenceProjectors:
    """Blockwise projectors of the reference basis ``U_ref`` ``(6, k)`` in ``(omega, v)`` order."""
    U_ref = np.asarray(U_ref, dtype=np.float64)
    if U_ref.ndim != 2 or U_ref.shape[0] != 6:
        raise ValueError(f"U_ref must have shape (6, k); got {U_ref.shape}")
    Q_omega, Q_v = column_space(U_ref[:3], rank_tol), column_space(U_ref[3:], rank_tol)
    P_omega, P_v = Q_omega @ Q_omega.T, Q_v @ Q_v.T
    eye = np.eye(3)
    return ReferenceProjectors(
        moment_par=P_omega, moment_perp=eye - P_omega, force_par=P_v, force_perp=eye - P_v
    )


def projected_norms(x: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Per-sample norms ``|P x_t|`` of a series ``(T, 3)`` for a symmetric projector ``P``."""
    return np.linalg.norm(np.asarray(x, dtype=np.float64) @ P, axis=1)


def impulse(x: np.ndarray, P: np.ndarray, dt: float) -> float:
    """Impulse ``sum_t |P x_t| dt`` (rectangle rule)."""
    return float(projected_norms(x, P).sum() * dt)


def peak(x: np.ndarray, P: np.ndarray, q: float = 95.0) -> float:
    """Percentile ``Q_q(|P x_t|)`` with linear interpolation (0 for an empty series)."""
    norms = projected_norms(x, P)
    return float(np.percentile(norms, q)) if norms.size else 0.0


def fraction(numerator: float, other: float) -> float:
    """``numerator / (numerator + other + 1e-8)``."""
    return numerator / (numerator + other + FRACTION_EPS)


def wrench_metrics(
    wrench: np.ndarray,
    dt: float,
    projectors: ReferenceProjectors,
    feedback_wrench: np.ndarray | None = None,
) -> dict[str, float]:
    """Complement impulses, peaks and fractions over an evaluation window.

    Args:
        wrench: Measured external wrench ``(T, 6)`` as ``(moment, force)``.
        dt: Sample interval [s].
        projectors: Reference projectors of the trial.
        feedback_wrench: Commanded feedback wrench ``(T, 6)`` (feedforward excluded);
            when given, ``beta_f`` and ``beta_m`` are added.

    Returns:
        ``ICF`` [N s], ``PCF`` [N], ``ICM`` [N m s], ``PCM`` [N m], ``lambda_f``,
        ``lambda_m`` and optionally ``beta_f``, ``beta_m``.
    """
    wrench = np.asarray(wrench, dtype=np.float64)
    moment, force = wrench[:, :3], wrench[:, 3:]
    out = {
        "ICF": impulse(force, projectors.force_perp, dt),
        "PCF": peak(force, projectors.force_perp),
        "ICM": impulse(moment, projectors.moment_perp, dt),
        "PCM": peak(moment, projectors.moment_perp),
    }
    out["lambda_f"] = fraction(out["ICF"], impulse(force, projectors.force_par, dt))
    out["lambda_m"] = fraction(out["ICM"], impulse(moment, projectors.moment_par, dt))
    if feedback_wrench is not None:
        fb = np.asarray(feedback_wrench, dtype=np.float64)
        fb_moment, fb_force = fb[:, :3], fb[:, 3:]
        out["beta_f"] = fraction(
            impulse(fb_force, projectors.force_par, dt), impulse(fb_force, projectors.force_perp, dt)
        )
        out["beta_m"] = fraction(
            impulse(fb_moment, projectors.moment_par, dt), impulse(fb_moment, projectors.moment_perp, dt)
        )
    return out


def principal_angles(U_a: np.ndarray, U_b: np.ndarray, alpha: float) -> np.ndarray:
    """Principal angles [deg] between ``range(U_a)`` and ``range(U_b)`` in ``G = blkdiag(alpha^2 I3, I3)``.

    Both bases ``(6, m)`` are mapped by ``L_alpha = blkdiag(alpha I3, I3)`` and
    orthonormalized; the angles are the arccosines of the singular values of
    ``Q_a^T Q_b`` (``min(m_a, m_b)`` angles, increasing).
    """
    L = np.diag([alpha] * 3 + [1.0] * 3)
    Q_a, _ = np.linalg.qr(L @ np.asarray(U_a, dtype=np.float64))
    Q_b, _ = np.linalg.qr(L @ np.asarray(U_b, dtype=np.float64))
    cosines = np.linalg.svd(Q_a.T @ Q_b, compute_uv=False)
    return np.sort(np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0))))


def subspace_error(U_est: np.ndarray, U_ref: np.ndarray, alpha: float) -> float:
    """Subspace error SE [deg]: the largest principal angle between the full subspaces in the PCA metric.

    ``U_est`` ``(6, m_est)`` and ``U_ref`` ``(6, m_ref)`` are bases (full column rank).
    Subspaces of different dimension cannot coincide, so they score 90: SE is the maximum
    principal angle of the full subspaces (Section 5, Appendix H), not of the smaller one against
    the larger one. Two empty subspaces give 0.
    """
    m_est, m_ref = np.shape(U_est)[1], np.shape(U_ref)[1]
    if m_est != m_ref:
        return 90.0
    if m_est == 0:
        return 0.0
    return float(principal_angles(U_est, U_ref, alpha).max())
