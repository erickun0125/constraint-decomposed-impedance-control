import math

import pytest
import torch
from synthetic import JOINT_BASES, make_ensemble, screw

from con_dec_imp.settings import ESTIMATION
from con_dec_imp.subspace import (
    SubspaceEstimate,
    characteristic_length,
    estimate_subspace,
    select_dimension,
    weighted_pca,
)
from con_dec_imp.twists import body_twists

torch.manual_seed(0)
F64 = dict(dtype=torch.float64)


def pca_metric(alpha):
    return torch.diag(torch.tensor([alpha**2] * 3 + [1.0] * 3, **F64))


def max_angle_deg(A, B, alpha):
    """Largest principal angle between range(A) and range(B) in blkdiag(alpha^2 I, I)."""
    L = torch.diag(torch.tensor([alpha] * 3 + [1.0] * 3, **F64))
    Qa, _ = torch.linalg.qr(L @ A)
    Qb, _ = torch.linalg.qr(L @ B)
    s = torch.linalg.svdvals(Qa.T @ Qb).clamp(0.0, 1.0)
    return math.degrees(torch.acos(s.min()).item())


def twist_matrix_of(name, **kw):
    poses = make_ensemble(JOINT_BASES[name], **kw)
    return body_twists(poses, ESTIMATION.dt).reshape(-1, 6).T


@pytest.mark.parametrize("name, m", [("revolute", 1), ("cylindrical", 2), ("planar", 3)])
def test_screw_ensembles_recover_dimension_and_subspace(name, m):
    est = estimate_subspace(twist_matrix_of(name))
    assert int(est.dim) == m
    assert est.basis().shape == (6, m)
    assert max_angle_deg(est.basis(), JOINT_BASES[name], est.alpha.item()) < 0.5
    # Off-subspace speeds stay well below the absolute threshold.
    assert est.rms_speeds[m:].max() < 0.5 * ESTIMATION.v_abs


def test_float32_matches_float64():
    W = twist_matrix_of("cylindrical")
    e64, e32 = estimate_subspace(W), estimate_subspace(W.float())
    assert int(e32.dim) == int(e64.dim) == 2
    assert e32.alpha.item() == pytest.approx(e64.alpha.item(), rel=1e-4)
    assert max_angle_deg(e32.basis().double(), e64.basis(), e64.alpha.item()) < 1e-2


def test_characteristic_length_of_a_single_screw():
    B = screw((0, 0, 1), (0.2, 0.1, 0.0))  # |v| / |omega| = 0.2236
    W = B[:, None] * torch.linspace(-1.0, 2.0, 50, **F64)
    torch.testing.assert_close(characteristic_length(W), torch.tensor(math.hypot(0.2, 0.1), **F64), rtol=1e-5, atol=0)


def test_characteristic_length_clips():
    t = torch.linspace(0.5, 1.0, 20, **F64)
    pure_translation = screw((1, 1, 0))[:, None] * t
    pure_rotation = screw((0, 1, 0), (0, 0, 0))[:, None] * t
    assert characteristic_length(pure_translation).item() == ESTIMATION.alpha_max
    assert characteristic_length(pure_rotation).item() == ESTIMATION.alpha_min
    far = screw((0, 0, 1), (3.0, 0, 0))[:, None] * t
    assert characteristic_length(far, alpha_max=1.0).item() == 1.0


def test_characteristic_length_ignores_sign_scale_and_order():
    W = twist_matrix_of("revolute")
    a = characteristic_length(W)
    torch.testing.assert_close(characteristic_length(-3.0 * W[:, torch.randperm(W.shape[1])]), a)


def test_weighted_pca_is_metric_orthonormal_and_uncentered():
    W = torch.randn(6, 300, **F64)
    alpha = 0.12
    U, rms = weighted_pca(W, alpha)
    torch.testing.assert_close(U.T @ pca_metric(alpha) @ U, torch.eye(6, **F64))
    assert (rms[:-1] >= rms[1:]).all()
    L = torch.diag(torch.tensor([alpha] * 3 + [1.0] * 3, **F64))
    torch.testing.assert_close(rms.square().sum(), (L @ W).square().sum() / W.shape[1])
    # A constant ensemble is pure signal for the uncentered PCA.
    V0 = torch.tensor([0.0, 0.0, 1.0, 0.1, -0.2, 0.0], **F64)
    U, rms = weighted_pca(V0[:, None].expand(6, 40), alpha)
    torch.testing.assert_close(rms[0], (L @ V0).norm())
    assert rms[1:].max() < 1e-12
    assert max_angle_deg(U[:, :1], V0[:, None], alpha) < 1e-6


def test_weighted_pca_rank_deficient_reconstruction():
    B = JOINT_BASES["planar"]
    W = B @ torch.randn(3, 100, generator=torch.Generator().manual_seed(1), **F64)
    U, rms = weighted_pca(W, 0.2)
    assert rms[3:].max() < 1e-12 * rms[0]
    # Principal angles near zero are resolved to about sqrt(eps) rad.
    assert max_angle_deg(U[:, :3], B, 0.2) < 1e-4


def test_select_dimension_rule():
    mm = 1e-3
    # Absolute test dominates; equality with the threshold does not count (strict >).
    assert select_dimension(torch.tensor([10, 5, 1, 1, 1, 1], **F64) * mm).item() == 1
    assert select_dimension(torch.tensor([10, 5.01, 1, 1, 1, 1], **F64) * mm).item() == 2
    # Relative test dominates: threshold = kappa_rel * sqrt(lambda_6) = 20 mm/s.
    rms = torch.tensor([100, 50, 20, 6, 5.5, 5], **F64) * mm
    assert select_dimension(rms).item() == 2
    assert select_dimension(rms, kappa_rel=1.0).item() == 5
    assert select_dimension(torch.full((6,), 4 * mm, **F64)).item() == 0
    batch = torch.stack([rms, torch.tensor([10, 9, 8, 1, 1, 1], **F64) * mm])
    torch.testing.assert_close(select_dimension(batch), torch.tensor([2, 3]))


def test_batched_estimate_matches_single_and_unbind():
    Ws = [twist_matrix_of(name, seed=i) for i, name in enumerate(JOINT_BASES)]
    est = estimate_subspace(torch.stack(Ws))
    assert est.directions.shape == (3, 6, 6) and est.dim.tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        est.basis()
    parts = est.unbind()
    assert len(parts) == 3 and all(isinstance(p, SubspaceEstimate) for p in parts)
    for part, W in zip(parts, Ws):
        single = estimate_subspace(W)
        torch.testing.assert_close(part.directions, single.directions)
        torch.testing.assert_close(part.alpha, single.alpha)
        assert int(part.dim) == int(single.dim)


def test_invalid_twist_matrix():
    with pytest.raises(ValueError):
        estimate_subspace(torch.randn(100, 6, **F64))
    with pytest.raises(ValueError, match="at least 6 twists"):
        estimate_subspace(torch.randn(6, 5, **F64))
    assert estimate_subspace(torch.randn(6, 6, **F64)).directions.shape == (6, 6)
