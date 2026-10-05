import math

import pytest
import torch

from con_dec_imp.gains import (
    decomposed_gains,
    inertia_metric,
    isotropic_gains,
    metric_projectors,
    modal_gains,
)
from con_dec_imp.settings import GAINS, GainSettings

torch.manual_seed(0)
F64 = dict(dtype=torch.float64)
EYE = torch.eye(6, **F64)


@pytest.fixture
def metric():
    return inertia_metric(dtype=torch.float64)


def test_inertia_metric_values():
    M = inertia_metric(dtype=torch.float64)
    ell = math.sqrt(GAINS.rotational_inertia / GAINS.mass)
    torch.testing.assert_close(M.diagonal(), torch.tensor([ell**2] * 3 + [1.0] * 3, **F64))
    assert abs(ell - 0.103) < 5e-4  # l_Lambda of the flying gripper (Appendix H)
    M_phys = inertia_metric(normalize=False, dtype=torch.float64)
    torch.testing.assert_close(M_phys, GAINS.mass * M)


@pytest.mark.parametrize("m", [1, 2, 3, 5])
def test_projectors_are_metric_orthogonal(metric, m):
    U = torch.randn(4, 6, m, **F64)
    P, Q = metric_projectors(U, metric)
    torch.testing.assert_close(P @ P, P)
    torch.testing.assert_close(P + Q, EYE.expand(4, 6, 6))
    torch.testing.assert_close(P @ U, U)
    torch.testing.assert_close(metric @ P, P.mT @ metric)  # self-adjoint in the metric
    torch.testing.assert_close(P.mT @ metric @ Q, torch.zeros(4, 6, 6, **F64), atol=1e-12, rtol=0)
    torch.testing.assert_close(torch.linalg.matrix_rank(P), torch.full((4,), m))


def test_projector_independent_of_basis_choice(metric):
    U = torch.randn(6, 3, **F64)
    A = torch.randn(3, 3, **F64) + 3 * torch.eye(3, **F64)
    torch.testing.assert_close(metric_projectors(U @ A, metric)[0], metric_projectors(U, metric)[0])


def test_projector_edge_dimensions(metric):
    P, Q = metric_projectors(torch.zeros(6, 0, **F64), metric)
    torch.testing.assert_close(P, torch.zeros(6, 6, **F64))
    torch.testing.assert_close(Q, EYE)
    P, Q = metric_projectors(torch.randn(6, 6, **F64), metric)
    torch.testing.assert_close(P, EYE)


def test_modal_gains_paper_values():
    k_par, k_perp, d_par, d_perp = modal_gains(GAINS.k0, GAINS.reduction_ratio, GAINS.zeta)
    assert (k_par, k_perp) == (250.0, 25.0)
    # Appendix H (numerical gain values), printed with two decimals.
    assert d_par == pytest.approx(24.21, abs=0.01)
    assert d_perp == pytest.approx(7.66, abs=0.01)
    assert d_perp == pytest.approx(math.sqrt(GAINS.reduction_ratio) * d_par)


def test_isotropic_gains(metric):
    K, D = isotropic_gains(metric)
    _, _, d0, _ = modal_gains(GAINS.k0, 1.0, GAINS.zeta)
    torch.testing.assert_close(K, GAINS.k0 * metric)
    torch.testing.assert_close(D, d0 * metric)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_decomposed_gains_symmetric_psd_with_modal_spectrum(metric, m):
    U = torch.randn(6, m, **F64)
    K, D = decomposed_gains(metric, U)
    k_par, k_perp, d_par, d_perp = modal_gains(GAINS.k0, GAINS.reduction_ratio, GAINS.zeta)
    for G, (par, perp) in ((K, (k_par, k_perp)), (D, (d_par, d_perp))):
        torch.testing.assert_close(G, G.mT)
        assert torch.linalg.eigvalsh(G).min() > 0
        # Generalized eigenvalues of (G, metric) are the modal gains.
        L = torch.linalg.cholesky(metric)
        Li = torch.linalg.inv(L)
        eig = torch.linalg.eigvalsh(Li @ G @ Li.mT)
        expected = torch.tensor(sorted([perp] * (6 - m) + [par] * m), **F64)
        torch.testing.assert_close(eig, expected)
    torch.testing.assert_close(K @ U, k_par * metric @ U)


def test_reduction_ratio_one_is_isotropic(metric):
    iso = GainSettings(reduction_ratio=1.0)
    K, D = decomposed_gains(metric, torch.randn(6, 2, **F64), iso)
    K0, D0 = isotropic_gains(metric)
    torch.testing.assert_close(K, K0)
    torch.testing.assert_close(D, D0)
    torch.testing.assert_close(K, GAINS.k0 * metric)


def test_zero_dimension_falls_back_to_isotropic(metric):
    K, D = decomposed_gains(metric, torch.zeros(3, 6, 0, **F64))
    K0, D0 = isotropic_gains(metric)
    assert K.shape == (3, 6, 6)
    torch.testing.assert_close(K, K0.expand(3, 6, 6))
    torch.testing.assert_close(D, D0.expand(3, 6, 6))


def test_batched_gains_match_single(metric):
    U = torch.randn(4, 6, 2, **F64)
    K, D = decomposed_gains(metric, U)
    for i in range(4):
        Ki, Di = decomposed_gains(metric, U[i])
        torch.testing.assert_close(K[i], Ki)
        torch.testing.assert_close(D[i], Di)
