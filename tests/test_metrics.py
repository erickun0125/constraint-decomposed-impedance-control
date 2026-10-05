import math

import numpy as np
import pytest

from con_dec_imp.metrics import (
    column_space,
    peak,
    principal_angles,
    reference_projectors,
    subspace_error,
    wrench_metrics,
)

rng = np.random.default_rng(0)


def rot_screw(axis, point):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate((a, np.cross(point, a)))


def trans_screw(axis):
    return np.concatenate((np.zeros(3), np.asarray(axis, float) / np.linalg.norm(axis)))


REVOLUTE = rot_screw((0, 0, 1), (0.0, -0.2, 0.0))[:, None]  # omega = z, v = -0.2 x
UNIVERSAL = np.stack([rot_screw(e, (0, 0, 0)) for e in np.eye(3)], axis=1)


def test_column_space_rank():
    block = np.outer([1.0, 2.0, 0.0], [1.0, -1.0]) + 1e-9 * rng.standard_normal((3, 2))
    assert column_space(block).shape == (3, 1)
    assert column_space(np.zeros((3, 2))).shape == (3, 0)
    assert column_space(np.zeros((3, 0))).shape == (3, 0)
    Q = column_space(rng.standard_normal((3, 2)))
    np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-12)


def test_reference_projectors_revolute():
    P = reference_projectors(REVOLUTE)
    ez, ex = np.diag([0.0, 0, 1]), np.diag([1.0, 0, 0])
    np.testing.assert_allclose(P.moment_par, ez, atol=1e-12)
    np.testing.assert_allclose(P.force_par, ex, atol=1e-12)
    for par, perp in ((P.moment_par, P.moment_perp), (P.force_par, P.force_perp)):
        np.testing.assert_allclose(par @ par, par, atol=1e-12)
        np.testing.assert_allclose(par + perp, np.eye(3), atol=1e-12)
        np.testing.assert_array_equal(par, par.T)


def test_reference_projectors_universal_has_empty_blocks():
    P = reference_projectors(UNIVERSAL)
    np.testing.assert_allclose(P.moment_perp, np.zeros((3, 3)), atol=1e-12)
    np.testing.assert_array_equal(P.force_par, np.zeros((3, 3)))
    np.testing.assert_array_equal(P.force_perp, np.eye(3))


def test_wrench_metrics_hand_computed():
    # Feasible force along x, feasible moment about z (REVOLUTE); 10 samples at dt = 0.1 s.
    n, dt = 10, 0.1
    wrench = np.tile([0.0, 0.5, 0.2, 3.0, 0.0, 4.0], (n, 1))  # moment (0, .5, .2), force (3, 0, 4)
    out = wrench_metrics(wrench, dt, reference_projectors(REVOLUTE))
    assert out["ICF"] == pytest.approx(4.0)  # |f_perp| = 4 N for 1 s
    assert out["PCF"] == pytest.approx(4.0)
    assert out["ICM"] == pytest.approx(0.5)
    assert out["PCM"] == pytest.approx(0.5)
    assert out["lambda_f"] == pytest.approx(4.0 / 7.0)
    assert out["lambda_m"] == pytest.approx(0.5 / 0.7)
    assert "beta_f" not in out


def test_peak_is_linear_interpolated_95th_percentile():
    force = np.zeros((4, 3))
    force[:, 2] = [1.0, 2.0, 3.0, 4.0]
    P = np.eye(3)
    assert peak(force, P) == pytest.approx(3.85)  # 3 + 0.85 * (4 - 3)
    assert peak(np.zeros((0, 3)), P) == 0.0


def test_feedback_fractions():
    n, dt = 20, 0.05
    measured = rng.standard_normal((n, 6))
    feedback = np.zeros((n, 6))
    feedback[:, 3] = 2.0  # feasible force (x)
    feedback[:, 5] = 2.0  # complement force (z)
    feedback[:, 2] = 1.0  # feasible moment (z)
    out = wrench_metrics(measured, dt, reference_projectors(REVOLUTE), feedback_wrench=feedback)
    assert out["beta_f"] == pytest.approx(0.5)
    assert out["beta_m"] == pytest.approx(1.0)
    # Structural values: empty feasible force block -> 0, empty moment complement -> 1.
    out = wrench_metrics(measured, dt, reference_projectors(UNIVERSAL), feedback_wrench=measured)
    assert out["beta_f"] == 0.0
    assert out["beta_m"] == pytest.approx(1.0)
    assert out["ICM"] < 1e-12 and out["lambda_m"] < 1e-12


def test_subspace_error_identical_and_reparametrized():
    U = rng.standard_normal((6, 2))
    assert subspace_error(U, U, 0.1) < 1e-5
    assert subspace_error(U @ np.array([[2.0, 1.0], [-1.0, 3.0]]), U, 0.1) < 1e-5


def test_subspace_error_in_the_pca_metric():
    # A rotation about x versus the same rotation with a lever: 45 deg when alpha equals the lever.
    U_a = np.array([[1.0, 0, 0, 0, 0, 0]]).T
    U_b = np.array([[1.0, 0, 0, 0, 0.1, 0]]).T
    assert subspace_error(U_a, U_b, 0.1) == pytest.approx(45.0)
    assert subspace_error(U_a, U_b, 0.2) == pytest.approx(math.degrees(math.atan(0.5)))
    assert subspace_error(np.eye(6)[:, :1], np.eye(6)[:, 1:2], 0.3) == pytest.approx(90.0)


def test_subspace_error_dimension_conventions():
    empty = np.zeros((6, 0))
    assert subspace_error(empty, empty, 0.1) == 0.0
    assert subspace_error(empty, REVOLUTE, 0.1) == 90.0
    angles = principal_angles(np.eye(6)[:, :3], np.eye(6)[:, 2:4], 0.1)
    np.testing.assert_allclose(angles, [0.0, 90.0], atol=1e-6)
    # Different dimensions: the full subspaces cannot coincide, even when one contains the other.
    assert subspace_error(np.eye(6)[:, :1], np.eye(6)[:, :3], 0.1) == 90.0
    assert subspace_error(np.eye(6)[:, :3], np.eye(6)[:, :1], 0.1) == 90.0
