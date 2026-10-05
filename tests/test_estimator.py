import torch
from synthetic import JOINT_BASES, make_ensemble

from con_dec_imp import lie
from con_dec_imp.estimator import (
    ensemble_gains,
    estimate_feasible_subspace,
    gains_from_estimate,
    twist_matrix,
)
from con_dec_imp.gains import decomposed_gains, inertia_metric, isotropic_gains, modal_gains
from con_dec_imp.settings import ESTIMATION, GAINS
from con_dec_imp.twists import body_twists

torch.manual_seed(0)
F64 = dict(dtype=torch.float64)


def test_twist_matrix_layout():
    poses = make_ensemble(JOINT_BASES["revolute"], num_samples=5, horizon=7)
    W = twist_matrix(poses, ESTIMATION.dt)
    tw = body_twists(poses, ESTIMATION.dt)
    assert W.shape == (6, 35)
    torch.testing.assert_close(W[:, 3 * 7 + 4], tw[3, 4])
    assert twist_matrix(poses[None].expand(2, -1, -1, -1, -1)).shape == (2, 6, 35)


def test_ensembles_of_different_objects_in_one_batch():
    poses = torch.stack([make_ensemble(B, seed=i) for i, B in enumerate(JOINT_BASES.values())])
    est = estimate_feasible_subspace(poses)
    assert est.dim.tolist() == [1, 2, 3]
    metric = inertia_metric(dtype=torch.float64)
    K, D = gains_from_estimate(est, metric)
    assert K.shape == D.shape == (3, 6, 6)
    for i, part in enumerate(est.unbind()):
        Ki, Di = decomposed_gains(metric, part.basis())
        torch.testing.assert_close(K[i], Ki)
        torch.testing.assert_close(D[i], Di)


def test_gains_are_stiff_along_the_feasible_subspace():
    B = JOINT_BASES["cylindrical"]
    metric = inertia_metric(dtype=torch.float64)
    K, D, est = ensemble_gains(make_ensemble(B), metric)
    k_par, k_perp, _, _ = modal_gains(GAINS.k0, GAINS.reduction_ratio, GAINS.zeta)
    torch.testing.assert_close(K, K.mT)
    assert torch.linalg.eigvalsh(K).min() > 0 and torch.linalg.eigvalsh(D).min() > 0
    # Stiffness k_par on the (estimated ~ true) feasible directions, k_perp on their metric complement.
    torch.testing.assert_close(K @ B, k_par * metric @ B, rtol=0, atol=0.05)
    complement = torch.linalg.svd((metric @ B).T).Vh[2:].T  # metric-orthogonal to range(B)
    torch.testing.assert_close(K @ complement, k_perp * metric @ complement, rtol=0, atol=0.05)


def test_no_motion_gives_isotropic_gains():
    poses = make_ensemble(JOINT_BASES["revolute"], num_samples=16)[:, :1].expand(16, 24, 4, 4)
    poses = poses @ lie.se3_exp(1e-5 * torch.randn(16, 24, 6, **F64))
    metric = inertia_metric(dtype=torch.float64)
    K, D, est = ensemble_gains(poses, metric)
    assert int(est.dim) == 0
    K0, D0 = isotropic_gains(metric)
    torch.testing.assert_close(K, K0)
    torch.testing.assert_close(D, D0)


def test_estimate_invariant_to_reference_frame():
    poses = make_ensemble(JOINT_BASES["planar"])
    A = lie.se3_exp(torch.tensor([0.4, -1.0, 0.2, 3.0, -2.0, 1.0], **F64))
    e1, e2 = estimate_feasible_subspace(poses), estimate_feasible_subspace(A @ poses)
    assert int(e1.dim) == int(e2.dim) == 3
    torch.testing.assert_close(e1.alpha, e2.alpha)
    torch.testing.assert_close(e1.rms_speeds, e2.rms_speeds)


def test_waypoints_planned_before_grasping_are_left_out():
    poses = make_ensemble(JOINT_BASES["revolute"], num_samples=32, horizon=24)
    grasped = torch.ones(32, 24, dtype=torch.bool)
    torch.testing.assert_close(twist_matrix(poses, grasped=grasped), twist_matrix(poses), rtol=0, atol=1e-12)
    # Some samples still approach the handle at their first waypoints: a translation that is
    # not a feasible motion of the joint.
    approach = poses.clone()
    approach[:8, 0, :3, 3] += torch.tensor([0.0, 0.0, -0.02], **F64)
    assert int(estimate_feasible_subspace(approach).dim) > 1
    grasped[:8, 0] = False
    assert int(estimate_feasible_subspace(approach, grasped=grasped).dim) == 1
    assert twist_matrix(approach, grasped=grasped).shape == (6, 32 * 24 - 8)


def test_samples_without_grasped_waypoints_are_left_out():
    poses = make_ensemble(JOINT_BASES["revolute"], num_samples=4, horizon=24)
    grasped = torch.ones(4, 24, dtype=torch.bool)
    grasped[0] = False           # a sample that keeps the gripper open over the whole horizon
    grasped[1, 1:] = False       # and one with a single closed waypoint
    W = twist_matrix(poses, grasped=grasped)
    torch.testing.assert_close(W, twist_matrix(poses[2:]), rtol=0, atol=1e-12)
    assert twist_matrix(poses, grasped=torch.zeros(4, 24, dtype=torch.bool)).shape == (6, 0)
