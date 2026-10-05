import torch

from con_dec_imp import lie
from con_dec_imp.gains import decomposed_gains, inertia_metric
from con_dec_imp.impedance import feedback_wrench, pose_error, twist_error

torch.manual_seed(0)
F64 = dict(dtype=torch.float64)


def random_transform(n):
    w = 0.8 * torch.randn(n, 3, **F64)
    return lie.make_transform(torch.randn(n, 3, **F64), lie.so3_exp(w))


def test_zero_errors_at_the_reference():
    T = random_transform(10)
    V = torch.randn(10, 6, **F64)
    torch.testing.assert_close(pose_error(T, T), torch.zeros(10, 6, **F64), atol=1e-12, rtol=0)
    torch.testing.assert_close(twist_error(T, T, V, V), torch.zeros(10, 6, **F64))


def test_pose_error_translation_and_rotation_blocks():
    T_b = random_transform(10)
    d = torch.randn(10, 3, **F64)
    T_d = lie.make_transform(T_b[:, :3, 3] + d, T_b[:, :3, :3])
    expected = torch.cat((torch.zeros(10, 3, **F64), (T_b[:, :3, :3].mT @ d[..., None])[..., 0]), -1)
    torch.testing.assert_close(pose_error(T_b, T_d), expected, atol=1e-12, rtol=0)
    w = 0.5 * torch.randn(10, 3, **F64)
    T_d = lie.make_transform(T_b[:, :3, 3], T_b[:, :3, :3] @ lie.so3_exp(w))
    torch.testing.assert_close(pose_error(T_b, T_d), torch.cat((w, torch.zeros(10, 3, **F64)), -1))


def test_pose_error_first_order_matches_body_twist():
    T_b = random_transform(10)
    xi = 1e-5 * torch.randn(10, 6, **F64)
    e = pose_error(T_b, T_b @ lie.se3_exp(xi))
    torch.testing.assert_close(e, xi, atol=1e-9, rtol=0)


def test_twist_error_transports_desired_twist():
    T_b, T_d = random_transform(10), random_transform(10)
    V_d, V_b = torch.randn(10, 6, **F64), torch.randn(10, 6, **F64)
    Ad = lie.adjoint(lie.se3_inverse(T_b) @ T_d)
    torch.testing.assert_close(twist_error(T_b, T_d, V_d, V_b), (Ad @ V_d[..., None])[..., 0] - V_b)


def test_feedback_wrench_batched_and_passive_stiffness():
    metric = inertia_metric(dtype=torch.float64)
    K, D = decomposed_gains(metric, torch.randn(5, 6, 2, **F64))
    e_T, e_V = torch.randn(5, 6, **F64), torch.randn(5, 6, **F64)
    F = feedback_wrench(K, D, e_T, e_V)
    torch.testing.assert_close(F, torch.einsum("bij,bj->bi", K, e_T) + torch.einsum("bij,bj->bi", D, e_V))
    assert (torch.einsum("bi,bij,bj->b", e_T, K, e_T) > 0).all()
