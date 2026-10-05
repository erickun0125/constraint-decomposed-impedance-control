import math

import pytest
import torch

from con_dec_imp import lie

torch.manual_seed(0)
F64 = dict(dtype=torch.float64)


def random_rotvec(n, max_angle=math.pi - 1e-3):
    w = torch.randn(n, 3, **F64)
    angle = torch.rand(n, 1, **F64) * max_angle
    return w / w.norm(dim=-1, keepdim=True) * angle


def random_transform(n):
    return lie.make_transform(torch.randn(n, 3, **F64), lie.so3_exp(random_rotvec(n)))


def test_skew_vee_and_cross_product():
    w, x = torch.randn(10, 3, **F64), torch.randn(10, 3, **F64)
    W = lie.skew(w)
    torch.testing.assert_close(W, -W.mT)
    torch.testing.assert_close(lie.vee(W), w)
    torch.testing.assert_close((W @ x[..., None])[..., 0], torch.linalg.cross(w, x))


@pytest.mark.parametrize("scale", [1.0, 1e-9])
def test_so3_exp_log_round_trip(scale):
    w = random_rotvec(200) * scale
    R = lie.so3_exp(w)
    torch.testing.assert_close(R @ R.mT, torch.eye(3, **F64).expand(200, 3, 3))
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(200, **F64))
    torch.testing.assert_close(lie.so3_log(R), w, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("dtype, tol", [(torch.float64, 1e-9), (torch.float32, 2e-6)])
@pytest.mark.parametrize("gap", [1.0, 1e-2, 2e-3, 5e-4, 1e-6, 0.0])
def test_so3_log_close_to_half_turn(gap, dtype, tol):
    axis = torch.nn.functional.normalize(torch.randn(50, 3, dtype=dtype), dim=-1)
    w = axis * (math.pi - gap)
    R = lie.so3_exp(w)
    torch.testing.assert_close(lie.so3_exp(lie.so3_log(R)), R, atol=tol, rtol=0)
    if gap > 0:  # at exactly a half turn, w and -w are the same rotation
        torch.testing.assert_close(lie.so3_log(R), w, atol=tol, rtol=0)


def test_se3_exp_log_round_trip():
    xi = torch.cat((random_rotvec(200), torch.randn(200, 3, **F64)), dim=-1)
    T = lie.se3_exp(xi)
    torch.testing.assert_close(T[:, 3], torch.tensor([0, 0, 0, 1], **F64).expand(200, 4))
    torch.testing.assert_close(lie.se3_log(T), xi, atol=1e-8, rtol=1e-7)


def test_se3_exp_pure_translation_and_small_twist():
    v = torch.randn(5, 3, **F64)
    T = lie.se3_exp(torch.cat((torch.zeros(5, 3, **F64), v), dim=-1))
    torch.testing.assert_close(T[:, :3, :3], torch.eye(3, **F64).expand(5, 3, 3))
    torch.testing.assert_close(T[:, :3, 3], v)
    xi = 1e-6 * torch.randn(5, 6, **F64)
    hat = torch.zeros(5, 4, 4, **F64)
    hat[:, :3, :3], hat[:, :3, 3] = lie.skew(xi[:, :3]), xi[:, 3:]
    torch.testing.assert_close(lie.se3_exp(xi), torch.eye(4, **F64) + hat, atol=1e-11, rtol=0)


def test_quaternion_round_trip_and_sign():
    R = lie.so3_exp(random_rotvec(100))
    q = lie.matrix_to_quat(R)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(100, **F64))
    torch.testing.assert_close(lie.quat_to_matrix(q), R)
    torch.testing.assert_close(lie.quat_to_matrix(-q), R)
    identity = lie.matrix_to_quat(torch.eye(3, **F64))
    torch.testing.assert_close(identity.abs(), torch.tensor([1.0, 0, 0, 0], **F64))


def test_rot6d_round_trip_and_gram_schmidt():
    R = lie.so3_exp(random_rotvec(100))
    torch.testing.assert_close(lie.rot6d_to_matrix(lie.matrix_to_rot6d(R)), R)
    # Gram-Schmidt of arbitrary 6-vectors (well-conditioned pairs: the 1e-8 guard in the
    # normalization perturbs orthonormality by about 1e-8 / norm).
    x = torch.randn(100, 6, **F64)
    x[:, 3:] += 2.0 * torch.nn.functional.normalize(torch.linalg.cross(x[:, :3], torch.randn(100, 3, **F64)), dim=-1)
    R2 = lie.rot6d_to_matrix(x)
    torch.testing.assert_close(R2 @ R2.mT, torch.eye(3, **F64).expand(100, 3, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.linalg.det(R2), torch.ones(100, **F64), atol=1e-6, rtol=0)


def test_transforms_and_inverse():
    T = random_transform(50)
    torch.testing.assert_close(T @ lie.se3_inverse(T), torch.eye(4, **F64).expand(50, 4, 4))
    q = lie.matrix_to_quat(T[:, :3, :3])
    torch.testing.assert_close(lie.pose_to_transform(T[:, :3, 3], q), T)


def test_adjoint_properties():
    T1, T2 = random_transform(30), random_transform(30)
    V = torch.randn(30, 6, **F64)
    torch.testing.assert_close((lie.adjoint(T1) @ V[..., None])[..., 0], lie.transform_twist(T1, V))
    torch.testing.assert_close(lie.adjoint(T1 @ T2), lie.adjoint(T1) @ lie.adjoint(T2))
    torch.testing.assert_close(lie.adjoint(lie.se3_inverse(T1)), torch.linalg.inv(lie.adjoint(T1)))
    # Conjugation: T exp(V) T^-1 = exp(Ad_T V).
    V = 0.5 * V
    torch.testing.assert_close(
        T1 @ lie.se3_exp(V) @ lie.se3_inverse(T1), lie.se3_exp(lie.transform_twist(T1, V))
    )
