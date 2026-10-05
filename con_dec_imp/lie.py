"""Rotation and rigid-body transform utilities (batched, torch).

Conventions used throughout the package:

* Twists are ordered ``(omega, v)`` (angular, then linear) and wrenches
  ``(moment, force)``.
* Quaternions are ``(w, x, y, z)``.
* The 6-D rotation representation is the first two columns of the rotation
  matrix, ``[R[:, 0], R[:, 1]]``, recovered by Gram-Schmidt orthonormalization.
* Homogeneous transforms are ``(..., 4, 4)`` tensors.
"""

from __future__ import annotations

import torch

_EPS = 1e-7


def skew(w: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix ``[w]`` of a 3-vector, shape ``(..., 3, 3)``."""
    x, y, z = w.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(w.shape[:-1] + (3, 3))


def vee(W: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`skew`."""
    return torch.stack((W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]), dim=-1)


def so3_exp(w: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Rodrigues formula ``exp([w])``."""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)[..., None]
    theta_sq = theta * theta
    W = skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(W.shape)
    theta_safe = torch.where(theta < eps, torch.ones_like(theta), theta)
    theta_sq_safe = torch.where(theta_sq < eps * eps, torch.ones_like(theta_sq), theta_sq)
    c1 = torch.where(theta < eps, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta_safe)
    c2 = torch.where(theta_sq < eps * eps, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / theta_sq_safe)
    return eye + c1 * W + c2 * (W @ W)


def so3_log(R: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Axis-angle vector ``log(R)^vee``.

    The angle is ``atan2(sin, cos)`` of the antisymmetric and trace parts. Beyond a
    quarter turn the axis ``n`` is recovered from the symmetric part,
    ``(R + R^T) / 2 = cos(theta) I + (1 - cos(theta)) n n^T``, which stays well
    conditioned up to a half turn; its sign is taken from the antisymmetric part.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    w_unscaled = vee((R - R.transpose(-1, -2)) / 2.0)
    sin_theta = torch.linalg.norm(w_unscaled, dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)
    scale = torch.where(theta < eps, torch.ones_like(theta), theta / torch.clamp(sin_theta, min=eps))
    w = scale[..., None] * w_unscaled

    eye = torch.eye(3, dtype=R.dtype, device=R.device).expand(R.shape)
    c = cos_theta[..., None, None]
    nnT = ((R + R.transpose(-1, -2)) / 2.0 - c * eye) / torch.clamp(1.0 - c, min=eps)
    diag = torch.diagonal(nnT, dim1=-2, dim2=-1)
    col = diag.argmax(dim=-1, keepdim=True)
    n = torch.gather(nnT, -1, col[..., None].expand(nnT.shape[:-1] + (1,)))[..., 0]
    n = n / torch.sqrt(torch.clamp(torch.gather(diag, -1, col), min=eps))
    n = torch.where((n * w_unscaled).sum(-1, keepdim=True) < 0.0, -n, n)
    return torch.where((theta > torch.pi / 2)[..., None], theta[..., None] * n, w)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix of a ``(w, x, y, z)`` quaternion."""
    r, i, j, k = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    m = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """``(w, x, y, z)`` quaternion of a rotation matrix (numerically robust branch selection)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R.reshape(R.shape[:-2] + (9,)).unbind(-1)
    q_abs = torch.sqrt(torch.clamp(torch.stack(
        (1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22),
        dim=-1,
    ), min=0.0))
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., None].clamp(min=0.1))
    best = q_abs.argmax(dim=-1)
    return torch.gather(candidates, -2, best[..., None, None].expand(best.shape + (1, 4)))[..., 0, :]


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from the 6-D representation (Gram-Schmidt on the two columns)."""
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 / (torch.linalg.norm(a1, dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """6-D representation ``[R[:, 0], R[:, 1]]`` of a rotation matrix."""
    return torch.cat((R[..., :, 0], R[..., :, 1]), dim=-1)


def make_transform(position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Homogeneous transform from a position ``(..., 3)`` and a rotation matrix ``(..., 3, 3)``."""
    T = torch.zeros(position.shape[:-1] + (4, 4), dtype=position.dtype, device=position.device)
    T[..., :3, :3] = rotation
    T[..., :3, 3] = position
    T[..., 3, 3] = 1.0
    return T


def pose_to_transform(position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Homogeneous transform from a position and a ``(w, x, y, z)`` quaternion."""
    return make_transform(position, quat_to_matrix(quat))


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Inverse of a homogeneous transform."""
    R_t = T[..., :3, :3].transpose(-1, -2)
    return make_transform(-(R_t @ T[..., :3, 3:4])[..., 0], R_t)


def se3_exp(xi: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Exponential map of a twist ``(omega, v)``."""
    w, v = xi[..., :3], xi[..., 3:]
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)[..., None]
    theta_sq, theta_cu = theta * theta, theta * theta * theta
    W = skew(w)
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(W.shape)
    theta_sq_safe = torch.where(theta_sq < eps * eps, torch.ones_like(theta_sq), theta_sq)
    theta_cu_safe = torch.where(theta_cu < eps ** 3, torch.ones_like(theta_cu), theta_cu)
    c1 = torch.where(theta_sq < eps * eps, torch.full_like(theta, 0.5), (1.0 - torch.cos(theta)) / theta_sq_safe)
    c2 = torch.where(theta_cu < eps ** 3, torch.full_like(theta, 1.0 / 6.0), (theta - torch.sin(theta)) / theta_cu_safe)
    J = torch.where(theta < eps, eye, eye + c1 * W + c2 * (W @ W))
    return make_transform((J @ v[..., None])[..., 0], so3_exp(w, eps))


def se3_log(T: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Logarithm map of a transform as a twist ``(omega, v)``."""
    w = so3_log(T[..., :3, :3], eps)
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)[..., None]
    theta_sq = theta * theta
    W = skew(w)
    eye = torch.eye(3, dtype=T.dtype, device=T.device).expand(W.shape)
    theta_safe = torch.where(theta < eps, torch.ones_like(theta), theta)
    theta_sq_safe = torch.where(theta_sq < eps * eps, torch.ones_like(theta_sq), theta_sq)
    sin_theta = torch.sin(theta)
    sin_safe = torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta)
    c = 1.0 / theta_sq_safe - (1.0 + torch.cos(theta)) / (2.0 * theta_safe * sin_safe)
    c = torch.where(theta_sq < eps * eps, torch.full_like(c, 1.0 / 12.0), c)
    J_inv = torch.where(theta < eps, eye, eye - 0.5 * W + c * (W @ W))
    v = (J_inv @ T[..., :3, 3:4])[..., 0]
    return torch.cat((w, v), dim=-1)


def adjoint(T: torch.Tensor) -> torch.Tensor:
    """Adjoint matrix ``[Ad_T]`` acting on ``(omega, v)`` twists, shape ``(..., 6, 6)``."""
    R, p = T[..., :3, :3], T[..., :3, 3]
    Ad = torch.zeros(T.shape[:-2] + (6, 6), dtype=T.dtype, device=T.device)
    Ad[..., :3, :3] = R
    Ad[..., 3:, :3] = skew(p) @ R
    Ad[..., 3:, 3:] = R
    return Ad


def transform_twist(T: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Apply ``[Ad_T]`` to a twist without forming the 6x6 matrix."""
    R, p = T[..., :3, :3], T[..., :3, 3]
    w = (R @ V[..., :3, None])[..., 0]
    v = torch.cross(p, w, dim=-1) + (R @ V[..., 3:, None])[..., 0]
    return torch.cat((w, v), dim=-1)
