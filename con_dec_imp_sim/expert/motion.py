"""Waypoint stepping of the scripted expert (paper, Appendix H, Table "Simulation policy data").

Poses are ``(N, 7)`` tensors ``(position, quaternion (w, x, y, z))`` in the world frame.
Every 0.1 s the expert moves its waypoint from the current gripper pose toward the phase
target along the SE(3) geodesic (ScLERP), limited to a linear and an angular speed; the
controller then tracks the ScLERP interpolation between the two poses at 150 Hz.
"""

from __future__ import annotations

import torch

from con_dec_imp.lie import matrix_to_quat, matrix_to_rot6d, pose_to_transform, quat_to_matrix, se3_exp, se3_inverse, se3_log

SCLERP_EPS = 1e-6
"""Small-angle threshold of the SE(3) exponential and logarithm used for ScLERP."""


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of ``(w, x, y, z)`` quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (inverse rotation) of unit ``(w, x, y, z)`` quaternions."""
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def quat_about_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Float32 quaternion of a rotation by ``angle`` ``(N,)`` about the unit ``axis`` ``(N, 3)``.

    The trigonometry runs in the precision of ``angle``.
    """
    half = angle * 0.5
    sin_half = torch.sin(half)
    return torch.stack(
        (torch.cos(half), axis[:, 0] * sin_half, axis[:, 1] * sin_half, axis[:, 2] * sin_half), dim=-1
    ).float()


def rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` ``(N, 3)`` by the quaternion ``q`` ``(N, 4)`` (``q v q*``)."""
    v_quat = torch.cat((torch.zeros_like(v[:, :1]), v), dim=-1)
    q_conj = torch.cat((q[:, :1], -q[:, 1:]), dim=-1)
    return quat_mul(quat_mul(q, v_quat), q_conj)[:, 1:]


def orientation_error(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Rotation angle between two quaternions, ``2 acos |<q1, q2>|``."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    dot = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    return 2.0 * torch.acos(dot.abs().clamp(0.0, 1.0))


def _geodesic(start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Start transform and body twist ``log(T_start^-1 T_end)`` of the ScLERP from ``start`` to ``end``."""
    T_start = pose_to_transform(start[:, :3], start[:, 3:7])
    T_end = pose_to_transform(end[:, :3], end[:, 3:7])
    return T_start, se3_log(se3_inverse(T_start) @ T_end, SCLERP_EPS)


def _pose_along(T_start: torch.Tensor, twist: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        t = t[:, None]
    T = T_start @ se3_exp(t * twist, SCLERP_EPS)
    return torch.cat((T[:, :3, 3], matrix_to_quat(T[:, :3, :3])), dim=-1)


def sclerp(start: torch.Tensor, end: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
    """Screw linear interpolation ``T_start exp(t log(T_start^-1 T_end))``; ``t`` scalar or ``(N,)``."""
    return _pose_along(*_geodesic(start, end), t)


def next_waypoint(
    current: torch.Tensor,
    target: torch.Tensor,
    max_linear_speed: float,
    max_angular_speed: float,
    dt: float,
) -> torch.Tensor:
    """Waypoint ``dt`` ahead of ``current`` on the geodesic to ``target``, speed-limited.

    The ScLERP fraction is the smaller of ``max_linear_speed * dt`` over the straight-line
    distance and ``max_angular_speed * dt`` over the rotation angle, capped at 1 (``target``
    itself when it is closer). The limits are exact for a pure translation or rotation and
    approximate for a combined screw motion.
    """
    distance = torch.norm(target[:, :3] - current[:, :3], dim=-1, keepdim=True)
    t_pos = torch.clamp(max_linear_speed * dt / (distance + 1e-8), max=1.0)
    dot = torch.sum(current[:, 3:7] * target[:, 3:7], dim=-1, keepdim=True)
    angle = 2.0 * torch.acos(torch.clamp(torch.abs(dot), max=1.0))
    t_ori = torch.clamp(max_angular_speed * dt / (angle + 1e-8), max=1.0)
    return sclerp(current, target, torch.min(t_pos, t_ori)[:, 0])


def interpolate_poses(start: torch.Tensor, end: torch.Tensor, num_steps: int) -> list[torch.Tensor]:
    """Control-rate references from ``start`` to ``end``: ScLERP at ``t = k / num_steps``, ``k = 1..num_steps``."""
    T_start, twist = _geodesic(start, end)
    return [_pose_along(T_start, twist, (k + 1) / num_steps) for k in range(num_steps)]


def pose_to_action(pose: torch.Tensor, gripper: torch.Tensor, origin: torch.Tensor | None = None) -> torch.Tensor:
    """10-D action ``(position, rot6d, gripper)``; positions relative to ``origin`` if given."""
    position = pose[:, :3] if origin is None else pose[:, :3] - origin
    rot6d = matrix_to_rot6d(quat_to_matrix(pose[:, 3:7]))
    return torch.cat((position, rot6d, gripper[:, None]), dim=-1)
