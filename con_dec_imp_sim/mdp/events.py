"""Reset events: gripper start pose, object joint configuration and randomized object pose.

The random draws use the global torch generator in a fixed order (gripper position x,
y, z, gripper roll, pitch, yaw; then object x, y, z, roll, pitch, yaw; the revolute task
also resets its joint with Isaac Lab's ``reset_joints_by_offset`` in between), so seeding
torch before a batched reset fixes the initial configurations of all environments. Draws
over zero-width ranges (the fixed gripper start, the object roll and pitch) still advance
the generator; they are part of the sequence that the evaluation seed reproduces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

Range = tuple[float, float]


def _uniform(n: int, bounds: Range, device: str) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(*bounds)


def _intrinsic_yxz(quat: torch.Tensor) -> torch.Tensor:
    """Angles ``(a, b, c)`` with ``R(quat) = Ry(a) Rx(b) Rz(c)``, the order of the virtual revolute joints."""
    R = matrix_from_quat(quat)
    beta = torch.asin((-R[:, 1, 2]).clamp(-1.0, 1.0))
    alpha = torch.atan2(R[:, 0, 2], R[:, 2, 2])
    gamma = torch.atan2(R[:, 1, 0], R[:, 1, 1])
    gimbal_lock = torch.cos(beta).abs() < 1e-6
    if gimbal_lock.any():
        alpha[gimbal_lock] = torch.atan2(-R[gimbal_lock, 2, 0], R[gimbal_lock, 0, 0])
        gamma[gimbal_lock] = 0.0
    return torch.stack((alpha, beta, gamma), dim=-1)


def reset_gripper_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: tuple[Range, Range, Range],
    base_orientation: tuple[float, float, float, float],
    orientation_range: Range = (0.0, 0.0),
) -> None:
    """Place the hand by setting the virtual joints; fingers and velocities go to their defaults.

    Args:
        position_range: Bounds of the hand position (x, y, z) in the environment frame [m].
        base_orientation: Hand orientation ``(w, x, y, z)`` before the perturbation.
        orientation_range: Bounds of the roll, pitch and yaw perturbation [rad].
    """
    asset = env.scene[asset_cfg.name]
    n, device = len(env_ids), env.device
    position = torch.stack([_uniform(n, bounds, device) for bounds in position_range], dim=-1)
    base = torch.tensor(base_orientation, device=device).expand(n, 4)
    roll, pitch, yaw = (_uniform(n, orientation_range, device) for _ in range(3))
    quat = quat_mul(base, quat_from_euler_xyz(roll, pitch, yaw))

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos[:, :6] = torch.cat((position, _intrinsic_yxz(quat)), dim=-1)
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)


def reset_joint_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    positions: dict[str, float],
) -> None:
    """Set the listed joints to fixed positions and all joint velocities to zero."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[env_ids].clone()
    for name, value in positions.items():
        joint_pos[:, asset.find_joints(name)[0][0]] = value
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)


def reset_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range_x: Range = (0.0, 0.0),
    position_range_y: Range = (0.0, 0.0),
    position_range_z: Range = (0.0, 0.0),
    roll_range: Range = (0.0, 0.0),
    pitch_range: Range = (0.0, 0.0),
    yaw_range: Range = (0.0, 0.0),
) -> None:
    """Offset the object's default root pose by uniform position and roll/pitch/yaw perturbations."""
    asset = env.scene[asset_cfg.name]
    n, device = len(env_ids), env.device
    root_state = asset.data.default_root_state[env_ids].clone()
    for axis, bounds in enumerate((position_range_x, position_range_y, position_range_z)):
        root_state[:, axis] += _uniform(n, bounds, device)
    roll, pitch, yaw = (_uniform(n, bounds, device) for bounds in (roll_range, pitch_range, yaw_range))
    root_state[:, 3:7] = quat_mul(root_state[:, 3:7].clone(), quat_from_euler_xyz(roll, pitch, yaw))
    root_state[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
