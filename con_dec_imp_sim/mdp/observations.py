"""Low-dimensional policy observations (Appendix H, Table "Diffusion Policy training configuration")."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from con_dec_imp.lie import matrix_to_rot6d, quat_to_matrix

from ..assets.flying_gripper import FINGER_JOINT, FINGER_OPEN

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def tcp_pose(env: ManagerBasedEnv, frame_cfg: SceneEntityCfg = SceneEntityCfg("gripper_frame")) -> torch.Tensor:
    """TCP pose ``(position, rot6d)``, ``(num_envs, 9)``; position relative to the environment origin."""
    frame = env.scene[frame_cfg.name]
    position = frame.data.target_pos_w[:, 0] - env.scene.env_origins
    rot6d = matrix_to_rot6d(quat_to_matrix(frame.data.target_quat_w[:, 0]))
    return torch.cat((position, rot6d), dim=-1)


def gripper_opening(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("gripper"),
    joint_name: str = FINGER_JOINT,
    open_position: float = FINGER_OPEN,
) -> torch.Tensor:
    """Finger opening normalized to ``[0, 1]`` (0 closed, 1 open), ``(num_envs, 1)``."""
    asset = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(joint_name)
    return (asset.data.joint_pos[:, joint_ids].mean(dim=-1, keepdim=True) / open_position).clamp(0.0, 1.0)
