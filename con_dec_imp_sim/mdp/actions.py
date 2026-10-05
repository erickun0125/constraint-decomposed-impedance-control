"""Impedance action term of the flying gripper (Eq. 2 with the gains of Section 4.2 and the errors of Eq. 9; Appendix A).

The action of each environment is a desired TCP pose and a gripper command,
``(position[3] (environment frame), rot6d[6], gripper[1])``. Every control step the
term computes the body-frame impedance wrench at the TCP

    F = K_v e_T + D_v e_V,     e_V = Ad_{T_b^{-1} T_d} V_d - V_b,

with ``V_d = 0`` and without the feedforward wrench of the general law (the gripper is
gravity free, and its Coriolis and reference-acceleration terms are neglected), and applies
it as torques ``tau = J_b^T F`` on the six virtual joints. The fingers track the open or
closed position depending on the gripper command (a command above
:data:`con_dec_imp.policy.actions.GRIPPER_CLOSED_THRESHOLD` closes).

Gains. Every environment starts with the isotropic gains ``(K_iso, D_iso)``. Installing
decomposed gains for an environment (at grasp closure) blends them in linearly over
``transition_steps`` control steps,

    K = (1 - a) K_iso + a K_dec,   a = min(1, n / transition_steps),

where ``n`` counts the control steps since installation, and the decomposed gains act
only while the fingers are physically closed; otherwise the isotropic gains apply.
A reset of an environment, or :meth:`ImpedanceAction.clear_decomposed_gains` (used when the
gripper is released), returns it to the isotropic gains.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from con_dec_imp.impedance import feedback_wrench, pose_error, twist_error
from con_dec_imp.lie import make_transform, quat_to_matrix, rot6d_to_matrix, skew
from con_dec_imp.policy.actions import ACTION_DIM, GRIPPER_CLOSED_THRESHOLD
from con_dec_imp.settings import CONTROL

from ..assets.flying_gripper import (
    FINGER_CLOSED,
    FINGER_JOINT,
    FINGER_OPEN,
    HAND_BODY,
    TCP_OFFSET_POS,
    TCP_OFFSET_ROT,
    VIRTUAL_JOINTS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ImpedanceAction(ActionTerm):
    """TCP impedance control through the virtual joints of the flying gripper."""

    cfg: ImpedanceActionCfg
    _asset: Articulation

    def __init__(self, cfg: ImpedanceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        body_ids, _ = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body matching {cfg.body_name!r}, found {len(body_ids)}.")
        self._body_idx = body_ids[0]
        # The fixed root link has no Jacobian block, so body indices shift by one.
        self._jacobian_idx = self._body_idx - 1
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)
        if len(self._joint_ids) != 6:
            raise ValueError(f"Expected six virtual joints matching {cfg.joint_names!r}, found {len(self._joint_ids)}.")
        finger_ids, _ = self._asset.find_joints(cfg.finger_joint_name)
        self._finger_id = finger_ids[0]
        self._finger_threshold = 0.5 * (cfg.finger_open + cfg.finger_closed)

        n, dev = self.num_envs, self.device
        self._offset_pos = torch.tensor(cfg.tcp_offset_pos, device=dev).expand(n, 3)
        self._offset_rot = torch.tensor(cfg.tcp_offset_rot, device=dev).expand(n, 4)

        self._raw_actions = torch.zeros(n, ACTION_DIM, device=dev)
        self._target_pose = torch.eye(4, device=dev).repeat(n, 1, 1)
        self._gripper_command = torch.zeros(n, device=dev)

        self._K_iso = torch.diag(torch.tensor(cfg.stiffness, device=dev))
        self._D_iso = torch.diag(torch.tensor(cfg.damping, device=dev))
        self._K_dec = self._K_iso.repeat(n, 1, 1)
        self._D_dec = self._D_iso.repeat(n, 1, 1)
        self._installed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._steps_since_install = torch.full((n,), float(cfg.transition_steps), device=dev)
        self._K = self._K_iso.repeat(n, 1, 1)
        self._D = self._D_iso.repeat(n, 1, 1)
        self._wrench = torch.zeros(n, 6, device=dev)

    # ------------------------------------------------------------------ action term interface

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Store the desired TCP pose (world frame) and the gripper command."""
        self._raw_actions[:] = actions
        self._target_pose[:, :3, :3] = rot6d_to_matrix(actions[:, 3:9])
        self._target_pose[:, :3, 3] = actions[:, :3] + self._env.scene.env_origins
        self._gripper_command[:] = actions[:, 9]

    def apply_actions(self) -> None:
        """Apply ``tau = J_b^T (K_v e_T + D_v e_V)`` and the finger position target."""
        fingers_closed = self.fingers_closed
        T_b, V_b, offset_w = self._tcp_state()
        e_T = pose_error(T_b, self._target_pose)
        e_V = twist_error(T_b, self._target_pose, torch.zeros_like(V_b), V_b)
        self._update_gains(fingers_closed)
        self._wrench = feedback_wrench(self._K, self._D, e_T, e_V)

        tau = (self._body_jacobian(T_b, offset_w).transpose(-1, -2) @ self._wrench[..., None])[..., 0]
        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)

        closing = self._gripper_command > GRIPPER_CLOSED_THRESHOLD
        finger_target = torch.where(closing, self.cfg.finger_closed, self.cfg.finger_open)
        self._asset.set_joint_position_target(finger_target[:, None], joint_ids=[self._finger_id])

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Hold the current TCP pose, open the gripper and return to the isotropic gains."""
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._target_pose[ids] = self.tcp_pose[ids]
        self._gripper_command[ids] = 0.0
        self.clear_decomposed_gains(ids)

    # ------------------------------------------------------------------ gains

    @property
    def stiffness(self) -> torch.Tensor:
        """Stiffness ``K_v`` used at the last control step, ``(num_envs, 6, 6)``."""
        return self._K

    @property
    def damping(self) -> torch.Tensor:
        """Damping ``D_v`` used at the last control step, ``(num_envs, 6, 6)``."""
        return self._D

    @property
    def isotropic_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Isotropic gains ``(K_iso, D_iso)``, each ``(6, 6)``."""
        return self._K_iso, self._D_iso

    def set_decomposed_gains(
        self,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Install decomposed gains for ``env_ids`` and start the transition from the isotropic gains.

        Args:
            stiffness: ``(len(env_ids), 6, 6)`` or ``(6, 6)`` stiffness ``K_v``.
            damping: Damping ``D_v`` of the same shape.
            env_ids: Environments to install into; the others are left unchanged.
        """
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._K_dec[ids] = stiffness.to(self.device, torch.float32)
        self._D_dec[ids] = damping.to(self.device, torch.float32)
        self._installed[ids] = True
        self._steps_since_install[ids] = 0.0

    def clear_decomposed_gains(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Return environments to the isotropic gains; decomposed gains are no longer installed there.

        Args:
            env_ids: Environment indices, a boolean mask ``(num_envs,)`` or a slice.
        """
        self._installed[env_ids] = False
        self._steps_since_install[env_ids] = float(self.cfg.transition_steps)

    @property
    def decomposed_gains_installed(self) -> torch.Tensor:
        """``(num_envs,)`` bool, True where decomposed gains are installed."""
        return self._installed

    def _update_gains(self, fingers_closed: torch.Tensor) -> None:
        """Gains of this control step (see the module docstring); advances the transition."""
        steps = self.cfg.transition_steps
        if steps > 0:
            a = (self._steps_since_install / steps).clamp(0.0, 1.0).view(-1, 1, 1)
            self._steps_since_install = torch.where(
                self._steps_since_install < steps, self._steps_since_install + 1.0, self._steps_since_install
            )
        else:
            a = torch.ones(self.num_envs, 1, 1, device=self.device)
        use_decomposed = (self._installed & fingers_closed).view(-1, 1, 1)
        self._K = torch.where(use_decomposed, (1.0 - a) * self._K_iso + a * self._K_dec, self._K_iso)
        self._D = torch.where(use_decomposed, (1.0 - a) * self._D_iso + a * self._D_dec, self._D_iso)

    # ------------------------------------------------------------------ state

    @property
    def fingers_closed(self) -> torch.Tensor:
        """``(num_envs,)`` bool, True where the fingers are physically closed (past mid-stroke)."""
        return self._asset.data.joint_pos[:, self._finger_id] < self._finger_threshold

    @property
    def target_pose(self) -> torch.Tensor:
        """Desired TCP pose ``T_d`` in the world frame, ``(num_envs, 4, 4)``."""
        return self._target_pose

    @property
    def tcp_pose(self) -> torch.Tensor:
        """Current TCP pose ``T_b`` in the world frame, ``(num_envs, 4, 4)``."""
        return self._tcp_state()[0]

    @property
    def tcp_twist(self) -> torch.Tensor:
        """Current TCP body twist ``V_b = (omega, v)``, ``(num_envs, 6)``."""
        return self._tcp_state()[1]

    @property
    def last_wrench(self) -> torch.Tensor:
        """Wrench ``(moment, force)`` commanded at the TCP in the last control step, ``(num_envs, 6)``."""
        return self._wrench

    @property
    def last_feedforward_wrench(self) -> torch.Tensor:
        """Feedforward part of :attr:`last_wrench`; zero, as the feedforward wrench is not used."""
        return torch.zeros_like(self._wrench)

    def _tcp_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TCP pose ``(N, 4, 4)``, body twist ``(N, 6)`` and the hand-to-TCP offset in the world frame."""
        data = self._asset.data
        hand_quat = data.body_quat_w[:, self._body_idx]
        offset_w = quat_apply(hand_quat, self._offset_pos)
        R = quat_to_matrix(quat_mul(hand_quat, self._offset_rot))
        T = make_transform(data.body_pos_w[:, self._body_idx] + offset_w, R)
        omega_w = data.body_ang_vel_w[:, self._body_idx]
        v_w = data.body_lin_vel_w[:, self._body_idx] + torch.cross(omega_w, offset_w, dim=-1)
        R_t = R.transpose(-1, -2)
        V_b = torch.cat(((R_t @ omega_w[..., None])[..., 0], (R_t @ v_w[..., None])[..., 0]), dim=-1)
        return T, V_b, offset_w

    def _body_jacobian(self, T_b: torch.Tensor, offset_w: torch.Tensor) -> torch.Tensor:
        """Body Jacobian ``J_b`` of the TCP w.r.t. the virtual joints, rows ``(omega, v)``, ``(N, 6, 6)``."""
        J = self._asset.root_physx_view.get_jacobians()[:, self._jacobian_idx, :, self._joint_ids]
        J_lin, J_ang = J[:, :3], J[:, 3:]
        J_lin = J_lin - skew(offset_w) @ J_ang  # hand origin -> TCP
        R_t = T_b[:, :3, :3].transpose(-1, -2)
        return torch.cat((R_t @ J_ang, R_t @ J_lin), dim=1)


@configclass
class ImpedanceActionCfg(ActionTermCfg):
    """Configuration of :class:`ImpedanceAction`."""

    class_type: type = ImpedanceAction
    asset_name: str = "gripper"
    body_name: str = HAND_BODY
    joint_names: str = VIRTUAL_JOINTS
    finger_joint_name: str = FINGER_JOINT
    finger_open: float = FINGER_OPEN
    finger_closed: float = FINGER_CLOSED
    tcp_offset_pos: tuple[float, float, float] = TCP_OFFSET_POS
    tcp_offset_rot: tuple[float, float, float, float] = TCP_OFFSET_ROT

    stiffness: tuple[float, float, float, float, float, float] = MISSING
    """Diagonal of the initial isotropic stiffness ``K_iso``, ordered ``(rotation, translation)``."""

    damping: tuple[float, float, float, float, float, float] = MISSING
    """Diagonal of the initial isotropic damping ``D_iso``, ordered ``(rotation, translation)``."""

    transition_steps: int = CONTROL.gain_transition_steps
    """Control steps of the linear blend from the isotropic to installed decomposed gains."""
