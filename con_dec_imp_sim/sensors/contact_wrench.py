"""Gripper-object contact wrench at the TCP (Appendix H, "Contact and force sensing").

Three filtered contact sensors (left finger, right finger, hand) report the contacts of
the gripper with the moving link that carries the handle. The wrench is the sum of the
normal and friction contact forces of these pairs and their moments about the TCP,
expressed in the TCP body frame and ordered ``(moment, force)``; it is the force the
object exerts on the gripper. It is used only for evaluation, not for control.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import ContactSensorCfg

from con_dec_imp.lie import quat_to_matrix

from ..assets.flying_gripper import FINGER_BODIES, HAND_BODY

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

CONTACT_BODIES = (*FINGER_BODIES, HAND_BODY)
CONTACT_SENSOR_NAMES = tuple(f"contact_{body}" for body in CONTACT_BODIES)
"""Scene keys of the contact sensors, one per gripper body."""

MAX_CONTACTS_PER_BODY = 32
"""Contact-point buffer per gripper body and environment."""


def contact_sensor_cfgs(target_prim_path: str) -> dict[str, ContactSensorCfg]:
    """Contact sensors of the three gripper bodies, filtered to contacts with ``target_prim_path``."""
    return {
        name: ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Gripper/{body}",
            filter_prim_paths_expr=[target_prim_path],
            max_contact_data_count_per_prim=MAX_CONTACTS_PER_BODY,
        )
        for name, body in zip(CONTACT_SENSOR_NAMES, CONTACT_BODIES)
    }


class ContactWrenchReader:
    """Reads the object-on-gripper contact wrench at the TCP, ``(num_envs, 6)``.

    Args:
        env: The environment; its scene holds the contact sensors and the TCP frame.
        sensor_names: Scene keys of the contact sensors.
        tcp_frame: Scene key of the frame transformer whose first target is the TCP.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        sensor_names: Sequence[str] = CONTACT_SENSOR_NAMES,
        tcp_frame: str = "gripper_frame",
    ) -> None:
        self._env = env
        self._views = [env.scene[name].contact_physx_view for name in sensor_names]
        self._tcp_frame = env.scene[tcp_frame]
        self._dt = env.physics_dt

    def read(self) -> torch.Tensor:
        """Contact wrench ``(moment, force)`` in the TCP body frame, ``(num_envs, 6)``."""
        p_tcp = self._tcp_frame.data.target_pos_w[:, 0]
        R_tcp = quat_to_matrix(self._tcp_frame.data.target_quat_w[:, 0])
        force = torch.zeros_like(p_tcp)
        moment = torch.zeros_like(p_tcp)
        for view in self._views:
            magnitude, points, normals, _, count, start = view.get_contact_data(dt=self._dt)
            f_n, m_n = self._sum_pairs(magnitude.view(-1, 1) * normals.view(-1, 3), points.view(-1, 3), count, start, p_tcp)
            friction, f_points, f_count, f_start = view.get_friction_data(dt=self._dt)
            f_t, m_t = self._sum_pairs(friction.view(-1, 3), f_points.view(-1, 3), f_count, f_start, p_tcp)
            force += f_n + f_t
            moment += m_n + m_t
        R_t = R_tcp.transpose(-1, -2)
        return torch.cat(((R_t @ moment[..., None])[..., 0], (R_t @ force[..., None])[..., 0]), dim=-1)

    def _sum_pairs(
        self,
        forces: torch.Tensor,
        points: torch.Tensor,
        count: torch.Tensor,
        start: torch.Tensor,
        p_tcp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-environment force and moment about the TCP of the contacts in the packed buffers.

        ``count`` and ``start`` index the contacts of every (environment, body, filter) pair;
        contacts with non-finite positions are skipped.
        """
        num_envs = p_tcp.shape[0]
        count = count.view(num_envs, -1).long()
        start = start.view(num_envs, -1).long()
        max_count = int(count.max()) if count.numel() else 0
        if max_count == 0 or forces.shape[0] == 0:
            return torch.zeros_like(p_tcp), torch.zeros_like(p_tcp)
        slot = torch.arange(max_count, device=forces.device)
        index = start[..., None] + slot  # (envs, pairs, max_count)
        valid = slot < count[..., None]
        index = torch.where(valid, index, 0)
        f, p = forces[index], points[index]
        valid &= torch.isfinite(p).all(dim=-1)
        f = torch.where(valid[..., None], f, 0.0)
        r = torch.where(valid[..., None], p - p_tcp[:, None, None], 0.0)
        return f.sum(dim=(1, 2)), torch.cross(r, f, dim=-1).sum(dim=(1, 2))
