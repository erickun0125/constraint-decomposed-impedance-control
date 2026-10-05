"""Body-frame tracking errors and the impedance feedback wrench (Eq. 2 and 9).

    e_T = [ log(R_b^T R_d)^vee ; R_b^T (p_d - p_b) ],
    e_V = Ad_{T_b^{-1} T_d} V_d - V_b,
    F_fb = K_v e_T + D_v e_V.

All quantities are expressed in the current body (control) frame ``{b}``; twists
are ``(omega, v)`` and wrenches ``(moment, force)``.
"""

from __future__ import annotations

import torch

from con_dec_imp.lie import se3_inverse, so3_log, transform_twist


def pose_error(T_b: torch.Tensor, T_d: torch.Tensor) -> torch.Tensor:
    """Double-geodesic pose error ``e_T`` of the desired pose ``T_d`` seen from ``T_b``."""
    R_bt = T_b[..., :3, :3].transpose(-1, -2)
    e_rot = so3_log(R_bt @ T_d[..., :3, :3])
    e_pos = (R_bt @ (T_d[..., :3, 3] - T_b[..., :3, 3])[..., None])[..., 0]
    return torch.cat((e_rot, e_pos), dim=-1)


def twist_error(
    T_b: torch.Tensor, T_d: torch.Tensor, V_d: torch.Tensor, V_b: torch.Tensor
) -> torch.Tensor:
    """Twist error ``e_V``: the desired body twist ``V_d`` transported into ``{b}``, minus ``V_b``."""
    return transform_twist(se3_inverse(T_b) @ T_d, V_d) - V_b


def feedback_wrench(
    K: torch.Tensor, D: torch.Tensor, e_T: torch.Tensor, e_V: torch.Tensor
) -> torch.Tensor:
    """Impedance feedback ``F_fb = K e_T + D e_V`` for batched gains ``(..., 6, 6)``."""
    return (K @ e_T[..., None])[..., 0] + (D @ e_V[..., None])[..., 0]
