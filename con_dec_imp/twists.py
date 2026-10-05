"""Body twists of pose trajectories by finite differences on SE(3) (Algorithm 1, line 3).

For a trajectory ``(T_1, ..., T_H)`` sampled at interval ``dt``, the body twist at
each waypoint is

    V_t = log(T_{t-1}^{-1} T_{t+1})^vee / (2 dt)      for 1 < t < H   (central),
    V_1 = log(T_1^{-1} T_2)^vee / dt                                  (forward),
    V_H = log(T_{H-1}^{-1} T_H)^vee / dt                              (backward),

so a trajectory of ``H`` poses yields ``H`` twists, ordered ``(omega, v)``.
The relative transform ``T_a^{-1} T_b`` is invariant to a common left transform of
the trajectory, so poses may be given in any fixed reference frame.
"""

from __future__ import annotations

import torch

from con_dec_imp.lie import se3_inverse, se3_log


def relative_twist(T_a: torch.Tensor, T_b: torch.Tensor, dt: float) -> torch.Tensor:
    """Constant body twist that moves ``T_a`` to ``T_b`` in time ``dt``: ``log(T_a^{-1} T_b)^vee / dt``."""
    batch = T_a.shape[:-2]
    T_rel = torch.bmm(se3_inverse(T_a.reshape(-1, 4, 4)), T_b.reshape(-1, 4, 4))
    return (se3_log(T_rel) / dt).reshape(batch + (6,))


def _num_waypoints(poses: torch.Tensor) -> int:
    """Number of waypoints ``H`` of pose trajectories ``(..., H, 4, 4)``; checks ``H >= 2``."""
    if poses.shape[-2:] != (4, 4) or poses.dim() < 3:
        raise ValueError(f"poses must have shape (..., H, 4, 4); got {tuple(poses.shape)}")
    H = poses.shape[-3]
    if H < 2:
        raise ValueError(f"a trajectory needs at least two poses; got H={H}")
    return H


def body_twists(poses: torch.Tensor, dt: float) -> torch.Tensor:
    """Per-waypoint body twists of pose trajectories.

    Args:
        poses: Homogeneous transforms ``(..., H, 4, 4)`` with ``H >= 2``.
        dt: Time between consecutive waypoints [s].

    Returns:
        Body twists ``(..., H, 6)`` ordered ``(omega, v)``: central differences inside
        the trajectory and one-sided differences at both ends. For ``H = 2`` the two
        one-sided twists coincide.
    """
    H = _num_waypoints(poses)
    forward = relative_twist(poses[..., 0:1, :, :], poses[..., 1:2, :, :], dt)
    central = relative_twist(poses[..., : H - 2, :, :], poses[..., 2:H, :, :], 2.0 * dt)
    backward = relative_twist(poses[..., H - 2 : H - 1, :, :], poses[..., H - 1 : H, :, :], dt)
    return torch.cat((forward, central, backward), dim=-2)


def selected_body_twists(
    poses: torch.Tensor, selected: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Body twists of the selected waypoints of each trajectory.

    The stencil of :func:`body_twists` is applied to the selected waypoints of a
    trajectory, taken in order: the twist at a selected waypoint is the relative motion
    between its neighbouring selected waypoints ``i_a < i_b`` (original indices) divided
    by the time between them, ``(i_b - i_a) dt``. Where unselected waypoints lie between
    two selected ones, this is the true time span, so a constant-velocity trajectory gives
    its exact twist at every selected waypoint. For a contiguous selection the result
    equals :func:`body_twists` of the selected waypoints. A trajectory with fewer than two
    selected waypoints contributes no twist.

    Args:
        poses: Homogeneous transforms ``(..., H, 4, 4)`` with ``H >= 2``.
        selected: Boolean mask ``(..., H)`` of the waypoints to use.
        dt: Time between consecutive waypoints [s].

    Returns:
        ``(twists, valid)``: twists ``(..., H, 6)`` of the selected waypoints, moved to the
        front of each trajectory, and the mask ``(..., H)`` of the valid entries.
    """
    H = _num_waypoints(poses)
    if selected.shape != poses.shape[:-2]:
        raise ValueError(f"selected must have shape {tuple(poses.shape[:-2])}; got {tuple(selected.shape)}")
    usable = selected.sum(-1, keepdim=True) >= 2
    # Trajectories without two selected waypoints are differentiated whole and then masked out.
    selected = selected | ~usable
    # order[..., k]: original index of the k-th selected waypoint (for k < length).
    order = torch.argsort((~selected).to(torch.int8), dim=-1, stable=True)
    compact = torch.gather(poses, -3, order[..., None, None].expand(poses.shape))
    length = selected.sum(-1, keepdim=True)
    j = torch.arange(H, device=poses.device).expand(selected.shape)
    # Neighbours of waypoint j in the compacted trajectory: forward difference at the first,
    # backward difference at the last, central differences in between.
    prev = torch.where(j >= length - 1, length - 2, (j - 1).clamp(min=0))
    nxt = torch.minimum(j + 1, length - 1)

    def take(index: torch.Tensor) -> torch.Tensor:
        return torch.gather(compact, -3, index[..., None, None].expand(poses.shape))

    span = (order.gather(-1, nxt) - order.gather(-1, prev)).to(poses.dtype) * dt
    twists = relative_twist(take(prev), take(nxt), 1.0) / span[..., None]
    return twists, (j < length) & usable
