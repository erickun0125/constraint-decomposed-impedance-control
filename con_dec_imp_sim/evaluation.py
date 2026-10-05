"""Execution of the policy chunks and grasp-closure detection in the evaluation (Section 5.1).

Both parts are Isaac-free (torch only) and batched over environments.

Executed reference. Every replan executes the first ``executed_waypoints`` waypoints of
the deterministic chunk. Between consecutive waypoints ``w_i`` and ``w_{i+1}`` the
controller receives ``steps_per_waypoint`` references at ``s = k / (steps_per_waypoint - 1)``,
``k = 0 .. steps_per_waypoint - 1``: the position and the gripper command are linearly
interpolated and the orientation is SLERP-interpolated (:func:`interpolate_chunk`).

Grasp closure. The phase of an episode follows the desired gripper command: a command
above :data:`~con_dec_imp.policy.actions.GRIPPER_CLOSED_THRESHOLD` (0.5) closes the gripper
(approach -> grasp), the next step starts the articulation phase, which is the evaluation
window, and a command at or below :data:`GRIPPER_RELEASE_THRESHOLD` (0.25) after closing
releases it and returns the episode to the approach phase, so a later close command starts
a new grasp. The lower release threshold is a hysteresis on the phase only: the fingers
follow the command itself (they close above 0.5), and the decomposed gains act only while
the fingers are physically closed. After a close command, the gripper counts as closed when
the fingers are physically closed or after :data:`CLOSURE_TIMEOUT_STEPS` control steps,
and grasp closure is reached :data:`CLOSURE_DELAY_STEPS` control steps later, one
wrist-camera frame after the fingers closed. At that step Ours samples its ensemble and
installs its gains (Oracle installs the reference gains); all controllers record the TCP
and object poses there. A release cancels a pending grasp closure.
"""

from __future__ import annotations

import torch

from con_dec_imp.lie import matrix_to_quat, matrix_to_rot6d, quat_to_matrix, rot6d_to_matrix
from con_dec_imp.policy.actions import GRIPPER_CLOSED_THRESHOLD

GRIPPER_RELEASE_THRESHOLD = 0.25
"""Once closed, a desired gripper command at or below this value ends the grasp attempt (the
phase returns to approach); the fingers themselves open as soon as the command is at or below 0.5."""

CLOSURE_TIMEOUT_STEPS = 10
"""Control steps after the close command after which the gripper counts as closed."""

CLOSURE_DELAY_STEPS = 15
"""Control steps from the closed gripper to grasp closure (one wrist-camera frame)."""

OPEN, CLOSING, ARTICULATION = range(3)
"""Episode phases driven by the desired gripper command."""


def slerp(q0: torch.Tensor, q1: torch.Tensor, s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Spherical linear interpolation of unit quaternions ``(N, 4)`` at fractions ``s`` ``(S,)``.

    Takes the shorter arc and falls back to linear interpolation for nearly equal
    quaternions. Returns ``(N, S, 4)``.
    """
    q0 = q0 / q0.norm(dim=-1, keepdim=True)
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    theta = torch.acos(dot.abs().clamp(max=1.0))[:, None]  # (N, 1, 1)
    s = s.view(1, -1, 1)
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < eps
    safe = torch.where(small, torch.ones_like(sin_theta), sin_theta)
    w0 = torch.where(small, 1.0 - s, torch.sin((1.0 - s) * theta) / safe)
    w1 = torch.where(small, s, torch.sin(s * theta) / safe)
    q = w0 * q0[:, None] + w1 * q1[:, None]
    return q / q.norm(dim=-1, keepdim=True)


def interpolate_chunk(chunk: torch.Tensor, steps_per_waypoint: int) -> torch.Tensor:
    """Control-rate references ``(N, (H - 1) * steps_per_waypoint, 10)`` of a chunk ``(N, H, 10)``.

    Segment ``i`` holds ``steps_per_waypoint`` references from waypoint ``i`` to waypoint
    ``i + 1``, both included, so reference ``i * steps_per_waypoint`` is waypoint ``i``.
    """
    position, rot6d, gripper = chunk[..., :3], chunk[..., 3:9], chunk[..., 9:]
    quat = matrix_to_quat(rot6d_to_matrix(rot6d))
    s = torch.linspace(0.0, 1.0, steps_per_waypoint, device=chunk.device, dtype=chunk.dtype)
    s3 = s.view(1, -1, 1)
    segments = []
    for i in range(chunk.shape[1] - 1):
        p = position[:, i, None] * (1.0 - s3) + position[:, i + 1, None] * s3
        g = gripper[:, i, None] * (1.0 - s3) + gripper[:, i + 1, None] * s3
        R = quat_to_matrix(slerp(quat[:, i], quat[:, i + 1], s))
        segments.append(torch.cat((p, matrix_to_rot6d(R), g), dim=-1))
    return torch.cat(segments, dim=1)


class GraspClosureDetector:
    """Per-environment episode phase and grasp-closure detection (see the module docstring).

    Each control step, call :meth:`command` with the desired gripper command before the
    environment step and :meth:`update` with the finger state after it.
    """

    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.phase = torch.full((num_envs,), OPEN, dtype=torch.long, device=self.device)
        self._commanded_closed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._pending = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._pending_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._delay_steps = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)

    def reset(self) -> None:
        """Start new episodes in all environments."""
        self.phase.fill_(OPEN)
        self._commanded_closed.zero_()
        self._pending.zero_()
        self._pending_steps.zero_()
        self._delay_steps.fill_(-1)

    def command(self, gripper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the phases with the desired gripper command ``(N,)``.

        Returns:
            ``(closed, released)``: environments whose gripper was commanded to close or
            released at this step, each ``(N,)`` bool.
        """
        closed_now = torch.where(
            self._commanded_closed, gripper > GRIPPER_RELEASE_THRESHOLD, gripper > GRIPPER_CLOSED_THRESHOLD
        )
        closed = closed_now & ~self._commanded_closed
        released = ~closed_now & self._commanded_closed
        phase = torch.where(self.phase == CLOSING, ARTICULATION, self.phase)
        phase = torch.where(closed, CLOSING, phase)
        self.phase = torch.where(released, OPEN, phase)
        self._commanded_closed = closed_now

        self._pending |= closed
        self._pending_steps[closed] = 0
        self._pending[released] = False
        self._pending_steps[released] = 0
        self._delay_steps[released] = -1
        return closed, released

    @property
    def articulating(self) -> torch.Tensor:
        """``(N,)`` bool, True in the articulation phase (the evaluation window)."""
        return self.phase == ARTICULATION

    def update(self, fingers_closed: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Count the control step that just ran; returns the environments reaching grasp closure ``(N,)``.

        Args:
            fingers_closed: ``(N,)`` bool, fingers physically closed after the step.
            active: ``(N,)`` bool, environments whose episode is still being evaluated.
        """
        reached = torch.zeros_like(self._pending)
        if not (self._pending.any() or (self._delay_steps >= 0).any()):
            return reached
        self._pending_steps[self._pending] += 1
        ready = self._pending & (fingers_closed | (self._pending_steps >= CLOSURE_TIMEOUT_STEPS)) & active
        self._delay_steps[ready & (self._delay_steps < 0)] = 0
        counting = self._delay_steps >= 0
        reached = counting & (self._delay_steps >= CLOSURE_DELAY_STEPS)
        self._delay_steps[counting & ~reached] += 1
        self._delay_steps[reached] = -1
        reached &= active
        self._pending[reached] = False
        self._pending_steps[reached] = 0
        return reached
