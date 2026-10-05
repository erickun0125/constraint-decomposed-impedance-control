"""Scripted finite-state experts of the four simulated objects (paper, Appendix H).

From a randomized object pose the expert approaches the handle, closes the gripper on it and
drives the object's feasible coordinates to a target configuration, mapping coordinate
progress to handle poses by analytic forward kinematics of the object. The expert uses this
model knowledge only to produce demonstrations.

Every environment runs the phases ``APPROACH -> REACH -> GRASP -> ARTICULATE -> DONE``; a
phase that exceeds its timeout ends the episode in ``ERROR``. Each policy period (0.1 s) the
collection loop calls::

    command = expert.command(obs)      # target pose, next waypoint, gripper command
    ...                                # track ScLERP(tcp_pose -> command.waypoint) at 150 Hz,
    expert.end_episodes(terminated, truncated)   # after every simulator step
    expert.advance(obs)                # same observation as command(): time and phase update

after ``expert.reset(obs)`` with the first observation of an episode batch. All poses are
``(N, 7)`` world-frame tensors ``(position, quaternion (w, x, y, z))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from con_dec_imp_sim.expert.motion import (
    next_waypoint,
    orientation_error,
    quat_about_axis,
    quat_conjugate,
    quat_mul,
    rotate_vector,
)
from con_dec_imp_sim.expert.timing import EqualProgressTiming, s_curve


class Phase(IntEnum):
    """Expert phases, in order."""

    APPROACH = 0
    """Move in front of the handle, gripper open."""
    REACH = 1
    """Move onto the handle, gripper open."""
    GRASP = 2
    """Close the gripper and wait."""
    ARTICULATE = 3
    """Drive the object coordinates along their timing profile, gripper closed."""
    DONE = 4
    """Target configuration reached; hold."""
    ERROR = 5
    """A phase timed out; hold with the gripper open."""


@dataclass
class ExpertObservation:
    """World-frame state the expert reads, one row per environment."""

    tcp_pose: torch.Tensor
    """Gripper control point (TCP) pose ``(N, 7)``."""
    handle_pose: torch.Tensor
    """Handle frame pose ``(N, 7)``."""
    base_pose: torch.Tensor
    """Object root pose ``(N, 7)``."""
    joint_pos: torch.Tensor
    """Object coordinates ``(N, n)`` in the order of the expert's ``coordinates``."""


@dataclass
class ExpertCommand:
    """Output of :meth:`ScriptedExpert.command`."""

    target: torch.Tensor
    """Pose the current phase moves to ``(N, 7)``."""
    waypoint: torch.Tensor
    """Next waypoint ``(N, 7)``: one policy period toward ``target``, speed-limited."""
    gripper: torch.Tensor
    """Gripper command ``(N,)``, 1 = close, 0 = open."""


@dataclass
class ExpertConfig:
    """Settings shared by all experts."""

    waypoint_period: float = 0.1
    """Policy period between waypoints [s]."""
    max_linear_speed: float = 0.5
    """Speed limit of the waypoint stepping [m/s]."""
    max_angular_speed: float = 3.0
    """Angular speed limit of the waypoint stepping [rad/s]."""
    approach_distance: float = 0.08
    """Stand-off of the approach pose in front of the handle [m]."""
    grasp_wait: float = 0.3
    """Time the gripper closes before the articulation starts [s]."""
    approach_tolerance: tuple[float, float] = (0.03, 0.15)
    """Position [m] and orientation [rad] errors that complete the approach."""
    reach_tolerance: tuple[float, float] = (0.015, 0.08)
    """Position [m] and orientation [rad] errors to the handle that complete the reach."""
    timeouts: tuple[float, float, float, float] = (10.0, 10.0, 3.0, 9.0)
    """Timeouts of the approach, reach, grasp and articulation phases [s]."""
    duration: float = 4.0
    """Articulation time T of the synchronous profile [s]."""


class ScriptedExpert:
    """Finite-state expert, vectorized over environments; subclasses define the object."""

    coordinates: tuple[str, ...] = ()
    """Names of the operated object coordinates (columns of ``ExpertObservation.joint_pos``)."""
    config_class = ExpertConfig

    def __init__(self, num_envs: int, config: ExpertConfig | None = None, device: str = "cpu", seed: int = 0):
        self.cfg = config or self.config_class()
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.generator = torch.Generator().manual_seed(seed)
        n = num_envs
        self.phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self.time = torch.zeros(n, device=self.device)
        self.phase_start = torch.zeros(n, device=self.device)
        self.done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.grasp_pose = torch.zeros(n, 7, device=self.device)
        self.initial_joint_pos = torch.zeros(n, len(self.coordinates), device=self.device)
        self.base_pose = torch.zeros(n, 7, device=self.device)
        self.base_yaw = torch.zeros(n, device=self.device)
        self._timeouts = torch.tensor((*self.cfg.timeouts, math.inf, math.inf), device=self.device)

    # ------------------------------------------------------------------ interface

    def reset(self, obs: ExpertObservation) -> None:
        """Start new episodes in all environments from their first observation."""
        self.phase.fill_(Phase.APPROACH)
        self.time.zero_()
        self.phase_start.zero_()
        self.done.zero_()
        self.success.zero_()
        self.grasp_pose.zero_()
        self.grasp_pose[:, 3] = 1.0
        self.initial_joint_pos.zero_()
        self.base_pose = obs.base_pose.clone()
        w, x, y, z = self.base_pose[:, 3:7].unbind(-1)
        self.base_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self._start_episode(obs)

    def command(self, obs: ExpertObservation) -> ExpertCommand:
        """Phase target, next waypoint and gripper command for the coming policy period."""
        phase = self.phase
        reach = phase == Phase.REACH
        if reach.any():
            self._hold_grasp(reach, obs)
        candidates = {
            Phase.APPROACH: self._approach_target(obs),
            Phase.REACH: self._reach_target(obs),
            Phase.GRASP: self.grasp_pose,
            Phase.ARTICULATE: self._articulation_target(self.time - self.phase_start),
        }
        target = obs.tcp_pose.clone()
        for p, pose in candidates.items():
            target = torch.where((phase == p)[:, None], pose, target)
        gripper = ((phase == Phase.GRASP) | (phase == Phase.ARTICULATE) | (phase == Phase.DONE)).float()
        waypoint = next_waypoint(
            obs.tcp_pose, target, self.cfg.max_linear_speed, self.cfg.max_angular_speed, self.cfg.waypoint_period
        )
        return ExpertCommand(target=target, waypoint=waypoint, gripper=gripper)

    def advance(self, obs: ExpertObservation) -> None:
        """Advance time by one policy period and take the phase transitions that ``obs`` allows."""
        self.time += self.cfg.waypoint_period
        state_time = self.time - self.phase_start
        phase = self.phase
        passed = (phase == Phase.APPROACH) & self._within(obs.tcp_pose, self._approach_target(obs), self.cfg.approach_tolerance)
        passed |= (phase == Phase.REACH) & self._reach_complete(obs, state_time)
        passed |= (phase == Phase.GRASP) & (state_time.double() > self.cfg.grasp_wait)
        passed |= (phase == Phase.ARTICULATE) & self._articulated(obs)
        timed_out = ~passed & (state_time > self._timeouts[phase])
        active = ~self.done & (phase < Phase.DONE)
        moved, failed = active & passed, active & timed_out
        finished = moved & (phase == Phase.ARTICULATE)
        self.phase = torch.where(failed, torch.full_like(phase, Phase.ERROR), torch.where(moved, phase + 1, phase))
        self.phase_start = torch.where(moved | failed, self.time, self.phase_start)
        self.success |= finished
        self.done |= finished | failed

    def end_episodes(self, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        """Mark episodes ended by the environment: success on termination, failure on time-out."""
        ended = (terminated | truncated) & ~self.done
        self.done |= ended
        self.success |= terminated & ended

    # ------------------------------------------------------------------ object definition

    def _start_episode(self, obs: ExpertObservation) -> None:
        """Per-episode object setup (geometry, timing draws)."""

    def _approach_target(self, obs: ExpertObservation) -> torch.Tensor:
        raise NotImplementedError

    def _reach_target(self, obs: ExpertObservation) -> torch.Tensor:
        return obs.handle_pose

    def _hold_grasp(self, mask: torch.Tensor, obs: ExpertObservation) -> None:
        """Remember the handle pose and object coordinates while reaching."""
        self.grasp_pose[mask] = obs.handle_pose[mask]
        self.initial_joint_pos[mask] = obs.joint_pos[mask]

    def _reach_complete(self, obs: ExpertObservation, state_time: torch.Tensor) -> torch.Tensor:
        return self._within(obs.tcp_pose, obs.handle_pose, self.cfg.reach_tolerance)

    def _articulation_target(self, state_time: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _articulated(self, obs: ExpertObservation) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _within(pose: torch.Tensor, target: torch.Tensor, tolerance: tuple[float, float]) -> torch.Tensor:
        position_error = torch.norm(pose[:, :3] - target[:, :3], dim=-1)
        rotation_error = orientation_error(pose[:, 3:7], target[:, 3:7])
        return (position_error.double() < tolerance[0]) & (rotation_error.double() < tolerance[1])

    def _front_direction(self) -> torch.Tensor:
        """Unit vector ``(N, 2)`` from the object toward the gripper side (object -x axis)."""
        yaw = self.base_yaw.double()
        return torch.stack((-torch.cos(yaw), -torch.sin(yaw)), dim=-1).float()

    def _offset_xy(self, pose: torch.Tensor, direction: torch.Tensor, distance: float) -> torch.Tensor:
        target = pose.clone()
        target[:, :2] = pose[:, :2] + direction * distance
        return target

    def _yawed(self, local_x: torch.Tensor, local_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World xy of a float64 offset in the yawed object frame, added to the object root."""
        yaw = self.base_yaw.double()
        c, s = torch.cos(yaw), torch.sin(yaw)
        x = self.base_pose[:, 0] + (c * local_x).float() - (s * local_y).float()
        y = self.base_pose[:, 1] + (s * local_x).float() + (c * local_y).float()
        return x, y


# ---------------------------------------------------------------------- revolute


@dataclass
class RevoluteConfig(ExpertConfig):
    """Door on a vertical hinge, opened by ``open_ratio * open_target`` on one S-curve profile."""

    open_target: float = 3.14159
    """Nominal hinge rotation [rad]."""
    open_ratio: float = 0.95
    """Fraction of ``open_target`` the expert commands."""
    completion_margin: float = 0.31
    """The articulation completes within this angle of the commanded opening [rad]."""
    hinge_offset: tuple[float, float] = (-0.21, -0.15)
    """Hinge position assumed by the expert in the object root frame (x, y) [m]; the asset's hinge
    is at (-0.215, -0.15). The demonstrations of the released policies were collected with this
    value; the 5 mm offset is absorbed by the compliant grasp."""
    timeouts: tuple[float, float, float, float] = (10.0, 6.0, 2.0, 8.0)


class RevoluteExpert(ScriptedExpert):
    """Pulls the handle along the arc about the hinge; the gripper turns with the door."""

    coordinates = ("hinge",)
    config_class = RevoluteConfig

    def _start_episode(self, obs: ExpertObservation) -> None:
        hinge_local = torch.zeros(self.num_envs, 3, device=self.device)
        hinge_local[:, 0], hinge_local[:, 1] = self.cfg.hinge_offset
        self.hinge = self.base_pose[:, :3] + rotate_vector(self.base_pose[:, 3:7], hinge_local)
        self.radius = torch.zeros(self.num_envs, device=self.device)
        self.initial_angle = torch.zeros(self.num_envs, device=self.device)

    def _approach_target(self, obs: ExpertObservation) -> torch.Tensor:
        offset = torch.zeros(self.num_envs, 3, device=self.device)
        offset[:, 0] = -self.cfg.approach_distance
        handle = obs.handle_pose
        return torch.cat((handle[:, :3] + rotate_vector(handle[:, 3:7], offset), handle[:, 3:7]), dim=-1)

    def _hold_grasp(self, mask: torch.Tensor, obs: ExpertObservation) -> None:
        super()._hold_grasp(mask, obs)
        arm = obs.handle_pose[:, :2] - self.hinge[:, :2]
        self.radius = torch.where(mask, torch.norm(arm, dim=-1), self.radius)
        self.initial_angle = torch.where(mask, torch.atan2(arm[:, 1], arm[:, 0]), self.initial_angle)

    def _articulation_target(self, state_time: torch.Tensor) -> torch.Tensor:
        progress = s_curve(state_time / self.cfg.duration) * self.cfg.open_ratio
        angle = self.cfg.open_target * progress
        heading = self.initial_angle + angle
        position = torch.stack(
            (
                self.hinge[:, 0] + self.radius * torch.cos(heading),
                self.hinge[:, 1] + self.radius * torch.sin(heading),
                self.grasp_pose[:, 2],
            ),
            dim=-1,
        )
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        return torch.cat((position, quat_mul(quat_about_axis(angle, z_axis), self.grasp_pose[:, 3:7])), dim=-1)

    def _articulated(self, obs: ExpertObservation) -> torch.Tensor:
        cfg = self.cfg
        return obs.joint_pos[:, 0] >= cfg.open_target * cfg.open_ratio - cfg.completion_margin


# ---------------------------------------------------------------------- multi-coordinate objects


@dataclass
class MultiCoordinateConfig(ExpertConfig):
    """Settings of the per-coordinate timing (paper, Appendix H)."""

    onset_range: tuple[float, float] = (0.0, 1.0)
    """Range of the per-coordinate onset phi_j (normalized time)."""
    spread_range: tuple[float, float] = (0.1, 0.3)
    """Range of the per-coordinate spread sigma_j (normalized time)."""


class MultiCoordinateExpert(ScriptedExpert):
    """Expert whose coordinates follow independent, equal-progress retimed profiles."""

    def __init__(self, num_envs: int, config: MultiCoordinateConfig | None = None, device: str = "cpu", seed: int = 0):
        super().__init__(num_envs, config, device, seed)
        self.timing = EqualProgressTiming(
            num_envs, len(self.coordinates), self.cfg.duration, self.cfg.onset_range, self.cfg.spread_range,
            device=self.device,
        )

    def _start_episode(self, obs: ExpertObservation) -> None:
        self.timing.resample(self.generator)

    def _coordinate_targets(self, state_time: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Float64 coordinates ``(N, n)`` on the way from the grasp configuration to ``goal``."""
        start = self.initial_joint_pos.double()
        return start + self.timing.progress(state_time).double() * (goal - start)


@dataclass
class CylindricalConfig(MultiCoordinateConfig):
    """Handle on a vertical rotating slider: rotation and translation along the same axis."""

    rotation: float = 3.14159
    """Commanded rotation [rad]."""
    translation: float = 0.30
    """Commanded translation [m]."""
    completion_margin: tuple[float, float] = (0.31, 0.03)
    """The articulation completes within these margins of the commanded rotation [rad] and translation [m]."""
    handle_offset: tuple[float, float] = (-0.12, 0.0)
    """Handle position in the rotating frame (x, y) [m]."""
    handle_height: float = 0.125
    """Handle height above the object root at zero translation [m]."""


class CylindricalExpert(MultiCoordinateExpert):
    """Turns and lifts the handle; coordinates (rotation about z, translation along z)."""

    coordinates = ("rotation", "translation")
    config_class = CylindricalConfig

    def _approach_target(self, obs: ExpertObservation) -> torch.Tensor:
        radial = obs.handle_pose[:, :2] - self.base_pose[:, :2]
        distance = torch.norm(radial, dim=-1, keepdim=True)
        fallback = torch.tensor([-1.0, 0.0], device=self.device).expand_as(radial)
        direction = torch.where(distance > 1e-6, radial / distance, fallback)
        return self._offset_xy(obs.handle_pose, direction, self.cfg.approach_distance)

    def _articulation_target(self, state_time: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        goal = self.initial_joint_pos.double() + torch.tensor([cfg.rotation, cfg.translation], device=self.device, dtype=torch.float64)
        q = self._coordinate_targets(state_time, goal)
        angle = self.base_yaw.double() + q[:, 0]
        hx, hy = cfg.handle_offset
        c, s = torch.cos(angle), torch.sin(angle)
        position = torch.stack(
            (
                self.base_pose[:, 0] + (hx * c).float() - (hy * s).float(),
                self.base_pose[:, 1] + (hx * s).float() + (hy * c).float(),
                self.base_pose[:, 2] + cfg.handle_height + q[:, 1].float(),
            ),
            dim=-1,
        )
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float64).expand(self.num_envs, 3)
        turn = quat_about_axis(q[:, 0] - self.initial_joint_pos[:, 0].double(), z_axis)
        return torch.cat((position, quat_mul(turn, self.grasp_pose[:, 3:7])), dim=-1)

    def _articulated(self, obs: ExpertObservation) -> torch.Tensor:
        cfg = self.cfg
        goal = self.initial_joint_pos.double() + torch.tensor([cfg.rotation, cfg.translation], device=self.device, dtype=torch.float64)
        margin = torch.tensor(cfg.completion_margin, device=self.device, dtype=torch.float64)
        return (obs.joint_pos.double() >= goal - margin).all(dim=-1)


@dataclass
class PlanarConfig(MultiCoordinateConfig):
    """Handle on a vertical plane: two in-plane translations and the rotation about the plane normal."""

    goal: tuple[float, float, float] = (-0.15, 0.15, 1.5708)
    """Absolute goal of (translation y [m], translation z [m], rotation x [rad])."""
    completion_margin: tuple[float, float, float] = (0.03, 0.03, 0.31)
    """The articulation completes within these distances of ``goal``."""
    pre_grasp_distance: float = 0.02
    """Stand-off of the reach pose in front of the handle [m]; the grasp closes it."""
    reach_tolerance: tuple[float, float] = (0.04, 0.15)
    reach_time: float = 3.0
    """The reach also completes after this time [s] (the free handle yields to the gripper)."""
    handle_x: float = -0.065
    """Handle position along the object x axis (plane normal) [m]."""
    timeouts: tuple[float, float, float, float] = (15.0, 20.0, 3.0, 19.0)


class PlanarExpert(MultiCoordinateExpert):
    """Slides and turns the handle in the plane; coordinates (translation y, translation z, rotation x)."""

    coordinates = ("translation_y", "translation_z", "rotation_x")
    config_class = PlanarConfig

    def _approach_target(self, obs: ExpertObservation) -> torch.Tensor:
        return self._offset_xy(obs.handle_pose, self._front_direction(), self.cfg.approach_distance)

    def _reach_target(self, obs: ExpertObservation) -> torch.Tensor:
        return self._offset_xy(obs.handle_pose, self._front_direction(), self.cfg.pre_grasp_distance)

    def _reach_complete(self, obs: ExpertObservation, state_time: torch.Tensor) -> torch.Tensor:
        return super()._reach_complete(obs, state_time) | (state_time.double() > self.cfg.reach_time)

    def _articulation_target(self, state_time: torch.Tensor) -> torch.Tensor:
        goal = torch.tensor(self.cfg.goal, device=self.device, dtype=torch.float64)
        q = self._coordinate_targets(state_time, goal)
        x, y = self._yawed(torch.full_like(q[:, 0], self.cfg.handle_x), q[:, 0])
        position = torch.stack((x, y, self.base_pose[:, 2] + q[:, 1].float()), dim=-1)
        yaw = self.base_yaw.double()
        normal = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=-1)
        turn = quat_about_axis(q[:, 2] - self.initial_joint_pos[:, 2].double(), normal)
        return torch.cat((position, quat_mul(turn, self.grasp_pose[:, 3:7])), dim=-1)

    def _articulated(self, obs: ExpertObservation) -> torch.Tensor:
        goal = torch.tensor(self.cfg.goal, device=self.device, dtype=torch.float64)
        margin = torch.tensor(self.cfg.completion_margin, device=self.device, dtype=torch.float64)
        return ((obs.joint_pos.double() - goal).abs() < margin).all(dim=-1)


@dataclass
class UniversalConfig(MultiCoordinateConfig):
    """Handle on a universal joint: rotations about the root x axis, then the rotated y axis."""

    goal: tuple[float, float] = (1.5708, 1.5708)
    """Absolute goal of (rotation x, rotation y) [rad]."""
    completion_margin: float = 0.16
    """The articulation completes when both rotations are within this angle of ``goal`` [rad]."""
    reach_tolerance: tuple[float, float] = (0.02, 0.10)
    handle_length: float = 0.08
    """Distance of the handle from the joint center [m]."""


class UniversalExpert(MultiCoordinateExpert):
    """Swings the handle about both joint axes; coordinates (rotation x, rotation y)."""

    coordinates = ("rotation_x", "rotation_y")
    config_class = UniversalConfig

    def _approach_target(self, obs: ExpertObservation) -> torch.Tensor:
        return self._offset_xy(obs.handle_pose, self._front_direction(), self.cfg.approach_distance)

    def _articulation_target(self, state_time: torch.Tensor) -> torch.Tensor:
        goal = torch.tensor(self.cfg.goal, device=self.device, dtype=torch.float64)
        q = self._coordinate_targets(state_time, goal)
        h = self.cfg.handle_length
        rx, ry = q[:, 0], q[:, 1]
        x, y = self._yawed(h * torch.sin(ry), (-h) * torch.sin(rx) * torch.cos(ry))
        z = self.base_pose[:, 2] + (h * torch.cos(rx) * torch.cos(ry)).float()
        # The handle turns by A(q) = Rx(q_x) Ry(q_y); the gripper, grasped at q_0, follows it with
        # the relative rotation A(q_0)^-1 A(q) expressed in its own frame at the grasp.
        q0 = self.initial_joint_pos.double()
        ones, zeros = torch.ones_like(rx), torch.zeros_like(rx)
        x_axis, y_axis = torch.stack((ones, zeros, zeros), dim=-1), torch.stack((zeros, ones, zeros), dim=-1)
        start = quat_mul(quat_about_axis(q0[:, 0], x_axis), quat_about_axis(q0[:, 1], y_axis))
        now = quat_mul(quat_about_axis(rx, x_axis), quat_about_axis(ry, y_axis))
        turn = quat_mul(quat_conjugate(start), now)
        return torch.cat((torch.stack((x, y, z), dim=-1), quat_mul(self.grasp_pose[:, 3:7], turn)), dim=-1)

    def _articulated(self, obs: ExpertObservation) -> torch.Tensor:
        goal = torch.tensor(self.cfg.goal, device=self.device, dtype=torch.float64)
        return (obs.joint_pos.double() >= goal - self.cfg.completion_margin).all(dim=-1)


EXPERTS: dict[str, type[ScriptedExpert]] = {
    "revolute": RevoluteExpert,
    "cylindrical": CylindricalExpert,
    "planar": PlanarExpert,
    "universal": UniversalExpert,
}
"""Expert class of each simulated object."""


def make_expert(name: str, num_envs: int, device: str = "cpu", seed: int = 0, **settings) -> ScriptedExpert:
    """Expert of object ``name`` with config values ``settings`` (e.g. the ``expert`` section of configs/collect)."""
    cls = EXPERTS[name]
    settings = {k: tuple(v) if isinstance(v, list) else v for k, v in settings.items()}
    return cls(num_envs, cls.config_class(**settings), device=device, seed=seed)
