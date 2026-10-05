"""Scripted expert: timing profiles, waypoint stepping and phase logic (CPU only)."""

import math
from pathlib import Path

import pytest
import torch
import yaml

from con_dec_imp_sim import OBJECTS
from con_dec_imp_sim.expert import (
    EqualProgressTiming,
    ExpertObservation,
    Phase,
    interpolate_poses,
    make_expert,
    next_waypoint,
    pose_to_action,
    s_curve,
)
from con_dec_imp_sim.expert.motion import orientation_error

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "collect"


# ---------------------------------------------------------------------- timing


def test_s_curve_endpoints_monotone_and_at_rest():
    tau = torch.linspace(-0.2, 1.2, 14001, dtype=torch.float64)
    s = s_curve(tau)
    assert s[tau <= 0].abs().max() == 0 and (s[tau >= 1] - 1).abs().max() == 0
    assert torch.all(s[1:] - s[:-1] >= 0)
    h = 1e-4
    ends = torch.tensor([0.0, h, 1.0 - h, 1.0], dtype=torch.float64)
    start_speed, end_speed = (s_curve(ends[1:2]) - s_curve(ends[:1])) / h, (s_curve(ends[3:]) - s_curve(ends[2:3])) / h
    cruise_speed = 1.0 / 0.9
    assert start_speed.abs().item() < 1e-3 * cruise_speed and end_speed.abs().item() < 1e-3 * cruise_speed
    assert abs(s_curve(torch.tensor([0.5], dtype=torch.float64)).item() - 0.5) < 1e-12


def _timing(seed, num_envs=8, num_dofs=3):
    timing = EqualProgressTiming(num_envs, num_dofs, duration=4.0)
    timing.resample(torch.Generator().manual_seed(seed))
    return timing


def test_timing_draws_are_in_range_and_seeded():
    a, b, c = _timing(3), _timing(3), _timing(4)
    assert torch.equal(a.onset, b.onset) and torch.equal(a.spread, b.spread) and torch.equal(a.curve, b.curve)
    assert not torch.equal(a.onset, c.onset)
    assert a.onset.min() >= 0.0 and a.onset.max() <= 1.0
    assert a.spread.min() >= 0.1 and a.spread.max() <= 0.3
    t = torch.full((8,), 1.3)
    assert torch.equal(a.progress(t), b.progress(t))


def test_velocity_shape_vanishes_at_both_ends():
    timing = _timing(0)
    slope = (timing.curve[:, 1:] - timing.curve[:, :-1]) / (timing.grid[1] - timing.grid[0])
    peak = slope.amax(dim=1)
    assert torch.all(slope[:, 0] < 1e-2 * peak) and torch.all(slope[:, -1] < 1e-2 * peak)


def test_equal_progress_retiming():
    timing = _timing(1)
    total = timing.total_duration
    n = timing.num_dofs
    assert torch.all(total >= 4.0 - 1e-5)
    assert torch.allclose(total, 4.0 * timing.length.double() / math.sqrt(n))

    times = torch.linspace(0.0, float(total.max()) + 1.0, 2001)
    progress = torch.stack([timing.progress(torch.full((8,), float(t))) for t in times], dim=1)  # (N, T, n)
    assert torch.all(progress[:, 0] == 0.0)
    assert torch.all(progress[:, 1:] - progress[:, :-1] >= -1e-6)  # monotone per coordinate
    for e in range(8):
        after = times > total[e].item()
        assert torch.allclose(progress[e, after], torch.ones_like(progress[e, after]))
        # constant speed sqrt(n) / T in progress space while moving
        inside = (times > 0.05 * total[e].item()) & (times < 0.95 * total[e].item())
        speed = torch.linalg.norm(progress[e, 1:] - progress[e, :-1], dim=-1) / (times[1] - times[0])
        moving = speed[inside[1:] & inside[:-1]]
        assert torch.allclose(moving.mean(), torch.tensor(math.sqrt(n) / 4.0), rtol=0.02)
        assert (moving.std() / moving.mean()) < 0.1


# ---------------------------------------------------------------------- waypoints


def _pose(position, yaw):
    yaw = torch.as_tensor(yaw, dtype=torch.float32)
    quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
    return torch.cat((torch.as_tensor(position, dtype=torch.float32), quat), -1)


def test_next_waypoint_respects_speed_limits():
    current = _pose([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], [0.0, 0.0, 0.0])
    target = _pose([[0.5, 0.0, 0.5], [0.01, 0.0, 0.5], [0.0, 0.0, 0.5]], [0.0, 0.0, 1.5])
    waypoint = next_waypoint(current, target, 0.5, 3.0, 0.1)
    step = torch.norm(waypoint[:, :3] - current[:, :3], dim=-1)
    turn = orientation_error(waypoint[:, 3:], current[:, 3:])
    assert step[0] == pytest.approx(0.05, abs=1e-5)  # linear limit
    assert torch.allclose(waypoint[1], target[1], atol=1e-6)  # within one step
    assert turn[2] == pytest.approx(0.3, abs=1e-4)  # angular limit


def test_interpolated_references_end_at_waypoint():
    start, end = _pose([[0.0, 0.0, 0.5]], [0.0]), _pose([[0.03, 0.01, 0.48]], [0.2])
    poses = interpolate_poses(start, end, 15)
    assert len(poses) == 15 and torch.allclose(poses[-1], end, atol=1e-6)
    shifted = _pose([[0.03, 0.01, 0.48]], [0.0])  # pure translation: equal steps on the chord
    first = interpolate_poses(start, shifted, 15)[0]
    assert torch.allclose(first[:, :3], start[:, :3] + (shifted[:, :3] - start[:, :3]) / 15, atol=1e-7)
    action = pose_to_action(end, torch.ones(1), origin=torch.tensor([[0.0, 0.0, 0.5]]))
    assert action.shape == (1, 10) and action[0, 2] == pytest.approx(-0.02) and action[0, 9] == 1.0


# ---------------------------------------------------------------------- phases


def _scene(name, num_envs):
    base = _pose([[0.5, 0.0, 0.5]] * num_envs, [0.1] * num_envs)
    handle = _pose([[0.38, 0.0, 0.62]] * num_envs, [0.1] * num_envs)
    tcp = _pose([[0.0, 0.0, 0.5]] * num_envs, [0.0] * num_envs)
    start = {"revolute": [0.0], "cylindrical": [0.0, 0.0], "planar": [0.15, -0.15, -1.5708], "universal": [0.0, 0.0]}[name]
    goal = {"revolute": [3.0], "cylindrical": [3.2, 0.3], "planar": [-0.15, 0.15, 1.5708], "universal": [1.6, 1.6]}[name]
    return base, handle, tcp, torch.tensor([start] * num_envs), torch.tensor([goal] * num_envs)


@pytest.mark.parametrize("name", OBJECTS)
def test_expert_runs_through_all_phases(name):
    num_envs = 2
    base, handle, tcp, joints, goal = _scene(name, num_envs)
    expert = make_expert(name, num_envs, seed=0)
    obs = ExpertObservation(tcp_pose=tcp, handle_pose=handle, base_pose=base, joint_pos=joints)
    expert.reset(obs)
    seen, grippers = [], {}
    for step in range(300):
        obs = ExpertObservation(tcp_pose=tcp.clone(), handle_pose=handle.clone(), base_pose=base, joint_pos=joints.clone())
        command = expert.command(obs)
        phase = Phase(int(expert.phase[0]))
        seen.append(phase)
        grippers.setdefault(phase, float(command.gripper[0]))
        tcp = command.waypoint  # perfect tracking
        if phase >= Phase.GRASP:
            handle = tcp.clone()  # the handle moves with the closed gripper
        if phase == Phase.ARTICULATE:
            joints = goal.clone()
        expert.advance(obs)
        if expert.done.all():
            break
    order = list(dict.fromkeys(seen))
    assert order == [Phase.APPROACH, Phase.REACH, Phase.GRASP, Phase.ARTICULATE]
    assert expert.done.all() and expert.success.all() and torch.all(expert.phase == Phase.DONE)
    assert grippers == {Phase.APPROACH: 0.0, Phase.REACH: 0.0, Phase.GRASP: 1.0, Phase.ARTICULATE: 1.0}
    # the 0.3 s grasp wait spans three to four policy periods (float32 time)
    assert 3 <= seen.count(Phase.GRASP) <= 4


def test_timeout_ends_in_error_and_environment_end_marks_episodes():
    base, handle, tcp, joints, _ = _scene("cylindrical", 3)
    expert = make_expert("cylindrical", 3)
    obs = ExpertObservation(tcp_pose=tcp, handle_pose=handle, base_pose=base, joint_pos=joints)
    expert.reset(obs)
    for _ in range(101):  # the gripper never moves: approach times out after 10 s
        expert.command(obs)
        expert.advance(obs)
    assert torch.all(expert.phase == Phase.ERROR) and expert.done.all() and not expert.success.any()
    assert torch.all(expert.command(obs).gripper == 0.0)

    expert.reset(obs)
    expert.end_episodes(torch.tensor([True, False, False]), torch.tensor([False, True, False]))
    assert expert.done.tolist() == [True, True, False] and expert.success.tolist() == [True, False, False]


@pytest.mark.parametrize("name", OBJECTS)
def test_collection_configs_build_experts(name):
    cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    assert cfg["object"] == name and cfg["num_demos"] == 400 and cfg["num_envs"] == 16
    expert = make_expert(name, cfg["num_envs"], seed=cfg["seed"], **cfg["expert"])
    assert expert.cfg.duration == 4.0 and expert.cfg.waypoint_period == 0.1


def test_universal_target_follows_the_handle_from_a_nonzero_configuration():
    # With the gripper aligned to the handle, the target orientation is the handle orientation
    # A(q) = Rx(q_x) Ry(q_y) for any configuration at the grasp, not only q_0 = 0.
    from con_dec_imp.lie import matrix_to_quat, quat_to_matrix, so3_exp, so3_log

    def handle_pose(q):
        R = so3_exp(torch.tensor([q[0], 0.0, 0.0])) @ so3_exp(torch.tensor([0.0, q[1], 0.0]))
        return torch.cat(((R @ torch.tensor([0.0, 0.0, 0.08])), matrix_to_quat(R))).view(1, 7)

    for q0 in ((0.0, 0.0), (0.3, 0.4), (-0.5, 0.2)):
        base = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        handle = handle_pose(q0)
        expert = make_expert("universal", 1)
        obs = ExpertObservation(handle, handle, base, torch.tensor([q0]))
        expert.reset(obs)
        expert._hold_grasp(torch.tensor([True]), obs)
        t = torch.tensor([2.0])
        target = expert._articulation_target(t)
        q = expert._coordinate_targets(t, torch.tensor(expert.cfg.goal, dtype=torch.float64))[0].float()
        expected = quat_to_matrix(handle_pose(q.tolist())[0, 3:])
        error = so3_log(expected.T @ quat_to_matrix(target[0, 3:])).norm()
        assert error < 1e-5, (q0, float(error))
