import math

import torch

from con_dec_imp.lie import matrix_to_rot6d, rot6d_to_matrix, so3_exp, so3_log
from con_dec_imp_sim.evaluation import (
    ARTICULATION,
    CLOSING,
    CLOSURE_DELAY_STEPS,
    CLOSURE_TIMEOUT_STEPS,
    OPEN,
    GraspClosureDetector,
    interpolate_chunk,
)


def make_chunk(num_envs: int = 2, horizon: int = 5) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    position = torch.randn(num_envs, horizon, 3, generator=gen, dtype=torch.float64)
    R = so3_exp(0.5 * torch.randn(num_envs, horizon, 3, generator=gen, dtype=torch.float64))
    gripper = torch.rand(num_envs, horizon, 1, generator=gen, dtype=torch.float64)
    return torch.cat((position, matrix_to_rot6d(R), gripper), dim=-1)


def test_interpolate_chunk_hits_waypoints():
    chunk, S = make_chunk(), 15
    refs = interpolate_chunk(chunk, S)
    assert refs.shape == (2, 4 * S, 10)
    for i in range(chunk.shape[1] - 1):
        torch.testing.assert_close(refs[:, i * S], chunk[:, i])
        torch.testing.assert_close(refs[:, i * S + S - 1], chunk[:, i + 1])


def test_interpolate_chunk_is_linear_and_geodesic():
    chunk, S = make_chunk(horizon=2), 15
    refs = interpolate_chunk(chunk, S)
    s = torch.linspace(0, 1, S, dtype=torch.float64)
    expected_p = chunk[:, :1, :3] * (1 - s[:, None]) + chunk[:, 1:, :3] * s[:, None]
    torch.testing.assert_close(refs[..., :3], expected_p)
    torch.testing.assert_close(refs[..., 9:], chunk[:, :1, 9:] * (1 - s[:, None]) + chunk[:, 1:, 9:] * s[:, None])
    # Constant angular rate along the relative rotation axis (SLERP).
    R0, R1 = rot6d_to_matrix(chunk[:, 0, 3:9]), rot6d_to_matrix(chunk[:, 1, 3:9])
    full = so3_log(R0.mT @ R1)
    for k in range(S):
        Rk = rot6d_to_matrix(refs[:, k, 3:9])
        torch.testing.assert_close(so3_log(R0.mT @ Rk), s[k] * full, atol=1e-7, rtol=0)


def run_detector(commands, fingers_closed_from=math.inf, active_until=math.inf):
    """Feed one environment ``commands``; returns (closed steps, reached steps, phases)."""
    det = GraspClosureDetector(1, "cpu")
    closed_at, reached_at, phases = [], [], []
    for t, g in enumerate(commands):
        closed, _ = det.command(torch.tensor([g]))
        if closed.item():
            closed_at.append(t)
        phases.append(int(det.phase))
        fingers = torch.tensor([t >= fingers_closed_from])
        if det.update(fingers, torch.tensor([t < active_until])).item():
            reached_at.append(t)
    return closed_at, reached_at, phases


def test_closure_after_timeout_and_delay():
    commands = [0.0] * 5 + [0.6] * 60
    closed_at, reached_at, phases = run_detector(commands)
    assert closed_at == [5]
    # Fallback: the gripper counts as closed 10 steps after the command (counted from the command step).
    assert reached_at == [5 + CLOSURE_TIMEOUT_STEPS - 1 + CLOSURE_DELAY_STEPS]
    assert phases[4] == OPEN and phases[5] == CLOSING and phases[6] == ARTICULATION


def test_closure_with_physical_closing():
    commands = [0.0] * 3 + [1.0] * 40
    _, reached_at, _ = run_detector(commands, fingers_closed_from=6)
    assert reached_at == [6 + CLOSURE_DELAY_STEPS]


def test_release_and_regrasp():
    # Hysteresis: 0.3 keeps the grasp phase; 0.2 ends it; the next close is a new grasp.
    commands = [1.0] * 30 + [0.3] * 5 + [0.2] + [1.0] * 40
    closed_at, reached_at, phases = run_detector(commands, fingers_closed_from=0)
    assert closed_at == [0, 36] and reached_at == [CLOSURE_DELAY_STEPS, 36 + CLOSURE_DELAY_STEPS]
    assert phases[34] == ARTICULATION and phases[35] == OPEN and phases[37] == ARTICULATION


def test_release_before_closure_then_regrasp():
    # The first grasp is released before grasp closure; the second one reaches it.
    commands = [1.0] * 5 + [0.0] * 2 + [1.0] * 40
    closed_at, reached_at, _ = run_detector(commands, fingers_closed_from=7)
    assert closed_at == [0, 7] and reached_at == [7 + CLOSURE_DELAY_STEPS]


def test_release_before_closure_cancels_it():
    commands = [1.0] * 5 + [0.0] * 40
    _, reached_at, _ = run_detector(commands)
    assert reached_at == []


def test_inactive_environment_does_not_reach_closure():
    _, reached_at, _ = run_detector([1.0] * 40, fingers_closed_from=0, active_until=10)
    assert reached_at == []
