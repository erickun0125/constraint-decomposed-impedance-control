import pytest
import torch

from con_dec_imp import lie
from con_dec_imp.twists import body_twists, relative_twist, selected_body_twists

torch.manual_seed(0)
F64 = dict(dtype=torch.float64)
DT = 0.1


def random_trajectories(*shape):
    """Random walks ``(*shape, 4, 4)`` on SE(3) with steps of about 0.3 rad / 0.3 m."""
    steps = lie.se3_exp(0.2 * torch.randn(*shape, 6, **F64))
    poses = [lie.se3_exp(torch.randn(6, **F64)).expand(shape[:-1] + (4, 4))]
    for t in range(1, shape[-1]):
        poses.append(poses[-1] @ steps[..., t, :, :])
    return torch.stack(poses, dim=-3)


def test_constant_screw_gives_exact_twist():
    V = torch.tensor([0.3, -0.2, 0.5, 0.1, 0.4, -0.3], **F64)
    t = torch.arange(24, dtype=torch.float64)[:, None]
    poses = lie.se3_exp(torch.randn(6, **F64)) @ lie.se3_exp(t * DT * V)
    torch.testing.assert_close(body_twists(poses, DT), V.expand(24, 6), atol=1e-10, rtol=0)


def test_stencil_definition():
    poses = random_trajectories(24)
    tw = body_twists(poses, DT)
    assert tw.shape == (24, 6)

    def log_rel(a, b):
        return lie.se3_log(lie.se3_inverse(poses[a]) @ poses[b])

    torch.testing.assert_close(tw[0], log_rel(0, 1) / DT)
    torch.testing.assert_close(tw[-1], log_rel(22, 23) / DT)
    for t in (1, 7, 22):
        torch.testing.assert_close(tw[t], log_rel(t - 1, t + 1) / (2 * DT))


def test_batched_matches_single_trajectories():
    poses = random_trajectories(2, 3, 10)
    tw = body_twists(poses, DT)
    assert tw.shape == (2, 3, 10, 6)
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(tw[i, j], body_twists(poses[i, j], DT))


def test_left_invariance_and_dt_scaling():
    poses = random_trajectories(4, 12)
    A = lie.se3_exp(torch.randn(6, **F64))
    torch.testing.assert_close(body_twists(A @ poses, DT), body_twists(poses, DT))
    torch.testing.assert_close(body_twists(poses, 2 * DT), body_twists(poses, DT) / 2)


def test_two_poses_and_invalid_input():
    poses = random_trajectories(5, 2)
    tw = body_twists(poses, DT)
    assert tw.shape == (5, 2, 6)
    torch.testing.assert_close(tw[:, 0], tw[:, 1])
    torch.testing.assert_close(tw[:, 0], relative_twist(poses[:, 0], poses[:, 1], DT))
    with pytest.raises(ValueError):
        body_twists(poses[:, :1], DT)
    with pytest.raises(ValueError):
        body_twists(torch.eye(4, **F64), DT)


def test_selected_twists_match_the_stencil_of_the_kept_waypoints():
    poses = random_trajectories(5, 9)
    everything = torch.ones(5, 9, dtype=torch.bool)
    twists, valid = selected_body_twists(poses, everything, DT)
    assert valid.all()
    torch.testing.assert_close(twists, body_twists(poses, DT), rtol=0, atol=1e-12)

    keep = everything.clone()
    keep[0, :2] = False          # waypoints dropped at the start: a contiguous selection
    keep[1, 4] = False           # and in the middle
    keep[2, :] = False
    keep[2, 3] = True            # a single selected waypoint: the trajectory contributes nothing
    keep[3, :] = False           # no selected waypoint: likewise
    twists, valid = selected_body_twists(poses, keep, DT)
    assert valid.sum(-1).tolist() == [7, 8, 0, 0, 9]
    torch.testing.assert_close(twists[0, :7], body_twists(poses[0, 2:], DT))
    # Kept waypoints 0, 1, 2, 3, 5, 6, 7, 8: the twists at 3 and 5 span 3 dt (2 -> 5 and 3 -> 6).
    expected = body_twists(poses[1, keep[1]], DT)
    expected[3:5] *= 2.0 / 3.0
    torch.testing.assert_close(twists[1, :8], expected)
    torch.testing.assert_close(twists[4], body_twists(poses[4], DT))


def test_selected_twists_of_a_constant_screw_are_exact_across_gaps():
    V = torch.tensor([0.3, -0.2, 0.5, 0.1, 0.4, -0.3], **F64)
    t = torch.arange(6, dtype=torch.float64)[:, None]
    poses = lie.se3_exp(torch.randn(6, **F64)) @ lie.se3_exp(t * DT * V)
    keep = torch.tensor([True, True, False, True, True, True])
    twists, valid = selected_body_twists(poses, keep, DT)
    assert valid.tolist() == [True] * 5 + [False]
    torch.testing.assert_close(twists[valid], V.expand(5, 6), atol=1e-10, rtol=0)


def test_selected_twists_invalid_input():
    poses = random_trajectories(3, 4)
    with pytest.raises(ValueError):
        selected_body_twists(poses[:, :1], torch.ones(3, 1, dtype=torch.bool), DT)
    with pytest.raises(ValueError):
        selected_body_twists(poses, torch.ones(3, 3, dtype=torch.bool), DT)
