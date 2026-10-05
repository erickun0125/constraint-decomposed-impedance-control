import torch

from con_dec_imp import lie
from con_dec_imp.policy.actions import (
    ACTION_DIM,
    absolute_to_relative,
    chunk_to_transforms,
    relative_to_absolute,
)

torch.manual_seed(0)
F64 = dict(dtype=torch.float64)


def random_chunk(batch, H):
    R = lie.so3_exp(torch.randn(batch, H, 3, **F64))
    return torch.cat(
        (torch.randn(batch, H, 3, **F64), lie.matrix_to_rot6d(R), torch.rand(batch, H, 1, **F64)), dim=-1
    )


def random_anchor(batch):
    return random_chunk(batch, 1)[:, 0, :9]


def test_action_dim():
    assert random_chunk(1, 1).shape[-1] == ACTION_DIM == 10


def test_relative_absolute_round_trip():
    chunk, anchor = random_chunk(4, 24), random_anchor(4)
    torch.testing.assert_close(relative_to_absolute(absolute_to_relative(chunk, anchor), anchor), chunk)
    torch.testing.assert_close(absolute_to_relative(relative_to_absolute(chunk, anchor), anchor), chunk)


def test_anchor_pose_maps_to_identity_and_gripper_is_unchanged():
    chunk, anchor = random_chunk(3, 5), random_anchor(3)
    chunk[:, 0, :9] = anchor
    rel = absolute_to_relative(chunk, anchor)
    torch.testing.assert_close(rel[:, 0, :3], torch.zeros(3, 3, **F64), atol=1e-12, rtol=0)
    torch.testing.assert_close(lie.rot6d_to_matrix(rel[:, 0, 3:9]), torch.eye(3, **F64).expand(3, 3, 3))
    torch.testing.assert_close(rel[..., 9], chunk[..., 9])


def test_relative_chunk_matches_relative_transforms():
    chunk, anchor = random_chunk(2, 6), random_anchor(2)
    T_c = lie.make_transform(anchor[:, :3], lie.rot6d_to_matrix(anchor[:, 3:9]))
    expected = lie.se3_inverse(T_c)[:, None] @ chunk_to_transforms(chunk)
    torch.testing.assert_close(chunk_to_transforms(absolute_to_relative(chunk, anchor)), expected)


def test_relative_chunk_invariant_to_common_transform():
    chunk, anchor = random_chunk(2, 6), random_anchor(2)
    A = lie.se3_exp(torch.randn(6, **F64))

    def moved(x):
        T = A @ lie.make_transform(x[..., :3], lie.rot6d_to_matrix(x[..., 3:9]))
        return torch.cat((T[..., :3, 3], lie.matrix_to_rot6d(T[..., :3, :3]), x[..., 9:]), dim=-1)

    rel = absolute_to_relative(chunk, anchor)
    rel_moved = absolute_to_relative(moved(chunk), moved(anchor))
    torch.testing.assert_close(rel_moved, rel)


def test_chunk_to_transforms():
    chunk = random_chunk(2, 7)
    T = chunk_to_transforms(chunk)
    assert T.shape == (2, 7, 4, 4)
    torch.testing.assert_close(T[..., :3, 3], chunk[..., :3])
    torch.testing.assert_close(lie.matrix_to_rot6d(T[..., :3, :3]), chunk[..., 3:9])
