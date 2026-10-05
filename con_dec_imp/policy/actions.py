"""Pose-chunk representation of the policy actions.

An action step is a 10-vector ``(position[3], rot6d[6], gripper[1])``. The policy is
trained on *relative* chunks: every pose of an H-step chunk is expressed in the
end-effector frame at the observation time (the chunk anchor), ``p_rel = R_c^T (p_t - p_c)``
and ``R_rel = R_c^T R_t``. The gripper command is not transformed.
"""

from __future__ import annotations

import torch

from con_dec_imp.lie import make_transform, matrix_to_rot6d, rot6d_to_matrix

ACTION_DIM = 10
"""Position (3) + 6-D rotation (6) + gripper command (1)."""

GRIPPER_CLOSED_THRESHOLD = 0.5
"""A gripper command above this value closes the gripper."""


def absolute_to_relative(chunk: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Express an absolute chunk ``(..., H, 10)`` in the anchor frame ``(..., 9)`` (pos + rot6d)."""
    p_c, R_c = anchor[..., :3], rot6d_to_matrix(anchor[..., 3:9])
    p_rel = torch.einsum("...ji,...hj->...hi", R_c, chunk[..., :3] - p_c.unsqueeze(-2))
    R_rel = torch.einsum("...ji,...hjk->...hik", R_c, rot6d_to_matrix(chunk[..., 3:9]))
    return torch.cat((p_rel, matrix_to_rot6d(R_rel), chunk[..., 9:10]), dim=-1)


def relative_to_absolute(chunk: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`absolute_to_relative`."""
    p_c, R_c = anchor[..., :3], rot6d_to_matrix(anchor[..., 3:9])
    p_abs = torch.einsum("...ij,...hj->...hi", R_c, chunk[..., :3]) + p_c.unsqueeze(-2)
    R_abs = torch.einsum("...ij,...hjk->...hik", R_c, rot6d_to_matrix(chunk[..., 3:9]))
    return torch.cat((p_abs, matrix_to_rot6d(R_abs), chunk[..., 9:10]), dim=-1)


def chunk_to_transforms(chunk: torch.Tensor) -> torch.Tensor:
    """Homogeneous transforms ``(..., H, 4, 4)`` of the poses in a chunk ``(..., H, 10)``."""
    return make_transform(chunk[..., :3], rot6d_to_matrix(chunk[..., 3:9]))
