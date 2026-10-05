"""Reference feasible twist subspaces of the four simulated objects (Section 5, Appendix H).

The reference subspace of a rollout is the analytic feasible subspace of the mechanism
at the grasp pose of that rollout: each joint contributes a screw generator, written in
the object root frame and mapped to the TCP body frame at the measured TCP and object
poses. A rotation about axis ``a`` through point ``q`` gives the body twist
``(R^T a, R^T (a x (p - q)))`` and a translation along ``a`` gives ``(0, R^T a)``, where
``(R, p)`` is the TCP pose. The axis directions are fixed in the object root frame, and
where an axis moves with the configuration (the planar rotation) the spanned subspace does
not change, so the basis needs no joint positions. The universal
joint uses its global envelope, the three rotations about the joint centre.

Also defines the fixed evaluation length ``alpha*`` per object used by the subspace
error (SE): it is set by the geometry of each mechanism.

This module needs only torch.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch

from con_dec_imp.lie import quat_to_matrix


class ScrewGenerator(NamedTuple):
    """A feasible motion of the object in its root frame."""

    kind: Literal["rotation", "translation"]
    axis: tuple[float, float, float]
    point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """A point on the rotation axis (unused for translations)."""


GENERATORS: dict[str, tuple[ScrewGenerator, ...]] = {
    # Door hinge along z through the hinge line.
    "revolute": (ScrewGenerator("rotation", (0.0, 0.0, 1.0), (-0.215, -0.15, 0.0)),),
    # Rotation about and translation along the vertical axis.
    "cylindrical": (
        ScrewGenerator("rotation", (0.0, 0.0, 1.0), (0.0, 0.0, 0.10)),
        ScrewGenerator("translation", (0.0, 0.0, 1.0)),
    ),
    # Rotation about x and translations along y and z; the span does not depend on the axis point.
    "planar": (
        ScrewGenerator("rotation", (1.0, 0.0, 0.0), (-0.025, 0.0, 0.0)),
        ScrewGenerator("translation", (0.0, 1.0, 0.0)),
        ScrewGenerator("translation", (0.0, 0.0, 1.0)),
    ),
    # Global envelope: all rotations about the joint centre (the two joint axes and their normal).
    "universal": (
        ScrewGenerator("rotation", (1.0, 0.0, 0.0)),
        ScrewGenerator("rotation", (0.0, 1.0, 0.0)),
        ScrewGenerator("rotation", (0.0, 0.0, 1.0)),
    ),
}

EVALUATION_LENGTH = {
    # Hinge-to-handle distance.
    "revolute": math.hypot(-0.035, 0.24),
    # Handle radius combined with the axial travel per radian of rotation.
    "cylindrical": math.sqrt((-0.12) ** 2 + (0.30 / math.pi) ** 2),
    # Translation range over rotation range (the handle lies on the rotation axis).
    "planar": math.sqrt(0.30**2 + 0.30**2) / math.pi,
    # Handle distance from the joint centre, RMS over the three envelope rotations (two with lever 0.08 m, one with 0).
    "universal": math.dist((0.0, 0.0, 0.08), (0.0, 0.0, 0.0)) * math.sqrt(2.0 / 3.0),
}
"""Fixed evaluation length ``alpha*`` [m] of the subspace error, per object."""


def reference_basis(object_name: str, tcp_pose: torch.Tensor, object_pose: torch.Tensor) -> torch.Tensor:
    """Basis ``(..., 6, m)`` of the reference feasible subspace in the TCP body frame, rows ``(omega, v)``.

    Args:
        object_name: ``"revolute"``, ``"cylindrical"``, ``"planar"`` or ``"universal"``.
        tcp_pose: TCP pose ``(..., 7)`` as position and quaternion ``(x, y, z, qw, qx, qy, qz)``.
        object_pose: Object root pose ``(..., 7)`` in the same world frame.

    Returns:
        One column per generator of :data:`GENERATORS`, not orthonormalized.
    """
    dtype = torch.promote_types(tcp_pose.dtype, torch.float32)
    tcp, obj = tcp_pose.to(dtype), object_pose.to(dtype)
    p_ee, R_ee = tcp[..., :3], quat_to_matrix(tcp[..., 3:7])
    p_ob, R_ob = obj[..., :3], quat_to_matrix(obj[..., 3:7])

    def to_world(vec: tuple[float, float, float]) -> torch.Tensor:
        v = torch.tensor(vec, dtype=dtype, device=tcp.device).expand(*R_ob.shape[:-2], 3)
        return torch.einsum("...ij,...j->...i", R_ob, v)

    columns = []
    for generator in GENERATORS[object_name]:
        axis_w = to_world(generator.axis)
        if generator.kind == "translation":
            omega, v = torch.zeros_like(axis_w), axis_w
        else:
            omega = axis_w
            v = torch.linalg.cross(axis_w, p_ee - (p_ob + to_world(generator.point)), dim=-1)
        columns.append(
            torch.cat(
                (torch.einsum("...ji,...j->...i", R_ee, omega), torch.einsum("...ji,...j->...i", R_ee, v)), dim=-1
            )
        )
    return torch.stack(columns, dim=-1)
