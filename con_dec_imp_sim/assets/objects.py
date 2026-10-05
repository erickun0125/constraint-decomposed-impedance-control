"""The four articulated objects of Section 5.1, built from primitive shapes.

Every object has a fixed base link, a chain of free (zero-stiffness, damped) joints,
and a cuboid handle attached to the last moving link. Positions are in metres and
expressed in the parent frame; the object root sits at the base-link origin.

==============  ============================================  ==================
object          joints (limits)                               feasible motion
==============  ============================================  ==================
revolute        door hinge about z ([0, pi])                  1 rotation
cylindrical     slide along z ([0, 0.30]) + rotate about z    rotation + axial
                ([0, 1.05 pi])                                translation
planar          slide along y and z ([-0.15, 0.15]) +         2 translations +
                rotate about x ([-pi/2, pi/2])                1 rotation
universal       rotate about x, then y ([-100 deg, 100 deg])  2 rotations
==============  ============================================  ==================
"""

from __future__ import annotations

import math
from dataclasses import MISSING, field
from typing import Literal

from isaaclab.utils import configclass

BLUE_GRAY = (0.7, 0.75, 0.8)
BROWN = (0.55, 0.40, 0.25)
GRAY = (0.5, 0.5, 0.5)
RED = (0.8, 0.2, 0.2)


@configclass
class LinkCfg:
    """A rigid link made of one primitive shape."""

    shape: Literal["cuboid", "cylinder", "sphere"] = MISSING
    size: tuple[float, ...] = MISSING
    """``(x, y, z)`` for a cuboid, ``(radius, height)`` for a z-axis cylinder, ``(radius,)`` for a sphere."""
    mass: float = MISSING
    color: tuple[float, float, float] = GRAY
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Base link: position in the object frame. Other links: link centre relative to its parent joint."""
    is_base: bool = False
    """The base link is welded to the world."""
    collision: bool = True
    visible: bool = True


@configclass
class JointCfg:
    """A joint between two links."""

    type: Literal["revolute", "prismatic", "fixed"] = MISSING
    parent: str = MISSING
    child: str = MISSING
    axis: Literal["X", "Y", "Z"] = "Z"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Joint origin in the parent-link frame."""
    lower: float = 0.0
    """Lower limit [rad or m]."""
    upper: float = 0.0
    """Upper limit [rad or m]."""
    damping: float = 0.0
    """Joint damping [N m s/rad or N s/m]; the joints have no stiffness."""
    effort_limit: float = 0.0


@configclass
class HandleCfg:
    """The cuboid the gripper grasps; part of (and moving with) its parent link."""

    parent: str = MISSING
    position: tuple[float, float, float] = MISSING
    size: tuple[float, float, float] = (0.05, 0.025, 0.08)
    color: tuple[float, float, float] = RED


@configclass
class ArticulatedObjectCfg:
    """An articulated object assembled by :func:`~con_dec_imp_sim.assets.spawner.spawn_articulated_object`."""

    links: dict[str, LinkCfg] = MISSING
    joints: dict[str, JointCfg] = MISSING
    handle: HandleCfg = MISSING
    init_joint_pos: dict[str, float] = field(default_factory=dict)

    @property
    def base_link(self) -> str:
        """Name of the link welded to the world."""
        return next(name for name, link in self.links.items() if link.is_base)


REVOLUTE_OBJECT = ArticulatedObjectCfg(
    links={
        "base": LinkCfg(shape="cuboid", size=(0.40, 0.30, 0.30), mass=10.0, color=BLUE_GRAY, is_base=True),
        # The hinge sits on the door's left edge, 0.14 m from the door centre.
        "door": LinkCfg(shape="cuboid", size=(0.02, 0.28, 0.30), mass=0.1, color=BROWN, position=(0.0, 0.14, 0.0)),
    },
    joints={
        "door_joint": JointCfg(
            type="revolute", parent="base", child="door", axis="Z", position=(-0.215, -0.15, 0.0),
            lower=0.0, upper=math.pi, damping=2.0, effort_limit=50.0,
        ),
    },
    handle=HandleCfg(parent="door", position=(-0.035, 0.10, 0.0)),
    init_joint_pos={"door_joint": 0.0},
)

CYLINDRICAL_OBJECT = ArticulatedObjectCfg(
    links={
        "base": LinkCfg(shape="cylinder", size=(0.08, 0.20), mass=5.0, color=BLUE_GRAY, is_base=True, collision=False),
        "slider": LinkCfg(
            shape="cylinder", size=(0.015, 0.30), mass=0.01, color=BROWN, position=(0.0, 0.0, -0.15), collision=False
        ),
        "handle_part": LinkCfg(
            shape="cylinder", size=(0.085, 0.05), mass=0.1, color=BROWN, position=(0.0, 0.0, 0.175), collision=False
        ),
    },
    joints={
        "prismatic_joint": JointCfg(
            type="prismatic", parent="base", child="slider", axis="Z", position=(0.0, 0.0, 0.10),
            lower=0.0, upper=0.30, damping=2.0, effort_limit=100.0,
        ),
        "revolute_joint": JointCfg(
            type="revolute", parent="slider", child="handle_part", axis="Z",
            lower=0.0, upper=math.pi * 1.05, damping=2.0, effort_limit=30.0,
        ),
    },
    handle=HandleCfg(parent="handle_part", position=(-0.12, 0.0, 0.0)),
    init_joint_pos={"prismatic_joint": 0.0, "revolute_joint": 0.0},
)

PLANAR_OBJECT = ArticulatedObjectCfg(
    links={
        "board": LinkCfg(shape="cuboid", size=(0.02, 0.40, 0.40), mass=5.0, color=BLUE_GRAY, is_base=True),
        "slider_y": LinkCfg(shape="cuboid", size=(0.01, 0.01, 0.01), mass=0.01, collision=False, visible=False),
        "slider_z": LinkCfg(shape="cuboid", size=(0.01, 0.01, 0.01), mass=0.01, collision=False, visible=False),
        "pad": LinkCfg(shape="cuboid", size=(0.03, 0.08, 0.06), mass=0.1, color=BROWN, collision=False),
    },
    joints={
        "prismatic_y_joint": JointCfg(
            type="prismatic", parent="board", child="slider_y", axis="Y", position=(-0.025, 0.0, 0.0),
            lower=-0.15, upper=0.15, damping=2.0, effort_limit=100.0,
        ),
        "prismatic_z_joint": JointCfg(
            type="prismatic", parent="slider_y", child="slider_z", axis="Z",
            lower=-0.15, upper=0.15, damping=2.0, effort_limit=100.0,
        ),
        "revolute_x_joint": JointCfg(
            type="revolute", parent="slider_z", child="pad", axis="X",
            lower=-math.pi / 2, upper=math.pi / 2, damping=2.0, effort_limit=30.0,
        ),
    },
    handle=HandleCfg(parent="pad", position=(-0.04, 0.0, 0.0)),
    init_joint_pos={"prismatic_y_joint": 0.15, "prismatic_z_joint": -0.15, "revolute_x_joint": -math.pi / 2},
)

UNIVERSAL_OBJECT = ArticulatedObjectCfg(
    links={
        "base": LinkCfg(shape="sphere", size=(0.05,), mass=5.0, color=BLUE_GRAY, is_base=True, collision=False),
        "axis1_link": LinkCfg(shape="sphere", size=(0.04,), mass=0.2, collision=False),
        "handle_part": LinkCfg(shape="sphere", size=(0.045,), mass=0.3, color=BROWN, collision=False),
        # Visual pedestal welded to the base.
        "pedestal": LinkCfg(shape="cuboid", size=(0.18, 0.18, 0.04), mass=0.2, color=(0.35, 0.35, 0.4), collision=False),
    },
    joints={
        "revolute_x_joint": JointCfg(
            type="revolute", parent="base", child="axis1_link", axis="X",
            lower=-math.radians(100.0), upper=math.radians(100.0), damping=0.5, effort_limit=30.0,
        ),
        "revolute_y_joint": JointCfg(
            type="revolute", parent="axis1_link", child="handle_part", axis="Y",
            lower=-math.radians(100.0), upper=math.radians(100.0), damping=0.5, effort_limit=30.0,
        ),
        "pedestal_joint": JointCfg(type="fixed", parent="base", child="pedestal", position=(0.0, 0.0, -0.04)),
    },
    handle=HandleCfg(parent="handle_part", position=(0.0, 0.0, 0.08), size=(0.025, 0.025, 0.08)),
    init_joint_pos={"revolute_x_joint": 0.0, "revolute_y_joint": 0.0},
)
