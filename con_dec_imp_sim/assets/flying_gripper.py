"""Flying gripper: a Franka hand carried by six virtual joints.

The articulation is fixed to the world at ``root_link`` and reaches the hand through
three prismatic joints (x, y, z) and three revolute joints (y, x, z), so the hand pose
is a joint-space configuration and the hand is driven by joint torques
``tau = J_b^T F`` (Section 3.1, Eq. 2). The virtual joints have no drive gains and gravity is
disabled, so the commanded wrench reaches the hand unchanged. The fingers are
position controlled.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

USD_PATH = os.path.join(os.path.dirname(__file__), "data", "flying_gripper.usda")

HAND_BODY = "panda_hand"
FINGER_BODIES = ("panda_leftfinger", "panda_rightfinger")
VIRTUAL_JOINTS = "root_.*_joint"
"""Regex of the six virtual joints, ordered (x, y, z, Ry, Rx, Rz) in the articulation."""

FINGER_JOINT = "panda_finger_joint1"
"""Actuated finger joint; ``panda_finger_joint2`` mimics it."""

FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0

TCP_OFFSET_POS = (0.0, 0.0, 0.1034)
TCP_OFFSET_ROT = (0.0, 0.7071, 0.0, 0.7071)
"""TCP frame in the hand frame (quaternion ``w, x, y, z``): the TCP lies between the
fingertips, with x toward the fingertips and y along the finger motion axis."""

FLYING_GRIPPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Gripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "root_x_axis_joint": 0.0,
            "root_y_axis_joint": 0.0,
            "root_z_axis_joint": 0.5,
            # R = Ry(pi/2) Rx(0) Rz(pi): the fingertips (hand +z) point along world +x.
            "root_y_rot_joint": 1.570796,
            "root_x_rot_joint": 0.0,
            "root_z_rot_joint": 3.141593,
            FINGER_JOINT: FINGER_OPEN,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "virtual_joints": ImplicitActuatorCfg(
            joint_names_expr=[VIRTUAL_JOINTS],
            effort_limit_sim=1000.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[FINGER_JOINT],
            effort_limit_sim=400.0,
            velocity_limit_sim=0.2,
            stiffness=4000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Flying gripper articulation (hand 0.558 kg, two fingers 0.014 kg each)."""
