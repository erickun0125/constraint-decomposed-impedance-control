"""Universal joint: turn both revolute axes (x, then y) by pi/2.

An evaluation episode succeeds once both joint angles exceed 65 deg at the same time.
"""

from __future__ import annotations

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.utils import configclass

from .. import mdp
from ..assets import UNIVERSAL_OBJECT
from .base import (
    COLLECT_NUM_ENVS,
    ENV_SPACING,
    EVAL_NUM_ENVS,
    OBJECT_POSE_RANGES,
    WRIST_CAMERA_CFG,
    EventsCfg,
    FlyingGripperEnvCfg,
    FlyingGripperSceneCfg,
    handle_frame_cfg,
    object_cfg,
)

ROTATION_TARGET = math.pi / 2


@configclass
class UniversalSceneCfg(FlyingGripperSceneCfg):
    object: ArticulationCfg = object_cfg(UNIVERSAL_OBJECT, "Universal")
    handle_frame: FrameTransformerCfg = handle_frame_cfg(UNIVERSAL_OBJECT, "Universal")
    wrist_cam: TiledCameraCfg = WRIST_CAMERA_CFG


@configclass
class UniversalEventsCfg(EventsCfg):
    reset_object_joints = EventTerm(
        func=mdp.reset_joint_positions,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"), "positions": {"revolute_x_joint": 0.0, "revolute_y_joint": 0.0}},
    )
    reset_object_pose = EventTerm(
        func=mdp.reset_object_pose, mode="reset", params={"asset_cfg": SceneEntityCfg("object"), **OBJECT_POSE_RANGES}
    )


@configclass
class UniversalEnvCfg(FlyingGripperEnvCfg):
    """Universal joint, evaluation setting."""

    scene: UniversalSceneCfg = UniversalSceneCfg(num_envs=EVAL_NUM_ENVS, env_spacing=ENV_SPACING)
    events: UniversalEventsCfg = UniversalEventsCfg()
    eval_success = {
        "revolute_x_joint": ("greater", math.radians(65.0)),
        "revolute_y_joint": ("greater", math.radians(65.0)),
    }
    collect_success = {
        "revolute_x_joint": ("greater", ROTATION_TARGET - 0.16),
        "revolute_y_joint": ("greater", ROTATION_TARGET - 0.16),
    }
    viewer: ViewerCfg = ViewerCfg(eye=(0.8, 1.0, 0.8), lookat=(0.40, 0.0, 0.40))


@configclass
class UniversalCollectEnvCfg(UniversalEnvCfg):
    """Universal joint, demonstration-collection setting."""

    scene: UniversalSceneCfg = UniversalSceneCfg(num_envs=COLLECT_NUM_ENVS, env_spacing=ENV_SPACING)
    collection: bool = True
