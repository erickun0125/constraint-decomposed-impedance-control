"""Cylindrical joint: rotate the handle by pi while pulling it 0.30 m along the axis.

An evaluation episode succeeds once the rotation exceeds 145 deg and the translation
exceeds 0.26 m at the same time.
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
from ..assets import CYLINDRICAL_OBJECT
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

ROTATION_TARGET = math.pi
TRANSLATION_TARGET = 0.30


@configclass
class CylindricalSceneCfg(FlyingGripperSceneCfg):
    object: ArticulationCfg = object_cfg(CYLINDRICAL_OBJECT, "Cylindrical")
    handle_frame: FrameTransformerCfg = handle_frame_cfg(CYLINDRICAL_OBJECT, "Cylindrical")
    wrist_cam: TiledCameraCfg = WRIST_CAMERA_CFG


@configclass
class CylindricalEventsCfg(EventsCfg):
    reset_object_joints = EventTerm(
        func=mdp.reset_joint_positions,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"), "positions": {"revolute_joint": 0.0, "prismatic_joint": 0.0}},
    )
    reset_object_pose = EventTerm(
        func=mdp.reset_object_pose, mode="reset", params={"asset_cfg": SceneEntityCfg("object"), **OBJECT_POSE_RANGES}
    )


@configclass
class CylindricalEnvCfg(FlyingGripperEnvCfg):
    """Cylindrical joint, evaluation setting."""

    scene: CylindricalSceneCfg = CylindricalSceneCfg(num_envs=EVAL_NUM_ENVS, env_spacing=ENV_SPACING)
    events: CylindricalEventsCfg = CylindricalEventsCfg()
    eval_success = {
        "revolute_joint": ("greater", math.radians(145.0)),
        "prismatic_joint": ("greater", TRANSLATION_TARGET - 0.040),
    }
    collect_success = {
        "revolute_joint": ("greater", ROTATION_TARGET - 0.16),
        "prismatic_joint": ("greater", TRANSLATION_TARGET - 0.015),
    }
    viewer: ViewerCfg = ViewerCfg(eye=(0.8, 1.0, 0.8), lookat=(0.35, 0.0, 0.35))


@configclass
class CylindricalCollectEnvCfg(CylindricalEnvCfg):
    """Cylindrical joint, demonstration-collection setting."""

    scene: CylindricalSceneCfg = CylindricalSceneCfg(num_envs=COLLECT_NUM_ENVS, env_spacing=ENV_SPACING)
    collection: bool = True
