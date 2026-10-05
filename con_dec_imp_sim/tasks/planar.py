"""Planar joint: slide the pad from (y, z) = (0.15, -0.15) to (-0.15, 0.15) m while turning it from -pi/2 to pi/2.

An evaluation episode succeeds once y < -0.12 m, z > 0.12 m and the rotation exceeds
pi/2 - 0.32 rad at the same time.
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
from ..assets import PLANAR_OBJECT
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

TRANSLATION_Y_TARGET = -0.15
TRANSLATION_Z_TARGET = 0.15
ROTATION_TARGET = math.pi / 2


@configclass
class PlanarSceneCfg(FlyingGripperSceneCfg):
    object: ArticulationCfg = object_cfg(PLANAR_OBJECT, "Planar")
    handle_frame: FrameTransformerCfg = handle_frame_cfg(PLANAR_OBJECT, "Planar")
    wrist_cam: TiledCameraCfg = WRIST_CAMERA_CFG


@configclass
class PlanarEventsCfg(EventsCfg):
    reset_object_joints = EventTerm(
        func=mdp.reset_joint_positions,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "positions": {"prismatic_y_joint": 0.15, "prismatic_z_joint": -0.15, "revolute_x_joint": -1.5708},
        },
    )
    reset_object_pose = EventTerm(
        func=mdp.reset_object_pose, mode="reset", params={"asset_cfg": SceneEntityCfg("object"), **OBJECT_POSE_RANGES}
    )


@configclass
class PlanarEnvCfg(FlyingGripperEnvCfg):
    """Planar joint, evaluation setting."""

    scene: PlanarSceneCfg = PlanarSceneCfg(num_envs=EVAL_NUM_ENVS, env_spacing=ENV_SPACING)
    events: PlanarEventsCfg = PlanarEventsCfg()
    eval_success = {
        "prismatic_y_joint": ("less", TRANSLATION_Y_TARGET + 0.030),
        "prismatic_z_joint": ("greater", TRANSLATION_Z_TARGET - 0.030),
        "revolute_x_joint": ("greater", ROTATION_TARGET - 0.32),
    }
    collect_success = {
        "prismatic_y_joint": ("less", TRANSLATION_Y_TARGET + 0.015),
        "prismatic_z_joint": ("greater", TRANSLATION_Z_TARGET - 0.015),
        "revolute_x_joint": ("greater", ROTATION_TARGET - 0.16),
    }
    viewer: ViewerCfg = ViewerCfg(eye=(0.8, 1.0, 0.8), lookat=(0.40, 0.0, 0.40))


@configclass
class PlanarCollectEnvCfg(PlanarEnvCfg):
    """Planar joint, demonstration-collection setting."""

    scene: PlanarSceneCfg = PlanarSceneCfg(num_envs=COLLECT_NUM_ENVS, env_spacing=ENV_SPACING)
    collection: bool = True
