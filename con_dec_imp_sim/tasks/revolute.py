"""Revolute door: open the hinge.

The expert opens the door to 0.95 pi. An evaluation episode succeeds once the hinge
exceeds 0.95 pi - 0.32 rad (152.7 deg).
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
from ..assets import REVOLUTE_OBJECT
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

OPEN_TARGET = 0.95 * math.pi


@configclass
class RevoluteSceneCfg(FlyingGripperSceneCfg):
    object: ArticulationCfg = object_cfg(REVOLUTE_OBJECT, "Revolute")
    handle_frame: FrameTransformerCfg = handle_frame_cfg(REVOLUTE_OBJECT, "Revolute")
    wrist_cam: TiledCameraCfg = WRIST_CAMERA_CFG


@configclass
class RevoluteEventsCfg(EventsCfg):
    reset_object_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    reset_object_pose = EventTerm(
        func=mdp.reset_object_pose, mode="reset", params={"asset_cfg": SceneEntityCfg("object"), **OBJECT_POSE_RANGES}
    )


@configclass
class RevoluteEnvCfg(FlyingGripperEnvCfg):
    """Revolute door, evaluation setting."""

    scene: RevoluteSceneCfg = RevoluteSceneCfg(num_envs=EVAL_NUM_ENVS, env_spacing=ENV_SPACING)
    events: RevoluteEventsCfg = RevoluteEventsCfg()
    eval_success = {"door_joint": ("greater", OPEN_TARGET - 0.32)}
    collect_success = {"door_joint": ("greater", OPEN_TARGET - 0.16)}
    viewer: ViewerCfg = ViewerCfg(eye=(1.2, 1.2, 1.2), lookat=(0.2, 0.0, 0.35))


@configclass
class RevoluteCollectEnvCfg(RevoluteEnvCfg):
    """Revolute door, demonstration-collection setting."""

    scene: RevoluteSceneCfg = RevoluteSceneCfg(num_envs=COLLECT_NUM_ENVS, env_spacing=ENV_SPACING)
    collection: bool = True
