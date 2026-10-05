"""Scene and environment settings shared by the four flying-gripper tasks (Section 5.1, Appendix H).

Each environment holds the flying gripper, one articulated object and a wrist camera
inside four walls. The impedance controller runs at 150 Hz (one physics step per
control step) and the camera renders at 10 Hz, the policy rate.

Two settings are provided per task:

* evaluation: 2000-step (13.3 s) episodes, the isotropic gains of Iso as the initial
  gains, and contact sensors for the gripper-object contact wrench;
* demonstration collection: 20 s episodes and the scripted expert's gains.

The success thresholds of the two settings differ (see the task modules). During collection an
episode also ends when the scripted expert completes its articulation, whichever comes first.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass

from con_dec_imp.gains import inertia_metric, isotropic_gains
from con_dec_imp.settings import CONTROL

from .. import mdp
from ..assets import FLYING_GRIPPER_CFG, HAND_BODY, TCP_OFFSET_POS, TCP_OFFSET_ROT
from ..assets.objects import ArticulatedObjectCfg
from ..assets.spawner import ArticulatedObjectSpawnerCfg
from ..sensors import contact_sensor_cfgs

ENV_SPACING = 2.5
EVAL_NUM_ENVS = 100
COLLECT_NUM_ENVS = 16

EVAL_EPISODE_STEPS = 2000
"""Evaluation episode budget in control steps (13.3 s); an episode that has not succeeded by then fails."""

COLLECT_EPISODE_LENGTH_S = 20.0

OBJECT_ORIGIN = (0.5, 0.0, 0.5)
"""Nominal object root position in the environment frame (Appendix H, Table "Simulation policy data")."""

OBJECT_POSE_RANGES = {
    "position_range_x": (-0.15, 0.15),
    "position_range_y": (-0.2, 0.2),
    "position_range_z": (-0.1, 0.1),
    "yaw_range": (-math.radians(15), math.radians(15)),
}
"""Uniform object placement around :data:`OBJECT_ORIGIN`."""

GRIPPER_START = (-0.1, 0.0, 0.5)
"""Hand origin at reset in the environment frame; the TCP starts at (0.0034, 0, 0.5)."""

GRIPPER_START_ORIENTATION = (0.0, 0.7071, 0.0, 0.7071)
"""Hand orientation at reset ``(w, x, y, z)``: the TCP axes align with the world axes."""

# Diagonal gains of the scripted expert (not the Iso gains), ordered (rotation, translation).
EXPERT_STIFFNESS = (12.0, 12.0, 12.0, 250.0, 250.0, 250.0)
EXPERT_DAMPING = (0.24, 0.24, 0.24, 20.0, 20.0, 20.0)

WALL_HEIGHT = 1.5
WALL_THICKNESS = 0.02
WALL_LENGTH = 2.4
WALL_COLOR = (0.3, 0.3, 0.35)

WRIST_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Gripper/panda_hand/wrist_cam",
    update_period=1.0 / CONTROL.policy_rate,
    height=224,
    width=224,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, -0.2), rot=(0.7011, -0.0923, -0.0923, 0.7011), convention="ros"),
)
"""224x224 RGB wrist camera on the hand, pitched 15 deg toward the fingertips."""


def _wall(name: str, pos: tuple[float, float], along_x: bool) -> AssetBaseCfg:
    size = (WALL_LENGTH, WALL_THICKNESS, WALL_HEIGHT) if along_x else (WALL_THICKNESS, WALL_LENGTH, WALL_HEIGHT)
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*pos, WALL_HEIGHT / 2)),
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=WALL_COLOR),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        collision_group=-1,
    )


def object_cfg(obj: ArticulatedObjectCfg, prim_name: str) -> ArticulationCfg:
    """Scene entry of an articulated object at :data:`OBJECT_ORIGIN` with free, damped joints."""
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=ArticulatedObjectSpawnerCfg(object_cfg=obj),
        init_state=ArticulationCfg.InitialStateCfg(pos=OBJECT_ORIGIN, joint_pos=dict(obj.init_joint_pos)),
        actuators={
            name: ImplicitActuatorCfg(joint_names_expr=[name], stiffness=0.0, damping=joint.damping)
            for name, joint in obj.joints.items()
            if joint.type != "fixed"
        },
    )


def handle_frame_cfg(obj: ArticulatedObjectCfg, prim_name: str) -> FrameTransformerCfg:
    """Frame transformer whose target is the handle centre (orientation of the handle's link)."""
    return FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}/{obj.base_link}",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{prim_name}/{obj.handle.parent}",
                name="handle",
                offset=OffsetCfg(pos=obj.handle.position),
            )
        ],
    )


@configclass
class FlyingGripperSceneCfg(InteractiveSceneCfg):
    """Gripper, TCP frame, ground, light and walls; task scenes add the object, handle frame and camera."""

    gripper: ArticulationCfg = FLYING_GRIPPER_CFG
    gripper_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Gripper/{HAND_BODY}",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Gripper/{HAND_BODY}",
                name="tcp",
                offset=OffsetCfg(pos=TCP_OFFSET_POS, rot=TCP_OFFSET_ROT),
            )
        ],
    )
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    wall_left: AssetBaseCfg = _wall("WallLeft", (0.0, 1.2), along_x=True)
    wall_right: AssetBaseCfg = _wall("WallRight", (0.0, -1.2), along_x=True)
    wall_back: AssetBaseCfg = _wall("WallBack", (1.2, 0.0), along_x=False)
    wall_front: AssetBaseCfg = _wall("WallFront", (-1.2, 0.0), along_x=False)


@configclass
class ObservationsCfg:
    """Policy observations: TCP pose (position + 6-D rotation), gripper opening and wrist image."""

    @configclass
    class PolicyCfg(ObsGroup):
        eef_pose = ObsTerm(func=mdp.tcp_pose)
        gripper = ObsTerm(func=mdp.gripper_opening)
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    impedance: mdp.ImpedanceActionCfg = mdp.ImpedanceActionCfg()


@configclass
class EventsCfg:
    """Gripper reset; task event configs add the object joint reset and the object pose randomization."""

    reset_gripper = EventTerm(
        func=mdp.reset_gripper_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("gripper"),
            "position_range": tuple((c, c) for c in GRIPPER_START),
            "base_orientation": GRIPPER_START_ORIENTATION,
        },
    )


@configclass
class RewardsCfg:
    """No rewards; the environments are used for imitation and evaluation only."""


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success: DoneTerm = MISSING


Thresholds = dict[str, tuple[Literal["greater", "less"], float]]


@configclass
class FlyingGripperEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment; a task sets ``scene``, ``events`` and the two success-threshold maps."""

    scene: FlyingGripperSceneCfg = MISSING
    events: EventsCfg = MISSING
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    collection: bool = False
    """Demonstration-collection setting instead of the evaluation setting."""

    eval_success: Thresholds = MISSING
    """Success thresholds of the evaluation setting (joint name -> comparison, value)."""

    collect_success: Thresholds = MISSING
    """Success thresholds of the collection setting."""

    decimation: int = 1
    episode_length_s: float = MISSING
    num_rerenders_on_reset: int = 1
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / CONTROL.control_rate,
        render_interval=round(CONTROL.control_rate / CONTROL.policy_rate),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    def __post_init__(self):
        thresholds = self.collect_success if self.collection else self.eval_success
        self.terminations.success = DoneTerm(func=mdp.joints_reached, params={"thresholds": thresholds})
        if self.collection:
            self.episode_length_s = COLLECT_EPISODE_LENGTH_S
            self.actions.impedance.stiffness = EXPERT_STIFFNESS
            self.actions.impedance.damping = EXPERT_DAMPING
        else:
            self.episode_length_s = EVAL_EPISODE_STEPS * self.sim.dt * self.decimation
            K, D = isotropic_gains(inertia_metric())
            self.actions.impedance.stiffness = tuple(K.diagonal().tolist())
            self.actions.impedance.damping = tuple(D.diagonal().tolist())
            self.scene.gripper.spawn.activate_contact_sensors = True
            handle_link_path = f"{self.scene.object.prim_path}/{self.scene.object.spawn.object_cfg.handle.parent}"
            for name, sensor in contact_sensor_cfgs(handle_link_path).items():
                setattr(self.scene, name, sensor)
