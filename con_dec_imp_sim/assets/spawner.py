"""USD spawner that assembles an :class:`ArticulatedObjectCfg` from primitive shapes.

The object is authored once under ``env_0``; the interactive scene clones it to the
other environments. Layout of the spawned prim::

    <root>                  articulation root, object pose
    <root>/<link>           one rigid body per link (child links at the root origin)
    <root>/<link>/<joint>   joint whose child is <link>
    <root>/FixedJoint       welds the base link to the world
    <root>/<parent>/handle  handle shape, part of its parent rigid body
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import MISSING

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils import configclass

from .objects import ArticulatedObjectCfg, HandleCfg, JointCfg, LinkCfg

_IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _spawn_link(prim_path: str, link: LinkCfg, is_child: bool) -> Usd.Prim:
    """Spawn one link; a child link is placed at the root origin and positioned by its joint."""
    common = dict(
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=link.color),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=link.mass),
        collision_props=sim_utils.CollisionPropertiesCfg() if link.collision else None,
    )
    if link.shape == "cuboid":
        cfg = sim_utils.CuboidCfg(size=link.size, **common)
    elif link.shape == "cylinder":
        radius, height = link.size
        cfg = sim_utils.CylinderCfg(radius=radius, height=height, axis="Z", **common)
    elif link.shape == "sphere":
        (radius,) = link.size
        cfg = sim_utils.SphereCfg(radius=radius, **common)
    else:
        raise ValueError(f"Unsupported link shape: {link.shape}")
    translation = (0.0, 0.0, 0.0) if is_child else link.position
    cfg.func(prim_path, cfg, translation=translation, orientation=_IDENTITY)
    prim = get_current_stage().GetPrimAtPath(prim_path)
    if not link.visible:
        UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def _create_joint(name: str, joint: JointCfg, parent: Usd.Prim, child: Usd.Prim, child_offset) -> None:
    """Create ``joint`` under its child link; the child's centre sits at ``child_offset`` from the joint."""
    stage = get_current_stage()
    path = f"{child.GetPath()}/{name}"
    if joint.type == "revolute":
        usd_joint = UsdPhysics.RevoluteJoint.Define(stage, path)
        usd_joint.GetAxisAttr().Set(joint.axis)
        usd_joint.GetLowerLimitAttr().Set(math.degrees(joint.lower))
        usd_joint.GetUpperLimitAttr().Set(math.degrees(joint.upper))
    elif joint.type == "prismatic":
        usd_joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        usd_joint.GetAxisAttr().Set(joint.axis)
        usd_joint.GetLowerLimitAttr().Set(joint.lower)
        usd_joint.GetUpperLimitAttr().Set(joint.upper)
    elif joint.type == "fixed":
        usd_joint = UsdPhysics.FixedJoint.Define(stage, path)
    else:
        raise ValueError(f"Unsupported joint type: {joint.type}")

    usd_joint.GetBody0Rel().SetTargets([parent.GetPath()])
    usd_joint.GetBody1Rel().SetTargets([child.GetPath()])
    usd_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*joint.position))
    usd_joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    usd_joint.GetLocalPos1Attr().Set(Gf.Vec3f(*(-c for c in child_offset)))
    usd_joint.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    if joint.type != "fixed":
        drive = UsdPhysics.DriveAPI.Apply(usd_joint.GetPrim(), "angular" if joint.type == "revolute" else "linear")
        drive.GetStiffnessAttr().Set(0.0)
        drive.GetDampingAttr().Set(joint.damping)
        drive.GetMaxForceAttr().Set(joint.effort_limit)


def _spawn_handle(parent_path: str, handle: HandleCfg) -> None:
    """Add the handle as a collision shape of its parent rigid body."""
    cfg = sim_utils.CuboidCfg(
        size=handle.size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=handle.color),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    cfg.func(f"{parent_path}/handle", cfg, translation=handle.position, orientation=_IDENTITY)


def spawn_articulated_object(
    prim_path: str,
    cfg: ArticulatedObjectSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn ``cfg.object_cfg`` at ``prim_path`` (an ``env_.*`` pattern resolves to ``env_0``)."""
    stage = get_current_stage()
    prim_path = re.sub(r"/env_\.\*", "/env_0", prim_path)
    obj = cfg.object_cfg

    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    w, x, y, z = orientation or _IDENTITY
    root.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    root_prim = root.GetPrim()

    links = {name: _spawn_link(f"{prim_path}/{name}", link, not link.is_base) for name, link in obj.links.items()}

    # Weld the base link to the world and make the object root the (fixed-base) articulation root.
    base = links[obj.base_link]
    UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/FixedJoint").GetBody1Rel().SetTargets([base.GetPath()])
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(root_prim).GetArticulationEnabledAttr().Set(True)

    for name, joint in obj.joints.items():
        _create_joint(name, joint, links[joint.parent], links[joint.child], obj.links[joint.child].position)

    _spawn_handle(f"{prim_path}/{obj.handle.parent}", obj.handle)
    return root_prim


@configclass
class ArticulatedObjectSpawnerCfg(SpawnerCfg):
    """Spawner configuration for an :class:`ArticulatedObjectCfg`."""

    func: Callable = spawn_articulated_object
    object_cfg: ArticulatedObjectCfg = MISSING
