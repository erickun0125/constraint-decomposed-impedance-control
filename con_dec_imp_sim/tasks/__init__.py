"""Flying-gripper environments of the four articulated objects, registered with gymnasium.

Each object has an evaluation environment ``ConDecImp-FlyingGripper-<Object>-v0`` and a
demonstration-collection environment ``ConDecImp-FlyingGripper-<Object>-Collect-v0``;
create them with ``gym.make(task_id, cfg=cfg)`` or ``ManagerBasedRLEnv(cfg)``.
"""

import gymnasium as gym

from .. import OBJECTS
from .base import EVAL_EPISODE_STEPS, FlyingGripperEnvCfg
from .cylindrical import CylindricalCollectEnvCfg, CylindricalEnvCfg
from .planar import PlanarCollectEnvCfg, PlanarEnvCfg
from .revolute import RevoluteCollectEnvCfg, RevoluteEnvCfg
from .universal import UniversalCollectEnvCfg, UniversalEnvCfg

EVAL_ENV_CFGS: dict[str, type[FlyingGripperEnvCfg]] = {
    "revolute": RevoluteEnvCfg,
    "cylindrical": CylindricalEnvCfg,
    "planar": PlanarEnvCfg,
    "universal": UniversalEnvCfg,
}
COLLECT_ENV_CFGS: dict[str, type[FlyingGripperEnvCfg]] = {
    "revolute": RevoluteCollectEnvCfg,
    "cylindrical": CylindricalCollectEnvCfg,
    "planar": PlanarCollectEnvCfg,
    "universal": UniversalCollectEnvCfg,
}


def task_id(object_name: str, collection: bool = False) -> str:
    """Gymnasium id of an object's evaluation or collection environment."""
    return f"ConDecImp-FlyingGripper-{object_name.capitalize()}{'-Collect' if collection else ''}-v0"


for _name in OBJECTS:
    for _collection, _cfgs in ((False, EVAL_ENV_CFGS), (True, COLLECT_ENV_CFGS)):
        gym.register(
            id=task_id(_name, _collection),
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{_cfgs[_name].__module__}:{_cfgs[_name].__name__}"},
        )
