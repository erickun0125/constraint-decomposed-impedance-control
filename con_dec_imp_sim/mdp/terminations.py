"""Task success: every listed object joint is past its threshold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joints_reached(
    env: ManagerBasedEnv,
    thresholds: dict[str, tuple[Literal["greater", "less"], float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """True where every joint ``name`` satisfies ``q > value`` (``"greater"``) or ``q < value`` (``"less"``)."""
    asset = env.scene[asset_cfg.name]
    reached = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name, (comparison, value) in thresholds.items():
        q = asset.data.joint_pos[:, asset.find_joints(name)[0][0]]
        reached &= q > value if comparison == "greater" else q < value
    return reached
