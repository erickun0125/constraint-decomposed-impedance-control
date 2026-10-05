"""Collect scripted demonstrations of one simulated object (paper, Appendix H, "Simulation demonstration data").

usage: python scripts/collect_demos.py --object revolute --out datasets/revolute.hdf5 --headless

The scripted expert (:mod:`con_dec_imp_sim.expert`) drives the flying gripper in the
collection environment ``ConDecImp-FlyingGripper-<Object>-Collect-v0``: ``num_envs``
episodes run in parallel from one batched reset, each with a 20 s limit. Every policy
period (0.1 s) the expert commands the next waypoint and the impedance controller tracks
the ScLERP interpolation from the measured TCP pose toward it for 15 control steps (150 Hz);
the evaluation instead interpolates between consecutive policy waypoints. A demonstration step
stores the observation at the start of the period and the commanded waypoint as action
(absolute TCP pose in the environment frame, rot6d, gripper command).

Batches run until ``num_demos`` successful demonstrations are written, in the order the
episodes finish; failed episodes are discarded. The first ``train_fraction`` of the
demonstrations form ``mask/train`` and the rest ``mask/valid`` (file layout:
:mod:`con_dec_imp_sim.demos`). Settings come from ``configs/collect/<object>.yaml``; the
impedance gains are those of the collection environment (:mod:`con_dec_imp_sim.tasks.base`).

Seeds: ``--seed`` seeds the expert's timing draws (the multi-coordinate objects) and, as
``1000003 * seed + b``, the reset of batch ``b``, so the initial object poses of a batch
depend only on both.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher

from con_dec_imp_sim import OBJECTS

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "collect"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--object", required=True, choices=OBJECTS)
parser.add_argument("--config", type=Path, default=None, help="settings (default: configs/collect/<object>.yaml)")
parser.add_argument("--num_demos", type=int, default=None, help="successful demonstrations to write (default: config)")
parser.add_argument("--num_envs", type=int, default=None, help="environments simulated in parallel (default: config)")
parser.add_argument("--seed", type=int, default=None, help="seed of the expert and the resets (default: config)")
parser.add_argument("--out", type=Path, default=None, help="HDF5 file to write (default: datasets/<object>.hdf5)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
settings = yaml.safe_load((args.config or CONFIGS / f"{args.object}.yaml").read_text())
if settings["object"] != args.object:
    parser.error(f"the config describes the {settings['object']} object, not {args.object}")
num_demos = args.num_demos if args.num_demos is not None else settings["num_demos"]
num_envs = args.num_envs if args.num_envs is not None else settings["num_envs"]
seed = args.seed if args.seed is not None else settings["seed"]
out = args.out or Path("datasets") / f"{args.object}.hdf5"
if num_demos < 1 or num_envs < 1:
    parser.error("--num_demos and --num_envs must be positive")
if out.exists():
    parser.error(f"{out} exists")
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from con_dec_imp_sim.demos import Demonstration, DemoWriter  # noqa: E402
from con_dec_imp_sim.expert import (  # noqa: E402
    ExpertObservation,
    Phase,
    ScriptedExpert,
    interpolate_poses,
    make_expert,
    pose_to_action,
)
from con_dec_imp_sim.tasks import COLLECT_ENV_CFGS, task_id  # noqa: E402

EXPERT_JOINTS = {
    "revolute": ("door_joint",),
    "cylindrical": ("revolute_joint", "prismatic_joint"),
    "planar": ("prismatic_y_joint", "prismatic_z_joint", "revolute_x_joint"),
    "universal": ("revolute_x_joint", "revolute_y_joint"),
}
"""Object joints in the order of each expert's ``coordinates``."""

SEED_STRIDE = 1000003
"""Offset between the reset seeds of consecutive ``--seed`` values; batches add their index."""

MAX_EPISODES_PER_DEMO = 2
"""Collection stops with an error after this many episodes per requested demonstration."""


def frame_pose(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """World pose ``(N, 7)`` of the first target of a frame transformer."""
    data = env.scene[name].data
    return torch.cat((data.target_pos_w[:, 0], data.target_quat_w[:, 0]), dim=-1)


def observe(env: ManagerBasedRLEnv, joint_ids: list[int]) -> ExpertObservation:
    """Snapshot of the state the expert reads."""
    obj = env.scene["object"].data
    return ExpertObservation(
        tcp_pose=frame_pose(env, "gripper_frame"),
        handle_pose=frame_pose(env, "handle_frame"),
        base_pose=torch.cat((obj.root_pos_w, obj.root_quat_w), dim=-1),
        joint_pos=obj.joint_pos[:, joint_ids],
    )


def run_batch(env: ManagerBasedRLEnv, expert: ScriptedExpert, joint_ids: list[int], reset_seed: int):
    """Run one batch of episodes from a seeded reset.

    Yields ``(demonstration, success, coordinates, phase)`` for each episode as it ends, in
    order of the policy step and then of the environment; ``coordinates`` are the object
    coordinates at the start of the last step and ``phase`` the last phase the expert was
    working in (the phase that timed out if it failed). An episode ends when the expert
    finishes or fails, or when the environment terminates it (success threshold or the 20 s
    time limit, which bounds the batch); its environment keeps running unrecorded until the
    whole batch has ended.
    """
    steps_per_waypoint = round(expert.cfg.waypoint_period / env.step_dt)
    origins = env.scene.env_origins
    obs, _ = env.reset(seed=reset_seed)
    expert.reset(observe(env, joint_ids))
    working = expert.phase.clone()
    records: list[list[dict[str, np.ndarray]]] = [[] for _ in range(env.num_envs)]
    ended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    while not ended.all():
        state = observe(env, joint_ids)
        command = expert.command(state)
        step = {key: value.cpu().numpy() for key, value in obs["policy"].items()}
        reward = torch.zeros(env.num_envs, device=env.device)
        for reference in interpolate_poses(state.tcp_pose, command.waypoint, steps_per_waypoint):
            obs, step_reward, terminated, truncated, _ = env.step(pose_to_action(reference, command.gripper, origins))
            expert.end_episodes(terminated, truncated)
            reward += step_reward
        step["actions"] = pose_to_action(command.waypoint, command.gripper, origins).cpu().numpy()
        step["rewards"] = reward.cpu().numpy()
        for i in (~ended).nonzero()[:, 0].tolist():
            records[i].append({key: value[i] for key, value in step.items()})
        coordinates = state.joint_pos.cpu().numpy()
        finished = expert.done & ~ended
        for i in finished.nonzero()[:, 0].tolist():
            yield Demonstration.from_steps(records[i]), bool(expert.success[i]), coordinates[i], Phase(int(working[i]))
        ended |= finished
        expert.advance(state)
        working = torch.where(expert.phase < Phase.DONE, expert.phase, working)
    for i in (~ended).nonzero()[:, 0].tolist():
        yield Demonstration.from_steps(records[i]), bool(expert.success[i]), coordinates[i], Phase(int(working[i]))


def summary(coordinate_names: tuple[str, ...], lengths: list[int], coordinates: list[np.ndarray]) -> str:
    """Episode lengths and final object coordinates (mean and range) of successful episodes."""
    if not lengths:
        return "no successful episode"
    q = np.stack(coordinates)
    final = ", ".join(
        f"{name} {q[:, j].mean():.3f} ({q[:, j].min():.3f} to {q[:, j].max():.3f})"
        for j, name in enumerate(coordinate_names)
    )
    return f"length {np.mean(lengths):.1f} steps ({min(lengths)} to {max(lengths)}); final {final}"


def collect(env: ManagerBasedRLEnv, expert: ScriptedExpert, joint_ids: list[int], writer: DemoWriter) -> Counter:
    """Run batches until ``num_demos`` successful demonstrations are written; count episodes and batches."""
    totals = Counter()
    while writer.num_demos < num_demos:
        if totals["episodes"] >= MAX_EPISODES_PER_DEMO * num_demos:
            raise RuntimeError(f"only {writer.num_demos} of {totals['episodes']} episodes succeeded")
        reset_seed = SEED_STRIDE * seed + totals["batches"]
        lengths, finals, failures = [], [], Counter()
        for demo, success, coordinates, phase in run_batch(env, expert, joint_ids, reset_seed):
            totals["episodes"] += 1
            if not success:
                failures[phase.name] += 1
                continue
            lengths.append(len(demo))
            finals.append(coordinates)
            writer.write(demo)
            if writer.num_demos == num_demos:
                break
        failed = ", ".join(f"{count} in {name}" for name, count in failures.items()) or "none"
        print(
            f"[batch {totals['batches']}, reset seed {reset_seed}] {len(lengths)} successful, failed: {failed}; "
            f"{summary(expert.coordinates, lengths, finals)}; written {writer.num_demos}/{num_demos}",
            flush=True,
        )
        totals["batches"] += 1
    return totals


def main() -> None:
    env_cfg = COLLECT_ENV_CFGS[args.object]()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env = ManagerBasedRLEnv(cfg=env_cfg)
    joint_ids = [env.scene["object"].find_joints(name)[0][0] for name in EXPERT_JOINTS[args.object]]
    expert = make_expert(args.object, env.num_envs, env.device, seed=seed, **settings["expert"])

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = DemoWriter(out, task_id(args.object, collection=True))
    try:
        totals = collect(env, expert, joint_ids, writer)
        path = writer.close(settings["train_fraction"])
    except BaseException:
        writer.discard()
        raise
    print(
        f"{args.object}: {writer.num_demos} demonstrations from {totals['episodes']} finished episodes "
        f"in {totals['batches']} batches -> {path}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
    simulation_app.close()
