"""Evaluate Iso, Ours or Oracle on one simulated object with a Diffusion Policy checkpoint (Section 5.1).

usage: python scripts/evaluate.py --object cylindrical --controller ours \
           --checkpoint checkpoints/cylindrical.pth --out runs/single_task/cylindrical/ours --headless

Every environment runs one trial. All environments start from one batched reset, seeded with
``--seed``, and each trial runs until success or the 2000-step time-out; an environment whose
trial has ended keeps running (it is reset by the simulator) and is ignored. All environments
replan together every 16 waypoints (240 control steps) with the deterministic chunk of the
policy, executed through the control-rate references of
:func:`con_dec_imp_sim.evaluation.interpolate_chunk`.

All controllers start with the isotropic gains of Iso. At grasp closure
(:class:`con_dec_imp_sim.evaluation.GraspClosureDetector`):

* Ours samples N = 128 trajectories for that environment alone (one batched DDIM pass with a
  per-trial generator seeded with ``seed + 1 + trial index``), estimates the feasible subspace
  from the waypoints at which each sample commands a closed gripper, and installs the
  decomposed gains (Algorithm 1); with m = 0 it keeps the isotropic gains;
* Oracle installs the decomposed gains of the reference subspace at the measured grasp;
* Iso keeps the isotropic gains.

Installed gains blend in over 50 control steps and act while the fingers are closed; a
release command returns the environment to the isotropic gains. The run is written in the
record format of :mod:`con_dec_imp_sim.records` (score it with ``scripts/score.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from con_dec_imp_sim import OBJECTS

PAPER_SEED = 1296387670
"""Seed of the paper's evaluation; reproduces its initial object configurations."""

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--object", required=True, choices=OBJECTS)
parser.add_argument("--controller", required=True, choices=("iso", "ours", "oracle"))
parser.add_argument("--checkpoint", type=Path, required=True, help="release checkpoint of the policy")
parser.add_argument("--out", type=Path, required=True, help="run directory to write")
parser.add_argument(
    "--num_envs",
    type=int,
    default=100,
    help="trials, one per environment, all reset together (the paper's initial configurations need 100)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=PAPER_SEED,
    help="seed of the batched reset (Ours samples each trial's ensemble with seed + 1 + trial index); "
    "the default is the seed of the paper's evaluation",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import hashlib  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from con_dec_imp.estimator import gains_from_estimate, twist_matrix  # noqa: E402
from con_dec_imp.gains import decomposed_gains, inertia_metric  # noqa: E402
from con_dec_imp.policy.actions import GRIPPER_CLOSED_THRESHOLD, chunk_to_transforms  # noqa: E402
from con_dec_imp.policy.diffusion_policy import DiffusionPolicy  # noqa: E402
from con_dec_imp.settings import CONTROL, ESTIMATION, GAINS  # noqa: E402
from con_dec_imp.subspace import estimate_subspace  # noqa: E402
from con_dec_imp_sim.evaluation import GraspClosureDetector, interpolate_chunk  # noqa: E402
from con_dec_imp_sim.records import Episode, Estimate, Run, Trace, trace_file, write_run, write_trace  # noqa: E402
from con_dec_imp_sim.reference import reference_basis  # noqa: E402
from con_dec_imp_sim.sensors import ContactWrenchReader  # noqa: E402
from con_dec_imp_sim.tasks import EVAL_ENV_CFGS, EVAL_EPISODE_STEPS  # noqa: E402

STEPS_PER_WAYPOINT = round(CONTROL.control_rate / CONTROL.policy_rate)
REPLAN_STEPS = CONTROL.executed_waypoints * STEPS_PER_WAYPOINT


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    """Seed the global generators that the reset events draw from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tcp_poses(env: ManagerBasedRLEnv) -> torch.Tensor:
    """TCP poses ``(N, 7)`` in the world frame."""
    frame = env.scene["gripper_frame"].data
    return torch.cat((frame.target_pos_w[:, 0], frame.target_quat_w[:, 0]), dim=-1)


def object_poses(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Object root poses ``(N, 7)`` in the world frame."""
    data = env.scene["object"].data
    return torch.cat((data.root_pos_w, data.root_quat_w), dim=-1)


def synchronize(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class Evaluation:
    """One controller on one object: one trial per environment from a single batched reset."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        policy: DiffusionPolicy,
        object_name: str,
        controller: str,
        seed: int,
        out_dir: Path,
    ):
        self.env = env
        self.policy = policy
        self.object_name = object_name
        self.controller = controller
        self.seed = seed
        self.out_dir = out_dir
        self.term = env.action_manager.get_term("impedance")
        self.reader = ContactWrenchReader(env)
        self.detector = GraspClosureDetector(env.num_envs, env.device)
        self.metric = inertia_metric(device=env.device)

    def run(self) -> list[Episode]:
        """Reset all environments and run until every trial has ended; returns the trials by index."""
        env, term, n, device = self.env, self.term, self.env.num_envs, self.env.device
        seed_all(self.seed)
        obs, _ = env.reset()
        self.detector.reset()
        initial_object_poses = object_poses(env).clone()
        initial_object_poses[:, :3] -= env.scene.env_origins

        rows = env.max_episode_length
        contact = torch.zeros(rows, n, 6, device=device)
        commanded = torch.zeros(rows, n, 6, device=device)
        feedforward = torch.zeros(rows, n, 6, device=device)
        window = torch.zeros(rows, n, dtype=torch.bool, device=device)
        trace_start = torch.full((n,), -1, dtype=torch.long)
        closure_step = torch.full((n,), -1, dtype=torch.long)
        closure_tcp = torch.zeros(n, 7, dtype=torch.float64)
        closure_object = torch.zeros(n, 7, dtype=torch.float64)
        installed_basis: dict[int, np.ndarray] = {}
        estimates: dict[int, Estimate] = {}
        episodes: list[Episode] = []
        done = torch.zeros(n, dtype=torch.bool, device=device)
        start_time = time.time()

        step = 0
        while not done.all():
            if step >= rows:
                raise RuntimeError("a trial exceeded the evaluation time-out")
            if step % REPLAN_STEPS == 0:
                chunk = self.policy.predict(obs["policy"])
                references = interpolate_chunk(chunk[:, : CONTROL.executed_waypoints + 1], STEPS_PER_WAYPOINT)
            action = references[:, step % REPLAN_STEPS]
            closed, released = self.detector.command(action[:, 9])
            # A close command starts a grasp attempt: the trace and the grasp-closure state of the
            # environment restart there (a later attempt replaces an earlier, released one).
            for i in (closed & ~done).nonzero()[:, 0].tolist():
                trace_start[i] = step
                closure_step[i] = -1
                installed_basis.pop(i, None)
                estimates.pop(i, None)
            term.clear_decomposed_gains(released)

            obs, _, terminated, truncated, _ = env.step(action)
            contact[step] = self.reader.read()
            commanded[step] = term.last_wrench
            feedforward[step] = term.last_feedforward_wrench
            window[step] = self.detector.articulating

            # Environments whose trial ends at this step were already reset by the simulator.
            reached = self.detector.update(term.fingers_closed, ~done & ~(terminated | truncated))
            for i in reached.nonzero()[:, 0].tolist():
                closure_step[i] = step
                closure_tcp[i] = tcp_poses(env)[i].double().cpu()
                closure_object[i] = object_poses(env)[i].double().cpu()
                if self.controller == "ours":
                    installed_basis[i], estimates[i] = self.install_ours(obs["policy"], i)
                elif self.controller == "oracle":
                    self.install_oracle(closure_tcp[i], closure_object[i], i)

            finished = ((terminated | truncated) & ~done).nonzero()[:, 0].tolist()
            for i in finished:
                trace_name = None
                if trace_start[i] >= 0:
                    # The terminating step is not recorded: the simulator resets the
                    # environment within that step, before its wrench could be read.
                    rows_i = slice(int(trace_start[i]), step)
                    reached_closure = bool(closure_step[i] >= 0)
                    trace_name = trace_file(i)
                    write_trace(
                        self.out_dir / trace_name,
                        Trace(
                            contact_wrench=contact[rows_i, i].cpu().numpy(),
                            commanded_wrench=commanded[rows_i, i].cpu().numpy(),
                            feedforward_wrench=feedforward[rows_i, i].cpu().numpy(),
                            window=window[rows_i, i].cpu().numpy(),
                            dt=env.step_dt,
                            object_initial_pose=initial_object_poses[i].double().cpu().numpy(),
                            closure_tcp_pose=closure_tcp[i].numpy() if reached_closure else None,
                            closure_object_pose=closure_object[i].numpy() if reached_closure else None,
                            installed_basis=installed_basis.get(i),
                        ),
                    )
                episodes.append(
                    Episode(
                        index=i,
                        success=bool(terminated[i]),
                        steps=step + 1,
                        closure_index=int(closure_step[i] - trace_start[i]) if closure_step[i] >= 0 else None,
                        trace=trace_name,
                        estimate=estimates.get(i),
                    )
                )
            done[finished] = True
            step += 1

        print(f"{step} control steps, {time.time() - start_time:.0f} s", flush=True)
        return sorted(episodes, key=lambda e: e.index)

    def install_ours(self, observation: dict, env_id: int) -> tuple[np.ndarray, Estimate]:
        """Algorithm 1 for one environment: sample the ensemble, estimate, install the gains."""
        generator = torch.Generator().manual_seed(self.seed + 1 + env_id)
        single = {key: value[env_id : env_id + 1] for key, value in observation.items()}
        synchronize(self.env.device)
        start = time.perf_counter()
        chunks = self.policy.sample_ensemble(single, [generator])
        grasped = chunks[0][..., 9] > GRIPPER_CLOSED_THRESHOLD
        W = twist_matrix(chunk_to_transforms(chunks[0]), ESTIMATION.dt, grasped)
        if W.shape[-1] < 6:
            # Too few sampled waypoints with a closed gripper to estimate six principal
            # directions: no estimate, the isotropic gains stay.
            synchronize(self.env.device)
            record = Estimate(dim=0, alpha=None, rms_speeds=(), time=time.perf_counter() - start)
            return np.zeros((6, 0)), record
        estimate = estimate_subspace(W)
        if int(estimate.dim) > 0:
            K, D = gains_from_estimate(estimate, self.metric)
            self.term.set_decomposed_gains(K, D, [env_id])
        synchronize(self.env.device)
        elapsed = time.perf_counter() - start
        record = Estimate(
            dim=int(estimate.dim),
            alpha=float(estimate.alpha),
            rms_speeds=tuple(estimate.rms_speeds.tolist()),
            time=elapsed,
        )
        return estimate.basis().double().cpu().numpy(), record

    def install_oracle(self, tcp_pose: torch.Tensor, object_pose: torch.Tensor, env_id: int) -> None:
        """Decomposed gains of the reference subspace at the measured grasp."""
        basis = reference_basis(self.object_name, tcp_pose, object_pose)
        K, D = decomposed_gains(inertia_metric(dtype=torch.float64), basis)
        self.term.set_decomposed_gains(K, D, [env_id])


def main() -> None:
    cfg = EVAL_ENV_CFGS[args.object]()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=cfg)
    torch.backends.cudnn.benchmark = False
    policy = DiffusionPolicy.load(args.checkpoint, env.device)

    episodes = Evaluation(env, policy, args.object, args.controller, args.seed, args.out).run()
    write_run(
        args.out,
        Run(
            object=args.object,
            controller=args.controller,
            checkpoint=args.checkpoint.name,
            checkpoint_sha256=file_sha256(args.checkpoint),
            seed=args.seed,
            settings={
                "estimation": asdict(ESTIMATION),
                "gains": asdict(GAINS),
                "control": asdict(CONTROL),
                "num_envs": args.num_envs,
                "episode_steps": EVAL_EPISODE_STEPS,
            },
            episodes=tuple(episodes),
        ),
    )
    successes = sum(e.success for e in episodes)
    print(f"{args.object} / {args.controller}: {successes}/{len(episodes)} successes -> {args.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
