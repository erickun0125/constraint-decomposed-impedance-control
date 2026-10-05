"""Score evaluation runs: per-trial wrench metrics and subspace error (Section 5, Appendix H).

Each trial is scored over its evaluation window against the reference feasible subspace of
its own grasp, built from the TCP and object poses at grasp closure
(:func:`con_dec_imp_sim.reference.reference_basis`). The subspace error (SE) of Ours compares
its installed basis with this reference in the metric of the object's fixed evaluation length
``alpha*``.

usage: python scripts/score.py <run or tree> [<run or tree> ...] --out scores.jsonl

Writes one JSON line per trial: the run identity (object, controller, checkpoint and its
SHA-256, seed, number of environments), the episode index, success, episode steps, the
selected dimension of Ours, the number of window samples, the force and moment complement
dimensions of the reference, ICF, PCF, ICM, PCM, lambda_f, lambda_m, beta_f, beta_m and, for
Ours, SE. Trials that did not reach grasp closure carry null metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from con_dec_imp.metrics import reference_projectors, subspace_error, wrench_metrics
from con_dec_imp_sim import OBJECTS
from con_dec_imp_sim.records import Trace, find_runs, read_run, read_trace
from con_dec_imp_sim.reference import EVALUATION_LENGTH, reference_basis

TRIAL_COLUMNS = (
    "window_samples",
    "force_complement_dim",
    "moment_complement_dim",
    "ICF",
    "PCF",
    "ICM",
    "PCM",
    "lambda_f",
    "lambda_m",
    "beta_f",
    "beta_m",
    "SE",
)
"""Per-trial score columns (null when the trial did not reach grasp closure)."""


def score_trace(object_name: str, controller: str, trace: Trace) -> dict[str, Any]:
    """Metrics of one trial that reached grasp closure."""
    U_ref = reference_basis(
        object_name, torch.from_numpy(trace.closure_tcp_pose), torch.from_numpy(trace.closure_object_pose)
    ).numpy()
    projectors = reference_projectors(U_ref)
    window = trace.window
    scores: dict[str, Any] = {
        "window_samples": int(window.sum()),
        "force_complement_dim": int(round(np.trace(projectors.force_perp))),
        "moment_complement_dim": int(round(np.trace(projectors.moment_perp))),
        **wrench_metrics(trace.contact_wrench[window], trace.dt, projectors, trace.feedback_wrench[window]),
        "SE": None,
    }
    if controller == "ours":
        if trace.installed_basis is None:
            raise ValueError("an Ours trace with grasp closure must hold the installed basis")
        scores["SE"] = subspace_error(trace.installed_basis, U_ref, EVALUATION_LENGTH[object_name])
    return scores


def score_run(run_dir: Path) -> list[dict[str, Any]]:
    """Per-trial rows of one run directory."""
    run = read_run(run_dir)
    if run.object not in OBJECTS:
        raise ValueError(f"{run_dir}: unknown object {run.object!r}")
    rows = []
    for episode in run.episodes:
        row: dict[str, Any] = {
            "object": run.object,
            "controller": run.controller,
            "checkpoint": run.checkpoint,
            "checkpoint_sha256": run.checkpoint_sha256,
            "seed": run.seed,
            "num_envs": run.settings.get("num_envs"),
            "episode": episode.index,
            "success": episode.success,
            "steps": episode.steps,
            "dim": None if episode.estimate is None else episode.estimate.dim,
            **dict.fromkeys(TRIAL_COLUMNS),
        }
        if episode.trace is not None:
            trace = read_trace(run_dir / episode.trace)
            if trace.has_closure:
                row.update(score_trace(run.object, run.controller, trace))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories or directory trees holding runs")
    parser.add_argument("--out", type=Path, required=True, help="per-trial table to write (JSON lines)")
    args = parser.parse_args()

    run_dirs = sorted({run_dir for root in args.runs for run_dir in find_runs(root)})
    if not run_dirs:
        raise SystemExit(f"no results.json found under {', '.join(map(str, args.runs))}")
    rows = []
    for run_dir in run_dirs:
        run_rows = score_run(run_dir)
        rows.extend(run_rows)
        successes = sum(row["success"] for row in run_rows)
        print(f"{run_dir}: {successes}/{len(run_rows)} successes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"{len(rows)} trials -> {args.out}")


if __name__ == "__main__":
    main()
