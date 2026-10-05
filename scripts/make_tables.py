"""Paper tables from scored trials: simulation results (Table 1), multi-task results, diagnostics.

usage: python scripts/make_tables.py --single_task scores.jsonl [--multi_task scores_mt.jsonl]
                                     [--out tables.md] [--csv summary.csv]

The inputs are per-trial tables written by ``scripts/score.py``, each holding the Iso, Ours
and Oracle runs of one policy setting on the four objects. Per object, the success rate (SR)
counts all trials of each controller; every other entry is the mean over the common-success
set, the episodes in which all three controllers succeed (paired by episode index) and reach
grasp closure. SE is reported for Ours. A complement whose reference dimension is zero on every trial (the angular
complement of the universal joint) is printed as ``--``. Bold marks the better of Iso and Ours
at the printed precision; an empty common-success set gives ``nan`` means and no bold.
``--csv`` also writes the unrounded per-controller summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from con_dec_imp_sim import OBJECTS
from con_dec_imp_sim.records import CONTROLLERS

LOADS = (("ICF", "force"), ("PCF", "force"), ("ICM", "moment"), ("PCM", "moment"))
"""Complement loads and the block whose complement they measure."""
DIAGNOSTICS = {"lambda_f": 2, "lambda_m": 3, "beta_f": 2, "beta_m": 2}
"""Diagnostic fractions and their printed decimals."""
MEANS = tuple(name for name, _ in LOADS) + tuple(DIAGNOSTICS) + ("SE",)


@dataclass(frozen=True)
class ObjectSummary:
    """Aggregates of the three controllers on one object."""

    object: str
    trials: dict[str, int]
    successes: dict[str, int]
    common: tuple[int, ...]
    means: dict[str, dict[str, float]]
    """Controller -> metric -> mean over the common-success set (SE for Ours only)."""
    empty_complement: dict[str, bool]
    """Block (``force``/``moment``) -> complement of the reference is zero-dimensional."""


def read_trials(path: Path) -> list[dict[str, Any]]:
    """Rows of a per-trial table written by ``scripts/score.py``."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else math.nan


PAIRING_KEYS = ("checkpoint", "checkpoint_sha256", "seed", "num_envs")
"""Run identity that the three controllers of an object must share for paired trials."""


def _check_paired(name: str, runs: dict[str, dict[int, dict[str, Any]]]) -> None:
    """Trials are paired by episode index, which is meaningful only for runs of the same
    checkpoint, seed and number of environments (the same initial configurations)."""
    for key in PAIRING_KEYS:
        values = {c: {row.get(key) for row in runs[c].values()} for c in CONTROLLERS}
        if any(len(v) != 1 for v in values.values()) or len(set.union(*values.values())) != 1:
            found = ", ".join(f"{c}: {sorted(map(str, v))}" for c, v in values.items())
            raise ValueError(f"{name}: the controllers' runs differ in {key} ({found}); trials cannot be paired")
    episodes = {c: set(runs[c]) for c in CONTROLLERS}
    if len({frozenset(e) for e in episodes.values()}) != 1:
        raise ValueError(f"{name}: the controllers' runs hold different episodes; trials cannot be paired")


def summarize(rows: list[dict[str, Any]]) -> list[ObjectSummary]:
    """Per-object summaries, in the paper's object order."""
    trials: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for row in rows:
        by_episode = trials.setdefault(row["object"], {}).setdefault(row["controller"], {})
        if row["episode"] in by_episode:
            raise ValueError(
                f"{row['object']}/{row['controller']}: episode {row['episode']} appears twice; "
                "score one run per object and controller"
            )
        by_episode[row["episode"]] = row
    summaries = []
    for name in (o for o in OBJECTS if o in trials):
        runs = trials[name]
        missing = [c for c in CONTROLLERS if c not in runs]
        if missing:
            raise ValueError(f"{name}: no trials of {', '.join(missing)}")
        _check_paired(name, runs)
        succeeded = sorted(
            ep for ep in runs["iso"] if all(ep in runs[c] and runs[c][ep]["success"] for c in CONTROLLERS)
        )
        # A trial can succeed without a detected grasp closure (the policy reopens the gripper
        # before the gains are installed and grasps again); it has no evaluation window.
        common = tuple(ep for ep in succeeded if all(runs[c][ep]["ICF"] is not None for c in CONTROLLERS))
        unscored = sorted(set(succeeded) - set(common))
        if unscored:
            print(f"{name}: trials {unscored} succeeded without a grasp closure and are not averaged", file=sys.stderr)
        means = {
            c: {
                m: _mean([runs[c][ep][m] for ep in common])
                for m in MEANS
                if m != "SE" or c == "ours"
            }
            for c in CONTROLLERS
        }
        empty = {
            block: bool(common)
            and all(runs[c][ep][f"{block}_complement_dim"] == 0 for c in CONTROLLERS for ep in common)
            for block in ("force", "moment")
        }
        summaries.append(
            ObjectSummary(
                object=name,
                trials={c: len(runs[c]) for c in CONTROLLERS},
                successes={c: sum(row["success"] for row in runs[c].values()) for c in CONTROLLERS},
                common=common,
                means=means,
                empty_complement=empty,
            )
        )
    return summaries


def number(value: float, decimals: int, strip_zero: bool = False) -> str:
    """Fixed-point text; ``strip_zero`` prints ``0.29`` as ``.29``."""
    text = f"{value:.{decimals}f}"
    return text[1:] if strip_zero and text.startswith("0.") else text


def load_text(metric: str, value: float) -> str:
    """Paper rounding of a complement load: one decimal from 10 up, else two; moments without leading zero."""
    return number(value, 1 if abs(value) >= 10 else 2, strip_zero=metric in ("ICM", "PCM"))


def triple(texts: list[str], higher_is_better: bool = False) -> str:
    """``Iso / Ours / Oracle`` with the better of Iso and Ours in bold (no bold on a printed tie or a NaN)."""
    iso, ours, oracle = texts
    a, b = float(iso), float(ours)
    if a != b and not (math.isnan(a) or math.isnan(b)):
        if (b > a) == higher_is_better:
            ours = f"**{ours}**"
        else:
            iso = f"**{iso}**"
    return f"{iso} / {ours} / {oracle}"


def results_table(summaries: list[ObjectSummary]) -> list[str]:
    """Markdown rows of the results table (layout of Table 1)."""
    lines = [
        "| Task | SR (%) | SE (°) | ICF (N·s) | PCF (N) | ICM (N·m·s) | PCM (N·m) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        sr = triple([f"{100 * s.successes[c] / s.trials[c]:.0f}" for c in CONTROLLERS], higher_is_better=True)
        cells = [sr, number(s.means["ours"]["SE"], 2)]
        for metric, block in LOADS:
            if s.empty_complement[block]:
                cells.append("-- / -- / --")
            else:
                cells.append(triple([load_text(metric, s.means[c][metric]) for c in CONTROLLERS]))
        lines.append(f"| {s.object.capitalize()} | " + " | ".join(cells) + " |")
    return lines


def diagnostics_table(sections: list[tuple[str, list[ObjectSummary]]]) -> list[str]:
    """Markdown rows of the diagnostics table (n_c, load fractions, feedback fractions)."""
    lines = ["| Task | n_c | λ_f | λ_m | β_f | β_m |", "|---|---|---|---|---|---|"]
    for title, summaries in sections:
        lines.append(f"| *{title}* | | | | | |")
        for s in summaries:
            cells = [str(len(s.common))]
            for metric, decimals in DIAGNOSTICS.items():
                cells.append(" / ".join(number(s.means[c][metric], decimals, strip_zero=True) for c in CONTROLLERS))
            lines.append(f"| {s.object.capitalize()} | " + " | ".join(cells) + " |")
    return lines


RESULTS_CAPTION = (
    "Triples are Iso / Ours / Oracle; SR counts all trials, and the other entries are means over the "
    "trials in which all three controllers succeed and reach grasp closure; SE reports Ours only; bold marks the better value "
    "between Iso and Ours; dashes indicate structurally empty complement components."
)
DIAGNOSTICS_CAPTION = (
    "Triples are Iso / Ours / Oracle. n_c is the number of trials in which all three controllers succeed "
    "and reach grasp closure; λ_f, λ_m are "
    "the force/moment complement load fractions and β_f, β_m the force/moment feedback feasible "
    "fractions, means over the common-success set."
)


def write_csv(path: Path, sections: list[tuple[str, list[ObjectSummary]]]) -> None:
    """Unrounded summaries, one row per policy setting, object and controller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "object", "controller", "trials", "successes", "SR", "n_c", *MEANS])
        for title, summaries in sections:
            for s in summaries:
                for c in CONTROLLERS:
                    writer.writerow(
                        [title, s.object, c, s.trials[c], s.successes[c], 100 * s.successes[c] / s.trials[c],
                         len(s.common), *(s.means[c].get(m, "") for m in MEANS)]
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--single_task", type=Path, help="per-trial table of the single-task policies")
    parser.add_argument("--multi_task", type=Path, help="per-trial table of the multi-task policy")
    parser.add_argument("--out", type=Path, help="also write the markdown tables here")
    parser.add_argument("--csv", type=Path, help="also write the unrounded summaries here")
    args = parser.parse_args()
    inputs = [("Single-task policies", args.single_task), ("Multi-task policy", args.multi_task)]
    sections = [(title, summarize(read_trials(path))) for title, path in inputs if path is not None]
    if not sections:
        parser.error("give --single_task and/or --multi_task")

    lines: list[str] = []
    for title, summaries in sections:
        lines += [f"## Simulation results: {title.lower()}", "", RESULTS_CAPTION, "", *results_table(summaries), ""]
    lines += ["## Simulation rollout diagnostics", "", DIAGNOSTICS_CAPTION, "", *diagnostics_table(sections)]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.csv is not None:
        write_csv(args.csv, sections)


if __name__ == "__main__":
    main()
