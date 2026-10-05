import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from con_dec_imp_sim.records import Episode, Run, Trace, trace_file, write_run, write_trace
from con_dec_imp_sim.reference import reference_basis

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    module_spec = importlib.util.spec_from_file_location(f"scripts_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


score = load_script("score")
make_tables = load_script("make_tables")

OBJECT_POSE = np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
TCP_POSE = np.array([0.6, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0])
DT = 0.01
T = 11  # sample 0 lies outside the window


def revolute_directions():
    """Feasible moment and force directions of the revolute door at TCP_POSE (body = world axes)."""
    hinge = OBJECT_POSE[:3] + np.array([-0.215, -0.15, 0.0])
    lever = np.cross([0.0, 0.0, 1.0], TCP_POSE[:3] - hinge)
    return np.array([0.0, 0.0, 1.0]), lever / np.linalg.norm(lever)


def constant_trace(moment, force, feedback=None, basis=None, closure=True) -> Trace:
    wrench = np.tile(np.concatenate((moment, force)), (T, 1))
    wrench[0] = 1e3  # excluded by the window
    window = np.arange(T) > 0
    commanded = wrench if feedback is None else np.tile(feedback, (T, 1))
    return Trace(
        contact_wrench=wrench,
        commanded_wrench=commanded,
        feedforward_wrench=np.zeros((T, 6)),
        window=window,
        dt=DT,
        object_initial_pose=OBJECT_POSE,
        closure_tcp_pose=TCP_POSE if closure else None,
        closure_object_pose=OBJECT_POSE if closure else None,
        installed_basis=basis,
    )


def write_runs(root: Path, object_name: str, traces: dict, success: dict) -> None:
    """``traces[controller][episode]`` and ``success[controller][episode]``."""
    for controller, by_episode in traces.items():
        run_dir = root / object_name / controller
        episodes = []
        for index, trace in by_episode.items():
            write_trace(run_dir / trace_file(index), trace)
            episodes.append(
                Episode(
                    index=index,
                    success=success[controller][index],
                    steps=500,
                    closure_index=24,
                    trace=trace_file(index),
                )
            )
        write_run(
            run_dir,
            Run(object=object_name, controller=controller, checkpoint=f"{object_name}.pth",
                checkpoint_sha256="ab" * 32, seed=0, settings={},
                episodes=tuple(episodes)),
        )


def test_score_trace_hand_computed():
    z, u_par = revolute_directions()
    trace = constant_trace(moment=np.array([1.0, 0.0, 2.0]), force=3.0 * z + 4.0 * u_par,
                           feedback=np.concatenate((2.0 * z, 5.0 * u_par)))
    out = score.score_trace("revolute", "iso", trace)
    n = T - 1
    assert out["window_samples"] == n
    assert (out["force_complement_dim"], out["moment_complement_dim"]) == (2, 2)
    assert out["ICF"] == pytest.approx(3.0 * n * DT)
    assert out["PCF"] == pytest.approx(3.0)
    assert out["ICM"] == pytest.approx(1.0 * n * DT)
    assert out["PCM"] == pytest.approx(1.0)
    assert out["lambda_f"] == pytest.approx(3.0 / 7.0)
    assert out["lambda_m"] == pytest.approx(1.0 / 3.0)
    assert out["beta_f"] == pytest.approx(1.0) and out["beta_m"] == pytest.approx(1.0)
    assert out["SE"] is None


def test_subspace_error_of_ours():
    U_ref = reference_basis("cylindrical", torch.from_numpy(TCP_POSE), torch.from_numpy(OBJECT_POSE)).numpy()
    zero = np.zeros(3)

    def se(basis):
        return score.score_trace("cylindrical", "ours", constant_trace(zero, zero, basis=basis))["SE"]

    assert se(U_ref) == pytest.approx(0.0, abs=1e-5)
    assert se(U_ref[:, :1]) == 90.0  # a lower dimension than the reference
    assert se(np.zeros((6, 0))) == 90.0
    with pytest.raises(ValueError):
        score.score_trace("cylindrical", "ours", constant_trace(zero, zero))


def test_score_run_without_closure(tmp_path):
    zero = np.zeros(3)
    traces = {"iso": {0: constant_trace(zero, zero), 1: constant_trace(zero, zero, closure=False)}}
    write_runs(tmp_path, "planar", traces, {"iso": {0: True, 1: False}})
    rows = score.score_run(tmp_path / "planar" / "iso")
    assert [r["episode"] for r in rows] == [0, 1]
    assert rows[0]["ICF"] == 0.0 and rows[0]["force_complement_dim"] == 1
    assert all(rows[1][key] is None for key in score.TRIAL_COLUMNS)


def build_tables(tmp_path: Path) -> list:
    """Universal (empty moment complement) and revolute runs; Iso fails episode 1 of revolute."""
    z, u_par = revolute_directions()
    force_scale = {"iso": 4.0, "ours": 1.0, "oracle": 0.5}
    U_ref = {
        name: reference_basis(name, torch.from_numpy(TCP_POSE), torch.from_numpy(OBJECT_POSE)).numpy()
        for name in ("revolute", "universal")
    }
    for name in ("revolute", "universal"):
        traces = {
            c: {ep: constant_trace(np.array([0.3, 0.0, 0.1]), force_scale[c] * np.array([0.0, 0.0, 1.0]) + u_par,
                                   basis=U_ref[name] if c == "ours" else None)
                for ep in range(3)}
            for c in force_scale
        }
        success = {c: {0: True, 1: not (name == "revolute" and c == "iso"), 2: True} for c in force_scale}
        write_runs(tmp_path, name, traces, success)
    rows = [row for run_dir in sorted(tmp_path.glob("*/*")) for row in score.score_run(run_dir)]
    return make_tables.summarize(json.loads(json.dumps(rows)))


def test_tables(tmp_path):
    summaries = build_tables(tmp_path)
    assert [s.object for s in summaries] == ["revolute", "universal"]
    revolute, universal = summaries
    assert revolute.common == (0, 2) and universal.common == (0, 1, 2)
    assert revolute.successes == {"iso": 2, "ours": 3, "oracle": 3}
    assert universal.empty_complement == {"force": False, "moment": True}
    assert revolute.empty_complement == {"force": False, "moment": False}
    n = T - 1
    assert revolute.means["iso"]["ICF"] == pytest.approx(4.0 * n * DT)
    assert revolute.means["ours"]["SE"] == pytest.approx(0.0, abs=1e-5)
    lines = make_tables.results_table(summaries)
    assert lines[2] == (
        "| Revolute | 67 / **100** / 100 | 0.00 | 0.40 / **0.10** / 0.05 | 4.00 / **1.00** / 0.50 "
        "| .03 / .03 / .03 | .30 / .30 / .30 |"
    )
    assert lines[3].endswith("| -- / -- / -- | -- / -- / -- |")
    diagnostics = make_tables.diagnostics_table([("Single-task policies", summaries)])
    assert diagnostics[2] == "| *Single-task policies* | | | | | |"
    assert diagnostics[4].startswith("| Universal | 3 |") and ".000 / .000 / .000" in diagnostics[4]


def test_formatting():
    assert make_tables.load_text("ICF", 15.34) == "15.3"
    assert make_tables.load_text("ICF", 0.834) == "0.83"
    assert make_tables.load_text("PCM", 0.294) == ".29"
    assert make_tables.number(1.0, 2, strip_zero=True) == "1.00"
    assert make_tables.number(0.0, 3, strip_zero=True) == ".000"
    assert make_tables.triple(["7.18", "2.62", "2.13"]) == "7.18 / **2.62** / 2.13"
    assert make_tables.triple([".21", ".22", ".22"]) == "**.21** / .22 / .22"
    assert make_tables.triple(["99", "99", "99"], higher_is_better=True) == "99 / 99 / 99"
    assert make_tables.triple(["99", "100", "100"], higher_is_better=True) == "99 / **100** / 100"
    # An empty common-success set gives NaN means: nothing is bold.
    nan = make_tables.load_text("ICF", math.nan)
    assert make_tables.triple([nan, nan, nan]) == "nan / nan / nan"
    assert make_tables.triple(["1.00", nan, nan]) == "1.00 / nan / nan"


def test_duplicate_trials_rejected(tmp_path):
    rows = [{"object": "planar", "controller": "iso", "episode": 0, "success": True}] * 2
    with pytest.raises(ValueError):
        make_tables.summarize(rows)


def test_successful_trials_without_grasp_closure_are_not_averaged(capsys):
    row = dict.fromkeys(score.TRIAL_COLUMNS) | {"object": "planar", "episode": 0, "success": True}
    rows = [row | {"controller": c} for c in ("iso", "ours", "oracle")]
    (summary,) = make_tables.summarize(rows)
    assert summary.common == () and summary.successes["iso"] == 1
    assert "succeeded without a grasp closure" in capsys.readouterr().err
    assert math.isnan(make_tables._mean([]))


@pytest.mark.parametrize("key, value", [("seed", 1), ("checkpoint", "other.pth"), ("checkpoint_sha256", "f00"), ("num_envs", 50)])
def test_unpaired_runs_are_rejected(key, value):
    base = dict.fromkeys(score.TRIAL_COLUMNS) | {
        "object": "planar", "episode": 0, "success": True, "seed": 0,
        "checkpoint": "planar.pth", "checkpoint_sha256": "abc", "num_envs": 100,
    }
    rows = [base | {"controller": c} for c in ("iso", "ours", "oracle")]
    make_tables.summarize(rows)
    rows[1] = rows[1] | {key: value}
    with pytest.raises(ValueError, match="cannot be paired"):
        make_tables.summarize(rows)


def test_runs_with_different_episodes_are_rejected():
    base = dict.fromkeys(score.TRIAL_COLUMNS) | {"object": "planar", "success": True}
    rows = [base | {"controller": c, "episode": 0} for c in ("iso", "ours", "oracle")]
    rows.append(base | {"controller": "iso", "episode": 1})
    with pytest.raises(ValueError, match="different episodes"):
        make_tables.summarize(rows)
