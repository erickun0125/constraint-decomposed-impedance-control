"""Rollout records of the simulation evaluation, written by the evaluation and read by the scorer.

A run is one controller on one object with one checkpoint and one evaluation seed. Its
directory holds ``results.json`` and one compressed trace per episode::

    <run>/results.json
    <run>/traces/episode_000.npz
    ...

``results.json``::

    {
      "format_version": 1,
      "object": "revolute",          # revolute | cylindrical | planar | universal
      "controller": "ours",          # iso | ours | oracle
      "checkpoint": "revolute.pth",  # file name
      "checkpoint_sha256": "5f12...",  # SHA-256 of the checkpoint file
      "seed": 0,                     # evaluation seed
      "settings": {...},             # hyperparameters used (informational)
      "episodes": [
        {
          "index": 0,                # episode index; pairs the trials of the three controllers
          "success": true,           # success termination before the time-out
          "steps": 907,              # control steps from the reset to the end of the episode
          "closure_index": 24,       # trace sample at which grasp closure was detected and the
                                     # gains were installed (null if the episode ended first)
          "trace": "traces/episode_000.npz",  # null if the gripper was never commanded to close
          "estimate": {              # Ours only, null otherwise and before grasp closure
            "dim": 1,                # selected feasible dimension m
            "alpha": 0.2385,         # characteristic length [m] (null: fewer than six grasped twists)
            "rms_speeds": [...],     # sqrt(lambda_k), k = 1..6, decreasing
            "time": 0.071            # sampling and estimation time [s]
          }
        },
        ...
      ]
    }

Episode trace ``traces/episode_XXX.npz``. The ``T`` samples are read after every environment
step, from the close command of the episode's last grasp attempt up to the step before the
episode ends (at the terminating step the simulator has already reset the environment)::

    contact_wrench       float32 (T, 6)  gripper-object contact wrench acting on the gripper,
                                         TCP body frame, (moment, force) [N m, N]
    commanded_wrench     float32 (T, 6)  wrench commanded by the impedance controller, same frame
    feedforward_wrench   float32 (T, 6)  feedforward part of the commanded wrench (zero here:
                                         the controller uses no feedforward)
    window               bool    (T,)    evaluation window: the articulation phase, from the
                                         step after the close command
    dt                   float64 ()      sample interval [s]
    object_initial_pose  float64 (7,)    object root pose after the reset, environment frame
    closure_tcp_pose     float64 (7,)    TCP pose at grasp closure, world frame         } absent if
    closure_object_pose  float64 (7,)    object root pose at grasp closure, world frame } not reached
    installed_basis      float64 (6, m)  feasible basis installed by Ours, rows (omega, v)  (Ours only)

Poses are ``(x, y, z, qw, qx, qy, qz)``. The grasp-closure poses are recorded for every
controller at the same instant, so each trial is scored against the reference subspace of
its own grasp (Appendix H, "Evaluation metrics").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1
RESULTS_FILE = "results.json"
CONTROLLERS = ("iso", "ours", "oracle")


@dataclass(frozen=True)
class Estimate:
    """Constraint estimate of Ours at grasp closure (Algorithm 1)."""

    dim: int
    alpha: float | None
    """Characteristic length; None when the ensemble gave fewer than six twists of waypoints with a
    closed gripper (no estimate; the isotropic gains stay)."""
    rms_speeds: tuple[float, ...]
    time: float


@dataclass(frozen=True)
class Episode:
    """One trial of a run."""

    index: int
    success: bool
    steps: int
    closure_index: int | None = None
    trace: str | None = None
    estimate: Estimate | None = None


@dataclass(frozen=True)
class Run:
    """One controller evaluated on one object."""

    object: str
    controller: str
    checkpoint: str
    checkpoint_sha256: str
    seed: int
    settings: dict[str, Any]
    episodes: tuple[Episode, ...]


@dataclass(frozen=True)
class Trace:
    """Signals of one episode needed for scoring (see the module docstring)."""

    contact_wrench: np.ndarray
    commanded_wrench: np.ndarray
    feedforward_wrench: np.ndarray
    window: np.ndarray
    dt: float
    object_initial_pose: np.ndarray
    closure_tcp_pose: np.ndarray | None = None
    closure_object_pose: np.ndarray | None = None
    installed_basis: np.ndarray | None = None

    @property
    def feedback_wrench(self) -> np.ndarray:
        """Feedback part of the commanded wrench ``(T, 6)``, feedforward excluded."""
        return self.commanded_wrench.astype(np.float64) - self.feedforward_wrench.astype(np.float64)

    @property
    def has_closure(self) -> bool:
        """Whether grasp closure was reached (and the closure poses recorded)."""
        return self.closure_tcp_pose is not None


_WRENCHES = ("contact_wrench", "commanded_wrench", "feedforward_wrench")


def trace_file(index: int) -> str:
    """Path of an episode trace relative to its run directory."""
    return f"traces/episode_{index:03d}.npz"


def _check_trace(trace: Trace) -> None:
    if trace.window.ndim != 1:
        raise ValueError(f"window must have shape (T,); got {trace.window.shape}")
    T = trace.window.shape[0]
    for name in _WRENCHES:
        if getattr(trace, name).shape != (T, 6):
            raise ValueError(f"{name} must have shape ({T}, 6); got {getattr(trace, name).shape}")
    if not trace.dt > 0:
        raise ValueError(f"dt must be positive; got {trace.dt}")
    for name in ("object_initial_pose", "closure_tcp_pose", "closure_object_pose"):
        pose = getattr(trace, name)
        if pose is not None and pose.shape != (7,):
            raise ValueError(f"{name} must have shape (7,); got {pose.shape}")
    if (trace.closure_tcp_pose is None) != (trace.closure_object_pose is None):
        raise ValueError("closure_tcp_pose and closure_object_pose must be given together")
    basis = trace.installed_basis
    if basis is not None and (basis.ndim != 2 or basis.shape[0] != 6 or basis.shape[1] > 6):
        raise ValueError(f"installed_basis must have shape (6, m), m <= 6; got {basis.shape}")


def _normalized(trace: Trace) -> Trace:
    """Copy of ``trace`` with the stored dtypes, validated."""

    def cast(value: Any, dtype: type) -> np.ndarray | None:
        return None if value is None else np.asarray(value, dtype=dtype)

    out = Trace(
        contact_wrench=cast(trace.contact_wrench, np.float32),
        commanded_wrench=cast(trace.commanded_wrench, np.float32),
        feedforward_wrench=cast(trace.feedforward_wrench, np.float32),
        window=cast(trace.window, bool),
        dt=float(trace.dt),
        object_initial_pose=cast(trace.object_initial_pose, np.float64),
        closure_tcp_pose=cast(trace.closure_tcp_pose, np.float64),
        closure_object_pose=cast(trace.closure_object_pose, np.float64),
        installed_basis=cast(trace.installed_basis, np.float64),
    )
    _check_trace(out)
    return out


def write_trace(path: str | Path, trace: Trace) -> None:
    """Write an episode trace as a compressed npz; absent optional fields are omitted."""
    trace = _normalized(trace)
    arrays = {f.name: getattr(trace, f.name) for f in fields(trace) if getattr(trace, f.name) is not None}
    arrays["dt"] = np.float64(trace.dt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez_compressed(f, **arrays)


def read_trace(path: str | Path) -> Trace:
    """Read an episode trace written by :func:`write_trace`."""
    with np.load(path) as data:
        stored = {name: data[name] for name in data.files}
    stored["dt"] = float(stored["dt"])
    return _normalized(Trace(**stored))


def write_run(run_dir: str | Path, run: Run) -> Path:
    """Write ``results.json`` of a run; returns its path."""
    if run.controller not in CONTROLLERS:
        raise ValueError(f"controller must be one of {CONTROLLERS}; got {run.controller!r}")
    content = {"format_version": FORMAT_VERSION, **asdict(run)}
    path = Path(run_dir) / RESULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")
    return path


def read_run(run_dir: str | Path) -> Run:
    """Read ``results.json`` of a run directory."""
    content = json.loads((Path(run_dir) / RESULTS_FILE).read_text(encoding="utf-8"))
    version = content.pop("format_version", None)
    if version != FORMAT_VERSION:
        raise ValueError(f"{run_dir}: unsupported record format version {version}")
    if content["controller"] not in CONTROLLERS:
        raise ValueError(f"{run_dir}: unknown controller {content['controller']!r}")
    episodes = []
    for entry in content.pop("episodes"):
        estimate = entry.pop("estimate", None)
        if estimate is not None:
            estimate = Estimate(**{**estimate, "rms_speeds": tuple(estimate["rms_speeds"])})
        episodes.append(Episode(**entry, estimate=estimate))
    return Run(**content, episodes=tuple(episodes))


def find_runs(root: str | Path) -> list[Path]:
    """Run directories (those holding ``results.json``) at or below ``root``, sorted."""
    root = Path(root)
    return sorted(path.parent for path in root.rglob(RESULTS_FILE))
