import json

import numpy as np
import pytest

from con_dec_imp_sim.records import (
    FORMAT_VERSION,
    RESULTS_FILE,
    Episode,
    Estimate,
    Run,
    Trace,
    find_runs,
    read_run,
    read_trace,
    trace_file,
    write_run,
    write_trace,
)

rng = np.random.default_rng(0)


def make_trace(T: int = 12, closure: bool = True, basis_dim: int | None = 2) -> Trace:
    window = np.ones(T, dtype=bool)
    window[0] = False
    pose = np.array([0.5, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0])
    return Trace(
        contact_wrench=rng.standard_normal((T, 6)),
        commanded_wrench=rng.standard_normal((T, 6)),
        feedforward_wrench=np.zeros((T, 6)),
        window=window,
        dt=1.0 / 150.0,
        object_initial_pose=pose,
        closure_tcp_pose=pose + 0.1 if closure else None,
        closure_object_pose=pose if closure else None,
        installed_basis=None if basis_dim is None else rng.standard_normal((6, basis_dim)),
    )


def make_run(controller: str = "ours") -> Run:
    estimate = Estimate(dim=2, alpha=0.15, rms_speeds=(0.2, 0.1, 1e-3, 1e-3, 1e-4, 1e-4), time=0.07)
    return Run(
        object="cylindrical",
        controller=controller,
        checkpoint="cylindrical.pth",
        checkpoint_sha256="ab" * 32,
        seed=0,
        settings={"estimation": {"num_samples": 128}},
        episodes=(
            Episode(index=0, success=True, steps=900, closure_index=24, trace=trace_file(0), estimate=estimate),
            Episode(index=1, success=False, steps=2000, closure_index=None, trace=trace_file(1)),
            Episode(index=2, success=False, steps=2000),
        ),
    )


def test_trace_roundtrip(tmp_path):
    trace = make_trace()
    path = tmp_path / trace_file(3)
    write_trace(path, trace)
    back = read_trace(path)
    assert path.name == "episode_003.npz"
    for name in ("contact_wrench", "commanded_wrench", "feedforward_wrench"):
        assert getattr(back, name).dtype == np.float32
        np.testing.assert_array_equal(getattr(back, name), getattr(trace, name).astype(np.float32))
    np.testing.assert_array_equal(back.window, trace.window)
    assert back.dt == trace.dt
    for name in ("object_initial_pose", "closure_tcp_pose", "closure_object_pose", "installed_basis"):
        assert getattr(back, name).dtype == np.float64
        np.testing.assert_array_equal(getattr(back, name), getattr(trace, name))
    assert back.has_closure
    np.testing.assert_allclose(
        back.feedback_wrench, trace.commanded_wrench.astype(np.float32).astype(np.float64), rtol=0, atol=0
    )


def test_trace_optional_fields(tmp_path):
    write_trace(tmp_path / "a.npz", make_trace(closure=False, basis_dim=None))
    back = read_trace(tmp_path / "a.npz")
    assert not back.has_closure
    assert back.closure_object_pose is None and back.installed_basis is None
    with np.load(tmp_path / "a.npz") as data:
        assert set(data.files) == {
            "contact_wrench", "commanded_wrench", "feedforward_wrench", "window", "dt", "object_initial_pose"
        }
    write_trace(tmp_path / "b.npz", make_trace(basis_dim=0))
    assert read_trace(tmp_path / "b.npz").installed_basis.shape == (6, 0)


@pytest.mark.parametrize(
    "change",
    [
        {"contact_wrench": np.zeros((5, 6))},
        {"window": np.ones((12, 1), dtype=bool)},
        {"dt": 0.0},
        {"closure_tcp_pose": np.zeros(6)},
        {"closure_object_pose": None},
        {"installed_basis": np.zeros((3, 1))},
    ],
)
def test_trace_validation(tmp_path, change):
    fields = make_trace().__dict__ | change
    with pytest.raises(ValueError):
        write_trace(tmp_path / "bad.npz", Trace(**fields))


def test_run_roundtrip(tmp_path):
    run = make_run()
    path = write_run(tmp_path / "run", run)
    assert path.name == RESULTS_FILE
    content = json.loads(path.read_text())
    assert content["format_version"] == FORMAT_VERSION
    assert content["episodes"][2]["trace"] is None and content["episodes"][1]["estimate"] is None
    assert read_run(tmp_path / "run") == run


def test_run_validation(tmp_path):
    with pytest.raises(ValueError):
        write_run(tmp_path, make_run(controller="impedance"))
    path = write_run(tmp_path, make_run())
    content = json.loads(path.read_text())
    content["format_version"] = FORMAT_VERSION + 1
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        read_run(tmp_path)


def test_find_runs(tmp_path):
    for sub in ("b/ours", "a/iso", "a/oracle"):
        write_run(tmp_path / sub, make_run(sub.split("/")[1]))
    (tmp_path / "a" / "notes").mkdir()
    assert find_runs(tmp_path) == [tmp_path / "a/iso", tmp_path / "a/oracle", tmp_path / "b/ours"]
    assert find_runs(tmp_path / "b" / "ours") == [tmp_path / "b/ours"]
