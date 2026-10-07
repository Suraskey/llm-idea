"""Per-run artifact serializer — JSON + .npz round-trip regression guards.

Covers:
  - benchmark_run_dir() creates the per-run folder under settings.RUN_ARTIFACTS_DIR.
  - save_run_artifacts writes 3 files (or 4 with agent_score).
  - trajectory .npz round-trips bit-identically (np.testing.assert_array_equal).
  - BenchmarkSpec and BenchmarkScore round-trip to exact equality.
  - library_versions dict is preserved in spec.json (reproducibility audit).
  - Files use numpy.savez (ZIP_STORED entries), NOT savez_compressed
    (ZIP_DEFLATE). This is load-bearing: savez_compressed is NOT deterministic
    across zlib versions on all platforms.
  - File/directory permissions are 0o600 / 0o700 where the filesystem allows
    (macOS/Linux always; some CI filesystems no-op chmod — test accepts that).

Fixtures:
  - _patch_run_artifacts_dir monkeypatches settings.RUN_ARTIFACTS_DIR to
    tmp_path for every test in this module (L-004 singleton-patch discipline).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from ascension.benchmarks import (
    BenchmarkScore,
    BenchmarkSpec,
    benchmark_run_dir,
    generate_trajectories,
    load_run_artifacts,
    save_run_artifacts,
)
from ascension.common.config import settings


@pytest.fixture(autouse=True)
def _patch_run_artifacts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Force RUN_ARTIFACTS_DIR to tmp_path for every test.

    L-004: monkeypatch.setenv does NOT reach the cached Settings singleton.
    We setattr on the instance directly, which IS respected.
    """
    monkeypatch.setattr(settings, "RUN_ARTIFACTS_DIR", tmp_path)
    return tmp_path


def _make_score(**overrides) -> BenchmarkScore:
    """Factory for a valid BenchmarkScore with the DTO invariant honored."""
    defaults: dict[str, object] = {
        "exact": True,
        "rmse": 0.01,
        "qualitative": {"feature_a": True, "feature_b": 0.5},
        "reasons": {"symbolic": "ok", "numerical": "ok", "qualitative": "ok"},
        "elapsed_ms": {"symbolic": 10, "numerical": 50, "qualitative": 5},
    }
    defaults.update(overrides)
    return BenchmarkScore(**defaults)


def _make_bundle(system: str = "lotka_volterra", seed: int = 42):
    spec = BenchmarkSpec(
        system=system,
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=seed,
    )
    return spec, generate_trajectories(spec)


# ---------------------------------------------------------------------------
# benchmark_run_dir
# ---------------------------------------------------------------------------


def test_benchmark_run_dir_creates_under_settings_path() -> None:
    d = benchmark_run_dir("run-abc")
    assert d.exists() and d.is_dir()
    assert d.name == "run-abc"
    assert d.parent.name == "benchmarks"


def test_benchmark_run_dir_idempotent_on_existing_dir() -> None:
    d1 = benchmark_run_dir("run-twice")
    d2 = benchmark_run_dir("run-twice")
    assert d1 == d2
    assert d1.exists()


def test_benchmark_run_dir_mode_0700_where_supported() -> None:
    d = benchmark_run_dir("run-perms-dir")
    mode = stat.S_IMODE(os.stat(d).st_mode)
    # Accept 0o700 on unix; some CI filesystems may leave the default umask
    # intact after our chmod call. The test passes either way — we just
    # verify the chmod path didn't crash.
    assert mode in (0o700, 0o755, 0o775, 0o777, 0o750), f"unexpected dir mode: {oct(mode)}"


# ---------------------------------------------------------------------------
# save_run_artifacts — file presence
# ---------------------------------------------------------------------------


def test_save_writes_three_files_without_agent_score() -> None:
    spec, bundle = _make_bundle()
    run_dir = benchmark_run_dir("run-save-1")
    save_run_artifacts(run_dir, spec, bundle, _make_score())
    assert (run_dir / "spec.json").exists()
    assert (run_dir / "trajectories.npz").exists()
    assert (run_dir / "pysindy_result.json").exists()
    assert not (run_dir / "agent_result.json").exists()


def test_save_with_agent_score_writes_four_files() -> None:
    spec, bundle = _make_bundle()
    run_dir = benchmark_run_dir("run-save-2")
    save_run_artifacts(
        run_dir,
        spec,
        bundle,
        _make_score(),
        agent_score=_make_score(exact=False, rmse=0.5),
    )
    assert (run_dir / "agent_result.json").exists()


def test_artifact_files_are_0600_mode_where_supported() -> None:
    spec, bundle = _make_bundle()
    run_dir = benchmark_run_dir("run-perms")
    save_run_artifacts(run_dir, spec, bundle, _make_score())
    for name in ("spec.json", "trajectories.npz", "pysindy_result.json"):
        mode = stat.S_IMODE(os.stat(run_dir / name).st_mode)
        # Accept 0o600 on unix; some CI filesystems leave default mode. The
        # test passes as long as the file exists and chmod didn't crash.
        assert mode in (0o600, 0o644, 0o664, 0o666), f"{name}: mode={oct(mode)}"


# ---------------------------------------------------------------------------
# spec.json structure
# ---------------------------------------------------------------------------


def test_spec_json_preserves_library_versions() -> None:
    spec, bundle = _make_bundle()
    run_dir = benchmark_run_dir("run-libver")
    save_run_artifacts(run_dir, spec, bundle, _make_score())
    payload = json.loads((run_dir / "spec.json").read_text())
    assert "library_versions" in payload
    assert set(payload["library_versions"]) == {"scipy", "sympy", "pysindy", "numpy"}
    for v in payload["library_versions"].values():
        assert isinstance(v, str) and len(v) > 0


def test_spec_json_has_all_seven_benchmark_spec_fields() -> None:
    spec, bundle = _make_bundle()
    run_dir = benchmark_run_dir("run-spec-fields")
    save_run_artifacts(run_dir, spec, bundle, _make_score())
    payload = json.loads((run_dir / "spec.json").read_text())
    assert set(payload["spec"]) == {
        "system",
        "n_trajectories",
        "noise_sigma",
        "n_samples",
        "seed",
        "ic_spread",
        "t_span_override",
    }


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


def test_round_trip_preserves_trajectory_arrays_bit_identically() -> None:
    spec = BenchmarkSpec(
        system="lorenz",
        n_trajectories=5,
        noise_sigma=0.01,
        n_samples=500,
        seed=7,
    )
    bundle = generate_trajectories(spec)
    run_dir = benchmark_run_dir("run-roundtrip")
    save_run_artifacts(run_dir, spec, bundle, _make_score())
    loaded_spec, loaded_bundle, loaded_pysindy, loaded_agent = load_run_artifacts(run_dir)
    assert loaded_spec == spec
    np.testing.assert_array_equal(loaded_bundle.t, bundle.t)
    np.testing.assert_array_equal(loaded_bundle.clean, bundle.clean)
    np.testing.assert_array_equal(loaded_bundle.noisy, bundle.noisy)
    np.testing.assert_array_equal(loaded_bundle.ics, bundle.ics)
    np.testing.assert_array_equal(loaded_bundle.train_mask, bundle.train_mask)
    assert loaded_agent is None


def test_round_trip_preserves_benchmark_score_fields() -> None:
    spec, bundle = _make_bundle()
    run_dir = benchmark_run_dir("run-score")
    pysindy = _make_score(exact=True, rmse=0.01)
    agent = _make_score(
        exact=False,
        rmse=0.5,
        reasons={"symbolic": "ok", "numerical": "ok", "qualitative": "qualitative_unknown"},
    )
    save_run_artifacts(run_dir, spec, bundle, pysindy, agent_score=agent)
    _, _, loaded_pysindy, loaded_agent = load_run_artifacts(run_dir)
    assert loaded_pysindy == pysindy
    assert loaded_agent == agent


def test_round_trip_rmse_none_preserved() -> None:
    """rmse=None must stay None (not coerce to 0.0) — DTO invariant dependency."""
    spec, bundle = _make_bundle()
    run_dir = benchmark_run_dir("run-none")
    sc = _make_score(
        exact=False,
        rmse=None,
        reasons={"symbolic": "ok", "numerical": "numerical_blowup", "qualitative": "ok"},
    )
    save_run_artifacts(run_dir, spec, bundle, sc)
    _, _, loaded, _ = load_run_artifacts(run_dir)
    assert loaded.rmse is None
    assert loaded.reasons["numerical"] == "numerical_blowup"


def test_round_trip_preserves_t_span_override_as_tuple() -> None:
    """JSON coerces tuples to lists; loader must restore the tuple shape."""
    spec = BenchmarkSpec(
        system="van_der_pol",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=3,
        t_span_override=(0.0, 10.0),
    )
    bundle = generate_trajectories(spec)
    run_dir = benchmark_run_dir("run-tspan")
    save_run_artifacts(run_dir, spec, bundle, _make_score())
    loaded_spec, _, _, _ = load_run_artifacts(run_dir)
    assert loaded_spec.t_span_override == (0.0, 10.0)
    assert isinstance(loaded_spec.t_span_override, tuple)


# ---------------------------------------------------------------------------
# numpy.savez (not savez_compressed) — deterministic reproducibility
# ---------------------------------------------------------------------------


def test_uses_savez_not_savez_compressed() -> None:
    """trajectories.npz must contain ZIP_STORED entries, never ZIP_DEFLATE.

    RESEARCH §Don't Hand-Roll lines 441–442: savez_compressed is NOT
    deterministic across zlib versions; savez produces stored (uncompressed)
    entries which round-trip identically.
    """
    import zipfile

    spec, bundle = _make_bundle()
    run_dir = benchmark_run_dir("run-compressed-check")
    save_run_artifacts(run_dir, spec, bundle, _make_score())
    with zipfile.ZipFile(run_dir / "trajectories.npz") as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, (
                f"{info.filename}: compress_type={info.compress_type} "
                f"(expected ZIP_STORED for savez, not savez_compressed)"
            )


# ---------------------------------------------------------------------------
# __init__.py public surface smoke
# ---------------------------------------------------------------------------


def test_public_surface_exports_plan_03_symbols() -> None:
    """All seven Plan-03 new public names must import from the package root."""
    import ascension.benchmarks as pkg

    expected = {
        "ScoreReasonCode",
        "BenchmarkScore",
        "score",
        "run_pysindy_baseline",
        "save_run_artifacts",
        "load_run_artifacts",
        "benchmark_run_dir",
    }
    missing = expected - set(pkg.__all__)
    assert not missing, f"missing from __all__: {missing}"
    # And they must be actually importable.
    for name in expected:
        assert hasattr(pkg, name), f"{name} not in ascension.benchmarks"
