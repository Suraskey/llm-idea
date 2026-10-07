"""Tests for benchmark DTOs (06.0 RESEARCH §Pattern 1).

Covers:
  - BenchmarkSpec frozen-slots dataclass with exactly 7 fields.
  - TrajectoryBundle shape invariants (clean/noisy/ics/train_mask).
  - library_versions dict has exactly 4 keys (scipy/sympy/pysindy/numpy).

Invariant tested at module-load level (grep of source):
  - BenchmarkSpec.noise_sigma convention must be documented literally in
    the source as "fraction of per-trajectory RMS" (RESEARCH Pitfall #7
    — prevents silent drift on what "1% noise" means across systems).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from ascension.benchmarks import types as benchmarks_types
from ascension.benchmarks.types import BenchmarkSpec, TrajectoryBundle


def _make_spec(**overrides: object) -> BenchmarkSpec:
    defaults: dict[str, object] = {
        "system": "lotka_volterra",
        "n_trajectories": 4,
        "noise_sigma": 0.0,
        "n_samples": 200,
        "seed": 42,
    }
    defaults.update(overrides)
    return BenchmarkSpec(**defaults)  # type: ignore[arg-type]


def test_benchmark_spec_frozen_slots() -> None:
    spec = _make_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.seed = 99  # type: ignore[misc]
    # slots=True means no __dict__ on instances.
    assert not hasattr(spec, "__dict__")


def test_benchmark_spec_required_fields() -> None:
    fields = dataclasses.fields(BenchmarkSpec)
    names = {f.name for f in fields}
    assert names == {
        "system",
        "n_trajectories",
        "noise_sigma",
        "n_samples",
        "seed",
        "ic_spread",
        "t_span_override",
    }
    assert len(fields) == 7


def test_benchmark_spec_defaults() -> None:
    spec = _make_spec()
    # Phase 6.1 D-04 amended (2026-04-24): ic_spread default is the sentinel
    # None, not 1.0. None signals "look up per-system value from
    # DEFAULT_PARAMS". A caller can still pass ic_spread=1.0 explicitly —
    # that wins over the registry (verified by
    # test_caller_wins_with_explicit_one_oh in tests/benchmarks/test_audit.py).
    assert spec.ic_spread is None
    assert spec.t_span_override is None


def test_trajectory_bundle_shape_invariants(lv_small_bundle: TrajectoryBundle) -> None:
    bundle = lv_small_bundle
    assert bundle.clean.shape == (4, 200, 2)
    assert bundle.noisy.shape == (4, 200, 2)
    assert bundle.ics.shape == (4, 2)
    assert bundle.train_mask.shape == (4,)
    assert bundle.train_mask.dtype == np.bool_
    assert bundle.t.shape == (200,)


def test_trajectory_bundle_library_versions_has_four_keys(
    lv_small_bundle: TrajectoryBundle,
) -> None:
    versions = lv_small_bundle.library_versions
    assert set(versions.keys()) == {"scipy", "sympy", "pysindy", "numpy"}
    for key, value in versions.items():
        assert isinstance(value, str), f"version for {key} must be str"
        assert value, f"version for {key} must be non-empty"


def test_benchmark_spec_source_mentions_rms_convention() -> None:
    """RESEARCH Pitfall #7: noise_sigma convention must be documented in source.

    Prevents silent doc-drift on what "1% noise" means. Reviewers expect
    fraction-of-per-trajectory-RMS (PySINDy convention).
    """
    src = Path(benchmarks_types.__file__).read_text()
    assert "fraction of per-trajectory RMS" in src, (
        "types.py must document the 'fraction of per-trajectory RMS' "
        "noise_sigma convention (RESEARCH Pitfall #7)."
    )


def test_trajectory_bundle_is_frozen(lv_small_bundle: TrajectoryBundle) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        lv_small_bundle.t = np.zeros(5)  # type: ignore[misc]
