"""Tests for benchmark typed exception hierarchy (06.0 PATTERNS §exceptions.py).

One test per class:
  1. Subclasses BenchmarkError.
  2. Constructor accepts documented fields and stores them as public attrs.
  3. str(exc) surfaces identifying fields.

Negative guard (PATTERNS §Shared Pattern D — analog to sandbox D-11):
the benchmarks hierarchy MUST NOT define per-scoring-failure classes like
ScoreFailure / SymbolicTimeout / NumericalBlowup / etc. Those failure
modes are string reason codes on the Plan 02 `BenchmarkScore.reasons`
dict, NOT exceptions. Only host-level failures raise.
"""

from __future__ import annotations

import pytest

from ascension.benchmarks import exceptions as benchmarks_exceptions
from ascension.benchmarks.exceptions import (
    BenchmarkError,
    BenchmarkIntegrationError,
    UnknownBenchmarkSystem,
)


def test_benchmark_error_is_exception_subclass() -> None:
    assert issubclass(BenchmarkError, Exception)


def test_unknown_benchmark_system_subclasses_benchmark_error() -> None:
    assert issubclass(UnknownBenchmarkSystem, BenchmarkError)
    # Accepts a plain message string.
    exc = UnknownBenchmarkSystem("Unknown benchmark system: banana")
    assert "Unknown benchmark system: banana" in str(exc)


def test_benchmark_integration_error_carries_system_seed_message() -> None:
    assert issubclass(BenchmarkIntegrationError, BenchmarkError)
    exc = BenchmarkIntegrationError(
        system="lorenz", seed=7, message="solve_ivp failed on traj 0: stiff"
    )
    assert exc.system == "lorenz"
    assert exc.seed == 7
    assert exc.message == "solve_ivp failed on traj 0: stiff"
    s = str(exc)
    assert "BenchmarkIntegrationError" in s
    assert "system='lorenz'" in s
    assert "seed=7" in s


def test_unknown_benchmark_system_catchable_as_base() -> None:
    with pytest.raises(BenchmarkError):
        raise UnknownBenchmarkSystem("Unknown benchmark system: banana")


def test_benchmark_integration_error_catchable_as_base() -> None:
    with pytest.raises(BenchmarkError):
        raise BenchmarkIntegrationError(system="lorenz", seed=0, message="x")


def test_no_per_run_failure_exception_classes_exist() -> None:
    """D-11-analog divergence: per-scoring-tier failures are DTO reason
    strings on Plan 02's BenchmarkScore.reasons, NOT exception classes.
    Assert absence so a future dev cannot drift into raising on them.
    """
    forbidden = (
        "ScoreFailure",
        "ScoringError",
        "SymbolicTimeout",
        "SymbolicParseFail",
        "NumericalBlowup",
        "NumericalDiverged",
        "QualitativeUnknown",
    )
    for name in forbidden:
        assert not hasattr(benchmarks_exceptions, name), (
            f"{name} must not exist in ascension.benchmarks.exceptions — "
            "per-scoring-tier failures are reason strings on Plan 02's "
            "BenchmarkScore.reasons dict (PATTERNS §Shared Pattern D)."
        )
