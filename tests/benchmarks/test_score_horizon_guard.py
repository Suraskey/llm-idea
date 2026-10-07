"""WR-01 regression — _DEFAULT_HORIZONS.get() missing-key path is loud.

The original code called ``_DEFAULT_HORIZONS.get(truth_spec.system)`` with
no second argument. If a future phase adds a new system to DEFAULT_PARAMS
but forgets to wire a horizon entry, the scorer would silently fall
through to horizon=None (full t-span). For a chaotic system that
produces meaningless RMSE — fine numerically but silently wrong.

The fix uses a sentinel-based lookup with an explicit logger.warning
(CLAUDE.md no-silent-drops rule). This test patches _DEFAULT_HORIZONS at
the module level to remove the lotka_volterra key, then confirms the
warning fires when score() runs.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from ascension.benchmarks import BenchmarkSpec, generate_trajectories, scoring
from ascension.benchmarks.scoring import score


def test_missing_horizon_key_logs_loud_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A system absent from _DEFAULT_HORIZONS triggers a warning.

    The invariant: if somebody adds a system to DEFAULT_PARAMS without a
    horizon, the scorer MUST emit a warning. Silent fall-through to
    full-t-span RMSE is a no-cover-ups violation.
    """
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=100,
        seed=0,
    )
    bundle = generate_trajectories(spec)

    # Patch _DEFAULT_HORIZONS to an empty dict — simulates a newly-added
    # system whose maintainer forgot to wire the horizon entry.
    empty_horizons: dict[str, float | None] = {}
    with patch.object(scoring, "_DEFAULT_HORIZONS", empty_horizons):
        with caplog.at_level(logging.WARNING, logger="ascension.benchmarks.scoring"):
            result = score(["x - y", "x*y"], spec, bundle)

    # Warning fired.
    warning_lines = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("not in _DEFAULT_HORIZONS" in msg for msg in warning_lines), (
        f"Expected loud warning for missing horizon; got warnings: " f"{warning_lines!r}"
    )
    # score() still returns a usable BenchmarkScore (warning-only, not hard-fail).
    assert result.reasons["numerical"] in (
        "ok",
        "numerical_diverged",
        "numerical_blowup",
    )


def test_known_system_does_not_log_missing_horizon_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The happy path — lotka_volterra is in _DEFAULT_HORIZONS — no warning.

    Confirms the warning is CONDITIONAL on missing-key, not a regression
    that fires on every call.
    """
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=100,
        seed=0,
    )
    bundle = generate_trajectories(spec)

    with caplog.at_level(logging.WARNING, logger="ascension.benchmarks.scoring"):
        score(["x - 10*x*y", "10*x*y - 2*y"], spec, bundle)

    warning_lines = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not any(
        "not in _DEFAULT_HORIZONS" in msg for msg in warning_lines
    ), f"Unexpected horizon warning on known system: {warning_lines!r}"
