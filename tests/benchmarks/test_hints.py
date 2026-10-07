"""Tests for ascension.benchmarks.hints — alpha-14 two-phase trajectory
sample, structural_prior exposure, and the check_structural_prior_violation
detection helper.
"""

from __future__ import annotations

import pytest

from ascension.benchmarks.hints import (
    check_structural_prior_violation,
    compute_benchmark_hints,
)
from ascension.benchmarks.ode import generate_trajectories
from ascension.benchmarks.types import BenchmarkSpec


@pytest.fixture
def lv_bundle():
    """Standard LV seed=42 bundle used across the Phase 7 alpha series.

    Phase 6.1 retune: default_ic moved from [5.0, 5.0] to [0.3, 0.1], so the
    historical ic_spread=0.3 (10× the new registry value of 0.03) jitters ICs
    into negative populations and stalls LV integration. ic_spread=0.03
    matches the new registry default and preserves proportional jitter.
    """
    spec = BenchmarkSpec(
        system="lotka_volterra",
        seed=42,
        n_trajectories=8,
        ic_spread=0.03,
        noise_sigma=0.05,
        n_samples=100,
        t_span_override=None,
    )
    return generate_trajectories(spec, strict=False)


# ---------------------------------------------------------------------------
# Two-phase trajectory_sample (A1 — alpha-14 fix for alpha-13 regression)
# ---------------------------------------------------------------------------


def test_trajectory_sample_has_transient_and_steady_phases(lv_bundle) -> None:
    hints = compute_benchmark_hints(lv_bundle)
    sample = hints["trajectory_sample"]
    assert set(sample.keys()) == {"transient", "steady"}
    assert set(sample["transient"].keys()) == {"t", "vals"}
    assert set(sample["steady"].keys()) == {"t", "vals"}


def test_transient_covers_early_time_steady_covers_post_transient(lv_bundle) -> None:
    """Alpha-13 failure: single-sample trajectory over t_span=[0,20] was
    dominated by the initial crash (t<2). Alpha-14 splits so the agent
    sees BOTH phases. Transient must be concentrated at early t; steady
    must start AFTER ~30% of n_samples."""
    hints = compute_benchmark_hints(lv_bundle)
    t_end = float(hints["t_span"][1])

    transient_t = hints["trajectory_sample"]["transient"]["t"]
    steady_t = hints["trajectory_sample"]["steady"]["t"]

    # Transient is 1-10 points at the front; last transient time should
    # be well within the first third of t_span.
    assert len(transient_t) <= 10
    assert transient_t[-1] < t_end * 0.35

    # Steady starts at or after 30% of t_span and covers the post-transient region.
    assert steady_t[0] >= t_end * 0.25
    # Steady sample ideally has 20 points on n_samples=100 (70 post-transient / 20 = stride 3.5 → 20 points).
    assert len(steady_t) >= 15


def test_steady_sample_vals_have_correct_dim(lv_bundle) -> None:
    hints = compute_benchmark_hints(lv_bundle)
    for row in hints["trajectory_sample"]["steady"]["vals"]:
        assert len(row) == hints["dim"]


# ---------------------------------------------------------------------------
# structural_prior exposure (A2)
# ---------------------------------------------------------------------------


def test_hints_include_lv_structural_prior(lv_bundle) -> None:
    hints = compute_benchmark_hints(lv_bundle)
    prior = hints["structural_prior"]
    assert prior is not None
    assert prior["benchmark"] == "lotka_volterra"
    assert "constraints" in prior
    assert set(prior["constraints"].keys()) == {
        "alpha_positive",
        "beta_positive",
        "gamma_positive",
        "delta_positive",
    }
    assert "canonical_form" in prior
    assert "notes" in prior


# ---------------------------------------------------------------------------
# check_structural_prior_violation — the detection helper
# ---------------------------------------------------------------------------


def test_check_prior_truth_form_no_violations() -> None:
    """Canonical LV form should violate no priors."""
    r = check_structural_prior_violation(
        ["1.0*x0 - 10.0*x0*x1", "10.0*x0*x1 - 2.0*x1"],
        "lotka_volterra",
    )
    assert r["applicable"] is True
    assert r["parse_ok"] is True
    assert r["violations"] == []
    coefs = r["coefficients"]
    assert abs(coefs["alpha"] - 1.0) < 1e-9
    assert abs(coefs["beta"] - 10.0) < 1e-9
    assert abs(coefs["gamma"] - 2.0) < 1e-9
    assert abs(coefs["delta"] - 10.0) < 1e-9


def test_check_prior_alpha13_iter14_form_alpha_violation() -> None:
    """Alpha-13 iter 14 form had α=-1 (biologically impossible). The
    violation detector must flag alpha_positive specifically."""
    r = check_structural_prior_violation(
        ["-1.0*x0 - 1.0*x0*x1", "1.0*x0*x1 - 3.0*x1"],
        "lotka_volterra",
    )
    assert r["applicable"] is True
    assert r["parse_ok"] is True
    assert "alpha_positive" in r["violations"]
    # The other three should NOT be violated (β=1, γ=3, δ=1 all positive).
    assert "beta_positive" not in r["violations"]
    assert "gamma_positive" not in r["violations"]
    assert "delta_positive" not in r["violations"]
    assert r["coefficients"]["alpha"] == -1.0


def test_check_prior_alpha11_iter3_form_no_violations() -> None:
    """Alpha-11 iter 3 form (α=2, β=δ=10, γ=2). Structurally correct
    even though α=2 vs truth α=1 — all coefficients still positive, so
    no prior violation."""
    r = check_structural_prior_violation(
        ["2.0*x0 - 10.0*x0*x1", "10.0*x0*x1 - 2.0*x1"],
        "lotka_volterra",
    )
    assert r["parse_ok"] is True
    assert r["violations"] == []


def test_check_prior_unknown_benchmark_not_applicable() -> None:
    r = check_structural_prior_violation(
        ["1.0*x0", "1.0*x1"],
        "unknown_system_xyz",
    )
    assert r["applicable"] is False
    assert r["parse_ok"] is False
    assert r["violations"] == []


def test_check_prior_parse_error_returns_parse_ok_false() -> None:
    """Malformed symbolic_form strings must not crash the detector —
    they return applicable=True, parse_ok=False so callers can treat as
    'no signal' rather than raising."""
    r = check_structural_prior_violation(
        ["((( bad syntax", "x0"],
        "lotka_volterra",
    )
    assert r["applicable"] is True
    assert r["parse_ok"] is False
    assert r["violations"] == []


def test_check_prior_empty_form_returns_parse_ok_false() -> None:
    r = check_structural_prior_violation([], "lotka_volterra")
    assert r["applicable"] is True
    assert r["parse_ok"] is False


def test_check_prior_wrong_arity_returns_parse_ok_false() -> None:
    """LV expects exactly 2 equations. One or three should not be
    treated as parseable."""
    r = check_structural_prior_violation(["1.0*x0"], "lotka_volterra")
    assert r["parse_ok"] is False
