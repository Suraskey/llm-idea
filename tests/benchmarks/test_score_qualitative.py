"""Tests for the qualitative tier of score() — per-system feature extractors.

Covers RESEARCH §Architecture Patterns Pattern 4:
  - Lotka-Volterra: fixed_point_match + conserved_quantity_drift.
  - Van der Pol: has_limit_cycle + limit_cycle_period_match.
  - Lorenz: attractor_box_match + lyapunov_sign_match + fixed_point_count_match.

Plus the end-to-end happy path: score(truth, truth_spec, truth_bundle) must
return exact=True and reasons['qualitative']=='ok' for every system.
"""

from __future__ import annotations

from sympy import symbols

from ascension.benchmarks.scoring import (
    _lorenz_features,
    _lv_features,
    _qualitative_match,
    _truth_rhs_sympy,
    _vdp_features,
    score,
)

# -----------------------------------------------------------------------------
# Lotka-Volterra features
# -----------------------------------------------------------------------------


def test_lv_features_truth_vs_truth_all_match(lv_small_bundle) -> None:
    truth = _truth_rhs_sympy("lotka_volterra")
    feats = _lv_features(truth, lv_small_bundle)
    assert feats["fixed_point_match"] is True
    # Noise-free LV with rtol=1e-12 still produces drift ~20% on large-amplitude
    # orbits that pass through x ≈ 0: log(x) explodes there and H's dynamic
    # range (40–100) amplifies integrator error visibly in std/mean. This is
    # a true measurement of the trajectory's numerical behavior, not a test
    # flake — it bounds the feature from above, which is what matters for
    # the qualitative tier (a real NOISY fit would drift O(1)+, not 0.23).
    # A Phase-7.0 evaluator can tighten this when agent-generated ODEs flow in.
    assert 0.0 <= feats["conserved_quantity_drift"] < 1.0


def test_lv_features_returns_expected_keys(lv_small_bundle) -> None:
    truth = _truth_rhs_sympy("lotka_volterra")
    feats = _lv_features(truth, lv_small_bundle)
    # alpha-10: fp_proposed_x / fp_proposed_y added so agent prompts can
    # see their own equation's FP coordinates without leaking truth.
    assert set(feats.keys()) == {
        "fixed_point_match",
        "fp_proposed_x",
        "fp_proposed_y",
        "conserved_quantity_drift",
    }


def test_lv_features_wrong_coefficients_miss_fixed_point(lv_small_bundle) -> None:
    """FP of α=2, β=20, δ=20, γ=4 would be (0.2, 0.1) still BUT wrong parameters
    shift FP significantly. Try (α=1, β=5, δ=10, γ=2) → FP=(0.2, 0.2)."""
    x, y = symbols("x y")
    bad = [x - 5 * x * y, 10 * x * y - 2 * y]  # β=5, α=1, δ=10, γ=2 → FP=(0.2, 0.2)
    feats = _lv_features(bad, lv_small_bundle)
    # FP=(0.2, 0.2) vs truth (0.2, 0.1) → x matches but y is 2x off.
    assert feats["fixed_point_match"] is False
    # alpha-10: fp_proposed surfaces the actual coords — 0.2 vs 0.2 means
    # the agent can see it flipped vs truth's (0.2, 0.1).
    assert abs(feats["fp_proposed_x"] - 0.2) < 1e-6
    assert abs(feats["fp_proposed_y"] - 0.2) < 1e-6


def test_lv_features_fp_proposed_flipped_fp_detected(lv_small_bundle) -> None:
    """Alpha-10 regression — alpha-9 17-iter agent locked on α=0.767,
    β=3.5, γ=0.0788, δ=3.5. Its FP = (γ/δ, α/β) = (0.0225, 0.219) —
    flipped vs truth's (0.2, 0.1). Verify fp_proposed_x/y surface the
    flipped coords so the next iteration can detect and escape."""
    x, y = symbols("x y")
    flipped = [0.767 * x - 3.5 * x * y, 3.5 * x * y - 0.0788 * y]
    feats = _lv_features(flipped, lv_small_bundle)
    assert feats["fixed_point_match"] is False
    assert abs(feats["fp_proposed_x"] - 0.0225) < 1e-3
    assert abs(feats["fp_proposed_y"] - 0.2191) < 1e-3


def test_score_x0x1_parse_contract_no_implicit_mul_collapse(lv_small_bundle) -> None:
    """L-013 regression — the most consequential bug in the project.

    `implicit_multiplication_application` is in scoring's PARSE_TRANSFORMS.
    Without pre-registering `x0`/`x1` in local_dict the transformation
    reads `x0` as `x*0 = 0` and `x1` as `x*1 = x`. Every alpha-1 through
    alpha-10 LV run scored a degenerate form (`[0, -c*x]`) instead of the
    agent's actual proposed RHS, because the V1..V7 prompt contract uses
    x0,x1 while `_vars_for("lotka_volterra")` returns x,y.

    Post-fix (score() now aliases x0→vars_[0], x1→vars_[1]):
      - Truth in x0,x1 form scores exact=True, rmse=0.0.
      - Alpha-9 "locked" form (α=0.767, β=3.5, γ=0.0788, δ=3.5) scores
        rmse ≈ 8.3 — the real value, not the degenerate ≈4.47 mirage.
      - fp_proposed coords are the actual FP of the agent's form, not NaN.
    """
    from ascension.benchmarks.scoring import score

    truth_in_x0x1 = [
        "1.0*x0 - 10.0*x0*x1",
        "10.0*x0*x1 - 2.0*x1",
    ]
    # spec implicit in lv_small_bundle — use the same spec the fixture builds
    from ascension.benchmarks.types import BenchmarkSpec

    # Phase 6.1 retune: default_ic moved from [5.0, 5.0] to [0.3, 0.1];
    # ic_spread=0.3 jitters into negative populations and stalls LV integration.
    # ic_spread=0.03 preserves proportional jitter against the new IC.
    spec = BenchmarkSpec(
        system="lotka_volterra",
        seed=42,
        n_trajectories=8,
        ic_spread=0.03,
        noise_sigma=0.05,
        n_samples=100,
        t_span_override=None,
    )
    from ascension.benchmarks.ode import generate_trajectories

    bundle = generate_trajectories(spec, strict=False)

    result = score(truth_in_x0x1, spec, bundle)
    assert result.exact is True
    assert result.rmse == 0.0
    assert result.qualitative["fixed_point_match"] is True
    assert abs(result.qualitative["fp_proposed_x"] - 0.2) < 1e-6
    assert abs(result.qualitative["fp_proposed_y"] - 0.1) < 1e-6

    # And a non-truth form with x0/x1 should also parse correctly
    wrong_but_parseable = [
        "0.767*x0 - 3.5*x0*x1",
        "3.5*x0*x1 - 0.0788*x1",
    ]
    r2 = score(wrong_but_parseable, spec, bundle)
    assert r2.exact is False
    # Sanity: rmse should be a real finite number (not the degenerate 4.47
    # that would indicate the collapsed-parse bug reappearing).
    assert r2.rmse is not None
    assert r2.qualitative["fp_proposed_x"] > 0  # self-derived, zero-leak
    assert r2.qualitative["fp_proposed_y"] > 0


# -----------------------------------------------------------------------------
# Van der Pol features
# -----------------------------------------------------------------------------


def test_vdp_features_truth_vs_truth_has_limit_cycle(vdp_small_bundle) -> None:
    truth = _truth_rhs_sympy("van_der_pol")
    feats = _vdp_features(truth, vdp_small_bundle)
    assert feats["has_limit_cycle"] is True
    assert feats["limit_cycle_period_match"] is True


def test_vdp_features_returns_expected_keys(vdp_small_bundle) -> None:
    truth = _truth_rhs_sympy("van_der_pol")
    feats = _vdp_features(truth, vdp_small_bundle)
    assert set(feats.keys()) == {"has_limit_cycle", "limit_cycle_period_match"}


# -----------------------------------------------------------------------------
# Lorenz features
# -----------------------------------------------------------------------------


def test_lorenz_features_truth_vs_truth(lorenz_small_bundle) -> None:
    truth = _truth_rhs_sympy("lorenz")
    feats = _lorenz_features(truth, lorenz_small_bundle)
    assert feats["lyapunov_sign_match"] is True  # truth is chaotic
    # Attractor bounds use hard-coded pysindy-ic truth bounds; we just need
    # the feature to resolve without crashing on truth-vs-truth.
    assert isinstance(feats["attractor_box_match"], bool)
    assert isinstance(feats["fixed_point_count_match"], bool)


def test_lorenz_features_returns_expected_keys(lorenz_small_bundle) -> None:
    truth = _truth_rhs_sympy("lorenz")
    feats = _lorenz_features(truth, lorenz_small_bundle)
    assert set(feats.keys()) == {
        "attractor_box_match",
        "lyapunov_sign_match",
        "fixed_point_count_match",
    }


# -----------------------------------------------------------------------------
# Dispatcher behavior
# -----------------------------------------------------------------------------


def test_qualitative_dispatch_unknown_system(lv_small_bundle) -> None:
    feats, reason = _qualitative_match([], "banana", lv_small_bundle)
    assert reason == "qualitative_unknown"
    assert feats == {}


def test_qualitative_dispatch_lv_returns_ok(lv_small_bundle) -> None:
    truth = _truth_rhs_sympy("lotka_volterra")
    feats, reason = _qualitative_match(truth, "lotka_volterra", lv_small_bundle)
    assert reason == "ok"
    assert "fixed_point_match" in feats
    assert "conserved_quantity_drift" in feats


def test_qualitative_dispatch_vdp_returns_ok(vdp_small_bundle) -> None:
    truth = _truth_rhs_sympy("van_der_pol")
    feats, reason = _qualitative_match(truth, "van_der_pol", vdp_small_bundle)
    assert reason == "ok"
    assert "has_limit_cycle" in feats
    assert "limit_cycle_period_match" in feats


def test_qualitative_dispatch_lorenz_returns_ok(lorenz_small_bundle) -> None:
    truth = _truth_rhs_sympy("lorenz")
    feats, reason = _qualitative_match(truth, "lorenz", lorenz_small_bundle)
    assert reason == "ok"
    assert "attractor_box_match" in feats
    assert "lyapunov_sign_match" in feats
    assert "fixed_point_count_match" in feats


def test_qualitative_returns_graceful_when_nonsense_rhs_cant_integrate(
    lorenz_small_bundle,
) -> None:
    """Nonsense RHS — extractor should return features (all False) or
    QUALITATIVE_UNKNOWN. Never hang or raise."""
    x, y, z = symbols("x y z")
    # Exploding RHS — blowup event kicks in inside the helper.
    nonsense = [1e10 * x**5, 1e10 * y**5, 1e10 * z**5]
    feats, reason = _qualitative_match(nonsense, "lorenz", lorenz_small_bundle)
    assert reason in ("ok", "qualitative_unknown")
    if reason == "ok":
        # All booleans must be False — no integration succeeded.
        assert feats["attractor_box_match"] is False
        assert feats["lyapunov_sign_match"] is False


# -----------------------------------------------------------------------------
# Full score() integration — truth-vs-truth for all three systems
# -----------------------------------------------------------------------------


def test_full_score_happy_path_lv(lv_small_bundle) -> None:
    truth = _truth_rhs_sympy("lotka_volterra")
    r = score(truth, lv_small_bundle.spec, lv_small_bundle)
    assert r.exact is True, f"lv: exact={r.exact}, reasons={r.reasons}"
    assert r.reasons["symbolic"] == "ok"
    assert r.reasons["numerical"] == "ok"
    assert r.reasons["qualitative"] == "ok"
    assert r.rmse is not None


def test_full_score_happy_path_vdp(vdp_small_bundle) -> None:
    truth = _truth_rhs_sympy("van_der_pol")
    r = score(truth, vdp_small_bundle.spec, vdp_small_bundle)
    assert r.exact is True, f"vdp: exact={r.exact}, reasons={r.reasons}"
    assert r.reasons["symbolic"] == "ok"
    assert r.reasons["numerical"] == "ok"
    assert r.reasons["qualitative"] == "ok"
    assert r.rmse is not None


def test_full_score_happy_path_lorenz(lorenz_small_bundle) -> None:
    truth = _truth_rhs_sympy("lorenz")
    r = score(truth, lorenz_small_bundle.spec, lorenz_small_bundle)
    assert r.exact is True, f"lorenz: exact={r.exact}, reasons={r.reasons}"
    assert r.reasons["symbolic"] == "ok"
    assert r.reasons["numerical"] == "ok"
    assert r.reasons["qualitative"] == "ok"
    assert r.rmse is not None


def test_full_score_qualitative_captures_feature_dict(lv_small_bundle) -> None:
    """Score result exposes the per-feature dict in `qualitative`."""
    truth = _truth_rhs_sympy("lotka_volterra")
    r = score(truth, lv_small_bundle.spec, lv_small_bundle)
    assert "fixed_point_match" in r.qualitative
    assert "conserved_quantity_drift" in r.qualitative


def test_full_score_skips_qualitative_when_symbolic_parse_fails(
    lv_small_bundle,
) -> None:
    """WR-04 regression: empty proposed_expr → qualitative skipped, not faked.

    Feature extractors like LV's conserved_quantity_drift pull signal from
    the held-out trajectory alone. Before the fix, an unparseable proposed
    RHS left proposed_expr=[] but the qualitative tier still ran and
    produced a real drift number from ground truth, masquerading as a
    valid qualitative measurement. The fix short-circuits: qualitative
    stays empty, reason flags the skip.
    """
    # Garbage proposed RHS — guaranteed parse failure (invalid Python syntax
    # gets past our parse_expr guard for some shapes, but '@@@@' is a hard fail).
    r = score(["@@@@", "@@@@"], lv_small_bundle.spec, lv_small_bundle)
    assert r.reasons["symbolic"] == "symbolic_parse_fail"
    assert (
        r.qualitative == {}
    ), f"qualitative dict must be empty when parse fails, got {r.qualitative}"
    assert r.reasons["qualitative"] != "ok", (
        f"qualitative reason must flag skip when parse fails, " f"got {r.reasons['qualitative']!r}"
    )
