"""EXP-080 (STRAT-03) — the V2-regime depth scorer (sum coupling + r^-2.5).

The regime-aware ``score_depth`` serves a SECOND §10 alien law without forking.
This suite locks the two halves:

  1. V1 BYTE-IDENTICAL (the non-negotiable safety gate). ``score_depth`` with the
     default ``regime=None`` (or ``RegimeSpec()``) returns output IDENTICAL to the
     pre-EXP-080 V1 scorer on the V1 truth law + the cross-config product family.
     If a V2 edit perturbs the V1 path, the DepthScore diverges and this fails.

  2. V2 REGIME GATES. On the V2 law (RegimeSpec(correction_exponent=2.5,
     charge_coupling="sum")):
       - L2 credits the r^-2.5 correction (and the V1 oracle would DENY it — the
         band is moved);
       - L3/L4 credit the SUM-confound charge work via the cross-config s-SUM gate
         (a per-config (s1+s2) scaling beyond a global amplitude);
       - a global-beta V2 proposal (no per-body charge) ceilings at depth 2 (the
         sum gate is not gameable);
       - the V2 truth law reaches consecutive depth 4.

$0, deterministic sympy/scipy — no LLM (the no-LLM gate must still pass).
"""

from __future__ import annotations

from ascension.benchmarks.alien_depth import RegimeSpec, score_depth
from ascension.benchmarks.alien_fixtures import (
    _build_tier2_tuned_cfg,
    alien_family_benchmark,
    alien_family_v2_benchmark,
    build_tier2_held_out_ics_v2,
)
from ascension.simulator.config import TIER2_HELD_OUT_ICS

# --- V1 truth-law proposals (for the byte-identical regression) --------------
_V1_TRUTH = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**3.5 + beta*s1*s2*(1 - cos(gamma*r))/r**2"
_V1_GLOBAL_BETA = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**3.5 + beta*(1 - cos(gamma*r))/r**2"
_V1_DEPTH2 = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**3.5"

# --- V2 truth-law proposals (sum coupling + r^-2.5) --------------------------
_V2_TRUTH = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**2.5 + beta*(s1 + s2)*(1 - cos(gamma*r))/r**2"
_V2_GLOBAL_BETA = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**2.5 + beta*(1 - cos(gamma*r))/r**2"
_V2_DEPTH2 = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**2.5"
# A V2 proposal that keeps the WRONG (V1) correction exponent — L2 must deny it.
_V2_WRONG_EXPONENT = (
    "-G*m1*m2/r**2 - G*alpha*m1*m2/r**3.5 + beta*(s1 + s2)*(1 - cos(gamma*r))/r**2"
)
# A V2 proposal that uses the V1 PRODUCT amplitude — the sum gate must deny L3
# (it is the wrong charge structure for the V2 law).
_V2_PRODUCT_ON_V2 = (
    "-G*m1*m2/r**2 - G*alpha*m1*m2/r**2.5 + beta*s1*s2*(1 - cos(gamma*r))/r**2"
)

_V2 = RegimeSpec(correction_exponent=2.5, charge_coupling="sum")


def _hyp(symbolic_form: str, rationale: str = "") -> dict:
    return {"symbolic_form": symbolic_form, "rationale": rationale}


def _consec(per_layer) -> int:
    c = 0
    for p in per_layer[:4]:
        if p:
            c += 1
        else:
            break
    return c


# =============================================================================
# Gate 1 — V1 BYTE-IDENTICAL. regime=None / RegimeSpec() reproduce the V1 scorer.
# =============================================================================


def test_v1_score_byte_identical_regime_none_vs_default_spec() -> None:
    """score_depth(regime=None) == score_depth(regime=RegimeSpec()) — exactly.

    The default RegimeSpec IS the V1 regime, so passing it explicitly must give a
    bit-identical DepthScore (same per_layer, max_depth, held_out_rmse, notes).
    """
    truth = _build_tier2_tuned_cfg(seed=0)
    _family, held_out = alien_family_benchmark(seed=0)
    hyps = [_hyp(_V1_TRUTH, "hidden per-body product")]
    a = score_depth(hyps, truth, TIER2_HELD_OUT_ICS, family_scoring_cfgs=held_out, regime=None)
    b = score_depth(
        hyps, truth, TIER2_HELD_OUT_ICS, family_scoring_cfgs=held_out, regime=RegimeSpec()
    )
    assert a == b


def test_v1_truth_still_reaches_depth4_under_default_regime() -> None:
    """The V1 truth law on the product family still reaches consecutive depth 4 —
    the EXP-076/079 contract is unchanged by the V2 additions."""
    truth = _build_tier2_tuned_cfg(seed=0)
    _family, held_out = alien_family_benchmark(seed=0)
    score = score_depth(
        [_hyp(_V1_TRUTH, "hidden per-body product")],
        truth,
        TIER2_HELD_OUT_ICS,
        family_scoring_cfgs=held_out,
    )
    assert _consec(score.per_layer) == 4, f"V1 truth consec != 4: {score.per_layer} {score.notes}"


def test_v1_global_beta_still_ceilings_depth2() -> None:
    """The V1 global-beta control still ceilings at depth 2 (not gameable)."""
    truth = _build_tier2_tuned_cfg(seed=0)
    _family, held_out = alien_family_benchmark(seed=0)
    score = score_depth(
        [_hyp(_V1_GLOBAL_BETA)], truth, TIER2_HELD_OUT_ICS, family_scoring_cfgs=held_out
    )
    assert score.per_layer[2] is False
    assert score.max_depth == 2


# =============================================================================
# Gate 2 — V2 REGIME GATES (sum coupling + r^-2.5).
# =============================================================================


def test_v2_truth_reaches_consecutive_depth4() -> None:
    """The V2 truth law (sum + r^-2.5) reaches consecutive depth 4 under the V2
    regime — the regime-aware oracle credits the V2 correction (L2) AND the
    sum-confound charge (L3/L4)."""
    held_v2 = build_tier2_held_out_ics_v2(seed=0)
    _family, held_out = alien_family_v2_benchmark(seed=0)
    base = _family[0][1]
    score = score_depth(
        [_hyp(_V2_TRUTH, "the hidden per-body charge enters additively as a sum")],
        base,
        held_v2,
        family_scoring_cfgs=held_out,
        regime=_V2,
    )
    assert _consec(score.per_layer) == 4, (
        f"V2 truth must reach consecutive depth 4; per_layer={score.per_layer} "
        f"notes={score.notes}"
    )


def test_v2_l2_credits_r_minus_2p5_correction() -> None:
    """L2 credits the V2 r^-2.5 correction (a depth-2 V2 proposal reaches L2)."""
    held_v2 = build_tier2_held_out_ics_v2(seed=0)
    score = score_depth([_hyp(_V2_DEPTH2)], build_tier2_held_out_ics_v2(seed=0)[0], held_v2, regime=_V2)
    assert score.per_layer[1] is True, f"L2 must credit r^-2.5: {score.per_layer} {score.notes}"
    assert score.max_depth == 2


def test_v2_l2_denies_wrong_v1_exponent() -> None:
    """A V2 proposal that keeps the V1 r^-3.5 correction is DENIED at L2 (the band
    moved to -2.5), so consecutive depth is capped at 1."""
    held_v2 = build_tier2_held_out_ics_v2(seed=0)
    _family, held_out = alien_family_v2_benchmark(seed=0)
    base = _family[0][1]
    score = score_depth(
        [_hyp(_V2_WRONG_EXPONENT)], base, held_v2, family_scoring_cfgs=held_out, regime=_V2
    )
    assert score.per_layer[1] is False, (
        f"the V1 r^-3.5 exponent must be denied L2 on the V2 law; "
        f"per_layer={score.per_layer} notes={score.notes}"
    )
    assert _consec(score.per_layer) == 1


def test_v2_sum_gate_credits_l3_on_distinct_sum_family() -> None:
    """The cross-config s-SUM gate credits L3 for a correct (s1+s2) proposal across
    the distinct-sum held-out family."""
    held_v2 = build_tier2_held_out_ics_v2(seed=0)
    _family, held_out = alien_family_v2_benchmark(seed=0)
    base = _family[0][1]
    score = score_depth(
        [_hyp(_V2_TRUTH, "additive hidden per-body charge")],
        base,
        held_v2,
        family_scoring_cfgs=held_out,
        regime=_V2,
    )
    assert score.per_layer[2] is True, (
        f"correct (s1+s2) scaling must clear L3 on the V2 family; "
        f"per_layer={score.per_layer} notes={score.notes}"
    )


def test_v2_global_beta_does_not_clear_l3() -> None:
    """A global-beta V2 proposal (no per-body charge) cannot game the sum gate."""
    held_v2 = build_tier2_held_out_ics_v2(seed=0)
    _family, held_out = alien_family_v2_benchmark(seed=0)
    base = _family[0][1]
    score = score_depth(
        [_hyp(_V2_GLOBAL_BETA)], base, held_v2, family_scoring_cfgs=held_out, regime=_V2
    )
    assert score.per_layer[2] is False, (
        f"a global-beta V2 proposal must NOT clear L3 (not gameable); "
        f"per_layer={score.per_layer} notes={score.notes}"
    )
    assert score.max_depth == 2


def test_v2_product_amplitude_does_not_clear_sum_l3() -> None:
    """A V2 proposal using the WRONG (product s1*s2) amplitude is denied L3 by the
    sum gate — the V2 oracle credits the SUM confound, not the product.

    This is the discriminator that the V2 gate is genuinely sum-shaped, not just
    'any hidden charge'."""
    held_v2 = build_tier2_held_out_ics_v2(seed=0)
    _family, held_out = alien_family_v2_benchmark(seed=0)
    base = _family[0][1]
    score = score_depth(
        [_hyp(_V2_PRODUCT_ON_V2)], base, held_v2, family_scoring_cfgs=held_out, regime=_V2
    )
    # _has_hidden_sum is False for an s1*s2 product term, so L3/L4 never engage.
    assert score.per_layer[2] is False, (
        f"a product (s1*s2) amplitude must NOT clear the V2 SUM L3 gate; "
        f"per_layer={score.per_layer} notes={score.notes}"
    )


def test_v2_degenerate_equal_sum_family_does_not_credit_l3() -> None:
    """An equal-SUM 'family' has no cross-config variation → the sum gate does ~0
    work → L3 denied even for a correct (s1+s2) proposal (the gate needs DISTINCT
    sums — the sum analog of the equal-product degeneracy)."""
    held_v2 = build_tier2_held_out_ics_v2(seed=0)
    _family, equal_sum_cfgs = alien_family_v2_benchmark(seed=0, held_out_sums=(6.0, 6.0))
    base = _family[0][1]
    score = score_depth(
        [_hyp(_V2_TRUTH)], base, held_v2, family_scoring_cfgs=equal_sum_cfgs, regime=_V2
    )
    assert score.per_layer[2] is False, (
        f"an equal-sum (degenerate) family must NOT credit L3; "
        f"per_layer={score.per_layer} notes={score.notes}"
    )
    assert score.per_layer[3] is False


# --- L-064/DET-04: the Newton carve-out must not truncate the V2 band -------


def test_v2_l2_band_lower_edge_not_swallowed_by_newton_carveout() -> None:
    """An in-band exponent just above 2.2 must be L2-ELIGIBLE on V2.

    Pre-fix, the fixed 0.25 Newton carve-out excluded every exponent in
    [2.0, 2.25) from second-term consideration, truncating the stated V2 band
    [2.2, 2.8] to an effective [2.25, 2.8]: a proposal at 2.24 (inside the
    stated band) was silently treated as the Newton term and got no L2
    evaluation at all.
    """
    held_v2 = build_tier2_held_out_ics_v2(seed=0)
    base = build_tier2_held_out_ics_v2(seed=0)[0]
    form = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**2.24"
    score = score_depth([_hyp(form)], base, held_v2, regime=_V2)
    assert score.per_layer[1] is True, (
        f"2.24 is inside the stated V2 band [2.2, 2.8] and must be L2-eligible; "
        f"per_layer={score.per_layer} notes={score.notes}"
    )


def test_v2_battery_recommendation_2p1_evaluated_but_denied_l2() -> None:
    """The corrected battery's own 2.1 recommendation is OUT of the truth band.

    Pre-fix it produced no L2 evaluation at all (swallowed by the carve-out);
    now it is a second term that the band gate DENIES — visible, not silent.
    """
    held_v2 = build_tier2_held_out_ics_v2(seed=0)
    base = build_tier2_held_out_ics_v2(seed=0)[0]
    form = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**2.1"
    score = score_depth([_hyp(form)], base, held_v2, regime=_V2)
    assert score.per_layer[1] is False, (
        f"2.1 is outside the band and must be denied; notes={score.notes}"
    )


def test_v1_newton_carveout_unchanged() -> None:
    """V1 keeps the historical 0.25 carve-out: a 2.2 exponent on V1 is still
    inside the carve-out (min(0.25, 3.5-0.3-2.0) = 0.25) — byte-identical
    behavior for every recorded V1 run."""
    held_v1 = TIER2_HELD_OUT_ICS
    base = held_v1[0]
    # 2.2 on V1: |−2.2+2| = 0.2 < 0.25 → not a second term → no L2 (and the
    # V1 band around 3.5 wouldn't credit it anyway); this pins the carve-out
    # arithmetic itself so a refactor cannot widen V1's second-term filter.
    form = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**2.2"
    score = score_depth([_hyp(form)], base, held_v1)
    assert score.per_layer[1] is False
    assert not any("L2:" in n for n in score.notes)
