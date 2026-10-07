"""Phase 15.2-02 — the 6-layer discovery-depth scorer suite (make-or-break).

TDD RED→GREEN. This suite fully specifies the ``score_depth`` contract and the
``DepthScore`` DTO. It has three arms (15.2 PLAN <behavior>):

  POSITIVE — the truth law reaches its §10 layers; Newton-only stops at L1;
    Newton+α reaches L2 with the held-out RMSE-improvement gate (the TIER2-04
    bar).
  ADVERSARIAL / negative-control (D-08) — gaming is defeated by the
    behavioral/numerical gates, NOT the pattern match: decorative-s ⇏ L3,
    wrong-frequency cos ⇏ L4, drifting-Q ⇏ L5, false-symmetry ⇏ L6,
    rationale-only ⇏ any layer its math doesn't support.
  DETERMINISM + parser-safety — same input → identical output (sorted
    iteration, tuple-not-dict per L-020); the s1*s2 subscripted-name regression
    (s1 not collapsed to s*1, the L-013 guard).

$0 LLM — deterministic sympy/scipy/structural + the locked-truth simulator.
The truth couplings are ``TIER2_TUNED_PARAMS`` (G=1, α=0.05, β=0.02, γ=0.7,
κ=0.3); the held-out set is ``TIER2_HELD_OUT_ICS``.

Truth force-law signatures the predicates match (15.2 RESEARCH §six predicates):
  L1: ~ c/r**2            L2: + alpha/r**3.5
  L3/L4: + beta*s1*s2*(1 - cos(gamma*r))/r**2
  L5: Q = kinetic + U(r) + kappa*s**2*v_perp**2 (conserved)
  L6: parity-charge symmetry P:(x->-x, v->-v, s->-s)
"""

from __future__ import annotations

from ascension.benchmarks.alien_depth import DepthScore, score_depth
from ascension.simulator.config import (
    TIER2_HELD_OUT_ICS,
    build_tier2_held_out_ics,
)
from ascension.simulator.types import AlienConfig

# -----------------------------------------------------------------------------
# Fixtures — the locked truth config + symbolic_form string builders
# -----------------------------------------------------------------------------


def _truth_cfg() -> AlienConfig:
    """The TIER2 tuned operating point as an AlienConfig (a near-circular IC).

    Couplings come from TIER2_TUNED_PARAMS verbatim; the IC is the first
    held-out config (a real, integrable orbit) so the truth cfg the scorer
    compares against is the same physics the held-out set spans.
    """
    base = TIER2_HELD_OUT_ICS[0]
    return base


def _hyp(symbolic_form: str, rationale: str = "an orbit law", confidence: float = 0.8) -> dict:
    """A hypothesis row in the dict shape the depth scorer reads from nodes.

    Mirrors round_close.py's ``content`` JSONB read: a dict carrying
    ``symbolic_form`` / ``rationale`` / ``confidence``.
    """
    return {
        "symbolic_form": symbolic_form,
        "rationale": rationale,
        "confidence": confidence,
    }


# The §10 truth-law signatures as parseable symbolic_form strings (whitelist:
# G, alpha, beta, gamma, kappa, r, s1, s2, m1, m2, cos, sin, exp, sqrt).
_FULL_TRUTH = "G*m1*m2*(1/r**2 + alpha/r**3.5) + beta*s1*s2*(1 - cos(gamma*r))/r**2"
_NEWTON_ONLY = "G*m1*m2/r**2"
_NEWTON_ALPHA = "G*m1*m2*(1/r**2 + alpha/r**3.5)"
_NEWTON_ALPHA_HIDDEN = "G*m1*m2*(1/r**2 + alpha/r**3.5) + beta*s1*s2*(1 - cos(gamma*r))/r**2"
# A conservation-claim expression (kinetic + velocity-coupled hidden term).
_CONSERVED_Q = "m1*v1**2/2 + U + kappa*s1**2*v_perp**2/2"


# -----------------------------------------------------------------------------
# DTO contract
# -----------------------------------------------------------------------------


def test_depthscore_is_frozen_slots_with_invariant_docstring() -> None:
    """DepthScore is a frozen, slotted DTO carrying the documented invariant.

    Mirrors AlienAuditResult discipline: a grep-able docstring literal states
    ``max_depth == (highest index i where per_layer[i] is True) + 1, or 0``.
    """
    score = score_depth([_hyp(_NEWTON_ONLY)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert isinstance(score, DepthScore)
    # frozen
    try:
        score.max_depth = 99  # type: ignore[misc]
        raise AssertionError("DepthScore must be frozen")
    except (AttributeError, Exception):
        pass
    # slots (no __dict__)
    assert not hasattr(score, "__dict__"), "DepthScore must use slots"
    # per_layer is a length-6 tuple of bools (tuple-not-dict, L-020)
    assert isinstance(score.per_layer, tuple)
    assert len(score.per_layer) == 6
    assert all(isinstance(b, bool) for b in score.per_layer)
    # invariant stated as a docstring literal for a grep regression
    assert "max_depth ==" in (DepthScore.__doc__ or "")


def test_max_depth_matches_per_layer_invariant() -> None:
    """max_depth == highest True index + 1 (or 0 if none) — checked on truth."""
    score = score_depth([_hyp(_FULL_TRUTH)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    highest = max((i for i, ok in enumerate(score.per_layer) if ok), default=-1)
    expected = highest + 1
    assert score.max_depth == expected


# -----------------------------------------------------------------------------
# POSITIVE arm — the truth law reaches its layers
# -----------------------------------------------------------------------------


def test_full_truth_reaches_l1_through_l4() -> None:
    """The full truth law scores L1=L2=L3=L4 True; max_depth >= 4."""
    score = score_depth([_hyp(_FULL_TRUTH)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.per_layer[0] is True, "L1 (inverse-square) must hold"
    assert score.per_layer[1] is True, "L2 (alpha correction) must hold"
    assert score.per_layer[2] is True, "L3 (hidden-charge existence) must hold"
    assert score.per_layer[3] is True, "L4 (hidden-force form) must hold"
    assert score.max_depth >= 4


def test_newton_only_reaches_l1_not_l2() -> None:
    """A Newton-only proposal scores L1 True, L2 False; max_depth == 1."""
    score = score_depth([_hyp(_NEWTON_ONLY)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.per_layer[0] is True
    assert score.per_layer[1] is False
    assert score.max_depth == 1


def test_newton_plus_alpha_reaches_l2_with_numerical_gate() -> None:
    """Newton+alpha reaches L2 (TIER2-04 bar) — structural AND held-out RMSE gate.

    max_depth >= 2 requires BOTH the second-radial-term structural test AND the
    held-out RMSE-improvement gate (Newton+correction beats Newton-only on
    TIER2_HELD_OUT_ICS). The numerical oracle, not the pattern match, makes the
    depth claim falsifiable.
    """
    score = score_depth([_hyp(_NEWTON_ALPHA)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.per_layer[1] is True, "L2 must hold for Newton+alpha"
    assert score.max_depth >= 2
    # the gate recorded the held-out RMSE it used to credit L2
    assert score.held_out_rmse is not None


def test_depth_two_gate_blocks_when_correction_does_not_improve() -> None:
    """L2 is NOT credited if the held-out RMSE gate can't run / doesn't improve.

    The negative control for the TIER2-04 bar: a proposal with a SECOND radial
    term whose exponent does NOT match -3.5 and does not improve the Newton-only
    held-out residual must NOT score L2 (a vacuous "second term" is not depth 2).
    Uses a +1/r**1.5 *repulsive-shaped* term far from the truth's attractive
    alpha/r**3.5 — structurally a second power but numerically not the truth's
    correction.
    """
    bogus = "G*m1*m2*(1/r**2 + 5.0/r**1.2)"
    score = score_depth([_hyp(bogus)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.per_layer[0] is True, "L1 still holds (it has 1/r**2)"
    assert score.per_layer[1] is False, (
        "a second radial term that does not improve held-out RMSE must NOT "
        "score L2 — the numerical gate is non-negotiable (TIER2-04)"
    )


# -----------------------------------------------------------------------------
# ADVERSARIAL / negative-control arm (D-08) — gaming defeated by the gates
# -----------------------------------------------------------------------------


def test_decorative_s_does_not_reach_l3() -> None:
    """Decorative-s: a free symbol that does NO work does NOT score L3.

    The proposal sprinkles s1*s2 into a term that, when the hidden symbols go to
    0, leaves the held-out RMSE unchanged (the s does no work). The L3 behavioral
    gate (held-out RMSE with hidden->0 must be materially worse) defeats this.
    Here the s-term is multiplied by a tiny coefficient (~0) so it contributes
    nothing to the dynamics.
    """
    decorative = "G*m1*m2*(1/r**2 + alpha/r**3.5) + 1e-12*s1*s2*(1 - cos(gamma*r))/r**2"
    score = score_depth(
        [_hyp(decorative, rationale="this posits a hidden per-body charge")],
        _truth_cfg(),
        TIER2_HELD_OUT_ICS,
    )
    assert score.per_layer[2] is False, (
        "a decorative s that does no work (held-out RMSE unchanged with s->0) "
        "must NOT score L3 even though s1*s2 appears structurally and the "
        "rationale claims a hidden charge"
    )


def test_wrong_frequency_cos_does_not_reach_l4() -> None:
    """Wrong-frequency cos: cos(k*r) with k far from gamma=0.7 does NOT score L4.

    The L4 numerical gate requires gamma_est ~ 0.7 AND held-out RMSE < 1e-3.
    A cos(5*r) hidden term has the right STRUCTURE (s1*s2, 1/r**2 envelope, a
    cos) but the wrong frequency, so the numerical reproduction fails.
    """
    wrong_freq = "G*m1*m2*(1/r**2 + alpha/r**3.5) + beta*s1*s2*(1 - cos(5.0*r))/r**2"
    score = score_depth([_hyp(wrong_freq)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.per_layer[3] is False, (
        "a cos with frequency far from gamma=0.7 must NOT score L4 — the "
        "numerical gate (gamma_est~0.7 AND held-out RMSE<1e-3) defeats it"
    )


def test_drifting_q_does_not_reach_l5() -> None:
    """Drifting-Q: a claimed 'conserved' quantity that drifts does NOT score L5.

    A quantity built from the WRONG structure (e.g. a plain kinetic+U with no
    tangential-inertia term, or a coefficient that makes it drift) is evaluated
    along a fresh held-out trajectory; its drift exceeds the _Q_DRIFT_BAR (1e-6),
    so L5 fails even if the rationale asserts conservation.
    """
    # kinetic + U but with NO kappa*s**2*v_perp**2 augmentation -> NOT invariant
    # (RESEARCH: §10's literal Q without tangential inertia drifts ~28%).
    drifting = "m1*v1**2/2 + U"
    score = score_depth(
        [_hyp(drifting, rationale="this quantity Q is conserved over the orbit")],
        _truth_cfg(),
        TIER2_HELD_OUT_ICS,
    )
    assert score.per_layer[4] is False, (
        "a quantity that drifts on a held-out trajectory must NOT score L5 — "
        "rationale claiming conservation is never sufficient (traj.Q drift gate)"
    )


def test_force_law_with_kappa_velocity_does_not_reach_l5() -> None:
    """Phase 15.2 TIER2-03 regression: a FORCE LAW must NOT false-positive into L5.

    The live TIER2-03 run produced ``G*m2/r**2 - kappa*v1**2`` — the acceleration
    RHS the prompt actually asks for — and the prior L5 gate wrongly credited it
    (bare ``kappa`` matched "coupled", and the numerical gate checked the TRUTH
    Q's always-bounded drift). A force law is NOT a conserved quantity: it has no
    potential ``U`` and no hidden charge. It must score L1 (inverse-square) only.
    """
    force_law = "G*m2/r**2 - kappa*v1**2"
    score = score_depth([_hyp(force_law)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.per_layer[4] is False, (
        "a force law with a kappa*v**2 term must NOT score L5 — it is not a "
        "conserved quantity (no potential U, no hidden charge)"
    )
    assert (
        score.max_depth == 1
    ), "the honest depth of G*m2/r**2 - kappa*v1**2 is L1 (inverse-square) only"


def test_force_law_with_hidden_velocity_coupling_no_potential_does_not_reach_l5() -> None:
    """A force law carrying a hidden-charge velocity coupling but NO U ⇏ L5.

    Even a force law that includes a genuine hidden-charge product coupled to a
    velocity (``kappa*s1**2*v1**2``) is still a force law, not a conserved
    quantity — without the potential ``U`` term it cannot be the energy-like
    invariant Q. The ``U`` requirement is the force-law discriminator.
    """
    force_law = "G/r**2 + kappa*s1**2*v1**2"
    score = score_depth([_hyp(force_law)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.per_layer[4] is False, (
        "a force law with a hidden-charge velocity coupling but no potential U "
        "must NOT score L5 (it is not a conserved energy-like quantity)"
    )


def test_genuine_conserved_quantity_still_reaches_l5() -> None:
    """The genuine conserved Q (T + U + kappa*s**2*v_perp**2) STILL reaches L5.

    Guards the TIER2-03 fix in the other direction: tightening the structural
    check (require U + a hidden-charge product) must not reject a genuine
    conserved-quantity proposal. NOTE: the current Council prompt does not elicit
    such a proposal (it asks for the force law); this test documents that the L5
    gate stays correct for the day a prompt DOES ask for a conserved quantity.
    """
    score = score_depth([_hyp(_CONSERVED_Q)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.per_layer[4] is True, (
        "the genuine conserved quantity m1*v1**2/2 + U + kappa*s1**2*v_perp**2/2 "
        "must still score L5 (T + U + hidden-charge coupling)"
    )


def test_false_symmetry_does_not_reach_l6() -> None:
    """False-symmetry: a law NOT invariant under P:(x->-x,v->-v,s->-s) ⇏ L6.

    A law with an ODD power of r in a way that breaks the parity-charge
    invariance (or one whose hidden term is linear in s rather than the s1*s2
    even product) is checked by sympy substitution; the invariance fails, so L6
    is not credited even when the rationale claims a parity symmetry.
    """
    # A hidden term LINEAR in a single s (odd under s->-s) breaks the symmetry.
    asym = "G*m1*m2/r**2 + beta*s1*(1 - cos(gamma*r))/r**2"
    score = score_depth(
        [_hyp(asym, rationale="invariant under parity and charge sign flip")],
        _truth_cfg(),
        TIER2_HELD_OUT_ICS,
    )
    assert score.per_layer[5] is False, (
        "a law that is NOT genuinely invariant under P:(x->-x,v->-v,s->-s) "
        "must NOT score L6 — the sympy invariance check, not the prose, decides"
    )


def test_rationale_only_confers_no_layer_beyond_its_math() -> None:
    """Rationale-only: prose claims do NOT confer a layer the math doesn't support.

    A bare Newton law whose rationale name-drops "hidden charge", "conserved",
    and "symmetric" scores at most L1 (the layer its symbolic_form supports).
    Rationale is at most a logged tiebreaker, never authoritative.
    """
    score = score_depth(
        [
            _hyp(
                _NEWTON_ONLY,
                rationale=(
                    "there is a hidden per-body charge, an unobserved conserved "
                    "quantity, and a parity symmetry in this system"
                ),
            )
        ],
        _truth_cfg(),
        TIER2_HELD_OUT_ICS,
    )
    assert score.per_layer[2] is False, "rationale alone never confers L3"
    assert score.per_layer[4] is False, "rationale alone never confers L5"
    assert score.per_layer[5] is False, "rationale alone never confers L6"
    assert score.max_depth <= 1, "Newton-only math caps depth at L1 regardless of prose"


# -----------------------------------------------------------------------------
# DETERMINISM + parser-safety arm
# -----------------------------------------------------------------------------


def test_determinism_identical_output_across_two_calls() -> None:
    """Same hypothesis set -> identical max_depth + per_layer tuple, twice."""
    hyps = [_hyp(_NEWTON_ONLY), _hyp(_FULL_TRUTH), _hyp(_NEWTON_ALPHA)]
    a = score_depth(hyps, _truth_cfg(), TIER2_HELD_OUT_ICS)
    b = score_depth(hyps, _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert a.max_depth == b.max_depth
    assert a.per_layer == b.per_layer


def test_order_independence_shuffled_input_same_output() -> None:
    """Output is independent of hypothesis input order (sorted set iteration)."""
    hyps = [_hyp(_NEWTON_ONLY), _hyp(_FULL_TRUTH), _hyp(_NEWTON_ALPHA)]
    forward = score_depth(hyps, _truth_cfg(), TIER2_HELD_OUT_ICS)
    reversed_ = score_depth(list(reversed(hyps)), _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert forward.max_depth == reversed_.max_depth
    assert forward.per_layer == reversed_.per_layer


def test_subscripted_name_parses_without_collapse_l013() -> None:
    """s1*s2 parses correctly (s1 NOT collapsed to s*1) — the L-013 regression.

    The full truth law carries s1*s2; if the parser collapsed s1->s*1 and
    s2->s*2 the hidden term would degenerate to 2*s**... and the structural L3
    match would break. Reaching L3 on the full truth proves the subscripted
    names survived parsing.
    """
    score = score_depth([_hyp(_FULL_TRUTH)], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.per_layer[2] is True, (
        "s1*s2 must parse as two distinct symbols (L-013): if collapsed to s*1, "
        "the hidden-charge-product structural test would not fire"
    )


def test_missing_symbolic_form_is_skipped_not_fatal() -> None:
    """A hypothesis with no symbolic_form is logged+skipped, never silent-drop.

    Mirrors round_close.py:420-453 read-and-iterate: skip+WARN. The scorer must
    still score the remaining valid hypothesis (no crash, no silent drop).
    """
    rows = [
        {"rationale": "no symbolic form here", "confidence": 0.5},  # missing
        _hyp(_NEWTON_ONLY),
    ]
    score = score_depth(rows, _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.max_depth == 1, "the valid Newton hypothesis still scores L1"


def test_unparseable_symbolic_form_is_skipped_not_fatal() -> None:
    """A garbage symbolic_form is logged+skipped, never crashes the scorer."""
    rows = [
        _hyp("this is not @@ valid sympy ++"),
        _hyp(_NEWTON_ONLY),
    ]
    score = score_depth(rows, _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.max_depth == 1, "the valid Newton hypothesis still scores L1"


def test_empty_hypothesis_set_scores_zero() -> None:
    """No hypotheses -> max_depth 0, all per_layer False (no vacuous credit)."""
    score = score_depth([], _truth_cfg(), TIER2_HELD_OUT_ICS)
    assert score.max_depth == 0
    assert score.per_layer == (False, False, False, False, False, False)


def test_held_out_ics_are_nonvacuous_multiple_orbits() -> None:
    """Sanity: TIER2_HELD_OUT_ICS spans >1 orbit shape (non-vacuity guarantee).

    The depth predicates lean on the held-out set spanning a RANGE of r0/frac so
    a layer can't pass on one orbit shape. This pins that the set is the >=12
    multi-orbit set the scorer relies on (15.1 D-09).
    """
    ics = build_tier2_held_out_ics()
    assert len(ics) >= 12
    # at least two distinct starting radii
    r0s = {cfg.ics_pos[1][0] for cfg in ics}
    assert len(r0s) >= 2
