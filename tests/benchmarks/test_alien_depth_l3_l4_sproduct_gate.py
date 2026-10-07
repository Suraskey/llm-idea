"""Phase 19-04 (AUTO-04, headline) — L3/L4 cross-config s-product scorer gate.

The §10 cross-config gate credits the hidden per-body charge ONLY when per-config
charge-product scaling does >= 5% RSS work BEYOND a single global amplitude:

  (a) a correct per-config ``s1*s2`` proposal scored across the multi-charge
      family clears L3 (the cross-config work is real) => max_depth >= 3;
  (b) a global-beta proposal (NO s1*s2) on the SAME family does NOT clear L3 =>
      max_depth == 2 (not gameable — structural ``_has_hidden_product`` denies it);
  (c) the single-config path is byte-unchanged: a depth-2 proposal on the
      equal-charge benchmark ceilings at max_depth == 2 (the control);
  (d) a DEGENERATE equal-product family yields ~0 cross-config work, so even an
      ``s1*s2`` proposal ceilings at depth 2 — the gate genuinely needs DISTINCT
      products (the single-charge degeneracy that Session 029 proved).

L1/L2 run on the product-1 ``TIER2_HELD_OUT_ICS`` in BOTH modes (charge-
independent + calibrated there). $0, deterministic sympy/scipy — no LLM (the
``test_no_llm_in_alien_depth`` gate must still pass after this extension).
"""

from __future__ import annotations

from ascension.benchmarks.alien_depth import score_depth
from ascension.benchmarks.alien_fixtures import (
    _build_tier2_tuned_cfg,
    alien_family_benchmark,
)
from ascension.simulator.config import TIER2_HELD_OUT_ICS

# Force-law proposals (symbolic_form RHS strings). The TRUTH law:
#   -G*m_j/r**2 - G*alpha*m_j/r**3.5 + beta*s1*s2*(1 - cos(gamma*r))/r**2
_PROPOSAL_S1S2 = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**3.5 + beta*s1*s2*(1 - cos(gamma*r))/r**2"
_PROPOSAL_GLOBAL_BETA = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**3.5 + beta*(1 - cos(gamma*r))/r**2"
_PROPOSAL_DEPTH2 = "-G*m1*m2/r**2 - G*alpha*m1*m2/r**3.5"


def _hyp(symbolic_form: str, rationale: str = "") -> dict:
    return {"symbolic_form": symbolic_form, "rationale": rationale}


def test_a_s1s2_proposal_on_family_clears_l3() -> None:
    """Correct per-config s1*s2 scaling on the distinct-product family => L3."""
    base_cfg = _build_tier2_tuned_cfg(seed=0)
    _family, held_out = alien_family_benchmark(seed=0)  # held-out products {3,5}
    score = score_depth(
        [_hyp(_PROPOSAL_S1S2, "the hidden per-body factor enters as a product")],
        base_cfg,
        TIER2_HELD_OUT_ICS,
        family_scoring_cfgs=held_out,
    )
    assert score.per_layer[2] is True, (
        f"correct s1*s2 scaling must clear L3 on the family; "
        f"per_layer={score.per_layer} notes={score.notes}"
    )
    assert (
        score.max_depth >= 3
    ), f"max_depth should reach >= 3; got {score.max_depth} notes={score.notes}"


def test_b_global_beta_proposal_on_family_stays_depth2() -> None:
    """A global-beta proposal (no per-body charge) cannot game L3 on the family."""
    base_cfg = _build_tier2_tuned_cfg(seed=0)
    _family, held_out = alien_family_benchmark(seed=0)
    score = score_depth(
        [_hyp(_PROPOSAL_GLOBAL_BETA)],
        base_cfg,
        TIER2_HELD_OUT_ICS,
        family_scoring_cfgs=held_out,
    )
    assert score.per_layer[2] is False, (
        f"a global-beta proposal must NOT clear L3 (not gameable); "
        f"per_layer={score.per_layer} notes={score.notes}"
    )
    assert (
        score.max_depth == 2
    ), f"global-beta should ceiling at L2; got {score.max_depth} notes={score.notes}"


def test_c_single_config_control_ceilings_at_depth2() -> None:
    """The single-config path (family_scoring_cfgs=None) is byte-unchanged."""
    truth_cfg = _build_tier2_tuned_cfg(seed=0)
    score = score_depth([_hyp(_PROPOSAL_DEPTH2)], truth_cfg, TIER2_HELD_OUT_ICS)
    assert score.max_depth == 2, (
        f"single-config depth-2 proposal must ceiling at 2 (control unchanged); "
        f"got {score.max_depth} per_layer={score.per_layer} notes={score.notes}"
    )


def test_d_degenerate_equal_product_family_does_not_credit_l3() -> None:
    """An equal-product 'family' has no cross-config variation, so even an s1*s2
    proposal does no per-config work => L3 denied (the gate needs DISTINCT
    products — the single-charge degeneracy)."""
    base_cfg = _build_tier2_tuned_cfg(seed=0)
    # Two scoring configs at the SAME product => zero cross-config variation.
    _family, equal_product_cfgs = alien_family_benchmark(seed=0, held_out_products=(7.0, 7.0))
    score = score_depth(
        [_hyp(_PROPOSAL_S1S2)],
        base_cfg,
        TIER2_HELD_OUT_ICS,
        family_scoring_cfgs=equal_product_cfgs,
    )
    # The HEADLINE gate (the L3/L4 force-law layers) is DENIED — an equal-product
    # family carries no cross-config signal, so the per-config s-product scaling
    # does ~0 RSS work beyond a global amplitude.
    assert score.per_layer[2] is False, (
        f"an equal-product (degenerate) family must NOT credit L3; "
        f"per_layer={score.per_layer} notes={score.notes}"
    )
    assert score.per_layer[3] is False, (
        f"L4 must also be denied on a degenerate family; "
        f"per_layer={score.per_layer} notes={score.notes}"
    )
    # NOTE: L6 (parity-charge symmetry) DOES fire here — sympy verifies that any
    # s1*s2 law is invariant under s->-s, which is ORTHOGONAL to the cross-config
    # identifiability gate this plan governs. That symbolic credit is the
    # pre-existing, documented L5/L6-not-honestly-elicited behavior (the prompt
    # asks only for the force law, so L1-L4 is the honest depth metric — see the
    # alien_depth module docstring). The headline L3/L4 gate correctly denies the
    # degenerate family, which is exactly what this test asserts.
