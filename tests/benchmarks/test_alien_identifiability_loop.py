"""Burst C unit 2 (Layer 1) — the identifiability loop on the REAL alien physics.

The autonomous scientist's move, end-to-end and $0: single charge product → diagnose
(the product-scaling is non-identifiable) → propose product-sets → rank → adopt a
multi-distinct-product design → re-diagnose → SOLVED. Grounded in the production force
law (amplitude is linear in the charge product), framed as identifying the
product-scaling exponent p (task #9 reconciliation).
"""

from __future__ import annotations

import numpy as np

from ascension.benchmarks.alien_identifiability_loop import (
    amplitude_predict,
    propose_charge_product_designs,
    real_amplitude_at_product,
    run_alien_charge_scaling_loop,
)
from ascension.diagnostics import practical_identifiability


def test_real_hidden_amplitude_is_linear_in_charge_product() -> None:
    """Grounding: the production hidden_charge_accel makes the amplitude EXACTLY
    beta*product (p=1 truth). amplitude(P)/P is constant across products."""
    beta = 0.7
    ratios = [real_amplitude_at_product(P, beta=beta) / P for P in (1.0, 2.0, 4.0, 6.0)]
    # All ratios equal beta → amplitude ∝ product (linear, exponent p=1).
    assert np.allclose(
        ratios, beta, rtol=1e-9
    ), f"amplitude must be linear in product; ratios={ratios}"


def test_single_product_design_is_non_identifiable() -> None:
    # One charge product → 1 observation, 2 params ([beta, p]) → rank ≤ 1 → non-id.
    v = practical_identifiability(amplitude_predict([1.0]), [0.7, 1.0], param_names=["beta", "p"])
    assert v.identifiable is False
    assert v.rank == 1


def test_distinct_products_design_is_identifiable() -> None:
    # ≥2 DISTINCT products → full rank → [beta, p] recoverable. The family fix.
    v = practical_identifiability(
        amplitude_predict([1.0, 2.0, 4.0]), [0.7, 1.0], param_names=["beta", "p"]
    )
    assert v.identifiable is True
    assert v.rank == 2


def test_equal_products_design_stays_non_identifiable() -> None:
    # Multiple configs but the SAME product → rows collinear → still rank 1. (Why the
    # family must span DISTINCT products — the L-051 enrichment, recovered.)
    v = practical_identifiability(
        amplitude_predict([2.0, 2.0, 2.0]), [0.7, 1.0], param_names=["beta", "p"]
    )
    assert v.identifiable is False
    assert v.rank == 1


def test_full_loop_solves_the_alien_charge_scaling_autonomously() -> None:
    """The headline: the loop starts on the single-product plateau, diagnoses
    non-identifiability, the ranker picks a multi-distinct-product experiment, and
    [beta, p] becomes identifiable → outcome 'solved'."""
    result = run_alien_charge_scaling_loop(beta=0.7, p_truth=1.0)
    assert result.final_identifiable is True
    assert result.outcome == "solved"
    # Round 0: the plateau was non-identifiable and an enriching design was chosen.
    s0 = result.steps[0]
    assert s0.identifiable is False
    assert s0.chosen in ("two_distinct_products", "family_4_products")
    assert "single_product" in s0.proposed


def test_proposer_offers_the_confounded_and_enriched_candidates() -> None:
    designs = propose_charge_product_designs(verdict=None)
    assert {"single_product", "two_distinct_products", "family_4_products"} <= set(designs)
