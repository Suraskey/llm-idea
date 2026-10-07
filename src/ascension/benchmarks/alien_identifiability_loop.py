"""Burst C unit 2 (Layer 1) — the identifiability-driven loop instantiated on the
REAL Alien Universe, end-to-end, no-LLM.

This wires the domain-agnostic engine (`diagnostics.engine.run_identifiability_loop`)
to the actual alien physics, demonstrating the full scientist's move autonomously:

    plateau (one charge configuration)
      → diagnose: the hidden-charge scaling is NON-identifiable from a single product
      → propose candidate experiments (charge-product sets)
      → identifiability-driven ranking picks one with DISTINCT products
      → "service" it (observe those configurations)
      → re-diagnose: now identifiable → SOLVED

The reconciliation that makes this a clean PARAMETER-identifiability problem (task #9):
the alien hidden charges s1, s2 are only-product-identifiable (only s1*s2 ever acts), so
"identify s1, s2" is the wrong question. The RIGHT question — and the one the
multi-charge family answers — is whether the oscillatory amplitude SCALES with the
(oracle-known) per-config charge product or is a fixed constant. Model that as

    amplitude(config) = beta * product(config) ** p

where ``p`` is the product-scaling exponent (p=0 → constant / no hidden charge; p=1 →
the true charge scaling). Identifying ``[beta, p]`` is exactly "is there a hidden charge,
and how does it scale?". With ONE product the sensitivity matrix is 1×2 → rank ≤ 1 →
non-identifiable; with ≥2 DISTINCT products it is full rank → identifiable. That is the
multi-charge-family fix, recovered as parameter identifiability.

The model is FAITHFUL to the real force law: ``simulator.physics.hidden_charge_accel``
makes the hidden-force amplitude exactly ``beta * (s_i*s_j) * (1-cos(gamma r))/r**2`` — i.e.
linear in the charge product (p=1 is the ground truth). ``real_amplitude_at_product``
confirms this against the production simulator, so the loop is grounded, not a toy.

Layer 1 (this module) uses a DETERMINISTIC proposer (no LLM) so the whole loop is $0 and
testable — the substrate. Layer 2 swaps the proposer for the live Council (ACTEXP-02);
the loop, ranking, and fit are unchanged and stay no-LLM (§22.6 / §17).

LLM-free: imports numpy + the diagnostics engine + the simulator physics only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ascension.diagnostics.engine import LoopResult, run_identifiability_loop
from ascension.diagnostics.identifiability import PredictFn
from ascension.simulator.physics import hidden_charge_accel

# Reference radii at which the hidden-charge amplitude is "observed". 1 - cos(gamma r)
# is non-degenerate here so the hidden term is present.
_REF_RADII = (1.0, 1.5, 2.0, 2.5, 3.0)
_TRUTH_GAMMA = 0.7  # SCOPE §10 / TIER2; the oscillation frequency.


def real_amplitude_at_product(
    product: float, *, beta: float = 1.0, gamma: float = _TRUTH_GAMMA
) -> float:
    """The REAL simulator's hidden-charge force amplitude at a charge product, isolated.

    Evaluates ``hidden_charge_accel`` for a 2-body config whose charges multiply to
    ``product`` (charges = (1, product)), summed over the reference radii, divided by the
    charge-independent oscillatory envelope ``sum (1-cos(gamma r))/r**2``. By construction
    of the force law this returns ``beta * product`` exactly — confirming the amplitude is
    LINEAR in the product (the ``p=1`` ground truth the model in this module assumes).
    """
    charges = np.array([1.0, float(product)], dtype=np.float64)
    envelope = 0.0
    measured = 0.0
    for r in _REF_RADII:
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        # hidden_charge_accel uses cfg.beta=beta, cfg.gamma=gamma via its args.
        a = hidden_charge_accel(pos, charges, beta, gamma)
        measured += a[1][0]  # radial (x) hidden-force on body 2
        envelope += (1.0 - np.cos(gamma * r)) / r**2
    # measured = -beta*product*envelope (attractive sign); return the magnitude/envelope.
    return abs(measured) / envelope if envelope != 0 else 0.0


def amplitude_predict(products: Sequence[float]) -> PredictFn:
    """A candidate OBSERVATION DESIGN: model the amplitude across the observed charge
    ``products`` as ``beta * product ** p``. Returns predict([beta, p]) -> amplitudes.

    The design IS the set of products observed; identifiability of [beta, p] is a property
    of that set (≥2 distinct products ⇒ full rank ⇒ identifiable).
    """
    prods = np.asarray(products, dtype=np.float64)

    def predict(theta: np.ndarray) -> np.ndarray:
        beta, p = float(theta[0]), float(theta[1])
        return beta * prods**p

    return predict


def propose_charge_product_designs(verdict: object) -> Mapping[str, PredictFn]:
    """Deterministic stand-in for the Council's ACTEXP-02 move (Layer 1).

    On a non-identifiable plateau, offer candidate charge-product sets — including the
    confounded single-product baseline AND multi-distinct-product enrichments. The
    identifiability-driven ranker selects which one actually breaks the degeneracy; this
    proposer does NOT pre-judge that (it just generates candidates), mirroring how the
    Council proposes and the deterministic ranker decides.
    """
    return {
        "single_product": amplitude_predict([1.0]),
        "two_distinct_products": amplitude_predict([1.0, 2.0]),
        "family_4_products": amplitude_predict([1.0, 2.0, 4.0, 6.0]),
    }


def run_alien_charge_scaling_loop(
    *,
    beta: float = 1.0,
    p_truth: float = 1.0,
    max_rounds: int = 3,
    propose=propose_charge_product_designs,
) -> LoopResult:
    """Run the full diagnose → design → re-diagnose loop on the alien charge-scaling
    question, starting from the single-product PLATEAU.

    Returns the LoopResult. With the default proposer this reaches outcome 'solved':
    the single-product start is non-identifiable, the ranker picks a multi-distinct-product
    design, and [beta, p] becomes identifiable. The ``propose`` arg is the ACTEXP-02 seam —
    pass the live Council here in Layer 2.
    """
    initial = amplitude_predict([1.0])  # the plateau: one charge product observed
    return run_identifiability_loop(
        initial,
        [beta, p_truth],
        propose_designs=propose,
        max_rounds=max_rounds,
        param_names=["beta", "p"],
    )
