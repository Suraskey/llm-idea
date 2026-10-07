"""Burst C unit 2 (Layer 2) — the identifiability loop CLOSED BY THE LIVE COUNCIL.

Layer 1 (`alien_identifiability_loop.py`) ran the diagnose → design → re-diagnose loop with a
DETERMINISTIC proposer ($0 substrate). Layer 2 swaps in the live Council's director (ACTEXP-02):
on the plateau the Council DECIDES the next experiment, an adapter translates that decision into
candidate observation designs, the deterministic OED ranker INDEPENDENTLY picks the rank-restoring
one, and the diagnostic re-checks. The loop, ranking, and fit stay no-LLM (§22.6 / §17) — only the
proposer calls an LLM. The whole thing is the contribution as ONE artifact: an agent that hits an
identifiability wall, designs the experiment the math confirms breaks it, and re-diagnoses to solved.

HONESTY BOUNDARY — what counts as the Council "generating the design" vs flipping a fixed switch
(the AUDIT-036 STRAT-02 question, decided here, in the open):

  The director's move vocabulary is FIXED — two moves: ``request_new_initial_condition`` and
  ``request_multi_charge_family`` (the model emits the leak-safe ``request_varied_family``). So the
  Council does NOT emit free-form charge products; it CHOOSES A CATEGORY of experiment. The closure
  this module demonstrates is therefore:

    1. the live Council, on a degenerate single-configuration plateau, AUTONOMOUSLY chooses to VARY
       THE CHARGE FAMILY rather than request a new initial condition (its design DECISION);
    2. the adapter maps that choice to the corresponding observation designs (a multi-distinct-
       product family) alongside the status-quo single-product design;
    3. the deterministic OED ranker — truth-blind, no LLM — INDEPENDENTLY selects the family as the
       rank-restoring experiment and rejects the single/equal-product designs;
    4. the diagnostic re-checks → identifiable → SOLVED.

  The Council's reasoning (which experiment category breaks the degeneracy) and the math (which design
  restores rank) AGREE, end-to-end, in one trace. The Council's choice is LOAD-BEARING: if it instead
  picks a new initial condition (same single charge product), the adapter offers only single-product
  designs, the OED cannot restore rank, and the loop honestly fails to solve — so the demonstration is
  NOT a relabeled switch-flip, it is the agent's decision determining whether the degeneracy breaks.

  LIMITATION (stated, for the paper's limitations section): the fixed two-move vocabulary means the
  agent picks a category, not free-form products; the adapter maps category → products. A free-form
  design-generation vocabulary (the Council emits the products) is future work and strengthens the
  claim — it does not change the closure demonstrated here.

LLM-free at its core: this module imports numpy + the Layer-1 designs + the director's move-name
constants only. The injected ``decide_move`` MAY call an LLM (the live director); this module does not.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ascension.benchmarks.alien_identifiability_loop import (
    amplitude_predict,
    run_alien_charge_scaling_loop,
)
from ascension.diagnostics.engine import LoopResult
from ascension.diagnostics.identifiability import PredictFn

# The director's INTERNAL move names (imported as constants, not magic strings, so a rename in
# director.py is caught by import, not silently mismatched). Deferred to avoid any council↔benchmarks
# module-level import cycle.


def _move_names() -> tuple[str, str]:
    # The live Council proposer is not part of the public release; these are the
    # two move tokens its director emits (council/director.py in the development repo).
    try:
        from ascension.council.director import (  # type: ignore[import-not-found]
            REQUEST_MULTI_CHARGE_FAMILY,
            REQUEST_NEW_INITIAL_CONDITION,
        )
    except ImportError:
        return "request_multi_charge_family", "request_new_initial_condition"

    return REQUEST_MULTI_CHARGE_FAMILY, REQUEST_NEW_INITIAL_CONDITION


# A minimal structural protocol for the director's decision: anything with a ``.move`` string.
DecideMoveFn = Callable[[object], object]  # (verdict) -> NextMove-like (has .move: str)


def council_proposer(next_move: object) -> Mapping[str, PredictFn]:
    """Translate the Council director's move DECISION into candidate observation designs.

    The honesty boundary lives here (see module docstring): the Council picks a move CATEGORY; this
    maps the category to the designs the deterministic OED ranker then scores. The Council's choice
    is load-bearing — ``request_multi_charge_family`` offers the rank-restoring family alongside the
    status quo; ``request_new_initial_condition`` offers only single-product designs (no rank restore).

    Args:
      next_move: the director's decision; only its ``.move`` attribute (a string) is read.

    Returns:
      {label: PredictFn} candidate designs for the OED ranker, or ``{}`` if the move is unrecognized
      (the loop then stops honestly — no design proposed).
    """
    family_move, ic_move = _move_names()
    move = getattr(next_move, "move", None)

    if move == family_move:
        # The Council chose to vary the charge family. Offer the status-quo single product plus
        # multi-distinct-product enrichments; the OED ranker decides which actually restores rank.
        return {
            "council→single_product (status quo)": amplitude_predict([1.0]),
            "council→vary_family_2_distinct": amplitude_predict([1.0, 2.0]),
            "council→vary_family_4_distinct": amplitude_predict([1.0, 2.0, 4.0, 6.0]),
        }
    if move == ic_move:
        # A new initial condition keeps the SAME single charge product → cannot break the charge
        # degeneracy. Offer only single-product variants; the OED will not restore rank (honest fail).
        return {"council→new_initial_condition (still single product)": amplitude_predict([1.0])}
    # Unknown / no move → no proposal; the loop stops honestly stuck.
    return {}


def run_alien_council_loop(
    decide_move: DecideMoveFn,
    *,
    beta: float = 1.0,
    p_truth: float = 1.0,
    max_rounds: int = 3,
) -> LoopResult:
    """Run the alien charge-scaling loop with the LIVE Council as the proposer (ACTEXP-02).

    Args:
      decide_move: the Council director's decision function — ``(verdict) -> NextMove`` (anything with
        a ``.move`` string). In production this wraps the live ``council.director`` call (one LLM call
        on the plateau); in tests it is a $0 stub. The loop/ranking/diagnosis stay no-LLM.
      beta, p_truth: the true hidden-charge amplitude scaling (p=1 is the simulator ground truth).
      max_rounds: design-experiment round budget.

    Returns:
      LoopResult. If the Council chooses to vary the family, the loop reaches outcome 'solved'
      (the OED independently confirms a multi-distinct-product design restores rank); if it chooses a
      new initial condition, the loop honestly fails to solve (the degeneracy persists).
    """

    def propose(verdict: object) -> Mapping[str, PredictFn]:
        return council_proposer(decide_move(verdict))

    return run_alien_charge_scaling_loop(
        beta=beta, p_truth=p_truth, max_rounds=max_rounds, propose=propose
    )


__all__ = ["council_proposer", "run_alien_council_loop", "DecideMoveFn"]
