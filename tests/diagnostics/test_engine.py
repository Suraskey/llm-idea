"""PIVOT-001 Burst C unit 1 — the identifiability-driven experiment loop (ACTEXP).

The pivot thesis as control flow: on a plateau, distinguish "not solved yet"
(identifiable — keep reasoning) from "unsolvable from this data" (non-identifiable —
design the disambiguating experiment), and act. Deterministic core; the proposer
(Council) + servicer (simulator+fitter) are injected stubs here.
"""

from __future__ import annotations

import numpy as np

from ascension.diagnostics import run_identifiability_loop

_XS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


def _sum_only(theta: np.ndarray) -> np.ndarray:
    # a, b confounded — only the sum acts (rank 1).
    return (theta[0] + theta[1]) * _XS


def _distinct_powers(theta: np.ndarray) -> np.ndarray:
    # a, b enter with distinct x-dependence — identifiable (rank 2).
    return theta[0] * _XS + theta[1] * _XS**2


def _another_confound(theta: np.ndarray) -> np.ndarray:
    # still product-confounded under a different basis (never identifies a,b).
    return (theta[0] * theta[1]) * _XS**2


def test_capability_plateau_stops_without_designing_an_experiment() -> None:
    """Identifiable from round 0 => the data already determines the structure =>
    'not solved yet' (capability limit). The loop must NOT request an experiment."""
    proposed = {"n": 0}

    def propose(verdict):
        proposed["n"] += 1
        return {"x": _sum_only}

    result = run_identifiability_loop(
        _distinct_powers, [1.0, 1.0], propose_designs=propose, param_names=["a", "b"]
    )
    assert result.final_identifiable is True
    assert result.outcome == "capability_plateau"
    assert result.rounds_used == 0
    assert proposed["n"] == 0, "must not ask for an experiment when already identifiable"


def test_observability_plateau_designs_the_experiment_and_solves() -> None:
    """Non-identifiable => ask the proposer for candidates, the ranker picks the
    identifying one, adopt it, re-diagnose => solved. The full thesis loop."""

    def propose(verdict):
        # The Council offers a confounded option AND the disambiguating one; the
        # identifiability-driven ranker must pick the latter.
        return {"keep_confounded": _sum_only, "enrich": _distinct_powers}

    result = run_identifiability_loop(
        _sum_only,
        [1.0, 1.0],
        propose_designs=propose,
        max_rounds=3,
        param_names=["a", "b"],
    )
    assert result.final_identifiable is True
    assert result.outcome == "solved"
    assert result.rounds_used == 1
    # Round 0: non-identifiable, proposed both, chose the enriching design.
    s0 = result.steps[0]
    assert s0.identifiable is False and s0.chosen == "enrich"
    assert set(s0.proposed) == {"keep_confounded", "enrich"}


def test_service_and_refit_hook_is_used() -> None:
    """When a servicer is supplied (gather data + fit the proposer's own structure),
    the loop uses its returned predict as the new effective model."""
    serviced = {"calls": []}

    def propose(verdict):
        return {"enrich": _distinct_powers}

    def service(label, predict):
        serviced["calls"].append(label)
        return predict  # stub: data gathered, structure fit → the enriched model applies

    result = run_identifiability_loop(
        _sum_only,
        [1.0, 1.0],
        propose_designs=propose,
        service_and_refit=service,
        param_names=["a", "b"],
    )
    assert result.final_identifiable is True and result.outcome == "solved"
    assert serviced["calls"] == ["enrich"]


def test_no_proposal_stops_honestly() -> None:
    """Non-identifiable and the proposer gives up => honest 'no_proposal' stop."""
    result = run_identifiability_loop(
        _sum_only, [1.0, 1.0], propose_designs=lambda v: {}, param_names=["a", "b"]
    )
    assert result.final_identifiable is False
    assert result.outcome == "no_proposal"


def test_exhausted_when_no_design_ever_identifies() -> None:
    """Proposer keeps offering confounded designs => loop exhausts the round budget
    and reports honestly (no false 'solved')."""
    result = run_identifiability_loop(
        _sum_only,
        [1.0, 1.0],
        propose_designs=lambda v: {"still_bad": _another_confound},
        max_rounds=2,
        param_names=["a", "b"],
    )
    assert result.final_identifiable is False
    assert result.outcome == "exhausted"
    assert result.rounds_used == 2


def test_engine_module_is_llm_free() -> None:
    from pathlib import Path

    import ascension.diagnostics.engine as mod

    forbidden = ("google", "genai", "ascension.llm", "llm_client", "generate_structured")
    for line in Path(mod.__file__).read_text().splitlines():
        s = line.strip().lower()
        if s.startswith(("import ", "from ")):
            for tok in forbidden:
                assert tok not in s, f"engine.py imports {tok!r} — core loop must be LLM-free"
