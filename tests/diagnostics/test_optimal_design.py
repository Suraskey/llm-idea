"""PIVOT-001 Burst B — identifiability-driven experiment design (IDENT-03).

The second novelty pillar: given candidate observation designs, pick the one that
makes the confounded structure observable. The principle behind the alien
multi-charge-family fix, made explicit + domain-agnostic.
"""

from __future__ import annotations

import numpy as np

from ascension.diagnostics import rank_designs_by_identifiability

_XS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


def test_ranker_recommends_the_design_that_breaks_the_confound() -> None:
    # Design A observes only (a+b)*x → a,b confounded (rank 1, only the sum acts).
    def design_sum_only(theta: np.ndarray) -> np.ndarray:
        return (theta[0] + theta[1]) * _XS

    # Design B observes a*x + b*x**2 → a,b enter with distinct x-dependence → rank 2.
    def design_distinct(theta: np.ndarray) -> np.ndarray:
        return theta[0] * _XS + theta[1] * _XS**2

    ranking = rank_designs_by_identifiability(
        {"sum_only": design_sum_only, "distinct_powers": design_distinct},
        [1.0, 1.0],
        param_names=["a", "b"],
    )
    assert ranking.best == "distinct_powers"
    by = {s.label: s for s in ranking.ranked}
    assert by["sum_only"].identifiable is False and by["sum_only"].rank == 1
    assert by["distinct_powers"].identifiable is True and by["distinct_powers"].rank == 2
    # The identifying design must outscore the confounded one.
    assert by["distinct_powers"].score > by["sum_only"].score
    assert "distinct_powers" in ranking.rationale


def test_ranker_reports_when_no_design_identifies() -> None:
    # Both designs leave a,b confounded (a hidden product, like the alien charges):
    # no candidate breaks it → the ranker says so honestly.
    def design_product_a(theta: np.ndarray) -> np.ndarray:
        return (theta[0] * theta[1]) * _XS

    def design_product_b(theta: np.ndarray) -> np.ndarray:
        return (theta[0] * theta[1]) * _XS**2  # different basis, still product-only

    ranking = rank_designs_by_identifiability(
        {"prod_a": design_product_a, "prod_b": design_product_b},
        [1.0, 1.0],
        param_names=["s1", "s2"],
    )
    assert all(not s.identifiable for s in ranking.ranked)
    assert "NO candidate design fully identifies" in ranking.rationale


def test_ranker_prefers_better_conditioning_at_equal_rank() -> None:
    # Two full-rank designs; one is near-degenerate (sloppy), the other clean.
    # Equal rank (2) → the better-conditioned design wins on the tiebreaker.
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    omega = 1e-5 * np.array([1.0, -1.0, 1.0, -1.0, 1.0])

    def clean(theta: np.ndarray) -> np.ndarray:
        return theta[0] * a + theta[1] * np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    def sloppy(theta: np.ndarray) -> np.ndarray:
        return theta[0] * a + theta[1] * (a + omega)

    ranking = rank_designs_by_identifiability({"clean": clean, "sloppy": sloppy}, [1.0, 1.0])
    by = {s.label: s for s in ranking.ranked}
    assert by["clean"].rank == 2 and by["sloppy"].rank == 2
    assert ranking.best == "clean"
    assert by["clean"].condition_number < by["sloppy"].condition_number


def test_empty_designs_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        rank_designs_by_identifiability({}, [1.0])
