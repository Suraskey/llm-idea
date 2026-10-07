"""AUDIT A2 — adversarial stress tests for the diagnostics instrument.

Probes the numerical/symbolic/loop edge cases the happy-path tests don't, including
robustness to broken inputs (engineering_discipline_no_coverups: never halt the whole
analysis on one transient/bad component). Tests assert the ROBUST behavior; where the
production code did not meet it, the code was fixed alongside.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from ascension.diagnostics import (
    practical_identifiability,
    rank_designs_by_identifiability,
    run_identifiability_loop,
    sensitivity_matrix,
    structural_identifiability,
)

_XS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


# ---------------------------------------------------------------------------
# practical_identifiability — numerical edges
# ---------------------------------------------------------------------------


def test_nonfinite_prediction_fails_loud() -> None:
    def predict(theta):
        return np.array([np.nan, 1.0])

    with pytest.raises(ValueError):
        sensitivity_matrix(predict, np.array([1.0]))


def test_constant_model_zero_jacobian_is_non_identifiable_not_crash() -> None:
    # predict ignores params entirely → zero sensitivity → rank 0, all confounded.
    def predict(theta):
        return np.array([1.0, 2.0, 3.0])

    v = practical_identifiability(predict, [1.0, 2.0], param_names=["a", "b"])
    assert v.rank == 0 and v.identifiable is False
    assert np.isinf(v.condition_number)
    assert len(v.confounded_directions) == 2  # both directions collapsed


def test_single_parameter_identifiable() -> None:
    def predict(theta):
        return theta[0] * _XS

    v = practical_identifiability(predict, [3.0], param_names=["a"])
    assert v.identifiable is True and v.rank == 1


def test_more_params_than_observations_is_non_identifiable() -> None:
    # 1 observation, 3 params → rank ≤ 1 → cannot identify 3.
    def predict(theta):
        return np.array([theta[0] + theta[1] + theta[2]])

    v = practical_identifiability(predict, [1.0, 1.0, 1.0])
    assert v.identifiable is False and v.rank <= 1


def test_parameter_at_zero_uses_step_floor_and_still_sees_sensitivity() -> None:
    # theta[1] sits at 0; rel_step*|0| = 0 → the abs_step_floor must still probe it.
    def predict(theta):
        return theta[0] * _XS + theta[1] * _XS**2

    v = practical_identifiability(predict, [1.0, 0.0], param_names=["a", "b"])
    assert v.identifiable is True, "param at 0 must still be probed via the step floor"


def test_extreme_scale_difference_is_flagged_practically_non_identifiable() -> None:
    # b's effect is ~1e-11 of a's → genuinely there but swamped → sloppy, NOT a crash,
    # NOT a false 'clean identifiable'.
    def predict(theta):
        return theta[0] * _XS + theta[1] * 1e-11 * _XS

    v = practical_identifiability(predict, [1.0, 1.0], param_names=["a", "b"])
    assert v.identifiable is False  # either rank-collapsed or sloppy; must not be "clean"
    assert v.status in ("structurally_non_identifiable", "practically_non_identifiable")


def test_verdict_is_robust_to_finite_difference_step_choice() -> None:
    # A smooth identifiable model should give the same identifiable verdict across a
    # range of FD steps (no step-sensitivity flip on a well-behaved problem).
    def predict(theta):
        return theta[0] * _XS + theta[1] * _XS**2

    for step in (1e-4, 1e-6, 1e-8):
        v = practical_identifiability(predict, [1.0, 1.0], rel_step=step)
        assert v.identifiable is True, f"verdict flipped at rel_step={step}"


# ---------------------------------------------------------------------------
# structural_identifiability — symbolic edges
# ---------------------------------------------------------------------------


def test_structural_transcendental_is_identifiable() -> None:
    a, b, x = sympy.symbols("a b x")
    v = structural_identifiability(sympy.sin(a * x) + b * x, [a, b], [x])
    assert v.identifiable is True  # cos(a x) ratio carries x → distinct dependence


def test_structural_params_in_denominator() -> None:
    a, b, x = sympy.symbols("a b x")
    v = structural_identifiability(a / (b + x), [a, b], [x])
    assert v.identifiable is True


def test_structural_three_way_product_all_confounded() -> None:
    a, b, c, x = sympy.symbols("a b c x")
    v = structural_identifiability(a * b * c * x, [a, b, c], [x])
    assert v.identifiable is False
    assert {"a", "b", "c"} in [set(g) for g in v.confounded_groups]


def test_structural_param_in_two_terms_is_identifiable() -> None:
    a, b, x = sympy.symbols("a b x")
    # a enters as x + x**2, b as x → ratio (x+x**2)/x = 1+x carries x → identifiable.
    v = structural_identifiability(a * x + a * x**2 + b * x, [a, b], [x])
    assert v.identifiable is True


def test_structural_does_not_falsely_confound_distinct_dependence() -> None:
    a, b, x = sympy.symbols("a b x")
    v = structural_identifiability(a * sympy.exp(x) + b * sympy.log(x + 2), [a, b], [x])
    assert v.identifiable is True


# ---------------------------------------------------------------------------
# optimal_design + engine — robustness to a BROKEN candidate (no-coverups)
# ---------------------------------------------------------------------------


def _good(theta):
    return theta[0] * _XS + theta[1] * _XS**2  # identifiable


def _confounded(theta):
    return (theta[0] + theta[1]) * _XS  # rank 1


def _broken(theta):
    raise RuntimeError("synthetic broken design (e.g. a malformed Council proposal)")


def test_ranker_skips_a_broken_design_instead_of_crashing() -> None:
    # A broken candidate must NOT crash the whole ranking — it is scored worst and the
    # good design still wins (best-effort, never halt on one bad component).
    ranking = rank_designs_by_identifiability(
        {"broken": _broken, "good": _good, "confounded": _confounded},
        [1.0, 1.0],
        param_names=["a", "b"],
    )
    assert ranking.best == "good"
    labels = {s.label for s in ranking.ranked}
    assert {"broken", "good", "confounded"} == labels
    broken = next(s for s in ranking.ranked if s.label == "broken")
    assert broken.identifiable is False  # broken ranked as worst, not crashed


def test_loop_survives_a_broken_proposed_design() -> None:
    # The proposer offers a broken design AND a good one; the loop must route around the
    # broken one and still solve, not crash.
    def propose(verdict):
        return {"broken": _broken, "enrich": _good}

    result = run_identifiability_loop(
        _confounded, [1.0, 1.0], propose_designs=propose, param_names=["a", "b"]
    )
    assert result.final_identifiable is True and result.outcome == "solved"


def test_loop_handles_all_designs_broken_without_crashing() -> None:
    def propose(verdict):
        return {"broken1": _broken, "broken2": _broken}

    result = run_identifiability_loop(
        _confounded,
        [1.0, 1.0],
        propose_designs=propose,
        max_rounds=2,
        param_names=["a", "b"],
    )
    # No good design ever → honest non-solved terminal, no crash.
    assert result.final_identifiable is False
    assert result.outcome in ("exhausted", "no_proposal")
