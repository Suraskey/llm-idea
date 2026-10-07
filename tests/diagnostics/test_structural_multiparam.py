"""EXP-261 regressions: the exact multi-parameter dependence test in the structural
checker, and precomputed (exact) sensitivities in the practical verdict + ranker.
$0, no LLM, no DB."""

from __future__ import annotations

import numpy as np

from ascension.diagnostics.identifiability import practical_identifiability
from ascension.diagnostics.optimal_design import rank_designs_by_identifiability
from ascension.diagnostics.structural import structural_identifiability


def test_rc_circuit_three_way_redundancy_is_found() -> None:
    """ODEBench id 1: (c_0 - x_0/c_1)/c_2 has three constants but only two
    combinations (c_0/c_2 and 1/(c_1 c_2)) enter the vector field."""
    v = structural_identifiability("(c_0 - x_0/c_1)/c_2", ["c_0", "c_1", "c_2"], ["x_0"])
    assert v.identifiable is False
    assert v.confounded_groups == ()  # no PAIR is proportional; the pairwise test is silent
    assert v.rank_deficiency == 1
    assert v.dependence_groups == (("c_0", "c_1", "c_2"),)
    assert len(v.null_directions) == 1 and len(v.null_directions[0]) == 3


def test_multiparam_test_does_not_flag_identifiable_forms() -> None:
    v = structural_identifiability("c_0*x_0*(1 - x_0/c_1)", ["c_0", "c_1"], ["x_0"])
    assert v.identifiable is True
    assert v.rank_deficiency == 0


def test_alien_form_keeps_pairwise_groups_and_reports_the_rank_deficiency() -> None:
    """(s1*s2)*(a/r^2 + b/r^3.5): pairwise finds {s1, s2}; the exact test adds that
    only two combinations (s1 s2 a, s1 s2 b) are recoverable from four parameters."""
    v = structural_identifiability("(s1*s2)*(a/r**2 + b/r**3.5)", ["s1", "s2", "a", "b"], ["r"])
    assert v.identifiable is False
    assert v.confounded_groups == (("s1", "s2"),)
    assert v.rank_deficiency == 2
    lorenz = structural_identifiability(
        "w_0*(c_0*(x_1 - x_0)) + w_1*(x_0*(c_2 - x_2) - x_1) + w_2*(x_0*x_1 - c_1*x_2)",
        ["c_0", "c_1", "c_2"], ["x_0", "x_1", "x_2", "w_0", "w_1", "w_2"],
    )
    assert lorenz.identifiable is True


def test_vector_field_weighted_sum_trick_keeps_components_separate() -> None:
    """Two components combined as w_0*f_0 + w_1*f_1 with w free: a parameter that is
    redundant in ONE component but not the other must stay identifiable."""
    expr = "w_0*((a + b)*x_0) + w_1*(a*x_1)"
    v = structural_identifiability(expr, ["a", "b"], ["x_0", "x_1", "w_0", "w_1"])
    assert v.identifiable is True


def test_precomputed_sensitivity_matches_finite_difference_verdict() -> None:
    theta = np.array([1.0, 2.0])
    r = np.linspace(1.0, 3.0, 25)

    def predict(th):
        return th[0] / r**2 + th[1] / r**3.5

    s_exact = np.column_stack([1 / r**2, 1 / r**3.5])
    v_fd = practical_identifiability(predict, theta)
    v_pre = practical_identifiability(predict, theta, sensitivity=s_exact)
    assert v_fd.identifiable and v_pre.identifiable
    assert v_fd.rank == v_pre.rank == 2
    assert abs(np.log10(v_fd.condition_number) - np.log10(v_pre.condition_number)) < 1e-3


def test_precomputed_sensitivity_exposes_null_direction_finite_difference_blurs() -> None:
    """A true null direction with a finite-difference floor of ~1e-8 lands in the
    'practical' band; the exact matrix returns the structural verdict."""
    r = np.linspace(1.0, 3.0, 40)
    s_true = np.column_stack([1 / r**2, 1 / r**3.5, 1 / r**2 + 1 / r**3.5])  # col3 = col1 + col2
    v = practical_identifiability(lambda th: np.zeros(0), [1.0, 1.0, 1.0], sensitivity=s_true)
    assert v.status == "structurally_non_identifiable"
    assert v.rank == 2
    d = np.asarray(v.confounded_directions[-1])
    d = d / np.linalg.norm(d)
    assert abs(float(d @ (np.array([1.0, 1.0, -1.0]) / np.sqrt(3)))) > 0.999


def test_ranker_accepts_precomputed_sensitivity_matrices() -> None:
    r = np.linspace(1.0, 3.0, 30)
    good = np.column_stack([1 / r**2, 1 / r**3.5])
    bad = np.column_stack([1 / r**2, 2 / r**2])  # collinear
    ranking = rank_designs_by_identifiability({"good": good, "bad": bad}, [1.0, 1.0])
    assert ranking.best == "good"
    by = {s.label: s for s in ranking.ranked}
    assert by["good"].rank == 2 and by["bad"].rank == 1
