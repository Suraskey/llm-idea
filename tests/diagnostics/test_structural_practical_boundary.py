"""Burst-B boundary regression — pins the structural↔practical complementarity.

Originally (EXP-082) the structural checker was pairwise-only and this file pinned
its blind spot on a 3-way confound, with the practical (FIM) check as the necessary
complement. As of 2026-09-08 (EXP-261) the structural checker also runs an exact
multi-parameter dependence test, so both instruments now agree on this case; the
practical check remains the complement for black-box models and for PRACTICAL
(sloppy) degeneracy, which no symbolic test sees.

$0, no LLM, no DB.
"""

from __future__ import annotations

import numpy as np

from ascension.diagnostics.identifiability import practical_identifiability
from ascension.diagnostics.structural import structural_identifiability

# f = a·r + b·r² + c·(r + r²): ∂f/∂c = ∂f/∂a + ∂f/∂b exactly → rank 2/3.
# No two sensitivity functions are proportional (ratios all carry r), so the
# pairwise structural method cannot see the degeneracy.
_EXPR = "a*r + b*r**2 + c*(r + r**2)"
_R = np.linspace(0.7, 2.5, 60)


def _predict(theta):
    a, b, c = theta
    return np.array([a * r + b * r**2 + c * (r + r**2) for r in _R], dtype=float)


def test_structural_now_catches_higher_order_confound() -> None:
    """Until 2026-09-08 this test pinned a documented blind spot: the pairwise
    ratio test returned identifiable here. EXP-261 (the ODEBench sweep) hit the
    same blind spot on the RC circuit and the checker gained an exact
    multi-parameter dependence test. The 3-way confound is now found exactly."""
    v = structural_identifiability(_EXPR, ["a", "b", "c"], ["r"])
    assert v.identifiable is False
    assert v.confounded_groups == ()  # still no pairwise proportionality
    assert v.rank_deficiency == 1
    assert v.dependence_groups == (("a", "b", "c"),)
    d = np.asarray(v.null_directions[0])
    d = d / np.linalg.norm(d)
    assert abs(float(d @ (np.array([1.0, 1.0, -1.0]) / np.sqrt(3)))) > 0.999


def test_practical_catches_the_same_confound_with_the_exact_null_direction() -> None:
    """The complement: the FIM sees the whole Jacobian → rank 2/3, and the null
    direction is the planted (1,1,-1) (∂c = ∂a + ∂b)."""
    v = practical_identifiability(_predict, [1.0, 1.0, 1.0], param_names=["a", "b", "c"])
    assert v.identifiable is False
    assert v.rank == 2
    assert v.n_params == 3
    d = np.asarray(v.confounded_directions[-1], dtype=float)
    d = d / np.linalg.norm(d)
    planted = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
    # direction is sign-ambiguous; compare up to sign via |cosine| ≈ 1
    cos = abs(float(d @ planted))
    assert cos > 0.99, f"null direction {d} is not ±(1,1,-1)/sqrt3 (|cos|={cos:.3f})"


def test_structural_stays_exact_on_pairwise_confounds() -> None:
    """The reach side: the structural method IS exact on the confounds it covers —
    product (alien hidden charge) and sum (V2 coupling)."""
    prod = structural_identifiability("s1*s2/r**2", ["s1", "s2"], ["r"])
    assert prod.identifiable is False and prod.confounded_groups == (("s1", "s2"),)
    summ = structural_identifiability("(a + b)/r**2", ["a", "b"], ["r"])
    assert summ.identifiable is False and summ.confounded_groups == (("a", "b"),)
    distinct = structural_identifiability("a/r**2 + b/r**3.5", ["a", "b"], ["r"])
    assert distinct.identifiable is True


def test_structural_string_parse_cannot_execute_code() -> None:
    """DET-06 — expression strings route through no-eval parsing.

    eval-based sympify would EXECUTE `__import__('os')...` during parse; the
    empty-__builtins__ parse_expr path must raise instead of running code, and
    plain explicit-syntax strings must still parse exactly (s1 stays s1).
    """
    import pytest as _pytest

    with _pytest.raises(Exception):
        structural_identifiability(
            "__import__('os').system('true')", ["s1"], ["r"]
        )
    # Well-formed strings keep exact multi-char symbol names.
    v = structural_identifiability("s1*s2/r**2", ["s1", "s2"], ["r"])
    assert v.confounded_groups == (("s1", "s2"),)
