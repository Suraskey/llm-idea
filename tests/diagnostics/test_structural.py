"""PIVOT-001 Burst B — exact symbolic structural identifiability (IDENT-01..04).

The clean-form complement to the numerical practical-ID checker: where the model is
an explicit expression, decide identifiability EXACTLY. The alien force law is the
headline case — the hidden charges are confounded but the radial-power coefficients
are not, exactly as L-051 found.
"""

from __future__ import annotations

from pathlib import Path

import sympy

from ascension.diagnostics.structural import (
    StructuralVerdict,
    structural_identifiability,
)

r, t, v = sympy.symbols("r t v")
s1, s2, alpha, beta, gamma = sympy.symbols("s1 s2 alpha beta gamma")


def _groups_as_sets(v: StructuralVerdict) -> list[set[str]]:
    return [set(g) for g in v.confounded_groups]


def test_product_is_structurally_non_identifiable() -> None:
    # f = (s1*s2)/r**2 — the alien hidden charge: only the PRODUCT is recoverable.
    verdict = structural_identifiability((s1 * s2) / r**2, [s1, s2], [r])
    assert verdict.identifiable is False
    assert {"s1", "s2"} in _groups_as_sets(verdict)


def test_sum_is_structurally_non_identifiable() -> None:
    verdict = structural_identifiability((alpha + beta) * r, [alpha, beta], [r])
    assert verdict.identifiable is False
    assert {"alpha", "beta"} in _groups_as_sets(verdict)


def test_v2_hidden_charge_sum_is_structurally_non_identifiable() -> None:
    """EXP-080 P-EXP-080-2 — the V2 sum-confound is auto-detected, NO agent needed.

    The V2 hidden-charge law enters the charges ADDITIVELY: f = (s1 + s2)·g(r)
    with g(r) = β(1 − cos γr)/r² (the V2 hidden radial envelope). The structural
    diagnostic ALREADY decides this case (structural.py line 18: a sum gives a
    derivative ratio of 1, free of observables → confounded), so pointing it at
    the V2 model form is a FREE deterministic generalization win — the diagnostic
    side transfers the moment we aim it at V2, before any agent runs.

    Asserts the exact P-EXP-080-2 contract: identifiable=False, the {s1,s2}
    confounded group (canonical tuple form), AND the break-the-proportionality
    disambiguation hint — exactly as the V1 product case returns.
    """
    g = beta * (1 - sympy.cos(gamma * r)) / r**2
    verdict = structural_identifiability((s1 + s2) * g, [s1, s2], [r])
    assert verdict.identifiable is False
    # The canonical confounded-group tuple form (sorted names, sorted groups).
    assert verdict.confounded_groups == (("s1", "s2"),)
    assert {"s1", "s2"} in _groups_as_sets(verdict)
    assert verdict.absent_params == ()
    # The disambiguation hint points at breaking the proportionality (the
    # experiment the engine should request) — the same hint the product case gives.
    assert "break" in verdict.disambiguation_hint.lower()
    assert "proportionality" in verdict.disambiguation_hint.lower()


def test_v2_sum_vs_v1_product_confound_verdicts_match_in_status() -> None:
    """Both the V1 product (s1*s2) and the V2 sum (s1+s2) confound the hidden
    charges identically: identifiable=False with the {s1,s2} group. This is the
    side-by-side V1/V2 verdict the $0 tabulation script reports for the paper."""
    g = beta * (1 - sympy.cos(gamma * r)) / r**2
    v1 = structural_identifiability((s1 * s2) * g, [s1, s2], [r])
    v2 = structural_identifiability((s1 + s2) * g, [s1, s2], [r])
    assert v1.identifiable is v2.identifiable is False
    assert v1.confounded_groups == v2.confounded_groups == (("s1", "s2"),)


def test_distinct_radial_powers_are_identifiable() -> None:
    # f = alpha/r**2 + beta/r**3.5 — different observable dependence => both recoverable.
    expr = alpha / r**2 + beta / r ** sympy.Rational(7, 2)
    verdict = structural_identifiability(expr, [alpha, beta], [r])
    assert verdict.identifiable is True
    assert verdict.confounded_groups == ()


def test_alien_force_law_charges_confounded_coeffs_identifiable() -> None:
    """The headline: f = (s1*s2)*(alpha/r**2 + beta/r**3.5). The hidden charges s1,s2
    are confounded (only the product acts), but alpha and beta enter with DISTINCT
    radial powers so they are separately identifiable — exactly the L-051 picture."""
    expr = (s1 * s2) * (alpha / r**2 + beta / r ** sympy.Rational(7, 2))
    verdict = structural_identifiability(expr, [s1, s2, alpha, beta], [r])
    assert verdict.identifiable is False
    groups = _groups_as_sets(verdict)
    assert {"s1", "s2"} in groups
    # alpha, beta must NOT be confounded with each other (distinct powers of r).
    assert not any({"alpha", "beta"} <= g for g in groups)


def test_absent_parameter_is_flagged() -> None:
    # gamma does not appear => it has no effect => unidentifiable (absent).
    verdict = structural_identifiability(alpha * r, [alpha, gamma], [r])
    assert verdict.identifiable is False
    assert "gamma" in verdict.absent_params
    assert "alpha" not in verdict.absent_params


def test_fully_identifiable_form() -> None:
    verdict = structural_identifiability(alpha * r + beta * t, [alpha, beta], [r, t])
    assert verdict.identifiable is True
    assert verdict.confounded_groups == ()
    assert verdict.absent_params == ()


def test_disambiguation_hint_points_at_breaking_proportionality() -> None:
    verdict = structural_identifiability((s1 * s2) / r**2, [s1, s2], [r])
    assert "distinct products" in verdict.disambiguation_hint.lower()
    assert isinstance(verdict.describe(), str)


def test_accepts_string_expression() -> None:
    verdict = structural_identifiability("s1*s2/r**2", ["s1", "s2"], ["r"])
    assert verdict.identifiable is False
    assert {"s1", "s2"} in _groups_as_sets(verdict)


def test_structural_module_is_llm_free() -> None:
    import ascension.diagnostics.structural as mod

    src_path = Path(mod.__file__)
    forbidden = ("google", "genai", "ascension.llm", "llm_client", "generate_structured")
    for line in src_path.read_text().splitlines():
        stripped = line.strip().lower()
        if stripped.startswith(("import ", "from ")):
            for tok in forbidden:
                assert tok not in stripped, f"structural.py imports {tok!r} — must be LLM-free"
