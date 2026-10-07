"""$0 BURST-B CHARACTERIZATION — where the cheap structural diagnostic's reach ends,
and the practical (FIM) diagnostic takes over. Deterministic, no LLM, no DB.

PIVOT-001 Burst B names this boundary as itself a publishable finding: "how far
structural-identifiability analysis generalizes before the math gets hard." The
paper (§10.1) promised it as future work and asserted — without demonstrating —
that practical identifiability is the catch-all fallback past the structural
method's scope. This battery turns the assertion into a measured result.

`structural_identifiability` decides identifiability by PAIRWISE derivative
proportionality (params p_i, p_j confounded iff ∂f/∂p_i ÷ ∂f/∂p_j is free of
observables). That is exact for the common confounds (product, sum, absent
parameter) and, by construction, BLIND to a higher-order confound: k parameters
whose sensitivity functions are linearly dependent over observable-space but with
NO two pairwise proportional. The practical FIM/sensitivity-rank diagnostic, which
sees the whole Jacobian at once, catches exactly that case.

The battery is a graded ladder of model classes with KNOWN identifiability answers
(by construction). For each we run BOTH unmodified diagnostics and tabulate:
structural verdict, practical verdict, ground truth, and agreement. The load-
bearing row is the higher-order confound where structural says "identifiable"
(a FALSE NEGATIVE for non-identifiability — it misses the degeneracy) and practical
recovers the exact null direction. That single disagreement IS the boundary, made
concrete: it shows precisely what the $0 algebraic check can and cannot certify,
and that the numerical path is not a redundant backup but the necessary complement.

Run:  poetry run python scripts/structural_boundary_battery.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ascension.diagnostics.identifiability import practical_identifiability
from ascension.diagnostics.structural import structural_identifiability


@dataclass(frozen=True)
class Case:
    name: str
    klass: str  # model class (the ladder rung)
    expr: str  # sympy string in params + observables
    params: tuple[str, ...]
    observables: tuple[str, ...]
    # predict(theta) -> observation vector, for the practical diagnostic.
    predict: object
    theta: tuple[float, ...]
    truth_identifiable: bool
    truth_note: str


_R = np.linspace(0.7, 2.5, 60)  # an observation grid that is generic (no accidental rank loss)


def _vec(fn):
    return lambda th: np.array([fn(th, float(r)) for r in _R], dtype=float)


def _cases() -> list[Case]:
    cases: list[Case] = []

    # --- Rung 1: pairwise confounds the structural method is EXACT for ---------
    cases.append(Case(
        "identifiable_distinct_powers", "linear-in-params (distinct basis)",
        "a/r**2 + b/r**3.5", ("a", "b"), ("r",),
        _vec(lambda th, r: th[0] / r**2 + th[1] / r**3.5), (1.0, 1.0),
        True, "distinct radial dependence → both recoverable (the EXP-076 L2 case)",
    ))
    cases.append(Case(
        "product_confound", "bilinear (product)",
        "s1*s2*(1/r**2)", ("s1", "s2"), ("r",),
        _vec(lambda th, r: th[0] * th[1] / r**2), (1.0, 1.0),
        False, "only the product s1*s2 is recoverable (alien hidden-charge L-051)",
    ))
    cases.append(Case(
        "sum_confound", "affine (sum)",
        "(a + b)*(1/r**2)", ("a", "b"), ("r",),
        _vec(lambda th, r: (th[0] + th[1]) / r**2), (1.0, 1.0),
        False, "only the sum a+b is recoverable (the EXP-080 V2 coupling)",
    ))
    cases.append(Case(
        "absent_param", "degenerate (no effect)",
        "a/r**2 + 0*c", ("a", "c"), ("r",),
        _vec(lambda th, r: th[0] / r**2), (1.0, 1.0),
        False, "c has zero effect on the prediction → unidentifiable",
    ))

    # --- Rung 2: THE BOUNDARY — higher-order confound, NO pairwise proportionality
    # f = a·r + b·r² + c·(r + r²). Pairwise ratios r/r²=1/r, r/(r+r²)=1/(1+r),
    # r²/(r+r²)=r/(1+r) all carry the observable r → structural sees NO pairwise
    # confound and returns identifiable. BUT ∂f/∂c = ∂f/∂a + ∂f/∂b exactly, so the
    # sensitivity matrix is rank 2: only (a+c) and (b+c) are recoverable. A genuine
    # 3-way degeneracy with no pairwise proportionality — the structural method's
    # documented blind spot; the FIM rank sees it at once.
    cases.append(Case(
        "higher_order_no_pairwise", "rank-deficient, no pairwise prop. (THE BOUNDARY)",
        "a*r + b*r**2 + c*(r + r**2)", ("a", "b", "c"), ("r",),
        _vec(lambda th, r: th[0] * r + th[1] * r**2 + th[2] * (r + r**2)), (1.0, 1.0, 1.0),
        False, "∂c = ∂a+∂b → rank 2/3; only (a+c),(b+c) recoverable; NO pair is proportional",
    ))
    # A fully-identifiable sibling of the same class (distinct independent bases),
    # to show rung 2 is not the method failing on all 3-param linear models.
    cases.append(Case(
        "three_independent_powers", "rank-full, 3 distinct bases",
        "a*r + b*r**2 + c*r**3", ("a", "b", "c"), ("r",),
        _vec(lambda th, r: th[0] * r + th[1] * r**2 + th[2] * r**3), (1.0, 1.0, 1.0),
        True, "three independent observable bases → full rank, all recoverable",
    ))

    # --- Rung 3: nonlinear / transcendental — structural algebra still applies ---
    cases.append(Case(
        "nonlinear_freq_amp_identifiable", "transcendental (distinct roles)",
        "A*sin(w*r)", ("A", "w"), ("r",),
        _vec(lambda th, r: th[0] * np.sin(th[1] * r)), (1.0, 0.7),
        True, "amplitude and frequency act differently → both recoverable",
    ))
    cases.append(Case(
        "nonlinear_scale_confound", "transcendental (scale product)",
        "A*k*exp(-r)", ("A", "k"), ("r",),
        _vec(lambda th, r: th[0] * th[1] * np.exp(-r)), (1.0, 1.0),
        False, "A and k enter only as the product A*k → confounded",
    ))

    return cases


def _structural_says_identifiable(c: Case) -> tuple[bool, str]:
    v = structural_identifiability(c.expr, list(c.params), list(c.observables))
    detail = "identifiable" if v.identifiable else (
        "NON-id: " + "; ".join("{" + ",".join(g) + "}" for g in v.confounded_groups)
        + (f" absent={v.absent_params}" if v.absent_params else "")
    )
    return v.identifiable, detail


def _practical_says_identifiable(c: Case) -> tuple[bool, str]:
    v = practical_identifiability(c.predict, list(c.theta), param_names=list(c.params))
    detail = f"{v.status} rank={v.rank}/{v.n_params} cond={v.condition_number:.1e}"
    if v.confounded_directions:
        d = np.asarray(v.confounded_directions[-1])
        detail += " null≈(" + ",".join(f"{x:+.2f}" for x in d) + ")"
    return v.identifiable, detail


def main() -> None:
    cases = _cases()
    print("=" * 108)
    print("STRUCTURAL-vs-PRACTICAL IDENTIFIABILITY — the Burst-B boundary battery ($0, no LLM)")
    print("  structural = exact pairwise-proportionality (symbolic) | practical = FIM/sensitivity rank (numerical)")
    print("-" * 108)
    hdr = f"{'case':<34}{'truth':<7}{'struct':<7}{'pract':<7}{'agree':<7}detail"
    print(hdr)
    print("-" * 108)
    boundary_rows = []
    n_struct_correct = n_pract_correct = 0
    for c in cases:
        s_id, s_detail = _structural_says_identifiable(c)
        p_id, p_detail = _practical_says_identifiable(c)
        s_ok = (s_id == c.truth_identifiable)
        p_ok = (p_id == c.truth_identifiable)
        n_struct_correct += s_ok
        n_pract_correct += p_ok
        agree = "yes" if s_id == p_id else "**NO**"
        if s_id != p_id:
            boundary_rows.append((c, s_id, p_id, s_detail, p_detail))
        truth = "ID" if c.truth_identifiable else "non-ID"
        s_str = ("ID" if s_id else "non-ID") + ("" if s_ok else "✗")
        p_str = ("ID" if p_id else "non-ID") + ("" if p_ok else "✗")
        print(f"{c.name:<34}{truth:<7}{s_str:<7}{p_str:<7}{agree:<7}{c.klass}")
        print(f"{'':<34}{'':<21}struct: {s_detail}")
        print(f"{'':<34}{'':<21}pract:  {p_detail}")
        print(f"{'':<34}{'':<21}truth:  {c.truth_note}")
    print("-" * 108)
    n = len(cases)
    print(f"structural correct on {n_struct_correct}/{n} | practical correct on {n_pract_correct}/{n}")
    print("-" * 108)
    print("THE BOUNDARY (cases where the two diagnostics DISAGREE):")
    if not boundary_rows:
        print("  (none — battery did not exercise the boundary)")
    for c, s_id, p_id, s_detail, p_detail in boundary_rows:
        print(f"  • {c.name} [{c.klass}]")
        print(f"      structural: {'identifiable' if s_id else 'non-id'}  ← {s_detail}")
        print(f"      practical : {'identifiable' if p_id else 'non-id'}  ← {p_detail}")
        print(f"      ground truth: {'identifiable' if c.truth_identifiable else 'NON-identifiable'} ({c.truth_note})")
        if (not c.truth_identifiable) and s_id and (not p_id):
            print("      → structural FALSE-NEGATIVE for non-identifiability (its documented")
            print("        pairwise blind spot); practical recovers the degeneracy. This is the")
            print("        exact complementarity the paper claims, demonstrated not asserted.")
    print("=" * 108)


if __name__ == "__main__":
    main()
