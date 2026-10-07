"""PySINDy external baseline — SciML reference scored against the same score() function.

Plain English: PySINDy is a library that tries to find a short equation for your
data using regression on polynomial terms. We run it on the same trajectories our
AI agents will see, then feed its output through our three-tier scorer so the
resulting numbers can sit next to each other in the paper's Tier 1 table — no
apples-to-oranges with PySINDy's native R²-on-derivatives.

Binding decisions:
  - 06.0 RESEARCH §Pattern 5 lines 395–420 (PySINDy → sympy bridge; same score()
    entry point that agents will use).
  - SCOPE §9 / TIER1-03 (PySINDy is the canonical non-LLM SciML baseline).
  - 06.0 PATTERNS §src/ascension/benchmarks/pysindy_baseline.py (thin-wrapper
    shape — one public function, one private string→sympy helper).

Pitfalls honored:
  - RESEARCH #5 (feature_names default hides axis bugs) — ALWAYS explicitly
    ``feature_names=list(feature_names)`` fetched from DEFAULT_PARAMS; never
    default to x0/x1/x2.
  - RESEARCH #6 (multiple-trajectory 3D-slice footgun) — train_data is built as
    a LIST COMPREHENSION over ``range(N)`` with ``if bundle.train_mask[i]``,
    never ``bundle.noisy[bundle.train_mask]`` (which would be a 3D ndarray that
    PySINDy interprets differently).
  - RESEARCH #7 (noise convention) — inherited from generate_trajectories; this
    module is noise-convention-agnostic.

PySINDy 2.1.0 API notes (differences from RESEARCH §Pattern 5 pseudo-code —
documented as deviations in the Plan-03 SUMMARY):

  1. ``feature_names`` lives on ``SINDy.fit(...)``, NOT on ``SINDy.__init__``.
     Passing ``feature_names=`` to the constructor raises TypeError in 2.1.0.
  2. ``multiple_trajectories=True`` kwarg was removed from ``SINDy.fit``. A
     list of 2D arrays is auto-detected as multi-trajectory training data.
  3. ``SINDy.equations()`` returns raw RHS strings WITHOUT the ``(x)' = ``
     LHS prefix. PySINDy's ``print()`` still emits the LHS-prefixed form. The
     parser below handles both shapes so hand-written tests that use the
     ``(x)' = `` format (documented in the plan) continue to work.
  4. The equation strings use ``x^2`` / implicit-multiplication syntax (e.g.
     ``' 1.000 x + -10.000 x y'``). Vanilla ``sympify`` trips on both; we use
     ``parse_expr`` with ``standard_transformations + (convert_xor,
     implicit_multiplication_application)`` so ``x^2`` becomes ``x**2`` and
     ``10 x y`` becomes ``10*x*y``.

Threat-model mitigations:
  - T-06.0-10 (Tampering on sympify of PySINDy output): CR-01 fix (2026-04-22):
    ``parse_expr(..., local_dict={name: symbols(name) for name in feature_names},
    global_dict=_SAFE_SYMPY_GLOBALS)``. The local_dict restricts name
    resolution to the 2–3 known state variables, and the global_dict has an
    empty ``__builtins__`` mapping so crafted strings cannot resolve
    ``__import__``, ``eval``, ``exec``, etc. during parsing. Unknown
    identifiers raise NameError → caught → returned as ``symbolic_parse_fail``
    via score().
"""

from __future__ import annotations

import re
import tokenize

import numpy as np
import pysindy as ps
import sympy
from sympy import symbols
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ascension.benchmarks.score_types import BenchmarkScore
from ascension.benchmarks.scoring import _SAFE_SYMPY_GLOBALS, score
from ascension.benchmarks.systems import params_for
from ascension.benchmarks.types import TrajectoryBundle
from ascension.common.logging import get_logger

logger = get_logger(__name__)

# Optional LHS prefix: "(x)' = " / "(x0)' = " — pysindy.print() format. When
# this pattern matches we strip it and sympify the RHS. When it doesn't, we
# treat the whole string as a standalone RHS (pysindy.equations() format).
_LHS_RE = re.compile(r"^\s*\(([a-zA-Z_]\w*)\)'\s*=\s*")

# Transformations that let sympy parse "1.000 x + -10.000 x y + 11.243 x^3"
# as "1.000*x + -10.000*x*y + 11.243*x**3":
#   - convert_xor: ^ → ** (so "x^2" parses as "x**2")
#   - implicit_multiplication_application: insert * between juxtaposed factors
#     (so "10.0 x" parses as "10.0*x", and "x y" parses as "x*y")
_PARSE_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)


def _pysindy_strings_to_sympy(
    raw_eqs: list[str],
    system: str,
) -> list[sympy.Expr]:
    """Parse PySINDy ``equations()`` (or ``print()``) strings into sympy Expr list.

    Plain English: takes pysindy's human-readable equation strings — e.g.
    ``'(x)' = 1.000 x + -10.000 x y'`` or its LHS-less cousin
    ``' 1.000 x + -10.000 x y'`` — and turns each one into a sympy expression
    for the right-hand side of dx/dt. Order is preserved (matches the order
    ``feature_names`` was passed to ``SINDy.fit``).

    Raises ``sympy.SympifyError`` when the RHS won't parse (caller surfaces it
    via score()'s ``symbolic_parse_fail`` branch) OR when the number of
    equations doesn't match the system's feature count.
    """
    feature_names = params_for(system)["feature_names"]
    sym_locals = {n: symbols(n) for n in feature_names}
    parsed: list[sympy.Expr] = []
    for eq in raw_eqs:
        # Strip optional "(x)' = " LHS if present (print() format); keep the
        # whole thing otherwise (equations() format).
        m = _LHS_RE.match(eq)
        rhs_str = eq[m.end() :] if m else eq
        rhs_str = rhs_str.strip()
        if not rhs_str:
            raise sympy.SympifyError(f"pysindy emitted empty RHS: {eq!r}")
        try:
            # CR-01: pass global_dict=_SAFE_SYMPY_GLOBALS (empty __builtins__)
            # so crafted PySINDy output cannot resolve __import__/eval/exec.
            # Input here is pysindy.model.equations() which is low risk
            # today, but the same hardening applies as a matter of
            # consistent discipline — a compromised pysindy install or
            # crafted noise->coefficient chain would otherwise be a vector.
            rhs_expr = parse_expr(
                rhs_str,
                local_dict=sym_locals,
                global_dict=_SAFE_SYMPY_GLOBALS,
                transformations=_PARSE_TRANSFORMS,
            )
        except (
            SyntaxError,
            ValueError,
            TypeError,
            NameError,
            AttributeError,
            tokenize.TokenError,
        ) as exc:
            # Normalize all parser failures to SympifyError so callers have
            # one exception class to catch.
            raise sympy.SympifyError(f"pysindy output failed to parse: {eq!r}") from exc
        parsed.append(rhs_expr)
    if len(parsed) != len(feature_names):
        raise sympy.SympifyError(
            f"pysindy returned {len(parsed)} equations; "
            f"expected {len(feature_names)} for system {system!r}",
        )
    return parsed


def run_pysindy_baseline(bundle: TrajectoryBundle) -> BenchmarkScore:
    """Fit PySINDy on bundle.noisy (train mask only), parse equations, score via score().

    Plain English: runs the reference library on the same data an agent would
    see, extracts the equation it proposes, and feeds that through our scorer
    so the paper's Tier 1 table compares like-to-like numbers (same scoring
    function that Phase 7.0+ agent loops will use).

    Returns a ``BenchmarkScore`` — ``exact`` (bool), ``rmse`` (float | None),
    ``qualitative`` (dict), ``reasons`` (dict, three tier keys), ``elapsed_ms``
    (dict, three tier keys). The scoring function's invariants hold
    transitively: ``rmse is None iff reasons['numerical'] != 'ok'``.
    """
    system = bundle.spec.system
    cfg = params_for(system)
    feature_names = cfg["feature_names"]
    dt = float(bundle.t[1] - bundle.t[0])

    # RESEARCH Pitfall #6: list comprehension over ``range(N)`` filtered by
    # ``train_mask``. NEVER ``bundle.noisy[bundle.train_mask]`` — that produces
    # a 3D ndarray which PySINDy 2.1.0 treats as a single (N_train, T, D)
    # training tensor rather than N_train separate (T, D) trajectories.
    train_data: list[np.ndarray] = [
        bundle.noisy[i] for i in range(bundle.spec.n_trajectories) if bundle.train_mask[i]
    ]
    if not train_data:
        # No training trajectories — cannot fit. Return a parse-fail score via
        # the scoring function so the invariant chain stays intact.
        logger.warning("pysindy baseline: empty train_data (train_mask all False)")
        return score([""], bundle.spec, bundle)

    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=3),
        optimizer=ps.STLSQ(threshold=0.1),
    )
    # RESEARCH Pitfall #5: feature_names MUST be explicit per-system. pysindy
    # 2.1.0 accepts this on .fit() (2.0.x accepted it on __init__; API moved).
    # Passing list(feature_names) creates a fresh list so downstream mutation
    # can't affect our registry.
    model.fit(train_data, t=dt, feature_names=list(feature_names))
    raw_eqs: list[str] = model.equations()
    logger.info("pysindy equations for %s: %s", system, raw_eqs)

    try:
        proposed_expr = _pysindy_strings_to_sympy(raw_eqs, system)
    except sympy.SympifyError as exc:
        logger.warning("pysindy output failed to parse: %s", exc)
        # Fall through via the scoring function's string-input path — score()
        # will sympify and land on ``symbolic_parse_fail`` for the symbolic
        # tier, plus ``numerical_diverged`` for the numerical tier since
        # proposed_expr is empty.
        return score(raw_eqs, bundle.spec, bundle)

    return score(proposed_expr, bundle.spec, bundle)
