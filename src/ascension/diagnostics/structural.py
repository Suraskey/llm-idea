"""Structural (exact, symbolic) identifiability for clean algebraic / rational forms.

PIVOT-001 Burst B, the complement to :func:`practical_identifiability` (numerical,
local). When the model is an explicit sympy expression in parameters + observables,
we can decide identifiability EXACTLY and GLOBALLY over observable-space — not just
locally at one operating point. This is the clean structural path the 2026-05-31
novelty pressure-test said to reserve for rational/ODE forms (reviewer-attack #4):
practical ID is the general fallback; structural ID is the exact answer when the
algebra is available.

Method (exact for the common confounds). Parameters ``p_i`` and ``p_j`` are
structurally confounded iff ``d(expr)/dp_i`` and ``d(expr)/dp_j`` are proportional
with a ratio FREE of the observables — the two parameters move the prediction in
lockstep at every observable point, so only one combination of them is recoverable:

  * product   ``f = (s1*s2)*g(r)``  -> ratio ``s2/s1`` (no observable) -> confounded
    (the alien hidden-charge L-051 case: s1, s2 not separately identifiable)
  * sum       ``f = (a+b)*g(r)``    -> ratio ``1``                     -> confounded
  * a parameter absent from expr (zero derivative) -> has no effect -> unidentifiable
  * distinct dependence ``f = a/r**2 + b/r**3.5`` -> ratio ``r**1.5`` (an observable)
    -> a and b ARE separately identifiable

Scope, stated honestly (the boundary is itself a Burst B finding):
  * Exact for confounds expressible as pairwise derivative-proportionality. Higher-
    order confounding (k params with only m < k-1 independent combinations and NO
    pairwise proportionality) is not fully captured — fall back to
    :func:`practical_identifiability` sampled at several points, or a full structural
    tool (SIAN / StructuralIdentifiability.jl) for rational ODE systems.
  * Conservative on un-simplifiable ratios: if sympy cannot decide a ratio is
    observable-free, the pair is treated as NOT-proven-confounded (so we never
    FALSELY claim non-identifiability; we may miss an exotic confound, which the
    numerical path then catches).

Provably LLM-free (IDENT-04): imports sympy + stdlib only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

# T-15.2-06/CR-01 (audit DET-06): expression STRINGS must never reach eval-based
# sympify. parse_expr with an explicit empty __builtins__ blocks __import__/
# eval/exec/open from resolving during parse (mirrors benchmarks.scoring's
# _safe_parse_expr; replicated locally so the diagnostic stays a standalone,
# sympy+stdlib-only instrument — the Track-A open-source boundary).
_SAFE_SYMPY_GLOBALS: dict = {k: v for k, v in sympy.__dict__.items() if not k.startswith("_")}
_SAFE_SYMPY_GLOBALS["__builtins__"] = {}
# standard + convert_xor ONLY — implicit_multiplication_application would
# split multi-char names ("s1*s2" -> 2*s**2) absent a local_dict of known
# symbols; the diagnostic accepts explicit-syntax strings.
_PARSE_TRANSFORMS = standard_transformations + (convert_xor,)


def _safe_sympify(s: str) -> sympy.Expr:
    """Parse a string into a sympy Expr without executing code."""
    return parse_expr(s, global_dict=_SAFE_SYMPY_GLOBALS, transformations=_PARSE_TRANSFORMS)



@dataclass(frozen=True)
class StructuralVerdict:
    """Exact symbolic identifiability result (IDENT-01..03). No LLM produced it."""

    identifiable: bool
    # Each group is a set of parameter names that enter only through a fixed
    # combination => not separately identifiable (IDENT-02). Singletons are
    # identifiable and omitted; a group may be a single name only when that
    # parameter is absent from the expression (no effect).
    confounded_groups: tuple[tuple[str, ...], ...]
    absent_params: tuple[str, ...]
    rationale: str
    disambiguation_hint: str
    # EXP-261 (2026-09-08): exact multi-parameter dependences. Each null direction is
    # a constant-coefficient linear relation among the derivative FUNCTIONS (a
    # dependence that needs three or more parameters, invisible to the pairwise
    # test). ``dependence_groups`` are their supports; ``rank_deficiency`` is the
    # number of independent such relations (the count of parameter combinations the
    # model form cannot determine). Pairwise groups keep their own, finer meaning.
    dependence_groups: tuple[tuple[str, ...], ...] = ()
    null_directions: tuple[tuple[float, ...], ...] = ()
    rank_deficiency: int = 0

    def describe(self) -> str:
        lines = [f"identifiable={self.identifiable}"]
        for g in self.confounded_groups:
            lines.append(
                f"  confounded (only a fixed combination is recoverable): {{{', '.join(g)}}}"
            )
        if self.absent_params:
            lines.append(f"  absent / no effect: {{{', '.join(self.absent_params)}}}")
        lines.append(f"rationale: {self.rationale}")
        lines.append(f"disambiguation: {self.disambiguation_hint}")
        return "\n".join(lines)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _ratio_is_observable_free(
    di: sympy.Expr, dj: sympy.Expr, observables: set[sympy.Symbol]
) -> bool:
    """True iff di/dj simplifies to something with no observable symbols (=> the
    two derivatives are proportional everywhere in observable-space)."""
    try:
        ratio = sympy.simplify(di / dj)
    except (
        TypeError,
        ValueError,
        ZeroDivisionError,
        NotImplementedError,
    ):  # pragma: no cover - defensive
        return False
    if ratio in (sympy.nan, sympy.zoo, sympy.oo, -sympy.oo):
        return False
    return not (ratio.free_symbols & observables)


def _exact_linear_dependence_groups(
    derivs: Sequence[sympy.Expr],
    active: Sequence[int],
    observables: set[sympy.Symbol],
    *,
    n_points: int | None = None,
    dps: int = 50,
    rel_tol: float = 1e-30,
    seed: int = 0,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]:
    """Find every linear dependence among the parameter-derivative FUNCTIONS.

    EXP-261 (2026-09-08). The pairwise ratio test above finds two derivatives that
    are proportional everywhere. It cannot see a dependence that needs three or
    more of them, e.g. the RC circuit ``(c_0 - x_0/c_1)/c_2`` where the three
    constants enter only as ``c_0/c_2`` and ``1/(c_1 c_2)``, or
    ``a r + b r^2 + c (r + r^2)`` where ``d/dc = d/da + d/db``. Those are exactly
    the cases where the derivatives ``d f / d p_i``, as functions of the
    observables, are linearly dependent with CONSTANT coefficients.

    Test: evaluate the derivative functions at ``n_points`` random rational
    observable points at ``dps`` decimal digits and take the rank of the
    (points x params) matrix. Generic points make the rank of that matrix equal
    the dimension of the span of the functions (a dependence that holds at
    every point holds at these; an accidental rank drop at all sampled points
    has probability zero). Fifty digits with a 1e-30 relative floor separates a
    true zero from round-off by twenty orders of magnitude. Each null vector's
    support is one confounded group; supports are unioned.

    Returns (groups as index tuples, null directions over ``active`` order).
    """
    import mpmath

    k = len(active)
    if k < 2:
        return (), ()
    n_pts = n_points if n_points is not None else k + 6
    obs = sorted(observables, key=str)
    # the parameter symbols themselves may appear in the derivatives (any form that
    # is not linear in its parameters); the dependence is STRUCTURAL when it holds
    # at generic parameter values, so it is checked at two independent random
    # rational parameter points and must hold at both with the same support.
    # floats in the model form (r**3.5) would pull the 50-digit evaluation down to
    # double precision and turn a true zero singular value into ~1e-16; rationalize.
    derivs = [sympy.nsimplify(d, rational=True) for d in derivs]
    psyms = sorted(
        {sym for i in active for sym in derivs[i].free_symbols} - set(obs), key=str
    )
    rng = __import__("random").Random(seed)

    def _null_space_at(theta_pt: dict) -> tuple[list[tuple[float, ...]], list[tuple[int, ...]]] | None:
        rows: list[list[mpmath.mpf]] = []
        with mpmath.workdps(dps):
            attempts = 0
            while len(rows) < n_pts and attempts < 20 * n_pts:
                attempts += 1
                # rational points away from 0 (avoid the poles most model forms carry)
                pt = {o: sympy.Rational(rng.randint(3, 97), rng.randint(2, 23)) for o in obs}
                pt.update(theta_pt)
                try:
                    vals = [sympy.N(derivs[i].subs(pt), dps) for i in active]
                    if any(v.free_symbols for v in vals):
                        return None  # something stayed symbolic: refuse to guess
                    mp_vals = [mpmath.mpf(str(sympy.re(v))) for v in vals]
                    if any(not mpmath.isfinite(v) for v in mp_vals) or any(
                        abs(sympy.im(v)) > 0 for v in vals
                    ):
                        continue
                except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                    continue
                rows.append(mp_vals)
            if len(rows) < k:
                return None
            mat = mpmath.matrix(rows)
            _u, sv, v = mpmath.svd_r(mat)
            s_max = max(sv[i] for i in range(len(sv))) if len(sv) else mpmath.mpf(0)
            if s_max == 0:
                return [tuple(1.0 for _ in active)], [tuple(range(k))]
            null_rows = [i for i in range(len(sv)) if sv[i] < rel_tol * s_max]
            # svd_r returns V with singular vectors as ROWS (k x k when rows >= k)
            directions = []
            for i in null_rows:
                vec = [float(v[i, j]) for j in range(k)]
                norm = sum(x * x for x in vec) ** 0.5
                vec = [x / norm for x in vec] if norm > 0 else vec
                directions.append(tuple(vec))
            # the projector onto the null space is basis-invariant (an SVD basis of a
            # 2-dimensional null space is an arbitrary rotation, so vector SUPPORTS
            # are not); P[i, j] != 0 iff parameters i and j share a dependence.
            proj = [[0.0] * k for _ in range(k)]
            for d in directions:
                for a in range(k):
                    for b in range(k):
                        proj[a][b] += d[a] * d[b]
        return directions, proj

    draws = []
    for _ in range(2 if psyms else 1):
        theta_pt = {p: sympy.Rational(rng.randint(3, 97), rng.randint(2, 23)) for p in psyms}
        res = _null_space_at(theta_pt)
        if res is None:
            return (), ()
        draws.append(res)
    directions, proj = draws[0]
    if len(draws) == 2 and len(draws[1][0]) != len(directions):
        # the deficiency must be the same at a second generic parameter point to
        # count as structural; otherwise it was an accident of the first draw
        return (), ()
    if not directions:
        return (), ()
    uf = _UnionFind(k)
    for a in range(k):
        for b in range(a + 1, k):
            if abs(proj[a][b]) > 1e-9:
                uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for a in range(k):
        if abs(proj[a][a]) > 1e-9:
            groups.setdefault(uf.find(a), []).append(a)
    out_groups = tuple(
        tuple(active[j] for j in sorted(set(idxs))) for idxs in groups.values()
    )
    return out_groups, tuple(directions)


def structural_identifiability(
    expr: sympy.Expr | str,
    params: Sequence[sympy.Symbol | str],
    observables: Sequence[sympy.Symbol | str],
) -> StructuralVerdict:
    """Decide exact structural identifiability of ``params`` in ``expr``.

    Args:
      expr: the model's predicted quantity as a sympy expression (or a string
        sympify can parse) in the parameter + observable symbols.
      params: the parameter symbols whose identifiability is in question.
      observables: the measured/independent variables (r, t, v, masses, ...).

    Returns:
      StructuralVerdict. ``identifiable`` is True iff every parameter has a
      non-zero derivative AND no two parameters are derivative-proportional with
      an observable-free ratio.
    """
    if isinstance(expr, str):
        expr = _safe_sympify(expr)
    else:
        expr = sympy.sympify(expr)
    psyms = [sympy.Symbol(str(p)) if not isinstance(p, sympy.Symbol) else p for p in params]
    osyms = {sympy.Symbol(str(o)) if not isinstance(o, sympy.Symbol) else o for o in observables}
    names = [str(p) for p in psyms]
    n = len(psyms)

    derivs = [sympy.diff(expr, p) for p in psyms]
    absent = tuple(names[i] for i in range(n) if derivs[i] == 0)

    uf = _UnionFind(n)
    active = [i for i in range(n) if derivs[i] != 0]
    for a in range(len(active)):
        for b in range(a + 1, len(active)):
            i, j = active[a], active[b]
            if _ratio_is_observable_free(derivs[i], derivs[j], osyms):
                uf.union(i, j)

    # EXP-261: the exact multi-parameter dependence test (constant-coefficient
    # linear dependence among the derivative functions). Reported in its own
    # fields; the pairwise groups keep their finer "proportional everywhere" meaning.
    dep_groups_idx, dep_dirs = _exact_linear_dependence_groups(derivs, active, osyms)
    dependence_groups = tuple(
        sorted(tuple(sorted(names[i] for i in g)) for g in dep_groups_idx)
    )
    # the directions are over ``active`` order; re-embed over all params (absent = 0)
    null_directions = []
    for d in dep_dirs:
        full = [0.0] * n
        for pos, i in enumerate(active):
            full[i] = d[pos]
        null_directions.append(tuple(full))
    rank_deficiency = len(null_directions)

    groups: dict[int, list[int]] = {}
    for i in active:
        groups.setdefault(uf.find(i), []).append(i)
    # BUG-05 (AUDIT-036): canonicalize ordering — sort names within each group, then
    # sort the groups. The SET of confounded params was always correct; only tuple
    # order varied with input parameter order, making order-sensitive downstream
    # assertions fragile. Confounding is a property of the math, not the arg order.
    confounded_groups = tuple(
        sorted(tuple(sorted(names[i] for i in idxs)) for idxs in groups.values() if len(idxs) > 1)
    )

    identifiable = not confounded_groups and not absent and rank_deficiency == 0
    if identifiable:
        rationale = (
            f"every parameter {{{', '.join(names)}}} has a distinct observable-"
            f"dependent effect on the prediction; all are exactly recoverable."
        )
        hint = "Identifiable from this model form; no degeneracy."
    else:
        parts = []
        if confounded_groups:
            parts.append(
                "confounded groups (only a fixed combination is recoverable): "
                + "; ".join("{" + ", ".join(g) + "}" for g in confounded_groups)
            )
        if absent:
            parts.append(f"parameters with no effect on the prediction: {{{', '.join(absent)}}}")
        if rank_deficiency:
            parts.append(
                f"{rank_deficiency} exact multi-parameter dependence(s) among the derivative "
                "functions (only n - " + str(rank_deficiency) + " combinations are recoverable): "
                + "; ".join("{" + ", ".join(g) + "}" for g in dependence_groups)
            )
        rationale = "; ".join(parts)
        hint = (
            "Add a measurement/configuration that breaks the proportionality — i.e. "
            "where the confounded parameters affect the prediction DIFFERENTLY (for a "
            "hidden product s1*s2, observe configurations at DISTINCT products so the "
            "amplitude varies across them). Absent parameters need a regime where they "
            "actually act. This is the experiment the engine should request."
        )

    return StructuralVerdict(
        identifiable=identifiable,
        confounded_groups=confounded_groups,
        absent_params=absent,
        rationale=rationale,
        disambiguation_hint=hint,
        dependence_groups=dependence_groups,
        null_directions=tuple(null_directions),
        rank_deficiency=rank_deficiency,
    )
