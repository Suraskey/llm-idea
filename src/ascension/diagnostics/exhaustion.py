"""Design-class exhaustion certificates: "no experiment in this class can help."

The diagnostic (identifiability.py) says a parameter direction ``d`` is invisible under
the CURRENT observation design. The optimal-design ranker (optimal_design.py) picks
the best candidate among a finite list. Neither answers the question an autonomous
scientist actually faces on a plateau:

    Is ``d`` invisible under THIS design, or under EVERY design I am allowed to run?

If it is invisible under every design in the class (every sampling time, every
initial condition, every dose, every configuration the class parameterizes), then
more experiments of the same kind are wasted, and the right move is to CHANGE THE
CLASS: add a new observable (an IV arm), a new structural element (a family of
systems with distinct hidden charges), or report the invariant combination and stop.

This module certifies that distinction.

Two routes:

* ``certify_class_symmetry_symbolic`` (a proof). Given the model as a sympy
  expression in (parameters, observables, design controls), fix the parameters at
  the operating point, keep every observable and design control SYMBOLIC, and check
  that the directional derivative along ``d`` is identically zero. If it is, no
  choice of observable value or design setting changes the prediction along ``d``
  TO FIRST ORDER. That alone is not a proof of non-identifiability: at a singular
  point of the parameter-to-prediction map a first-order-invisible direction can
  still be identifiable (``f = a x + b**3 x**2`` at ``b = 0``: every design has
  zero sensitivity to ``b`` there, yet two design points recover ``b`` exactly).
  The certificate therefore also checks that the operating point is REGULAR: the
  rank of the class sensitivity (the derivative functions over every free
  variable) at ``theta`` equals its rank at nearby generic points. By the constant
  rank theorem the class map then has constant rank near ``theta``, its level set
  through ``theta`` is a manifold tangent to ``d``, and every parameter on it gives
  identical predictions for every design in the class (Rothenberg 1971's regularity
  condition, applied to the whole class rather than one design). Sound when it
  returns certified; conservative otherwise (a singular point is refused).

* ``certify_class_symmetry_numeric`` (a test, for black-box / ODE models). Sample
  many designs from the class, stack their sensitivity matrices, and check whether
  ``d`` is still a null direction of the STACKED matrix. If ``d`` survives every
  sampled design it is (numerically) class-level; if any sampled design moves the
  prediction along ``d``, the class contains a resolving experiment and the ranker
  should be asked for it.

``triage_plateau`` composes them with the diagnostic into the three-way verdict:
``capability`` (identifiable: the data suffice, keep reasoning),
``design_within_class`` (non-identifiable but resolvable: design the experiment),
``change_class`` (non-identifiable and certified class-level: no experiment of this
kind helps).

No LLM anywhere. Deterministic numpy/sympy.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import sympy

from ascension.diagnostics.identifiability import (
    IdentifiabilityVerdict,
    PredictFn,
    practical_identifiability,
    sensitivity_matrix,
)

_DEFAULT_NULL_TOL = 1e-7


@dataclass(frozen=True)
class ExhaustionCertificate:
    """Whether a confounded direction is a symmetry of the WHOLE design class."""

    certified: bool  # True: no design in the class can identify ``direction``
    method: str  # "symbolic" | "numeric"
    direction: tuple[float, ...]
    param_names: tuple[str, ...]
    # symbolic route: the simplified directional derivative (str) when non-zero
    residual: str | None
    # numeric route: designs sampled and the worst normalized response along d
    n_designs_sampled: int
    max_response_along_direction: float
    rationale: str
    recommendation: str
    # regularity of the operating point (constant class rank nearby). None when the
    # check was not run; False means the point is singular and the certificate was
    # refused even if the first-order response along ``direction`` vanished.
    regular_point: bool | None = None
    class_rank_at_point: int | None = None
    class_rank_nearby: int | None = None

    def describe(self) -> str:
        head = "CERTIFIED class-level symmetry" if self.certified else "NOT class-level"
        return f"{head} ({self.method}): {self.rationale}\n  -> {self.recommendation}"


def _fmt_dir(direction: Sequence[float], names: Sequence[str]) -> str:
    terms = [f"{w:+.3f}*{n}" for w, n in zip(direction, names, strict=True) if abs(w) > 1e-6]
    return " ".join(terms) if terms else "(zero)"


def _class_rank_symbolic(
    grads: Sequence[sympy.Expr],
    free: Sequence[sympy.Symbol],
    point: dict,
    rng: random.Random,
    *,
    dps: int = 50,
    rel_tol: float = 1e-30,
) -> int | None:
    """Rank of the class sensitivity at one parameter point.

    ``grads`` are the parameter derivatives as functions of the free (observable and
    design) symbols. Their span over the whole class has the dimension of the rank of
    the (free points x params) matrix at generic free points, so random rational
    points at ``dps`` digits give it with probability one; the 1e-30 floor separates
    a true zero from round-off (as in ``structural._exact_linear_dependence_groups``).
    Returns None when an evaluation stays symbolic or no finite rows are found.
    """
    import mpmath

    k = len(grads)
    n_pts = k + 6
    subbed = [g.subs(point) for g in grads]
    rows: list[list] = []
    with mpmath.workdps(dps):
        attempts = 0
        while len(rows) < n_pts and attempts < 20 * n_pts:
            attempts += 1
            pt = {f: sympy.Rational(rng.randint(3, 97), rng.randint(2, 23)) for f in free}
            try:
                vals = [sympy.N(g.subs(pt), dps) for g in subbed]
                if any(v.free_symbols for v in vals):
                    return None
                if any(abs(sympy.im(v)) > 0 for v in vals):
                    continue
                mp_vals = [mpmath.mpf(str(sympy.re(v))) for v in vals]
                if any(not mpmath.isfinite(v) for v in mp_vals):
                    continue
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                continue
            rows.append(mp_vals)
        if len(rows) < k:
            return None
        sv = mpmath.svd_r(mpmath.matrix(rows), compute_uv=False)
        s_max = max(sv[i] for i in range(len(sv)))
        if s_max == 0:
            return 0
        return sum(1 for i in range(len(sv)) if sv[i] >= rel_tol * s_max)


def _nearby_points(
    theta_sym: Sequence[sympy.Expr], rng: random.Random, n: int
) -> list[list[sympy.Expr]]:
    """Exact rational points within a few percent of ``theta`` (additive for zeros)."""
    out = []
    for _ in range(n):
        pt = []
        for t in theta_sym:
            scale = max(abs(t), sympy.Integer(1))
            pt.append(t + sympy.Rational(rng.choice((-1, 1)) * rng.randint(1, 9), 100) * scale)
        out.append(pt)
    return out


def certify_class_symmetry_symbolic(
    expr: sympy.Expr | str,
    params: Sequence[sympy.Symbol | str],
    theta: Sequence[float],
    direction: Sequence[float],
    *,
    free_vars: Sequence[sympy.Symbol | str] = (),
    snap_rational: bool = True,
    check_regular: bool = True,
    seed: int = 0,
) -> ExhaustionCertificate:
    """Prove (or fail to prove) that ``direction`` is a symmetry of the design class.

    Args:
      expr: the predicted observable as a sympy expression in ``params`` plus the
        observables and design controls (time, radius, dose, initial condition, ...).
      params: parameter symbols, in the order ``theta``/``direction`` use.
      theta: the operating point; substituted as exact rationals so a true zero
        reduces to literal 0 rather than round-off.
      direction: the confounded direction (from the diagnostic's null vector).
      free_vars: the observable / design-control symbols that must stay free. Any
        symbol in ``expr`` that is not a parameter is treated as free anyway; this
        argument exists to make the class explicit and to catch typos.
      snap_rational: snap the numeric direction to small rationals (``nsimplify``)
        before differentiating, so a numerically-estimated null vector such as
        (0.707, -0.707) becomes exactly (1, -1). The certificate is then checked
        for the snapped direction, which is reported.
      check_regular: also require the operating point to be regular (class rank at
        ``theta`` equal to the largest class rank at three nearby rational points).
        Without it the result is only a first-order statement at ``theta``, which a
        singular point can defeat; turn it off only to reproduce pre-2026-09-28 runs.
      seed: seeds the random evaluation points of the regularity check.

    Returns:
      ExhaustionCertificate. ``certified`` is True iff the directional derivative
      simplifies to literal 0 with every free variable still symbolic AND (when
      ``check_regular``) the operating point is regular.
    """
    if isinstance(expr, str):
        expr = sympy.sympify(expr)
    psyms = [sympy.Symbol(str(p)) if not isinstance(p, sympy.Symbol) else p for p in params]
    names = tuple(str(p) for p in psyms)
    if len(theta) != len(psyms) or len(direction) != len(psyms):
        raise ValueError("theta and direction must have one entry per parameter")

    declared_free = {
        sympy.Symbol(str(v)) if not isinstance(v, sympy.Symbol) else v for v in free_vars
    }
    actual_free = expr.free_symbols - set(psyms)
    missing = declared_free - expr.free_symbols
    if missing:
        raise ValueError(f"free_vars not present in expr: {sorted(map(str, missing))}")

    d = np.asarray(direction, dtype=float)
    if snap_rational:
        scale = float(np.max(np.abs(d))) or 1.0
        dsym = [sympy.nsimplify(float(x / scale), rational=True, tolerance=1e-3) for x in d]
    else:
        dsym = [sympy.Float(float(x)) for x in d]
    theta_sym = [sympy.nsimplify(float(t), rational=True, tolerance=1e-9) for t in theta]

    directional = sum(
        (dj * sympy.diff(expr, pj) for dj, pj in zip(dsym, psyms, strict=True)), sympy.Integer(0)
    )
    at_point = directional.subs(dict(zip(psyms, theta_sym, strict=True)))
    residual = sympy.simplify(at_point)
    first_order_null = residual == 0
    d_reported = tuple(float(x) for x in dsym)
    free_list = ", ".join(sorted(str(s) for s in actual_free)) or "(none)"

    regular: bool | None = None
    rank_at = rank_near = None
    if first_order_null and check_regular:
        # float exponents (r**3.5) would drop the 50-digit evaluation to double precision
        grads = [sympy.nsimplify(sympy.diff(expr, p), rational=True) for p in psyms]
        free = sorted(actual_free, key=str)
        rng = random.Random(seed)
        rank_at = _class_rank_symbolic(grads, free, dict(zip(psyms, theta_sym, strict=True)), rng)
        near = [
            _class_rank_symbolic(grads, free, dict(zip(psyms, pt, strict=True)), rng)
            for pt in _nearby_points(theta_sym, rng, 3)
        ]
        if rank_at is not None and all(r is not None for r in near):
            rank_near = max(near)
            regular = rank_at == rank_near
        else:
            regular = False  # could not establish regularity: refuse rather than guess
    certified = first_order_null and (regular is True or not check_regular)

    if first_order_null and not certified:
        why = (
            f"the class sensitivity has rank {rank_at} at the operating point but rank "
            f"{rank_near} nearby"
            if rank_at is not None and rank_near is not None
            else "the class rank could not be evaluated at generic points"
        )
        rationale = (
            f"the directional derivative along {_fmt_dir(d_reported, names)} vanishes for "
            f"every value of {{{free_list}}} at the operating point, but the point is "
            f"singular ({why}): first-order invisibility there does not imply that no "
            f"design can identify the direction."
        )
        rec = (
            "No certificate. Treat the plateau as resolvable in principle: ask the design "
            "ranker, or re-diagnose at a perturbed operating point."
        )
    elif certified:
        rationale = (
            f"the directional derivative along {_fmt_dir(d_reported, names)} is identically "
            f"zero for every value of {{{free_list}}}: no observable value and no design "
            f"setting in this class responds to that direction."
        )
        rec = (
            "Change the design class (add an observable or a structural element that "
            "breaks the symmetry), or report the identifiable combination and stop. "
            "More experiments of this kind cannot help."
        )
    else:
        rationale = (
            f"the directional derivative along {_fmt_dir(d_reported, names)} is "
            f"{residual} (not identically zero): some setting of {{{free_list}}} responds."
        )
        rec = (
            "A resolving experiment exists inside this class: ask the design ranker for "
            "the setting that maximizes the response along this direction."
        )
    return ExhaustionCertificate(
        certified=certified,
        method="symbolic",
        direction=d_reported,
        param_names=names,
        residual=None if certified else str(residual),
        n_designs_sampled=0,
        max_response_along_direction=0.0 if first_order_null else float("nan"),
        rationale=rationale,
        recommendation=rec,
        regular_point=regular,
        class_rank_at_point=rank_at,
        class_rank_nearby=rank_near,
    )


def certify_class_symmetry_numeric(
    design_sampler: Callable[[np.random.Generator], PredictFn],
    theta: Sequence[float],
    direction: Sequence[float],
    *,
    n_designs: int = 32,
    seed: int = 0,
    rel_step: float = 1e-6,
    null_tol: float = _DEFAULT_NULL_TOL,
    param_names: Sequence[str] | None = None,
    check_regular: bool = True,
) -> ExhaustionCertificate:
    """Test whether ``direction`` stays invisible across many designs from the class.

    Args:
      design_sampler: draws ONE design from the class and returns its predict
        closure (params -> observations). Each draw may have a different output
        length; the sensitivity matrices are stacked row-wise.
      theta: operating point.
      direction: the confounded direction to test.
      n_designs: how many designs to sample.
      null_tol: ``d`` counts as a null direction of the stacked sensitivity matrix
        when ||S d|| <= null_tol * ||S|| (spectral norm).
      check_regular: also require the stacked numeric rank at ``theta`` to equal the
        largest stacked rank at three nearby points (same sampled designs). A rank
        that rises off the point marks a singular point, where a first-order null
        direction need not be a symmetry; the certificate is then refused.

    Returns:
      ExhaustionCertificate. Certified means: across every sampled design the
      prediction did not move along ``d`` (numerically), at a point whose stacked rank
      is locally constant. This is evidence, not a proof; the symbolic route is the
      proof when a closed form exists.
    """
    theta_arr = np.asarray(theta, dtype=float).ravel()
    n = theta_arr.size
    names = (
        tuple(param_names) if param_names is not None else tuple(f"theta[{i}]" for i in range(n))
    )
    d = np.asarray(direction, dtype=float).ravel()
    if d.size != n:
        raise ValueError("direction must have one entry per parameter")
    d = d / (np.linalg.norm(d) or 1.0)

    rng = np.random.default_rng(seed)
    predicts: list[PredictFn] = []
    blocks: list[np.ndarray] = []
    worst = 0.0
    for _ in range(n_designs):
        predict = design_sampler(rng)
        predicts.append(predict)
        s_mat = sensitivity_matrix(predict, theta_arr, rel_step=rel_step)
        blocks.append(s_mat)
        s_norm = float(np.linalg.norm(s_mat, 2)) or 1.0
        worst = max(worst, float(np.linalg.norm(s_mat @ d)) / s_norm)
    stacked = np.vstack(blocks)
    stacked_norm = float(np.linalg.norm(stacked, 2)) or 1.0
    response = float(np.linalg.norm(stacked @ d)) / stacked_norm
    first_order_null = response <= null_tol

    def _numeric_rank(mat: np.ndarray) -> int:
        sv = np.linalg.svd(mat, compute_uv=False)
        return int(np.sum(sv >= null_tol * (sv[0] if sv.size and sv[0] > 0 else 1.0)))

    regular: bool | None = None
    rank_at = rank_near = None
    if first_order_null and check_regular:
        rank_at = _numeric_rank(stacked)
        scale = np.maximum(np.abs(theta_arr), 1.0)
        near = []
        for _ in range(3):
            jitter = rng.choice((-1.0, 1.0), size=n) * rng.uniform(0.01, 0.09, size=n) * scale
            theta_j = theta_arr + jitter
            near.append(
                _numeric_rank(
                    np.vstack([sensitivity_matrix(p, theta_j, rel_step=rel_step) for p in predicts])
                )
            )
        rank_near = max(near)
        regular = rank_at == rank_near
    certified = first_order_null and (regular is True or not check_regular)

    if first_order_null and not certified:
        rationale = (
            f"across {n_designs} sampled designs the prediction did not move along "
            f"{_fmt_dir(tuple(d), names)} at the operating point, but the stacked rank is "
            f"{rank_at} there and {rank_near} nearby: a singular point, so first-order "
            f"invisibility is not evidence of a class-level symmetry."
        )
        rec = "No certificate: ask the design ranker, or re-diagnose at a perturbed point."
    elif certified:
        rationale = (
            f"across {n_designs} sampled designs the prediction did not move along "
            f"{_fmt_dir(tuple(d), names)} (normalized response {response:.1e} <= {null_tol:.0e})."
        )
        rec = (
            "No sampled design in this class resolves the direction: change the design "
            "class or report the identifiable combination. (Numerical evidence over the "
            "sampled class, not a proof.)"
        )
    else:
        rationale = (
            f"at least one of {n_designs} sampled designs moves the prediction along "
            f"{_fmt_dir(tuple(d), names)} (worst single-design normalized response {worst:.2e})."
        )
        rec = "A resolving experiment exists inside this class: ask the design ranker for it."
    return ExhaustionCertificate(
        certified=certified,
        method="numeric",
        direction=tuple(float(x) for x in d),
        param_names=names,
        residual=None,
        n_designs_sampled=n_designs,
        max_response_along_direction=worst,
        rationale=rationale,
        recommendation=rec,
        regular_point=regular,
        class_rank_at_point=rank_at,
        class_rank_nearby=rank_near,
    )


@dataclass(frozen=True)
class PlateauTriage:
    """The three-way verdict an autonomous scientist needs on a plateau."""

    verdict: str  # "capability" | "design_within_class" | "change_class"
    diagnosis: IdentifiabilityVerdict
    certificates: tuple[ExhaustionCertificate, ...]
    rationale: str

    def describe(self) -> str:
        lines = [f"TRIAGE: {self.verdict}", f"  {self.rationale}"]
        for c in self.certificates:
            lines.append("  " + c.describe().replace("\n", "\n  "))
        return "\n".join(lines)


def triage_plateau(
    predict: PredictFn,
    theta: Sequence[float],
    *,
    certify: Callable[[Sequence[float]], ExhaustionCertificate],
    param_names: Sequence[str] | None = None,
    sigma: float = 1.0,
    rel_step: float = 1e-6,
) -> PlateauTriage:
    """Diagnose the current design, then certify each confounded direction.

    ``certify(direction)`` is the class-level check for ONE direction (bind either
    route with functools.partial). The verdict is ``capability`` if the diagnosis is
    identifiable; ``change_class`` if EVERY confounded direction is certified
    class-level (no experiment of this kind can help); ``design_within_class``
    otherwise (at least one direction is resolvable inside the class).
    """
    diag = practical_identifiability(
        predict, theta, sigma=sigma, rel_step=rel_step, param_names=param_names
    )
    if diag.identifiable:
        return PlateauTriage(
            verdict="capability",
            diagnosis=diag,
            certificates=(),
            rationale=(
                "the current data already determine every parameter direction; the "
                "plateau is not an observability artifact. Keep reasoning / searching."
            ),
        )
    certs = tuple(certify(d) for d in diag.confounded_directions)
    if certs and all(c.certified for c in certs):
        return PlateauTriage(
            verdict="change_class",
            diagnosis=diag,
            certificates=certs,
            rationale=(
                f"{len(certs)} confounded direction(s), every one a symmetry of the whole "
                "design class: no experiment of this kind can resolve them. Change the "
                "class or report the invariant combination."
            ),
        )
    n_res = sum(1 for c in certs if not c.certified)
    return PlateauTriage(
        verdict="design_within_class",
        diagnosis=diag,
        certificates=certs,
        rationale=(
            f"{n_res} of {len(certs)} confounded direction(s) respond to some design in the "
            "class: design the disambiguating experiment (rank candidates)."
        ),
    )
