"""Three-tier scorer — symbolic equivalence + numerical RMSE + qualitative match.

Public surface: `score(proposed_rhs, truth_spec, held_out) -> BenchmarkScore`.
Private sub-scorers `_symbolic_match`, `_numerical_rmse`, `_qualitative_match`
are exported for Plan 03 (PySINDy baseline) to call at finer granularity if
needed.

Plain English: given a candidate differential equation and the real data,
answer three questions — does the equation exactly match the truth symbolically
(algebra equality)? Do its predictions match numerically on data it hasn't
seen (RMSE)? And does the overall shape match (fixed points, limit cycles,
attractor box)? Each question produces a yes/no/number AND a reason string so
failures are loud rather than silent (CLAUDE.md no-cover-ups).

Priority rules per tier (first match wins within a tier):

  Symbolic tier:
    1. parse_expr(proposed_str) raises (SympifyError/NameError/...) → reasons['symbolic'] = 'symbolic_parse_fail', exact=False
    2. Poly-canonicalized equality holds                → reasons['symbolic'] = 'ok', exact=True
    3. simplify(diff).equals(0) under timeout           → reasons['symbolic'] = 'ok', exact=True
    4. timeout expires on simplify fallback             → reasons['symbolic'] = 'symbolic_timeout', exact=False
    5. otherwise                                        → reasons['symbolic'] = 'ok', exact=False (clean no-match)

  Numerical tier:
    1. solve_ivp(proposed).success is False             → reasons['numerical'] = 'numerical_diverged', rmse=None
    2. any(isfinite(y) == False) in returned y          → reasons['numerical'] = 'numerical_blowup', rmse=None
    3. any(y.span > box_mult * truth_span) per axis     → reasons['numerical'] = 'numerical_blowup', rmse=None
    4. otherwise                                        → reasons['numerical'] = 'ok', rmse=float

  Qualitative tier:
    1. per-system feature extractor raises              → reasons['qualitative'] = 'qualitative_unknown', qualitative=<partial>
    2. otherwise                                        → reasons['qualitative'] = 'ok', qualitative=<dict>

Invariant (enforced at BenchmarkScore construction): rmse is None iff
reasons['numerical'] != 'ok'. Violation raises AssertionError (loud, per
CLAUDE.md). See Shared Pattern I + score_types.py module docstring.

Binding decisions:
  - 06.0 RESEARCH §Pattern 3 (three-tier scoring with explicit reason codes).
  - 06.0 RESEARCH §Pattern 4 (system-specific qualitative features).
  - 06.0 RESEARCH §Pitfall #3 (Lorenz RMSE horizon = 2.0, not full t_span).
  - PATTERNS §Shared Pattern D (reason-string-on-DTO, not exception-per-mode).
  - PATTERNS §Shared Pattern I (invariant-doc-in-source + grep regression).

Pitfalls honored:
  - Pitfall #2 (sympy simplify hangs) → ThreadPoolExecutor timeout-wrap.
  - Pitfall #3 (Lorenz chaotic RMSE) → _DEFAULT_HORIZONS['lorenz'] = 2.0.
  - Pitfall #5 (feature-name ordering) → variables fetched from DEFAULT_PARAMS
    so truth/proposed share the same x/y/z symbol identity.
  - Pitfall #8 (blowup detection) → explicit sol.success check + isfinite
    check + bounding-box check + scipy terminal event on ||y|| > hard cap
    (stops LSODA in O(ms) on pathological RHS), each mapping to a distinct
    reason code.

Threat-model mitigations (Phase 6.0 threat register T-06.0-05 through -09):
  - T-06.0-05 DoS via pathological sympy expr → timeout-wrapped simplify.
  - T-06.0-06 Tampered sympify payload → CR-01 fix (2026-04-22): route
    agent strings through parse_expr with global_dict carrying empty
    __builtins__. Blocks __import__/eval/exec from resolving during parse.
  - T-06.0-07 DoS via diverging RHS → bounded t_eval + bounding-box + isfinite.
  - T-06.0-08 Silent invariant drift → AssertionError on rmse/reasons mismatch.
"""

from __future__ import annotations

import time
import tokenize
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import numpy as np
import sympy
from scipy.integrate import solve_ivp
from sympy import Poly, Rational, nsimplify, symbols
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ascension.benchmarks.reason_codes import ScoreReasonCode
from ascension.benchmarks.score_types import BenchmarkScore
from ascension.benchmarks.systems import INTEGRATOR_KEYWORDS, params_for
from ascension.benchmarks.types import BenchmarkSpec, TrajectoryBundle
from ascension.common.logging import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Safe parse_expr globals — RCE mitigation (T-06.0-06)
# -----------------------------------------------------------------------------
#
# CR-01 fix (2026-04-22): `sympy.sympify` calls `eval()` internally. Passing
# `locals={v: v for v}` DOES NOT block access to Python builtins; an
# agent-controlled string like `__import__("os").system("id")` executes
# during the parse step before any exception can be caught.
#
# The fix: route every agent/third-party string through
# `sympy.parsing.sympy_parser.parse_expr(...)` with an explicit
# `global_dict` that carries sympy's public namespace PLUS an EMPTY
# `__builtins__` mapping (which blocks `__import__`, `eval`, `exec`,
# `open`, etc. from resolving inside the parser).
#
# The `local_dict` still restricts name resolution to the known state
# variables (Tampering defense, T-06.0-06). `parse_expr` uses AST
# transformations rather than `eval`, so this combination is safe even
# against crafted strings.

_SAFE_SYMPY_GLOBALS: dict = {k: v for k, v in sympy.__dict__.items() if not k.startswith("_")}
# Explicit empty builtins — blocks __import__, eval, exec, open from
# resolving during parse. This is the critical line; without it parse_expr
# still finds Python builtins via the default global namespace.
_SAFE_SYMPY_GLOBALS["__builtins__"] = {}

# Transformations: standard + convert_xor ("x^2" → "x**2") +
# implicit_multiplication ("10 x y" → "10*x*y"). Matches pysindy_baseline.
_PARSE_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)


def _safe_parse_expr(
    s: str,
    local_dict: dict[str, sympy.Symbol],
) -> sympy.Expr:
    """Parse an agent/third-party string into a sympy Expr without executing code.

    Plain English: sympify() used to call eval() which let attacker strings
    like `__import__('os').system(...)` run on the host. This wrapper uses
    parse_expr with an empty __builtins__ dict so those names resolve to
    nothing and parsing fails cleanly instead of running a command.
    """
    return parse_expr(
        s,
        local_dict=local_dict,
        global_dict=_SAFE_SYMPY_GLOBALS,
        transformations=_PARSE_TRANSFORMS,
    )


# -----------------------------------------------------------------------------
# Per-system defaults and helpers
# -----------------------------------------------------------------------------

# Lorenz default RMSE horizon (RESEARCH Pitfall #3 — ≈ Lyapunov time scale).
# Non-chaotic systems get None = full t_span.
_DEFAULT_HORIZONS: dict[str, float | None] = {
    "lotka_volterra": None,  # full t_span is fine (non-chaotic, bounded orbits)
    "van_der_pol": None,  # limit cycle — long-horizon RMSE is still meaningful
    "lorenz": 2.0,  # ~2 Lyapunov times; see RESEARCH Pitfall #3
}


def _vars_for(system: str) -> list[sympy.Symbol]:
    """Return the sympy Symbol list for `system`, in feature-name order.

    Plain English: the ODE's state variables as sympy names — e.g. for
    Lorenz, `[x, y, z]` — so symbolic comparison is done with identical
    Symbol identity on both truth and proposed.
    """
    names = params_for(system)["feature_names"]
    return list(symbols(" ".join(names)))


def _canonicalize_expr(
    expr: sympy.Expr,
    vars_: list[sympy.Symbol],
    coeff_tol: float,
) -> sympy.Expr:
    """Snap floating-point coefs to rationals within `coeff_tol`, then Poly-canonicalize.

    Plain English: takes a polynomial like `0.9999 * x - 0.1001 * x*y`, snaps
    the messy decimals to nearby clean fractions within `coeff_tol`, then asks
    sympy to write it in a canonical polynomial form so two
    equivalent-but-differently-typed expressions compare equal.
    """
    snapped = nsimplify(expr, rational=False, tolerance=coeff_tol)
    # Poly() will raise if the expression isn't polynomial in vars_; caller
    # decides whether to treat that as a clean no-match or a parse failure.
    return Poly(snapped, *vars_).as_expr()


# -----------------------------------------------------------------------------
# Symbolic tier
# -----------------------------------------------------------------------------


def _symbolic_match(
    proposed: list[sympy.Expr],
    truth: list[sympy.Expr],
    vars_: list[sympy.Symbol],
    *,
    timeout_s: float,
    coeff_tol: float,
) -> tuple[bool, str]:
    """Per-state-variable symbolic equivalence. ALL equations must match.

    Returns (True, 'ok') iff every proposed[i] is equivalent to truth[i]
    under rational-coefficient canonicalization or timeout-bounded simplify
    fallback. On timeout → (False, 'symbolic_timeout'). On mismatch →
    (False, 'ok') — a clean NO is not an error.
    """
    if len(proposed) != len(truth):
        return False, ScoreReasonCode.OK.value  # clean no-match, not an error

    for p, t in zip(proposed, truth, strict=False):
        # Try Poly canonicalization first (bounded-time rational arithmetic).
        try:
            canon_p = _canonicalize_expr(p, vars_, coeff_tol)
            canon_t = _canonicalize_expr(t, vars_, coeff_tol)
            if canon_p == canon_t:
                continue  # cheap path — exact canonical equality
            # Fallback: simplify the difference with a hard thread-level timeout.
            diff = canon_p - canon_t
        except (sympy.PolynomialError, AttributeError, TypeError):
            # Non-polynomial expression — fall back to simplify-the-difference.
            # p and t are ALREADY sympy.Expr per the function signature (see
            # the `list[sympy.Expr]` annotation). Subtract them directly
            # rather than round-tripping through sympify, which would bypass
            # our CR-01 parse_expr guard if a string ever leaks past typing.
            diff = p - t

        def _job(d: sympy.Expr = diff) -> bool:
            # simplify() may hang on pathological inputs — wrapped below.
            return bool(sympy.simplify(d) == 0)

        with ThreadPoolExecutor(max_workers=1) as ex:
            try:
                eq = ex.submit(_job).result(timeout=timeout_s)
            except FuturesTimeout:
                return False, ScoreReasonCode.SYMBOLIC_TIMEOUT.value
            except Exception as exc:  # noqa: BLE001 — reason-code captures it
                logger.warning("symbolic simplify raised: %s: %s", type(exc).__name__, exc)
                return False, ScoreReasonCode.OK.value

        if not eq:
            return False, ScoreReasonCode.OK.value  # clean no-match

    return True, ScoreReasonCode.OK.value


# -----------------------------------------------------------------------------
# Numerical tier
# -----------------------------------------------------------------------------


def _numerical_rmse(
    proposed_rhs: Callable[[float, np.ndarray], np.ndarray],
    held_out: TrajectoryBundle,
    *,
    horizon: float | None,
    box_mult: float,
) -> tuple[float | None, str]:
    """Integrate proposed RHS on held-out ICs; compute RMSE vs truth clean trajectories.

    Failure modes (priority order):
      1. sol.success is False                           → (None, 'numerical_diverged')
      2. any non-finite value in returned y             → (None, 'numerical_blowup')
      3. any axis span > box_mult * truth span          → (None, 'numerical_blowup')

    Plain English: re-runs the proposed equation from each held-out starting
    point. If scipy gives up, or the numbers explode to infinity / NaN, or the
    trajectory wanders outside a sensible multiple of the true system's range,
    we return None + a loud reason code. Otherwise we compute average
    squared error vs the real trajectory.
    """
    # Slice time axis to horizon if requested (Lorenz default 2.0 per Pitfall #3).
    t_full = held_out.t
    if horizon is not None:
        # Use t relative to the bundle's t[0] so horizon is a span, not an absolute cap.
        horizon_mask = t_full <= (t_full[0] + horizon)
        t = t_full[horizon_mask]
        if len(t) < 2:
            # Horizon too short for the bundle's sample density — surface as diverged.
            logger.warning(
                "horizon=%s produces <2 timepoints in bundle t[0]=%s; returning diverged",
                horizon,
                t_full[0],
            )
            return None, ScoreReasonCode.NUMERICAL_DIVERGED.value
    else:
        t = t_full

    held_mask = ~held_out.train_mask
    if not held_mask.any():
        # No held-out trajectories — can't compute RMSE. Surface as diverged.
        logger.warning("no held-out trajectories (train_mask all True); returning diverged")
        return None, ScoreReasonCode.NUMERICAL_DIVERGED.value

    held_clean = held_out.clean[held_mask][:, : len(t), :]
    held_ics = held_out.ics[held_mask]

    # Truth bbox + allowed span (box_mult times the truth range, per axis).
    truth_all = held_out.clean
    truth_span = truth_all.max(axis=(0, 1)) - truth_all.min(axis=(0, 1))
    # Guard against zero-width axes (e.g. dim that's always 0).
    allowed_span = np.maximum(truth_span * box_mult, 1e-6)

    # Blowup event: terminates solve_ivp immediately when ||y||_inf exceeds a
    # hard cap. Without this, a pathological RHS (e.g. `y**3 + 1e3`) makes
    # LSODA repeatedly shrink step size trying to meet tight tolerances and
    # can spin for minutes. The event fires in O(ms) and we route the early
    # termination through the existing `y.shape[0] != len(t)` branch below
    # as NUMERICAL_BLOWUP.
    #
    # The cap is set generously: box_mult * max(|truth|) per axis, with a
    # floor of 1e6 for zero-centered systems. Real trajectories stay well
    # inside; diverging ones burst through within a handful of steps.
    truth_absmax = np.abs(truth_all).max(axis=(0, 1))
    blowup_cap = float(max((truth_absmax * box_mult).max(), 1e6))

    def _blowup_event(_t: float, y: np.ndarray) -> float:
        # Positive while safe, crosses zero downward at blowup.
        return blowup_cap - float(np.abs(y).max())

    _blowup_event.terminal = True  # type: ignore[attr-defined]
    _blowup_event.direction = -1.0  # type: ignore[attr-defined]

    predicted = np.empty_like(held_clean)
    for i, ic in enumerate(held_ics):
        try:
            sol = solve_ivp(
                proposed_rhs,
                (t[0], t[-1]),
                ic,
                t_eval=t,
                events=_blowup_event,
                **INTEGRATOR_KEYWORDS,
            )
        except Exception as exc:  # noqa: BLE001 — reason-code captures it
            logger.warning(
                "solve_ivp raised on traj %d: %s: %s",
                i,
                type(exc).__name__,
                exc,
            )
            return None, ScoreReasonCode.NUMERICAL_DIVERGED.value

        if not sol.success:
            return None, ScoreReasonCode.NUMERICAL_DIVERGED.value
        y = sol.y.T
        # Early termination (blowup event fired) → sol.status == 1 AND/OR
        # y.shape[0] < len(t). Treat as blowup rather than diverged.
        if getattr(sol, "status", 0) == 1 or y.shape[0] != len(t):
            return None, ScoreReasonCode.NUMERICAL_BLOWUP.value
        if not np.all(np.isfinite(y)):
            return None, ScoreReasonCode.NUMERICAL_BLOWUP.value
        if np.any((y.max(axis=0) - y.min(axis=0)) > allowed_span):
            return None, ScoreReasonCode.NUMERICAL_BLOWUP.value
        predicted[i] = y

    rmse = float(np.sqrt(((predicted - held_clean) ** 2).mean()))
    return rmse, ScoreReasonCode.OK.value


# -----------------------------------------------------------------------------
# Sympy → scipy callable bridge
# -----------------------------------------------------------------------------


def _sympy_to_callable(
    exprs: list[sympy.Expr],
    vars_: list[sympy.Symbol],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Compile a list of sympy Expr into a scipy.solve_ivp-compatible f(t, y).

    Plain English: turns the symbolic equation list into a Python function
    that scipy's ODE integrator can call with numbers.
    """
    funcs = [sympy.lambdify(vars_, e, "numpy") for e in exprs]

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([float(f(*y)) for f in funcs])

    return rhs


# -----------------------------------------------------------------------------
# Ground-truth RHS lookup (sympy form)
# -----------------------------------------------------------------------------


def _truth_rhs_sympy(system: str) -> list[sympy.Expr]:
    """Ground-truth ODE RHS as a list of sympy Expr, one per state variable.

    Parameters match `pysindy.utils.odes.py` defaults — re-verified on
    2026-04-22 (Assumption A1 in RESEARCH §Assumptions Log). The
    `test_truth_rhs_sympy_matches_pysindy_numerically_*` tests regression-guard
    this table against silent parameter drift.

    Plain English: the ground-truth equations, translated into sympy so the
    symbolic-match tier has something to compare candidate equations against.
    """
    x, y, z = symbols("x y z")
    if system == "lotka_volterra":
        # pysindy.utils.lotka p=[1, 10]:
        #   dx = p[0]*x - p[1]*x*y     = 1*x - 10*x*y
        #   dy = p[1]*x*y - 2*p[0]*y   = 10*x*y - 2*y
        return [x - 10 * x * y, 10 * x * y - 2 * y]
    if system == "van_der_pol":
        # pysindy.utils.van_der_pol p=[0.5]:
        #   dx = y
        #   dy = p[0]*(1 - x**2)*y - x = 0.5*(1 - x**2)*y - x
        return [y, Rational(1, 2) * (1 - x**2) * y - x]
    if system == "lorenz":
        # pysindy.utils.lorenz sigma=10, beta=2.66667, rho=28:
        #   dx = sigma*(y - x)           = 10*(y - x)
        #   dy = x*(rho - z) - y         = x*(28 - z) - y
        #   dz = x*y - beta*z            = x*y - 2.66667*z
        # NOTE: pysindy uses the float literal 2.66667 (not 8/3). The
        # difference is ~3e-6 numerically; we match the literal so
        # numerical regression tests pass to rtol=1e-4.
        beta = nsimplify(2.66667, rational=True)
        return [10 * (y - x), x * (28 - z) - y, x * y - beta * z]
    raise ValueError(f"_truth_rhs_sympy: unknown system {system!r}")


# -----------------------------------------------------------------------------
# Qualitative tier — per-system feature dispatcher + helpers
# -----------------------------------------------------------------------------
#
# Features per system come from RESEARCH §Architecture Patterns Pattern 4:
#
#   Lotka-Volterra:
#     - fixed_point_match  : non-trivial FP within 10% of truth (γ/δ, α/β).
#     - conserved_quantity_drift : std(H)/|mean(H)| of
#         H = δx - γ ln(x) + βy - α ln(y) along one held-out trajectory.
#
#   Van der Pol:
#     - has_limit_cycle           : autocorrelation peak / zero-crossing count
#                                   in the last 70% of a long trajectory.
#     - limit_cycle_period_match  : measured period within 10% of μ=0.5 truth
#                                   (≈ 7.63 time units).
#
#   Lorenz:
#     - attractor_box_match       : per-axis min/max of post-transient
#                                   proposed trajectory within 20% of truth.
#     - lyapunov_sign_match       : sign of finite-time Lyapunov estimate
#                                   from a trajectory pair 1e-6 apart.
#     - fixed_point_count_match   : 3 real fixed points (origin + 2 lobes).


def _qualitative_match(
    proposed_expr: list[sympy.Expr],
    system: str,
    held_out: TrajectoryBundle,
) -> tuple[dict[str, bool | float], str]:
    """Per-system qualitative-tier feature extractor.

    Plain English: asks the high-level "did it get the overall shape right?"
    questions that RMSE alone doesn't capture — does the predicted system have
    the right kind of fixed point, the right kind of oscillation, the right
    kind of chaotic attractor box?

    Returns (feature_dict, reason_code). Reason 'ok' means the extractor
    produced a dict (whose values may themselves report individual feature
    failures as False). Reason 'qualitative_unknown' means the extractor
    itself raised — feature dict may be empty.
    """
    try:
        if system == "lotka_volterra":
            return _lv_features(proposed_expr, held_out), ScoreReasonCode.OK.value
        if system == "van_der_pol":
            return _vdp_features(proposed_expr, held_out), ScoreReasonCode.OK.value
        if system == "lorenz":
            return _lorenz_features(proposed_expr, held_out), ScoreReasonCode.OK.value
        # Unknown system — bubble up as QUALITATIVE_UNKNOWN rather than crashing.
        logger.warning("_qualitative_match: unknown system %r", system)
        return {}, ScoreReasonCode.QUALITATIVE_UNKNOWN.value
    except Exception as exc:  # noqa: BLE001 — reason-code captures it
        logger.warning(
            "qualitative feature extraction failed for %s: %s: %s",
            system,
            type(exc).__name__,
            exc,
        )
        return {}, ScoreReasonCode.QUALITATIVE_UNKNOWN.value


def _safe_integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    ic: np.ndarray,
    t_eval: np.ndarray,
    *,
    blowup_cap: float = 1e6,
):
    """solve_ivp wrapper with a terminal blowup event.

    Same pattern as _numerical_rmse — keeps qualitative-tier feature
    extractors from hanging on a pathological proposed RHS.
    """

    def _evt(_t: float, y: np.ndarray) -> float:
        return blowup_cap - float(np.abs(y).max())

    _evt.terminal = True  # type: ignore[attr-defined]
    _evt.direction = -1.0  # type: ignore[attr-defined]

    return solve_ivp(rhs, t_span, ic, t_eval=t_eval, events=_evt, **INTEGRATOR_KEYWORDS)


def _lv_features(
    proposed_expr: list[sympy.Expr],
    held_out: TrajectoryBundle,
) -> dict[str, bool | float]:
    """Lotka-Volterra: fixed-point match + conserved-quantity drift.

    Non-trivial fixed point of dx/dt = α x - β x y, dy/dt = δ x y - γ y is
    (γ/δ, α/β). For pysindy defaults p=[1, 10] (dx = x - 10 x y ; dy = 10 x y - 2 y),
    α=1, β=10, δ=10, γ=2 → FP = (2/10, 1/10) = (0.2, 0.1).

    Plain English: "fixed point" is the population level where neither predator
    nor prey grows or shrinks — the system can sit there forever. The conserved
    quantity H is an invariant energy-like number that stays constant along a
    Lotka-Volterra orbit; if it drifts, the candidate equation is wrong.
    """
    x_sym, y_sym = symbols("x y")
    # Symbolic fixed-point solve on proposed RHS (set each equation = 0).
    # alpha-10: also surface the proposed FP coordinates (self-derived from
    # the agent's own coefficients — zero leak) so the agent can see WHERE
    # its FP sits vs its own hints.mean. NaN sentinel when no non-trivial
    # co-existence FP exists (degenerate or parse-empty form).
    fp_match: bool = False
    fp_proposed_x: float = float("nan")
    fp_proposed_y: float = float("nan")
    if len(proposed_expr) == 2:
        try:
            sols = sympy.solve(proposed_expr, [x_sym, y_sym], dict=True)
            candidates: list[tuple[float, float]] = []
            for s in sols:
                xv = s.get(x_sym)
                yv = s.get(y_sym)
                if xv is None or yv is None:
                    continue
                # Require real and positive (non-trivial co-existence FP).
                if not (xv.is_real and yv.is_real):
                    continue
                try:
                    xf = float(xv)
                    yf = float(yv)
                except (TypeError, ValueError):
                    continue
                if xf > 1e-6 and yf > 1e-6:
                    candidates.append((xf, yf))
            # pysindy p=[1,10]: α=1 β=10 δ=10 γ=2·1=2 → FP = (γ/δ, α/β) = (0.2, 0.1)
            truth_fp = (0.2, 0.1)
            if candidates:
                fp_prop = candidates[0]
                fp_proposed_x = float(fp_prop[0])
                fp_proposed_y = float(fp_prop[1])
                fp_match = (
                    abs(fp_prop[0] - truth_fp[0]) / truth_fp[0] < 0.10
                    and abs(fp_prop[1] - truth_fp[1]) / truth_fp[1] < 0.10
                )
        except Exception as exc:  # noqa: BLE001 — feature extractor
            logger.warning("LV fixed-point solve raised: %s: %s", type(exc).__name__, exc)

    # Conserved quantity H along one held-out clean trajectory.
    # LEAK BOUNDARY: H uses truth's hard-coded (α, β, γ, δ). `drift` is
    # informative but derived from truth's specific Hamiltonian; callers
    # that surface qualitative features to the agent prompt MUST NOT
    # include conserved_quantity_drift. `fixed_point_match` and
    # `fp_proposed_{x,y}` are safe (boolean + self-derived coordinates).
    held_mask = ~held_out.train_mask
    drift: float
    if not held_mask.any():
        drift = float("nan")
    else:
        traj = held_out.clean[held_mask][0]  # shape (T, 2)
        # α=1 β=10 δ=10 γ=2 : H = δx - γ ln(x) + βy - α ln(y) = 10x - 2ln(x) + 10y - ln(y)
        alpha, beta, delta, gamma = 1.0, 10.0, 10.0, 2.0
        xs = np.clip(traj[:, 0], 1e-8, None)
        ys = np.clip(traj[:, 1], 1e-8, None)
        H = delta * xs - gamma * np.log(xs) + beta * ys - alpha * np.log(ys)
        drift = float(np.std(H) / max(abs(float(np.mean(H))), 1e-8))

    return {
        "fixed_point_match": bool(fp_match),
        "fp_proposed_x": fp_proposed_x,
        "fp_proposed_y": fp_proposed_y,
        "conserved_quantity_drift": drift,
    }


def _vdp_features(
    proposed_expr: list[sympy.Expr],
    held_out: TrajectoryBundle,
) -> dict[str, bool | float]:
    """Van der Pol: has_limit_cycle + limit_cycle_period_match.

    Plain English: a "limit cycle" is when the solution trajectory settles into
    repeating the same closed loop forever, like a stable orbit. For Van der
    Pol with μ=0.5 the loop has period ≈ 7.63 time units.
    """
    held_mask = ~held_out.train_mask
    if not held_mask.any() or len(proposed_expr) != 2:
        return {"has_limit_cycle": False, "limit_cycle_period_match": False}
    ic = held_out.ics[held_mask][0]

    # Integrate a long trajectory past transient so we can measure period.
    t_long = np.linspace(0.0, 50.0, 5000)
    try:
        rhs = _sympy_to_callable(proposed_expr, _vars_for("van_der_pol"))
        sol = _safe_integrate(rhs, (0.0, 50.0), np.asarray(ic, dtype=float), t_long)
    except Exception as exc:  # noqa: BLE001
        logger.warning("VdP integration raised: %s: %s", type(exc).__name__, exc)
        return {"has_limit_cycle": False, "limit_cycle_period_match": False}

    if not sol.success or sol.y.shape[1] != len(t_long):
        return {"has_limit_cycle": False, "limit_cycle_period_match": False}
    x_traj = sol.y[0]
    if not np.all(np.isfinite(x_traj)):
        return {"has_limit_cycle": False, "limit_cycle_period_match": False}

    # Use last 70% to skip transient.
    start = int(0.3 * len(x_traj))
    tail = x_traj[start:]
    tail_t = t_long[start:]

    # Zero-crossings in tail → period = 2 * (elapsed / crossings).
    signs = np.sign(tail)
    crossings_mask = np.abs(np.diff(signs)) > 0
    crossings = int(np.sum(crossings_mask))
    has_cycle = crossings >= 4  # at least ~2 periods in tail window

    truth_period = 7.63  # VdP μ=0.5 canonical period
    # WR-02: guard against tail_t length < 2. With current constants
    # (t_long hardcoded to 5000 points, start = int(0.3 * 5000) = 1500),
    # tail_t always has ~3500 entries — but the guard is future-proof
    # against anyone reducing the sample count or overriding the
    # integration window. Also covers the edge case where start ends up
    # indexing to the last point.
    if crossings > 0 and len(tail_t) >= 2:
        period_prop = 2.0 * (tail_t[-1] - tail_t[0]) / crossings
        period_match = abs(period_prop - truth_period) / truth_period < 0.20
    else:
        period_match = False

    return {
        "has_limit_cycle": bool(has_cycle),
        "limit_cycle_period_match": bool(period_match),
    }


def _lorenz_features(
    proposed_expr: list[sympy.Expr],
    held_out: TrajectoryBundle,
) -> dict[str, bool | float]:
    """Lorenz: attractor_box_match + lyapunov_sign_match + fixed_point_count_match.

    Plain English: Lorenz is chaotic — trajectories never repeat exactly but
    stay inside a butterfly-shaped attractor. We check (a) does the proposed
    system stay inside roughly the same box, (b) does it expand small
    perturbations like a chaotic system does (positive Lyapunov sign), and
    (c) does it have the same number of "equilibrium" points (3 for truth).
    """
    held_mask = ~held_out.train_mask
    if not held_mask.any() or len(proposed_expr) != 3:
        return {
            "attractor_box_match": False,
            "lyapunov_sign_match": False,
            "fixed_point_count_match": False,
        }
    ic = np.asarray(held_out.ics[held_mask][0], dtype=float)

    # Re-integrate proposed for 5 time units past transient.
    t_long = np.linspace(0.0, 5.0, 5000)
    try:
        rhs = _sympy_to_callable(proposed_expr, _vars_for("lorenz"))
        sol_p = _safe_integrate(rhs, (0.0, 5.0), ic, t_long)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lorenz integration raised: %s: %s", type(exc).__name__, exc)
        return {
            "attractor_box_match": False,
            "lyapunov_sign_match": False,
            "fixed_point_count_match": False,
        }

    if not sol_p.success or sol_p.y.shape[1] != len(t_long) or not np.all(np.isfinite(sol_p.y)):
        return {
            "attractor_box_match": False,
            "lyapunov_sign_match": False,
            "fixed_point_count_match": False,
        }

    # --- Attractor bounding box (skip first 20% as transient) ---
    post_transient = sol_p.y[:, int(0.2 * len(t_long)) :]
    prop_min = post_transient.min(axis=1)
    prop_max = post_transient.max(axis=1)
    # Truth Lorenz post-transient bounds (σ=10, β=2.66667, ρ=28, ic=[-8,8,27]).
    truth_min = np.array([-19.0, -27.0, 0.8])
    truth_max = np.array([19.0, 27.0, 48.0])
    box_match = all(
        abs(prop_min[i] - truth_min[i]) / max(abs(truth_min[i]), 1e-3) < 0.20
        and abs(prop_max[i] - truth_max[i]) / max(abs(truth_max[i]), 1e-3) < 0.20
        for i in range(3)
    )

    # --- Lyapunov sign via trajectory-pair divergence ---
    ic2 = ic + np.array([1e-6, 0.0, 0.0])
    try:
        sol_p2 = _safe_integrate(rhs, (0.0, 5.0), ic2, t_long)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lorenz pair integration raised: %s: %s", type(exc).__name__, exc)
        lyap_sign_match = False
    else:
        if (
            not sol_p2.success
            or sol_p2.y.shape[1] != len(t_long)
            or not np.all(np.isfinite(sol_p2.y))
        ):
            lyap_sign_match = False
        else:
            delta0 = float(np.linalg.norm(sol_p.y[:, 0] - sol_p2.y[:, 0]))
            delta_f = float(np.linalg.norm(sol_p.y[:, -1] - sol_p2.y[:, -1]))
            if delta0 == 0 or delta_f == 0:
                lyap_sign_match = False
            else:
                lyap_est = float(np.log(delta_f / delta0)) / float(t_long[-1])
                # Truth Lorenz Lyapunov ≈ +0.9 (chaotic). Match on sign only.
                lyap_sign_match = lyap_est > 0

    # --- Fixed-point count — sympy.solve on proposed RHS = 0, count real solutions ---
    x_sym, y_sym, z_sym = symbols("x y z")
    try:
        sols = sympy.solve(proposed_expr, [x_sym, y_sym, z_sym], dict=True)
        real_count = 0
        for s in sols:
            vals = (s.get(x_sym), s.get(y_sym), s.get(z_sym))
            if any(v is None for v in vals):
                continue
            if all(v.is_real for v in vals):
                real_count += 1
        fp_count_match = real_count == 3
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lorenz fixed-point solve raised: %s: %s", type(exc).__name__, exc)
        fp_count_match = False

    return {
        "attractor_box_match": bool(box_match),
        "lyapunov_sign_match": bool(lyap_sign_match),
        "fixed_point_count_match": bool(fp_count_match),
    }


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def score(
    proposed_rhs: list[sympy.Expr] | list[str],
    truth_spec: BenchmarkSpec,
    held_out: TrajectoryBundle,
    *,
    rmse_horizon: float | None = None,
    blowup_box_multiplier: float = 10.0,
    symbolic_timeout_s: float = 5.0,
    symbolic_coeff_tolerance: float = 1e-3,
) -> BenchmarkScore:
    """Three-tier score. See module docstring for priority rules per tier.

    Plain English: run the three checks (does the equation match
    symbolically, do its predictions match numerically, does the overall
    shape match), wrap them in a BenchmarkScore DTO with per-tier timing
    and reason codes.
    """
    vars_ = _vars_for(truth_spec.system)
    reasons: dict[str, str] = {}
    elapsed_ms: dict[str, int] = {}

    # --- Symbolic tier --------------------------------------------------
    sym_start = time.monotonic()
    proposed_expr: list[sympy.Expr] = []
    sym_reason = ScoreReasonCode.OK.value
    exact = False
    try:
        if proposed_rhs and isinstance(proposed_rhs[0], str):
            # CR-01: route strings through _safe_parse_expr (parse_expr with
            # empty __builtins__) rather than sympify. sympify calls eval()
            # and does NOT block __import__/__builtins__ access even with
            # locals= restriction — an agent string like
            # `__import__("os").system("id")` would execute during parsing.
            # parse_expr uses AST transformations and the empty __builtins__
            # means those names resolve to nothing → NameError → caught here.
            #
            # Alpha-11 (L-013 fix): the `implicit_multiplication_application`
            # transformation collapses `x0` → `x*0` = 0 and `x1` → `x*1` = x
            # when `x0`/`x1` are NOT pre-registered in local_dict. Every
            # alpha-1..alpha-10 live run scored a DEGENERATE form because
            # agents propose in x0,x1 per the V1..V7 prompt contract while
            # the scorer's locals_ only registered x,y. The alias mapping
            # below makes `x0` → vars_[0], `x1` → vars_[1], etc., so the
            # parser resolves the name before implicit-multiplication fires.
            # Keeps the canonical feature-name bindings intact (x,y for LV
            # so Poly canonicalization still compares correctly).
            locals_ = {str(v): v for v in vars_}
            for i, v in enumerate(vars_):
                locals_[f"x{i}"] = v
            proposed_expr = [_safe_parse_expr(s, local_dict=locals_) for s in proposed_rhs]
        else:
            proposed_expr = list(proposed_rhs)  # type: ignore[arg-type]
        truth_expr = _truth_rhs_sympy(truth_spec.system)
        exact, sym_reason = _symbolic_match(
            proposed_expr,
            truth_expr,
            vars_,
            timeout_s=symbolic_timeout_s,
            coeff_tol=symbolic_coeff_tolerance,
        )
    except (
        sympy.SympifyError,
        SyntaxError,
        TypeError,
        ValueError,
        NameError,
        AttributeError,
        tokenize.TokenError,
    ) as exc:
        # NameError / AttributeError: blocked-builtin resolution (e.g.
        #   `__import__` becomes a sympy Symbol under our locked global_dict
        #   and `.system(...)` on a Symbol raises AttributeError).
        # ValueError / SyntaxError / TokenError: parse_expr on malformed
        #   syntax (e.g. unclosed bracket, tokenizer EOF).
        # TypeError: operator mis-applied (e.g. Symbol * str).
        logger.warning(
            "parse_expr raised on proposed_rhs: %s: %s",
            type(exc).__name__,
            exc,
        )
        exact = False
        sym_reason = ScoreReasonCode.SYMBOLIC_PARSE_FAIL.value
        proposed_expr = []
    reasons["symbolic"] = sym_reason
    elapsed_ms["symbolic"] = int((time.monotonic() - sym_start) * 1000)

    # --- Numerical tier -------------------------------------------------
    num_start = time.monotonic()
    # WR-01: dict.get() without a default returns None silently for missing
    # keys. If a future phase adds a new system to DEFAULT_PARAMS but forgets
    # to add an entry in _DEFAULT_HORIZONS, the scorer would silently use
    # horizon=None (full t-span). For a chaotic system that produces
    # meaningless RMSE. Use a sentinel + explicit warning so the drop is
    # loud (CLAUDE.md no-silent-drops).
    _MISSING = object()
    _system_horizon = _DEFAULT_HORIZONS.get(truth_spec.system, _MISSING)
    if _system_horizon is _MISSING:
        logger.warning(
            "system %r not in _DEFAULT_HORIZONS; defaulting to full t_span. "
            "Add an entry to _DEFAULT_HORIZONS if this system is chaotic.",
            truth_spec.system,
        )
        _system_horizon = None
    horizon = rmse_horizon if rmse_horizon is not None else _system_horizon
    if proposed_expr:
        try:
            proposed_callable = _sympy_to_callable(proposed_expr, vars_)
            rmse, num_reason = _numerical_rmse(
                proposed_callable,
                held_out,
                horizon=horizon,
                box_mult=blowup_box_multiplier,
            )
        except Exception as exc:  # noqa: BLE001 — reason-code captures it
            logger.warning(
                "_numerical_rmse raised unexpectedly: %s: %s",
                type(exc).__name__,
                exc,
            )
            rmse, num_reason = None, ScoreReasonCode.NUMERICAL_DIVERGED.value
    else:
        # No parseable RHS → numerical tier can't run; surface as diverged
        # so the invariant `rmse is None iff reasons['numerical'] != 'ok'` holds.
        rmse, num_reason = None, ScoreReasonCode.NUMERICAL_DIVERGED.value
    reasons["numerical"] = num_reason
    elapsed_ms["numerical"] = int((time.monotonic() - num_start) * 1000)

    # --- Qualitative tier (Task 3 fills the stub) -----------------------
    # WR-04: when symbolic parse failed (proposed_expr empty), feature
    # extractors that pull signal from held_out alone — e.g., conserved-
    # quantity drift in LV — would return a real-looking number even
    # though no equation was proposed. That masquerades as a valid
    # qualitative measurement when it's actually ground-truth-only data.
    # Short-circuit: if nothing parsed, emit an empty qualitative dict
    # and mark the reason so downstream consumers see the skip.
    qual_start = time.monotonic()
    qualitative: dict[str, bool | float] = {}
    if not proposed_expr:
        qual_reason = ScoreReasonCode.QUALITATIVE_UNKNOWN.value
    else:
        qual_reason = ScoreReasonCode.OK.value
        try:
            qualitative, qual_reason = _qualitative_match(
                proposed_expr,
                truth_spec.system,
                held_out,
            )
        except Exception as exc:  # noqa: BLE001 — reason-code captures it
            logger.warning(
                "qualitative match failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            qualitative = {}
            qual_reason = ScoreReasonCode.QUALITATIVE_UNKNOWN.value
    reasons["qualitative"] = qual_reason
    elapsed_ms["qualitative"] = int((time.monotonic() - qual_start) * 1000)

    # --- Invariant enforcement (fail loud, never silent) ----------------
    if rmse is None and num_reason == ScoreReasonCode.OK.value:
        raise AssertionError(
            "invariant violation: rmse=None with numerical reason 'ok' "
            f"(system={truth_spec.system!r})",
        )
    if rmse is not None and num_reason != ScoreReasonCode.OK.value:
        raise AssertionError(
            f"invariant violation: rmse={rmse} with numerical reason {num_reason!r} "
            f"(system={truth_spec.system!r})",
        )

    return BenchmarkScore(
        exact=exact,
        rmse=rmse,
        qualitative=qualitative,
        reasons=reasons,
        elapsed_ms=elapsed_ms,
    )


# Public re-exports for convenience. Private helpers (``_foo``) are
# intentionally NOT listed here — they are imported explicitly by name
# (e.g. ``from ascension.benchmarks.scoring import _truth_rhs_sympy``),
# which bypasses ``__all__``. Keeping ``__all__`` to public-only makes
# ``from ascension.benchmarks.scoring import *`` give only the supported
# surface and signals intent (IN-01).
__all__ = ["score"]
