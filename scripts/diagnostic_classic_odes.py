"""STRAT-04 — run the Ascension identifiability diagnostic on three classic ODE
systems (Lotka-Volterra, Van der Pol, Lorenz) and tabulate against the
literature-known identifiability answers.

Why this exists (the credibility argument):
  A diagnostic that only ever runs on the authors' invented "alien" physics is
  weak evidence. The SAME domain-agnostic instrument
  (`practical_identifiability`, the FIM/sensitivity checker, and
  `structural_identifiability`, the exact symbolic checker) is run here on
  textbook systems whose identifiability answers are settled in the
  structural-ID literature (SIAN / StructuralIdentifiability.jl / DAISY-class
  analyses). Reproducing the KNOWN answers — and honestly flagging the one
  place a LOCAL FIM check cannot see a GLOBAL structural fact — is what makes
  the instrument defensible.

Determinism + safety:
  * NO LLM, NO network. Pure numpy / scipy / sympy.
  * Every trajectory is integrated with scipy LSODA at rtol=atol=1e-12 (the
    same tolerance the benchmark registry pins for Lorenz chaos, systems.py
    INTEGRATOR_KEYWORDS), on a FIXED time grid so the observation design is a
    fixed-length vector (the diagnostic requires predict() to return a
    fixed-length flat vector — identifiability.sensitivity_matrix).

A KNOWN parameterization caveat (verified 2026-06-02, observation #2884):
  pysindy's `lotka(t, x, p=[1, 10])` exposes only TWO free parameters and ties
  the predator-gain to the prey-loss (p[1]) and the predator-death to 2*p[0].
  The famous Lotka-Volterra partial-identifiability result is a property of the
  FULL FOUR-parameter model (alpha, beta, delta, gamma). So this script runs
  BOTH:
    (a) the pysindy-anchored RHS exactly as the repo registry defines it
        (honest to systems.py — labeled "pysindy 2-param reduction"), AND
    (b) a self-contained FULL 4-parameter LV RHS (alpha, beta, delta, gamma),
        which is where the textbook degeneracy actually lives and where the
        literature comparison is meaningful.
  Reporting both is the honest thing; claiming the 2-param form reproduces the
  4-param literature answer would be the overstatement #2884 caught.

Run:  poetry run python scripts/diagnostic_classic_odes.py
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import sympy
from scipy.integrate import solve_ivp

from ascension.benchmarks.systems import DEFAULT_PARAMS, INTEGRATOR_KEYWORDS
from ascension.diagnostics.identifiability import (
    IdentifiabilityVerdict,
    practical_identifiability,
)
from ascension.diagnostics.structural import (
    StructuralVerdict,
    structural_identifiability,
)

# ---------------------------------------------------------------------------
# Trajectory closures.  Each builds a predict(theta) -> flat observation vector
# under a FIXED observation design (which state coordinates, over a fixed time
# grid).  The diagnostic linearizes at the nominal theta via central differences,
# so the only requirement is a smooth, fixed-length predict().
# ---------------------------------------------------------------------------

RhsBuilder = Callable[[np.ndarray], Callable[[float, np.ndarray], Sequence[float]]]


def _integrate(
    rhs: Callable[[float, np.ndarray], Sequence[float]],
    ic: Sequence[float],
    t_grid: np.ndarray,
) -> np.ndarray:
    """Integrate `rhs` from `ic` and sample the solution at `t_grid`.

    Uses the SAME stiff/non-stiff LSODA + 1e-12 tolerance the benchmark registry
    pins (systems.py INTEGRATOR_KEYWORDS) so the sensitivity finite-differences
    are not polluted by integrator error masquerading as parameter sensitivity.
    Returns an (n_states, n_times) array.
    """
    sol = solve_ivp(
        rhs,
        (float(t_grid[0]), float(t_grid[-1])),
        np.asarray(ic, dtype=np.float64),
        t_eval=t_grid,
        method=INTEGRATOR_KEYWORDS["method"],
        rtol=INTEGRATOR_KEYWORDS["rtol"],
        atol=INTEGRATOR_KEYWORDS["atol"],
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.y  # (n_states, n_times)


def make_predict(
    rhs_builder: RhsBuilder,
    ic: Sequence[float],
    t_grid: np.ndarray,
    observed_states: Sequence[int],
) -> Callable[[np.ndarray], np.ndarray]:
    """Return predict(theta) -> flat observations for the given observation design.

    rhs_builder(theta) returns the parameterized RHS; observed_states selects which
    state coordinates the design measures (e.g. [0] = prey-only, [0, 1] = full).
    """
    obs = list(observed_states)

    def predict(theta: np.ndarray) -> np.ndarray:
        rhs = rhs_builder(np.asarray(theta, dtype=np.float64))
        y = _integrate(rhs, ic, t_grid)  # (n_states, n_times)
        return y[obs, :].ravel()

    return predict


# ---- RHS builders ----------------------------------------------------------


def _lotka_pysindy_builder(theta: np.ndarray):
    """pysindy 2-param LV exactly as the registry uses it: lotka(t, x, p=theta)."""
    rhs_fn = DEFAULT_PARAMS["lotka_volterra"]["rhs"]

    def rhs(t: float, x: np.ndarray):
        return rhs_fn(t, x, list(theta))

    return rhs


def _lotka_full4_builder(theta: np.ndarray):
    """FULL 4-parameter Lotka-Volterra (the textbook form, where the known
    identifiability results live):

        dx/dt = alpha*x - beta*x*y       (prey: growth - predation)
        dy/dt = delta*x*y - gamma*y      (predator: gain from predation - death)

    theta = (alpha, beta, delta, gamma).
    """
    alpha, beta, delta, gamma = (float(v) for v in theta)

    def rhs(t: float, z: np.ndarray):
        x, y = z
        return [alpha * x - beta * x * y, delta * x * y - gamma * y]

    return rhs


def _van_der_pol_builder(theta: np.ndarray):
    """pysindy Van der Pol: van_der_pol(t, x, p=[mu]). theta = (mu,)."""
    rhs_fn = DEFAULT_PARAMS["van_der_pol"]["rhs"]

    def rhs(t: float, x: np.ndarray):
        return rhs_fn(t, x, list(theta))

    return rhs


def _lorenz_builder(theta: np.ndarray):
    """pysindy Lorenz: lorenz(t, x, sigma, beta, rho). theta = (sigma, beta, rho)."""
    rhs_fn = DEFAULT_PARAMS["lorenz"]["rhs"]
    sigma, beta, rho = (float(v) for v in theta)

    def rhs(t: float, x: np.ndarray):
        return rhs_fn(t, x, sigma, beta, rho)

    return rhs


# ---------------------------------------------------------------------------
# Case table.  Each case is one (system, observation-design) row.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    system: str
    design: str
    rhs_builder: RhsBuilder
    ic: Sequence[float]
    t_grid: np.ndarray
    observed_states: Sequence[int]
    theta: Sequence[float]
    param_names: Sequence[str]
    # Optional sympy structural model: (expr, params, observables). Only set
    # where a closed algebraic prediction form makes structural ID meaningful.
    structural: tuple[sympy.Expr, list[str], list[str]] | None = None


def _build_cases() -> list[Case]:
    # Time grids: dense enough to expose the dynamics, modest length to keep the
    # finite-difference SVD well-posed. LV/VdP windows match the registry.
    lv_t = np.linspace(0.0, 20.0, 200)
    vdp_t = np.linspace(0.0, 20.0, 200)
    # Lorenz: keep the window short. Chaos makes long-horizon sensitivities blow
    # up (the diagnostic is a LOCAL linearization; an exponentially diverging
    # trajectory makes finite differences unreliable past a few Lyapunov times).
    lorenz_t = np.linspace(0.0, 3.0, 150)

    lv_reg = DEFAULT_PARAMS["lotka_volterra"]
    vdp_reg = DEFAULT_PARAMS["van_der_pol"]
    lorenz_reg = DEFAULT_PARAMS["lorenz"]

    cases: list[Case] = []

    # ---- Lotka-Volterra: pysindy 2-param reduction --------------------------
    cases.append(
        Case(
            system="LV (pysindy 2-param)",
            design="full-state (x,y)",
            rhs_builder=_lotka_pysindy_builder,
            ic=lv_reg["default_ic"],
            t_grid=lv_t,
            observed_states=[0, 1],
            theta=[1.0, 10.0],
            param_names=["p0", "p1"],
        )
    )
    cases.append(
        Case(
            system="LV (pysindy 2-param)",
            design="prey-only (x)",
            rhs_builder=_lotka_pysindy_builder,
            ic=lv_reg["default_ic"],
            t_grid=lv_t,
            observed_states=[0],
            theta=[1.0, 10.0],
            param_names=["p0", "p1"],
        )
    )

    # ---- Lotka-Volterra: FULL 4-param (textbook form) -----------------------
    # Nominal rates chosen to give a clean multi-cycle limit oscillation in the
    # window so every parameter is exercised.
    lv4_theta = [1.0, 1.0, 1.0, 1.0]  # alpha, beta, delta, gamma
    lv4_ic = [1.0, 0.5]
    lv4_names = ["alpha", "beta", "delta", "gamma"]
    cases.append(
        Case(
            system="LV (full 4-param)",
            design="full-state (x,y)",
            rhs_builder=_lotka_full4_builder,
            ic=lv4_ic,
            t_grid=lv_t,
            observed_states=[0, 1],
            theta=lv4_theta,
            param_names=lv4_names,
        )
    )
    cases.append(
        Case(
            system="LV (full 4-param)",
            design="prey-only (x)",
            rhs_builder=_lotka_full4_builder,
            ic=lv4_ic,
            t_grid=lv_t,
            observed_states=[0],
            theta=lv4_theta,
            param_names=lv4_names,
        )
    )

    # ---- Van der Pol --------------------------------------------------------
    cases.append(
        Case(
            system="Van der Pol",
            design="full-state (x,y)",
            rhs_builder=_van_der_pol_builder,
            ic=vdp_reg["default_ic"],
            t_grid=vdp_t,
            observed_states=[0, 1],
            theta=[0.5],
            param_names=["mu"],
        )
    )
    cases.append(
        Case(
            system="Van der Pol",
            design="position-only (x)",
            rhs_builder=_van_der_pol_builder,
            ic=vdp_reg["default_ic"],
            t_grid=vdp_t,
            observed_states=[0],
            theta=[0.5],
            param_names=["mu"],
        )
    )

    # ---- Lorenz -------------------------------------------------------------
    lorenz_theta = [10.0, 8.0 / 3.0, 28.0]  # sigma, beta, rho
    lorenz_names = ["sigma", "beta", "rho"]
    cases.append(
        Case(
            system="Lorenz",
            design="full-state (x,y,z)",
            rhs_builder=_lorenz_builder,
            ic=lorenz_reg["default_ic"],
            t_grid=lorenz_t,
            observed_states=[0, 1, 2],
            theta=lorenz_theta,
            param_names=lorenz_names,
        )
    )
    cases.append(
        Case(
            system="Lorenz",
            design="x-only (x)",
            rhs_builder=_lorenz_builder,
            ic=lorenz_reg["default_ic"],
            t_grid=lorenz_t,
            observed_states=[0],
            theta=lorenz_theta,
            param_names=lorenz_names,
        )
    )

    return cases


# ---------------------------------------------------------------------------
# Structural-ID demonstrations on the closed RHS forms (polynomial), where the
# exact symbolic checker is meaningful.  This complements the numerical local
# FIM check with an EXACT global statement about the *RHS coefficients* — the
# vector-field-level identifiability that matches what SINDy/regression recovers
# when the full state derivative is available.
# ---------------------------------------------------------------------------


def _structural_demos() -> list[tuple[str, StructuralVerdict]]:
    x, y, z = sympy.symbols("x y z")
    alpha, beta, delta, gamma, mu, sigma, rho = sympy.symbols("alpha beta delta gamma mu sigma rho")
    bsym = sympy.Symbol("b")  # lorenz beta (avoid clash with LV beta naming in print)

    demos: list[tuple[str, StructuralVerdict]] = []

    # Full 4-param LV, full state observed: treat the two RHS components as the
    # predicted quantities. Concatenate them so a single expression carries every
    # coefficient with a distinct monomial. dx/dt has {alpha*x, -beta*x*y};
    # dy/dt has {delta*x*y, -gamma*y}. Each parameter multiplies a DISTINCT
    # monomial in (x,y) => all four separately identifiable.
    lv_pred = (alpha * x - beta * x * y) + sympy.Symbol("q") * (delta * x * y - gamma * y)
    # The dummy symbol q tags the second equation so its monomials cannot cancel
    # against the first; it is treated as an observable (a design index).
    demos.append(
        (
            "LV full 4-param, full-state RHS coefficients",
            structural_identifiability(
                lv_pred,
                [alpha, beta, delta, gamma],
                [x, y, sympy.Symbol("q")],
            ),
        )
    )

    # Van der Pol full state: dy/dt = mu*(1 - x^2)*y - x. Only mu is a parameter
    # and it multiplies a non-vanishing monomial => identifiable.
    vdp_pred = mu * (1 - x**2) * y - x
    demos.append(
        (
            "Van der Pol, full-state RHS coefficient",
            structural_identifiability(vdp_pred, [mu], [x, y]),
        )
    )

    # Lorenz full state: pack all three equations with distinct design tags so
    # every coefficient sits on its own monomial.
    q1, q2 = sympy.symbols("q1 q2")
    lorenz_pred = (sigma * (y - x)) + q1 * (x * (rho - z) - y) + q2 * (x * y - bsym * z)
    demos.append(
        (
            "Lorenz, full-state RHS coefficients",
            structural_identifiability(
                lorenz_pred,
                [sigma, rho, bsym],
                [x, y, z, q1, q2],
            ),
        )
    )

    return demos


# ---------------------------------------------------------------------------
# Runner / reporting.
# ---------------------------------------------------------------------------


def _confounded_summary(v: IdentifiabilityVerdict) -> str:
    if not v.confounded_directions:
        return "(none)"
    parts = []
    for d in v.confounded_directions:
        terms = [f"{w:+.3f}*{n}" for w, n in zip(d, v.param_names, strict=False) if abs(w) > 1e-6]
        parts.append("[" + " ".join(terms) + "]" if terms else "[degenerate]")
    return "; ".join(parts)


def run() -> None:
    print("=" * 100)
    print("STRAT-04  Identifiability diagnostic on classic ODE systems")
    print("Deterministic. No LLM. No network. (numpy / scipy LSODA 1e-12 / sympy)")
    print("=" * 100)

    cases = _build_cases()

    header = (
        f"{'system':<22} {'design':<20} {'n_p':>3} {'rank':>4} "
        f"{'ident?':>7} {'status':<30} {'cond':>11}"
    )
    print("\nPRACTICAL (FIM / sensitivity) identifiability — local at nominal theta\n")
    print(header)
    print("-" * len(header))

    results: list[tuple[Case, IdentifiabilityVerdict]] = []
    for case in cases:
        predict = make_predict(case.rhs_builder, case.ic, case.t_grid, case.observed_states)
        verdict = practical_identifiability(
            predict,
            list(case.theta),
            param_names=list(case.param_names),
        )
        results.append((case, verdict))
        cond = (
            "inf"
            if not np.isfinite(verdict.condition_number)
            else f"{verdict.condition_number:.3e}"
        )
        print(
            f"{case.system:<22} {case.design:<20} {verdict.n_params:>3} "
            f"{verdict.rank:>4} {str(verdict.identifiable):>7} "
            f"{verdict.status:<30} {cond:>11}"
        )

    # Per-case detail (singular values + confounded directions).
    print("\n" + "-" * 100)
    print("DETAIL per case (singular values, confounded directions, hint)\n")
    for case, verdict in results:
        print(f"### {case.system}  |  {case.design}")
        sv = ", ".join(f"{s:.3e}" for s in verdict.singular_values)
        print(f"    params       : {list(case.param_names)}  @ theta={list(case.theta)}")
        print(f"    rank/n       : {verdict.rank}/{verdict.n_params}   status={verdict.status}")
        cond = (
            "inf"
            if not np.isfinite(verdict.condition_number)
            else f"{verdict.condition_number:.6e}"
        )
        print(f"    cond number  : {cond}")
        print(f"    sing. values : [{sv}]")
        print(f"    confounded   : {_confounded_summary(verdict)}")
        print(f"    rationale    : {verdict.rationale}")
        print()

    # Structural demos.
    print("=" * 100)
    print("STRUCTURAL (exact symbolic) identifiability — RHS coefficients, full-state")
    print("=" * 100 + "\n")
    for label, sv in _structural_demos():
        print(f"### {label}")
        print(f"    identifiable      : {sv.identifiable}")
        print(
            f"    confounded_groups : {sv.confounded_groups if sv.confounded_groups else '(none)'}"
        )
        print(f"    absent_params     : {sv.absent_params if sv.absent_params else '(none)'}")
        print(f"    rationale         : {sv.rationale}")
        print()

    print("Done. (deterministic — re-running yields byte-identical verdicts)")


if __name__ == "__main__":
    run()
