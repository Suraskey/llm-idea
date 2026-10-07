"""ODE system registry — PySINDy-anchored defaults (SCOPE §9).

Values verified against `pysindy/utils/odes.py` on UPDATED_ON. Bumping
PYSINDY_VERSION_ANCHOR requires re-verification against the upstream file
— a six-month-later reviewer sees the anchor date and re-runs the check
(PATTERNS §`src/ascension/benchmarks/systems.py`, analog =
`src/ascension/llm/pricing.py` staleness discipline).

Plain English: this file is the single place that remembers which math
functions we use to simulate each toy physics system, what the default
knob values are, and the numerical accuracy we demand from the solver.
We do NOT hand-write the equations — we import them from PySINDy so the
"anchored to PySINDy benchmarks" requirement in SCOPE §9 is literal.

Binding decisions:
  - 06.0 RESEARCH §Pattern 1 lines 243-266: DEFAULT_PARAMS verbatim values.
  - 06.0 RESEARCH §Common Pitfalls #1: INTEGRATOR_KEYWORDS is LSODA +
    rtol=atol=1e-12, never scipy defaults.
  - 06.0 RESEARCH §Common Pitfalls #5: feature_names pinned per-system
    (x/y or x/y/z) to match the symbols a human-readable equation uses;
    consumed by Plan 03 `pysindy_baseline.py`.
  - SCOPE §9: "anchored to PySINDy benchmark suite" — we import the RHS
    functions from `pysindy.utils` rather than hand-rolling them.

Pitfalls honored:
  - #1 (scipy default tolerance wrecks Lorenz) — INTEGRATOR_KEYWORDS is
    the fix.
  - #5 (PySINDy default feature_names x0/x1/x2) — DEFAULT_PARAMS[...]
    ['feature_names'] pins x/y/z; consumed by Plan 03 pysindy_baseline.
  - #7 (noise convention) — documented on BenchmarkSpec, not here.

Threat-model mitigations: N/A — pure-compute registry module, no
network/auth/secret surface per RESEARCH §Security Domain lines 690–702.
"""

from __future__ import annotations

from datetime import date

from pysindy.utils import lorenz, lotka, van_der_pol

from ascension.benchmarks.exceptions import UnknownBenchmarkSystem

# Anchor date: bumping PYSINDY_VERSION_ANCHOR below requires re-verifying
# the constants in DEFAULT_PARAMS against upstream `pysindy/utils/odes.py`.
UPDATED_ON = date(2026, 4, 22)
PYSINDY_VERSION_ANCHOR = "2.1.0"

# PySINDy's canonical integrator keywords — LSODA auto-switches stiff/non-stiff,
# rtol/atol=1e-12 is required for Lorenz chaos (RESEARCH Pitfall #1). Scipy
# defaults (rtol=1e-3, atol=1e-6) would produce visibly wrong Lorenz
# trajectories — 9 orders of magnitude looser than what the anchor expects.
INTEGRATOR_KEYWORDS: dict[str, object] = {
    "method": "LSODA",
    "rtol": 1e-12,
    "atol": 1e-12,
}


# DEFAULT_PARAMS row schema (per system key):
#   rhs           — callable matching scipy.integrate.solve_ivp signature
#                   f(t, y, *args) -> dy/dt; imported verbatim from pysindy.utils.
#   args          — tuple of positional args passed through `solve_ivp(..., args=...)`.
#   dim           — state-space dimension (2 for LV/VdP, 3 for Lorenz).
#   default_ic    — canonical initial condition used as the centroid around
#                   which BenchmarkSpec.ic_spread perturbs.
#   t_span        — default integration window (t_start, t_end). Overridable
#                   via BenchmarkSpec.t_span_override.
#   feature_names — human-readable state variable names (Pitfall #5). Passed
#                   to ps.SINDy(feature_names=...) in Plan 03 so symbolic
#                   scoring doesn't trip on x0/x1/x2 vs x/y/z.
#   ic_spread     — per-system default Gaussian σ consumed by ode.py when
#                   BenchmarkSpec.ic_spread is None (sentinel). LV is retuned
#                   to 0.03 for the small-orbit FP neighborhood (Phase 6.1
#                   D-04 amended); VdP/Lorenz keep 1.0 (legacy behavior).
DEFAULT_PARAMS: dict[str, dict[str, object]] = {
    "lotka_volterra": {
        "rhs": lotka,
        "args": ([1, 10],),
        "dim": 2,
        # Phase 6.1 D-01 (locked 2026-04-24): default_ic retuned from
        # [5.0, 5.0] to [0.3, 0.1] to close the LV seed=42 instance of
        # L-015. At IC=[5.0, 5.0] with p=[1, 10, 2, 10] the orbit period
        # far exceeds t_span=[0, 20] — truth spends 99% of the window
        # near-zero quiescence, making the benchmark undiscoverable
        # (scoring treats LV and exponential-decay as indistinguishable).
        # Verified 2026-04-24: at IC=[0.3, 0.1] the orbit period is
        # T ≈ 4.55s so ~4 full cycles fit in t_span=[0, 20], with
        # x ∈ [0.125, 0.30] and y ∈ [0.050, 0.175] visibly oscillatory
        # throughout. See .planning/phases/06.1-benchmark-discoverability-audit/
        # 06.1-RESEARCH.md §1 and §2 for the numerical verification and
        # the rejected alternatives (path-b extending t_span; path-c both).
        # Do NOT revert without reading 06.1-CONTEXT.md D-01 first.
        "default_ic": [0.3, 0.1],
        "t_span": (0.0, 20.0),
        "feature_names": ["x", "y"],
        # Phase 6.1 D-04 amended (2026-04-24): ic_spread scaled to the
        # new small-orbit neighborhood. 3σ = 0.09 so y-3σ = 0.01 stays
        # positive (populations must be non-negative for LV). See
        # CONTEXT.md §Risks "IC jitter into negative populations".
        "ic_spread": 0.03,
    },
    "van_der_pol": {
        "rhs": van_der_pol,
        "args": ([0.5],),
        "dim": 2,
        "default_ic": [2.0, 0.0],
        "t_span": (0.0, 20.0),
        "feature_names": ["x", "y"],
        # Phase 6.1 D-04 amended: preserves pre-6.1 global-default behavior.
        "ic_spread": 1.0,
    },
    "lorenz": {
        "rhs": lorenz,
        # IN-02 (decision log): beta = 8/3 → 2.6666666666666665 (Python float).
        # PySINDy's ensembling examples quote the constant as `2.66667`. The
        # scoring canonicalization layer (scoring.py:_truth_rhs_sympy) uses
        # nsimplify(2.66667, rational=True) to match what PySINDy's
        # model.equations() emits after fitting on a trajectory integrated
        # with any beta in the 2.666...67 neighbourhood — the fit rounds
        # coefficients to 5 significant digits. An agent proposing the exact
        # rational `Rational(8,3)` will score `symbolic_parse_fail` against
        # this canonical form; that is a known trade-off documented here so
        # future reviewers see the intent rather than "fixing" it by
        # switching to Rational(8,3) (which would then mis-match PySINDy's
        # discovered coefficient in the Tier 1 baseline comparison).
        "args": (10, 8 / 3, 28),
        "dim": 3,
        "default_ic": [-8.0, 8.0, 27.0],
        "t_span": (0.0, 10.0),
        "feature_names": ["x", "y", "z"],
        # Phase 6.1 D-04 amended: preserves pre-6.1 global-default behavior;
        # Lorenz is robust to ~1-unit IC perturbations (attractor is a
        # strange attractor, initial transient rapidly folds onto the same
        # manifold regardless of small IC shifts).
        "ic_spread": 1.0,
    },
}


def params_for(system: str) -> dict[str, object]:
    """Return the DEFAULT_PARAMS row for ``system`` or raise UnknownBenchmarkSystem.

    Plain English: given a string like ``"lorenz"``, hand back the recipe
    row for that system (RHS function, default params, integration window,
    etc.). If the name isn't in our registry, shout loudly — don't
    return None or a default.
    """
    if system not in DEFAULT_PARAMS:
        raise UnknownBenchmarkSystem(f"Unknown benchmark system: {system}")
    return DEFAULT_PARAMS[system]
