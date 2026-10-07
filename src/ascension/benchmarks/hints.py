"""Alpha-8 — fair training-trajectory statistics for the agent prompt.

Computes summary statistics from a ``TrajectoryBundle`` that any
scientist would produce in the first five minutes of looking at the
data: initial conditions, time span, amplitude, mean, rough period
estimate. These are injected into the user prompt as a
``<benchmark_hints>`` block so the agent can pick plausible y0 / tspan
/ coefficient magnitudes instead of guessing blind.

What this does NOT include:
  * The full trajectory (cost / leakage-prevention)
  * The ground-truth parameters (the agent is supposed to rediscover)
  * The symbolic form (ditto)
  * Any scoring internals

Why this is fair: an analyst given a black-box dataset always starts by
computing mean/std/amplitude/period. Withholding these from the agent
makes the task unrepresentative of real scientific discovery — the
agent should be judged on symbolic-form identification, not blind
numerical fishing.

Returned shape:

    {
        "dim": 2,
        "y0":  [5.0, 5.0],
        "t_span": [0.0, 20.0],
        "n_samples": 100,
        "per_dim": [
            {
                "name": "x0",
                "mean": float,
                "amplitude": float,   # max - min
                "min": float,
                "max": float,
                "std": float,
                "est_period": float | None,
            },
            ...
        ],
        "n_train_trajectories": int,
    }

Serialized via ``json.dumps(..., indent=2, sort_keys=True)`` by callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ascension.benchmarks.types import TrajectoryBundle


def _estimate_period(t: np.ndarray, vals: np.ndarray) -> float | None:
    """Rough period estimate via autocorrelation peak detection.

    Returns None if the signal is too short (<3 samples) or no clear
    period is found (the autocorrelation is monotonic — signal is
    aperiodic / decaying / chaotic).
    """
    if vals.size < 6:
        return None
    centered = vals - vals.mean()
    if centered.std() < 1e-12:
        return None
    ac = np.correlate(centered, centered, mode="full")
    ac = ac[ac.size // 2 :]  # positive lags only
    # The first local max after lag 0 is the dominant period.
    # Smoothing guards against single-sample noise spikes.
    if ac.size < 4:
        return None
    peaks = []
    for i in range(2, ac.size - 1):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.3 * ac[0]:
            peaks.append(i)
            break
    if not peaks:
        return None
    lag = peaks[0]
    dt = float(t[1] - t[0])
    return float(lag * dt)


def compute_benchmark_hints(bundle: TrajectoryBundle) -> dict[str, Any]:
    """Derive training-trajectory statistics from a bundle.

    Uses ``bundle.clean`` restricted to ``bundle.train_mask`` so we don't
    leak held-out trajectory statistics to the agent. The hints reflect
    what an analyst would see from the training split alone.
    """
    clean = bundle.clean  # shape (n_traj, n_samples, dim)
    mask = bundle.train_mask
    t = bundle.t

    train = clean[mask]  # (n_train, n_samples, dim)
    if train.size == 0:  # pragma: no cover — defensive, bundles always have train rows
        train = clean

    dim = int(train.shape[-1])
    n_samples = int(train.shape[1])

    # Take one representative training trajectory for the "typical y0"
    # hint. The spec's default_ic is the centroid; ics[0] is the first
    # perturbed starting point. Report both so the agent knows the
    # training split spans a small neighbourhood, not a single point.
    first_ic = [float(x) for x in bundle.ics[mask][0]]

    per_dim = []
    for d in range(dim):
        col = train[:, :, d].ravel()
        # Per-trajectory period estimate averaged across train runs.
        periods = []
        for tr in train[:, :, d]:
            p = _estimate_period(t, tr)
            if p is not None:
                periods.append(p)
        mean_period = float(np.mean(periods)) if periods else None
        # Alpha-9: per-dim quantiles + spike-ratio so the agent can tell
        # "trajectory decays to origin" (mean ≈ 0, std small) from
        # "trajectory spikes near zero" (mean ≈ 0, std huge, q90 >> q50).
        # Lotka-Volterra is the latter; a pure decay is the former. Alpha-8
        # misread "mean ≈ 0" as decay and locked on a wrong structural form.
        mean_val = float(col.mean())
        std_val = float(col.std())
        q10 = float(np.quantile(col, 0.10))
        q50 = float(np.quantile(col, 0.50))
        q90 = float(np.quantile(col, 0.90))
        spike_ratio = (std_val / abs(mean_val)) if abs(mean_val) > 1e-9 else None
        # Shape classifier — string label so the agent can read it directly.
        if q90 > 3 * max(abs(q50), 1e-9):
            shape = "spikes_near_zero"
        elif std_val < 0.1 * abs(q50 - q10):
            shape = "smooth_decay"
        elif std_val < 0.1 * abs(mean_val):
            shape = "smooth_stationary"
        else:
            shape = "oscillatory_or_mixed"
        per_dim.append(
            {
                "name": f"x{d}",
                "min": float(col.min()),
                "max": float(col.max()),
                "amplitude": float(col.max() - col.min()),
                "mean": mean_val,
                "std": std_val,
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "spike_ratio": (float(spike_ratio) if spike_ratio is not None else None),
                "shape": shape,
                "est_period": mean_period,
            }
        )

    # Alpha-10 + Alpha-14: downsampled training trajectory sample so the
    # agent can read phase and period directly instead of inferring from
    # aggregate stats.
    #
    # Alpha-14 fix (root cause of alpha-13 regression): the original
    # alpha-10 sampler took 20 evenly-spaced points over the full
    # n_samples. For trajectories starting far from equilibrium
    # (e.g. LV seed=42 IC=[5.3, 4] where orbits swing through extreme
    # transient values before settling into cycles), the first 10-15
    # points are the initial crash — the 20-point sample was
    # transient-dominated and led alpha-13's agent to conclude "no
    # oscillation, shape is decay-to-zero" and leave the LV family.
    #
    # Alpha-14 splits the sample into TWO phases:
    #   - transient: first ~10 points covering the initial crash/ramp
    #   - steady:    20 points sampled from the POST-transient region
    #                (skip first 30% of n_samples, then sample)
    # The agent sees BOTH. Structural interpretation ("is this
    # oscillatory or decay?") uses the steady block; initial-condition
    # reasoning uses the transient block.
    #
    # 30% skip is a benchmark-agnostic pragmatic choice for alpha-14.
    # L-015 candidate: adaptive transient detection via
    # |y(t+dt)-y(t)| dropping below a threshold.
    first_traj = train[0]  # (n_samples, dim)

    transient_n = min(10, max(1, n_samples // 4))
    transient_idx = list(range(transient_n))

    steady_start = max(transient_n, int(n_samples * 0.30))
    steady_available = n_samples - steady_start
    if steady_available >= 20:
        steady_stride = max(1, steady_available // 20)
        steady_idx = [steady_start + i * steady_stride for i in range(20)]
        steady_idx = [i for i in steady_idx if i < n_samples][:20]
    elif steady_available > 0:
        steady_idx = list(range(steady_start, n_samples))
    else:
        # Degenerate: whole trajectory is "transient" — fall back to
        # uniform sample over the full window.
        steady_idx = transient_idx

    def _sample_block(idx: list[int]) -> dict[str, Any]:
        return {
            "t": [float(t[i]) for i in idx],
            "vals": [[float(first_traj[i, d]) for d in range(dim)] for i in idx],
        }

    trajectory_sample = {
        "transient": _sample_block(transient_idx),
        "steady": _sample_block(steady_idx),
    }

    # Alpha-14 (A2): physical / structural priors for this benchmark.
    # Populated only for known benchmarks; unknown ones get None so the
    # agent prompt can treat absence gracefully. These priors encode
    # domain knowledge ANY specialist would apply (e.g. prey growth
    # α > 0 in predator-prey systems) without leaking truth parameters.
    # The agent may override these with empirical evidence + low
    # confidence per the V11 soft-rail discipline.
    structural_prior = _STRUCTURAL_PRIORS.get(bundle.spec.system)

    return {
        "dim": dim,
        "y0": first_ic,
        "t_span": [float(t[0]), float(t[-1])],
        "n_samples": n_samples,
        "n_train_trajectories": int(mask.sum()),
        "per_dim": per_dim,
        "trajectory_sample": trajectory_sample,
        "structural_prior": structural_prior,
    }


# Per-benchmark physical / structural priors. A violation (e.g. α < 0
# for lotka_volterra) is a soft-rail trigger — the agent must lower
# confidence and justify empirically. NOT a hard block; agent judgment
# still rules per the V11 discipline (see SYSTEM_PROMPT_V11).
_STRUCTURAL_PRIORS: dict[str, dict[str, Any]] = {
    "lotka_volterra": {
        "benchmark": "lotka_volterra",
        "canonical_form": "x0' = alpha*x0 - beta*x0*x1 ; x1' = delta*x0*x1 - gamma*x1",
        "constraints": {
            "alpha_positive": (
                "alpha (prey growth, coefficient of x0 in eq 1) > 0 — "
                "populations do not spontaneously decline without interaction"
            ),
            "beta_positive": (
                "beta (prey-predator interaction, coefficient of x0*x1 in eq 1, "
                "with a minus sign in front) > 0 — predation rate is non-negative"
            ),
            "gamma_positive": (
                "gamma (predator mortality, coefficient of x1 in eq 2, with a "
                "minus sign in front) > 0 — decay rate is non-negative"
            ),
            "delta_positive": (
                "delta (predator-prey interaction, coefficient of x0*x1 in eq 2) > 0 "
                "— hunting success is non-negative"
            ),
        },
        "notes": (
            "Classic predator-prey dynamics. Non-trivial fixed point = "
            "(gamma/delta, alpha/beta) when all four coefficients positive. "
            "Violating any positivity prior implies non-standard biology "
            "(e.g. extinction-only dynamics, cooperative predation). "
            "Propose violations ONLY with strong empirical evidence "
            "(e.g. rmse dropped >10x on a violating form vs the best "
            "prior-respecting form) AND set confidence <= 0.3."
        ),
    },
}


def check_structural_prior_violation(
    symbolic_form: list[str],
    benchmark: str,
) -> dict[str, Any]:
    """Check whether a proposed symbolic_form violates this benchmark's
    structural priors (alpha-14).

    Plain English: each benchmark has domain-knowledge constraints (e.g.
    predator-prey systems require alpha > 0 because populations do not
    spontaneously decline without interaction). This function parses the
    proposed form, extracts the relevant coefficients, and returns which
    priors — if any — are violated.

    Returns a dict:
      applicable: bool    — is there a prior registered for this benchmark
      parse_ok:   bool    — could we extract coefficient values
      violations: list[str] — names of violated priors (may be empty)
      coefficients: dict[str, float] — extracted values by name

    Never raises. Unknown benchmark → applicable=False. Parse failure →
    applicable=True, parse_ok=False.
    """
    prior = _STRUCTURAL_PRIORS.get(benchmark)
    if prior is None:
        return {
            "applicable": False,
            "parse_ok": False,
            "violations": [],
            "coefficients": {},
        }
    if benchmark == "lotka_volterra":
        return _check_lotka_volterra_prior(symbolic_form)
    # Known benchmark but no checker implemented yet.
    return {
        "applicable": True,
        "parse_ok": False,
        "violations": [],
        "coefficients": {},
    }


def _check_lotka_volterra_prior(symbolic_form: list[str]) -> dict[str, Any]:
    """Extract alpha/beta/gamma/delta from a parsed LV form and flag
    any non-positive coefficient.

    Parses the form with the same x0/x1 aliasing as the scorer (L-013
    fix) so the LLM's preferred variable names work.
    """
    # Imports are lazy to keep hints.py importable when the scoring
    # chain isn't needed (e.g. in fast unit tests that only touch
    # compute_benchmark_hints).
    import tokenize

    import sympy
    from sympy import symbols

    from ascension.benchmarks.scoring import _safe_parse_expr

    result: dict[str, Any] = {
        "applicable": True,
        "parse_ok": False,
        "violations": [],
        "coefficients": {},
    }

    if not symbolic_form or len(symbolic_form) != 2:
        return result

    vars_ = list(symbols("x y"))
    locals_ = {str(v): v for v in vars_}
    for i, v in enumerate(vars_):
        locals_[f"x{i}"] = v

    try:
        parsed = [_safe_parse_expr(s, local_dict=locals_) for s in symbolic_form]
    except (
        sympy.SympifyError,
        SyntaxError,
        TypeError,
        ValueError,
        NameError,
        AttributeError,
        tokenize.TokenError,
    ):
        return result

    x, y = vars_
    try:
        alpha_expr = parsed[0].coeff(x, 1).coeff(y, 0)
        neg_beta_expr = parsed[0].coeff(x, 1).coeff(y, 1)
        delta_expr = parsed[1].coeff(x, 1).coeff(y, 1)
        neg_gamma_expr = parsed[1].coeff(y, 1).coeff(x, 0)
    except Exception:  # noqa: BLE001 — never fail prior-check
        return result

    coefs: dict[str, float] = {}
    if getattr(alpha_expr, "is_number", False):
        try:
            coefs["alpha"] = float(alpha_expr)
        except (TypeError, ValueError):
            pass
    if getattr(neg_beta_expr, "is_number", False):
        try:
            coefs["beta"] = -float(neg_beta_expr)
        except (TypeError, ValueError):
            pass
    if getattr(delta_expr, "is_number", False):
        try:
            coefs["delta"] = float(delta_expr)
        except (TypeError, ValueError):
            pass
    if getattr(neg_gamma_expr, "is_number", False):
        try:
            coefs["gamma"] = -float(neg_gamma_expr)
        except (TypeError, ValueError):
            pass

    result["parse_ok"] = True
    result["coefficients"] = coefs

    violations: list[str] = []
    if "alpha" in coefs and coefs["alpha"] <= 0:
        violations.append("alpha_positive")
    if "beta" in coefs and coefs["beta"] <= 0:
        violations.append("beta_positive")
    if "gamma" in coefs and coefs["gamma"] <= 0:
        violations.append("gamma_positive")
    if "delta" in coefs and coefs["delta"] <= 0:
        violations.append("delta_positive")

    result["violations"] = violations
    return result


__all__ = [
    "compute_benchmark_hints",
    "check_structural_prior_violation",
]
