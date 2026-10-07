"""generate_trajectories — seeded, reproducible, bundled ODE trajectory producer.

Plain English: for a chosen toy physics system (predator-prey, a wobbly
oscillator, or the Lorenz butterfly), this simulates N slightly-different
starting conditions forward in time, adds measurement noise, and tucks away
a hidden 20% we'll check predictions against later. Same seed in, same
trajectories out — forever.

Binding decisions:
  - 06.0 RESEARCH §Pattern 1 lines 229–329: verbatim reference implementation.
  - 06.0 RESEARCH §Common Pitfalls #1: INTEGRATOR_KEYWORDS (LSODA, rtol=1e-12)
    from systems.py — NEVER scipy defaults.
  - 06.0 RESEARCH §Common Pitfalls #4: ``np.random.default_rng(seed)``; never
    ``np.random.seed`` (PATTERNS §Shared Pattern G).
  - 06.0 RESEARCH §Common Pitfalls #7: noise_sigma is fraction-of-per-trajectory-RMS
    (PySINDy convention).
  - 06.0 RESEARCH §Common Pitfalls #8: sol.success=False on TRUTH raises
    BenchmarkIntegrationError (no cover-ups, CLAUDE.md).
  - SCOPE §22.2 reproducibility: library versions captured in bundle.

Pitfalls honored: #1, #4, #7, #8 (see above).

Threat-model mitigations: N/A — pure numerical compute, no network/auth/secret
surface per RESEARCH §Security Domain lines 690–702.
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

import numpy as np
from scipy.integrate import solve_ivp

from ascension.benchmarks.exceptions import BenchmarkIntegrationError
from ascension.benchmarks.systems import INTEGRATOR_KEYWORDS, params_for
from ascension.benchmarks.types import BenchmarkSpec, TrajectoryBundle
from ascension.common.logging import get_logger

logger = get_logger(__name__)


def generate_trajectories(
    spec: BenchmarkSpec,
    *,
    strict: bool = True,
) -> TrajectoryBundle:
    """Produce clean + noisy trajectories with a seeded 80/20 IC-level split.

    Plain English: simulates the chosen toy physics system N times from
    slightly-different starting points, adds measurement noise, keeps 20%
    of the runs aside as a hidden held-out set, and packages everything in
    a ``TrajectoryBundle``. Same ``spec`` twice → bit-identical output
    (RESEARCH Pitfall #4 + SCOPE §22.2).

    Args:
      spec: The problem specification — system name, count, noise level,
        samples, seed, optional IC spread / time-span overrides.
      strict: Phase 6.1 D-05 gate. When True (the default), this call
        runs ``check_benchmark_discoverability(spec)`` as a pre-flight
        self-consistency audit — if the scorer can't call TRUTH exact,
        we raise ``BenchmarkIntegrationError`` before the integration
        loop runs. When False, the audit is skipped (a WARN log with
        the full spec summary is emitted so the skip is never silent —
        engineering_discipline_no_coverups). ``audit.py`` itself calls
        this function with ``strict=False`` to produce the bundle it
        scores (otherwise we'd recurse).

    Sentinel semantics on ``spec.ic_spread`` (D-04 amended):
      ``spec.ic_spread is None`` means "look up the per-system default
      from ``DEFAULT_PARAMS``" (LV=0.03, VdP=1.0, Lorenz=1.0). An
      explicit float — including ``1.0`` — means "use my value, ignore
      the registry". This is the plan-check BLOCKER-2 fix: a caller who
      legitimately typed ``ic_spread=1.0`` is now distinguishable from
      a caller who didn't pass the kwarg.

    Returns:
      A ``TrajectoryBundle`` with shape-compatible ``.clean`` /``.noisy``
      arrays of shape ``(n_trajectories, n_samples, dim)``, initial
      conditions, a bool ``train_mask`` ~80% True, and a
      ``library_versions`` dict pinning scipy/sympy/pysindy/numpy for
      reproducibility audit.

    Raises:
      UnknownBenchmarkSystem: if ``spec.system`` is not in the registry.
      ValueError: if ``spec.n_trajectories < 2``. With n=1 the 80/20 split
        puts the sole trajectory in train and leaves zero held-out, which
        causes every downstream ``score(..., held_out=empty)`` call to
        silently return ``numerical_diverged`` (WR-03). Fail loudly at
        the source rather than let the signal vanish mid-pipeline.
      BenchmarkIntegrationError: if scipy.integrate.solve_ivp returns
        ``sol.success=False`` on the TRUTH RHS (CLAUDE.md "no
        cover-ups" — host-level failure, not a reason code) OR, under
        ``strict=True``, if the Phase 6.1 discoverability audit rejects
        the spec (message prefix ``"discoverability audit failed:"``).
    """
    if spec.n_trajectories < 2:
        raise ValueError(
            "generate_trajectories requires n_trajectories >= 2 so the 80/20 "
            "split yields at least one held-out trajectory; got "
            f"n_trajectories={spec.n_trajectories}"
        )
    cfg = params_for(spec.system)
    if strict:
        # Lazy import — audit.py imports generate_trajectories with
        # strict=False to produce the bundle it scores. Top-level
        # import would cycle (audit.py <-> ode.py).
        from ascension.benchmarks.audit import check_benchmark_discoverability

        result = check_benchmark_discoverability(spec)
        if not result.passed:
            raise BenchmarkIntegrationError(
                system=spec.system,
                seed=spec.seed,
                message=(
                    f"discoverability audit failed: reason={result.reason}; "
                    f"rmse={result.score.rmse if result.score else 'None'}"
                ),
            )
    else:
        logger.warning(
            "generate_trajectories called with strict=False — "
            "discoverability audit bypassed. spec=system=%s seed=%d "
            "n_traj=%d noise=%g n_samples=%d ic_spread=%s",
            spec.system,
            spec.seed,
            spec.n_trajectories,
            spec.noise_sigma,
            spec.n_samples,
            spec.ic_spread,
        )
    rng = np.random.default_rng(spec.seed)
    t_span = spec.t_span_override if spec.t_span_override is not None else cfg["t_span"]
    t = np.linspace(t_span[0], t_span[1], spec.n_samples)
    dim: int = int(cfg["dim"])  # type: ignore[arg-type]

    # Initial conditions perturbed around the default IC (seeded Gaussian).
    # IMPORTANT: draw ICs BEFORE the integration loop so the RNG stream order is
    # deterministic (Pitfall #4). Any reshuffling here breaks bit-reproducibility.
    base_ic = np.asarray(cfg["default_ic"], dtype=np.float64)
    # Phase 6.1 D-04 amended (2026-04-24): spec.ic_spread is a sentinel.
    # None means "look up per-system default from DEFAULT_PARAMS". An
    # explicit float (including 1.0) means "use my value, ignore the
    # registry". This is the plan-check BLOCKER-2 resolution — true
    # caller-wins semantics with no silent override.
    if spec.ic_spread is None:
        effective_ic_spread = float(cfg.get("ic_spread", 1.0))
    else:
        effective_ic_spread = float(spec.ic_spread)
    ics = base_ic + effective_ic_spread * rng.standard_normal((spec.n_trajectories, dim))

    clean = np.empty((spec.n_trajectories, spec.n_samples, dim), dtype=np.float64)
    for i, ic in enumerate(ics):
        sol = solve_ivp(
            cfg["rhs"],
            t_span,
            ic,
            t_eval=t,
            args=cfg["args"],
            **INTEGRATOR_KEYWORDS,
        )
        if not sol.success:
            # No cover-ups: surface explicit host-level failure.
            raise BenchmarkIntegrationError(
                system=spec.system,
                seed=spec.seed,
                message=f"solve_ivp failed on traj {i}: {sol.message}",
            )
        clean[i] = sol.y.T

    # Noise: Gaussian with sigma = noise_sigma × per-trajectory RMS
    # (PySINDy convention — RESEARCH Pitfall #7). Per-trajectory RMS so the
    # meaning of "1% noise" is comparable across systems with different
    # value ranges (Lorenz ~[-20,20] vs VdP ~[-2,2]).
    rms = np.sqrt((clean**2).mean(axis=(1, 2), keepdims=True))  # shape (N, 1, 1)
    noise = rng.standard_normal(clean.shape) * rms * spec.noise_sigma
    noisy = clean + noise

    # 80/20 IC-level split (seeded permutation; train_mask ~80% True).
    # IC-level (not sample-level) so train and held-out live on distinct
    # initial conditions — agents never see the test ICs at all.
    n_train = int(round(0.8 * spec.n_trajectories))
    order = rng.permutation(spec.n_trajectories)
    train_mask = np.zeros(spec.n_trajectories, dtype=bool)
    train_mask[order[:n_train]] = True

    return TrajectoryBundle(
        spec=spec,
        t=t,
        clean=clean,
        noisy=noisy,
        ics=ics,
        train_mask=train_mask,
        library_versions={
            "scipy": _pkg_version("scipy"),
            "sympy": _pkg_version("sympy"),
            "pysindy": _pkg_version("pysindy"),
            "numpy": np.__version__,
        },
    )
