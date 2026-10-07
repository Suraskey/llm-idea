"""Benchmarks — ODE rediscovery harness for Tier 1 validation (Phase 6.0).

Anchored to PySINDy's `pysindy.utils` benchmark RHS functions (SCOPE §9).
Wraps `scipy.integrate.solve_ivp` with PySINDy's canonical integrator
keywords (LSODA, rtol=atol=1e-12) and seeds every RNG draw through a
scoped ``np.random.default_rng(seed)`` for bit-reproducibility
(SCOPE §22.2). Per-run scoring failures are string reason codes on the
Plan 02 `BenchmarkScore` DTO — ONLY host-level integrator breakage on
TRUTH RHS raises `BenchmarkIntegrationError` here.

What this module does (plain English): takes a "benchmark recipe"
(`BenchmarkSpec` — which toy physics system, how many runs, how much
noise, which seed), runs the reference simulator, sprinkles in the
specified amount of measurement noise, splits 80/20 train/held-out,
and hands back a packing box (`TrajectoryBundle`) that downstream plans
(scoring in Plan 02, PySINDy baseline in Plan 03, and every agent /
ablation in Phase 7.0+) consume. Bit-reproducible: same recipe in,
identical numbers out.

Public surface (re-exported from submodules):
  - BenchmarkSpec, TrajectoryBundle (types.py)
  - BenchmarkError, BenchmarkIntegrationError, UnknownBenchmarkSystem
    (exceptions.py)
  - DEFAULT_PARAMS, INTEGRATOR_KEYWORDS, PYSINDY_VERSION_ANCHOR,
    params_for (systems.py)
  - generate_trajectories (ode.py)

Binding decisions: see 06.0 RESEARCH §Pattern 1 (generate_trajectories)
and §Pattern 3 (reason-code discipline for per-run failures); 06.0
PATTERNS §Shared Patterns A/B/C/D/E/G/I for the module shape.

Pitfalls honored (from 06.0 RESEARCH §Common Pitfalls):
  - #1 (scipy default tolerance wrecks Lorenz) — `INTEGRATOR_KEYWORDS`
    pins LSODA + rtol=atol=1e-12.
  - #4 (numpy legacy RNG state leaks) — `np.random.default_rng(seed)`
    everywhere; never `np.random.seed`.
  - #7 (noise sigma convention varies across papers) — documented on
    `BenchmarkSpec.noise_sigma` as fraction of per-trajectory RMS
    (PySINDy convention).
  - #8 (blowup detection on proposed ODEs) — TRUTH integration failures
    raise `BenchmarkIntegrationError`; proposed-ODE failures are
    Plan 02's reason codes, not exceptions.

Threat-model mitigations: N/A — pure-compute phase, no network / auth /
secrets surface per 06.0 RESEARCH §Security Domain lines 690–702.
"""

from ascension.benchmarks.audit import (
    AuditReasonCode,
    AuditResult,
    check_benchmark_discoverability,
    clear_audit_cache,
)
from ascension.benchmarks.exceptions import (
    BenchmarkError,
    BenchmarkIntegrationError,
    UnknownBenchmarkSystem,
)
from ascension.benchmarks.io import (
    benchmark_run_dir,
    load_run_artifacts,
    save_run_artifacts,
)
from ascension.benchmarks.ode import generate_trajectories
from ascension.benchmarks.pysindy_baseline import run_pysindy_baseline
from ascension.benchmarks.reason_codes import ScoreReasonCode
from ascension.benchmarks.score_types import BenchmarkScore
from ascension.benchmarks.scoring import score
from ascension.benchmarks.systems import (
    DEFAULT_PARAMS,
    INTEGRATOR_KEYWORDS,
    PYSINDY_VERSION_ANCHOR,
    params_for,
)
from ascension.benchmarks.types import BenchmarkSpec, TrajectoryBundle

__all__ = [
    # Plan 01 (Wave 1) surface
    "BenchmarkSpec",
    "TrajectoryBundle",
    "BenchmarkError",
    "UnknownBenchmarkSystem",
    "BenchmarkIntegrationError",
    "DEFAULT_PARAMS",
    "INTEGRATOR_KEYWORDS",
    "PYSINDY_VERSION_ANCHOR",
    "params_for",
    "generate_trajectories",
    # Plan 02 (Wave 1) surface
    "ScoreReasonCode",
    "BenchmarkScore",
    "score",
    # Plan 03 (Wave 2) surface
    "run_pysindy_baseline",
    "save_run_artifacts",
    "load_run_artifacts",
    "benchmark_run_dir",
    # Phase 6.1 surface (discoverability audit)
    "AuditReasonCode",
    "AuditResult",
    "check_benchmark_discoverability",
    "clear_audit_cache",
]
