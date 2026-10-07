"""Phase 11.0 Tier-2 validation fixtures — typed BenchmarkSpec + TrajectoryBundle
ready to feed into the live Orchestrator + RoundCloseCoordinator.

Plain English: every live Council run needs a "real" BenchmarkSpec wired
through cli.py → build_council → 5 per-role agents (carry as attr) AND
cli.py → Orchestrator → RoundCloseCoordinator (uses for benchmark.scoring.
score()-based best/worst selection per L-016 mitigation R-1). Before this
module the only way to assemble a spec was inside tests/benchmarks/conftest.py.

This module exposes the three Tier-4 stability fixtures (LV, VdP, Lorenz)
and a default-helper `default_benchmark()` that the orchestrator CLI picks
when `ASCENSION_BENCHMARK_SYSTEM` is unset. Each fixture returns a
`(BenchmarkSpec, TrajectoryBundle)` pair so the caller does not have to
separately drive `generate_trajectories()`.

D-15 (Reward Collapse mitigation R-1) binding: this is the producer side
of the benchmark-spec plumbing. The 6-commit chain into Tier-2 wires the
spec through 5 layers; this module is layer 5 (the source). Without this
module, cli.py:155-161 has no spec to forward, so RoundCloseCoordinator
falls back to confidence-based best/worst selection — the L-016 circular
feedback we explicitly built the plumbing to avoid.

L-018 cycle-defense: benchmarks → council import direction is FORBIDDEN
(would inject the heavy LLM transitive dependency tree into the benchmarks
package). This module only depends on the existing benchmarks public
surface (`BenchmarkSpec`, `TrajectoryBundle`, `generate_trajectories`) —
zero council imports.

Reuses Phase 6.1 LV retuned IC + canonical seeds from EXP-001 record so
six months from now the same fixture call yields bit-identical
trajectories (SCOPE §22.2 reproducibility).
"""

from __future__ import annotations

from ascension.benchmarks.ode import generate_trajectories
from ascension.benchmarks.types import BenchmarkSpec, SystemName, TrajectoryBundle


def lotka_volterra_retuned(
    seed: int = 42, n_trajectories: int = 4, n_samples: int = 200
) -> tuple[BenchmarkSpec, TrajectoryBundle]:
    """LV (predator-prey) benchmark — Phase 6.1 retuned IC, Phase 11.0 default.

    EXP-001 binding: this is the canonical "warm" Council benchmark.
    `noise_sigma=0.0` keeps the held-out scoring deterministic so
    P5/P7/P8 in EXP-003 predictions can pin a tight tolerance.
    """
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=n_trajectories,
        noise_sigma=0.0,
        n_samples=n_samples,
        seed=seed,
    )
    bundle = generate_trajectories(spec)
    return spec, bundle


def van_der_pol_default(
    seed: int = 42, n_trajectories: int = 5, n_samples: int = 500
) -> tuple[BenchmarkSpec, TrajectoryBundle]:
    """VdP (self-sustaining oscillator) benchmark — Tier-4 cell 4-6.

    `t_span_override=(0.0, 50.0)` per Phase 6.1 conftest convention so
    the trajectory captures ≥ 5 limit-cycle periods at default mu.
    """
    spec = BenchmarkSpec(
        system="van_der_pol",
        n_trajectories=n_trajectories,
        noise_sigma=0.0,
        n_samples=n_samples,
        seed=seed,
        t_span_override=(0.0, 50.0),
    )
    bundle = generate_trajectories(spec)
    return spec, bundle


def lorenz_default(
    seed: int = 42, n_trajectories: int = 5, n_samples: int = 1000
) -> tuple[BenchmarkSpec, TrajectoryBundle]:
    """Lorenz (chaotic butterfly) benchmark — Tier-4 cell 7-9."""
    spec = BenchmarkSpec(
        system="lorenz",
        n_trajectories=n_trajectories,
        noise_sigma=0.0,
        n_samples=n_samples,
        seed=seed,
    )
    bundle = generate_trajectories(spec)
    return spec, bundle


_FIXTURES: dict[SystemName, callable] = {
    "lotka_volterra": lotka_volterra_retuned,
    "van_der_pol": van_der_pol_default,
    "lorenz": lorenz_default,
}


def default_benchmark(
    system: SystemName = "lotka_volterra", seed: int = 42
) -> tuple[BenchmarkSpec, TrajectoryBundle]:
    """Orchestrator CLI default. Reads ASCENSION_BENCHMARK_SYSTEM /
    ASCENSION_BENCHMARK_SEED env vars at the call site (cli.py:155-161)."""
    if system not in _FIXTURES:
        raise ValueError(
            f"default_benchmark: unknown system {system!r}; " f"expected one of {sorted(_FIXTURES)}"
        )
    return _FIXTURES[system](seed=seed)


__all__ = [
    "default_benchmark",
    "lorenz_default",
    "lotka_volterra_retuned",
    "van_der_pol_default",
]
