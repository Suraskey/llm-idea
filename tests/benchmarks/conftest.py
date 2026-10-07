"""Benchmarks-test fixtures. Inherits L-004 singleton-patch pattern.

The `settings` singleton caches .env values at import time; clearing env
vars via ``monkeypatch.delenv`` alone does NOT reach the cached fields.
Phase 6.0 Plan 01 adds NO benchmark-specific `Settings` fields (per
RESEARCH §Runtime State Inventory), so there are no `monkeypatch.setattr`
calls here yet — but the skeleton is kept in place (including the
``BENCHMARK_`` prefix in ``_ENV_PREFIXES``) so future phases that
introduce BENCHMARK_* knobs inherit the L-004 discipline out of the
box.

Also provides the canonical per-system fixtures — tiny noise-free
bundles reused across multiple test files:
  - ``lv_small_bundle``: Lotka-Volterra, 4 trajectories × 200 samples,
    seed=42 (noise-free). ``train_mask.sum() == 3``, ``held_out == 1``.
  - ``vdp_small_bundle``: Van der Pol, 5 trajectories × 500 samples,
    seed=123, t_span (0, 50) (~6 oscillation cycles), noise-free.
  - ``lorenz_small_bundle``: Lorenz, 5 trajectories × 5000 samples,
    seed=7, t_span (0, 10) (chaos-density timeline), noise-free.

Binding decisions:
  - 06.0 PATTERNS §`tests/benchmarks/conftest.py`: L-004 autouse
    env-scrubber + per-system small bundles.
  - L-004 (LEARNINGS.md): monkeypatch.setattr(settings, "FIELD", value)
    is the only pattern that reliably overrides Settings fields in
    tests.

Pitfalls honored:
  - RESEARCH Pitfall #1: bundles go through the real
    ``generate_trajectories`` entrypoint so integrator kwargs (LSODA,
    rtol=atol=1e-12) are exercised on every fixture use.
  - RESEARCH Pitfall #4: every RNG use lives in Plan 01's generator;
    no ``np.random.seed`` anywhere in the test tree.
"""

from __future__ import annotations

import os

import pytest

# Set GOOGLE_API_KEY before any test module triggers `from ascension.benchmarks
# import ...`, which transitively imports `ascension.common.config` and
# instantiates the Settings singleton (which REQUIRES GOOGLE_API_KEY). pytest
# loads conftest.py before test modules, so module-scope env mutation here
# happens before any benchmark import chain runs.
#
# This is test-only; production callers are expected to provide a real key
# via their own environment.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")


_ENV_PREFIXES = (
    "ASCENSION_",
    "SANDBOX_",
    "BENCHMARK_",
    "MODEL_",
    "EMBEDDING_",
    "GOOGLE_",
    "POSTGRES_",
    "DATABASE_",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip benchmark-related env overrides and patch the Settings singleton.

    L-004: setenv does NOT reach the cached Settings singleton. Plan 01
    introduces no BENCHMARK_* Settings fields, so there are no setattr
    calls here yet. Add ``monkeypatch.setattr(settings, "BENCHMARK_FOO",
    <locked default>)`` here if a future phase introduces BENCHMARK_*
    knobs on the Settings class.
    """
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    # Settings import requires GOOGLE_API_KEY; supply a dummy test value.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


@pytest.fixture
def lv_small_bundle():
    """Tiny noise-free LV bundle — 4 trajectories, 200 samples, seed=42.

    Plain English: a cheap pre-baked bundle many tests reuse so we don't
    re-integrate the same trajectory five times across the suite.
    80/20 split: train_mask.sum() == 3, held_out == 1.
    """
    from ascension.benchmarks import BenchmarkSpec, generate_trajectories

    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=42,
    )
    return generate_trajectories(spec)


@pytest.fixture
def vdp_small_bundle():
    """Tiny Van der Pol bundle — 5 trajs, period ≈ 7.63 so 50 units covers ~6 cycles.

    80/20 split: train_mask.sum() == 4, held_out == 1. Seed=123, noise_sigma=0.
    """
    from ascension.benchmarks import BenchmarkSpec, generate_trajectories

    spec = BenchmarkSpec(
        system="van_der_pol",
        n_trajectories=5,
        noise_sigma=0.0,
        n_samples=500,
        seed=123,
        t_span_override=(0.0, 50.0),
    )
    return generate_trajectories(spec)


@pytest.fixture
def lorenz_small_bundle():
    """Tiny Lorenz bundle — 5 trajs, 5000 samples, chaos-density timeline.

    80/20 split: train_mask.sum() == 4, held_out == 1. Seed=7, t_span (0, 10),
    noise_sigma=0. Chaos requires dense sampling — hence 5000 samples.
    """
    from ascension.benchmarks import BenchmarkSpec, generate_trajectories

    spec = BenchmarkSpec(
        system="lorenz",
        n_trajectories=5,
        noise_sigma=0.0,
        n_samples=5000,
        seed=7,
        t_span_override=(0.0, 10.0),
    )
    return generate_trajectories(spec)
