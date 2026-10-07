"""Phase 15.2 Plan 01 Task 1 — alien_benchmark() feed-shape + data-only-boundary tests.

Plain English: ``alien_benchmark()`` is the alien-native twin of
``benchmarks/fixtures.py:default_benchmark`` — it builds the data-only
``ObservationBundle`` the Council sees, plus the alien truth ``AlienConfig`` +
held-out IC set the SCORER (Plan 02) sees on a SEPARATE path the Council never
touches.

These tests pin the five Wave-1 behaviors (15.2-01-PLAN.md Task 1):

  1. shape — returns (ObservationBundle, AlienConfig, tuple[AlienConfig, ...]);
     held-out has >= 12 entries.
  2. data-only boundary (T-15.2-leak / the §10 invariant) — the returned
     ObservationBundle exposes ONLY {t, positions, velocities, masses}: no
     ``charges``, no ``Q``, no ``config``, no true law. Pinned as a REGRESSION so
     a future field-add to ObservationBundle can't silently re-leak hidden state.
  3. locked universe (T-15.2-03) — the truth cfg has masses==(1.0, 1.0) (NOT the
     non-integrable (1000, 1)) and the §10 couplings (alpha=0.05, beta=0.02,
     gamma=0.7, kappa=0.3) — built from TIER2_TUNED_PARAMS, never
     DEFAULT_ALIEN_PARAMS.
  4. determinism — two ``alien_benchmark(seed=0)`` calls return bit-identical
     positions (np.array_equal).
  5. reuse, not recreate — the held-out tuple IS ``config.TIER2_HELD_OUT_ICS``
     (identity), not a freshly hand-built set.

$0 LLM, deterministic numpy. No agent in any path.
"""

from __future__ import annotations

import numpy as np

from ascension.benchmarks.alien_fixtures import alien_benchmark
from ascension.simulator import config as sim_config
from ascension.simulator.types import AlienConfig, ObservationBundle


def test_alien_benchmark_returns_three_tuple_shape() -> None:
    """alien_benchmark() returns (ObservationBundle, AlienConfig, held_out tuple)."""
    bundle, truth_cfg, held_out = alien_benchmark(seed=0)

    assert isinstance(bundle, ObservationBundle)
    assert isinstance(truth_cfg, AlienConfig)
    assert isinstance(held_out, tuple)
    assert all(isinstance(c, AlienConfig) for c in held_out)
    # >= 12 held-out configs (D-09 generalization set).
    assert len(held_out) >= 12


def test_alien_benchmark_bundle_is_data_only() -> None:
    """T-15.2-leak / §10 invariant — the Council-visible bundle carries NO hidden state.

    The ObservationBundle field set is exactly {t, positions, velocities, masses}.
    A future field-add (charges / Q / config / true law) that re-leaked hidden
    state would break this regression. This is the load-bearing zero-shot guard.
    """
    bundle, _truth_cfg, _held_out = alien_benchmark(seed=0)

    # The hidden fields the Council must NEVER see.
    assert not hasattr(bundle, "charges"), "ObservationBundle leaked charges"
    assert not hasattr(bundle, "Q"), "ObservationBundle leaked Q"
    assert not hasattr(bundle, "config"), "ObservationBundle leaked config (true law)"
    assert not hasattr(bundle, "library_versions")

    # Field set is exactly the data-only four.
    field_names = {f.name for f in __import__("dataclasses").fields(bundle)}
    assert field_names == {"t", "positions", "velocities", "masses"}


def test_alien_benchmark_uses_locked_tuned_universe() -> None:
    """T-15.2-03 — truth cfg is the locked (1,1) TIER2 universe, not (1000,1)."""
    _bundle, truth_cfg, _held_out = alien_benchmark(seed=0)

    # masses=(1,1) — the integrable regime. NEVER (1000,1) (DEFAULT_ALIEN_PARAMS).
    assert tuple(truth_cfg.masses) == (1.0, 1.0)
    assert tuple(truth_cfg.masses) != tuple(
        sim_config.DEFAULT_ALIEN_PARAMS["masses"]  # type: ignore[index]
    )

    # The §10 couplings, sourced from TIER2_TUNED_PARAMS.
    assert truth_cfg.alpha == 0.05
    assert truth_cfg.beta == 0.02
    assert truth_cfg.gamma == 0.7
    assert truth_cfg.kappa == 0.3
    assert tuple(truth_cfg.charges) == (1.0, 1.0)


def test_alien_benchmark_is_deterministic() -> None:
    """Two seed=0 calls return bit-identical positions (D-06 reproducibility)."""
    bundle_a, _ta, _ha = alien_benchmark(seed=0)
    bundle_b, _tb, _hb = alien_benchmark(seed=0)

    assert np.array_equal(bundle_a.positions, bundle_b.positions)
    assert np.array_equal(bundle_a.velocities, bundle_b.velocities)
    assert np.array_equal(bundle_a.t, bundle_b.t)


def test_alien_benchmark_reuses_tier2_held_out_ics() -> None:
    """The held-out tuple IS config.TIER2_HELD_OUT_ICS — reused, not recreated."""
    _bundle, _truth_cfg, held_out = alien_benchmark(seed=0)

    # Identity: the feed reuses the module-bound 15.1 set, not a fresh build.
    assert held_out is sim_config.TIER2_HELD_OUT_ICS
