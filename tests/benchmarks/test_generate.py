"""Tests for generate_trajectories (06.0 RESEARCH §Pattern 1, ode.py).

Covers all the <behavior> bullets from the plan:
  - shape invariants (clean/noisy/ics/train_mask).
  - bit-reproducibility (same spec twice → identical arrays).
  - noise_sigma=0.0 → noisy == clean.
  - noise_sigma=0.01 → measured RMS-relative noise ≈ 0.01.
  - Lorenz trajectory bounding box is in the correct scale (regression
    guard for Pitfall #1: scipy defaults would produce visibly wrong
    numbers).
  - Unknown system raises UnknownBenchmarkSystem.

These tests are under the default ``-m 'not smoke'`` pytest config — they
are unit-level and run every CI pass. The "smoke" marker is reserved for
future live-PySINDy end-to-end happy paths in Plan 03.
"""

from __future__ import annotations

import numpy as np
import pytest

from ascension.benchmarks import (
    BenchmarkSpec,
    TrajectoryBundle,
    UnknownBenchmarkSystem,
    generate_trajectories,
)


def test_generate_shapes(lv_small_bundle: TrajectoryBundle) -> None:
    bundle = lv_small_bundle
    assert bundle.clean.shape == (4, 200, 2)
    assert bundle.noisy.shape == (4, 200, 2)
    assert bundle.ics.shape == (4, 2)
    assert bundle.train_mask.shape == (4,)
    assert bundle.train_mask.dtype == np.bool_
    # 80% of 4 = 3.2 → round to 3 True values.
    assert bundle.train_mask.sum() == 3


def test_generate_is_bit_reproducible() -> None:
    """SCOPE §22.2: same spec → identical arrays. RESEARCH Pitfall #4."""
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.01,
        n_samples=200,
        seed=42,
    )
    a = generate_trajectories(spec)
    b = generate_trajectories(spec)
    np.testing.assert_array_equal(a.clean, b.clean)
    np.testing.assert_array_equal(a.noisy, b.noisy)
    np.testing.assert_array_equal(a.ics, b.ics)
    np.testing.assert_array_equal(a.train_mask, b.train_mask)
    np.testing.assert_array_equal(a.t, b.t)


def test_generate_different_seeds_produce_different_output() -> None:
    spec1 = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=42,
    )
    spec2 = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=43,
    )
    a = generate_trajectories(spec1)
    b = generate_trajectories(spec2)
    # Different ICs → different trajectories.
    assert not np.array_equal(a.ics, b.ics)
    assert not np.array_equal(a.clean, b.clean)


def test_generate_noise_zero_means_clean_equals_noisy(
    lv_small_bundle: TrajectoryBundle,
) -> None:
    np.testing.assert_array_equal(lv_small_bundle.clean, lv_small_bundle.noisy)


def test_generate_noise_one_percent_within_tolerance() -> None:
    """noise_sigma=0.01 → measured per-trajectory relative noise ≈ 0.01.

    Within ±30% relative tolerance to account for small-sample variance
    (4 trajectories × 200 samples × 2 dims).
    """
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.01,
        n_samples=200,
        seed=42,
    )
    bundle = generate_trajectories(spec)
    # Per-trajectory RMS (what the noise scales to).
    rms = np.sqrt((bundle.clean**2).mean(axis=(1, 2), keepdims=True))
    measured = ((bundle.noisy - bundle.clean) / rms).std()
    assert (
        abs(measured - 0.01) / 0.01 < 0.3
    ), f"measured relative noise {measured} not within 30% of 0.01"
    # And the noisy array is NOT bit-equal to clean when sigma > 0.
    assert not np.array_equal(bundle.clean, bundle.noisy)


def test_generate_train_mask_80_20_split() -> None:
    """Train mask: 80% of n_trajectories rounded. 4 → 3, 10 → 8."""
    for n, expected_train in [(4, 3), (10, 8), (5, 4), (20, 16)]:
        spec = BenchmarkSpec(
            system="lotka_volterra",
            n_trajectories=n,
            noise_sigma=0.0,
            n_samples=100,
            seed=0,
        )
        bundle = generate_trajectories(spec)
        assert bundle.train_mask.sum() == expected_train, (
            f"n_trajectories={n}: expected {expected_train} True, " f"got {bundle.train_mask.sum()}"
        )
        assert bundle.train_mask.shape == (n,)


def test_generate_lorenz_uses_tight_tolerance() -> None:
    """Regression guard for RESEARCH Pitfall #1.

    Lorenz z-coordinate canonically ranges up to ~45 on the canonical
    attractor (sigma=10, beta=8/3, rho=28). With scipy defaults
    (rtol=1e-3), the integrator drifts off the attractor within a few
    time units; with LSODA + rtol=atol=1e-12, the trajectory stays in
    the correct bounding box. Asserting max z > 25 is a cheap
    fingerprint that the integrator pipeline is wired through correctly.
    """
    spec = BenchmarkSpec(
        system="lorenz",
        n_trajectories=2,
        noise_sigma=0.0,
        n_samples=500,
        seed=7,
        t_span_override=(0.0, 10.0),
    )
    bundle = generate_trajectories(spec)
    assert np.isfinite(bundle.clean).all(), "Lorenz trajectory has NaN/Inf"
    # Max over (trajectory, time) axis → per-dim maximum.
    max_per_axis = bundle.clean.max(axis=(0, 1))
    assert max_per_axis[2] > 25.0, (
        f"Lorenz z max {max_per_axis[2]} < 25 — integrator tolerance "
        "may have drifted (Pitfall #1)"
    )


def test_generate_rejects_n_trajectories_below_two() -> None:
    """WR-03 regression: n<2 silently produces empty held-out; fail loudly.

    With n=1 the 80/20 split gives train_mask.sum()==1 and zero held-out,
    causing every downstream score(...) to land on 'numerical_diverged'
    without explanation. Validation at the source (generate_trajectories)
    prevents the phantom signal from leaking into ablation runs.
    """
    import pytest

    for n in (0, 1):
        spec = BenchmarkSpec(
            system="lotka_volterra",
            n_trajectories=n,
            noise_sigma=0.0,
            n_samples=50,
            seed=0,
        )
        with pytest.raises(ValueError, match="n_trajectories >= 2"):
            generate_trajectories(spec)


def test_generate_lorenz_shapes() -> None:
    spec = BenchmarkSpec(
        system="lorenz",
        n_trajectories=2,
        noise_sigma=0.0,
        n_samples=300,
        seed=99,
    )
    bundle = generate_trajectories(spec)
    assert bundle.clean.shape == (2, 300, 3)
    assert bundle.ics.shape == (2, 3)


def test_generate_van_der_pol_shapes() -> None:
    spec = BenchmarkSpec(
        system="van_der_pol",
        n_trajectories=3,
        noise_sigma=0.0,
        n_samples=150,
        seed=11,
    )
    bundle = generate_trajectories(spec)
    assert bundle.clean.shape == (3, 150, 2)


def test_generate_unknown_system_raises() -> None:
    with pytest.raises(UnknownBenchmarkSystem):
        generate_trajectories(
            BenchmarkSpec(
                system="banana",  # type: ignore[arg-type]
                n_trajectories=2,
                noise_sigma=0.0,
                n_samples=10,
                seed=0,
            )
        )


def test_generate_library_versions_populated() -> None:
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=2,
        noise_sigma=0.0,
        n_samples=50,
        seed=0,
    )
    bundle = generate_trajectories(spec)
    assert set(bundle.library_versions.keys()) == {"scipy", "sympy", "pysindy", "numpy"}
    for key, value in bundle.library_versions.items():
        assert value, f"library_versions[{key}] must be non-empty"


def test_generate_t_span_override_is_honored() -> None:
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=2,
        noise_sigma=0.0,
        n_samples=11,
        seed=0,
        t_span_override=(0.0, 5.0),
    )
    bundle = generate_trajectories(spec)
    assert bundle.t[0] == 0.0
    assert bundle.t[-1] == 5.0
    assert bundle.t.shape == (11,)
