"""Wave-3 bit-identical reproducibility at the public ``run()`` entrypoint (D-06).

Plain English: the same recipe (``AlienConfig``) run twice through the REAL
``AlienUniverse.run()`` must produce trajectories that are bit-for-bit identical —
not "close" (``allclose``), but exactly equal (``np.array_equal``). This is the
SCOPE §22.2 reproducibility contract at the assembly level, the partner to the
integrator-level bit-identity already proved in
``test_integrator_conservation.py``. It exercises every code path the production
simulator runs: the canonical-polar reduction, the implicit-midpoint loop, the
Cartesian reconstruction, and the Q series.

RESEARCH D-06 + benchmarks Pitfall #4 (test_generate.py:42-57): assert with
``np.testing.assert_array_equal`` — NEVER ``allclose``. A float-noise branch in
the inner fixed-point solve, an unseeded RNG draw, or a non-float64 cast would
break exact equality even when ``allclose`` would mask it.

Binding: 15.0 PLAN Task 1 <behavior>; property→test-map row 6
(``test_reproducibility.py::test_bit_identical_reproducibility``). $0 LLM.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.simulator.alien import AlienUniverse
from ascension.simulator.fixtures import reproducibility_fixture


def test_bit_identical_reproducibility() -> None:
    """D-06: same config run twice through ``run()`` → bit-identical arrays.

    Uses ``reproducibility_fixture`` (full alien physics, fixed seed) so the
    bit-identity guarantee covers Newton + α-correction + hidden-charge +
    tangential-inertia code paths, the polar reduction, and the Cartesian
    reconstruction. ``np.array_equal`` (exact), NOT ``allclose`` (RESEARCH D-06,
    test_generate.py:42-57).
    """
    cfg, _traj_throwaway = reproducibility_fixture()

    a = AlienUniverse(cfg).run()
    b = AlienUniverse(cfg).run()

    # Float64 throughout — a non-float64 cast would silently break bit-identity.
    assert a.positions.dtype == np.float64
    assert b.positions.dtype == np.float64

    # Bit-for-bit, every emitted array (positions, velocities, Q, t, masses).
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.velocities, b.velocities)
    np.testing.assert_array_equal(a.masses, b.masses)
    np.testing.assert_array_equal(a.charges, b.charges)
    np.testing.assert_array_equal(a.Q, b.Q)


def test_reproducibility_is_array_equal_not_merely_allclose() -> None:
    """Guard the discipline itself: the trajectory is EXACTLY equal, not merely
    close. A regression that introduced float-noise branching could still pass
    ``allclose`` while failing ``array_equal`` — assert the strict property holds
    so the weaker check can never silently substitute for it (D-06).
    """
    cfg, _ = reproducibility_fixture()
    a = AlienUniverse(cfg).run()
    b = AlienUniverse(cfg).run()
    # array_equal is True (the contract). allclose is necessarily also True, but
    # it is the WEAKER claim; we pin the strong one.
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.Q, b.Q)


def test_different_config_produces_different_trajectory() -> None:
    """A perturbed config (different alpha) must NOT reproduce the baseline.

    Non-vacuity guard for the reproducibility claim: if changing the physics did
    not change the trajectory, bit-identity would be a trivial constant. Bump
    alpha and assert the position arrays genuinely differ.
    """
    cfg, _ = reproducibility_fixture()
    bumped = dataclasses.replace(cfg, alpha=cfg.alpha + 0.02)
    base = AlienUniverse(cfg).run()
    other = AlienUniverse(bumped).run()
    assert not np.array_equal(base.positions, other.positions), (
        "perturbing alpha left the trajectory unchanged — reproducibility " "would be vacuous"
    )
