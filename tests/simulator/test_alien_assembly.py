"""Wave-2 assembly tests: the AlienUniverse system object + analytical fixtures.

Plain English: Wave 1 proved the physics and integrator in isolation. This
module exercises the assembled system: ``AlienUniverse.run()`` must integrate
forward from initial conditions, emit a FULL ``AlienTrajectory`` (positions,
velocities, charges, Q, library_versions) AND a DATA-ONLY ``ObservationBundle``
that structurally cannot leak hidden state. The five analytical-orbit fixtures
must all drive the REAL ``run()`` entrypoint (mirror benchmarks/fixtures.py:
producers that go through the real generator).

Binding decisions:
  - 15.0 RESEARCH §Architecture Patterns / Pattern 3 (hidden/observable split):
    the ObservationBundle field set is exactly {t, positions, velocities,
    masses}; it is built by EXPLICIT field selection, never ``asdict`` of the
    full trajectory (RESEARCH Pitfall 4 — the leak path).
  - T-15.0-01 mitigation: the bundle has no charges/Q attribute at the type
    level (structurally incapable of leaking; the runtime leak-guard test is
    Wave 3).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ascension.simulator import fixtures as F
from ascension.simulator.alien import AlienUniverse
from ascension.simulator.types import (
    AlienConfig,
    AlienTrajectory,
    ObservationBundle,
)


def _newtonian_limit_cfg() -> AlienConfig:
    """A small Newtonian-limit config (α=0, β=0, charges=0) for fast assembly checks."""
    return AlienConfig(
        G=1.0,
        alpha=0.0,
        beta=0.0,
        gamma=0.7,
        kappa=0.0,
        masses=(1.0, 1.0),
        charges=(0.0, 0.0),
        ics_pos=((0.0, 0.0), (1.0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 1.0)),
        dt=0.005,
        n_steps=200,
        seed=0,
    )


def test_run_emits_full_trajectory() -> None:
    """run(cfg) returns a populated AlienTrajectory with library_versions."""
    cfg = _newtonian_limit_cfg()
    traj = AlienUniverse(cfg).run()

    assert isinstance(traj, AlienTrajectory)
    n = cfg.n_steps + 1
    assert traj.t.shape == (n,)
    assert traj.positions.shape == (n, 2, 2)
    assert traj.velocities.shape == (n, 2, 2)
    assert traj.charges.shape == (2,)
    assert traj.Q.shape == (n,)
    # Everything finite (no silent NaN — loud blowup would have raised).
    assert np.all(np.isfinite(traj.positions))
    assert np.all(np.isfinite(traj.velocities))
    assert np.all(np.isfinite(traj.Q))
    # Library versions pinned (SCOPE §22.2) — scipy/sympy/numpy present.
    for lib in ("numpy", "scipy", "sympy"):
        assert lib in traj.library_versions
        assert isinstance(traj.library_versions[lib], str)
        assert traj.library_versions[lib]


def test_observation_bundle_is_data_only() -> None:
    """The emitted bundle has EXACTLY {t, positions, velocities, masses}.

    It must NOT carry charges or Q, and it must NOT be producible from
    ``dataclasses.asdict`` of the full trajectory (the leak path, RESEARCH
    Pitfall 4). We assert the field set and the absence of any hidden attribute.
    """
    cfg = _newtonian_limit_cfg()
    universe = AlienUniverse(cfg)
    traj = universe.run()
    bundle = universe.to_observation_bundle(traj)

    assert isinstance(bundle, ObservationBundle)

    field_names = {f.name for f in dataclasses.fields(bundle)}
    assert field_names == {"t", "positions", "velocities", "masses"}

    # Structurally incapable of leaking: no charges / Q / config / law attribute.
    for forbidden in ("charges", "Q", "config", "law", "library_versions"):
        assert not hasattr(
            bundle, forbidden
        ), f"ObservationBundle leaked a hidden attribute: {forbidden!r}"

    # The full-trajectory asdict carries the hidden fields — proving the bundle
    # was NOT built from it (if it were, charges/Q would be present).
    full = dataclasses.asdict(traj)
    assert "charges" in full and "Q" in full

    # The observable data matches the trajectory's observable slice (sanity:
    # explicit selection copied the right arrays).
    np.testing.assert_array_equal(bundle.t, traj.t)
    np.testing.assert_array_equal(bundle.positions, traj.positions)
    np.testing.assert_array_equal(bundle.velocities, traj.velocities)
    np.testing.assert_array_equal(bundle.masses, traj.masses)


@pytest.mark.parametrize(
    "fixture_fn",
    [
        F.circular_orbit_fixture,
        F.parabolic_orbit_fixture,
        F.newtonian_limit_fixture,
        F.eccentric_fixture,
        F.reproducibility_fixture,
    ],
)
def test_fixtures_run_through_real_entrypoint(fixture_fn) -> None:
    """Each fixture returns a valid (AlienConfig, AlienTrajectory) via run()."""
    cfg, traj = fixture_fn()
    assert isinstance(cfg, AlienConfig)
    assert isinstance(traj, AlienTrajectory)
    n = cfg.n_steps + 1
    assert traj.positions.shape == (n, 2, 2)
    assert np.all(np.isfinite(traj.positions))
    assert np.all(np.isfinite(traj.Q))


def test_reproducibility_fixture_is_bit_identical() -> None:
    """reproducibility_fixture runs the same config twice → np.array_equal (D-06)."""
    cfg, traj_a = F.reproducibility_fixture()
    _cfg_b, traj_b = F.reproducibility_fixture()
    np.testing.assert_array_equal(traj_a.positions, traj_b.positions)
    np.testing.assert_array_equal(traj_a.velocities, traj_b.velocities)
    np.testing.assert_array_equal(traj_a.Q, traj_b.Q)
