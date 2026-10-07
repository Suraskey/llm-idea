"""Analytical-orbit correctness (D-03): circular holds r₀; parabolic escapes at E≈0.

Plain English: closed-form orbits are the ground truth a conservation test
alone can miss (D-03 / RESEARCH Q3). Two anchors:
  - Circular: initialized at the RESEARCH Q3 balance speed
    ``v_circ = sqrt(r₀·dU/dr|_{r₀}/mu)``, the separation r(t) must NOT drift
    radially — a circular orbit stays on its circle.
  - Parabolic (E=0 marginal escape): initialized at ``v_esc = sqrt(−2U(r)/mu)``
    pointing outward, the trajectory must NOT return (r monotonically increasing
    past a horizon) AND the conserved quantity must stay ≈ 0 throughout (the
    E=0 signature).

These exercise the assembled ``AlienUniverse.run()`` through the Task-1 fixtures.
RESEARCH Q3 closed forms are cited in each test.
"""

from __future__ import annotations

import numpy as np

from ascension.simulator.fixtures import (
    circular_orbit_fixture,
    parabolic_orbit_fixture,
)

# A circular orbit must hold its radius to a tight tolerance over the run. The
# implicit-midpoint integrator keeps r essentially at machine precision for the
# exact-balance v_circ IC; 1e-6 is a generous contract bound (RESEARCH Q3).
CIRCULAR_RADIUS_TOL = 1e-6
# Parabolic (E=0) marginal escape: |Q| stays near zero. The IC sets E=0 from the
# single-source potential; residual is finite-difference / step error only.
PARABOLIC_ENERGY_TOL = 1e-3


def _separation(traj) -> np.ndarray:
    """2-body separation |r₁ − r₀| over time."""
    rel = traj.positions[:, 1, :] - traj.positions[:, 0, :]
    return np.hypot(rel[:, 0], rel[:, 1])


def test_circular_orbit() -> None:
    """Circular IC keeps r(t) within tolerance of r₀ (D-03, RESEARCH Q3).

    With ``v_circ`` from the radial-force balance the orbit is a circle; any
    radial drift would signal an integrator or force-balance bug.
    """
    _cfg, traj = circular_orbit_fixture()
    r = _separation(traj)
    r0 = r[0]
    drift = float(np.max(np.abs(r - r0)))
    assert np.all(np.isfinite(r))
    assert drift < CIRCULAR_RADIUS_TOL, (
        f"circular orbit drifted radially by {drift:.3e} (> {CIRCULAR_RADIUS_TOL:.0e}); "
        "v_circ balance broken"
    )


def test_parabolic_orbit() -> None:
    """E=0 IC → non-returning trajectory with Q≈0 throughout (D-03, RESEARCH Q3).

    The marginal-escape speed ``v_esc = sqrt(−2U(r)/mu)`` gives total energy ≈ 0,
    so the body climbs out of the well without turning back: r increases
    monotonically past a horizon, and the conserved quantity stays near zero.
    """
    _cfg, traj = parabolic_orbit_fixture()
    r = _separation(traj)
    assert np.all(np.isfinite(r))

    # Non-returning: r increases monotonically (escape, never a turning point).
    assert np.all(np.diff(r) > 0), "parabolic orbit returned (r not monotonic)"
    # Past a clear horizon (well beyond the starting radius).
    assert (
        r[-1] > 2.0 * r[0]
    ), f"parabolic orbit did not escape far enough: r0={r[0]:.3f} -> r_end={r[-1]:.3f}"

    # E ≈ 0 throughout (the marginal-escape signature). Q is the trajectory's
    # conserved quantity; in the Newtonian-limit parabolic case it equals the
    # mechanical energy, which the IC pins to ~0.
    assert np.all(np.isfinite(traj.Q))
    assert float(np.max(np.abs(traj.Q))) < PARABOLIC_ENERGY_TOL, (
        f"parabolic energy |Q|max={np.max(np.abs(traj.Q)):.3e} not ~0 "
        f"(> {PARABOLIC_ENERGY_TOL:.0e})"
    )
