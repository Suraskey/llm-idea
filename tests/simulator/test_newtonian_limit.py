"""Newtonian-limit correctness (D-02): simulator vs an independent DOP853 oracle.

Plain English: when the alien corrections are switched off (α=0, β=0, charges=0)
the dynamics MUST collapse to pure Newtonian gravity. We prove it by integrating
the same initial condition two ways — once with the production simulator
(implicit-midpoint in the canonical-polar coordinate) and once with an
independent, high-order, ADAPTIVE scipy DOP853 integration of the analytic
Kepler RHS — and asserting the two relative-coordinate orbits agree to < 1e-3
RMSE.

⚠️ TEST-TIME ORACLE ONLY: ``scipy.integrate.solve_ivp(method="DOP853")`` is an
independent cross-check here and NEVER enters the production path (RESEARCH
rejects adaptive solvers for the conservation bar + bit-identity; T-15.0-05).
The oracle is valid ONLY because the fixture sets α=0 — with α≠0 the orbit is
non-Keplerian (RESEARCH Q3) and a Kepler-ellipse oracle would be wrong.

Tolerance rationale (mirror tests/benchmarks/test_score_numerical.py): the bar
is D-02's 1e-3 RMSE. Observed this session is ~1.9e-5 (≈50× margin); we assert
< 1e-3 to track the contract, not the incidental tightness.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ascension.simulator.fixtures import newtonian_limit_fixture

# D-02 contract: Newtonian limit reproduces pure Newton to this RMSE.
NEWTONIAN_LIMIT_RMSE_TOL = 1e-3


def test_newtonian_limit() -> None:
    """α=0, β=0, charges=0 reproduces pure Newton to < 1e-3 RMSE (D-02).

    The simulator integrates the full Q-consistent machinery with the alien
    corrections zeroed; the oracle integrates the analytic Kepler RHS for the
    SAME relative-coordinate IC. With the corrections off these must coincide.
    """
    cfg, traj = newtonian_limit_fixture()

    # Sanity: the limit fixture really has the corrections off (oracle validity).
    assert cfg.alpha == 0.0
    assert cfg.beta == 0.0
    assert all(s == 0.0 for s in cfg.charges)

    # Simulator relative coordinate (body1 relative to body0).
    rel_pos = traj.positions[:, 1, :] - traj.positions[:, 0, :]
    rel_vel = traj.velocities[:, 1, :] - traj.velocities[:, 0, :]
    assert np.all(np.isfinite(rel_pos))

    # Independent Kepler oracle for the relative coordinate. The reduced-mass
    # equation mu·r'' = −G m0 m1 / r² · r̂ gives r'' = −GM_eff/r² · r̂ with
    # GM_eff = G·m0·m1/mu = G·(m0+m1) (the standard two-body relative motion).
    m = np.asarray(cfg.masses, dtype=np.float64)
    mu = float(m[0] * m[1] / (m[0] + m[1]))
    gm_eff = cfg.G * m[0] * m[1] / mu

    def kepler_rhs(_t: float, y: np.ndarray) -> list[float]:
        x, yy, vx, vy = y
        r = np.hypot(x, yy)
        return [vx, vy, -gm_eff * x / r**3, -gm_eff * yy / r**3]

    y0 = [rel_pos[0, 0], rel_pos[0, 1], rel_vel[0, 0], rel_vel[0, 1]]
    sol = solve_ivp(
        kepler_rhs,
        (float(traj.t[0]), float(traj.t[-1])),
        y0,
        t_eval=traj.t,
        method="DOP853",  # TEST-TIME ORACLE ONLY — never the production path.
        rtol=1e-12,
        atol=1e-12,
    )
    assert sol.success, f"DOP853 oracle failed: {sol.message}"
    oracle = sol.y[:2].T  # (n, 2)
    assert np.all(np.isfinite(oracle))

    rmse = float(np.sqrt(np.mean((rel_pos - oracle) ** 2)))
    assert rmse < NEWTONIAN_LIMIT_RMSE_TOL, (
        f"Newtonian-limit RMSE {rmse:.3e} exceeds {NEWTONIAN_LIMIT_RMSE_TOL:.0e} "
        "vs the DOP853 Kepler oracle (D-02)"
    )
