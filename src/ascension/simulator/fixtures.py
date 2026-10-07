"""Canonical analytical-orbit + reproducibility fixtures for the simulator.

Plain English: every correctness test needs a "known" configuration whose answer
is dictated by closed-form physics — a circle, a marginal-escape parabola, the
pure-Newton limit, an eccentric orbit, and a run-twice reproducibility pair.
Each fixture builds an ``AlienConfig`` and drives it through the REAL
``AlienUniverse.run()`` entrypoint (mirror benchmarks/fixtures.py:40-112 —
producers that go through the real generator, not hand-assembled trajectories),
returning a ``(AlienConfig, AlienTrajectory)`` pair.

The circular / parabolic initial speeds use the RESEARCH Q3 closed forms:
  - circular at r₀:  v_circ = sqrt(r₀ · g(r₀) / mu)  where g(r₀) = dU/dr|_{r₀}
    (the radial force magnitude). With α=0 this is the textbook sqrt(GM/r₀).
  - parabolic (E=0): v_esc(r) = sqrt(−2·U(r)/mu)  (marginal escape).
These are computed from the SAME ``physics.potential_U`` the integrator uses, so
the IC and the dynamics can never disagree (single source of truth).

⚠️ Newtonian-limit discipline: ``newtonian_limit_fixture`` sets α=0, β=0,
charges=0 so the analytic Kepler oracle (Task 3) is VALID. With α≠0 the orbit is
non-Keplerian (RESEARCH Q3) and a Kepler-ellipse oracle would be wrong — do NOT
wire α-on into a Kepler comparison.

Binding decisions:
  - 15.0 RESEARCH Q3: circular / parabolic closed-form IC conditions.
  - L-018 cycle guard: import ONLY from the simulator public surface
    (``alien``, ``types``, ``physics``) — zero heavy / agent imports.

Threat-model mitigations: N/A — pure-compute fixture surface.
"""

from __future__ import annotations

import numpy as np

from ascension.simulator import physics as P
from ascension.simulator.alien import AlienUniverse
from ascension.simulator.types import AlienConfig, AlienTrajectory


def _reduced_mass(masses: tuple[float, ...]) -> float:
    """Reduced mass mu = m0·m1/(m0+m1) for the 2-body relative coordinate."""
    m = np.asarray(masses, dtype=np.float64)
    return float(m[0] * m[1] / (m[0] + m[1]))


def _U_and_dUdr(cfg: AlienConfig):
    """Return ``(U, dUdr)`` closures over the 2-body separation r (single source).

    Both come from ``physics.potential_U`` so a fixture's IC speed is consistent
    with the force the integrator will apply.
    """
    m = np.asarray(cfg.masses, dtype=np.float64)
    ch = np.asarray(cfg.charges, dtype=np.float64)

    def U(r: float) -> float:
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, cfg)

    def dUdr(r: float) -> float:
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    return U, dUdr


def _two_body_config(
    *,
    r0: float,
    rel_speed: float,
    tangential: bool,
    alpha: float,
    beta: float,
    gamma: float,
    kappa: float,
    charges: tuple[float, ...],
    masses: tuple[float, ...],
    dt: float,
    n_steps: int,
    G: float = 1.0,
    seed: int = 0,
) -> AlienConfig:
    """Assemble a 2-body AlienConfig from a relative speed at separation r₀.

    body0 sits at the origin, body1 at ``(r0, 0)``. The relative velocity is
    purely tangential ``(0, rel_speed)`` for orbits, or purely radial
    ``(rel_speed, 0)`` for escape. ``run()`` reduces this to the canonical-polar
    coordinate internally; we only need the Cartesian IC here.
    """
    if tangential:
        vel1 = (0.0, rel_speed)
    else:
        vel1 = (rel_speed, 0.0)
    return AlienConfig(
        G=G,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kappa=kappa,
        masses=masses,
        charges=charges,
        ics_pos=((0.0, 0.0), (r0, 0.0)),
        ics_vel=((0.0, 0.0), vel1),
        dt=dt,
        n_steps=n_steps,
        seed=seed,
    )


def circular_orbit_fixture() -> tuple[AlienConfig, AlienTrajectory]:
    """Circular orbit at r₀=1.0 (Newtonian-limit params, α optionally on=0 here).

    RESEARCH Q3: balance ``mu·v²/r₀ = g(r₀)`` ⇒ ``v_circ = sqrt(r₀·dU/dr|_{r₀}/mu)``.
    Serves ``test_circular_orbit`` (D-03): r(t) must stay within tolerance of r₀
    over the run — a circular orbit cannot drift radially. Equal unit masses keep
    v_circ ~ 1 so dt=0.005 is in the RESEARCH-validated regime.
    """
    r0 = 1.0
    masses = (1.0, 1.0)
    charges = (0.0, 0.0)
    cfg_probe = _two_body_config(
        r0=r0,
        rel_speed=0.0,
        tangential=True,
        alpha=0.0,
        beta=0.0,
        gamma=0.7,
        kappa=0.0,
        charges=charges,
        masses=masses,
        dt=0.005,
        n_steps=1200,
    )
    mu = _reduced_mass(masses)
    _U, dUdr = _U_and_dUdr(cfg_probe)
    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    cfg = _two_body_config(
        r0=r0,
        rel_speed=v_circ,
        tangential=True,
        alpha=0.0,
        beta=0.0,
        gamma=0.7,
        kappa=0.0,
        charges=charges,
        masses=masses,
        dt=0.005,
        n_steps=1200,
    )
    return cfg, AlienUniverse(cfg).run()


def parabolic_orbit_fixture() -> tuple[AlienConfig, AlienTrajectory]:
    """Marginal-escape (E=0) orbit at r₀=1.0 pointing outward.

    RESEARCH Q3: ``v_esc(r) = sqrt(−2·U(r)/mu)`` gives total energy E≈0, so the
    trajectory does not return (r monotonically increasing past a horizon).
    Serves ``test_parabolic_orbit`` (D-03). Newtonian-limit params so U is the
    pure ``−GM/r`` well and the E=0 condition is exact.
    """
    r0 = 1.0
    masses = (1.0, 1.0)
    charges = (0.0, 0.0)
    cfg_probe = _two_body_config(
        r0=r0,
        rel_speed=0.0,
        tangential=False,
        alpha=0.0,
        beta=0.0,
        gamma=0.7,
        kappa=0.0,
        charges=charges,
        masses=masses,
        dt=0.005,
        n_steps=600,
    )
    mu = _reduced_mass(masses)
    U, _dUdr = _U_and_dUdr(cfg_probe)
    v_esc = float(np.sqrt(-2.0 * U(r0) / mu))
    # Point outward (radial) so the body climbs out of the well marginally.
    cfg = _two_body_config(
        r0=r0,
        rel_speed=v_esc,
        tangential=False,
        alpha=0.0,
        beta=0.0,
        gamma=0.7,
        kappa=0.0,
        charges=charges,
        masses=masses,
        dt=0.005,
        n_steps=600,
    )
    return cfg, AlienUniverse(cfg).run()


def newtonian_limit_fixture() -> tuple[AlienConfig, AlienTrajectory]:
    """Pure-Newton limit: α=0, β=0, charges=0 (Kepler oracle is VALID here).

    Serves ``test_newtonian_limit`` (D-02): with the alien corrections off the
    dynamics reduce to ``mu·v² = GM/r²`` central-force motion, so an independent
    DOP853 integration of the analytic Kepler RHS is a legitimate oracle. A
    moderately eccentric IC (tangential speed 0.85·v_circ) gives a real ellipse
    (not a degenerate circle) for a meaningful RMSE comparison.
    """
    r0 = 1.0
    masses = (1.0, 1.0)
    charges = (0.0, 0.0)
    cfg_probe = _two_body_config(
        r0=r0,
        rel_speed=0.0,
        tangential=True,
        alpha=0.0,
        beta=0.0,
        gamma=0.7,
        kappa=0.0,
        charges=charges,
        masses=masses,
        dt=0.005,
        n_steps=1000,
    )
    mu = _reduced_mass(masses)
    _U, dUdr = _U_and_dUdr(cfg_probe)
    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    cfg = _two_body_config(
        r0=r0,
        rel_speed=0.85 * v_circ,
        tangential=True,
        alpha=0.0,
        beta=0.0,
        gamma=0.7,
        kappa=0.0,
        charges=charges,
        masses=masses,
        dt=0.005,
        n_steps=1000,
    )
    return cfg, AlienUniverse(cfg).run()


def eccentric_fixture() -> tuple[AlienConfig, AlienTrajectory]:
    """Eccentric Q-consistent orbit (the Wave-1 make-or-break Q recipe).

    Tangential speed 0.8·v_circ with the FULL alien physics on (α, β, κ, unit
    charges) — the canonical Q-conservation recipe (RESEARCH Q-conservation +
    Wave-1 conftest ``eccentric_fixture``). Reused here as a public-surface
    fixture so Wave-2/3 tests can drive it through the real ``run()`` entrypoint
    without re-deriving the IC. dt=0.005 sits in the RESEARCH-validated regime.
    """
    r0 = 1.0
    masses = (1.0, 1.0)
    charges = (1.0, 1.0)
    cfg_probe = _two_body_config(
        r0=r0,
        rel_speed=0.0,
        tangential=True,
        alpha=0.05,
        beta=0.02,
        gamma=0.7,
        kappa=0.3,
        charges=charges,
        masses=masses,
        dt=0.005,
        n_steps=1000,
    )
    # v_circ uses the reduced-mass radial balance against the FULL potential.
    mu = _reduced_mass(masses)
    _U, dUdr = _U_and_dUdr(cfg_probe)
    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    cfg = _two_body_config(
        r0=r0,
        rel_speed=0.8 * v_circ,
        tangential=True,
        alpha=0.05,
        beta=0.02,
        gamma=0.7,
        kappa=0.3,
        charges=charges,
        masses=masses,
        dt=0.005,
        n_steps=1000,
    )
    return cfg, AlienUniverse(cfg).run()


def reproducibility_fixture() -> tuple[AlienConfig, AlienTrajectory]:
    """A deterministic config run through ``run()`` — for bit-identity tests (D-06).

    Calling this twice and comparing the trajectories with ``np.array_equal``
    must hold: the integrator is deterministic (fixed-step, fixed inner-iteration
    cap, no RNG draw on deterministic ICs), float64 throughout. Uses the full
    alien physics so reproducibility covers every code path.
    """
    cfg = _two_body_config(
        r0=1.0,
        rel_speed=0.85,
        tangential=True,
        alpha=0.05,
        beta=0.02,
        gamma=0.7,
        kappa=0.3,
        charges=(1.0, 1.0),
        masses=(1.0, 1.0),
        dt=0.005,
        n_steps=400,
        seed=12345,
    )
    return cfg, AlienUniverse(cfg).run()
