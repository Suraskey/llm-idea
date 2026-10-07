"""Wave-3 branch-coverage tests for the simulator core (physics/alien/integrator).

These exercise REAL branches the analytical/symmetry/audit tests don't reach —
not trivial padding (15.0 PLAN Task 2 coverage clause):

  - ``physics.conserved_Q`` directly (the Cartesian-coordinate Q form). The
    assembly path reads ``traj.Q`` (the polar-Hamiltonian invariant), so the
    Cartesian ``conserved_Q`` is never hit by ``run()``; it is still a public
    function with a real contract (Q = radial + tangential-inertia kinetic + U)
    and deserves a direct test.
  - ``physics.v_perp_squared`` raising ValueError on a non-2-body input (15.0
    ships the 2-body case; general-N v_perp is deferred).
  - ``AlienUniverse._validate`` ValueError branches (n_bodies, ragged arrays,
    wrong integrator, bad dt/n_steps) — loud rejection, never silent.
  - The zero-separation degenerate IC raising ValueError.
  - ``integrator.integrate`` raising SimulatorIntegrationError on a divergent
    config (blowup → non-finite state, the loud host-level failure, T-15.0-04).

Binding: 15.0 PLAN Task 2 coverage clause. $0 LLM — deterministic numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from ascension.simulator import physics as P
from ascension.simulator.alien import AlienUniverse
from ascension.simulator.exceptions import SimulatorIntegrationError
from ascension.simulator.integrator import integrate
from ascension.simulator.types import AlienConfig


def _cfg(**overrides) -> AlienConfig:
    base = dict(
        G=1.0,
        alpha=0.05,
        beta=0.02,
        gamma=0.7,
        kappa=0.3,
        masses=(1.0, 1.0),
        charges=(1.0, 1.0),
        ics_pos=((0.0, 0.0), (1.0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 0.9)),
        dt=0.005,
        n_steps=100,
        seed=0,
    )
    base.update(overrides)
    return AlienConfig(**base)


# --------------------------------------------------------------------------
# physics.conserved_Q (the Cartesian-coordinate Q form, lines 281-302)
# --------------------------------------------------------------------------


def test_conserved_q_is_kinetic_plus_potential() -> None:
    """conserved_Q = Σ[½m·v_radial² + ½(m+κs²)·v_perp²] + U(r) on a known state.

    Hand-build a 2-body Cartesian state and assert conserved_Q equals the sum of
    the radial kinetic, tangential-inertia kinetic, and potential energies
    computed independently from the same physics primitives.
    """
    cfg = _cfg()
    pos = np.array([[0.0, 0.0], [1.5, 0.0]], dtype=np.float64)
    vel = np.array([[0.0, 0.0], [0.3, 0.7]], dtype=np.float64)
    masses = np.asarray(cfg.masses, dtype=np.float64)
    charges = np.asarray(cfg.charges, dtype=np.float64)

    q = P.conserved_Q(pos, vel, masses, charges, cfg)

    # Independent recomputation of the same decomposition.
    vp2 = P.v_perp_squared(pos, vel)
    kinetic = 0.0
    for i in range(2):
        other = 1 - i
        rel = pos[i] - pos[other]
        r = np.sqrt(np.dot(rel, rel))
        rhat = rel / r
        v_radial = np.dot(vel[i], rhat)
        m = masses[i]
        s2 = charges[i] * charges[i]
        kinetic += 0.5 * m * v_radial * v_radial
        kinetic += 0.5 * (m + cfg.kappa * s2) * vp2[i]
    u = P.potential_U(pos, masses, charges, cfg)
    assert np.isclose(q, kinetic + u, atol=1e-12)


def test_conserved_q_kappa_increases_tangential_energy() -> None:
    """A larger κ raises conserved_Q (more tangential inertia) at a moving state.

    The tangential-inertia term ½(m+κs²)v_perp² grows with κ, so for a state with
    non-zero v_perp a bigger κ gives a bigger Q. Confirms the κ coupling is wired
    into conserved_Q (not dropped)."""
    pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    vel = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)  # pure tangential
    masses = np.asarray((1.0, 1.0), dtype=np.float64)
    charges = np.asarray((1.0, 1.0), dtype=np.float64)
    q_lo = P.conserved_Q(pos, vel, masses, charges, _cfg(kappa=0.1))
    q_hi = P.conserved_Q(pos, vel, masses, charges, _cfg(kappa=0.9))
    assert q_hi > q_lo


def test_v_perp_squared_rejects_non_two_body() -> None:
    """v_perp_squared raises ValueError for n != 2 (15.0 ships the 2-body case)."""
    pos = np.zeros((3, 2), dtype=np.float64)
    vel = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="2-body"):
        P.v_perp_squared(pos, vel)


# --------------------------------------------------------------------------
# AlienUniverse._validate loud-rejection branches (alien.py 96/101/107/112)
# --------------------------------------------------------------------------


def test_validate_rejects_non_two_body() -> None:
    with pytest.raises(ValueError, match="2-body"):
        AlienUniverse(
            _cfg(
                masses=(1.0, 1.0, 1.0),
                charges=(1.0, 1.0, 1.0),
                ics_pos=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
                ics_vel=((0.0, 0.0), (0.0, 0.9), (0.0, 0.5)),
            )
        )


def test_validate_rejects_ragged_arrays() -> None:
    with pytest.raises(ValueError, match="must all have length"):
        AlienUniverse(_cfg(charges=(1.0,)))


def test_validate_rejects_unknown_integrator() -> None:
    with pytest.raises(ValueError, match="implicit_midpoint"):
        AlienUniverse(_cfg(integrator="rk45"))


def test_validate_rejects_bad_dt_and_nsteps() -> None:
    with pytest.raises(ValueError, match="dt must be"):
        AlienUniverse(_cfg(dt=0.0))
    with pytest.raises(ValueError, match="dt must be"):
        AlienUniverse(_cfg(n_steps=0))


def test_run_rejects_zero_separation_ic() -> None:
    """A degenerate IC with both bodies at the same point raises ValueError."""
    cfg = _cfg(ics_pos=((1.0, 1.0), (1.0, 1.0)))
    with pytest.raises(ValueError, match="initial separation is zero"):
        AlienUniverse(cfg).run()


# --------------------------------------------------------------------------
# integrator loud blowup (integrator.py:135 — SimulatorIntegrationError)
# --------------------------------------------------------------------------


def test_integrate_raises_on_nonfinite_blowup() -> None:
    """A divergent rhs drives the state non-finite → SimulatorIntegrationError.

    The integrator must surface a blowup LOUD (host-level exception), never a
    silent NaN-filled array (T-15.0-04, CLAUDE.md no-cover-ups)."""
    y0 = np.array([1.0], dtype=np.float64)

    def divergent_rhs(y: np.ndarray) -> np.ndarray:
        # Explosive positive feedback: |y| grows ~e^t per step, overflowing to
        # inf within a handful of large steps.
        return 1e308 * np.sign(y) * (np.abs(y) + 1.0)

    cfg = _cfg()
    with pytest.raises(SimulatorIntegrationError) as exc:
        integrate(y0, dt=10.0, n_steps=50, rhs=divergent_rhs, config=cfg, seed=0)
    assert exc.value.seed == 0
    assert exc.value.config is cfg


def test_run_defense_in_depth_nonfinite(monkeypatch) -> None:
    """alien.run() re-checks finiteness after integration (defense-in-depth).

    The integrator already guards, but alien.run() asserts finiteness again
    (alien.py:214). Patch ``integrate`` to return a NaN-laced polar array and
    confirm run() raises SimulatorIntegrationError rather than emitting a NaN
    trajectory."""
    import ascension.simulator.alien as alien_mod

    cfg = _cfg(n_steps=10)

    def _nan_integrate(y0, dt, n_steps, rhs, *, config=None, seed=0, **kw):  # noqa: ANN001
        arr = np.zeros((n_steps + 1, 4), dtype=np.float64)
        arr[:] = y0
        arr[-1, 0] = np.nan  # inject a non-finite after the integrator "succeeds"
        return arr

    monkeypatch.setattr(alien_mod, "integrate", _nan_integrate)
    with pytest.raises(SimulatorIntegrationError, match="non-finite"):
        AlienUniverse(cfg).run()
