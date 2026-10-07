"""AlienUniverse system object — assembles forces + integrator, runs forward.

Implements the Q-consistent §10 formulation (tangential-inertia coupling),
reconciled into SCOPE §10 (2026-05-23).

Plain English: this is the "machine" that turns an ``AlienConfig`` recipe into a
trajectory. It wires the physics force laws (``physics.py``) and the
deterministic integrator (``integrator.py``) together, runs the system forward
from its initial conditions, and produces TWO outputs:
  - ``AlienTrajectory`` (FULL): positions, velocities, the hidden charges, the
    conserved quantity Q over time, and the pinned library versions.
  - ``ObservationBundle`` (DATA-ONLY): positions, velocities, masses, time —
    and NOTHING else. This is the only thing the Council (Phase 15.2) ever sees.

``alien.py`` is the SINGLE place an ``ObservationBundle`` is constructed, and it
does so by EXPLICIT positional field selection (``to_observation_bundle``),
never by ``dataclasses.asdict`` of the full trajectory (RESEARCH Pitfall 4 — a
future field addition would silently re-leak the hidden charges/Q).

Why the canonical-polar relative coordinate (lifted from the Wave-1 conftest):
the Q-consistent dynamics have a position-dependent tangential effective inertia
``m_t = mu + κ·s₁s₂`` (the §10 correction), making the Hamiltonian NON-SEPARABLE.
RESEARCH Q2 + the Wave-1 make-or-break proved that integrating the 2-body
RELATIVE coordinate in canonical-polar state ``[r, θ, p_r, p_θ]`` — with the
radial force taken from ``physics.potential_U``'s OWN gradient (single source of
truth) — holds Q-drift to ~1e-10. An independently hand-coded ``dU/dr`` produced
a fixed ~8e-3 offset (Wave-1 LEARNINGS). The integration therefore happens in
polar; the FULL Cartesian positions/velocities are reconstructed in the
centre-of-mass frame for the trajectory + bundle. ``AlienTrajectory.Q`` carries
the true polar-Hamiltonian invariant (the quantity the integrator preserves);
the per-body Cartesian ``physics.conserved_Q`` reconciliation is the Wave-3
audit's job, not this assembly's.

Binding decisions:
  - 15.0 RESEARCH §Architecture Patterns / Pattern 3: hidden/observable type
    split; ObservationBundle by explicit field selection.
  - 15.0 RESEARCH Q4: v_perp is the velocity component orthogonal to the radial
    direction (the polar ``p_θ``/``m_t r²`` term).
  - benchmarks/ode.py:125-194: seed ``np.random.default_rng(seed)`` (never
    ``np.random.seed``); draw any IC jitter BEFORE the loop; pin library
    versions via ``importlib.metadata.version`` (ode.py:189-194).
  - benchmarks/ode.py:157-163 + integrator.py: a non-finite state raises
    ``SimulatorIntegrationError`` (loud, no cover-ups — CLAUDE.md).

Threat-model mitigations: T-15.0-01 (information disclosure) —
``to_observation_bundle`` is the single construction site, explicit selection.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_metadata_version

import numpy as np

from ascension.simulator import physics as P
from ascension.simulator.exceptions import SimulatorIntegrationError
from ascension.simulator.integrator import integrate
from ascension.simulator.types import AlienConfig, AlienTrajectory, ObservationBundle


def _pkg_version(name: str) -> str:
    """Best-effort installed-version string (mirror benchmarks/ode.py:_pkg_version)."""
    try:
        return _pkg_metadata_version(name)
    except PackageNotFoundError:  # pragma: no cover - all three are locked deps
        return "unknown"


class AlienUniverse:
    """A runnable 2-body Alien-Universe system assembled from an ``AlienConfig``.

    Plain English: hand it a recipe (``AlienConfig``), call ``run()``, get the
    full trajectory back. Call ``to_observation_bundle(traj)`` to get the
    data-only view the Council is allowed to see.

    15.0 ships the 2-body case (the analytical anchors and Q-consistency proof
    are 2-body; RESEARCH Q8 keeps the FORCES N-body but the analytical
    validation and the canonical-polar reduction are unambiguous only for two
    bodies). A non-2-body config raises ``ValueError`` at construction.
    """

    def __init__(self, config: AlienConfig) -> None:
        self._validate(config)
        self.config = config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(cfg: AlienConfig) -> None:
        """Reject configs the 15.0 system cannot honor (loud, never silent)."""
        n = len(cfg.masses)
        if n != 2:
            raise ValueError(
                "AlienUniverse ships the 2-body case in 15.0 (analytical anchors "
                f"+ canonical-polar reduction are 2-body); got n_bodies={n}"
            )
        if len(cfg.charges) != n or len(cfg.ics_pos) != n or len(cfg.ics_vel) != n:
            raise ValueError(
                "AlienConfig masses/charges/ics_pos/ics_vel must all have length "
                f"{n}; got charges={len(cfg.charges)}, ics_pos={len(cfg.ics_pos)}, "
                f"ics_vel={len(cfg.ics_vel)}"
            )
        if cfg.integrator != "implicit_midpoint":
            raise ValueError(
                "AlienUniverse 15.0 ships only the 'implicit_midpoint' integrator; "
                f"got {cfg.integrator!r}"
            )
        if cfg.dt <= 0 or cfg.n_steps < 1:
            raise ValueError(
                f"dt must be > 0 and n_steps >= 1; got dt={cfg.dt}, " f"n_steps={cfg.n_steps}"
            )

    # ------------------------------------------------------------------
    # Canonical-polar wiring (single source of truth: physics.potential_U)
    # ------------------------------------------------------------------
    def _polar_terms(
        self,
    ) -> tuple[float, float, Callable[[float], float], Callable[[float], float]]:
        """Return ``(mu, m_t, U, dUdr)`` for the 2-body relative coordinate.

        ``mu`` is the reduced mass; ``m_t = mu + κ·s₁s₂`` is the tangential
        effective inertia (the §10 correction). ``U(r)`` and its 4th-order
        gradient come from ``physics.potential_U`` so the force the integrator
        uses and the energy Q reports share ONE source of truth.
        """
        cfg = self.config
        m = np.asarray(cfg.masses, dtype=np.float64)
        ch = np.asarray(cfg.charges, dtype=np.float64)
        total_mass = float(m[0] + m[1])
        mu = float(m[0] * m[1] / total_mass)
        s_prod = float(ch[0] * ch[1])
        m_t = mu + cfg.kappa * s_prod

        def U(r: float) -> float:
            pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
            return P.potential_U(pos, m, ch, cfg)

        def dUdr(r: float) -> float:
            h = 1e-6
            return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

        return mu, m_t, U, dUdr

    def _initial_polar_state(self, mu: float, m_t: float) -> np.ndarray:
        """Build the initial canonical-polar state ``y0 = [r, θ, p_r, p_θ]``.

        Derived from the config's Cartesian relative IC (body1 relative to
        body0). The relative position sets ``r`` and ``θ``; the relative
        velocity sets ``p_r = mu·ṙ`` and ``p_θ = m_t·r²·θ̇``.
        """
        cfg = self.config
        pos = np.asarray(cfg.ics_pos, dtype=np.float64)
        vel = np.asarray(cfg.ics_vel, dtype=np.float64)
        rel_pos = pos[1] - pos[0]
        rel_vel = vel[1] - vel[0]
        r = float(np.hypot(rel_pos[0], rel_pos[1]))
        if r == 0.0:
            raise ValueError("initial separation is zero — degenerate 2-body IC")
        theta = float(np.arctan2(rel_pos[1], rel_pos[0]))
        rhat = rel_pos / r
        that = np.array([-rhat[1], rhat[0]], dtype=np.float64)
        rdot = float(np.dot(rel_vel, rhat))
        theta_dot = float(np.dot(rel_vel, that) / r)
        p_r = mu * rdot
        p_theta = m_t * r * r * theta_dot
        return np.array([r, theta, p_r, p_theta], dtype=np.float64)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self) -> AlienTrajectory:
        """Integrate forward from the config ICs; emit the FULL AlienTrajectory.

        Seeds ``np.random.default_rng(cfg.seed)`` (NEVER ``np.random.seed``) so
        any future IC jitter is deterministic; 15.0 configs have deterministic
        ICs so the RNG draws nothing yet, but the stream is established before
        the loop (benchmarks Pitfall #4). On a non-finite state the integrator
        raises ``SimulatorIntegrationError`` (loud).
        """
        cfg = self.config
        # Establish the deterministic RNG stream BEFORE integration. 15.0 ICs are
        # deterministic so no draw happens, but reshuffling this later would break
        # bit-reproducibility (benchmarks Pitfall #4 / ode.py:131-132).
        _rng = np.random.default_rng(cfg.seed)  # noqa: F841 - reserved for IC jitter

        mu, m_t, U, dUdr = self._polar_terms()

        def rhs(y: np.ndarray) -> np.ndarray:
            r, _theta, p_r, p_theta = y
            return np.array(
                [
                    p_r / mu,
                    p_theta / (m_t * r**2),
                    p_theta**2 / (m_t * r**3) - dUdr(r),
                    0.0,
                ],
                dtype=np.float64,
            )

        y0 = self._initial_polar_state(mu, m_t)
        polar = integrate(y0, cfg.dt, cfg.n_steps, rhs, config=cfg, seed=cfg.seed)  # (n_steps+1, 4)

        if not np.all(np.isfinite(polar)):  # defense-in-depth (integrator also guards)
            raise SimulatorIntegrationError(
                config=cfg,
                seed=cfg.seed,
                message="non-finite polar state after integration",
            )

        t = np.arange(cfg.n_steps + 1, dtype=np.float64) * cfg.dt
        positions, velocities = self._reconstruct_cartesian(polar, mu, m_t)
        q_series = np.array(
            [self._polar_Q(polar[k], mu, m_t, U) for k in range(polar.shape[0])],
            dtype=np.float64,
        )

        return AlienTrajectory(
            config=cfg,
            t=t,
            positions=positions,
            velocities=velocities,
            masses=np.asarray(cfg.masses, dtype=np.float64),
            charges=np.asarray(cfg.charges, dtype=np.float64),
            Q=q_series,
            library_versions={
                "numpy": np.__version__,
                "scipy": _pkg_version("scipy"),
                "sympy": _pkg_version("sympy"),
            },
        )

    @staticmethod
    def _polar_Q(y: np.ndarray, mu: float, m_t: float, U: Callable[[float], float]) -> float:
        """The true polar-Hamiltonian invariant Q the integrator preserves.

        ``Q = p_r²/(2 mu) + p_θ²/(2 m_t r²) + U(r)``. The two kinetic terms are
        the radial (``½ mu ṙ²``) and tangential (``½ m_t (r θ̇)²``) energies; the
        ``m_t`` tangential inertia is what makes Q conserved (RESEARCH
        make-or-break). Equals §10's Q symbol in the relative coordinate.
        """
        r, _theta, p_r, p_theta = y
        return float(p_r**2 / (2 * mu) + p_theta**2 / (2 * m_t * r**2) + U(r))

    def _reconstruct_cartesian(
        self, polar: np.ndarray, mu: float, m_t: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Polar relative coordinate → full 2-body Cartesian state (COM frame).

        With the centre of mass fixed at the origin, the relative coordinate
        ``rel = (r cosθ, r sinθ)`` distributes onto the two bodies as
        ``pos0 = −(m1/M)·rel`` and ``pos1 = +(m0/M)·rel`` (and likewise for
        velocity), so the trajectory is the genuine two-body motion. Returns
        ``(positions, velocities)`` each shaped ``(n_steps+1, 2, 2)``.
        """
        cfg = self.config
        m = np.asarray(cfg.masses, dtype=np.float64)
        total_mass = float(m[0] + m[1])
        n = polar.shape[0]
        positions = np.empty((n, 2, 2), dtype=np.float64)
        velocities = np.empty((n, 2, 2), dtype=np.float64)
        for k in range(n):
            r, theta, p_r, p_theta = polar[k]
            rdot = p_r / mu
            theta_dot = p_theta / (m_t * r**2)
            c, s = np.cos(theta), np.sin(theta)
            rhat = np.array([c, s], dtype=np.float64)
            that = np.array([-s, c], dtype=np.float64)
            rel = r * rhat
            rel_vel = rdot * rhat + r * theta_dot * that
            positions[k, 0] = -(m[1] / total_mass) * rel
            positions[k, 1] = +(m[0] / total_mass) * rel
            velocities[k, 0] = -(m[1] / total_mass) * rel_vel
            velocities[k, 1] = +(m[0] / total_mass) * rel_vel
        return positions, velocities

    # ------------------------------------------------------------------
    # Data-only emission (the single ObservationBundle construction site)
    # ------------------------------------------------------------------
    @staticmethod
    def to_observation_bundle(traj: AlienTrajectory) -> ObservationBundle:
        """Project the FULL trajectory to the DATA-ONLY ObservationBundle.

        EXPLICIT positional field selection — copies only ``t``, ``positions``,
        ``velocities``, ``masses``. NEVER ``dataclasses.asdict(traj)`` then
        filter: a future field added to ``AlienTrajectory`` must not be able to
        silently re-leak through here (RESEARCH Pitfall 4 / T-15.0-01). This is
        the single construction site for an ``ObservationBundle`` in 15.0.
        """
        return ObservationBundle(
            t=traj.t,
            positions=traj.positions,
            velocities=traj.velocities,
            masses=traj.masses,
        )
