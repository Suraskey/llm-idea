"""Simulator-test fixtures. Inherits the L-004 singleton-patch discipline.

Mirrors ``tests/benchmarks/conftest.py``: the L-004 autouse env-scrubber
(``monkeypatch.setattr`` on the cached Settings singleton, not just setenv) plus
``os.environ.setdefault("GOOGLE_API_KEY", "test-key")`` so import chains that
touch ``common.config`` don't fail in CI. The simulator modules themselves are
import-light (numpy/scipy/sympy only), but the conftest keeps the discipline so
any future Settings-touching import inherits it.

Provides ``eccentric_fixture`` — a 2-body ``AlienConfig`` plus the canonical
polar-Hamiltonian ``rhs`` and ``Q`` closures, wired through the REAL integrator
entrypoint (``integrator.integrate`` + ``physics.total_accel``) so every
conservation test exercises the production path (RESEARCH Q-conservation recipe;
benchmarks conftest Pitfall #1 "go through the real entrypoint").

NO ``np.random.seed`` anywhere (conftest.py:32-33 hard rule). The analytical
configs here are deterministic ICs, so no RNG is needed; if any randomness is
ever added, use ``np.random.default_rng``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest

from ascension.simulator import physics as P
from ascension.simulator.types import AlienConfig

# Set GOOGLE_API_KEY before any test module triggers an import chain that
# transitively instantiates the common.config Settings singleton. Test-only;
# production callers provide a real key.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")


_ENV_PREFIXES = (
    "ASCENSION_",
    "SANDBOX_",
    "BENCHMARK_",
    "SIMULATOR_",
    "MODEL_",
    "EMBEDDING_",
    "GOOGLE_",
    "POSTGRES_",
    "DATABASE_",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip simulator-related env overrides; supply a dummy GOOGLE_API_KEY.

    L-004: setenv does NOT reach the cached Settings singleton. Plan 01
    introduces no SIMULATOR_* Settings fields, so there are no setattr calls
    here yet — add ``monkeypatch.setattr(settings, "SIMULATOR_FOO", <default>)``
    if a future phase introduces SIMULATOR_* knobs.
    """
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


# -----------------------------------------------------------------------------
# Canonical 2-body polar-Hamiltonian wiring (single source of truth: potential_U)
# -----------------------------------------------------------------------------
#
# State y = [r, theta, p_r, p_theta] for the relative 2-body coordinate.
# Hamiltonian:  H = p_r^2/(2 mu) + p_theta^2/(2 m_t r^2) + U(r)
#   mu  = reduced mass m1 m2/(m1+m2)
#   m_t = mu + kappa*(s1 s2)  (tangential effective inertia — the §10 correction)
# Hamilton's eqns:
#   rdot     = p_r / mu
#   thetadot = p_theta / (m_t r^2)
#   p_r_dot  = p_theta^2 / (m_t r^3) - dU/dr
#   p_th_dot = 0
# The radial force -dU/dr is taken from physics.potential_U's OWN gradient so
# the force used by the integrator and the energy used by Q come from ONE source
# of truth (this is what makes Q conserve to ~1e-10; an independently hand-coded
# dU/dr that disagreed with U by a constant produced a fixed ~8e-3 offset). The
# gradient is computed with a 4th-order central stencil, consistent with the
# physics-invariant test that already proved total_accel == -grad U to 1e-9.


@dataclass(frozen=True, slots=True)
class _PolarSystem:
    """Bundle of the config + the rhs/Q closures + the analytical v_circ."""

    config: AlienConfig
    rhs: object  # Callable[[np.ndarray], np.ndarray]
    Q: object  # Callable[[np.ndarray], float]
    y0: np.ndarray
    mu: float
    m_t: float
    v_circ: float


def _build_polar_system(cfg: AlienConfig, *, tangential_frac: float, r0: float) -> _PolarSystem:
    """Build the canonical-polar (rhs, Q, y0) for a 2-body AlienConfig."""
    m = np.asarray(cfg.masses, dtype=np.float64)
    ch = np.asarray(cfg.charges, dtype=np.float64)
    mu = float(m[0] * m[1] / (m[0] + m[1]))
    s_prod = float(ch[0] * ch[1])
    m_t = mu + cfg.kappa * s_prod

    def U(r: float) -> float:
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, cfg)

    def dUdr(r: float) -> float:
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    def rhs(y: np.ndarray) -> np.ndarray:
        r, _th, pr, pth = y
        return np.array(
            [pr / mu, pth / (m_t * r**2), pth**2 / (m_t * r**3) - dUdr(r), 0.0],
            dtype=np.float64,
        )

    def Q(y: np.ndarray) -> float:
        r, _th, pr, pth = y
        return float(pr**2 / (2 * mu) + pth**2 / (2 * m_t * r**2) + U(r))

    # Circular-orbit speed: balance mu*v^2/r0 = -dU/dr (the radial force is
    # attractive, dU/dr > 0). v_circ = sqrt(r0 * dU/dr / mu).
    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    theta_dot0 = tangential_frac * v_circ / r0
    p_theta0 = m_t * r0**2 * theta_dot0
    y0 = np.array([r0, 0.0, 0.0, p_theta0], dtype=np.float64)
    return _PolarSystem(config=cfg, rhs=rhs, Q=Q, y0=y0, mu=mu, m_t=m_t, v_circ=v_circ)


@pytest.fixture
def eccentric_fixture() -> _PolarSystem:
    """Eccentric 2-body system for the make-or-break Q-conservation bar.

    GM ~ O(1) (equal unit masses) so v_circ ~ 1 and dt=0.005 is in the
    RESEARCH-validated regime (RESEARCH Q2). tangential_frac=0.8 gives a moderate
    eccentricity (perihelion well above the 0.5 wall) so dt=0.005 clears the
    1e-6 bar with margin while still being genuinely non-circular (a vacuous
    near-circular orbit would not exercise v_perp's variation).
    """
    cfg = AlienConfig(
        G=1.0,
        alpha=0.05,
        beta=0.02,
        gamma=0.7,
        kappa=0.3,
        masses=(1.0, 1.0),
        charges=(1.0, 1.0),
        ics_pos=((0.0, 0.0), (1.0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 1.0)),
        dt=0.005,
        n_steps=1000,
        seed=0,
    )
    return _build_polar_system(cfg, tangential_frac=0.8, r0=1.0)
