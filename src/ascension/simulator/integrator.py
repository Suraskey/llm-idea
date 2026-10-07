"""Deterministic fixed-step implicit-midpoint integrator (no scipy in the path).

Implements the Q-consistent §10 integration: a hand-rolled, fixed-step,
float64 implicit-midpoint rule (the lowest Gauss-Legendre member). It conserves
all quadratic invariants exactly and is symplectic, so the conserved quantity
Q has BOUNDED (non-secular) drift regardless of horizon (15.0 RESEARCH Q2).

Why hand-rolled and NOT scipy: D-01/D-06 require a deterministic fixed-step
integrator. scipy's adaptive solvers (RK45/LSODA/Radau) branch their internal
step selection on float noise, which breaks D-06 bit-identity, and the
non-symplectic ones (RK45) show SECULAR Q-drift that blows up near perihelion.
RESEARCH measured implicit-midpoint at dt=0.005 holding Q-drift to 1.2e-9 over
50,000 steps. scipy DOP853 is kept ONLY as a test-time cross-check oracle
(Plan 02) — never in this production path.

  - Pitfall 2 (RESEARCH): do not pick the integrator by short-horizon RMSE;
    symplectic bounded drift, not adaptive RK accuracy, is the requirement.
  - Pitfall 3 (RESEARCH): velocity-Verlet needs a SEPARABLE Hamiltonian; the
    tangential-inertia mass ``(m+κs²)r²`` is position-dependent → non-separable
    → Verlet does NOT conserve Q. Implicit-midpoint handles non-separable H.

D-06 determinism (load-bearing): the inner fixed-point solve uses a FIXED
``max_iter`` cap and a tolerance compared with ``<`` on a deterministic
reduction (``np.max(np.abs(...))``). There is no RNG inside the step and no
adaptive dt, so the same ``y``/``dt`` produces the same iteration trace →
bit-identical output (asserted by ``np.array_equal``, not ``allclose``).

Binding decisions:
  - 15.0 PATTERNS §integrator.py / §Shared Pattern reproducibility:
    deterministic fixed-step; float64 throughout; pre-allocate the output.
  - 15.0 RESEARCH Code Examples §"Verified: Q conservation under
    implicit-midpoint" — this is the verbatim step rule.
  - benchmarks/ode.py:147 — pre-allocate ``np.empty((n+1, ...), dtype=float64)``.
  - benchmarks/ode.py:157-163 — non-finite state raises a host-level loud
    failure (no cover-ups, CLAUDE.md), never silent.

Threat-model mitigations: T-15.0-03 (non-reproducible run) — float64 +
deterministic fixed step. T-15.0-04 (blowup) — non-finite state raises
SimulatorIntegrationError, loud + replayable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ascension.simulator.exceptions import SimulatorIntegrationError


def implicit_midpoint_step(
    y: np.ndarray,
    dt: float,
    rhs: Callable[[np.ndarray], np.ndarray],
    *,
    max_iter: int = 64,
    tol: float = 1e-15,
) -> np.ndarray:
    """One deterministic implicit-midpoint step.

    The implicit-midpoint rule advances ``y`` by solving the fixed point
    ``y_mid = y + ½·dt·rhs(y_mid)`` and returning ``y_{n+1} = 2·y_mid − y``.
    The inner solve is fixed-point iteration with an explicit-Euler predictor.

    D-06 determinism: ``max_iter`` is a fixed cap and ``tol`` is compared with
    ``<`` on ``np.max(np.abs(new − y_mid))`` — a deterministic reduction. Same
    ``y``/``dt`` → same iteration trace → bit-identical output. No RNG, no
    adaptive dt, float64 throughout.

    Args:
      y: Current state vector (float64).
      dt: Fixed step size.
      rhs: Right-hand side ``f(y) -> dy/dt``. For the Alien Universe this calls
        ``physics.total_accel`` (via the canonical-polar wiring built by the
        system object / fixture).
      max_iter: Fixed iteration cap for the inner fixed-point solve.
      tol: Convergence threshold on the max-abs state increment.

    Returns:
      The next state ``y_{n+1}`` (float64), same shape as ``y``.
    """
    y = np.asarray(y, dtype=np.float64)
    y_mid = y + 0.5 * dt * rhs(y)  # explicit-Euler predictor
    for _ in range(max_iter):  # fixed cap → deterministic, no float-noise branch
        new = y + 0.5 * dt * rhs(y_mid)
        if np.max(np.abs(new - y_mid)) < tol:
            y_mid = new
            break
        y_mid = new
    return 2.0 * y_mid - y


def integrate(
    y0: np.ndarray,
    dt: float,
    n_steps: int,
    rhs: Callable[[np.ndarray], np.ndarray],
    *,
    config: object | None = None,
    seed: int = 0,
    max_iter: int = 64,
    tol: float = 1e-15,
) -> np.ndarray:
    """Deterministic fixed-step integration loop.

    Pre-allocates the trajectory array (mirror benchmarks/ode.py:147) and steps
    ``n_steps`` times with ``implicit_midpoint_step``. float64 throughout. On a
    non-finite state (NaN/Inf — typically a blowup near perihelion) it raises
    ``SimulatorIntegrationError`` (host-level loud failure, no cover-ups), never
    a silent NaN-filled array.

    Args:
      y0: Initial state vector (float64).
      dt: Fixed step size.
      n_steps: Number of steps to take.
      rhs: Right-hand side ``f(y) -> dy/dt``.
      config: Optional ``AlienConfig`` carried into the exception for replay.
      seed: Seed carried into the exception for replay (D-06).
      max_iter: Inner-solve iteration cap (passed through).
      tol: Inner-solve convergence threshold (passed through).

    Returns:
      Array of shape ``(n_steps + 1, *y0.shape)``, float64; row 0 is ``y0``.

    Raises:
      SimulatorIntegrationError: if any integrated state is non-finite.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    out = np.empty((n_steps + 1, *y0.shape), dtype=np.float64)
    out[0] = y0
    y = y0
    for k in range(n_steps):
        y = implicit_midpoint_step(y, dt, rhs, max_iter=max_iter, tol=tol)
        if not np.all(np.isfinite(y)):
            raise SimulatorIntegrationError(
                config=config,
                seed=seed,
                message=f"non-finite state at step {k + 1} (blowup); state={y!r}",
            )
        out[k + 1] = y
    return out
