"""Value-object DTOs for the Alien Universe simulator.

Implements the Q-consistent §10 formulation (tangential-inertia coupling),
reconciled into SCOPE §10 (2026-05-23).

Frozen slots dataclasses, same pattern as ``src/ascension/benchmarks/types.py``
and ``src/ascension/sandbox/types.py``.

DTO-idiom resolution ([CONFLICT] D-07 wording vs benchmarks idiom): CONTEXT
D-07 literally says "frozen Pydantic DTOs", but the package it directs us to
mirror (``benchmarks/types.py:1-7``) uses ``@dataclass(frozen=True,
slots=True)`` and reserves Pydantic for Agora nodes (Phase 5.0). The
mirror-benchmarks directive wins here for three reasons documented in 15.0
RESEARCH §Standard Stack ("Architecture decision the planner must resolve") and
15.0 PATTERNS gotcha 3:
  1. These are internal-only transport records, identical in role to
     ``BenchmarkSpec`` / ``TrajectoryBundle``.
  2. numpy arrays do not coerce cleanly under Pydantic v2 without
     ``arbitrary_types_allowed`` config noise.
  3. D-06 bit-reproducibility is cleaner without Pydantic coercion.

Plain English: these are the "packing boxes" the simulator produces.
``AlienConfig`` is the full recipe for a run (physics constants, hidden charges,
initial conditions, integrator settings, seed). ``AlienTrajectory`` is the FULL
internal result, including the hidden charges and the conserved quantity Q.
``ObservationBundle`` is the data-only twin handed downstream — it deliberately
carries NOTHING the Council could use to shortcut discovery.

Invariants (documented here so a grep regression test can confirm presence —
see ``tests/simulator/``):
  - ObservationBundle carries NO charges, NO true law, NO Q. Its field set is
    exactly {t, positions, velocities, masses}. This is D-07's "hidden /
    observable split enforced at the type boundary" (T-15.0-01 mitigation).

Binding decisions:
  - 15.0 PATTERNS §types.py: frozen slots dataclass; required fields first,
    sentinel-None optionals last; data-only contract as a grep-asserted
    docstring literal.
  - 15.0 RESEARCH §Architecture Patterns / Pattern 3: hidden/observable type
    split; ObservationBundle constructed by explicit field selection, never by
    ``dataclasses.asdict`` of the full trajectory (which would leak).

Threat-model mitigations: T-15.0-01 (information disclosure) — the data-only
ObservationBundle field set is frozen + documented; the leak-guard test is
Plan 03.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class AlienConfig:
    """Immutable problem specification for one Alien Universe run.

    Plain English: the "recipe card" for an alien-physics run. It lists the
    physics constants (gravity strength G, the short-range correction α, the
    hidden-charge coupling β/γ, the tangential-inertia coupling κ), the per-body
    masses and HIDDEN charges, the initial positions and velocities, the
    integrator step size + count, which integrator to use, and the random seed
    that locks any IC jitter so two runs are bit-identical.

    Attributes:
      G: Newtonian gravitational constant for the run.
      alpha: Short-range correction strength (the ``α/r^p`` term, exponent
        ``correction_exponent``; attractive).
      beta: Hidden-charge force coupling (the ``β·(s₁ coupling s₂)·(1−cos γr)/r²``
        term, where the coupling is set by ``charge_coupling``).
      gamma: Periodic length-scale parameter inside the hidden-charge force.
      kappa: Tangential-inertia coupling — the hidden charge augments tangential
        inertia via ``m_eff = m + κ·s²`` (the Q-consistent §10 correction). This
        coupling is INDEPENDENT of ``charge_coupling`` (which governs the β hidden
        FORCE, not the inertia term) and so is unchanged across V1/V2 regimes.
      charge_coupling: How the two bodies' hidden charges combine in the β hidden
        FORCE term. ``"product"`` (default, the V1 §10 law) uses ``s_i·s_j``;
        ``"sum"`` (the EXP-080 V2 regime) uses ``s_i+s_j``. The default preserves
        V1 byte-for-byte. The matching ``potential_U`` + ``conserved_Q`` branch on
        this field so the force law, potential, and conserved Q stay mutually
        consistent (the integrable §10 invariant).
      correction_exponent: The radial power of the short-range α correction —
        the ``p`` in the attractive force ``−G·α·m_j/r^p``. ``3.5`` (default, the
        V1 §10 law) preserves V1 byte-for-byte; ``2.5`` is the EXP-080 V2 regime.
        The matching correction potential ``U_corr = −(G·m_i·m_j·α/(p−1))·r^(−(p−1))``
        is derived from this exponent in ``potential_U`` so ``−dU_corr/dr`` exactly
        reproduces the force (the integrable §10 invariant).
      masses: Per-body masses, shape ``(n_bodies,)`` as a tuple for hashability.
      charges: Per-body HIDDEN scalar charges ``s``, shape ``(n_bodies,)``.
        These never enter an ObservationBundle.
      ics_pos: Initial positions, ``(n_bodies, dim)`` as a tuple-of-tuples.
      ics_vel: Initial velocities, ``(n_bodies, dim)`` as a tuple-of-tuples.
      dt: Fixed integrator step size (RESEARCH Q2: 0.005 clears the 1e-6 bar).
      n_steps: Number of integration steps.
      integrator: Integrator name. Only ``"implicit_midpoint"`` is shipped in
        15.0 (deterministic, fixed-step, conserves quadratic invariants).
      seed: Integer fed into ``np.random.default_rng(seed)`` — only used if ICs
        are jittered. Same seed → bit-identical output (D-06).
    """

    G: float
    alpha: float
    beta: float
    gamma: float
    kappa: float
    masses: tuple[float, ...]
    charges: tuple[float, ...]
    ics_pos: tuple[tuple[float, ...], ...]
    ics_vel: tuple[tuple[float, ...], ...]
    dt: float
    n_steps: int
    seed: int
    integrator: str = "implicit_midpoint"
    # V2 regime knobs (EXP-080). Defaults preserve the V1 §10 law BYTE-FOR-BYTE:
    # "product" + 3.5 reproduce exactly the force/potential/Q the V1 build shipped.
    charge_coupling: Literal["product", "sum"] = "product"
    correction_exponent: float = 3.5


@dataclass(frozen=True, slots=True)
class BodyState:
    """Per-body instantaneous state (internal, transient).

    Plain English: one body's position and velocity at one instant. Used inside
    physics/integrator helpers; not serialized.

    Attributes:
      pos: Position vector, shape ``(dim,)``, float64.
      vel: Velocity vector, shape ``(dim,)``, float64.
    """

    pos: np.ndarray
    vel: np.ndarray


@dataclass(frozen=True, slots=True)
class AlienTrajectory:
    """FULL internal trajectory record — includes hidden state.

    Plain English: the complete result of one run. It carries everything,
    INCLUDING the hidden charges and the conserved quantity Q. This is the
    internal record; downstream consumers get the data-only ``ObservationBundle``
    instead. ``library_versions`` pins the exact numpy/scipy/sympy versions at
    generation time so the run can be re-derived (SCOPE §22.2).

    Attributes:
      config: The ``AlienConfig`` used. Stored so the trajectory is
        self-describing.
      t: Time axis, shape ``(n_steps + 1,)``.
      positions: Body positions over time, shape
        ``(n_steps + 1, n_bodies, dim)``, float64.
      velocities: Body velocities over time, same shape as ``positions``.
      masses: Per-body masses, shape ``(n_bodies,)``, float64.
      charges: HIDDEN per-body charges, shape ``(n_bodies,)``, float64. Never
        crosses into an ObservationBundle.
      Q: The conserved quantity over time, shape ``(n_steps + 1,)``. Hidden —
        recovering it requires positing the hidden coupling.
      library_versions: Dict mapping ``"numpy"`` / ``"scipy"`` / ``"sympy"`` to
        the versions active during generation.
    """

    config: AlienConfig
    t: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    charges: np.ndarray
    Q: np.ndarray
    library_versions: dict[str, str]


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    """DATA-ONLY downstream view of a run.

    ObservationBundle carries NO charges, NO true law, NO Q.

    Plain English: this is the only thing the Council (Phase 15.2) ever sees. It
    is the data-only twin of ``AlienTrajectory`` with every hidden field removed.
    Its field set is exactly {t, positions, velocities, masses} — nothing the
    swarm could use to shortcut the discovery task (no charges, no conserved
    quantity, no config carrying the physics constants, no reference to the true
    law). ``alien.py`` is the only place that constructs one, and it does so by
    explicit field selection — NEVER by ``dataclasses.asdict`` of the full
    trajectory (which would silently re-leak hidden fields if one were added).

    Attributes:
      t: Time axis, shape ``(n_steps + 1,)``.
      positions: Observable body positions, shape
        ``(n_steps + 1, n_bodies, dim)``, float64.
      velocities: Observable body velocities, same shape as ``positions``.
      masses: Per-body masses, shape ``(n_bodies,)``, float64.
    """

    t: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
