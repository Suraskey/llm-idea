"""Simulator module constants — the §10 STARTING parameter set + artifact layout.

Mirrors ``src/ascension/benchmarks/config.py`` (non-tunable artifact-directory
layout) plus the ``benchmarks/systems.py:44-47`` ``UPDATED_ON`` staleness
anchor. Values that are NOT user-tunable live here; the per-run physics knobs
live on the ``AlienConfig`` DTO (per-call).

NOTE — ``DEFAULT_ALIEN_PARAMS`` is the §10 STARTING set (physics-CORRECTNESS
tests only). Phase 15.1 (TIER2-02) adds the provenance-stamped TUNED operating
point ``TIER2_TUNED_PARAMS`` + the ``TIER2_HELD_OUT_ICS`` generalization set as
SEPARATE named constants (D-09) — ``DEFAULT_ALIEN_PARAMS`` is left un-mutated as
the documented baseline the analytical-anchor fixtures read.

The default orbit is deliberately near-circular (tangential speed 0.92·v_circ
at r0=1.0) so the implicit-midpoint dt=0.005 step clears the 1e-6 Q-drift bar
with margin (15.0 RESEARCH Q2). Eccentric / high-drift fixtures are built
explicitly in tests, not from this default.

Binding decisions:
  - 15.0 PATTERNS §config.py: artifact-layout constants + the default α/β/γ/κ
    parameter set live here; tuning is 15.1.
  - 15.0 RESEARCH §Make-or-break: the Q-consistent §10 formulation
    (``m_eff = m + κs²``, attractive α-correction). UPDATED_ON anchors when the
    Q-consistency derivation was verified so a reviewer re-checks if §10 changes.
  - benchmarks/systems.py:44-47: UPDATED_ON staleness-discipline pattern.

Pitfalls honored: N/A — this file is structural / default-constants only.
Pitfall handling is in ``physics.py`` and ``integrator.py``.

Threat-model mitigations: N/A — pure constants surface.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from ascension.simulator import physics as P
from ascension.simulator.types import AlienConfig

# Date the Q-consistency derivation was verified against SCOPE §10 (RESEARCH),
# bumped 2026-05-23 when Phase 15.1 added TIER2_TUNED_PARAMS / TIER2_HELD_OUT_ICS
# and corrected the DEFAULT_ALIEN_PARAMS mass-ratio integrability caveat (D-09).
# Re-check Q-consistency + the integrator drift bound if §10 changes after this.
UPDATED_ON = date(2026, 5, 23)

# Per-run artifact layout under settings.RUN_ARTIFACTS_DIR. Consumed by Plan 02
# `io.py` when it writes spec / trajectories / manifest under
# $RUN_ARTIFACTS_DIR/<ARTIFACT_SUBDIR>/<run_id>/.
ARTIFACT_SUBDIR = "simulator"
SPEC_FILENAME = "spec.json"
TRAJECTORIES_FILENAME = "trajectories.npz"
MANIFEST_FILENAME = "manifest.json"

# -----------------------------------------------------------------------------
# DEFAULT_ALIEN_PARAMS — the §10 STARTING set (a 2-body near-circular config).
# -----------------------------------------------------------------------------
# r0 = 1.0; the orbiting body's tangential speed is `tangential_frac` of the
# circular-orbit speed v_circ(r0) = sqrt(G·m_center/m_orbit · (1/r0 + α/r0^2.5)),
# computed in fixtures.py from these values (NOT hardcoded here, so it stays
# consistent if a knob changes). tangential_frac = 0.92 keeps the orbit
# near-circular so dt=0.005 is safe (RESEARCH Q2).
#
# masses = (1000.0, 1.0): a heavy near-fixed center + a light orbiting body, so
# the 2-body analytical anchors (circular / parabolic) apply.
# charges = (1.0, 1.0): both bodies carry unit hidden charge s, so the
# tangential-inertia coupling and the F_s hidden force are both active.
#
# ⚠️ DOC-CORRECTION (2026-05-23, Phase 15.1 RESEARCH make-or-break): the
# "tangential_frac=0.92 keeps the orbit near-circular so dt=0.005 is safe" claim
# above is FALSE at masses=(1000,1). At that mass ratio v_circ≈32, giving only
# ~39 integrator steps/orbit at dt=0.005 — wildly under-resolved; measured
# Q-drift is ~1e-1..1e-2 (frac 0.80–0.95), grossly violating the locked 1e-6 bar.
# The 15.0 fixtures and TIER2_TUNED_PARAMS use masses=(1.0,1.0) (≈869 steps/orbit,
# Q-drift ~1e-8). This constant is the documented §10 STARTING set, NOT an
# integrable run config — the VALUES are left unchanged (D-09: this is the
# baseline the analytical-anchor fixtures read), only the caveat is added. To
# actually integrate the (1000,1) regime to the bar, dt would need to drop to
# ≈2e-5 (~250× cost) — rejected; tune in the (1,1) regime instead.
DEFAULT_ALIEN_PARAMS: dict[str, object] = {
    "G": 1.0,
    "alpha": 0.05,
    "beta": 0.02,
    "gamma": 0.7,
    "kappa": 0.3,
    # ⚠️ masses=(1000,1) is NOT integrable to the 1e-6 Q-drift bar at dt=0.005
    # (~39 steps/orbit; measured Q-drift ~1e-1, 15.1 RESEARCH). The 15.0 fixtures
    # and TIER2_TUNED_PARAMS use masses=(1,1). This constant is the documented §10
    # STARTING set, not an integrable run config.
    "masses": (1000.0, 1.0),
    "charges": (1.0, 1.0),
    "r0": 1.0,
    "tangential_frac": 0.92,
    "dt": 0.005,
    "n_steps": 1000,
    "integrator": "implicit_midpoint",
    "seed": 0,
}

# -----------------------------------------------------------------------------
# TIER2_TUNED_PARAMS — the Phase 15.1 TUNED operating point (D-09).
# -----------------------------------------------------------------------------
# The provenance-stamped tuned set for the Tier-2 discovery task (TIER2-02),
# distinct from the un-mutated §10 baseline above. Every value is the
# RESEARCH-recommended operating point (15.1 RESEARCH §Recommended Operating
# Point), and the comment records WHY each value (D-01 "document why, not just
# that"), not merely that it works:
#
#   - couplings α=0.05, β=0.02, γ=0.7, κ=0.3 stay at the §10 STARTING values.
#     The sweep showed NO coupling increase is needed — the Newton-fit residual
#     structure already clears p<0.01 by ~200 orders of magnitude (nested-model
#     F-stats 10⁴–10⁷). Increasing couplings only stresses the integrator for
#     zero statistical gain, and keeps the tuned set maximally close to the
#     documented baseline (cleaner provenance story).
#   - masses=(1.0, 1.0), NOT (1000,1). The make-or-break finding: (1000,1) is
#     not integrable to the 1e-6 Q-drift bar at dt=0.005 (~39 steps/orbit). The
#     (1,1) regime is the one 15.0 actually validated (~869 steps/orbit, Q-drift
#     ~1e-8); all 2-body analytical anchors still apply via the reduced mass, and
#     the discovery story is unchanged (the hidden charge still surfaces only in
#     the angular dynamics). The "heavy fixed center" framing is cosmetic.
#   - charges=(1.0, 1.0): unit hidden charge on both bodies so the κ tangential-
#     inertia coupling and the F_s hidden force are both active (and observable).
#   - tangential_frac=0.85 is the difficulty⇄solvability⇄stability sweet spot:
#     Q-drift 4.13e-8 (24× margin on the 1e-6 bar), r-range [1.0, 1.41] (wide
#     enough that the orbit precesses and the hidden DOF dominates the residual —
#     the hidden-F_s term explains 91% of the post-(Newton+α) residual), real
#     apsidal precession (~8 perihelion passages / 10k steps). frac=0.80 is too
#     narrow (orbit accidentally polynomial-fittable on its visited radii);
#     frac=0.90 shrinks the conservation margin to ~5×; frac≤~0.55 plunges.
#   - dt=0.005, n_steps=1000, implicit_midpoint, seed=0: the 15.0-validated
#     deterministic regime; no dt change is needed at this operating point.
#
# r0=1.0 and tangential_frac are INPUTS to the IC-builder (the fixtures._U_and_dUdr
# idiom), baked into ics_pos/ics_vel for any run — they are NOT AlienConfig fields,
# but are carried in this dict (mirroring DEFAULT_ALIEN_PARAMS' shape verbatim) so
# the tuned operating point is self-describing.
TIER2_TUNED_PARAMS: dict[str, object] = {
    "G": 1.0,
    "alpha": 0.05,
    "beta": 0.02,
    "gamma": 0.7,
    "kappa": 0.3,
    "masses": (1.0, 1.0),  # <-- NOT (1000,1); see make-or-break finding above
    "charges": (1.0, 1.0),
    "r0": 1.0,
    "tangential_frac": 0.85,
    "dt": 0.005,
    "n_steps": 1000,
    "integrator": "implicit_midpoint",
    "seed": 0,
}


# -----------------------------------------------------------------------------
# TIER2_HELD_OUT_ICS — the held-out generalization IC set (D-03 / D-09).
# -----------------------------------------------------------------------------
def _reduced_mass(masses: tuple[float, ...]) -> float:
    """Reduced mass mu = m0·m1/(m0+m1) (mirror fixtures._reduced_mass)."""
    m = np.asarray(masses, dtype=np.float64)
    return float(m[0] * m[1] / (m[0] + m[1]))


def _tier2_two_body_config(*, r0: float, tangential_frac: float, seed: int) -> AlienConfig:
    """Build one held-out 2-body AlienConfig with a potential_U-derived IC speed.

    Derives the tangential speed from ``physics.potential_U``'s OWN gradient using
    the EXACT ``fixtures._U_and_dUdr`` idiom (4th-order finite-difference stencil
    over the FULL tuned potential) — NEVER a hardcoded v_circ — so the IC and the
    force the integrator applies can never disagree (single source of truth). The
    held-out IC carries the TIER2 couplings + masses=(1,1) + unit charges, dt=0.005,
    n_steps=1000; only ``r0`` and ``tangential_frac`` vary, spanning distinct orbit
    shapes (different perihelion/aphelion, precession rates).
    """
    alpha = float(TIER2_TUNED_PARAMS["alpha"])
    beta = float(TIER2_TUNED_PARAMS["beta"])
    gamma = float(TIER2_TUNED_PARAMS["gamma"])
    kappa = float(TIER2_TUNED_PARAMS["kappa"])
    G = float(TIER2_TUNED_PARAMS["G"])
    masses = TIER2_TUNED_PARAMS["masses"]  # type: ignore[assignment]
    charges = TIER2_TUNED_PARAMS["charges"]  # type: ignore[assignment]
    dt = float(TIER2_TUNED_PARAMS["dt"])
    n_steps = int(TIER2_TUNED_PARAMS["n_steps"])

    # Probe config (rel_speed=0) so potential_U sees the right couplings; the
    # speed is then derived from this config's own gradient, exactly like the
    # fixtures.py idiom — no hardcoded v_circ.
    probe = AlienConfig(
        G=G,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kappa=kappa,
        masses=tuple(masses),
        charges=tuple(charges),
        ics_pos=((0.0, 0.0), (r0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 0.0)),
        dt=dt,
        n_steps=n_steps,
        seed=seed,
    )
    m = np.asarray(probe.masses, dtype=np.float64)
    ch = np.asarray(probe.charges, dtype=np.float64)

    def U(r: float) -> float:
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, probe)

    def dUdr(r: float) -> float:
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    mu = _reduced_mass(probe.masses)
    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    rel_speed = tangential_frac * v_circ
    return AlienConfig(
        G=G,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kappa=kappa,
        masses=tuple(masses),
        charges=tuple(charges),
        ics_pos=((0.0, 0.0), (r0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, rel_speed)),
        dt=dt,
        n_steps=n_steps,
        seed=seed,
    )


def build_tier2_held_out_ics() -> tuple[AlienConfig, ...]:
    """Construct the ≥12-config held-out generalization set (D-03 / D-09).

    Honest-split rationale (documented inline per RESEARCH §Held-out IC design):
    each held-out IC differs in its STARTING orbit (r0, tangential_frac) but the
    hidden charge / κ coupling is the SAME physics across all of them — that is
    the point: the true law must generalize to UNSEEN orbit shapes. The split
    does not leak the hidden structure because the hypothetical discoverer never
    sees charges/Q — only positions/velocities/masses/t (the ObservationBundle
    data-only contract, types.py). The anti-honest trap to avoid (drawing every
    held-out IC at the fit-set frac, which only tests reproducibility of ONE
    orbit shape) is sidestepped by spanning a RANGE of frac and r0.

    Draw: a deterministic 3×3 grid over r0∈{0.9,1.05,1.2} × frac∈{0.80,0.85,0.90}
    (9 configs, all inside the stable band [0.70,0.90] so no IC blows up) PLUS a
    seeded ``np.random.default_rng(7)`` jitter of 4 more configs over the same
    r0∈[0.9,1.2], frac∈[0.80,0.90] box → 13 configs total (≥12, D-09). NO
    ``np.random.seed`` (conftest hard rule) — ``default_rng(7)`` only.
    """
    cfgs: list[AlienConfig] = []
    seed = 0
    # Deterministic grid (9 configs).
    for r0 in (0.9, 1.05, 1.2):
        for frac in (0.80, 0.85, 0.90):
            cfgs.append(_tier2_two_body_config(r0=r0, tangential_frac=frac, seed=seed))
            seed += 1
    # Seeded jitter (4 more configs) over the same stable box.
    rng = np.random.default_rng(7)
    for _ in range(4):
        r0 = float(rng.uniform(0.9, 1.2))
        frac = float(rng.uniform(0.80, 0.90))
        cfgs.append(_tier2_two_body_config(r0=r0, tangential_frac=frac, seed=seed))
        seed += 1
    return tuple(cfgs)


# ≥12 held-out configs spanning r0∈[0.9,1.2] and frac∈[0.80,0.90], each with a
# potential_U-derived IC speed (never hardcoded). Bound at module load (D-09).
TIER2_HELD_OUT_ICS: tuple[AlienConfig, ...] = build_tier2_held_out_ics()
