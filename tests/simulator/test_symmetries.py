"""Wave-3 symmetry tests for the assembled ``AlienUniverse.run()`` (D-03/Q5).

Plain English: physical laws have symmetries. The Alien Universe forces are
central (depend only on the inter-body separation r), so the dynamics are
invariant under rigid rotations and translations of the whole system, and the
trajectory reverses cleanly under time reversal. The hidden charges enter ONLY
through products (``s₁s₂`` in both the tangential inertia ``m_t`` and the hidden
force), so flipping the sign of EVERY charge leaves the dynamics unchanged — and
combined with a parity flip of the initial conditions gives a reflected
trajectory (RESEARCH Q5, the provable all-charge-flip variant).

Each test drives the REAL public entrypoint ``AlienUniverse.run()`` (never a
hand-assembled trajectory) so the symmetry is proved through the production
canonical-polar reduction + Cartesian reconstruction, not just the force law.

Tolerance discipline (mirror tests/benchmarks/test_score_numerical.py:32-109):
  - Translation is EXACT (``array_equal``): the assembly reconstructs in the
    centre-of-mass frame from ``rel = pos[1] − pos[0]`` only, so a uniform shift
    of both bodies is structurally invisible to ``run()``.
  - Rotation / time-reversal / parity↔charge involve a re-derivation of the
    polar IC through ``arctan2`` / sign flips, so they hold to integrator
    tolerance (``allclose`` at a documented ``SYMMETRY_RTOL``), not bitwise.

Binding: 15.0 PLAN Task 1 <behavior>; property→test-map rows 7-10. $0 LLM.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.simulator.alien import AlienUniverse
from ascension.simulator.fixtures import eccentric_fixture

# Non-exact symmetries (rotation / time-reversal / parity↔charge) re-derive the
# polar IC through trig + sign flips, so they hold to integrator tolerance, not
# bitwise. 1e-9 absolute is ~3 orders above the float64 trig round-off of a
# single arctan2/cos/sin and well below the orbit's ~O(1) scale, so it catches a
# genuine symmetry break while tolerating reconstruction round-off.
SYMMETRY_ATOL = 1e-9


def _eccentric_cfg():
    """The full-alien-physics eccentric config (drives every code path)."""
    cfg, _traj = eccentric_fixture()
    return cfg


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _rotate_pairs(arr: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Apply a 2x2 rotation to a tuple-of-2-vectors (the IC pos/vel)."""
    out = []
    for v in arr:
        out.append(tuple(rot @ np.asarray(v, dtype=np.float64)))
    return tuple(out)


def test_rotation_invariance() -> None:
    """Rotating the IC by θ rotates the whole trajectory by θ (central force).

    The Alien forces depend only on the separation r, so the dynamics commute
    with rigid rotations. We rotate both the initial positions and velocities by
    θ, run, and assert each frame equals the θ-rotation of the un-rotated run.
    Holds to integrator tolerance (the polar θ is re-derived via ``arctan2``).
    """
    cfg = _eccentric_cfg()
    theta = 0.7
    rot = _rotation_matrix(theta)

    base = AlienUniverse(cfg).run()
    rotated_cfg = dataclasses.replace(
        cfg,
        ics_pos=_rotate_pairs(cfg.ics_pos, rot),
        ics_vel=_rotate_pairs(cfg.ics_vel, rot),
    )
    rotated = AlienUniverse(rotated_cfg).run()

    # Rotate every body, every frame of the base trajectory by θ; must match.
    expected_pos = base.positions @ rot.T
    expected_vel = base.velocities @ rot.T
    assert np.allclose(
        rotated.positions, expected_pos, atol=SYMMETRY_ATOL
    ), "trajectory is not rotation-invariant (central-force symmetry broken)"
    assert np.allclose(rotated.velocities, expected_vel, atol=SYMMETRY_ATOL)


def test_translation_invariance() -> None:
    """A uniform shift of both bodies leaves the trajectory identical (COM frame).

    ``AlienUniverse`` reconstructs in the centre-of-mass frame from the relative
    coordinate ``rel = pos[1] − pos[0]`` ONLY — the absolute IC position never
    enters. So shifting both bodies by the same vector is structurally invisible:
    the trajectory is BIT-identical (``array_equal``), the strongest claim.
    """
    cfg = _eccentric_cfg()
    shift = np.array([3.5, -1.25], dtype=np.float64)

    base = AlienUniverse(cfg).run()
    shifted_cfg = dataclasses.replace(
        cfg,
        ics_pos=tuple(tuple(np.asarray(p, dtype=np.float64) + shift) for p in cfg.ics_pos),
    )
    shifted = AlienUniverse(shifted_cfg).run()

    # COM-frame reconstruction makes this exact, not merely close.
    np.testing.assert_array_equal(shifted.positions, base.positions)
    np.testing.assert_array_equal(shifted.velocities, base.velocities)


def test_time_reversal() -> None:
    """t→−t, v→−v reproduces the time-reversed trajectory (reversible dynamics).

    The equations of motion are second-order and velocity-independent (the
    forces are position-only; the tangential-inertia term is an inertia, not a
    velocity-dependent force), so they are time-reversible. The clean way to
    prove it on a forward-only integrator: run base forward N steps, then take
    the FINAL state, negate every velocity, and run forward N more steps. A
    reversible system retraces its own path, so frame k of the reversed run
    equals frame (N−k) of the base run for positions (even under v→−v) and the
    negation of it for velocities (odd). Holds to integrator tolerance.

    The implicit-midpoint rule is symmetric (self-adjoint), so this reversal is
    exact up to round-off — the strongest available statement of time-reversal
    symmetry for this production path.
    """
    cfg = _eccentric_cfg()
    base = AlienUniverse(cfg).run()

    # Start the reverse leg from base's FINAL Cartesian state with v negated.
    final_pos = base.positions[-1]
    final_vel = -base.velocities[-1]
    reversed_cfg = dataclasses.replace(
        cfg,
        ics_pos=tuple(tuple(p) for p in final_pos),
        ics_vel=tuple(tuple(v) for v in final_vel),
    )
    rev = AlienUniverse(reversed_cfg).run()

    # Reversible: rev positions retrace base positions read from the end;
    # rev velocities are the negation of base velocities read from the end.
    base_pos_reversed = base.positions[::-1]
    base_vel_reversed = -base.velocities[::-1]
    assert np.allclose(
        rev.positions, base_pos_reversed, atol=SYMMETRY_ATOL
    ), "dynamics are not time-reversal symmetric (positions did not retrace)"
    assert np.allclose(
        rev.velocities, base_vel_reversed, atol=SYMMETRY_ATOL
    ), "dynamics are not time-reversal symmetric (velocities did not reverse)"


def test_parity_charge_symmetry() -> None:
    """All-charge-flip + parity: x_B(t) = −x_A(t) (RESEARCH Q5, provable variant).

    Config A has charges (s₁, s₂) and IC (x₀, v₀); config B has charges
    (−s₁, −s₂) and IC (−x₀, −v₀). The charges enter the dynamics ONLY through
    products (``s₁s₂`` in the tangential inertia ``m_t = μ + κ·s₁s₂`` and in the
    hidden force ``∝ s₁s₂``), so flipping BOTH signs leaves the dynamics
    identical; the parity flip of the IC then reflects the whole motion through
    the origin. Hence x_B(t) = −x_A(t) to integrator tolerance.

    NOTE: the SINGLE-charge-flip variant (flip only s₁) is NOT a clean symmetry
    of this physics (it flips the product sign) and is explicitly flagged to
    Phase 15.1 — see SEED-016. We test only the provable all-flip version here.
    """
    cfg = _eccentric_cfg()
    base = AlienUniverse(cfg).run()

    flipped_cfg = dataclasses.replace(
        cfg,
        charges=tuple(-c for c in cfg.charges),
        ics_pos=tuple(tuple(-np.asarray(p, dtype=np.float64)) for p in cfg.ics_pos),
        ics_vel=tuple(tuple(-np.asarray(v, dtype=np.float64)) for v in cfg.ics_vel),
    )
    flipped = AlienUniverse(flipped_cfg).run()

    assert np.allclose(flipped.positions, -base.positions, atol=SYMMETRY_ATOL), (
        "all-charge-flip + parity did not reflect the trajectory (Q5 symmetry "
        "broken — charges must enter only through products)"
    )
    assert np.allclose(flipped.velocities, -base.velocities, atol=SYMMETRY_ATOL)
