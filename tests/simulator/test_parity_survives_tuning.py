"""Parity↔charge symmetry survives the Phase 15.1 tuning (D-07).

Plain English: Phase 15.0 proved the all-charge-flip parity symmetry on the
eccentric default fixture — flip the sign of EVERY hidden charge and reflect the
initial conditions through the origin (x -> -x, v -> -v), and the whole trajectory
reflects (x_B(t) = -x_A(t)). This file re-validates that SAME symmetry at the
TUNED operating point (``config.TIER2_TUNED_PARAMS``), because tuning changed the
operating point and a symmetry proven at the default must be re-checked where the
benchmark actually runs.

Why all-flip is a clean symmetry (RESEARCH §Parity): every alien force depends on
the charge PRODUCT ``s₁·s₂`` (the hidden force ``∝ s₁s₂`` and the tangential
inertia ``m_t = μ + κ·s₁s₂``), and ``(−s₁)(−s₂) = s₁s₂``. So flipping BOTH signs
leaves the dynamics identical — the all-charge-flip is really PURE SPATIAL PARITY
(the charge flip is a no-op on the product). The parity flip of the IC then
reflects the motion through the origin.

⚠️ D-08: the SINGLE-charge-flip variant (flip only s₁) is NOT a clean symmetry —
it flips the product sign, giving max|pos_D − (−pos_A)| ≈ 1.17 (the whole orbit
scale). Making it one would need new physics analysis; it stays DEFERRED in
SEED-016 and is deliberately NOT added here.

Binding: 15.1-03 PLAN Task 2 <behavior>; analog tests/simulator/test_symmetries.py
:158-192. $0 LLM — deterministic numpy only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.simulator.alien import AlienUniverse
from ascension.simulator.config import TIER2_TUNED_PARAMS
from ascension.simulator.tuning import tuned_config

# Mirror tests/simulator/test_symmetries.py:36-41 — non-exact symmetries
# (parity↔charge re-derives the polar IC through sign flips) hold to integrator
# tolerance, not bitwise. 1e-9 absolute is ~3 orders above the float64 trig
# round-off of a single arctan2/cos/sin and well below the orbit's ~O(1) scale,
# so it catches a genuine symmetry break while tolerating reconstruction
# round-off. Measured at the tuned point: ~1.4e-14 (far inside the bar).
SYMMETRY_ATOL = 1e-9


def test_parity_charge_symmetry_survives_tuning() -> None:
    """All-charge-flip + parity holds at TIER2_TUNED_PARAMS: x_B(t) = −x_A(t).

    Config A is the tuned operating point (charges (s₁, s₂), IC (x₀, v₀)); config
    B has charges (−s₁, −s₂) and IC (−x₀, −v₀). The charges enter only through
    products, so the dynamics are identical and the parity-reflected IC reflects
    the whole motion. We drive the REAL ``AlienUniverse.run()`` for both and
    assert the reflection to ``SYMMETRY_ATOL`` (the analog of the 15.0
    ``test_parity_charge_symmetry`` at the tuned constant rather than the default
    eccentric fixture).
    """
    cfg = tuned_config()
    # Sanity: this really is the tuned operating point (charges from the constant).
    assert tuple(cfg.charges) == tuple(TIER2_TUNED_PARAMS["charges"])  # type: ignore[arg-type]

    base = AlienUniverse(cfg).run()

    flipped_cfg = dataclasses.replace(
        cfg,
        charges=tuple(-c for c in cfg.charges),
        ics_pos=tuple(tuple(-np.asarray(p, dtype=np.float64)) for p in cfg.ics_pos),
        ics_vel=tuple(tuple(-np.asarray(v, dtype=np.float64)) for v in cfg.ics_vel),
    )
    flipped = AlienUniverse(flipped_cfg).run()

    residual_pos = float(np.max(np.abs(flipped.positions - (-base.positions))))
    assert np.allclose(flipped.positions, -base.positions, atol=SYMMETRY_ATOL), (
        "all-charge-flip + parity did not reflect the trajectory at "
        f"TIER2_TUNED_PARAMS (residual {residual_pos:.3e} > {SYMMETRY_ATOL:.0e}) — "
        "charges must enter only through products (Q5 / D-07 symmetry broken)"
    )
    assert np.allclose(
        flipped.velocities, -base.velocities, atol=SYMMETRY_ATOL
    ), "velocities did not reflect under all-charge-flip + parity at the tuned point"
