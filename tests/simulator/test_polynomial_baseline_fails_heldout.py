"""D-11 beyond-the-bar guard: the universe is NOT accidentally polynomial-solvable.

Plain English: a reviewer's sharpest attack on the Tier-2 bet is "your alien
universe is secretly just a polynomial force law — a baseline that fits a
polynomial to the observed motion would crack it, so the architecture proved
nothing." This module shoots that down HONESTLY: a polynomial-FORCE central law
fit to the SEEN orbit, then FORWARD-INTEGRATED from a held-out IC, FAILS at
RMSE > 1e-2. The polynomial force has no κ tangential-inertia degree of freedom,
so it cannot reproduce the held-out ANGULAR dynamics (the §10 discovery-layer-3
signal). Observed ≈0.51 (≫ the 1e-2 bar; RESEARCH measured ≈4.83 on a different
held-out IC + fit basis — same catastrophic-failure verdict).

⚠️ WHY THE HELD-OUT FORWARD INTEGRATION IS THE HONEST TEST (and the per-orbit
polynomial-RESIDUAL fit is the L-015 TRAP / RESEARCH Pitfall 3): over a single
near-circular orbit's NARROW visited r-range, a high-degree polynomial can crush
the per-orbit acceleration residual to ~0 and FALSELY "prove" the universe is
polynomial-fittable — because it has memorized one orbit's radii, not learned the
force LAW. The honest guard fits on the seen orbit but TESTS on a HELD-OUT IC's
forward integration: a polynomial that merely memorized the seen orbit's r-range
cannot generalize, and the missing κ DOF makes the held-out angular dynamics
diverge. That divergence (RMSE ≫ 1e-2) is the real, non-vacuous failure.

This is the D-11 guard at the FORWARD-INTEGRATION layer — complementary to the
15.0 ``test_hidden_charge_non_polynomial`` wide-grid quartic-fit floor (which is a
property of the force FUNCTION over a wide r-range). That test stays UNTOUCHED
here; the two guards attack the polynomial-fittability claim from different angles.

Drives the REAL ``AlienUniverse.run()`` via ``tuning.py``. NO ``np.random.seed``
(conftest hard rule).

Binding: 15.1 PLAN-02 Task 2 <behavior>; EXP-071 P-5; T-15.1-01 mitigation;
RESEARCH §Polynomial-Fittability §2; PATTERNS "polynomial-anti-pattern" / L-015.
$0 LLM — deterministic numpy/scipy only.
"""

from __future__ import annotations

from ascension.simulator import config, tuning

# Contract bar: the polynomial-force baseline must FAIL held-out forward
# integration above this RMSE (assert the CONTRACT, not the incidental tightness).
POLY_HELD_OUT_FAILURE_BAR = 1e-2  # P-5 / D-11 (measured ≈0.51 here; RESEARCH ≈4.83).


def test_polynomial_force_baseline_fails_held_out_forward_integration() -> None:
    """A polynomial-FORCE baseline fit on the seen orbit FAILS held-out (D-11 / P-5).

    Fit ``a_radial ~ Σ c_k·(1/r)^k`` on the SEEN (TIER2_TUNED_PARAMS) orbit, then
    forward-integrate that central force from a HELD-OUT IC and compare to the true
    ``AlienUniverse`` trajectory. The held-out RMSE must exceed the 1e-2 bar —
    the polynomial law lacks the κ tangential-inertia DOF, so it cannot reproduce
    the held-out angular dynamics. Observed ≈0.51 (≫ the 1e-2 bar).

    ⚠️ This is the HELD-OUT FORWARD-INTEGRATION test, NOT a per-orbit polynomial-
    residual fit. The per-orbit residual fit is the L-015 trap: a degree-8 poly
    crushes a single near-circular orbit's residual and FALSELY passes (memorizing
    one orbit's r-range is not learning the force law). The honest guard generalizes
    to a held-out IC, where the missing κ DOF makes the divergence catastrophic.
    """
    fit_cfg = tuning.tuned_config()
    held_out_cfg = config.TIER2_HELD_OUT_ICS[0]

    rmse = tuning.polynomial_baseline_heldout_rmse(fit_cfg, held_out_cfg)

    assert rmse > POLY_HELD_OUT_FAILURE_BAR, (
        f"polynomial-force baseline SUCCEEDED held-out (RMSE {rmse:.4f} <= bar "
        f"{POLY_HELD_OUT_FAILURE_BAR:.0e}) — the alien universe would be accidentally "
        "polynomial-solvable and the Tier-2 architecture bet would be vacuous "
        "(D-11 / P-5 / T-15.1-01). NOTE this MUST be the held-out forward-"
        "integration RMSE; if it were computed as a per-orbit polynomial-residual "
        "fit it would falsely pass (the L-015 trap)."
    )
