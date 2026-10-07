"""Make-or-break statistical surface: Newton provably fails at the tuned point.

Plain English: the WHOLE Tier-2 bet is that the Alien Universe is genuinely
non-Newtonian — a position-only discoverer who fits the textbook inverse-square
law to the observed motion is LEFT WITH STRUCTURE they cannot explain. This module
proves that claim is BOTH true AND falsifiable, per IC.

Two halves (mirroring the wide-grid / negative-control discipline of
``test_hidden_charge.py::..._narrow_grid_is_the_failure_mode``):

  1. POSITIVE (the make-or-break): at ``TIER2_TUNED_PARAMS`` couplings, across
     ≥7 eccentric held-out ICs, the nested-model F-test rejects the white-noise
     null at ``p_alpha < 0.01 AND p_hidden < 0.01`` AND the Newton-only fit leaves
     a relative residual ABOVE a 1e-3 effect-size floor. Asserted PER IC (the
     deterministic seed-analog for SCOPE §22.1 — there is no RNG here, so
     robustness-across-initial-conditions is the rigor unit, D-02). The F-stat,
     p-values, and resid_newton_rel ride in every failure message — never
     collapsed to one aggregate number.

  2. NEGATIVE CONTROL (non-vacuity): in the TRUE Newtonian limit (α=0, β=0,
     **charges=(0,0)** — charges=0 is REQUIRED, because with charges=(1,1) the κ
     tangential-inertia coupling STILL bends the orbit and it is NOT a clean
     control; Plan-01 AS-BUILT finding) the Newton-only relative residual sits at
     the finite-difference-truncation FLOOR (< 1e-3). Gate on the EFFECT SIZE, not
     the p-value.

⚠️ WHY NOT THE p-VALUE FOR THE CONTROL (RESEARCH Pitfall 2 / Plan-01 finding): a
bare F-test p-value is VACUOUS on smooth DETERMINISTIC residuals — even the true
Newtonian limit's pure FD-truncation residual (~1.5e-5 relative) "rejects" the
white-noise null at p≈0 (the F-stat is still ~5e4). So a p<0.01 alone proves
NOTHING; it fires for any smooth residual. The discriminator that makes the
positive test meaningful is the EFFECT SIZE: the tuned point's Newton residual is
~5.6e-2 (orders above the floor — real, physically-explained structure), the
Newtonian limit's is ~1.5e-5 (below the floor — pure numerical noise). 1e-3 sits
cleanly between them. This is exactly the autocorrelation-is-vacuous warning
re-skinned for the F-test (see ANTI-AUTOCORRELATION note below).

⚠️ ANTI-AUTOCORRELATION GUARD (D-02, documented not gated): this module does NOT
gate on Ljung-Box / runs / autocorrelation tests. Those are VACUOUS on smooth
deterministic residuals — a deterministic FD-truncation residual is HIGHLY
autocorrelated even in the Newtonian limit (consecutive samples are nearly equal),
so an autocorrelation test would "reject white noise" in the clean negative
control too, exactly like the bare p-value does. The locked structure test is the
nested-model F-test on the per-step radial acceleration GATED BY the effect-size
floor; the ``test_negative_control_pvalue_is_vacuous_not_the_bar`` test below
makes this concrete by showing the negative control's bare p-value DOES fire
(p≈0) — proving why the p-value alone cannot be the bar.

Drives the REAL ``AlienUniverse.run()`` via ``tuning.py`` (never a hand-assembled
trajectory). NO ``np.random.seed`` (conftest hard rule); the eccentric ICs are
deterministic from ``config._tier2_two_body_config`` (the potential_U-derived-speed
idiom).

Binding: 15.1 PLAN-02 Task 1 <behavior>; EXP-071 P-1 / P-4; RESEARCH Pitfall 2;
PATTERNS gotcha 3 (narrow-grid trap). $0 LLM — deterministic numpy/scipy only.
"""

from __future__ import annotations

import dataclasses

from ascension.simulator import config, tuning

# Contract bars (assert the CONTRACT, not the incidental tightness — the
# documented-tolerance discipline of test_newtonian_limit.py:29-30). Observed
# margins are documented inline at each assertion.
P_BAR = 0.01  # nested-model F-test rejects white-noise null below this.
RESID_FLOOR = 1e-3  # Newton-only relative-residual effect-size floor (P-1 / P-4).


def _eccentric_held_out_ics() -> tuple[config.AlienConfig, ...]:
    """Build ≥7 ECCENTRIC held-out ICs (frac ≥ 0.85) spanning distinct orbits.

    The deterministic ``TIER2_HELD_OUT_ICS`` grid contributes only 6 configs at
    frac ≥ 0.85 (RESEARCH/PATTERNS gotcha 3: a narrow near-circular grid lets the
    1/r² envelope dominate and the test would pass for the WRONG reason — the
    hidden structure must be detectable, which needs eccentric orbits). To clear
    the ≥7 bar DETERMINISTICALLY (and without coupling to the jitter-RNG draw
    order in ``build_tier2_held_out_ics``), build a dedicated 3×3 eccentric set
    over r0 ∈ {0.95, 1.05, 1.2} × frac ∈ {0.85, 0.88, 0.90} = 9 ICs, each with a
    ``physics.potential_U``-derived IC speed via the SAME ``config`` builder the
    named held-out constant uses (NEVER a hardcoded v_circ). All fracs sit inside
    the stable band [0.70, 0.90] so no IC blows up.
    """
    cfgs: list[config.AlienConfig] = []
    seed = 100
    for r0 in (0.95, 1.05, 1.2):
        for frac in (0.85, 0.88, 0.90):
            cfgs.append(config._tier2_two_body_config(r0=r0, tangential_frac=frac, seed=seed))
            seed += 1
    return tuple(cfgs)


def test_newton_fit_residual_is_structured_per_eccentric_ic() -> None:
    """POSITIVE: Newton provably fails on EVERY eccentric held-out IC (P-1).

    For each of ≥7 eccentric held-out ICs, the nested-model F-test must reject
    the white-noise null on BOTH the +α and the +hidden-charge step
    (p_alpha < 0.01 AND p_hidden < 0.01) AND leave a Newton-only relative residual
    above the 1e-3 effect-size floor. Asserted PER IC (the deterministic
    seed-analog — robustness across initial conditions, D-02), with the F-stat,
    p-values, and resid_newton_rel in the failure message.

    Observed (this run, at TIER2_TUNED_PARAMS couplings): F_alpha ~ 1e6–1e7,
    F_hidden ~ 1e5, p_alpha = p_hidden = 0.0, resid_newton_rel ~ 5.5e-2..1.2e-1
    across the set (min ~5.46e-2 — orders above the floor). We assert the CONTRACT
    bars (p<0.01, resid>1e-3), not the incidental tightness.
    """
    ics = _eccentric_held_out_ics()
    assert len(ics) >= 7, f"need >=7 eccentric ICs for the seed-analog; got {len(ics)}"

    for i, cfg in enumerate(ics):
        res = tuning.nested_model_ftest(cfg)
        p_alpha = res["p_alpha"]
        p_hidden = res["p_hidden"]
        resid = res["resid_newton_rel"]
        # The conjunction IS the bar: both F-tests reject (Newton + α both
        # insufficient) AND the effect size is meaningful (NOT FD-truncation
        # noise). The effect-size floor is what makes p<0.01 non-vacuous (P-4).
        assert p_alpha < P_BAR and p_hidden < P_BAR and resid > RESID_FLOOR, (
            f"eccentric held-out IC #{i}: Newton-fit residual is NOT structured "
            f"at the contract bar — F_alpha={res['F_alpha']:.3e} "
            f"p_alpha={p_alpha:.3e}, F_hidden={res['F_hidden']:.3e} "
            f"p_hidden={p_hidden:.3e}, resid_newton_rel={resid:.3e} "
            f"(need p_alpha<{P_BAR} AND p_hidden<{P_BAR} AND resid>{RESID_FLOOR}); "
            "the make-or-break Tier-2 claim (Newton provably fails) would be unmet "
            "on this IC (P-1)"
        )


def test_negative_control_newtonian_limit_is_not_structured() -> None:
    """NEGATIVE CONTROL: the TRUE Newtonian limit leaves NO structure (P-4).

    Build the true Newtonian limit ``replace(tuned, alpha=0, beta=0,
    charges=(0,0))`` — charges=0 is REQUIRED (with charges=(1,1) the κ
    tangential-inertia coupling STILL bends the orbit, so it is NOT a clean
    control; Plan-01 AS-BUILT finding). Assert ``resid_newton_rel < 1e-3``: the
    Newton model already fits, so the residual is at the FD-truncation floor.

    ⚠️ Gate on the EFFECT SIZE, NOT the p-value — the p-value is vacuous here
    (proven explicitly in the next test). Observed: resid_newton_rel ~ 1.5e-5
    (~67× below the floor), vs the tuned point's ~5.6e-2. This is what makes the
    positive p<0.01 above MEANINGFUL: the test does NOT fire for any smooth
    deterministic residual, only for one with real physical effect size.
    """
    tuned = tuning.tuned_config()
    neg = dataclasses.replace(tuned, alpha=0.0, beta=0.0, charges=(0.0, 0.0))
    res = tuning.nested_model_ftest(neg)
    resid = res["resid_newton_rel"]
    assert resid < RESID_FLOOR, (
        f"the TRUE Newtonian-limit negative control IS structured "
        f"(resid_newton_rel={resid:.3e} >= floor {RESID_FLOOR}) — the residual "
        "effect-size gate is not falsifiable, so the positive p<0.01 above could "
        "be firing for FD-truncation noise rather than real physics (P-4). Check "
        "that charges=(0,0) (charges=(1,1) leaves the κ coupling active and is NOT "
        "a clean control)."
    )


def test_negative_control_pvalue_is_vacuous_not_the_bar() -> None:
    """The negative control's BARE p-value DOES fire — proving it can't be the bar.

    This is the RESEARCH-Pitfall-2 negative control made explicit (the F-test
    analog of test_hidden_charge's narrow-grid failure-mode test). On a smooth
    DETERMINISTIC residual, the nested-model F-test p-value rejects the
    white-noise null (p≈0, F-stat ~1e4–1e5) EVEN in the true Newtonian limit,
    because the FD-truncation residual is structured numerical noise, not white
    noise. Asserting this is NOT vacuous: it proves WHY the effect-size floor
    (not the p-value) is the locked discriminator, and documents in-code why an
    autocorrelation/Ljung-Box gate would be equally vacuous (a deterministic
    residual is highly autocorrelated in the Newtonian limit too).
    """
    tuned = tuning.tuned_config()
    neg = dataclasses.replace(tuned, alpha=0.0, beta=0.0, charges=(0.0, 0.0))
    res = tuning.nested_model_ftest(neg)
    # The bare p-value fires (p < 0.01) even though there is NO real structure —
    # so a p-value-only gate would FALSELY pass the negative control. Observed:
    # p_alpha = 0.0, F_alpha ~ 5e4. This is the whole point: the p-value is
    # vacuous on smooth deterministic residuals; the effect-size floor is the bar.
    assert res["p_alpha"] < P_BAR, (
        "expected the bare F-test p-value to FIRE on the smooth deterministic "
        "Newtonian-limit residual (the vacuity that motivates the effect-size "
        f"floor) — got p_alpha={res['p_alpha']:.3e}. If this no longer fires, the "
        "FD-truncation characterization in tuning.py has changed; revisit the "
        "RESEARCH-Pitfall-2 rationale."
    )
