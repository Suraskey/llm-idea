"""Smoke tests for the deterministic tuning-analysis library (Phase 15.1, TIER2-02).

Plain English: ``tuning.py`` is the $0-LLM numpy/scipy analysis surface every
downstream 15.1 plan calls. These smoke tests drive the REAL ``AlienUniverse.run()``
(never a hand-assembled trajectory — ``fixtures.py:6-9`` idiom) through each
analysis function at ``TIER2_TUNED_PARAMS`` + a 2-IC slice of ``TIER2_HELD_OUT_ICS``
and assert the make-or-break statistics exist and fire.

Documented-tolerance discipline (mirror ``test_newtonian_limit.py:17-19`` /
``test_hidden_charge.py:45-54``): each assertion tracks the CONTRACT bar
(p<0.01, RMSE<1e-3, Q-drift<1e-6, poly-baseline RMSE>1e-2), NOT the incidental
tightness. Observed margins are noted inline so a reviewer sees the headroom.

$0 LLM — deterministic numpy/scipy/sympy, no agent in any path (D-10).
Binding: 15.1 PLAN Task 2 <behavior>; RESEARCH §Residual-Structure Test,
§Solvability, §Polynomial-Fittability.
"""

from __future__ import annotations

import numpy as np

from ascension.simulator import tuning
from ascension.simulator.config import TIER2_HELD_OUT_ICS, TIER2_TUNED_PARAMS

# Contract bars (D-01/D-02/D-03/D-05/D-11). Assert the bar, not the tightness.
P_BAR = 0.01  # nested-model F-test rejects white-noise null below this.
RMSE_BAR = 1e-3  # true-law held-out RMSE bar (D-03). Observed ≈0 (bit-identical).
Q_DRIFT_BAR = 1e-6  # locked 15.0 conservation bar (D-05). Observed ≈4e-8 here.
POLY_FAIL_BAR = 1e-2  # poly-force baseline must FAIL held-out above this. Obs ≈4.8.


def _tuned_cfg():
    """Build the tuned AlienConfig from TIER2_TUNED_PARAMS via the real IC-builder."""
    return tuning.tuned_config()


def test_newton_fit_residual_returns_finite_trimmed_residual() -> None:
    """newton_fit_residual returns (r, a_radial, e_newton) of the trimmed length.

    Central finite difference of the relative coordinate trims 3 endpoints
    (RESEARCH §Residual-Structure "What is fit" step 2); the residual must be the
    same trimmed length and entirely finite (no silent NaNs).
    """
    cfg = _tuned_cfg()
    r, a_radial, e_newton = tuning.newton_fit_residual(cfg)
    n = cfg.n_steps + 1
    expected = n - 3  # central FD over rel[k-1..k+1] then trim 3 endpoints
    assert r.shape == (expected,)
    assert a_radial.shape == (expected,)
    assert e_newton.shape == (expected,)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(a_radial))
    assert np.all(np.isfinite(e_newton))


def test_nested_model_ftest_fires_at_tuned_point() -> None:
    """The make-or-break statistic exists and fires: p_alpha<0.01 AND p_hidden<0.01.

    The nested-model F-test (Newton → +α → +hidden) rejects the white-noise null
    on both the α-correction and the hidden-charge term at the tuned point.
    Observed: F-stats 10⁴–10⁷, p == 0.0 — ~200 orders below the 0.01 bar. We
    assert the contract bar (RESEARCH §the structure test / D-02).
    """
    cfg = _tuned_cfg()
    out = tuning.nested_model_ftest(cfg)
    assert out["p_alpha"] < P_BAR, f"p_alpha={out['p_alpha']} not < {P_BAR}"
    assert out["p_hidden"] < P_BAR, f"p_hidden={out['p_hidden']} not < {P_BAR}"
    # F-stats are positive (variance is genuinely reduced, not inflated).
    assert out["F_alpha"] > 0.0
    assert out["F_hidden"] > 0.0


def test_held_out_rmse_true_law_near_zero() -> None:
    """held_out_rmse over a 2-IC slice returns RMSE < 1e-3 (true-law-vs-itself).

    The proposed law for the audit IS the true law, so this collapses to "does
    the true law reproduce held-out trajectories to integrator precision". Bar:
    RMSE<1e-3 (D-03); observed ≈0 (bit-identical) — many orders of margin.
    """
    cfg = _tuned_cfg()
    held = TIER2_HELD_OUT_ICS[:2]
    out = tuning.held_out_rmse(cfg, held)
    assert out["max_rmse"] < RMSE_BAR, f"max_rmse={out['max_rmse']} not < {RMSE_BAR}"
    assert len(out["per_ic_rmse"]) == 2
    assert all(v < RMSE_BAR for v in out["per_ic_rmse"])
    assert out["reason"] == "ok"


def test_q_drift_reads_traj_Q_under_bar() -> None:
    """q_drift reads traj.Q (NOT physics.conserved_Q) and returns < 1e-6.

    PATTERNS gotcha 8: the invariant the integrator preserves is the polar-
    Hamiltonian traj.Q, not physics.conserved_Q of the reconstructed Cartesian
    state. Bar 1e-6 (D-05); observed ≈4.1e-8 (24× margin).
    """
    cfg = _tuned_cfg()
    qd = tuning.q_drift(cfg)
    assert qd < Q_DRIFT_BAR, f"q_drift={qd} not < {Q_DRIFT_BAR}"
    assert qd >= 0.0


def test_polynomial_baseline_fails_held_out() -> None:
    """A polynomial-force baseline FAILS held-out forward integration (RMSE>1e-2).

    The D-11 beyond-the-bar guard, RIGHT framing (RESEARCH §Polynomial-Fittability
    §2): fit a polynomial-FORCE central law on the SEEN orbit, forward-integrate
    it (NO κ DOF) from a HELD-OUT IC, compare to truth. The poly force cannot
    reproduce the hidden tangential-inertia dynamics, so held-out RMSE is large
    (observed ≈4.8). We assert it exceeds the 1e-2 floor — the universe is NOT
    accidentally polynomial-solvable. (NOT a per-orbit residual fit — that is the
    L-015 trap; see RESEARCH Pitfall 3.)
    """
    cfg = _tuned_cfg()
    held_out = TIER2_HELD_OUT_ICS[0]
    rmse = tuning.polynomial_baseline_heldout_rmse(cfg, held_out)
    assert rmse > POLY_FAIL_BAR, (
        f"poly-force baseline RMSE {rmse:.3e} <= floor {POLY_FAIL_BAR}; the "
        "universe would be accidentally polynomial-solvable (D-11 broken)"
    )


def test_sweep_marks_tuned_cell_in_band() -> None:
    """sweep returns rows with the EXP-071 in_band flag; the tuned cell is in band.

    sweep is the EXP-071 measurement object (Plan 02 runs the verdict). A single
    cell at the tuned tangential_frac must report in_band=True (all four sub-bars
    cleared) — the smoke proves the row schema and the in_band conjunction exist.
    """
    held = TIER2_HELD_OUT_ICS[:2]
    grid = [{"tangential_frac": float(TIER2_TUNED_PARAMS["tangential_frac"])}]
    rows = tuning.sweep(grid, held)
    assert len(rows) == 1
    row = rows[0]
    for key in ("params", "p_alpha", "p_hidden", "held_out_rmse", "q_drift", "in_band"):
        assert key in row, f"sweep row missing key {key!r}"
    assert row["in_band"] is True
    assert row["p_alpha"] < P_BAR and row["p_hidden"] < P_BAR
    assert row["held_out_rmse"] < RMSE_BAR and row["q_drift"] < Q_DRIFT_BAR


def test_sweep_negative_control_not_in_band() -> None:
    """Negative control: in the true Newtonian limit the residual is NOT structured.

    P-4 in the EXP-071 pre-registration. The negative control is the TRUE
    Newtonian limit (α=0, β=0, charges=0 — the ``newtonian_limit_fixture`` regime),
    NOT merely α=0,β=0: with charges=(1,1) the κ tangential-inertia coupling
    (m_eff=mu+κ·s₁s₂) still bends the orbit, so its radial acceleration is
    genuinely non-Keplerian and the residual IS structured. Only zeroing the
    charges collapses to pure Newton.

    ⚠️ DEVIATION (documented in SUMMARY): the plan's illustrative grid override
    ``{"alpha":0.0,"beta":0.0}`` does NOT produce a clean negative control because
    κ·charges remains active; the physically correct control adds ``charges=(0,0)``.
    AND a bare F-test p-value is vacuous on smooth deterministic residuals
    (RESEARCH Pitfall 2) — even the Newtonian limit's FD-truncation residual
    (~2e-5 relative) rejects white-noise at p≈0. So ``in_band`` is gated on the
    Newton-fit relative-residual EFFECT SIZE (>1e-3), which the Newtonian limit
    fails — that is the falsifiable discriminator. We assert in_band is False AND
    that the residual sits at the FD floor (the physical statement).
    """
    held = TIER2_HELD_OUT_ICS[:2]
    grid = [{"alpha": 0.0, "beta": 0.0, "charges": (0.0, 0.0)}]
    rows = tuning.sweep(grid, held)
    assert len(rows) == 1
    # True Newtonian limit → Newton fits the radial law to the FD floor → the
    # residual effect size is below the floor → NOT in band (the negative control
    # proves the structure test is falsifiable, not vacuous).
    assert rows[0]["in_band"] is False
    assert rows[0]["resid_newton_rel"] < 1e-3, (
        "Newtonian-limit Newton residual should sit at the FD-truncation floor; "
        f"got {rows[0]['resid_newton_rel']:.3e} (the structure test would be vacuous)"
    )


# --- L-064/DET-07: regime knobs thread through, unknown keys fail loudly ----


def test_config_from_params_threads_regime_knobs() -> None:
    from ascension.simulator.config import TIER2_TUNED_PARAMS
    from ascension.simulator.tuning import config_from_params

    params = dict(TIER2_TUNED_PARAMS)
    params["charge_coupling"] = "sum"
    params["correction_exponent"] = 2.5
    cfg = config_from_params(params)
    assert cfg.charge_coupling == "sum"
    assert cfg.correction_exponent == 2.5


def test_config_from_params_rejects_unknown_keys_loudly() -> None:
    import pytest

    from ascension.simulator.config import TIER2_TUNED_PARAMS
    from ascension.simulator.tuning import config_from_params

    params = dict(TIER2_TUNED_PARAMS)
    params["charge_couplng"] = "sum"  # typo'd regime knob must NOT be dropped
    with pytest.raises(ValueError, match="unrecognized param key"):
        config_from_params(params)


def test_nested_model_ftest_uses_the_configs_own_correction_basis() -> None:
    """A V2 config's F-test must fit the V2 basis (r^-2.5), not V1's r^-3.5:
    with the correct basis the +alpha term is decisively detected."""
    from ascension.benchmarks.alien_fixtures import _build_tier2_tuned_cfg_v2
    from ascension.simulator.tuning import nested_model_ftest

    out = nested_model_ftest(_build_tier2_tuned_cfg_v2(seed=0))
    assert out["p_alpha"] < 0.01, f"V2 basis must detect the correction: {out}"
