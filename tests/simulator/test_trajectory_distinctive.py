"""ROADMAP criterion 3 / D-07: trajectories visually distinctive but plausible.

Plain English: ROADMAP success criterion 3 (D-07) asks that the tuned Alien
Universe's orbits be VISUALLY DISTINCTIVE from textbook Newtonian gravity (so the
hidden physics is observable, not a Newton look-alike) yet PHYSICALLY PLAUSIBLE
(a real bound orbit, not a numerical explosion or an escape to infinity). The
graphical rendering (VIZ-02) is out of scope here; this module is the VERIFIABLE
NUMERIC proxy for "looks different but is real" — not a plot, an assertion.

Three properties (all from TIER2_TUNED_PARAMS, the recommended operating point):

  1. DISTINCTIVE: the true-law trajectory and a Newton-only twin
     (``replace(cfg, alpha=0, beta=0, charges=(0,0))``) from the SAME IC diverge by
     position RMSE > 1e-2 — well above the 1e-3 numerical-match floor, so the
     hidden physics VISIBLY reshapes the orbit. Observed ≈0.58.

  2. SCALES WITH THE HIDDEN CHARGE: for charges in {(0.3,0.3),(0.6,0.6),(1.0,1.0)}
     the true-vs-Newton divergence is STRICTLY monotonically increasing in
     |s₁·s₂| — the hidden charge's visible signature at the TRAJECTORY level
     (§10 discovery layer 3), complementing the accel-level F-test in
     test_tuning_residual_structure.py. Observed: 0.054 → 0.284 → 0.578.

  3. PLAUSIBLE (bounded): the true orbit is all-finite, never collapses to the
     singularity (min|rel| > 0.1), and never escapes (max|rel| < 5·r0). Observed
     r-range ≈ [1.00, 1.41] with real apsidal precession (RESEARCH §Recommended
     Operating Point) — a genuine bound, precessing orbit.

Drives the REAL ``AlienUniverse.run()`` (NOT a hand-assembled trajectory). NO
``np.random.seed`` (conftest hard rule).

Binding: 15.1 PLAN-02 Task 2 <behavior>; EXP-071 distinctiveness characterization;
ROADMAP success criterion 3 / D-07; §10 discovery layer 3. $0 LLM — deterministic
numpy only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.simulator import config, tuning
from ascension.simulator.alien import AlienUniverse

# Contract bars (assert the CONTRACT, not the incidental tightness).
DISTINCTIVE_RMSE_BAR = 1e-2  # true-vs-Newton must exceed this (≫ the 1e-3 match floor).
COLLAPSE_FLOOR = 0.1  # min|rel| must stay above this (no singularity collapse).
ESCAPE_CEILING_FACTOR = 5.0  # max|rel| must stay below this × r0 (no escape).


def _newton_twin(cfg: config.AlienConfig) -> config.AlienConfig:
    """The Newton-only twin of a config: corrections + charges zeroed.

    ``alpha=0, beta=0, charges=(0,0)`` — charges=0 is REQUIRED so the κ
    tangential-inertia coupling is inert (with charges=(1,1) the κ term still
    bends the orbit and it is NOT a clean Newtonian comparator; the Plan-01
    AS-BUILT negative-control finding).
    """
    return dataclasses.replace(cfg, alpha=0.0, beta=0.0, charges=(0.0, 0.0))


def _position_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_true_orbit_is_distinctive_from_newton() -> None:
    """The true orbit visibly diverges from its Newton twin (D-07 distinctive).

    Run the true law and a Newton-only twin from the SAME IC through the real
    ``AlienUniverse.run()``; the position RMSE between them must exceed 1e-2 —
    well above the 1e-3 numerical-match floor, so the hidden physics VISIBLY bends
    the orbit (it is NOT a Newton look-alike). Observed ≈0.58 (≈58× the bar).
    """
    cfg = tuning.tuned_config()
    newton = _newton_twin(cfg)
    true_traj = AlienUniverse(cfg).run()
    newton_traj = AlienUniverse(newton).run()

    assert np.all(np.isfinite(true_traj.positions)), "true trajectory has non-finite positions"
    assert np.all(np.isfinite(newton_traj.positions)), "newton twin has non-finite positions"

    rmse = _position_rmse(true_traj.positions, newton_traj.positions)
    assert rmse > DISTINCTIVE_RMSE_BAR, (
        f"the true-law orbit is NOT visually distinctive from Newton "
        f"(position RMSE {rmse:.4f} <= bar {DISTINCTIVE_RMSE_BAR:.0e}) — at the "
        "tuned operating point the universe is a Newton look-alike and ROADMAP "
        "criterion 3 / D-07 is not met"
    )


def test_distinctiveness_scales_with_hidden_charge() -> None:
    """True-vs-Newton divergence grows monotonically with |s₁·s₂| (D-07 / layer 3).

    For charges in {(0.3,0.3),(0.6,0.6),(1.0,1.0)} (same IC, only the charge
    product varies), the divergence from the Newton twin must STRICTLY increase
    with |s₁·s₂| — the hidden charge's visible trajectory-level signature (§10
    discovery layer 3), complementing the accel-level F-test. Observed:
    s₁s₂=0.09 → 0.054, 0.36 → 0.284, 1.00 → 0.578 (strictly increasing).
    """
    cfg = tuning.tuned_config()
    newton_positions = AlienUniverse(_newton_twin(cfg)).run().positions

    charge_pairs = [(0.3, 0.3), (0.6, 0.6), (1.0, 1.0)]
    divergences: list[tuple[float, float]] = []
    for ch in charge_pairs:
        true_traj = AlienUniverse(dataclasses.replace(cfg, charges=ch)).run()
        assert np.all(np.isfinite(true_traj.positions)), f"charges={ch} produced non-finite"
        s_prod = abs(ch[0] * ch[1])
        div = _position_rmse(true_traj.positions, newton_positions)
        divergences.append((s_prod, div))

    # Strict monotone increase in |s₁s₂| (the charge pairs are already ordered).
    # NOTE: this is a sliding-window pairwise zip — `divergences[1:]` is one shorter
    # by construction, so NO strict=True here (that would falsely raise on the
    # intended off-by-one window).
    for (s_lo, d_lo), (s_hi, d_hi) in zip(divergences, divergences[1:], strict=False):
        assert d_hi > d_lo, (
            f"true-vs-Newton divergence did NOT increase with the hidden charge: "
            f"|s₁s₂|={s_lo:.2f}→div={d_lo:.4f} then |s₁s₂|={s_hi:.2f}→div={d_hi:.4f} "
            "— the hidden charge has no monotone observable trajectory signature "
            f"(D-07 / §10 layer 3). All: {[(round(s,2), round(d,4)) for s, d in divergences]}"
        )


def test_true_orbit_is_physically_plausible_bounded() -> None:
    """The true orbit is bounded — finite, no collapse, no escape (D-07 plausible).

    A distinctive orbit is only useful if it is also a REAL bound orbit. Assert
    the true trajectory is all-finite, never collapses toward the singularity
    (min|rel| > 0.1), and never escapes (max|rel| < 5·r0). Observed r-range
    ≈ [1.00, 1.41] (a bound, precessing orbit; RESEARCH §Recommended Operating
    Point — real apsidal precession, ~8 perihelion passages / 10k steps).
    """
    cfg = tuning.tuned_config()
    r0 = float(config.TIER2_TUNED_PARAMS["r0"])
    traj = AlienUniverse(cfg).run()

    assert np.all(np.isfinite(traj.positions)), "true orbit has non-finite positions (blowup)"

    rel = traj.positions[:, 1, :] - traj.positions[:, 0, :]
    r = np.linalg.norm(rel, axis=1)
    r_min = float(r.min())
    r_max = float(r.max())

    assert r_min > COLLAPSE_FLOOR, (
        f"true orbit collapses toward the singularity (min|rel| {r_min:.4f} <= "
        f"{COLLAPSE_FLOOR}) — not a physically plausible bound orbit (D-07)"
    )
    assert r_max < ESCAPE_CEILING_FACTOR * r0, (
        f"true orbit escapes (max|rel| {r_max:.4f} >= {ESCAPE_CEILING_FACTOR}·r0="
        f"{ESCAPE_CEILING_FACTOR * r0}) — not a physically plausible bound orbit "
        "(D-07)"
    )
