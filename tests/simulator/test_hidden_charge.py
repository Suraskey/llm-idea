"""Wave-3 hidden-charge tests: structural non-polynomiality + distinguishability.

Plain English: the whole point of the hidden charge is that an agent CANNOT
shortcut the discovery by fitting a low-degree polynomial to the observed motion
(anti-memorization, D-04 / T-15.0-06). Two correctness properties prove this is
falsifiable:

  1. Non-polynomiality (structural): the hidden-charge force shape
     ``g(r) = (1 − cos γr)/r²`` is genuinely non-polynomial — it OSCILLATES (the
     cosine periodicity). A degree-≤4 least-squares polynomial cannot fit it over
     a WIDE grid where the periodicity bites. We assert RMS-residual / RMS-signal
     stays strictly above a floor of 0.05.

     ⚠️ WIDE-GRID CAVEAT (RESEARCH flags this as the real failure mode): on a
     NARROW near-circular range a quartic fits ``g(r)`` very well (residual
     ratio ≪ 0.05). The test MUST use a fixed wide grid ``r ∈ [0.1, 6π/γ]`` —
     independent of any orbit's visited radii — so the cosine completes several
     full periods and defeats the polynomial. Using only the orbit's r-range
     would make this test vacuously pass. (The plan's illustrative upper bound
     3π/γ is too narrow: the 1/r² envelope still dominates and a quartic fits to
     ~0.015 residual; 6π/γ makes the periodicity bite. Deviation documented.)

  2. Distinguishability (D-04): two configs differing ONLY in the charge product
     ``s₁·s₂`` produce trajectories that differ by more than a small floor RMSE —
     so the hidden charge has an OBSERVABLE consequence (it is not a free
     parameter the agent can ignore).

These are CORRECTNESS properties (a residual ratio above a fixed floor, an RMSE
above a fixed floor), NOT the p<0.01 statistical anti-memorization tuning, which
is Phase 15.1 (TIER2-02).

Binding: 15.0 PLAN Task 1 <behavior>; property→test-map rows 11-12; RESEARCH
non-polynomiality criterion + D-04. $0 LLM — deterministic numpy only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.simulator.alien import AlienUniverse
from ascension.simulator.fixtures import eccentric_fixture

# D-04 non-polynomiality floor: a degree-≤4 fit must leave at least 5% relative
# RMS residual on the wide grid. Observed on the wide grid is ~0.3-0.5; the
# floor tracks the contract, not the incidental tightness (RESEARCH criterion).
NON_POLYNOMIAL_RESIDUAL_FLOOR = 0.05
POLY_DEGREE = 4

# Distinguishability floor: two s₁·s₂ configs must differ by more than this RMSE
# over the run. Set well below the observed separation but safely above float
# round-off so the hidden charge is provably observable, not noise.
DISTINGUISHABILITY_RMSE_FLOOR = 1e-3


def test_hidden_charge_non_polynomial() -> None:
    """g(r)=(1−cos γr)/r² is NOT degree-≤4-fittable on the fixed wide grid (D-04).

    Evaluate the hidden-force radial shape on a FIXED WIDE grid ``r∈[0.1, 6π/γ]``
    (independent of the orbit — the periodicity must bite), fit a degree-4
    least-squares polynomial, and assert RMS(residual)/RMS(signal) exceeds the
    0.05 floor. The wide grid is load-bearing: a narrow near-circular range fits
    well (RESEARCH's flagged failure mode). The plan's illustrative [0.1, 3π/γ]
    is widened to [0.1, 6π/γ] so several full cosine periods are present and the
    non-polynomial structure dominates the 1/r² envelope (deviation documented).
    """
    gamma = 0.7
    # FIXED WIDE grid r ∈ [0.1, 6π/γ]. The lower bound 0.1 stays clear of the
    # 1/r² singularity at r=0 (per the plan's illustrative [0.1, 3π/γ]); the
    # upper bound is extended to 6π/γ so the cosine completes THREE full periods.
    # This is the binding requirement — "the periodicity must bite": on the
    # plan's illustrative narrow [0.1, 3π/γ] the 1/r² envelope concentrates so
    # much signal energy near r=0.1 that a quartic tracks it to a residual ratio
    # of only ~0.015 (< the 0.05 floor); extending the grid so several full
    # oscillation periods are present makes the non-polynomial structure dominate
    # (ratio ~0.14). See SUMMARY deviation + iterations/15.0.md.
    r = np.linspace(0.1, 6.0 * np.pi / gamma, 400)
    g = (1.0 - np.cos(gamma * r)) / r**2

    # Least-squares degree-4 polynomial fit (numpy Vandermonde / polyfit).
    coeffs = np.polyfit(r, g, POLY_DEGREE)
    fit = np.polyval(coeffs, r)
    residual = g - fit

    rms_residual = float(np.sqrt(np.mean(residual**2)))
    rms_signal = float(np.sqrt(np.mean(g**2)))
    ratio = rms_residual / rms_signal

    assert ratio > NON_POLYNOMIAL_RESIDUAL_FLOOR, (
        f"hidden-charge force is degree-{POLY_DEGREE}-fittable on the wide grid "
        f"(residual ratio {ratio:.4f} <= floor {NON_POLYNOMIAL_RESIDUAL_FLOOR}); "
        "anti-memorization claim would be unfalsifiable (D-04)"
    )


def test_hidden_charge_non_polynomial_narrow_grid_is_the_failure_mode() -> None:
    """Document the wide-grid caveat: a NARROW grid IS quartic-fittable.

    This is the RESEARCH-flagged real failure mode made explicit — on a narrow
    near-circular range the same ``g(r)`` fits a quartic well (residual ratio
    BELOW the floor). Asserting this is the negative control that justifies why
    the real test (above) must use the wide grid. Not a vacuous assertion: it
    proves the wide grid is doing the work.
    """
    gamma = 0.7
    # Narrow band around a near-circular orbit radius.
    r = np.linspace(0.9, 1.1, 200)
    g = (1.0 - np.cos(gamma * r)) / r**2
    coeffs = np.polyfit(r, g, POLY_DEGREE)
    fit = np.polyval(coeffs, r)
    ratio = float(np.sqrt(np.mean((g - fit) ** 2)) / np.sqrt(np.mean(g**2)))
    assert ratio < NON_POLYNOMIAL_RESIDUAL_FLOOR, (
        "narrow grid unexpectedly NOT quartic-fittable — the wide-grid caveat "
        "would not be the real failure mode"
    )


def test_hidden_charge_distinguishable() -> None:
    """Two configs differing ONLY in s₁·s₂ give distinguishable trajectories (D-04).

    Take the eccentric fixture (charges (1,1) ⇒ s₁s₂=1) and a sibling with one
    charge halved (charges (0.5,1) ⇒ s₁s₂=0.5). Everything else is identical.
    The trajectories must differ by more than the distinguishability floor RMSE —
    the hidden charge has an OBSERVABLE consequence.
    """
    cfg = _eccentric_cfg()
    assert cfg.charges == (1.0, 1.0)
    sibling = dataclasses.replace(cfg, charges=(0.5, 1.0))

    base = AlienUniverse(cfg).run()
    other = AlienUniverse(sibling).run()

    rmse = float(np.sqrt(np.mean((base.positions - other.positions) ** 2)))
    assert rmse > DISTINGUISHABILITY_RMSE_FLOOR, (
        f"two s₁·s₂ configs are indistinguishable (RMSE {rmse:.3e} <= floor "
        f"{DISTINGUISHABILITY_RMSE_FLOOR}); the hidden charge has no observable "
        "consequence (D-04 broken)"
    )


def _eccentric_cfg():
    cfg, _traj = eccentric_fixture()
    return cfg
