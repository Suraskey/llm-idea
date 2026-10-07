"""Solvability half of the two-sided objective: the true law fits held-out (D-03).

Plain English: proving Newton FAILS (test_tuning_residual_structure.py) is only
half the bet. The OTHER half is that the TRUE law actually SOLVES the universe —
it reproduces held-out trajectories to numerical-match precision. If the true law
could not reproduce its own held-out orbits, the universe would be unsolvable and
the Tier-2 discovery target would be ill-posed. This module proves the §10
numerical-match (solvability) half, plus the D-05 admissibility that the
solvability claim is not computed against a degraded integrator regime.

  - SOLVABILITY (D-03 / P-2): ``tuning.held_out_rmse`` over the full
    ``TIER2_HELD_OUT_ICS`` set returns max RMSE < 1e-3. Here the proposed law IS
    the true law, so this is the true-law-vs-itself reproducibility (observed ≈0,
    bit-identical) — the §10 numerical-match method (analog
    benchmarks/scoring.py:_numerical_rmse; the 15.0 Newtonian-limit RMSE already
    hits ~1.9e-5). We assert the <1e-3 CONTRACT bar, not the incidental tightness.

  - ADMISSIBILITY within the held-out set (D-05 / P-3): max Q-drift across
    ``TIER2_HELD_OUT_ICS`` < 1e-6 (read ``traj.Q`` via ``tuning.q_drift``, the
    polar-Hamiltonian invariant — NOT ``physics.conserved_Q`` of the reconstructed
    Cartesian state; PATTERNS gotcha 8), so no held-out IC sits in a degraded
    regime that would make the solvability number meaningless.

Drives the REAL ``AlienUniverse.run()`` via ``tuning.py`` (fixtures.py:6-9
posture). NO ``np.random.seed`` (conftest hard rule); the held-out ICs are
deterministic from config.

Binding: 15.1 PLAN-02 Task 2 <behavior>; EXP-071 P-2 / P-3. $0 LLM —
deterministic numpy/scipy only.
"""

from __future__ import annotations

from ascension.simulator import config, tuning

# Contract bars (assert the CONTRACT, not the incidental tightness — the
# documented-tolerance discipline of test_newtonian_limit.py:29-30).
HELD_OUT_RMSE_BAR = 1e-3  # §10 numerical-match solvability bar (P-2).
Q_DRIFT_BAR = 1e-6  # the locked 15.0 conservation bar (P-3 / D-05).


def test_true_law_reproduces_held_out_trajectories() -> None:
    """The true law solves every held-out IC at RMSE < 1e-3 (D-03 / P-2).

    Call ``tuning.held_out_rmse`` over the full TIER2_HELD_OUT_ICS set and assert
    the max RMSE clears the <1e-3 numerical-match bar. The proposed law IS the
    true law here, so the RMSE is the true-law-vs-itself reproducibility:
    observed ≈0 (bit-identical) — an effectively infinite margin on the bar. We
    assert the CONTRACT (<1e-3), not the incidental tightness; the reason field
    must be 'ok' (a numerical blowup would surface as max_rmse=inf + reason loudly,
    never a silent NaN — engineering_discipline_no_coverups).
    """
    tuned = tuning.tuned_config()
    result = tuning.held_out_rmse(tuned, config.TIER2_HELD_OUT_ICS)

    assert result["reason"] == "ok", (
        f"held-out RMSE evaluation did not complete cleanly: reason="
        f"{result['reason']!r}, max_rmse={result['max_rmse']} — a held-out IC blew "
        "up (the true law failed to integrate); the solvability claim cannot be "
        "made on a degraded run"
    )
    max_rmse = result["max_rmse"]
    assert max_rmse < HELD_OUT_RMSE_BAR, (
        f"true-law held-out max RMSE {max_rmse:.3e} exceeds the {HELD_OUT_RMSE_BAR:.0e} "
        f"§10 numerical-match bar across {len(config.TIER2_HELD_OUT_ICS)} held-out "
        "ICs — the universe is not solvable to the contract precision (P-2). "
        f"Per-IC RMSE: {[f'{x:.2e}' for x in result['per_ic_rmse']]}"
    )


def test_held_out_set_conserves_within_bar() -> None:
    """Every held-out IC holds Q-drift < 1e-6 (D-05 admissibility / P-3).

    Read ``traj.Q`` via ``tuning.q_drift`` (the polar-Hamiltonian invariant the
    integrator preserves — NOT physics.conserved_Q of the reconstructed Cartesian
    state, PATTERNS gotcha 8) for every held-out IC and assert the max drift over
    1000 steps clears the locked 1e-6 bar. This guards the solvability claim from
    being computed against a held-out IC sitting in a degraded integrator regime.
    Observed: max Q-drift ~2.9e-7 (≈3× margin) across the set.
    """
    drifts = {i: tuning.q_drift(ic) for i, ic in enumerate(config.TIER2_HELD_OUT_ICS)}
    worst_i = max(drifts, key=drifts.get)
    worst = drifts[worst_i]
    assert worst < Q_DRIFT_BAR, (
        f"held-out IC #{worst_i} violates the locked conservation bar "
        f"(Q-drift {worst:.3e} >= {Q_DRIFT_BAR:.0e}) — it sits in a degraded "
        "integrator regime, so the solvability RMSE on it would be meaningless "
        f"(D-05 / P-3). All per-IC Q-drift: "
        f"{ {i: f'{d:.2e}' for i, d in drifts.items()} }"
    )
