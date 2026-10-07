"""Tests for the Phase 15.1 Alien-Universe discoverability audit (the L-015 ratchet).

This is the Tier-2 analog of ``tests/benchmarks/test_audit.py`` (Phase 6.1). It
proves the Alien Universe's KNOWN-GOOD law is recognised as ``exact=True`` by the
scoring pipeline at ``config.TIER2_TUNED_PARAMS`` BEFORE any Council agent touches
the simulator (D-04). If a future ``physics.py`` edit, a ``dt`` change, or a sign
regression ever makes the true law stop scoring exact, the first ratchet test
below FAILS LOUD — that is the whole point of the audit (T-15.1-06).

Covered behaviors (15.1-03 PLAN Task 1 <behavior>):
  - The ratchet: at TIER2_TUNED_PARAMS + TIER2_HELD_OUT_ICS the audit returns
    ``passed=True``, ``reason is None`` (OK), symbolic-exact True, held-out
    RMSE < 1e-3, Q-drift < 1e-6. Mirrors ``test_audit.py:57-77``.
  - Fail-loud-but-never-raise: a regressed / mismatched reference returns
    ``passed=False`` with ``RMSE_ABOVE_TOLERANCE`` and NO exception (the façade
    discipline, ``audit.py:426-428``).
  - The grep-regression invariant literal lives in the result type's docstring
    (the "reason is None iff passed is True" literal — Shared Pattern I,
    ``test_audit.py:230-237``).
  - The audit REUSES the Phase 6.1 ``AuditReasonCode`` vocabulary verbatim — no
    7th code (D-04 / T-15.1-08).

$0 LLM — deterministic numpy/scipy/sympy only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.benchmarks.alien_audit import (
    AlienAuditResult,
    alien_discoverability_audit,
)
from ascension.benchmarks.audit import AuditReasonCode
from ascension.simulator.config import TIER2_HELD_OUT_ICS
from ascension.simulator.tuning import tuned_config

# -----------------------------------------------------------------------------
# Test 1 — the L-015 ratchet: the true law scores exact=True at the tuned point.
# -----------------------------------------------------------------------------


def test_alien_audit_passes_at_tuned_params() -> None:
    """At TIER2_TUNED_PARAMS the true law is recognised as exact (the ratchet).

    If this test ever starts failing, a regression has degraded the benchmark
    truth — a physics.py edit, a dt change, or a sign flip has made the true law
    stop scoring exact=True. That is the loud failure the audit exists to raise
    (T-15.1-06 / D-04 / [[audit_first_pattern_is_generic]]). Mirrors
    ``test_audit.py::test_audit_passes_on_default_spec``.
    """
    cfg = tuned_config()
    result = alien_discoverability_audit(cfg, TIER2_HELD_OUT_ICS)

    assert isinstance(result, AlienAuditResult)
    assert result.passed, (
        f"true law failed the audit at TIER2_TUNED_PARAMS: reason={result.reason} "
        f"symbolic_exact={result.symbolic_exact} max_rmse={result.max_held_out_rmse} "
        f"max_q_drift={result.max_q_drift}"
    )
    # Invariant: reason is None iff passed is True.
    assert result.reason is None
    # Symbolic half: total_accel == -grad potential_U (the alien "exact").
    assert result.symbolic_exact is True
    # Numerical half: true-law-vs-itself held-out RMSE is ~0 by construction.
    assert result.max_held_out_rmse < 1e-3
    # Conservation: Q-drift across the held-out set clears the locked 1e-6 bar.
    assert result.max_q_drift < 1e-6


def test_alien_audit_reuses_phase_6_1_reason_code_vocabulary() -> None:
    """The audit REUSES the 6-code Phase 6.1 vocabulary verbatim — no 7th code.

    D-04 / T-15.1-08: a silently-added 7th AuditReasonCode breaks downstream
    histograms. Q-drift failures fold into RMSE_ABOVE_TOLERANCE; any new code is
    a STATE.md decision, not a drive-by. This pins the audit to the SAME enum
    object exported from ``benchmarks.audit``.
    """
    assert {c.value for c in AuditReasonCode} == {
        "ok",
        "integration_fail",
        "symbolic_fail",
        "not_exact",
        "rmse_above_tolerance",
        "unknown_system",
    }
    # The OK path uses None (passed) — the audit never invents a new "ok" reason.
    cfg = tuned_config()
    result = alien_discoverability_audit(cfg, TIER2_HELD_OUT_ICS)
    assert result.reason is None
    assert isinstance(AuditReasonCode.OK, AuditReasonCode)


# -----------------------------------------------------------------------------
# Test 2 — fail-loud-but-never-raise on a regressed / mismatched truth.
# -----------------------------------------------------------------------------


def test_alien_audit_fails_loud_on_regressed_truth_without_raising() -> None:
    """A mismatched reference returns passed=False (RMSE_ABOVE_TOLERANCE), no raise.

    Façade discipline (audit.py:426-428): a correctness failure surfaces as a
    reason code on the DTO, NEVER as an exception. We simulate a regression by
    perturbing ONE held-out IC's velocity so the audited true-law run no longer
    reproduces the (perturbed) reference — the held-out RMSE then exceeds the
    tolerance and the audit reports the failure loudly.

    This is the L-015 ratchet's negative arm: it proves the audit actually
    *detects* a divergence rather than passing vacuously.
    """
    cfg = tuned_config()
    # Build a reference set where ONE IC is perturbed relative to what the audit
    # will re-run: we hand the audit a held-out set, but inject a perturbed
    # reference trajectory via the `reference_override` seam so pred != ref.
    #
    # The simulator reconstructs in the COM frame from the RELATIVE velocity
    # (vel[1] - vel[0]), so we must perturb ONLY body 1's velocity — a delta on
    # both bodies cancels in the relative coordinate (translation invariance) and
    # would leave the reference identical to the base (RMSE 0). Bumping just body
    # 1's tangential speed by 5% changes the relative IC and hence the orbit.
    base_ic = TIER2_HELD_OUT_ICS[0]
    v0, v1 = base_ic.ics_vel
    perturbed_v1 = tuple(np.asarray(v1, dtype=np.float64) + np.array([0.0, 0.05]))
    perturbed_ic = dataclasses.replace(base_ic, ics_vel=(v0, perturbed_v1))

    # The audited law runs `base_ic`; the reference is the PERTURBED trajectory,
    # so pred (true law on base_ic) != ref (true law on perturbed_ic) → big RMSE.
    result = alien_discoverability_audit(
        cfg,
        (base_ic,),
        reference_override=(perturbed_ic,),
    )

    assert result.passed is False
    assert result.reason == AuditReasonCode.RMSE_ABOVE_TOLERANCE
    assert result.max_held_out_rmse >= 1e-3
    # No exception was raised — we got here with a structured DTO.


# -----------------------------------------------------------------------------
# Test 3 — grep-regression invariant literal (Shared Pattern I).
# -----------------------------------------------------------------------------


def test_alien_audit_result_docstring_contains_invariant_literal() -> None:
    """The AlienAuditResult docstring MUST carry the invariant literal.

    Shared Pattern I (test_audit.py:230-237): a future reader cannot quietly
    break the ``reason is None iff passed is True`` invariant without this test
    catching it. The alien result DTO carries the SAME invariant as the Phase
    6.1 ``AuditResult``.
    """
    from ascension.benchmarks.alien_audit import AlienAuditResult as _AlienAuditResult

    assert "reason is None iff passed is True" in (_AlienAuditResult.__doc__ or "")
