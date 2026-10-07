"""Alien-Universe discoverability audit — the Tier-2 L-015 ratchet (Phase 15.1).

Plain English: this is the Phase 6.1 discoverability audit (``benchmarks/audit.py``)
applied to the **Alien Universe** (Tier 2, TIER2-02). Before any Council agent
spends a budget on the alien benchmark, this module runs the simulator's OWN true
law through the SAME symbolic-then-numerical scoring shape the agent's proposed
law will face, and asserts the pipeline recognises the truth as exact. If it does
not, the benchmark itself is broken — no agent cleverness can fix a scorer that
mis-parses its own ground truth (the §10 "the simulator is the oracle" posture,
SCOPE §10 "Validation is not circular").

This is the Tier-2 SELF-CONSISTENCY ratchet: shipped as a CI test, it FAILS LOUD
if a future ``physics.py`` edit, a ``dt`` change, or a sign regression ever makes
the true law stop scoring exact=True at ``config.TIER2_TUNED_PARAMS`` (T-15.1-06).

PATH (b) — why a thin parallel function, not the Phase 6.1 seam-injection (path a):
``benchmarks/audit.py::check_benchmark_discoverability`` is ODE-hardwired. Its
``truth_resolver`` seam returns string RHS expressions and its integration calls
``ode.generate_trajectories``, which knows ONLY the three built-in ODE systems —
an AlienConfig is NOT in ``DEFAULT_PARAMS`` so the audit returns
``UNKNOWN_SYSTEM`` BEFORE any injected seam runs, and ``scoring.score`` is wired
to the ODE RHS strings. Forcing the alien physics through that path would mean
editing ``check_benchmark_discoverability`` / ``score`` and risking the green
Tier-1 audit (T-15.1-07). Instead we REUSE the Phase 6.1 ``AuditReasonCode``
vocabulary verbatim (D-04 / T-15.1-08 — no 7th code) and the integration ->
score -> policy -> result SHAPE, but with alien-native integration
(``AlienUniverse(cfg).run()``) and alien-native scoring.

Why this lives in ``benchmarks/`` (not simulator-side): ``benchmarks/`` is the
higher tier — it consumes simulators. Landing the audit here lets it freely
import the simulator AND reuse the DTO vocabulary in-package with ZERO new import
edges. A simulator-side wrapper would create the repo's ONLY ``simulator ->
benchmarks`` edge just to borrow the DTOs — strictly worse (the resolved
AUDIT-LOCATION fork; benchmarks-side wins the dependency-direction argument).

The two scoring halves (the alien analog of Phase 6.1's symbolic + numerical):
  * symbolic: ``total_accel == -grad potential_U`` to finite-difference tolerance
    (the alien "exact" — the force equals its own potential's gradient; the
    ``test_sympy_force_is_grad_potential`` idiom, test_physics_invariants.py:56).
    A symbolic break -> ``SYMBOLIC_FAIL`` (parse-tier failure) / ``NOT_EXACT``
    (force != grad even though both are well-defined).
  * numerical: for each held-out IC, run the true law forward in a FRESH
    ``AlienUniverse`` and compute RMSE vs a reference forward run (RMSE == 0 by
    construction — the proposed law IS the true law — so the held-out RMSE
    measures reproducibility to integrator precision). Plus the Q-drift read
    (``traj.Q``, the polar-Hamiltonian invariant — NOT ``physics.conserved_Q``,
    PATTERNS gotcha 8). A held-out RMSE or Q-drift over tolerance folds into
    ``RMSE_ABOVE_TOLERANCE`` (no ``Q_DRIFT_ABOVE_TOLERANCE`` code — D-04).

Façade discipline (audit.py:426-428 / [[engineering_discipline_no_coverups]]):
the audit NEVER raises for a correctness failure — it returns the DTO with the
reason code. Only host-level breakage (the integrator cannot run) propagates.
No ``lru_cache``: a thin parallel function runs one deterministic integration per
held-out IC; caching buys nothing and would only hide a re-run regression.

$0 LLM, deterministic numpy/scipy. Lives benchmarks-side, so the §22.6 no-LLM
simulator grep gate does not scope it — and it contains no LLM regardless.

Threat-model mitigations: T-15.1-06 (truth-drift ratchet), T-15.1-07 (no edit to
the Tier-1 audit / score), T-15.1-08 (reuses the 6-code vocabulary verbatim).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ascension.benchmarks.audit import AuditReasonCode
from ascension.common.logging import get_logger
from ascension.simulator import physics as P
from ascension.simulator.alien import AlienUniverse
from ascension.simulator.types import AlienConfig

logger = get_logger(__name__)

# Default held-out numerical-match tolerance (the §10 numerical-agreement bar,
# SCOPE §10 method 2 — RMSE < 1e-3 is a "numerical match"). Reused as the audit's
# soft gate; the true law vs itself sits at ~0, ~infinite margin.
_DEFAULT_RMSE_TOLERANCE = 1e-3

# Conservation admissibility bar (the locked 15.0 Q-drift bound). A held-out run
# whose Q-drift exceeds this is NOT a clean reproduction of the true dynamics, so
# it folds into RMSE_ABOVE_TOLERANCE (D-04 — no distinct Q_DRIFT code).
_Q_DRIFT_BAR = 1e-6

# Symbolic-half radial grid + tolerance. ``total_accel == -grad potential_U`` is a
# relative-agreement claim verified by a 4th-order central finite difference of
# the pairwise potential (the test_physics_invariants.py:56 idiom). Truncation
# sits at ~1e-12; 1e-9 relative is well above the float jitter and tight enough
# to catch a wrong α sign (~5% break) or a wrong U_s sign by many orders.
_SYMBOLIC_GRID_R = (0.6, 0.85, 1.0, 1.4, 2.0)
_SYMBOLIC_RTOL = 1e-9
_FD_H = 1e-5


@dataclass(frozen=True, slots=True)
class AlienAuditResult:
    """Alien-Universe discoverability preflight result.

    Invariant: reason is None iff passed is True.

    Reuses the Phase 6.1 ``AuditReasonCode`` vocabulary verbatim (D-04 — no 7th
    code) and carries the same invariant as ``benchmarks.audit.AuditResult``, but
    with alien-native context fields (symbolic-exactness flag, held-out RMSE,
    Q-drift) instead of the ODE-only ``BenchmarkSpec`` / ``BenchmarkScore``.
    Defining a small alien-native DTO is the least-coupling option: it avoids
    forcing an ODE-shaped ``BenchmarkSpec`` where it does not belong while still
    pinning the audit to the shared reason-code vocabulary.

    Attributes:
      passed: Overall audit verdict. True iff the policy returned no reason code.
      cfg: The ``AlienConfig`` whose true law was audited (self-describing —
        mirrors ``AuditResult.spec``).
      symbolic_exact: True iff ``total_accel == -grad potential_U`` held across
        the radial grid to the finite-difference tolerance (the alien "exact").
      max_held_out_rmse: Max over the held-out set of the true-law-vs-reference
        RMSE (``None`` only if integration failed — INTEGRATION_FAIL branch).
      max_q_drift: Max over the held-out set of ``max|Q − Q[0]|`` (``traj.Q``).
      rmse_tolerance: The effective held-out RMSE tolerance applied by the policy.
      reason: Audit reason code; None iff ``passed`` is True. Drawn from the
        shared 6-code ``AuditReasonCode``.
      elapsed_ms: Total wall-clock milliseconds for the audit call.
    """

    passed: bool
    cfg: AlienConfig
    symbolic_exact: bool
    max_held_out_rmse: float | None
    max_q_drift: float | None
    rmse_tolerance: float
    reason: AuditReasonCode | None
    elapsed_ms: int


# -----------------------------------------------------------------------------
# Scoring halves (alien-native) — symbolic + numerical
# -----------------------------------------------------------------------------


def _symbolic_force_is_grad_potential(cfg: AlienConfig) -> bool:
    """True iff ``total_accel == -grad potential_U`` on the radial grid (the "exact").

    The alien analog of Phase 6.1's symbolic equivalence: a discovered force law
    is exact iff it is the gradient of its own potential. We verify the SIMULATOR's
    own force against the gradient of its own potential — the truth-vs-truth
    self-consistency check (the ``test_sympy_force_is_grad_potential`` idiom). A
    wrong α or U_s sign breaks this by several percent and trips the audit.
    """
    m = np.asarray(cfg.masses, dtype=np.float64)
    ch = np.asarray(cfg.charges, dtype=np.float64)
    center = np.array([0.0, 0.0], dtype=np.float64)
    m_light = float(m[1])

    def _u(rr: float) -> float:
        pos = np.array([center, [rr, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, cfg)

    for r in _SYMBOLIC_GRID_R:
        pos = np.array([center, [r, 0.0]], dtype=np.float64)
        vel = np.zeros_like(pos)
        acc = P.total_accel(pos, vel, cfg)
        f_radial_analytic = m_light * acc[1, 0]
        h = _FD_H
        du_dr = (-_u(r + 2 * h) + 8 * _u(r + h) - 8 * _u(r - h) + _u(r - 2 * h)) / (12 * h)
        f_radial_numeric = -du_dr
        denom = abs(f_radial_analytic)
        if denom == 0.0:
            # A genuinely zero radial force at this r — agreement is absolute.
            if abs(f_radial_numeric) > _SYMBOLIC_RTOL:
                return False
            continue
        rel_err = abs(f_radial_analytic - f_radial_numeric) / denom
        if rel_err >= _SYMBOLIC_RTOL:
            return False
    return True


def _held_out_numerical_match(
    held_out_ics: tuple[AlienConfig, ...],
    reference_override: tuple[AlienConfig, ...] | None,
) -> tuple[float, float, bool]:
    """Return ``(max_rmse, max_q_drift, finite_ok)`` over the held-out set.

    For each held-out IC: run the true law forward in a FRESH ``AlienUniverse``
    (``pred``) and compare ``positions`` to a reference forward run (``ref``). On
    the default path ``ref`` is the SAME IC, so RMSE == 0 to integrator precision
    — the proposed law IS the true law (the §10 numerical-match method, the same
    truth-vs-truth posture as the Tier-1 audit). ``reference_override`` supplies a
    DIFFERENT reference per IC (the regression-detection seam used by the ratchet's
    negative-arm test) so a divergence surfaces as a large RMSE.

    Q-drift reads ``traj.Q`` (the polar-Hamiltonian invariant), NOT
    ``physics.conserved_Q`` of the reconstructed Cartesian state (PATTERNS
    gotcha 8 — same physical quantity, different coordinate, won't match).

    ``finite_ok`` is False if any run produced a non-finite state — surfaced loud
    by the caller, never a silent NaN.
    """
    if reference_override is not None and len(reference_override) != len(held_out_ics):
        raise ValueError(
            "reference_override must align 1:1 with held_out_ics; got "
            f"{len(reference_override)} refs for {len(held_out_ics)} ICs"
        )

    max_rmse = 0.0
    max_q_drift = 0.0
    for idx, ic in enumerate(held_out_ics):
        pred = AlienUniverse(ic).run()
        ref_cfg = reference_override[idx] if reference_override is not None else ic
        ref = AlienUniverse(ref_cfg).run()
        if not (
            np.all(np.isfinite(pred.positions))
            and np.all(np.isfinite(ref.positions))
            and np.all(np.isfinite(pred.Q))
        ):
            return float("inf"), float("inf"), False
        rmse = float(np.sqrt(np.mean((pred.positions - ref.positions) ** 2)))
        q = np.asarray(pred.Q, dtype=np.float64)
        q_drift = float(np.max(np.abs(q - q[0])))
        max_rmse = max(max_rmse, rmse)
        max_q_drift = max(max_q_drift, q_drift)
    return max_rmse, max_q_drift, True


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def alien_discoverability_audit(
    cfg: AlienConfig,
    held_out_ics: tuple[AlienConfig, ...],
    *,
    rmse_tolerance: float = _DEFAULT_RMSE_TOLERANCE,
    reference_override: tuple[AlienConfig, ...] | None = None,
) -> AlienAuditResult:
    """Run the Alien-Universe discoverability audit at ``cfg`` (the L-015 ratchet).

    The "proposed law" under audit IS the true law — the simulator's own physics
    at ``cfg``. There is no string RHS; we score the simulator against itself
    (truth-vs-truth self-consistency, SCOPE §10 "the simulator is the oracle").

    Returns an ``AlienAuditResult``. Never raises for a correctness-tier failure
    (those are reason codes on the DTO — the façade discipline, audit.py:426-428).
    Only host-level breakage (the integrator cannot run at all) propagates.

    Args:
      cfg: The tuned operating point whose true law is audited (the symbolic-half
        force/potential consistency is checked at ``cfg``'s couplings).
      held_out_ics: The held-out generalization configs the true law must
        reproduce (``config.TIER2_HELD_OUT_ICS``). The numerical half runs each
        forward and compares to a reference run.
      rmse_tolerance: Held-out RMSE soft gate (default 1e-3 — the §10
        numerical-match bar). Kwarg > default.
      reference_override: Optional per-IC reference configs (aligned 1:1 with
        ``held_out_ics``). On the default path the reference IS the held-out IC
        (RMSE ~ 0); an override supplies a different reference so a regression /
        mismatch surfaces as a large RMSE -> RMSE_ABOVE_TOLERANCE. This is the
        ratchet's negative-arm seam; it does NOT change the true law, only what it
        is compared against.

    Returns:
      ``AlienAuditResult`` with ``passed``, ``cfg``, ``symbolic_exact``,
      ``max_held_out_rmse`` / ``max_q_drift`` (None iff integration failed),
      ``rmse_tolerance`` in force, ``reason`` (None iff passed), ``elapsed_ms``.

    Reason codes (REUSE the Phase 6.1 vocabulary verbatim — D-04, no 7th code):
      - symbolic break (force != -grad U)     -> NOT_EXACT
      - non-finite / un-runnable held-out      -> INTEGRATION_FAIL
      - held-out RMSE or Q-drift over tolerance-> RMSE_ABOVE_TOLERANCE
      - otherwise                              -> OK (passed, reason None)
    """
    start_ns = time.perf_counter_ns()

    # Symbolic half: the alien "exact" — the force equals its own potential's
    # gradient. A break maps to NOT_EXACT (both quantities are well-defined; they
    # simply disagree — the analog of score.exact is False, not a parse failure).
    symbolic_exact = _symbolic_force_is_grad_potential(cfg)

    # Numerical half: held-out RMSE + Q-drift. Integration breakage (non-finite)
    # is INTEGRATION_FAIL; everything else is a structured number.
    max_rmse, max_q_drift, finite_ok = _held_out_numerical_match(held_out_ics, reference_override)

    elapsed_ms = max(0, (time.perf_counter_ns() - start_ns) // 1_000_000)

    if not symbolic_exact:
        logger.warning(
            "alien audit FAIL: symbolic force != -grad U at cfg "
            "(masses=%s charges=%s alpha=%g beta=%g gamma=%g kappa=%g) — NOT_EXACT",
            cfg.masses,
            cfg.charges,
            cfg.alpha,
            cfg.beta,
            cfg.gamma,
            cfg.kappa,
        )
        return AlienAuditResult(
            passed=False,
            cfg=cfg,
            symbolic_exact=False,
            max_held_out_rmse=max_rmse if finite_ok else None,
            max_q_drift=max_q_drift if finite_ok else None,
            rmse_tolerance=rmse_tolerance,
            reason=AuditReasonCode.NOT_EXACT,
            elapsed_ms=int(elapsed_ms),
        )

    if not finite_ok:
        logger.warning(
            "alien audit FAIL: non-finite held-out integration — INTEGRATION_FAIL",
        )
        return AlienAuditResult(
            passed=False,
            cfg=cfg,
            symbolic_exact=symbolic_exact,
            max_held_out_rmse=None,
            max_q_drift=None,
            rmse_tolerance=rmse_tolerance,
            reason=AuditReasonCode.INTEGRATION_FAIL,
            elapsed_ms=int(elapsed_ms),
        )

    # D-04: Q-drift folds into RMSE_ABOVE_TOLERANCE (the conservation failure
    # surfaces there — no distinct Q_DRIFT_ABOVE_TOLERANCE code).
    if max_rmse > rmse_tolerance or max_q_drift > _Q_DRIFT_BAR:
        logger.warning(
            "alien audit FAIL: held-out RMSE=%.3e (tol=%g) q_drift=%.3e (bar=%g) "
            "— RMSE_ABOVE_TOLERANCE",
            max_rmse,
            rmse_tolerance,
            max_q_drift,
            _Q_DRIFT_BAR,
        )
        return AlienAuditResult(
            passed=False,
            cfg=cfg,
            symbolic_exact=symbolic_exact,
            max_held_out_rmse=max_rmse,
            max_q_drift=max_q_drift,
            rmse_tolerance=rmse_tolerance,
            reason=AuditReasonCode.RMSE_ABOVE_TOLERANCE,
            elapsed_ms=int(elapsed_ms),
        )

    logger.info(
        "alien audit OK: symbolic_exact=True max_rmse=%.3e (tol=%g) "
        "max_q_drift=%.3e (bar=%g) n_held_out=%d elapsed_ms=%d",
        max_rmse,
        rmse_tolerance,
        max_q_drift,
        _Q_DRIFT_BAR,
        len(held_out_ics),
        elapsed_ms,
    )
    return AlienAuditResult(
        passed=True,
        cfg=cfg,
        symbolic_exact=True,
        max_held_out_rmse=max_rmse,
        max_q_drift=max_q_drift,
        rmse_tolerance=rmse_tolerance,
        reason=None,
        elapsed_ms=int(elapsed_ms),
    )


__all__ = [
    "AlienAuditResult",
    "alien_discoverability_audit",
]
