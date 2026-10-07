"""Pre-flight discoverability audit — truth-vs-truth self-consistency gate.

Plain English: before Phase 7+ agents spend a budget on a benchmark, this
module runs the REAL equations through the SAME scoring pipeline the agent
will face and asserts the pipeline recognises the truth as exact. If it
doesn't, the benchmark itself is broken — no amount of agent cleverness
can fix a scorer that mis-parses its own ground truth.

This is a SELF-CONSISTENCY check, not an information-content gate. It
catches:
  (a) scoring-pipeline regressions (L-013-class — a `_safe_parse_expr`
      change that silently collapses `x0`/`x1` into zero / x),
  (b) DEFAULT_PARAMS drift (registry says one thing, scoring another),
  (c) integrator non-determinism across CI hosts (soft-gate RMSE threshold),
  (d) unknown-system plumbing (typos in spec.system).

It does NOT try to detect held-out-data-quiescence (L-015-class) — that
class requires information-content telemetry that Phase 11 Council
orchestration hasn't produced yet. See D-08 → Phase 6.2 reservation.

Binding decisions (Phase 6.1, locked 2026-04-24):

  * D-02: hard gate = score.exact is True. No rmse threshold on the
    primary decision — L-014's cliff-threshold brittleness lesson. A
    scorer that can't call TRUTH exact cannot be used for any agent run.

  * D-03: soft gate rmse ≤ 1e-6 WARN-only under the default tolerance;
    environment variable ``ASCENSION_AUDIT_RMSE_TOLERANCE`` loosens to
    1e-4 on flake-prone CI hosts. Kwarg > env var > default.

  * D-06: ``@lru_cache(maxsize=256)`` on the internal ``_audit_cached``
    helper; cache key carries a DEFAULT_PARAMS fingerprint so in-process
    edits to the registry invalidate stale results. Public
    ``clear_audit_cache()`` is the test-teardown seam for tests that
    monkeypatch DEFAULT_PARAMS or scoring internals.

  * D-07: unknown spec.system returns
    ``AuditResult(passed=False, reason=UNKNOWN_SYSTEM)``. We do NOT
    raise ValueError — reason codes are DTO strings, exceptions are
    host-level failures only (Shared Pattern D).

  * L-013 retrospective defense — this preflight would catch a parse-
    pipeline regression (e.g., ``_safe_parse_expr`` silently returning
    ``Integer(0)`` for every input). Because the audit runs TRUTH
    through the same ``_safe_parse_expr`` path an agent uses, a
    degenerate parse scores exact=False → reason=NOT_EXACT, and the
    gate raises before any agent burns a budget.

  * L-015 retrospective defense — the known LV instance at IC=[5,5] is
    closed by the D-01 retune to IC=[0.3,0.1] in systems.py. The 3×8
    grid smoke (test_audit.py Test 1) would catch additional
    VdP/Lorenz self-consistency traps if any exist.
    Information-content gating deferred to Phase 6.2 per D-08.

Pitfalls honored:

  * Shared Pattern D (reason-string-on-DTO, no exception-per-failure):
    all audit-time failures surface as ``AuditResult.reason`` codes,
    NOT exceptions. The exception only happens at the
    ``generate_trajectories(..., strict=True)`` wrapper in ode.py.

  * Shared Pattern E (structured logging): every audit run logs INFO on
    OK, WARN on failure with reason + spec summary, DEBUG on cache hit.
    Never silent — engineering_discipline_no_coverups.

  * Shared Pattern G (no exception swallowing): ``BenchmarkIntegrationError``
    from the underlying ``generate_trajectories(strict=False)`` call is
    caught exactly once, at the integration boundary, and converted into
    ``AuditResult(reason=INTEGRATION_FAIL)``. All other exceptions
    propagate.

  * Shared Pattern I (grep-regression invariant): the AuditResult
    docstring contains the literal substring
    ``"reason is None iff passed is True"``, enforced by
    test_audit_result_docstring_contains_invariant_literal.

Threat-model mitigations: see the `<threat_model>` section of PLAN-06.1-01.

  * T-6.1-01 DEFAULT_PARAMS drift → fingerprint-keyed lru_cache.
  * T-6.1-03 DoS on the 24-cell grid → lru_cache keeps per-process cost
    to one integration per unique (spec, tolerance, registry-fingerprint).
  * T-6.1-04 silent-skip repudiation → WARN log on strict=False in
    ode.py; audit.py itself never silences a failure.
  * T-6.1-07 malformed ``ASCENSION_AUDIT_RMSE_TOLERANCE`` → float-parse
    inside try/except; malformed value logs WARN and falls back to the
    default 1e-6. Never crashes the audit on bad env input.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from ascension.benchmarks.score_types import BenchmarkScore
from ascension.benchmarks.systems import DEFAULT_PARAMS
from ascension.benchmarks.types import BenchmarkSpec
from ascension.common.logging import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Reason-code vocabulary (D-07, mirrors reason_codes.py discipline)
# -----------------------------------------------------------------------------


class AuditReasonCode(StrEnum):
    """Fixed 6-code audit failure vocabulary.

    Adding a 7th code is a real decision — it requires a STATE.md
    decision-log entry, not a drive-by commit. Mirrors
    ``reason_codes.ScoreReasonCode`` discipline: downstream consumers
    (session-close summaries, future Phase 16.2 ablation stats) key on
    the raw JSON string, and a silently-appearing new code breaks
    histograms.

    D-08 reservation: information-content codes (e.g.,
    ``INFORMATION_POOR``) are explicitly OUT OF SCOPE for Phase 6.1.
    Those land in Phase 6.2 post-Council telemetry.
    """

    OK = "ok"
    INTEGRATION_FAIL = "integration_fail"
    SYMBOLIC_FAIL = "symbolic_fail"
    NOT_EXACT = "not_exact"
    RMSE_ABOVE_TOLERANCE = "rmse_above_tolerance"
    UNKNOWN_SYSTEM = "unknown_system"


# -----------------------------------------------------------------------------
# AuditResult DTO
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Benchmark discoverability preflight result.

    Invariant: reason is None iff passed is True.

    Attributes:
      passed: Overall audit verdict. True iff the default (or injected)
        policy returned no reason code.
      spec: The ``BenchmarkSpec`` audited. Mirrors
        ``TrajectoryBundle.spec`` — bundle/result are self-describing so
        downstream consumers never have to pair a result with its input
        manually.
      score: The ``BenchmarkScore`` produced by running TRUTH through
        the full ``score()`` pipeline, or ``None`` if the underlying
        integration raised ``BenchmarkIntegrationError`` (INTEGRATION_FAIL
        branch) or the system was unknown (UNKNOWN_SYSTEM branch).
      rmse_tolerance: The effective tolerance applied by the policy.
        Records whether the default (1e-6), env-loosened (1e-4), or
        kwarg-supplied value was in force — useful for session-close
        receipts and D-03 compliance audits.
      reason: Audit reason code; None iff ``passed`` is True.
      elapsed_ms: Total wall-clock milliseconds for the audit call
        (integration + score + policy). Separate from ``score.elapsed_ms``
        so cache-hit latency shows up as ~0ms here even when the
        underlying score took 500ms on the first call.
    """

    passed: bool
    spec: BenchmarkSpec
    score: BenchmarkScore | None
    rmse_tolerance: float
    reason: AuditReasonCode | None
    elapsed_ms: int


# -----------------------------------------------------------------------------
# Internal helpers — fingerprint, truth resolver, default policy
# -----------------------------------------------------------------------------


def _freeze_for_hash(v):
    """Recursively convert mutable containers to hashable tuples.

    Handles the DEFAULT_PARAMS value shapes: ``args`` is a tuple that may
    itself contain lists (e.g., LV's ``([1, 10],)``); ``default_ic`` /
    ``feature_names`` are lists; ``t_span`` is a tuple of floats;
    ``rhs`` is a callable.
    """
    if isinstance(v, list | tuple):
        return tuple(_freeze_for_hash(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze_for_hash(vv)) for k, vv in v.items()))
    if callable(v):
        # rhs / other callables — identity-based hash; stable within a process.
        return ("<callable>", id(v))
    return v


def _default_params_fingerprint() -> int:
    """Stable hash of the DEFAULT_PARAMS registry for lru_cache key use.

    Plain English: if someone edits systems.py in a test
    (monkeypatching DEFAULT_PARAMS), this fingerprint changes, so the
    cached audit result for the old registry gets invalidated
    automatically the next time check_benchmark_discoverability is
    called.

    Not cryptographic — cheap and stable within a process. Uses
    ``_freeze_for_hash`` so nested lists (e.g., LV's ``args=([1, 10],)``
    where the inner list is mutable) are converted to tuples recursively.
    """
    items = []
    for name in sorted(DEFAULT_PARAMS.keys()):
        cfg = DEFAULT_PARAMS[name]
        row_items = tuple(sorted((k, _freeze_for_hash(cfg[k])) for k in cfg.keys()))
        items.append((name, row_items))
    return hash(tuple(items))


def _resolve_truth_rhs(spec: BenchmarkSpec) -> tuple[str, ...]:
    """Default truth resolver — strs of ``scoring._truth_rhs_sympy(spec.system)``.

    Phase 15.1 Alien Universe can inject a different resolver via the
    ``truth_resolver`` kwarg on ``check_benchmark_discoverability``; the
    default path here targets the three built-in ODE systems.
    """
    from ascension.benchmarks.scoring import _truth_rhs_sympy  # lazy, avoids cycle

    return tuple(str(e) for e in _truth_rhs_sympy(spec.system))


def _default_ode_policy(
    sc: BenchmarkScore,
    rmse_tolerance: float,
) -> tuple[bool, AuditReasonCode | None]:
    """Default decision tree for the built-in ODE systems (D-02, D-03).

    First-match-wins: symbolic failure takes precedence over exactness
    (a failed parse can't be called exact or inexact — it's unknown).
    ``exact is False`` with a clean symbolic reason is NOT_EXACT (D-02
    hard gate). RMSE threshold is the D-03 soft gate, applied last.
    """
    if sc.reasons.get("symbolic") != "ok":
        return (False, AuditReasonCode.SYMBOLIC_FAIL)
    if sc.exact is False:
        return (False, AuditReasonCode.NOT_EXACT)  # D-02
    if sc.rmse is not None and sc.rmse > rmse_tolerance:
        return (False, AuditReasonCode.RMSE_ABOVE_TOLERANCE)  # D-03
    return (True, None)


# Per-system default rmse tolerances for the D-03 soft gate (Phase 6.1 added
# 2026-04-24 after discovering truth-vs-truth rmse on Lorenz is ~3.9e-5, not
# ≤ 1e-6, because the scoring truth uses ``nsimplify(2.66667)`` whereas
# DEFAULT_PARAMS['lorenz']['args'] carries ``8/3``. The discrepancy is
# documented at scoring.py:408 and is intentional to match PySINDy's quoted
# literal. Non-chaotic systems (LV/VdP) stay at 1e-6 because their rmse
# floor really is machine-precision for truth-vs-truth. Overridden by
# explicit kwarg or by ASCENSION_AUDIT_RMSE_TOLERANCE (env var); the
# env-var global sweep remains supported per D-03.
_PER_SYSTEM_DEFAULT_RMSE_TOL: dict[str, float] = {
    "lotka_volterra": 1e-6,
    "van_der_pol": 1e-6,
    "lorenz": 1e-3,  # absorbs the beta=8/3 vs 2.66667 sympy discrepancy
}
_FALLBACK_DEFAULT_RMSE_TOL = 1e-6


def _resolve_rmse_tolerance(kwarg: float | None, system: str) -> float:
    """Resolve the effective D-03 rmse tolerance.

    Precedence: kwarg > ``ASCENSION_AUDIT_RMSE_TOLERANCE`` env var >
    per-system default (``_PER_SYSTEM_DEFAULT_RMSE_TOL``) > global fallback
    1e-6.

    Malformed env values fall back to the per-system default with a WARN
    log (T-6.1-07 threat-model mitigation).
    """
    if kwarg is not None:
        return float(kwarg)
    env = os.environ.get("ASCENSION_AUDIT_RMSE_TOLERANCE")
    if env is not None:
        try:
            return float(env)
        except ValueError:
            logger.warning(
                "ASCENSION_AUDIT_RMSE_TOLERANCE=%r is not a valid float; "
                "falling back to per-system default for %s",
                env,
                system,
            )
    return _PER_SYSTEM_DEFAULT_RMSE_TOL.get(system, _FALLBACK_DEFAULT_RMSE_TOL)


# -----------------------------------------------------------------------------
# Core audit machinery — cached helper + public entry point
# -----------------------------------------------------------------------------


def _run_audit(
    spec: BenchmarkSpec,
    rmse_tolerance: float,
    policy,
    truth_resolver,
) -> AuditResult:
    """Uncached audit core. Integration → score → policy → AuditResult.

    Called directly when ``policy`` or ``truth_resolver`` is injected
    (callable identity makes caching unsafe); called via
    ``_audit_cached`` on the default path.
    """
    # Lazy imports to avoid top-level cycle with ode.py (which imports
    # check_benchmark_discoverability from this module when strict=True).
    from ascension.benchmarks.exceptions import BenchmarkIntegrationError
    from ascension.benchmarks.ode import generate_trajectories
    from ascension.benchmarks.scoring import score

    start_ns = time.perf_counter_ns()

    # D-07: unknown system returns structured failure, does NOT raise.
    if spec.system not in DEFAULT_PARAMS:
        elapsed_ms = max(0, (time.perf_counter_ns() - start_ns) // 1_000_000)
        logger.warning(
            "audit unknown system: spec.system=%r not in DEFAULT_PARAMS",
            spec.system,
        )
        return AuditResult(
            passed=False,
            spec=spec,
            score=None,
            rmse_tolerance=rmse_tolerance,
            reason=AuditReasonCode.UNKNOWN_SYSTEM,
            elapsed_ms=int(elapsed_ms),
        )

    truth_rhs = tuple(truth_resolver(spec))

    # strict=False to avoid recursion — we ARE the audit. The WARN log
    # in ode.py's strict=False branch is expected here.
    try:
        bundle = generate_trajectories(spec, strict=False)
    except BenchmarkIntegrationError as exc:
        elapsed_ms = max(0, (time.perf_counter_ns() - start_ns) // 1_000_000)
        logger.warning(
            "audit integration failed: system=%s seed=%d msg=%s",
            spec.system,
            spec.seed,
            exc,
        )
        return AuditResult(
            passed=False,
            spec=spec,
            score=None,
            rmse_tolerance=rmse_tolerance,
            reason=AuditReasonCode.INTEGRATION_FAIL,
            elapsed_ms=int(elapsed_ms),
        )

    sc = score(list(truth_rhs), spec, bundle)
    passed, reason = policy(sc, rmse_tolerance)
    elapsed_ms = max(0, (time.perf_counter_ns() - start_ns) // 1_000_000)

    if passed:
        logger.info(
            "audit OK: system=%s seed=%d rmse=%s tol=%g elapsed_ms=%d",
            spec.system,
            spec.seed,
            f"{sc.rmse:.3e}" if sc.rmse is not None else "None",
            rmse_tolerance,
            elapsed_ms,
        )
    else:
        logger.warning(
            "audit FAIL: system=%s seed=%d reason=%s rmse=%s tol=%g",
            spec.system,
            spec.seed,
            reason,
            f"{sc.rmse:.3e}" if sc.rmse is not None else "None",
            rmse_tolerance,
        )

    return AuditResult(
        passed=passed,
        spec=spec,
        score=sc,
        rmse_tolerance=rmse_tolerance,
        reason=reason,
        elapsed_ms=int(elapsed_ms),
    )


@lru_cache(maxsize=256)
def _audit_cached(
    spec: BenchmarkSpec,
    rmse_tolerance: float,
    params_fingerprint: int,  # noqa: ARG001 — present in key for invalidation
) -> AuditResult:
    """Cached default-path audit. Key includes the DEFAULT_PARAMS fingerprint
    so in-process registry edits invalidate stale entries (D-06, T-6.1-01).
    """
    logger.debug(
        "audit cache miss: system=%s seed=%d rmse_tol=%g fingerprint=%d",
        spec.system,
        spec.seed,
        rmse_tolerance,
        params_fingerprint,
    )
    return _run_audit(
        spec=spec,
        rmse_tolerance=rmse_tolerance,
        policy=_default_ode_policy,
        truth_resolver=_resolve_truth_rhs,
    )


def clear_audit_cache() -> None:
    """Testing helper — drops the internal lru_cache on ``_audit_cached``.

    Callers outside tests/benchmarks/ should NEVER touch this. Production
    code relies on the lru_cache staying warm within a process. This
    shim exists so test suites that monkeypatch DEFAULT_PARAMS (or
    scoring._safe_parse_expr) can force a fresh re-audit in the next
    call.
    """
    _audit_cached.cache_clear()


def check_benchmark_discoverability(
    spec: BenchmarkSpec,
    *,
    policy=None,
    truth_resolver=None,
    rmse_tolerance: float | None = None,
) -> AuditResult:
    """Run the pre-flight discoverability audit for ``spec``.

    Returns an ``AuditResult``. Never raises for correctness-tier
    failures (those are reason codes on the result DTO). Only
    propagates unexpected exceptions (e.g., bugs in an injected policy).

    Args:
      spec: The benchmark spec to audit. Same object the caller will
        later pass to ``generate_trajectories(spec)``.
      policy: Optional decision-tree override. Callable
        ``(BenchmarkScore, float) -> (bool, AuditReasonCode | None)``.
        Phase 15.1 Alien Universe injects its own policy when the
        default ODE branching doesn't map to its success criteria.
        Default: ``_default_ode_policy`` (D-02 + D-03).
      truth_resolver: Optional truth RHS override. Callable
        ``(BenchmarkSpec) -> Iterable[str]`` returning the RHS
        expressions (as strings) to score against the integrated bundle.
        Default: ``_resolve_truth_rhs`` (stringified
        ``scoring._truth_rhs_sympy(spec.system)``).
      rmse_tolerance: Optional override of the D-03 soft-gate threshold.
        Resolution order: kwarg > ``ASCENSION_AUDIT_RMSE_TOLERANCE`` env
        var > per-system default (LV/VdP=1e-6, Lorenz=1e-3 to absorb
        the documented beta=8/3 vs 2.66667 scoring discrepancy from
        scoring.py:408) > global fallback 1e-6.

    Returns:
      ``AuditResult`` with ``passed``, ``spec``, ``score`` (None iff
      integration failed or system unknown), ``rmse_tolerance`` in force,
      ``reason`` (None iff passed), and ``elapsed_ms``.

    Cache semantics:
      - When both ``policy`` and ``truth_resolver`` are None, the result
        is cached under (spec, rmse_tolerance, DEFAULT_PARAMS fingerprint).
      - When either seam is injected, the cache is bypassed entirely —
        callable identity makes caching unsafe across injections.
      - Tests that monkeypatch DEFAULT_PARAMS or scoring internals should
        call ``clear_audit_cache()`` before the audit call.
    """
    effective_tolerance = _resolve_rmse_tolerance(rmse_tolerance, spec.system)

    if policy is None and truth_resolver is None:
        return _audit_cached(
            spec,
            effective_tolerance,
            _default_params_fingerprint(),
        )

    return _run_audit(
        spec=spec,
        rmse_tolerance=effective_tolerance,
        policy=policy if policy is not None else _default_ode_policy,
        truth_resolver=truth_resolver if truth_resolver is not None else _resolve_truth_rhs,
    )


__all__ = [
    "AuditReasonCode",
    "AuditResult",
    "check_benchmark_discoverability",
    "clear_audit_cache",
]
