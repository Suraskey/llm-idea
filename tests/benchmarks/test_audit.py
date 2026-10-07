"""Tests for the Phase 6.1 discoverability audit.

Covers:
  - 3×8 grid smoke (24 cells) across the three built-in systems — catches
    any latent L-015-style self-consistency trap today.
  - L-013 retrospective defense via a ``scoring._safe_parse_expr``
    monkeypatch through the DEFAULT truth_resolver path.
  - ic_spread positivity at the retuned LV IC (D-04, negative-population
    regression).
  - UNKNOWN_SYSTEM branch (D-07) — structured failure, no ValueError.
  - generate_trajectories(strict=True) raises ``BenchmarkIntegrationError``
    when the audit fails (D-05).
  - generate_trajectories(strict=False) logs WARN and returns the bundle
    (D-05).
  - Grep-regression invariant literal in AuditResult docstring.
  - AuditReasonCode vocabulary locked at 6 — no INFORMATION_POOR
    (D-08 reserves that for Phase 6.2).
  - Caller-wins-with-explicit-1.0 sentinel semantics (BLOCKER-2, D-04).
  - ASCENSION_AUDIT_RMSE_TOLERANCE env var loosens the D-03 soft gate
    (WARNING-1).
  - Import-ordering circular-avoidance (WARNING-3) — all 6 permutations
    of ascension.benchmarks / .audit / .ode import cleanly.

Binding decisions (Phase 6.1): D-01 through D-08 per
``.planning/phases/06.1-benchmark-discoverability-audit/06.1-CONTEXT.md``.
"""

from __future__ import annotations

import itertools
import sys

import pytest
import sympy

from ascension.benchmarks import (
    AuditReasonCode,
    AuditResult,
    BenchmarkIntegrationError,
    BenchmarkSpec,
    check_benchmark_discoverability,
    clear_audit_cache,
    generate_trajectories,
)
from ascension.benchmarks.audit import _default_ode_policy  # Test 10 only
from ascension.benchmarks.score_types import BenchmarkScore  # Test 10 only

# -----------------------------------------------------------------------------
# Test 1 — 3 × 8 parametrize grid (24 cells), RESEARCH 6.1-D
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system",
    ["lotka_volterra", "van_der_pol", "lorenz"],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_audit_passes_on_default_spec(system: str, seed: int) -> None:
    """3×8 grid — every built-in system passes truth-vs-truth at every seed.

    If this test ever starts failing at a specific (system, seed) cell, a
    latent L-015-style self-consistency trap has been introduced by a
    DEFAULT_PARAMS edit, a scoring-pipeline change, or a truth-RHS drift.
    """
    clear_audit_cache()
    spec = BenchmarkSpec(
        system=system,  # type: ignore[arg-type]
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=seed,
    )
    result = check_benchmark_discoverability(spec)
    assert isinstance(result, AuditResult)
    assert result.passed, (
        f"{system} seed={seed} failed: reason={result.reason} "
        f"rmse={result.score.rmse if result.score else 'None'}"
    )
    assert result.reason is None  # Invariant: reason is None iff passed is True
    assert result.score is not None
    assert result.score.exact is True
    # INFO-2: self-describing AuditResult — exercise the spec invariant once.
    if system == "lotka_volterra" and seed == 0:
        assert result.spec == spec


# -----------------------------------------------------------------------------
# Test 2 — L-013 retrospective guard via DEFAULT truth_resolver path (BLOCKER-4)
# -----------------------------------------------------------------------------


def test_audit_catches_parse_regression_via_default_path(monkeypatch) -> None:
    """L-013 retrospective defense — recreates the real parse-pipeline failure.

    If ``scoring._safe_parse_expr`` regresses and silently returns a degenerate
    form (e.g., ``sympy.Integer(0)`` regardless of input), the audit's DEFAULT
    truth_resolver path (no injection) scores truth-vs-parsed-garbage and
    yields ``exact=False`` → reason=``NOT_EXACT``.

    Reason-code ordering note (INFO-1): the default ODE policy returns
    ``NOT_EXACT`` because the zero-form proposed RHS parses cleanly
    (``symbolic`` reason is ``'ok'``), so the first-match-wins rule in
    ``_default_ode_policy`` skips ``SYMBOLIC_FAIL`` and lands on
    ``NOT_EXACT`` — which is the correct retrospective verdict for L-013's
    failure mode.

    We clear the lru_cache first (BLOCKER-3) because a prior test's
    cached-pass result would mask the regression.
    """
    from ascension.benchmarks import scoring

    # Force every parse to return Integer(0) — the L-013-class regression signature.
    monkeypatch.setattr(
        scoring,
        "_safe_parse_expr",
        lambda s, local_dict=None: sympy.Integer(0),
    )
    clear_audit_cache()

    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=42,
    )
    result = check_benchmark_discoverability(spec)

    assert not result.passed
    assert result.reason == AuditReasonCode.NOT_EXACT, (
        f"Expected NOT_EXACT (first-match-wins after 'symbolic'=='ok' skip); "
        f"got {result.reason}"
    )


# -----------------------------------------------------------------------------
# Test 3 — ic_spread negative-population regression (D-04 sanity)
# -----------------------------------------------------------------------------


def test_lv_jittered_bundle_has_no_negative_populations() -> None:
    """With retuned default_ic=[0.3, 0.1] and per-system ic_spread=0.03,
    jittered ICs and the integrated clean trajectories must not go
    negative. Uses generate_trajectories(strict=False) to isolate this
    from the audit path.
    """
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=8,
        noise_sigma=0.0,
        n_samples=200,
        seed=42,
    )
    bundle = generate_trajectories(spec, strict=False)
    assert (bundle.ics >= 0).all(), f"negative IC: {bundle.ics}"
    assert (
        bundle.clean >= -1e-9
    ).all(), f"negative population sample: min={bundle.clean.min():.3e}"


# -----------------------------------------------------------------------------
# Test 4 — UNKNOWN_SYSTEM branch (D-07)
# -----------------------------------------------------------------------------


def test_audit_unknown_system_returns_structured_failure() -> None:
    """D-07: unknown system → AuditResult(passed=False, UNKNOWN_SYSTEM),
    NOT ValueError. Reason codes are DTO strings; exceptions are reserved
    for host-level failures (Shared Pattern D).
    """
    spec = BenchmarkSpec(
        system="rossler_not_in_registry",  # type: ignore[arg-type]
        n_trajectories=2,
        noise_sigma=0.0,
        n_samples=10,
        seed=0,
    )
    result = check_benchmark_discoverability(spec)
    assert not result.passed
    assert result.reason == AuditReasonCode.UNKNOWN_SYSTEM
    assert result.score is None


# -----------------------------------------------------------------------------
# Test 5 — strict=True raises BenchmarkIntegrationError on audit failure (D-05)
# -----------------------------------------------------------------------------


def test_generate_trajectories_strict_true_raises_on_audit_fail(
    monkeypatch,
) -> None:
    """Forces audit failure by monkeypatching ``scoring._safe_parse_expr``,
    then confirms ``generate_trajectories`` with ``strict=True`` (the default)
    raises ``BenchmarkIntegrationError`` with ``"discoverability audit failed"``
    substring in the message.
    """
    from ascension.benchmarks import scoring

    monkeypatch.setattr(
        scoring,
        "_safe_parse_expr",
        lambda s, local_dict=None: sympy.Integer(0),
    )
    clear_audit_cache()

    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=42,
    )
    with pytest.raises(BenchmarkIntegrationError) as exc_info:
        generate_trajectories(spec)  # strict=True default

    assert "discoverability audit failed" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Test 6 — strict=False logs WARN and returns bundle (D-05)
# -----------------------------------------------------------------------------


def test_generate_trajectories_strict_false_warns(caplog) -> None:
    """D-05 no-silent-skip — strict=False MUST emit a WARN with the spec
    summary and return the bundle unconditionally.
    """
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=42,
    )
    with caplog.at_level("WARNING"):
        bundle = generate_trajectories(spec, strict=False)

    assert bundle is not None
    assert any(
        "strict=False" in rec.message or "discoverability audit bypassed" in rec.message
        for rec in caplog.records
    ), f"no WARN log seen; records={[r.message for r in caplog.records]}"


# -----------------------------------------------------------------------------
# Test 7 — AuditResult invariant literal (Shared Pattern I grep regression)
# -----------------------------------------------------------------------------


def test_audit_result_docstring_contains_invariant_literal() -> None:
    """Shared Pattern I: the AuditResult docstring MUST carry the literal
    substring "reason is None iff passed is True" so a future reader
    cannot quietly break the invariant without this test catching it.
    """
    from ascension.benchmarks.audit import AuditResult as _AuditResult

    assert "reason is None iff passed is True" in (_AuditResult.__doc__ or "")


# -----------------------------------------------------------------------------
# Test 8 — Fixed enum cardinality (reason_codes.py-style discipline, D-08)
# -----------------------------------------------------------------------------


def test_audit_reason_code_vocabulary_locked() -> None:
    """If this test breaks, you are adding a 7th code. That requires a
    STATE.md decision-log entry, not a drive-by commit.

    Note: D-08 reserves information-content codes for Phase 6.2 — any
    ``INFORMATION_POOR``-style addition is a Phase 6.2 decision, not 6.1.
    """
    assert {c.value for c in AuditReasonCode} == {
        "ok",
        "integration_fail",
        "symbolic_fail",
        "not_exact",
        "rmse_above_tolerance",
        "unknown_system",
    }


# -----------------------------------------------------------------------------
# Test 9 — Sentinel caller-wins with explicit 1.0 (BLOCKER-2)
# -----------------------------------------------------------------------------


def test_caller_wins_with_explicit_one_oh() -> None:
    """BLOCKER-2 regression — an LV caller who explicitly passes
    ``ic_spread=1.0`` gets a 1.0-jittered bundle (NOT silently downgraded
    to 0.03 from the registry). True caller-wins via None sentinel.

    The test compares IC jitter magnitudes between a sentinel-None spec
    (registry=0.03) and an explicit-1.0 spec. At seed=42 with n_traj=8
    the 1.0-jittered bundle has strictly larger jitter radius than the
    0.03-jittered bundle.
    """
    # Tiny t_span_override keeps integration trivial. The test only reads
    # bundle.ics (pre-integration jitter), so we don't need a full window.
    # Without this, ic_spread=1.0 against LV's small default_ic=[0.3, 0.1]
    # jitters ICs to negative populations that blow up under LV dynamics
    # and stall solve_ivp.
    base = dict(
        system="lotka_volterra",
        n_trajectories=2,
        noise_sigma=0.0,
        n_samples=10,
        seed=42,
        t_span_override=(0.0, 0.01),
    )
    spec_registry = BenchmarkSpec(**base)  # type: ignore[arg-type]  # sentinel None → 0.03
    spec_explicit = BenchmarkSpec(**base, ic_spread=1.0)  # type: ignore[arg-type]  # caller wins → 1.0

    assert spec_registry.ic_spread is None
    assert spec_explicit.ic_spread == 1.0

    b_reg = generate_trajectories(spec_registry, strict=False)
    b_exp = generate_trajectories(spec_explicit, strict=False)

    # Default_ic=[0.3, 0.1]; std across trajectories captures jitter magnitude.
    reg_jitter = float(b_reg.ics.std(axis=0).mean())
    exp_jitter = float(b_exp.ics.std(axis=0).mean())
    # explicit 1.0 should produce ~33× larger jitter than registry 0.03.
    assert exp_jitter > 10 * reg_jitter, (
        f"caller-wins broken: registry-jitter={reg_jitter:.4f}, "
        f"explicit-1.0-jitter={exp_jitter:.4f} (expected >10× ratio)"
    )


# -----------------------------------------------------------------------------
# Test 10 — ASCENSION_AUDIT_RMSE_TOLERANCE env var loosens soft gate
# (WARNING-1, D-03)
# -----------------------------------------------------------------------------


def test_rmse_tolerance_env_var_loosens_soft_gate(monkeypatch) -> None:
    """D-03 escape hatch — env var ``ASCENSION_AUDIT_RMSE_TOLERANCE`` raises
    the soft-gate threshold.

    First half: inject a synthetic BenchmarkScore with ``exact=True`` and
    ``rmse=1e-5``. Under the default 1e-6 tolerance the default ODE policy
    trips ``RMSE_ABOVE_TOLERANCE``; under a loosened 1e-4 tolerance the
    same score passes.

    Second half: end-to-end env-var path — set the env var to 1e-4 and
    audit the default LV spec. Confirm the AuditResult records the
    env-loosened tolerance and passes (truth-vs-truth passes at any sane
    tolerance).
    """
    synth = BenchmarkScore(
        exact=True,
        rmse=1e-5,
        qualitative={},
        reasons={"symbolic": "ok"},
        elapsed_ms={},
    )

    # Default tolerance (1e-6): should fail with RMSE_ABOVE_TOLERANCE.
    passed_default, reason_default = _default_ode_policy(synth, 1e-6)
    assert passed_default is False
    assert reason_default == AuditReasonCode.RMSE_ABOVE_TOLERANCE

    # Loosened tolerance (1e-4): should pass.
    passed_loose, reason_loose = _default_ode_policy(synth, 1e-4)
    assert passed_loose is True
    assert reason_loose is None

    # End-to-end env-var path: set the env var and audit a default spec;
    # confirm check_benchmark_discoverability honors it.
    monkeypatch.setenv("ASCENSION_AUDIT_RMSE_TOLERANCE", "1e-4")
    clear_audit_cache()
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=42,
    )
    result = check_benchmark_discoverability(spec)
    assert result.rmse_tolerance == pytest.approx(1e-4)
    assert result.passed  # truth-vs-truth passes at any sane tolerance


# -----------------------------------------------------------------------------
# Test 11 — Circular-import regression (WARNING-3)
# -----------------------------------------------------------------------------


def test_no_top_level_import_ordering_dependency() -> None:
    """WARNING-3 — audit.py <-> ode.py cycle is avoided via lazy imports.
    If a future refactor introduces a top-level import cycle, this test
    fails.

    State-leak fix: do NOT delete `ascension.benchmarks.scoring` from
    sys.modules. Other test modules import symbols from scoring at module
    load (e.g., test_score_numerical.py imports `_numerical_rmse`
    directly) and would hold stale references after a re-import,
    breaking their `monkeypatch.setattr(scoring_mod, "solve_ivp", ...)`
    setup. The audit↔ode cycle check only needs those two modules
    re-imported in different orders.
    """
    import importlib

    mods = [
        "ascension.benchmarks",
        "ascension.benchmarks.audit",
        "ascension.benchmarks.ode",
    ]
    # Snapshot pre-test state so we can restore at end.
    saved = {name: sys.modules[name] for name in mods if name in sys.modules}
    try:
        for order in itertools.permutations(mods):
            # Drop only the three modules under test; do NOT touch scoring.
            for name in mods:
                sys.modules.pop(name, None)
            for name in order:
                importlib.import_module(name)  # must not raise
    finally:
        # Restore the original module objects so any other test holding
        # stale references continues to work.
        for name in mods:
            sys.modules.pop(name, None)
        for name, mod in saved.items():
            sys.modules[name] = mod
