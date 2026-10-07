"""Audit-first self-consistency gate for the Alien Universe simulator.

Implements the Q-consistent §10 formulation's verification gate. AUDIT-FIRST:
this module shipped in Wave 1 with all five checks raising ``NotImplementedError``
so the gate failed LOUD until this plan flips each green. Verification gate
FIRST, optimization SECOND ([[audit_first_pattern_is_generic]], L-017): Phase
7.2 burned ~$8 + 2 sessions doing it backwards.

Two complementary analogs (15.0 PATTERNS §simulator_audit):
  - ``benchmarks/audit.py`` — reason-enum + frozen-slots result DTO + façade
    that NEVER raises on a correctness failure (returns the DTO).
  - ``agora/decay_audit.py`` — the stub→flip-green idiom + the defense-in-depth
    ``except NotImplementedError: raise`` so a regressed stub on a
    previously-green check loud-fails rather than passing spuriously.

The five checks (D-07), all flipped GREEN in Plan 03:
  1. ``_check_property_has_test`` — every claimed physical property in the FROZEN
     PROPERTY_TEST_MAP has a collected pytest node (introspect-by-NAME, NOT a
     file count). A deleted/renamed mapped test FAILS the gate (T-15.0-07).
  2. ``_check_hash_reproducible`` — simulator_sha256 recorded + reproducible,
     and sensitive to a physics-config change.
  3. ``_check_no_hidden_leak``    — ObservationBundle leaks no hidden state.
  4. ``_check_no_llm``            — no LLM in the simulator path.
  5. ``_check_drift_bound``       — Q drift < 1e-6 over 1000 steps (D-01).

Flip-green schedule (mirror decay_audit.py:183-188):
  - Plan 01: all 5 checks raise NotImplementedError (audit-first LOUD stubs).
  - Plan 03 (this): all 5 GREEN.

Reason codes live in ``reason_codes.SimulatorReasonCode`` (the closed vocab).

⚠️ The FROZEN PROPERTY_TEST_MAP below is the canonical property→test contract
(15.0 PLAN Task 2). It is asserted by NAME against live pytest collection. If a
mapped test is legitimately renamed, this map AND the plan's ``<property_test_map>``
table change together, and that change is a STATE.md-worthy decision (mirror the
reason_codes "adding a code is a real decision" posture).

Threat-model mitigations: this gate is the phase-close defense-in-depth for
T-15.0-01 (leak), T-15.0-02 (no-LLM), T-15.0-03 (reproducibility),
T-15.0-04 (drift/blowup), T-15.0-07 (a future plan deleting/renaming a
claimed-property test or re-introducing a stub). $0 LLM — deterministic.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ascension.simulator.reason_codes import SimulatorReasonCode

# -----------------------------------------------------------------------------
# The FROZEN property→test-node map (15.0 PLAN <property_test_map>, B3).
# Each entry pairs a claimed physical property with the EXACT pytest node id
# (file::function) that proves it. _check_property_has_test introspects pytest
# collection for these node ids — a missing OR renamed test FAILS the gate. This
# is NOT "count N files". The 16 rows are the canonical list; they change only
# alongside the plan table (a STATE.md-worthy decision).
# -----------------------------------------------------------------------------
PROPERTY_TEST_MAP: tuple[tuple[str, str], ...] = (
    (
        "Q-conservation drift < 1e-6 / 1000 steps (D-01)",
        "tests/simulator/test_integrator_conservation.py::test_q_conservation",
    ),
    (
        "Q-drift bounded / non-secular over 10k steps (D-09)",
        "tests/simulator/test_integrator_conservation.py::test_q_conservation_stretch",
    ),
    (
        "Newtonian limit <= 1e-3 RMSE vs DOP853 alpha=0 oracle (D-02)",
        "tests/simulator/test_newtonian_limit.py::test_newtonian_limit",
    ),
    (
        "Circular orbit holds r0 (D-03)",
        "tests/simulator/test_analytical_orbits.py::test_circular_orbit",
    ),
    (
        "Parabolic E=0 non-returning, E~0 (D-03)",
        "tests/simulator/test_analytical_orbits.py::test_parabolic_orbit",
    ),
    (
        "Bit-identical reproducibility (np.array_equal, D-06)",
        "tests/simulator/test_reproducibility.py::test_bit_identical_reproducibility",
    ),
    ("Rotation invariance", "tests/simulator/test_symmetries.py::test_rotation_invariance"),
    (
        "Translation invariance (COM frame)",
        "tests/simulator/test_symmetries.py::test_translation_invariance",
    ),
    ("Time-reversal symmetry", "tests/simulator/test_symmetries.py::test_time_reversal"),
    (
        "Parity<->charge (all-charge-flip) symmetry (Q5)",
        "tests/simulator/test_symmetries.py::test_parity_charge_symmetry",
    ),
    (
        "Hidden-charge force non-polynomial, residual ratio > 0.05 (D-04)",
        "tests/simulator/test_hidden_charge.py::test_hidden_charge_non_polynomial",
    ),
    (
        "Two s1.s2 configs distinguishable (D-04)",
        "tests/simulator/test_hidden_charge.py::test_hidden_charge_distinguishable",
    ),
    (
        "ObservationBundle no-leak (field set exact)",
        "tests/simulator/test_observation_contract.py::test_observation_bundle_no_leak",
    ),
    (
        "simulator_sha256 stable on plumbing edits, changes on physics/config",
        "tests/simulator/test_provenance.py::test_simulator_hash_stable",
    ),
    (
        "sympy invariant: total_accel == -grad U AND Lagrangian energy == Q",
        "tests/simulator/test_physics_invariants.py::test_sympy_force_is_grad_potential",
    ),
    (
        "no-LLM grep gate over the simulator path (§22.6)",
        "tests/simulator/test_no_llm_in_simulator.py::test_no_llm_in_simulator_path",
    ),
)

# Repo root (…/src/ascension/simulator/simulator_audit.py → parents[3]).
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class SimulatorAuditResult:
    """Simulator self-consistency audit result.

    Invariant: reason is None iff passed is True.

    Attributes:
      passed: Overall audit verdict. True iff every check returned no reason
        code (i.e., each ``_check_*`` returned None).
      reason: The first failing check's reason code, or None iff ``passed``.
      details: Human-readable context for the verdict (which check, what failed).
      elapsed_ms: Total wall-clock milliseconds for the audit call.
    """

    passed: bool
    reason: SimulatorReasonCode | None
    details: str
    elapsed_ms: int


# -----------------------------------------------------------------------------
# Collection introspection helper for check #1.
# -----------------------------------------------------------------------------


def _collect_pytest_node_ids() -> frozenset[str]:
    """Return the set of pytest node ids collected from tests/simulator/.

    Runs ``pytest --collect-only -q`` in a SUBPROCESS (no plugins, collection
    only — no test bodies execute, no LLM, deterministic) and parses the
    ``file::function`` node ids it prints. Subprocess (not in-process
    ``pytest.main``) keeps the audit free of pytest-session global state and
    re-entrancy when this gate is itself invoked from within a pytest run.
    """
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, repo-local
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/simulator/",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    node_ids: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and line.startswith("tests/"):
            node_ids.add(line)
    return frozenset(node_ids)


# -----------------------------------------------------------------------------
# The five checks — flipped GREEN (Plan 03). Each returns None on pass or a
# SimulatorAuditResult on failure (first-non-None-wins in the façade).
# -----------------------------------------------------------------------------


def _check_property_has_test() -> SimulatorAuditResult | None:
    """D-07 check 1: every claimed property has a NAMED, collected pytest test.

    B3 — introspect, do NOT count files. Asserts every node id in the FROZEN
    ``PROPERTY_TEST_MAP`` is present in live pytest collection. A deleted test,
    a renamed file, or a renamed function makes its node id absent → FAIL with
    ``PROPERTY_UNTESTED`` and the missing node id(s) in details (T-15.0-07).
    """
    started = time.monotonic()
    collected = _collect_pytest_node_ids()
    missing = [node for _prop, node in PROPERTY_TEST_MAP if node not in collected]
    if missing:
        return SimulatorAuditResult(
            passed=False,
            reason=SimulatorReasonCode.PROPERTY_UNTESTED,
            details=(
                "property_has_test: claimed-property test(s) not collected by "
                f"pytest (deleted/renamed): {missing}"
            ),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    return None


def _check_hash_reproducible() -> SimulatorAuditResult | None:
    """D-07 check 2: simulator_sha256 is stable on re-run + sensitive to config.

    Hashes the default config twice (must match) and a perturbed-alpha config
    (must differ). A non-deterministic digest or one insensitive to a physics
    change is a ``HASH_MISMATCH`` failure (RESEARCH Q7 / T-15.0-03).
    """
    started = time.monotonic()
    from ascension.simulator.io import simulator_sha256

    cfg = _default_audit_config()
    d1 = simulator_sha256(cfg)
    d2 = simulator_sha256(cfg)
    if d1 != d2:
        return SimulatorAuditResult(
            passed=False,
            reason=SimulatorReasonCode.HASH_MISMATCH,
            details=f"hash_reproducible: same config gave {d1} != {d2}",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    bumped = dataclasses.replace(cfg, alpha=cfg.alpha + 0.01)
    if simulator_sha256(bumped) == d1:
        return SimulatorAuditResult(
            passed=False,
            reason=SimulatorReasonCode.HASH_MISMATCH,
            details=(
                "hash_reproducible: digest did not change on a perturbed-alpha "
                "config — the hash is insensitive to a physics-config change"
            ),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    return None


def _check_no_hidden_leak() -> SimulatorAuditResult | None:
    """D-07 check 3: ObservationBundle leaks no hidden state (charges / Q / law).

    Builds an ObservationBundle from a real run and asserts its field set is
    exactly {t, positions, velocities, masses} with no charges/Q/config
    attributes. A leak is a ``HIDDEN_STATE_LEAK`` failure (T-15.0-01).
    """
    started = time.monotonic()
    from ascension.simulator.alien import AlienUniverse
    from ascension.simulator.types import ObservationBundle

    allowed = {"t", "positions", "velocities", "masses"}
    forbidden = {"charges", "Q", "config", "library_versions"}

    field_names = {f.name for f in dataclasses.fields(ObservationBundle)}
    if field_names != allowed:
        return SimulatorAuditResult(
            passed=False,
            reason=SimulatorReasonCode.HIDDEN_STATE_LEAK,
            details=(
                f"no_hidden_leak: ObservationBundle field set {sorted(field_names)} "
                f"!= {sorted(allowed)}"
            ),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    cfg = _default_audit_config()
    traj = AlienUniverse(cfg).run()
    bundle = AlienUniverse.to_observation_bundle(traj)
    leaked = [name for name in forbidden if hasattr(bundle, name)]
    if leaked:
        return SimulatorAuditResult(
            passed=False,
            reason=SimulatorReasonCode.HIDDEN_STATE_LEAK,
            details=f"no_hidden_leak: bundle exposes hidden field(s) {leaked}",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    return None


def _check_no_llm() -> SimulatorAuditResult | None:
    """D-07 check 4: no LLM in the simulator path (§22.6 / T-15.0-02).

    Greps every simulator source file for the LLM-client class name, the
    Google generative-AI SDK module name, and the embed-call pattern. Any hit is
    an ``LLM_IN_PATH`` failure. Reuses the no-LLM gate logic the
    test_no_llm_in_simulator suite asserts at the test bar.

    The three regex tokens are assembled from fragments at runtime so this gate's
    OWN source never contains the bare trigger tokens (which would self-match and
    spuriously fail the gate — the same reason the test suite greps the src
    files, not itself).
    """
    started = time.monotonic()
    import re

    simulator_dir = Path(__file__).resolve().parent
    # Assemble the trigger tokens from fragments so they do not appear verbatim
    # in this file (self-match avoidance).
    patterns = (
        r"\b" + "LLM" + "Client" + r"\b",
        r"\b" + "gen" + "ai" + r"\b",
        r"\." + "embed" + r"\s*\(",
    )
    for path in sorted(simulator_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if re.search(pattern, text):
                return SimulatorAuditResult(
                    passed=False,
                    reason=SimulatorReasonCode.LLM_IN_PATH,
                    details=(
                        f"no_llm: pattern {pattern!r} found in {path.name} — "
                        "no LLM may originate from the simulator path (§22.6)"
                    ),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
    return None


def _check_drift_bound() -> SimulatorAuditResult | None:
    """D-07 check 5: Q-drift < 1e-6 over 1000 steps (the make-or-break bar, D-01).

    Runs the eccentric fixture through the REAL ``AlienUniverse.run()`` for 1000
    steps and reads ``traj.Q`` — the genuine polar-Hamiltonian invariant the
    integrator preserves (NOT ``physics.conserved_Q`` of the reconstructed
    Cartesian state, which is the same physical quantity in a different
    coordinate system and would not numerically match a 2-body run). A drift past
    the 1e-6 bound is a ``CONSERVATION_VIOLATION`` failure.
    """
    started = time.monotonic()
    from ascension.simulator.alien import AlienUniverse

    cfg = _default_audit_config()
    traj = AlienUniverse(cfg).run()
    q = traj.Q
    drift = float(max(abs(q - q[0])))
    if drift >= 1e-6:
        return SimulatorAuditResult(
            passed=False,
            reason=SimulatorReasonCode.CONSERVATION_VIOLATION,
            details=(
                f"drift_bound: max|Q-Q0|={drift:.3e} >= 1e-6 over "
                f"{cfg.n_steps} steps (the D-01 make-or-break bar)"
            ),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    return None


def _default_audit_config():
    """The eccentric Q-consistent 2-body config the audit probes (1000 steps).

    Mirrors ``fixtures.eccentric_fixture`` / the Wave-1 conftest recipe: full
    alien physics (α, β, κ, unit charges), tangential speed 0.8·v_circ at r0=1.0,
    dt=0.005, 1000 steps — the canonical make-or-break Q-conservation case.
    Built here (not imported from fixtures.py) so the audit does not depend on
    the test package and so io.py's digest-stability contract holds (fixtures.py
    is not in the hashed dynamics set).
    """
    import numpy as np

    from ascension.simulator import physics as P
    from ascension.simulator.types import AlienConfig

    r0 = 1.0
    masses = (1.0, 1.0)
    charges = (1.0, 1.0)
    alpha, beta, gamma, kappa = 0.05, 0.02, 0.7, 0.3
    probe = AlienConfig(
        G=1.0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kappa=kappa,
        masses=masses,
        charges=charges,
        ics_pos=((0.0, 0.0), (r0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 0.0)),
        dt=0.005,
        n_steps=1000,
        seed=0,
    )
    m = np.asarray(masses, dtype=np.float64)
    ch = np.asarray(charges, dtype=np.float64)
    mu = float(m[0] * m[1] / (m[0] + m[1]))

    def U(r: float) -> float:
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, probe)

    def dUdr(r: float) -> float:
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    return dataclasses.replace(probe, ics_vel=((0.0, 0.0), (0.0, 0.8 * v_circ)))


def run_simulator_audit() -> SimulatorAuditResult:
    """Façade — fixed evaluation order, first-failure-wins, never raises on
    correctness, BUT re-raises NotImplementedError LOUD (defense-in-depth).

    Returns the ``SimulatorAuditResult`` DTO. A correctness failure becomes a
    reason code on the DTO, never an exception (§13 / SCOPE.md:630). But if a
    future plan REGRESSES a green check back to a ``NotImplementedError`` stub,
    the exception propagates so the gate loud-fails rather than silently passing
    (mirror decay_audit.py:1194-1199 / T-15.0-07).

    All references to ``alien.py`` / ``io.py`` / ``physics.py`` are lazy-imported
    inside the check bodies to break any import cycle with this module.
    """
    started_at = time.monotonic()
    checks = (
        ("property_has_test", _check_property_has_test),
        ("hash_reproducible", _check_hash_reproducible),
        ("no_hidden_leak", _check_no_hidden_leak),
        ("no_llm", _check_no_llm),
        ("drift_bound", _check_drift_bound),
    )
    for _label, check in checks:
        try:
            result = check()
        except NotImplementedError:
            # Defense-in-depth: a stub (or a regressed-to-stub) check must
            # loud-fail. This propagates rather than silently passing the gate.
            raise
        if result is not None:
            return result
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    return SimulatorAuditResult(
        passed=True,
        reason=SimulatorReasonCode.OK,
        details="all simulator audit checks passed",
        elapsed_ms=elapsed_ms,
    )
