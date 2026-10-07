"""Simulator reason-code vocabulary — the fixed failure surface.

These are the only failure reasons the simulator audit / result code can
report. Anything not covered by one of these strings is an unexpected crash
that should propagate up (a host-level exception, see ``exceptions.py``), not
get swept under a new code.

Plain English: when a simulator self-check or a per-property gate finishes, it
reports a reason string. "ok" means the check passed cleanly. Everything else
names a specific failure mode so downstream stats and Argus-style audits can
bucket them without guessing.

Binding decisions:
  - 15.0 PATTERNS §reason_codes.py: mirror ``benchmarks/reason_codes.py``
    discipline — ``OK = "ok"`` first, then a closed failure vocabulary.
  - 03.0 D-09 discipline (sandbox 8-code vocabulary): adding a new code is a
    real decision — STATE.md decision-log entry required, not a drive-by
    commit. Downstream readers key on the raw JSON string, so a new code
    silently appearing breaks their histograms.

Pitfalls honored:
  - 15.0 RESEARCH §Make-or-break: a non-conserved trajectory is a
    CONSERVATION_VIOLATION reason code on a result DTO, NOT a raised
    exception (§13 / SCOPE.md:630).
  - 15.0 RESEARCH §Pitfall: a hidden-state leak through ObservationBundle is a
    HIDDEN_STATE_LEAK reason code; the type-boundary split is the control.

Threat-model mitigations: N/A — this file is a pure constants surface.
"""

from __future__ import annotations

from enum import StrEnum


class SimulatorReasonCode(StrEnum):
    """Fixed simulator failure vocabulary.

    Adding a new code is a real decision — it requires a STATE.md
    decision-log entry, not a drive-by commit. Mirrors
    ``benchmarks.reason_codes.ScoreReasonCode`` discipline: downstream
    consumers (session-close summaries, future Phase 15.1/15.2 stats) key on
    the raw JSON string, and a silently-appearing new code breaks histograms.

    The vocabulary maps onto the audit-gate checks (simulator_audit.py):
      - ``NONCONVERGENCE`` — the integrator's inner fixed-point solve did not
        converge within the fixed max_iter cap.
      - ``NONFINITE_STATE`` — a NaN/Inf appeared in the integrated state
        (blowup near perihelion); host-level loud failure (also raises).
      - ``CONSERVATION_VIOLATION`` — Q drifted past the 1e-6 bound over the
        proof horizon (drift-bound check).
      - ``POLYNOMIAL_FITTABLE`` — the hidden-charge force was well-fit by a
        low-degree polynomial (anti-memorization guard; Plan 02/03).
      - ``HIDDEN_STATE_LEAK`` — ObservationBundle carried hidden state
        (charges / Q / true law) it must not.
      - ``HASH_MISMATCH`` — a re-run did not reproduce the recorded
        ``simulator_sha256`` (provenance check; Plan 02/03).
      - ``PROPERTY_UNTESTED`` — a claimed physical property in the frozen
        PROPERTY_TEST_MAP has no collected pytest node (test deleted, file
        renamed, or function renamed). Added Plan 03 (STATE.md decision-log
        entry): the ``_check_property_has_test`` gate introspects pytest
        collection by NAME, so a missing/renamed mapped test must FAIL with a
        dedicated code — NOT be miscoded as POLYNOMIAL_FITTABLE (T-15.0-07).
      - ``LLM_IN_PATH`` — an LLM client / Google generative-AI SDK / embed
        reference was found in a simulator source file (no-LLM gate; Plan 03,
        T-15.0-02).
    """

    OK = "ok"
    NONCONVERGENCE = "nonconvergence"
    NONFINITE_STATE = "nonfinite_state"
    CONSERVATION_VIOLATION = "conservation_violation"
    POLYNOMIAL_FITTABLE = "polynomial_fittable"
    HIDDEN_STATE_LEAK = "hidden_state_leak"
    HASH_MISMATCH = "hash_mismatch"
    PROPERTY_UNTESTED = "property_untested"
    LLM_IN_PATH = "llm_in_path"
