"""Scoring reason-code vocabulary — the fixed 6-string failure surface.

These are the only failure reasons the scoring code can report. Anything
not covered by one of these six strings is an unexpected crash that should
propagate up, not get swept under a seventh code.

Plain English: when `score()` runs over a proposed ODE, each of its three
tiers (symbolic / numerical / qualitative) finishes with a reason string.
"ok" means the tier completed cleanly. Everything else names a specific
failure mode so downstream stats and Argus-style audits can bucket them
without guessing.

Binding decisions:
  - 06.0 RESEARCH §Pattern 3 lines 346–354: six-code scoring vocabulary.
  - 03.0 D-09 discipline (sandbox 8-code vocabulary): adding a 7th code is
    a real decision — STATE.md decision-log entry required, not a drive-by
    commit. Downstream readers (Plan 03 io.py, future Phase 16 stats
    pipeline) key on the raw JSON string, so a new code silently appearing
    breaks their histograms.

Pitfalls honored:
  - RESEARCH §Common Pitfalls #2 (sympy hang) → SYMBOLIC_TIMEOUT surfaces
    the timeout as a DTO field, not as a hang.
  - RESEARCH §Common Pitfalls #8 (blowup detection) → NUMERICAL_BLOWUP
    distinguishes `sol.success=True` + non-finite values from the clean
    `sol.success=False` diverged case.

Threat-model mitigations: N/A — this file is a pure constants surface.
"""

from __future__ import annotations

from enum import StrEnum


class ScoreReasonCode(StrEnum):
    """Fixed 6-code scoring failure vocabulary.

    Adding a 7th code is a real decision — it requires a STATE.md
    decision-log entry, not a drive-by commit. Mirrors 03.0 D-09 discipline
    (sandbox 8-code vocabulary) for the same reason: downstream (Plan 03
    io.py, future Phase 16 stats pipeline) reads the raw JSON string and a
    new code silently appearing breaks their histograms.
    """

    OK = "ok"
    SYMBOLIC_PARSE_FAIL = "symbolic_parse_fail"
    SYMBOLIC_TIMEOUT = "symbolic_timeout"
    NUMERICAL_BLOWUP = "numerical_blowup"
    NUMERICAL_DIVERGED = "numerical_diverged"
    QUALITATIVE_UNKNOWN = "qualitative_unknown"
