"""Value-object DTO for score().

Frozen slots dataclass — no Pydantic (internal transport, not user-facing
input). Same pattern as src/ascension/sandbox/types.py and
src/ascension/llm/types.py.

Invariants (tested in tests/benchmarks/test_score_types.py):
  - BenchmarkScore.rmse is None iff reasons['numerical'] != 'ok'.
    (rmse is None iff reasons['numerical'] != 'ok' — stated literally here
    so a grep-for-invariant test can confirm the doc presence, same as
    sandbox types.py L5-8 Shared Pattern I.)
  - reasons dict has exactly three keys: 'symbolic', 'numerical',
    'qualitative'. Each value is one of ScoreReasonCode string values.
  - elapsed_ms dict has exactly three keys: 'symbolic', 'numerical',
    'qualitative'.

Binding decisions:
  - 06.0 RESEARCH §Pattern 3 lines 355–376: five-field BenchmarkScore
    shape.
  - PATTERNS Shared Pattern D (reason-string-on-DTO, not
    exception-per-failure-mode).
  - PATTERNS Shared Pattern I (invariant-documented-in-source + grep
    regression test).

Pitfalls honored: N/A (this file is pure DTO).
Threat-model mitigations: N/A.

Plain English: this is the object `score()` returns — three yes/no
answers about whether the proposed equation matches the real one, one
number for how close the predictions came, and a failure-reason string
per answer in case anything went sideways.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    """Three-tier score: symbolic + numerical + qualitative.

    Invariant: rmse is None iff reasons['numerical'] != 'ok'.
    """

    exact: bool  # symbolic equivalence up to tolerance
    rmse: float | None  # None iff numerical branch failed
    qualitative: dict[str, bool | float]  # per-system feature comparison
    reasons: dict[str, str]  # keys: 'symbolic', 'numerical', 'qualitative'
    elapsed_ms: dict[str, int]  # keys: 'symbolic', 'numerical', 'qualitative'
