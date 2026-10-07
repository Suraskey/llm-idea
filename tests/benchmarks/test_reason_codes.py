"""Tests for ScoreReasonCode — 6-code locked vocabulary (06.0 Plan 02 Task 1).

Mirrors tests/sandbox/test_sandbox_types.py:206-221 — StrEnum exact-members
+ string-values + is-str-subclass checks.
"""

from __future__ import annotations

from ascension.benchmarks.reason_codes import ScoreReasonCode


def test_score_reason_code_has_exactly_six_members() -> None:
    assert len(list(ScoreReasonCode)) == 6


def test_score_reason_code_string_values_match_research_pattern_3() -> None:
    assert ScoreReasonCode.OK == "ok"
    assert ScoreReasonCode.SYMBOLIC_PARSE_FAIL == "symbolic_parse_fail"
    assert ScoreReasonCode.SYMBOLIC_TIMEOUT == "symbolic_timeout"
    assert ScoreReasonCode.NUMERICAL_BLOWUP == "numerical_blowup"
    assert ScoreReasonCode.NUMERICAL_DIVERGED == "numerical_diverged"
    assert ScoreReasonCode.QUALITATIVE_UNKNOWN == "qualitative_unknown"


def test_score_reason_code_is_str_enum() -> None:
    # StrEnum members ARE strings — they coerce cleanly into JSON.
    assert isinstance(ScoreReasonCode.OK, str)
    assert isinstance(ScoreReasonCode.SYMBOLIC_TIMEOUT, str)


def test_score_reason_code_names_in_expected_order() -> None:
    names = [m.name for m in ScoreReasonCode]
    assert names == [
        "OK",
        "SYMBOLIC_PARSE_FAIL",
        "SYMBOLIC_TIMEOUT",
        "NUMERICAL_BLOWUP",
        "NUMERICAL_DIVERGED",
        "QUALITATIVE_UNKNOWN",
    ]
