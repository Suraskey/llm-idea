"""Tests for BenchmarkScore DTO — frozen slots + invariant-doc (06.0 Plan 02 Task 1).

Mirrors tests/sandbox/test_sandbox_types.py:32-117. Invariant-doc-in-source
assertion is Shared Pattern I from PATTERNS.md — the literal string
``rmse is None iff reasons['numerical'] != 'ok'`` must appear in source.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from ascension.benchmarks import score_types
from ascension.benchmarks.score_types import BenchmarkScore


def _make_score(**overrides: object) -> BenchmarkScore:
    defaults: dict[str, object] = {
        "exact": True,
        "rmse": 0.01,
        "qualitative": {"fixed_point_match": True},
        "reasons": {"symbolic": "ok", "numerical": "ok", "qualitative": "ok"},
        "elapsed_ms": {"symbolic": 10, "numerical": 50, "qualitative": 5},
    }
    defaults.update(overrides)
    return BenchmarkScore(**defaults)  # type: ignore[arg-type]


def test_benchmark_score_construction_with_all_five_fields() -> None:
    s = _make_score()
    assert s.exact is True
    assert s.rmse == 0.01
    assert s.qualitative == {"fixed_point_match": True}
    assert s.reasons == {"symbolic": "ok", "numerical": "ok", "qualitative": "ok"}
    assert s.elapsed_ms == {"symbolic": 10, "numerical": 50, "qualitative": 5}


def test_benchmark_score_is_frozen_raises_on_mutation() -> None:
    s = _make_score()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.exact = False  # type: ignore[misc]


def test_benchmark_score_uses_slots() -> None:
    # slots=True means no __dict__ — memory footprint discipline.
    s = _make_score()
    assert not hasattr(s, "__dict__")


def test_benchmark_score_rejects_missing_fields() -> None:
    # Missing `elapsed_ms` → TypeError from dataclass __init__.
    with pytest.raises(TypeError):
        BenchmarkScore(  # type: ignore[call-arg]
            exact=True,
            rmse=0.0,
            qualitative={},
            reasons={},
        )


def test_benchmark_score_fields_count() -> None:
    fields = dataclasses.fields(BenchmarkScore)
    names = {f.name for f in fields}
    assert names == {"exact", "rmse", "qualitative", "reasons", "elapsed_ms"}
    assert len(fields) == 5


def test_benchmark_score_accepts_none_rmse_when_numerical_failed() -> None:
    # DTO accepts the invariant shape; enforcement lives in score().
    s = _make_score(
        rmse=None,
        reasons={"symbolic": "ok", "numerical": "numerical_blowup", "qualitative": "ok"},
    )
    assert s.rmse is None
    assert s.reasons["numerical"] == "numerical_blowup"


def test_benchmark_score_source_contains_invariant() -> None:
    """Shared Pattern I: the invariant string must appear verbatim in source.

    A future dev refactoring the DTO cannot accidentally drop the invariant
    documentation — this grep-regression-test catches the removal.
    """
    src = Path(score_types.__file__).read_text()
    assert "rmse is None iff reasons['numerical'] != 'ok'" in src, (
        "BenchmarkScore source must document the "
        "\"rmse is None iff reasons['numerical'] != 'ok'\" invariant "
        "(Shared Pattern I)."
    )
