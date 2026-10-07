"""PIVOT-001 Burst B — the practical-identifiability diagnostic (IDENT-01..04).

These tests are also the worked demonstrations: the alien hidden-charge degeneracy
(L-051) is the product-confound case, and the multi-charge-family fix is the
"add an observation that senses the confounded direction" case — both recovered
from first principles by the generic FIM/sensitivity instrument.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ascension.diagnostics.identifiability import (
    practical_identifiability,
    sensitivity_matrix,
)

_XS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _parallel(a: tuple[float, ...], b) -> float:
    """abs cosine similarity of two direction vectors (1.0 == parallel)."""
    return abs(float(np.dot(_unit(a), _unit(b))))


# --- IDENT-01: identifiable model -------------------------------------------


def test_full_rank_linear_is_identifiable() -> None:
    # f = a*x + b — a and b enter independently across the design => full rank.
    def predict(theta: np.ndarray) -> np.ndarray:
        return theta[0] * _XS + theta[1]

    v = practical_identifiability(predict, [2.0, -1.0], param_names=["a", "b"])
    assert v.identifiable is True
    assert v.status == "identifiable"
    assert v.rank == 2 and v.n_params == 2
    assert math.isfinite(v.condition_number)
    assert v.confounded_directions == ()


# --- IDENT-02: structural non-identifiability + the named degeneracy ---------


def test_sum_confounded_recovers_difference_direction() -> None:
    # f = (a+b)*x — only the SUM a+b is observable; the difference is free.
    def predict(theta: np.ndarray) -> np.ndarray:
        return (theta[0] + theta[1]) * _XS

    v = practical_identifiability(predict, [3.0, 5.0], param_names=["a", "b"])
    assert v.identifiable is False
    assert v.status == "structurally_non_identifiable"
    assert v.rank == 1 and v.n_params == 2
    assert len(v.confounded_directions) == 1
    # The unobservable direction is the difference a - b (i.e. ~[1, -1]).
    assert _parallel(v.confounded_directions[0], [1.0, -1.0]) > 0.999


def test_product_confound_is_the_alien_hidden_charge_case() -> None:
    """L-051 in miniature: f = (s1*s2)*g — at s1 == s2 the two sensitivity columns
    (d/ds1 = s2*g, d/ds2 = s1*g) become collinear, so s1 and s2 are NOT separately
    identifiable. The confounded direction at s1==s2 is exactly [1, -1] (trade one
    charge for the other with no observable effect — the equal-charge degeneracy)."""

    def predict(theta: np.ndarray) -> np.ndarray:
        return (theta[0] * theta[1]) * _XS

    v = practical_identifiability(predict, [1.0, 1.0], param_names=["s1", "s2"])
    assert v.identifiable is False
    assert v.status == "structurally_non_identifiable"
    assert v.rank == 1
    assert _parallel(v.confounded_directions[0], [1.0, -1.0]) > 0.999
    # The disambiguation hint must point at varying the design (the family fix).
    assert (
        "distinct products" in v.disambiguation_hint.lower()
        or "vary" in v.disambiguation_hint.lower()
    )


def test_product_confound_direction_tracks_the_operating_point() -> None:
    # Away from s1==s2 the collapsed direction rotates to [s1, -s2] (here [2, -3]):
    # columns are [s2*g, s1*g], so the null vector w obeys w0*s2 + w1*s1 = 0.
    def predict(theta: np.ndarray) -> np.ndarray:
        return (theta[0] * theta[1]) * _XS

    v = practical_identifiability(predict, [2.0, 3.0], param_names=["s1", "s2"])
    assert v.rank == 1
    assert _parallel(v.confounded_directions[0], [2.0, -3.0]) > 0.999


# --- IDENT-03: adding the disambiguating observation restores identifiability -


def test_adding_an_observation_that_senses_the_confound_restores_rank() -> None:
    """The engine's move (Burst C in miniature): the product s1*s2 alone is rank-1,
    but ALSO observing a configuration that senses s1 on its own breaks the
    collinearity => full rank => identifiable. This is exactly why a multi-charge
    family (configs at distinct products) restores the hidden charge."""

    def predict_product_only(theta: np.ndarray) -> np.ndarray:
        return (theta[0] * theta[1]) * _XS

    def predict_with_extra_config(theta: np.ndarray) -> np.ndarray:
        product_obs = (theta[0] * theta[1]) * _XS
        s1_sensing_obs = theta[0] * _XS  # a config whose amplitude scales with s1 alone
        return np.concatenate([product_obs, s1_sensing_obs])

    before = practical_identifiability(predict_product_only, [1.0, 1.0])
    after = practical_identifiability(predict_with_extra_config, [1.0, 1.0])
    assert before.identifiable is False and before.rank == 1
    assert after.identifiable is True and after.rank == 2


# --- practical (sloppy) non-identifiability ----------------------------------


def test_sloppy_model_is_practically_non_identifiable() -> None:
    # Two nearly-parallel sensitivity columns: full rank, but a ~1e6 condition
    # number => the weak direction is swamped by realistic noise.
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    omega = 1e-6 * np.array([1.0, -1.0, 1.0, -1.0, 1.0])

    def predict(theta: np.ndarray) -> np.ndarray:
        return theta[0] * a + theta[1] * (a + omega)

    v = practical_identifiability(predict, [1.0, 1.0], cond_threshold=1e4)
    assert v.status == "practically_non_identifiable"
    assert v.identifiable is False
    assert v.rank == 2  # full rank — it is sloppy, not structurally collapsed
    assert v.condition_number > 1e4


def test_insensitive_parameter_is_flagged() -> None:
    # b does nothing at all => its direction collapses (rank 1 of 2).
    def predict(theta: np.ndarray) -> np.ndarray:
        return theta[0] * _XS + 0.0 * theta[1]

    v = practical_identifiability(predict, [2.0, 9.0], param_names=["a", "b"])
    assert v.identifiable is False
    assert v.rank == 1
    # The collapsed direction is the pure-b axis.
    assert _parallel(v.confounded_directions[0], [0.0, 1.0]) > 0.999


# --- robustness / discipline -------------------------------------------------


def test_sensitivity_matrix_loud_on_inconsistent_length() -> None:
    # A design that changes output length under perturbation is a bug, not a
    # rank drop — must fail loud (engineering_discipline_no_coverups), not silently.
    calls = {"n": 0}

    def bad_predict(theta: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return np.ones(3 if calls["n"] % 2 else 4)

    with pytest.raises(ValueError):
        sensitivity_matrix(bad_predict, np.array([1.0]))


def test_verdict_describe_is_deterministic_text() -> None:
    def predict(theta: np.ndarray) -> np.ndarray:
        return (theta[0] + theta[1]) * _XS

    v = practical_identifiability(predict, [1.0, 1.0], param_names=["a", "b"])
    text = v.describe()
    assert isinstance(text, str) and "identifiable=" in text and "confounded" in text


def test_diagnostics_package_is_llm_free() -> None:
    """IDENT-04: the diagnostic is provably LLM-free (grep gate). No google.genai,
    no ascension.llm, no model client anywhere in the package source."""
    import ascension.diagnostics as pkg

    pkg_dir = Path(pkg.__file__).parent
    # Check IMPORT statements, not prose: a docstring may legitimately *say*
    # "no google.genai"; what matters is that nothing is actually imported.
    forbidden = ("google", "genai", "ascension.llm", "llm_client", "generate_structured")
    for py in pkg_dir.glob("*.py"):
        for line in py.read_text().splitlines():
            stripped = line.strip().lower()
            if stripped.startswith(("import ", "from ")):
                for tok in forbidden:
                    assert (
                        tok not in stripped
                    ), f"{py.name} imports {tok!r} — diagnostic must be LLM-free (IDENT-04)"
