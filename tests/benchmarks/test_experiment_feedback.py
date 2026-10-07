"""Phase C — the active-experimentation instrument suite ($0 deterministic).

Specifies the ``experiment_feedback`` contract: the depth-2 lever (SEED-018; the
SCOPE §10 active-experimentation amendment, Surya signed off 2026-05-26). Three
arms:

  CORRECTNESS — release-from-rest radial-drop probes recover a clean a(r) from the
    OBSERVED trajectory; fitting the exponent of the Council's OWN proposed
    correction term to that clean data lands inside the L2 ``near_35`` band
    (truth 3.5; recovers ~3.4). THE HEADLINE: adopting the reported form lifts a
    depth-1 plateau to depth 2 on the SEPARATE post-hoc ``score_depth`` oracle,
    while the plateau forms stay at depth 1.
  INTEGRITY — anti-leak (the block names ONLY the agent's own fitted exponent +
    RSS reduction, never a truth coupling / oscillation frequency / domain word,
    and is brace-free); the block never reveals the disentangling METHOD (no
    radial-drop / release-from-rest / v_perp wording); data-only (a(r) comes from
    the trajectory, never from reading the force law); §22.6 (no LLM in the
    module).
  ROBUSTNESS — a bare Newton law (no correction of the agent's to fit) → None;
    garbage → None (skip-not-crash); determinism (same input → identical output).
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest

from ascension.benchmarks.alien_depth import score_depth
from ascension.benchmarks.experiment_feedback import (
    FittedForm,
    compute_experiment_feedback,
    fit_proposed_structure,
    run_clean_probes,
)
from ascension.simulator.config import TIER2_HELD_OUT_ICS
from ascension.simulator.tuning import tuned_config

# Plateau-state hypotheses (the EXP-072 / EXP-073 result): the Council proposed a
# steeper correction but guessed the exponent (integer r**-3 / r**-4, fractional
# undershoots, or a symbolic r**p). All have a correction term of the agent's to
# fit; none is the truth's ~-3.5.
_PLATEAU_FORMS = (
    "G*m1*m2/r**2 + C/r**3",
    "G*m1*m2/r**2 - C/r**4",
    "G*m1*m2/r**2 + C/r**p",
    "1.0/r**2 + 0.05/r**2.5",
)
_BARE_NEWTON = "G*m1*m2/r**2"

EXPERIMENT_FEEDBACK_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "ascension"
    / "benchmarks"
    / "experiment_feedback.py"
)


@pytest.fixture(scope="module")
def clean_probes() -> tuple[np.ndarray, np.ndarray]:
    """The clean radial-drop probe data, computed ONCE (6 short integrations)."""
    return run_clean_probes(seed=0)


def _hyp(sf: str) -> dict:
    """A score_depth-shaped hypothesis row (flat dict, _extract_symbolic_form)."""
    return {"symbolic_form": sf, "rationale": "r"}


# -----------------------------------------------------------------------------
# CORRECTNESS — probes are clean, the fit lands in the L2 band, depth lifts 1->2
# -----------------------------------------------------------------------------


def test_clean_probes_are_well_posed(clean_probes) -> None:
    """The release-from-rest battery returns a clean, finite mid-r a(r) sample."""
    r, a = clean_probes
    assert r.size > 100
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(a))
    assert r.min() >= 0.5 - 1e-9  # the probe r-floor (FD-clean window)
    assert a.max() < 50.0  # no slingshot / singularity garbage survived the cut


@pytest.mark.parametrize("form", _PLATEAU_FORMS)
def test_fit_recovers_correction_exponent_in_L2_band(form, clean_probes) -> None:
    """Fitting the agent's correction on clean data lands in the L2 near_35 band.

    The L2 structural gate accepts a second radial exponent within 0.3 of -3.5
    (i.e. -3.2 .. -3.8). The instrument recovers ~3.4 and removes nearly all of
    the Newton-only leftover on the short-range-dominated window.
    """
    r, a = clean_probes
    ff = fit_proposed_structure(form, r, a)
    assert ff is not None
    assert (
        3.2 < ff.fitted_exponent < 3.8
    ), f"fitted exponent {ff.fitted_exponent} outside the L2 near_35 band"
    assert ff.rss_reduction > 0.5  # the correction explains the leftover


def test_adopting_reported_form_lifts_depth_1_to_2() -> None:
    """THE HEADLINE: the plateau scores depth 1; adopting the fit scores depth 2.

    Reproduces the EXP-072 / EXP-073 negative (plateau forms = max_depth 1) and
    shows the instrument resolves it: a Council that commits to the reported
    inverse-power exponent clears L2 on the SEPARATE post-hoc score_depth oracle
    (scored on TIER2_HELD_OUT_ICS the Council did not choose — §22.6 / §10).
    """
    truth = tuned_config()
    ff = compute_experiment_feedback("G*m1*m2/r**2 + C/r**3", seed=0)
    assert ff is not None

    # Plateau: a wrong-exponent guess stays at depth 1 (L1 yes, L2 no).
    plateau = score_depth([_hyp("G*m1*m2/r**2 + C/r**3")], truth, TIER2_HELD_OUT_ICS)
    assert plateau.max_depth == 1, plateau.notes

    # Adopting the instrument's reported form clears L2 → depth 2.
    adopted_sf = f"G*m1*m2/r**2 + alpha/r**{ff.fitted_exponent:.1f}"
    adopted = score_depth([_hyp(adopted_sf)], truth, TIER2_HELD_OUT_ICS)
    assert adopted.max_depth >= 2, (adopted_sf, adopted.notes)
    assert adopted.per_layer[1] is True  # L2 (the correction term) credited


# -----------------------------------------------------------------------------
# INTEGRITY — anti-leak, method-blind, data-only, no-LLM
# -----------------------------------------------------------------------------

# The block MAY name the agent's OWN fitted exponent (the blessed §10 boundary).
# It must NOT name any OTHER truth coupling value, the oscillation frequency, a
# coupling symbol, or a domain word.
_FORBIDDEN_LEAK_TOKENS = (
    "0.05",
    "0.02",
    "0.7",
    "0.3",  # the OTHER truth couplings / freq
    "beta",
    "gamma",
    "kappa",  # truth coupling symbol names
    "cos",
    "oscillat",  # the hidden periodic term
    "hidden",
    "charge",
    "gravity",
    "newton",
    "kepler",
    "alien",  # domain words
)
# The disentangling METHOD must stay the Council's insight (esp. for Stage 2):
# the block never names the radial drop / release-from-rest / velocity trick.
_FORBIDDEN_METHOD_TOKENS = (
    "radial drop",
    "radial-drop",
    "release from rest",
    "release-from-rest",
    "tangential",
    "v_perp",
    "vperp",
    "v=0",
    "v = 0",
    "drop the",
    "drop a",
)


def test_rendered_block_leaks_no_other_coupling_and_is_brace_free() -> None:
    """The block names only the agent's own fitted exponent — never the answer."""
    ff = compute_experiment_feedback("G*m1*m2/r**2 + C/r**3", seed=0)
    assert ff is not None
    block = ff.rendered_block
    lowered = block.lower()
    for tok in _FORBIDDEN_LEAK_TOKENS:
        assert tok not in lowered, f"experiment block leaks {tok!r}: {block!r}"
    assert (
        "{" not in block and "}" not in block
    ), f"experiment block contains a brace (Pitfall 4): {block!r}"


def test_rendered_block_does_not_reveal_the_method() -> None:
    """The block describes 'cleaner/simpler motion', never the radial-drop trick.

    The disentangling experiment design is the Council's discovery (Stage 2); the
    Stage-1 system-triggered block must not hand it over.
    """
    ff = compute_experiment_feedback("G*m1*m2/r**2 + C/r**3", seed=0)
    assert ff is not None
    lowered = ff.rendered_block.lower()
    for tok in _FORBIDDEN_METHOD_TOKENS:
        assert tok not in lowered, f"experiment block reveals the method {tok!r}"


def _code_only(text: str) -> str:
    """EXECUTABLE tokens only (drop comments + strings) — mirrors the §22.6 gate."""
    kept: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if tok.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
                continue
            kept.append(tok.string)
    except tokenize.TokenError:
        return text
    return " ".join(kept)


def test_no_llm_in_experiment_feedback_module() -> None:
    """§22.6 grep gate: no LLM client / genai SDK / embed call in the module."""
    code = _code_only(EXPERIMENT_FEEDBACK_PATH.read_text())
    assert not re.search(r"\bLLMClient\b", code)
    assert not re.search(r"\bgenai\b", code)
    assert not re.search(r"\.embed\s*\(", code)


def test_data_only_recovers_from_trajectory_not_the_force_law() -> None:
    """§10 data-only: a(r) comes from the OBSERVED trajectory, never the force law.

    The instrument must not read ``physics.total_accel`` / the conserved Q / the
    hidden charges / the potential — that would be reading the answer, not running
    an experiment. It uses only AlienUniverse(cfg).run().positions (the observable).
    """
    code = _code_only(EXPERIMENT_FEEDBACK_PATH.read_text())
    for forbidden in ("total_accel", "conserved_Q", "potential_U", ".charges", ".Q"):
        assert (
            forbidden not in code
        ), f"experiment_feedback reads the force law via {forbidden!r} — not data-only"


# -----------------------------------------------------------------------------
# ROBUSTNESS — only fit the agent's OWN structure, skip-not-crash, determinism
# -----------------------------------------------------------------------------


def test_bare_newton_returns_none(clean_probes) -> None:
    """A bare inverse-square law has no correction of the agent's to fit → None.

    The blessed boundary fits the agent's OWN proposed structure; with nothing
    proposed beyond Newton the instrument stays silent (the residual loop's job is
    to get them to propose a correction first).
    """
    r, a = clean_probes
    assert fit_proposed_structure(_BARE_NEWTON, r, a) is None


def test_non_correction_form_returns_none(clean_probes) -> None:
    """A law whose only extra term is non-radial (no steeper r-power) → None."""
    r, a = clean_probes
    assert fit_proposed_structure("G*m1*m2/r**2 + C*(x2 - x1)", r, a) is None


def test_garbage_form_returns_none(clean_probes) -> None:
    """An unparseable law → None (loud-logged, skip) — never a raised exception."""
    r, a = clean_probes
    assert fit_proposed_structure("this is not @ valid law $$", r, a) is None


def test_compute_short_circuits_before_probes_on_bare_newton() -> None:
    """compute_experiment_feedback returns None for bare Newton (no probe run)."""
    assert compute_experiment_feedback(_BARE_NEWTON, seed=0) is None


def test_determinism_identical_fit_across_two_calls(clean_probes) -> None:
    """Same input → identical FittedForm (no hidden RNG / ordering nondeterminism)."""
    r, a = clean_probes
    a1 = fit_proposed_structure("G*m1*m2/r**2 + C/r**3", r, a)
    a2 = fit_proposed_structure("G*m1*m2/r**2 + C/r**3", r, a)
    assert a1 == a2
    assert isinstance(a1, FittedForm)
