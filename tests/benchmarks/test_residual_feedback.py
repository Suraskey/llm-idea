"""Phase B — the residual-feedback tool suite ($0 deterministic).

Specifies the ``compute_residual_feedback`` contract: a LEAK-SAFE, ground-truth-
anchored "how wrong is this law" signal the Council reads to refine (SEED-017,
the EXP-072 depth-1 plateau fix). Three arms:

  CORRECTNESS — bare Newton leaves a MATERIAL, real leftover (a steeper-power
    band AND an oscillatory component); the full truth law drives that leftover
    below the materiality threshold (the discriminating property). The exact
    numbers are pinned as a REGRESSION baseline against the finite-difference-
    recovered observed acceleration — they are real, not vacuous.
  INTEGRITY — anti-leak (the rendered block reveals only the leftover SHAPE,
    never the truth couplings / exact exponent / wavelength / a domain word, and
    is brace-free per Pitfall 4); L-016 (the signal is confidence-blind — the
    function has no confidence channel); §22.6 (no LLM in the module).
  ROBUSTNESS — garbage → None + WARN (skip-not-crash); determinism (same input
    → byte-identical block).

Key calibrated finding (documented, not a bug): the FD-recovered radial
acceleration has a ~5.5% noise / non-radial floor, so EVEN THE TRUTH LAW caps at
fit_fraction ≈ 0.945. fit_fraction therefore barely moves between bare Newton
(~0.935) and truth (~0.945); the DISCRIMINATING signal is the
steeper_power_present / oscillation_present FLAGS, which correctly flip
True → False as the law approaches truth. The leftover-after-Newton is dominated
by the hidden OSCILLATORY term, so the power and oscillation detectors are not
orthogonal (a hard superposition, SEED-017) — the block honestly reports BOTH.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

from ascension.benchmarks.residual_feedback import (
    ResidualFeedback,
    compute_residual_feedback,
)
from ascension.simulator.config import TIER2_HELD_OUT_ICS

# The §10 truth-law signatures as parseable symbolic_form strings (the SAME
# pinned constants tests/benchmarks/test_alien_depth.py uses).
_NEWTON_ONLY = "G*m1*m2/r**2"
_NEWTON_ALPHA = "G*m1*m2*(1/r**2 + alpha/r**3.5)"
_FULL_TRUTH = "G*m1*m2*(1/r**2 + alpha/r**3.5) + beta*s1*s2*(1 - cos(gamma*r))/r**2"

# truth_cfg is unused by compute_residual_feedback (the fit anchors to the
# OBSERVED acceleration, L-016) — kept for signature symmetry. The first
# held-out IC stands in (mirrors test_alien_depth._truth_cfg()).
_TRUTH_CFG = TIER2_HELD_OUT_ICS[0]

RESIDUAL_FEEDBACK_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "ascension"
    / "benchmarks"
    / "residual_feedback.py"
)


def _fb(form: str) -> ResidualFeedback | None:
    return compute_residual_feedback(form, _TRUTH_CFG, TIER2_HELD_OUT_ICS)


# -----------------------------------------------------------------------------
# CORRECTNESS — the leftover is real, and the truth drives it away
# -----------------------------------------------------------------------------


def test_bare_newton_leaves_a_material_real_leftover() -> None:
    """Bare 1/r**2 explains ~94% but leaves a MATERIAL steeper-power + oscillation.

    The regression that the characterization is REAL (not vacuous): both the
    steeper-power column and the oscillation column buy a material RSS reduction
    against the FD-recovered observed acceleration. These numbers are the pinned
    baseline (tolerant to small platform FD jitter).
    """
    fb = _fb(_NEWTON_ONLY)
    assert fb is not None
    assert fb.n_radial_terms_fit == 1
    assert 0.90 <= fb.fit_fraction <= 0.96
    assert fb.steeper_power_present is True
    assert fb.oscillation_present is True
    # The reductions are SUBSTANTIAL — the leftover is structured, not noise.
    assert fb.steeper_power_rss_reduction >= 0.10
    assert fb.oscillation_rss_reduction >= 0.10


def test_full_truth_law_drives_the_leftover_below_materiality() -> None:
    """The full §10 truth law leaves NO material standard-correction structure.

    The discriminating property: the truth's leftover is only the FD noise floor,
    so both presence flags go FALSE. fit_fraction caps at ~0.945 (the FD floor),
    NOT 1.0 — documented above, asserted here so a future change to the recovery
    path that breaks the ceiling is caught.
    """
    fb = _fb(_FULL_TRUTH)
    assert fb is not None
    assert 0.92 <= fb.fit_fraction <= 0.97
    assert fb.steeper_power_present is False
    assert fb.oscillation_present is False
    assert fb.steeper_power_rss_reduction < 0.05
    assert fb.oscillation_rss_reduction < 0.05


def test_truth_explains_more_than_bare_newton() -> None:
    """The truth law fits at least as well as bare Newton (monotone sanity)."""
    assert _fb(_FULL_TRUTH).fit_fraction >= _fb(_NEWTON_ONLY).fit_fraction


def test_newton_alpha_still_flags_remaining_structure() -> None:
    """Newton+alpha (genuine L2) still leaves the hidden OSCILLATORY structure.

    SEED-017 hard-superposition: after the power correction, the dominant
    leftover is the hidden periodic term, so at least one flag remains True —
    the feedback correctly keeps pushing toward the next (L3/L4) layer rather
    than declaring victory at L2.
    """
    fb = _fb(_NEWTON_ALPHA)
    assert fb is not None
    assert fb.n_radial_terms_fit == 2
    assert fb.steeper_power_present or fb.oscillation_present


# -----------------------------------------------------------------------------
# INTEGRITY — anti-leak, confidence-blind, no-LLM
# -----------------------------------------------------------------------------

_FORBIDDEN_LEAK_TOKENS = (
    "3.5",
    "0.7",
    "0.05",
    "0.02",
    "0.3",  # truth exponents / couplings
    "alpha",
    "beta",
    "gamma",
    "kappa",  # truth coupling symbol names
    "hidden",
    "charge",
    "spin",
    "gravity",
    "newton",  # domain words
)


def test_rendered_block_leaks_no_truth_and_is_brace_free() -> None:
    """The block reveals only leftover SHAPE — never the answer (SCOPE §10).

    No truth coupling / exact exponent / wavelength / domain word, and no
    '{'/'}' (else str.format on the live prompt raises KeyError — Pitfall 4).
    Checked across the structurally-distinct rendered cases.
    """
    for form in (_NEWTON_ONLY, _NEWTON_ALPHA, _FULL_TRUTH):
        block = _fb(form).rendered_block
        lowered = block.lower()
        for tok in _FORBIDDEN_LEAK_TOKENS:
            assert tok not in lowered, f"rendered_block for {form!r} leaks {tok!r}: {block!r}"
        assert (
            "{" not in block and "}" not in block
        ), f"rendered_block for {form!r} contains a brace (Pitfall 4): {block!r}"


def test_signal_is_confidence_blind_by_construction() -> None:
    """L-016: the signal has NO confidence channel — it cannot read self-report.

    ``compute_residual_feedback`` takes only (symbolic_form, truth_cfg,
    held_out_ics); there is structurally no way to pass a confidence. Two calls
    for the same law are byte-identical regardless of any caller belief.
    """
    import inspect

    params = set(inspect.signature(compute_residual_feedback).parameters)
    assert "confidence" not in params
    assert params == {"symbolic_form", "truth_cfg", "held_out_ics"}


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


def test_no_llm_in_residual_feedback_module() -> None:
    """§22.6 grep gate: no LLM client / genai SDK / embed call in the module code.

    The residual-feedback tool is a DETERMINISTIC numpy/scipy/sympy signal the
    agents reason over — it must never originate a model call (that would put an
    LLM inside the feedback loop's measurement). Mirrors
    test_no_llm_in_alien_depth.py, comment/string-stripped so the docstring may
    cite the forbidden tokens in prose.
    """
    code = _code_only(RESIDUAL_FEEDBACK_PATH.read_text())
    assert not re.search(r"\bLLMClient\b", code)
    assert not re.search(r"\bgenai\b", code)
    assert not re.search(r"\.embed\s*\(", code)


# -----------------------------------------------------------------------------
# ROBUSTNESS — skip-not-crash, determinism
# -----------------------------------------------------------------------------


def test_garbage_form_returns_none_not_crash() -> None:
    """An unparseable law → None (loud-logged, skip) — never a raised exception."""
    assert _fb("this is not @ valid law $$") is None


def test_non_radial_only_form_returns_none() -> None:
    """A law with only per-body kinematic components (no radial term) → None.

    ``(x2 - x1)`` carries Cartesian components and cannot be reduced to f(r);
    with no usable radial term the round simply carries no feedback that cycle.
    """
    assert _fb("G*m1*m2*(x2 - x1)") is None


def test_determinism_identical_block_across_two_calls() -> None:
    """Same input → byte-identical output (no hidden RNG / ordering nondeterminism)."""
    a = _fb(_NEWTON_ONLY)
    b = _fb(_NEWTON_ONLY)
    assert a == b
    assert a.rendered_block == b.rendered_block
