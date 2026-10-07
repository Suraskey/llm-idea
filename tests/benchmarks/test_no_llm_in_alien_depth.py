"""Phase 15.2-02 §22.6 grep gate — NO LLM in the alien depth scorer.

SCOPE §22.6 binding: the discovery-depth scorer is a DETERMINISTIC
sympy/scipy/structural classifier. No live-model call originates from
``score_depth`` or any code path it transitively invokes — the ONLY parser is
``_safe_parse_expr`` (sympy ``parse_expr``), never an LLM, never ``sympify``.

This test fails the build if the LLM client class name, the google generative
AI SDK module name, or an embed-call pattern appears anywhere in
``src/ascension/benchmarks/alien_depth.py`` (T-15.2-04 mitigation).

Mirrors ``tests/agora/test_no_llm_in_verifier.py`` verbatim, retargeted at the
depth scorer. Uses comment-stripped grep hygiene (drop ``#``-prefixed lines
before token counting) so the module's own header prose citing "no LLM" can't
self-invalidate the gate.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

ALIEN_DEPTH_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "ascension" / "benchmarks" / "alien_depth.py"
)


def _code_only(text: str) -> str:
    """Return only the EXECUTABLE-code tokens of the source (no comments/strings).

    The module docstring + the per-layer comments legitimately describe the
    §22.6 no-LLM guard and the safe-parse discipline, so they MUST be allowed to
    cite the forbidden tokens (LLMClient / genai / .embed( / sympify) in prose
    without self-invalidating the gate. We tokenize the Python source (the same
    ``tokenize``-based hygiene scoring.py uses) and drop COMMENT and STRING
    tokens, keeping only NAME / OP / NUMBER tokens — so the gate measures genuine
    CODE references, never documentation.
    """
    kept: list[str] = []
    try:
        toks = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in toks:
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if tok.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
                continue  # f-string literal text (3.12+) — documentation, not code
            kept.append(tok.string)
    except tokenize.TokenError:
        # Fall back to full text if tokenization fails — fail SAFE (stricter).
        return text
    return " ".join(kept)


def test_no_llm_client_reference_in_alien_depth() -> None:
    """``LLMClient`` must not appear in the code of alien_depth.py.

    The depth scorer is a pure deterministic gate (§22.6). Reintroducing the
    LLM client class would reintroduce the violation of "no LLM in the
    validation pipeline".
    """
    code = _code_only(ALIEN_DEPTH_PATH.read_text())
    assert not re.search(r"\bLLMClient\b", code), (
        "alien_depth.py contains LLMClient reference — §22.6 binding "
        "(no LLM in validation pipeline)."
    )


def test_no_genai_reference_in_alien_depth() -> None:
    """``genai`` (the google-genai SDK module name) must not appear in code."""
    code = _code_only(ALIEN_DEPTH_PATH.read_text())
    assert not re.search(
        r"\bgenai\b", code
    ), "alien_depth.py contains genai reference — §22.6 binding."


def test_no_embed_call_in_alien_depth() -> None:
    """No ``.embed(`` call pattern in alien_depth.py code.

    Embedding is an LLM operation (Gemini Embedding model). The depth scorer is
    a sympy + structured-comparison gate; it must not embed.
    """
    code = _code_only(ALIEN_DEPTH_PATH.read_text())
    assert not re.search(
        r"\.embed\s*\(", code
    ), "alien_depth.py contains .embed( call — §22.6 binding."


def test_no_sympify_in_alien_depth() -> None:
    """``sympify`` must not appear in code — the ONLY parser is _safe_parse_expr.

    ``sympify`` calls ``eval`` internally and does NOT block ``__import__`` even
    with a locals dict (the CR-01 finding, scoring.py:96-99). The depth scorer
    parses untrusted agent ``symbolic_form`` strings; it MUST route every parse
    through ``_safe_parse_expr`` (empty ``__builtins__``), never ``sympify``
    (T-15.2-06 mitigation).
    """
    code = _code_only(ALIEN_DEPTH_PATH.read_text())
    assert not re.search(r"\bsympify\b", code), (
        "alien_depth.py contains sympify — use _safe_parse_expr only "
        "(T-15.2-06: empty __builtins__ RCE guard)."
    )
