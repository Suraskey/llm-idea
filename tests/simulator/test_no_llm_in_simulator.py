"""No-LLM grep gate over the simulator path (SCOPE §22.6 + §10 isolation).

The Alien Universe simulator is a deterministic numpy/scipy/sympy library. No
live-model call may originate from any code path in ``src/ascension/simulator/``.
This test FAILS THE BUILD if the LLM client class name, the google-genai SDK
module name, or an embed-call pattern appears in any simulator source file.

Mirrors ``tests/agora/test_no_llm_in_verifier.py:30-62`` (D-10 no-LLM gate). Uses
the L-018 ``if not path.exists(): continue`` pattern so coverage is established
before every simulator file lands (Wave 1 ships only a subset of the modules).

Threat-model mitigation: T-15.0-02 (LLM creeping into the deterministic
simulator path). $0 LLM — this gate is the §22.6 enforcement.
"""

from __future__ import annotations

import re
from pathlib import Path

_SIMULATOR_DIR = Path(__file__).resolve().parents[2] / "src" / "ascension" / "simulator"

# The full intended simulator module set (D-07). Files not yet shipped are
# skipped via `if not path.exists()` (L-018) so the gate has coverage from
# Wave 1 onward and automatically tightens as later waves add files.
SIMULATOR_FILES = (
    _SIMULATOR_DIR / "__init__.py",
    _SIMULATOR_DIR / "types.py",
    _SIMULATOR_DIR / "reason_codes.py",
    _SIMULATOR_DIR / "exceptions.py",
    _SIMULATOR_DIR / "config.py",
    _SIMULATOR_DIR / "physics.py",
    _SIMULATOR_DIR / "integrator.py",
    _SIMULATOR_DIR / "alien.py",
    _SIMULATOR_DIR / "io.py",
    _SIMULATOR_DIR / "fixtures.py",
    _SIMULATOR_DIR / "simulator_audit.py",
    _SIMULATOR_DIR / "tuning.py",
)


def test_no_llm_client_reference_in_simulator() -> None:
    """``LLMClient`` must not appear in any simulator source file."""
    for path in SIMULATOR_FILES:
        if not path.exists():
            continue
        text = path.read_text()
        assert not re.search(r"\bLLMClient\b", text), (
            f"{path.name} contains LLMClient reference — §22.6 binding "
            "(no LLM in the simulator path)."
        )


def test_no_genai_reference_in_simulator() -> None:
    """``genai`` (the google-genai SDK module name) must not appear."""
    for path in SIMULATOR_FILES:
        if not path.exists():
            continue
        text = path.read_text()
        assert not re.search(
            r"\bgenai\b", text
        ), f"{path.name} contains genai reference — §22.6 binding."


def test_no_embed_call_in_simulator() -> None:
    """No ``.embed(`` call pattern anywhere in the simulator path."""
    for path in SIMULATOR_FILES:
        if not path.exists():
            continue
        text = path.read_text()
        assert not re.search(
            r"\.embed\s*\(", text
        ), f"{path.name} contains .embed( call — §22.6 binding."


def test_no_llm_in_simulator_path() -> None:
    """Consolidated §22.6 no-LLM gate over the whole simulator path.

    This is the canonical node id the simulator audit gate asserts against by
    name (PROPERTY_TEST_MAP row 16). It re-runs all three grep checks
    (``LLMClient`` / ``genai`` / ``.embed(``) over every simulator source file in
    one place, so the audit's ``_check_property_has_test`` has a single, stably
    named test proving the no-LLM property (T-15.0-02). The three granular tests
    above remain for pinpoint diagnostics.
    """
    patterns = (
        (r"\bLLMClient\b", "LLMClient reference"),
        (r"\bgenai\b", "genai (google-genai SDK) reference"),
        (r"\.embed\s*\(", ".embed( call"),
    )
    for path in SIMULATOR_FILES:
        if not path.exists():
            continue
        text = path.read_text()
        for pattern, label in patterns:
            assert not re.search(pattern, text), (
                f"{path.name} contains {label} — §22.6 binding " "(no LLM in the simulator path)."
            )
