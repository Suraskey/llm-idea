"""CR-01 regression — agent-controlled strings cannot execute host commands.

The original `score()` path used `sympy.sympify(s, locals=...)` which
internally calls Python `eval()`; the `locals=` kwarg restricts name
resolution but does NOT strip Python builtins. An agent-provided string like
``'__import__("os").system("...")'`` would execute on the host process
during parsing — before any exception was caught and mapped to a reason
code.

The fix routes every agent/third-party string through
`sympy.parsing.sympy_parser.parse_expr` with a locked `global_dict` that
contains an empty `__builtins__` mapping. `parse_expr` uses AST
transformations rather than `eval`, and the empty builtins block
`__import__` / `eval` / `exec` / `open` from resolving during parse. The
tests below confirm that:

  1. The import of `__import__` fails cleanly (no side effect on disk).
  2. The import of `open` fails cleanly (no side effect on disk).
  3. The `score()` entry point returns `symbolic_parse_fail` rather than
     executing the payload, and downstream invariants hold.

If this guard regresses, a Phase 7.0+ agent could escape scoring and run
arbitrary host commands as the benchmark user.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import sympy
from sympy import symbols

from ascension.benchmarks import BenchmarkSpec, generate_trajectories
from ascension.benchmarks.scoring import (
    _SAFE_SYMPY_GLOBALS,
    _safe_parse_expr,
    score,
)


def test_safe_parse_expr_blocks_import_based_rce(tmp_path: Path) -> None:
    """parse_expr(_SAFE_SYMPY_GLOBALS) refuses to resolve __import__.

    If this fails, an agent string like
    '__import__("os").system("touch /tmp/pwned")' would execute on the host.
    """
    canary = tmp_path / "pwned.txt"
    assert not canary.exists()
    x, y = symbols("x y")
    payload = f'__import__("os").system("touch {canary}")'
    # The empty __builtins__ in _SAFE_SYMPY_GLOBALS means __import__ resolves
    # to nothing → NameError (which parse_expr wraps in SympifyError/ValueError).
    with pytest.raises(
        (
            sympy.SympifyError,
            NameError,
            ValueError,
            SyntaxError,
            TypeError,
            AttributeError,
        ),
    ):
        _safe_parse_expr(payload, local_dict={"x": x, "y": y})
    # The crucial assertion: the command did NOT run. Guarded even if
    # parse_expr raises LATE in the call — the command must never execute.
    assert not canary.exists(), f"CR-01 regression: {canary} was created — RCE guard broken"


def test_safe_parse_expr_blocks_open_based_rce(tmp_path: Path) -> None:
    """parse_expr refuses to resolve open() — builtins stripped."""
    canary = tmp_path / "opened.txt"
    assert not canary.exists()
    x, y = symbols("x y")
    payload = f'open({str(canary)!r}, "w").write("pwned")'
    with pytest.raises(
        (
            sympy.SympifyError,
            NameError,
            ValueError,
            SyntaxError,
            TypeError,
            AttributeError,
        ),
    ):
        _safe_parse_expr(payload, local_dict={"x": x, "y": y})
    assert not canary.exists(), f"CR-01 regression: open() resolved and wrote to {canary}"


def test_safe_parse_expr_allows_legitimate_polynomial() -> None:
    """The happy path still works — legitimate sympy expressions parse."""
    x, y = symbols("x y")
    expr = _safe_parse_expr("x - 10*x*y", local_dict={"x": x, "y": y})
    assert expr == x - 10 * x * y


def test_safe_parse_expr_handles_implicit_multiplication() -> None:
    """parse_expr transformations apply: '10 x y' → '10*x*y'."""
    x, y = symbols("x y")
    expr = _safe_parse_expr("10 x y", local_dict={"x": x, "y": y})
    assert expr == 10 * x * y


def test_safe_parse_expr_handles_caret_exponent() -> None:
    """convert_xor transformation: 'x^2' → 'x**2'."""
    x, y = symbols("x y")
    expr = _safe_parse_expr("x^2 + y", local_dict={"x": x, "y": y})
    assert expr == x**2 + y


def test_score_rejects_rce_string_returns_parse_fail_reason(tmp_path: Path) -> None:
    """End-to-end: score() on an RCE payload returns symbolic_parse_fail, no side effects.

    This is the full defense-in-depth check — even if _safe_parse_expr had a
    gap, the surrounding try/except in score() would map the exception to
    'symbolic_parse_fail' rather than crashing with a traceback. But the
    command must STILL not run, which is the invariant this test enforces.
    """
    canary = tmp_path / "score_pwned.txt"
    assert not canary.exists()

    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=100,
        seed=0,
    )
    bundle = generate_trajectories(spec)

    rce_payload = f'__import__("os").system("touch {canary}")'
    # Pass TWO strings — one RCE payload, one legitimate — because
    # score() routes list[str] through _safe_parse_expr for every element.
    result = score([rce_payload, "x - y"], spec, bundle)

    # The canary must NOT exist — the shell command never ran.
    assert not canary.exists(), f"CR-01 regression: score() executed {rce_payload!r}"
    # score() mapped the failure to symbolic_parse_fail.
    assert result.reasons["symbolic"] == "symbolic_parse_fail"
    assert result.exact is False
    # Numerical tier surfaces as diverged since proposed_expr is empty.
    assert result.rmse is None
    assert result.reasons["numerical"] == "numerical_diverged"


def test_safe_sympy_globals_has_empty_builtins() -> None:
    """Structural invariant: __builtins__ is the empty dict.

    If a future refactor accidentally inherits Python's real builtins
    (e.g. by omitting the explicit assignment), this test fires first.
    """
    assert "__builtins__" in _SAFE_SYMPY_GLOBALS
    assert _SAFE_SYMPY_GLOBALS["__builtins__"] == {}
    # Sanity: sympy symbols/functions are still there (sin, cos, etc).
    assert "sin" in _SAFE_SYMPY_GLOBALS
    assert "cos" in _SAFE_SYMPY_GLOBALS
    # And the namespace does NOT carry __import__ or eval.
    assert "__import__" not in _SAFE_SYMPY_GLOBALS
    assert "eval" not in _SAFE_SYMPY_GLOBALS


def test_safe_parse_expr_direct_import_attempt_blocked() -> None:
    """Direct __import__ resolution fails with a disarming exception class.

    No file touched; no subprocess spawned. The exception is thrown, the
    caller handles it, life goes on.
    """
    x, y = symbols("x y")
    # Wrap with a harmless scan counter so we're sure we'd observe a side-effect.
    before = len(os.listdir(tempfile.gettempdir()))
    with pytest.raises(
        (
            sympy.SympifyError,
            NameError,
            ValueError,
            SyntaxError,
            TypeError,
            AttributeError,
        ),
    ):
        _safe_parse_expr(
            '__import__("sys").exit(0)',
            local_dict={"x": x, "y": y},
        )
    after = len(os.listdir(tempfile.gettempdir()))
    # Indirect check — sys.exit(0) would kill the pytest worker; we'd never
    # reach here. The `raises` context proves control returned.
    assert after == before or after == before + 1  # pytest may write its own files
