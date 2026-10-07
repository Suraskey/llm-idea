"""Tests for the symbolic tier of score() — Poly + nsimplify + timeout.

Covers RESEARCH Pitfall #2 (sympy.simplify hang protection via
ThreadPoolExecutor timeout) and the Poly-canonicalization pattern with
floating-point coefficient tolerance (RESEARCH §Code Examples lines
552–584).
"""

from __future__ import annotations

import time

import sympy
from sympy import symbols

from ascension.benchmarks.scoring import _canonicalize_expr, _symbolic_match


def test_symbolic_match_identical_returns_ok_true() -> None:
    x, y = symbols("x y")
    truth = [x - 10 * x * y, 10 * x * y - 2 * y]
    matched, reason = _symbolic_match(truth, truth, [x, y], timeout_s=5.0, coeff_tol=1e-3)
    assert matched is True
    assert reason == "ok"


def test_symbolic_match_noisy_coefficients_snapped_to_rational() -> None:
    """nsimplify(tolerance=1e-3) rounds 0.9999 → 1, 10.0001 → 10."""
    x, y = symbols("x y")
    truth = [x - 10 * x * y, 10 * x * y - 2 * y]
    proposed = [
        sympy.sympify("0.9999*x - 10.0001*x*y"),
        sympy.sympify("10.0003*x*y - 2.0001*y"),
    ]
    matched, reason = _symbolic_match(
        proposed,
        truth,
        [x, y],
        timeout_s=5.0,
        coeff_tol=1e-3,
    )
    assert matched is True
    assert reason == "ok"


def test_symbolic_match_different_structure() -> None:
    """Different polynomial shape → clean no-match, not an error."""
    x, y = symbols("x y")
    truth = [x * y]
    proposed = [x * y + x]
    matched, reason = _symbolic_match(
        proposed,
        truth,
        [x, y],
        timeout_s=5.0,
        coeff_tol=1e-3,
    )
    assert matched is False
    assert reason == "ok"  # clean NO, not an error string


def test_symbolic_match_different_length_returns_clean_no_match() -> None:
    """Proposed has wrong number of equations → clean no-match."""
    x, y = symbols("x y")
    truth = [x, y]
    proposed = [x]  # only one equation
    matched, reason = _symbolic_match(
        proposed,
        truth,
        [x, y],
        timeout_s=5.0,
        coeff_tol=1e-3,
    )
    assert matched is False
    assert reason == "ok"


def test_symbolic_match_respects_coefficient_tolerance_floor() -> None:
    """Tight tolerance (1e-6) keeps 0.5*x vs 0.4*x distinct."""
    x, y = symbols("x y")
    matched, reason = _symbolic_match(
        [sympy.Float(0.5) * x],
        [sympy.Float(0.4) * x],
        [x, y],
        timeout_s=5.0,
        coeff_tol=1e-6,
    )
    assert matched is False
    assert reason == "ok"


def test_canonicalize_snaps_floats_to_rationals_within_tolerance() -> None:
    """_canonicalize_expr with coeff_tol=1e-3 maps 0.5000001*x → 0.5*x."""
    x, y = symbols("x y")
    a = _canonicalize_expr(sympy.sympify("0.5000001*x"), [x, y], 1e-3)
    b = _canonicalize_expr(sympy.sympify("0.5*x"), [x, y], 1e-3)
    assert a == b


def test_symbolic_match_timeout_triggers_reason() -> None:
    """Pathological non-polynomial expression that simplify chokes on.

    The invariant: never hang past ~2.5s on timeout_s=0.5. Either the
    timeout fires (preferred → SYMBOLIC_TIMEOUT) or simplify finishes
    quickly with a clean no-match. Both acceptable; the key assertion is
    wall-time bound.
    """
    x, y = symbols("x y")
    # Deeply nested trig/log expression chosen to stress simplify without
    # being easily reducible. Poly() will fail (non-polynomial); fallback
    # path runs sympy.simplify inside the ThreadPoolExecutor timeout.
    from sympy import cos, log, sin

    pathological = sin(cos(log(sin(cos(x + y**2)) + 2))) + x * y
    start = time.monotonic()
    matched, reason = _symbolic_match(
        [pathological],
        [x],
        [x, y],
        timeout_s=0.5,
        coeff_tol=1e-3,
    )
    elapsed = time.monotonic() - start

    assert matched is False
    # Either timed out OR ruled clean-no-match via Poly fallback finishing
    # quickly. Both are valid; the invariant is wall-time bound.
    assert reason in ("ok", "symbolic_timeout")
    assert elapsed < 2.5, f"symbolic match took {elapsed:.2f}s, timeout_s=0.5"


def test_symbolic_match_poly_failure_falls_back_to_simplify() -> None:
    """Non-polynomial input where simplify CAN resolve quickly."""
    x, y = symbols("x y")
    # Both expressions equivalent (sin(x)^2 + cos(x)^2 = 1). Poly fails;
    # simplify proves equality.
    a = [sympy.sin(x) ** 2 + sympy.cos(x) ** 2]
    b = [sympy.Integer(1)]
    matched, reason = _symbolic_match(a, b, [x, y], timeout_s=5.0, coeff_tol=1e-3)
    # Simplify fallback should prove equality.
    assert matched is True
    assert reason == "ok"
