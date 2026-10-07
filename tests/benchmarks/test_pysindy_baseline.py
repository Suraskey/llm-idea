"""PySINDy baseline — regression guards on the feature_names / train_data pitfalls.

Covers:
  - _pysindy_strings_to_sympy: canonical LV parse (LHS-prefixed form AND the
    RHS-only form pysindy 2.1.0 actually emits via equations()); malformed
    input raises SympifyError; equation-count mismatch raises SympifyError.
  - run_pysindy_baseline(lv_bundle) recovers LV within score()'s rmse<0.1
    threshold (end-to-end regression guard for Pitfalls #5 + #6 combined).
  - monkeypatch-spy on ps.SINDy.fit asserts feature_names is passed explicitly
    (Pitfall #5 anti-drift).
  - monkeypatch-spy on ps.SINDy.fit asserts train_data is a list[ndarray], not
    a 3D ndarray (Pitfall #6 anti-drift).

NOTE on pysindy 2.1.0 API shift from the plan's <pysindy_reference>:
  The plan's reference code fragment passes ``feature_names=`` to
  ``ps.SINDy(...)`` and ``multiple_trajectories=True`` to ``model.fit(...)``.
  Both changed in pysindy 2.1.0 — ``feature_names`` moved to ``fit()``,
  ``multiple_trajectories`` was removed (list input is auto-detected). The
  implementation and the spy tests here target the real 2.1.0 API. See the
  Plan-03 SUMMARY deviation log for details.
"""

from __future__ import annotations

import numpy as np
import pysindy as ps
import pytest
import sympy
from sympy import symbols

from ascension.benchmarks import BenchmarkSpec, generate_trajectories
from ascension.benchmarks.pysindy_baseline import (
    _pysindy_strings_to_sympy,
    run_pysindy_baseline,
)

# ---------------------------------------------------------------------------
# _pysindy_strings_to_sympy — parser regression tests
# ---------------------------------------------------------------------------


def test_pysindy_strings_to_sympy_lv_canonical_lhs_prefixed() -> None:
    """Plan-documented LHS-prefixed form — pysindy's print() format."""
    raw = [
        "(x)' = 1.000 x + -10.000 x y",
        "(y)' = 10.000 x y + -2.000 y",
    ]
    exprs = _pysindy_strings_to_sympy(raw, "lotka_volterra")
    assert len(exprs) == 2
    # Free symbols should be exactly x and y (no stray x0/x1 etc.).
    syms_0 = sorted(str(s) for s in exprs[0].free_symbols)
    syms_1 = sorted(str(s) for s in exprs[1].free_symbols)
    assert syms_0 == ["x", "y"]
    assert syms_1 == ["x", "y"]
    # Numerical spot-check at (x=1, y=1): dx/dt = 1 - 10 = -9, dy/dt = 10 - 2 = 8.
    x_s, y_s = symbols("x y")
    assert abs(float(exprs[0].subs({x_s: 1.0, y_s: 1.0})) - (-9.0)) < 1e-6
    assert abs(float(exprs[1].subs({x_s: 1.0, y_s: 1.0})) - 8.0) < 1e-6


def test_pysindy_strings_to_sympy_rhs_only_format() -> None:
    """pysindy 2.1.0's equations() actually returns RHS-only strings (no LHS)."""
    raw = [
        " 1.000 x + -10.000 x y",
        " 10.000 x y + -2.000 y",
    ]
    exprs = _pysindy_strings_to_sympy(raw, "lotka_volterra")
    assert len(exprs) == 2
    x_s, y_s = symbols("x y")
    assert abs(float(exprs[0].subs({x_s: 1.0, y_s: 1.0})) - (-9.0)) < 1e-6


def test_pysindy_strings_to_sympy_handles_caret_exponent() -> None:
    """pysindy emits x^2 (caret); parser must map caret to ** before sympify."""
    raw = [
        " 1.000 x^2 + -5.000 x y^2",
        " 2.000 y",
    ]
    exprs = _pysindy_strings_to_sympy(raw, "lotka_volterra")
    x_s, y_s = symbols("x y")
    # At (x=2, y=3): 1*2^2 + -5*2*3^2 = 4 - 90 = -86
    assert abs(float(exprs[0].subs({x_s: 2.0, y_s: 3.0})) - (-86.0)) < 1e-6


def test_pysindy_strings_to_sympy_handles_lorenz_three_vars() -> None:
    """Lorenz has 3 state variables; parser must accept x, y, z all resolved."""
    raw = [
        " 10.000 y + -10.000 x",
        " 28.000 x + -1.000 x z + -1.000 y",
        " 1.000 x y + -2.666 z",
    ]
    exprs = _pysindy_strings_to_sympy(raw, "lorenz")
    assert len(exprs) == 3
    for e in exprs:
        assert all(str(s) in ("x", "y", "z") for s in e.free_symbols)


def test_pysindy_strings_to_sympy_raises_on_malformed() -> None:
    """Non-parseable equation surfaces as SympifyError (not a silent success)."""
    with pytest.raises(sympy.SympifyError):
        _pysindy_strings_to_sympy(["no lhs here +++"], "lotka_volterra")


def test_pysindy_strings_to_sympy_raises_on_wrong_equation_count() -> None:
    """LV expects 2 equations; passing 1 is a contract violation → SympifyError."""
    with pytest.raises(sympy.SympifyError):
        _pysindy_strings_to_sympy(["(x)' = 1.0"], "lotka_volterra")


def test_pysindy_strings_to_sympy_raises_on_empty_rhs() -> None:
    """Empty RHS after LHS stripping is a parse failure."""
    with pytest.raises(sympy.SympifyError):
        _pysindy_strings_to_sympy(["(x)' = ", "(y)' = 2 y"], "lotka_volterra")


# ---------------------------------------------------------------------------
# run_pysindy_baseline — end-to-end regression guard (Pitfalls #5 + #6)
# ---------------------------------------------------------------------------


def test_run_pysindy_baseline_lv_noise_free_recovers() -> None:
    """Noise-free LV: pysindy baseline score should satisfy rmse<0.1.

    Deviation from plan: n_samples bumped from 500 → 2000. With
    pysindy.utils.lotka at p=[1, 10] (Plan 01's canonical anchor) + degree=3
    STLSQ(threshold=0.1), the 500-sample fit produces coefficients so far off
    that integration rmse lands ~2.2 (pathological fit, well above the 0.1
    threshold). 2000 samples drops it to ~0.008 comfortably. See Plan-03
    SUMMARY for the full rationale.
    """
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=5,
        noise_sigma=0.0,
        n_samples=2000,
        seed=42,
    )
    bundle = generate_trajectories(spec)
    result = run_pysindy_baseline(bundle)
    assert result.reasons["numerical"] == "ok", f"numerical: {result.reasons!r}"
    assert (
        result.rmse is not None and result.rmse < 0.1
    ), f"rmse={result.rmse}; reasons={result.reasons}"


def test_run_pysindy_baseline_returns_benchmark_score_with_three_tier_keys() -> None:
    """Return shape must expose all three-tier keys in reasons + elapsed_ms."""
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=5,
        noise_sigma=0.0,
        n_samples=2000,
        seed=42,
    )
    bundle = generate_trajectories(spec)
    result = run_pysindy_baseline(bundle)
    assert set(result.reasons) == {"symbolic", "numerical", "qualitative"}
    assert set(result.elapsed_ms) == {"symbolic", "numerical", "qualitative"}
    assert isinstance(result.exact, bool)


def test_run_pysindy_baseline_uses_explicit_feature_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pitfall #5 anti-drift: feature_names must be passed explicitly to .fit().

    pysindy 2.1.0 moved ``feature_names`` from SINDy.__init__ to SINDy.fit, so
    the spy targets .fit's kwargs (not __init__'s). The defaults-to-x0/x1 bug
    this prevents is unchanged — only the API surface moved.
    """
    captured: dict[str, object] = {}
    real_fit = ps.SINDy.fit

    def spy_fit(self, x, t=None, **kwargs):
        captured.update(kwargs)
        captured["x"] = x
        captured["t"] = t
        return real_fit(self, x, t=t, **kwargs)

    monkeypatch.setattr(ps.SINDy, "fit", spy_fit)

    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=4,
        noise_sigma=0.0,
        n_samples=200,
        seed=1,
    )
    run_pysindy_baseline(generate_trajectories(spec))

    # feature_names must be [x, y] for LV — never default x0/x1.
    assert captured.get("feature_names") == [
        "x",
        "y",
    ], f"feature_names not explicit: {captured.get('feature_names')!r}"


def test_run_pysindy_baseline_uses_list_comprehension_for_train_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pitfall #6 anti-drift: train_data must be list[ndarray], not a 3D slice.

    A 3D ndarray silently misinterprets the time axis in pysindy 2.1.0 (the
    `x: list ⇒ multi-trajectory` auto-detection only fires when x is a Python
    list). This test ensures we don't regress to ``bundle.noisy[bundle.train_mask]``.
    """
    capture: dict[str, object] = {}
    real_fit = ps.SINDy.fit

    def spy_fit(self, x, t=None, **kwargs):
        capture["train_data"] = x
        return real_fit(self, x, t=t, **kwargs)

    monkeypatch.setattr(ps.SINDy, "fit", spy_fit)

    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=5,
        noise_sigma=0.0,
        n_samples=300,
        seed=2,
    )
    run_pysindy_baseline(generate_trajectories(spec))

    train_data = capture["train_data"]
    assert isinstance(train_data, list), f"train_data must be list, got {type(train_data).__name__}"
    assert all(
        isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in train_data
    ), "each element must be a 2D ndarray of shape (T, D)"


def test_run_pysindy_baseline_uses_train_mask_subset_not_full_noisy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_data length must equal train_mask.sum(), not n_trajectories.

    Redundant with the list-comprehension grep but valuable as a behavior
    test — catches the subtle bug where someone replaces the comprehension
    with ``list(bundle.noisy)`` (which would also pass the list-type check).
    """
    capture: dict[str, object] = {}
    real_fit = ps.SINDy.fit

    def spy_fit(self, x, t=None, **kwargs):
        capture["train_data"] = x
        return real_fit(self, x, t=t, **kwargs)

    monkeypatch.setattr(ps.SINDy, "fit", spy_fit)

    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=5,
        noise_sigma=0.0,
        n_samples=300,
        seed=2,
    )
    bundle = generate_trajectories(spec)
    run_pysindy_baseline(bundle)

    n_train = int(bundle.train_mask.sum())
    assert len(capture["train_data"]) == n_train, (
        f"train_data length {len(capture['train_data'])} != " f"train_mask.sum() {n_train}"
    )
