"""Tests for the ODE system registry (06.0 RESEARCH §Pattern 1, systems.py).

Covers:
  - DEFAULT_PARAMS has exactly 3 systems at canonical keys.
  - Lorenz args are the canonical (sigma=10, beta=8/3, rho=28).
  - RHS callables come from pysindy.utils (SCOPE §9 anchor discipline).
  - INTEGRATOR_KEYWORDS are LSODA + rtol=atol=1e-12 (Pitfall #1).
  - params_for raises UnknownBenchmarkSystem with the expected message.
  - Every system has feature_names of the correct length (Pitfall #5).
"""

from __future__ import annotations

import pytest

from ascension.benchmarks.exceptions import UnknownBenchmarkSystem
from ascension.benchmarks.systems import (
    DEFAULT_PARAMS,
    INTEGRATOR_KEYWORDS,
    PYSINDY_VERSION_ANCHOR,
    params_for,
)


def test_default_params_has_exactly_three_systems() -> None:
    assert set(DEFAULT_PARAMS) == {"lotka_volterra", "van_der_pol", "lorenz"}


def test_default_params_lorenz_args_are_canonical() -> None:
    assert DEFAULT_PARAMS["lorenz"]["args"] == (10, 8 / 3, 28)


def test_default_params_lotka_volterra_args() -> None:
    assert DEFAULT_PARAMS["lotka_volterra"]["args"] == ([1, 10],)


def test_default_params_van_der_pol_args() -> None:
    assert DEFAULT_PARAMS["van_der_pol"]["args"] == ([0.5],)


def test_default_params_rhs_comes_from_pysindy() -> None:
    """SCOPE §9 anchor: RHS functions MUST come from pysindy, not hand-rolled."""
    for system_name, row in DEFAULT_PARAMS.items():
        rhs = row["rhs"]
        assert callable(rhs), f"{system_name} rhs must be callable"
        module = getattr(rhs, "__module__", "")
        assert module.startswith(
            "pysindy"
        ), f"{system_name} rhs must be from pysindy.* — got {module!r}"


def test_integrator_keywords_are_lsoda_1e_minus_12() -> None:
    """Pitfall #1: scipy defaults wreck Lorenz; we pin LSODA + rtol=atol=1e-12."""
    assert INTEGRATOR_KEYWORDS == {
        "method": "LSODA",
        "rtol": 1e-12,
        "atol": 1e-12,
    }


def test_params_for_returns_registry_row() -> None:
    row = params_for("lorenz")
    assert row["dim"] == 3
    assert row["feature_names"] == ["x", "y", "z"]


def test_params_for_raises_unknown_benchmark_system() -> None:
    with pytest.raises(UnknownBenchmarkSystem, match="Unknown benchmark system: banana"):
        params_for("banana")


def test_every_system_has_feature_names_of_correct_length() -> None:
    """Pitfall #5: PySINDy's default feature_names are x0/x1/x2; we pin x/y/z
    per-system so symbolic scoring in Plan 02 doesn't see symbol mismatch."""
    for system_name, row in DEFAULT_PARAMS.items():
        names = row["feature_names"]
        dim = row["dim"]
        assert isinstance(names, list), f"{system_name}: feature_names must be list"
        assert len(names) == dim, f"{system_name}: feature_names length {len(names)} != dim {dim}"
        for name in names:
            assert (
                isinstance(name, str) and name
            ), f"{system_name}: feature_names entries must be non-empty strings"


def test_every_system_has_default_ic_matching_dim() -> None:
    for system_name, row in DEFAULT_PARAMS.items():
        ic = row["default_ic"]
        dim = row["dim"]
        assert len(ic) == dim, f"{system_name}: default_ic length != dim"


def test_pysindy_version_anchor_string() -> None:
    assert PYSINDY_VERSION_ANCHOR == "2.1.0"
