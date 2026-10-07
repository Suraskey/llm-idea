"""Benchmarks smoke test — gated via @pytest.mark.smoke.

pyproject.toml sets ``addopts = -m 'not smoke'`` so these tests are
excluded from the default pytest run. Invoke explicitly:

    poetry run pytest -m smoke tests/benchmarks/test_smoke.py -q

or via the module's CLI entrypoint:

    poetry run python -m ascension.benchmarks.smoke

Both tests exercise the full pipeline on a real pysindy install:
  1. ``test_smoke_entry_exits_zero`` — invokes smoke.main() end-to-end.
  2. ``test_smoke_pysindy_recovers_lotka_volterra_noiseless`` — the
     acceptance regression: reasons['numerical']=='ok' AND rmse<0.1 OR exact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ascension.benchmarks import (
    BenchmarkSpec,
    generate_trajectories,
    run_pysindy_baseline,
)
from ascension.common.config import settings


@pytest.mark.smoke
def test_smoke_entry_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run the smoke entry and assert exit code 0. Requires live pysindy install."""
    monkeypatch.setattr(settings, "RUN_ARTIFACTS_DIR", tmp_path)
    from ascension.benchmarks.smoke import main

    rc = main()
    assert rc == 0


@pytest.mark.smoke
def test_smoke_pysindy_recovers_lotka_volterra_noiseless() -> None:
    """End-to-end recovery: PySINDy on LV noise-free should return rmse<0.1 or exact.

    n_samples=2000 per the Plan-03 SUMMARY calibration (500 samples on
    p=[1,10] LV produces a pathological fit).
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
    assert result.reasons["numerical"] == "ok"
    assert result.rmse is not None
    assert result.exact or result.rmse < 0.1, (
        f"PySINDy baseline failed to recover LV: " f"exact={result.exact} rmse={result.rmse}"
    )
