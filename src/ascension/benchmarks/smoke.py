"""Gated end-to-end smoke for the benchmarks module.

Runs the full happy path: generate_trajectories → run_pysindy_baseline →
save_run_artifacts on a real pysindy install. NOT a default CI test
(marker @pytest.mark.smoke excludes it from ``pytest`` default).

Invoke directly:
    poetry run python -m ascension.benchmarks.smoke

Exits 0 and prints one stdout line starting ``smoke.benchmarks OK:`` on
success; exits 1 and prints ``smoke.benchmarks FAIL:`` to stderr on any
exception (CLAUDE.md no-cover-ups — broad except by design).

Success criterion: PySINDy's output scored against our score() lands at
``reasons['numerical'] == 'ok'`` AND (``exact=True`` OR ``rmse < 0.1``).
Uses n_samples=2000 so the degree=3 STLSQ fit recovers cleanly on Plan 01's
canonical LV params p=[1, 10] (500 samples produces a pathological overfit;
see Plan-03 SUMMARY).

Binding decisions:
  - 06.0 PATTERNS §smoke.py: sync entrypoint mirrors common/smoke.py shape
    (no asyncio — benchmarks is pure compute; no Docker preflight).
  - 06.0 RESEARCH §Pattern 5 + TIER1-03: the happy path being exercised IS
    the Tier 1 baseline flow.

Threat-model mitigations:
  - run_id uses uuid.uuid4() — no user-supplied path traversal (T-06.0-11
    Phase 6.0 disposition: accept; Phase 7.0 CONTEXT adds run_id
    sanitization for external callers).
"""

from __future__ import annotations

import sys
import uuid

from ascension.benchmarks.io import benchmark_run_dir, save_run_artifacts
from ascension.benchmarks.ode import generate_trajectories
from ascension.benchmarks.pysindy_baseline import run_pysindy_baseline
from ascension.benchmarks.types import BenchmarkSpec


def run() -> int:
    """Execute the smoke happy path; return 0 on success, 1 on any failure.

    Plain English: generate LV trajectories, run PySINDy on them, score,
    write the artifacts, verify the score crossed the acceptance threshold,
    print one line of result. Returns 0 if everything passed, 1 otherwise.
    """
    run_id = f"smoke-benchmarks-{uuid.uuid4()}"
    # n_samples=2000 is deliberate — 500 samples on p=[1,10] LV produces a
    # pathological overfit (rmse~2.2); 2000 samples lands rmse~0.008 for a
    # reliable smoke. See Plan-03 SUMMARY for the full calibration story.
    spec = BenchmarkSpec(
        system="lotka_volterra",
        n_trajectories=5,
        noise_sigma=0.0,
        n_samples=2000,
        seed=42,
    )
    try:
        bundle = generate_trajectories(spec)
        result = run_pysindy_baseline(bundle)
        run_dir = benchmark_run_dir(run_id)
        save_run_artifacts(run_dir, spec, bundle, result)

        # Acceptance: numerical tier clean + (exact OR rmse<0.1).
        ok = (
            result.reasons["numerical"] == "ok"
            and result.rmse is not None
            and (result.exact or result.rmse < 0.1)
        )
        if not ok:
            print(
                f"smoke.benchmarks FAIL: system=lotka_volterra "
                f"exact={result.exact} rmse={result.rmse} "
                f"reasons={result.reasons}",
                file=sys.stderr,
            )
            return 1

        print(
            f"smoke.benchmarks OK: system=lotka_volterra "
            f"exact={result.exact} rmse={result.rmse:.6f} "
            f"reasons={result.reasons} artifacts={run_dir}",
        )
        return 0
    except Exception as exc:  # broad by design — smoke must surface ALL failures
        print(f"smoke.benchmarks FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    """Entry point for ``python -m ascension.benchmarks.smoke``."""
    return run()


if __name__ == "__main__":
    sys.exit(main())
