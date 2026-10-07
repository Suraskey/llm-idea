"""Benchmarks module constants — non-tunable artifact-directory layout.

Mirrors the 03.0 `src/ascension/sandbox/config.py` design analog: values
that are NOT user-tunable live here (mount paths, file naming conventions,
locked vocabulary); user-tunable knobs live on the `Settings` class in
`src/ascension/common/config.py`. Phase 6.0 Plan 01 adds NO tunable
BENCHMARK_* Settings knobs (per RESEARCH §Runtime State Inventory) — this
file holds only the artifact-directory layout consumed by Plan 03's
`io.py`.

Plain English: this file is the "name tags" for the per-run artifact
folder. It does NOT hold tolerance knobs or retry counts — those are
either on `BenchmarkSpec` (per-call) or would go on `Settings`
(sprint-wide) in a future phase.

NOTE: The `ScoreReasonCode` StrEnum for per-tier scoring failure codes
belongs to Plan 02 (Scoring), NOT this plan. Do not add it here.

Binding decisions:
  - 06.0 PATTERNS §`src/ascension/benchmarks/config.py`: artifact-layout
    constants consumed by Plan 03 `io.py` live here.
  - 03.0 `src/ascension/sandbox/config.py`: design analog — non-tunable
    constants + (locked-vocabulary StrEnum, added by Plan 02 here).

Pitfalls honored: N/A — this file is structural only; pitfall handling
is in `ode.py` and `systems.py`.

Threat-model mitigations: N/A — pure-compute phase, no network/auth/secret
surface per RESEARCH §Security Domain lines 690–702.
"""

from __future__ import annotations

# Per-run artifact layout under settings.RUN_ARTIFACTS_DIR.
# Consumed by Plan 03 `io.py` when it writes spec.json / trajectories.npz /
# *_result.json under $RUN_ARTIFACTS_DIR/<ARTIFACT_SUBDIR>/<run_id>/.

ARTIFACT_SUBDIR = "benchmarks"
SPEC_FILENAME = "spec.json"
TRAJECTORIES_FILENAME = "trajectories.npz"
PYSINDY_RESULT_FILENAME = "pysindy_result.json"
AGENT_RESULT_FILENAME = "agent_result.json"
