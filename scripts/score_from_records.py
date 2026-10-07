"""Re-score an Alien Universe run from its released record, with no database.

This is scripts/score_alien_run.py with the Postgres reads replaced by the files
in records/<EXP>/<batch>/runs/<run_id>/ (run.json, nodes.jsonl, llm_calls.jsonl.gz).
The scorer itself (ascension.benchmarks.alien_depth.score_depth) is the same
deterministic sympy/scipy code; no language model is involved.

Usage:
    python scripts/score_from_records.py records/EXP-076/exp076_20260530_155507/runs/<run_id> [--family | --family-v2]

  (no flag)    single-configuration run: EXP-072, EXP-074, EXP-075, EXP-077, the EXP-076
               control arm, and the EXP-089 instrument_only arm
  --family     multi-charge family run, first regime (product coupling, r^-3.5):
               the EXP-076 treatment arm and the EXP-089 family_only arm
  --family-v2  second regime (sum coupling, r^-2.5): EXP-080, EXP-081, EXP-083, EXP-083b

Which flag a batch used is recorded in its config.txt (score_flag=...). The
consecutive depth printed at the end is the number the paper reports.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from ascension.benchmarks.alien_depth import RegimeSpec, score_depth
from ascension.benchmarks.alien_fixtures import alien_benchmark


def main(run_dir: Path, family: bool, family_v2: bool) -> None:
    _bundle, truth_cfg, held_out_ics = alien_benchmark(seed=0)
    family_scoring_cfgs = None
    regime = None
    if family_v2:
        from ascension.benchmarks.alien_fixtures import (
            alien_family_v2_benchmark,
            build_tier2_held_out_ics_v2,
        )

        _family, held_out_cfgs = alien_family_v2_benchmark(seed=0)
        family_scoring_cfgs = held_out_cfgs
        held_out_ics = build_tier2_held_out_ics_v2(seed=0)
        truth_cfg = _family[0][1]
        regime = RegimeSpec(correction_exponent=2.5, charge_coupling="sum")
    elif family:
        from ascension.benchmarks.alien_fixtures import alien_family_benchmark

        _family, held_out_cfgs = alien_family_benchmark(seed=0)
        family_scoring_cfgs = held_out_cfgs

    run = json.loads((run_dir / "run.json").read_text())
    nodes = [json.loads(line) for line in (run_dir / "nodes.jsonl").read_text().splitlines() if line.strip()]
    hyp_nodes = [n for n in nodes if n.get("type") == "hypothesis"]
    hyps = [n.get("content") for n in hyp_nodes]
    status_breakdown: dict = {}
    for n in hyp_nodes:
        status_breakdown[n.get("status")] = status_breakdown.get(n.get("status"), 0) + 1
    node_types: dict = {}
    for n in nodes:
        node_types[n.get("type")] = node_types.get(n.get("type"), 0) + 1

    n_calls, total_cost, by_provider = 0, 0.0, {}
    calls_path = run_dir / "llm_calls.jsonl.gz"
    if calls_path.exists():
        with gzip.open(calls_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                c = json.loads(line)
                n_calls += 1
                cost = float(c.get("cost_usd") or 0.0)
                total_cost += cost
                p = by_provider.setdefault(c.get("provider"), [0, 0.0])
                p[0] += 1
                p[1] += cost

    depth = score_depth(hyps, truth_cfg, held_out_ics, family_scoring_cfgs=family_scoring_cfgs, regime=regime)

    config = run.get("config") or {}
    print("=" * 72)
    print(f"ALIEN RUN {run.get('id')}")
    print(f"  run.status={run.get('status')}  halt_reason={run.get('halt_reason')}")
    print(f"  PROVENANCE: benchmark_system={config.get('benchmark_system')!r}  "
          f"included_in_distillation_v1={config.get('included_in_distillation_v1')!r}")
    mode = "FAMILY-V2 (sum coupling + r^-2.5)" if family_v2 else "FAMILY (product coupling)" if family else "single-config"
    print(f"  SCORING MODE: {mode}")
    print(f"  node types: {node_types}")
    print(f"  hypothesis nodes={len(hyp_nodes)}  status_breakdown={status_breakdown}")
    print(f"  SPEND: {n_calls} llm_calls, ${total_cost:.4f} total")
    for prov, (c, cost) in sorted(by_provider.items(), key=lambda kv: str(kv[0])):
        print(f"    provider={prov!r}: {c} calls, ${cost:.4f}")
    print("-" * 72)
    print(f"  DISCOVERY DEPTH: max_depth={depth.max_depth}")
    print(f"    per_layer L1..L6 = {depth.per_layer}")
    consec = 0
    for passed in depth.per_layer[:4]:
        if passed:
            consec += 1
        else:
            break
    print(f"    consecutive force-law depth (L1-L4) = {consec}   [L5={depth.per_layer[4]} L6={depth.per_layer[5]} reported separately]")
    print(f"    held_out_rss_reduction={depth.held_out_rmse}  q_drift={depth.q_drift}")
    print(f"  scorer notes ({len(depth.notes)}):")
    for note in depth.notes[:60]:
        print(f"    - {note}")
    print("=" * 72)


if __name__ == "__main__":
    args = sys.argv[1:]
    family_v2 = "--family-v2" in args
    family = "--family" in args
    args = [a for a in args if not a.startswith("--")]
    if len(args) != 1 or (family and family_v2):
        sys.exit(__doc__)
    main(Path(args[0]), family, family_v2)
