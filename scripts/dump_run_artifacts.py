"""Dump every Postgres row from a single Council run into JSONL artifacts.

Usage:
    poetry run python scripts/dump_run_artifacts.py \\
        --run-id <uuid> --out-dir records/<EXP>/runs/<run-id>/

One SELECT per table; writes JSONL files alongside cost_timeline.csv +
manifest.json. No pandas, no analysis here — pure data extraction.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "hex"):  # UUID
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    return str(obj)


async def _dump_table(pool, sql, out_path: Path, params=()):
    """Run SQL, write one JSONL line per row."""
    n = 0
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            with out_path.open("w") as f:
                async for row in cur:
                    rec = dict(zip(cols, row))
                    f.write(json.dumps(rec, default=_json_default))
                    f.write("\n")
                    n += 1
    return n


async def main(run_id: str, out_dir: Path) -> int:
    from ascension.common.db import get_async_pool

    out_dir.mkdir(parents=True, exist_ok=True)
    pool = await get_async_pool()

    tables = {
        "llm_calls.jsonl": (
            "SELECT * FROM llm_calls WHERE run_id = %s ORDER BY created_at",
            (run_id,),
        ),
        "hypotheses.jsonl": (
            "SELECT n.*, a.name AS author_name FROM nodes n "
            "LEFT JOIN agents a ON a.id = n.author_agent_id "
            "WHERE n.created_at >= (SELECT MIN(created_at) FROM llm_calls WHERE run_id = %s) "
            "AND n.type = 'hypothesis' ORDER BY n.created_at",
            (run_id,),
        ),
        "posteriors.jsonl": (
            "SELECT * FROM council_hypothesis_posteriors "
            "WHERE hypothesis_node_id IN ("
            "  SELECT id FROM nodes WHERE created_at >= "
            "  (SELECT MIN(created_at) FROM llm_calls WHERE run_id = %s) "
            "  AND type = 'hypothesis'"
            ") ORDER BY created_at",
            (run_id,),
        ),
        "edges.jsonl": (
            "SELECT * FROM edges WHERE from_node IN ("
            "  SELECT id FROM nodes WHERE created_at >= "
            "  (SELECT MIN(created_at) FROM llm_calls WHERE run_id = %s)"
            ") ORDER BY created_at",
            (run_id,),
        ),
        "criteria.jsonl": (
            "SELECT * FROM agora_criteria WHERE created_at >= "
            "(SELECT MIN(created_at) FROM llm_calls WHERE run_id = %s) "
            "ORDER BY created_at",
            (run_id,),
        ),
        "round_close.jsonl": (
            "SELECT * FROM criteria_distillation_calls WHERE created_at >= "
            "(SELECT MIN(created_at) FROM llm_calls WHERE run_id = %s) "
            "ORDER BY created_at",
            (run_id,),
        ),
    }

    summary = {}
    for fname, (sql, params) in tables.items():
        try:
            n = await _dump_table(pool, sql, out_dir / fname, params)
            summary[fname] = n
            print(f"[dump] {fname}: {n} rows")
        except Exception as exc:
            print(f"[dump] WARN {fname}: {type(exc).__name__}: {exc}")
            summary[fname] = -1

    # Cost timeline CSV (derived from llm_calls). The schema has no
    # `role` column — agent_id is the FK to agents.name. Join for the
    # human-readable label so the CSV is grep-friendly per benchmark.
    cost_csv = out_dir / "cost_timeline.csv"
    cumulative = 0.0
    with cost_csv.open("w") as f:
        f.write("ts,agent_name,model,in_tok,out_tok,cost_usd,cumulative_usd\n")
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT lc.created_at, COALESCE(a.name, 'unknown'), "
                    "lc.model, lc.input_tokens, lc.output_tokens, lc.cost_usd "
                    "FROM llm_calls lc "
                    "LEFT JOIN agents a ON a.id = lc.agent_id "
                    "WHERE lc.run_id = %s ORDER BY lc.created_at",
                    (run_id,),
                )
                async for ts, agent_name, model, in_tok, out_tok, cost in cur:
                    cumulative += float(cost or 0)
                    f.write(
                        f"{ts.isoformat()},{agent_name},{model},{in_tok or 0},"
                        f"{out_tok or 0},{cost or 0:.6f},{cumulative:.6f}\n"
                    )
    summary["cost_timeline.csv"] = "ok"

    # Manifest.
    manifest = {
        "run_id": run_id,
        "exp_id": os.environ.get("ASCENSION_EXP_ID", "unknown"),
        "dumped_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": summary,
        "total_cost_usd": cumulative,
        "env": {
            "ASCENSION_BENCHMARK_SYSTEM": os.environ.get("ASCENSION_BENCHMARK_SYSTEM"),
            "ASCENSION_BENCHMARK_SEED": os.environ.get("ASCENSION_BENCHMARK_SEED"),
            "ASCENSION_PER_RUN_CEILING_USD": os.environ.get("ASCENSION_PER_RUN_CEILING_USD"),
            "GOOGLE_GENAI_USE_VERTEXAI": os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"),
            "GOOGLE_CLOUD_PROJECT": os.environ.get("GOOGLE_CLOUD_PROJECT"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[dump] manifest.json + cost_timeline.csv written")
    print(f"[dump] total_cost_usd = ${cumulative:.6f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.run_id, args.out_dir)))
