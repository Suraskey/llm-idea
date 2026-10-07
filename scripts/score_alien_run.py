"""Post-hoc alien discovery-depth scorer for a completed orchestrator run (EXP-072).

$0 LLM — deterministic sympy/scipy via ``benchmarks.alien_depth.score_depth``
(no model judges ground truth, SCOPE §22.6). Reads the run's persisted
hypothesis nodes from Postgres, rebuilds the alien truth_cfg + held-out set
(seed-independent — the 15.0 ICs draw nothing from the RNG), scores discovery
depth, and reports provenance flags (P-15.2-6), node status (P-15.2-5 sign-off),
spend, and provider mix.

Usage: poetry run python scripts/score_alien_run.py <run_id> [--family | --family-v2]

  <run_id>     the orchestrator run to score (reads its persisted hypothesis nodes)
  --family     score a multi-charge FAMILY run (EXP-076, Phase 19-04/05): the
               L3/L4 hidden-charge gates fit a(r) against the HELD-OUT family
               configs (charge products strictly off the observed family,
               P-EXP-076-7), while L1/L2/L5 stay on the single-config product-1
               baseline (L-052 — the charge-independent layers must not run on
               high-product data, which would inflate the Newton residual and
               sink the relative reduction). Without the flag, scores a single-
               config run exactly as before (byte-unchanged path).
  --family-v2  score an EXP-080 V2-regime FAMILY run (STRAT-03): the SECOND §10
               law (charge_coupling="sum", correction_exponent=2.5). The scorer
               is regime-aware — L2 centers on r^-2.5 and L3/L4 credit the
               SUM-confound charge via the cross-config s-SUM gate against the
               held-out V2 family sums (strictly off the observed sums,
               P-EXP-080-6). L1/L2/L5 score on the V2-law baseline held-out set
               (build_tier2_held_out_ics_v2 — the V2 analog of the product-1
               baseline). Scoring a V2 run with --family (the V1 oracle) would
               DENY every legitimate V2 L2/L3 and manufacture a spurious null.
"""
from __future__ import annotations

import asyncio
import sys


async def _amain(run_id: str, *, family: bool = False, family_v2: bool = False) -> None:
    from ascension.benchmarks.alien_depth import RegimeSpec, score_depth
    from ascension.benchmarks.alien_fixtures import alien_benchmark
    from ascension.common.db import close_async_pool, get_async_pool

    # Seed-independent truth (the bundle is discarded; only truth_cfg + held_out
    # feed the scorer). seed=0 mirrors the run launcher (ASCENSION_BENCHMARK_SEED=0).
    _bundle, truth_cfg, held_out_ics = alien_benchmark(seed=0)

    # FAMILY scoring (EXP-076): the L3/L4 cross-config s-product gate recovers
    # a(r) from the HELD-OUT family configs (products {3,5} — strictly off the
    # observed {1,2,4,6} family, P-EXP-076-7 anti-reward-hack). L1/L2/L5 still
    # score on the product-1 baseline above (L-052 decoupling). family_scoring_cfgs
    # stays None for a single-config run so that path is byte-identical.
    family_scoring_cfgs = None
    regime = None  # None = the V1 regime (byte-identical scorer path)
    if family_v2:
        # EXP-080 V2 regime (STRAT-03): the SECOND §10 law (sum coupling + r^-2.5).
        # The regime-aware oracle credits the V2 r^-2.5 correction (L2) + the
        # SUM-confound charge (L3/L4 via the cross-config s-SUM gate) against the
        # held-out V2 family sums {4,6} (off the observed {2,3,5,7}, P-EXP-080-6).
        # L1/L2/L5 score on the V2-law baseline held-out set (the V2 analog of the
        # product-1 baseline) so the V2 L2 gate finds the r^-2.5 correction.
        from ascension.benchmarks.alien_fixtures import (
            alien_family_v2_benchmark,
            build_tier2_held_out_ics_v2,
        )

        _family, held_out_cfgs = alien_family_v2_benchmark(seed=0)
        family_scoring_cfgs = held_out_cfgs
        held_out_ics = build_tier2_held_out_ics_v2(seed=0)
        truth_cfg = _family[0][1]  # the V2 tuned truth (for provenance)
        regime = RegimeSpec(correction_exponent=2.5, charge_coupling="sum")
    elif family:
        from ascension.benchmarks.alien_fixtures import alien_family_benchmark

        _family, held_out_cfgs = alien_family_benchmark(seed=0)
        family_scoring_cfgs = held_out_cfgs

    pool = await get_async_pool()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT config, status, halt_reason FROM runs WHERE id=%s",
                    (run_id,),
                )
                run_row = await cur.fetchone()
                await cur.execute(
                    "SELECT id, content, status FROM nodes "
                    "WHERE run_id=%s AND type='hypothesis'",
                    (run_id,),
                )
                hyp_rows = await cur.fetchall()
                await cur.execute(
                    "SELECT count(*), coalesce(sum(cost_usd),0) FROM llm_calls "
                    "WHERE run_id=%s",
                    (run_id,),
                )
                n_calls, total_cost = await cur.fetchone()
                await cur.execute(
                    "SELECT provider, count(*), coalesce(sum(cost_usd),0) "
                    "FROM llm_calls WHERE run_id=%s GROUP BY provider ORDER BY provider",
                    (run_id,),
                )
                provider_rows = await cur.fetchall()
                # total nodes by type (run-shape sanity)
                await cur.execute(
                    "SELECT type, count(*) FROM nodes WHERE run_id=%s "
                    "GROUP BY type ORDER BY type",
                    (run_id,),
                )
                node_type_rows = await cur.fetchall()
    finally:
        await close_async_pool()

    config = (run_row[0] if run_row else {}) or {}
    status = run_row[1] if run_row else None
    halt_reason = run_row[2] if run_row else None

    hyps = [r[1] for r in hyp_rows]
    status_breakdown: dict = {}
    for r in hyp_rows:
        status_breakdown[r[2]] = status_breakdown.get(r[2], 0) + 1

    depth = score_depth(
        hyps,
        truth_cfg,
        held_out_ics,
        family_scoring_cfgs=family_scoring_cfgs,
        regime=regime,
    )

    print("=" * 72)
    print(f"ALIEN RUN {run_id}")
    print(f"  run.status={status}  halt_reason={halt_reason}")
    print(
        f"  PROVENANCE (P-15.2-6): benchmark_system="
        f"{config.get('benchmark_system')!r}  included_in_distillation_v1="
        f"{config.get('included_in_distillation_v1')!r}"
    )
    if family_v2:
        _mode = "FAMILY-V2 (V2 law: sum coupling + r^-2.5; L3/L4 vs held-out sums {4,6})"
    elif family:
        _mode = "FAMILY (L3/L4 vs held-out products {3,5})"
    else:
        _mode = "single-config"
    print(f"  SCORING MODE: {_mode}")
    print(f"  node types: {dict(node_type_rows)}")
    print(
        f"  hypothesis nodes={len(hyp_rows)}  status_breakdown={status_breakdown}"
    )
    print(f"  SPEND: {n_calls} llm_calls, ${float(total_cost):.4f} total")
    for prov, c, cost in provider_rows:
        print(f"    provider={prov!r}: {c} calls, ${float(cost):.4f}")
    print("-" * 72)
    print(f"  DISCOVERY DEPTH (P-15.2-1/2/3): max_depth={depth.max_depth}")
    print(f"    per_layer L1..L6 = {depth.per_layer}")
    # HONEST consecutive force-law depth (L-057): walk L1->L4, stop at the first
    # False. max_depth above is the highest-FIRING layer, which the multi-charge
    # family can inflate via the L3 charge-credit + the automatic L6 parity credit
    # (L-051) WITHOUT consecutive discovery. The citeable depth is this consecutive
    # value; L5/L6 are orthogonal (L-039) and reported separately, never folded in.
    _pl = depth.per_layer
    _consec = 0
    for _passed in _pl[:4]:
        if _passed:
            _consec += 1
        else:
            break
    print(
        f"    HONEST consecutive force-law depth (L1-L4, L-057) = {_consec}"
        f"   [L5={_pl[4]} L6={_pl[5]} reported separately, non-consecutive]"
    )
    print(
        f"    held_out_rss_reduction={depth.held_out_rmse}  q_drift={depth.q_drift}"
    )
    print(f"  scorer notes ({len(depth.notes)}):")
    for n in depth.notes[:60]:
        print(f"    - {n}")
    forms = sorted(
        {
            (h or {}).get("symbolic_form", "")
            for h in hyps
            if (h or {}).get("symbolic_form")
        }
    )
    print(f"  distinct symbolic_form count={len(forms)} (sample up to 30):")
    for f in forms[:30]:
        print(f"    {f}")
    print("=" * 72)


if __name__ == "__main__":
    args = sys.argv[1:]
    family = False
    family_v2 = False
    if "--family-v2" in args:
        family_v2 = True
        args = [a for a in args if a != "--family-v2"]
    if "--family" in args:
        family = True
        args = [a for a in args if a != "--family"]
    if family and family_v2:
        print("error: pass only ONE of --family / --family-v2 (a run is one regime)")
        raise SystemExit(2)
    if len(args) != 1:
        print(
            "usage: poetry run python scripts/score_alien_run.py <run_id> "
            "[--family | --family-v2]"
        )
        raise SystemExit(2)
    asyncio.run(_amain(args[0], family=family, family_v2=family_v2))
