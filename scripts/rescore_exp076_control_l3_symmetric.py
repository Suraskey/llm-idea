"""STRAT-09 — symmetric L3 rescore of the EXP-076 CONTROL arm ($0, post-hoc).

WHY (EXP-076 honest caveat 1, the referee-probe asymmetry). The confirmed
EXP-076 result scored the CONTROL arm with the *single-config* L3 gate
(``score_alien_run.py`` WITHOUT ``--family``): a control hypothesis carrying a
structural ``s1*s2`` earned L3 whenever the hidden basis ``(1-cos(gamma*r))/r**2``
bought a generic single-config RSS reduction over Newton+alpha (control seed 2:
0.132 >= 0.05 -> L3, consecutive depth 3). The TREATMENT arm, by contrast, scored
L3 with the *family cross-config identifiability* gate
(``_cross_config_sproduct_reduction``): the hidden charge earns L3 ONLY when
per-config charge-product scaling does material RSS work BEYOND a single global
amplitude, fit across configs at DISTINCT charge products. That is a measurement
ASYMMETRY: the control's L3 bar was looser than the treatment's.

THE SYMMETRIC CRITERION (this script). Apply the SAME cross-config identifiability
function the family L3 uses to the control's L3 credit, scored against the configs
the CONTROL RUN ACTUALLY DISTINGUISHED — its single-charge operating point
(``TIER2_HELD_OUT_ICS``, all at charge product 1.0). A control run saw ONE charge
product, so its ``s1*s2`` is mathematically degenerate with a global amplitude
(L-051: at product 1, s1*s2 == 1 == beta). ``_cross_config_sproduct_reduction``
over a single-product set returns 0.0 (its own ``len({products}) < 2`` guard) ->
0.0 < _L3_RSS_REDUCTION_REL (0.05) -> L3 DENIED. This is the honest symmetric gate:
the family earns L3 from REAL cross-config charge-product variation; the control,
having generated none, cannot.

NOTE on the naive ``--family`` alternative (REJECTED). Feeding the control the
FAMILY's held-out cfgs {3,5} would grant it the 0.212 cross-config reduction the
control RUN never produced (that number is a property of the family data, not the
control's hypothesis). That would NOT tighten the gate — it would launder the
family's identifiability onto the control. The symmetric gate scores each arm
against the configs THAT ARM could distinguish: family -> {3,5}; control -> {1}.

$0 / NO LLM (SCOPE §22.6): deterministic sympy/scipy via ``score_depth``. ADDITIVE
amendment (SCOPE §22.1): this NEVER edits the confirmed EXP-076 numbers in place;
it re-scores the PERSISTED control hypothesis nodes and reports the corrected
control distribution alongside (not over) the headline.

Usage: poetry run python scripts/rescore_exp076_control_l3_symmetric.py
"""

from __future__ import annotations

import asyncio

# The 8 CONTROL run_ids from the registered EXP-076 batch
# (records/EXP-076/exp076_20260530_155507/, control_seed{N}_<run_id>.log).
_CONTROL_RUNS: tuple[tuple[int, str], ...] = (
    (0, "cf1b445b-7774-41a0-a390-ba8b2fc99048"),
    (1, "69fb2bd4-517f-4288-a984-21b104d9ef10"),
    (2, "4091bfa8-7f3a-44a9-9868-0da6ba932842"),  # the s2=3 anomaly
    (3, "4b612516-fa88-4242-a666-90c2fd2e4d62"),
    (4, "3750f571-efae-4b2d-a657-e502c610b44d"),
    (5, "e9a9d2b1-9b3c-4eb3-bfb5-6580731991aa"),
    (6, "50947eba-0161-4cc4-b8c2-018f83c83ba7"),
    (7, "75eaf77e-12de-4844-886a-a8114412a00a"),
)


def _consec(per_layer: tuple[bool, ...]) -> int:
    """HONEST consecutive force-law depth (L1->L4, stop at first False) — L-057."""
    c = 0
    for passed in per_layer[:4]:
        if passed:
            c += 1
        else:
            break
    return c


async def _fetch_hyps(pool, run_id: str) -> list:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT content FROM nodes WHERE run_id=%s AND type='hypothesis'",
                (run_id,),
            )
            rows = await cur.fetchall()
    return [r[0] for r in rows]


async def _amain() -> None:
    from ascension.benchmarks.alien_depth import score_depth
    from ascension.benchmarks.alien_fixtures import alien_benchmark
    from ascension.common.db import close_async_pool, get_async_pool

    # Control's own single-charge truth + held-out set (charge product 1.0, the
    # operating point the control RUN actually distinguished). seed=0 mirrors the
    # launcher (ASCENSION_BENCHMARK_SEED=0); the 15.0 ICs draw nothing from the RNG
    # so this truth is seed-independent (matches score_alien_run.py).
    _bundle, truth_cfg, held_out_ics = alien_benchmark(seed=0)

    # The SYMMETRIC L3 gate: feed the SAME cross-config identifiability function the
    # family uses (family_scoring_cfgs), but with the CONTROL's own single-charge
    # held-out configs. One distinct product -> _cross_config_sproduct_reduction
    # returns 0.0 -> L3 denied unless a control hypothesis somehow generated genuine
    # cross-config charge-product variation (it cannot — it ran at one product).
    symmetric_cfgs = held_out_ics

    pool = await get_async_pool()
    rows_out: list[dict] = []
    try:
        for seed, run_id in _CONTROL_RUNS:
            hyps = await _fetch_hyps(pool, run_id)

            # ORIGINAL gate (single-config, as the confirmed EXP-076 scored control).
            orig = score_depth(hyps, truth_cfg, held_out_ics, family_scoring_cfgs=None)
            # SYMMETRIC gate (family cross-config identifiability on control's own
            # single-charge configs).
            sym = score_depth(hyps, truth_cfg, held_out_ics, family_scoring_cfgs=symmetric_cfgs)
            rows_out.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "n_hyp": len(hyps),
                    "orig_per_layer": orig.per_layer,
                    "orig_consec": _consec(orig.per_layer),
                    "orig_L3": orig.per_layer[2],
                    "sym_per_layer": sym.per_layer,
                    "sym_consec": _consec(sym.per_layer),
                    "sym_L3": sym.per_layer[2],
                }
            )
    finally:
        await close_async_pool()

    print("=" * 88)
    print("EXP-076 CONTROL L3 SYMMETRIC RESCORE (STRAT-09, $0, post-hoc, additive)")
    print("  symmetric L3 = family cross-config identifiability gate applied to")
    print("  control's own single-charge configs (product 1.0 -> no cross-config")
    print("  variation -> charge not identifiable -> L3 denied; L-051).")
    print("=" * 88)
    hdr = f"{'seed':>4} {'orig_consec':>11} {'orig_L3':>8} {'sym_consec':>10} {'sym_L3':>7}  run_id"
    print(hdr)
    print("-" * 88)
    for r in rows_out:
        flag = "   <== s2 anomaly" if r["seed"] == 2 else ""
        print(
            f"{r['seed']:>4} {r['orig_consec']:>11} {str(r['orig_L3']):>8} "
            f"{r['sym_consec']:>10} {str(r['sym_L3']):>7}  {r['run_id']}{flag}"
        )
    print("-" * 88)

    orig_dist: dict[int, int] = {}
    sym_dist: dict[int, int] = {}
    for r in rows_out:
        orig_dist[r["orig_consec"]] = orig_dist.get(r["orig_consec"], 0) + 1
        sym_dist[r["sym_consec"]] = sym_dist.get(r["sym_consec"], 0) + 1

    def _fmt(d: dict[int, int]) -> str:
        return ", ".join(f"depth{k}={v}" for k, v in sorted(d.items()))

    n = len(rows_out)
    orig_ge3 = sum(1 for r in rows_out if r["orig_consec"] >= 3)
    sym_ge3 = sum(1 for r in rows_out if r["sym_consec"] >= 3)
    orig_mean = sum(r["orig_consec"] for r in rows_out) / n
    sym_mean = sum(r["sym_consec"] for r in rows_out) / n

    print(f"ORIGINAL control consec-depth dist (single-config L3): {_fmt(orig_dist)}")
    print(f"   >=3 on {orig_ge3}/{n} seeds; mean {orig_mean:.3f}")
    print(f"SYMMETRIC control consec-depth dist (cross-config L3) : {_fmt(sym_dist)}")
    print(f"   >=3 on {sym_ge3}/{n} seeds; mean {sym_mean:.3f}")
    print("-" * 88)
    s2 = next(r for r in rows_out if r["seed"] == 2)
    verdict = (
        "DROPS to <=2 (anomaly does NOT survive the symmetric gate)"
        if s2["sym_consec"] <= 2
        else "SURVIVES at >=3 (anomaly persists under the symmetric gate)"
    )
    print(
        f"s2 ANOMALY VERDICT: orig consec={s2['orig_consec']} -> sym consec={s2['sym_consec']}  =>  {verdict}"
    )
    print("=" * 88)


if __name__ == "__main__":
    asyncio.run(_amain())
