"""$0 DESCRIPTIVE ROBUSTNESS RESCORE — proposal-coupled L2 over stored hypotheses.

NOT a pre-registered headline and NOT a re-score of the confirmed records' default
metric (the locked `score_depth` is byte-unchanged; the confirmed EXP-075/076/077
numbers stand). This is the L-060 follow-up: it asks, post-hoc and at $0, whether
the EXP-075 depth separation survives an L2 gate that, instead of the FIXED truth
basis (−1/r**3.5) the soft gate uses, fits EACH PROPOSAL'S OWN correction exponent
to the held-out observed acceleration and gates on a material RSS reduction.

Why this matters (the critique, L-060): in EXP-075 BOTH arms clear L1 and L3; only
L2 separates them. The soft L2 gate reduces to a ±0.3 r-exponent band test — the
Council wrote ~r**-3.5 (in band), the Solo wrote a charge term at r**-3 (out of
band). The question this answers: when we drop the band and instead ask "does the
proposal's OWN correction exponent, fitted to the data, do real residual work?",
does the Council still pass and the Solo still fail — or does the Solo's r**-3
term clear the bar too (in which case δ=1.0 was carried by the soft band alone)?

Honest scope (stated, per L-060): the proposals are SYMBOLIC — the correction
coefficient is a free symbol (β, s1, s2), so a per-proposal numeric gate MUST fit
the coefficient, and a free-fit coefficient absorbs sign. So the proposal's STATED
sign cannot be gated at the single-config level; the discriminator is purely the
proposal's own EXPONENT fit-quality. We report the fitted coefficient sign for
transparency but it is not the gate. The genuinely proposal-coupled, sign-and-
charge-aware signal remains the cross-config L3/L4 gate (unchanged).

Run:  poetry run python scripts/rescore_l2_proposal_coupled.py
"""

from __future__ import annotations

import asyncio
import csv

import numpy as np

from ascension.benchmarks.alien_depth import (
    _L2_RSS_REDUCTION_REL,
    _corr_basis,
    _newton_basis,
    _observed_accel,
    _parse,
    _r_power_exponents,
)
from ascension.benchmarks.alien_fixtures import alien_benchmark
from ascension.common.db import close_async_pool, get_async_pool

# The SAME L1/L2 baseline held-out set score_alien_run.py --family uses for L2.
_HELD_OUT = alien_benchmark(seed=0)[2]


def _fit(a, cols):
    """Least-squares fit a -> cols; return (coeffs, rss)."""
    X = np.vstack(cols).T
    c, *_ = np.linalg.lstsq(X, a, rcond=None)
    rss = float(np.sum((a - X @ c) ** 2))
    return c, rss


def proposal_coupled_l2(expr) -> dict:
    """Tightened L2 for ONE parsed hypothesis.

    A 'correction' = a second radial term shorter-range than inverse-square
    (exponent e < -2.25). For each, fit a(r) ~ c0·(−1/r²) + c1·(−1/r**(−e)) on the
    held-out span and measure the relative RSS reduction the proposal's OWN
    exponent buys over a Newton-only fit. Pass = reduction ≥ 0.05 (the soft gate's
    margin). The fitted coefficient SIGN is NOT gated (ungateable at the
    single-config level — the L-060 sharpened finding; the true r^-3.5
    correction itself fits c1<0 here); c1 is reported for transparency only.
    """
    exps = _r_power_exponents(expr)
    second = sorted({round(float(e), 4) for e in exps if e < -2.25})
    if not second:
        return {"passed": False, "best_red": 0.0, "best_exp": None, "c1": None}
    r, a = _observed_accel(_HELD_OUT)
    newton = _newton_basis(r)
    _, rss_base = _fit(a, [newton])
    rows = []
    for e in second:
        corr = _corr_basis(r, -e)  # e<0 → correction_exponent = -e > 0 → −1/r**(−e)
        c, rss_aug = _fit(a, [newton, corr])
        red = (rss_base - rss_aug) / rss_base if rss_base > 0 else 0.0
        c1 = float(c[1])
        # Gate = the proposal's OWN exponent buys a material RSS reduction. We do
        # NOT gate the fitted coefficient SIGN: for symbolic proposals the coeff
        # is free, so the least-squares fit takes whatever sign fits the data —
        # the true r^-3.5 correction itself fits c1<0 here. Sign is ungateable at
        # the single-config level (the L-060 finding, sharpened); c1 reported only
        # for transparency. |c1| materiality is implied by a >=5% reduction.
        rows.append((e, red, c1, red >= _L2_RSS_REDUCTION_REL))
    rows.sort(key=lambda t: -t[1])
    best_e, best_red, best_c1, _ = rows[0]
    return {
        "passed": any(t[3] for t in rows),
        "best_red": best_red,
        "best_exp": best_e,
        "c1": best_c1,
    }


def run_level_tight_l2(symbolic_forms) -> dict:
    """Run-level tightened L2 = ANY hypothesis passes (mirrors score_depth's OR)."""
    best = {"passed": False, "best_red": 0.0, "best_exp": None, "c1": None}
    for sf in symbolic_forms:
        expr = _parse(sf)
        if expr is None:
            continue
        v = proposal_coupled_l2(expr)
        if v["passed"]:
            best = {"passed": True, **{k: v[k] for k in ("best_red", "best_exp", "c1")}}
            # keep scanning to report the largest reduction seen even once passed
        if v["best_red"] > best["best_red"]:
            best["best_red"] = v["best_red"]
            best["best_exp"] = v["best_exp"]
            best["c1"] = v["c1"]
    return best


def _negative_control() -> None:
    """Non-vacuity sanity: a genuine r**-3.5 correction passes; a bare inverse-
    square (no second term) fails. The gate is fit-quality, not a band — so we also
    print a long-range r**-1 term to show the gate is not trivially passed by any
    second power. (Sign is NOT asserted — it is ungateable for free-symbol coeffs.)"""
    good = proposal_coupled_l2(_parse("-G*m2/r**2 - beta/r**3.5"))
    longrange = proposal_coupled_l2(_parse("-G*m2/r**2 - beta/r**1"))
    bare = proposal_coupled_l2(_parse("-G*m2/r**2"))
    assert good["passed"], f"control: true r^-3.5 correction must pass, got {good}"
    assert not bare["passed"], f"control: bare inverse-square must fail, got {bare}"
    print(f"non-vacuity control: r^-3.5 PASSES (red={good['best_red']:.3f}, c1={good['c1']:+.3f}); "
          f"bare 1/r^2 fails (no 2nd term); "
          f"r^-1 long-range red={longrange['best_red']:.3f} pass={longrange['passed']}")


async def _amain() -> None:
    runs: dict[tuple[str, str], str] = {}
    with open("records/EXP-075/exp075_20260601_140400/results.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            runs[(row["seed"], row["arm"])] = row["run_id"]

    pool = await get_async_pool()
    forms: dict[str, list[str]] = {}
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for rid in runs.values():
                    await cur.execute(
                        "SELECT content FROM nodes WHERE run_id=%s AND type='hypothesis'",
                        (rid,),
                    )
                    rows = await cur.fetchall()
                    forms[rid] = [
                        (c or {}).get("symbolic_form", "")
                        for (c,) in rows
                        if (c or {}).get("symbolic_form")
                    ]
    finally:
        await close_async_pool()

    print("=" * 78)
    print("PROPOSAL-COUPLED L2 RESCORE — EXP-075 (descriptive, $0; default records unchanged)")
    print("  gate: proposal's OWN second-exponent fitted to held-out a(r); "
          f"reduction >= {_L2_RSS_REDUCTION_REL} (sign NOT gated — ungateable, L-060)")
    print("  (L1 & L3 are T for BOTH arms in the confirmed records, so consec depth")
    print("   = 3 if tightened-L2 passes else 1)")
    print("-" * 78)
    _negative_control()
    print("-" * 78)
    print(f"{'seed':>4} | {'COUNCIL L2tight (red@exp, c1 shown)':>40} | {'SOLO L2tight (red@exp, c1 shown)':>38}")
    print("-" * 78)
    council_depth, solo_depth = [], []
    for seed in sorted({s for (s, _a) in runs}):
        c = run_level_tight_l2(forms[runs[(seed, "council")]])
        s = run_level_tight_l2(forms[runs[(seed, "single_model")]])
        cd = 3 if c["passed"] else 1
        sd = 3 if s["passed"] else 1
        council_depth.append(cd)
        solo_depth.append(sd)

        def fmt(v):
            e = f"{-v['best_exp']:.2f}" if v["best_exp"] is not None else "—"
            return f"{'PASS' if v['passed'] else 'fail'} (red={v['best_red']:.3f}@r^-{e}, c1={'+' if (v['c1'] or 0)>0 else '-'})"

        print(f"{seed:>4} | depth {cd}: {fmt(c):>33} | depth {sd}: {fmt(s):>31}")
    print("-" * 78)
    import statistics

    diffs = [cd - sd for cd, sd in zip(council_depth, solo_depth)]
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    print(f"Council consec depth (tightened L2): {council_depth}  mean {statistics.mean(council_depth):.3f}")
    print(f"Solo    consec depth (tightened L2): {solo_depth}  mean {statistics.mean(solo_depth):.3f}")
    print(f"paired diffs (Council−Solo): {diffs}  ({pos} positive, {neg} negative)")
    if pos == 8 and neg == 0:
        print("VERDICT: separation SURVIVES the proposal-coupled gate (Council>Solo on 8/8) "
              "→ δ=1.0 is NOT a soft-band artifact; the tightened-gate re-run is worth firing.")
    elif pos == 0:
        print("VERDICT: separation COLLAPSES under the proposal-coupled gate "
              "→ the soft ±0.3 band was the discriminator; restructure so the "
              "cross-config L3/L4 result carries the observability claim (exit b).")
    else:
        print(f"VERDICT: separation PARTIALLY survives ({pos}/8) → report honestly; "
              "the headline weakens from δ=1.0 under a tightened gate.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(_amain())
