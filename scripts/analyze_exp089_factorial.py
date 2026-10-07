"""EXP-089 — the 2x2 factorial (family x instrument) assembled from four cells.
$0, deterministic, no LLM. Reads the per-seed per-layer vectors written by
scripts/score_alien_run.py into each batch's results.tsv and computes the
pre-committed contrasts (predictions.md, block dated 2026-08-25):

  * L3 credit, family versus single configuration (Fisher's exact, two-sided)
  * L2 credit, instrument versus none (Fisher's exact, two-sided)
  * exact binomial 95% CIs per cell (Clopper-Pearson)
  * P-EXP-089-1..4 scored against their stated falsifiers

Cells:
  treatment       = EXP-076 treatment  (family + instrument)   exp076_20260530_155507
  control         = EXP-076 control    (single + none)         exp076_20260530_155507
  family_only     = EXP-089            (family + none)         exp076_20260825_171201 (seeds 0-3, mixed roster)
                                                               exp076_20260908_122226 (seeds 4-7, all-Gemini, Amendment 2)
  instrument_only = EXP-089            (single + instrument)   same two batches

Validity rule (predictions.md): a seed counts only on a duration-cap or natural
termination with at least 30 model calls; starved runs are listed, never dropped
silently. Call counts come from the score logs ("SPEND: N llm_calls").

Run: poetry run python scripts/analyze_exp089_factorial.py
"""

from __future__ import annotations

import re
from pathlib import Path

from scipy.stats import beta, fisher_exact

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "records" / "EXP-089"
BATCHES = {
    "exp076": REPO / "records" / "EXP-076" / "exp076_20260530_155507",
    "exp089a": RES / "exp076_20260825_171201",
    "exp089b": RES / "exp076_20260908_122226",   # seeds 4, 5 (family_only); seed 5 instrument_only killed by hand (laptop), not scored
    "exp089c": RES / "exp076_20260908_141716",   # seed 5 instrument_only, relaunch 2
}
# relaunch 2 second batch (seeds 6, 7 both arms) gets its own timestamped dir
for _d in sorted(RES.glob("exp076_20260908_*")):
    if _d.name not in {b.name for b in BATCHES.values()} and _d.name != "exp076_20260908_121552":
        BATCHES[f"exp089d_{_d.name[-6:]}"] = _d
# exp076_20260908_121552 = the aborted mixed-roster relaunch (NVIDIA 410), never valid
ROSTER = {k: ("mixed" if k in ("exp076", "exp089a") else "all-gemini") for k in BATCHES}
MIN_CALLS = 30


def parse_results(tsv: Path) -> list[dict]:
    rows = []
    for line in tsv.read_text().splitlines()[1:]:
        seed, arm, run_id, max_depth, per_layer = line.split("\t")
        m = re.search(r"\((.*)\)", per_layer)
        layers = [x.strip() == "True" for x in m.group(1).split(",")] if m else None
        rows.append(dict(seed=int(seed), arm=arm, run_id=run_id, layers=layers, raw_depth=max_depth))
    return rows


def layers_from_score_log(batch: Path, arm: str, seed: int) -> tuple[list[bool] | None, str | None]:
    """Fallback when results.tsv holds ERR: the (re)score log has the per-layer
    vector. The rescored log (Amendment 2) is preferred over the original."""
    for name in (f"score_{arm}_seed{seed}_rescored_20260908.log", f"score_{arm}_seed{seed}.log"):
        p = batch / name
        if p.exists():
            txt = p.read_text()
            m = re.search(r"per_layer L1\.\.L6 = \((.*?)\)", txt)
            if m:
                return [x.strip() == "True" for x in m.group(1).split(",")], name
    return None, None


def calls_for(batch: Path, arm: str, seed: int) -> int | None:
    for name in (f"score_{arm}_seed{seed}_rescored_20260908.log", f"score_{arm}_seed{seed}.log"):
        p = batch / name
        if p.exists():
            m = re.search(r"SPEND: (\d+) llm_calls", p.read_text())
            if m:
                return int(m.group(1))
    return None


def consecutive_depth(layers: list[bool]) -> int:
    d = 0
    for ok in layers[:4]:
        if not ok:
            break
        d += 1
    return d


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def main() -> None:
    cells: dict[str, list[dict]] = {"treatment": [], "control": [], "family_only": [], "instrument_only": []}
    for key, batch in BATCHES.items():
        tsv = batch / "results.tsv"
        if not tsv.exists():
            continue
        for r in parse_results(tsv):
            r["batch"] = key
            r["roster"] = ROSTER[key]
            r["calls"] = calls_for(batch, r["arm"], r["seed"])
            if r["layers"] is None:
                r["layers"], src = layers_from_score_log(batch, r["arm"], r["seed"])
                if r["layers"] is not None:
                    r["batch"] = f"{key}*"  # * = recovered from {src}
            if r["layers"] is None:
                r["valid"] = False
                r["why"] = "no per-layer vector (run never scored)"
            elif r["calls"] is not None and r["calls"] < MIN_CALLS:
                r["valid"] = False
                r["why"] = f"starved: {r['calls']} calls < {MIN_CALLS}"
            else:
                r["valid"] = True
                r["why"] = ""
            cells[r["arm"]].append(r)

    # the 2026-08-25 EXP-089 batch had a rescore for instrument_only seed 3 (Amendment 2)
    print("EXP-089 factorial: per-cell rows")
    for arm, rows in cells.items():
        rows.sort(key=lambda r: (r["seed"], r["batch"]))
        print(f"\n[{arm}]")
        for r in rows:
            L = "".join("T" if x else "F" for x in r["layers"]) if r["layers"] else "------"
            cd = consecutive_depth(r["layers"]) if r["layers"] else "-"
            flag = "" if r["valid"] else f"  INVALID ({r['why']})"
            print(f"  seed {r['seed']}  {r['batch']:7s} {r['roster']:9s} calls={r['calls']}  layers={L}  consec={cd}{flag}")

    def credit(arm: str, layer_idx: int) -> tuple[int, int]:
        rows = [r for r in cells[arm] if r["valid"]]
        return sum(r["layers"][layer_idx] for r in rows), len(rows)

    def depth3(arm: str) -> tuple[int, int]:
        rows = [r for r in cells[arm] if r["valid"]]
        return sum(consecutive_depth(r["layers"]) >= 3 for r in rows), len(rows)

    print("\nPer-cell credit (valid seeds only), exact binomial 95% CI")
    table = {}
    for arm in cells:
        l2 = credit(arm, 1)
        l3 = credit(arm, 2)
        d3 = depth3(arm)
        table[arm] = dict(l2=l2, l3=l3, d3=d3)
        print(
            f"  {arm:16s} L2 {l2[0]}/{l2[1]} CI[{clopper_pearson(*l2)[0]:.2f},{clopper_pearson(*l2)[1]:.2f}]"
            f"   L3 {l3[0]}/{l3[1]} CI[{clopper_pearson(*l3)[0]:.2f},{clopper_pearson(*l3)[1]:.2f}]"
            f"   consecutive depth>=3 {d3[0]}/{d3[1]}"
        )

    # pre-committed contrasts
    fam_l3 = [table["treatment"]["l3"], table["family_only"]["l3"]]
    sin_l3 = [table["control"]["l3"], table["instrument_only"]["l3"]]
    a = sum(x[0] for x in fam_l3); n_a = sum(x[1] for x in fam_l3)
    b = sum(x[0] for x in sin_l3); n_b = sum(x[1] for x in sin_l3)
    p_l3 = fisher_exact([[a, n_a - a], [b, n_b - b]], alternative="two-sided")[1]
    print(f"\nContrast 1 (L3 credit): family {a}/{n_a} vs single-configuration {b}/{n_b}; Fisher two-sided p = {p_l3:.4g}")

    ins_l2 = [table["treatment"]["l2"], table["instrument_only"]["l2"]]
    non_l2 = [table["control"]["l2"], table["family_only"]["l2"]]
    c = sum(x[0] for x in ins_l2); n_c = sum(x[1] for x in ins_l2)
    d = sum(x[0] for x in non_l2); n_d = sum(x[1] for x in non_l2)
    p_l2 = fisher_exact([[c, n_c - c], [d, n_d - d]], alternative="two-sided")[1]
    print(f"Contrast 2 (L2 credit): instrument {c}/{n_c} vs none {d}/{n_d}; Fisher two-sided p = {p_l2:.4g}")

    # predictions
    fl3 = table["family_only"]["l3"]; il2 = table["instrument_only"]["l2"]
    id3 = table["instrument_only"]["d3"]; fd3 = table["family_only"]["d3"]
    print("\nPre-registered predictions (P-EXP-089-*), scored on valid seeds:")
    print(f"  P-1 family_only passes L3 on >=5/8 (falsified <=3/8): {fl3[0]}/{fl3[1]} -> "
          f"{'HELD' if fl3[0] >= 5 else ('FALSIFIED' if fl3[0] <= 3 and fl3[1] == 8 else 'indeterminate')}")
    print(f"  P-2 instrument_only passes L2 on >=5/8 (falsified <=3/8): {il2[0]}/{il2[1]} -> "
          f"{'HELD' if il2[0] >= 5 else ('FALSIFIED' if il2[0] <= 3 and il2[1] == 8 else 'indeterminate')}")
    print(f"  P-3 instrument_only consecutive depth>=3 on <=2/8 (falsified >=4/8): {id3[0]}/{id3[1]} -> "
          f"{'HELD' if id3[0] <= 2 and id3[1] == 8 else ('FALSIFIED' if id3[0] >= 4 else 'indeterminate')}")
    print(f"  P-4 family_only consecutive depth>=3 on <=4/8 (>=5/8 = family alone suffices): {fd3[0]}/{fd3[1]} -> "
          f"{'HELD' if fd3[0] <= 4 and fd3[1] == 8 else ('stronger result: family alone suffices' if fd3[0] >= 5 else 'indeterminate')}")

    # roster strata for the EXP-089 cells
    print("\nRoster strata (EXP-089 cells):")
    for arm in ("family_only", "instrument_only"):
        for roster in ("mixed", "all-gemini"):
            rows = [r for r in cells[arm] if r["valid"] and r["roster"] == roster]
            if rows:
                print(f"  {arm:16s} {roster:10s} n={len(rows)}  L2={sum(r['layers'][1] for r in rows)}  "
                      f"L3={sum(r['layers'][2] for r in rows)}  depth>=3={sum(consecutive_depth(r['layers'])>=3 for r in rows)}")


if __name__ == "__main__":
    main()
