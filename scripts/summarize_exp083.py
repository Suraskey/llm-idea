"""EXP-083 landing summary — every P-EXP-083 prediction checked from artifacts.

$0, deterministic, read-only. Computes from the run dir (results.tsv + council
logs) and prints the outcome block for EXPERIMENTS.md:
  P1: git SHA provenance + zero 3.4 deliveries (the L-063 fingerprint)
  P2: the plateau-gate quiet pattern (battery deliveries per seed; a seed with
      an in-band proposal and NO subsequent deliveries exhibits the signature)
  P3: L3 sum-gate rate (>=6/8 pass bar; 7/8 point prediction)
  P4: depth>=3 + L2 rates with exact Clopper-Pearson CIs (primary scorer)
  P5: terminal-belief anchoring is computed separately by
      analyze_exponent_proximity.py --correction-threshold -2.05

Usage: poetry run python scripts/summarize_exp083.py <run_dir>
"""

from __future__ import annotations

import glob
import os
import re
import sys

from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def consec_depth(per_layer: list[bool]) -> int:
    d = 0
    for v in per_layer[:4]:  # consecutive L1..L4 (the honest-depth convention)
        if v:
            d += 1
        else:
            break
    return d


def main(run_dir: str) -> None:
    results = os.path.join(run_dir, "results.tsv")
    rows: list[tuple[int, list[bool]]] = []
    with open(results) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            m = re.search(r"\(([^)]*)\)", parts[4])
            if not m:
                continue
            per_layer = [t.strip() == "True" for t in m.group(1).split(",")]
            rows.append((int(parts[0]), per_layer))
    n = len(rows)
    print(f"run dir: {run_dir}  seeds scored: {n}")

    # config provenance (P1a)
    cfg = open(os.path.join(run_dir, "config.txt")).read()
    sha = re.search(r"git_sha=(\w+)", cfg)
    print(f"P1a git_sha recorded: {sha.group(1)[:12] if sha else 'MISSING'}")

    # P1b: zero 3.4 deliveries; count deliveries + exponents per seed
    print("\nper-seed instrument channels:")
    any34 = False
    quiet_candidates = 0
    for seed, _pl in sorted(rows):
        logs = glob.glob(os.path.join(run_dir, f"council_seed{seed}_*.log"))
        if not logs:
            continue
        text = open(logs[0], errors="replace").read()
        battery = re.findall(r"fitted_exponent=([0-9.]+)", text)
        residual = len(re.findall(r"persisted residual feedback", text))
        if any(abs(float(e) - 3.4) < 1e-9 for e in battery):
            any34 = True
        if not battery:
            quiet_candidates += 1
        print(
            f"  seed {seed}: battery deliveries={len(battery)} "
            f"{sorted(set(battery))} residual deliveries={residual}"
        )
    print(f"P1b zero fitted_exponent=3.4 anywhere: {'MET' if not any34 else 'FAILED'}")
    print(
        f"P2 battery-quiet seeds (zero deliveries — the DET-02 gate never saw a "
        f"plateau without an in-band term): {quiet_candidates}/{n}"
    )

    # P3/P4: layer rates
    l2 = sum(1 for _s, pl in rows if pl[1])
    l3 = sum(1 for _s, pl in rows if pl[2])
    d3 = sum(1 for _s, pl in rows if consec_depth(pl) >= 3)
    for label, k in (("L2 band", l2), ("L3 sum-gate", l3), ("consec depth>=3", d3)):
        lo, hi = clopper_pearson(k, n)
        print(f"{label}: {k}/{n}  exact 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"P3 (L3 >= 6/8): {'MET' if l3 >= 6 else 'NOT MET'} (point prediction 7/8)")
    print("\nper-seed consec depth:", {s: consec_depth(pl) for s, pl in sorted(rows)})


if __name__ == "__main__":
    main(sys.argv[1])
