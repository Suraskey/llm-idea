"""Figure 5 (EXP-262): criterion head-to-head, regret relative to the best candidate.

Descriptive view of the pre-registered head-to-head: for each task and criterion,
the recovery error of the criterion's pick divided by the error of the best
candidate available on that task (1 = picked the best). Medians over tasks with
bootstrap 95% CIs (10,000 resamples, seed 262), per protocol. The registered
statistics (medians of M2, Wilcoxon vs RANK) are in results/exp262.
"""
from __future__ import annotations

import csv
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401

SRC = "results/exp262/picks.tsv"
OUT = "figures/fig5-criterion-head-to-head"
CRIT = ["AOPT", "DOPT", "VOI", "RANK", "DISAGREE", "RANDOM"]
LABEL = {
    "AOPT": "A-optimal (CARTOGRAPH-style)",
    "DOPT": "D-optimal (= rank-then-D hybrid)",
    "VOI": "Value of information (MDA-style)",
    "RANK": "Rank, then condition number (ours)",
    "DISAGREE": "Disagreement (LLM-ACES-style)",
    "RANDOM": "Random pick (expected)",
}
COLOR = {"T1": "#2a78d6", "T2": "#eb6834"}  # validated pair (dataviz validator, light)
PROTO = {"T1": "full state + derivative", "T2": "first coordinate only"}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "savefig.bbox": "tight", "figure.dpi": 150})

rows = list(csv.DictReader(open(SRC), delimiter="\t"))
ratios: dict[tuple[str, str], list[float]] = {}
for r in rows:
    if r["in_P2_setting"] != "True":
        continue
    pm, om = float(r["pick_m2"]), float(r["oracle_best_m2"])
    if not (math.isfinite(pm) and math.isfinite(om)) or om <= 0:
        continue
    ratios.setdefault((r["protocol"], r["criterion"]), []).append(pm / om)

rng = np.random.default_rng(262)


def med_ci(x):
    x = np.asarray(x)
    boots = np.median(rng.choice(x, size=(10000, x.size), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


fig, ax = plt.subplots(figsize=(5.4, 2.7))
y = np.arange(len(CRIT))[::-1]
for k, proto in enumerate(["T1", "T2"]):
    off = 0.14 if proto == "T1" else -0.14
    for yi, c in zip(y, CRIT, strict=True):
        m, lo, hi = med_ci(ratios[(proto, c)])
        ax.plot([lo, hi], [yi + off] * 2, color=COLOR[proto], lw=2, solid_capstyle="round")
        ax.plot(m, yi + off, "o", ms=5, color=COLOR[proto], mec="#fcfcfb", mew=1.2,
                label=f"{proto}: {PROTO[proto]}" if yi == y[0] else None)
ax.axvline(1.0, color=MUTED, lw=0.6, ls=(0, (2, 2)))
ax.set_yticks(y)
ax.set_yticklabels([LABEL[c] for c in CRIT], color=INK)
ax.set_xscale("log")
ax.set_xlabel("recovery error of the pick / error of the best candidate on that task", color=INK)
ax.xaxis.grid(True, color="#e2e1dc", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=7)
ax.set_xticks([1, 1.5, 2, 3, 5])
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_title("Median regret over 61 ODEBench tasks per protocol (95% bootstrap CI)",
             fontsize=8, color=INK, loc="left")
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}")
for proto in ("T1", "T2"):
    print(proto, {c: round(med_ci(ratios[(proto, c)])[0], 3) for c in CRIT})
