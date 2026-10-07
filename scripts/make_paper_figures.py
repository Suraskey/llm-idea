"""Generate the paper's figures from the recorded results. $0, deterministic.

All numbers are transcribed from the committed run records (EXPERIMENTS.md +
results.tsv) and the deterministic analyses; this script only renders them. Run:
  poetry run python scripts/make_paper_figures.py
Outputs PDF + PNG to figures/.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.splines.top" if False else "axes.grid": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.linewidth": 0.8,
})
INK = "#1a1a1a"
ACC = "#2b6cb0"
WARN = "#c05621"
MUTE = "#a0aec0"


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf/.png")


# --- Fig 0: the loop closes, both ways (EXP-078 live trace) --------------------
def fig_loop_closure_trace():
    """The annotated trace the paper's Results section opens with. Numbers are
    transcribed from records/EXP-078/loop_closure_alien_LIVE.md
    (EXP-078: 5 live director calls; family 3/5 -> SOLVED, new IC 2/5 -> EXHAUSTED)."""
    from matplotlib.patches import FancyBboxPatch

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9))
    LLM = WARN
    DET = ACC
    GOOD = "#2f855a"

    def box(ax, y, text, color, bold=False):
        ax.add_patch(FancyBboxPatch((0.04, y - 0.075), 0.92, 0.15,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc="white", ec=color, lw=1.4))
        ax.text(0.5, y, text, ha="center", va="center", fontsize=7,
                color=INK, fontweight="bold" if bold else "normal", wrap=True)

    def arrow(ax, y0, y1):
        ax.annotate("", xy=(0.5, y1 + 0.078), xytext=(0.5, y0 - 0.078),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))

    # left: the family choice -> SOLVED
    ax = axes[0]
    ax.set_title("director chose a multi-charge family (3 of 5 calls)", fontsize=8)
    steps = [
        ("Round 0: diagnose\nrank 1/2, NON-IDENTIFIABLE\nblind along [0.71, -0.71] over (s1, s2)", DET),
        ("live director (the only model call)\nmove: request_multi_charge_family", LLM),
        ("OED ranker (deterministic)\ncandidates: single product / 2 distinct / 4 distinct\nchosen: vary_family_2_distinct", DET),
        ("Round 1: re-diagnose\nrank 2/2, IDENTIFIABLE", DET),
        ("SOLVED in one design round", GOOD),
    ]
    ys = np.linspace(0.9, 0.1, len(steps))
    for (txt, c), y in zip(steps, ys, strict=True):
        box(ax, y, txt, c, bold=(c == GOOD))
    for y0, y1 in zip(ys[:-1], ys[1:], strict=True):
        arrow(ax, y0, y1)

    # right: the new-IC choice -> EXHAUSTED
    ax = axes[1]
    ax.set_title("director chose a new initial condition (2 of 5 calls)", fontsize=8)
    steps = [
        ("Round 0: diagnose\nrank 1/2, NON-IDENTIFIABLE\nsame blind direction", DET),
        ("live director (the only model call)\nmove: request_new_initial_condition", LLM),
        ("OED ranker (deterministic)\nonly candidate: new IC, still a single product\ncannot restore rank", DET),
        ("Rounds 1, 2, 3: re-diagnose\nrank 1/2 each time", DET),
        ("EXHAUSTED after 3 rounds:\nreport the identifiable combination, not its factors", WARN),
    ]
    for i, ((txt, c), y) in enumerate(zip(steps, ys, strict=True)):
        box(ax, y, txt, c, bold=(i == len(steps) - 1))
    for y0, y1 in zip(ys[:-1], ys[1:], strict=True):
        arrow(ax, y0, y1)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.text(0.5, 0.005,
             "orange border = the one step that calls a language model; blue = deterministic numpy/scipy; "
             "same engine, same plateau, adjudicated both ways ($0.11 total, EXP-078)",
             ha="center", fontsize=6.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig0-loop-closure-trace")


# --- Fig 1: observability is the lever (EXP-076 treatment vs control) ---------
def fig_observability_lift():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    arms = ["control\n(single config)", "treatment\n(multi-charge family)"]
    depth_ge3 = [1, 8]  # /8 seeds reaching consec depth >=3 (EXP-076)
    bars = ax.bar(arms, depth_ge3, color=[MUTE, ACC], width=0.6)
    ax.set_ylabel("seeds reaching depth ≥3  (of 8)")
    ax.set_ylim(0, 8.6)
    ax.set_title("Observability lift (EXP-076)", fontsize=9)
    for b, v in zip(bars, depth_ge3, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v), ha="center", fontsize=9, color=INK)
    ax.text(0.03, 0.94, "Wilcoxon p=0.0078, Cliff's δ=0.906", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color=INK)
    fig.tight_layout()
    _save(fig, "fig1-observability-lift")


# --- Fig 2: the load-bearing gate is instrument-independent (EXP-080/081/083b)
def fig_transfer_l3_stable():
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    groups = ["L3 sum-gate\n(load-bearing)", "L2 band\n(construct-caveated)", "depth ≥3"]
    x = np.arange(len(groups))
    w = 0.26
    ok = "#2f855a"
    exp080 = [7, 6, 5]  # /8 (EXPERIMENTS.md EXP-080)
    exp081 = [7, 4, 3]  # /8 (EXP-081)
    exp083b = [6, 2, 2]  # /8 (EXP-083b, clean-weather both-channels-corrected)
    b1 = ax.bar(x - w, exp080, w, label="EXP-080 (poisoned instrument)", color=MUTE)
    b2 = ax.bar(x, exp081, w, label="EXP-081 (battery corrected)", color=ACC)
    b3 = ax.bar(x + w, exp083b, w, label="EXP-083b (both channels corrected)", color=ok)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("seeds passing (of 8)")
    ax.set_ylim(0, 9.4)
    ax.set_title("The load-bearing gate is instrument-independent", fontsize=9)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                    str(int(b.get_height())), ha="center", fontsize=8, color=INK)
    # annotate the stable L3 against the registered bar
    ax.annotate("7, 7, 6: holds at the registered bar (≥6/8)",
                xy=(0, 7), xytext=(0.12, 8.55), ha="center", fontsize=7, color=WARN)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig2-transfer-l3-stable")


# --- Fig 3: the diagnostic on its own metric — OED ranks the scorer last ------
def fig_oed_self_diagnosis():
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    # indifference-band widths over e for the four designs (from the diagnostic run)
    designs = ["D1 orbit\n(the scorer's set)", "D3 small-r", "D2 battery", "D4 orbit+battery"]
    bands = [(2.05, 2.40), (2.05, 6.00), (2.05, 4.55), (3.15, 5.50)]  # (lo, hi) e-indiff
    oed_rank = [4, 3, 2, 1]  # OED ranking (1=best)
    colors = [WARN if r == 4 else (ACC if r == 1 else MUTE) for r in oed_rank]
    y = np.arange(len(designs))[::-1]
    for yi, (lo, hi), c in zip(y, bands, colors, strict=False):
        ax.plot([lo, hi], [yi, yi], color=c, lw=5, solid_capstyle="round")
    # the gate band and the two contested exponents
    ax.axvspan(3.2, 3.8, color="#000", alpha=0.06)
    ax.axvline(3.0, color=INK, ls=":", lw=0.9)
    ax.axvline(3.5, color=INK, ls="--", lw=0.9)
    ax.text(3.0, len(designs) - 0.4, "r⁻³", fontsize=7, ha="center", color=INK)
    ax.text(3.5, len(designs) - 0.4, "r⁻³·⁵", fontsize=7, ha="center", color=INK)
    ax.text(3.5, -0.7, "gate band [3.2,3.8]", fontsize=6.5, ha="center", color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}\n(OED #{r})" for d, r in zip(designs, oed_rank, strict=False)], fontsize=7)
    ax.set_xlabel("correction-exponent indifference band (red ≥ 0.95·max)")
    ax.set_xlim(1.9, 6.1)
    ax.set_ylim(-1.3, len(designs) + 0.2)
    ax.set_title("The engine ranks its own benchmark's design last", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig3-oed-self-diagnosis")


# --- Fig 4: feedback-dose response (terminal-exponent accuracy) ---------------
def fig_dose_response():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    # in-regime terminal fraction vs feedback-delivery dose (V1 cells, truth 3.5)
    cells = ["N=15\n(≈0 deliv.)", "N=40\n(1 deliv.)", "EXP-075\n(many)"]
    frac = [0 / 8, 4 / 8, 7 / 8]
    ax.plot(range(3), frac, "-o", color=ACC, lw=1.6, markersize=6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cells, fontsize=7.5)
    ax.set_ylabel("Council seeds with in-regime\nterminal exponent (of 8)")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "2", "4", "6", "8"])
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("Terminal accuracy tracks instrument dose", fontsize=9)
    for xi, f in zip(range(3), frac, strict=False):
        ax.text(xi, f + 0.05, f"{int(f*8)}/8", ha="center", fontsize=8, color=INK)
    fig.tight_layout()
    _save(fig, "fig4-dose-response")


if __name__ == "__main__":
    print("Rendering paper figures...")
    fig_loop_closure_trace()
    fig_observability_lift()
    fig_transfer_l3_stable()
    fig_oed_self_diagnosis()
    fig_dose_response()
    print("done.")
