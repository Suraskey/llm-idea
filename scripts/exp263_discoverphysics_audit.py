#!/usr/bin/env python3
"""EXP-263 — identifiability audit of the DiscoverPhysics benchmark (pre-registered).

Pre-registration: ``predictions.md`` block "Pre-registered block added 2026-09-28
(Session 047 — EXP-263 ...)" (commit 8f746eb). This script implements that procedure
mechanically. No LLM is called anywhere.

Pinned inputs
  * DiscoverPhysics 33b7fa9 at external/DiscoverPhysics (or $DISCOVERPHYSICS_DIR) (engine='nbody', the
    benchmark's own JAX N-body simulator; noise_frac 5%).
  * DiscoverPhysicsLeaderboard 8e9c858 at external/DiscoverPhysicsLeaderboard (or $DISCOVERPHYSICS_LEADERBOARD_DIR).

Two simulation paths, both built from the benchmark's own code:
  * EXECUTOR PATH — ``get_world(name, engine='nbody')['executor'].run([...])`` with the
    world constants patched on the executor (operator params / class attributes).
    Used for every TRUE-world prediction and for the registered central-difference
    sensitivities.
  * FAST PATH — the same simulator internals (``make_acceleration_fn``, ``_make_step``
    from ``physchool.worlds.nbody_sampler``, the ``force_laws`` kernels, the same
    lax.scan/fori_loop recording) re-orchestrated with parameters as traced JAX
    arguments so that alternative hypotheses can be fitted with exact Jacobians and
    vmapped over designs. Parity against the executor path is checked on all 65
    designs of every world and reported (parity.tsv).

Stages: 0 leaderboard copy (first) -> 1 claims -> 2 designs -> per world: parity, FD
sensitivities, CRLBs, ranker BEST, mechanism fits -> dark-matter count/coupling fits ->
sympy certificates -> verdicts, prediction statistics, README.

Usage
  python scripts/exp263_discoverphysics_audit.py --stage leaderboard
  python scripts/exp263_discoverphysics_audit.py --smoke --worlds gravity   # code paths only
  python scripts/exp263_discoverphysics_audit.py --stage worlds --jobs 4
  python scripts/exp263_discoverphysics_audit.py --stage aggregate
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "exp263"
CACHE = OUT / "cache"
LB = Path(os.environ.get("DISCOVERPHYSICS_LEADERBOARD_DIR", REPO / "external" / "DiscoverPhysicsLeaderboard"))  # pinned 8e9c858
DP = Path(os.environ.get("DISCOVERPHYSICS_DIR", REPO / "external" / "DiscoverPhysics"))  # pinned 33b7fa9
for p in (DP / "ScienceAgent", DP / "PhysicsSchool", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PUBLIC_WORLDS = ["gravity", "yukawa", "hubble", "ether", "oscillator", "circle",
                 "extra_dimensions", "fractional", "dark_matter", "three_species", "coulomb"]
CODE = {w: w for w in PUBLIC_WORLDS}
CODE["coulomb"] = "coulomb_easy"
TWO_P = {"gravity", "yukawa", "fractional", "oscillator", "extra_dimensions", "coulomb"}
PROBES = {"dark_matter", "three_species"}
PROBES_MASS = {"ether", "hubble"}

SEED = 263
N_SAMPLED = 64
NOISE_FRAC = 0.05
N_ROUNDS = 16
DT = 0.005
SOFT = 0.05
INTEG = "yoshida4"
KMAX = 10
FD_STEPS = (1e-4, 1e-3)          # registered relative step and the stability check
RESID_DIST = 1.0                 # residual (RMS / sigma) above the 5% noise level
RESID_EQUIV = 0.01               # 1% of the noise level
EXPECTED_DP = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
EXPECTED_LB = "8e9c858a95f9282717771e52bab30c45cf253259"

THETA = {
    "gravity": [("G", 1.0)],
    "yukawa": [("G", 1.0), ("lambda", 2.0)],
    "fractional": [("G", 1.0), ("alpha", 0.5)],
    "circle": [("G", 1.0), ("alpha", 0.75)],
    "oscillator": [("G0", 5.0), ("omega", math.pi / 2.0), ("phi", 0.0)],
    "extra_dimensions": [("G", 1.0), ("R_c", 0.5)],
    "coulomb": [("k", 1.0)],
    "ether": [("Q", 50.0), ("alpha", 0.05)],
    "hubble": [("Q", 50.0), ("H", 0.05)],
    "three_species": [("s_A", 1.0), ("s_B", 3.0), ("s_C", -2.0)],
    "dark_matter": [("s_vis", 1.0), ("s_dark", 5.0)],
}


# ════════════════════════════════════════════════════════════════════════════
# small utilities
# ════════════════════════════════════════════════════════════════════════════

def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return "inf" if o > 0 else ("-inf" if o < 0 else "nan")
    return o


def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1))


def fnum(x):
    if isinstance(x, str):
        return float(x)
    return float(x)


def world_vars() -> dict:
    """Parse _WORLD_VARS from the pinned run_benchmark.py (noise_frac definition)."""
    src = (DP / "scripts" / "run_benchmark.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.AnnAssign, ast.Assign)):
            tgt = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if getattr(tgt, "id", None) == "_WORLD_VARS":
                return ast.literal_eval(node.value)
    raise RuntimeError("_WORLD_VARS not found")


def sigma_of(world: str) -> float:
    """run_benchmark.py: absolute sigma = noise_frac * sqrt(_WORLD_VARS[world])."""
    return NOISE_FRAC * math.sqrt(world_vars()[CODE[world]])


# ════════════════════════════════════════════════════════════════════════════
# Stage 0 — leaderboard copy (runs FIRST, before any verdict is computed)
# ════════════════════════════════════════════════════════════════════════════

def copy_leaderboard() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    head = git_head(LB)
    data = json.loads((LB / "data" / "results.json").read_text())
    rows = []
    for r in data["results"]:
        for w in PUBLIC_WORLDS:
            pw = r["per_world"][w]
            rows.append((r["model"], w, pw["n"], pw["explanation_score"]["mean"],
                         pw["explanation_score"].get("se"), pw["geom_pos_err"]["mean"],
                         pw["geom_pos_err"].get("up"), pw["geom_pos_err"].get("down")))
    p = OUT / "leaderboard_cells.tsv"
    with p.open("w") as f:
        f.write(f"# source: DiscoverPhysicsLeaderboard data/results.json @ {head}\n")
        f.write(f"# benchmark_version: {data['metadata'].get('benchmark_version')}; "
                f"last_updated: {data['metadata'].get('last_updated')}\n")
        f.write("model\tworld\tn\texpl_mean\texpl_se\tgeom_pos_err_mean\tgeom_up\tgeom_down\n")
        for row in rows:
            f.write("\t".join("" if v is None else str(v) for v in row) + "\n")
    return p


def read_leaderboard() -> list[dict]:
    rows = []
    with (OUT / "leaderboard_cells.tsv").open() as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    hdr = lines[0].split("\t")
    for ln in lines[1:]:
        d = dict(zip(hdr, ln.split("\t")))
        rows.append({"model": d["model"], "world": d["world"], "n": int(d["n"]),
                     "expl": float(d["expl_mean"]), "geom": float(d["geom_pos_err_mean"])})
    return rows


# ════════════════════════════════════════════════════════════════════════════
# Stage 1 — graded claims (top band, verbatim fragments) and their typing
# ════════════════════════════════════════════════════════════════════════════
# Typing rules (fixed before any computation; see README "Implementation choices"):
#  PARAMETER  — a numeric constant with a stated tolerance/range in the top band.
#  MECHANISM  — a structural top-band assertion for which a LOWER band names a specific,
#               implementable alternative law/structure. Every such named alternative is
#               tested; the claim's class is the LEAST distinguishable one.
#  UNTYPED    — the lower-band counterpart is only an omission/vagueness ("omits",
#               "fails to characterise", generic "wrong operator family"), or no lower
#               band names an alternative. Listed, not scored.
#  NOT_SCORED — the named alternative (time-evolving / wave-like field) cannot be
#               represented by the pinned nbody engine. Listed, not scored.
#  NOTE       — grader convention without a tolerance (oscillator amplitude note).

def P(cid, frag, q, lo, hi, truth, kind, **kw):
    return dict(id=cid, frag=frag, type="PARAMETER", q=q, lo=lo, hi=hi, truth=truth, kind=kind, **kw)


def M(cid, frag, alts, **kw):
    return dict(id=cid, frag=frag, type="MECHANISM", alts=alts, **kw)


def U(cid, frag, why, typ="UNTYPED"):
    return dict(id=cid, frag=frag, type=typ, why=why)


STATIC_WHY = ("named alternative is a time-evolving / wave-like field; the pinned nbody engine "
              "has only instantaneous pairwise forces, so it cannot be simulated here")

CLAIMS = {
    "gravity": [
        U("GR-1", "Identifies a static scalar field", STATIC_WHY, "NOT_SCORED"),
        M("GR-2", "with a Laplacian/Poisson operator (∇²φ = source); states the resulting attractive "
                  "force falls off as 1/r (not 1/r²) in 2D, or equivalently that the Green's function "
                  "is logarithmic",
          [("7–9", "asserts the 3D-Newtonian 1/r² falloff", "inv2"),
           ("4–6", "commits to a clearly wrong decay law (exponential, inverse-cube)", "exp"),
           ("4–6", "commits to a clearly wrong decay law (exponential, inverse-cube)", "inv3"),
           ("1–3", "a qualitatively wrong force law (repulsive, constant)", "const")]),
        M("GR-3", "correctly distinguishes p1 as source coupling and p2 as particle inertia",
          [("7–9", "muddles the p1/p2 roles", "swap"),
           ("4–6", "fails to distinguish source coupling from inertia", "product")]),
    ],
    "yukawa": [
        U("YU-1", "Identifies a static", STATIC_WHY, "NOT_SCORED"),
        M("YU-2", "screened / Helmholtz operator (∇²φ − φ/λ² = source, or equivalent Yukawa form); "
                  "names or describes exponential suppression at long range",
          [("4–6", "treats it as a plain Laplacian or generic power-law and misses the screening "
                   "structure", "laplace"),
           ("4–6", "treats it as a plain Laplacian or generic power-law and misses the screening "
                   "structure", "pow")]),
        P("YU-3", "with screening length within roughly 1–4 (ground truth λ = 2)",
          "log_lambda", 1.0, 4.0, 2.0, "mult"),
        M("YU-4", "correctly identifies p1 as source coupling and p2 as inertia",
          [("7–9", "muddles the p1/p2 roles", "swap")]),
    ],
    "fractional": [
        U("FR-1", "Identifies a static", STATIC_WHY, "NOT_SCORED"),
        M("FR-2", "non-local / fractional-Laplacian operator of the form −(−∇²)^α",
          [("4–6", "assumes a standard Laplacian or ordinary power-law, missing the non-local / "
                   "fractional character", "laplace"),
           ("4–6", "assumes a standard Laplacian or ordinary power-law, missing the non-local / "
                   "fractional character", "pow")], certify="pow"),
        P("FR-3", "with α in a plausible range (roughly 0.3–0.8; ground truth α = 0.5)",
          "alpha", 0.3, 0.8, 0.5, "add"),
        M("FR-4", "describes the force as decaying more slowly with distance than the standard 2D "
                  "Laplacian case (enhanced long-range interaction)",
          [("7–9", "articulates the direction of the anomaly vaguely or incorrectly", "pow_fast")],
          claim_family="pow_slow"),
        U("FR-5", "correctly names p1 as source and p2 as inertia",
          "no lower band of the fractional rubric names a p1/p2 alternative"),
    ],
    "circle": [
        U("CI-1", "Identifies a static", STATIC_WHY, "NOT_SCORED"),
        M("CI-2", "non-local / fractional-Laplacian operator −(−∇²)^α",
          [("7–9", "Identifies a non-local / anomalous-decay operator", "pow"),
           ("4–6", "assumes a standard Laplacian (or Newtonian 1/r) and misses the fractional "
                   "character", "laplace")], certify="pow"),
        P("CI-3", "with α in a plausible range (roughly 0.5–1.0; ground truth α = 0.75)",
          "alpha", 0.5, 1.0, 0.75, "add"),
        U("CI-4", "states that the coupling is uniform across all 11 particles",
          "lower band only says 'omits the uniform-coupling claim' (an omission, no alternative law)"),
        M("CI-5", "describes force behaviour intermediate between the logarithmic 2D Laplacian case "
                  "and pure long-range attraction",
          [("7–9", "muddles the direction of the anomaly", "pow_fast")], claim_family="pow_slow"),
    ],
    "oscillator": [
        U("OS-1", "Identifies the standard 2D Laplacian spatial form (∇²φ = source, force ∝ 1/r in 2D)",
          "no lower band names an alternative spatial form (the 4–6 band keeps the 1/r form)"),
        M("OS-2", "with a *time-modulated overall coupling* G(t) — i.e. the strength (and SIGN) of the "
                  "interaction depends on absolute time, not just on geometry",
          [("4–6", "treats the coupling as constant (missing the entire time modulation)",
            "const_coupling")]),
        U("OS-3", "explicitly notes that the same configuration is attractive at some phases and "
                  "*repulsive* at others",
          "lower band only says 'omits the sign-flipping behaviour' (an omission)"),
        M("OS-4", "identifies a sinusoidal modulation G(t) = G₀·cos(ω t + φ) (or sin equivalent)",
          [("4–6", "treats the variation as transient / decaying or otherwise non-periodic",
            "transient")]),
        P("OS-5", "estimates the period within roughly a factor of 2 of T = 4",
          "log_T", 2.0, 8.0, 4.0, "mult"),
        M("OS-6", "correctly keeps p1 as source coupling and p2 as inertia",
          [("7–9", "muddles the p1/p2 roles", "swap")]),
        dict(id="OS-N", frag="the amplitude can be reported either as G₀ ≈ 5", type="NOTE",
             why="grader convention; no tolerance stated (CRLB of G0 reported for information)"),
    ],
    "extra_dimensions": [
        M("ED-1", "Identifies that the visible 2D world has *one* extra spatial dimension "
                  "compactified at radius R; states the force is the Kaluza-Klein image sum over the "
                  "compact dimension, ∝ Σ_n r/(r²+(nL)²)^(3/2) with L = 2π R",
          [("7–9", "misses the geometric image-sum structure", "crossover"),
           ("4–6", "proposes a wrong functional form (e.g. Yukawa screening, fractional Laplacian, "
                   "pure power law of intermediate exponent)", "yukawa"),
           ("4–6", "proposes a wrong functional form (e.g. Yukawa screening, fractional Laplacian, "
                   "pure power law of intermediate exponent)", "pow"),
           ("1–3", "Fits the data with a single canonical law (plain 2D gravity, plain 3D gravity, "
                   "fractional, Yukawa, …)", "laplace"),
           ("1–3", "Fits the data with a single canonical law (plain 2D gravity, plain 3D gravity, "
                   "fractional, Yukawa, …)", "inv2")]),
        M("ED-2", "correctly identifies the two asymptotic regimes (r ≫ R: 2D Poisson F ∝ p1/(2π r); "
                  "r ≲ R: 3D Newtonian F ∝ p1/(4π r²) becoming exponentially divergent at r → 0)",
          [("4–6", "proposes a wrong functional form (e.g. Yukawa screening, fractional Laplacian, "
                   "pure power law of intermediate exponent)", "same-as-ED-1")], shared="ED-1"),
        P("ED-3", "estimates R within roughly a factor of 2 of 0.5 (i.e. R ∈ [0.2, 1.5])",
          "log_R_c", 0.2, 1.5, 0.5, "mult"),
        P("ED-4", "gets the long-range coupling within ~30 % of G = 1", "log_G", 0.7, 1.3, 1.0, "mult"),
        U("ED-5", "correctly assigns p1 as source coupling and p2 as inertia",
          "no lower band of the extra_dimensions rubric names a p1/p2 alternative"),
    ],
    "coulomb": [
        U("CO-1", "Identifies a static", "named alternative (time-evolving, wave-like): the pinned nbody engine "
          "has only instantaneous pairwise forces, so it cannot be simulated here", "NOT_SCORED"),
        M("CO-2", "central pairwise attractive force with 1/r² (inverse-square) falloff",
          [("4–6", "identifies a clearly wrong falloff (1/r, exponential, constant)", "inv1"),
           ("4–6", "identifies a clearly wrong falloff (1/r, exponential, constant)", "exp"),
           ("4–6", "identifies a clearly wrong falloff (1/r, exponential, constant)", "const")]),
        M("CO-3", "correctly identifies the role of p1 and p2 as charges (or charge magnitudes) "
                  "entering the force as a product (F ∝ p1 · p2 / r²)",
          [("7–9", "claims one of them is inertia", "ratio_12"),
           ("7–9", "claims one of them is inertia", "ratio_21"),
           ("7–9", "that the force depends additively on them", "additive")]),
        U("CO-4", "states that particle 1 is held fixed and that particle 2's inertia is 1",
          "no lower band names an alternative"),
        P("CO-5", "estimates the coupling/strength constant within roughly a factor of 2 of 1",
          "log_k", 0.5, 2.0, 1.0, "mult"),
    ],
    "dark_matter": [
        U("DM-1", "Identifies a static Laplacian with force −∇φ",
          "lower band names only a generic 'wrong operator family'"),
        M("DM-2", "concludes that hidden / unseen sources exist based on visible particles "
                  "accelerating toward apparently empty regions",
          [("4–6", "attributes the visible particles' anomalous behaviour to noise, the probes, or "
                   "measurement error rather than to hidden sources", "no_hidden")]),
        M("DM-3", "estimates the hidden population roughly correctly (count in 5–15; ground truth 10) "
                  "with coupling stronger than the visible population (roughly 3–8×; ground truth 5×)",
          [("7–9", "gets their count or coupling strength badly wrong", "count5"),
           ("7–9", "gets their count or coupling strength badly wrong", "count20")], special="dm_count"),
        P("DM-3b", "with coupling stronger than the visible population (roughly 3–8×; ground truth 5×)",
          "log_sdark_over_svis", 3.0, 8.0, 5.0, "mult",
          conditional="conditional on the true count (10) and true dark positions"),
        U("DM-4", "identifies the probes (agent indices 20–24) as neutral (non-sourcing but responsive)",
          "lower band only says 'fails to characterise the probes as neutral' (an omission)"),
    ],
    "three_species": [
        U("TS-1", "Identifies a static Laplacian with force −∇φ",
          "lower band names only a generic 'wrong operator family'"),
        M("TS-2", "identifies three distinct source species among particles 0–29",
          [("7–9", "the particle-index partitioning is slightly wrong", "reassign_one"),
           ("4–6", "finds only two species", "merge_AB"),
           ("4–6", "finds only two species", "merge_AC"),
           ("4–6", "finds only two species", "merge_BC"),
           ("1–3", "Treats all particles as identical", "all_identical")]),
        M("TS-3", "correctly identifies one species as repulsive (negative coupling)",
          [("4–6", "misses the repulsive species (all three treated as attractive)", "nonneg")]),
        P("TS-4a", "estimates coupling ratios approximately matching +1 : +3 : −2 (each within roughly "
                   "a factor of 2, with signs correct)", "log_sB_over_sA", 1.5, 6.0, 3.0, "mult"),
        P("TS-4b", "estimates coupling ratios approximately matching +1 : +3 : −2 (each within roughly "
                   "a factor of 2, with signs correct)", "log_abs_sC_over_sA", 1.0, 4.0, 2.0, "mult"),
        M("TS-5", "identifies the probes as having approximately zero source coupling",
          [("7–9", "probes are lumped with one of the species", "probes_A"),
           ("7–9", "probes are lumped with one of the species", "probes_B"),
           ("7–9", "probes are lumped with one of the species", "probes_C")]),
    ],
    "ether": [
        U("ET-1", "Identifies a static 2D Laplacian central force sourced by the single anchor particle "
                  "(index 0)", "lower bands only 'fails to identify which particle is the anchor' / "
                               "generic 'wrong operator family'"),
        U("ET-2", "identifies the 20 orbiters and 5 probes as test particles responding to that field",
          "no lower band names an alternative"),
        M("ET-3", "identifies a uniform northward drift acceleration on every particle (or, "
                  "equivalently, a body-force proportional to mass producing a mass-independent "
                  "acceleration)",
          [("7–9", "misses the F ∝ m / a = const equivalence", "mass_dep_drift"),
           ("4–6", "mis-attributes the drift to a directional Laplacian / wind / repulsion between "
                   "specific particles", "repulsor")]),
        P("ET-4", "estimates the drift acceleration α within roughly a factor of 2 of 0.05",
          "log_alpha", 0.025, 0.1, 0.05, "mult"),
        M("ET-5", "recognises that orbiter masses ∈ {1, 2, 4} but that with the mass-proportional "
                  "ether force the drift looks identical for all particles in absolute coordinates",
          [("7–9", "misses the F ∝ m / a = const equivalence", "mass_dep_drift")], shared="ET-3"),
    ],
    "hubble": [
        U("HU-1", "Identifies a static 2D Laplacian central force sourced by the single anchor particle "
                  "(index 0)", "lower band names only a generic 'wrong operator family'"),
        U("HU-2", "identifies the 20 orbiters and 5 probes as test particles",
          "no lower band names an alternative"),
        M("HU-3", "identifies a *position-dependent* outward body-force that grows linearly with "
                  "distance from the anchor (a = H · r, mass-independent)",
          [("7–9", "misses the linear-in-r structure (e.g. assumes a constant outward force)",
            "const_push"),
           ("7–9", "attributes the outward push to a particular particle rather than to space itself",
            "central_only")]),
        U("HU-4", "recognises a critical radius beyond which probes accelerate outward and orbits unbind",
          "consequence of HU-3/HU-5; the lower band is a vagueness demotion, no alternative law"),
        P("HU-5", "estimates H within roughly a factor of 2 of 0.05", "log_H", 0.025, 0.1, 0.05, "mult"),
    ],
}


def rubric_texts():
    from scienceagent.worlds import WORLDS
    out = {}
    for w in PUBLIC_WORLDS:
        rub = WORLDS[CODE[w]]["explanation_rubric"]
        top = rub.split("\n 7")[0]
        out[w] = (re.sub(r"\s+", " ", top).strip(), re.sub(r"\s+", " ", rub).strip())
    return out


def tolerance_thresholds(c):
    """tau (registered reading) and tau_halflog (alternative reading), both reported."""
    lo, hi, t = c["lo"], c["hi"], c["truth"]
    if c["kind"] == "add":
        tau = 0.5 * min(t - lo, hi - t)
        return tau, tau, f"[{lo}, {hi}] absolute; tau = half the distance to the nearer bound"
    tau = 0.5 * min(1.0 - lo / t, hi / t - 1.0)
    tau_hl = 0.5 * min(math.log(t / lo), math.log(hi / t))
    return tau, tau_hl, (f"[{lo:g}, {hi:g}] around {t:g}; SE in log units; tau = 0.5*min(1-lo/t, hi/t-1) "
                         f"(=0.25 for x2, the registered example); half-log reading = {tau_hl:.3f}")


def write_claims_tsv():
    tops = rubric_texts()
    rows = []
    for w in PUBLIC_WORLDS:
        top, full = tops[w]
        for c in CLAIMS[w]:
            assert c["frag"] in top, (w, c["id"], "fragment not verbatim in top band")
            alts = ""
            if c["type"] == "MECHANISM":
                for band, txt, impl in c["alts"]:
                    assert txt in full, (w, c["id"], "alternative text not verbatim in rubric", txt)
                alts = " | ".join(f"[{b}] \"{t}\" -> {i}" for b, t, i in c["alts"])
            tol = ""
            if c["type"] == "PARAMETER":
                tau, tau_hl, desc = tolerance_thresholds(c)
                tol = f"{desc}; tau={tau:.3f}"
            rows.append((w, c["id"], c["type"], c["frag"], alts, tol,
                         c.get("why", c.get("conditional", ""))))
    p = OUT / "claims.tsv"
    with p.open("w") as f:
        f.write("world\tclaim_id\ttype\ttop_band_text_verbatim\tnamed_alternatives[band]\ttolerance\tnote\n")
        for r in rows:
            f.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")
    return p


# ════════════════════════════════════════════════════════════════════════════
# Stage 2 — designs (DEFAULT + 64 interface-legal designs, seed 263)
# ════════════════════════════════════════════════════════════════════════════

def _parse_format(fmt: str):
    body = fmt.split("<run_experiment>")[1].split("</run_experiment>")[0]
    return json.loads(body)[0]


def default_design(world):
    from scienceagent.worlds import WORLDS
    from scienceagent.evaluator import _DEFAULT_TEST_CASES
    entry = WORLDS[CODE[world]]
    if "experiment_format" in entry:
        d = dict(_parse_format(entry["experiment_format"]))
        src = "world experiment_format"
    else:
        d = json.loads(json.dumps(_DEFAULT_TEST_CASES[0]))
        src = "evaluator _DEFAULT_TEST_CASES[0] (world has no experiment_format)"
    floor = 10.0 if world == "dark_matter" else 5.0
    d["duration"] = max(float(max(d["measurement_times"])), floor)   # what the executor computes
    if world == "oscillator":
        d["start_time"] = 0.0
    d["label"] = "DEFAULT"
    d["source"] = src
    return d


def sample_designs(world, n=N_SAMPLED):
    rng = np.random.default_rng(SEED)
    out = []
    for i in range(n):
        if world in TWO_P:
            p1, p2 = rng.uniform(0.1, 10.0, 2)
            pos2 = rng.uniform(-10.0, 10.0, 2)
            v2 = rng.uniform(-5.0, 5.0, 2)
            dur = rng.uniform(5.0, 10.0)
            d = {"p1": float(p1), "p2": float(p2), "pos2": pos2.tolist(), "velocity2": v2.tolist()}
            if world == "oscillator":
                d["start_time"] = float(rng.uniform(0.0, 10.0))
        elif world in PROBES:
            dur = 10.0
            d = {"probe_positions": rng.uniform(-15.0, 15.0, (5, 2)).tolist(),
                 "probe_velocities": rng.uniform(-2.0, 2.0, (5, 2)).tolist()}
        elif world in PROBES_MASS:
            dur = 10.0
            d = {"probe_positions": rng.uniform(-22.0, 22.0, (5, 2)).tolist(),
                 "probe_velocities": rng.uniform(-3.0, 3.0, (5, 2)).tolist(),
                 "probe_masses": rng.choice([1.0, 2.0, 4.0], 5).tolist()}
        elif world == "circle":
            dur = 10.0
            d = {"ring_radius": float(rng.uniform(2.0, 10.0)),
                 "initial_tangential_velocity": float(rng.uniform(0.0, 2.0))}
        else:
            raise ValueError(world)
        d["duration"] = float(dur)
        d["measurement_times"] = [float(dur * k / 10.0) for k in range(1, 11)]
        d["label"] = f"S{i:02d}"
        out.append(d)
    return out


def all_designs(world, n_sampled=N_SAMPLED):
    return [default_design(world)] + sample_designs(world, n_sampled)


EXP_KEYS = {"p1", "p2", "pos2", "velocity2", "measurement_times", "duration", "start_time",
            "probe_positions", "probe_velocities", "probe_masses", "ring_radius",
            "initial_tangential_velocity", "visible_velocity_sign"}


def exp_of(design):
    return {k: v for k, v in design.items() if k in EXP_KEYS}


# ════════════════════════════════════════════════════════════════════════════
# Executor path (the benchmark's own simulator, constants patched on the executor)
# ════════════════════════════════════════════════════════════════════════════

def build_executor(world, th):
    from scienceagent.worlds import get_world
    from scienceagent import executor as EX
    code = CODE[world]
    th = [float(x) for x in th]
    if world == "gravity":
        return get_world(code, engine="nbody",
                         operators=[{"type": "laplacian", "params": {"strength": th[0]}}])["executor"]
    if world == "yukawa":
        return get_world(code, engine="nbody", operators=[{"type": "screening", "params": {
            "strength": th[0], "screening_length": th[1]}}])["executor"]
    if world in ("fractional", "circle"):
        return get_world(code, engine="nbody", operators=[{"type": "fractional_laplacian", "params": {
            "strength": th[0], "alpha": th[1]}}])["executor"]
    if world == "coulomb":
        return get_world(code, engine="nbody",
                         operators=[{"type": "coulomb", "params": {"strength": th[0]}}])["executor"]
    if world == "extra_dimensions":
        cls = type("PatchedExtraDimensions", (EX.NBodyExtraDimensionsExecutor,),
                   {"G": th[0], "R_COMPACT": th[1]})
        return cls()
    ex = get_world(code, engine="nbody")["executor"]
    if world == "oscillator":
        ex.G_0, ex.OMEGA, ex.PHI = th
    elif world == "ether":
        ex.ANCHOR_SOURCE, ex.ETHER_ALPHA = th
    elif world == "hubble":
        ex.ANCHOR_SOURCE, ex.HUBBLE_H = th
    elif world == "three_species":
        ex.SOURCE_A, ex.SOURCE_B, ex.SOURCE_C = th
    elif world == "dark_matter":
        ex.SOURCE_VISIBLE, ex.SOURCE_DARK = th
    else:
        raise ValueError(world)
    return ex


def result_to_obs(world, r):
    if world in TWO_P:
        return np.stack([np.asarray(r["pos1"], float), np.asarray(r["pos2"], float)], axis=1)
    return np.asarray(r["positions"], float)


def executor_obs(ex, world, designs):
    with ex.noise_disabled():
        res = ex.run([exp_of(d) for d in designs])
    return [result_to_obs(world, r) for r in res]


def check_truth_constants():
    from scienceagent.worlds import get_world
    from scienceagent import executor as EX
    got = {}
    for w in PUBLIC_WORLDS:
        ex = get_world(CODE[w], engine="nbody")["executor"]
        if w in ("gravity", "yukawa", "fractional", "circle", "coulomb"):
            prm = ex.operators[0]["params"]
            vals = {"gravity": [prm["strength"]], "coulomb": [prm["strength"]],
                    "yukawa": [prm["strength"], prm.get("screening_length")],
                    "fractional": [prm["strength"], prm.get("alpha")],
                    "circle": [prm["strength"], prm.get("alpha")]}[w]
        elif w == "extra_dimensions":
            vals = [EX.NBodyExtraDimensionsExecutor.G, EX.NBodyExtraDimensionsExecutor.R_COMPACT]
        elif w == "oscillator":
            vals = [ex.G_0, ex.OMEGA, ex.PHI]
        elif w in ("ether",):
            vals = [ex.ANCHOR_SOURCE, ex.ETHER_ALPHA]
        elif w == "hubble":
            vals = [ex.ANCHOR_SOURCE, ex.HUBBLE_H]
        elif w == "three_species":
            vals = [ex.SOURCE_A, ex.SOURCE_B, ex.SOURCE_C]
        else:
            vals = [ex.SOURCE_VISIBLE, ex.SOURCE_DARK]
        want = [v for _, v in THETA[w]]
        assert np.allclose(vals, want), (w, vals, want)
        assert ex.dt == DT and ex.softening == SOFT and ex.integrator == INTEG, w
        got[w] = vals
    return got


# ════════════════════════════════════════════════════════════════════════════
# Fast path (benchmark simulator internals, traced parameters)
# ════════════════════════════════════════════════════════════════════════════

_JAX = {}


def jx():
    if not _JAX:
        from physchool.worlds import nbody_sampler as NS  # sets jax_enable_x64
        from physchool.worlds import force_laws as FL
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        tq = np.linspace(0.0, 14.0, 1121)
        wq = np.full(tq.size, tq[1] - tq[0])
        wq[0] *= 0.5
        wq[-1] *= 0.5
        _JAX.update(NS=NS, FL=FL, jax=jax, jnp=jnp, CH=jnp.cosh(jnp.asarray(tq)), WQ=jnp.asarray(wq))
    return _JAX


def k1_jax(x):
    """Modified Bessel K1 by trapezoid quadrature of int_0^inf exp(-x cosh t) cosh t dt.

    Used only for Yukawa-family ALTERNATIVE hypotheses (the benchmark's own kernel uses a
    scipy pure_callback that has no JVP rule). Max relative error vs scipy.special.k1 on
    x in [3e-4, 63]: 3.6e-15 (checked at development time and again in parity.tsv).
    """
    J = jx()
    jnp = J["jnp"]
    return jnp.sum(jnp.exp(-x[..., None] * J["CH"]) * J["CH"] * J["WQ"], axis=-1)


INV2PI = 1.0 / (2.0 * math.pi)


def k_poisson(G):
    FL = jx()["FL"]
    return lambda r, qi, qj, mi, mj: FL.poisson_2d_force(r, qi, qj, mi, mj, G=G)


def k_yukawa(G, lam, sgn=1.0):
    return lambda r, qi, qj, mi, mj: sgn * G * qi * qj * INV2PI * k1_jax(r / lam) / lam


def k_riesz(G, alpha_float):
    FL = jx()["FL"]
    return lambda r, qi, qj, mi, mj: FL.riesz_2d_force(r, qi, qj, mi, mj, G=G, alpha=float(alpha_float))


def k_ed(G, R):
    FL = jx()["FL"]
    return lambda r, qi, qj, mi, mj: FL.extra_dimensions_2d_force(r, qi, qj, mi, mj, G=G,
                                                                  R_compact=R, n_images=20)


def k_coulomb(k):
    FL = jx()["FL"]
    return lambda r, qi, qj, mi, mj: FL.coulomb_force(r, qi, qj, mi, mj, k=k)


class Model:
    """A hypothesis simulated with the benchmark's integrator/acceleration code.

    kernel(u) -> pairwise force law; charges(u, s) -> (pos0, vel0, masses, src, frc);
    mod(u) -> f(t) | None (force_modulation); ext(u, s, masses) -> f(pos, t) | None
    (external_acceleration); vis = agent-visible particle indices (static).
    """

    def __init__(self, name, kernel, charges=None, mod=None, ext=None, vis=None, n_steps=2000):
        self.name, self.kernel, self.charges, self.mod, self.ext = name, kernel, charges, mod, ext
        self.vis = np.asarray(vis) if vis is not None else None
        self.n_steps = int(n_steps)
        self._sim = self._jac = None

    def sim_one(self, u, s):
        J = jx()
        jax, jnp, NS = J["jax"], J["jnp"], J["NS"]
        if self.charges is None:
            pos0, vel0, masses, src, frc = s["pos0"], s["vel0"], s["masses"], s["src"], s["frc"]
        else:
            pos0, vel0, masses, src, frc = self.charges(u, s)
        acc = NS.make_acceleration_fn(self.kernel(u), SOFT)
        modf = self.mod(u) if self.mod is not None else None
        extf = self.ext(u, s, masses) if self.ext is not None else None

        def apt(pos, t):
            a = acc(pos, src, frc, masses)
            if modf is not None:
                a = modf(t) * a
            if extf is not None:
                a = a + extf(pos, t)
            return a

        step = NS._make_step(INTEG, apt, DT)

        def chunk(state, _):
            state = jax.lax.fori_loop(0, 1, lambda _i, st: step(st), state)
            return state, state

        _, rec = jax.lax.scan(chunk, (pos0, vel0, s["t0"]), jnp.arange(self.n_steps))
        traj = jnp.concatenate([pos0[None], rec[0]], axis=0)
        vis = self.vis if self.vis is not None else np.arange(pos0.shape[0])
        return traj[s["idx"]][:, vis, :] - s["centre"]

    def traj_one(self, u, s):
        """Full trajectory (for geometry diagnostics)."""
        J = jx()
        jax, jnp, NS = J["jax"], J["jnp"], J["NS"]
        pos0, vel0, masses, src, frc = (self.charges(u, s) if self.charges is not None else
                                        (s["pos0"], s["vel0"], s["masses"], s["src"], s["frc"]))
        acc = NS.make_acceleration_fn(self.kernel(u), SOFT)
        modf = self.mod(u) if self.mod is not None else None
        extf = self.ext(u, s, masses) if self.ext is not None else None

        def apt(pos, t):
            a = acc(pos, src, frc, masses)
            if modf is not None:
                a = modf(t) * a
            if extf is not None:
                a = a + extf(pos, t)
            return a

        step = NS._make_step(INTEG, apt, DT)

        def chunk(state, _):
            state = jax.lax.fori_loop(0, 1, lambda _i, st: step(st), state)
            return state, state

        _, rec = jax.lax.scan(chunk, (pos0, vel0, s["t0"]), jnp.arange(self.n_steps))
        return jnp.concatenate([pos0[None], rec[0]], axis=0) - s["centre"]

    def sim(self, u, S):
        J = jx()
        if self._sim is None:
            self._sim = J["jax"].jit(J["jax"].vmap(self.sim_one, in_axes=(None, 0)))
        return np.asarray(self._sim(J["jnp"].asarray(u, dtype=J["jnp"].float64), S))

    def jac(self, u, S):
        J = jx()
        if self._jac is None:
            self._jac = J["jax"].jit(J["jax"].vmap(J["jax"].jacfwd(self.sim_one), in_axes=(None, 0)))
        return np.asarray(self._jac(J["jnp"].asarray(u, dtype=J["jnp"].float64), S))


def system_one(world, ex, d):
    """Replicates the executor's _run_one initial state for design d (numpy)."""
    c = float(ex.domain_size) / 2.0
    s = {}
    if world in TWO_P:
        x, y = d["pos2"]
        vx, vy = d["velocity2"]
        p1, p2 = float(d["p1"]), float(d["p2"])
        s["pos0"] = np.array([[c, c], [c + x, c + y]], float)
        s["vel0"] = np.array([[0.0, 0.0], [vx, vy]], float)
        if world == "coulomb":
            s["masses"] = np.array([1e15, 1.0])
            s["src"] = np.array([abs(p1), -abs(p2)])
            s["frc"] = np.array([0.0, -abs(p2)])
        else:
            s["masses"] = np.array([1e15, p2])
            s["src"] = np.array([p1, 1.0])
            s["frc"] = np.array([0.0, 1.0])
        s["p1"], s["p2"] = np.float64(p1), np.float64(p2)
        s["t0"] = np.float64(d.get("start_time", 0.0)) if world == "oscillator" else np.float64(0.0)
    elif world == "circle":
        R = float(d.get("ring_radius", 5.0))
        vt = float(d.get("initial_tangential_velocity", 0.0))
        ang = np.linspace(0, 2 * np.pi, ex.N_RING, endpoint=False)
        ring = np.column_stack([c + R * np.cos(ang), c + R * np.sin(ang)])
        s["pos0"] = np.vstack([[[c, c]], ring])
        s["vel0"] = np.vstack([[[0.0, 0.0]], np.column_stack([-vt * np.sin(ang), vt * np.cos(ang)])])
        s["masses"] = np.ones(ex.N_TOTAL)
        s["src"] = np.ones(ex.N_TOTAL)
        s["frc"] = np.ones(ex.N_TOTAL)
        s["t0"] = np.float64(0.0)
    elif world == "three_species":
        pp = np.asarray(d["probe_positions"], float)
        pv = np.asarray(d["probe_velocities"], float)
        s["pos0"] = np.vstack([ex._bg_positions_rel + c, pp + c])
        s["vel0"] = np.vstack([ex._bg_velocities, pv])
        s["masses"] = np.ones(ex.N_TOTAL)
        src = np.zeros(ex.N_TOTAL)
        src[ex.SPECIES_A] = ex.SOURCE_A
        src[ex.SPECIES_B] = ex.SOURCE_B
        src[ex.SPECIES_C] = ex.SOURCE_C
        s["src"] = src
        s["frc"] = np.ones(ex.N_TOTAL)
        s["t0"] = np.float64(0.0)
    elif world == "dark_matter":
        pp = np.asarray(d["probe_positions"], float)
        pv = np.asarray(d["probe_velocities"], float)
        sign = float(d.get("visible_velocity_sign", 1.0))
        s["pos0"] = np.vstack([ex._visible_positions_rel + c, ex._dark_positions_rel + c, pp + c])
        s["vel0"] = np.vstack([sign * ex._visible_velocities, ex._dark_velocities, pv])
        s["masses"] = np.ones(ex.N_TOTAL)
        src = np.zeros(ex.N_TOTAL)
        src[ex.VISIBLE] = ex.SOURCE_VISIBLE
        src[ex.DARK] = ex.SOURCE_DARK
        s["src"] = src
        s["frc"] = np.ones(ex.N_TOTAL)
        s["t0"] = np.float64(0.0)
        s["vis_pos"] = ex._visible_positions_rel + c
        s["vis_vel"] = sign * ex._visible_velocities
        s["probe_pos"] = pp + c
        s["probe_vel"] = pv
    elif world in PROBES_MASS:
        pp = np.asarray(d["probe_positions"], float)
        pv = np.asarray(d["probe_velocities"], float)
        pm = (np.asarray(d["probe_masses"], float) if d.get("probe_masses") is not None
              else np.full(ex.N_PROBES, ex.DEFAULT_PROBE_MASS))
        s["pos0"] = np.vstack([ex._bg_positions_rel + c, pp + c])
        s["vel0"] = np.vstack([ex._bg_velocities, pv])
        s["masses"] = np.concatenate([ex._bg_masses, pm])
        src = np.zeros(ex.N_TOTAL)
        src[ex.ANCHOR_INDEX] = ex.ANCHOR_SOURCE
        s["src"] = src
        frc = np.zeros(ex.N_TOTAL)
        frc[ex.RING_INDICES] = ex._ring_masses
        frc[ex.PROBE_INDICES] = pm
        s["frc"] = frc
        s["t0"] = np.float64(0.0)
    else:
        raise ValueError(world)
    n_total = max(int(round(float(d["duration"]) / DT)), 1)
    idx = [max(0, min(n_total, int(round(float(t) / DT)))) for t in sorted(d["measurement_times"])]
    K = len(idx)
    s["idx"] = np.array(idx + [0] * (KMAX - K), dtype=np.int64)
    s["mask"] = np.array([1.0] * K + [0.0] * (KMAX - K))
    s["centre"] = np.float64(c)
    s["n_total"] = n_total
    return s


def stack_systems(sl, keys=None):
    J = jx()
    jnp = J["jnp"]
    keys = keys or [k for k in sl[0] if k not in ("mask", "n_total")]
    return {k: jnp.asarray(np.stack([s[k] for s in sl])) for k in keys}


def n_steps_for(sl):
    return int(max(s["n_total"] for s in sl))


# ── truth models (fast path) ────────────────────────────────────────────────

def truth_model(world, n_steps, vis=None):
    jnp = jx()["jnp"]
    th_alpha = dict(THETA[world]).get("alpha")
    if world == "gravity":
        return Model("truth", lambda u: k_poisson(u[0]), n_steps=n_steps, vis=vis)
    if world == "yukawa":
        return Model("truth", lambda u: k_yukawa(u[0], u[1]), n_steps=n_steps, vis=vis)
    if world in ("fractional", "circle"):
        return Model("truth", lambda u: k_riesz(u[0], th_alpha), n_steps=n_steps, vis=vis)
    if world == "extra_dimensions":
        return Model("truth", lambda u: k_ed(u[0], u[1]), n_steps=n_steps, vis=vis)
    if world == "coulomb":
        return Model("truth", lambda u: k_coulomb(u[0]), n_steps=n_steps, vis=vis)
    if world == "oscillator":
        return Model("truth", lambda u: k_poisson(1.0),
                     mod=lambda u: (lambda t: u[0] * jnp.cos(u[1] * t + u[2])), n_steps=n_steps, vis=vis)
    if world == "ether":
        def ch(u, s):
            return s["pos0"], s["vel0"], s["masses"], s["src"].at[0].set(u[0]), s["frc"]

        def ext(u, s, m):
            e = jnp.stack([0.0 * u[1], u[1]])
            return lambda pos, t: jnp.broadcast_to(e[None, :], pos.shape)
        return Model("truth", lambda u: k_poisson(1.0), charges=ch, ext=ext, n_steps=n_steps, vis=vis)
    if world == "hubble":
        def ch(u, s):
            return s["pos0"], s["vel0"], s["masses"], s["src"].at[0].set(u[0]), s["frc"]

        def ext(u, s, m):
            cv = jnp.stack([s["centre"], s["centre"]])
            return lambda pos, t: u[1] * (pos - cv[None, :])
        return Model("truth", lambda u: k_poisson(1.0), charges=ch, ext=ext, n_steps=n_steps, vis=vis)
    if world == "three_species":
        def ch(u, s):
            src = jnp.concatenate([jnp.full(10, u[0]), jnp.full(10, u[1]), jnp.full(10, u[2]),
                                   jnp.zeros(5)])
            return s["pos0"], s["vel0"], s["masses"], src, s["frc"]
        return Model("truth", lambda u: k_poisson(1.0), charges=ch, n_steps=n_steps, vis=vis)
    if world == "dark_matter":
        def ch(u, s):
            src = jnp.concatenate([jnp.full(20, u[0]), jnp.full(10, u[1]), jnp.zeros(5)])
            return s["pos0"], s["vel0"], s["masses"], src, s["frc"]
        return Model("truth", lambda u: k_poisson(1.0), charges=ch, n_steps=n_steps,
                     vis=list(range(20)) + list(range(30, 35)))
    raise ValueError(world)


AD_TRACEABLE = {"gravity": [0], "yukawa": [0, 1], "fractional": [0], "circle": [0],
                "extra_dimensions": [0, 1], "coulomb": [0], "oscillator": [0, 1, 2],
                "ether": [0, 1], "hubble": [0, 1], "three_species": [0, 1, 2], "dark_matter": [0, 1]}


# ── alternative models ──────────────────────────────────────────────────────

def _kernel_alt(kind, sgn):
    """Pairwise-kernel alternative families; u unconstrained except bounded exponents."""
    jnp = jx()["jnp"]
    if kind in ("inv1", "inv2", "inv3"):
        n = float(kind[-1])
        return lambda u: (lambda r, qi, qj, mi, mj: sgn * jnp.exp(u[0]) * qi * qj * r ** (-n))
    if kind in ("pow", "pow_fast", "pow_slow"):
        return lambda u: (lambda r, qi, qj, mi, mj: sgn * jnp.exp(u[0]) * qi * qj * r ** (-u[1]))
    if kind == "exp":
        return lambda u: (lambda r, qi, qj, mi, mj: sgn * jnp.exp(u[0]) * qi * qj * jnp.exp(-r / jnp.exp(u[1])))
    if kind == "const":
        return lambda u: (lambda r, qi, qj, mi, mj: sgn * jnp.exp(u[0]) * qi * qj * jnp.ones_like(r))
    if kind == "laplace":
        return lambda u: (lambda r, qi, qj, mi, mj: sgn * jnp.exp(u[0]) * qi * qj * INV2PI / r)
    if kind == "yukawa":
        return lambda u: k_yukawa(jnp.exp(u[0]), jnp.exp(u[1]), sgn)
    if kind == "crossover":
        return lambda u: (lambda r, qi, qj, mi, mj: sgn * qi * qj * (jnp.exp(u[0]) / r + jnp.exp(u[1]) / r ** 2))
    raise ValueError(kind)


def kernel_alt_spec(kind, F0, r0s, sgn):
    """(n_params, starts, bounds) for a kernel alternative, starts matched to the true force
    F0(r0) at reference radii r0 (unit charges)."""
    starts = []
    inf = np.inf
    lb, ub = None, None
    for r0 in r0s:
        f = F0(r0)
        if kind in ("inv1", "inv2", "inv3"):
            n = float(kind[-1])
            starts.append([math.log(f * r0 ** n)])
        elif kind in ("pow", "pow_fast", "pow_slow"):
            ns = {"pow": [1.0, 1.5, 2.0], "pow_fast": [1.2, 2.0], "pow_slow": [0.5, 0.95]}[kind]
            for n in ns:
                starts.append([math.log(f * r0 ** n), n])
            lo_n, hi_n = {"pow": (0.0, 4.0), "pow_fast": (1.0, 4.0), "pow_slow": (0.0, 1.0)}[kind]
            lb, ub = [-inf, lo_n], [inf, hi_n]
        elif kind == "exp":
            for L in (r0, 3 * r0):
                starts.append([math.log(f * math.exp(r0 / L)), math.log(L)])
        elif kind == "const":
            starts.append([math.log(f)])
        elif kind == "laplace":
            starts.append([math.log(2 * math.pi * r0 * f)])
        elif kind == "yukawa":
            import scipy.special as sp
            for lam in (0.5 * r0, 2.0 * r0):
                g = f / (INV2PI * sp.k1(r0 / lam) / lam)
                starts.append([math.log(g), math.log(lam)])
        elif kind == "crossover":
            starts.append([math.log(0.9 * f * r0), math.log(0.1 * f * r0 ** 2)])
            starts.append([math.log(0.5 * f * r0), math.log(0.5 * f * r0 ** 2)])
    n = len(starts[0])
    if lb is None:
        lb, ub = [-inf] * n, [inf] * n
    return starts, (lb, ub)


def alt_model(world, alt, n_steps, S_np=None):
    """Return (Model, starts, bounds, param_names) for a named alternative."""
    jnp = jx()["jnp"]
    inf = np.inf
    two_vis = [0, 1]
    if world in TWO_P or world == "circle":
        sgn = -1.0 if world == "coulomb" else 1.0
        if world == "circle":
            tk = k_riesz(1.0, 0.75)
            r0s = [2.0, 5.0]
        else:
            tk = {"gravity": k_poisson(1.0), "yukawa": k_yukawa(1.0, 2.0),
                  "fractional": k_riesz(1.0, 0.5), "extra_dimensions": k_ed(1.0, 0.5),
                  "coulomb": k_coulomb(1.0), "oscillator": k_poisson(1.0)}[world]
            r0s = [1.0, 3.0]

        def F0(r0, tk=tk):
            one = jnp.asarray(1.0)
            q = -1.0 if world == "coulomb" else 1.0
            return abs(float(tk(jnp.asarray([r0]), q, one, one, one)[0]))

        vis = two_vis if world in TWO_P else None
        if alt in ("inv1", "inv2", "inv3", "pow", "pow_fast", "pow_slow", "exp", "const", "laplace",
                   "yukawa", "crossover"):
            starts, bounds = kernel_alt_spec(alt, F0, r0s, sgn)
            names = {"inv1": ["logA"], "inv2": ["logA"], "inv3": ["logA"], "const": ["logA"],
                     "laplace": ["logG"], "exp": ["logA", "logL"], "yukawa": ["logG", "loglambda"],
                     "crossover": ["logA", "logB"]}.get(alt, ["logA", "n"])
            mod = None
            if world == "oscillator":
                raise ValueError("kernel alternatives not used in oscillator")
            return Model(alt, _kernel_alt(alt, sgn), vis=vis, n_steps=n_steps), starts, bounds, names
        # role alternatives: transform how p1/p2 enter, keep the true-family kernel free
        if alt in ("swap", "product", "ratio_12", "ratio_21", "additive"):
            def ch(u, s, alt=alt):
                p1, p2 = s["p1"], s["p2"]
                one = jnp.asarray(1.0)
                if world == "coulomb":
                    qsrc = {"ratio_12": p1 / p2, "ratio_21": p2 / p1, "additive": p1 + p2}[alt]
                    return (s["pos0"], s["vel0"], jnp.stack([1e15 * one, one]),
                            jnp.stack([qsrc, -one]), jnp.stack([0.0 * one, -one]))
                if alt == "swap":
                    return (s["pos0"], s["vel0"], jnp.stack([1e15 * one, p1]), jnp.stack([p2, one]), s["frc"])
                return (s["pos0"], s["vel0"], jnp.stack([1e15 * one, one]), jnp.stack([p1 * p2, one]), s["frc"])
            if world == "coulomb":
                return (Model(alt, lambda u: k_coulomb(jnp.exp(u[0])), charges=ch, vis=vis, n_steps=n_steps),
                        [[0.0], [math.log(3.0)], [math.log(0.3)]], ([-inf], [inf]), ["logk"])
            if world == "gravity":
                return (Model(alt, lambda u: k_poisson(jnp.exp(u[0])), charges=ch, vis=vis, n_steps=n_steps),
                        [[0.0], [math.log(3.0)], [math.log(0.3)]], ([-inf], [inf]), ["logG"])
            if world == "yukawa":
                return (Model(alt, lambda u: k_yukawa(jnp.exp(u[0]), jnp.exp(u[1])), charges=ch, vis=vis,
                              n_steps=n_steps),
                        [[0.0, math.log(2.0)], [0.0, math.log(1.0)], [0.0, math.log(4.0)]],
                        ([-inf] * 2, [inf] * 2), ["logG", "loglambda"])
            if world == "oscillator":
                return (Model(alt, lambda u: k_poisson(1.0), charges=ch,
                              mod=lambda u: (lambda t: u[0] * jnp.cos(jnp.exp(u[1]) * t + u[2])),
                              vis=vis, n_steps=n_steps),
                        [[5.0, math.log(math.pi / 2), 0.0], [3.0, math.log(math.pi / 2), 0.5],
                         [5.0, math.log(math.pi), 0.0], [1.0, math.log(math.pi / 4), 0.0]],
                        ([-inf] * 3, [inf] * 3), ["G0", "logomega", "phi"])
        if world == "oscillator" and alt == "const_coupling":
            return (Model(alt, lambda u: k_poisson(1.0), mod=lambda u: (lambda t: u[0] + 0.0 * t),
                          vis=vis, n_steps=n_steps),
                    [[0.5], [2.0], [-2.0], [5.0]], ([-inf], [inf]), ["G"])
        if world == "oscillator" and alt == "transient":
            return (Model(alt, lambda u: k_poisson(1.0),
                          mod=lambda u: (lambda t: u[0] + u[1] * jnp.exp(-t / jnp.exp(u[2]))),
                          vis=vis, n_steps=n_steps),
                    [[0.0, 5.0, math.log(2.0)], [0.0, 5.0, math.log(8.0)], [1.0, 3.0, 0.0],
                     [-1.0, 6.0, math.log(0.5)]], ([-inf] * 3, [inf] * 3), ["A", "B", "logtau"])
    if world == "ether":
        if alt == "mass_dep_drift":
            def ch(u, s):
                return s["pos0"], s["vel0"], s["masses"], s["src"].at[0].set(jnp.exp(u[0])), s["frc"]

            def ext(u, s, m):
                return lambda pos, t: jnp.stack([jnp.zeros_like(m), jnp.exp(u[1]) / m], axis=-1)
            return (Model(alt, lambda u: k_poisson(1.0), charges=ch, ext=ext, n_steps=n_steps),
                    [[math.log(50.0), math.log(0.05)], [math.log(50.0), math.log(0.1)],
                     [math.log(45.0), math.log(0.03)]], ([-inf] * 2, [inf] * 2), ["logQ", "logalpha'"])
        if alt == "repulsor":
            def ch(u, s):
                c = s["centre"]
                pos0 = jnp.concatenate([s["pos0"], jnp.stack([c + u[1], c + u[2]])[None, :]], axis=0)
                vel0 = jnp.concatenate([s["vel0"], jnp.zeros((1, 2))], axis=0)
                masses = jnp.concatenate([s["masses"], jnp.array([1e15])])
                src = jnp.concatenate([s["src"].at[0].set(jnp.exp(u[0])), -jnp.exp(u[3])[None]])
                frc = jnp.concatenate([s["frc"], jnp.zeros(1)])
                return pos0, vel0, masses, src, frc
            return (Model(alt, lambda u: k_poisson(1.0), charges=ch, vis=list(range(26)), n_steps=n_steps),
                    [[math.log(50.0), 0.0, -10.0, math.log(3.0)], [math.log(50.0), 0.0, -20.0, math.log(6.0)],
                     [math.log(50.0), 0.0, -40.0, math.log(12.0)]],
                    ([-inf] * 4, [inf] * 4), ["logQ", "x_rep", "y_rep", "logs_rep"])
    if world == "hubble":
        def ch(u, s):
            return s["pos0"], s["vel0"], s["masses"], s["src"].at[0].set(jnp.exp(u[0])), s["frc"]
        if alt == "const_push":
            def ext(u, s, m):
                cv = jnp.stack([s["centre"], s["centre"]])

                def f(pos, t):
                    dd = pos - cv[None, :]
                    rr = jnp.sqrt(jnp.maximum(jnp.sum(dd * dd, axis=-1), 1e-30))
                    return u[1] * dd / rr[:, None]
                return f
            return (Model(alt, lambda u: k_poisson(1.0), charges=ch, ext=ext, n_steps=n_steps),
                    [[math.log(50.0), 0.5], [math.log(50.0), 0.2], [math.log(60.0), 1.0]],
                    ([-inf] * 2, [inf] * 2), ["logQ", "c_push"])
        if alt == "central_only":
            return (Model(alt, lambda u: k_poisson(1.0), charges=ch, n_steps=n_steps),
                    [[math.log(50.0)], [math.log(35.0)]], ([-inf], [inf]), ["logQ"])
    if world == "three_species":
        A, B, C = np.arange(0, 10), np.arange(10, 20), np.arange(20, 30)

        def mk(fn, starts, names, bounds=None):
            def ch(u, s):
                return s["pos0"], s["vel0"], s["masses"], fn(u), s["frc"]
            n = len(starts[0])
            return (Model(alt, lambda u: k_poisson(1.0), charges=ch, n_steps=n_steps), starts,
                    bounds or ([-inf] * n, [inf] * n), names)
        z5 = jnp.zeros(5)
        if alt == "merge_AB":
            return mk(lambda u: jnp.concatenate([jnp.full(20, u[0]), jnp.full(10, u[1]), z5]),
                      [[2.0, -2.0], [1.5, -1.0]], ["s_AB", "s_C"])
        if alt == "merge_AC":
            return mk(lambda u: jnp.concatenate([jnp.full(10, u[0]), jnp.full(10, u[1]), jnp.full(10, u[0]), z5]),
                      [[-0.5, 3.0], [0.5, 3.0]], ["s_AC", "s_B"])
        if alt == "merge_BC":
            return mk(lambda u: jnp.concatenate([jnp.full(10, u[0]), jnp.full(20, u[1]), z5]),
                      [[1.0, 0.5], [1.0, 1.0]], ["s_A", "s_BC"])
        if alt == "all_identical":
            return mk(lambda u: jnp.concatenate([jnp.full(30, u[0]), z5]), [[0.67], [1.0]], ["s"])
        if alt == "nonneg":
            return mk(lambda u: jnp.concatenate([jnp.full(10, u[0]), jnp.full(10, u[1]), jnp.full(10, u[2]), z5]),
                      [[1.0, 3.0, 0.01], [1.0, 2.0, 0.1], [0.5, 1.5, 0.5]], ["s_A", "s_B", "s_C"],
                      ([0.0] * 3, [inf] * 3))
        if alt.startswith("probes_"):
            k = "ABC".index(alt[-1])
            return mk(lambda u: jnp.concatenate([jnp.full(10, u[0]), jnp.full(10, u[1]), jnp.full(10, u[2]),
                                                 jnp.full(5, u[k])]),
                      [[1.0, 3.0, -2.0], [0.8, 2.5, -1.5]], ["s_A", "s_B", "s_C"])
        if alt.startswith("partition:"):
            part = np.array(json.loads(alt.split(":", 1)[1]))

            def fn(u, part=part):
                return jnp.concatenate([jnp.stack([u[0], u[1], u[2]])[part], z5])
            return mk(fn, [[1.0, 3.0, -2.0], [0.9, 2.8, -1.8]], ["s_A", "s_B", "s_C"])
    if world == "dark_matter" and alt == "no_hidden":
        def ch(u, s):
            src = jnp.concatenate([jnp.full(20, u[0]), jnp.zeros(10), jnp.full(5, u[1])])
            return s["pos0"], s["vel0"], s["masses"], src, s["frc"]
        return (Model(alt, lambda u: k_poisson(1.0), charges=ch, n_steps=n_steps,
                      vis=list(range(20)) + list(range(30, 35))),
                [[1.0, 0.0], [3.5, 0.0], [1.0, 1.0], [3.5, 2.0]], ([-inf] * 2, [inf] * 2), ["s_vis", "s_probe"])
    raise ValueError((world, alt))


def dm_count_model(N, n_steps):
    jnp = jx()["jnp"]
    q = 50.0 / N

    def ch(u, s):
        dark = s["centre"] + u.reshape(N, 2)
        pos0 = jnp.concatenate([s["vis_pos"], dark, s["probe_pos"]], axis=0)
        vel0 = jnp.concatenate([s["vis_vel"], jnp.zeros((N, 2)), s["probe_vel"]], axis=0)
        masses = jnp.ones(25 + N)
        src = jnp.concatenate([jnp.ones(20), jnp.full(N, q), jnp.zeros(5)])
        frc = jnp.ones(25 + N)
        return pos0, vel0, masses, src, frc
    return Model(f"count{N}", lambda u: k_poisson(1.0), charges=ch, n_steps=n_steps,
                 vis=list(range(20)) + list(range(20 + N, 25 + N)))


# ════════════════════════════════════════════════════════════════════════════
# fitting
# ════════════════════════════════════════════════════════════════════════════

def per_design_rms(res, mask, nvis):
    """res: (D, KMAX, nvis, 2) normalized residuals -> RMS per design over valid entries."""
    sq = (res ** 2).sum(axis=(2, 3))            # (D, KMAX)
    cnt = mask.sum(axis=1) * nvis * 2
    return np.sqrt((sq * mask).sum(axis=1) / cnt), np.abs(res * mask[:, :, None, None]).max(axis=(1, 2, 3))


def fit_model(model, starts, bounds, S, truth, mask, sig, max_nfev=200, ck_prefix=None):
    from scipy.optimize import least_squares
    D = truth.shape[0]
    nvis = truth.shape[2]
    w = mask[:, :, None, None]

    def fun(u):
        y = model.sim(u, S)
        r = ((y - truth) / sig) * w
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6).ravel()

    def jac(u):
        Jm = model.jac(u, S)                     # (D, KMAX, nvis, 2, n)
        Jm = (Jm / sig) * w[..., None]
        return np.nan_to_num(Jm, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, len(u))

    from types import SimpleNamespace
    best, log = None, []
    t0 = time.time()
    for si, x0 in enumerate(starts):
        x0 = np.asarray(x0, float)
        lb, ub = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        x0 = np.clip(x0, lb + 1e-9 * (np.isfinite(lb)), ub - 1e-9 * (np.isfinite(ub)))
        ckp = Path(f"{ck_prefix}.start{si}.json") if ck_prefix else None
        try:
            if ckp is not None and ckp.exists():
                d = json.loads(ckp.read_text())
                r = SimpleNamespace(x=np.asarray(d["x"], float), cost=float(d["cost"]), nfev=int(d["nfev"]),
                                    status=int(d["status"]), fun=None)
            else:
                r = least_squares(fun, x0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac",
                                  max_nfev=max_nfev, ftol=1e-13, xtol=1e-13, gtol=1e-13)
                if ckp is not None:
                    jdump({"x0": x0.tolist(), "x": r.x.tolist(), "cost": float(r.cost), "nfev": int(r.nfev),
                           "status": int(r.status)}, ckp)
            log.append({"x0": x0.tolist(), "cost": float(r.cost), "nfev": int(r.nfev), "status": int(r.status)})
            if best is None or r.cost < best.cost:
                best = r
        except Exception as e:  # noqa: BLE001
            log.append({"x0": x0.tolist(), "error": repr(e)[:300]})
    if best is None:
        return {"ok": False, "log": log}
    fvec = best.fun if best.fun is not None else fun(best.x)   # deterministic re-evaluation at saved x
    res = fvec.reshape(D, KMAX, nvis, 2)
    rms, mx = per_design_rms(res, mask, nvis)
    return {"ok": True, "u": best.x.tolist(), "cost": float(best.cost), "rms": rms.tolist(),
            "max": mx.tolist(), "log": log, "seconds": time.time() - t0}


# ════════════════════════════════════════════════════════════════════════════
# per-world pipeline
# ════════════════════════════════════════════════════════════════════════════

def graded_grad(c, theta):
    """Gradient of the graded quantity g(theta) (log units for 'mult')."""
    names = [n for n, _ in THETA_W[c["world"]]]
    g = np.zeros(len(theta))
    q = c["q"]
    ix = {n: i for i, n in enumerate(names)}
    if q == "alpha":
        g[ix["alpha"]] = 1.0
    elif q == "log_lambda":
        g[ix["lambda"]] = 1.0 / theta[ix["lambda"]]
    elif q == "log_T":
        g[ix["omega"]] = -1.0 / theta[ix["omega"]]
    elif q == "log_R_c":
        g[ix["R_c"]] = 1.0 / theta[ix["R_c"]]
    elif q == "log_G":
        g[ix["G"]] = 1.0 / theta[ix["G"]]
    elif q == "log_k":
        g[ix["k"]] = 1.0 / theta[ix["k"]]
    elif q == "log_alpha":
        g[ix["alpha"]] = 1.0 / theta[ix["alpha"]]
    elif q == "log_H":
        g[ix["H"]] = 1.0 / theta[ix["H"]]
    elif q == "log_sB_over_sA":
        g[ix["s_B"]] = 1.0 / theta[ix["s_B"]]
        g[ix["s_A"]] = -1.0 / theta[ix["s_A"]]
    elif q == "log_abs_sC_over_sA":
        g[ix["s_C"]] = 1.0 / theta[ix["s_C"]]
        g[ix["s_A"]] = -1.0 / theta[ix["s_A"]]
    elif q == "log_sdark_over_svis":
        g[ix["s_dark"]] = 1.0 / theta[ix["s_dark"]]
        g[ix["s_vis"]] = -1.0 / theta[ix["s_vis"]]
    elif q == "log_G0":
        g[ix["G0"]] = 1.0 / theta[ix["G0"]]
    else:
        raise ValueError(q)
    return g


THETA_W = {w: THETA[w] for w in THETA}


def crlb_se(S, sig, grad):
    """SE of g for ONE experiment scaled by 1/sqrt(16). S: (m x p) unwhitened sensitivities."""
    Sw = S / sig
    u_, sv, vt = np.linalg.svd(Sw, full_matrices=False)
    if sv.size == 0 or sv[0] <= 0 or sv[-1] <= 1e-13 * sv[0]:
        return float("inf")
    cov = (vt.T / sv ** 2) @ vt
    return float(math.sqrt(max(grad @ cov @ grad, 0.0)) / math.sqrt(N_ROUNDS))


FIT_JOBS = int(os.environ.get("EXP263_FIT_JOBS", "1"))


def alt_fit_task(task):
    """Fit one named alternative (joint + solo DEFAULT + solo BEST + DEFAULT+BEST)."""
    world, cid, alt, n_steps, S_np, Y, mask, sig, sub, labels, smoke = task
    J = jx()
    S_all = {k: J["jnp"].asarray(v) for k, v in S_np.items()}
    m, starts, bounds, pnames = alt_model(world, alt, n_steps)
    rec = {"claim": cid, "alt": alt, "params": pnames}
    for key, ids in sub.items():
        if smoke and key != "joint":
            continue
        Ssub = {k: v[np.array(ids)] for k, v in S_all.items()}
        r = fit_model(m, starts, bounds, Ssub, Y[ids], mask[ids], sig, max_nfev=(20 if smoke else 200))
        r["labels"] = [labels[i] for i in ids]
        rec[key] = r
        J["jax"].clear_caches()
    return rec


def run_world(world, n_sampled=N_SAMPLED, smoke=False, log=print, base_only=False):
    t_start = time.time()
    from scienceagent.worlds import get_world
    J = jx()
    sig = sigma_of(world)
    theta = np.array([v for _, v in THETA[world]], float)
    names = [n for n, _ in THETA[world]]
    designs = all_designs(world, n_sampled)
    labels = [d["label"] for d in designs]
    D = len(designs)
    out = {"world": world, "sigma": sig, "theta": dict(THETA[world]), "designs": designs}

    # ── parity (1): benchmark executor as shipped vs my executor runner at DEFAULT
    ex_ship = get_world(CODE[world], engine="nbody")["executor"]
    with ex_ship.noise_disabled():
        ship = result_to_obs(world, ex_ship.run([exp_of(designs[0])])[0])
    ex_true = build_executor(world, theta)
    truth_obs = executor_obs(ex_true, world, designs)          # registered truth predictions
    J["jax"].clear_caches()
    par = {"runner_vs_shipped_default_maxabs": float(np.abs(truth_obs[0] - ship).max())}
    log(f"[{world}] truth predictions done ({time.time() - t_start:.0f}s)")

    # stacked truth (padded to KMAX) and systems
    nvis = truth_obs[0].shape[1]
    Y = np.zeros((D, KMAX, nvis, 2))
    mask = np.zeros((D, KMAX))
    for i, y in enumerate(truth_obs):
        Y[i, : y.shape[0]] = y
        mask[i, : y.shape[0]] = 1.0
    ref_ex = build_executor(world, theta)
    systems = [system_one(world, ref_ex, d) for d in designs]
    n_steps = n_steps_for(systems)
    S_all = stack_systems(systems)

    # ── parity (2): fast path (benchmark internals, truth kernel) vs executor on all designs
    tm = truth_model(world, n_steps, vis=[0, 1] if world in TWO_P else None)
    Yf = tm.sim(theta, S_all)
    diff = np.abs(Yf - Y) * mask[:, :, None, None]
    par["fast_vs_executor_maxabs"] = float(diff.max())
    par["fast_vs_executor_maxabs_over_sigma"] = float(diff.max() / sig)
    par["fast_vs_executor_per_design_max_over_sigma"] = (diff.max(axis=(1, 2, 3)) / sig).tolist()
    out["parity"] = par
    log(f"[{world}] parity {par['runner_vs_shipped_default_maxabs']:.2e} / fast {par['fast_vs_executor_maxabs']:.2e}")

    # ── central-difference sensitivities through the executor (registered)
    fd = {}
    for step in FD_STEPS:
        cols = [[None] * len(theta) for _ in range(D)]
        for j in range(len(theta)):
            h = step * abs(theta[j]) if theta[j] != 0 else step
            tp, tmn = theta.copy(), theta.copy()
            tp[j] += h
            tmn[j] -= h
            yp = executor_obs(build_executor(world, tp), world, designs)
            ym = executor_obs(build_executor(world, tmn), world, designs)
            J["jax"].clear_caches()   # executor runs compile one executable each; free them
            for i in range(D):
                cols[i][j] = ((yp[i] - ym[i]) / (2 * h)).ravel()
        fd[step] = [np.column_stack(c) for c in cols]
        log(f"[{world}] FD step {step:g} done ({time.time() - t_start:.0f}s)")

    # ── AD cross-check (fast path jacfwd) on traceable constants
    adx = AD_TRACEABLE[world]
    Jad = tm.jac(theta, S_all)                                   # (D, KMAX, nvis, 2, p)
    ad_rel = []
    for i in range(D):
        K = int(mask[i].sum())
        A = Jad[i, :K].reshape(-1, len(theta))[:, adx]
        F = fd[FD_STEPS[0]][i][:, adx]
        ad_rel.append(float(np.linalg.norm(F - A) / max(np.linalg.norm(A), 1e-300)))
    out["fd_vs_ad"] = {"params": [names[j] for j in adx], "rel_frobenius_per_design": ad_rel}

    # ── ranker (Ascension diagnostics) on whitened, log-scaled sensitivities
    from ascension.diagnostics.optimal_design import rank_designs_by_identifiability
    scale = np.array([abs(t) if t != 0 else 1.0 for t in theta])
    dmap = {labels[i]: fd[FD_STEPS[0]][i] * scale[None, :] / sig for i in range(D)}   # DEFAULT first
    rk = rank_designs_by_identifiability(dmap, np.ones(len(theta)), param_names=names)
    best = rk.best
    ib = labels.index(best)
    out["ranker"] = {"best": best, "rationale": rk.rationale,
                     "top5": [(s.label, s.rank, s.condition_number, s.score) for s in rk.ranked[:5]],
                     "default_score": [(s.label, s.rank, s.condition_number, s.score)
                                       for s in rk.ranked if s.label == "DEFAULT"][0],
                     "n_tied_with_best": sum(1 for s in rk.ranked if s.score == rk.ranked[0].score)}
    log(f"[{world}] ranker BEST = {best}")

    # ── PARAMETER claims: CRLBs per design, both steps
    pclaims = [dict(c, world=world) for c in CLAIMS[world] if c["type"] == "PARAMETER"]
    if world == "oscillator":
        pclaims.append(dict(id="OS-N(G0)", q="log_G0", lo=2.5, hi=10.0, truth=5.0, kind="mult",
                            world=world, info_only=True))
    pres = {}
    for c in pclaims:
        g = graded_grad(c, theta)
        tau, tau_hl, _ = tolerance_thresholds(c)
        se = {str(st): [crlb_se(fd[st][i], sig, g) for i in range(D)] for st in FD_STEPS}
        main = se[str(FD_STEPS[0])]
        opt_i = int(np.argmin(main))
        pres[c["id"]] = {"tau": tau, "tau_halflog": tau_hl, "se": se,
                         "se_default": main[0], "se_best": main[ib],
                         "se_default_step1e-3": se[str(FD_STEPS[1])][0],
                         "se_best_step1e-3": se[str(FD_STEPS[1])][ib],
                         "ident_default": main[0] < tau, "ident_best": main[ib] < tau,
                         "ident_default_halflog": main[0] < tau_hl, "ident_best_halflog": main[ib] < tau_hl,
                         "n_designs_identifying": int(sum(x < tau for x in main)),
                         "info_optimal_design": labels[opt_i], "se_info_optimal": main[opt_i],
                         "info_only": c.get("info_only", False)}
    out["parameter"] = pres

    # ── MECHANISM claims: fits of every named alternative
    mres = {}
    alt_list = []
    for c in CLAIMS[world]:
        if c["type"] != "MECHANISM" or c.get("shared") or c.get("special"):
            continue
        for band, txt, impl in c["alts"]:
            alt_list.append((c["id"], impl))
        if c.get("claim_family"):
            alt_list.append((c["id"], c["claim_family"]))
    if world == "dark_matter":
        alt_list = [(cid, a) for cid, a in alt_list if a == "no_hidden"]
    sub = {"joint": list(range(D)), "solo_default": [0], "solo_best": [ib], "pair_default_best": sorted({0, ib})}

    # three_species: pick the least-distinguishable single-particle reassignment first
    if world == "three_species":
        part_true = np.array([0] * 10 + [1] * 10 + [2] * 10)
        m_scan = Model("scan", lambda u: k_poisson(1.0), charges=(lambda u, s: (
            s["pos0"], s["vel0"], s["masses"], J["jnp"].concatenate([u, J["jnp"].zeros(5)]), s["frc"])),
            n_steps=n_steps)
        scan = []
        for i in range(30):
            for sp in range(3):
                if sp == part_true[i]:
                    continue
                pt = part_true.copy()
                pt[i] = sp
                y = m_scan.sim(theta[pt], S_all)
                rms, _ = per_design_rms((y - Y) / sig * mask[:, :, None, None], mask, nvis)
                scan.append((float(np.sqrt(np.mean(rms ** 2))), i, sp))
        scan.sort()
        pbest = part_true.copy()
        pbest[scan[0][1]] = scan[0][2]
        out["ts_reassign_scan"] = {"best": scan[0], "worst": scan[-1], "n": len(scan)}
        alt_list = [(cid, ("partition:" + json.dumps(pbest.tolist())) if a == "reassign_one" else a)
                    for cid, a in alt_list]

    if base_only:
        geo = {}
        for key, i in (("DEFAULT", 0), ("BEST", ib)):
            if world in TWO_P:
                geo[key] = float(np.linalg.norm(truth_obs[i][:, 1] - truth_obs[i][:, 0], axis=-1).min())
        out["geometry_min_separation"] = geo
        out["alt_list"] = alt_list
        out["sub"] = sub
        out["n_steps"] = n_steps
        out["seconds_base"] = time.time() - t_start
        return out, {"Y": Y, "mask": mask}
    S_np = {k: np.asarray(v) for k, v in S_all.items()}
    tasks = [(world, cid, alt, n_steps, S_np, Y, mask, sig, sub, labels, smoke) for cid, alt in alt_list]
    if FIT_JOBS > 1 and len(tasks) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=min(FIT_JOBS, len(tasks)), mp_context=mp.get_context("spawn"),
                                    max_tasks_per_child=1) as pool:
            recs = list(pool.map(alt_fit_task, tasks))
    else:
        recs = [alt_fit_task(t) for t in tasks]
    for rec in recs:
        mres[f"{rec['claim']}::{rec['alt']}"] = rec
        log(f"[{world}] fitted {rec['claim']}::{rec['alt'][:20]} ({rec['joint'].get('seconds', 0):.0f}s)")
    out["mechanism"] = mres

    # geometry of BEST (for P4): min separation / probe radius over measurement times
    geo = {}
    for key, i in (("DEFAULT", 0), ("BEST", ib)):
        y = truth_obs[i]
        if world in TWO_P:
            geo[key] = float(np.linalg.norm(y[:, 1] - y[:, 0], axis=-1).min())
    out["geometry_min_separation"] = geo
    out["seconds"] = time.time() - t_start
    return out


# ── dark matter: hidden COUNT vs per-particle COUPLING at fixed total dark source 50 ──

CK = CACHE / "ckpt"


def _dm_elapsed(t0, commit=False):
    """Fit time spent in the count-vs-coupling stage, persisted across restarts (budget 3600 s)."""
    p = CK / "dark_matter" / "dmcount_elapsed.json"
    prev = json.loads(p.read_text())["seconds"] if p.exists() else 0.0
    tot = prev + (time.time() - t0)
    if commit:
        jdump({"seconds": tot}, p)
    return tot


def run_dm_count(wres, smoke=False, log=print, budget_s=3600.0):
    t0 = time.time()
    world = "dark_matter"
    J = jx()
    jnp = J["jnp"]
    sig = wres["sigma"]
    designs = wres["designs"]
    labels = [d["label"] for d in designs]
    ex = build_executor(world, np.array([1.0, 5.0]))
    systems = [system_one(world, ex, d) for d in designs]
    n_steps = n_steps_for(systems)
    S_all = stack_systems(systems)
    tm = truth_model(world, n_steps)
    Yt = executor_obs(ex, world, designs)
    Y = np.stack(Yt)                                            # all DM designs have 10 times
    mask = np.ones((len(designs), KMAX))
    # geometry: min distance of any probe to the dark centroid over the full trajectory
    trajf = J["jax"].jit(J["jax"].vmap(tm.traj_one, in_axes=(None, 0)))
    tr = np.asarray(trajf(jnp.asarray([1.0, 5.0]), S_all))     # (D, T, 35, 2)
    cen = tr[:, :, 20:30].mean(axis=2)
    dmin = np.linalg.norm(tr[:, :, 30:35] - cen[:, :, None], axis=-1).min(axis=(1, 2))
    rad = np.linalg.norm(tr[:, :, 20:30] - cen[:, :, None], axis=-1).max(axis=2)   # cluster radius(t)
    best = wres["ranker"]["best"]
    ib = labels.index(best)
    order = [int(i) for i in np.argsort(dmin) if int(i) not in (0, ib)]
    central = order[:3]
    chosen = list(dict.fromkeys([0, ib] + central))
    # chaos diagnostic: 1e-9 perturbation of the dark positions
    Sp = dict(S_all)
    pert = np.zeros((len(designs), 35, 2))
    pert[:, 20:30] = 1e-9 * np.random.default_rng(SEED).standard_normal((len(designs), 10, 2))
    Sp["pos0"] = S_all["pos0"] + jnp.asarray(pert)
    yp = tm.sim(np.array([1.0, 5.0]), Sp)
    chaos = (np.abs(yp - tm.sim(np.array([1.0, 5.0]), S_all)).max(axis=(1, 2, 3)) / 1e-9)
    info = {"chosen": [labels[i] for i in chosen], "central_designs": [labels[i] for i in central],
            "best": labels[ib],
            "min_probe_to_dark_centroid": {labels[i]: float(dmin[i]) for i in chosen},
            "min_probe_to_dark_centroid_all_sampled_median": float(np.median(dmin[1:])),
            "dark_cluster_max_radius_t0": float(rad[0, 0]),
            "dark_cluster_max_radius_over_run_DEFAULT": float(rad[0].max()),
            "perturbation_growth_1e-9": {labels[i]: float(chaos[i]) for i in chosen}}
    true_dark = np.asarray(ex._dark_positions_rel)
    rng = np.random.default_rng(SEED)
    n_restarts = 1 if smoke else 3
    results = {}
    for N in (5, 20):
        m = dm_count_model(N, n_steps)
        starts = []
        for _ in range(n_restarts):
            if N == 5:
                base = true_dark[rng.choice(10, 5, replace=False)]
            else:
                base = np.repeat(true_dark, 2, axis=0)
            starts.append((base + rng.normal(0.0, 0.1, base.shape)).ravel().tolist())
        bounds = ([-np.inf] * (2 * N), [np.inf] * (2 * N))
        for key, ids in [("solo_" + labels[i], [i]) for i in chosen] + [("joint_chosen", chosen)]:
            ckp = CK / world / f"dmcount_N{N}_{key}.json"
            if ckp.exists():
                results[f"N{N}::{key}"] = json.loads(ckp.read_text())
                continue
            if _dm_elapsed(t0) > budget_s:
                results[f"N{N}::{key}"] = {"ok": False, "skipped": "compute budget exhausted"}
                continue
            Ssub = {k: v[np.array(ids)] for k, v in S_all.items()}
            r = fit_model(m, starts, bounds, Ssub, Y[ids], mask[ids], sig, max_nfev=(5 if smoke else 60))
            r["labels"] = [labels[i] for i in ids]
            r.pop("u", None)
            results[f"N{N}::{key}"] = r
            if not smoke:
                jdump(r, ckp)
                _dm_elapsed(t0, commit=True)
                t0 = time.time()
            J["jax"].clear_caches()
            log(f"[dm_count] N={N} {key} done ({time.time() - t0:.0f}s)")
    info["fits"] = results
    info["seconds"] = _dm_elapsed(t0)
    return info


def worker(world, n_sampled, smoke):
    try:
        res = run_world(world, n_sampled=n_sampled, smoke=smoke,
                        log=lambda m: print(m, flush=True))
        if world == "dark_matter":
            res["dm_count"] = run_dm_count(res, smoke=smoke, log=lambda m: print(m, flush=True))
        jdump(res, CACHE / f"{'smoke_' if smoke else ''}world_{world}.json")
        return world, "ok", res.get("seconds")
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        (CACHE / f"{'smoke_' if smoke else ''}world_{world}.error.txt").write_text(tb)
        print(tb, flush=True)
        return world, "FAILED", tb[-500:]


# ════════════════════════════════════════════════════════════════════════════
# checkpointed resume driver (harness only; identical method)
# ════════════════════════════════════════════════════════════════════════════

def _safe(x):
    import hashlib
    return x if re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", x) else hashlib.md5(x.encode()).hexdigest()[:12]


def _pin_init(counter, lock):
    with lock:
        i = counter.value
        counter.value += 1
    try:
        os.sched_setaffinity(0, {i % (os.cpu_count() or 1)})
    except Exception:  # noqa: BLE001
        pass


def base_task(world):
    ckj, ckn = CK / world / "base.json", CK / world / "base.npz"
    if ckj.exists() and ckn.exists():
        return world, "cached"
    out, arr = run_world(world, log=lambda m: print(m, flush=True), base_only=True)
    ckn.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ckn, Y=arr["Y"], mask=arr["mask"])
    jdump(out, ckj)
    jx()["jax"].clear_caches()
    return world, "ok"


def fit_ck_path(world, cid, alt, key):
    return CK / world / f"fit_{_safe(cid)}__{_safe(alt)}__{key}.json"


def fit_task(args):
    world, cid, alt, key = args
    ck = fit_ck_path(world, cid, alt, key)
    if ck.exists():
        return args, "cached", 0.0
    t0 = time.time()
    base = json.loads((CK / world / "base.json").read_text())
    arr = np.load(CK / world / "base.npz")
    Y, mask = arr["Y"], arr["mask"]
    theta = np.array([v for _, v in THETA[world]], float)
    ex = build_executor(world, theta)
    S_all = stack_systems([system_one(world, ex, d) for d in base["designs"]])
    ids = base["sub"][key]
    labels = [d["label"] for d in base["designs"]]
    m, starts, bounds, pnames = alt_model(world, alt, int(base["n_steps"]))
    Ssub = {k: v[np.array(ids)] for k, v in S_all.items()}
    r = fit_model(m, starts, bounds, Ssub, Y[ids], mask[ids], float(base["sigma"]), max_nfev=200,
                  ck_prefix=str(ck)[:-5])
    r["labels"] = [labels[i] for i in ids]
    jdump({"claim": cid, "alt": alt, "params": pnames, "key": key, "fit": r}, ck)
    jx()["jax"].clear_caches()
    return args, "ok", time.time() - t0


def dmcount_task(_=None):
    ck = CK / "dark_matter" / "dmcount.json"
    if ck.exists():
        return "dmcount", "cached", 0.0
    base = json.loads((CK / "dark_matter" / "base.json").read_text())
    res = run_dm_count(base, log=lambda m: print(m, flush=True))
    jdump(res, ck)
    return "dmcount", "ok", res.get("seconds")


def assemble_world(world):
    out = json.loads((CK / world / "base.json").read_text())
    mres = {}
    for cid, alt in out["alt_list"]:
        rec = None
        for key in out["sub"]:
            d = json.loads(fit_ck_path(world, cid, alt, key).read_text())
            rec = rec or {"claim": cid, "alt": alt, "params": d["params"]}
            rec[key] = d["fit"]
        mres[f"{cid}::{alt}"] = rec
    out["mechanism"] = mres
    if world == "dark_matter":
        out["dm_count"] = json.loads((CK / "dark_matter" / "dmcount.json").read_text())
    jdump(out, CACHE / f"world_{world}.json")


def world_complete(world):
    if not (CK / world / "base.json").exists():
        return False
    base = json.loads((CK / world / "base.json").read_text())
    ok = all(fit_ck_path(world, cid, alt, key).exists() for cid, alt in base["alt_list"] for key in base["sub"])
    if world == "dark_matter":
        ok = ok and (CK / "dark_matter" / "dmcount.json").exists()
    return ok


def resume(worlds, jobs=4, skip_joint=(), fit_worlds=None, pin=True):
    import concurrent.futures as cf
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    counter, lock = ctx.Value("i", 0), ctx.Lock()
    kw = dict(initializer=_pin_init, initargs=(counter, lock)) if pin else {}
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, **kw) as pool:
        for f in cf.as_completed([pool.submit(base_task, w) for w in worlds]):
            print("== base", f.result(), flush=True)
        futs = []
        if "dark_matter" in worlds:
            futs.append(pool.submit(dmcount_task))
        tasks = []
        for w in (fit_worlds or worlds):
            base = json.loads((CK / w / "base.json").read_text())
            for cid, alt in base["alt_list"]:
                for key in base["sub"]:
                    if key == "joint" and w in skip_joint:
                        continue
                    if not fit_ck_path(w, cid, alt, key).exists():
                        tasks.append((w, cid, alt, key))
        cost = {"three_species": 0, "dark_matter": 1, "ether": 2}
        tasks.sort(key=lambda t: (t[3] != "joint", cost.get(t[0], 3)))
        futs += [pool.submit(fit_task, t) for t in tasks]
        for f in cf.as_completed(futs):
            a, st, sec = f.result()
            print(f"== fit {a} {st} {sec:.0f}s" if isinstance(sec, float) else f"== {a} {st}", flush=True)
    for w in (fit_worlds or worlds):
        if world_complete(w):
            assemble_world(w)
            print("== assembled", w, flush=True)
        else:
            print("== not yet complete", w, flush=True)


# ════════════════════════════════════════════════════════════════════════════
# sympy certificates
# ════════════════════════════════════════════════════════════════════════════

def sympy_certificates():
    import sympy as sp
    r, G, A = sp.symbols("r G A", positive=True)
    a, n = sp.symbols("alpha n", real=True)
    c_a = sp.gamma(1 - a) / (2 ** (2 * a) * sp.pi * sp.gamma(a))
    riesz = G * c_a * (2 - 2 * a) / r ** (3 - 2 * a)            # force_laws.riesz_2d_force (q=1)
    power = A / r ** n
    fwd = sp.simplify(riesz - power.subs({n: 3 - 2 * a, A: G * c_a * (2 - 2 * a)}))
    inv = sp.simplify(power - riesz.subs({a: (3 - n) / 2}).subs(
        {G: A / (c_a.subs(a, (3 - n) / 2) * (2 - 2 * (3 - n) / 2))}))
    out = {"riesz_to_power(n=3-2a, A=G c_a (2-2a))": str(fwd), "power_to_riesz(a=(3-n)/2)": str(inv),
           "certified": bool(fwd == 0 and inv == 0),
           "domain": "alpha in (0,1) <-> n in (1,3); the same map applies to the softened radius "
                     "r_eff = sqrt(r^2+0.05^2) because both kernels are evaluated on r_eff by make_acceleration_fn"}
    for al in (sp.Rational(1, 2), sp.Rational(3, 4)):
        out[f"alpha={al}"] = {"n": str(3 - 2 * al), "A_over_G": str(sp.nsimplify(sp.simplify(c_a.subs(a, al) * (2 - 2 * al)))),
                              "A_over_G_float": float(c_a.subs(a, al) * (2 - 2 * al))}
    # numeric: the benchmark's own kernel vs the power law on a grid
    J = jx()
    FL, jnp = J["FL"], J["jnp"]
    rr = np.logspace(-1.3, 1.5, 200)
    for al in (0.5, 0.75):
        f = np.asarray(FL.riesz_2d_force(jnp.asarray(rr), 1.0, 1.0, 1.0, 1.0, G=1.0, alpha=al))
        Aal = float(c_a.subs(a, al) * (2 - 2 * al))
        out[f"numeric_max_rel_diff_alpha={al}"] = float(np.max(np.abs(f - Aal * rr ** (-(3 - 2 * al))) / f))
    return out


# ════════════════════════════════════════════════════════════════════════════
# classification, verdicts, prediction statistics, README
# ════════════════════════════════════════════════════════════════════════════

ORDER = ["EQUIVALENT (certified)", "EQUIVALENT (numeric)", "INDETERMINATE",
         "DISTINGUISHABLE-BY-DESIGN", "DISTINGUISHABLE"]


def classify(r_default, r_sampled_max, r_all_max):
    if r_all_max < RESID_EQUIV:
        return "EQUIVALENT (numeric)"
    if r_default > RESID_DIST:
        return "DISTINGUISHABLE"
    if r_sampled_max > RESID_DIST:
        return "DISTINGUISHABLE-BY-DESIGN"
    return "INDETERMINATE"


def summarize_alt(rec, best_label):
    j = rec["joint"]
    if not j.get("ok"):
        return {"literal": "FAILED", "solo": "FAILED"}
    rms = np.array(j["rms"])
    labels = j["labels"]
    ib = labels.index(best_label)
    lit = classify(rms[0], rms[1:].max(), rms.max())
    out = {"r_default_joint": float(rms[0]), "r_best_joint": float(rms[ib]),
           "r_sampled_max_joint": float(rms[1:].max()), "r_all_max_joint": float(rms.max()),
           "r_sampled_median_joint": float(np.median(rms[1:])),
           "n_sampled_above_noise_joint": int((rms[1:] > RESID_DIST).sum()),
           "joint_params": j.get("u"), "literal": lit}
    for key in ("solo_default", "solo_best", "pair_default_best"):
        if key in rec and rec[key].get("ok"):
            out[f"r_{key}"] = float(max(rec[key]["rms"]))
    if "r_solo_default" in out:
        rd = out["r_solo_default"]
        if rms.max() < RESID_EQUIV:
            out["solo"] = "EQUIVALENT (numeric)"
        elif rd > RESID_DIST:
            out["solo"] = "DISTINGUISHABLE"
        elif rms[1:].max() > RESID_DIST:
            out["solo"] = "DISTINGUISHABLE-BY-DESIGN"
        else:
            out["solo"] = "INDETERMINATE"
    return out


def least(classes):
    cl = [c for c in classes if c in ORDER]
    return min(cl, key=ORDER.index) if cl else "FAILED"


def evaluate_world(w, res, certs):
    best = res["ranker"]["best"]
    claims = []
    for c in CLAIMS[w]:
        row = {"id": c["id"], "type": c["type"], "frag": c["frag"]}
        if c["type"] == "PARAMETER":
            p = res["parameter"][c["id"]]
            row.update(p)
            if not p["ident_best"]:
                row["class"] = "NON-IDENTIFIABLE at BEST"
            elif not p["ident_default"]:
                row["class"] = "IDENTIFIABLE only at BEST"
            else:
                row["class"] = "IDENTIFIABLE at DEFAULT"
        elif c["type"] == "MECHANISM":
            if c.get("special") == "dm_count":
                row.update(dm_classify(res.get("dm_count")))
            else:
                src = c.get("shared", c["id"])
                allowed = {impl for _, _, impl in c["alts"]}
                alts = {}
                for key, rec in res["mechanism"].items():
                    cid, alt = key.split("::", 1)
                    if cid != src:
                        continue
                    ok = ("same-as-ED-1" in allowed or alt in allowed or alt == c.get("claim_family")
                          or (alt.startswith("partition:") and "reassign_one" in allowed))
                    if not ok:
                        continue
                    s = summarize_alt(rec, best)
                    if c.get("claim_family") and alt == c["claim_family"]:
                        row["claim_family_fit"] = s
                        continue
                    if c.get("certify") == alt and certs.get("certified"):
                        s["literal"] = s["solo"] = "EQUIVALENT (certified)"
                    alts[alt] = s
                row["alts"] = alts
                row["class"] = least([a["literal"] for a in alts.values()])
                row["class_solo"] = least([a.get("solo", a["literal"]) for a in alts.values()])
                if c.get("claim_family") and "claim_family_fit" in row:
                    cf = row["claim_family_fit"]
                    row["contradicted"] = ("CONTRADICTED at DEFAULT" if cf["r_default_joint"] > RESID_DIST
                                           else ("CONTRADICTED by design" if cf["r_sampled_max_joint"] > RESID_DIST
                                                 else "not contradicted"))
        claims.append(row)

    def verdict(key):
        scored = [r for r in claims if r["type"] in ("PARAMETER", "MECHANISM")]
        nonid = any(r["type"] == "MECHANISM" and r.get(key, r.get("class", "")).startswith("EQUIVALENT")
                    for r in scored) or any(r["type"] == "PARAMETER" and not r["ident_best"] for r in scored)
        if nonid:
            return "NONID"
        resolv = any(r["type"] == "MECHANISM" and r.get(key, r.get("class")) == "DISTINGUISHABLE-BY-DESIGN"
                     for r in scored) or any(r["type"] == "PARAMETER" and r["ident_best"] and not r["ident_default"]
                                             for r in scored)
        return "RESOLVABLE" if resolv else "ID"

    v = verdict("class")
    v_solo = verdict("class_solo")
    certified = any(r["type"] == "MECHANISM" and r.get("class") == "EQUIVALENT (certified)" for r in claims)
    # resolved-at-DEFAULT / at-BEST summaries (registered-literal residuals)
    fail_def, fail_best = [], []
    for r in claims:
        if r["type"] == "PARAMETER":
            if not r["ident_default"]:
                fail_def.append(r["id"])
            if not r["ident_best"]:
                fail_best.append(r["id"])
        elif r["type"] == "MECHANISM":
            if r.get("special_dm"):
                if not r.get("dist_default"):
                    fail_def.append(r["id"])
                if not r.get("dist_best"):
                    fail_best.append(r["id"])
                continue
            for a, s in r.get("alts", {}).items():
                if s["literal"].startswith("EQUIVALENT") or s.get("r_default_joint", 0) <= RESID_DIST:
                    fail_def.append(f"{r['id']}:{a}")
                if s["literal"].startswith("EQUIVALENT") or s.get("r_best_joint", 0) <= RESID_DIST:
                    fail_best.append(f"{r['id']}:{a}")
    return {"claims": claims, "verdict": v, "verdict_solo_default": v_solo, "certified_nonid": certified and v == "NONID",
            "unresolved_at_default": sorted(set(fail_def)), "unresolved_at_best": sorted(set(fail_best)),
            "best": best}


def dm_classify(dm):
    if not dm:
        return {"class": "NOT SCORED", "class_solo": "NOT SCORED", "special_dm": True}
    fits = dm["fits"]
    out = {"special_dm": True, "dm_fits": {}}
    per_N = {}
    for N in (5, 20):
        d = {}
        for key, r in fits.items():
            if not key.startswith(f"N{N}::"):
                continue
            if r.get("ok"):
                d[key.split("::")[1]] = float(max(r["rms"])) if key.endswith("joint_chosen") is False else r["rms"]
        per_N[N] = d
    out["dm_fits"] = per_N
    classes = []
    dd, db = [], []
    chosen = dm["chosen"]
    for N in (5, 20):
        d = per_N[N]
        rdef = d.get("solo_DEFAULT")
        rbest = d.get(f"solo_{dm['best']}")
        others = [d.get(f"solo_{l}") for l in chosen if l != "DEFAULT" and d.get(f"solo_{l}") is not None]
        if rdef is None:
            classes.append("NOT SCORED")
            continue
        allv = [rdef] + others
        if max(allv) < RESID_EQUIV:
            classes.append("EQUIVALENT (numeric)")
        elif rdef > RESID_DIST:
            classes.append("DISTINGUISHABLE")
        elif others and max(others) > RESID_DIST:
            classes.append("DISTINGUISHABLE-BY-DESIGN")
        else:
            classes.append("INDETERMINATE")
        dd.append(rdef > RESID_DIST)
        db.append(rbest is not None and rbest > RESID_DIST)
    out["class"] = out["class_solo"] = least(classes) if "NOT SCORED" not in classes else "NOT SCORED"
    out["class_by_N"] = dict(zip(("N=5", "N=20"), classes))
    out["dist_default"] = all(dd) if dd else False
    out["dist_best"] = all(db) if db else False
    return out


EVAL_WORLDS = list(PUBLIC_WORLDS)

PREDICTED = {"fractional": "NONID", "circle": "NONID", "extra_dimensions": "RESOLVABLE",
             "gravity": "ID", "yukawa": "ID", "hubble": "ID", "ether": "ID", "oscillator": "ID",
             "three_species": "ID", "coulomb": "ID"}


def leaderboard_stats(cells):
    st = {}
    for w in PUBLIC_WORLDS:
        cs = [c for c in cells if c["world"] == w]
        ex = [c["expl"] for c in cs]
        ge = [c["geom"] for c in cs]
        tg = [c for c in cs if c["geom"] <= 0.1]
        st[w] = {"n_models": len(cs), "max_expl": max(ex), "mean_expl": float(np.mean(ex)),
                 "median_geom": float(np.median(ge)), "n_traj_good": len(tg),
                 "n_expl_fail": sum(1 for c in cs if c["expl"] < 0.9),
                 "n_tg_and_ef": sum(1 for c in tg if c["expl"] < 0.9)}
    return st


def predictions(wv, lbs, results):
    from scipy.stats import fisher_exact, mannwhitneyu
    out = {}
    # P1
    rows = []
    for w, pv in PREDICTED.items():
        if w not in EVAL_WORLDS:
            continue
        rows.append((w, pv, wv[w]["verdict"], pv == wv[w]["verdict"]))
    nmatch = sum(r[3] for r in rows)
    out["P1"] = {"rows": rows, "n_match": nmatch, "n": len(rows),
                 "outcome": "CONFIRMED" if nmatch >= 9 else ("FALSIFIED" if nmatch < 8 else "NOT CONFIRMED (8/10: between the pass and falsification thresholds)"),
                 "dark_matter_computed": wv.get("dark_matter", {}).get("verdict", "not evaluated"),
                 "rows_solo_variant": [(w, pv, wv[w]["verdict_solo_default"], pv == wv[w]["verdict_solo_default"])
                                       for w, pv in PREDICTED.items() if w in EVAL_WORLDS]}
    # P2
    grp_n = [w for w in EVAL_WORLDS if wv[w]["verdict"] in ("NONID", "RESOLVABLE")]
    grp_i = [w for w in EVAL_WORLDS if wv[w]["verdict"] == "ID"]
    xn = [lbs[w]["max_expl"] for w in grp_n]
    xi = [lbs[w]["max_expl"] for w in grp_i]
    p2 = {"NONID_or_RESOLVABLE": {w: lbs[w]["max_expl"] for w in grp_n},
          "ID": {w: lbs[w]["max_expl"] for w in grp_i},
          "verdicts_as_predicted": all(wv[w]["verdict"] == PREDICTED[w] for w in PREDICTED)}
    if xn and xi:
        u = mannwhitneyu(xn, xi, alternative="less")
        p2.update({"U": float(u.statistic), "p_one_sided": float(u.pvalue)})
        nonid_max = [lbs[w]["max_expl"] for w in EVAL_WORLDS if wv[w]["verdict"] == "NONID"]
        p2["perfect_separation_NONID_vs_ID"] = bool(nonid_max and max(nonid_max) < min(xi))
    else:
        p2["U"] = None
        p2["p_one_sided"] = None
        p2["note"] = "one group empty; test undefined"
    out["P2"] = p2
    # P3
    cert = [w for w in EVAL_WORLDS if wv[w]["certified_nonid"]]
    idw = [w for w in EVAL_WORLDS if wv[w]["verdict"] == "ID"]
    a = sum(lbs[w]["n_tg_and_ef"] for w in cert)
    b = sum(lbs[w]["n_traj_good"] - lbs[w]["n_tg_and_ef"] for w in cert)
    c = sum(lbs[w]["n_tg_and_ef"] for w in idw)
    d = sum(lbs[w]["n_traj_good"] - lbs[w]["n_tg_and_ef"] for w in idw)
    p3 = {"certified_nonid_worlds": cert, "id_worlds": idw, "table": [[a, b], [c, d]]}
    if (a + b) > 0 and (c + d) > 0:
        orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        p3.update({"frac_certified": a / (a + b), "frac_id": c / (c + d), "odds_ratio": float(orr),
                   "p_one_sided": float(p)})
        p3["outcome"] = ("CONFIRMED" if (a / (a + b) > c / (c + d) and p < 0.05) else
                         ("FALSIFIED" if a / (a + b) <= c / (c + d) else "NOT CONFIRMED (higher but p >= 0.05)"))
    else:
        p3["outcome"] = "UNDEFINED (empty group)"
    out["P3"] = p3
    # P4
    p4 = {}
    for w in EVAL_WORLDS:
        if wv[w]["verdict"] != "RESOLVABLE":
            continue
        un_best = wv[w]["unresolved_at_best"]
        claims = wv[w]["claims"]
        dbd = [r["id"] for r in claims if r.get("class") in ("DISTINGUISHABLE-BY-DESIGN", "IDENTIFIABLE only at BEST")]
        ok_claims = {}
        for r in claims:
            if r["type"] == "PARAMETER" and r["id"] in dbd:
                ok_claims[r["id"]] = r["ident_best"]
            elif r["type"] == "MECHANISM" and r["id"] in dbd:
                if r.get("special_dm"):
                    ok_claims[r["id"]] = r.get("dist_best", False)
                else:
                    ok_claims[r["id"]] = all(s.get("r_best_joint", 0) > RESID_DIST for s in r["alts"].values()
                                             if s["literal"] == "DISTINGUISHABLE-BY-DESIGN")
        entry = {"best": wv[w]["best"], "design_dependent_claims": dbd, "resolved_at_best": ok_claims,
                 "unresolved_at_best_any": un_best}
        if w == "extra_dimensions":
            entry["best_min_separation"] = results[w]["geometry_min_separation"].get("BEST")
            entry["default_min_separation"] = results[w]["geometry_min_separation"].get("DEFAULT")
            entry["probe_r_le_1.5"] = (entry["best_min_separation"] or 99) <= 1.5
        entry["ok"] = all(ok_claims.values()) and (entry.get("probe_r_le_1.5", True))
        p4[w] = entry
    out["P4"] = {"worlds": p4, "outcome": ("CONFIRMED" if p4 and all(e["ok"] for e in p4.values()) else
                                          ("FALSIFIED" if p4 else "VACUOUS (no RESOLVABLE world)"))}
    return out


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "yes" if x else "no"
    if not math.isfinite(x):
        return "inf"
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def aggregate(provisional=False):
    global EVAL_WORLDS
    certs = sympy_certificates()
    results, missing = {}, []
    for w in PUBLIC_WORLDS:
        p = CACHE / f"world_{w}.json"
        if p.exists():
            results[w] = json.loads(p.read_text())
        else:
            missing.append(w)
    if missing and not provisional:
        raise SystemExit(f"missing world results: {missing}")
    EVAL_WORLDS = [w for w in PUBLIC_WORLDS if w in results]
    wv = {w: evaluate_world(w, results[w], certs) for w in EVAL_WORLDS}
    cells = read_leaderboard()
    lbs = leaderboard_stats(cells)
    pr = predictions(wv, lbs, results)
    summary = {"certificates": certs, "world_verdicts": wv, "leaderboard": lbs, "predictions": pr,
               "worlds_evaluated": EVAL_WORLDS, "worlds_missing": missing,
               "pins": {"DiscoverPhysics": git_head(DP), "Leaderboard": git_head(LB),
                        "ascension": git_head(REPO)}}
    if provisional:
        summary["PROVISIONAL"] = (f"PROVISIONAL: {len(EVAL_WORLDS)} of 11 worlds; missing {missing}. Prediction "
                                  "statistics below are computed on the available worlds only and are NOT the "
                                  "registered outcomes.")
        tag = f"aggregate_provisional_{len(EVAL_WORLDS)}worlds"
        jdump(summary, OUT / f"{tag}.json")
        write_readme(results, wv, lbs, pr, certs, path=OUT / f"{tag}.md", banner=summary["PROVISIONAL"])
        return summary
    jdump(summary, OUT / "summary.json")
    write_tables(results, wv, lbs)
    write_readme(results, wv, lbs, pr, certs)
    return summary


def write_tables(results, wv, lbs):
    with (OUT / "parity.tsv").open("w") as f:
        f.write("world\tsigma\trunner_vs_shipped_DEFAULT_maxabs\tfast_vs_executor_maxabs_65designs\t"
                "fast_vs_executor_max_over_sigma\tFD_vs_AD_median_rel\tFD_vs_AD_max_rel\tAD_params\n")
        for w in EVAL_WORLDS:
            r = results[w]
            fa = r["fd_vs_ad"]["rel_frobenius_per_design"]
            f.write(f"{w}\t{r['sigma']:.6f}\t{r['parity']['runner_vs_shipped_default_maxabs']:.3e}\t"
                    f"{r['parity']['fast_vs_executor_maxabs']:.3e}\t{r['parity']['fast_vs_executor_maxabs_over_sigma']:.3e}\t"
                    f"{np.median(fa):.3e}\t{np.max(fa):.3e}\t{','.join(r['fd_vs_ad']['params'])}\n")
    with (OUT / "crlb.tsv").open("w") as f:
        f.write("world\tclaim\tdesign\tSE_step1e-4\tSE_step1e-3\ttau\ttau_halflog\tidentifiable(tau)\n")
        for w in EVAL_WORLDS:
            r = results[w]
            labels = [d["label"] for d in r["designs"]]
            for cid, p in r["parameter"].items():
                a, b = p["se"][str(FD_STEPS[0])], p["se"][str(FD_STEPS[1])]
                for i, lab in enumerate(labels):
                    tag = lab + (" (BEST)" if lab == r["ranker"]["best"] else "")
                    f.write(f"{w}\t{cid}\t{tag}\t{fnum(a[i]):.4e}\t{fnum(b[i]):.4e}\t{p['tau']:.4f}\t"
                            f"{p['tau_halflog']:.4f}\t{fnum(a[i]) < p['tau']}\n")
    with (OUT / "mechanism_residuals.tsv").open("w") as f:
        f.write("world\tclaim\talternative\tfit\tdesign\trms_over_sigma\tmax_over_sigma\tfit_params\n")
        for w in EVAL_WORLDS:
            for key, rec in results[w]["mechanism"].items():
                for fk in ("joint", "solo_default", "solo_best", "pair_default_best"):
                    fr = rec.get(fk)
                    if not fr or not fr.get("ok"):
                        if fr:
                            f.write(f"{w}\t{rec['claim']}\t{rec['alt']}\t{fk}\t—\tFAILED\t\t\n")
                        continue
                    for lab, rm, mx in zip(fr["labels"], fr["rms"], fr["max"]):
                        f.write(f"{w}\t{rec['claim']}\t{rec['alt']}\t{fk}\t{lab}\t{rm:.4e}\t{mx:.4e}\t"
                                f"{json.dumps(dict(zip(rec['params'], [round(x, 6) for x in fr['u']])))}\n")
    with (OUT / "world_verdicts.tsv").open("w") as f:
        f.write("world\tpredicted\tverdict_registered\tverdict_soloDEFAULT_variant\tBEST\tmax_expl\tmean_expl\t"
                "median_geom\tn_traj_good\tn_expl_fail\tn_TG_and_EF\n")
        for w in EVAL_WORLDS:
            s = lbs[w]
            f.write(f"{w}\t{PREDICTED.get(w, 'RESOLVABLE or NONID')}\t{wv[w]['verdict']}\t"
                    f"{wv[w]['verdict_solo_default']}\t{wv[w]['best']}\t{s['max_expl']:.2f}\t{s['mean_expl']:.3f}\t"
                    f"{s['median_geom']:.3f}\t{s['n_traj_good']}\t{s['n_expl_fail']}\t{s['n_tg_and_ef']}\n")
    dm = results.get("dark_matter", {}).get("dm_count")
    if dm:
        with (OUT / "dm_count_coupling.tsv").open("w") as f:
            f.write("fit\tdesigns\trms_over_sigma_per_design\tcost\tnfev_per_restart\tseconds\n")
            for k, r in dm["fits"].items():
                if not r.get("ok"):
                    f.write(f"{k}\t\t{r.get('skipped', 'FAILED')}\t\t\t\n")
                    continue
                f.write(f"{k}\t{','.join(r['labels'])}\t{','.join(f'{x:.3f}' for x in r['rms'])}\t"
                        f"{r['cost']:.4e}\t{','.join(str(l.get('nfev')) for l in r['log'])}\t{r['seconds']:.0f}\n")


def write_readme(results, wv, lbs, pr, certs, path=None, banner=None):
    L = []
    A = L.append
    if banner:
        A(f"> **{banner}**\n")
    pins = (git_head(DP), git_head(LB))
    A("# EXP-263 — Identifiability audit of the DiscoverPhysics benchmark\n")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by "
      "`scripts/exp263_discoverphysics_audit.py`. Pre-registration: `predictions.md`, block "
      "\"Pre-registered block added 2026-09-28 (Session 047 — EXP-263 ...)\" (commit 8f746eb). "
      "Retrospective test: the per-world outcomes were seen before registration; the verdicts were not.\n")
    A(f"- DiscoverPhysics pinned `{pins[0]}` (engine `nbody`, yoshida4, dt = 0.005, softening 0.05); "
      f"leaderboard pinned `{pins[1]}`.")
    A("- Step 1 was run first: `leaderboard_cells.tsv` (13 models x 11 worlds = 143 cells, per-cell means "
      "over 5 seeds as published) was copied from the pinned leaderboard before any verdict was computed.")
    A("- No LLM anywhere. Noise: sigma = 0.05 * sqrt(Var_GT[world]) (the benchmark's `noise_frac` "
      "definition in `scripts/run_benchmark.py`), i.i.d. Gaussian on every agent-visible position.\n")

    A("## Per-world results\n")
    A("| world | predicted | **verdict (registered)** | variant: solo-DEFAULT fits | BEST | claims unresolved at DEFAULT | "
      "unresolved at BEST | max expl | mean expl | median geom err | TG | EF | TG&EF |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for w in EVAL_WORLDS:
        s = lbs[w]
        v = wv[w]
        A(f"| {w} | {PREDICTED.get(w, 'RES or NONID')} | **{v['verdict']}** | {v['verdict_solo_default']} | "
          f"{v['best']} | {', '.join(v['unresolved_at_default']) or '—'} | {', '.join(v['unresolved_at_best']) or '—'} | "
          f"{s['max_expl']:.2f} | {s['mean_expl']:.2f} | {s['median_geom']:.3f} | {s['n_traj_good']} | "
          f"{s['n_expl_fail']} | {s['n_tg_and_ef']} |")
    A("\nTG = trajectory-good cells (geom_pos_err <= 0.1); EF = explanation-fail cells (score < 0.9); "
      "13 model cells per world.\n")

    A("## Per-claim results\n")
    A("PARAMETER: relative CRLB standard error of the graded quantity (log units for multiplicative "
      "tolerances, absolute for alpha ranges) for ONE experiment divided by sqrt(16); identifiable if "
      "SE < tau. Step 1e-4 (registered) and 1e-3 (stability) shown. MECHANISM: RMS residual / sigma of the "
      "best-fit named alternative; `joint` = one fit over DEFAULT + 64 sampled designs (registered), "
      "`solo` = the alternative refitted to that design alone (diagnostic).\n")
    for w in EVAL_WORLDS:
        A(f"### {w} — verdict **{wv[w]['verdict']}** (BEST = {wv[w]['best']})\n")
        r = results[w]
        A(f"sigma = {r['sigma']:.4f}; ranker: {r['ranker']['n_tied_with_best']} design(s) tied at the top score; "
          f"{r['ranker']['rationale'][:160]}\n")
        A("| claim | type | class | numbers |")
        A("|---|---|---|---|")
        for c in wv[w]["claims"]:
            if c["type"] == "PARAMETER":
                nums = (f"SE DEFAULT {fmt(c['se_default'])} (1e-3: {fmt(c['se_default_step1e-3'])}); "
                        f"SE BEST {fmt(c['se_best'])} (1e-3: {fmt(c['se_best_step1e-3'])}); tau {c['tau']:.3f} "
                        f"(half-log {c['tau_halflog']:.3f}); {c['n_designs_identifying']}/65 designs identify; "
                        f"info-optimal {c['info_optimal_design']} SE {fmt(c['se_info_optimal'])}")
                A(f"| {c['id']} | PARAMETER | {c['class']} | {nums} |")
            elif c["type"] == "MECHANISM":
                if c.get("special_dm"):
                    A(f"| {c['id']} | MECHANISM (count vs coupling) | {c['class']} | "
                      f"{json.dumps(c.get('class_by_N'))}; residuals: {json.dumps(c.get('dm_fits'))[:600]} |")
                    continue
                parts = []
                for a, s in c.get("alts", {}).items():
                    if s["literal"] == "FAILED":
                        parts.append(f"{a}: FAILED")
                        continue
                    parts.append(f"{a[:28]}: {s['literal']} [joint DEF {fmt(s['r_default_joint'])}, BEST "
                                 f"{fmt(s['r_best_joint'])}, max {fmt(s['r_sampled_max_joint'])}, "
                                 f"{s['n_sampled_above_noise_joint']}/64 >1; solo DEF {fmt(s.get('r_solo_default'))}, "
                                 f"solo BEST {fmt(s.get('r_solo_best'))}, DEF+BEST {fmt(s.get('r_pair_default_best'))}]")
                extra = ""
                if c.get("contradicted"):
                    cf = c["claim_family_fit"]
                    extra = (f"; top-band claim's own family (decay no faster than 1/r): {c['contradicted']} "
                             f"(joint DEF {fmt(cf['r_default_joint'])}, max {fmt(cf['r_all_max_joint'])})")
                A(f"| {c['id']} | MECHANISM | {c['class']} (solo variant: {c.get('class_solo')}) | "
                  f"{'<br>'.join(parts)}{extra} |")
            else:
                A(f"| {c['id']} | {c['type']} | — | {CLAIMS_BY_ID[w][c['id']].get('why', '')} |")
        for cid, p in r["parameter"].items():
            if p.get("info_only"):
                A(f"\nInformation only ({cid}, no tolerance registered): SE(log G0) DEFAULT {fmt(p['se_default'])}, "
                  f"BEST {fmt(p['se_best'])}.")
        A("")

    A("## Predictions\n")
    p1 = pr["P1"]
    A(f"**P-EXP-263-1 (verdicts): {p1['outcome']}** — {p1['n_match']}/{p1['n']} match "
      f"(pass >= 9, falsified < 8). Dark matter (not predicted): {p1['dark_matter_computed']}.\n")
    A("| world | predicted | computed | match | solo-DEFAULT variant |")
    A("|---|---|---|---|---|")
    for (w, pv, cv, m), (_, _, cs, _) in zip(p1["rows"], p1["rows_solo_variant"]):
        A(f"| {w} | {pv} | {cv} | {'yes' if m else 'NO'} | {cs} |")
    p2 = pr["P2"]
    A(f"\n**P-EXP-263-2 (ceilings, consistency check only):** NONID∪RESOLVABLE maxima "
      f"{json.dumps(p2['NONID_or_RESOLVABLE'])} vs ID maxima {json.dumps(p2['ID'])}; "
      f"Mann–Whitney U = {p2.get('U')}, one-sided p = {fmt(p2.get('p_one_sided'), 4)}; "
      f"perfect separation of NONID below every ID world: {p2.get('perfect_separation_NONID_vs_ID')}; "
      f"condition 'verdicts as predicted' met: {p2['verdicts_as_predicted']}.\n")
    p3 = pr["P3"]
    A(f"**P-EXP-263-3 (Proposition 3 signature): {p3['outcome']}** — certified-NONID worlds "
      f"{p3['certified_nonid_worlds']} vs ID worlds {p3['id_worlds']}; 2x2 [[TG&EF, TG&not-EF] certified; "
      f"[.., ..] ID] = {p3['table']}; fractions {fmt(p3.get('frac_certified'))} vs {fmt(p3.get('frac_id'))}; "
      f"Fisher one-sided p = {fmt(p3.get('p_one_sided'), 5)}.\n")
    p4 = pr["P4"]
    A(f"**P-EXP-263-4 (ranker finds the resolving design): {p4['outcome']}**\n")
    for w, e in p4["worlds"].items():
        A(f"- {w}: BEST = {e['best']}; design-dependent claims {e['design_dependent_claims']}; resolved at BEST "
          f"{json.dumps(e['resolved_at_best'])}" +
          (f"; min probe separation at BEST {fmt(e.get('best_min_separation'))} (DEFAULT "
           f"{fmt(e.get('default_min_separation'))}); r <= 1.5: {e.get('probe_r_le_1.5')}" if w == "extra_dimensions" else ""))
    A("\n## Symbolic certificate (sympy)\n")
    A("```\n" + json.dumps(certs, indent=1) + "\n```\n")
    A("## Parity and finite-difference stability\n")
    A("| world | runner vs shipped executor (DEFAULT) | fast path vs executor (65 designs), max abs | / sigma | "
      "FD(1e-4) vs AD, median rel | max rel |")
    A("|---|---|---|---|---|---|")
    for w in EVAL_WORLDS:
        r = results[w]
        fa = r["fd_vs_ad"]["rel_frobenius_per_design"]
        A(f"| {w} | {fmt(r['parity']['runner_vs_shipped_default_maxabs'])} | {fmt(r['parity']['fast_vs_executor_maxabs'])} | "
          f"{fmt(r['parity']['fast_vs_executor_maxabs_over_sigma'])} | {fmt(float(np.median(fa)))} | {fmt(float(np.max(fa)))} |")
    dm = results.get("dark_matter", {}).get("dm_count", {})
    A("\n## Dark matter: hidden count vs per-particle coupling (total dark source 50)\n")
    if dm:
        A(f"Designs fitted: {dm['chosen']} (DEFAULT, BEST, and the three sampled designs whose probes pass "
          f"closest to the dark centroid). Min probe-to-dark-centroid distance: "
          f"{json.dumps({k: round(v, 3) for k, v in dm['min_probe_to_dark_centroid'].items()})}; median over "
          f"sampled designs {dm['min_probe_to_dark_centroid_all_sampled_median']:.3f}; dark cluster radius at t=0 "
          f"{dm['dark_cluster_max_radius_t0']:.3f}, max over the DEFAULT run {dm['dark_cluster_max_radius_over_run_DEFAULT']:.3f}. "
          f"Growth of a 1e-9 perturbation of the dark positions (max probe/visible displacement / 1e-9): "
          f"{json.dumps({k: float(f'{v:.3g}') for k, v in dm['perturbation_growth_1e-9'].items()})}. Compute "
          f"{dm['seconds']:.0f} s. Full residuals in `dm_count_coupling.tsv`.\n")
    A("## Implementation choices (literal readings of ambiguous points)\n")
    for i, ch in enumerate(IMPLEMENTATION_CHOICES, 1):
        A(f"{i}. {ch}")
    A("\n## Limitations\n")
    for i, ch in enumerate(LIMITATIONS, 1):
        A(f"{i}. {ch}")
    A("\n## Files\n")
    A("- `leaderboard_cells.tsv` — per-(model, world) outcomes copied first from the pinned leaderboard.")
    A("- `claims.tsv` — every top-band claim, verbatim, typed, with named alternatives and tolerances.")
    A("- `crlb.tsv` — per-design CRLB SE for every PARAMETER claim at both FD steps.")
    A("- `mechanism_residuals.tsv` — per-design residuals for every alternative and fit type.")
    A("- `dm_count_coupling.tsv`, `parity.tsv`, `world_verdicts.tsv`, `summary.json`, `cache/world_*.json`.")
    (path or (OUT / "README.md")).write_text("\n".join(L) + "\n")


CLAIMS_BY_ID = {w: {c["id"]: c for c in CLAIMS[w]} for w in CLAIMS}

IMPLEMENTATION_CHOICES = [
    "Claim typing (fixed before any computation): PARAMETER = numeric constant with a stated tolerance; "
    "MECHANISM = top-band structure for which some lower band names a specific implementable alternative; "
    "every named alternative is fitted and the claim takes the LEAST distinguishable class. Claims whose "
    "lower-band counterpart is only an omission ('omits', 'fails to characterise') or a generic 'wrong "
    "operator family' are UNTYPED; 'static vs time-evolving/wave-like' claims are NOT SCORED (the pinned "
    "nbody engine cannot simulate a time-evolving field). Neither enters the verdict.",
    "p1/p2 role claims are MECHANISM claims where the rubric's lower band names 'muddles the p1/p2 roles' "
    "(gravity, yukawa, oscillator: implemented as a role swap; gravity 4–6 'fails to distinguish' as a "
    "product; coulomb 7–9 as p1/p2, p2/p1 and additive). The fractional and extra_dimensions rubrics name no "
    "p1/p2 alternative, so there the role claim is UNTYPED.",
    "fractional/circle operator claim: the alternatives are the rubric's 'ordinary power-law' / "
    "'anomalous-decay operator' (A r^-n, both free) and 'standard Laplacian' (A/r). The 'direction of the "
    "anomaly' claims are tested twice: the named lower-band alternative (opposite direction, n in [1,4]) "
    "and the top-band claim's own family (n in [0,1]); a claim refuted by the true world is flagged "
    "CONTRADICTED (outside the registered rule; it does not change the verdict).",
    "DEFAULT = the world's own `experiment_format` where the WORLDS entry defines one (coulomb, circle, "
    "dark_matter, three_species, ether, hubble); otherwise evaluator `_DEFAULT_TEST_CASES[0]` "
    "(p1 = p2 = 1, pos2 = [3,0], v2 = [0,0.5], times 1..10) for gravity, yukawa, fractional, oscillator "
    "(start_time 0) and extra_dimensions. (get_world() shows those agents a generic example with "
    "p1 = p2 = 1, pos2 = [3,0], times [0.5,1,2]; the registered text names test case 1.)",
    "Design sampling (seed 263, numpy default_rng, one fresh stream per world, uniform): 2-particle p1, p2 "
    "U[0.1,10], pos2 U[-10,10]^2, v2 U[-5,5]^2, duration U[5,10], oscillator start_time U[0,10] (range not "
    "documented; >= 2.5 periods); probes (dark_matter, three_species) positions U[-15,15]^2, velocities "
    "U[-2,2]^2; ether/hubble use their own documented interface (probes_with_masses: U[-22,22]^2, "
    "U[-3,3]^2, masses drawn from {1,2,4}); circle ring_radius U[2,10], v_tang U[0,2]. Probe/circle "
    "durations fixed at 10 (the documented minimum). Always 10 evenly spaced measurement times ending at "
    "the duration.",
    "World constants patched on the constructed executor (operator params, class attributes, a subclass "
    "for extra_dimensions). Background initial conditions computed in __init__ from the constants "
    "(hubble ring speeds, dark-matter visible speeds) are held at their true values: they are observed "
    "inputs, not model outputs.",
    "Sensitivities: all agent-visible positions (2-particle: pos1 and pos2; dark_matter: the 25 agent "
    "indices; others: all particles) at every measurement time; velocities are also returned to agents "
    "but the registration names positions only. Relative central-difference step 1e-4 (registered) and "
    "1e-3 (stability); for phi = 0 the steps are absolute (1e-4, 1e-3 rad). FIM = S^T S / sigma^2; "
    "CRLB SE = sqrt(g^T FIM^-1 g) / sqrt(16) with all world constants as joint unknowns. For "
    "three_species and dark_matter the Laplacian strength is not a separate constant (it is exactly "
    "confounded with the couplings).",
    "Tolerance threshold tau: the registered example (x2 -> 0.25) is reproduced by tau = 0.5*min(1 - lo/t, "
    "hi/t - 1) for multiplicative ranges (SE in log units) and tau = half the distance to the nearer bound "
    "for additive alpha ranges. The alternative 'half of log tolerance' reading (0.347 for x2) is reported "
    "alongside; verdicts use tau.",
    "Ranker = ascension.diagnostics.rank_designs_by_identifiability on per-design sensitivity matrices "
    "whitened by sigma and scaled to log-parameters (x theta_j; phi unscaled), DEFAULT first then S00..S63. "
    "Its score is rank + 1/(1+log10 cond); ties keep insertion order, so for one-constant worlds "
    "(gravity, coulomb) every design ties and BEST = DEFAULT.",
    "MECHANISM fits (registered): ONE least-squares fit of the alternative's constants jointly over "
    "DEFAULT + 64 designs to the true world's noiseless executor predictions; per-design residual = "
    "RMS over all visible coordinates / sigma. EQUIVALENT (numeric) if every design < 0.01; DISTINGUISHABLE "
    "if DEFAULT > 1; DISTINGUISHABLE-BY-DESIGN if DEFAULT <= 1 and some sampled design > 1; otherwise "
    "INDETERMINATE (a gap in the registered classes). Because a joint fit can mispredict DEFAULT merely "
    "through the compromise with other designs, each alternative is ALSO refitted to DEFAULT alone, BEST "
    "alone and DEFAULT+BEST; the 'solo-DEFAULT variant' verdict (DISTINGUISHABLE only if DEFAULT alone "
    "refutes the alternative) is reported as a sensitivity analysis, not as the registered verdict.",
    "Alternatives are simulated on the fast path (benchmark integrator, acceleration assembly and "
    "kernels with traced constants; exact Jacobians by forward-mode AD); multi-start trf least squares "
    "(<= 200 evaluations per start). Yukawa-family alternatives use a JAX K1 quadrature "
    "(max rel. error 3.6e-15 vs scipy) because the benchmark kernel's scipy callback has no JVP.",
    "three_species 'slightly wrong partition': all 60 single-particle reassignments were scanned at the "
    "true couplings; the least distinguishable one (lowest overall RMS) was refitted with free couplings.",
    "dark_matter count vs coupling: N in {5, 20} dark particles with coupling 50/N and zero initial "
    "velocity, 2N initial positions fitted (starts: random subset (N=5) or duplication (N=20) of the true "
    "positions + N(0, 0.1) jitter, 3 restarts, seed 263, <= 60 evaluations each) per design for DEFAULT, "
    "BEST and the three sampled designs whose probes pass closest to the dark centroid, plus one joint fit "
    "over those five. Joint fitting over all 65 designs was not attempted (40 chaotic nuisance "
    "parameters); the class uses per-design fits, which favour the alternative. DM-3b (coupling ratio) is "
    "a conditional CRLB with count and positions at truth.",
    "Execution note (no method change): the first pooled run (4 worlds per process, sequential "
    "alternatives) failed for 6 light worlds with an XLA runtime error ('Failed to materialize symbols') "
    "in a worker that had accumulated thousands of compiled executables, and a later OOM kill broke the "
    "pool, losing the in-progress ether/dark_matter/three_species runs (hubble and yukawa had completed and "
    "were kept). All other worlds were re-run from scratch in fresh processes with jax.clear_caches() "
    "between executor batches and with a world's alternative fits distributed over worker processes "
    "(three_species 3, ether 2). Designs, seeds, starts, tolerances and budgets were unchanged; the "
    "rerun reproduced the first run's ranker BEST designs. First-run tracebacks are kept in "
    "cache/failed_run1/.",
    "Execution note 2 (harness only): a container restart at ~01:30 UTC 2026-09-29 killed the in-flight "
    "ether/three_species/dark_matter reruns. They were re-run from scratch with the same method, seeds, "
    "starts, tolerances and the same 3600 s dark-matter count-fit budget (persisted across restarts), "
    "under a checkpointing driver (`--stage resume`): the base stage of each world (truth predictions, "
    "parity, FD, ranker, CRLBs) and every individual alternative fit (joint / solo DEFAULT / solo BEST / "
    "DEFAULT+BEST, and each dark-matter count fit) is written to cache/ckpt/ as soon as it finishes and "
    "skipped on restart. Parallelism: 4 long-lived worker processes, each pinned to one core, with "
    "BLAS/OpenMP and XLA intra-op threads set to 1 (the earlier load of ~16 on 4 cores came from "
    "oversubscribed threads).",
    "Execution note 3 (harness only): on the second driver, pinning its two workers to cores 2-3 "
    "(05:13 UTC) starved the two first-driver workers already pinned there (runnable, ~0 CPU) until "
    "all workers were unpinned at 11:40 UTC; about 6 h of wall time lost, no effect on results. A second "
    "container restart (~16:00 UTC) killed both drivers; the run resumed from checkpoints with non-"
    "three_species fits first. From then on each multi-start of a fit was also checkpointed as soon as that "
    "start finished (x, cost, nfev, status); a restarted fit skips finished starts and re-evaluates the "
    "residual at the best start's saved x. Warm-restarting a start mid-way was NOT done: scipy's trf keeps "
    "internal state (trust radius, jac-based x_scale, evaluation count against max_nfev) that a restart "
    "from saved parameters would reset, which would change the registered optimizer behaviour.",
    "Execution note 4: after the second restart the orchestrator briefly launched a duplicate driver "
    "(pid 637, ~16:31–16:32 UTC). Stopping it left its 4 pool workers orphaned; they were killed at "
    "~16:34 UTC. That driver re-read cached stages and completed exactly one fit, ether ET-3 "
    "mass_dep_drift solo_default (checkpoint 16:31:59 UTC). It was produced by the same code "
    "(deterministic), parses cleanly and was kept; no other checkpoint came from it.",
    "Execution note 5: a third container restart (~23:50 UTC 2026-09-29, the coordinator's count: fourth) "
    "killed the drivers again; per-start checkpoints (ether repulsor joint start 0, three_species nonneg "
    "joint start 0) survived. An orchestrator relaunch as a background shell command was killed by the "
    "harness time limit before any fit completed (nothing written). The agent then relaunched a single "
    "detached launcher (scripts run via setsid/nohup: ether with 1 worker followed by the provisional "
    "10-world aggregate, three_species with 3 workers, then the final aggregate), skipping every "
    "checkpointed fit and start.",
    "Execution note 6: after the ether driver finished (07:54 UTC 2026-09-30), dark_matter (all fits "
    "checkpointed earlier but never assembled) was assembled and the provisional 10-world aggregate was "
    "written; the three_species TS-5 probes_C joint fit was started in a separate detached process "
    "(08:05 UTC) to use the freed core. Fits are deterministic, so any duplicate pickup by the main "
    "driver could only waste CPU.",
    "Scored cells for P-EXP-263-3: the 13 per-(model, world) means published on the leaderboard (not "
    "per-seed values). Mann–Whitney: scipy mannwhitneyu, alternative 'less'; Fisher: scipy fisher_exact, "
    "alternative 'greater'.",
]

LIMITATIONS = [
    "Local (linearised) CRLBs at the true constants; no global identifiability analysis for PARAMETER claims.",
    "Static-vs-time-evolving claims are not scored: the pinned nbody engine cannot represent diffusion/wave "
    "fields and the FFT field engine was not used (engine difference; the leaderboard ran nbody).",
    "Alternatives the rubric names but that are not deterministic force laws (noise, drag, 'wind', a "
    "directional Laplacian) were not implemented.",
    "Least-squares fits can miss the global optimum; a missed optimum biases toward DISTINGUISHABLE. "
    "Multi-starts reduce but do not remove this, most of all for the chaotic dark-matter fits.",
    "Close encounters (softening 0.05) in random designs make trajectories stiff; FD derivatives and "
    "residuals on those designs are dominated by near-singular dynamics. FD-vs-AD agreement is reported "
    "per world as a check on the registered finite differences.",
    "The leaderboard reports per-cell means over 5 seeds; trajectory-good/explanation-fail are applied to "
    "those means, not to individual runs.",
    "Private worlds were out of scope; none of the public-code worlds outside the 11 was scored.",
]


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["leaderboard", "claims", "worlds", "resume", "aggregate",
                                                       "aggregate_provisional", "all"])
    ap.add_argument("--worlds", default=",".join(PUBLIC_WORLDS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-joint", default="", help="resume: worlds whose joint fits this driver leaves alone")
    ap.add_argument("--fit-worlds", default="", help="resume: worlds whose fits this driver runs")
    ap.add_argument("--no-pin", action="store_true")
    args = ap.parse_args()
    if git_head(DP) != EXPECTED_DP or git_head(LB) != EXPECTED_LB:
        raise SystemExit("pinned commits do not match")
    lb = OUT / "leaderboard_cells.tsv"
    if args.stage in ("leaderboard", "all") or not lb.exists():
        print("leaderboard copied:", copy_leaderboard())
        if args.stage == "leaderboard":
            return
    if args.stage in ("claims", "all"):
        check_truth_constants()
        print("claims:", write_claims_tsv())
        for w in PUBLIC_WORLDS:
            jdump(all_designs(w), OUT / "designs" / f"{w}.json")
        if args.stage == "claims":
            return
    if args.stage == "aggregate_provisional":
        aggregate(provisional=True)
        return
    if args.stage == "resume":
        resume([w for w in args.worlds.split(",") if w], jobs=args.jobs,
               skip_joint=[w for w in args.skip_joint.split(",") if w],
               fit_worlds=[w for w in args.fit_worlds.split(",") if w] or None, pin=not args.no_pin)
        return
    if args.stage in ("worlds", "all"):
        CACHE.mkdir(parents=True, exist_ok=True)
        worlds = [w for w in args.worlds.split(",") if w]
        n_s = 3 if args.smoke else N_SAMPLED
        pri = ["dark_matter", "three_species", "ether", "hubble", "yukawa", "circle", "oscillator",
               "extra_dimensions", "fractional", "coulomb", "gravity"]
        worlds.sort(key=lambda w: pri.index(w))
        if args.jobs <= 1:
            for w in worlds:
                print(worker(w, n_s, args.smoke)[:2], flush=True)
        else:
            import concurrent.futures as cf
            import multiprocessing as mp
            with cf.ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn"),
                                        max_tasks_per_child=1) as pool:
                futs = [pool.submit(worker, w, n_s, args.smoke) for w in worlds]
                for f in cf.as_completed(futs):
                    w, status, info = f.result()
                    print(f"== {w}: {status} {info if status == 'ok' else ''}", flush=True)
        if args.smoke:
            print("smoke run complete (no aggregation, no verdicts)")
            return
    if args.stage in ("aggregate", "all"):
        s = aggregate()
        print("aggregate written:", OUT / "README.md")


if __name__ == "__main__":
    main()
