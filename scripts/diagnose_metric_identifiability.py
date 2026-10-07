"""$0 ANALYSIS — the identifiability instrument turned on the benchmark's OWN metric.

The question (Session 039): the EXP-075 δ=1.0 depth separation collapsed under a
proposal-coupled L2 gate because, over the scorer's orbit-range held-out data,
r^-3 and r^-3.5 are nearly collinear (the L2 rescore + precursor). But "two
parameter values the data cannot distinguish" is EXACTLY the statement the
project's practical-identifiability diagnostic exists to make formally. So:
treat the correction EXPONENT itself as a free parameter,

    a(r; c0, c1, e) = c0·(−1/r²) + c1·(−1/r^e),   θ = (c0, c1, e)

and run the UNCHANGED diagnostic stack (practical_identifiability +
rank_designs_by_identifiability) over the observation designs the project
actually uses:

  D1 orbit-held-out : the r-points of the L2 scorer's held-out set (r∈[1,1.41])
  D2 probe-battery  : the clean radial-drop battery (run_clean_probes, r∈[0.5,~4])
  D3 small-r        : the battery restricted to r∈[0.5,1.0] (correction-dominated)
  D4 orbit+battery  : D1 ∪ D2 (what a scorer COULD have used)

If the diagnostic flags the exponent direction as sloppy/ill-conditioned on D1
and better-conditioned on the drop designs, then the depth-gate collapse was
PREDICTABLE by the project's own instrument before any paid run — the metric's
±0.3 exponent band asserted a resolution the observation design does not have.

Also reported: the empirical RSS-reduction "indifference band" per design — the
set {e : red(e) ≥ 0.95·red_max} of exponents the design's own data cannot
materially distinguish from its best — directly comparable to the gate's ±0.3
band, using each design's real observed accelerations (so D1 includes the
charge-term oscillation exactly as the real scorer's data does).

Deterministic, no LLM, no DB. Confirmed records untouched.

Run:  poetry run python scripts/diagnose_metric_identifiability.py
"""

from __future__ import annotations

import numpy as np

from ascension.benchmarks.alien_depth import (
    _corr_basis,
    _newton_basis,
    _observed_accel,
)
from ascension.benchmarks.alien_fixtures import alien_benchmark
from ascension.benchmarks.experiment_feedback import run_clean_probes
from ascension.diagnostics.identifiability import practical_identifiability
from ascension.diagnostics.optimal_design import rank_designs_by_identifiability

_TRUTH_EXPONENT = 3.5
_PARAM_NAMES = ["c0", "c1", "e"]
_E_GRID = np.round(np.arange(2.0, 6.0001, 0.05), 4)
_INDIFF_FRAC = 0.95  # indifference band = e's achieving >=95% of the best reduction


def _designs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    held_out = alien_benchmark(seed=0)[2]
    r1, a1 = _observed_accel(held_out)
    r2, a2 = run_clean_probes(seed=0)
    small = (r2 >= 0.5) & (r2 <= 1.0)
    r4 = np.concatenate([r1, r2])
    a4 = np.concatenate([a1, a2])
    return {
        "D1 orbit-held-out": (np.asarray(r1, float), np.asarray(a1, float)),
        "D2 probe-battery": (np.asarray(r2, float), np.asarray(a2, float)),
        "D3 small-r": (np.asarray(r2[small], float), np.asarray(a2[small], float)),
        "D4 orbit+battery": (r4.astype(float), a4.astype(float)),
    }


def _make_predict(r: np.ndarray):
    def predict(theta: np.ndarray) -> np.ndarray:
        c0, c1, e = float(theta[0]), float(theta[1]), float(theta[2])
        return c0 * _newton_basis(r) + c1 * _corr_basis(r, e)

    return predict


def _fit_theta_star(r: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Least-squares (c0, c1) at the truth exponent — a realistic θ* per design."""
    X = np.vstack([_newton_basis(r), _corr_basis(r, _TRUTH_EXPONENT)]).T
    c, *_ = np.linalg.lstsq(X, a, rcond=None)
    return np.array([c[0], c[1], _TRUTH_EXPONENT], dtype=float)


def _indifference_band(r: np.ndarray, a: np.ndarray) -> tuple[float, float, float]:
    """(e_lo, e_hi, e_best) of the RSS-reduction indifference band over _E_GRID."""
    newton = _newton_basis(r)
    _, rss_base = _lstsq_rss(a, [newton])
    reds = []
    for e in _E_GRID:
        _, rss_aug = _lstsq_rss(a, [newton, _corr_basis(r, float(e))])
        reds.append((rss_base - rss_aug) / rss_base if rss_base > 0 else 0.0)
    reds = np.asarray(reds)
    best_i = int(np.argmax(reds))
    ok = _E_GRID[reds >= _INDIFF_FRAC * reds[best_i]]
    return float(ok.min()), float(ok.max()), float(_E_GRID[best_i])


def _lstsq_rss(a, cols):
    X = np.vstack(cols).T
    c, *_ = np.linalg.lstsq(X, a, rcond=None)
    return c, float(np.sum((a - X @ c) ** 2))


def main() -> None:
    designs = _designs()
    print("=" * 96)
    print("METRIC IDENTIFIABILITY — θ=(c0, c1, e) under the project's real observation designs")
    print(f"  model a(r)=c0·(−1/r²)+c1·(−1/r^e); θ* per design = LSQ fit at truth e={_TRUTH_EXPONENT}")
    print("-" * 96)

    closures = {}
    for label, (r, a) in designs.items():
        theta_star = _fit_theta_star(r, a)
        predict = _make_predict(r)
        v = practical_identifiability(predict, theta_star, param_names=_PARAM_NAMES)
        lo, hi, best = _indifference_band(r, a)
        sloppy = ""
        if v.confounded_directions:
            d = np.asarray(v.confounded_directions[-1])
            sloppy = " sloppy_dir(c0,c1,e)=(" + ", ".join(f"{x:+.2f}" for x in d) + ")"
        print(
            f"{label:<18} n={len(r):>6}  r∈[{r.min():.2f},{r.max():.2f}]  "
            f"status={v.status:<28} rank={v.rank}/3 cond={v.condition_number:.2e}{sloppy}"
        )
        print(
            f"{'':<18} θ*=(c0={theta_star[0]:.3f}, c1={theta_star[1]:.3f}, e={_TRUTH_EXPONENT})  "
            f"indifference band (red≥{_INDIFF_FRAC:.2f}·max): e∈[{lo:.2f},{hi:.2f}] best={best:.2f} "
            f"width={hi - lo:.2f}  [gate band width 0.60: e∈[3.2,3.8]]"
        )
        # OED closures share ONE θ — use the battery-fit θ* (most physical) for all
        closures[label] = predict

    print("-" * 96)
    theta_shared = _fit_theta_star(*designs["D2 probe-battery"])
    ranking = rank_designs_by_identifiability(closures, theta_shared, param_names=_PARAM_NAMES)
    print("OED ranking over the four designs (shared θ* from D2 fit):")
    for s in ranking.ranked:
        print(
            f"  {s.label:<18} identifiable={s.identifiable} rank={s.rank}/3 "
            f"cond={s.condition_number:.2e} score={s.score:.3f}"
        )
    print(f"  BEST: {ranking.best}")
    print(f"  rationale: {ranking.rationale}")
    print("=" * 96)


if __name__ == "__main__":
    main()
