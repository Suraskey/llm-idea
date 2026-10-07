"""A real, published non-identifiability as a true-positive test for the diagnostic.

The one-compartment oral-dosing pharmacokinetic model is the textbook example of
structural non-identifiability (Rowland & Tozer; Gabrielsson & Weiner): after an
oral dose D, the plasma concentration is

    y(t) = (F * D * ka) / (V * (ka - ke)) * (exp(-ke t) - exp(-ka t))

so bioavailability F and volume of distribution V enter ONLY through F/V, and the
absorption/elimination rates ka, ke can be swapped ("flip-flop") with the change
absorbed into F/V. Pharmacologists resolve F/V by adding an intravenous arm
(an "absolute bioavailability" study: IV dose has F = 1 by definition).

This script runs the UNCHANGED engine (practical_identifiability +
rank_designs_by_identifiability + structural_identifiability) on that model and
reports: (1) the diagnostic's verdict on oral-only sampling (expect rank 3/4 with
the confounded direction along the (V, F) scaling axis); (2) which of three
candidate follow-up designs the OED ranker picks (expect the IV arm); (3) the
flip-flop as a demonstrated LOCAL-vs-GLOBAL scope limit: the swapped parameter
point produces identical oral observations, and the local check returns the SAME
verdict at both points (rank 3/4, the F/V confound), so it cannot see that a
second, disconnected solution exists. The IV arm breaks the flip-flop too, since
the IV curve pins ke on its own.

$0, deterministic, no LLM. Run:
    poetry run python scripts/diagnostic_pk_bioavailability.py
"""

from __future__ import annotations

import numpy as np
import sympy

from ascension.diagnostics.identifiability import practical_identifiability
from ascension.diagnostics.optimal_design import rank_designs_by_identifiability
from ascension.diagnostics.structural import structural_identifiability

DOSE = 100.0  # mg, known
THETA = np.array([1.2, 0.25, 30.0, 0.6])  # ka [1/h], ke [1/h], V [L], F [-]
NAMES = ["ka", "ke", "V", "F"]


def _oral(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    ka, ke, v, f = theta
    return (f * DOSE * ka) / (v * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def _iv(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    _ka, ke, v, _f = theta
    return (DOSE / v) * np.exp(-ke * t)


def design_oral(t: np.ndarray):
    def predict(theta: np.ndarray) -> np.ndarray:
        return _oral(np.asarray(theta, dtype=float), t)

    return predict


def design_oral_plus_iv(t_oral: np.ndarray, t_iv: np.ndarray):
    def predict(theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=float)
        return np.concatenate([_oral(th, t_oral), _iv(th, t_iv)])

    return predict


def main() -> None:
    t_std = np.linspace(0.25, 24.0, 24)
    t_dense = np.linspace(0.25, 24.0, 96)
    t_long = np.linspace(0.25, 72.0, 48)

    print("=" * 78)
    print("Oral one-compartment PK model: theta = (ka, ke, V, F), dose known")
    print("=" * 78)

    # (1) the diagnostic on the oral-only design
    v_oral = practical_identifiability(design_oral(t_std), THETA, param_names=NAMES)
    print("\n[1] oral-only sampling (24 points over 24 h)")
    print(v_oral.describe())
    assert v_oral.rank == 3 and v_oral.n_params == 4, v_oral
    d = np.array(v_oral.confounded_directions[0])
    # the confounded direction must lie along the (V, F) scaling axis: (0, 0, V, F)
    expect = np.array([0.0, 0.0, THETA[2], THETA[3]])
    expect /= np.linalg.norm(expect)
    cos = abs(float(d @ expect))
    print(f"    |cos(angle to the (0,0,V,F) scaling axis)| = {cos:.6f}")
    assert cos > 0.999, (d, expect)

    # exact symbolic confirmation on the closed form
    t = sympy.Symbol("t", positive=True)
    ka, ke, V, F = sympy.symbols("ka ke V F", positive=True)
    y = (F * DOSE * ka) / (V * (ka - ke)) * (sympy.exp(-ke * t) - sympy.exp(-ka * t))
    sv = structural_identifiability(y, [ka, ke, V, F], [t])
    print(
        f"    structural (exact symbolic): identifiable={sv.identifiable}; "
        f"confounded groups={sv.confounded_groups}"
    )
    assert sv.confounded_groups == (("F", "V"),), sv

    # (2) the OED ranker over three candidate follow-up designs
    designs = {
        "oral, 4x denser sampling": design_oral(t_dense),
        "oral, 72 h window": design_oral(t_long),
        "oral + intravenous arm (F=1 by definition)": design_oral_plus_iv(t_std, t_std),
    }
    ranking = rank_designs_by_identifiability(designs, THETA, param_names=NAMES)
    print("\n[2] OED ranking of candidate follow-up experiments")
    for s in ranking.ranked:
        print(f"    rank {s.rank}/{s.n_params}  cond={s.condition_number:9.3e}  {s.label}")
    print(f"    -> recommended: {ranking.best}")
    assert "intravenous" in ranking.best, ranking.best

    v_iv = practical_identifiability(designs[ranking.best], THETA, param_names=NAMES)
    assert v_iv.identifiable and v_iv.rank == 4, v_iv
    print(
        f"    re-diagnosis under the chosen design: rank {v_iv.rank}/{v_iv.n_params} "
        f"({v_iv.status})"
    )

    # (3) the flip-flop: a GLOBAL non-identifiability the local check cannot see
    ka0, ke0, v0, f0 = THETA
    swapped = np.array([ke0, ka0, v0, f0 * ka0 / ke0])
    y0 = design_oral(t_std)(THETA)
    y1 = design_oral(t_std)(swapped)
    gap = float(np.max(np.abs(y0 - y1)))
    v_swapped = practical_identifiability(design_oral(t_std), swapped, param_names=NAMES)
    print("\n[3] flip-flop (ka <-> ke, F/V rescaled): a second, disconnected solution")
    print(f"    max |y(theta) - y(theta_swapped)| = {gap:.2e}  (identical observations)")
    print(f"    local diagnostic at the swapped point: rank {v_swapped.rank}/{v_swapped.n_params}")
    assert gap < 1e-9
    # with the IV arm both points are locally full-rank AND predict different data:
    # the IV curve pins ke on its own, so the swap is broken globally as well. A
    # local check can only report the rank at each point, never the global fact.
    v_iv_swapped = practical_identifiability(designs[ranking.best], swapped, param_names=NAMES)
    print(
        f"    local diagnostic at the swapped point, oral+IV: rank "
        f"{v_iv_swapped.rank}/{v_iv_swapped.n_params}"
    )
    y_iv0 = designs[ranking.best](THETA)
    y_iv1 = designs[ranking.best](swapped)
    print(
        f"    max |y_oral+iv(theta) - y_oral+iv(theta_swapped)| = "
        f"{float(np.max(np.abs(y_iv0 - y_iv1))):.3e}  (no longer identical)"
    )
    print(
        "\nDONE: F/V confound caught (true positive on a published example); the ranker "
        "chose the IV arm (the absolute-bioavailability design); the flip-flop is the "
        "local-vs-global scope limit, demonstrated on a real model."
    )


if __name__ == "__main__":
    main()
