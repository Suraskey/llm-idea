"""Pins the real-literature true positive: the oral one-compartment PK model.

Textbook structural non-identifiability (Rowland & Tozer; Gabrielsson & Weiner):
after an oral dose, bioavailability F and volume V enter the plasma curve only as
F/V, and ka <-> ke can be swapped ("flip-flop") with the change absorbed into F/V.
The documented fix is an intravenous arm (absolute-bioavailability study).

These tests pin, on the UNCHANGED engine: the F/V confound is caught (rank 3/4,
direction along the (V, F) scaling axis; the exact symbolic checker agrees), the
OED ranker chooses the IV arm over denser or longer oral sampling, and the
flip-flop is the local-vs-global scope limit (identical oral data at two
disconnected points, same local verdict at both). $0, no LLM, no DB.
"""

from __future__ import annotations

import numpy as np
import sympy

from ascension.diagnostics.identifiability import practical_identifiability
from ascension.diagnostics.optimal_design import rank_designs_by_identifiability
from ascension.diagnostics.structural import structural_identifiability

DOSE = 100.0
THETA = np.array([1.2, 0.25, 30.0, 0.6])  # ka, ke, V, F
NAMES = ["ka", "ke", "V", "F"]
T = np.linspace(0.25, 24.0, 24)


def _oral(theta, t):
    ka, ke, v, f = theta
    return (f * DOSE * ka) / (v * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def _iv(theta, t):
    _ka, ke, v, _f = theta
    return (DOSE / v) * np.exp(-ke * t)


def _oral_only(theta):
    return _oral(np.asarray(theta, dtype=float), T)


def _oral_plus_iv(theta):
    th = np.asarray(theta, dtype=float)
    return np.concatenate([_oral(th, T), _iv(th, T)])


def test_oral_only_confounds_f_and_v_along_the_scaling_axis() -> None:
    v = practical_identifiability(_oral_only, THETA, param_names=NAMES)
    assert v.identifiable is False
    assert (v.rank, v.n_params) == (3, 4)
    d = np.asarray(v.confounded_directions[0], dtype=float)
    axis = np.array([0.0, 0.0, THETA[2], THETA[3]])
    axis /= np.linalg.norm(axis)
    assert abs(float(d @ axis)) > 0.999


def test_symbolic_checker_agrees_on_the_f_v_group() -> None:
    t = sympy.Symbol("t", positive=True)
    ka, ke, V, F = sympy.symbols("ka ke V F", positive=True)
    y = (F * DOSE * ka) / (V * (ka - ke)) * (sympy.exp(-ke * t) - sympy.exp(-ka * t))
    sv = structural_identifiability(y, [ka, ke, V, F], [t])
    assert sv.identifiable is False
    assert sv.confounded_groups == (("F", "V"),)


def test_ranker_picks_the_iv_arm_over_more_oral_sampling() -> None:
    t_dense = np.linspace(0.25, 24.0, 96)
    t_long = np.linspace(0.25, 72.0, 48)
    designs = {
        "oral dense": lambda th: _oral(np.asarray(th, dtype=float), t_dense),
        "oral long": lambda th: _oral(np.asarray(th, dtype=float), t_long),
        "oral + iv": _oral_plus_iv,
    }
    r = rank_designs_by_identifiability(designs, THETA, param_names=NAMES)
    assert r.best == "oral + iv"
    by_label = {s.label: s for s in r.ranked}
    assert by_label["oral + iv"].rank == 4
    assert by_label["oral dense"].rank == 3
    assert by_label["oral long"].rank == 3
    assert practical_identifiability(_oral_plus_iv, THETA, param_names=NAMES).identifiable


def test_flip_flop_is_a_global_limit_the_local_check_cannot_see() -> None:
    ka, ke, v, f = THETA
    swapped = np.array([ke, ka, v, f * ka / ke])
    assert np.max(np.abs(_oral_only(THETA) - _oral_only(swapped))) < 1e-12
    a = practical_identifiability(_oral_only, THETA, param_names=NAMES)
    b = practical_identifiability(_oral_only, swapped, param_names=NAMES)
    assert (a.rank, b.rank) == (3, 3)  # same local verdict at two disconnected points
    # the IV arm pins ke on its own, so the swap is broken globally as well
    assert np.max(np.abs(_oral_plus_iv(THETA) - _oral_plus_iv(swapped))) > 1e-3
