"""EXP-087 — the pre-registered confirmatory cell for the design-class exhaustion
certificate (predictions locked in predictions.md, commit 7b3aa61, BEFORE this file).

Michaelis-Menten with substrate observed: dS/dt = -kcat*E0*S/(Km+S). Literature: only
Vmax = kcat*E0 is identifiable from substrate time courses; separating kcat from E0
needs an independent enzyme-concentration measurement. No closed form exists, so this
is the black-box (numeric) route of the certificate. $0, no LLM, no DB.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from scipy.integrate import solve_ivp

from ascension.diagnostics.exhaustion import certify_class_symmetry_numeric, triage_plateau
from ascension.diagnostics.identifiability import practical_identifiability

THETA = [10.0, 0.1, 2.0]  # kcat, E0, Km  (Vmax = 1)
NAMES = ["kcat", "E0", "Km"]


def _substrate(theta, s0: float, t: np.ndarray) -> np.ndarray:
    kcat, e0, km = (float(x) for x in theta)

    def rhs(_t, y):
        return [-kcat * e0 * y[0] / (km + y[0])]

    sol = solve_ivp(
        rhs, (0.0, float(t[-1])), [s0], t_eval=t, method="LSODA", rtol=1e-10, atol=1e-10
    )
    assert sol.success, sol.message
    return sol.y[0]


def _design(s0: float, t: np.ndarray):
    return lambda th: _substrate(th, s0, t)


STD_T = np.linspace(0.1, 20.0, 24)
STD = _design(5.0, STD_T)


def _sampler(rng: np.random.Generator):
    s0 = float(rng.uniform(0.2, 20.0))
    n = int(rng.integers(8, 41))
    t = np.sort(rng.uniform(0.05, 30.0, size=n))
    return _design(s0, t)


def test_p1_diagnosis_rank_2_of_3() -> None:
    v = practical_identifiability(STD, THETA, param_names=NAMES)
    assert (v.rank, v.n_params) == (2, 3)
    assert v.status == "structurally_non_identifiable"


def test_p2_null_direction_is_the_kcat_e0_scaling_axis() -> None:
    v = practical_identifiability(STD, THETA, param_names=NAMES)
    d = np.asarray(v.confounded_directions[0], dtype=float)
    axis = np.array([THETA[0], -THETA[1], 0.0])
    axis /= np.linalg.norm(axis)
    assert abs(float(d @ axis)) > 0.99


def test_p3_class_level_certificate_and_change_class_triage() -> None:
    v = practical_identifiability(STD, THETA, param_names=NAMES)
    cert = certify_class_symmetry_numeric(
        _sampler, THETA, v.confounded_directions[0], n_designs=32, seed=87, param_names=NAMES
    )
    assert cert.certified, cert.describe()
    assert cert.n_designs_sampled == 32

    certify = partial(
        certify_class_symmetry_numeric, _sampler, THETA, n_designs=32, seed=87, param_names=NAMES
    )
    tri = triage_plateau(STD, THETA, certify=certify, param_names=NAMES)
    assert tri.verdict == "change_class", tri.describe()


def test_p4_class_change_restores_rank_but_a_within_class_design_does_not() -> None:
    certify = partial(
        certify_class_symmetry_numeric, _sampler, THETA, n_designs=32, seed=87, param_names=NAMES
    )

    # the class change: an independent enzyme-concentration assay joins the observation set
    def with_e0_assay(th):
        return np.concatenate([STD(th), [float(th[1])]])

    v = practical_identifiability(with_e0_assay, THETA, param_names=NAMES)
    assert (v.rank, v.n_params) == (3, 3)
    assert (
        triage_plateau(with_e0_assay, THETA, certify=certify, param_names=NAMES).verdict
        == "capability"
    )

    # negative control: a second substrate-only run at a different S0 stays inside the class
    second = _design(0.5, np.linspace(0.1, 20.0, 24))

    def two_runs(th):
        return np.concatenate([STD(th), second(th)])

    v2 = practical_identifiability(two_runs, THETA, param_names=NAMES)
    assert (v2.rank, v2.n_params) == (2, 3)
