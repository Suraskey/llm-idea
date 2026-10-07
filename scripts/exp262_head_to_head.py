"""EXP-262 — head-to-head of experiment-selection criteria on identical candidate sets.

Pre-registered in predictions.md, block "Pre-registered block added 2026-09-28
(Session 047 — EXP-262 ...)", committed in 8f746eb BEFORE any criterion was computed.
$0, deterministic, no LLM, no network.

What it does (all of it fixed by the pre-registration; the choices the registration
left open are fixed in IMPLEMENTATION_CHOICES below, before the first run):

  Tasks
    T1  62 ODEBench systems with free constants, EXP-261 LLM-ACES round-0 protocol:
        one trajectory from IC-0 on [0, 1], 100 points, full state + analytic derivative.
    T2  the same 62 systems, SPARSE protocol: first state coordinate only, no
        derivative, 20 uniform points on [0, 1], from IC-0.
    T3  descriptive only: oral PK, Michaelis-Menten, Alien amplitude, sequence ansatz.
  Candidates (T1, T2): the 10 EXP-261 box ICs + the ODEBench held-out IC-1, each one
    added trajectory under the task's own observation protocol.
  Criteria: RANK, DOPT, AOPT, VOI, DISAGREE, RANDOM (exact expectation), HYBRID.
  Outcomes: M1 rank restoration, M2 empirical recovery (5 noise seeds, LM fits on
    log-parameters), M3 experiments spent on certified change_class tasks.

Run:
  PYTHONPATH=src poetry run python scripts/exp262_head_to_head.py            # full run
  ... --ids 1,29,53 --smoke --out <scratch dir>                              # code-path smoke
  ... --aggregate-only                                                       # re-aggregate units.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy as sp
from scipy import stats
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import sweep_odebench_identifiability as s261  # noqa: E402  (EXP-261; pins LLM-ACES ODEBench)

from ascension.diagnostics.exhaustion import (  # noqa: E402
    certify_class_symmetry_numeric,
    certify_class_symmetry_symbolic,
)
from ascension.diagnostics.identifiability import practical_identifiability  # noqa: E402
from ascension.diagnostics.optimal_design import rank_designs_by_identifiability  # noqa: E402

strogatz_ode = s261.strogatz_ode
ACES = s261.ACES

# ---------------------------------------------------------------------------------
# Registered constants (predictions.md, EXP-262 block)
# ---------------------------------------------------------------------------------
T1_GRID = s261.T_TRAIN  # np.linspace(0, 1, 100), EXP-261 / LLM-ACES generate_ode.py
T2_GRID = np.linspace(0.0, 1.0, 20)  # SPARSE: 20 uniform points on [0, 1]
NOISE_FRAC = 0.01  # sigma = 1% of the RMS of each observed channel over round 0
SCALE_FLOOR = 1e-3  # S~ = S diag(max(|theta*|, 1e-3))
RANK_TOL = 1e-9  # M1 relative rank floor (the library default)
COND_THRESHOLD = 1e8  # the paper's practical threshold (round-0 verdict, as EXP-261)
EPS_RIDGE = 1e-10  # eps = 1e-10 tr(F0)
TAU = 1.0  # VOI / DISAGREE prior sd in log-parameters
K_DRAWS = 8  # DISAGREE
DISAGREE_SEED = 262
NOISE_SEEDS = (262, 263, 264, 265, 266)
START_SD = 0.1  # LM start: log theta* + N(0, 0.1^2)
BOOT_N = 10_000
BOOT_SEED = 262
CRITERIA = ["RANK", "HYBRID", "DOPT", "AOPT", "VOI", "DISAGREE", "RANDOM"]
DET_CRITERIA = [c for c in CRITERIA if c != "RANDOM"]
WILCOXON_PAIRS = ["DOPT", "AOPT", "VOI", "DISAGREE", "RANDOM", "HYBRID"]  # each vs RANK

# ---------------------------------------------------------------------------------
# Numerics that are ours (integrator settings, not protocol)
# ---------------------------------------------------------------------------------
TRUTH_TOL = 1e-12  # exact forward sensitivities and noiseless data (EXP-261 Amendment 3)
FIT_TOL = 1e-9  # model evaluations inside the LM fits
ROLL_TOL = 1e-10  # DISAGREE rollouts (EXP-261's INTEG tolerance)
P4_TOL = 1e-12  # rollouts for the P4 certified-direction contribution
MAX_RHS_CALLS = 200_000  # per integration; a stiff/diverging trial point fails instead of hanging
BLOWUP = 1e6  # |x| event, as EXP-261
PENALTY = 1e6  # residual returned at an LM trial point whose integration fails
GUARD_SECONDS = 600  # per (system, protocol) unit
INF_SENTINEL = 1e300  # rank-preserving stand-in for +inf in bootstrap / Wilcoxon

IMPLEMENTATION_CHOICES = [
    "Candidate order is EXP-261's: IC-1 (ODEBench held-out IC) first, then box IC 0..9 "
    "(np.random.default_rng(0).uniform over the ic_bounds.json box, exactly EXP-261). The "
    "EXP-261 OOD window is not a candidate (the registered set is 10 box ICs + IC-1). Every "
    "argmax/argmin breaks ties by this order (first wins); RANK's ties are broken by the "
    "library ranker's stable sort, which is the same order.",
    "Scaled sensitivity is formed with SIGNED scales, S diag(sign(theta*) max(|theta*|, 1e-3)): "
    "the exact Jacobian with respect to the sign-preserving log-parameters delta that DISAGREE "
    "draws and LM fits (theta = theta* exp(delta)). It differs from the registered "
    "S diag(max(|theta*|, 1e-3)) only by column signs (ids 16, 57, 61 have negative constants), "
    "which leaves rank, condition number, DOPT, AOPT and VOI scores identical and makes the "
    "posterior covariance Sigma's correlations correct for the draws. Constants with "
    "|theta*| < 1e-3 use the additive coordinate theta = theta* + 1e-3 delta (none in ODEBench; "
    "theta1 = 0 in the T3 sequence ansatz).",
    "Every criterion, M1 and the round-0 verdict use the same whitened matrix "
    "A = W S~, W = diag(1/sigma_row), so F = A^T A. RANK is the library ranker "
    "(rank_designs_by_identifiability) applied to [A0; Ac]; M1 is rank([A0; Ac]) at the "
    "library's 1e-9 relative floor (practical_identifiability), so RANK and M1 read the "
    "same number. The round-0 verdict (and so the certificate input) is "
    "practical_identifiability on A0 with the 1e8 condition threshold.",
    "Certificate: certify_class_symmetry_symbolic on the vector field sum_k w_k f_k(x; c) "
    "with x and w free (exactly EXP-261), applied to each round-0 null direction mapped to "
    "theta-space. For T2 this field-level certificate is sound (a symmetry of the field is "
    "invisible to any observation) but conservative (an x0-only symmetry that is not a field "
    "symmetry would not be certified). Triage: capability if round 0 is identifiable; "
    "change_class if every round-0 direction is certified; design_within_class otherwise.",
    "A candidate whose trajectory at theta* leaves the finite domain (|x| > 1e6) or fails to "
    "integrate at 1e-12 cannot be acquired; it is dropped from that task's candidate set "
    "(EXP-261 behaviour) and listed.",
    "P1 setting 'at least one candidate restores the maximum rank' is read as: "
    "max over candidates of rank([A0; Ac]) > rank(A0). A pick 'is max-rank' iff its rank "
    "equals that maximum.",
    "DISAGREE: the K = 8 draws delta_k = Sigma^(1/2) z_k (symmetric square root, "
    "z = default_rng(262).standard_normal((8, n))) are shared by every candidate of a task. "
    "Each draw is rolled out with the true simulator from the candidate IC and observed "
    "under the task protocol (T1: states + analytic derivatives on 100 points; T2: x0 on 20 "
    "points), at rtol = atol = 1e-10. Score = mean over pairs of ||y_i - y_j||_2 divided by "
    "the RMS of the mean prediction (LLM-ACES normalises by the mean trajectory). A draw whose "
    "rollout fails is excluded; with fewer than 2 valid rollouts the score is NaN and the "
    "candidate is never picked.",
    "M2 noise and starts use common random numbers: for seed s, rng = default_rng(s) draws "
    "the start delta0 ~ N(0, 0.1^2 I), then round-0 noise, then candidate noise, so the start "
    "and the round-0 noise are identical across candidates (and criteria) for a seed, and the "
    "candidate noise vector is identical across candidates of equal length.",
    "M2 fit: scipy least_squares(method='lm') on delta with residuals (model - data)/sigma_row "
    "and the exact Jacobian (forward sensitivities at the current theta, rtol = atol = 1e-9). "
    "A trial point whose integration fails returns a constant 1e6 residual so LM backs off. "
    "A fit FAILS (NaN, counted) if least_squares raises, status <= 0 (including the default "
    "max_nfev = 100 n reached), the final point does not integrate, or x/cost are non-finite. "
    "The per-candidate mean over seeds is +inf if any seed failed (the registered 'error = "
    "+inf in rankings').",
    "M2 score: e = delta_hat (error in log-parameters; delta* = 0). For a rank-deficient "
    "combined design with null space N (the collapsed right singular vectors of [A0; Ac] at "
    "the 1e-9 floor), e is replaced by (I - N N^T) e and the RMS is ||e|| / sqrt(rank) "
    "(the RMS over the rank-dimensional orthogonal complement; full rank reduces to the "
    "plain RMS over all parameters).",
    "RANDOM: per task, rank-restoration = fraction of candidates at max rank; M2 = mean of the "
    "candidates' M2 (so +inf if any candidate's fit failed); agreement with a criterion = "
    "1/n_candidates.",
    "P2's 'p < 0.05' is read with the registered Holm correction across the 6 pairs with RANK "
    "(each pair: one-sided Wilcoxon signed-rank, H1: criterion's M2 < RANK's M2, paired by "
    "task, zero differences dropped, scipy default method). Raw p is reported alongside.",
    "Populations: P1 = tasks in the P1 setting; P2 = tasks where >= 2 candidates attain the "
    "maximum rank. P3 and P5 state no setting, so their medians are over ALL scored tasks "
    "(T1 and T2 pooled); P3's rank-restoration part is scored on the P1 setting. The other "
    "populations are reported as secondary. All predictions are scored on T1+T2 pooled; "
    "per-protocol numbers are descriptive. T3 is never pooled into a prediction.",
    "'Within 5%' (P3) means |median_HYBRID / median_DOPT - 1| <= 0.05. 'Best on median' (P5) "
    "means DISAGREE's median is strictly lower than each of DOPT, AOPT and VOI.",
    "Status vocabulary: HELD (the prediction as stated is met); FAILED (the registered "
    "falsifier fired); NOT MET (the prediction as stated is not met but the registered "
    "falsifier did not fire); NOT SCORED (the registered descriptive fallback applies).",
    "P4 DISAGREE contribution along a certified direction d (unit, in delta-space, from the "
    "certificate's snapped theta-direction): with the task's 8 draws, rollouts at 1e-12, "
    "(a) score of the draws projected onto d divided by the full score, and (b) "
    "|full score - score of the draws with the d-component removed| / full score; the "
    "prediction is scored on the max over candidates of both.",
    "sigma floor: if a round-0 channel has RMS exactly 0, sigma = 1% of the RMS over all "
    "round-0 channels (and 0.01 absolute if that is 0 too); every use is listed in the notes.",
    "Guard: 10 min wall clock per (system, protocol) unit (SIGALRM in the worker); an "
    "errored or timed-out unit is excluded from every denominator and listed with its reason.",
    "Bootstrap: percentile 95% CI of the median, 10,000 resamples of tasks with "
    "default_rng(262), the same resampled task sets for every criterion; +inf is replaced by "
    "1e300 (rank-preserving) and any bound >= 1e299 is reported as inf. Wilcoxon uses the "
    "same substitution (it only uses signs and ranks).",
    "T3 definitions (fixed before running). PK: theta = (ka, ke, V, F) = (1.2, 0.25, 30, 0.6), "
    "dose 100; round 0 = oral, 24 points on [0.25, 24] h; candidates = the three follow-ups of "
    "scripts/diagnostic_pk_bioavailability.py as added data: oral 96 points on [0.25, 24], "
    "oral 48 points on [0.25, 72], oral 24 points + IV 24 points on [0.25, 24]; one channel "
    "(plasma concentration, the IV arm measures the same quantity) with sigma from round 0; "
    "certificate symbolic on the oral closed form (t free). Michaelis-Menten: theta = (kcat, "
    "E0, Km) = (10, 0.1, 2); round 0 = substrate from S0 = 5 at 24 points on [0.1, 20] "
    "(EXP-087 STD); candidates = substrate from S0 in geomspace(0.2, 20, 10) on the same grid, "
    "plus an E0 assay (one direct observation of E0, sigma = 1% of E0* since the channel is "
    "absent at round 0); certificate = EXP-087's numeric route (its sampler, seed 87, 32 "
    "designs). Alien amplitude: A(P) = beta P^p, theta = (1, 1); round 0 = P = [1]; "
    "candidates 'single product' P = [1] and 'two distinct products' P = [1, 2] "
    "(propose_charge_product_designs); certificate symbolic (P free). Sequence ansatz: "
    "log f(n) = log th0 + th1 log n + n log th2 at theta = (1, 0, 1) (test_exhaustion.py); "
    "round 0 = narrow window n = [400, 401, 402]; candidates 'narrow window' (the same window "
    "again) and 'geometric spread' n = [2, 5, 20, 100, 400]; round-0 RMS is 0 at this theta, "
    "so sigma = 0.01 absolute on log f (1% multiplicative on f); certificate symbolic "
    "(n free, snap_rational=False as in the test).",
]


class IntegrationFailure(Exception):
    pass


class ProtocolUnrealizable(Exception):
    pass


class GuardTimeout(BaseException):
    """BaseException so that no broad `except Exception` inside a fit swallows it."""


# ---------------------------------------------------------------------------------
# ODE machinery (exact forward sensitivities; lambdified once per process)
# ---------------------------------------------------------------------------------
def _integrate(fun, z0, t_eval, tol, dim):
    calls = [0]

    def wrapped(t, z):
        calls[0] += 1
        if calls[0] > MAX_RHS_CALLS:
            raise IntegrationFailure(f"RHS-call budget {MAX_RHS_CALLS} exceeded")
        return fun(t, z)

    def blowup(t, z):
        return BLOWUP - float(np.max(np.abs(z[:dim])))

    blowup.terminal = True
    sol = solve_ivp(
        wrapped, (0.0, float(t_eval[-1])), np.asarray(z0, dtype=np.float64), t_eval=t_eval,
        method="LSODA", rtol=tol, atol=tol, events=blowup,
    )
    if not sol.success or sol.y.shape[1] != t_eval.size or not np.all(np.isfinite(sol.y)):
        raise IntegrationFailure(f"integration failed: {sol.message} ({sol.y.shape[1]}/{t_eval.size} pts)")
    return sol.y


class OdeModel:
    def __init__(self, eq_str: str, n_params: int, dim: int):
        self.dim, self.n = dim, n_params
        self.xs = sp.symbols([f"x_{i}" for i in range(dim)])
        self.cs = sp.symbols([f"c_{i}" for i in range(n_params)])
        comps = [sp.sympify(s) for s in eq_str.split("|")]
        # id 44: Abs(x)*x -> sqrt(x^2)*x, exactly as EXP-261 (lambdifiable derivative)
        comps = [c.replace(sp.Abs, lambda a: sp.sqrt(a**2)) for c in comps]
        assert len(comps) == dim
        self.comps = comps
        jx = sp.Matrix(comps).jacobian(self.xs)
        jc = sp.Matrix(comps).jacobian(self.cs)
        flat = list(comps) + list(jx) + list(jc)  # Matrix iterates row-major
        self.flat_lam = sp.lambdify([self.xs, self.cs], flat, "numpy", cse=True)
        self.f_lam = sp.lambdify([self.xs, self.cs], comps, "numpy", cse=True)
        self.ws = sp.symbols([f"w_{i}" for i in range(dim)])
        self.field_expr = sum(w * c for w, c in zip(self.ws, comps, strict=True))

    def _vec(self, lam, X, th):
        nt = X.shape[1]
        vals = lam([X[i] for i in range(self.dim)], th)
        return np.array([np.broadcast_to(np.asarray(v, dtype=np.float64), (nt,)) for v in vals])

    def solve(self, th, ic, t_eval, *, sens: bool, tol: float):
        th = np.asarray(th, dtype=np.float64)
        dim, n = self.dim, self.n
        ic = np.asarray(ic, dtype=np.float64)
        if not sens:
            def rhs(t, x):
                return np.array(self.f_lam(x, th), dtype=np.float64)

            return _integrate(rhs, ic, t_eval, tol, dim), None
        o1, o2 = dim, dim + dim * dim

        def aug(t, z):
            x = z[:dim]
            S = z[dim:].reshape(dim, n)
            v = np.array(self.flat_lam(x, th), dtype=np.float64)
            return np.concatenate([v[:o1], (v[o1:o2].reshape(dim, dim) @ S + v[o2:].reshape(dim, n)).ravel()])

        Z = _integrate(aug, np.concatenate([ic, np.zeros(dim * n)]), t_eval, tol, dim)
        X = Z[:dim]
        S = Z[dim:].T.reshape(t_eval.size, dim, n)  # (nt, dim, n)
        return X, S

    def observe(self, X, S, th, obs: str):
        """obs 'full' (T1): [u.ravel(), du.ravel()] as EXP-261; 'first' (T2): u[0]."""
        dim, n = self.dim, self.n
        nt = X.shape[1]
        th = np.asarray(th, dtype=np.float64)
        if obs == "first":
            y = X[0].copy()
            J = None if S is None else S[:, 0, :].copy()
            ch = np.zeros(nt, dtype=int)
            return y, J, ch
        ch = np.concatenate([np.repeat(np.arange(dim), nt), dim + np.repeat(np.arange(dim), nt)])
        if S is None:
            dX = self._vec(self.f_lam, X, th)
            return np.concatenate([X.ravel(), dX.ravel()]), None, ch
        V = self._vec(self.flat_lam, X, th)  # (E, nt)
        o1, o2 = dim, dim + dim * dim
        dX = V[:o1]
        jx = V[o1:o2].T.reshape(nt, dim, dim)
        jc = V[o2:].T.reshape(nt, dim, n)
        dS = jx @ S + jc  # (nt, dim, n)
        y = np.concatenate([X.ravel(), dX.ravel()])
        J = np.vstack([S.transpose(1, 0, 2).reshape(dim * nt, n), dS.transpose(1, 0, 2).reshape(dim * nt, n)])
        return y, J, ch


# ---------------------------------------------------------------------------------
# Parameterisation: sign-preserving log coordinates (additive below 1e-3)
# ---------------------------------------------------------------------------------
class Reparam:
    def __init__(self, theta):
        self.theta = np.asarray(theta, dtype=np.float64)
        self.log_mask = np.abs(self.theta) >= SCALE_FLOOR
        # d theta / d delta at delta = 0: theta* (signed) for log coords, 1e-3 additive
        self.dscale = np.where(self.log_mask, self.theta, SCALE_FLOOR)

    def theta_of(self, d):
        d = np.asarray(d, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(self.log_mask, self.theta * np.exp(d), self.theta + SCALE_FLOOR * d)

    def dtheta(self, d):
        d = np.asarray(d, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(self.log_mask, self.theta * np.exp(d), SCALE_FLOOR)


# ---------------------------------------------------------------------------------
# Task container
# ---------------------------------------------------------------------------------
@dataclass
class Task:
    protocol: str
    sys_id: Any
    name: str
    names: list
    theta: np.ndarray
    rp: Reparam
    y0: np.ndarray
    J0: np.ndarray
    ch0: np.ndarray
    sigma_ch: np.ndarray
    cands: list  # dicts: label, y, J, ch
    evaluate: Callable  # (theta, which (-1 = round 0), jac: bool, tol) -> (y, J | None)
    certify: Callable | None
    dropped: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _sigma_from_round0(y0, ch0, n_ch, notes):
    sig = np.zeros(n_ch)
    all_rms = float(np.sqrt(np.mean(y0**2))) if y0.size else 0.0
    for k in range(n_ch):
        v = y0[ch0 == k]
        rms = float(np.sqrt(np.mean(v**2))) if v.size else 0.0
        if rms > 0 and np.isfinite(rms):
            sig[k] = NOISE_FRAC * rms
        elif all_rms > 0:
            sig[k] = NOISE_FRAC * all_rms
            notes.append(f"channel {k} has zero round-0 RMS; sigma = 1% of all-channel RMS")
        else:
            sig[k] = NOISE_FRAC
            notes.append(f"channel {k} has zero round-0 RMS and so do all channels; sigma = 0.01")
    return sig


_MODEL_CACHE: dict = {}


def _eq_by_id(eq_id):
    for e in strogatz_ode.equations:
        if e["id"] == eq_id:
            return e
    raise KeyError(eq_id)


def build_ode_task(eq_id: int, protocol: str) -> Task:
    eq = _eq_by_id(eq_id)
    bounds = json.load(open(ACES / "llm-aces/ic_bounds.json"))
    dim = eq["dim"]
    theta = np.asarray(eq["consts"][0], dtype=np.float64)
    n = theta.size
    if eq_id not in _MODEL_CACHE:
        _MODEL_CACHE[eq_id] = OdeModel(eq["eq"], n, dim)
    model = _MODEL_CACHE[eq_id]
    grid, obs = (T1_GRID, "full") if protocol == "T1" else (T2_GRID, "first")
    notes: list = []

    box = np.asarray(bounds[s261.bounds_key(eq["name"], bounds)], dtype=np.float64)
    if box.shape != (dim, 2):
        virtual_ics = np.empty((0, dim))
        notes.append(f"ic_bounds.json malformed ({box.shape} vs dim {dim}); box class empty (as EXP-261)")
    else:
        rng = np.random.default_rng(s261.ACQ_SEED)
        virtual_ics = rng.uniform(box[:, 0], box[:, 1], size=(s261.N_VIRTUAL, dim))
    specs = [("IC-1 (ODEBench held-out IC)", np.asarray(eq["init"][1], dtype=np.float64))]
    specs += [(f"box IC {k}", ic) for k, ic in enumerate(virtual_ics)]
    ic0 = np.asarray(eq["init"][0], dtype=np.float64)

    def ev(th, ic, jac, tol):
        X, S = model.solve(th, ic, grid, sens=jac, tol=tol)
        y, J, _ = model.observe(X, S, th, obs)
        return y, J

    try:
        X0, S0 = model.solve(theta, ic0, grid, sens=True, tol=TRUTH_TOL)
    except IntegrationFailure as ex:
        raise ProtocolUnrealizable(f"round-0 trajectory fails from IC-0: {ex}") from ex
    y0, J0, ch0 = model.observe(X0, S0, theta, obs)
    cands, dropped, cand_ics = [], [], []
    for label, ic in specs:
        try:
            Xc, Sc = model.solve(theta, ic, grid, sens=True, tol=TRUTH_TOL)
        except IntegrationFailure as ex:
            dropped.append(f"{label}: {ex}")
            continue
        yc, Jc, chc = model.observe(Xc, Sc, theta, obs)
        cands.append(dict(label=label, y=yc, J=Jc, ch=chc, ic=[float(v) for v in ic]))
        cand_ics.append(ic)
    if not cands:
        raise ProtocolUnrealizable("every candidate diverges at theta*")
    n_ch = 2 * dim if obs == "full" else 1
    sigma_ch = _sigma_from_round0(y0, ch0, n_ch, notes)

    def evaluate(th, which, jac, tol):
        ic = ic0 if which < 0 else cand_ics[which]
        return ev(th, ic, jac, tol)

    certify = partial(
        certify_class_symmetry_symbolic, model.field_expr, model.cs, [float(t) for t in theta],
        free_vars=list(model.xs) + list(model.ws),
    )
    return Task(
        protocol=protocol, sys_id=eq_id, name=eq["name"], names=[str(c) for c in model.cs],
        theta=theta, rp=Reparam(theta), y0=y0, J0=J0, ch0=ch0, sigma_ch=sigma_ch, cands=cands,
        evaluate=evaluate, certify=certify, dropped=dropped, notes=notes,
    )


# ---------------------------------------------------------------------------------
# T3 descriptive cases
# ---------------------------------------------------------------------------------
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _closed_form_task(case, name, names, theta, expr_by_piece, params, free_sym, round0, cand_defs,
                      channel_of_piece, sigma_ch_fn, certify, notes=None):
    """Generic closed-form task. expr_by_piece: piece name -> sympy expr in (params, free_sym).
    round0 / cand_defs: list of (piece, grid) blocks."""
    lam = {}
    for piece, expr in expr_by_piece.items():
        f = sp.lambdify([free_sym, params], expr, "numpy")
        g = sp.lambdify([free_sym, params], [sp.diff(expr, p) for p in params], "numpy")
        lam[piece] = (f, g)
    theta = np.asarray(theta, dtype=np.float64)

    def eval_blocks(blocks, th, jac):
        ys, Js, chs = [], [], []
        for piece, grid in blocks:
            grid = np.asarray(grid, dtype=np.float64)
            f, g = lam[piece]
            with np.errstate(all="ignore"):
                ys.append(np.broadcast_to(np.asarray(f(grid, th), dtype=np.float64), grid.shape).copy())
                if jac:
                    Js.append(np.column_stack([np.broadcast_to(np.asarray(v, dtype=np.float64), grid.shape)
                                               for v in g(grid, th)]))
            chs.append(np.full(grid.size, channel_of_piece[piece], dtype=int))
        y = np.concatenate(ys)
        if not np.all(np.isfinite(y)):
            raise IntegrationFailure("non-finite closed-form prediction")
        J = np.vstack(Js) if jac else None
        if jac and not np.all(np.isfinite(J)):
            raise IntegrationFailure("non-finite closed-form sensitivity")
        return y, J, np.concatenate(chs)

    y0, J0, ch0 = eval_blocks(round0, theta, True)
    cands = []
    for label, blocks in cand_defs:
        yc, Jc, chc = eval_blocks(blocks, theta, True)
        cands.append(dict(label=label, y=yc, J=Jc, ch=chc))
    notes = list(notes or [])
    sigma_ch = sigma_ch_fn(y0, ch0, notes)

    def evaluate(th, which, jac, tol):
        blocks = round0 if which < 0 else cand_defs[which][1]
        y, J, _ = eval_blocks(blocks, np.asarray(th, dtype=np.float64), jac)
        return y, J

    return Task(protocol="T3", sys_id=case, name=name, names=names, theta=theta, rp=Reparam(theta),
                y0=y0, J0=J0, ch0=ch0, sigma_ch=sigma_ch, cands=cands, evaluate=evaluate,
                certify=certify, notes=notes)


def build_t3_task(case: str) -> Task:
    if case == "PK":
        pk = _load_module(REPO / "scripts/diagnostic_pk_bioavailability.py", "pk262")
        t = sp.Symbol("t", positive=True)
        ka, ke, V, F = sp.symbols("ka ke V F", positive=True)
        D = pk.DOSE
        oral = (F * D * ka) / (V * (ka - ke)) * (sp.exp(-ke * t) - sp.exp(-ka * t))
        iv = (D / V) * sp.exp(-ke * t)
        t_std = np.linspace(0.25, 24.0, 24)
        t_dense = np.linspace(0.25, 24.0, 96)
        t_long = np.linspace(0.25, 72.0, 48)
        theta = [float(v) for v in pk.THETA]
        cert = partial(certify_class_symmetry_symbolic, oral, [ka, ke, V, F], theta, free_vars=[t])

        def sig_fn(y0, ch0, notes):
            return _sigma_from_round0(y0, ch0, 1, notes)

        task = _closed_form_task(
            "PK", "Oral one-compartment PK", pk.NAMES, theta, {"oral": oral, "iv": iv},
            [ka, ke, V, F], t, [("oral", t_std)],
            [("oral, 4x denser sampling (96 pts, 24 h)", [("oral", t_dense)]),
             ("oral, 72 h window (48 pts)", [("oral", t_long)]),
             ("oral + intravenous arm (24 + 24 pts)", [("oral", t_std), ("iv", t_std)])],
            {"oral": 0, "iv": 0}, sig_fn, cert,
        )
        # cross-check the sympy closed form against the script's numpy model
        assert np.allclose(task.y0, pk._oral(np.asarray(theta), t_std), rtol=1e-12, atol=1e-12)
        return task
    if case == "ALIEN":
        from ascension.benchmarks.alien_identifiability_loop import amplitude_predict

        P = sp.Symbol("P", positive=True)
        beta, p = sp.symbols("beta p", positive=True)
        amp = beta * P**p
        theta = [1.0, 1.0]
        cert = partial(certify_class_symmetry_symbolic, amp, [beta, p], theta, free_vars=[P])

        def sig_fn(y0, ch0, notes):
            return _sigma_from_round0(y0, ch0, 1, notes)

        task = _closed_form_task(
            "ALIEN", "Alien amplitude A = beta * product^p", ["beta", "p"], theta, {"amp": amp},
            [beta, p], P, [("amp", [1.0])],
            [("single product [1]", [("amp", [1.0])]),
             ("two distinct products [1, 2]", [("amp", [1.0, 2.0])])],
            {"amp": 0}, sig_fn, cert,
        )
        assert np.allclose(task.cands[1]["y"], amplitude_predict([1.0, 2.0])(np.asarray(theta)))
        return task
    if case == "SEQ":
        n = sp.Symbol("n", positive=True)
        th0, th1, th2 = sp.symbols("theta0 theta1 theta2", positive=True)
        logf = sp.log(th0) + th1 * sp.log(n) + n * sp.log(th2)
        theta = [1.0, 0.0, 1.0]
        cert = partial(certify_class_symmetry_symbolic, logf, [th0, th1, th2], theta, free_vars=[n],
                       snap_rational=False)

        def sig_fn(y0, ch0, notes):
            notes.append("round-0 RMS of log f is 0 at theta = (1, 0, 1); sigma = 0.01 absolute on log f")
            return np.array([NOISE_FRAC])

        return _closed_form_task(
            "SEQ", "Sequence ansatz log f = log th0 + th1 log n + n log th2", ["theta0", "theta1", "theta2"],
            theta, {"logf": logf}, [th0, th1, th2], n, [("logf", [400.0, 401.0, 402.0])],
            [("narrow window n = [400, 401, 402]", [("logf", [400.0, 401.0, 402.0])]),
             ("geometric spread n = [2, 5, 20, 100, 400]", [("logf", [2.0, 5.0, 20.0, 100.0, 400.0])])],
            {"logf": 0}, sig_fn, cert,
        )
    if case == "MM":
        mm = _load_module(REPO / "tests/diagnostics/test_exp087_michaelis_menten.py", "mm262")
        model = OdeModel("-c_0*c_1*x_0/(c_2 + x_0)", 3, 1)
        theta = np.asarray(mm.THETA, dtype=np.float64)
        grid = np.asarray(mm.STD_T, dtype=np.float64)
        s0_round0 = 5.0
        s0_cands = [float(v) for v in np.geomspace(0.2, 20.0, 10)]

        def evaluate(th, which, jac, tol):
            th = np.asarray(th, dtype=np.float64)
            if which == len(s0_cands):  # E0 assay
                y = np.array([th[1]])
                return y, (np.array([[0.0, 1.0, 0.0]]) if jac else None)
            s0 = s0_round0 if which < 0 else s0_cands[which]
            X, S = model.solve(th, [s0], grid, sens=jac, tol=tol)
            return X[0].copy(), (S[:, 0, :].copy() if jac else None)

        y0, J0 = evaluate(theta, -1, True, TRUTH_TOL)
        assert np.allclose(y0, mm._substrate(theta, 5.0, grid), rtol=1e-8, atol=1e-10)
        ch0 = np.zeros(y0.size, dtype=int)
        cands = []
        for i, s0 in enumerate(s0_cands):
            yc, Jc = evaluate(theta, i, True, TRUTH_TOL)
            cands.append(dict(label=f"substrate run S0 = {s0:.3g}", y=yc, J=Jc, ch=np.zeros(yc.size, dtype=int)))
        yc, Jc = evaluate(theta, len(s0_cands), True, TRUTH_TOL)
        cands.append(dict(label="E0 assay (direct enzyme measurement)", y=yc, J=Jc, ch=np.ones(1, dtype=int)))
        notes: list = ["E0-assay channel absent at round 0: sigma = 1% of E0*"]
        sig = _sigma_from_round0(y0, ch0, 1, notes)
        sigma_ch = np.array([sig[0], NOISE_FRAC * abs(theta[1])])

        def certify(direction):
            return certify_class_symmetry_numeric(mm._sampler, list(theta), direction, n_designs=32,
                                                  seed=87, param_names=["kcat", "E0", "Km"])

        return Task(protocol="T3", sys_id="MM", name="Michaelis-Menten (substrate observed)",
                    names=["kcat", "E0", "Km"], theta=theta, rp=Reparam(theta), y0=y0, J0=J0, ch0=ch0,
                    sigma_ch=sigma_ch, cands=cands, evaluate=evaluate, certify=certify, notes=notes)
    raise KeyError(case)


# ---------------------------------------------------------------------------------
# Criteria, M1, M2
# ---------------------------------------------------------------------------------
def _sv(*blocks):
    return np.linalg.svd(np.vstack(blocks), compute_uv=False)


def disagree_score(preds):
    P = [p for p in preds if p is not None]
    if len(P) < 2:
        return float("nan"), len(P)
    P = np.asarray(P)
    ybar = P.mean(axis=0)
    rms = float(np.sqrt(np.mean(ybar**2)))
    if not (np.isfinite(rms) and rms > 0):
        return float("nan"), len(P)
    d = [np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i + 1, len(P))]
    return float(np.mean(d)) / rms, len(P)


def _rollouts(task, deltas, which, tol):
    out = []
    for d in deltas:
        th = task.rp.theta_of(d)
        if not np.all(np.isfinite(th)):
            out.append(None)
            continue
        try:
            y, _ = task.evaluate(th, which, False, tol)
            out.append(y if np.all(np.isfinite(y)) else None)
        except Exception:  # noqa: BLE001 — a failed rollout is recorded as excluded, never fatal
            out.append(None)
    return out


def fit_once(task, which, y_obs, sig, d_start):
    rp = task.rp
    m, n = y_obs.size, task.theta.size
    last: dict = {}
    fails: list = []

    def model(d):
        key = np.asarray(d, dtype=np.float64).tobytes()
        if key in last:
            return last[key]
        out = (None, None, False)
        th = rp.theta_of(d)
        if np.all(np.isfinite(th)):
            try:
                y0m, J0m = task.evaluate(th, -1, True, FIT_TOL)
                ycm, Jcm = task.evaluate(th, which, True, FIT_TOL)
                y = np.concatenate([y0m, ycm])
                J = np.vstack([J0m, Jcm]) * rp.dtheta(d)[None, :]
                if np.all(np.isfinite(y)) and np.all(np.isfinite(J)):
                    out = (y, J, True)
            except Exception as ex:  # noqa: BLE001 — trial point fails; LM backs off
                fails.append(type(ex).__name__)
        last.clear()
        last[key] = out
        return out

    def fun(d):
        y, _, ok = model(d)
        return (y - y_obs) / sig if ok else np.full(m, PENALTY)

    def jac(d):
        _, J, ok = model(d)
        return J / sig[:, None] if ok else np.zeros((m, n))

    try:
        res = least_squares(fun, d_start, jac=jac, method="lm")
    except Exception as ex:  # noqa: BLE001 — recorded as a failed fit
        return None, f"least_squares raised {type(ex).__name__}: {str(ex)[:120]}", 0
    if not np.all(np.isfinite(res.x)) or not np.isfinite(res.cost):
        return None, "non-finite estimate or cost", res.nfev
    if res.status <= 0:
        return None, f"lm status {res.status} ({res.message})", res.nfev
    if not model(res.x)[2]:
        return None, f"final point does not integrate ({sorted(set(fails))})", res.nfev
    return res.x.copy(), None, res.nfev


def analyze_task(task: Task, *, light: bool = False) -> dict:
    t_start = time.time()
    n = task.theta.size
    ds = task.rp.dscale
    sig0 = task.sigma_ch[task.ch0]
    A0 = task.J0 * ds[None, :] / sig0[:, None]
    Acs = [c["J"] * ds[None, :] / task.sigma_ch[c["ch"]][:, None] for c in task.cands]
    labels = [c["label"] for c in task.cands]
    nc = len(labels)

    # --- round-0 verdict and certificate --------------------------------------------
    v0 = practical_identifiability(lambda _t: np.zeros(0), task.theta, sensitivity=A0,
                                   param_names=task.names, cond_threshold=COND_THRESHOLD)
    certs = []
    if not v0.identifiable and task.certify is not None:
        for d_delta in v0.confounded_directions:
            d_theta = ds * np.asarray(d_delta)
            d_theta = d_theta / (np.linalg.norm(d_theta) or 1.0)
            try:
                c = task.certify(d_theta)
                certs.append(dict(certified=bool(c.certified), method=c.method,
                                  direction_theta=[float(v) for v in c.direction],
                                  residual=c.residual, regular_point=c.regular_point,
                                  class_rank_at_point=c.class_rank_at_point,
                                  class_rank_nearby=c.class_rank_nearby, error=None))
            except Exception as ex:  # conservative: an erroring certificate does not certify
                certs.append(dict(certified=False, method="symbolic", direction_theta=list(map(float, d_theta)),
                                  residual=None, regular_point=None, class_rank_at_point=None,
                                  class_rank_nearby=None, error=f"{type(ex).__name__}: {ex}"))
    if v0.identifiable:
        triage = "capability"
    elif certs and all(c["certified"] for c in certs):
        triage = "change_class"
    else:
        triage = "design_within_class"

    # --- M1 and the Fisher criteria -----------------------------------------------------
    F0 = A0.T @ A0
    eps = EPS_RIDGE * float(np.trace(F0))
    ridge = math.sqrt(eps) * np.eye(n)
    prior = np.eye(n) / TAU
    base_voi = float(np.sum(np.log(_sv(A0, prior))))
    ranks, conds, nulls, dopt, aopt, voi = [], [], [], [], [], []
    for Ac in Acs:
        v = practical_identifiability(lambda _t: np.zeros(0), task.theta, sensitivity=np.vstack([A0, Ac]),
                                      param_names=task.names, cond_threshold=COND_THRESHOLD)
        ranks.append(v.rank)
        conds.append(float(v.condition_number))
        nulls.append(np.array(v.confounded_directions).T if v.rank < n else np.zeros((n, 0)))
        s = _sv(A0, Ac, ridge)
        with np.errstate(divide="ignore"):
            dopt.append(float(2.0 * np.sum(np.log(s))))
            aopt.append(float(np.sum(1.0 / s**2)))
        voi.append(float(np.sum(np.log(_sv(A0, Ac, prior)))) - base_voi)
    max_rank = max(ranks)
    ranking = rank_designs_by_identifiability(
        {lab: np.vstack([A0, Ac]) for lab, Ac in zip(labels, Acs, strict=True)}, task.theta,
        param_names=task.names,
    )
    rank_score = {d.label: float(d.score) for d in ranking.ranked}

    # --- DISAGREE ------------------------------------------------------------------------
    lam, V = np.linalg.eigh(F0 + np.eye(n) / TAU**2)
    sig_half = V @ np.diag(1.0 / np.sqrt(lam)) @ V.T  # Sigma^(1/2), Sigma = (F0 + I/tau^2)^-1
    Z = np.random.default_rng(DISAGREE_SEED).standard_normal((K_DRAWS, n))
    deltas = Z @ sig_half.T
    dis, dis_valid = [], []
    for ci in range(nc):
        s, k = disagree_score(_rollouts(task, deltas, ci, ROLL_TOL))
        dis.append(s)
        dis_valid.append(k)

    # --- picks -----------------------------------------------------------------------------
    def argbest(vals, maximize=True):
        a = np.array([(-np.inf if maximize else np.inf) if not np.isfinite(x) and np.isnan(x) else x
                      for x in vals], dtype=float)
        return int(np.argmax(a) if maximize else np.argmin(a))

    picks = {
        "RANK": labels.index(ranking.best),
        "DOPT": argbest(dopt, True),
        "AOPT": argbest(aopt, False),
        "VOI": argbest(voi, True),
        "DISAGREE": argbest(dis, True),
    }
    max_idx = [i for i in range(nc) if ranks[i] == max_rank]
    picks["HYBRID"] = max_idx[int(np.argmax([dopt[i] for i in max_idx]))]

    # --- M2 -----------------------------------------------------------------------------
    m2 = [[float("nan")] * len(NOISE_SEEDS) for _ in range(nc)]
    m2_fail = [[None] * len(NOISE_SEEDS) for _ in range(nc)]
    nfev_total = 0
    if not light:
        m0 = task.y0.size
        for si, seed in enumerate(NOISE_SEEDS):
            for ci, c in enumerate(task.cands):
                rng = np.random.default_rng(seed)
                d_start = rng.normal(0.0, START_SD, size=n)
                z0 = rng.standard_normal(m0)
                zc = rng.standard_normal(c["y"].size)
                sig = np.concatenate([sig0, task.sigma_ch[c["ch"]]])
                y_obs = np.concatenate([task.y0, c["y"]]) + sig * np.concatenate([z0, zc])
                est, why, nfev = fit_once(task, ci, y_obs, sig, d_start)
                nfev_total += nfev
                if est is None:
                    m2_fail[ci][si] = why
                    continue
                e = est.copy()
                N = nulls[ci]
                if N.shape[1]:
                    e = e - N @ (N.T @ e)
                r = ranks[ci]
                m2[ci][si] = float(np.linalg.norm(e) / math.sqrt(r)) if r > 0 else float("nan")
    m2_mean = []
    for ci in range(nc):
        vals = np.array(m2[ci], dtype=float)
        m2_mean.append(float("inf") if np.any(~np.isfinite(vals)) else float(np.mean(vals)))

    # --- P4 contribution along certified directions -----------------------------------
    p4 = None
    if triage == "change_class" and not light:
        p4 = []
        for c in certs:
            d_delta = np.asarray(c["direction_theta"]) / ds
            d_delta = d_delta / np.linalg.norm(d_delta)
            par = np.outer(deltas @ d_delta, d_delta)
            perp = deltas - par
            per_cand = []
            for ci in range(nc):
                full, _ = disagree_score(_rollouts(task, deltas, ci, P4_TOL))
                spar, _ = disagree_score(_rollouts(task, par, ci, P4_TOL))
                sperp, _ = disagree_score(_rollouts(task, perp, ci, P4_TOL))
                per_cand.append(dict(label=labels[ci], full=full, along_d=spar, without_d=sperp,
                                     ratio_along=spar / full if full else float("nan"),
                                     rel_change_without=abs(full - sperp) / full if full else float("nan")))
            p4.append(dict(direction_delta=[float(v) for v in d_delta], per_candidate=per_cand,
                           max_ratio_along=max(x["ratio_along"] for x in per_cand),
                           max_rel_change_without=max(x["rel_change_without"] for x in per_cand)))

    cand_rows = []
    for ci in range(nc):
        cand_rows.append(dict(
            idx=ci, label=labels[ci], rank=int(ranks[ci]), cond=conds[ci], is_max_rank=ranks[ci] == max_rank,
            scores=dict(RANK=rank_score[labels[ci]], DOPT=dopt[ci], AOPT=aopt[ci], VOI=voi[ci], DISAGREE=dis[ci]),
            disagree_valid=dis_valid[ci], m2_seeds=m2[ci], m2_mean=m2_mean[ci],
            m2_fail=m2_fail[ci], null_dim=int(nulls[ci].shape[1]),
        ))
    return dict(
        protocol=task.protocol, sys_id=task.sys_id, name=task.name, n_params=int(n),
        names=list(task.names), theta=[float(v) for v in task.theta],
        rank0=int(v0.rank), cond0=float(v0.condition_number), status0=v0.status,
        round0_null=[list(map(float, d)) for d in v0.confounded_directions] if v0.rank < n else [],
        triage=triage, certificates=certs, max_rank=int(max_rank), candidates=cand_rows, picks=picks,
        n_candidates=nc, dropped=task.dropped, notes=task.notes, sigma_ch=[float(s) for s in task.sigma_ch],
        eps=eps, p4=p4, nfev_total=int(nfev_total), elapsed_s=round(time.time() - t_start, 2), error=None,
    )


# ---------------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------------
def _alarm(_signum, _frame):
    raise GuardTimeout(f"per-unit wall-clock guard ({GUARD_SECONDS} s) fired")


def _init_worker():
    np.seterr(all="ignore")
    signal.signal(signal.SIGALRM, _alarm)


def run_unit(unit) -> dict:
    protocol, sys_id = unit
    t0 = time.time()
    signal.alarm(GUARD_SECONDS)
    try:
        task = build_t3_task(sys_id) if protocol == "T3" else build_ode_task(sys_id, protocol)
        out = analyze_task(task)
    except GuardTimeout as ex:
        out = dict(protocol=protocol, sys_id=sys_id, error=f"GuardTimeout: {ex}")
    except ProtocolUnrealizable as ex:
        out = dict(protocol=protocol, sys_id=sys_id, error=f"ProtocolUnrealizable: {ex}")
    except Exception as ex:  # loud, recorded, excluded from denominators
        out = dict(protocol=protocol, sys_id=sys_id, error=f"{type(ex).__name__}: {ex}",
                   traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
    out["wall_s"] = round(time.time() - t0, 2)
    if "name" not in out:
        try:
            out["name"] = _eq_by_id(sys_id)["name"] if protocol != "T3" else str(sys_id)
        except Exception:
            out["name"] = str(sys_id)
    return out


# ---------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------
def _fin(x: float):
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return _fin(o)
    return o


def task_outcomes(t) -> dict:
    cands = t["candidates"]
    maxr = t["max_rank"]
    out = {}
    for c in DET_CRITERIA:
        i = t["picks"][c]
        out[c] = dict(idx=i, label=cands[i]["label"], rank=cands[i]["rank"], hit=float(cands[i]["rank"] == maxr),
                      m2=float(cands[i]["m2_mean"]))
    m2s = np.array([c["m2_mean"] for c in cands], dtype=float)
    out["RANDOM"] = dict(idx=None, label="uniform (expectation)", rank=None,
                         hit=float(np.mean([c["rank"] == maxr for c in cands])),
                         m2=float("inf") if np.any(~np.isfinite(m2s)) else float(np.mean(m2s)))
    return out


def _s(x):
    return np.where(np.isfinite(x), x, INF_SENTINEL)


def _disp(x):
    return float("inf") if x >= INF_SENTINEL / 10 else float(x)


def median_ci(M: np.ndarray):
    """M: (N tasks, C criteria). Paired percentile bootstrap of the median."""
    N = M.shape[0]
    if N == 0:
        return [None] * M.shape[1], [(None, None)] * M.shape[1]
    Ms = _s(M)
    med = np.median(Ms, axis=0)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, N, size=(BOOT_N, N))
    boots = np.median(Ms[idx], axis=1)  # (B, C)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return [_disp(v) for v in med], [(_disp(a), _disp(b)) for a, b in zip(lo, hi, strict=True)]


def binom(k, n):
    if n == 0:
        return dict(k=k, n=n, rate=None, ci=[None, None])
    ci = stats.binomtest(int(k), int(n)).proportion_ci(0.95, method="exact")
    return dict(k=int(k), n=int(n), rate=k / n, ci=[float(ci.low), float(ci.high)])


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank_i, j in enumerate(order):
        running = max(running, min(1.0, (m - rank_i) * pvals[j]))
        adj[j] = running
    return adj


def wilcoxon_family(tasks):
    """Each of WILCOXON_PAIRS vs RANK: one-sided, H1: M2(criterion) < M2(RANK)."""
    rows = []
    for c in WILCOXON_PAIRS:
        x = _s(np.array([task_outcomes(t)[c]["m2"] for t in tasks], dtype=float))
        y = _s(np.array([task_outcomes(t)["RANK"]["m2"] for t in tasks], dtype=float))
        d = x - y
        nz = int(np.sum(d != 0))
        if nz == 0:
            rows.append(dict(criterion=c, n_tasks=len(tasks), n_nonzero=0, statistic=None, p_raw=1.0,
                             note="no non-zero paired differences (identical errors on every task)"))
            continue
        r = stats.wilcoxon(x, y, alternative="less", zero_method="wilcox")
        rows.append(dict(criterion=c, n_tasks=len(tasks), n_nonzero=nz, statistic=float(r.statistic),
                         p_raw=float(r.pvalue), n_lower=int(np.sum(d < 0)), n_higher=int(np.sum(d > 0))))
    adj = holm(np.array([r["p_raw"] for r in rows]))
    for r, a in zip(rows, adj, strict=True):
        r["p_holm"] = float(a)
    return rows


def aggregate(results: list, out: Path, runtime: dict) -> dict:
    errored = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]
    ode = [r for r in ok if r["protocol"] in ("T1", "T2")]
    t3 = [r for r in ok if r["protocol"] == "T3"]
    by_proto = {p: sorted([r for r in ode if r["protocol"] == p], key=lambda r: r["sys_id"]) for p in ("T1", "T2")}
    pooled = by_proto["T1"] + by_proto["T2"]

    def in_p1(t):
        return t["rank0"] < t["n_params"] and t["max_rank"] > t["rank0"]

    def in_p2(t):
        return sum(c["rank"] == t["max_rank"] for c in t["candidates"]) >= 2

    pops = {}
    for tag, lst in (("T1", by_proto["T1"]), ("T2", by_proto["T2"]), ("pooled", pooled)):
        pops[f"ALL_{tag}"] = lst
        pops[f"P1_{tag}"] = [t for t in lst if in_p1(t)]
        pops[f"P2_{tag}"] = [t for t in lst if in_p2(t)]
    cert_tasks = [t for t in pooled if t["triage"] == "change_class"]

    # ---- rank restoration --------------------------------------------------------------
    restoration = {}
    for pop in ("P1_T1", "P1_T2", "P1_pooled", "ALL_T1", "ALL_T2", "ALL_pooled"):
        lst = pops[pop]
        row = {}
        for c in CRITERIA:
            hits = [task_outcomes(t)[c]["hit"] for t in lst]
            if c == "RANDOM":
                row[c] = dict(expected_rate=float(np.mean(hits)) if hits else None, n=len(hits))
            else:
                row[c] = binom(int(round(sum(hits))), len(hits))
        restoration[pop] = row

    # ---- median M2 ----------------------------------------------------------------------
    medians = {}
    for pop in ("ALL_T1", "ALL_T2", "ALL_pooled", "P2_T1", "P2_T2", "P2_pooled"):
        lst = pops[pop]
        M = np.array([[task_outcomes(t)[c]["m2"] for c in CRITERIA] for t in lst], dtype=float).reshape(len(lst), len(CRITERIA))
        med, ci = median_ci(M)
        n_inf = {c: int(np.sum(~np.isfinite(M[:, j]))) if len(lst) else 0 for j, c in enumerate(CRITERIA)}
        medians[pop] = {c: dict(median=med[j], ci95=list(ci[j]), n=len(lst), n_inf=n_inf[c])
                        for j, c in enumerate(CRITERIA)}

    # ---- Wilcoxon ------------------------------------------------------------------------
    wil = {pop: wilcoxon_family(pops[pop]) for pop in ("P2_pooled", "P2_T1", "P2_T2", "ALL_pooled")
           if len(pops[pop]) > 0}

    # ---- agreement -------------------------------------------------------------------------
    agreement = {}
    for pop in ("ALL_pooled", "ALL_T1", "ALL_T2"):
        lst = pops[pop]
        mat = {}
        for a in CRITERIA:
            mat[a] = {}
            for b in CRITERIA:
                if not lst:
                    mat[a][b] = None
                    continue
                if a == "RANDOM" and b == "RANDOM":
                    v = float(np.mean([1.0 / t["n_candidates"] for t in lst]))
                elif "RANDOM" in (a, b):
                    v = float(np.mean([1.0 / t["n_candidates"] for t in lst]))
                else:
                    v = float(np.mean([t["picks"][a] == t["picks"][b] for t in lst]))
                mat[a][b] = v
        agreement[pop] = mat

    # ---- M2 failures -----------------------------------------------------------------------
    fail_counts = {}
    for p in ("T1", "T2", "T3"):
        lst = by_proto.get(p, t3 if p == "T3" else [])
        nfits = sum(len(NOISE_SEEDS) * t["n_candidates"] for t in lst)
        nfail = sum(1 for t in lst for c in t["candidates"] for f in c["m2_fail"] if f)
        reasons: dict = {}
        for t in lst:
            for c in t["candidates"]:
                for f in c["m2_fail"]:
                    if f:
                        key = f.split(" (")[0][:60]
                        reasons[key] = reasons.get(key, 0) + 1
        dis_fail = sum(K_DRAWS - c["disagree_valid"] for t in lst for c in t["candidates"])
        dis_nan = sum(1 for t in lst for c in t["candidates"] if not math.isfinite(c["scores"]["DISAGREE"]))
        inf_cands = sum(1 for t in lst for c in t["candidates"] if not math.isfinite(c["m2_mean"]))
        fail_counts[p] = dict(n_fits=nfits, n_failed=nfail, reasons=reasons, disagree_failed_rollouts=dis_fail,
                              disagree_nan_scores=dis_nan, candidates_with_inf_m2=inf_cands)

    # ---- predictions -------------------------------------------------------------------------
    preds = {}
    # P1
    p1 = restoration["P1_pooled"]
    n_p1 = len(pops["P1_pooled"])
    r = {c: (p1[c]["rate"] if c != "RANDOM" else p1[c]["expected_rate"]) for c in CRITERIA}
    if n_p1 < 5:
        st, why = "NOT SCORED", f"only {n_p1} tasks in the P1 setting across T1+T2 (< 5): reported descriptively"
        met = fals = None
    else:
        met = (r["RANK"] == 1 and r["HYBRID"] == 1 and all(r[c] >= 0.9 for c in ("DOPT", "AOPT", "VOI"))
               and r["DISAGREE"] >= 0.75)
        fals = any(r[c] < 0.75 for c in ("DOPT", "AOPT", "VOI")) or r["DISAGREE"] < 0.5
        st = "FAILED" if fals else ("HELD" if met else "NOT MET")
        why = ", ".join(f"{c} {p1[c]['k']}/{p1[c]['n']}" for c in DET_CRITERIA)
    preds["P-EXP-262-1"] = dict(status=st, n_tasks=n_p1, rates=r, prediction_met=met, falsifier_fired=fals, detail=why)
    # P2
    med2 = medians["P2_pooled"]
    n_p2 = len(pops["P2_pooled"])
    if n_p2 == 0:
        preds["P-EXP-262-2"] = dict(status="NOT SCORED", detail="no task with >= 2 max-rank candidates")
    else:
        mR, mD = med2["RANK"]["median"], med2["DOPT"]["median"]
        wd = next(w for w in wil["P2_pooled"] if w["criterion"] == "DOPT")
        met = (mR >= mD) and wd["p_holm"] < 0.05
        fals = mR < 0.95 * mD
        st = "FAILED" if fals else ("HELD" if met else "NOT MET")
        preds["P-EXP-262-2"] = dict(status=st, n_tasks=n_p2, median_RANK=mR, median_DOPT=mD,
                                    ratio_RANK_over_DOPT=(mR / mD if mD else None),
                                    wilcoxon_DOPT_lt_RANK=wd, prediction_met=met, falsifier_fired=fals)
    # P3
    medA = medians["ALL_pooled"]
    mH, mD = medA["HYBRID"]["median"], medA["DOPT"]["median"]
    rrH = restoration["P1_pooled"]["HYBRID"]
    within = abs(mH / mD - 1) <= 0.05 if mD else (mH == mD)
    rr_ok = (rrH["rate"] == 1.0) if rrH["n"] else True
    fals = mH > 1.10 * mD
    met = within and rr_ok
    preds["P-EXP-262-3"] = dict(
        status="FAILED" if fals else ("HELD" if met else "NOT MET"), population="ALL tasks, T1+T2 pooled",
        n_tasks=len(pops["ALL_pooled"]), median_HYBRID=mH, median_DOPT=mD, ratio=(mH / mD if mD else None),
        hybrid_rank_restoration_P1=rrH, secondary_P2=dict(median_HYBRID=med2["HYBRID"]["median"] if n_p2 else None,
                                                          median_DOPT=med2["DOPT"]["median"] if n_p2 else None),
        prediction_met=met, falsifier_fired=fals)
    # P4
    if not cert_tasks:
        preds["P-EXP-262-4"] = dict(status="NOT MET", detail="no task received a certified change_class verdict "
                                    "(the prediction names at minimum T1 id 1)", n_tasks=0)
    else:
        raised = []
        for t in cert_tasks:
            o = task_outcomes(t)
            for c in DET_CRITERIA:
                if o[c]["rank"] != t["rank0"]:
                    raised.append((t["protocol"], t["sys_id"], c, o[c]["rank"], t["rank0"]))
            if any(cc["rank"] != t["rank0"] for cc in t["candidates"]):
                raised.append((t["protocol"], t["sys_id"], "RANDOM(any candidate)", t["max_rank"], t["rank0"]))
        contrib = max((max(p["max_ratio_along"], p["max_rel_change_without"]) for t in cert_tasks
                       for p in (t["p4"] or [])), default=float("nan"))
        rc_t1 = any(t["protocol"] == "T1" and t["sys_id"] == 1 for t in cert_tasks)
        met = (not raised) and contrib <= 1e-8 and rc_t1
        fals = bool(raised)
        preds["P-EXP-262-4"] = dict(
            status="FAILED" if fals else ("HELD" if met else "NOT MET"),
            certified_tasks=[f"{t['protocol']} id {t['sys_id']} ({t['name']})" for t in cert_tasks],
            rc_circuit_T1_certified=rc_t1, picks_raising_rank=raised,
            max_disagree_contribution_along_certified=contrib,
            experiments_saved=f"1 per round per task on {len(cert_tasks)} tasks (RANK + certificate spends 0; "
                              "every pure selection criterion spends 1 per round)",
            prediction_met=met, falsifier_fired=fals)
    # P5
    mX = {c: medA[c]["median"] for c in CRITERIA}
    met = all(mX["DISAGREE"] >= mX[c] for c in ("DOPT", "AOPT", "VOI"))
    fals = all(mX["DISAGREE"] < mX[c] for c in ("DOPT", "AOPT", "VOI"))
    preds["P-EXP-262-5"] = dict(
        status="FAILED" if fals else ("HELD" if met else "NOT MET"), population="ALL tasks, T1+T2 pooled",
        medians={c: mX[c] for c in ("DOPT", "AOPT", "VOI", "DISAGREE")},
        lowest_among_all_7=min(CRITERIA, key=lambda c: mX[c]),
        secondary_P2={c: med2[c]["median"] for c in ("DOPT", "AOPT", "VOI", "DISAGREE")} if n_p2 else None,
        prediction_met=met, falsifier_fired=fals)

    # ---- M3 ----------------------------------------------------------------------------------
    m3 = dict(n_change_class_tasks=len(cert_tasks),
              spent_per_round={**{c: len(cert_tasks) for c in CRITERIA}, "RANK+certificate": 0})

    # ---- sanity --------------------------------------------------------------------------------
    sanity = dict(
        RANK_hits_max_rank_all_tasks=all(task_outcomes(t)["RANK"]["hit"] == 1 for t in pooled),
        HYBRID_hits_max_rank_all_tasks=all(task_outcomes(t)["HYBRID"]["hit"] == 1 for t in pooled),
        rc_circuit={t["protocol"]: dict(rank0=t["rank0"], n=t["n_params"],
                                        candidate_ranks=[c["rank"] for c in t["candidates"]],
                                        pick_ranks={c: task_outcomes(t)[c]["rank"] for c in DET_CRITERIA})
                    for t in pooled if t["sys_id"] == 1},
    )

    counts = dict(
        units_total=len(results), units_errored=len(errored),
        tasks=dict(T1=len(by_proto["T1"]), T2=len(by_proto["T2"]), T3=len(t3)),
        round0_rank_deficient=dict(T1=sum(t["rank0"] < t["n_params"] for t in by_proto["T1"]),
                                   T2=sum(t["rank0"] < t["n_params"] for t in by_proto["T2"])),
        round0_status={p: {s: sum(t["status0"] == s for t in by_proto[p]) for s in
                           ("identifiable", "structurally_non_identifiable", "practically_non_identifiable")}
                       for p in ("T1", "T2")},
        triage={p: {s: sum(t["triage"] == s for t in by_proto[p]) for s in
                    ("capability", "design_within_class", "change_class")} for p in ("T1", "T2")},
        P1_setting={p: len(pops[f"P1_{p}"]) for p in ("T1", "T2", "pooled")},
        P2_setting={p: len(pops[f"P2_{p}"]) for p in ("T1", "T2", "pooled")},
        P1_setting_ids={p: [t["sys_id"] for t in pops[f"P1_{p}"]] for p in ("T1", "T2")},
        P1_setting_full_rank_restorable={p: sum(t["max_rank"] == t["n_params"] for t in pops[f"P1_{p}"])
                                         for p in ("T1", "T2")},
        rank_deficient_not_restorable={p: [t["sys_id"] for t in by_proto[p]
                                           if t["rank0"] < t["n_params"] and t["max_rank"] == t["rank0"]]
                                       for p in ("T1", "T2")},
        dropped_candidates=[dict(protocol=t["protocol"], id=t["sys_id"], dropped=t["dropped"]) for t in ok if t["dropped"]],
        single_candidate_tasks=[f"{t['protocol']} id {t['sys_id']}" for t in pooled if t["n_candidates"] < 2],
        notes=[dict(protocol=t["protocol"], id=t["sys_id"], notes=t["notes"]) for t in ok if t["notes"]],
    )
    summary = dict(
        experiment="EXP-262", preregistration="predictions.md, block 2026-09-28 (Session 047), commit 8f746eb",
        llm_aces_pin="60d4df7", predictions=preds, counts=counts, rank_restoration=restoration,
        median_m2=medians, wilcoxon=wil, agreement=agreement, m2_fit_failures=fail_counts, m3=m3, sanity=sanity,
        errored=[dict(protocol=r["protocol"], id=r["sys_id"], name=r.get("name"), error=r["error"]) for r in errored],
        t3=[t3_summary(t) for t in sorted(t3, key=lambda r: str(r["sys_id"]))],
        runtime=runtime, implementation_choices=IMPLEMENTATION_CHOICES,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=1))
    write_tsvs(ok, out)
    write_readme(summary, out)
    return summary


def t3_summary(t):
    o = task_outcomes(t)
    return dict(
        case=t["sys_id"], name=t["name"], n_params=t["n_params"], rank0=t["rank0"], cond0=t["cond0"],
        triage=t["triage"], certified=[c["certified"] for c in t["certificates"]],
        candidates=[dict(label=c["label"], rank=c["rank"], m2_mean=c["m2_mean"], n_failed=sum(1 for f in c["m2_fail"] if f))
                    for c in t["candidates"]],
        picks={c: dict(label=o[c]["label"], rank=o[c]["rank"], m2=o[c]["m2"]) for c in CRITERIA},
        notes=t["notes"],
    )


def _f(x, fmt="{:.6g}"):
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return fmt.format(x)


def write_tsvs(ok, out: Path):
    with open(out / "per_task_candidate.tsv", "w") as fh:
        cols = ["protocol", "sys_id", "name", "n_params", "cand_idx", "cand_label", "rank_before", "rank_after",
                "max_rank_over_candidates", "is_max_rank", "cond_after",
                "score_RANK", "score_DOPT", "score_AOPT", "score_VOI", "score_DISAGREE", "disagree_valid_rollouts",
                "m2_mean"] + [f"m2_seed{s}" for s in NOISE_SEEDS] + ["m2_n_failed", "m2_fail_reasons",
                                                                      "round0_status", "certificate_triage"]
        fh.write("\t".join(cols) + "\n")
        for t in sorted(ok, key=lambda r: (r["protocol"], str(r["sys_id"]).zfill(4))):
            for c in t["candidates"]:
                row = [t["protocol"], t["sys_id"], t["name"], t["n_params"], c["idx"], c["label"], t["rank0"],
                       c["rank"], t["max_rank"], c["is_max_rank"], _f(c["cond"])]
                row += [_f(c["scores"][k]) for k in ("RANK", "DOPT", "AOPT", "VOI", "DISAGREE")]
                row += [c["disagree_valid"], _f(c["m2_mean"])] + [_f(v) for v in c["m2_seeds"]]
                fails = [f for f in c["m2_fail"] if f]
                row += [len(fails), " | ".join(fails), t["status0"], t["triage"]]
                fh.write("\t".join(str(v) for v in row) + "\n")
    with open(out / "picks.tsv", "w") as fh:
        cols = ["protocol", "sys_id", "name", "criterion", "pick_label", "pick_is_max_rank", "pick_rank",
                "max_rank", "rank_before", "pick_m2", "oracle_best_label", "oracle_best_m2", "in_P1_setting",
                "in_P2_setting", "certificate_triage"]
        fh.write("\t".join(cols) + "\n")
        for t in sorted(ok, key=lambda r: (r["protocol"], str(r["sys_id"]).zfill(4))):
            o = task_outcomes(t)
            m2s = [c["m2_mean"] for c in t["candidates"]]
            ob = int(np.argmin(_s(np.array(m2s, dtype=float))))
            in_p1 = t["rank0"] < t["n_params"] and t["max_rank"] > t["rank0"]
            in_p2 = sum(c["rank"] == t["max_rank"] for c in t["candidates"]) >= 2
            for c in CRITERIA:
                hit = o[c]["hit"]
                row = [t["protocol"], t["sys_id"], t["name"], c, o[c]["label"],
                       (bool(hit) if c != "RANDOM" else _f(hit, "{:.4f}")), o[c]["rank"] if o[c]["rank"] is not None else "NA",
                       t["max_rank"], t["rank0"], _f(o[c]["m2"]), t["candidates"][ob]["label"], _f(m2s[ob]),
                       in_p1, in_p2, t["triage"]]
                fh.write("\t".join(str(v) for v in row) + "\n")


def _pct(d):
    if d.get("rate") is None:
        return "NA (n = 0)"
    return f"{d['k']}/{d['n']} = {100 * d['rate']:.1f}% [{100 * d['ci'][0]:.1f}, {100 * d['ci'][1]:.1f}]"


def write_readme(S: dict, out: Path):
    L = []
    a = L.append
    a("# EXP-262 — head-to-head of experiment-selection criteria on identical candidate sets\n")
    a("Pre-registration: `predictions.md`, block \"Pre-registered block added 2026-09-28 (Session 047 — EXP-262 ...)\", "
      "commit 8f746eb. Script: `scripts/exp262_head_to_head.py`. $0, no LLM, no network. LLM-ACES pinned at 60d4df7.\n")
    a("Outputs in this directory: `per_task_candidate.tsv` (one row per protocol x system x candidate), `picks.tsv` "
      "(one row per protocol x task x criterion), `summary.json` (every number below), `units.jsonl` (raw per-task "
      "results), `run.log`.\n")
    a("## Predictions\n")
    a("| Prediction | Status | Measured |")
    a("|---|---|---|")
    P = S["predictions"]
    p1 = P["P-EXP-262-1"]
    a(f"| P-EXP-262-1 rank restoration | **{p1['status']}** | n = {p1['n_tasks']} tasks (T1+T2 pooled). "
      + ", ".join(f"{c} {_f(p1['rates'][c], '{:.3f}')}" for c in CRITERIA) + " |")
    p2 = P["P-EXP-262-2"]
    if "median_RANK" in p2:
        w = p2["wilcoxon_DOPT_lt_RANK"]
        a(f"| P-EXP-262-2 RANK tiebreak vs DOPT | **{p2['status']}** | n = {p2['n_tasks']} tasks with >= 2 max-rank "
          f"candidates; median M2 RANK {_f(p2['median_RANK'])} vs DOPT {_f(p2['median_DOPT'])} (ratio "
          f"{_f(p2['ratio_RANK_over_DOPT'], '{:.4f}')}); Wilcoxon DOPT < RANK: {w.get('n_nonzero')} non-zero pairs, "
          f"p_raw = {_f(w['p_raw'], '{:.4g}')}, p_Holm = {_f(w['p_holm'], '{:.4g}')} |")
    else:
        a(f"| P-EXP-262-2 | **{p2['status']}** | {p2.get('detail')} |")
    p3 = P["P-EXP-262-3"]
    a(f"| P-EXP-262-3 HYBRID keeps both | **{p3['status']}** | all tasks pooled (n = {p3['n_tasks']}): median M2 "
      f"HYBRID {_f(p3['median_HYBRID'])} vs DOPT {_f(p3['median_DOPT'])} (ratio {_f(p3['ratio'], '{:.4f}')}); HYBRID "
      f"rank restoration on P1 setting {_pct(p3['hybrid_rank_restoration_P1'])} |")
    p4 = P["P-EXP-262-4"]
    if p4.get("certified_tasks"):
        a(f"| P-EXP-262-4 certified directions | **{p4['status']}** | certified tasks: {', '.join(p4['certified_tasks'])}; "
          f"picks raising rank: {len(p4['picks_raising_rank'])}; max DISAGREE contribution along the certified "
          f"direction {_f(p4['max_disagree_contribution_along_certified'], '{:.2e}')}; {p4['experiments_saved']} |")
    else:
        a(f"| P-EXP-262-4 | **{p4['status']}** | {p4.get('detail')} |")
    p5 = P["P-EXP-262-5"]
    a(f"| P-EXP-262-5 DISAGREE weakest | **{p5['status']}** | all tasks pooled: median M2 "
      + ", ".join(f"{c} {_f(v)}" for c, v in p5["medians"].items()) + f"; lowest of all 7: {p5['lowest_among_all_7']} |")
    a("\nStatus vocabulary: HELD = the prediction as stated is met; FAILED = the registered falsifier fired; NOT MET = "
      "not met but the falsifier did not fire; NOT SCORED = the registered descriptive fallback applies.\n")

    C = S["counts"]
    a("## Counts\n")
    a(f"- Units run: {C['units_total']} ({C['units_errored']} errored). Scored tasks: T1 {C['tasks']['T1']}, "
      f"T2 {C['tasks']['T2']}, T3 {C['tasks']['T3']}.")
    a(f"- Round-0 status: T1 {C['round0_status']['T1']}; T2 {C['round0_status']['T2']}.")
    a(f"- Round-0 rank-deficient: T1 {C['round0_rank_deficient']['T1']}, T2 {C['round0_rank_deficient']['T2']} "
      "(the T2 number was not predicted).")
    a(f"- Triage (certificate): T1 {C['triage']['T1']}; T2 {C['triage']['T2']}.")
    a(f"- P1 setting (rank-deficient round 0, some candidate raises rank): T1 {C['P1_setting']['T1']} "
      f"(ids {C['P1_setting_ids']['T1']}), T2 {C['P1_setting']['T2']} (ids {C['P1_setting_ids']['T2']}); of these, "
      f"full rank reachable: T1 {C['P1_setting_full_rank_restorable']['T1']}, T2 {C['P1_setting_full_rank_restorable']['T2']}.")
    a(f"- Rank-deficient with no candidate raising rank: T1 ids {C['rank_deficient_not_restorable']['T1']}, "
      f"T2 ids {C['rank_deficient_not_restorable']['T2']}.")
    a(f"- P2 setting (>= 2 candidates at max rank): T1 {C['P2_setting']['T1']}, T2 {C['P2_setting']['T2']}, "
      f"pooled {C['P2_setting']['pooled']}.")
    a(f"- Single-candidate tasks: {C['single_candidate_tasks']}.")
    for d in C["dropped_candidates"]:
        a(f"- Dropped candidates, {d['protocol']} id {d['id']}: {'; '.join(d['dropped'])}")
    for d in C["notes"]:
        a(f"- Note, {d['protocol']} id {d['id']}: {'; '.join(d['notes'])}")
    a("\n### Errored units (excluded from every denominator)\n")
    if S["errored"]:
        for e in S["errored"]:
            a(f"- {e['protocol']} id {e['id']} ({e['name']}): {e['error']}")
    else:
        a("- none")
    a("\n### M2 fit failures (NaN, counted; +inf in rankings)\n")
    for p, d in S["m2_fit_failures"].items():
        a(f"- {p}: {d['n_failed']} of {d['n_fits']} fits failed; candidates with +inf mean M2: "
          f"{d['candidates_with_inf_m2']}; reasons {d['reasons']}; DISAGREE failed rollouts {d['disagree_failed_rollouts']}, "
          f"NaN DISAGREE scores {d['disagree_nan_scores']}.")

    a("\n## Rank restoration (M1): pick attains the maximum rank over candidates\n")
    a("Exact binomial (Clopper-Pearson) 95% CIs; RANDOM is the exact expectation of a uniform pick.\n")
    R = S["rank_restoration"]
    for pop in ("P1_T1", "P1_T2", "P1_pooled", "ALL_pooled"):
        a(f"**{pop}**\n")
        a("| Criterion | Rate |")
        a("|---|---|")
        for c in CRITERIA:
            d = R[pop][c]
            a(f"| {c} | " + (_f(d['expected_rate'], '{:.3f}') + f" (expected, n = {d['n']})" if c == "RANDOM" else _pct(d)) + " |")
        a("")

    a("## Median M2 (RMS log-parameter error of the pick; mean over 5 noise seeds)\n")
    a("Percentile bootstrap 95% CI, 10,000 resamples of tasks, seed 262. `n_inf` = tasks where the pick's M2 is +inf "
      "(some fit failed).\n")
    M = S["median_m2"]
    pops = ("ALL_T1", "ALL_T2", "ALL_pooled", "P2_T1", "P2_T2", "P2_pooled")
    a("| Criterion | " + " | ".join(f"{p} (n={M[p]['RANK']['n']})" for p in pops) + " |")
    a("|---|" + "---|" * len(pops))
    for c in CRITERIA:
        cells = []
        for p in pops:
            d = M[p][c]
            cells.append(f"{_f(d['median'], '{:.4g}')} [{_f(d['ci95'][0], '{:.4g}')}, {_f(d['ci95'][1], '{:.4g}')}]"
                         + (f" (inf: {d['n_inf']})" if d['n_inf'] else ""))
        a(f"| {c} | " + " | ".join(cells) + " |")

    a("\n## Paired one-sided Wilcoxon signed-rank vs RANK (H1: criterion's M2 < RANK's M2), Holm across 6 pairs\n")
    for pop, rows in S["wilcoxon"].items():
        tag = "registered family (P2 setting, pooled)" if pop == "P2_pooled" else "secondary"
        a(f"**{pop}** — {tag}\n")
        a("| Criterion | n tasks | non-zero pairs | lower / higher than RANK | W | p raw | p Holm |")
        a("|---|---|---|---|---|---|---|")
        for w in rows:
            a(f"| {w['criterion']} | {w['n_tasks']} | {w['n_nonzero']} | {w.get('n_lower', 0)} / {w.get('n_higher', 0)} | "
              f"{_f(w['statistic'])} | {_f(w['p_raw'], '{:.4g}')} | {_f(w['p_holm'], '{:.4g}')} |")
        a("")

    a("## Top-pick agreement (fraction of tasks where two criteria pick the same candidate; ALL tasks pooled)\n")
    A = S["agreement"]["ALL_pooled"]
    a("| | " + " | ".join(CRITERIA) + " |")
    a("|---|" + "---|" * len(CRITERIA))
    for x in CRITERIA:
        a(f"| {x} | " + " | ".join(_f(A[x][y], '{:.2f}') for y in CRITERIA) + " |")
    a("\nRANDOM entries are the expected agreement of a uniform pick (mean of 1/n_candidates).\n")

    a("## Certified cases (P4) and M3\n")
    if p4.get("certified_tasks"):
        a(f"- Certified change_class tasks: {', '.join(p4['certified_tasks'])}.")
        a(f"- Picks that raised rank on a certified task: {p4['picks_raising_rank'] or 'none'}.")
        a(f"- Max DISAGREE contribution along the certified direction (max over tasks, candidates, and the two "
          f"registered-in-spirit measures): {_f(p4['max_disagree_contribution_along_certified'], '{:.3e}')}.")
    a(f"- M3: {S['m3']['n_change_class_tasks']} change_class tasks. Experiments spent per round: "
      + ", ".join(f"{k} {v}" for k, v in S["m3"]["spent_per_round"].items()) + ".")
    sa = S["sanity"]
    a("\n### Sanity checks\n")
    a(f"- RANK picks a max-rank candidate on every T1/T2 task: {sa['RANK_hits_max_rank_all_tasks']}. HYBRID: "
      f"{sa['HYBRID_hits_max_rank_all_tasks']}.")
    for p, d in sa["rc_circuit"].items():
        a(f"- RC circuit (id 1) {p}: round-0 rank {d['rank0']}/{d['n']}; candidate ranks {d['candidate_ranks']}; "
          f"pick ranks {d['pick_ranks']}.")

    a("\n## T3 (descriptive only)\n")
    a("| Case | round-0 rank | triage | candidate: rank, mean M2 | picks (criterion: candidate, rank, M2) |")
    a("|---|---|---|---|---|")
    for t in S["t3"]:
        cands = "<br>".join(f"{c['label']}: {c['rank']}/{t['n_params']}, {_f(c['m2_mean'], '{:.4g}')}"
                            + (f" ({c['n_failed']} failed)" if c['n_failed'] else "") for c in t["candidates"])
        picks = "<br>".join(f"{k}: {v['label']}, {v['rank'] if v['rank'] is not None else '-'}, {_f(v['m2'], '{:.4g}')}"
                            for k, v in t["picks"].items())
        a(f"| {t['name']} | {t['rank0']}/{t['n_params']} (cond {_f(t['cond0'], '{:.3g}')}) | {t['triage']} "
          f"(certified {t['certified']}) | {cands} | {picks} |")
    for t in S["t3"]:
        if t["notes"]:
            a(f"\n- {t['case']} notes: {'; '.join(t['notes'])}")

    a("\n## Runtime\n")
    rt = S["runtime"]
    a(f"- Wall clock {rt.get('wall_s')} s on {rt.get('processes')} processes; per-unit wall times in `units.jsonl` "
      f"(slowest: {rt.get('slowest')}).")

    a("\n## Implementation choices (fixed before the first run; none changed after outcomes were seen)\n")
    for i, c in enumerate(S["implementation_choices"], 1):
        a(f"{i}. {c}")
    (out / "README.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "results" / "exp262")
    ap.add_argument("--ids", type=str, default="", help="comma-separated ODEBench ids (smoke)")
    ap.add_argument("--no-t3", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="print only runtime / errors; no aggregation")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    units_path = args.out / "units.jsonl"

    if args.aggregate_only:
        results = [json.loads(line) for line in open(units_path)]
        rt = json.loads((args.out / "runtime.json").read_text()) if (args.out / "runtime.json").exists() else {}
        aggregate(results, args.out, rt)
        print("aggregated", len(results), "units ->", args.out)
        return

    eqs = [e for e in strogatz_ode.equations if len(e["consts"][0]) > 0]
    if args.ids:
        keep = {int(x) for x in args.ids.split(",")}
        eqs = [e for e in eqs if e["id"] in keep]
    units = [("T1", e["id"]) for e in eqs] + [("T2", e["id"]) for e in eqs]
    if not args.no_t3:
        units += [("T3", c) for c in ("PK", "MM", "ALIEN", "SEQ")]

    def cost(u):
        if u[0] == "T3":
            return 1
        e = _eq_by_id(u[1])
        n = len(e["consts"][0])
        return e["dim"] * (n + 1) * n * (3 if u[0] == "T1" else 1)

    units.sort(key=cost, reverse=True)
    print(f"EXP-262: {len(units)} units on {args.processes} processes; out = {args.out}", flush=True)
    t0 = time.time()
    results = []
    with open(units_path, "w") as fh, mp.get_context("fork").Pool(args.processes, initializer=_init_worker,
                                                                   maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(run_unit, units, chunksize=1):
            results.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if args.smoke:
                nf = sum(1 for c in r.get("candidates", []) for f in c["m2_fail"] if f)
                print(f"[{len(results):3d}/{len(units)}] {r['protocol']} {r['sys_id']!s:>5} wall={r['wall_s']:7.1f}s "
                      f"cands={r.get('n_candidates')} dropped={len(r.get('dropped', []))} "
                      f"m2_failed_fits={nf} nfev={r.get('nfev_total')} "
                      + (f"ERROR={r['error']}" if r.get("error") else "OK"), flush=True)
            else:
                print(f"[{len(results):3d}/{len(units)}] {r['protocol']} {r['sys_id']!s:>5} wall={r['wall_s']:7.1f}s "
                      + (f"ERROR={r['error']}" if r.get("error") else "ok"), flush=True)
    wall = time.time() - t0
    slow = sorted(results, key=lambda r: -r["wall_s"])[:5]
    runtime = dict(wall_s=round(wall, 1), processes=args.processes,
                   slowest=[f"{r['protocol']} {r['sys_id']}: {r['wall_s']} s" for r in slow],
                   sum_unit_wall_s=round(sum(r["wall_s"] for r in results), 1))
    (args.out / "runtime.json").write_text(json.dumps(runtime, indent=1))
    print(f"done in {wall:.1f} s", flush=True)
    if args.smoke:
        for r in results:
            if r.get("error"):
                print(r.get("traceback", r["error"]))
        return
    aggregate(results, args.out, runtime)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
