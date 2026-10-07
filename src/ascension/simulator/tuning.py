"""Deterministic tuning-analysis library for the Alien Universe (Phase 15.1, TIER2-02).

Plain English: this is the $0-LLM numpy/scipy analysis surface the Tier-2 tuning
sprint runs over the LOCKED 15.0 simulator. It does NOT change any physics — it
chooses constants (``config.TIER2_TUNED_PARAMS``) and ANALYZES the trajectories the
real ``AlienUniverse.run()`` produces, to answer three questions:

  1. Is the universe NON-TRIVIAL? — a pure-Newtonian central-force fit to the
     observed per-step radial acceleration leaves residuals that a nested-model
     F-test rejects as white noise at p<0.01 (Newton alone provably fails). The
     make-or-break statistic. [15.1 RESEARCH §Residual-Structure Test]
  2. Is the true law SOLVABLE? — the true law reproduces held-out trajectories at
     RMSE<1e-3 (here ≈0 / bit-identical, since the proposed law IS the true law).
     [15.1 RESEARCH §Solvability]
  3. Is the universe NOT accidentally polynomial-solvable? — a polynomial-FORCE
     baseline fit on the seen orbit FAILS held-out forward integration (RMSE>1e-2),
     because it lacks the hidden tangential-inertia DOF. The honest D-11 guard
     (held-out forward integration, NOT a per-orbit residual fit — that is the
     L-015 trap). [15.1 RESEARCH §Polynomial-Fittability]

Plus ``q_drift`` (the D-05 admissibility read — reads ``traj.Q``, the polar-
Hamiltonian invariant, NOT ``physics.conserved_Q`` of the reconstructed Cartesian
state — PATTERNS gotcha 8) and ``sweep`` (the EXP-071 measurement object; Plan 02
runs the verdict).

The recommended operating point (couplings stay at §10 starting values; masses=(1,1)
NOT (1000,1); tangential_frac=0.85) is documented on ``config.TIER2_TUNED_PARAMS``.

Import-light posture (mirror ``physics.py:21-22``): numpy + scipy ONLY, NO logging,
NO DB, NO agent imports. L-018 cycle guard: imports only the simulator public
surface (``alien``, ``physics``, ``types``, ``config``). If it ever needs the
alien-audit fn (Plan 03), lazy-import inside the function body.

⚠️ $0 LLM, deterministic numpy/scipy/sympy — no agent in any path (D-10). The §22.6
no-LLM grep gate guards this file (``tests/simulator/test_no_llm_in_simulator.py``
SIMULATOR_FILES); it stays OUT of ``io.SIMULATOR_HASHED_FILES`` (plumbing, not
dynamics — its edits must not change a run's provenance digest).

Binding decisions:
  - D-01/D-02: nested-model F-test on per-step radial acceleration is the locked
    structure test (autocorrelation/runs/Ljung-Box are VACUOUS on smooth
    deterministic residuals — RESEARCH Pitfall 2).
  - D-03: §10 numerical-match held-out RMSE (analog benchmarks/scoring.py:_numerical_rmse).
  - D-05: every candidate cell re-validated against the 1e-6 Q-drift bar via q_drift.
  - D-11: the RIGHT polynomial guard is held-out forward integration, not a
    per-orbit residual fit (L-015 trap, RESEARCH Pitfall 3).

Threat-model mitigations: T-15.1-01 (poly-fittability guard), T-15.1-03 (no-LLM).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp

from ascension.simulator import physics as P
from ascension.simulator.alien import AlienUniverse
from ascension.simulator.config import TIER2_TUNED_PARAMS
from ascension.simulator.types import AlienConfig

# Endpoints trimmed by the central finite-difference accel recovery (rel[k-1..k+1]).
_FD_TRIM = 3

# Band sub-bars for `sweep` (the EXP-071 conjunction). Assert the CONTRACT, not
# the incidental tightness (documented-tolerance discipline).
_P_BAR = 0.01
_RMSE_BAR = 1e-3
_Q_DRIFT_BAR = 1e-6
# Effect-size floor for the Newton-only relative residual. ⚠️ A bare F-test
# p-value is vacuous on smooth deterministic residuals (RESEARCH Pitfall 2): even
# the Newtonian limit's FD-truncation residual (~1e-5 relative) rejects white
# noise. A universe is only genuinely "Newton-fails" if Newton leaves a residual
# that is a MEANINGFUL fraction of the signal. Observed: tuned point ≈6e-2 (and
# the post-(N+α) hidden share is 91%); Newtonian limit ≈2e-5. 1e-3 sits cleanly
# between them — far above the FD floor, far below the tuned structure.
_RESID_FLOOR = 1e-3

# Polynomial-force baseline: solve_ivp settings + a blowup guard span (mirror
# benchmarks/scoring.py:_numerical_rmse blowup-guard posture — non-finite/runaway
# is a loud reason, never a silent NaN).
_POLY_INTEGRATOR_KWARGS = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-12}
_POLY_BLOWUP_SPAN = 1e3  # |state| beyond this on a held-out integration = blowup.


# -----------------------------------------------------------------------------
# Config builders (IC speed derived from physics.potential_U's own gradient)
# -----------------------------------------------------------------------------
def _U_and_dUdr(cfg: AlienConfig):
    """Return ``(U, dUdr)`` closures over the 2-body separation r (single source).

    Mirrors ``fixtures._U_and_dUdr`` (fixtures.py:46-63) verbatim so a tuned IC's
    speed is consistent with the force the integrator applies — NEVER a hardcoded
    v_circ.
    """
    m = np.asarray(cfg.masses, dtype=np.float64)
    ch = np.asarray(cfg.charges, dtype=np.float64)

    def U(r: float) -> float:
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, cfg)

    def dUdr(r: float) -> float:
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    return U, dUdr


def config_from_params(params: dict[str, object]) -> AlienConfig:
    """Build a 2-body ``AlienConfig`` from a TIER2_TUNED_PARAMS-shaped dict.

    ``r0`` and ``tangential_frac`` are INPUTS to the IC-builder (not AlienConfig
    fields): body0 at the origin, body1 at ``(r0, 0)`` with a purely tangential
    relative velocity ``frac · v_circ``, where ``v_circ = sqrt(r0·dUdr(r0)/mu)`` is
    derived from ``physics.potential_U``'s OWN gradient over the FULL tuned
    potential (the fixtures.py idiom). NEVER a hardcoded v_circ.
    """
    # L-064/DET-07: unknown keys must FAIL LOUDLY. This function used to consume
    # a fixed key set and silently ignore everything else — including the regime
    # knobs charge_coupling/correction_exponent — so a future V2-regime sweep
    # through this module would silently measure the V1 law. Regime knobs are now
    # threaded; anything unrecognized raises.
    _known = {
        "G", "alpha", "beta", "gamma", "kappa", "masses", "charges", "r0",
        "tangential_frac", "dt", "n_steps", "integrator", "seed",
        "charge_coupling", "correction_exponent",
    }
    unknown = set(params) - _known
    if unknown:
        raise ValueError(
            f"config_from_params: unrecognized param key(s) {sorted(unknown)} — "
            "refusing to silently ignore them (L-064: a silently-dropped regime "
            "knob measures the wrong universe)."
        )

    G = float(params["G"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    gamma = float(params["gamma"])
    kappa = float(params["kappa"])
    masses = tuple(params["masses"])  # type: ignore[arg-type]
    charges = tuple(params["charges"])  # type: ignore[arg-type]
    r0 = float(params["r0"])
    frac = float(params["tangential_frac"])
    dt = float(params["dt"])
    n_steps = int(params["n_steps"])
    seed = int(params["seed"])

    regime_kwargs: dict[str, object] = {}
    if "charge_coupling" in params:
        regime_kwargs["charge_coupling"] = str(params["charge_coupling"])
    if "correction_exponent" in params:
        regime_kwargs["correction_exponent"] = float(params["correction_exponent"])  # type: ignore[arg-type]

    probe = AlienConfig(
        G=G,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kappa=kappa,
        masses=masses,
        charges=charges,
        ics_pos=((0.0, 0.0), (r0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 0.0)),
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        **regime_kwargs,  # type: ignore[arg-type]
    )
    m = np.asarray(masses, dtype=np.float64)
    mu = float(m[0] * m[1] / (m[0] + m[1]))
    _U, dUdr = _U_and_dUdr(probe)
    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    rel_speed = frac * v_circ
    return dataclasses.replace(probe, ics_vel=((0.0, 0.0), (0.0, rel_speed)))


def tuned_config() -> AlienConfig:
    """The ``AlienConfig`` for ``TIER2_TUNED_PARAMS`` (the recommended operating point)."""
    return config_from_params(TIER2_TUNED_PARAMS)


# -----------------------------------------------------------------------------
# Newton-fit residual + nested-model F-test (the make-or-break statistic)
# -----------------------------------------------------------------------------
def _observed_radial_acceleration(
    cfg: AlienConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``(r, a_radial)`` a position-only discoverer would compute.

    Procedure (RESEARCH §Residual-Structure "What is fit"):
      rel = positions[:,1,:] − positions[:,0,:];
      a_obs[k] = (rel[k+1] − 2·rel[k] + rel[k−1]) / dt² (central FD, trim 3 ends);
      r = |rel|, r̂ = rel/r, a_radial = a_obs · r̂.
    Per-step acceleration (the PySINDy posture), NOT a trajectory least-squares —
    isolates the FORCE-law discrepancy, closed-form, no IC/integrator entanglement.
    """
    traj = AlienUniverse(cfg).run()
    rel = traj.positions[:, 1, :] - traj.positions[:, 0, :]
    dt = cfg.dt
    a_obs = (rel[2:] - 2.0 * rel[1:-1] + rel[:-2]) / dt**2  # central FD
    rel_mid = rel[1:-1]
    r = np.linalg.norm(rel_mid, axis=1)
    rhat = rel_mid / r[:, None]
    a_radial = np.sum(a_obs * rhat, axis=1)
    # Trim 3 endpoints total for FD edge effects (RESEARCH §Residual-Structure
    # step 2). The central FD already drops the two outermost rows of `rel`
    # (indices 0 and n-1); drop ONE additional leading sample so the kept length
    # is exactly (n_steps+1) − 3, the contract length downstream tests assert.
    keep = slice(1, None)
    return r[keep], a_radial[keep]


def newton_fit_residual(
    cfg: AlienConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Newton-only ``a_radial ≈ −k/r²`` and return ``(r, a_radial, e_newton)``.

    One free parameter ``k`` (= effective G·M) by ``np.linalg.lstsq`` against the
    single basis ``−1/r²``; residual ``e_newton = a_radial − (−k/r²)``. This is the
    Model-0 fit of the nested-model F-test; a structured ``e_newton`` is the signal
    that Newton alone does not explain the force.
    """
    r, a_radial = _observed_radial_acceleration(cfg)
    basis = (-1.0 / r**2)[:, None]
    k, _res, _rank, _sv = np.linalg.lstsq(basis, a_radial, rcond=None)
    fit = basis @ k
    e_newton = a_radial - fit
    return r, a_radial, e_newton


def _rss(a_radial: np.ndarray, columns: list[np.ndarray]) -> float:
    """Residual sum of squares for a least-squares fit of ``a_radial`` to ``columns``."""
    X = np.vstack(columns).T
    c, _res, _rank, _sv = np.linalg.lstsq(X, a_radial, rcond=None)
    return float(np.sum((a_radial - X @ c) ** 2))


def nested_model_ftest(cfg: AlienConfig) -> dict:
    """Two concordant nested-model F-tests on the per-step radial acceleration.

    Three least-squares fits to ``a_radial`` (RESEARCH §the structure test):
      Model 0: basis [−1/r²]                                   (RSS0)
      Model 1: [−1/r², −1/r^ce]                                (RSS1, +α)
      Model 2: [−1/r², −1/r^ce, (1−cos(γr))/r²]                (RSS2, +hidden)
    where ce = cfg.correction_exponent (V1 3.5, V2 2.5 — the config's own regime).
    F_alpha = ((RSS0−RSS1)/1)/(RSS1/(n−2)); p_alpha = f.sf(F_alpha, 1, n−2).
    F_hidden = ((RSS1−RSS2)/1)/(RSS2/(n−3)); p_hidden = f.sf(F_hidden, 1, n−3).

    Returns ``{F_alpha, p_alpha, F_hidden, p_hidden, n}``. Both p-values below 0.01
    reject the white-noise null (Newton alone provably fails). ⚠️ NOT autocorrelation
    / runs / Ljung-Box — those are vacuous on smooth deterministic residuals
    (RESEARCH Pitfall 2 / D-02). Guards against a degenerate fit (n too small or a
    zero-variance RSS) with a loud structured result rather than a bare divide.
    """
    r, a_radial = _observed_radial_acceleration(cfg)
    n = int(r.shape[0])
    if n < 4:
        raise ValueError(
            f"nested_model_ftest needs n>=4 trimmed samples for the d.o.f.; got n={n} "
            "(increase n_steps or check the trajectory)."
        )
    gamma = cfg.gamma
    # L-064/DET-07: the correction basis follows the CONFIG'S OWN regime (V1
    # default 3.5 -> unchanged; a V2 cfg is no longer silently fit with V1's basis).
    ce = float(getattr(cfg, "correction_exponent", 3.5))
    rss0 = _rss(a_radial, [-1.0 / r**2])
    rss1 = _rss(a_radial, [-1.0 / r**2, -1.0 / r**ce])
    rss2 = _rss(
        a_radial,
        [-1.0 / r**2, -1.0 / r**ce, (1.0 - np.cos(gamma * r)) / r**2],
    )

    # F = (extra variance explained per added param) / (residual variance per dof).
    # A zero residual RSS (perfect fit) makes F → +inf and p → 0 (a maximally
    # confident rejection); np handles inf cleanly via f.sf, but guard the 0/0 case.
    def _ftest(rss_small: float, rss_big: float, df_resid: int) -> tuple[float, float]:
        delta = rss_small - rss_big
        if rss_big <= 0.0:
            # Bigger model fits to machine zero → the extra term explains
            # essentially all remaining variance → maximal rejection.
            F = float("inf") if delta > 0.0 else 0.0
        else:
            F = float((delta / 1.0) / (rss_big / df_resid))
        F = max(F, 0.0)  # numerical floor: negative F means no improvement.
        p = float(stats.f.sf(F, 1, df_resid))
        return F, p

    F_alpha, p_alpha = _ftest(rss0, rss1, n - 2)
    F_hidden, p_hidden = _ftest(rss1, rss2, n - 3)

    # Newton-only relative residual magnitude — the PHYSICAL effect-size gate that
    # makes the negative control falsifiable. ⚠️ On smooth DETERMINISTIC residuals
    # a bare p-value rejects white-noise even for pure FD-truncation/round-off
    # structure (RESEARCH Pitfall 2): in the true Newtonian limit (α=0,β=0,
    # charges=0) the F-stat is still ~1e4 with p≈0, but the Newton fit's relative
    # residual sits at the ~1e-5 FD-truncation FLOOR. At the tuned operating point
    # the relative residual is ORDERS larger (real, physically-explained structure).
    # `resid_newton_rel` is what distinguishes a structured universe from the
    # Newtonian limit; the in-band conjunction (sweep) requires it ABOVE a floor.
    basis0 = (-1.0 / r**2)[:, None]
    k0, _r0, _rk0, _sv0 = np.linalg.lstsq(basis0, a_radial, rcond=None)
    e_newton0 = a_radial - basis0 @ k0
    rms_signal = float(np.sqrt(np.mean(a_radial**2)))
    rms_resid = float(np.sqrt(np.mean(e_newton0**2)))
    resid_newton_rel = rms_resid / rms_signal if rms_signal > 0.0 else 0.0
    return {
        "F_alpha": F_alpha,
        "p_alpha": p_alpha,
        "F_hidden": F_hidden,
        "p_hidden": p_hidden,
        "resid_newton_rel": resid_newton_rel,
        "n": n,
    }


# -----------------------------------------------------------------------------
# Held-out RMSE (the §10 numerical-match solvability half)
# -----------------------------------------------------------------------------
def held_out_rmse(cfg: AlienConfig, held_out_ics: tuple[AlienConfig, ...]) -> dict:
    """True-law held-out RMSE (the §10 numerical-match method; D-03).

    For each held-out IC: run the true law forward in a FRESH ``AlienUniverse``,
    compare ``positions`` to a reference forward run of the SAME held-out IC
    (the proposed law IS the true law here, so this is the true law vs itself —
    reproducibility to integrator precision). RMSE = sqrt(mean((pred−ref)²)).
    Blowup-guard (non-finite → loud reason, mirror scoring.py:342-345), never a
    bare crash. Returns ``{per_ic_rmse, max_rmse, reason}``.

    ``cfg`` is accepted for interface symmetry with the audit shape (the tuned
    operating point whose law is under test); the forward runs use each held-out
    IC's own config (which already carries the tuned couplings).
    """
    per_ic: list[float] = []
    for ic in held_out_ics:
        ref = AlienUniverse(ic).run()
        pred = AlienUniverse(ic).run()  # fresh run of the same (true) law
        if not (np.all(np.isfinite(ref.positions)) and np.all(np.isfinite(pred.positions))):
            return {
                "per_ic_rmse": per_ic,
                "max_rmse": float("inf"),
                "reason": "numerical_blowup",
            }
        rmse = float(np.sqrt(np.mean((pred.positions - ref.positions) ** 2)))
        per_ic.append(rmse)
    max_rmse = max(per_ic) if per_ic else 0.0
    return {"per_ic_rmse": per_ic, "max_rmse": max_rmse, "reason": "ok"}


# -----------------------------------------------------------------------------
# Polynomial-force baseline (the D-11 beyond-the-bar guard, RIGHT framing)
# -----------------------------------------------------------------------------
def polynomial_baseline_heldout_rmse(
    fit_cfg: AlienConfig,
    held_out_cfg: AlienConfig,
    *,
    degree_terms: tuple[int, ...] = (2, 3, 4, 5, 6),
) -> float:
    """Fit a polynomial-FORCE central law on the seen orbit; forward-integrate held-out.

    RIGHT framing (RESEARCH §Polynomial-Fittability §2 / D-11): fit
    ``a_radial ~ Σ c_k·(1/r)^k`` (k in ``degree_terms``) on the SEEN (``fit_cfg``)
    orbit, then FORWARD-INTEGRATE that radial central force (NO κ DOF) from the
    HELD-OUT IC and compare ``positions`` to the true ``AlienUniverse`` trajectory.
    Returns the held-out RMSE (observed ≈4.8 — the poly baseline FAILS).

    ⚠️ NOT a per-orbit polynomial-RESIDUAL fit: over a near-circular orbit's narrow
    r-range a degree-8 poly crushes the residual and FALSELY "proves" polynomial-
    fittability (the L-015 trap, RESEARCH Pitfall 3). The honest guard is held-out
    forward integration only.

    Blowup is a LARGE finite RMSE (loud) — the integration is capped by an event so
    a runaway state returns a big number rather than NaN/exception, which still
    correctly says "the poly baseline fails the <1e-3 bar".
    """
    # 1. Fit the polynomial force a_radial ~ Σ c_k (1/r)^k on the SEEN orbit.
    r_seen, a_radial_seen = _observed_radial_acceleration(fit_cfg)
    columns = [(1.0 / r_seen) ** k for k in degree_terms]
    X = np.vstack(columns).T
    coeffs, _res, _rank, _sv = np.linalg.lstsq(X, a_radial_seen, rcond=None)

    def poly_radial_accel(r: float) -> float:
        return float(sum(c * (1.0 / r) ** k for c, k in zip(coeffs, degree_terms, strict=True)))

    # 2. Forward-integrate that central force from the HELD-OUT IC in the relative
    #    coordinate. The fitted poly law is already the RELATIVE radial ACCELERATION
    #    (recovered as the second derivative of `rel`), so the rhs uses it directly —
    #    no reduced-mass conversion is needed. Crucially the poly force has NO κ
    #    tangential-inertia DOF, which is why it cannot reproduce the held-out
    #    angular dynamics (the §10 discovery-layer-3 signal).
    pos = np.asarray(held_out_cfg.ics_pos, dtype=np.float64)
    vel = np.asarray(held_out_cfg.ics_vel, dtype=np.float64)
    rel0 = pos[1] - pos[0]
    relv0 = vel[1] - vel[0]
    y0 = np.array([rel0[0], rel0[1], relv0[0], relv0[1]], dtype=np.float64)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        x, yy, vx, vy = y
        r = float(np.hypot(x, yy))
        a_mag = poly_radial_accel(r)  # signed radial accel (per the basis sign)
        ax = a_mag * x / r
        ay = a_mag * yy / r
        return np.array([vx, vy, ax, ay], dtype=np.float64)

    def _blowup(_t: float, y: np.ndarray) -> float:
        return _POLY_BLOWUP_SPAN - float(np.max(np.abs(y)))

    _blowup.terminal = True  # type: ignore[attr-defined]
    _blowup.direction = -1.0  # type: ignore[attr-defined]

    t = np.arange(held_out_cfg.n_steps + 1, dtype=np.float64) * held_out_cfg.dt

    # 3. Reference: the TRUE law's relative coordinate over the same held-out IC.
    truth = AlienUniverse(held_out_cfg).run()
    rel_true = truth.positions[:, 1, :] - truth.positions[:, 0, :]

    sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, events=_blowup, **_POLY_INTEGRATOR_KWARGS)
    if not sol.success or sol.y.shape[1] != len(t) or not np.all(np.isfinite(sol.y)):
        # Blowup / divergence: the poly baseline failed catastrophically. Return a
        # large finite RMSE (loud, > the 1e-2 bar) rather than a NaN or exception.
        return float(_POLY_BLOWUP_SPAN)
    rel_poly = sol.y[:2, :].T  # (n+1, 2) relative position from the poly force
    return float(np.sqrt(np.mean((rel_poly - rel_true) ** 2)))


# -----------------------------------------------------------------------------
# Q-drift (the D-05 admissibility read) + sweep (the EXP-071 measurement object)
# -----------------------------------------------------------------------------
def q_drift(cfg: AlienConfig) -> float:
    """Run the universe and return ``max|Q − Q[0]|`` from ``traj.Q`` (D-05).

    Reads ``traj.Q`` — the polar-Hamiltonian invariant the integrator preserves —
    NOT ``physics.conserved_Q`` of the reconstructed Cartesian state (same physical
    quantity, different coordinate, won't numerically match; PATTERNS gotcha 8).
    """
    traj = AlienUniverse(cfg).run()
    Q = np.asarray(traj.Q, dtype=np.float64)
    return float(np.max(np.abs(Q - Q[0])))


def sweep(param_grid: list[dict], held_out_ics: tuple[AlienConfig, ...]) -> list[dict]:
    """Deterministic grid sweep → EXP-071 measurement rows (Plan 02 runs the verdict).

    Each grid entry is a PARTIAL override dict (e.g. ``{"tangential_frac": 0.85}``
    or ``{"alpha": 0.0, "beta": 0.0}``) merged over ``TIER2_TUNED_PARAMS``. For each
    cell, build the config (IC speed re-derived from potential_U), run the analysis,
    and return a row:
        {params, p_alpha, p_hidden, held_out_rmse, q_drift, in_band}
    where ``in_band = (p_alpha<0.01 and p_hidden<0.01 and held_out_rmse<1e-3 and
    q_drift<1e-6)`` — the conjunction of the EXP-071 target band. Re-validates every
    cell against the locked 15.0 conservation bar via ``q_drift`` (D-05).

    Pure-function + deterministic: no RNG, no I/O, no LLM. The negative control
    (α=0,β=0) returns ``in_band=False`` because Newton then fits the radial law
    (P-4 in the EXP-071 pre-registration).
    """
    rows: list[dict] = []
    for overrides in param_grid:
        params = dict(TIER2_TUNED_PARAMS)
        params.update(overrides)
        cfg = config_from_params(params)

        ftest = nested_model_ftest(cfg)
        rmse_out = held_out_rmse(cfg, held_out_ics)
        qd = q_drift(cfg)

        p_alpha = ftest["p_alpha"]
        p_hidden = ftest["p_hidden"]
        resid_rel = ftest["resid_newton_rel"]
        ho_rmse = rmse_out["max_rmse"]
        # in_band conjuncts the EXP-071 target band: Newton provably fails
        # (p<0.01 on both F-tests AND a residual above the effect-size floor —
        # the latter is what keeps the negative control falsifiable, P-4), the
        # true law is solvable (held-out RMSE<1e-3), and the 15.0 conservation
        # bar holds (Q-drift<1e-6).
        in_band = bool(
            p_alpha < _P_BAR
            and p_hidden < _P_BAR
            and resid_rel > _RESID_FLOOR
            and ho_rmse < _RMSE_BAR
            and qd < _Q_DRIFT_BAR
        )
        rows.append(
            {
                "params": dict(overrides),
                "p_alpha": p_alpha,
                "p_hidden": p_hidden,
                "resid_newton_rel": resid_rel,
                "held_out_rmse": ho_rmse,
                "q_drift": qd,
                "in_band": in_band,
            }
        )
    return rows
