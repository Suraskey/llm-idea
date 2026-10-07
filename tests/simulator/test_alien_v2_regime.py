"""EXP-080 V2 regime — the SECOND hidden-physics law (sum coupling + r^-2.5).

The V2 regime is a SECOND integrable §10 alien law, added to test whether the
identifiability instruments GENERALIZE or MEMORIZE (STRAT-03). It differs from V1
in exactly two places, both gated behind new ``AlienConfig`` defaults so V1 stays
byte-for-byte:

  - the hidden-charge FORCE combines the pair's charges as a SUM ``s1 + s2``
    (``charge_coupling="sum"``), not the V1 PRODUCT ``s1*s2``;
  - the short-range correction is ``r^-2.5`` (``correction_exponent=2.5``), not
    the V1 ``r^-3.5``.

This module proves the two NON-NEGOTIABLE safety gates for the V2 simulator edit:

  1. V1 BYTE-IDENTICAL — with the defaults (product / 3.5), ``total_accel`` and
     ``potential_U`` are bit-for-bit identical to the values BEFORE the V2 fields
     existed (frozen reference values captured from the V1 build). A regression
     that perturbs V1 fails here.
  2. Q-CONSISTENCY (the §10 invariant) — for the V2 law the force ``== −∇U`` to
     ~1e-9 (the sign-bug catcher), AND the polar-Hamiltonian Q drifts ~1e-10 over
     a 1000-step eccentric integration (mirrors the V1 make-or-break bar,
     test_integrator_conservation.py::test_q_conservation). A non-integrable V2
     would fail here — and a non-integrable V2 is WRONG, not a result.

Plus the behavioral V2 deltas: the force scales with the charge SUM (doubling one
charge does NOT double the hidden force, unlike the product law), and the second
radial power-law term is ~r^-2.5.

$0 LLM — deterministic numpy/scipy/sympy only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.simulator import physics as P
from ascension.simulator.alien import AlienUniverse
from ascension.simulator.integrator import integrate
from ascension.simulator.types import AlienConfig

_DT = 0.005


def _v1_cfg(**overrides) -> AlienConfig:
    """A V1 (default-regime) 2-body config — product coupling, r^-3.5 correction."""
    base = dict(
        G=1.0,
        alpha=0.05,
        beta=0.02,
        gamma=0.7,
        kappa=0.3,
        masses=(1.0, 1.0),
        charges=(1.0, 1.0),
        ics_pos=((0.0, 0.0), (1.0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 0.9)),
        dt=_DT,
        n_steps=100,
        seed=0,
    )
    base.update(overrides)
    return AlienConfig(**base)


# =============================================================================
# Gate 1 — V1 BYTE-IDENTICAL. The defaults reproduce the V1 force/potential
# bit-for-bit. The reference values below were captured from the V1 build (the
# product / r^-3.5 law) BEFORE the V2 fields were added; if a V2 edit perturbs
# the default path, these frozen values diverge and the test fails.
# =============================================================================


def test_v1_defaults_are_product_and_3p5() -> None:
    """The new AlienConfig fields default to the V1 law (no opt-in needed)."""
    cfg = _v1_cfg()
    assert cfg.charge_coupling == "product"
    assert cfg.correction_exponent == 3.5


def test_v1_total_accel_byte_identical() -> None:
    """Default-regime total_accel matches the frozen V1 reference EXACTLY.

    Bit-for-bit (``array_equal``, not allclose): the V2 branch must not change a
    single floating-point bit of the V1 force path. The reference was produced by
    the V1 code (product coupling, hardcoded r**3.5 / r**(-2.5)).
    """
    cfg = _v1_cfg(charges=(1.0, 1.3))
    pos = np.array([[0.0, 0.0], [1.4, 0.0]], dtype=np.float64)
    vel = np.zeros_like(pos)
    acc = P.total_accel(pos, vel, cfg)
    # Frozen V1 reference (the exact float the product / r**3.5 build produces at
    # this state — regression anchor).
    # Body 0 (at origin) is pulled toward the +x body → +x; body 1 (at +x) is
    # pulled toward the origin → −x (attractive, Newton's third law).
    expected = np.array(
        [
            [0.5197278911948675, 0.0],
            [-0.5197278911948675, 0.0],
        ],
        dtype=np.float64,
    )
    # The V1 force is computed identically here; assert the new default path
    # reproduces the independent hand-recomputation bit-for-bit.
    G, alpha, beta, gamma = cfg.G, cfg.alpha, cfg.beta, cfg.gamma
    r = 1.4
    s1, s2 = 1.0, 1.3
    # Radial magnitude on body 1 (at +x) along its outward r̂ (+x): negative =
    # attractive (points back toward the origin, −x). Body 0 feels +mag (Newton's
    # third law). So acc[1]=mag, acc[0]=−mag.
    mag = (
        -G * 1.0 / r**2  # newton (m_j = 1)
        - G * alpha * 1.0 / r**3.5  # correction, exponent 3.5
        + beta * (s1 * s2) * (1.0 - np.cos(gamma * r)) / r**2  # hidden, PRODUCT
    )
    hand = np.array([[-mag, 0.0], [mag, 0.0]], dtype=np.float64)
    np.testing.assert_array_equal(acc, hand)
    # And the deterministic frozen literal (regression anchor) — bit-for-bit.
    np.testing.assert_array_equal(acc, expected)


def test_v1_potential_byte_identical() -> None:
    """Default-regime potential_U matches the frozen V1 closed form EXACTLY.

    Asserts the new default path reproduces the V1 ``r**(-2.5)`` correction
    potential + ``s1*s2`` hidden potential bit-for-bit against an independent
    hand recomputation using the OLD literal forms.
    """
    from scipy.special import sici

    cfg = _v1_cfg(charges=(1.0, 1.3))
    r = 1.4
    pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
    masses = np.asarray(cfg.masses, dtype=np.float64)
    charges = np.asarray(cfg.charges, dtype=np.float64)
    u = P.potential_U(pos, masses, charges, cfg)

    G, alpha, beta, gamma = cfg.G, cfg.alpha, cfg.beta, cfg.gamma
    mm = 1.0
    u_newton = -G * mm / r
    u_corr = -(G * mm * alpha / 2.5) * r ** (-2.5)  # OLD V1 literal (p-1 = 2.5)
    si, _ci = sici(gamma * r)
    u_s = beta * (1.0 * 1.3) * ((1.0 - np.cos(gamma * r)) / r - gamma * si)  # PRODUCT
    expected = u_newton + u_corr + u_s
    assert u == expected  # bit-for-bit


def test_v1_explicit_kwargs_match_defaults() -> None:
    """Passing charge_coupling='product' / correction_exponent=3.5 explicitly to the
    primitives reproduces the default-arg result (no hidden default drift)."""
    pos = np.array([[0.0, 0.0], [1.25, 0.0]], dtype=np.float64)
    charges = np.asarray((1.0, 2.0), dtype=np.float64)
    masses = np.asarray((1.0, 1.0), dtype=np.float64)
    a_default = P.hidden_charge_accel(pos, charges, 0.02, 0.7)
    a_explicit = P.hidden_charge_accel(pos, charges, 0.02, 0.7, charge_coupling="product")
    np.testing.assert_array_equal(a_default, a_explicit)
    c_default = P.correction_accel(pos, masses, 1.0, 0.05)
    c_explicit = P.correction_accel(pos, masses, 1.0, 0.05, correction_exponent=3.5)
    np.testing.assert_array_equal(c_default, c_explicit)


# =============================================================================
# Gate 2 — V2 Q-CONSISTENCY (the §10 integrable invariant). force == −∇U and
# the polar Q drifts ~1e-10 on an eccentric V2 orbit.
# =============================================================================


def _v2_cfg(**overrides) -> AlienConfig:
    """A V2-regime 2-body config — SUM coupling, r^-2.5 correction.

    Distinct charges (s1 != s2) so the sum s1+s2 is non-trivial and the κ
    tangential-inertia term (s1*s2) is exercised; couplings match the V1 tuned
    scale so the orbit is well-behaved at dt=0.005.
    """
    base = dict(
        G=1.0,
        alpha=0.05,
        beta=0.02,
        gamma=0.7,
        kappa=0.3,
        masses=(1.0, 1.0),
        charges=(1.0, 2.0),
        ics_pos=((0.0, 0.0), (1.0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 0.9)),
        dt=_DT,
        n_steps=1000,
        seed=0,
        charge_coupling="sum",
        correction_exponent=2.5,
    )
    base.update(overrides)
    return AlienConfig(**base)


def test_v2_force_is_grad_potential() -> None:
    """V2: total_accel == −∇U to ~1e-9 (the sign-bug catcher for the V2 law).

    Mirrors test_physics_invariants::test_sympy_force_is_grad_potential but on the
    V2 config (sum coupling + r^-2.5). A wrong sign on either the r^-2.5
    correction potential or the sum hidden-charge potential breaks this by percent.
    """
    cfg = _v2_cfg()
    m_light = cfg.masses[1]
    center = np.array([0.0, 0.0], dtype=np.float64)
    masses = np.asarray(cfg.masses, dtype=np.float64)
    charges = np.asarray(cfg.charges, dtype=np.float64)

    for r in (0.6, 0.85, 1.0, 1.4, 2.0):
        pos = np.array([center, [r, 0.0]], dtype=np.float64)
        vel = np.zeros_like(pos)
        acc = P.total_accel(pos, vel, cfg)
        f_radial_analytic = m_light * acc[1, 0]

        def _u(rr):
            p = np.array([center, [rr, 0.0]], dtype=np.float64)
            return P.potential_U(p, masses, charges, cfg)

        h = 1e-5
        du_dr = (-_u(r + 2 * h) + 8 * _u(r + h) - 8 * _u(r - h) + _u(r - 2 * h)) / (12 * h)
        f_radial_numeric = -du_dr
        rel_err = abs(f_radial_analytic - f_radial_numeric) / abs(f_radial_analytic)
        assert rel_err < 1e-9, (
            f"V2 force != -grad U at r={r}: analytic={f_radial_analytic} "
            f"numeric={f_radial_numeric} rel_err={rel_err}"
        )


def _build_v2_polar_system(cfg: AlienConfig, *, tangential_frac: float, r0: float):
    """Canonical-polar (rhs, Q, y0) for a 2-body V2 AlienConfig.

    A verbatim mirror of tests/simulator/conftest._build_polar_system, kept local
    so the V2 Q-drift bar runs through the SAME single-source-of-truth wiring the
    V1 make-or-break bar uses: the radial force is −dU/dr of physics.potential_U
    (which now branches on the V2 fields) and m_t = mu + κ·s1·s2 (κ inertia
    coupling, regime-independent). Returns (rhs, Q, y0, mu, m_t).
    """
    m = np.asarray(cfg.masses, dtype=np.float64)
    ch = np.asarray(cfg.charges, dtype=np.float64)
    mu = float(m[0] * m[1] / (m[0] + m[1]))
    s_prod = float(ch[0] * ch[1])
    m_t = mu + cfg.kappa * s_prod

    def U(r: float) -> float:
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, cfg)

    def dUdr(r: float) -> float:
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    def rhs(y: np.ndarray) -> np.ndarray:
        r, _th, pr, pth = y
        return np.array(
            [pr / mu, pth / (m_t * r**2), pth**2 / (m_t * r**3) - dUdr(r), 0.0],
            dtype=np.float64,
        )

    def Q(y: np.ndarray) -> float:
        r, _th, pr, pth = y
        return float(pr**2 / (2 * mu) + pth**2 / (2 * m_t * r**2) + U(r))

    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    theta_dot0 = tangential_frac * v_circ / r0
    p_theta0 = m_t * r0**2 * theta_dot0
    y0 = np.array([r0, 0.0, 0.0, p_theta0], dtype=np.float64)
    return rhs, Q, y0, mu, m_t


def _v2_drift(*, tangential_frac: float, dt: float, n_steps: int) -> tuple[float, float]:
    """Run a V2 eccentric orbit and return (Q-drift, r-range). Helper for the bar
    + the dt-convergence integrability proof."""
    cfg = _v2_cfg(dt=dt, n_steps=n_steps)
    rhs, Q, y0, _mu, _m_t = _build_v2_polar_system(cfg, tangential_frac=tangential_frac, r0=1.0)
    traj = integrate(y0, dt, n_steps, rhs, config=cfg, seed=cfg.seed)
    q_series = np.array([Q(traj[i]) for i in range(traj.shape[0])])
    drift = float(np.max(np.abs(q_series - Q(y0))))
    r_series = traj[:, 0]
    return drift, float(r_series.max() - r_series.min())


def test_v2_q_conservation_drift_bar() -> None:
    """THE V2 make-or-break bar: max|Q − Q₀| ~1e-8 over 1000 steps (V2 law).

    Runs a V2-regime eccentric orbit (sum coupling + r^-2.5) through the REAL
    integrator with the polar rhs driven by physics.potential_U's OWN gradient
    (single source of truth) and reads the polar Q at every step. At frac=0.7 the
    V2 orbit is genuinely eccentric (r-range ~0.18, real radial motion) yet the
    Q-drift sits at ~1e-8 — the same ~1e-10..1e-8 integrable scale as the V1
    make-or-break bar (test_integrator_conservation::test_q_conservation), well
    below the 1e-6 bar. A non-integrable V2 (wrong potential / bad derivation)
    would drift past the bar — a STOP-and-fix defect, not a result.
    """
    drift, r_range = _v2_drift(tangential_frac=0.7, dt=_DT, n_steps=1000)
    assert r_range > 0.05, f"V2 fixture vacuously near-circular (r-range {r_range:.3f})"
    assert drift < 1e-6, f"V2 Q drift {drift:.3e} exceeds the 1e-6 make-or-break bar"
    # Integrable single-source-of-truth wiring conserves to the ~1e-8 scale here
    # (caught a silently-degraded-but-<1e-6 regression).
    assert drift < 1e-7, f"V2 Q drift {drift:.3e} is above the integrable ~1e-8 scale"


def test_v2_q_drift_is_truncation_not_secular_leak() -> None:
    """The V2 Q-drift CONVERGES toward zero as dt→0 — the integrability proof.

    The sharpest test that the V2 derivation is genuinely Q-consistent (not just
    'small drift on this orbit'): a NON-integrable system has a SECULAR energy
    leak whose drift is roughly dt-INDEPENDENT (the Q it tracks is not a true
    invariant). An integrable system's drift is pure integrator TRUNCATION error,
    which shrinks with dt (implicit-midpoint is 2nd order, so ~dt²). We run the
    SAME aggressively-eccentric V2 orbit at dt and dt/4 and require the drift to
    drop by a large factor — proving the §10 invariant holds for the V2 law and
    the 1e-7-scale number above is numerical, not physical.
    """
    # Aggressive eccentricity (frac=0.8, r-range ~1.6) makes the truncation error
    # large enough to see the convergence cleanly.
    drift_coarse, _ = _v2_drift(tangential_frac=0.8, dt=_DT, n_steps=1000)
    drift_fine, _ = _v2_drift(tangential_frac=0.8, dt=_DT / 4.0, n_steps=4000)
    # 2nd-order: dt/4 should cut the drift by ~16x. Require >= 4x (a generous floor)
    # so a secular (dt-independent) leak — which would NOT shrink — fails loudly.
    assert drift_fine < drift_coarse / 4.0, (
        f"V2 Q-drift did not shrink with dt (coarse={drift_coarse:.3e} "
        f"fine={drift_fine:.3e}) — looks SECULAR, i.e. a non-integrable V2 derivation"
    )


def test_v2_q_conservation_bit_reproducible() -> None:
    """V2 integration is bit-identical on re-run (determinism, D-06)."""
    cfg = _v2_cfg()
    rhs, _Q, y0, _mu, _m_t = _build_v2_polar_system(cfg, tangential_frac=0.8, r0=1.0)
    a = integrate(y0, _DT, 400, rhs, config=cfg, seed=cfg.seed)
    b = integrate(y0, _DT, 400, rhs, config=cfg, seed=cfg.seed)
    np.testing.assert_array_equal(a, b)


def test_v2_traj_Q_drift_via_run() -> None:
    """The full AlienUniverse.run() on a V2 config holds traj.Q drift ~1e-10.

    The end-to-end path (the one the scorer's _proposed_q_drift reads) must also
    conserve on the V2 law — the assembly reconstructs an eccentric V2 orbit and
    its polar-Hamiltonian Q stays bounded.
    """
    # Derive the tangential IC speed from the V2 config's own gradient (single
    # source of truth) so the orbit is eccentric and the integrator force matches.
    cfg0 = _v2_cfg(ics_vel=((0.0, 0.0), (0.0, 0.0)))
    m = np.asarray(cfg0.masses, dtype=np.float64)
    ch = np.asarray(cfg0.charges, dtype=np.float64)
    mu = float(m[0] * m[1] / (m[0] + m[1]))

    def U(r):
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, cfg0)

    def dUdr(r):
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    v_circ = float(np.sqrt(1.0 * dUdr(1.0) / mu))
    cfg = _v2_cfg(ics_vel=((0.0, 0.0), (0.0, 0.85 * v_circ)))
    traj = AlienUniverse(cfg).run()
    q = np.asarray(traj.Q, dtype=np.float64)
    drift = float(np.max(np.abs(q - q[0])))
    assert drift < 1e-6, f"V2 run() traj.Q drift {drift:.3e} exceeds 1e-6"


# =============================================================================
# V2 behavioral deltas — the sum coupling + the r^-2.5 correction are really there.
# =============================================================================


def test_v2_hidden_force_scales_with_charge_sum_not_product() -> None:
    """V2 hidden force ∝ (s1 + s2): doubling one charge does NOT double it.

    Under the SUM coupling the amplitude is s1+s2, so going (1,1)->(2,1) raises
    the factor 2->3 (×1.5), NOT ×2 as the product law would. And a zero charge
    does NOT kill the force (s2=0 still leaves s1) — the sharpest product-vs-sum
    discriminator.
    """
    pos = np.array([[0.0, 0.0], [1.3, 0.0]], dtype=np.float64)
    beta, gamma = 0.02, 0.7
    a_11 = P.hidden_charge_accel(pos, np.asarray((1.0, 1.0)), beta, gamma, charge_coupling="sum")
    a_21 = P.hidden_charge_accel(pos, np.asarray((2.0, 1.0)), beta, gamma, charge_coupling="sum")
    a_10 = P.hidden_charge_accel(pos, np.asarray((1.0, 0.0)), beta, gamma, charge_coupling="sum")
    # sum: (1+1)=2 -> (2+1)=3, ratio 3/2 = 1.5 (NOT 2.0 as product would give).
    np.testing.assert_allclose(a_21, 1.5 * a_11, rtol=1e-12)
    # zero charge does NOT kill the sum force (s1=1 still contributes): factor 1.
    np.testing.assert_allclose(a_10, 0.5 * a_11, rtol=1e-12)
    assert not np.allclose(a_10, 0.0), "sum coupling: a single zero charge must NOT zero the force"


def test_v2_correction_is_r_minus_2p5() -> None:
    """V2 correction force scales as r^-2.5 (vs V1 r^-3.5).

    Fit log|a_corr| vs log(r) over a radial grid; the slope is the (negative)
    exponent. V2 must land near -2.5, V1 near -3.5.
    """
    masses = np.asarray((1.0, 1.0), dtype=np.float64)
    rs = np.array([0.8, 1.0, 1.3, 1.7, 2.2], dtype=np.float64)

    def corr_mag(r, exp):
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        a = P.correction_accel(pos, masses, 1.0, 0.05, correction_exponent=exp)
        return abs(a[1, 0])

    mags_v2 = np.array([corr_mag(r, 2.5) for r in rs])
    mags_v1 = np.array([corr_mag(r, 3.5) for r in rs])
    slope_v2 = np.polyfit(np.log(rs), np.log(mags_v2), 1)[0]
    slope_v1 = np.polyfit(np.log(rs), np.log(mags_v1), 1)[0]
    assert abs(slope_v2 - (-2.5)) < 1e-9, f"V2 correction slope {slope_v2} != -2.5"
    assert abs(slope_v1 - (-3.5)) < 1e-9, f"V1 correction slope {slope_v1} != -3.5"


def test_invalid_charge_coupling_raises() -> None:
    """An unknown charge_coupling fails LOUD (no silent fallthrough)."""
    import pytest

    pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="product.*sum|charge_coupling"):
        P.hidden_charge_accel(pos, np.asarray((1.0, 1.0)), 0.02, 0.7, charge_coupling="bogus")
