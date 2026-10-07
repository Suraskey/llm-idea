"""Physics invariant tests for the Q-consistent §10 formulation.

These tests prove the two hardest physics facts of Phase 15.0 in isolation,
BEFORE any system object or agent touches the simulator:

  1. ``total_accel == −∇U`` numerically on a radial grid — this is the test
     that catches the α-correction SIGN BUG (15.0 RESEARCH Pitfall 1: a wrong
     ``+`` sign injects ~5% energy drift that masquerades as an integrator bug).
  2. The Lagrangian ``L = ½m·ṙ² + ½(m+κs²)·r²θ̇² − U(r)`` has energy that
     simplifies EXACTLY to ``Q = ½m·v² + U(r) + ½κs²·v_perp²`` (sympy
     ``simplify(Q − target) == 0``) — proving Q is a true invariant of the
     corrected dynamics (15.0 RESEARCH Make-or-break / Code Examples).

Plus the pure-Newton sub-case (α=0, β=0 → −G·m/r² toward center) and the
``v_perp_squared`` 2D identity.

Binding: 15.0 PLAN Task 2 <behavior>; 15.0 RESEARCH §Code Examples (sympy
1.14.0 invariant check). $0 LLM — deterministic numpy/sympy only.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ascension.simulator.physics import (
    correction_accel,
    hidden_charge_accel,
    newtonian_accel,
    potential_U,
    total_accel,
    v_perp_squared,
)
from ascension.simulator.types import AlienConfig


def _cfg(*, alpha=0.05, beta=0.02, gamma=0.7, kappa=0.3, charges=(1.0, 1.0)):
    """A 2-body AlienConfig for physics-only tests (ICs are placeholders here;
    the physics functions consume positions/velocities passed directly)."""
    return AlienConfig(
        G=1.0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kappa=kappa,
        masses=(1000.0, 1.0),
        charges=charges,
        ics_pos=((0.0, 0.0), (1.0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 1.0)),
        dt=0.005,
        n_steps=10,
        seed=0,
    )


def test_sympy_force_is_grad_potential() -> None:
    """Finite-difference of potential_U matches the radial force to ~1e-6.

    This is the SIGN-BUG catcher (RESEARCH Pitfall 1). For a 2-body system with
    a heavy center at the origin and a light body at radius r along +x, the
    net force on the light body is purely radial. We compute the radial force
    component two ways and require agreement:

      analytic:  F_radial = m_light * (total_accel on the light body) · r̂
      numeric:   F_radial = −dU/dr  via central finite difference of potential_U

    A wrong (``+``) sign on the α term flips the correction force and breaks
    this to several percent.
    """
    cfg = _cfg()
    m_light = cfg.masses[1]
    center = np.array([0.0, 0.0], dtype=np.float64)

    for r in (0.6, 0.85, 1.0, 1.4, 2.0):
        pos = np.array([center, [r, 0.0]], dtype=np.float64)
        # Zero velocity: only the radial (position-dependent) forces act; the
        # tangential-inertia term contributes nothing to the FORCE (it lives in
        # the integrator's effective mass / Q, never as a force here).
        vel = np.zeros_like(pos)
        acc = total_accel(pos, vel, cfg)
        # Radial component of force on the light body (body index 1), along +x.
        f_radial_analytic = m_light * acc[1, 0]

        # Numeric −dU/dr via a 4th-order central difference of the pairwise
        # potential. The relationship total_accel == −∇U is a relative-agreement
        # claim; a 4th-order stencil drives finite-difference TRUNCATION error
        # to ~1e-12, so a 1e-9 relative tolerance is both correct (well above
        # observed ~1e-12 jitter) and tight enough to catch a wrong α sign
        # (~5% break) or a wrong U_s hidden-charge sign by many orders of
        # magnitude. [4th-order stencil chosen over the plan's bare difference
        # so truncation, not physics, never trips this — see SUMMARY deviation.]
        def _u(rr):
            p = np.array([center, [rr, 0.0]], dtype=np.float64)
            return potential_U(p, np.asarray(cfg.masses), np.asarray(cfg.charges), cfg)

        h = 1e-5
        # 4th-order central first derivative.
        du_dr = (-_u(r + 2 * h) + 8 * _u(r + h) - 8 * _u(r - h) + _u(r - 2 * h)) / (12 * h)
        f_radial_numeric = -du_dr

        rel_err = abs(f_radial_analytic - f_radial_numeric) / abs(f_radial_analytic)
        assert rel_err < 1e-9, (
            f"force != -grad U at r={r}: analytic={f_radial_analytic} "
            f"numeric={f_radial_numeric} rel_err={rel_err}; a wrong α or U_s "
            "sign breaks this by several percent"
        )


def test_sympy_lagrangian_energy_equals_Q() -> None:
    """sympy: the corrected Lagrangian's energy simplifies EXACTLY to §10's Q.

    L = ½m·ṙ² + ½(m+κs²)·r²θ̇² − U(r).  Its conserved energy
    E = ṙ·∂L/∂ṙ + θ̇·∂L/∂θ̇ − L equals
    Q = ½m·ṙ² + ½(m+κs²)·r²θ̇² + U(r) = ½m·v² + U(r) + ½κs²·v_perp².
    RESEARCH ran this exact check (sympy 1.14.0) and got True.
    """
    t = sp.symbols("t")
    m, k, s2 = sp.symbols("m kappa s2", positive=True)
    r = sp.Function("r")(t)
    th = sp.Function("theta")(t)
    U = sp.Function("U")

    L = (
        sp.Rational(1, 2) * m * sp.diff(r, t) ** 2
        + sp.Rational(1, 2) * (m + k * s2) * r**2 * sp.diff(th, t) ** 2
        - U(r)
    )
    pr = sp.diff(L, sp.diff(r, t))
    pth = sp.diff(L, sp.diff(th, t))
    Q = sp.simplify(pr * sp.diff(r, t) + pth * sp.diff(th, t) - L)
    target = (
        sp.Rational(1, 2) * m * sp.diff(r, t) ** 2
        + sp.Rational(1, 2) * (m + k * s2) * r**2 * sp.diff(th, t) ** 2
        + U(r)
    )
    assert sp.simplify(Q - target) == 0


def test_pure_newton_subcase() -> None:
    """α=0, β=0, charges→0: total_accel reduces to −G·m/r² toward center.

    Heavy center at origin, light body at (r, 0). The light body must feel
    acceleration −G·m_center/r² along −x (toward the center). Hand value at
    r=2.0 with G=1, m_center=1000: a_x = −1000/4 = −250.0.
    """
    cfg = _cfg(alpha=0.0, beta=0.0, charges=(0.0, 0.0))
    r = 2.0
    pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
    vel = np.zeros_like(pos)
    acc = total_accel(pos, vel, cfg)
    expected_ax = -cfg.G * cfg.masses[0] / r**2  # = -250.0
    assert np.isclose(
        acc[1, 0], expected_ax, rtol=0, atol=1e-12
    ), f"pure-Newton a_x={acc[1, 0]} != {expected_ax}"
    assert np.isclose(acc[1, 1], 0.0, atol=1e-12)


def test_v_perp_squared_2d_identity() -> None:
    """v_perp_squared equals (x·v_y − y·v_x)² / r² on a known 2D state.

    RESEARCH Q4: v_perp² = |v|² − (v·r̂)² = (x·v_y − y·v_x)²/r² in 2D, taking the
    radial direction from the force center (other body). For a 2-body state we
    measure body 1 relative to body 0.
    """
    pos = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)  # r = 5
    vel = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    vp2 = v_perp_squared(pos, vel)
    x, y = 3.0, 4.0
    vx, vy = 1.0, 2.0
    r2 = x * x + y * y
    expected = (x * vy - y * vx) ** 2 / r2  # (3*2 - 4*1)^2 / 25 = 4/25 = 0.16
    assert np.isclose(vp2[1], expected, atol=1e-12), f"{vp2[1]} != {expected}"


def test_hidden_charge_accel_uses_charges() -> None:
    """hidden_charge_accel scales with the charge product s₁·s₂.

    Doubling one body's charge doubles the hidden force (it is ∝ s₁s₂). Zero
    charge on either body kills the hidden force entirely.
    """
    cfg = _cfg(charges=(1.0, 1.0))
    pos = np.array([[0.0, 0.0], [1.3, 0.0]], dtype=np.float64)
    a1 = hidden_charge_accel(pos, np.asarray((1.0, 1.0)), cfg.beta, cfg.gamma)
    a2 = hidden_charge_accel(pos, np.asarray((2.0, 1.0)), cfg.beta, cfg.gamma)
    a0 = hidden_charge_accel(pos, np.asarray((0.0, 1.0)), cfg.beta, cfg.gamma)
    assert np.allclose(a2, 2.0 * a1), "hidden force must scale with s₁·s₂"
    assert np.allclose(a0, 0.0), "zero charge → zero hidden force"


def test_correction_accel_attractive() -> None:
    """The α-correction is ATTRACTIVE (points toward the center), i.e. −x for a
    body on the +x axis. A wrong sign would point it +x (repulsive)."""
    cfg = _cfg()
    pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    ac = correction_accel(pos, np.asarray(cfg.masses), cfg.G, cfg.alpha)
    assert ac[1, 0] < 0.0, "α-correction must be attractive (toward center)"
    # And on the heavy body it points the other way (Newton's third law).
    assert ac[0, 0] > 0.0


def test_newtonian_accel_attractive() -> None:
    """Pure Newtonian accel points toward the center for a body on +x axis."""
    cfg = _cfg()
    pos = np.array([[0.0, 0.0], [1.5, 0.0]], dtype=np.float64)
    an = newtonian_accel(pos, np.asarray(cfg.masses), cfg.G)
    assert an[1, 0] < 0.0
    assert np.isclose(an[1, 1], 0.0, atol=1e-12)
