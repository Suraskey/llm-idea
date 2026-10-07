"""Scale-invariance resolution for the Alien Universe (D-06, Phase 15.1).

Plain English: a "scale-invariant" law looks the same if you zoom in or out — you
can rescale all distances by λ and all times by some power λ^p and the dynamics
are unchanged. SCOPE §10's draft promised "scale invariance at specific exponents"
but named none (SEED-016). This file RESOLVES that claim two ways:

  1. ``test_newtonian_limit_scale_invariance`` — in the Newtonian limit (α=0, β=0,
     charges=0) the alien physics collapses to pure Kepler gravity, which IS
     scale-invariant under the classic mechanical-similarity transform
     ``r -> λr, t -> λ^{3/2}t`` (Kepler's third law). We prove it by running the
     production ``AlienUniverse.run()`` twice — a reference orbit and a λ-scaled
     orbit (positions × λ, velocities / √λ, dt × λ^{3/2}, same n_steps) — and
     asserting ``scaled.positions == λ · base.positions`` to integrator tolerance.

  2. ``test_full_physics_breaks_scale_invariance`` — the FULL physics is provably
     NOT globally scale-invariant. A sympy argument shows the three force terms
     demand THREE inconsistent time-scaling exponents under ``r -> λr``: the
     inverse-square term needs ``p = 3/2`` (Kepler), the ``α/r^3.5`` correction
     needs ``p = 9/4``, and the hidden-charge term ``(1 − cos γr)/r²`` is not a
     power law at all (γ is a fixed inverse-length scale). No single ``p`` exists,
     so the symmetry is deliberately broken — and that broken symmetry is a
     FEATURE: the γ length scale and the mismatched correction exponent are part
     of what makes the universe non-trivial and non-polynomial-regressible.

This is the test backing the dated 2026-05-23 §10 resolution note (Task 3).

Binding: 15.1-03 PLAN Task 2 <behavior>; 15.1 RESEARCH §Scale-Invariance.
Analogs: tests/simulator/test_newtonian_limit.py (Newtonian-limit oracle posture),
tests/simulator/test_symmetries.py:84-89 (transform-then-compare). $0 LLM —
deterministic numpy/sympy only.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import sympy as sp

from ascension.simulator.alien import AlienUniverse
from ascension.simulator.fixtures import _two_body_config, newtonian_limit_fixture

# The Kepler mechanical-similarity transform re-derives the polar IC through a
# scaled dt + scaled velocity, so the match holds to integrator tolerance (not
# bitwise). Both runs use the SAME n_steps and the implicit-midpoint rule; the
# only difference is the uniform λ rescaling. Over 1000 steps with the λ^{3/2}
# dt rescaling the round-off accumulates to ~1.8e-9 absolute (observed); 1e-7
# absolute sits ~2 orders above that floor and FOUR orders below the >1e-3
# discrepancy the full-physics break test asserts — so this tolerance cleanly
# separates "scale-invariant Newtonian limit" from "scale-broken full physics"
# (the documented-tolerance discipline, test_newtonian_limit.py:17-19).
_SCALE_RTOL = 1e-7
_SCALE_ATOL = 1e-7


def test_newtonian_limit_scale_invariance() -> None:
    """Kepler ``r -> λr, t -> λ^{3/2}t`` is a clean symmetry ONLY in the Newtonian limit.

    With α=0, β=0, charges=0 the dynamics are pure ``-GM/r²`` gravity, which is
    scale-invariant by mechanical similarity. We rescale the IC (positions × λ,
    relative velocity / √λ) and the step (dt × λ^{3/2}, same n_steps) and assert
    the whole trajectory is exactly λ × the reference orbit.
    """
    base_cfg, base = newtonian_limit_fixture()

    # Oracle validity guard: the limit fixture really has the corrections off.
    assert base_cfg.alpha == 0.0
    assert base_cfg.beta == 0.0
    assert all(s == 0.0 for s in base_cfg.charges)

    lam = 1.7  # an arbitrary non-trivial scale factor.
    # Rescale the IC: r -> λr (positions), v -> v/√λ (so t -> λ^{3/2}t). The
    # reference IC is body0 at origin, body1 at (r0, 0) with tangential speed v.
    scaled_pos = tuple(tuple(lam * np.asarray(p, dtype=np.float64)) for p in base_cfg.ics_pos)
    scaled_vel = tuple(
        tuple(np.asarray(v, dtype=np.float64) / np.sqrt(lam)) for v in base_cfg.ics_vel
    )
    scaled_cfg = dataclasses.replace(
        base_cfg,
        ics_pos=scaled_pos,
        ics_vel=scaled_vel,
        dt=base_cfg.dt * lam**1.5,  # t -> λ^{3/2}t, same n_steps -> scaled times.
    )
    scaled = AlienUniverse(scaled_cfg).run()

    # Mechanical similarity: the scaled orbit is exactly λ × the reference orbit,
    # frame-for-frame (since both share n_steps and the scaled dt samples at the
    # corresponding scaled times).
    assert np.allclose(
        scaled.positions, lam * base.positions, rtol=_SCALE_RTOL, atol=_SCALE_ATOL
    ), (
        "Newtonian-limit Kepler scale-invariance broken: "
        f"max|scaled - λ·base| = {np.max(np.abs(scaled.positions - lam * base.positions)):.3e}"
    )
    # Velocities scale as v/√λ -> the scaled run's velocities are base/√λ.
    assert np.allclose(
        scaled.velocities,
        base.velocities / np.sqrt(lam),
        rtol=_SCALE_RTOL,
        atol=_SCALE_ATOL,
    )


def test_full_physics_breaks_scale_invariance() -> None:
    """The FULL physics admits NO single scale-invariance exponent (sympy proof).

    Under ``r -> λr`` with ``t -> λ^p t``, the radial equation of motion
    ``r'' = f(r)`` is invariant iff ``f`` scales as ``λ^{1-2p}`` (since r'' picks
    up ``λ^{1-2p}``). Solve for the exponent ``p`` term-by-term:

      - Newton ``f ∝ -1/r²``  ->  scales as λ^{-2}  ->  1-2p = -2  ->  p = 3/2.
      - α-correction ``f ∝ -1/r^3.5``  ->  scales as λ^{-3.5}  ->  1-2p = -3.5
        ->  p = 9/4.
      - hidden ``f ∝ (1 − cos γr)/r²`` is NOT a power of r: cos(γ·λr) ≠ any power
        of λ times cos(γr) because γ carries a FIXED inverse-length scale. So no
        exponent p makes this term self-similar.

    Three irreconcilable demands ⇒ no global scale symmetry. We assert all of it
    symbolically so the proof is machine-checked, not asserted by hand.
    """
    lam, r, p, gamma = sp.symbols("lambda r p gamma", positive=True)

    # The required scaling exponent that makes a pure power-law force f ∝ r^q
    # scale-invariant under r->λr, t->λ^p t. r'' scales as λ^{1-2p}; f(λr) scales
    # as λ^q. Invariance: 1 - 2p = q  =>  p = (1 - q)/2.
    def p_for_power(q):
        return sp.Rational(1, 1) * (1 - q) / 2

    # Newton: f ∝ r^{-2}  =>  q = -2.
    p_newton = p_for_power(-2)
    # α-correction: f ∝ r^{-3.5}  =>  q = -7/2.
    p_alpha = p_for_power(sp.Rational(-7, 2))

    assert p_newton == sp.Rational(3, 2), f"Newton exponent wrong: {p_newton}"
    assert p_alpha == sp.Rational(9, 4), f"α-correction exponent wrong: {p_alpha}"
    # The two power-law terms demand DIFFERENT exponents — already no single p.
    assert p_newton != p_alpha, (
        "Newton and α-correction must demand different scale exponents "
        f"(got {p_newton} == {p_alpha}) — scale-invariance would NOT be broken"
    )

    # Hidden-charge term: f_hidden ∝ (1 - cos(γ r)) / r². Under r -> λr it becomes
    # (1 - cos(γ λ r)) / (λ r)². For this to be λ^c · f_hidden(r) for some constant
    # c (independent of r) we would need (1 - cos(γ λ r)) = λ^{c+2} (1 - cos(γ r))
    # for ALL r. The ratio is NOT r-independent because γ is a fixed inverse-length.
    f_hidden = (1 - sp.cos(gamma * r)) / r**2
    f_hidden_scaled = (1 - sp.cos(gamma * lam * r)) / (lam * r) ** 2
    ratio = sp.simplify(f_hidden_scaled / f_hidden)
    # If the ratio depended on λ alone (a pure power of λ), d(ratio)/dr would be 0.
    dratio_dr = sp.simplify(sp.diff(ratio, r))
    assert dratio_dr != 0, (
        "hidden-charge term unexpectedly self-similar — it must NOT be a power "
        "law (γ is a fixed inverse-length scale that breaks scale-invariance)"
    )
    # And concretely: at a generic γ, λ, r the ratio is r-dependent (two distinct
    # r values give two distinct ratios), confirming non-power-law structure.
    subs_a = {gamma: 0.7, lam: 1.7, r: 1.0}
    subs_b = {gamma: 0.7, lam: 1.7, r: 2.0}
    ratio_a = float(ratio.subs(subs_a))
    ratio_b = float(ratio.subs(subs_b))
    assert abs(ratio_a - ratio_b) > 1e-6, (
        "hidden-charge scaling ratio is r-independent — would imply a power law "
        f"(ratio@r=1: {ratio_a:.6f}, ratio@r=2: {ratio_b:.6f})"
    )


def test_full_physics_orbit_is_not_scale_invariant_numerically() -> None:
    """Numerical companion: the FULL-physics orbit is NOT reproduced by Kepler scaling.

    Applying the Newtonian-limit Kepler transform (r->λr, v->v/√λ, dt->λ^{3/2}dt)
    to the FULL alien physics does NOT yield λ × the reference orbit — because the
    α-correction and the γ length scale break the similarity. This is the
    numerical mirror of the sympy proof above (defense-in-depth, never silent).
    """
    # Full alien physics (α, β, κ, unit charges) — a near-circular tuned-like orbit.
    base_cfg = _two_body_config(
        r0=1.0,
        rel_speed=0.85,
        tangential=True,
        alpha=0.05,
        beta=0.02,
        gamma=0.7,
        kappa=0.3,
        charges=(1.0, 1.0),
        masses=(1.0, 1.0),
        dt=0.005,
        n_steps=400,
    )
    base = AlienUniverse(base_cfg).run()

    lam = 1.7
    scaled_cfg = dataclasses.replace(
        base_cfg,
        ics_pos=tuple(tuple(lam * np.asarray(p, dtype=np.float64)) for p in base_cfg.ics_pos),
        ics_vel=tuple(
            tuple(np.asarray(v, dtype=np.float64) / np.sqrt(lam)) for v in base_cfg.ics_vel
        ),
        dt=base_cfg.dt * lam**1.5,
    )
    scaled = AlienUniverse(scaled_cfg).run()

    # If the full physics WERE scale-invariant this would be ~0; it is NOT.
    discrepancy = float(np.max(np.abs(scaled.positions - lam * base.positions)))
    assert discrepancy > 1e-3, (
        "full alien physics unexpectedly obeyed Kepler scale-invariance "
        f"(discrepancy {discrepancy:.3e}) — the α/γ terms should break it"
    )
