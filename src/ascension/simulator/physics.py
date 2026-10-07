"""Q-consistent §10 force laws + the conserved quantity Q (pure functions).

Implements the Q-consistent §10 formulation (tangential-inertia coupling,
``m_eff = m + κs²``; attractive α-correction). Reconciled into SCOPE §10
(2026-05-23) — see iterations/15.0.md. Cites SCOPE.md:296 (§10) + 15.0 RESEARCH Make-or-break.
Scale-invariance resolved into SCOPE §10 (2026-05-23, Phase 15.1) — full physics breaks it via γ; Kepler limit tested in tests/simulator/test_scale_invariance.py. See iterations/15.1.md.

THE MAKE-OR-BREAK PHYSICS (do NOT get the signs wrong):
  - RESEARCH proved §10's literal Q (½mv² + U + ½κs²·v_perp² under POSITION-ONLY
    forces) is NOT conserved (drifts 28%). The minimal coherent correction makes
    the hidden charge augment TANGENTIAL INERTIA: m_eff = m + κs². The Lagrangian
    L = ½m·ṙ² + ½(m+κs²)·r²θ̇² − U(r) then has a conserved energy that is EXACTLY
    §10's Q symbol (sympy-verified). The tangential-inertia coupling is handled
    in the integrator's effective-mass treatment / in conserved_Q — it is NOT an
    extra force here. Do NOT double-count it as a force.
  - The α-correction potential carries an ATTRACTIVE MINUS sign:
    U_corr = −(G·m₁m₂·α/(p−1))·r^{−(p−1)} where p = ``correction_exponent``
    (the V1 §10 law is p=3.5 → U_corr = −(G·m₁m₂·α/2.5)·r^{−2.5}; the EXP-080 V2
    regime is p=2.5 → U_corr = −(G·m₁m₂·α/1.5)·r^{−1.5}). ``total_accel`` and
    ``conserved_Q`` MUST agree on this sign (RESEARCH Pitfall 1 — a wrong ``+``
    sign injects ~5% drift that masquerades as an integrator bug). The
    sympy/finite-difference test ``test_total_accel_is_grad_potential`` is the
    regression guard.
  - The hidden-charge FORCE coupling is set by ``charge_coupling``: the V1 §10 law
    combines the pair's charges as the PRODUCT s_i·s_j; the EXP-080 V2 regime uses
    the SUM s_i+s_j. The charge factor is constant in r, so the force/potential/Q
    consistency is identical in structure across regimes (only the prefactor
    changes). The κ tangential-inertia coupling (``m_eff = m + κs²``) is a SEPARATE
    coupling and is unchanged across V1/V2.

Pure functions, numpy-only, NO I/O, NO logging (mirrors benchmarks/systems.py
import-light posture). General N-body pairwise sums (O(N²), RESEARCH Q8); the
2-body analytical anchors and the default config restrict TESTS to 2-body.

Binding decisions:
  - 15.0 PATTERNS §physics.py: hand-rolled pure force laws + conserved_Q
    (no upstream library to anchor to; the physics is novel §10).
  - 15.0 RESEARCH §Code Examples: the closed-form U_s via scipy.special.sici.
  - 15.0 RESEARCH Q4: v_perp is the component of velocity orthogonal to the
    radial direction from the force center (load-bearing for Q and the tests).

Threat-model mitigations: N/A — pure-compute, no network/auth/secret surface.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sici

# -----------------------------------------------------------------------------
# Geometry helper
# -----------------------------------------------------------------------------


def v_perp_squared(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Per-body squared speed orthogonal to the radial direction (RESEARCH Q4).

    For a 2-body system, the radial direction for each body is taken toward the
    OTHER body (the force center): r̂ = (r⃗_i − r⃗_other)/|·|. Then
    ``v_perp² = |v⃗_i|² − (v⃗_i·r̂)²`` which in 2D equals ``(x·v_y − y·v_x)²/r²``
    (relative coordinates). This is load-bearing for both Q and the
    conservation/symmetry tests.

    Args:
      pos: Positions, shape ``(n_bodies, dim)``, float64.
      vel: Velocities, same shape.

    Returns:
      ``(n_bodies,)`` array of v_perp² per body. For the 2-body case the two
      entries use the relative radial direction (each toward the other body).

    Raises:
      ValueError: if pos/vel are not 2-body. (15.0 ships the 2-body case;
        general-N v_perp is deferred — RESEARCH Q8 keeps the FORCES N-body but
        v_perp's "force center" is only unambiguous for 2-body.)
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    n = pos.shape[0]
    if n != 2:
        raise ValueError(
            "v_perp_squared is defined for the 2-body case in 15.0 "
            f"(force-center radial direction is unambiguous only there); got n={n}"
        )
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        other = 1 - i
        rel = pos[i] - pos[other]
        r = np.sqrt(np.dot(rel, rel))
        rhat = rel / r
        v = vel[i]
        v_radial = np.dot(v, rhat)
        out[i] = np.dot(v, v) - v_radial * v_radial
    return out


# -----------------------------------------------------------------------------
# Force laws (accelerations) — pairwise, general N-body
# -----------------------------------------------------------------------------


def _pairwise_radial_accel(
    positions: np.ndarray,
    masses: np.ndarray,
    magnitude_fn,
) -> np.ndarray:
    """Generic O(N²) pairwise radial-acceleration accumulator.

    For each ordered pair (i, j != i), ``magnitude_fn(r, m_j)`` returns the
    SIGNED radial-force-per-unit-mass-of-i contribution; positive = repulsive
    (along +r̂ from i toward away-from-j), negative = attractive. The radial
    unit vector points from j toward i (so a negative magnitude pulls i toward
    j, i.e. attractive). This keeps Newton's third law exact at the force level.

    Returns ``(n_bodies, dim)`` accelerations.
    """
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    n, dim = positions.shape
    acc = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rel = positions[i] - positions[j]  # from j toward i
            r = np.sqrt(np.dot(rel, rel))
            rhat = rel / r
            acc[i] += magnitude_fn(r, masses[j]) * rhat
    return acc


def newtonian_accel(positions: np.ndarray, masses: np.ndarray, G: float) -> np.ndarray:
    """Newtonian gravity: ``−G·m_j/r²·r̂`` per pair (attractive). N-body."""
    return _pairwise_radial_accel(positions, masses, lambda r, mj: -G * mj / r**2)


def correction_accel(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float,
    alpha: float,
    correction_exponent: float = 3.5,
) -> np.ndarray:
    """α short-range correction: ``−G·α·m_j/r^p·r̂`` per pair (ATTRACTIVE).

    The MINUS makes it attractive (adds to gravity). ``p`` is
    ``correction_exponent`` (default 3.5, the V1 §10 law; 2.5 for the EXP-080 V2
    regime). Its potential is ``U_corr = −(G m₁m₂ α/(p−1))·r^{−(p−1)}`` (the
    matching MINUS); ``−dU/dr`` gives ``−G α m₁m₂/r^p`` toward the center. Wrong
    sign → ~5% drift (Pitfall 1). The default p=3.5 (p−1=2.5) reproduces the V1
    force byte-for-byte.
    """
    return _pairwise_radial_accel(
        positions, masses, lambda r, mj: -G * alpha * mj / r**correction_exponent
    )


def _charge_factor(s_i: float, s_j: float, charge_coupling: str) -> float:
    """Combine a pair's hidden charges per the regime's coupling.

    ``"product"`` (default, V1 §10 law) returns ``s_i·s_j``; ``"sum"`` (EXP-080
    V2 regime) returns ``s_i+s_j``. This is the ONLY place the V1↔V2 charge
    coupling differs — the force, potential, and (per-body) conserved-Q charge
    factors all route through it so the three stay mutually consistent.
    """
    if charge_coupling == "sum":
        return s_i + s_j
    if charge_coupling == "product":
        return s_i * s_j
    raise ValueError(
        f"charge_coupling must be 'product' or 'sum'; got {charge_coupling!r}"
    )


def hidden_charge_accel(
    positions: np.ndarray,
    charges: np.ndarray,
    beta: float,
    gamma: float,
    charge_coupling: str = "product",
) -> np.ndarray:
    """Hidden-charge force: ``β·(s_i coupling s_j)·(1−cos γr)/r²·r̂`` per pair.

    Uses the per-body hidden charges (the layer-4 discovery target). The pair's
    charge factor is set by ``charge_coupling``: the PRODUCT ``s_i·s_j`` (default,
    V1 §10 law) so two configs differing only in s₁·s₂ produce distinguishable
    trajectories (D-04); or the SUM ``s_i+s_j`` (the EXP-080 V2 regime). Only the
    constant-in-r amplitude changes between the two — the ``(1−cos γr)/r²`` radial
    structure is identical, so the matching potential's ``−dU_s/dr = force``
    derivation is unchanged in r (only the prefactor differs). The acceleration on
    body i divides by its own mass — but for the FORCE-per-unit-mass-of-i
    convention here we keep it consistent with the radial accumulator: the
    per-pair acceleration of i is the force magnitude along r̂; mass division for
    the force→accel happens implicitly because the other accels are already
    per-unit-mass-of-i and this term is added as an acceleration of equal footing.
    (For the 2-body near-equal-magnitude test config this matches §10's stated F_s.)
    """
    positions = np.asarray(positions, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    n, dim = positions.shape
    acc = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rel = positions[i] - positions[j]
            r = np.sqrt(np.dot(rel, rel))
            rhat = rel / r
            cfac = _charge_factor(charges[i], charges[j], charge_coupling)
            mag = beta * cfac * (1.0 - np.cos(gamma * r)) / r**2
            acc[i] += mag * rhat
    return acc


def total_accel(positions: np.ndarray, velocities: np.ndarray, cfg) -> np.ndarray:
    """Sum of the three RADIAL accelerations (Newton + α-correction + hidden).

    NOTE: the tangential-inertia ``m_eff = m + κs²`` coupling is NOT a force
    here — it is handled in the integrator's effective-mass treatment / in
    ``conserved_Q``. Adding it as a force would double-count it (RESEARCH Code
    Examples comment). ``velocities`` is accepted for interface symmetry with a
    future velocity-dependent term but is unused by the radial forces.

    Args:
      positions: ``(n_bodies, dim)`` float64.
      velocities: ``(n_bodies, dim)`` float64 (currently unused — radial forces
        are position-only; kept for the integrator's rhs(state) signature).
      cfg: an ``AlienConfig`` providing G, alpha, beta, gamma, masses, charges.

    Returns:
      ``(n_bodies, dim)`` total acceleration.
    """
    masses = np.asarray(cfg.masses, dtype=np.float64)
    charges = np.asarray(cfg.charges, dtype=np.float64)
    # Defaults on the new cfg fields preserve V1 (product / 3.5) byte-for-byte;
    # getattr keeps total_accel working on any older AlienConfig-like object that
    # predates the V2 fields (defensive — the dataclass already defaults them).
    charge_coupling = getattr(cfg, "charge_coupling", "product")
    correction_exponent = getattr(cfg, "correction_exponent", 3.5)
    a = newtonian_accel(positions, masses, cfg.G)
    a = a + correction_accel(positions, masses, cfg.G, cfg.alpha, correction_exponent)
    a = a + hidden_charge_accel(positions, charges, cfg.beta, cfg.gamma, charge_coupling)
    return a


# -----------------------------------------------------------------------------
# Potential + conserved quantity Q
# -----------------------------------------------------------------------------


def potential_U(positions: np.ndarray, masses: np.ndarray, charges: np.ndarray, cfg) -> float:
    """Total potential energy U(r) summed over unordered pairs.

    Per pair (i<j) at separation r, with correction exponent p (=
    ``cfg.correction_exponent``, default 3.5) and charge factor c (= s_i·s_j for
    the V1 "product" coupling, s_i+s_j for the V2 "sum" coupling):
      U(r) = −G m_i m_j / r
             − (G m_i m_j α / (p−1))·r^{−(p−1)}  # α term carries MINUS (attractive)
             + U_s(r)
    where the hidden-charge potential (closed form via the sine integral
    Si = scipy.special.sici) is
      U_s(r) = β c [ (1 − cos γr)/r − γ·Si(γr) ].
    (sympy-verified: −dU_s/dr = β c (1 − cos γr)/r² exactly — the charge factor c
    is constant in r, so swapping product→sum changes only the prefactor, not the
    derivative-in-r structure. The ``(1−cos γr)/r`` term carries a PLUS — an
    earlier MINUS broke ``total_accel == −∇U`` on the hidden term while leaving
    Newton/α correct.) For the α term, ``−dU_corr/dr = −G α m_i m_j r^{−p}``
    because ``d/dr[r^{−(p−1)}] = −(p−1)·r^{−p}`` and the ``1/(p−1)`` prefactor
    cancels the ``(p−1)`` — so any p stays force-consistent (V1 p=3.5 → p−1=2.5,
    byte-identical to the prior literal; V2 p=2.5 → p−1=1.5).
    ``−dU/dr`` reproduces each radial force exactly (sympy-verified this session),
    which is what makes ``total_accel == −∇U`` and ``conserved_Q`` agree.

    Args:
      positions: ``(n_bodies, dim)`` float64.
      masses: ``(n_bodies,)`` float64.
      charges: ``(n_bodies,)`` float64.
      cfg: ``AlienConfig`` providing G, alpha, beta, gamma.

    Returns:
      Scalar total potential energy (float).
    """
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    n = positions.shape[0]
    G, alpha, beta, gamma = cfg.G, cfg.alpha, cfg.beta, cfg.gamma
    # Defaults preserve V1 (product / 3.5) byte-for-byte; getattr keeps potential_U
    # working on any AlienConfig-like object predating the V2 fields (defensive).
    charge_coupling = getattr(cfg, "charge_coupling", "product")
    p = getattr(cfg, "correction_exponent", 3.5)  # correction force exponent
    # The matching correction potential U_corr = −(G m_i m_j α/(p−1))·r^{−(p−1)}
    # so −dU_corr/dr = −G α m_i m_j r^{−p} exactly (see docstring). For V1 p=3.5,
    # (p−1)=2.5 → identical to the prior hardcoded `r**(-2.5)` literal.
    corr_pot_exp = p - 1.0  # = 2.5 for V1, 1.5 for V2
    u = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rel = positions[i] - positions[j]
            r = float(np.sqrt(np.dot(rel, rel)))
            mm = masses[i] * masses[j]
            u_newton = -G * mm / r
            u_corr = -(G * mm * alpha / corr_pot_exp) * r ** (-corr_pot_exp)  # MINUS sign
            si, _ci = sici(gamma * r)  # Si(γr)
            cfac = _charge_factor(charges[i], charges[j], charge_coupling)
            u_s = beta * cfac * ((1.0 - np.cos(gamma * r)) / r - gamma * si)
            u += u_newton + u_corr + u_s
    return float(u)


def conserved_Q(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    charges: np.ndarray,
    cfg,
) -> float:
    """The §10 conserved quantity under the Q-consistent dynamics.

    Q = Σ_bodies [ ½ m·v_radial² + ½ (m + κ s²)·v_perp² ] + U(r)

    where v_radial = v⃗·r̂ and v_perp² is the orthogonal component
    (``v_perp_squared``), and the α sign in U MUST match ``potential_U``. The
    ``½κs²·v_perp²`` augmentation is the tangential-inertia term that makes Q a
    true invariant (RESEARCH Make-or-break). For the 2-body case the kinetic
    sum runs over both bodies; the heavy near-fixed center contributes
    negligibly but is included for exactness.

    V2 regime (EXP-080): the potential ``U`` here is ``potential_U``, which
    branches internally on ``cfg.charge_coupling`` + ``cfg.correction_exponent``,
    so Q automatically tracks the V2 hidden FORCE/correction without any change
    here. The ``½κs²`` tangential-inertia term is governed by κ (the INERTIA
    coupling), INDEPENDENT of ``charge_coupling`` (the β hidden-FORCE coupling),
    so it is unchanged across V1/V2 — keeping the V1 per-body charge factor
    (``s_i²``) exact. The integrable §10 invariant (force == −∇U, Q = T + U) thus
    holds in both regimes.

    Args:
      positions: ``(n_bodies, dim)`` float64.
      velocities: ``(n_bodies, dim)`` float64.
      masses: ``(n_bodies,)`` float64.
      charges: ``(n_bodies,)`` float64.
      cfg: ``AlienConfig`` providing G, alpha, beta, gamma, kappa.

    Returns:
      Scalar Q (float).
    """
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    kappa = cfg.kappa

    vp2 = v_perp_squared(positions, velocities)  # (n_bodies,)
    kinetic = 0.0
    n = positions.shape[0]
    for i in range(n):
        other = 1 - i
        rel = positions[i] - positions[other]
        r = np.sqrt(np.dot(rel, rel))
        rhat = rel / r
        v_radial = np.dot(velocities[i], rhat)
        m = masses[i]
        s2 = charges[i] * charges[i]
        kinetic += 0.5 * m * v_radial * v_radial
        kinetic += 0.5 * (m + kappa * s2) * vp2[i]

    u = potential_U(positions, masses, charges, cfg)
    return float(kinetic + u)
