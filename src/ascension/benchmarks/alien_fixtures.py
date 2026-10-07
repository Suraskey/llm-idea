"""Phase 15.2 Plan 01 — alien_benchmark() feed (the §10 swap-point producer, D-03).

Plain English: this is the alien-native twin of
``benchmarks/fixtures.py:default_benchmark``. Hand it nothing (or a seed) and it
gives back THREE things:

  - ``bundle`` — the DATA-ONLY ``ObservationBundle`` ({t, positions, velocities,
    masses}). This is the ONLY surface the Council is ever allowed to see. It
    carries NO charges, NO conserved quantity Q, NO config, NO true law. This is
    the §10 hidden/observable boundary (T-15.2-leak) — the load-bearing zero-shot
    invariant. Breaching it invalidates the entire discovery claim (TIER2-03/04).
  - ``truth_cfg`` — the alien truth ``AlienConfig`` (the couplings + hidden
    charges that GENERATED the bundle). This goes to the SCORER (Plan 02) on a
    SEPARATE path the Council never touches.
  - ``held_out`` — ``config.TIER2_HELD_OUT_ICS`` (the >= 12 generalization ICs,
    already built in 15.1). Also scorer-side only.

Why this lives BENCHMARKS-side (L-018): ``benchmarks/`` is the higher tier — it
consumes simulators. Landing the feed here lets it import ``ascension.simulator``
freely at module top (the exact posture ``benchmarks/alien_audit.py:72-74`` uses)
with ZERO new import edges. A simulator-side feed reaching back into benchmarks
would create the repo's ONLY ``simulator -> benchmarks`` edge — strictly worse
(the resolved dependency-direction argument from alien_audit.py).

Build from ``TIER2_TUNED_PARAMS`` (masses=(1,1), the integrable §10 universe) —
NEVER ``DEFAULT_ALIEN_PARAMS`` (masses=(1000,1), proven non-integrable at
dt=0.005, L-037). The tangential IC speed is DERIVED from ``physics.potential_U``'s
own gradient (the ``simulator/fixtures._U_and_dUdr`` idiom, mirrored verbatim by
``config._tier2_two_body_config`` for the held-out set) — NEVER a hardcoded
v_circ, so the IC and the force the integrator applies can never disagree (single
source of truth).

The bundle is produced via ``AlienUniverse.to_observation_bundle`` — the audited
EXPLICIT-field-selection twin (alien.py:289-304). NEVER ``dataclasses.asdict`` of
the trajectory: that would re-leak the hidden charges/Q if a field were ever
added (RESEARCH Pitfall 4 / T-15.0-01).

$0 LLM, deterministic — this module is a PRODUCER, not a judge. It contains no
``LLMClient`` / ``genai`` / ``.embed`` (SCOPE §22.6). Citation: 15.2 RESEARCH
§Swap-Point Plan + 15.2 PATTERNS Pattern 1.

Threat-model mitigations: T-15.2-leak (data-only ObservationBundle is the only
Council-visible surface; truth_cfg + held_out travel a separate channel),
T-15.2-03 (TIER2_TUNED_PARAMS masses=(1,1), never DEFAULT_ALIEN_PARAMS).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.simulator import physics as P
from ascension.simulator.alien import AlienUniverse
from ascension.simulator.config import TIER2_HELD_OUT_ICS, TIER2_TUNED_PARAMS
from ascension.simulator.types import AlienConfig, ObservationBundle


def _reduced_mass(masses: tuple[float, ...]) -> float:
    """Reduced mass mu = m0·m1/(m0+m1) (mirror config._reduced_mass)."""
    m = np.asarray(masses, dtype=np.float64)
    return float(m[0] * m[1] / (m[0] + m[1]))


def _build_tier2_tuned_cfg(*, seed: int) -> AlienConfig:
    """Materialize the tuned-universe truth ``AlienConfig`` from TIER2_TUNED_PARAMS.

    Mirrors ``config._tier2_two_body_config`` VERBATIM (the IC-builder 15.1 used
    to build TIER2_HELD_OUT_ICS): derive the tangential IC speed from
    ``physics.potential_U``'s OWN 4th-order finite-difference gradient (the
    ``simulator/fixtures._U_and_dUdr`` idiom) — NEVER a hardcoded v_circ — so the
    IC speed and the force the integrator applies share ONE source of truth.

    ``config`` does not expose a ``build_tier2_tuned_cfg`` analog to
    ``build_tier2_held_out_ics`` (only the held-out builder is public), so we
    construct the tuned operating point inline here, reading r0 / tangential_frac
    / couplings / masses / charges from ``TIER2_TUNED_PARAMS`` — never recomputing
    or hardcoding any of them.
    """
    G = float(TIER2_TUNED_PARAMS["G"])
    alpha = float(TIER2_TUNED_PARAMS["alpha"])
    beta = float(TIER2_TUNED_PARAMS["beta"])
    gamma = float(TIER2_TUNED_PARAMS["gamma"])
    kappa = float(TIER2_TUNED_PARAMS["kappa"])
    masses = tuple(TIER2_TUNED_PARAMS["masses"])  # type: ignore[arg-type]
    charges = tuple(TIER2_TUNED_PARAMS["charges"])  # type: ignore[arg-type]
    r0 = float(TIER2_TUNED_PARAMS["r0"])
    tangential_frac = float(TIER2_TUNED_PARAMS["tangential_frac"])
    dt = float(TIER2_TUNED_PARAMS["dt"])
    n_steps = int(TIER2_TUNED_PARAMS["n_steps"])

    # Probe config (rel_speed=0) so potential_U sees the right couplings; the
    # speed is then derived from this config's own gradient — exactly the
    # config._tier2_two_body_config idiom. No hardcoded v_circ.
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
    )
    m = np.asarray(probe.masses, dtype=np.float64)
    ch = np.asarray(probe.charges, dtype=np.float64)

    def U(r: float) -> float:
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, probe)

    def dUdr(r: float) -> float:
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    mu = _reduced_mass(probe.masses)
    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    rel_speed = tangential_frac * v_circ
    return AlienConfig(
        G=G,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kappa=kappa,
        masses=masses,
        charges=charges,
        ics_pos=((0.0, 0.0), (r0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, rel_speed)),
        dt=dt,
        n_steps=n_steps,
        seed=seed,
    )


def alien_benchmark(
    *, seed: int = 0
) -> tuple[ObservationBundle, AlienConfig, tuple[AlienConfig, ...]]:
    """The alien-universe benchmark feed — the §10 swap-point producer (D-03).

    Returns ``(bundle, truth_cfg, held_out_ics)``:

      - ``bundle``: the DATA-ONLY ``ObservationBundle`` for the COUNCIL. The ONLY
        Council-visible surface — no charges, no Q, no config, no true law
        (T-15.2-leak / the §10 hidden/observable boundary).
      - ``truth_cfg``: the alien truth ``AlienConfig`` for the SCORER (Plan 02).
        The Council NEVER receives this.
      - ``held_out_ics``: ``config.TIER2_HELD_OUT_ICS`` for the SCORER. Reused,
        not recreated. The Council NEVER receives this either.

    ⚠️ The Council NEVER receives ``truth_cfg`` or the full ``AlienTrajectory`` —
    ONLY ``bundle``. ``truth_cfg`` + ``held_out_ics`` travel a SEPARATE channel
    (cli.py threads them on ``held_out``, Plan 03 wires round-close, Plan 02 reads
    them in the post-hoc depth scorer). This separation is the load-bearing
    zero-shot invariant.

    Args:
      seed: threaded into the truth ``AlienConfig`` so determinism is honoured
        (15.0 ICs are deterministic, so the RNG draws nothing yet, but the seed
        is carried so two calls are bit-identical — D-06). Kwarg-only.

    Returns:
      ``(ObservationBundle, AlienConfig, tuple[AlienConfig, ...])``.
    """
    # Build the alien truth config from the locked TIER2 universe (masses=(1,1)).
    truth_cfg = _build_tier2_tuned_cfg(seed=seed)

    # Run it forward and project to the data-only bundle via the audited
    # explicit-field-selection twin (NEVER dataclasses.asdict — would re-leak).
    traj = AlienUniverse(truth_cfg).run()
    bundle = AlienUniverse.to_observation_bundle(traj)

    # Reuse the 15.1 held-out generalization set (identity, not a fresh build).
    return bundle, truth_cfg, TIER2_HELD_OUT_ICS


# =============================================================================
# Phase 19-04 (AUTO-04, headline) — the multi-charge FAMILY proving ground.
# =============================================================================
#
# The autonomy loop TRIGGERS this family to break the single-charge degeneracy
# (Session 029: at equal charges s1*s2 == 1 == beta, so the per-body charge is
# DEGENERATE with a global amplitude — depth-2 is the honest ceiling). A FAMILY
# of universes at DISTINCT charge products makes the per-body charge identifiable
# from cross-config amplitude variation (the $0 discoverability audit, test_b,
# PROVED the fitted oscillatory amplitude TRACKS the charge product).
#
# Construction is the audit's proven pattern — same base law, vary ONLY the
# per-body charges via dataclasses.replace(base, charges=...). NO fresh / N-body
# builder (L-051 rule 2 explicitly rejected that).

# Observed family charge products (the Council sees these as data-only bundles).
_FAMILY_CHARGE_PRODUCTS: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0)

# Held-out SCORING products — STRICTLY OFF the observed family (P-EXP-076-7
# anti-reward-hack). Distinct from one another so the cross-config s-product gate
# (Task 2) has genuine amplitude variation to fit on the held-out set; disjoint
# from the observed products so a proposal must GENERALIZE its per-body-charge
# claim to UNSEEN products (it cannot memorize the observed configs).
_HELD_OUT_CHARGE_PRODUCTS: tuple[float, ...] = (3.0, 5.0)


def _charges_for_product(product: float) -> tuple[float, float]:
    """A ``(s1, s2)`` charge pair whose PRODUCT is ``product`` (s1 fixed at 1.0).

    The identifiable signal is the charge PRODUCT s1*s2 (the audit proved the
    fitted oscillatory amplitude tracks the product, not the individual charges),
    so fixing s1=1 and setting s2=product spans the products cleanly while every
    other parameter (masses/alpha/beta/gamma/kappa) stays identical across the
    family.
    """
    return (1.0, float(product))


def alien_family_benchmark(
    *,
    seed: int = 0,
    charge_products: tuple[float, ...] = _FAMILY_CHARGE_PRODUCTS,
    held_out_products: tuple[float, ...] = _HELD_OUT_CHARGE_PRODUCTS,
) -> tuple[list[tuple[ObservationBundle, AlienConfig]], tuple[AlienConfig, ...]]:
    """A multi-charge FAMILY of 2-body alien universes (one law, distinct charges).

    The headline proving ground (AUTO-04): the autonomy loop TRIGGERS this family
    to break the single-charge degeneracy. Every member shares ONE law (same
    G/alpha/beta/gamma/kappa/masses) and differs ONLY in per-body charges, built
    via ``dataclasses.replace(base_cfg, charges=...)`` — the exact pattern the $0
    discoverability audit proved (``audit_depth_discoverability._clean_probes_at_charges``).
    NO fresh many-body builder (L-051 rule 2 rejected a larger-body-count rebuild).

    Returns ``(family, held_out)``:
      - ``family``: list of ``(bundle, per_config_truth_cfg)`` for the OBSERVED
        charge products. ``bundle`` is the DATA-ONLY ObservationBundle the Council
        sees ({t, positions, velocities, masses} — NO charges/couplings/Q/law,
        P-EXP-076-6); ``per_config_truth_cfg`` is the scorer-side truth (the
        Council NEVER receives it).
      - ``held_out``: scorer-side truth configs at charge products STRICTLY OFF
        the observed family (P-EXP-076-7 anti-reward-hack). The cross-config
        L3/L4 gate (Task 2) recovers a(r) from THESE, so a proposal must
        GENERALIZE its per-body-charge claim to UNSEEN products. Each carries its
        own ICs (built by ``_build_tier2_tuned_cfg``), so a(r) is recoverable.

    Args:
      seed: threaded into the base truth config (determinism — D-06).
      charge_products: the OBSERVED products (default {1,2,4,6}).
      held_out_products: the held-out scoring products (default {3,5}); MUST be
        disjoint from ``charge_products`` (anti-reward-hack) and have >= 2 members
        (so the cross-config gate has variation).
    """
    base = _build_tier2_tuned_cfg(seed=seed)

    # The OBSERVED family — one data-only bundle per charge product. Reuses the
    # audited explicit-field-selection projection (NEVER dataclasses.asdict of the
    # trajectory — would re-leak the hidden charges/Q; RESEARCH Pitfall 4).
    family: list[tuple[ObservationBundle, AlienConfig]] = []
    for product in charge_products:
        cfg = dataclasses.replace(base, charges=_charges_for_product(product))
        traj = AlienUniverse(cfg).run()
        bundle = AlienUniverse.to_observation_bundle(traj)
        family.append((bundle, cfg))

    # The HELD-OUT scoring configs — off-family charges (anti-reward-hack). Truth
    # configs only (scorer-side); never projected to a Council-visible bundle.
    held_out = tuple(
        dataclasses.replace(base, charges=_charges_for_product(p)) for p in held_out_products
    )
    return family, held_out


# =============================================================================
# EXP-080 (STRAT-03) — the V2 regime FAMILY (sum coupling + r^-2.5).
# =============================================================================
#
# The V2 generalization-vs-overfit proving ground. Same construction pattern as
# the V1 family (one law, vary ONLY the per-body charges via
# dataclasses.replace), but on the V2 law (charge_coupling="sum",
# correction_exponent=2.5). Where the V1 family spans the charge PRODUCT s1*s2
# (the identifiable signal under the product coupling), the V2 family spans the
# charge SUM s1+s2 — the signal a sum-confound discoverer must recover. Held-out
# sums are STRICTLY OFF the observed set (the P-EXP-080-6 anti-reward-hack analog
# of the V1 {3,5} held-out products), so a proposal must GENERALIZE its per-body
# charge claim to UNSEEN sums.

# Observed family charge SUMS (the Council sees these as data-only bundles).
_FAMILY_V2_CHARGE_SUMS: tuple[float, ...] = (2.0, 3.0, 5.0, 7.0)

# Held-out SCORING sums — STRICTLY OFF the observed family (anti-reward-hack).
# Distinct from one another so the cross-config sum gate has genuine amplitude
# variation to fit; disjoint from the observed sums so the per-body-charge claim
# must generalize to UNSEEN sums.
_HELD_OUT_V2_CHARGE_SUMS: tuple[float, ...] = (4.0, 6.0)


def _build_tier2_tuned_cfg_v2(*, seed: int) -> AlienConfig:
    """The V2-regime tuned truth config — same operating point as V1 but on the
    V2 law (sum coupling + r^-2.5).

    Builds the V1 tuned config (single source of truth for masses / couplings /
    r0 / tangential_frac), then flips ONLY the two regime knobs via
    dataclasses.replace AND re-derives the tangential IC speed from the V2
    potential's own gradient. Re-deriving the IC is REQUIRED, not optional: the
    V2 potential differs (r^-2.5 correction + sum hidden term), so v_circ differs;
    reusing the V1 ICs would put the orbit off its intended tuned point. The
    derivation reuses the exact _U_and_dUdr idiom (no hardcoded v_circ).
    """
    v1 = _build_tier2_tuned_cfg(seed=seed)
    # Probe config on the V2 law with rel_speed=0 so potential_U sees the V2 law;
    # derive v_circ from its own gradient.
    probe = dataclasses.replace(
        v1,
        ics_vel=((0.0, 0.0), (0.0, 0.0)),
        charge_coupling="sum",
        correction_exponent=2.5,
    )
    m = np.asarray(probe.masses, dtype=np.float64)
    ch = np.asarray(probe.charges, dtype=np.float64)
    r0 = float(probe.ics_pos[1][0])

    def U(r: float) -> float:
        pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
        return P.potential_U(pos, m, ch, probe)

    def dUdr(r: float) -> float:
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    mu = _reduced_mass(probe.masses)
    # tangential_frac is carried in TIER2_TUNED_PARAMS, not on the cfg — read it
    # back from the V1 build by inverting v1's IC against the V1 gradient would be
    # circular; instead read the canonical value from the params dict directly.
    tangential_frac = float(TIER2_TUNED_PARAMS["tangential_frac"])
    v_circ = float(np.sqrt(r0 * dUdr(r0) / mu))
    rel_speed = tangential_frac * v_circ
    return dataclasses.replace(
        probe, ics_vel=((0.0, 0.0), (0.0, rel_speed))
    )


def _charges_for_sum(charge_sum: float) -> tuple[float, float]:
    """A ``(s1, s2)`` charge pair whose SUM is ``charge_sum`` (s1 fixed at 1.0).

    The identifiable signal under the V2 sum coupling is s1+s2, so fixing s1=1 and
    setting s2=charge_sum−1 spans the sums cleanly while every other parameter
    stays identical across the family (the sum analog of _charges_for_product).
    """
    return (1.0, float(charge_sum) - 1.0)


def alien_family_v2_benchmark(
    *,
    seed: int = 0,
    charge_sums: tuple[float, ...] = _FAMILY_V2_CHARGE_SUMS,
    held_out_sums: tuple[float, ...] = _HELD_OUT_V2_CHARGE_SUMS,
) -> tuple[list[tuple[ObservationBundle, AlienConfig]], tuple[AlienConfig, ...]]:
    """A multi-charge FAMILY of V2-regime universes (one V2 law, distinct sums).

    The EXP-080 V2 analog of :func:`alien_family_benchmark`: every member shares
    ONE V2 law (sum coupling + r^-2.5, same G/alpha/beta/gamma/kappa/masses) and
    differs ONLY in per-body charges, built via dataclasses.replace(base,
    charges=...). The family spans the charge SUM s1+s2 (the V2 identifiable
    signal) instead of the product.

    Returns ``(family, held_out)``:
      - ``family``: list of ``(bundle, per_config_truth_cfg)`` for the OBSERVED
        charge sums. ``bundle`` is the DATA-ONLY ObservationBundle the Council sees
        (no charges/couplings/Q/law); ``per_config_truth_cfg`` is scorer-side truth.
      - ``held_out``: scorer-side truth configs at charge sums STRICTLY OFF the
        observed family (anti-reward-hack). The cross-config sum gate recovers a(r)
        from THESE, so a proposal must GENERALIZE to UNSEEN sums.

    Args:
      seed: threaded into the base V2 truth config (determinism — D-06).
      charge_sums: the OBSERVED sums (default {2,3,5,7}).
      held_out_sums: the held-out scoring sums (default {4,6}); MUST be disjoint
        from ``charge_sums`` and have >= 2 members (cross-config variation).
    """
    base = _build_tier2_tuned_cfg_v2(seed=seed)

    family: list[tuple[ObservationBundle, AlienConfig]] = []
    for charge_sum in charge_sums:
        cfg = dataclasses.replace(base, charges=_charges_for_sum(charge_sum))
        traj = AlienUniverse(cfg).run()
        bundle = AlienUniverse.to_observation_bundle(traj)
        family.append((bundle, cfg))

    held_out = tuple(
        dataclasses.replace(base, charges=_charges_for_sum(s)) for s in held_out_sums
    )
    return family, held_out


def build_tier2_held_out_ics_v2(
    *, seed: int = 0, charge_sum: float = 2.0
) -> tuple[AlienConfig, ...]:
    """The V2-law analog of ``TIER2_HELD_OUT_ICS`` for the L1/L2/L5 baseline gates.

    The depth scorer's L1/L2/L5 gates run on a SINGLE-config (baseline-charge)
    held-out set, regardless of regime (L-052: the charge-independent layers must
    not run on high-charge data, which would inflate the Newton residual and sink
    L2's relative reduction). For V2 those gates must score against the V2 LAW
    (r^-2.5 correction), not the V1 law — else the V2 L2 gate (centered at -2.5)
    would find no correction in V1 (r^-3.5) data and manufacture a spurious null.

    This builds V2-law held-out configs at the BASELINE charge sum (default 2.0 =
    the lowest observed V2 family sum, the V2 analog of product-1), spanning the
    SAME r0 × tangential_frac grid as ``TIER2_HELD_OUT_ICS`` and re-deriving each
    IC speed from the V2 potential's OWN gradient (single source of truth — the V2
    potential differs, so v_circ differs). Deterministic; $0; no LLM.
    """
    import numpy as _np

    from ascension.simulator.config import TIER2_HELD_OUT_ICS

    base_v2 = dataclasses.replace(
        _build_tier2_tuned_cfg_v2(seed=seed),
        charges=_charges_for_sum(charge_sum),
    )
    m = _np.asarray(base_v2.masses, dtype=_np.float64)
    ch = _np.asarray(base_v2.charges, dtype=_np.float64)
    mu = _reduced_mass(base_v2.masses)

    def U(r: float) -> float:
        pos = _np.array([[0.0, 0.0], [r, 0.0]], dtype=_np.float64)
        return P.potential_U(pos, m, ch, base_v2)

    def dUdr(r: float) -> float:
        h = 1e-6
        return (-U(r + 2 * h) + 8 * U(r + h) - 8 * U(r - h) + U(r - 2 * h)) / (12 * h)

    out: list[AlienConfig] = []
    for v1_cfg in TIER2_HELD_OUT_ICS:
        # Reuse the V1 held-out's (r0, tangential_frac) ORBIT SHAPE: r0 is its
        # initial separation; tangential_frac = its tangential speed / V1 v_circ —
        # but re-derive the SPEED from the V2 gradient so the V2 orbit is at the
        # intended fraction of the V2 circular speed.
        r0 = float(v1_cfg.ics_pos[1][0])
        v1_rel_speed = float(v1_cfg.ics_vel[1][1])
        # The V1 IC was rel_speed = frac * v1_circ(r0); recover frac via the V1
        # circular speed at r0, then apply it to the V2 circular speed.
        m_v1 = _np.asarray(v1_cfg.masses, dtype=_np.float64)
        ch_v1 = _np.asarray(v1_cfg.charges, dtype=_np.float64)
        mu_v1 = float(m_v1[0] * m_v1[1] / (m_v1[0] + m_v1[1]))

        def _U_v1(r, _c=v1_cfg, _m=m_v1, _ch=ch_v1):
            pos = _np.array([[0.0, 0.0], [r, 0.0]], dtype=_np.float64)
            return P.potential_U(pos, _m, _ch, _c)

        def _dUdr_v1(r, _f=_U_v1):
            h = 1e-6
            return (-_f(r + 2 * h) + 8 * _f(r + h) - 8 * _f(r - h) + _f(r - 2 * h)) / (12 * h)

        v1_circ = float(_np.sqrt(r0 * _dUdr_v1(r0) / mu_v1))
        frac = v1_rel_speed / v1_circ if v1_circ > 0 else 0.85
        v2_circ = float(_np.sqrt(r0 * dUdr(r0) / mu))
        rel_speed = frac * v2_circ
        out.append(
            dataclasses.replace(
                base_v2,
                ics_pos=((0.0, 0.0), (r0, 0.0)),
                ics_vel=((0.0, 0.0), (0.0, rel_speed)),
            )
        )
    return tuple(out)


def alien_family_v2_observation(
    *,
    seed: int = 0,
    charge_sums: tuple[float, ...] = _FAMILY_V2_CHARGE_SUMS,
) -> AlienFamilyObservation:
    """Build the Council-visible :class:`AlienFamilyObservation` for the V2 regime.

    Convenience wrapper over :func:`alien_family_v2_benchmark` that keeps ONLY the
    data-only bundles (drops the scorer-side per-config truth). The held-out
    scoring configs the post-hoc depth gate needs come from
    ``alien_family_v2_benchmark`` directly (scorer channel, never Council-visible).
    """
    family, _held_out = alien_family_v2_benchmark(seed=seed, charge_sums=charge_sums)
    return AlienFamilyObservation(bundles=tuple(b for b, _cfg in family))


@dataclasses.dataclass(frozen=True)
class AlienFamilyObservation:
    """A Council-visible FAMILY of data-only alien observations (EXP-076 / AUTO-04).

    Wraps an ordered tuple of data-only ``ObservationBundle``s that share ONE law
    but differ in an unobserved per-body property (the hidden charge product —
    NEVER named or valued to the Council). The Council observes ALL members at
    once (Phase 19-05 family-feed consumption) so it can detect that a force
    component's amplitude VARIES across members — the cross-configuration signal
    that breaks the single-charge ``s1*s2 ≡ beta`` degeneracy (L-051). Sequential
    single-config episodes with shared Agora memory were REJECTED: memory carries
    forward prior hypotheses, not raw trajectories, so the cross-config amplitude
    signal would be lost.

    DUCK-TYPES THE BASELINE (first) MEMBER: it exposes ``.positions /.velocities /
    .masses /.t`` of ``bundles[0]`` so every EXISTING single-bundle consumer (the
    round-close in-loop scoring channel, the ``hasattr(spec,'positions') and not
    hasattr(spec,'system')`` alien discriminator) treats it as the baseline config
    UNCHANGED, while the render path keys on ``.bundles`` to show all members. It
    carries NO charges / Q / law / products — each member is data-only by
    construction (``alien.py:to_observation_bundle``); presenting several data-only
    bundles leaks nothing the §10 boundary forbids.
    """

    bundles: tuple[ObservationBundle, ...]

    def __post_init__(self) -> None:
        if len(self.bundles) < 2:
            raise ValueError(
                "AlienFamilyObservation needs >= 2 members to be a family; "
                f"got {len(self.bundles)}."
            )

    @property
    def positions(self):  # noqa: ANN201 - mirrors ObservationBundle attr
        return self.bundles[0].positions

    @property
    def velocities(self):  # noqa: ANN201
        return self.bundles[0].velocities

    @property
    def masses(self):  # noqa: ANN201
        return self.bundles[0].masses

    @property
    def t(self):  # noqa: ANN201
        return self.bundles[0].t


def alien_family_observation(
    *,
    seed: int = 0,
    charge_products: tuple[float, ...] = _FAMILY_CHARGE_PRODUCTS,
) -> AlienFamilyObservation:
    """Build the Council-visible :class:`AlienFamilyObservation` for ``seed``.

    Convenience wrapper over :func:`alien_family_benchmark` that keeps ONLY the
    data-only bundles (drops the scorer-side per-config truth). The held-out
    scoring configs the post-hoc depth gate needs come from
    ``alien_family_benchmark`` directly (scorer channel, never Council-visible).
    """
    family, _held_out = alien_family_benchmark(seed=seed, charge_products=charge_products)
    return AlienFamilyObservation(bundles=tuple(b for b, _cfg in family))


__all__ = [
    "alien_benchmark",
    "alien_family_benchmark",
    "AlienFamilyObservation",
    "alien_family_observation",
    # EXP-080 V2 regime (STRAT-03).
    "alien_family_v2_benchmark",
    "alien_family_v2_observation",
    "build_tier2_held_out_ics_v2",
]
