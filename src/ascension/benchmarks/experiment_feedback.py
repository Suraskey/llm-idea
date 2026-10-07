"""Phase C — the deterministic active-experimentation instrument (the depth-2 lever).

The EXP-072 / EXP-073 honest negatives plus the $0 identifiability post-mortem
(NOTEBOOK Session 025 cont. #2; L-044 RESOLUTION; SEED-018) established that the
truth's ``alpha/r**3.5`` correction is PERFECTLY identifiable from a controlled
radial-drop experiment but DESTROYED in the passive near-circular orbit observable
(the kappa tangential-inertia confound makes the finite-difference-recovered radial
acceleration multivalued in r at the correction's own ~5-9% magnitude). The Phase B
residual loop directed the search and broke the integer-exponent anchor but plateaued
at depth 1: LLMs GUESS symbolic forms (r**3, r**4, symbolic r**p, r**2.5) rather than
numerically self-fit the continuous exponent.

This module supplies the missing observability. Given a Council hypothesis's
``symbolic_form`` that ALREADY proposes a steeper inverse-power correction term (the
agent's OWN structure), it (1) runs CLEAN radial-drop probe experiments — bodies
released from rest so v_perp=0 switches off the kappa confound — recovering a clean
``a(r)`` from the OBSERVED trajectory via a 4th-order finite difference (NOT from the
force law: the §10 data-only boundary), and (2) fits the exponent of the agent's
proposed correction term to that clean data by a profiled grid, returning the fitted
value and the residual-RSS reduction. The agent can then commit to that exponent —
the move a real physicist makes after designing a cleaner measurement.

$0 LLM, deterministic numpy/scipy/sympy. The ground-truth depth judge
(``alien_depth.score_depth``) stays a SEPARATE post-hoc oracle (SCOPE §22.6 — no LLM
in the validation pipeline), scored on held-out ICs the Council does not choose.

Three integrity invariants make this safe (SCOPE §10 active-experimentation amendment,
2026-05-26, Surya signed off):
  - Data-only boundary: the clean a(r) is recovered from an OBSERVED radial-drop
    trajectory (positions over time), via finite difference — never by reading
    ``physics.total_accel`` / the force law. The simulator stays the oracle.
  - Fit-the-agent's-OWN-structure (the blessed boundary): the instrument fits a free
    parameter of the structure the AGENT proposed (a steeper inverse-power correction
    term) and reports its fitted exponent. It NEVER reveals the true law, the true
    couplings, the hidden-charge term, the oscillation frequency, or any structure the
    agent did not itself propose. If the agent has proposed no correction term, the
    instrument returns None (nothing of theirs to fit).
  - Non-circular (SCOPE §10): the creative discovery — conceiving the correction
    structure (Stage 1) and designing the disentangling experiment (Stage 2) — is the
    Council's; the headline metric remains the separate held-out ``score_depth`` oracle
    the agent cannot game; the EXP-075 Council-vs-single-model ablation controls for
    "the tool did the discovery."

Layering (L-018): benchmarks-side, reusing the simulator + the alien_depth
deterministic machinery (``_parse``, ``_r_power_exponents``, ``_rss_fit``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import sympy

from ascension.benchmarks.alien_depth import _parse, _r_power_exponents, _rss_fit
from ascension.common.logging import get_logger
from ascension.simulator.alien import AlienUniverse
from ascension.simulator.config import _tier2_two_body_config
from ascension.simulator.types import AlienConfig

logger = get_logger(__name__)

# --- Clean radial-drop probe battery -----------------------------------------
# Release-from-rest drops (tangential_frac=0.0 → v_perp=0 → kappa confound OFF).
# Finer dt than the tuned 5e-3 + more steps resolve the fast infall; multiple r0
# span the r-range so the small-r region (where the alpha correction is strongest)
# is covered. Pure radial motion keeps a(r) single-valued in r regardless of the
# leg, so there is no orbit confound to fight.
_PROBE_R0_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
_PROBE_DT: float = 5e-4
_PROBE_N_STEPS: int = 8000
_PROBE_R_MIN: float = 0.5  # discard small-r where FD truncation / the singularity bite
_PROBE_FD_TRIM: int = 4  # endpoint samples dropped for the 4th-order stencil

# --- Exponent fit -------------------------------------------------------------
# Profiled grid for the correction exponent p in a 1/r**p term. Resolution 0.1
# lands the minimum on the half-integer the L2 gate wants (truth = 3.5); the grid
# brackets the integer guesses (3, 4) and the fractional undershoots (2.5, 2.7)
# the residual loop produced.
_FIT_GRID: tuple[float, ...] = tuple(round(p, 4) for p in np.arange(2.0, 6.0001, 0.1))
# A fitted correction buys a "material" reduction if it removes at least this
# fraction of the Newton-only RSS on the clean data (mirrors the alien_depth L2 5%).
_MATERIAL_RSS_REDUCTION: float = 0.05
# A radial term counts as the agent's "correction" if it is steeper than Newton by
# at least this much (matches the alien_depth L2 ``second_terms`` 0.25 band), OR if
# it carries a symbolic (un-fitted) exponent.
_CORRECTION_STEEPER_THAN: float = -2.25
# The α correction is SHORT-RANGE (a steeper inverse power), so it is identifiable
# where it dominates: small r. Profiling over the full probe range lets the hidden
# (1-cos γr)/r² charge term — which oscillates and weakens the α signal at large r —
# bias the single-power fit (≈3.1 over [0.5,4], ≈3.4 over [0.5,1.0]). Fitting the
# small-r window is correct experimental design and LEAK-SAFE (uses no hidden γ /
# charge knowledge); it recovers ≈3.4, comfortably inside the L2 near_35 band.
_FIT_R_MAX: float = 1.0


@dataclass(frozen=True, slots=True)
class FittedForm:
    """The clean-data fit of the agent's OWN proposed correction term.

    ``fitted_exponent`` is the profiled inverse-power exponent p (the proposed
    ``1/r**p`` correction best-fits the clean radial-drop data at this p — truth is
    3.5). ``rss_reduction`` is the fraction of the Newton-only RSS the fitted
    correction removes on the clean data. ``rendered_block`` is the leak-bounded,
    brace-free prompt string (pre-rendered so the hot prompt-build path does no work).
    """

    fitted_exponent: float
    rss_reduction: float
    n_clean_points: int
    rendered_block: str


def run_clean_probes(
    seed: int, *, base_cfg: AlienConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Run the release-from-rest radial-drop battery; return clean ``(r, a_radial)``.

    For each r0 in the grid: build the probe config with ``tangential_frac=0.0``
    (release from rest → v_perp=0 → kappa inert), re-resolve it at a finer dt, run
    the simulator, and recover the radial acceleration from the OBSERVED relative
    trajectory via a 4th-order central second difference. Concatenate over r0,
    keeping only the clean mid-r window. $0, deterministic.

    ``base_cfg`` is the run's REGIME truth config (the round-close coordinator's
    ``held_out[0]``): when given, probes are release-from-rest drops in THAT
    universe (same law knobs — ``charge_coupling``, ``correction_exponent`` — and
    couplings/masses/charges), so the battery measures the regime the agents are
    actually in. When None, the V1 TIER2 builder is used (byte-identical legacy
    path). L-063: the V2 EXP-080 run left this unwired, so the battery simulated
    the V1 universe and fed ``fitted_exponent=3.4`` (V1's answer) to agents
    working in the V2 regime (truth 2.5) — a regime-blind instrument actively
    steering agents AWAY from the truth they were converging on.
    """
    rs: list[np.ndarray] = []
    ars: list[np.ndarray] = []
    for r0 in _PROBE_R0_GRID:
        if base_cfg is None:
            base = _tier2_two_body_config(r0=r0, tangential_frac=0.0, seed=seed)
        else:
            # Release-from-rest (v=0) needs no v_circ derivation; only the IC
            # geometry changes — every law knob comes from the regime config.
            base = replace(
                base_cfg,
                ics_pos=((0.0, 0.0), (float(r0), 0.0)),
                ics_vel=((0.0, 0.0), (0.0, 0.0)),
                seed=seed,
            )
        cfg = replace(base, dt=_PROBE_DT, n_steps=_PROBE_N_STEPS)
        traj = AlienUniverse(cfg).run()
        rel = traj.positions[:, 1, :] - traj.positions[:, 0, :]
        if rel.shape[0] < 2 * _PROBE_FD_TRIM + 3:
            continue
        # 4th-order central second difference: a[k] = (-rel[k+2] + 16 rel[k+1]
        # - 30 rel[k] + 16 rel[k-1] - rel[k-2]) / (12 dt^2).
        a_obs = (-rel[4:] + 16.0 * rel[3:-1] - 30.0 * rel[2:-2] + 16.0 * rel[1:-3] - rel[:-4]) / (
            12.0 * cfg.dt**2
        )
        rel_mid = rel[2:-2]
        r = np.linalg.norm(rel_mid, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rhat = rel_mid / r[:, None]
            a_radial = np.sum(a_obs * rhat, axis=1)
        # Keep ONLY the clean inbound infall leg: from release down to the first
        # perihelion (global min of r — the close passage). Integrating a head-on
        # radial drop THROUGH r~=0 injects spurious energy at the 1/r**2
        # singularity and slingshots the body to large r; the outbound leg re-enters
        # the r-window with garbage FD accelerations. The inbound leg recovers a(r)
        # to ~1e-3 (SEED-018). Combined with r >= _PROBE_R_MIN this also drops the
        # violent near-center samples.
        i_peri = int(np.argmin(r))
        inbound = np.zeros(r.size, dtype=bool)
        inbound[: i_peri + 1] = True
        keep = (
            inbound
            & np.isfinite(a_radial)
            & np.isfinite(r)
            & (r >= _PROBE_R_MIN)
            & (r <= r0 * 0.99)
        )
        if _PROBE_FD_TRIM > 0 and keep.size > 2 * _PROBE_FD_TRIM:
            keep[:_PROBE_FD_TRIM] = False
            keep[-_PROBE_FD_TRIM:] = False
        rs.append(r[keep])
        ars.append(a_radial[keep])
    if not rs:
        return np.empty(0), np.empty(0)
    return np.concatenate(rs), np.concatenate(ars)


def _has_fittable_correction(expr: sympy.Expr) -> bool:
    """True iff the agent proposed a steeper inverse-power OR symbolic-exponent term.

    The instrument only fits structure the AGENT itself proposed (the §10 blessed
    boundary). A bare Newton law (only an r**-2 term) has nothing to fit and returns
    False — the residual loop's job is to get them to propose a correction first.
    """
    for e in _r_power_exponents(expr):
        if e <= _CORRECTION_STEEPER_THAN:  # steeper inverse-power than Newton
            return True
    # Symbolic (un-fitted) exponent on r, e.g. ``C/r**p`` — the residual loop's frame.
    r = sympy.Symbol("r")
    for term in sympy.expand(expr).as_ordered_terms():
        factors = term.as_ordered_factors() if term.is_Mul else [term]
        for f in factors:
            base, exp = f.as_base_exp()
            if base == r and not exp.is_number:
                return True
    return False


def fit_proposed_structure(
    symbolic_form: str, clean_r: np.ndarray, clean_a: np.ndarray
) -> FittedForm | None:
    """Fit the exponent of the agent's proposed correction term to clean probe data.

    Returns None (graceful, logged) when the law cannot be parsed, proposes no
    correction term to fit, or the clean data is degenerate — the round simply
    carries no experiment feedback that cycle.
    """
    expr = _parse(symbolic_form)
    if expr is None:
        return None  # _parse already logged the parse failure
    if not _has_fittable_correction(expr):
        logger.info(
            "experiment_feedback: %r proposes no correction term to fit; skipped",
            symbolic_form,
        )
        return None

    r = np.asarray(clean_r, dtype=np.float64)
    a = np.asarray(clean_a, dtype=np.float64)
    if r.size < 8 or not np.all(np.isfinite(a)) or not np.all(r > 0):
        logger.warning("experiment_feedback: degenerate clean probe data; skipped")
        return None

    # Identify the short-range correction where it dominates (small r). Leak-safe.
    window = r <= _FIT_R_MAX
    if int(np.count_nonzero(window)) >= 8:
        r = r[window]
        a = a[window]

    newton = -1.0 / r**2
    rss_newton = _rss_fit(a, [newton])
    if rss_newton <= 0.0:
        logger.warning("experiment_feedback: Newton already fits clean data; skipped")
        return None

    # Profile the correction exponent: at each p, fit [Newton, -1/r**p] by free
    # least squares and take the RSS minimum. The agent's structure, fitted.
    best_p: float | None = None
    best_rss = float("inf")
    for p in _FIT_GRID:
        corr = -1.0 / r**p
        rss = _rss_fit(a, [newton, corr])
        if rss < best_rss:
            best_rss = rss
            best_p = p
    if best_p is None:
        return None

    rss_reduction = (rss_newton - best_rss) / rss_newton
    rendered = _render_block(best_p, rss_reduction)
    return FittedForm(
        fitted_exponent=best_p,
        rss_reduction=rss_reduction,
        n_clean_points=int(r.size),
        rendered_block=rendered,
    )


def compute_experiment_feedback(
    symbolic_form: str, seed: int, *, base_cfg: AlienConfig | None = None
) -> FittedForm | None:
    """Round-close entry point: run the clean probe battery, fit the agent's structure.

    $0, deterministic. Returns None when there is nothing of the agent's to fit or the
    probes are degenerate (the round carries no experiment feedback that cycle).
    ``base_cfg`` (the run's regime truth config) makes the battery regime-faithful —
    see :func:`run_clean_probes` (L-063).
    """
    # Short-circuit BEFORE the (relatively expensive) probe battery when the
    # agent has proposed no correction term to fit — the blessed boundary only
    # fits the agent's OWN structure.
    expr = _parse(symbolic_form)
    if expr is None or not _has_fittable_correction(expr):
        logger.info(
            "experiment_feedback: %r proposes no correction term to fit; " "battery not run",
            symbolic_form,
        )
        return None
    clean_r, clean_a = run_clean_probes(seed, base_cfg=base_cfg)
    if clean_r.size == 0:
        logger.warning("experiment_feedback: clean probe battery returned no points")
        return None
    return fit_proposed_structure(symbolic_form, clean_r, clean_a)


def _render_block(fitted_exponent: float, rss_reduction: float) -> str:
    """The leak-bounded, brace-free prompt block (the blessed §10 boundary).

    Names ONLY the agent's OWN proposed correction term and its fitted inverse-power
    exponent + RSS reduction. Contains NO truth coupling, NO hidden-charge term, NO
    oscillation frequency, NO domain word, and no '{'/'}' (the live prompt is built
    via str.format — brace literals would break it). Surya signed off this exact
    boundary 2026-05-26: the instrument fits and reports the agent's own hypothesis
    against data the agent gathered, never the answer.
    """
    pct = int(round(100 * rss_reduction))
    p = fitted_exponent
    lines = [
        "CONTROLLED EXPERIMENT RESULT. You requested simpler motion to isolate "
        "your proposed correction. On a controlled subset of the data where the "
        "motion is cleaner, your proposed correction term beyond the inverse-square "
        f"force best fits at an inverse-power exponent of about {p:.1f} "
        f"(a 1/r**{p:.1f} term), and at that exponent it removes about {pct}% of "
        "what the inverse-square force alone leaves over.",
        f"Consider committing to that form: an inverse-square term plus a 1/r**{p:.1f} "
        "correction, using only the allowed observable names and any constants you "
        "posit. Then refine further if a systematic leftover remains.",
    ]
    return "\n".join(lines)


__all__ = [
    "FittedForm",
    "compute_experiment_feedback",
    "fit_proposed_structure",
    "run_clean_probes",
]
