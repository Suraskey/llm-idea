"""Phase B — the deterministic residual-feedback tool (the "how wrong am I" signal).

Given a Council hypothesis's ``symbolic_form`` (a radial force / acceleration law
as a string), compute how much of the REAL observed acceleration it explains and a
LEAK-SAFE characterization of the leftover residual, so the Council can SEE how
wrong its current best law is and refine — the missing signal behind the EXP-072
depth-1 plateau (SEED-017). A scientist fits a guess, inspects the leftover, and
refines; this gives the Council that leftover.

$0 LLM, deterministic numpy/scipy/sympy. This is a TOOL the agents reason over;
the ground-truth depth judge (``alien_depth.score_depth``) stays a SEPARATE
post-hoc oracle (SCOPE §22.6 — no LLM in the validation pipeline).

Two integrity invariants make this safe:
  - L-016 (anchor to ground truth, never self-report): the fit is to the OBSERVED
    radial acceleration recovered from trajectory data (``_observed_accel``), and
    we fit FREE coefficients to the agent's law SHAPE — we never substitute the
    truth's couplings (that is what ``round_close._alien_fit_rmse`` does, and it
    would leak the answer). Nothing reads the agent's confidence.
  - Anti-leak (validation is not circular, SCOPE §10): the rendered block reports
    only the SHAPE of the leftover (a coarse "steeper power" band + whether an
    oscillatory component is present). It NEVER prints the truth couplings, the
    exact correction exponent, the oscillation wavelength, or any domain word —
    else depth-2 becomes a trivial copy of a revealed answer.

Layering (L-018): benchmarks-side, reusing the simulator + the alien_depth
deterministic machinery with zero new import edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy

from ascension.benchmarks.alien_depth import _observed_accel, _parse
from ascension.common.logging import get_logger
from ascension.simulator.types import AlienConfig

logger = get_logger(__name__)

# Candidate steeper-power exponents probed against the leftover (the L2 nudge).
# Reported only as a COARSE BAND, never the exact best value (anti-leak: the truth
# correction sits at 3.5; we must not hand that over — see _render_block).
_POWER_GRID: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0)
# Oscillation-frequency probe grid (presence-only; the fitted frequency is NEVER
# rendered — reporting the wavelength would reveal the hidden coupling).
_OSC_FREQ_GRID: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9, 1.2, 1.5)
# A correction buys a "material" reduction if it removes at least this fraction of
# the leftover RSS (mirrors the alien_depth L2/L3 5% margins).
_MATERIAL_RSS_REDUCTION = 0.05

# Per-body kinematic component stems. A term carrying one of these is NOT a pure
# radial law and cannot be faithfully reduced to f(r); such terms are excluded
# from the fit (with a note) rather than crudely collapsed.
_COMPONENT_STEMS: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "p",
)


@dataclass(frozen=True, slots=True)
class ResidualFeedback:
    """The leftover-after-your-law signal for one (best) hypothesis.

    ``fit_fraction``: 1 - RMSE(residual)/RMS(observed accel) — "explains X% of the
      motion." ``rendered_block`` is the leak-safe, brace-free prompt string
      (pre-rendered so the hot prompt-build path does no sympy work).
    """

    fit_fraction: float
    residual_rmse: float
    steeper_power_present: bool
    steeper_power_rss_reduction: float
    oscillation_present: bool
    oscillation_rss_reduction: float
    n_radial_terms_fit: int
    rendered_block: str


def _term_is_radial(term: sympy.Expr) -> bool:
    """True iff the term has no per-body kinematic component symbol.

    A term like ``G*m2/r**2`` is radial (G/m2 are constants); ``G*(x2-x1)/r**3``
    is NOT (it carries Cartesian components) and cannot be reduced to a function
    of the scalar distance r alone.
    """
    for sym in term.free_symbols:
        name = str(sym)
        if name == "r":
            continue
        # strip a trailing body index to get the stem (vx1 -> vx, x_0 -> x).
        stem = name.rstrip("0123456789").rstrip("_")
        if stem in _COMPONENT_STEMS:
            return False
    return True


def _term_r_shape(term: sympy.Expr, r_values: np.ndarray) -> np.ndarray | None:
    """Reduce a radial term to its r-shape column: set every non-r symbol to 1.

    The free coefficient absorbed later by least squares stands in for the
    constants (G, alpha, m1, ...) — we only need the term's DEPENDENCE on r.
    Returns the column evaluated over ``r_values``, or None on a non-finite /
    unlambdifiable shape.
    """
    r = sympy.Symbol("r")
    subs = {s: 1.0 for s in term.free_symbols if str(s) != "r"}
    shape = term.subs(subs)
    try:
        f = sympy.lambdify(r, shape, modules="numpy")
        col = np.asarray(f(r_values), dtype=np.float64)
    except Exception:  # noqa: BLE001 — unlambdifiable shape, skip the term
        return None
    if col.ndim == 0:  # a constant shape broadcast to the column length
        col = np.full_like(r_values, float(col))
    if col.shape != r_values.shape or not np.all(np.isfinite(col)):
        return None
    return col


def _lstsq_residual(columns: list[np.ndarray], target: np.ndarray) -> np.ndarray:
    """Residual of a free-coefficient least-squares fit (mirrors _rss_fit's lstsq).

    Returns ``target - X @ c`` (the leftover the columns cannot explain).
    """
    X = np.vstack(columns).T
    c, *_ = np.linalg.lstsq(X, target, rcond=None)
    return target - X @ c


def _rss(v: np.ndarray) -> float:
    return float(np.sum(v**2))


def compute_residual_feedback(
    symbolic_form: str,
    truth_cfg: AlienConfig,  # noqa: ARG001 — kept for signature symmetry / future
    held_out_ics: tuple[AlienConfig, ...],
) -> ResidualFeedback | None:
    """Compute the leak-safe residual-feedback signal for one law.

    Returns None (graceful, loud-logged) when the law cannot be parsed or has no
    usable radial term — the round simply carries no feedback that cycle.
    """
    expr = _parse(symbolic_form)
    if expr is None:
        return None  # _parse already logged the parse failure

    r_obs, a_obs = _observed_accel(held_out_ics)
    r_obs = np.asarray(r_obs, dtype=np.float64)
    a_obs = np.asarray(a_obs, dtype=np.float64)
    if r_obs.size < 4 or not np.all(np.isfinite(a_obs)):
        logger.warning("residual_feedback: degenerate observed accel; skipped")
        return None

    # Build one fit column per RADIAL additive term (non-radial terms excluded).
    columns: list[np.ndarray] = []
    n_excluded = 0
    for term in sympy.expand(expr).as_ordered_terms():
        if not _term_is_radial(term):
            n_excluded += 1
            continue
        col = _term_r_shape(term, r_obs)
        if col is not None:
            columns.append(col)
    if not columns:
        logger.warning(
            "residual_feedback: %r has no usable radial term (%d non-radial " "excluded); skipped",
            symbolic_form,
            n_excluded,
        )
        return None

    residual = _lstsq_residual(columns, a_obs)
    rss_base = _rss(residual)
    residual_rmse = float(np.sqrt(np.mean(residual**2)))
    rms_signal = float(np.sqrt(np.mean(a_obs**2)))
    fit_fraction = max(0.0, 1.0 - residual_rmse / rms_signal) if rms_signal > 0 else 0.0

    # What would a STEEPER inverse-power term buy on top of the agent's law?
    best_power_red = 0.0
    if rss_base > 0:
        for p in _POWER_GRID:
            added = -1.0 / r_obs**p
            red = (rss_base - _rss(_lstsq_residual(columns + [added], a_obs))) / rss_base
            best_power_red = max(best_power_red, red)
    steeper_power_present = best_power_red >= _MATERIAL_RSS_REDUCTION

    # Is there an OSCILLATORY component left over? (presence only — never the freq)
    best_osc_red = 0.0
    if rss_base > 0:
        for k in _OSC_FREQ_GRID:
            added = (1.0 - np.cos(k * r_obs)) / r_obs**2
            red = (rss_base - _rss(_lstsq_residual(columns + [added], a_obs))) / rss_base
            best_osc_red = max(best_osc_red, red)
    oscillation_present = best_osc_red >= _MATERIAL_RSS_REDUCTION

    rendered = _render_block(fit_fraction, steeper_power_present, oscillation_present)
    return ResidualFeedback(
        fit_fraction=fit_fraction,
        residual_rmse=residual_rmse,
        steeper_power_present=steeper_power_present,
        steeper_power_rss_reduction=best_power_red,
        oscillation_present=oscillation_present,
        oscillation_rss_reduction=best_osc_red,
        n_radial_terms_fit=len(columns),
        rendered_block=rendered,
    )


def _render_block(fit_fraction: float, steeper_power: bool, oscillation: bool) -> str:
    """The leak-safe, brace-free prompt block.

    Reports only the SHAPE of the leftover: a coarse "steeper inverse-power"
    band (NEVER the exact correction exponent) and whether an oscillatory
    component is present (NEVER its wavelength/frequency). Contains no truth
    couplings and no '{'/'}' (the live prompt is built via str.format — Pitfall 4).
    """
    pct = int(round(100 * fit_fraction))
    lines = [
        f"CURRENT BEST LAW (across the Council) explains about {pct}% of the "
        f"observed motion. A small but SYSTEMATIC leftover remains (it is not "
        f"noise — it repeats across the trajectory):",
    ]
    if steeper_power:
        lines.append(
            "  - a steeper inverse-power term than 1/r**2 would explain part of "
            "what is left. Its exponent need NOT be a whole number: do not just "
            "try 1/r**3 or 1/r**4 — treat the power as a CONTINUOUS value and fit "
            "it to the leftover (a non-integer exponent is allowed and likely)."
        )
    if oscillation:
        lines.append(
            "  - an OSCILLATORY component is also present: the leftover changes "
            "sign across the distance range (a periodic term, not a pure power)."
        )
    if not steeper_power and not oscillation:
        lines.append(
            "  - the leftover has no single dominant structure the standard "
            "corrections capture; look for a qualitatively different term."
        )
    lines.append(
        "Propose a symbolic_form that ALSO accounts for this leftover, using "
        "only the allowed observable names and any constants you posit."
    )
    return "\n".join(lines)


__all__ = ["ResidualFeedback", "compute_residual_feedback"]
