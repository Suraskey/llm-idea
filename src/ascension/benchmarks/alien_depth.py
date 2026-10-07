"""The 6-layer deterministic discovery-depth scorer (Phase 15.2-02, TIER2-03/04).

Plain English: this takes the Council's final proposed law(s) — each a
``symbolic_form`` string like ``G*m1*m2*(1/r**2 + alpha/r**3.5) + ...`` — and
asks how DEEP into the Alien Universe's §10 discovery layers the proposal got:

  L1  dominant inverse-power force        (~ c/r**2)
  L2  correction term                     (+ alpha/r**3.5)
  L3  hidden-charge EXISTENCE             (an unobserved per-body scalar s1*s2)
  L4  hidden-force functional form        (+ beta*s1*s2*(1 - cos(gamma*r))/r**2)
  L5  conservation law                    (Q = kinetic + U + kappa*s**2*v_perp**2)
  L6  hidden symmetry                      (parity-charge P:(x->-x, v->-v, s->-s))

It returns the MAX §10 layer reached across the run's hypotheses (L5/L6 read the
set jointly), as a ``DepthScore``.

⚠️ ELICITATION SCOPE (Phase 15.2 TIER2-03, 2026-05-24): the Council prompt asks
ONLY for the force law (the acceleration RHS) — it never asks for a conserved
quantity (L5) or a symmetry statement (L6). So L1-L4 (the force-law layers) are
the HONEST discovery-depth metric for the current prompt; L5/L6 are reachable
only by an agent volunteering a conserved-quantity / symmetry expression, which
the prompt does not request. The live TIER2-03 run exposed an L5 over-credit (a
force law ``G*m2/r**2 - kappa*v1**2`` tripping a too-loose structural check); the
L5 gate is now tightened to require a genuine conserved-quantity structure
(potential U + hidden-charge product), so a force law can no longer false-positive
into L5. The L5/L6 numerical gates remain conservative-quantity-shaped; treat
L1-L4 as the reported metric until a prompt elicits L5/L6 directly.

$0 LLM, deterministic sympy/scipy/structural — the ONLY parser is
_safe_parse_expr; rationale never alone confers a layer (SCOPE §22.6 no-LLM-judge
guard); runs POST-HOC over a run's hypothesis nodes, NOT through round-close.

Why post-hoc and off the round-close seam (T-15.2-01, 15.2 RESEARCH MAKE-OR-BREAK,
PATTERNS NON-OBVIOUS #2): routing alien proposals through the ODE ``score()``
returns UNKNOWN_SYSTEM and triggers the L-016 confidence-circularity. The depth
metric is a SEPARATE pass over the run's persisted hypothesis nodes, so the
headline result never depends on the in-loop selection path.

The no-LLM-judge guard (D-02, §22.6, T-15.2-04): the sole parser is sympy
``parse_expr`` via the existing ``_safe_parse_expr`` (scoring.py:128, empty
``__builtins__`` global_dict, L-013-safe subscripted-name binding). No model
interprets the proposal. ``rationale`` is read ONLY through a fixed keyword set,
logged as a tiebreaker, never authoritative — the symbolic+numerical predicate
decides every layer.

The anti-gaming discipline (D-08, 15.2 RESEARCH §adversarial table, T-15.2-05):
each layer pairs a structural sympy test with a BEHAVIORAL / numerical gate, so
a proposal cannot game depth by sprinkling a free symbol (decorative-s ⇏ L3 — the
held-out RMSE with the hidden symbols set to 0 must be materially worse), writing
a cos at the wrong frequency (wrong-freq ⇏ L4 — gamma_est ~ 0.7 AND held-out RMSE
< 1e-3), asserting "conserved" in prose (drifting-Q ⇏ L5 — the proposed Q's
``traj.Q``-style drift must be bounded), or claiming a parity symmetry it does not
have (false-symmetry ⇏ L6 — sympy verifies the invariance of the proposed LAW).

The numerical oracle is the locked simulator, read through the 15.1 nested-model
lens: a position-only discoverer recovers the per-step radial acceleration a(r)
by central finite difference (``tuning._observed_radial_acceleration``), and each
layer's numerical half measures the residual-sum-of-squares REDUCTION its term
buys when fit to a(r) (the α correction ~17%, the hidden term ~13% at the tuned
point). This isolates each force term's contribution with no IC/integrator
entanglement — the term-by-term, falsifiable oracle. (An earlier cut gated on the
position RMSE of partial-physics forward integration; that is physically wrong —
removing any term decoheres the whole orbit, so it cannot isolate a term. See the
NUMERICAL ORACLE note below and the SUMMARY deviation.) The simulator, not the
pattern match, makes every depth claim falsifiable (SCOPE §10 "validation is not
circular").

Layering (L-018): benchmarks-side, so it imports the simulator (config / types /
alien / tuning) and reuses ``benchmarks.scoring._safe_parse_expr`` +
``benchmarks.alien_audit._Q_DRIFT_BAR`` + ``simulator.tuning`` recovery idioms
with ZERO new import edges (the resolved AUDIT-LOCATION fork, alien_audit.py:29-34).

Threat-model mitigations: T-15.2-01 (post-hoc, off the seam — no UNKNOWN_SYSTEM
circularity), T-15.2-04 (no-LLM grep gate + sympy-only parser), T-15.2-05
(behavioral/numerical anti-gaming gates + per-layer negative controls), T-15.2-06
(``_safe_parse_expr`` empty-``__builtins__`` RCE guard, reused verbatim, never
sympify).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass

import sympy

from ascension.benchmarks.alien_audit import _Q_DRIFT_BAR
from ascension.benchmarks.scoring import _safe_parse_expr
from ascension.common.logging import get_logger
from ascension.simulator.alien import AlienUniverse
from ascension.simulator.types import AlienConfig

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Constants — the §10 layer signatures + the layer gate bars
# -----------------------------------------------------------------------------

# The observable alphabet: names an honest discoverer ALREADY sees in the
# ObservationBundle (positions/velocities/masses/time). Any free symbol OUTSIDE
# this set is a candidate HIDDEN per-body property (the L3 leap). r/r0 are radial
# observables; m1/m2 masses; x/y/v/t kinematics. NOT in here: s, s1, s2, kappa
# (hidden charge / its coupling) — those are the discovery targets.
_OBSERVABLE_NAMES: frozenset[str] = frozenset(
    {
        "r",
        "r0",
        "r1",
        "r2",
        "m1",
        "m2",
        "m3",
        "t",
        "x",
        "y",
        "z",
        "v",
        "v0",
        "v1",
        "v2",
        "v_perp",
        "vperp",
        "x0",
        "x1",
        "x2",
        "y0",
        "y1",
        "y2",
        # Phase 15.2 — the PINNED per-body velocity components the alien task
        # description now labels its data columns with (council/base.py). These are
        # OBSERVABLE kinematic quantities, NOT hidden per-body charges, so they must
        # be classified as observable (else _hidden_symbol_names would mistake a
        # Cartesian velocity component for the L3 hidden-charge leap).
        "vx0",
        "vy0",
        "vx1",
        "vy1",
        "vx2",
        "vy2",
    }
)

# The known physics CONSTANTS an honest discoverer can name without positing a
# hidden DOF (gravitational constant, the correction/coupling scalars). These are
# NOT hidden per-body properties even though they are "non-observable" in the raw
# data — so they must not be mistaken for the L3 hidden charge.
_KNOWN_CONSTANT_NAMES: frozenset[str] = frozenset({"G", "alpha", "beta", "gamma", "pi", "E", "e"})

# Hidden per-body symbols the truth uses (the s-family + the kappa coupling).
_HIDDEN_SYMBOL_NAMES: frozenset[str] = frozenset({"s", "s1", "s2", "kappa"})

# The pre-registered local_dict names (L-013: pre-register subscripted names so
# implicit-multiplication can't collapse s1->s*1 / x0->x*0). Superset of every
# name the alien alphabet + the truth law can use.
_LOCAL_SYMBOL_NAMES: tuple[str, ...] = (
    "r",
    "r0",
    "r1",
    "r2",
    "m1",
    "m2",
    "m3",
    "s",
    "s1",
    "s2",
    "G",
    "alpha",
    "beta",
    "gamma",
    "kappa",
    "t",
    "x",
    "y",
    "z",
    "v",
    "v0",
    "v1",
    "v2",
    "v_perp",
    "x0",
    "x1",
    "x2",
    "y0",
    "y1",
    "y2",
    "U",
    # Phase 15.2 — PINNED per-body velocity components (the alien task
    # description labels its columns with these). Pre-registering them as atomic
    # Symbols blocks the L-013 implicit-multiplication split (vx1 -> v*x1) so a
    # component-form hypothesis parses to the symbol the agent meant.
    "vx0",
    "vy0",
    "vx1",
    "vy1",
    "vx2",
    "vy2",
)

# NUMERICAL ORACLE — per-step radial-acceleration residual fit (the 15.1
# nested-model story, NOT full-trajectory position RMSE).
#
# ⚠️ DESIGN CORRECTION (this session, deviation Rule 1 — see SUMMARY): the first
# cut gated L1/L2/L3 on the POSITION RMSE of a partial-physics forward
# integration (Newton-only / Newton+α) vs the full truth orbit. That is
# physically wrong: the truth orbit is the joint product of ALL force terms, so
# REMOVING any term makes the trajectory diverge from truth regardless of which
# term survives (measured: Newton-only rel-RMSE ≈ 1.73, Newton+α ≈ 1.70 — both
# decohere; only the EXACT law gives RMSE→0; Newton+α is even slightly WORSE than
# Newton because α without the hidden term over-corrects). Position-RMSE of
# partial dynamics cannot isolate a single term's contribution.
#
# The correct, term-isolating, falsifiable oracle is the one Phase 15.1 built and
# tested (``tuning.nested_model_ftest`` / ``_observed_radial_acceleration``): a
# position-only discoverer recovers the per-step radial acceleration a(r) by
# central finite difference, then fits NESTED basis sets and measures the
# residual-sum-of-squares (RSS) REDUCTION each added term buys. This isolates the
# FORCE-law discrepancy with no IC/integrator entanglement (RESEARCH §the structure
# test; the alpha term buys ~17% RSS reduction over Newton, the hidden term ~13%
# over Newton+α — clean, non-vacuous, term-by-term). The simulator is still the
# oracle (the a(r) data comes from the locked AlienUniverse), so depth claims stay
# §10-"validation is not circular"-compliant.

# L2 RSS-reduction margin: adding the α-correction basis (−1/r**3.5) must reduce
# the Newton-only acceleration-fit RSS by at least this RELATIVE fraction over the
# held-out set. Measured at the tuned operating point: ~0.17 (17%). A 5% floor is
# conservative and non-vacuous (the negative control — a wrong second exponent —
# is rejected structurally before this gate runs).
_L2_RSS_REDUCTION_REL = 0.05

# L3 behavioral gate: adding the hidden-charge basis ((1−cos(γ_est r))/r**2) must
# reduce the Newton+α RSS by at least this RELATIVE fraction. Measured: ~0.13. A
# DECORATIVE s (coefficient ~0 in the proposal) is caught by the coefficient
# pre-filter BEFORE this gate — the proposal's own stated coefficient must be
# non-negligible, so a re-fit of a structurally-present-but-zero-coefficient term
# cannot launder it past the gate.
_L3_RSS_REDUCTION_REL = 0.05

# L4 structural gate: the proposal's STATED cos frequency (read from the
# symbolic_form literal) must be within this of the truth γ=0.7. The
# acceleration-residual fit does NOT cleanly re-identify γ (a free γ grid-fit
# lands at ~1.5 because the visited radii + FD noise under-determine it), so the
# AUTHORITATIVE frequency check is the proposal's stated γ, not a re-fit —
# the wrong-frequency negative control (cos(5r)) is rejected here. The hidden term
# must ALSO buy the L3-style RSS reduction (it does real work).
_L4_GAMMA_TOL = 0.1  # |gamma_est - 0.7| must be within this

# Heavy-sympy-op timeout (mirror scoring.py's ThreadPoolExecutor idiom) so a
# pathological expression can't hang the scorer.
_SYMPY_TIMEOUT_S = 5.0


# -----------------------------------------------------------------------------
# Regime selection (EXP-080) — V1 (product / r^-3.5) vs V2 (sum / r^-2.5).
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegimeSpec:
    """Which §10 alien LAW the scorer's L2/L3/L4 gates target (EXP-080 STRAT-03).

    The depth scorer's truth signatures are law-specific: the L2 correction band
    + fit basis center on a radial exponent, and the L3/L4 hidden-charge gates
    detect how the per-body charges enter. ``RegimeSpec`` carries the two knobs
    that the V2 regime moves OFF the V1-tuned bands so a single ``score_depth``
    serves both laws without forking:

      - ``correction_exponent``: the V1 law's correction is ``r^-3.5`` (default);
        the V2 law's is ``r^-2.5``. The L2 band centers on this exponent and the
        L2 fit basis is ``−1/r**correction_exponent``.
      - ``charge_coupling``: ``"product"`` (V1) detects the hidden charges as a
        PRODUCT ``s1*s2`` (``_has_hidden_product`` + the cross-config s-product
        gate); ``"sum"`` (V2) detects them as a SUM ``s1+s2`` (``_has_hidden_sum``
        + the cross-config s-SUM gate).

    The DEFAULT instance (``product`` / 3.5) reproduces the V1 scorer behavior
    BYTE-FOR-BYTE — ``score_depth(regime=None)`` and ``regime=RegimeSpec()`` take
    the identical V1 code path. V2 selects ``RegimeSpec(correction_exponent=2.5,
    charge_coupling="sum")``.
    """

    correction_exponent: float = 3.5
    charge_coupling: str = "product"


# The canonical V1 regime (the default everywhere — preserves the pre-EXP-080
# behavior). V2 callers build their own RegimeSpec.
_REGIME_V1 = RegimeSpec(correction_exponent=3.5, charge_coupling="product")


@dataclass(frozen=True, slots=True)
class DepthScore:
    """Per-layer discovery-depth result over a run's hypothesis set.

    Invariant: max_depth == (highest index i where per_layer[i] is True) + 1, or 0
    if none. ``per_layer`` is a length-6 tuple of bools (L1..L6) — a TUPLE, not a
    dict (L-020: downstream stats key on a stable ordered shape, not dict order).

    Mirrors the AlienAuditResult DTO discipline (frozen, slots, a grep-able
    docstring-literal invariant). Same input -> identical output: the scorer
    sorts its hypothesis iteration so the result is order-independent.

    Attributes:
      max_depth: Highest §10 layer reached across the hypothesis set (0..6).
      per_layer: Length-6 tuple of L1..L6 pass/fail.
      held_out_rmse: The held-out numerical-gate provenance value for the deepest
        credited radial layer — the RELATIVE acceleration-fit RSS reduction the
        L2/L3/L4 term bought over its nested base (None if no numerical gate ran).
        (Named ``held_out_rmse`` for continuity with the AlienAuditResult /
        DepthScore contract Plan 03/04 consume; it is the held-out RSS-reduction
        oracle, not a position RMSE — see the module NUMERICAL ORACLE note.)
      q_drift: The proposed-Q drift the L5 gate recorded (None if no L5 candidate).
      notes: Ordered, human-readable provenance (which predicate fired/skipped,
        any rationale-tiebreaker logged, parse skips) — never load-bearing.
    """

    max_depth: int
    per_layer: tuple[bool, ...]
    held_out_rmse: float | None
    q_drift: float | None
    notes: tuple[str, ...]


# -----------------------------------------------------------------------------
# Parsing + structural extraction
# -----------------------------------------------------------------------------


def _local_dict() -> dict[str, sympy.Symbol]:
    """Pre-registered sympy Symbols for the alien alphabet (L-013-safe).

    Binding every subscripted name (s1, s2, r0, m1, m2, x0...) as a real Symbol
    blocks implicit-multiplication from collapsing s1->s*1 / x0->x*0 (the L-013
    foundational bug). Mirrors the round_close/scoring locals discipline.
    """
    return {name: sympy.Symbol(name) for name in _LOCAL_SYMBOL_NAMES}


def _parse(symbolic_form: str) -> sympy.Expr | None:
    """Parse one symbolic_form via the ONLY parser (_safe_parse_expr).

    Never sympify, never an LLM (T-15.2-06 / T-15.2-04). A parse failure is
    logged + returns None (skip-not-fatal, [[engineering_discipline_no_coverups]]
    — loud, never a silent crash).
    """
    try:
        return _safe_parse_expr(symbolic_form, local_dict=_local_dict())
    except Exception as exc:  # noqa: BLE001 — surfaced loud, then skipped
        logger.warning(
            "alien_depth: could not parse symbolic_form %r (%s: %s); skipped",
            symbolic_form,
            type(exc).__name__,
            exc,
        )
        return None


def _r_power_exponents(expr: sympy.Expr) -> list[float]:
    """Return the set of r-power exponents present in ``expr`` (as a poly in 1/r).

    Walks the additive terms and, for each, reads the exponent of ``r`` (positive
    or negative). Used by L1 (is there a -2 term?) and L2 (is there a SECOND
    radial power-law term, ideally ~ -3.5?). Robust to products: a term like
    ``G*m1*m2/r**2`` contributes exponent -2.
    """
    r = sympy.Symbol("r")
    exps: list[float] = []
    expanded = sympy.expand(expr)
    terms = expanded.as_ordered_terms()
    for term in terms:
        # Degree of r in this term (handles r**n, 1/r**n, and bare r).
        try:
            deg = sympy.degree(term, gen=r)
            exps.append(float(deg))
            continue
        except (sympy.PolynomialError, TypeError):
            pass
        # Fall back: inspect Pow factors of r directly (covers fractional /
        # negative exponents like r**(-3.5) that sympy.degree rejects).
        e = _radial_exponent_of_term(term, r)
        if e is not None:
            exps.append(e)
    return exps


def _radial_exponent_of_term(term: sympy.Expr, r: sympy.Symbol) -> float | None:
    """Net exponent of ``r`` in a single multiplicative term, or None if absent."""
    total = 0.0
    found = False
    factors = term.as_ordered_factors() if term.is_Mul else [term]
    for f in factors:
        base, exp = f.as_base_exp()
        if base == r:
            try:
                total += float(exp)
                found = True
            except (TypeError, ValueError):
                return None
        elif f == r:
            total += 1.0
            found = True
    return total if found else None


def _free_symbol_names(expr: sympy.Expr) -> set[str]:
    """Names of free symbols in ``expr``."""
    return {str(s) for s in expr.free_symbols}


def _hidden_symbol_names(expr: sympy.Expr) -> set[str]:
    """Free symbols that are HIDDEN per-body properties (not observable/constant).

    The L3 leap: a name that is neither an observable (r/m/v/t/x...) nor a known
    physics constant (G/alpha/beta/gamma) — i.e. an UNOBSERVED per-body scalar the
    discoverer had to POSIT. kappa is treated as a hidden coupling for L5.
    """
    names = _free_symbol_names(expr)
    return {n for n in names if n not in _OBSERVABLE_NAMES and n not in _KNOWN_CONSTANT_NAMES}


def _has_hidden_product(expr: sympy.Expr) -> bool:
    """True iff a hidden symbol enters as a PRODUCT (s1*s2 or s**2 self-coupling).

    The L3 structural half (15.2 RESEARCH): the hidden DOF must enter as a
    per-body PRODUCT, not a lone additive nuisance term. We detect this by
    checking, term-by-term, whether the multiplicative factors include either two
    distinct hidden symbols (s1*s2) or a hidden symbol raised to an even power
    >= 2 (s**2).
    """
    hidden = _HIDDEN_SYMBOL_NAMES
    for term in sympy.expand(expr).as_ordered_terms():
        factors = term.as_ordered_factors() if term.is_Mul else [term]
        hidden_in_term: list[tuple[str, float]] = []
        for f in factors:
            base, exp = f.as_base_exp()
            name = str(base)
            if name in hidden:
                try:
                    hidden_in_term.append((name, float(exp)))
                except (TypeError, ValueError):
                    hidden_in_term.append((name, 1.0))
        if len(hidden_in_term) >= 2:
            # two distinct hidden factors (s1 * s2)
            distinct = {n for n, _ in hidden_in_term}
            if len(distinct) >= 2:
                return True
        for _name, exp in hidden_in_term:
            if exp >= 2:  # self-coupling s**2
                return True
    return False


def _hidden_symbols_in_term(term: sympy.Expr) -> list[tuple[str, float]]:
    """The (name, exponent) of each hidden symbol in a single multiplicative term."""
    hidden = _HIDDEN_SYMBOL_NAMES
    factors = term.as_ordered_factors() if term.is_Mul else [term]
    out: list[tuple[str, float]] = []
    for f in factors:
        base, exp = f.as_base_exp()
        name = str(base)
        if name in hidden:
            try:
                out.append((name, float(exp)))
            except (TypeError, ValueError):
                out.append((name, 1.0))
    return out


def _radial_structure_key(term: sympy.Expr) -> str:
    """A canonical string of a term's NON-hidden (radial/observable) structure.

    Strips the hidden symbols (s/s1/s2) and the leading numeric coefficient from a
    term so two terms ``s1*g(r)`` and ``s2*g(r)`` map to the SAME key — the
    signature the SUM detector keys on (same radial envelope, different single
    hidden symbol). Built from the term divided by its hidden symbols + its
    numeric coefficient.
    """
    hidden_syms = [sympy.Symbol(n) for n in _HIDDEN_SYMBOL_NAMES if sympy.Symbol(n) in term.free_symbols]
    stripped = term
    for s in hidden_syms:
        # divide out each hidden symbol to its power present in the term
        stripped = sympy.simplify(stripped / s ** _term_symbol_power(term, s))
    # drop the leading numeric coefficient so amplitudes don't split the key
    coeff, rest = stripped.as_coeff_Mul()
    return sympy.srepr(sympy.expand(rest))


def _term_symbol_power(term: sympy.Expr, sym: sympy.Symbol) -> float:
    """Net power of ``sym`` in a multiplicative term (0 if absent)."""
    e = _radial_exponent_of_term(term, sym)
    return e if e is not None else 0.0


def _has_hidden_sum(expr: sympy.Expr) -> bool:
    """True iff two DISTINCT hidden symbols enter ADDITIVELY as the amplitude.

    The L3 SUM-confound structural half (EXP-080 V2 regime): the V2 hidden DOF
    enters as ``(s1 + s2)·g(r)``, the ADDITIVE analog of the V1 product
    ``s1*s2·g(r)``. After ``sympy.expand``, ``(s1+s2)·g`` becomes two SEPARATE
    additive terms ``s1·g + s2·g``, each carrying exactly ONE hidden symbol
    (linear, exponent 1) and sharing the SAME radial/observable structure ``g``.
    We detect this signature: >= 2 additive terms, each with exactly one distinct
    linear hidden symbol, that share a common radial-structure key. A lone
    additive ``s1`` nuisance (no matching second symbol on the same structure)
    does NOT qualify, and a product ``s1*s2`` term (two hidden symbols in ONE
    term) is the V1 case, not this one.
    """
    # Map each radial-structure key -> set of distinct single-hidden-symbol names
    # that appear as a LINEAR amplitude on that structure.
    structure_to_syms: dict[str, set[str]] = {}
    for term in sympy.expand(expr).as_ordered_terms():
        hidden_in_term = _hidden_symbols_in_term(term)
        # Exactly one hidden symbol, entering linearly (exp ~ 1) — the additive
        # single-charge amplitude. (Two hidden symbols in one term = product = V1.)
        if len(hidden_in_term) != 1:
            continue
        name, exp = hidden_in_term[0]
        if abs(exp - 1.0) > 1e-9:
            continue
        try:
            key = _radial_structure_key(term)
        except (TypeError, ValueError, ZeroDivisionError):  # pragma: no cover - defensive
            continue
        structure_to_syms.setdefault(key, set()).add(name)
    # A genuine sum confound: some shared radial structure carries >= 2 distinct
    # single hidden symbols additively (s1·g + s2·g).
    return any(len(syms) >= 2 for syms in structure_to_syms.values())


def _term_coefficient_is_nonneglible(expr: sympy.Expr, has_hidden_fn) -> bool:
    """Heuristic: is the hidden-bearing term's coefficient appreciably nonzero?

    Cheap structural pre-filter for the obvious decorative-s case (a 1e-12
    coefficient). The AUTHORITATIVE anti-gaming check is the behavioral RMSE gate;
    this only avoids paying for the numerical gate on a term that is structurally
    ~0. Returns True unless every hidden-bearing term has a tiny numeric prefactor.
    """
    max_coeff = 0.0
    saw_hidden_term = False
    for term in sympy.expand(expr).as_ordered_terms():
        if not has_hidden_fn(term):
            continue
        saw_hidden_term = True
        # numeric prefactor of the term (constant multiplicative coefficient)
        coeff = term.as_coeff_Mul()[0]
        try:
            max_coeff = max(max_coeff, abs(float(coeff)))
        except (TypeError, ValueError):
            return True  # symbolic coeff — can't dismiss; let the gate decide
    if not saw_hidden_term:
        return False
    return max_coeff > 1e-6


# -----------------------------------------------------------------------------
# Numerical oracle — per-step radial-acceleration residual fit (the 15.1 story)
# -----------------------------------------------------------------------------


# Deterministic memo of the recovered (r, a_radial) over a held-out set. Each is a
# single fixed integration + a central FD per held-out IC; the SAME held-out set
# recurs across every hypothesis and every test. Caching on the held_out_ics
# identity preserves determinism (same input -> identical output) and cuts the
# repeated integration cost. It memoizes a PURE deterministic function of the
# locked simulator — NOT the public entry point (which would mask a re-run
# regression).
# BUG-02 (AUDIT-036): key the oracle memos on the held-out set's CONTENT (a tuple
# of frozen-hashable AlienConfig), not id(). id() is content-blind — a GC'd tuple
# whose address is reused would return another set's cached accel/drift. Bounded so
# a long-lived process can't grow it without limit (the distinct held-out sets are few).
_MEMO_MAX = 256
_ACCEL_MEMO: dict[tuple[AlienConfig, ...], tuple] = {}


def _observed_accel(held_out_ics: tuple[AlienConfig, ...]):
    """Return concatenated (r, a_radial) recovered over the held-out set.

    Reuses ``tuning._observed_radial_acceleration`` (the 15.1 PySINDy-posture
    central-FD recovery a position-only discoverer would compute) on each held-out
    IC and concatenates — so the residual fits run on the WHOLE held-out span
    (the non-vacuity guarantee: a layer can't pass on one orbit shape). Memoized.
    """
    import numpy as np

    from ascension.simulator.tuning import _observed_radial_acceleration

    key = held_out_ics
    cached = _ACCEL_MEMO.get(key)
    if cached is not None:
        return cached
    rs: list = []
    ars: list = []
    for cfg in held_out_ics:
        r, a = _observed_radial_acceleration(cfg)
        rs.append(r)
        ars.append(a)
    r_all = np.concatenate(rs)
    a_all = np.concatenate(ars)
    if len(_ACCEL_MEMO) >= _MEMO_MAX:
        _ACCEL_MEMO.clear()
    _ACCEL_MEMO[key] = (r_all, a_all)
    return r_all, a_all


def _rss_fit(a_radial, columns) -> float:
    """Residual sum of squares of a least-squares fit of a_radial to ``columns``.

    Mirror of ``tuning._rss`` — a free-coefficient linear fit of the recovered
    radial acceleration to the proposed basis set. The RSS REDUCTION an added
    basis column buys is each layer's term-isolating numerical signal.
    """
    import numpy as np

    X = np.vstack(columns).T
    c, _res, _rank, _sv = np.linalg.lstsq(X, a_radial, rcond=None)
    return float(np.sum((a_radial - X @ c) ** 2))


def _newton_basis(r):
    return -1.0 / r**2


def _alpha_basis(r):
    return -1.0 / r**3.5


def _corr_basis(r, correction_exponent: float = 3.5):
    """The L2 correction fit basis ``−1/r**p`` for a regime's correction exponent.

    Generalizes ``_alpha_basis`` (which is exactly ``_corr_basis`` at p=3.5, the
    V1 default — byte-identical) so the V2 regime can fit its ``r^-2.5`` correction
    with ``_corr_basis(r, 2.5)``. The MINUS makes it attractive (same convention
    as Newton/alpha), matching the simulator's correction force sign.
    """
    return -1.0 / r**correction_exponent


def _hidden_basis(r, gamma: float):
    import numpy as np

    return (1.0 - np.cos(gamma * r)) / r**2


def _rss_reduction(
    held_out_ics: tuple[AlienConfig, ...],
    base_columns_fn,
    added_column_fn,
) -> float:
    """Relative RSS reduction adding ``added`` to the ``base`` acceleration fit.

    Returns ``(RSS_base − RSS_base+added) / RSS_base`` over the held-out span.
    The term-isolating numerical gate: a basis column that explains real force
    structure drops the RSS materially; a vacuous / wrong-structure column does
    not. >= 0 by the lstsq nesting (a superset basis can't fit worse), so a
    near-0 reduction means the added term does no work.
    """
    r, a = _observed_accel(held_out_ics)
    base_cols = base_columns_fn(r)
    rss_base = _rss_fit(a, base_cols)
    if rss_base <= 0.0:
        return 0.0
    rss_aug = _rss_fit(a, base_cols + [added_column_fn(r)])
    return (rss_base - rss_aug) / rss_base


def _cross_config_sproduct_reduction(scoring_cfgs: tuple[AlienConfig, ...], gamma: float) -> float:
    """Relative RSS the PER-CONFIG charge-product scaling buys BEYOND a single
    GLOBAL amplitude, fitting the hidden basis across multiple charge configs.

    The §10 cross-config L3/L4 gate (Phase 19-04 / EXP-076). A per-body charge is
    DEGENERATE with a global amplitude on ONE config (s1*s2 absorbed into beta —
    Session 029), but across a FAMILY at DISTINCT charge products the oscillatory
    amplitude varies as ``beta*product_c`` — a signal ONLY a per-config product
    scaling can fit. For each config c (charge product ``p_c``) recover a(r_c) and
    build shared Newton + alpha columns; the hidden column is fit two ways:

      GLOBAL    : ``hidden_basis(r_c)``        — one amplitude for EVERY config;
      PER-CONFIG: ``p_c * hidden_basis(r_c)``  — the oracle-known product scales it.

    Returns ``(RSS_global - RSS_perconfig)/RSS_global`` over the concatenated span.
    Distinct products => the global fit cannot match the varying amplitude => a
    large reduction (the s1*s2 signal is real). Equal products (one config, or all
    equal) => the two columns are scalar multiples => reduction ~0, so a global-
    beta proposal cannot game L3 and the equal-charge control ceilings at depth 2.

    Deterministic, $0, NO LLM: the a(r) oracle is the locked simulator via
    ``tuning._observed_radial_acceleration`` (the SAME recovery the single-config
    gate uses). Requires >= 2 configs at DISTINCT products (else returns 0.0).
    """
    import numpy as np

    from ascension.simulator.tuning import _observed_radial_acceleration

    if len(scoring_cfgs) < 2:
        return 0.0
    a_parts: list = []
    newton_parts: list = []
    alpha_parts: list = []
    hidden_global_parts: list = []
    hidden_perconfig_parts: list = []
    products: list[float] = []
    for cfg in scoring_cfgs:
        r, a = _observed_radial_acceleration(cfg)
        if r.size == 0:
            continue
        p = float(cfg.charges[0]) * float(cfg.charges[1])
        products.append(p)
        a_parts.append(a)
        newton_parts.append(_newton_basis(r))
        alpha_parts.append(_alpha_basis(r))
        hb = _hidden_basis(r, gamma)
        hidden_global_parts.append(hb)
        hidden_perconfig_parts.append(p * hb)
    # Need >= 2 configs at DISTINCT products for genuine cross-config variation.
    if len(a_parts) < 2 or len({round(p, 9) for p in products}) < 2:
        return 0.0
    a_all = np.concatenate(a_parts)
    newton_all = np.concatenate(newton_parts)
    alpha_all = np.concatenate(alpha_parts)
    rss_global = _rss_fit(a_all, [newton_all, alpha_all, np.concatenate(hidden_global_parts)])
    rss_perconfig = _rss_fit(a_all, [newton_all, alpha_all, np.concatenate(hidden_perconfig_parts)])
    if rss_global <= 0.0:
        return 0.0
    return (rss_global - rss_perconfig) / rss_global


def _cross_config_ssum_reduction(
    scoring_cfgs: tuple[AlienConfig, ...],
    gamma: float,
    correction_exponent: float = 3.5,
) -> float:
    """The SUM analog of ``_cross_config_sproduct_reduction`` (EXP-080 V2 regime).

    Identical machinery, but the per-config oscillatory amplitude varies as
    ``beta*(s1+s2)_c`` (the V2 sum coupling) instead of ``beta*(s1*s2)_c``. Across
    a V2 family at DISTINCT charge SUMS, only a per-config SUM scaling can fit the
    varying amplitude; a single global amplitude cannot, so the relative RSS
    reduction is large ONLY when the sum coupling is real and the family spans
    distinct sums. An equal-sum family yields ~0 (not gameable). The correction
    (alpha) column uses ``_corr_basis(r, correction_exponent)`` so the V2 r^-2.5
    correction is the nested base, not the V1 r^-3.5.

    Deterministic, $0, NO LLM: the a(r) oracle is the locked simulator via
    ``tuning._observed_radial_acceleration``. Requires >= 2 configs at DISTINCT
    sums (else returns 0.0).
    """
    import numpy as np

    from ascension.simulator.tuning import _observed_radial_acceleration

    if len(scoring_cfgs) < 2:
        return 0.0
    a_parts: list = []
    newton_parts: list = []
    alpha_parts: list = []
    hidden_global_parts: list = []
    hidden_perconfig_parts: list = []
    sums: list[float] = []
    for cfg in scoring_cfgs:
        r, a = _observed_radial_acceleration(cfg)
        if r.size == 0:
            continue
        s = float(cfg.charges[0]) + float(cfg.charges[1])  # the SUM (not product)
        sums.append(s)
        a_parts.append(a)
        newton_parts.append(_newton_basis(r))
        alpha_parts.append(_corr_basis(r, correction_exponent))  # V2 correction basis
        hb = _hidden_basis(r, gamma)
        hidden_global_parts.append(hb)
        hidden_perconfig_parts.append(s * hb)  # per-config SUM scaling
    # Need >= 2 configs at DISTINCT sums for genuine cross-config variation.
    if len(a_parts) < 2 or len({round(s, 9) for s in sums}) < 2:
        return 0.0
    a_all = np.concatenate(a_parts)
    newton_all = np.concatenate(newton_parts)
    alpha_all = np.concatenate(alpha_parts)
    rss_global = _rss_fit(a_all, [newton_all, alpha_all, np.concatenate(hidden_global_parts)])
    rss_perconfig = _rss_fit(a_all, [newton_all, alpha_all, np.concatenate(hidden_perconfig_parts)])
    if rss_global <= 0.0:
        return 0.0
    return (rss_global - rss_perconfig) / rss_global


def _newton_fit_converges(held_out_ics: tuple[AlienConfig, ...]) -> bool:
    """L1 loose numerical gate: a finite, non-degenerate Newton (−1/r**2) fit.

    L1 alone misses the α + hidden terms, so the §10 walkthrough credits it to a
    curve-fittable inverse-square dominant force — NOT an exact reproduction. The
    honest loose gate: the recovered acceleration is finite and the single-basis
    Newton fit is well-posed (the dominant force IS ~1/r**2). A divergent / empty
    recovery fails.
    """
    import numpy as np

    r, a = _observed_accel(held_out_ics)
    if r.size < 4 or not (np.all(np.isfinite(r)) and np.all(np.isfinite(a))):
        return False
    rss_newton = _rss_fit(a, [_newton_basis(r)])
    rms_signal = float(np.sqrt(np.mean(a**2)))
    if rms_signal <= 0.0:
        return False
    # The Newton basis must explain the BULK of the signal (the dominant force is
    # inverse-square): relative residual well below 1 (measured ~0.065 at the
    # tuned point). A loose 0.5 bar credits a dominant-inverse-square fit while
    # still being non-vacuous.
    rel_resid = float(np.sqrt(rss_newton / r.size)) / rms_signal
    return rel_resid < 0.5


# Memo for the per-held-out-set truth Q drift (a single fixed integration).
_QDRIFT_MEMO: dict[tuple[AlienConfig, ...], float] = {}


# -----------------------------------------------------------------------------
# L4 — hidden-force functional-form match (cos frequency extraction)
# -----------------------------------------------------------------------------


# The truth periodic length-scale, named by the symbol ``gamma`` in the §10 law.
_TRUTH_GAMMA = 0.7


def _extract_cos_frequency(expr: sympy.Expr) -> float | None:
    """Return the k in a ``cos(k*r)`` factor of a hidden-bearing term, else None.

    L4 structural half: the truth hidden term carries ``(1 - cos(gamma*r))``. We
    look for a ``cos(Wild*r)`` inside any hidden-symbol-bearing additive term and
    read the constant multiplying r:
      - a NUMERIC literal (e.g. cos(5.0*r)) returns that number — the
        wrong-frequency negative control is caught here when it is far from 0.7;
      - the bare SYMBOL ``gamma`` (the proposal naming the truth coupling, e.g.
        cos(gamma*r)) returns the truth gamma 0.7 — the agent named the §10
        length-scale rather than guessing a number, which IS the correct form;
      - any OTHER symbolic frequency returns None (an unidentified coupling — not
        creditable as the truth form).
    The numeric/symbolic split keeps the gate honest: a wrong NUMBER fails, but
    naming the truth coupling symbol is the L4-correct expression.
    """
    r = sympy.Symbol("r")
    gamma_sym = sympy.Symbol("gamma")
    k_wild = sympy.Wild("k", exclude=[r])
    for term in sympy.expand(expr).as_ordered_terms():
        if not _has_hidden_product(term) and not _term_has_hidden(term):
            continue
        for sub in term.atoms(sympy.cos):
            arg = sub.args[0]
            m = arg.match(k_wild * r)
            if m is not None and k_wild in m:
                k = m[k_wild]
                if k == gamma_sym:
                    return _TRUTH_GAMMA  # named the truth coupling symbol
                try:
                    return float(k)
                except (TypeError, ValueError):
                    return None  # an unidentified symbolic coupling
            # arg could be exactly r (k=1)
            if arg == r:
                return 1.0
    return None


def _term_has_hidden(term: sympy.Expr) -> bool:
    """True iff this single term carries any hidden symbol (s/s1/s2/kappa)."""
    return bool(_free_symbol_names(term) & _HIDDEN_SYMBOL_NAMES)


def _has_inverse_square_envelope_on_hidden(expr: sympy.Expr) -> bool:
    """True iff a hidden-bearing term has a ~1/r**2 envelope (the L4 form)."""
    r = sympy.Symbol("r")
    for term in sympy.expand(expr).as_ordered_terms():
        if not _term_has_hidden(term):
            continue
        e = _radial_exponent_of_term(term, r)
        if e is not None and abs(e - (-2.0)) < 0.25:
            return True
    return False


# -----------------------------------------------------------------------------
# L5 — conservation: proposed-Q drift on a fresh held-out trajectory
# -----------------------------------------------------------------------------


def _is_conservation_candidate(expr: sympy.Expr) -> bool:
    """Structural L5 half: a CONSERVED ENERGY-like quantity Q (T + U + coupling).

    A genuine conserved quantity Q is ``kinetic(T) + potential(U) +
    hidden-charge-squared velocity coupling`` (Q = T + U + kappa*s**2*v_perp**2,
    SCOPE §10 L5). We require ALL of: a velocity symbol, the potential symbol
    ``U``, a ``v**2``-style kinetic factor, AND a term in which a velocity factor
    is coupled to a GENUINE hidden-charge product (s1*s2 or s**2).

    Phase 15.2 TIER2-03 fix (false-positive caught in the live run): the prior
    test credited "coupled" on bare ``kappa`` and did not require ``U``, so the
    FORCE LAW ``G*m2/r**2 - kappa*v1**2`` — the acceleration RHS the prompt
    actually asks for — wrongly matched as a conserved quantity and scored L5.
    A force law is NOT a conserved quantity: it has no potential ``U`` term and
    no hidden charge. Requiring ``U`` + a hidden-charge product distinguishes a
    genuine Q proposal from any force law carrying a kappa*v**2 term.

    NOTE (elicitation): the current Council prompt asks ONLY for the force law
    (acceleration RHS), never for a conserved quantity, so L5 is NOT honestly
    elicited — the honest discovery-depth metric for this prompt is L1-L4 (see
    module docstring). This predicate stays correct for the day a prompt DOES
    ask for a conserved quantity; it just refuses to be tricked by a force law.
    """
    names = _free_symbol_names(expr)
    has_velocity = bool(names & {"v", "v0", "v1", "v2", "v_perp", "vperp"})
    if not has_velocity:
        return False
    # A conserved energy-like quantity carries the potential symbol U; a force
    # law (acceleration RHS) does not. This is the primary force-law discriminator.
    if "U" not in names:
        return False
    # kinetic term: some velocity symbol squared
    has_kinetic = False
    has_coupled = False
    for term in sympy.expand(expr).as_ordered_terms():
        tnames = _free_symbol_names(term)
        vel_in_term = tnames & {"v", "v0", "v1", "v2", "v_perp", "vperp"}
        if not vel_in_term:
            continue
        # squared velocity?
        for vname in vel_in_term:
            e = _radial_exponent_of_term(term, sympy.Symbol(vname))
            if e is not None and e >= 2:
                has_kinetic = True
        # velocity coupled to a GENUINE hidden-charge product (s1*s2 or s**2).
        # Bare ``kappa`` is NOT sufficient — a force law can carry a kappa*v**2
        # term without being a conserved quantity (the TIER2-03 false positive).
        if _has_hidden_product(term):
            has_coupled = True
    return has_kinetic and has_coupled


def _proposed_q_drift(held_out_ics: tuple[AlienConfig, ...]) -> float:
    """Drift of the TRUE conserved Q along a fresh held-out trajectory (traj.Q).

    L5's numerical oracle: a genuinely-conserved quantity has bounded drift. We
    read ``traj.Q`` (the polar-Hamiltonian invariant — NOT physics.conserved_Q of
    reconstructed Cartesian state, gotcha 8) on the first held-out run; the truth
    Q sits near the _Q_DRIFT_BAR (1e-6) scale. A drifting proposal fails the gate.
    """
    import numpy as np

    key = held_out_ics
    cached = _QDRIFT_MEMO.get(key)
    if cached is not None:
        return cached
    traj = AlienUniverse(held_out_ics[0]).run()
    q = np.asarray(traj.Q, dtype=np.float64)
    drift = float(np.max(np.abs(q - q[0])))
    if len(_QDRIFT_MEMO) >= _MEMO_MAX:
        _QDRIFT_MEMO.clear()
    _QDRIFT_MEMO[key] = drift
    return drift


def _claimed_q_drifts(expr: sympy.Expr, held_out_ics: tuple[AlienConfig, ...]) -> bool:
    """True iff the proposed conserved quantity would DRIFT on a held-out orbit.

    The honest test: a structurally-complete proposed Q (kinetic + U +
    kappa*s**2*v_perp**2) matches the truth invariant -> bounded drift (passes);
    a structurally-incomplete one (missing the tangential-inertia augmentation)
    drifts. Since the proposed Q is symbolic, we map "structurally complete" to
    "the truth Q is conserved on this orbit" (drift < bar) and "incomplete" to a
    drift > bar by checking whether the candidate carries the kappa*s**2 coupling.
    """
    # If the candidate is NOT a full conservation candidate, it drifts by fiat.
    if not _is_conservation_candidate(expr):
        return True
    # A complete candidate matches the truth invariant: check the truth Q drift.
    truth_drift = _proposed_q_drift(held_out_ics)
    return truth_drift > _Q_DRIFT_BAR


# -----------------------------------------------------------------------------
# L6 — parity-charge symmetry verification (sympy substitution)
# -----------------------------------------------------------------------------


def _is_parity_charge_invariant(expr: sympy.Expr) -> bool:
    """Verify the proposed LAW is invariant under P:(x->-x, v->-v, s->-s).

    We don't trust the claim; we VERIFY it. The radial force law is written in
    r (= |x|, EVEN under x->-x, so r->r) and the hidden charges s1/s2. Under the
    parity-charge transform: r is invariant; each hidden charge flips sign
    (s_k -> -s_k). A genuinely-invariant law must be UNCHANGED, which (for the
    radial-force representation) requires the hidden charges to enter as an EVEN
    product (s1*s2 or s**2) — a lone odd s breaks it. We substitute s_k -> -s_k
    and assert sympy simplifies the difference to 0 (timeout-wrapped).
    """
    hidden_syms = [
        sympy.Symbol(n) for n in ("s", "s1", "s2") if sympy.Symbol(n) in expr.free_symbols
    ]
    if not hidden_syms:
        # No hidden charge at all -> a purely radial law IS trivially invariant
        # under the charge flip, but L6 is the HIDDEN symmetry (charge sign): a
        # law with no hidden DOF does not demonstrate the parity-CHARGE symmetry.
        return False
    subs = {s: -s for s in hidden_syms}
    transformed = expr.xreplace(subs)
    diff = expr - transformed

    def _job(d: sympy.Expr = diff) -> bool:
        return bool(sympy.simplify(d) == 0)

    with ThreadPoolExecutor(max_workers=1) as ex:
        try:
            return ex.submit(_job).result(timeout=_SYMPY_TIMEOUT_S)
        except FuturesTimeout:
            logger.warning("alien_depth: L6 invariance simplify timed out; not credited")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("alien_depth: L6 invariance check raised %s; not credited", exc)
            return False


# Fixed keyword sets — rationale CORROBORATION only (logged tiebreaker, never
# authoritative, §22.6). The symbolic+numerical predicate decides every layer.
_L3_RATIONALE_KEYWORDS = ("hidden", "unobserved", "charge", "per-body")
_L6_RATIONALE_KEYWORDS = ("parity", "symmetry", "symmetric", "charge sign", "invariant")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def score_depth(
    hypotheses: list,
    alien_truth_cfg: AlienConfig,
    held_out_ics: tuple[AlienConfig, ...],
    *,
    family_scoring_cfgs: tuple[AlienConfig, ...] | None = None,
    regime: RegimeSpec | None = None,
) -> DepthScore:
    """Score the max §10 discovery layer reached across a run's hypothesis set.

    DETERMINISTIC (same input -> identical output): the hypothesis iteration is
    sorted by symbolic_form string, so the result is order-independent. $0 LLM —
    the only parser is ``_safe_parse_expr``; rationale is a logged tiebreaker,
    never authoritative (§22.6). Runs POST-HOC over a run's hypothesis nodes, NOT
    through the round-close/distillation seam (T-15.2-01).

    Args:
      hypotheses: the run's final Council hypotheses. Each is either a dict row
        (``content`` JSONB: ``symbolic_form`` / ``rationale``) or a
        ``HypothesisContent``-like object exposing those attributes.
      alien_truth_cfg: the locked alien truth config (couplings the layers match;
        carried for provenance + the symbolic-grad anchor).
      held_out_ics: the held-out generalization set (``TIER2_HELD_OUT_ICS``, all
        at the product-1 operating point) the L1/L2/L5 numerical gates run each
        proposed model against. UNCHANGED in family mode — L1/L2 are
        charge-independent and calibrated here, so they always run on the
        product-1 set (running them on high-charge-product data would inflate the
        Newton residual and push L2's relative alpha-reduction below its bar).
      family_scoring_cfgs: when provided (Phase 19-04 family scoring), the L3/L4
        behavioral gate fits ACROSS these distinct-charge configs and credits the
        hidden charge ONLY when per-config charge scaling (PRODUCT for V1, SUM for
        V2 — selected by ``regime``) reduces RSS by >= _L3_RSS_REDUCTION_REL
        BEYOND a single global amplitude (so a global-beta proposal cannot game L3,
        and a degenerate equal-amplitude set ceilings at depth 2). None (default) =
        the single-config behavior: L3/L4 use the in-place _rss_reduction over
        held_out_ics (byte-unchanged; the equal-charge control).
      regime: which §10 alien LAW the L2/L3/L4 gates target (EXP-080 STRAT-03).
        None (default) or ``RegimeSpec()`` = the V1 law (product / r^-3.5) — the
        scorer behaves BYTE-FOR-BYTE as before. ``RegimeSpec(correction_exponent=
        2.5, charge_coupling="sum")`` = the V2 law: L2 centers its band + fit basis
        on r^-2.5, and L3/L4 detect the hidden charges as a SUM (s1+s2) via
        ``_has_hidden_sum`` + the cross-config s-SUM gate. The V1 path is unchanged
        regardless of this arg's presence (the default reproduces the old literals).

    Returns:
      ``DepthScore`` — per-layer L1..L6 pass/fail + ``max_depth`` (0..6) +
      ``held_out_rmse`` / ``q_drift`` provenance + ordered ``notes``.
    """
    spec = regime if regime is not None else _REGIME_V1
    is_sum = spec.charge_coupling == "sum"
    corr_exp = spec.correction_exponent
    # The per-EXPR hidden-amplitude detector for this regime: product (s1*s2) for
    # V1, sum (s1+s2) for V2. Used by the L3/L4 structural gates.
    has_hidden_amp = _has_hidden_sum if is_sum else _has_hidden_product
    start_ns = time.perf_counter_ns()
    notes: list[str] = []

    # --- Extract + parse (deterministic, sorted, skip-not-drop) -------------
    forms: list[str] = []
    for h in hypotheses:
        sf = _extract_symbolic_form(h)
        if not sf:
            logger.warning("alien_depth: hypothesis has no symbolic_form; skipped")
            notes.append("skipped: missing symbolic_form")
            continue
        forms.append(sf)
    forms.sort()  # sorted iteration -> order-independent output (L-020)

    parsed: list[tuple[str, sympy.Expr]] = []
    for sf in forms:
        expr = _parse(sf)
        if expr is None:
            notes.append(f"skipped: unparseable {sf!r}")
            continue
        parsed.append((sf, expr))
    # rationale text in the same sorted order (for the L3/L6 tiebreaker logging)
    rationale_by_form: dict[str, str] = {}
    for h in hypotheses:
        sf = _extract_symbolic_form(h)
        if sf:
            rationale_by_form[sf] = _extract_rationale(h)

    per_layer = [False] * 6  # L1..L6
    held_out_rmse: float | None = None
    # BUG-03 (AUDIT-036): held_out_rmse is the numerical-gate provenance of the
    # DEEPEST credited radial layer (per the DepthScore docstring), NOT last-writer
    # across the per-hypothesis loop. Track the deepest layer that set it so a
    # shallower layer (or a later hypothesis) can never clobber a deeper value.
    _deepest_rmse_layer = -1
    q_drift: float | None = None

    if not parsed:
        elapsed_ms = max(0, (time.perf_counter_ns() - start_ns) // 1_000_000)
        logger.info("alien_depth: no parseable hypotheses; max_depth=0")
        return DepthScore(
            max_depth=0,
            per_layer=tuple(per_layer),
            held_out_rmse=None,
            q_drift=None,
            notes=tuple(notes) or ("no parseable hypotheses",),
        )

    # --- Evaluate the six predicates over the parsed set --------------------
    for sf, expr in parsed:
        exps = _r_power_exponents(expr)

        # L1 — dominant inverse-power force (~ c/r**2) + loose numerical gate.
        if any(abs(e - (-2.0)) < 0.25 for e in exps):
            # Numerical gate (LOOSE): the recovered acceleration is well-posed and
            # the dominant force IS inverse-square (a finite, non-degenerate
            # Newton fit). L1 alone misses alpha + hidden, so this is a loose bar,
            # not an exact reproduction.
            if _newton_fit_converges(held_out_ics):
                per_layer[0] = True
                notes.append(f"L1: inverse-square term + Newton accel-fit converges ({sf!r})")

        # L2 — correction term (SECOND radial power-law, ~ -corr_exp) + RSS-
        # reduction. The band center + fit basis are regime-parameterized:
        # V1 -corr_exp=-3.5 (byte-identical), V2 -2.5.
        #
        # L-064/DET-04: the Newton carve-out must not swallow the regime band's
        # lower edge. With a fixed 0.25 tolerance the V2 band |e+2.5|<0.3 was
        # truncated to an effective [2.25, 2.8] (the paper states 2.2–2.8), and
        # exponents in [2.0, 2.25) — including the corrected battery's own 2.1
        # recommendation — produced NO L2 evaluation at all. The tolerance now
        # shrinks to the gap between Newton and the band edge (V1 unchanged:
        # min(0.25, 3.5-0.3-2.0)=0.25). Recorded EXP-080/081 were scored under
        # the fixed 0.25 carve-out; the paper discloses the effective band.
        newton_tol = min(0.25, max(0.05, (corr_exp - 0.3) - 2.0))
        second_terms = [e for e in exps if abs(e - (-2.0)) >= newton_tol]
        near_corr = any(abs(e - (-corr_exp)) < 0.3 for e in second_terms)
        if second_terms:
            # Numerical gate (NON-NEGOTIABLE, TIER2-04): adding the correction basis
            # (-1/r**corr_exp) to the Newton accel-fit must reduce the RSS by a
            # structured margin over the held-out span. We credit L2 only when the
            # proposed correction IS the truth's (a ~-corr_exp second exponent) AND
            # it buys real residual reduction — a wrong exponent is rejected
            # structurally, a vacuous term by the reduction gate.
            reduction = _rss_reduction(
                held_out_ics,
                lambda r: [_newton_basis(r)],
                lambda r, _p=corr_exp: _corr_basis(r, _p),
            )
            improved = reduction >= _L2_RSS_REDUCTION_REL
            if near_corr and improved:
                per_layer[1] = True
                if 1 > _deepest_rmse_layer:
                    held_out_rmse = reduction
                    _deepest_rmse_layer = 1
                notes.append(
                    f"L2: second radial term ~r**{-corr_exp:g} + correction-basis RSS "
                    f"reduction {reduction:.3f} >= {_L2_RSS_REDUCTION_REL} ({sf!r})"
                )
            elif near_corr and not improved:
                notes.append(
                    f"L2 DENIED: ~{-corr_exp:g} term present but correction-basis RSS "
                    f"reduction {reduction:.3f} below margin ({sf!r})"
                )
            else:
                notes.append(
                    f"L2 DENIED: second radial term exponent not ~{-corr_exp:g} ({sf!r})"
                )

        # L3 — hidden-charge EXISTENCE: structural hidden-amplitude (product for
        # V1, SUM for V2) AND behavioral. The structural detector + the cross-config
        # gate are regime-selected; the V1 path is byte-identical.
        gamma_est = _extract_cos_frequency(expr)
        # Per-TERM relevance for the coefficient pre-filter: product detection is a
        # per-term property (_has_hidden_product), but the SUM signature is a
        # whole-expression property — after expansion each V2 term carries ONE
        # linear hidden symbol, so the per-term relevance is "carries any hidden
        # symbol" (_term_has_hidden). Using _has_hidden_sum per-term would wrongly
        # see no hidden term and false-deny the decorative-s pre-filter.
        per_term_hidden = _term_has_hidden if is_sum else _has_hidden_product
        if has_hidden_amp(expr):
            if not _term_coefficient_is_nonneglible(expr, per_term_hidden):
                # Decorative-s: a hidden term with a ~0 STATED coefficient does no
                # work (the proposal itself zeroes it). The coefficient pre-filter
                # is authoritative here — a free re-fit of a structurally-present
                # term would launder a decorative s, so we honor the stated
                # coefficient.
                notes.append(f"L3 DENIED: hidden term coefficient ~0 (decorative-s) ({sf!r})")
            else:
                # Behavioral gate: adding the hidden basis (with the proposal's
                # stated frequency, or the truth's if unspecified) to Newton+
                # correction must reduce the accel-fit RSS materially — the hidden
                # DOF does real work.
                g_for_basis = gamma_est if gamma_est is not None else 0.7
                if family_scoring_cfgs is not None:
                    # Family scoring (Phase 19-04 / EXP-080): credit the hidden
                    # charge ONLY when per-config charge SCALING does material RSS
                    # work BEYOND a single global amplitude — PRODUCT scaling for
                    # V1, SUM scaling for V2. A global-beta proposal (no s1*s2 /
                    # s1+s2) never reaches here (structural has_hidden_amp above); a
                    # degenerate equal-amplitude family yields ~0 (not gameable; the
                    # equal-charge control ceilings at depth 2).
                    if is_sum:
                        reduction = _cross_config_ssum_reduction(
                            family_scoring_cfgs, g_for_basis, corr_exp
                        )
                    else:
                        reduction = _cross_config_sproduct_reduction(
                            family_scoring_cfgs, g_for_basis
                        )
                else:
                    reduction = _rss_reduction(
                        held_out_ics,
                        lambda r, _p=corr_exp: [_newton_basis(r), _corr_basis(r, _p)],
                        lambda r, _g=g_for_basis: _hidden_basis(r, _g),
                    )
                hidden_works = reduction >= _L3_RSS_REDUCTION_REL
                rat = rationale_by_form.get(sf, "").lower()
                corroborated = any(k in rat for k in _L3_RATIONALE_KEYWORDS)
                if hidden_works:
                    per_layer[2] = True
                    if 2 > _deepest_rmse_layer:
                        held_out_rmse = reduction
                        _deepest_rmse_layer = 2
                    tb = " (+rationale tiebreaker)" if corroborated else ""
                    amp_word = "sum" if is_sum else "product"
                    notes.append(
                        f"L3: hidden {amp_word} does real work, hidden-basis RSS "
                        f"reduction {reduction:.3f}{tb} ({sf!r})"
                    )
                else:
                    notes.append(
                        f"L3 DENIED: hidden symbols do no accel-fit work "
                        f"(decorative-s), reduction {reduction:.3f} ({sf!r})"
                    )

        # L4 — hidden-force functional form: cos(gamma_est*r), 1/r**2, hidden
        # amplitude (s1*s2 for V1, s1+s2 for V2) + numerical gate (gamma_est~0.7
        # STRUCTURAL AND the hidden term does work). The structural detector + the
        # cross-config gate are regime-selected; the V1 path is byte-identical.
        if (
            has_hidden_amp(expr)
            and gamma_est is not None
            and _has_inverse_square_envelope_on_hidden(expr)
        ):
            gamma_ok = abs(gamma_est - 0.7) < _L4_GAMMA_TOL
            if not gamma_ok:
                # Wrong-frequency cos is rejected HERE (structural): the proposal's
                # STATED frequency must match the truth gamma=0.7. The accel-fit
                # does not cleanly re-identify gamma (under-determined by the
                # visited radii + FD noise), so the stated literal is authoritative.
                notes.append(f"L4 DENIED: cos frequency {gamma_est:.3f} far from 0.7 ({sf!r})")
            else:
                # Numerical gate: the hidden term at gamma=0.7 buys real RSS
                # reduction over Newton+correction (it does work — the L3 oracle at
                # the correct frequency). In family mode it must do that work as a
                # PER-CONFIG charge scaling (PRODUCT for V1, SUM for V2) beyond a
                # global amplitude.
                if family_scoring_cfgs is not None:
                    if is_sum:
                        reduction = _cross_config_ssum_reduction(
                            family_scoring_cfgs, gamma_est, corr_exp
                        )
                    else:
                        reduction = _cross_config_sproduct_reduction(
                            family_scoring_cfgs, gamma_est
                        )
                else:
                    reduction = _rss_reduction(
                        held_out_ics,
                        lambda r, _p=corr_exp: [_newton_basis(r), _corr_basis(r, _p)],
                        lambda r, _g=gamma_est: _hidden_basis(r, _g),
                    )
                if reduction >= _L3_RSS_REDUCTION_REL:
                    per_layer[3] = True
                    if 3 > _deepest_rmse_layer:
                        held_out_rmse = reduction
                        _deepest_rmse_layer = 3
                    amp_repr = "(s1+s2)" if is_sum else "s1*s2"
                    notes.append(
                        f"L4: hidden form ~{amp_repr}*(1-cos({gamma_est:.3f}*r))/r**2 + "
                        f"hidden-basis RSS reduction {reduction:.3f} ({sf!r})"
                    )
                else:
                    notes.append(
                        f"L4 DENIED: structure + gamma match but hidden-basis RSS "
                        f"reduction {reduction:.3f} below margin ({sf!r})"
                    )

        # L5 — conservation: structural candidate AND non-drifting on held-out.
        if _is_conservation_candidate(expr):
            drifts = _claimed_q_drifts(expr, held_out_ics)
            d = _proposed_q_drift(held_out_ics)
            q_drift = d
            if not drifts:
                per_layer[4] = True
                notes.append(
                    f"L5: conserved-quantity structure + bounded held-out drift "
                    f"{d:.3e}<{_Q_DRIFT_BAR:g} ({sf!r})"
                )
            else:
                notes.append(f"L5 DENIED: proposed Q drifts on held-out trajectory ({sf!r})")

        # L6 — hidden symmetry: rationale names it AND sympy verifies invariance.
        rat = rationale_by_form.get(sf, "").lower()
        names_symmetry = any(k in rat for k in _L6_RATIONALE_KEYWORDS) or bool(
            expr.free_symbols & {sympy.Symbol("s"), sympy.Symbol("s1"), sympy.Symbol("s2")}
        )
        if names_symmetry and _is_parity_charge_invariant(expr):
            per_layer[5] = True
            notes.append(f"L6: law verified invariant under P:(x->-x,v->-v,s->-s) ({sf!r})")
        elif names_symmetry:
            notes.append(f"L6 DENIED: claimed symmetry not verified by sympy ({sf!r})")

    max_depth = max((i + 1 for i, ok in enumerate(per_layer) if ok), default=0)
    elapsed_ms = max(0, (time.perf_counter_ns() - start_ns) // 1_000_000)
    logger.info(
        "alien_depth: max_depth=%d per_layer=%s held_out_rmse=%s q_drift=%s "
        "n_hyp=%d elapsed_ms=%d",
        max_depth,
        tuple(per_layer),
        held_out_rmse,
        q_drift,
        len(parsed),
        elapsed_ms,
    )

    return DepthScore(
        max_depth=max_depth,
        per_layer=tuple(per_layer),
        held_out_rmse=held_out_rmse,
        q_drift=q_drift,
        notes=tuple(notes),
    )


def _extract_symbolic_form(h) -> str | None:
    """Read ``symbolic_form`` from a dict row OR a HypothesisContent-like object.

    Mirrors round_close.py:420-453: ``content.get("symbolic_form")``. Returns
    None (caller skips + WARNs) when absent — never a silent drop.
    """
    if isinstance(h, dict):
        val = h.get("symbolic_form")
    else:
        val = getattr(h, "symbolic_form", None)
    if isinstance(val, list):
        # score()-style list wrapping; take the first element.
        val = val[0] if val else None
    return val if isinstance(val, str) and val.strip() else None


def _extract_rationale(h) -> str:
    """Read ``rationale`` from a dict row OR a HypothesisContent-like object."""
    if isinstance(h, dict):
        val = h.get("rationale", "")
    else:
        val = getattr(h, "rationale", "")
    return val if isinstance(val, str) else ""


__all__ = [
    "DepthScore",
    "RegimeSpec",
    "score_depth",
]
