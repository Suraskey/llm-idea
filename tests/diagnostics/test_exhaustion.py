"""Design-class exhaustion certificates: "no experiment in this class can help."

Three plateaus, same primitive, three different answers:

1. Oral-dosing PK (a published non-identifiability): F/V is a scaling symmetry of the
   whole oral class (every sampling schedule). CERTIFIED -> change the class (IV arm).
2. Alien Universe at equal charges (the paper's testbed): the [1,-1] charge direction
   is a symmetry of the single-configuration class (every radius, every initial
   condition, every time grid). CERTIFIED -> change the class (a multi-charge family).
   This is EXP-078's EXHAUSTED branch as a proof instead of three rounds of budget.
3. The closed-form ansatz on a narrow sample window: [1, log n, n] is collinear on
   the window but NOT identically, so a different window in the same class resolves
   it. NOT certified -> design within the class (which is what the ranker did).

$0, no LLM, no DB.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import sympy

from ascension.diagnostics.exhaustion import (
    certify_class_symmetry_numeric,
    certify_class_symmetry_symbolic,
    triage_plateau,
)
from ascension.diagnostics.identifiability import practical_identifiability

# --- 1. oral PK ------------------------------------------------------------------
DOSE = 100.0
PK_THETA = [1.2, 0.25, 30.0, 0.6]  # ka, ke, V, F
PK_NAMES = ["ka", "ke", "V", "F"]


def _pk_oral(theta, t):
    ka, ke, v, f = theta
    return (f * DOSE * ka) / (v * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def _pk_expr():
    t = sympy.Symbol("t", positive=True)
    ka, ke, V, F = sympy.symbols("ka ke V F", positive=True)
    y = (F * DOSE * ka) / (V * (ka - ke)) * (sympy.exp(-ke * t) - sympy.exp(-ka * t))
    return y, [ka, ke, V, F], [t]


def test_pk_f_over_v_is_a_symmetry_of_the_whole_oral_class_symbolic() -> None:
    y, params, free = _pk_expr()
    t = np.linspace(0.25, 24.0, 24)
    v = practical_identifiability(
        lambda th: _pk_oral(np.asarray(th, float), t), PK_THETA, param_names=PK_NAMES
    )
    assert v.rank == 3
    cert = certify_class_symmetry_symbolic(
        y, params, PK_THETA, v.confounded_directions[0], free_vars=free
    )
    assert cert.certified, cert.describe()
    assert cert.method == "symbolic"
    # the snapped direction is the (V, F) scaling axis, exactly
    assert cert.direction[0] == 0 and cert.direction[1] == 0
    assert abs(cert.direction[3] / cert.direction[2] - PK_THETA[3] / PK_THETA[2]) < 1e-6


def test_pk_numeric_route_agrees_over_random_sampling_schedules() -> None:
    v = practical_identifiability(
        lambda th: _pk_oral(np.asarray(th, float), np.linspace(0.25, 24.0, 24)),
        PK_THETA,
        param_names=PK_NAMES,
    )

    def sampler(rng):
        n = int(rng.integers(6, 40))
        t = np.sort(rng.uniform(0.1, 96.0, size=n))
        return lambda th: _pk_oral(np.asarray(th, float), t)

    cert = certify_class_symmetry_numeric(
        sampler, PK_THETA, v.confounded_directions[0], n_designs=24, param_names=PK_NAMES
    )
    assert cert.certified, cert.describe()
    assert cert.n_designs_sampled == 24


def test_pk_triage_says_change_class_and_the_iv_arm_breaks_it() -> None:
    y, params, free = _pk_expr()
    t = np.linspace(0.25, 24.0, 24)
    oral = lambda th: _pk_oral(np.asarray(th, float), t)  # noqa: E731
    certify = partial(certify_class_symmetry_symbolic, y, params, PK_THETA, free_vars=free)
    tri = triage_plateau(oral, PK_THETA, certify=certify, param_names=PK_NAMES)
    assert tri.verdict == "change_class", tri.describe()

    # the class change: an IV arm (F = 1 by definition) added to the observation set
    def oral_plus_iv(th):
        ka, ke, v, f = np.asarray(th, float)
        return np.concatenate([_pk_oral((ka, ke, v, f), t), (DOSE / v) * np.exp(-ke * t)])

    tri2 = triage_plateau(oral_plus_iv, PK_THETA, certify=certify, param_names=PK_NAMES)
    assert tri2.verdict == "capability"


# --- 2. Alien Universe at equal charges -------------------------------------------
def test_alien_single_config_two_certified_directions_and_what_the_family_breaks() -> None:
    """The hidden force is beta*s1*s2*(1-cos(gamma r))/r^2. With theta = (beta, s1, s2)
    observed at ONE configuration, two directions are invisible:

      swap    (0, 1, -1):          s1 <-> s2 is a symmetry of the product law itself, so
                                   it is class-level for EVERY design (single config or
                                   family). The honest report is the product, never the
                                   factors: "report the identifiable combination".
      scaling (beta, -s1/2, -s2/2): beta up, product down leaves beta*s1*s2 fixed. It is
                                   class-level for the single-configuration class (any
                                   radius, any initial condition, any time grid), which
                                   is why EXP-078's new-IC branch was EXHAUSTED. A
                                   FAMILY (a second system sharing beta with its own
                                   charges) breaks it: the class change the paper's
                                   observability knob performs.
    """
    r = sympy.Symbol("r", positive=True)
    beta, s1, s2 = sympy.symbols("beta s1 s2", positive=True)
    gamma = 2
    force = beta * s1 * s2 * (1 - sympy.cos(gamma * r)) / r**2
    theta = [0.3, 1.0, 1.0]
    swap = (0.0, 0.7071, -0.7071)  # as the diagnostic would estimate it
    scaling = (0.3, -0.5, -0.5)  # (beta, -s1/2, -s2/2) at the operating point

    c_swap = certify_class_symmetry_symbolic(force, [beta, s1, s2], theta, swap, free_vars=[r])
    c_scale = certify_class_symmetry_symbolic(force, [beta, s1, s2], theta, scaling, free_vars=[r])
    assert c_swap.certified, c_swap.describe()
    assert c_scale.certified, c_scale.describe()
    assert c_swap.direction == (0.0, 1.0, -1.0)  # snapped exactly

    # the class change: a family. Two systems share beta; the second has its own charges.
    t1, t2 = sympy.symbols("t1 t2", positive=True)
    family = sympy.Matrix([force, force.subs({s1: t1, s2: t2})])
    params = [beta, s1, s2, t1, t2]
    point = {beta: sympy.Rational(3, 10), s1: 1, s2: 1, t1: 2, t2: sympy.Rational(1, 2)}

    def directional(d):
        return sympy.simplify(
            sum(
                (dj * family.diff(pj) for dj, pj in zip(d, params, strict=True)), sympy.zeros(2, 1)
            ).subs(point)
        )

    # swap stays a symmetry of the family (only the product ever enters the law)
    assert directional([0, 1, -1, 0, 0]) == sympy.zeros(2, 1)
    # scaling of system 1 no longer is: system 2 shares beta and responds
    assert directional(
        [sympy.Rational(3, 10), -sympy.Rational(1, 2), -sympy.Rational(1, 2), 0, 0]
    ) != sympy.zeros(2, 1)


def test_alien_numeric_route_over_random_radii_and_products_is_class_level() -> None:
    """Numeric twin of the symbolic proof: sample many single-configuration designs
    (random radius grids) at the operating point s1 == s2; the [1,-1] direction never
    moves the prediction."""

    def sampler(rng):
        rr = np.sort(rng.uniform(0.5, 4.0, size=int(rng.integers(5, 30))))

        def predict(th):
            s1, s2 = np.asarray(th, float)
            return 0.3 * s1 * s2 * (1 - np.cos(2 * rr)) / rr**2

        return predict

    cert = certify_class_symmetry_numeric(
        sampler, [1.0, 1.0], (1.0, -1.0), n_designs=20, param_names=["s1", "s2"]
    )
    assert cert.certified, cert.describe()


# --- 3. the sequence ansatz: NOT class-level ---------------------------------------
def test_sequence_ansatz_window_collapse_is_not_a_class_symmetry() -> None:
    """log f(n) = log(theta0) + theta1*log(n) + n*log(theta2). On a 3-point tail
    window the columns [1, log n, n] are numerically collinear (the EXP-084 rank
    drop), but the collapse is a property of the WINDOW, not the class: the
    directional derivative is a non-zero function of n, so a better window exists."""
    n = sympy.Symbol("n", positive=True)
    th0, th1, th2 = sympy.symbols("theta0 theta1 theta2", positive=True)
    logf = sympy.log(th0) + th1 * sympy.log(n) + n * sympy.log(th2)

    grid = np.array([400.0, 401.0, 402.0])
    v = practical_identifiability(
        lambda th: np.log(th[0]) + th[1] * np.log(grid) + grid * np.log(th[2]),
        [1.0, 0.0, 1.0],
        param_names=["theta0", "theta1", "theta2"],
        cond_threshold=1e6,
    )
    assert not v.identifiable  # the narrow-window collapse the ranker later fixed
    direction = v.confounded_directions[0]
    cert = certify_class_symmetry_symbolic(
        logf, [th0, th1, th2], [1.0, 0.0, 1.0], direction, free_vars=[n], snap_rational=False
    )
    assert not cert.certified, cert.describe()
    assert cert.residual is not None

    # numeric twin: sampling wide windows from the same class exposes the direction
    def sampler(rng):
        pts = np.sort(rng.uniform(2.0, 500.0, size=6))
        return lambda th: np.log(th[0]) + th[1] * np.log(pts) + pts * np.log(th[2])

    ncert = certify_class_symmetry_numeric(sampler, [1.0, 0.0, 1.0], direction, n_designs=8)
    assert not ncert.certified


def test_symbolic_route_rejects_unknown_free_vars() -> None:
    x, a = sympy.symbols("x a")
    try:
        certify_class_symmetry_symbolic(a * x, [a], [1.0], [1.0], free_vars=["zzz"])
    except ValueError as e:
        assert "free_vars" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


# --- 4. the loop: exhausted by PROOF at round 0 instead of by budget ---------------
def test_loop_stops_by_certificate_at_round_zero_when_class_is_exhausted() -> None:
    from ascension.diagnostics.engine import run_identifiability_loop

    y, params, free = _pk_expr()
    t = np.linspace(0.25, 24.0, 24)
    oral = lambda th: _pk_oral(np.asarray(th, float), t)  # noqa: E731

    def propose_only_oral(_verdict):
        # a proposer that can only vary sampling inside the oral class
        return {
            "oral dense": lambda th: _pk_oral(np.asarray(th, float), np.linspace(0.25, 24.0, 96)),
            "oral long": lambda th: _pk_oral(np.asarray(th, float), np.linspace(0.25, 72.0, 48)),
        }

    def certify(verdict):
        return [
            certify_class_symmetry_symbolic(y, params, PK_THETA, d, free_vars=free)
            for d in verdict.confounded_directions
        ]

    # without the certificate: three rounds of budget burned, then "exhausted"
    r0 = run_identifiability_loop(
        oral, PK_THETA, propose_designs=propose_only_oral, max_rounds=3, param_names=PK_NAMES
    )
    assert r0.outcome == "exhausted" and r0.rounds_used == 3

    # with it: stopped at round 0 by proof
    r1 = run_identifiability_loop(
        oral,
        PK_THETA,
        propose_designs=propose_only_oral,
        certify_exhaustion=certify,
        max_rounds=3,
        param_names=PK_NAMES,
    )
    assert r1.outcome == "exhausted_by_certificate", r1.rationale
    assert r1.rounds_used == 0
    assert "Change the design class" in r1.rationale

    # and the certificate does NOT fire when the class is resolvable: the sequence
    # ansatz on a narrow window still runs the normal design loop
    n = sympy.Symbol("n", positive=True)
    th0, th1, th2 = sympy.symbols("theta0 theta1 theta2", positive=True)
    logf = sympy.log(th0) + th1 * sympy.log(n) + n * sympy.log(th2)
    narrow = np.array([400.0, 401.0, 402.0])
    wide = np.array([2.0, 5.0, 20.0, 100.0, 400.0])
    seq_narrow = lambda th: np.log(th[0]) + th[1] * np.log(narrow) + narrow * np.log(th[2])  # noqa: E731
    seq_wide = lambda th: np.log(th[0]) + th[1] * np.log(wide) + wide * np.log(th[2])  # noqa: E731

    def certify_seq(verdict):
        return [
            certify_class_symmetry_symbolic(
                logf, [th0, th1, th2], [1.0, 0.0, 1.0], d, free_vars=[n], snap_rational=False
            )
            for d in verdict.confounded_directions
        ]

    r2 = run_identifiability_loop(
        seq_narrow,
        [1.0, 0.0, 1.0],
        propose_designs=lambda v: {"wide": seq_wide},
        certify_exhaustion=certify_seq,
        max_rounds=3,
        param_names=["theta0", "theta1", "theta2"],
    )
    assert r2.outcome == "solved" and r2.rounds_used == 1


# --- 5. soundness: a first-order null at a SINGULAR point is not a certificate ------
def test_singular_point_is_refused_symbolic_and_numeric() -> None:
    """f = a x + b**3 x**2 at b = 0: every design has zero sensitivity to b there, so
    the first-order check passes for every x. But b is identifiable (x = 1, 2 give
    y(2) - 2 y(1) = 2 b**3, injective in b). The class rank is 1 at b = 0 and 2 at
    every nearby point, so the point is singular and the certificate must refuse.
    Before 2026-09-28 both routes certified this case."""
    x, a, b = sympy.symbols("x a b")
    expr = a * x + b**3 * x**2
    sym = certify_class_symmetry_symbolic(expr, [a, b], [1.0, 0.0], [0.0, 1.0], free_vars=[x])
    assert not sym.certified, sym.describe()
    assert sym.regular_point is False
    assert (sym.class_rank_at_point, sym.class_rank_nearby) == (1, 2)
    # the pre-fix behaviour, kept reachable for reproducing old runs
    legacy = certify_class_symmetry_symbolic(
        expr, [a, b], [1.0, 0.0], [0.0, 1.0], free_vars=[x], check_regular=False
    )
    assert legacy.certified

    def sampler(rng):
        pts = rng.uniform(0.5, 3.0, size=4)
        return lambda th: th[0] * pts + th[1] ** 3 * pts**2

    num = certify_class_symmetry_numeric(sampler, [1.0, 0.0], [0.0, 1.0], n_designs=8)
    assert not num.certified, num.describe()
    assert num.regular_point is False

    # and at a regular point of a genuine symmetry both routes still certify
    y2 = a * b * x
    ok = certify_class_symmetry_symbolic(y2, [a, b], [2.0, 3.0], [2.0, -3.0], free_vars=[x])
    assert ok.certified and ok.regular_point, ok.describe()
    assert (ok.class_rank_at_point, ok.class_rank_nearby) == (1, 1)


# --- 6. mixed case: some directions resolvable in the class, one certified --------
def test_loop_reports_solved_up_to_class_symmetry_when_only_certified_directions_remain() -> None:
    """f(x; a, b, c) = a*b*x + c*x**2. One design point (x = 1) sees only a*b + c:
    rank 1 of 3. The (a, -b, 0) scaling is a symmetry of every design (only a*b
    enters); the c direction is resolvable (x**2 vs x). After one added design the
    rank is 2 of 3 = the class rank, and the only remaining null direction is
    certified. Before 2026-09-28 the loop called this "exhausted_by_certificate", as
    if no progress had been made."""
    from ascension.diagnostics.engine import run_identifiability_loop

    x, a, b, c = sympy.symbols("x a b c")
    expr = a * b * x + c * x**2
    theta = [2.0, 3.0, 0.5]
    one = lambda th: np.array([th[0] * th[1] * 1.0 + th[2] * 1.0])  # noqa: E731
    two = lambda th: np.array([th[0] * th[1] * xv + th[2] * xv**2 for xv in (1.0, 2.0)])  # noqa: E731

    def certify(verdict):
        return [
            certify_class_symmetry_symbolic(expr, [a, b, c], theta, d, free_vars=[x])
            for d in verdict.confounded_directions
        ]

    res = run_identifiability_loop(
        one,
        theta,
        propose_designs=lambda v: {"two points": two},
        certify_exhaustion=certify,
        max_rounds=3,
        param_names=["a", "b", "c"],
    )
    assert res.outcome == "solved_up_to_class_symmetry", res.rationale
    assert res.rounds_used == 1
    assert res.steps[-1].rank == 2
