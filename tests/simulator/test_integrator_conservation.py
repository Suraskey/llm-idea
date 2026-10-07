"""The make-or-break Q-conservation bar + determinism + audit-first stubs.

These tests prove the two integrator facts of Phase 15.0:
  1. THE BAR (D-01): implicit-midpoint holds Q-drift < 1e-6 over 1000 steps on
     an eccentric 2-body fixture, driven through the REAL integrator entrypoint.
  2. The symplectic signature (D-09 stretch): over 10,000 steps the drift stays
     the SAME ORDER of magnitude (bounded, non-secular), NOT a tighter absolute
     bound — proving it does not grow without limit.

Plus D-06 bit-identical reproducibility (``np.array_equal``, NOT ``allclose``)
and the audit-first defense: all five ``simulator_audit`` checks raise
``NotImplementedError`` at ship and the façade re-raises LOUD.

Binding: 15.0 PLAN Task 3 <behavior>; 15.0 RESEARCH Q2 (dt=0.005, bounded
drift). $0 LLM — deterministic numpy/scipy/sympy only.
"""

from __future__ import annotations

import numpy as np

from ascension.simulator.integrator import implicit_midpoint_step, integrate
from ascension.simulator.simulator_audit import run_simulator_audit

_DT = 0.005


def test_q_conservation(eccentric_fixture) -> None:
    """THE make-or-break bar: max|Q − Q₀| < 1e-6 over 1000 steps.

    Integrates the eccentric fixture with the real ``integrate`` loop (which
    uses ``implicit_midpoint_step``) and the ``physics``-backed polar rhs, then
    evaluates Q at every recorded state. Non-vacuous: the fixture is genuinely
    eccentric (tangential_frac=0.8), so v_perp varies over the orbit and Q has
    something real to conserve.
    """
    sys = eccentric_fixture
    traj = integrate(sys.y0, _DT, 1000, sys.rhs, config=sys.config, seed=sys.config.seed)
    assert traj.shape[0] == 1001  # n_steps + 1
    q_series = np.array([sys.Q(traj[i]) for i in range(traj.shape[0])])
    q0 = sys.Q(sys.y0)
    drift = float(np.max(np.abs(q_series - q0)))
    # Non-vacuous sanity: the orbit must actually move radially (eccentric).
    r_series = traj[:, 0]
    assert (
        r_series.max() - r_series.min() > 0.05
    ), "fixture is vacuously near-circular — Q conservation would be trivial"
    assert drift < 1e-6, f"Q drift {drift:.3e} exceeds the 1e-6 make-or-break bar"


def test_q_conservation_stretch(eccentric_fixture) -> None:
    """D-09 stretch: 10,000 steps → drift stays the SAME ORDER (bounded).

    The symplectic signature is BOUNDED, non-secular drift. We assert the 10k
    drift is within a small constant factor of the 1k drift — NOT a tighter
    absolute bound (which would wrongly demand the error shrink with horizon).
    """
    sys = eccentric_fixture
    q0 = sys.Q(sys.y0)

    traj_1k = integrate(sys.y0, _DT, 1000, sys.rhs, config=sys.config, seed=sys.config.seed)
    drift_1k = float(np.max(np.abs([sys.Q(traj_1k[i]) - q0 for i in range(traj_1k.shape[0])])))

    traj_10k = integrate(sys.y0, _DT, 10000, sys.rhs, config=sys.config, seed=sys.config.seed)
    drift_10k = float(np.max(np.abs([sys.Q(traj_10k[i]) - q0 for i in range(traj_10k.shape[0])])))

    # Bounded / non-secular: 10x the horizon must NOT 10x the drift. Allow a
    # small constant factor (symplectic drift oscillates within a band).
    assert drift_10k < 5.0 * max(drift_1k, 1e-12), (
        f"drift grew secularly: 1k={drift_1k:.3e} 10k={drift_10k:.3e} "
        "(expected bounded, same order)"
    )
    # And it still clears the bar at 10k (it should — symplectic).
    assert drift_10k < 1e-6, f"10k drift {drift_10k:.3e} exceeds 1e-6"


def test_bit_identical_reproducibility(eccentric_fixture) -> None:
    """D-06: same y0 + same rhs twice → np.array_equal (bitwise, NOT allclose).

    Proves the inner fixed-point solve does not branch on float noise: a fixed
    max_iter cap + a deterministic reduction give the same iteration trace
    every run, float64 throughout.
    """
    sys = eccentric_fixture
    a = integrate(sys.y0, _DT, 500, sys.rhs, config=sys.config, seed=sys.config.seed)
    b = integrate(sys.y0, _DT, 500, sys.rhs, config=sys.config, seed=sys.config.seed)
    assert a.dtype == np.float64 and b.dtype == np.float64
    np.testing.assert_array_equal(a, b)


def test_single_step_is_deterministic(eccentric_fixture) -> None:
    """A single implicit_midpoint_step is bit-identical on re-run."""
    sys = eccentric_fixture
    s1 = implicit_midpoint_step(sys.y0, _DT, sys.rhs)
    s2 = implicit_midpoint_step(sys.y0, _DT, sys.rhs)
    np.testing.assert_array_equal(s1, s2)


def test_audit_gate_is_green_after_plan03_flip() -> None:
    """Plan 03 flipped all 5 checks GREEN: run_simulator_audit() now passes.

    Wave 1 shipped the gate audit-first (all 5 stubs raised NotImplementedError;
    the façade re-raised LOUD). Plan 03 replaced each stub with a real probe, so
    the façade now returns a passing DTO. The defense-in-depth re-raise on a
    regressed stub is asserted in test_simulator_audit.py."""
    result = run_simulator_audit()
    assert result.passed is True


def test_individual_audit_checks_pass_after_flip() -> None:
    """Each of the five checks returns None (pass) on the default config (Plan 03).

    The audit-first NotImplementedError stubs are gone; each probe now returns
    None on pass or a SimulatorAuditResult on failure. The failure paths +
    defense-in-depth re-raise live in test_simulator_audit.py."""
    from ascension.simulator import simulator_audit as A

    for fn in (
        A._check_property_has_test,
        A._check_hash_reproducible,
        A._check_no_hidden_leak,
        A._check_no_llm,
        A._check_drift_bound,
    ):
        assert fn() is None
