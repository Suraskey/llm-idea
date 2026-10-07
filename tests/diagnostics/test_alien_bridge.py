"""PIVOT-001 Burst B — the worked-example bridge: the generic identifiability
diagnostic applied to the REAL Alien Universe simulator physics (not a toy model).

This is the headline validation of the pivot's Diagnostic pillar: the same
domain-agnostic `practical_identifiability` checker, handed a `predict(charges)`
closure built from the actual `simulator.physics.total_accel`, INDEPENDENTLY
recovers L-051 — the hidden per-body charges s1, s2 are NOT separately identifiable
from a single configuration's radial-acceleration profile (only the product s1*s2
acts, via `hidden_charge_accel`), and the unobservable direction is exactly [1, -1].

No LLM, no trajectory integration — just the force law evaluated at a few radii, the
finite-difference sensitivity matrix, and its SVD. The diagnostic was written with no
knowledge of the alien system; that it reproduces the simulator's own degeneracy is
the evidence the wedge is real, not retrofitted.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.benchmarks.alien_fixtures import _build_tier2_tuned_cfg
from ascension.diagnostics import practical_identifiability
from ascension.simulator.physics import total_accel

# Radii at which we "measure" the radial acceleration (the observation design).
# Spread across the orbit; 1 - cos(gamma*r) is non-degenerate here so the hidden
# term does real work.
_RADII = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


def _alien_radial_accel_predict(base_cfg):
    """Build predict(charges) -> radial-accel profile from the REAL force law.

    Releases body 2 from rest at separation r (velocities = 0 — the radial-drop
    setup that switches off the kappa velocity-coupling so the charge question is
    isolated), evaluates ``total_accel`` at each radius, and returns the radial
    (x) component on body 2 at each radius. Only ``hidden_charge_accel`` depends on
    the charges, so the entire sensitivity to (s1, s2) flows through the s1*s2 term.
    """

    def predict(theta: np.ndarray) -> np.ndarray:
        cfg = dataclasses.replace(base_cfg, charges=(float(theta[0]), float(theta[1])))
        out = []
        for r in _RADII:
            pos = np.array([[0.0, 0.0], [r, 0.0]], dtype=np.float64)
            vel = np.zeros((2, 2), dtype=np.float64)
            a = total_accel(pos, vel, cfg)
            out.append(a[1][0])  # radial (x) acceleration of body 2
        return np.array(out, dtype=np.float64)

    return predict


def _abs_cos(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def test_alien_hidden_charge_is_nonidentifiable_on_real_physics() -> None:
    """L-051 recovered on the actual simulator: at the equal TIER2 charges the two
    hidden charges are confounded (only s1*s2 is observable), unobservable
    direction ~[1, -1]."""
    cfg = _build_tier2_tuned_cfg(seed=0)
    predict = _alien_radial_accel_predict(cfg)

    # Sanity: the observable is non-trivial (the hidden term actually moves a(r)).
    a_unit = predict(np.array([1.0, 1.0]))
    a_double_product = predict(np.array([2.0, 2.0]))  # product 4 != 1 -> must differ
    assert np.linalg.norm(a_unit) > 0
    assert not np.allclose(
        a_unit, a_double_product
    ), "the charge product must affect a(r), or the test radii are degenerate"

    verdict = practical_identifiability(predict, [1.0, 1.0], param_names=["s1", "s2"])
    assert verdict.identifiable is False
    assert verdict.status == "structurally_non_identifiable"
    assert verdict.rank == 1, (
        f"single-config a(r) depends only on the product s1*s2 -> rank 1; "
        f"got rank {verdict.rank} ({verdict.rationale})"
    )
    assert len(verdict.confounded_directions) == 1
    assert _abs_cos(verdict.confounded_directions[0], [1.0, -1.0]) > 0.999, (
        f"the unobservable direction at equal charges must be ~[1,-1]; "
        f"got {verdict.confounded_directions[0]}"
    )


def test_alien_product_only_degeneracy_matches_L051() -> None:
    """Two charge pairs with the SAME product (1,1) and (2,0.5) are observationally
    identical under the single-config radial-accel design — the exact L-051
    observational-indistinguishability the discoverability audit proved."""
    cfg = _build_tier2_tuned_cfg(seed=0)
    predict = _alien_radial_accel_predict(cfg)
    a_11 = predict(np.array([1.0, 1.0]))
    a_205 = predict(np.array([2.0, 0.5]))  # same product = 1.0
    assert np.allclose(
        a_11, a_205, rtol=1e-9, atol=1e-12
    ), "equal-product charge pairs must produce identical a(r) (L-051)"
