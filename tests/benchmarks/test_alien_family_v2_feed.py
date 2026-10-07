"""EXP-080 (STRAT-03) — the V2-regime multi-charge family feed tests.

``alien_family_v2_benchmark`` produces a FAMILY of 2-body alien universes on the
SECOND hidden-physics law (sum coupling + r^-2.5), sharing ONE V2 law and
differing ONLY in per-body charges. The V2 analog of the V1 family feed; these
tests lock the same invariants but on the V2 regime:

  - every member runs the V2 law (charge_coupling="sum", correction_exponent=2.5);
  - the family spans the expected DISTINCT charge SUMS (the V2 identifiable
    signal — s1+s2, not s1*s2);
  - all members share one law (G/alpha/beta/gamma/kappa/masses identical);
  - each Council-visible bundle is DATA-ONLY — no charge/coupling/Q/law leak
    (P-EXP-080-5/6 / the §10 hidden-observable boundary);
  - the held-out scoring sums are STRICTLY OFF the observed family
    (P-EXP-080-6 anti-reward-hack): disjoint charge tuples AND disjoint sums;
  - determinism (D-06).

$0, deterministic — no LLM, no DB.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ascension.benchmarks.alien_fixtures import (
    _FAMILY_V2_CHARGE_SUMS,
    _HELD_OUT_V2_CHARGE_SUMS,
    AlienFamilyObservation,
    alien_family_v2_benchmark,
    alien_family_v2_observation,
)

# Any of these field names on a Council-visible bundle would re-leak the hidden
# universe (the §10 boundary). The bundle must be data-only.
_FORBIDDEN_BUNDLE_FIELDS = {
    "charge",
    "charges",
    "alpha",
    "beta",
    "gamma",
    "kappa",
    "q",
    "law",
    "config",
    "couplings",
    "charge_coupling",
    "correction_exponent",
}


def _charge_sum(cfg) -> float:
    s1, s2 = cfg.charges
    return float(s1) + float(s2)


def test_v2_family_members_run_the_v2_law() -> None:
    """Every observed + held-out member carries the V2 regime knobs."""
    family, held_out = alien_family_v2_benchmark(seed=0)
    for _bundle, cfg in family:
        assert cfg.charge_coupling == "sum"
        assert cfg.correction_exponent == 2.5
    for cfg in held_out:
        assert cfg.charge_coupling == "sum"
        assert cfg.correction_exponent == 2.5


def test_v2_family_spans_distinct_charge_sums() -> None:
    """The observed family spans exactly the declared distinct charge SUMS."""
    family, _held = alien_family_v2_benchmark(seed=0)
    sums = sorted(round(_charge_sum(cfg), 6) for _b, cfg in family)
    assert sums == sorted(round(float(s), 6) for s in _FAMILY_V2_CHARGE_SUMS)
    # Genuinely distinct (no collapsed duplicates → real cross-config variation).
    assert len(set(sums)) == len(sums) >= 2


def test_v2_all_members_share_one_law() -> None:
    """All observed members share G/alpha/beta/gamma/kappa/masses + V2 knobs."""
    family, _held = alien_family_v2_benchmark(seed=0)
    cfgs = [cfg for _b, cfg in family]
    base = cfgs[0]
    for cfg in cfgs[1:]:
        assert cfg.G == base.G
        assert cfg.alpha == base.alpha
        assert cfg.beta == base.beta
        assert cfg.gamma == base.gamma
        assert cfg.kappa == base.kappa
        assert cfg.masses == base.masses
        assert cfg.charge_coupling == base.charge_coupling
        assert cfg.correction_exponent == base.correction_exponent
        # ONLY the charges differ across the family.
        assert cfg.charges != base.charges


def test_v2_each_bundle_is_data_only() -> None:
    """T-15.2-leak / §10 boundary — each V2 family bundle carries NO hidden state."""
    family, _held = alien_family_v2_benchmark(seed=0)
    for bundle, _cfg in family:
        field_names = {f.name for f in dataclasses.fields(bundle)}
        leaked = field_names & _FORBIDDEN_BUNDLE_FIELDS
        assert not leaked, f"V2 family bundle leaks hidden fields: {leaked}"
        # exact data-only field set
        assert field_names == {"t", "positions", "velocities", "masses"}


def test_v2_held_out_sums_off_the_observed_family() -> None:
    """P-EXP-080-6 anti-reward-hack — held-out sums STRICTLY off the observed set.

    Disjoint charge TUPLES (no observed config reused) AND disjoint SUMS (the
    scorer tests generalization to UNSEEN sums, not memorization).
    """
    family, held_out = alien_family_v2_benchmark(seed=0)
    obs_tuples = {tuple(cfg.charges) for _b, cfg in family}
    held_tuples = {tuple(cfg.charges) for cfg in held_out}
    assert obs_tuples.isdisjoint(held_tuples), "a held-out config reuses an observed charge tuple"

    obs_sums = {round(_charge_sum(cfg), 6) for _b, cfg in family}
    held_sums = {round(_charge_sum(cfg), 6) for cfg in held_out}
    assert obs_sums.isdisjoint(held_sums), "held-out sums overlap the observed family"
    # >= 2 held-out members at distinct sums (cross-config gate needs variation).
    assert len(held_sums) >= 2
    assert held_sums == {round(float(s), 6) for s in _HELD_OUT_V2_CHARGE_SUMS}


def test_v2_held_out_shares_the_same_law_as_the_family() -> None:
    """Held-out configs share the family law (only charges differ)."""
    family, held_out = alien_family_v2_benchmark(seed=0)
    base = family[0][1]
    for cfg in held_out:
        assert cfg.G == base.G
        assert cfg.alpha == base.alpha
        assert cfg.beta == base.beta
        assert cfg.gamma == base.gamma
        assert cfg.kappa == base.kappa
        assert cfg.masses == base.masses
        assert cfg.charge_coupling == "sum"
        assert cfg.correction_exponent == 2.5


def test_v2_family_is_deterministic() -> None:
    """Two seed=0 builds → bit-identical observed positions (D-06)."""
    f1, h1 = alien_family_v2_benchmark(seed=0)
    f2, h2 = alien_family_v2_benchmark(seed=0)
    assert len(f1) == len(f2)
    for (b1, _c1), (b2, _c2) in zip(f1, f2):
        np.testing.assert_array_equal(b1.positions, b2.positions)
    assert tuple(c.charges for c in h1) == tuple(c.charges for c in h2)


def test_v2_observation_wrapper_is_council_visible_family() -> None:
    """alien_family_v2_observation builds an AlienFamilyObservation of >= 2 bundles."""
    obs = alien_family_v2_observation(seed=0)
    assert isinstance(obs, AlienFamilyObservation)
    assert len(obs.bundles) == len(_FAMILY_V2_CHARGE_SUMS)
    # duck-types the baseline member (the round-close consumer relies on this)
    assert obs.positions is obs.bundles[0].positions
    assert obs.masses is obs.bundles[0].masses


def test_v2_family_trajectories_differ_across_sums() -> None:
    """Distinct charge sums → distinguishable trajectories (the discoverable signal).

    Non-vacuity: if the sum did not affect the dynamics, the family would carry no
    cross-config signal and the V2 generalization test would be meaningless.
    """
    family, _held = alien_family_v2_benchmark(seed=0)
    # Compare the first two members' position arrays — they MUST differ.
    p0 = family[0][0].positions
    p1 = family[1][0].positions
    assert not np.array_equal(p0, p1), "different charge sums produced identical trajectories"
