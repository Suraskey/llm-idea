"""Phase 19-04 (AUTO-04, headline) — multi-charge family feed tests.

``alien_family_benchmark`` produces a FAMILY of 2-body alien universes that share
ONE law and differ ONLY in per-body charges (built via ``dataclasses.replace``),
so the autonomy loop can break the single-charge degeneracy. These tests lock:

  - the family spans the expected DISTINCT charge products;
  - all members share one law (G/alpha/beta/gamma/kappa/masses identical);
  - each Council-visible bundle is DATA-ONLY — no charge/coupling/Q/law leak
    (P-EXP-076-6 / the §10 hidden-observable boundary);
  - the held-out scoring configs are STRICTLY OFF the observed family
    (P-EXP-076-7 anti-reward-hack): disjoint charge tuples AND disjoint products.

$0, deterministic — no LLM, no DB.
"""

from __future__ import annotations

import dataclasses

from ascension.benchmarks.alien_fixtures import (
    _FAMILY_CHARGE_PRODUCTS,
    alien_family_benchmark,
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
}


def _product(cfg) -> float:
    s1, s2 = cfg.charges
    return float(s1) * float(s2)


def test_family_spans_distinct_charge_products() -> None:
    family, _held = alien_family_benchmark(seed=0)
    products = [round(_product(cfg), 6) for _bundle, cfg in family]
    assert products == [round(float(p), 6) for p in _FAMILY_CHARGE_PRODUCTS]
    # distinct products — the cross-config s-product signal needs variation
    assert len(set(products)) == len(products)
    assert len(family) >= 4


def test_all_members_share_one_law() -> None:
    family, _held = alien_family_benchmark(seed=0)
    laws = {
        (cfg.G, cfg.alpha, cfg.beta, cfg.gamma, cfg.kappa, cfg.masses) for _bundle, cfg in family
    }
    assert len(laws) == 1, f"family members must share ONE law; got {laws}"


def test_each_bundle_is_data_only() -> None:
    family, _held = alien_family_benchmark(seed=0)
    for bundle, _cfg in family:
        field_names = {f.name.lower() for f in dataclasses.fields(bundle)}
        leaked = field_names & _FORBIDDEN_BUNDLE_FIELDS
        assert not leaked, f"family bundle leaks hidden fields: {leaked}"
        # it DOES carry the observable data the Council legitimately sees
        assert "positions" in field_names and "velocities" in field_names


def test_held_out_charges_off_the_observed_family() -> None:
    family, held_out = alien_family_benchmark(seed=0)
    observed_tuples = {tuple(cfg.charges) for _b, cfg in family}
    held_tuples = {tuple(cfg.charges) for cfg in held_out}
    # charge TUPLES disjoint — no observed config is reused as a scoring config
    assert not (
        observed_tuples & held_tuples
    ), f"held-out charges overlap observed: {observed_tuples & held_tuples}"
    # products disjoint — the scorer tests generalization to UNSEEN products
    observed_products = {round(_product(cfg), 6) for _b, cfg in family}
    held_products = {round(_product(cfg), 6) for cfg in held_out}
    assert not (
        observed_products & held_products
    ), f"held-out products overlap observed: {observed_products & held_products}"
    # the cross-config gate needs >= 2 distinct held-out products
    assert len(held_products) >= 2


def test_held_out_shares_the_same_law_as_the_family() -> None:
    """The held-out scoring configs differ ONLY in charges (same universe)."""
    family, held_out = alien_family_benchmark(seed=0)
    _b, base_cfg = family[0]
    for cfg in held_out:
        assert (cfg.G, cfg.alpha, cfg.beta, cfg.gamma, cfg.kappa, cfg.masses) == (
            base_cfg.G,
            base_cfg.alpha,
            base_cfg.beta,
            base_cfg.gamma,
            base_cfg.kappa,
            base_cfg.masses,
        )


def test_family_is_deterministic() -> None:
    f1, h1 = alien_family_benchmark(seed=0)
    f2, h2 = alien_family_benchmark(seed=0)
    assert [tuple(c.charges) for _b, c in f1] == [tuple(c.charges) for _b, c in f2]
    assert [tuple(c.charges) for c in h1] == [tuple(c.charges) for c in h2]
