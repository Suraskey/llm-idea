"""Wave-3 observation-contract leak-guard + docstring grep + hash stability.

Plain English: the ONLY thing the Council (Phase 15.2) ever sees is the
``ObservationBundle``. It must carry EXACTLY {t, positions, velocities, masses}
and NOTHING else — no hidden charges, no conserved quantity Q, no config holding
the physics constants, no reference to the true law. If a future field were added
to ``AlienTrajectory`` and the bundle were built by ``dataclasses.asdict`` + a
filter (instead of the explicit positional selection ``alien.py`` uses), the
hidden state would silently re-leak (RESEARCH Pitfall 4 / T-15.0-01).

Three guards (mirror the benchmarks grep-regression discipline):
  1. ``dataclasses.fields(ObservationBundle)`` is exactly the four allowed names —
     a structural assertion, so adding a leaky field FAILS the build.
  2. The ``types.py`` source literally contains the docstring sentinel
     "ObservationBundle carries NO charges, NO true law, NO Q" — a grep-regression
     so the documented contract cannot be silently deleted.
  3. ``simulator_sha256`` is invariant to a plumbing edit (it hashes only the
     three dynamics files) and changes when a config field changes — re-confirmed
     at the test-bar level (the property is also covered by Plan 02's
     test_provenance, this is the contract-level restatement).

Binding: 15.0 PLAN Task 1 <behavior>; property→test-map row 13
(``test_observation_bundle_no_leak``); T-15.0-01. $0 LLM — pure introspection.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from ascension.simulator.alien import AlienUniverse
from ascension.simulator.fixtures import eccentric_fixture
from ascension.simulator.io import simulator_sha256
from ascension.simulator.types import ObservationBundle

# The exact, frozen field set the Council is allowed to see. Anything beyond
# these four is a hidden-state leak (T-15.0-01).
ALLOWED_OBSERVATION_FIELDS = frozenset({"t", "positions", "velocities", "masses"})

# Hidden-field names that must NEVER appear on an ObservationBundle.
FORBIDDEN_FIELDS = frozenset({"charges", "Q", "config", "library_versions"})

_TYPES_PY = Path(__file__).resolve().parents[2] / "src" / "ascension" / "simulator" / "types.py"
# The docstring sentinel that documents the no-leak contract (grep-regression).
_NO_LEAK_LITERAL = "ObservationBundle carries NO charges, NO true law, NO Q"


def test_observation_bundle_no_leak() -> None:
    """ObservationBundle field set is EXACTLY {t,positions,velocities,masses}.

    Structural leak-guard via ``dataclasses.fields`` (T-15.0-01): the bundle must
    expose only the four observable arrays — no charges, no Q, no config. We also
    build a real bundle from a run and confirm the forbidden hidden fields are
    truly absent as attributes.
    """
    field_names = {f.name for f in dataclasses.fields(ObservationBundle)}
    assert field_names == ALLOWED_OBSERVATION_FIELDS, (
        f"ObservationBundle field set {sorted(field_names)} != "
        f"{sorted(ALLOWED_OBSERVATION_FIELDS)} — a hidden field leaked (T-15.0-01)"
    )
    # Defensive: none of the forbidden names sneaked in.
    assert not (field_names & FORBIDDEN_FIELDS), (
        f"ObservationBundle exposes forbidden hidden field(s): "
        f"{sorted(field_names & FORBIDDEN_FIELDS)}"
    )

    # Build a real bundle from a run and confirm the hidden state is gone.
    cfg, traj = eccentric_fixture()
    bundle = AlienUniverse.to_observation_bundle(traj)
    for forbidden in FORBIDDEN_FIELDS:
        assert not hasattr(bundle, forbidden), f"bundle exposes forbidden attribute '{forbidden}'"
    # The observable arrays match the (full) trajectory's observable arrays.
    np.testing.assert_array_equal(bundle.t, traj.t)
    np.testing.assert_array_equal(bundle.positions, traj.positions)
    np.testing.assert_array_equal(bundle.velocities, traj.velocities)
    np.testing.assert_array_equal(bundle.masses, traj.masses)


def test_types_source_carries_no_leak_docstring_literal() -> None:
    """Grep-regression: the no-leak contract literal stays in types.py.

    Mirrors the benchmarks grep-regression idiom — the documented invariant
    "ObservationBundle carries NO charges, NO true law, NO Q" must not be
    silently deleted from the source (it is the human-readable half of the
    type-boundary control, T-15.0-01).
    """
    source = _TYPES_PY.read_text(encoding="utf-8")
    assert _NO_LEAK_LITERAL in source, (
        f"the no-leak docstring literal is missing from {_TYPES_PY.name}: "
        f"expected {_NO_LEAK_LITERAL!r}"
    )


def test_simulator_hash_stable_to_plumbing_unstable_to_config() -> None:
    """simulator_sha256 is stable to plumbing edits, changes on a config field.

    Contract-level restatement of the Plan-02 provenance property at the
    observation/test-bar boundary: the digest hashes only the three dynamics
    files (physics/integrator/alien), so editing io.py/types.py/fixtures.py does
    NOT change it; changing a physics config field (alpha) DOES (RESEARCH Q7).
    """
    cfg, _ = eccentric_fixture()
    # Stable: same config twice → identical digest.
    assert simulator_sha256(cfg) == simulator_sha256(cfg)
    # Unstable to a physics-config change.
    bumped = dataclasses.replace(cfg, alpha=cfg.alpha + 0.01)
    assert simulator_sha256(cfg) != simulator_sha256(bumped)
    # Structural: types.py / io.py / fixtures.py are NOT in the hashed file set,
    # so a plumbing edit is outside the digest's input by construction.
    from ascension.simulator.io import SIMULATOR_HASHED_FILES

    hashed_names = {p.name for p in SIMULATOR_HASHED_FILES}
    assert hashed_names == {"physics.py", "integrator.py", "alien.py"}
    for plumbing in ("io.py", "types.py", "fixtures.py", "config.py", "simulator_audit.py"):
        assert plumbing not in hashed_names, (
            f"{plumbing} is in the hashed set — a plumbing edit would change the "
            "digest (D-05 stability broken)"
        )
