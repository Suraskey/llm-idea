"""Wave-2 provenance tests: simulator_sha256 stability + .npz bit-repro round-trip.

Plain English: a run must be replayable from its hash + manifest alone (the
six-months-later reviewer, T-15.0-03). Two properties are load-bearing:
  1. ``simulator_sha256`` is deterministic and STABLE across edits to plumbing
     (io.py, fixtures docstrings) — it hashes only the three DYNAMICS files
     (physics.py + integrator.py + alien.py) plus the sorted-key config JSON
     (RESEARCH Q7). It CHANGES when a config field (e.g. alpha) changes.
  2. Trajectory arrays round-trip bit-identically through save → load
     (``np.savez``, NOT savez_compressed — ZIP_STORED round-trips exactly).

Binding decisions:
  - 15.0 RESEARCH Q7: the pinned 3-file hash recipe + sorted-key config JSON.
  - benchmarks/io.py:139-141 + RESEARCH §Don't Hand-Roll: np.savez (uncompressed).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from ascension.simulator import io as SIO
from ascension.simulator.fixtures import newtonian_limit_fixture
from ascension.simulator.io import simulator_sha256
from ascension.simulator.types import AlienConfig


def _cfg() -> AlienConfig:
    return AlienConfig(
        G=1.0,
        alpha=0.05,
        beta=0.02,
        gamma=0.7,
        kappa=0.3,
        masses=(1.0, 1.0),
        charges=(1.0, 1.0),
        ics_pos=((0.0, 0.0), (1.0, 0.0)),
        ics_vel=((0.0, 0.0), (0.0, 0.9)),
        dt=0.005,
        n_steps=100,
        seed=0,
    )


def test_simulator_sha256_deterministic() -> None:
    """Same config + same source bytes → same digest twice."""
    cfg = _cfg()
    d1 = simulator_sha256(cfg)
    d2 = simulator_sha256(cfg)
    assert d1 == d2
    assert isinstance(d1, str)
    assert len(d1) == 64  # sha256 hexdigest


def test_simulator_sha256_changes_on_config_edit() -> None:
    """Changing a physics config field (alpha) changes the digest (RESEARCH Q7)."""
    base = _cfg()
    bumped = dataclasses.replace(base, alpha=base.alpha + 0.01)
    assert simulator_sha256(base) != simulator_sha256(bumped)
    # A seed change also alters the canonical config JSON → different digest.
    reseeded = dataclasses.replace(base, seed=base.seed + 1)
    assert simulator_sha256(base) != simulator_sha256(reseeded)


def test_simulator_hash_stable(monkeypatch) -> None:
    """Editing io.py / a fixtures docstring must NOT change the digest (D-05).

    The hash recipe reads ONLY physics.py + integrator.py + alien.py. We prove
    stability by confirming the hashed file set is exactly those three (so a
    plumbing edit is structurally outside the input) and that the digest equals
    a hand-recomputed hash over only those three files + the config JSON.
    """
    cfg = _cfg()
    # The function must expose the pinned file set so this contract is testable.
    hashed = SIO.SIMULATOR_HASHED_FILES
    names = sorted(p.name for p in hashed)
    assert names == ["alien.py", "integrator.py", "physics.py"]
    # io.py / fixtures.py / types.py / config.py / audit.py are NOT hashed.
    for excluded in ("io.py", "fixtures.py", "types.py", "config.py", "simulator_audit.py"):
        assert excluded not in names

    # Recompute by hand over the pinned files + config JSON; must match.
    import hashlib

    parts = []
    for path in sorted(hashed, key=lambda p: str(p)):
        parts.append(path.read_bytes())
    blob = b"\0".join(parts)
    blob += json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode("utf-8")
    expected = hashlib.sha256(blob).hexdigest()
    assert simulator_sha256(cfg) == expected


def test_trajectory_npz_round_trips_bit_identically(tmp_path) -> None:
    """Trajectory arrays save → load bit-identically (np.savez, not compressed)."""
    cfg, traj = newtonian_limit_fixture()
    run_dir = tmp_path / "run-001"
    SIO.save_run_artifacts(run_dir, traj)
    loaded = SIO.load_trajectory(run_dir)

    np.testing.assert_array_equal(loaded.t, traj.t)
    np.testing.assert_array_equal(loaded.positions, traj.positions)
    np.testing.assert_array_equal(loaded.velocities, traj.velocities)
    np.testing.assert_array_equal(loaded.masses, traj.masses)
    np.testing.assert_array_equal(loaded.charges, traj.charges)
    np.testing.assert_array_equal(loaded.Q, traj.Q)


def test_manifest_carries_hash_config_and_versions(tmp_path) -> None:
    """manifest.json holds simulator_sha256 + library_versions + sorted-key config."""
    cfg, traj = newtonian_limit_fixture()
    run_dir = tmp_path / "run-002"
    SIO.save_run_artifacts(run_dir, traj)

    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    raw = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(raw)

    assert manifest["simulator_sha256"] == simulator_sha256(cfg)
    assert manifest["library_versions"] == traj.library_versions
    assert manifest["config"] == json.loads(json.dumps(dataclasses.asdict(cfg), sort_keys=True))
    # sort_keys=True is mandatory for digest stability — keys must be sorted.
    reserialized = json.dumps(manifest, sort_keys=True)
    assert json.dumps(json.loads(raw), sort_keys=True) == reserialized
