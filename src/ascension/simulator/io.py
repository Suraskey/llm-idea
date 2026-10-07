"""Per-run artifact serializer: manifest JSON + .npz + the simulator_sha256 digest.

simulator_sha256 hashes only the three dynamics files (physics+integrator+alien)
so the digest is stable across unrelated edits — RESEARCH Q7.

Plain English: every simulator run can be written to disk as two files — a
manifest (the recipe + pinned library versions + the provenance hash) and the
raw trajectory arrays — so a six-months-later reviewer can replay it. The
provenance hash is the load-bearing piece: it fingerprints the PHYSICS of a run
(the force laws, the stepping rule, the assembly, and the exact parameters) so
"does a re-run reproduce this trajectory?" has a one-line answer.

Mirrors ``src/ascension/benchmarks/io.py`` (single-shot JSON + ``np.savez``,
0o700 dir / 0o600 files best-effort, JSON-tuple coercion on load). The one new
thing with no benchmarks analog is ``simulator_sha256`` (D-05 / RESEARCH Q7).

Binding decisions:
  - 15.0 RESEARCH Q7: hash the RAW BYTES of EXACTLY physics.py + integrator.py +
    alien.py (the dynamics) in fixed sorted-path order, ``\\0``-separated, then
    append ``json.dumps(asdict(cfg), sort_keys=True)`` bytes; single
    ``hashlib.sha256(...).hexdigest()``. Do NOT hash io.py / types.py /
    fixtures.py / config.py / simulator_audit.py — editing plumbing must not
    change a physics run's digest (D-05 "stable across unrelated edits").
  - benchmarks/io.py:101-105: ``json.dumps(..., indent=2, sort_keys=True)`` —
    sort_keys is mandatory for digest + manifest stability.
  - benchmarks/io.py:139-141: ``np.savez`` (NOT savez_compressed) — ZIP_STORED
    round-trips bit-identically; ZIP_DEFLATE does not across zlib versions.
  - benchmarks/io.py:188-192: JSON serializes tuples as lists; coerce the
    tuple-typed AlienConfig fields back on load so the reconstructed config
    compares equal.
  - llm/sink.py:76-77 + common/prompts.py:94-99: ``hashlib.sha256`` helper idiom.

Threat-model mitigations:
  - T-15.0-03 (non-reproducible run / unstable digest): pinned 3-file dynamics
    set + sorted-key config JSON; np.savez bit-repro round-trip.
  - file-permission hygiene: 0o700 dir / 0o600 files (best-effort), mirroring
    benchmarks/io.py.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ascension.common.config import settings
from ascension.common.logging import get_logger
from ascension.simulator.config import (
    ARTIFACT_SUBDIR,
    MANIFEST_FILENAME,
    TRAJECTORIES_FILENAME,
)
from ascension.simulator.types import AlienConfig, AlienTrajectory

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# simulator_sha256 — the pinned DYNAMICS file set (RESEARCH Q7)
# -----------------------------------------------------------------------------
# ONLY these three files affect the numbers a run produces: physics.py (forces +
# potential + Q), integrator.py (the stepping rule), alien.py (assembly + the
# integration loop + the polar reduction). Hashing io.py / fixtures.py /
# types.py / config.py / simulator_audit.py would make the digest change on a
# docstring fix, defeating the "stable across unrelated edits" contract (D-05).
_SIMULATOR_DIR = Path(__file__).resolve().parent
SIMULATOR_HASHED_FILES: tuple[Path, ...] = (
    _SIMULATOR_DIR / "physics.py",
    _SIMULATOR_DIR / "integrator.py",
    _SIMULATOR_DIR / "alien.py",
)


def simulator_sha256(cfg: AlienConfig) -> str:
    """Provenance digest over the dynamics source + the run's config (RESEARCH Q7).

    Recipe (deterministic, stable across unrelated edits):
      1. Read the RAW BYTES (not text — avoids newline normalization) of EXACTLY
         the three dynamics files in FIXED SORTED PATH order:
         ``alien.py``, ``integrator.py``, ``physics.py``.
      2. Concatenate them with a ``\\0`` separator.
      3. Append ``json.dumps(dataclasses.asdict(cfg), sort_keys=True)`` bytes.
      4. Single ``hashlib.sha256(...).hexdigest()``.

    The hashed file set is exactly ``SIMULATOR_HASHED_FILES``. Editing io.py,
    fixtures.py, types.py, config.py, or simulator_audit.py does NOT change the
    digest; editing a force law / the integrator / the assembly, or any config
    field, DOES.

    Args:
      cfg: the ``AlienConfig`` for the run being fingerprinted.

    Returns:
      64-char hex sha256 digest.
    """
    parts: list[bytes] = []
    for path in sorted(SIMULATOR_HASHED_FILES, key=lambda p: str(p)):
        parts.append(path.read_bytes())
    blob = b"\0".join(parts)
    blob += json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# -----------------------------------------------------------------------------
# Run-directory + permission helpers (mirror benchmarks/io.py)
# -----------------------------------------------------------------------------


def simulator_run_dir(run_id: str) -> Path:
    """Return settings.RUN_ARTIFACTS_DIR / 'simulator' / run_id, created 0o700.

    Each run gets its own folder; same run_id twice is idempotent (no clobber).
    0o700 keeps the dir owner-only (best-effort — non-fatal on CI filesystems
    that restrict chmod).
    """
    root = Path(str(settings.RUN_ARTIFACTS_DIR)) / ARTIFACT_SUBDIR / run_id
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except PermissionError:
        logger.debug("could not chmod 0o700 on %s", root)
    return root


def _write_0600(path: Path, content: str | bytes) -> None:
    """Write file then chmod 0o600 (best-effort)."""
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    try:
        os.chmod(path, 0o600)
    except PermissionError:
        logger.debug("could not chmod 0o600 on %s", path)


def _manifest_json(traj: AlienTrajectory) -> str:
    """Serialize the run manifest: config + library_versions + simulator_sha256.

    sort_keys=True is mandatory — it keeps the manifest stable and makes the
    embedded config JSON identical to the one fed into ``simulator_sha256``.
    """
    return json.dumps(
        {
            "config": dataclasses.asdict(traj.config),
            "library_versions": traj.library_versions,
            "simulator_sha256": simulator_sha256(traj.config),
        },
        indent=2,
        sort_keys=True,
    )


def save_run_artifacts(run_dir: Path, traj: AlienTrajectory) -> None:
    """Write manifest.json + trajectories.npz for one simulator run.

    Args:
      run_dir: directory to write into (created if absent).
      traj: the FULL ``AlienTrajectory`` to serialize. The hidden charges + Q
        are part of the on-disk record (this is the internal artifact, NOT the
        data-only ObservationBundle the Council sees).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_0600(run_dir / MANIFEST_FILENAME, _manifest_json(traj))

    # np.savez (NOT compressed) — ZIP_STORED round-trips bit-identically
    # (benchmarks/io.py:139-141 / RESEARCH §Don't Hand-Roll).
    traj_path = run_dir / TRAJECTORIES_FILENAME
    np.savez(
        traj_path,
        t=traj.t,
        positions=traj.positions,
        velocities=traj.velocities,
        masses=traj.masses,
        charges=traj.charges,
        Q=traj.Q,
    )
    try:
        os.chmod(traj_path, 0o600)
    except PermissionError:
        logger.debug("could not chmod 0o600 on %s", traj_path)


def _config_from_dict(d: dict) -> AlienConfig:
    """Rehydrate an AlienConfig from a JSON-loaded dict (tuple-coercion gotcha).

    JSON serializes tuples as lists; coerce the tuple-typed fields back so the
    reconstructed AlienConfig compares equal to the original
    (benchmarks/io.py:188-192).
    """
    d = dict(d)
    d["masses"] = tuple(d["masses"])
    d["charges"] = tuple(d["charges"])
    d["ics_pos"] = tuple(tuple(row) for row in d["ics_pos"])
    d["ics_vel"] = tuple(tuple(row) for row in d["ics_vel"])
    return AlienConfig(**d)


def load_trajectory(run_dir: Path) -> AlienTrajectory:
    """Inverse of save_run_artifacts: reconstruct the FULL AlienTrajectory.

    Trajectory arrays come back bit-identical (np.testing.assert_array_equal);
    the config compares equal after tuple-coercion; library_versions are read
    from the manifest.
    """
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    cfg = _config_from_dict(manifest["config"])
    library_versions = dict(manifest["library_versions"])

    traj_path = run_dir / TRAJECTORIES_FILENAME
    with np.load(traj_path) as data:
        # Copy arrays out before the NpzFile context closes.
        return AlienTrajectory(
            config=cfg,
            t=np.asarray(data["t"]),
            positions=np.asarray(data["positions"]),
            velocities=np.asarray(data["velocities"]),
            masses=np.asarray(data["masses"]),
            charges=np.asarray(data["charges"]),
            Q=np.asarray(data["Q"]),
            library_versions=library_versions,
        )
