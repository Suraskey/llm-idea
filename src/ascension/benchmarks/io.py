"""Per-run artifact serializer: JSON + .npz on local disk.

Plain English: every benchmark run writes three files side by side — the
problem spec, the raw trajectory arrays, and the score result. Anyone re-
running the pinned scipy/sympy/pysindy versions against these files should
recompute the same answers. That's the reproducibility contract (SCOPE §22.2).

Binding decisions:
  - 06.0 RESEARCH §Architecture Patterns diagram lines 195–208 (per-run dir
    layout under settings.RUN_ARTIFACTS_DIR/benchmarks/<run_id>/).
  - 06.0 PATTERNS §io.py: divergence from sandbox sink — single-shot JSON +
    .npz, NOT async JSONL envelope + finalize. Pure compute, no async
    lifecycle, no secret scrubbing (benchmarks path touches no secrets).
  - Shared Pattern: artifact files are mode 0o600 (matches sandbox sink
    hygiene); run dir is 0o700. Both are best-effort — some CI filesystems
    don't honor chmod and we log.debug rather than fail.

Pitfalls honored:
  - RESEARCH §Don't Hand-Roll lines 441–442: numpy.savez (NOT
    savez_compressed) for deterministic bit-reproducibility — ZIP_STORED
    entries round-trip identically, ZIP_DEFLATE does not.

Threat-model mitigations:
  - T-06.0-12 (other users on shared host reading benchmark artifacts): files
    written with 0o600, dir with 0o700 (best-effort).
  - T-06.0-13 (loaded artifact differs from original): round-trip regression
    tests assert np.testing.assert_array_equal on all trajectory arrays and
    exact equality on BenchmarkScore fields. library_versions dict preserved
    in spec.json for version-audit trail.
  - T-06.0-11 (path-traversal via run_id): Phase 6.0 only caller is
    smoke.py which uses uuid.uuid4(); sanitization deferred to Phase 7.0
    CONTEXT when external callers reach this surface.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from ascension.benchmarks.config import (
    AGENT_RESULT_FILENAME,
    ARTIFACT_SUBDIR,
    PYSINDY_RESULT_FILENAME,
    SPEC_FILENAME,
    TRAJECTORIES_FILENAME,
)
from ascension.benchmarks.score_types import BenchmarkScore
from ascension.benchmarks.types import BenchmarkSpec, TrajectoryBundle
from ascension.common.config import settings
from ascension.common.logging import get_logger

logger = get_logger(__name__)


def benchmark_run_dir(run_id: str) -> Path:
    """Return settings.RUN_ARTIFACTS_DIR / 'benchmarks' / run_id, created 0o700.

    Plain English: each benchmark run gets its own folder under the project's
    runs/benchmarks/ tree. Same run_id twice is idempotent (we don't clobber
    an existing dir). 0o700 means only the owner user can list the dir —
    defense against shared-host snooping (T-06.0-12).
    """
    root = Path(str(settings.RUN_ARTIFACTS_DIR)) / ARTIFACT_SUBDIR / run_id
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except PermissionError:
        # Non-fatal on CI/test environments where chmod may be restricted.
        logger.debug("could not chmod 0o700 on %s", root)
    return root


def _write_0600(path: Path, content: str | bytes) -> None:
    """Write file then chmod 0o600 (best-effort).

    Plain English: write the bytes, then lock the file to owner-only read/write.
    If the filesystem doesn't support chmod, we log and continue — the data
    is still on disk.
    """
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    try:
        os.chmod(path, 0o600)
    except PermissionError:
        logger.debug("could not chmod 0o600 on %s", path)


def _spec_to_json(spec: BenchmarkSpec, library_versions: dict[str, str]) -> str:
    """Serialize BenchmarkSpec + library_versions as indented JSON.

    Plain English: the "what was this run" card — all seven BenchmarkSpec
    fields plus the pinned library versions, so a six-months-later reviewer
    can read off exactly which scipy/sympy/pysindy/numpy was in use.
    """
    return json.dumps(
        {"spec": dataclasses.asdict(spec), "library_versions": library_versions},
        indent=2,
        sort_keys=True,
    )


def _score_to_json(score: BenchmarkScore) -> str:
    """Serialize BenchmarkScore as indented JSON."""
    return json.dumps(dataclasses.asdict(score), indent=2, sort_keys=True)


def save_run_artifacts(
    run_dir: Path,
    spec: BenchmarkSpec,
    bundle: TrajectoryBundle,
    pysindy_score: BenchmarkScore,
    agent_score: BenchmarkScore | None = None,
) -> None:
    """Write spec.json + trajectories.npz + pysindy_result.json (+ agent_result.json).

    Plain English: dumps the full record of one benchmark run — the recipe,
    the raw numbers, and the score — to disk in three (or four) files. Same
    run_dir twice re-writes atomically; intended for one-shot serialization,
    not streaming.

    Args:
      run_dir: Directory to write into. Created if absent.
      spec: BenchmarkSpec used to generate the bundle.
      bundle: TrajectoryBundle with clean + noisy + ICs + train_mask +
        library_versions.
      pysindy_score: BenchmarkScore from run_pysindy_baseline(bundle).
      agent_score: Optional BenchmarkScore from a Phase 7.0+ agent loop.
        When provided, a fourth file ``agent_result.json`` is written.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_0600(run_dir / SPEC_FILENAME, _spec_to_json(spec, bundle.library_versions))

    # numpy.savez (NOT compressed) — per RESEARCH §Don't Hand-Roll deterministic
    # bit-repro. ZIP_STORED round-trips identically; ZIP_DEFLATE does not on all
    # platforms/zlib versions.
    traj_path = run_dir / TRAJECTORIES_FILENAME
    np.savez(
        traj_path,
        t=bundle.t,
        clean=bundle.clean,
        noisy=bundle.noisy,
        ics=bundle.ics,
        train_mask=bundle.train_mask,
    )
    try:
        os.chmod(traj_path, 0o600)
    except PermissionError:
        logger.debug("could not chmod 0o600 on %s", traj_path)

    _write_0600(run_dir / PYSINDY_RESULT_FILENAME, _score_to_json(pysindy_score))
    if agent_score is not None:
        _write_0600(run_dir / AGENT_RESULT_FILENAME, _score_to_json(agent_score))


def _score_from_dict(d: dict) -> BenchmarkScore:
    """Rehydrate a BenchmarkScore from a JSON-loaded dict.

    Plain English: inverse of _score_to_json — takes a {exact, rmse,
    qualitative, reasons, elapsed_ms} dict (with rmse potentially None) and
    returns a fresh BenchmarkScore instance.
    """
    return BenchmarkScore(
        exact=bool(d["exact"]),
        rmse=(None if d["rmse"] is None else float(d["rmse"])),
        qualitative=dict(d["qualitative"]),
        reasons=dict(d["reasons"]),
        elapsed_ms=dict(d["elapsed_ms"]),
    )


def load_run_artifacts(
    run_dir: Path,
) -> tuple[BenchmarkSpec, TrajectoryBundle, BenchmarkScore, BenchmarkScore | None]:
    """Inverse of save_run_artifacts. Agent slot may be missing (returns None).

    Plain English: read the three (or four) files back and reconstruct the
    original objects. Trajectory arrays come back bit-identical; spec and
    score fields compare equal to the originals.
    """
    spec_payload = json.loads((run_dir / SPEC_FILENAME).read_text(encoding="utf-8"))
    spec_dict = spec_payload["spec"]
    # JSON serializes tuples as lists — coerce t_span_override back to tuple so
    # the reconstructed BenchmarkSpec compares equal to the original.
    if spec_dict.get("t_span_override") is not None:
        spec_dict["t_span_override"] = tuple(spec_dict["t_span_override"])
    spec = BenchmarkSpec(**spec_dict)

    traj_path = run_dir / TRAJECTORIES_FILENAME
    with np.load(traj_path) as data:
        # np.load returns NpzFile; copy arrays out before the context exits so
        # the caller doesn't accidentally use closed-file handles.
        bundle = TrajectoryBundle(
            spec=spec,
            t=np.asarray(data["t"]),
            clean=np.asarray(data["clean"]),
            noisy=np.asarray(data["noisy"]),
            ics=np.asarray(data["ics"]),
            train_mask=np.asarray(data["train_mask"]),
            library_versions=dict(spec_payload["library_versions"]),
        )

    pysindy_score = _score_from_dict(
        json.loads((run_dir / PYSINDY_RESULT_FILENAME).read_text(encoding="utf-8")),
    )
    agent_path = run_dir / AGENT_RESULT_FILENAME
    agent_score = (
        _score_from_dict(json.loads(agent_path.read_text(encoding="utf-8")))
        if agent_path.exists()
        else None
    )
    return spec, bundle, pysindy_score, agent_score
