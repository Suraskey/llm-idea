"""Identifiability-driven experiment design (the second novelty pillar, IDENT-03).

PIVOT-001's defensible claim has two halves: (1) use identifiability as an
evaluation diagnostic [identifiability.py / structural.py], and (2) **close the loop
with identifiability-driven experiment design** — when the diagnostic says "this is
non-identifiable from the current data," choose the next measurement that makes the
hidden structure observable. This module is the deterministic, no-LLM core of (2):
given a set of CANDIDATE observation designs, rank them by how well they identify the
target parameters, and recommend the disambiguating one.

This is FIM-based optimal experimental design (systems biology / OED), specialized to
the diagnostic's verdict: a design that restores full rank (or improves conditioning
along the formerly-confounded direction) is preferred. It is the PRINCIPLE behind the
Alien Universe multi-charge-family fix — at a single configuration the hidden charge
product is confounded, so the engine requests configurations at DISTINCT products that
break the degeneracy. Here that move is made explicit, deterministic, and domain-
agnostic: the Council (Burst C) proposes candidate designs; THIS ranks them; the
chosen one is the experiment to run. No model is the judge — the ranking is sympy/
numpy on the agent's OWN proposed structure.

Provably LLM-free (IDENT-04): imports numpy + the local identifiability checker only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ascension.diagnostics.identifiability import (
    IdentifiabilityVerdict,
    PredictFn,
    practical_identifiability,
)

# Condition numbers above this are clamped for scoring (an effectively-unbounded
# sloppy direction shouldn't dominate the score arithmetic).
_COND_CLAMP = 1e12


@dataclass(frozen=True)
class DesignScore:
    """One candidate design's identifiability under the shared operating point."""

    label: str
    identifiable: bool
    rank: int
    n_params: int
    condition_number: float
    # Higher = more identifiable. Rank dominates (each recovered parameter direction
    # is worth a full point); conditioning is a sub-unit tiebreaker so that, among
    # designs of equal rank, the better-conditioned (less noise-sensitive) one wins.
    score: float
    # Non-None iff this candidate design FAILED to evaluate (its predict raised, e.g. a
    # malformed proposal). Such a design is scored worst (-inf) and never recommended,
    # but it does NOT crash the ranking (best-effort; never halt on one bad component).
    error: str | None = None


@dataclass(frozen=True)
class DesignRanking:
    best: str
    ranked: tuple[DesignScore, ...]  # sorted best-first
    rationale: str


def _score(verdict: IdentifiabilityVerdict) -> float:
    cond = verdict.condition_number
    if not math.isfinite(cond) or cond <= 0:
        cond = _COND_CLAMP
    cond = min(cond, _COND_CLAMP)
    # rank in whole points; conditioning in [0,1) so it never outranks a higher rank.
    # log10(cond) ranges ~[0, 12] over the clamp; map to a (0,1] bonus that shrinks as
    # conditioning worsens.
    cond_bonus = 1.0 / (1.0 + math.log10(max(cond, 1.0)))
    return float(verdict.rank) + cond_bonus


def rank_designs_by_identifiability(
    designs: Mapping[str, PredictFn | np.ndarray],
    theta: Sequence[float] | np.ndarray,
    *,
    sigma: float = 1.0,
    rel_step: float = 1e-6,
    param_names: Sequence[str] | None = None,
) -> DesignRanking:
    """Rank candidate observation designs by how well they identify ``theta``.

    Each design is a ``predict(params) -> observations`` closure encoding a DIFFERENT
    observation protocol (which inputs/configurations/times are measured). All are
    evaluated at the SAME parameter operating point ``theta`` (local analysis). The
    design that recovers the most parameter directions (highest rank), best
    conditioned, is recommended as the disambiguating experiment (IDENT-03).

    Args:
      designs: mapping label -> predict closure. Must be non-empty.
      theta: the shared nominal parameter point.
      sigma, rel_step, param_names: forwarded to ``practical_identifiability``.

    Returns:
      DesignRanking (best label + the full sorted scoreboard + a rationale).
    """
    if not designs:
        raise ValueError("rank_designs_by_identifiability needs at least one design")

    n_theta = int(np.asarray(theta, dtype=np.float64).ravel().size)
    scores: list[DesignScore] = []
    for label, predict in designs.items():
        try:
            if isinstance(predict, np.ndarray):
                # EXP-261: a precomputed (m x n) sensitivity matrix for this design
                # (exact forward sensitivities); see practical_identifiability.
                v = practical_identifiability(
                    lambda _th: np.zeros(0), theta, sigma=sigma, rel_step=rel_step,
                    param_names=param_names, sensitivity=predict,
                )
            else:
                v = practical_identifiability(
                    predict, theta, sigma=sigma, rel_step=rel_step, param_names=param_names
                )
        except Exception as exc:  # noqa: BLE001 — a malformed candidate must not halt
            # the whole ranking (no-coverups: best-effort, never halt on one bad
            # component). Score it worst (-inf), capture the error; good designs rank on.
            scores.append(
                DesignScore(
                    label=label,
                    identifiable=False,
                    rank=0,
                    n_params=n_theta,
                    condition_number=float("inf"),
                    score=float("-inf"),
                    error=str(exc),
                )
            )
            continue
        scores.append(
            DesignScore(
                label=label,
                identifiable=v.identifiable,
                rank=v.rank,
                n_params=v.n_params,
                condition_number=v.condition_number,
                score=_score(v),
                error=None,
            )
        )

    # Sort best-first: higher score wins; stable on ties for determinism.
    ranked = tuple(sorted(scores, key=lambda s: s.score, reverse=True))
    best = ranked[0]

    if best.error is not None:
        # Every candidate failed to evaluate — no usable design was proposed.
        rationale = (
            f"ALL {len(ranked)} candidate design(s) FAILED to evaluate (e.g. "
            f"'{best.label}': {best.error}). No usable disambiguating design available."
        )
    elif best.identifiable:
        rationale = (
            f"'{best.label}' fully identifies all {best.n_params} parameters "
            f"(rank {best.rank}/{best.n_params}, cond {best.condition_number:.2e}); "
            f"run it to disambiguate."
        )
    else:
        rationale = (
            f"NO candidate design fully identifies the parameters; '{best.label}' is "
            f"the best available (rank {best.rank}/{best.n_params}). The confounded "
            f"directions persist across all proposed designs — propose a design that "
            f"varies the inputs the confounded combination is insensitive to, or accept "
            f"the parameter is only-jointly-identifiable (e.g. a product/sum) and report "
            f"the identifiable combination, not its factors."
        )

    return DesignRanking(best=best.label, ranked=ranked, rationale=rationale)
