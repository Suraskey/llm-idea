"""The identifiability-driven experiment loop — the Engine, domain-agnostic (ACTEXP).

This is the pivot thesis as runnable control flow. On a plateau, it does NOT blindly
retry (ACTEXP-01): it runs the identifiability diagnostic and acts on the verdict.

  * IDENTIFIABLE  → "not solved yet." The data DOES determine the structure; the
                    plateau is a capability/reasoning limit. The loop stops and hands
                    back: keep reasoning, no new data will help.
  * NON-IDENTIFIABLE → "unsolvable from the data I have." The plateau is an
                    OBSERVABILITY limit. The loop asks the proposer (the Council —
                    NEVER Argus, ACTEXP-02) for candidate experiments/designs, uses the
                    identifiability-driven ranker to pick the one that best breaks the
                    degeneracy, services it (gather data + fit the proposer's OWN
                    structure, ACTEXP-03/04), and re-diagnoses. Repeat until identifiable
                    or out of rounds.

Knowing the difference between those two — and acting on it — is the defining skill the
thesis names. This module makes it explicit, deterministic at its core, and reusable.

Dependency injection keeps the CORE no-LLM and domain-agnostic (so it stays in the
open-sourceable instrument, dissemination Track A): the LLM proposer and the
simulator+fitter are passed in as callables. In production the proposer is the Council
and the servicer wraps the simulator + the deterministic fitter; in tests they are
stubs. The ranking + the diagnosis are sympy/numpy — no model judges (the §22.6 / §17
invariants: Argus never sets direction; the value comes from the data, not a leaked
oracle).

Provably LLM-free (IDENT-04): imports numpy + the local diagnostic/OED modules only.
The injected proposer MAY call an LLM, but this module does not import one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ascension.diagnostics.identifiability import PredictFn, practical_identifiability
from ascension.diagnostics.optimal_design import rank_designs_by_identifiability

# (verdict) -> candidate designs {label: predict}. The Council's move (ACTEXP-02).
# Receives the diagnosis so it can target the confounded direction. Returns {} when it
# has no proposal (the loop then stops, honestly stuck).
ProposeFn = Callable[[object], Mapping[str, PredictFn]]

# (chosen_label, chosen_predict) -> the NEW effective predict after gathering data under
# the chosen design and fitting the proposer's own structure to it (ACTEXP-03/04). When
# omitted, the loop simply adopts the chosen design's predict (a pure design switch).
ServiceFn = Callable[[str, PredictFn], PredictFn]

# (verdict) -> certificate per confounded direction, or None to skip. When EVERY
# confounded direction is certified as a symmetry of the whole design class, no
# experiment the proposer can offer inside that class will help, and the loop stops
# at once with outcome "exhausted_by_certificate" (a proof, not a budget). See
# diagnostics/exhaustion.py. Optional; the loop is unchanged when omitted.
CertifyFn = Callable[[object], Sequence[object]]


@dataclass(frozen=True)
class LoopStep:
    round: int
    identifiable: bool
    status: str  # the diagnostic status ("identifiable" / "structurally_..." / "practically_...")
    rank: int
    n_params: int
    proposed: tuple[str, ...]  # design labels offered this round ({} → stuck)
    chosen: (
        str | None
    )  # the OED-selected disambiguating design (None if identifiable / none proposed)


@dataclass(frozen=True)
class LoopResult:
    steps: tuple[LoopStep, ...]
    final_identifiable: bool
    rounds_used: int
    # "solved" (became identifiable), "capability_plateau" (identifiable from round 0 —
    # the data already suffices, keep reasoning), "no_proposal" (proposer gave up),
    # "exhausted" (hit max_rounds still non-identifiable), "solved_up_to_class_symmetry"
    # (experiments reached the class rank; only certified directions remain),
    # "exhausted_by_certificate"
    # (every confounded direction PROVEN to be a symmetry of the whole design class:
    # change the class or report the identifiable combination; no budget was spent).
    outcome: str
    rationale: str


def run_identifiability_loop(
    predict: PredictFn,
    theta: Sequence[float] | np.ndarray,
    *,
    propose_designs: ProposeFn,
    service_and_refit: ServiceFn | None = None,
    certify_exhaustion: CertifyFn | None = None,
    max_rounds: int = 3,
    sigma: float = 1.0,
    rel_step: float = 1e-6,
    param_names: Sequence[str] | None = None,
) -> LoopResult:
    """Run the diagnose → (design-experiment) → re-diagnose loop on a plateau.

    Args:
      predict: the current model under the current observation design (params ->
        observations). The plateau the loop is invoked on.
      theta: the parameter operating point.
      propose_designs: ACTEXP-02. Given the diagnosis, returns candidate designs. The
        Council in production; never Argus (§17). Returns {} to signal "no proposal".
      service_and_refit: ACTEXP-03/04. Given the chosen design, gather data + fit the
        proposer's own structure and return the new effective predict. None → adopt the
        chosen design directly (a design switch with no separate refit step).
      certify_exhaustion: optional. Given a non-identifiable verdict, return one
        ExhaustionCertificate per confounded direction (diagnostics/exhaustion.py). If
        every certificate is ``certified`` the loop returns "exhausted_by_certificate"
        immediately: the degeneracy is a symmetry of the entire design class, so no
        proposal inside the class can break it. Skipped when None.
      max_rounds: cap on design-experiment rounds (a budget rail).
      sigma, rel_step, param_names: forwarded to the diagnostic.

    Returns:
      LoopResult — the per-round trail + the terminal outcome.
    """
    steps: list[LoopStep] = []

    for r in range(max_rounds + 1):
        verdict = practical_identifiability(
            predict, theta, sigma=sigma, rel_step=rel_step, param_names=param_names
        )

        if verdict.identifiable:
            steps.append(
                LoopStep(
                    round=r,
                    identifiable=True,
                    status=verdict.status,
                    rank=verdict.rank,
                    n_params=verdict.n_params,
                    proposed=(),
                    chosen=None,
                )
            )
            if r == 0:
                return LoopResult(
                    steps=tuple(steps),
                    final_identifiable=True,
                    rounds_used=r,
                    outcome="capability_plateau",
                    rationale=(
                        "Identifiable from the current data at round 0: the data DOES "
                        "determine the structure, so this plateau is a capability/"
                        "reasoning limit, not an observability one. No new experiment "
                        "will help — keep reasoning."
                    ),
                )
            return LoopResult(
                steps=tuple(steps),
                final_identifiable=True,
                rounds_used=r,
                outcome="solved",
                rationale=(
                    f"Identifiability RESTORED after {r} design-experiment round(s): the "
                    f"chosen experiment(s) made the previously-confounded structure "
                    f"observable. The data now determines it."
                ),
            )

        # Non-identifiable: an OBSERVABILITY plateau. First, if a certifier is wired,
        # ask whether the degeneracy is a symmetry of the WHOLE design class — if so,
        # no experiment of this kind can help and we stop by proof, not by budget.
        if certify_exhaustion is not None:
            certs = tuple(certify_exhaustion(verdict))
            if certs and all(bool(getattr(c, "certified", False)) for c in certs):
                steps.append(
                    LoopStep(
                        round=r,
                        identifiable=False,
                        status=verdict.status,
                        rank=verdict.rank,
                        n_params=verdict.n_params,
                        proposed=(),
                        chosen=None,
                    )
                )
                if r > 0:
                    # Experiments already removed every direction the class CAN resolve;
                    # what is left is a class-level symmetry (Proposition 2 of the paper:
                    # the class rank, not full rank, is the reachable target).
                    return LoopResult(
                        steps=tuple(steps),
                        final_identifiable=False,
                        rounds_used=r,
                        outcome="solved_up_to_class_symmetry",
                        rationale=(
                            f"After {r} design-experiment round(s) the design reaches the "
                            f"class rank ({verdict.rank} of {verdict.n_params}); the "
                            f"{len(certs)} remaining confounded direction(s) are certified "
                            f"symmetries of the whole design class. Report the identifiable "
                            f"combinations; only a change of class can go further."
                        ),
                    )
                return LoopResult(
                    steps=tuple(steps),
                    final_identifiable=False,
                    rounds_used=r,
                    outcome="exhausted_by_certificate",
                    rationale=(
                        f"Non-identifiable ({verdict.status}) and every confounded "
                        f"direction is certified as a symmetry of the whole design "
                        f"class ({len(certs)} certificate(s)): no experiment of this kind "
                        f"can resolve it. Change the design class (a new observable or "
                        f"structural element) or report the identifiable combination."
                    ),
                )

        if r == max_rounds:
            steps.append(
                LoopStep(
                    round=r,
                    identifiable=False,
                    status=verdict.status,
                    rank=verdict.rank,
                    n_params=verdict.n_params,
                    proposed=(),
                    chosen=None,
                )
            )
            return LoopResult(
                steps=tuple(steps),
                final_identifiable=False,
                rounds_used=r,
                outcome="exhausted",
                rationale=(
                    f"Still non-identifiable after {max_rounds} round(s) "
                    f"({verdict.status}): no proposed experiment broke the degeneracy "
                    f"within budget. Report the identifiable combination, not its "
                    f"factors, or widen the design space."
                ),
            )

        designs = dict(propose_designs(verdict))
        if not designs:
            steps.append(
                LoopStep(
                    round=r,
                    identifiable=False,
                    status=verdict.status,
                    rank=verdict.rank,
                    n_params=verdict.n_params,
                    proposed=(),
                    chosen=None,
                )
            )
            return LoopResult(
                steps=tuple(steps),
                final_identifiable=False,
                rounds_used=r,
                outcome="no_proposal",
                rationale=(
                    "Non-identifiable and the proposer offered no candidate experiment. "
                    "Stuck honestly — the degeneracy persists and no design was proposed."
                ),
            )

        ranking = rank_designs_by_identifiability(
            designs, theta, sigma=sigma, rel_step=rel_step, param_names=param_names
        )
        # If EVERY proposed design failed to evaluate (all predicts raised), there is no
        # usable experiment this round — stop honestly rather than adopt a broken design
        # (which would crash the next diagnosis). The ranker scores failures -inf, so a
        # best with .error set means all candidates failed.
        if ranking.ranked[0].error is not None:
            steps.append(
                LoopStep(
                    round=r,
                    identifiable=False,
                    status=verdict.status,
                    rank=verdict.rank,
                    n_params=verdict.n_params,
                    proposed=tuple(designs.keys()),
                    chosen=None,
                )
            )
            return LoopResult(
                steps=tuple(steps),
                final_identifiable=False,
                rounds_used=r,
                outcome="no_proposal",
                rationale=(
                    "Non-identifiable, and every proposed candidate design FAILED to "
                    f"evaluate ({ranking.rationale}). No usable experiment — stopped."
                ),
            )
        chosen = ranking.best
        steps.append(
            LoopStep(
                round=r,
                identifiable=False,
                status=verdict.status,
                rank=verdict.rank,
                n_params=verdict.n_params,
                proposed=tuple(designs.keys()),
                chosen=chosen,
            )
        )

        # Service the chosen experiment (gather data + fit the proposer's own structure),
        # then re-diagnose the new effective model on the next iteration.
        chosen_predict = designs[chosen]
        predict = (
            service_and_refit(chosen, chosen_predict)
            if service_and_refit is not None
            else chosen_predict
        )

    # Unreachable (the r == max_rounds branch returns), but keeps the type checker happy.
    return LoopResult(  # pragma: no cover
        steps=tuple(steps),
        final_identifiable=False,
        rounds_used=max_rounds,
        outcome="exhausted",
        rationale="loop terminated",
    )
