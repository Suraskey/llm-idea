"""Ascension diagnostics — the pre-flight identifiability checker (PIVOT-001 Burst B).

The pivot's Diagnostic pillar: deterministic, no-LLM analysis that tells
*"unsolvable from the data I have"* apart from *"not solved yet."* Given a model
class and an observation protocol, decide whether the target structure is
recoverable, and — when it is not — characterize the degeneracy (which parameter
combinations collapse) and suggest the data regime that would restore
identifiability.

This package generalizes the alien-specific ``scripts/audit_depth_discoverability.py``
(L-051) into a domain-agnostic instrument (REQUIREMENT IDENT-01..04). It is the
piece nobody in the agent-discovery world uses and the likely commercial core, so
it lives in ``src/`` (importable, tested), not in ``scripts/``.

Provably LLM-free (IDENT-04): this package imports only numpy + stdlib. No
``ascension.llm``, no ``google.genai``. A grep gate in the test suite enforces it.
"""

from __future__ import annotations

from ascension.diagnostics.engine import (
    LoopResult,
    LoopStep,
    run_identifiability_loop,
)
from ascension.diagnostics.exhaustion import (
    ExhaustionCertificate,
    PlateauTriage,
    certify_class_symmetry_numeric,
    certify_class_symmetry_symbolic,
    triage_plateau,
)
from ascension.diagnostics.identifiability import (
    IdentifiabilityVerdict,
    practical_identifiability,
    sensitivity_matrix,
)
from ascension.diagnostics.optimal_design import (
    DesignRanking,
    DesignScore,
    rank_designs_by_identifiability,
)
from ascension.diagnostics.structural import (
    StructuralVerdict,
    structural_identifiability,
)

__all__ = [
    "DesignRanking",
    "DesignScore",
    "ExhaustionCertificate",
    "IdentifiabilityVerdict",
    "PlateauTriage",
    "certify_class_symmetry_numeric",
    "certify_class_symmetry_symbolic",
    "triage_plateau",
    "LoopResult",
    "LoopStep",
    "StructuralVerdict",
    "practical_identifiability",
    "rank_designs_by_identifiability",
    "run_identifiability_loop",
    "sensitivity_matrix",
    "structural_identifiability",
    "triage_plateau",
]
