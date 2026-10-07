"""Simulator — Alien Universe custom-physics simulator (numpy/scipy/sympy, no LLM).

Implements the Q-consistent §10 formulation (tangential-inertia coupling),
reconciled into SCOPE §10 (2026-05-23). Populated in Phase 15.0.

Public surface: the DTOs (``AlienConfig``, ``AlienTrajectory``,
``ObservationBundle``) plus the audit-gate façade (``run_simulator_audit``).
``run_simulator_audit`` is resolved lazily (PEP 562 ``__getattr__``) so importing
this package never eagerly pulls ``simulator_audit`` — which keeps the
import-light posture and avoids the ``alien.py ↔ simulator_audit.py`` cycle for
callers that only need the DTOs.
"""

from __future__ import annotations

from ascension.simulator.types import (
    AlienConfig,
    AlienTrajectory,
    BodyState,
    ObservationBundle,
)

__all__ = [
    "AlienConfig",
    "AlienTrajectory",
    "BodyState",
    "ObservationBundle",
    "run_simulator_audit",
]


def __getattr__(name: str):
    """Lazy attribute resolution for the audit façade (PEP 562).

    Keeps the package import-light: ``run_simulator_audit`` is only imported on
    first access, not at package import time.
    """
    if name == "run_simulator_audit":
        from ascension.simulator.simulator_audit import run_simulator_audit

        return run_simulator_audit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
