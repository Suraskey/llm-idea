"""Typed exceptions for the benchmarks module.

Only HOST-LEVEL failures raise. Per-run scoring failures are reason-string
DTO fields on the Plan 02 `BenchmarkScore` record — NOT exceptions. This is
the same discipline as `src/ascension/sandbox/exceptions.py` (03.0 D-11):
a per-trajectory blowup or a sympy timeout is data on the scorer's output,
not a stack-unwinding event on the host.

Plain English: "exceptions here" means the whole phase is broken — the
reference PySINDy integrator couldn't run, or someone asked for a system
name that doesn't exist. They are NOT for "the model the agent proposed
is bad" — that's a score with a reason code, not a crash.

Classes that DO NOT exist here and MUST NOT be added:
  - ``ScoreFailure``, ``ScoringError``
  - ``SymbolicTimeout``, ``SymbolicParseFail``
  - ``NumericalBlowup``, ``NumericalDiverged``
  - ``QualitativeUnknown``

Those failure modes are STRING reason codes on the Plan 02 `BenchmarkScore`
DTO (PATTERNS §Shared Pattern D). Adding them as classes here would break
the D-11-analog divergence rule and poison downstream ablation stats with
exceptions that should be data.

Binding decisions:
  - 06.0 RESEARCH §Pattern 3 lines 338-373: reason-string-on-DTO for per-run
    failure modes.
  - 06.0 PATTERNS §`src/ascension/benchmarks/exceptions.py`: three host-level
    classes only (base + UnknownBenchmarkSystem + BenchmarkIntegrationError).
  - CLAUDE.md "no cover-ups": sol.success=False on TRUTH raises loudly.

Pitfalls honored:
  - RESEARCH #8 (blowup detection on proposed ODEs): but only for the
    TRUTH integrator — proposed-ODE blowups are Plan 02 reason codes.

Threat-model mitigations: N/A — pure-compute phase, no network/auth/secret
surface per RESEARCH §Security Domain lines 690–702.
"""

from __future__ import annotations


class BenchmarkError(Exception):
    """Base class. All benchmarks-module typed errors subclass this."""


class UnknownBenchmarkSystem(BenchmarkError):
    """Raised when a `BenchmarkSpec.system` value has no entry in DEFAULT_PARAMS.

    Message format: ``'Unknown benchmark system: <name>'`` — the regex
    downstream callers key on is the literal ``Unknown benchmark system:``
    prefix.
    """


class BenchmarkIntegrationError(BenchmarkError):
    """Raised when scipy.integrate.solve_ivp returns sol.success=False on the
    TRUTH system during generate_trajectories().

    This is a HOST-LEVEL failure — the PySINDy reference RHS cannot be
    integrated, which means the benchmark itself is unusable this run.
    Surfaced loud (exception) rather than stuffed into a reason code: a
    trajectory bundle with an unintegrable truth is not recoverable by any
    downstream scorer.

    Per-run scoring failures (symbolic timeouts, proposed-ODE blowups, etc.)
    are NOT exceptions — they are string reason codes on the Plan 02
    `BenchmarkScore.reasons` dict. See PATTERNS §Shared Pattern D.

    Stores ``system`` + ``seed`` + ``message`` so the run is replayable from
    the exception alone (SCOPE §22.2 reproducibility).
    """

    def __init__(self, system: str, seed: int, message: str) -> None:
        self.system = system
        self.seed = seed
        self.message = message
        super().__init__(f"BenchmarkIntegrationError system={system!r} seed={seed} msg={message!r}")
