"""Typed exceptions for the simulator module.

Only HOST-LEVEL failures raise. Per-property simulator failures are
reason-string codes on a result DTO (``simulator_audit.SimulatorAuditResult``),
NOT exceptions. This is the same discipline as
``src/ascension/benchmarks/exceptions.py`` (03.0 D-11 analog): a Q-conservation
violation, a polynomial-fittable hidden force, or a leaked hidden field is data
on the audit's output, not a stack-unwinding event on the host.

Plain English: "exceptions here" mean the whole run is broken — the integrator
could not advance the state (a NaN/Inf blew up near perihelion, or the inner
solve diverged). They are NOT for "the trajectory failed a physics property" —
that's a result with a reason code, not a crash.

Classes that DO NOT exist here and MUST NOT be added:
  - ``ConservationViolation``, ``DriftError``
  - ``PolynomialFittable``
  - ``HiddenStateLeak``
  - ``HashMismatch``

Those failure modes are STRING reason codes on the
``SimulatorAuditResult`` DTO (PATTERNS §reason_codes / §simulator_audit).
Adding them as classes here would break the D-11-analog divergence rule and
poison downstream stats with exceptions that should be data.

Binding decisions:
  - 15.0 PATTERNS §exceptions.py: base class (``SimulatorError``) +
    host-level-only subclass (``SimulatorIntegrationError``).
  - 15.0 RESEARCH §Pitfall 2 (blowup near perihelion): non-finite state on the
    TRUTH integrator raises loudly (no cover-ups, CLAUDE.md).
  - SCOPE.md:630 (§13): failed executions return structured failure objects,
    not silent crashes — satisfied by the reason-code DTO, not by raising for
    per-property failures.

Threat-model mitigations: T-15.0-04 (integrator non-convergence / blowup) —
``SimulatorIntegrationError`` surfaces the failure loud + replayable, never
silent.
"""

from __future__ import annotations


class SimulatorError(Exception):
    """Base class. All simulator-module typed errors subclass this."""


class SimulatorIntegrationError(SimulatorError):
    """Raised when the integrator cannot advance the state of a run.

    Two host-level conditions raise this:
      1. The implicit-midpoint inner fixed-point solve produced a non-finite
         (NaN/Inf) state — a blowup, typically near perihelion of a
         high-eccentricity orbit (15.0 RESEARCH Pitfall 2).
      2. (reserved) Any other condition that makes the trajectory itself
         unusable for any downstream consumer.

    This is a HOST-LEVEL failure — surfaced loud (exception) rather than
    stuffed into a reason code, because a trajectory containing NaN is not
    recoverable by any downstream check.

    Per-property simulator failures (Q drift past bound, polynomial-fittable
    hidden force, ObservationBundle leak) are NOT exceptions — they are string
    reason codes on the ``SimulatorAuditResult`` DTO. See PATTERNS §simulator_audit.

    Stores ``config`` + ``seed`` + ``message`` so the run is replayable from
    the exception alone (SCOPE §22.2 reproducibility). ``config`` is held as an
    opaque object (the ``AlienConfig``) — exceptions.py stays import-light and
    does not import types.py to avoid any cycle.
    """

    def __init__(self, config: object, seed: int, message: str) -> None:
        self.config = config
        self.seed = seed
        self.message = message
        super().__init__(f"SimulatorIntegrationError seed={seed} msg={message!r}")
