"""Practical identifiability via the sensitivity matrix / Fisher information.

PIVOT-001 Burst B, IDENT-01..04. This is the GENERAL fallback the novelty
pressure-test (2026-05-31) flagged as load-bearing: structural-identifiability
tools assume rational-function ODEs, but an LLM's hypothesis can be black-box or
non-ODE. Practical identifiability needs only that you can *evaluate* the model's
predictions as a function of its parameters — it makes no assumption about the
model's algebraic form — so it covers the cases clean structural analysis cannot.

The idea (standard in systems biology / OED, e.g. FIM-based optimal experimental
design): a model is *locally identifiable* at a parameter point theta iff distinct
parameter values produce distinguishable predictions under the observation design.
The local linearization of that statement is the sensitivity matrix

    S[i, j] = d prediction_i / d theta_j   (evaluated at theta),

whose Fisher information is ``FIM = S^T S / sigma^2``. Then:

  * ``rank(S) < n_params``        => STRUCTURALLY non-identifiable at theta: some
                                     parameter combination changes nothing in the
                                     predictions. The right-singular vectors for
                                     the zero singular values ARE those combinations
                                     (IDENT-02 — the degeneracy, named).
  * full rank but huge condition  => PRACTICALLY non-identifiable ("sloppy"): a
    number (s_max / s_min)          direction is technically observable but so
                                     weakly that noise swamps it. The smallest
                                     singular direction is the sloppy one.
  * full rank, well conditioned   => identifiable at theta.

This is the linearized worked example of L-051: at equal alien charges the hidden
product ``s1*s2`` has sensitivity columns ``d/ds1 = s2*(...)`` and
``d/ds2 = s1*(...)`` that become collinear at ``s1 == s2`` — rank drops, and the
confounded direction is exactly ``[1, -1]`` (you can trade s1 for s2 with no
observable effect). Distinct charges across a family break the collinearity and
restore rank — which is the multi-charge-family fix, recovered from first
principles by the same instrument.

Scope/limits stated honestly (the boundary is itself a finding, per Burst B):
  * LOCAL analysis at the supplied ``theta`` — it certifies local, not global,
    identifiability. A globally-confounded model can look locally fine away from
    the symmetry point (and vice versa); sample a few theta if global behavior
    matters.
  * Finite-difference sensitivities assume the predict() map is smooth near theta.
    For noisy/stochastic simulators, supply an averaged or analytic predict().
  * Uniform-Gaussian-noise FIM. Heteroscedastic noise scales columns differently;
    pass a pre-whitened predict() (divide each observation by its sigma) if needed.

Provably LLM-free (IDENT-04): imports numpy + stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

# A model under an observation design: maps a parameter vector to the flat vector
# of predicted observations the design produces. The caller bakes the observation
# protocol (which inputs/times/configs are measured) into this closure, so the
# diagnostic stays domain-agnostic (IDENT-01: "given a model class + protocol").
PredictFn = Callable[[np.ndarray], np.ndarray]

# Default: a singular direction counts as "collapsed" (non-identifiable) when its
# singular value is below this fraction of the largest. 1e-9 ~ double-precision
# rank floor; anything below is numerically indistinguishable from no sensitivity.
_DEFAULT_RANK_TOL = 1e-9

# Default: full-rank but condition number above this is "practically" (not
# structurally) non-identifiable — the sloppy direction is ~8 orders weaker than
# the stiff one, so realistic noise makes it unrecoverable. Tunable per use case.
_DEFAULT_COND_THRESHOLD = 1e8


@dataclass(frozen=True)
class IdentifiabilityVerdict:
    """The structured result (IDENT-01..03). Self-describing; no LLM produced it."""

    identifiable: bool
    # "identifiable" | "structurally_non_identifiable" | "practically_non_identifiable"
    status: str
    rank: int
    n_params: int
    condition_number: float
    singular_values: tuple[float, ...]
    # IDENT-02 — the named degeneracy: each confounded direction is a unit vector
    # in parameter space (weights over the n_params parameters) along which the
    # predictions do not change (structural) or barely change (practical). For the
    # alien equal-charge case this comes back ~[0.707, -0.707] over (s1, s2).
    confounded_directions: tuple[tuple[float, ...], ...]
    rationale: str
    # IDENT-03 — what data regime would restore identifiability.
    disambiguation_hint: str
    param_names: tuple[str, ...] = field(default=())

    def describe(self) -> str:
        """One-block human summary (still no LLM — deterministic formatting)."""
        lines = [
            f"identifiable={self.identifiable} ({self.status})",
            f"rank={self.rank}/{self.n_params}  cond={self.condition_number:.3e}",
            f"singular_values={[f'{s:.3e}' for s in self.singular_values]}",
        ]
        for d in self.confounded_directions:
            lines.append(f"  confounded direction: {self._fmt_direction(d)}")
        lines.append(f"rationale: {self.rationale}")
        lines.append(f"disambiguation: {self.disambiguation_hint}")
        return "\n".join(lines)

    def _fmt_direction(self, direction: tuple[float, ...]) -> str:
        names = self.param_names or tuple(f"theta[{i}]" for i in range(self.n_params))
        terms = [f"{w:+.3f}*{n}" for w, n in zip(direction, names, strict=False) if abs(w) > 1e-6]
        return " ".join(terms) if terms else "(degenerate)"


def sensitivity_matrix(
    predict: PredictFn,
    theta: np.ndarray,
    *,
    rel_step: float = 1e-6,
    abs_step_floor: float = 1e-9,
) -> np.ndarray:
    """Central-difference sensitivity matrix S[i, j] = d predict_i / d theta_j.

    Per-parameter step ``h_j = rel_step * |theta_j|`` with an absolute floor so a
    parameter sitting exactly at zero still gets a finite, non-degenerate step.
    Central differences (O(h^2)) keep the linearization honest near sharp curvature
    (e.g., the alien collinearity at s1 == s2).

    Returns an (m_observations x n_params) array. Raises on a non-finite or
    inconsistent-length prediction (loud failure, per engineering discipline — a
    silent NaN would masquerade as a rank drop and mislabel a fine model).
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    n = theta.size
    base = np.asarray(predict(theta), dtype=np.float64).ravel()
    m = base.size
    if not np.all(np.isfinite(base)):
        raise ValueError("predict(theta) returned non-finite values at the nominal point")

    cols: list[np.ndarray] = []
    for j in range(n):
        h = rel_step * abs(theta[j])
        if h < abs_step_floor:
            h = abs_step_floor
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += h
        tm[j] -= h
        fp = np.asarray(predict(tp), dtype=np.float64).ravel()
        fm = np.asarray(predict(tm), dtype=np.float64).ravel()
        if fp.size != m or fm.size != m:
            raise ValueError(
                f"predict() changed output length under a theta[{j}] perturbation "
                f"({fp.size}/{fm.size} vs {m}); the observation design must be fixed"
            )
        if not (np.all(np.isfinite(fp)) and np.all(np.isfinite(fm))):
            raise ValueError(f"predict() returned non-finite values perturbing theta[{j}]")
        cols.append((fp - fm) / (2.0 * h))
    return np.column_stack(cols) if cols else np.empty((m, 0))


def practical_identifiability(
    predict: PredictFn,
    theta: Sequence[float] | np.ndarray,
    *,
    sigma: float = 1.0,
    rel_step: float = 1e-6,
    rank_tol: float = _DEFAULT_RANK_TOL,
    cond_threshold: float = _DEFAULT_COND_THRESHOLD,
    param_names: Sequence[str] | None = None,
    sensitivity: np.ndarray | None = None,
) -> IdentifiabilityVerdict:
    """Decide local identifiability of ``predict``'s parameters at ``theta``.

    IDENT-01 (recoverable?), IDENT-02 (which combinations collapse), IDENT-03
    (what would restore it). Deterministic, no LLM.

    Args:
      predict: params -> flat predicted-observation vector under the FIXED design.
      theta: nominal parameter point to linearize at.
      sigma: observation-noise std (uniform Gaussian). Scales the FIM but NOT the
        rank or the condition number, so the structural verdict is sigma-free;
        sigma only matters if you read absolute FIM eigenvalues downstream.
      rank_tol: a singular value below ``rank_tol * s_max`` is treated as zero
        (the direction is structurally unobservable).
      cond_threshold: full-rank but ``s_max / s_min`` above this => practically
        non-identifiable (sloppy).
      param_names: optional labels for a readable degeneracy description.

    Returns:
      IdentifiabilityVerdict.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    n = theta.size
    names = (
        tuple(param_names) if param_names is not None else tuple(f"theta[{i}]" for i in range(n))
    )
    if param_names is not None and len(names) != n:
        raise ValueError(f"param_names length {len(names)} != n_params {n}")

    # EXP-261 (2026-09-08): a caller that can compute exact sensitivities (forward
    # sensitivity equations for an ODE, an analytic Jacobian, ...) passes them in
    # and skips the central-difference estimate, whose round-off floor
    # (integrator tolerance / step) can hide a true null direction behind a
    # ~1e-8 singular value. The verdict logic below is unchanged.
    if sensitivity is not None:
        s_mat = np.asarray(sensitivity, dtype=np.float64)
        if s_mat.ndim != 2 or s_mat.shape[1] != n:
            raise ValueError(
                f"sensitivity must be (m x {n}); got {s_mat.shape}"
            )
        if not np.all(np.isfinite(s_mat)):
            raise ValueError("sensitivity contains non-finite entries")
    else:
        s_mat = sensitivity_matrix(predict, theta, rel_step=rel_step)
    # SVD: S = U @ diag(sv) @ Vt. Right-singular vectors (rows of Vt) are directions
    # in PARAMETER space; their singular values measure how strongly the predictions
    # respond along each. FIM = S^T S / sigma^2 shares these vectors with eigenvalues
    # sv^2 / sigma^2, so SVD of S is the numerically-stable route to the same answer.
    if s_mat.size == 0 or s_mat.shape[0] == 0:
        sv = np.zeros(n)
        vt = np.eye(n)
    else:
        # Thin SVD when there are at least as many observations as parameters (the only
        # case where full_matrices=True would allocate an m x m U; for the metric
        # pre-flight m ~ 4e4 rows that is ~15 GB and hangs the script). When m < n the
        # full V is needed so the (n - m) structurally-null directions are indexable.
        _u, sv, vt = np.linalg.svd(s_mat, full_matrices=(s_mat.shape[0] < s_mat.shape[1]))
        # svd returns min(m,n) singular values; pad to n so every parameter
        # direction has an associated (possibly zero) singular value + vector.
        if sv.size < n:
            sv = np.concatenate([sv, np.zeros(n - sv.size)])

    s_max = float(sv[0]) if sv.size and sv[0] > 0 else 0.0
    # A direction is "collapsed" when its singular value is below the relative floor
    # (or s_max is 0 — the model is insensitive to every parameter).
    if s_max <= 0.0:
        collapsed = np.ones(n, dtype=bool)
    else:
        collapsed = sv < (rank_tol * s_max)
    rank = int(np.count_nonzero(~collapsed))

    # Condition number is well-defined only over a full-rank Jacobian. If any
    # direction collapsed (rank < n), the condition number over parameter space is
    # infinite by definition, and the STRUCTURAL branch below owns that verdict.
    if rank < n or s_max <= 0.0:
        condition_number = float("inf")
    else:
        s_min = float(sv[n - 1])  # full rank => the smallest singular value is positive
        condition_number = s_max / s_min if s_min > 0.0 else float("inf")

    confounded: list[tuple[float, ...]] = []
    if rank < n:
        # Structural: the collapsed right-singular vectors are the unobservable
        # parameter combinations (IDENT-02). Vt rows align with sv entries.
        for i in range(n):
            if collapsed[i]:
                confounded.append(tuple(float(x) for x in vt[i]))
        status = "structurally_non_identifiable"
        identifiable = False
        rationale = (
            f"rank {rank} < {n} parameters: {n - rank} parameter combination(s) "
            f"leave the predictions unchanged under this observation design."
        )
        hint = (
            "Non-identifiable from this design. Add observations that respond to the "
            "confounded direction(s) below — i.e. vary the inputs/configurations the "
            "model is currently insensitive to along that combination (for a hidden "
            "product s1*s2, observe configurations at DISTINCT products so the columns "
            "stop being collinear). This is the experiment the engine should request."
        )
    elif condition_number > cond_threshold:
        # Practical (sloppy): full rank, but the weakest direction is swamped.
        status = "practically_non_identifiable"
        identifiable = False
        confounded.append(tuple(float(x) for x in vt[n - 1]))  # smallest-sv direction
        rationale = (
            f"full rank ({rank}/{n}) but condition number {condition_number:.2e} "
            f"> {cond_threshold:.0e}: the weakest direction is observable in theory "
            f"but realistic noise makes it unrecoverable (sloppy)."
        )
        hint = (
            "Technically identifiable but practically not under realistic noise. "
            "Either tighten measurement noise / add replicates, or extend the design "
            "to gain sensitivity along the sloppy direction below."
        )
    else:
        status = "identifiable"
        identifiable = True
        rationale = (
            f"full rank ({rank}/{n}), condition number {condition_number:.2e} "
            f"within tolerance: every parameter is locally recoverable from this design."
        )
        hint = "Identifiable at this operating point; no extra data needed for recovery."

    return IdentifiabilityVerdict(
        identifiable=identifiable,
        status=status,
        rank=rank,
        n_params=n,
        condition_number=condition_number,
        singular_values=tuple(float(s) for s in sv),
        confounded_directions=tuple(confounded),
        rationale=rationale,
        disambiguation_hint=hint,
        param_names=names,
    )
