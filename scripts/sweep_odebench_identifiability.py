"""EXP-261 — the IDEA diagnostic swept over the 63 ODEBench systems under the
LLM-ACES observation protocol. $0, deterministic, no LLM, no network.

What this answers (pre-registered in predictions.md, block dated 2026-09-08):
  Under the data LLM-ACES hands its proposer at round 0 (one trajectory from
  IC-0 on t in [0, 1], 100 points, every state and its TRUE derivative
  observed, noise-free), which of the 63 ODEBench systems have parameters that
  the data determine, which are practically non-identifiable, and which are
  structurally non-identifiable so that NO initial condition in the LLM-ACES
  acquisition box can ever separate them (certified exhaustion on a benchmark
  we did not build).

Protocol fidelity (verbatim from the pinned LLM-ACES release, commit 60d4df7):
  * equations, constants, IC-0, IC-1: scripts/strogatz_ode.py (63 systems)
  * time grid: np.linspace(0, 1, 100)  (generate_ode.py N_TRAIN, T_TRAIN_END)
  * derivatives: analytic f(u; theta) at the sampled states (generate_ode.py
    make_dataset, Y_true = fn(*X.T)); NOT finite differences
  * acquisition class: an initial condition sampled uniformly inside the
    per-system box in llm-aces/ic_bounds.json (active_llm_aces.py Step 3,
    n_virtual = 10, so 10 random candidates), integrated on the same grid
  * OOD window (1, 10] with 150 points is their held-out evaluation, not
    acquisition; included as a 13th candidate for completeness, labelled.

Numerics that are ours, not theirs: sensitivities are EXACT, by the forward
sensitivity equations (x' = f, S' = J_x S + J_c) integrated at 1e-12, with the
instrument's central-difference estimate kept as a cross-check field. The first
full run (2026-09-08, central differences at 1e-10) put the RC circuit at
cond 2.35e7, under the 1e8 practical threshold, because the round-off floor of
a finite difference (tolerance / step) hides a true null direction behind a
~1e-8 singular value; see predictions.md Amendment 3. LLM-ACES generates its
data at 1e-5 / 1e-7; tolerance is an integrator setting, not the protocol.

Structural check: the 63 systems are vector fields f_k(x; c). One sympy
expression per system is formed as sum_k w_k * f_k with the w_k declared as
free (observable) symbols, so two parameters are derivative-proportional with
an observable-free ratio iff they are so in EVERY component with the SAME
ratio. That is exactly the redundancy the structural checker is built to find.

Run:  poetry run python scripts/sweep_odebench_identifiability.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from ascension.diagnostics.exhaustion import certify_class_symmetry_symbolic
from ascension.diagnostics.identifiability import practical_identifiability
from ascension.diagnostics.optimal_design import rank_designs_by_identifiability
from ascension.diagnostics.structural import structural_identifiability

REPO = Path(__file__).resolve().parents[1]
ACES = Path(os.environ.get("LLM_ACES_DIR", REPO / "external" / "LLM-ACES"))  # pinned 60d4df7, see external/README.md
sys.path.insert(0, str(ACES / "scripts"))
import strogatz_ode  # noqa: E402  (the pinned LLM-ACES copy of ODEBench)

# --- the LLM-ACES protocol constants (generate_ode.py) ------------------------
T_TRAIN_END = 1.0
N_TRAIN = 100
N_OOD = 150
T_TRAIN = np.linspace(0.0, T_TRAIN_END, N_TRAIN)
T_OOD = np.linspace(T_TRAIN_END, 10.0, N_OOD + 1)[1:]
N_VIRTUAL = 10  # active_llm_aces.py --n_virtual default
ACQ_SEED = 0  # generate_ode.py sets np.random.seed(0); we mirror that for the box draws

# --- our numerics ------------------------------------------------------------
INTEG = dict(method="LSODA", rtol=1e-10, atol=1e-10)
COND_THRESHOLD = 1e8  # the paper's practical threshold (Section 3.1)


def slug(name: str) -> str:
    # keep unicode letters: the LLM-ACES ic_bounds.json keys Rössler as 'rössler-...'
    return re.sub(r"[^\w]+", "-", name.lower()).strip("-").replace("_", "-")


def bounds_key(name: str, bounds: dict) -> str:
    k = slug(name)
    if k in bounds:
        return k
    raise KeyError(f"no ic_bounds entry for {name!r} (tried {k!r})")


@dataclass
class SystemResult:
    id: int
    name: str
    dim: int
    n_params: int
    eq: str
    # structural (exact, symbolic; protocol-free)
    structural_identifiable: bool
    structural_confounded_groups: list[list[str]]
    structural_absent: list[str]
    # practical under the LLM-ACES round-0 protocol (IC-0, [0,1], u + du)
    round0_status: str
    round0_rank: int
    round0_cond: float
    round0_null_directions: list[list[float]]
    # certificate: is each round-0 null direction a symmetry of the whole
    # acquisition class (any IC, any time)? symbolic on the vector field.
    certified_class_level: list[bool]
    certificate_residuals: list[str | None]
    # acquisition: best single added design by the ranker, and the verdict
    # of (round-0 data + that design)
    best_added_design: str | None
    after_acquisition_status: str | None
    after_acquisition_rank: int | None
    after_acquisition_cond: float | None
    any_single_design_restores: bool | None
    # P4: LLM-ACES disagreement signal along a certified null direction
    max_rel_rollout_change_along_null: float | None
    triage: str
    elapsed_s: float
    error: str | None = None
    note: str | None = None
    # finite-difference cross-check of the round-0 verdict (the instrument's old path)
    round0_fd_status: str | None = None
    round0_fd_cond: float | None = None
    structural_dependence_groups: list[list[str]] | None = None
    structural_rank_deficiency: int = 0


class ProtocolUnrealizable(Exception):
    """The published protocol's own trajectory diverges before the window ends."""


def build_system(eq: dict, bounds: dict):
    dim = eq["dim"]
    theta = np.asarray(eq["consts"][0], dtype=np.float64)
    n = theta.size
    xs = sp.symbols([f"x_{i}" for i in range(dim)])
    cs = sp.symbols([f"c_{i}" for i in range(n)])
    comps = [sp.sympify(s) for s in eq["eq"].split("|")]
    # id 44 (quadratic damping) carries Abs(x_1)*x_1; sympy's derivative of Abs is
    # not lambdifiable. sqrt(x^2) is the same function with an almost-everywhere
    # derivative x/|x|, which is what the sensitivity equations need.
    comps = [c.replace(sp.Abs, lambda a: sp.sqrt(a**2)) for c in comps]
    assert len(comps) == dim, f"{eq['id']}: {len(comps)} components for dim {dim}"
    f_lam = sp.lambdify([xs, cs], comps, "numpy")

    def rhs_factory(th: np.ndarray):
        def rhs(t, x):
            return np.asarray(f_lam(x, th), dtype=np.float64)

        return rhs

    def _blowup(t, z):
        return 1e6 - float(np.max(np.abs(z[:dim])))

    _blowup.terminal = True

    def integrate(th: np.ndarray, ic, t_eval: np.ndarray) -> np.ndarray:
        sol = solve_ivp(
            rhs_factory(th),
            (0.0, float(t_eval[-1])),
            np.asarray(ic, dtype=np.float64),
            t_eval=t_eval,
            events=_blowup,
            **INTEG,
        )
        if not sol.success or sol.y.shape[1] != t_eval.size:
            raise RuntimeError(f"integration failed: {sol.message}")
        return sol.y  # (dim, n_t)

    # exact Jacobians for the forward sensitivity equations
    J_x = sp.Matrix(comps).jacobian(xs)
    J_c = sp.Matrix(comps).jacobian(cs)
    jx_lam = sp.lambdify([xs, cs], J_x, "numpy")
    jc_lam = sp.lambdify([xs, cs], J_c, "numpy")

    def sensitivity(ic, t_eval: np.ndarray, *, t_from: int = 0) -> np.ndarray:
        """Exact (m x n) sensitivity of [u.ravel(), du.ravel()] w.r.t. theta at the
        published constants, by the forward sensitivity equations
            x' = f(x, c),   S' = J_x S + J_c,   S(0) = 0   (the IC is not a parameter)
        so d(du)/dc = J_x S + J_c at every sample. Integrated at 1e-12."""
        th = theta

        def aug(t, z):
            x = z[:dim]
            S = z[dim:].reshape(dim, n)
            fx = np.asarray(f_lam(x, th), dtype=np.float64).ravel()
            jx = np.asarray(jx_lam(x, th), dtype=np.float64).reshape(dim, dim)
            jc = np.asarray(jc_lam(x, th), dtype=np.float64).reshape(dim, n)
            return np.concatenate([fx, (jx @ S + jc).ravel()])

        z0 = np.concatenate([np.asarray(ic, dtype=np.float64), np.zeros(dim * n)])
        sol = solve_ivp(aug, (0.0, float(t_eval[-1])), z0, t_eval=t_eval,
                        method="LSODA", rtol=1e-12, atol=1e-12, events=_blowup)
        if not sol.success or sol.y.shape[1] != t_eval.size:
            raise RuntimeError(f"sensitivity integration failed: {sol.message}")
        Z = sol.y[:, t_from:]
        nt = Z.shape[1]
        S_u = Z[dim:, :].T.reshape(nt, dim, n)          # (nt, dim, n)
        rows_u = S_u.reshape(nt * dim, n)               # u.ravel() is (dim, nt) row-major
        rows_u = np.transpose(S_u, (1, 0, 2)).reshape(dim * nt, n)
        rows_du = np.empty((dim, nt, n))
        for j in range(nt):
            x = Z[:dim, j]
            jx = np.asarray(jx_lam(x, th), dtype=np.float64).reshape(dim, dim)
            jc = np.asarray(jc_lam(x, th), dtype=np.float64).reshape(dim, n)
            rows_du[:, j, :] = jx @ S_u[j] + jc
        return np.vstack([rows_u, rows_du.reshape(dim * nt, n)])

    def design(ic, t_eval: np.ndarray):
        """predict(theta) -> [u.ravel(), du.ravel()] on this IC and grid (used by P4
        rollouts and as the finite-difference cross-check)."""

        def predict(th: np.ndarray) -> np.ndarray:
            u = integrate(th, ic, t_eval)
            du = np.column_stack([f_lam(u[:, j], th) for j in range(u.shape[1])])
            return np.concatenate([u.ravel(), np.asarray(du, dtype=np.float64).ravel()])

        return predict

    # the acquisition class: ICs uniform in the LLM-ACES box for this system
    box = np.asarray(bounds[bounds_key(eq["name"], bounds)], dtype=np.float64)
    box_note = None
    if box.shape != (dim, 2):
        # LLM-ACES release defect: e.g. SEIR (id 63) is 4-D but ic_bounds.json lists
        # 3 intervals, so their acquisition step cannot draw a valid IC for it.
        # Recorded, not hidden: the box class is empty for this system.
        box_note = f"ic_bounds.json malformed for this system: {box.shape} vs dim {dim}; box class empty"
        virtual_ics = np.empty((0, dim))
    else:
        rng = np.random.default_rng(ACQ_SEED)
        virtual_ics = rng.uniform(box[:, 0], box[:, 1], size=(N_VIRTUAL, dim))

    try:
        round0 = sensitivity(eq["init"][0], T_TRAIN)
    except RuntimeError as ex:
        # the round-0 trajectory itself leaves the finite domain inside [0, 1]:
        # the LLM-ACES protocol is not realizable for this system. Own category.
        raise ProtocolUnrealizable(str(ex)) from ex
    round0_predict = design(eq["init"][0], T_TRAIN)

    designs = {}
    dropped = []
    candidates = [("IC-1 (ODEBench held-out IC), [0,1]", eq["init"][1], T_TRAIN, 0)]
    candidates += [(f"box IC {k}, [0,1]", ic, T_TRAIN, 0) for k, ic in enumerate(virtual_ics)]
    # their OOD evaluation window on IC-0, labelled as not-acquisition
    t_full = np.concatenate([T_TRAIN, T_OOD])
    candidates.append(("OOD window (1,10] on IC-0 (evaluation, not acquisition)", eq["init"][0], t_full, N_TRAIN))
    for label, ic, grid, t_from in candidates:
        try:
            designs[label] = sensitivity(ic, grid, t_from=t_from)
        except RuntimeError as ex:
            # a candidate whose trajectory leaves the finite domain cannot be
            # acquired (LLM-ACES would receive inf data); dropped and recorded.
            dropped.append(f"{label}: {ex}")
    if dropped:
        box_note = (box_note + "; " if box_note else "") + "dropped designs: " + " | ".join(dropped)
    if not designs:
        raise ProtocolUnrealizable("every candidate design diverges")
    # symbolic vector field with component weights as free symbols
    ws = sp.symbols([f"w_{i}" for i in range(dim)])
    field_expr = sum(w * c for w, c in zip(ws, comps, strict=True))
    return dict(
        theta=theta, cs=cs, xs=xs, ws=ws, field_expr=field_expr, round0=round0,
        designs=designs, virtual_ics=virtual_ics, integrate=integrate, f_lam=f_lam,
        box_note=box_note, round0_predict=round0_predict,
    )


def combined(a, b):
    def predict(th):
        return np.concatenate([a(th), b(th)])

    return predict


def run_system(eq: dict, bounds: dict) -> SystemResult:
    t0 = time.time()
    if len(eq["consts"][0]) == 0:
        # ODEBench id 11 (naive critical slowing down, x' = -x^3) has no constants:
        # nothing to identify. Recorded as its own category, not as identifiable.
        return SystemResult(
            id=eq["id"], name=eq["name"], dim=eq["dim"], n_params=0, eq=eq["eq"],
            structural_identifiable=True, structural_confounded_groups=[], structural_absent=[],
            round0_status="no_free_parameters", round0_rank=0, round0_cond=1.0,
            round0_null_directions=[], certified_class_level=[], certificate_residuals=[],
            best_added_design=None, after_acquisition_status=None, after_acquisition_rank=None,
            after_acquisition_cond=None, any_single_design_restores=None,
            max_rel_rollout_change_along_null=None, triage="no_free_parameters",
            elapsed_s=round(time.time() - t0, 2), note="no constants in the equation",
        )
    S = build_system(eq, bounds)
    theta, cs = S["theta"], S["cs"]
    names = [str(c) for c in cs]

    # 1. structural (exact)
    sv = structural_identifiability(S["field_expr"], cs, list(S["xs"]) + list(S["ws"]))

    # 2. practical at round 0 (exact forward sensitivities; the finite-difference
    #    estimate is kept as a cross-check field)
    r0 = practical_identifiability(
        S["round0_predict"], theta, cond_threshold=COND_THRESHOLD, param_names=names,
        sensitivity=S["round0"],
    )
    r0_fd = practical_identifiability(
        S["round0_predict"], theta, cond_threshold=COND_THRESHOLD, param_names=names
    )

    # 3. certificate per null direction, symbolic on the vector field itself:
    #    if the field is invariant along d for every x, no IC and no time can help.
    certs = []
    for d in r0.confounded_directions:
        c = certify_class_symmetry_symbolic(
            S["field_expr"], cs, list(theta), list(d),
            free_vars=list(S["xs"]) + list(S["ws"]),
        )
        certs.append(c)

    # 4. acquisition: rank single added designs; verdict of round0 + best
    best_label = after_status = None
    after_rank = after_cond = None
    any_restores = None
    if not r0.identifiable:
        ranking = rank_designs_by_identifiability(
            {k: np.vstack([S["round0"], v]) for k, v in S["designs"].items()},
            theta, param_names=names,
        )
        best_label = ranking.best
        board = {d.label: d for d in ranking.ranked}
        bv = board[best_label]
        after_rank, after_cond = bv.rank, bv.condition_number
        after_status = (
            "identifiable" if bv.identifiable
            else ("structurally_non_identifiable" if bv.rank < bv.n_params
                  else "practically_non_identifiable")
        )
        any_restores = any(d.identifiable for d in board.values())

    # 5. P4: LLM-ACES disagreement has no signal along a certified null direction.
    #    Two models theta +/- eps*d, rolled out from every candidate IC: max
    #    relative change in the rollout.
    max_rel = None
    cert_dirs = [d for d, c in zip(r0.confounded_directions, certs, strict=True) if c.certified]
    if cert_dirs:
        eps = 1e-3
        worst = 0.0
        for d in cert_dirs:
            d = np.asarray(d)
            for ic in list(S["virtual_ics"]) + [np.asarray(eq["init"][1])]:
                up = S["integrate"](theta + eps * d, ic, T_TRAIN)
                um = S["integrate"](theta - eps * d, ic, T_TRAIN)
                scale = max(np.max(np.abs(up)), np.max(np.abs(um)), 1e-12)
                worst = max(worst, float(np.max(np.abs(up - um)) / scale))
        max_rel = worst

    if r0.identifiable:
        triage = "capability"
    elif certs and all(c.certified for c in certs):
        triage = "change_class"
    else:
        triage = "design_within_class"

    return SystemResult(
        id=eq["id"], name=eq["name"], dim=eq["dim"], n_params=int(theta.size), eq=eq["eq"],
        structural_identifiable=sv.identifiable,
        structural_confounded_groups=[list(g) for g in sv.confounded_groups],
        structural_absent=list(sv.absent_params),
        round0_status=r0.status, round0_rank=r0.rank, round0_cond=float(r0.condition_number),
        round0_null_directions=[list(map(float, d)) for d in r0.confounded_directions],
        certified_class_level=[c.certified for c in certs],
        certificate_residuals=[c.residual for c in certs],
        best_added_design=best_label, after_acquisition_status=after_status,
        after_acquisition_rank=after_rank,
        after_acquisition_cond=None if after_cond is None else float(after_cond),
        any_single_design_restores=any_restores,
        max_rel_rollout_change_along_null=max_rel,
        triage=triage, elapsed_s=round(time.time() - t0, 2), note=S["box_note"],
        round0_fd_status=r0_fd.status, round0_fd_cond=float(r0_fd.condition_number),
        structural_dependence_groups=[list(g) for g in sv.dependence_groups],
        structural_rank_deficiency=sv.rank_deficiency,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "results" / "exp261")
    ap.add_argument("--ids", type=str, default="", help="comma-separated subset for smoke")
    ap.add_argument("--guard-seconds", type=int, default=1800,
                    help="per-system wall-clock guard (id 62, a 4-D system with 4 params, needs ~10 min at 1e-12)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bounds = json.load(open(ACES / "llm-aces/ic_bounds.json"))
    eqs = strogatz_ode.equations
    if args.ids:
        keep = {int(x) for x in args.ids.split(",")}
        eqs = [e for e in eqs if e["id"] in keep]
    aces_hash = (ACES / ".git/HEAD").read_text().strip()
    try:
        ref = (ACES / ".git" / aces_hash.split(": ")[1]).read_text().strip()
    except Exception:
        ref = aces_hash
    print(f"LLM-ACES pinned at {ref}; {len(eqs)} systems; out={args.out}")

    import signal

    class _Timeout(Exception):
        pass

    def _alarm(_signum, _frame):
        raise _Timeout(f"per-system wall-clock guard ({args.guard_seconds} s) fired")

    signal.signal(signal.SIGALRM, _alarm)

    results: list[SystemResult] = []
    for eq in eqs:
        try:
            # a diverging trajectory (e.g. id 29, x_1' = x_1^2 - x_0^2 from (3.2, 1.4))
            # makes a 1e-12 integration crawl; guard each system and record the
            # timeout as an error rather than hanging the sweep.
            signal.alarm(args.guard_seconds)
            try:
                r = run_system(eq, bounds)
            finally:
                signal.alarm(0)
        except ProtocolUnrealizable as ex:
            r = SystemResult(
                id=eq["id"], name=eq["name"], dim=eq["dim"], n_params=len(eq["consts"][0]),
                eq=eq["eq"], structural_identifiable=True, structural_confounded_groups=[],
                structural_absent=[], round0_status="protocol_unrealizable", round0_rank=-1,
                round0_cond=float("nan"), round0_null_directions=[], certified_class_level=[],
                certificate_residuals=[], best_added_design=None, after_acquisition_status=None,
                after_acquisition_rank=None, after_acquisition_cond=None,
                any_single_design_restores=None, max_rel_rollout_change_along_null=None,
                triage="protocol_unrealizable", elapsed_s=0.0,
                note=f"round-0 trajectory diverges inside [0,1] from IC-0: {ex}",
            )
        except Exception as ex:  # loud, recorded, never silent
            r = SystemResult(
                id=eq["id"], name=eq["name"], dim=eq["dim"], n_params=len(eq["consts"][0]),
                eq=eq["eq"], structural_identifiable=False, structural_confounded_groups=[],
                structural_absent=[], round0_status="ERROR", round0_rank=-1, round0_cond=float("nan"),
                round0_null_directions=[], certified_class_level=[], certificate_residuals=[],
                best_added_design=None, after_acquisition_status=None, after_acquisition_rank=None,
                after_acquisition_cond=None, any_single_design_restores=None,
                max_rel_rollout_change_along_null=None, triage="ERROR", elapsed_s=0.0,
                error=f"{type(ex).__name__}: {ex}",
            )
        results.append(r)
        print(
            f"[{r.id:2d}] {r.name[:38]:38s} dim={r.dim} p={r.n_params} "
            f"struct={'ok' if r.structural_identifiable else 'NON-ID pairs=' + str(r.structural_confounded_groups) + ' dep=' + str(r.structural_dependence_groups)} "
            f"round0={r.round0_status} rank={r.round0_rank}/{r.n_params} cond={r.round0_cond:.2e} "
            f"triage={r.triage}"
            + (f" after={r.after_acquisition_status} via '{r.best_added_design}'" if r.best_added_design else "")
            + (f" P4 max_rel={r.max_rel_rollout_change_along_null:.1e}" if r.max_rel_rollout_change_along_null is not None else "")
            + (f" [FD: {r.round0_fd_status} cond={r.round0_fd_cond:.1e}]" if r.round0_fd_status and r.round0_fd_status != r.round0_status else "")
            + (f" ERROR={r.error}" if r.error else ""),
            flush=True,
        )

    (args.out / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=1))
    with open(args.out / "results.tsv", "w") as fh:
        fh.write("id\tname\tdim\tn_params\tstructural\tround0_status\trank\tcond\ttriage\tbest_added\tafter_status\tany_restores\tP4_max_rel\terror\n")
        for r in results:
            fh.write(
                f"{r.id}\t{r.name}\t{r.dim}\t{r.n_params}\t{r.structural_identifiable}\t{r.round0_status}\t"
                f"{r.round0_rank}\t{r.round0_cond:.3e}\t{r.triage}\t{r.best_added_design}\t{r.after_acquisition_status}\t"
                f"{r.any_single_design_restores}\t{r.max_rel_rollout_change_along_null}\t{r.error or ''}\n"
            )

    # summary
    ok = [r for r in results if not r.error and r.round0_status not in ("no_free_parameters", "protocol_unrealizable")]
    n = len(ok)
    n_no_params = sum(r.round0_status == "no_free_parameters" for r in results)
    n_unrealizable = sum(r.round0_status == "protocol_unrealizable" for r in results)
    fd_disagree = [r.id for r in ok if r.round0_fd_status != r.round0_status]
    ident = sum(r.round0_status == "identifiable" for r in ok)
    struct = sum(r.round0_status == "structurally_non_identifiable" for r in ok)
    prac = sum(r.round0_status == "practically_non_identifiable" for r in ok)
    sym_non = sum(not r.structural_identifiable for r in ok)
    cert_all = [r for r in ok if r.triage == "change_class"]
    prac_rest = [r for r in ok if r.round0_status == "practically_non_identifiable" and r.any_single_design_restores]
    summary = dict(
        n_systems=n, n_no_free_parameters=n_no_params, n_protocol_unrealizable=n_unrealizable,
        unrealizable_ids=[r.id for r in results if r.round0_status == "protocol_unrealizable"],
        n_errors=len(results) - n - n_no_params - n_unrealizable,
        finite_difference_disagrees_with_exact=fd_disagree,
        round0_identifiable=ident, round0_structural_non_identifiable=struct,
        round0_practical_non_identifiable=prac,
        symbolic_structural_non_identifiable=sym_non,
        certified_change_class=len(cert_all),
        certified_ids=[r.id for r in cert_all],
        practical_restored_by_one_design=len(prac_rest),
        practical_ids=[r.id for r in ok if r.round0_status == "practically_non_identifiable"],
        P4_max_rel_over_certified=max(
            [r.max_rel_rollout_change_along_null for r in cert_all if r.max_rel_rollout_change_along_null is not None],
            default=None,
        ),
        llm_aces_commit=ref,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
