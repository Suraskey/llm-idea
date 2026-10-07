<!-- Public copy of the development repository's predictions.md as of 2026-10-07. Two occurrences of a billing-project name were replaced by [GCP project]; nothing else was changed. The commit ordering that the paper cites (predictions committed before each run) is in the private development repository and is available from the authors on request. -->

# Pre-Registered Predictions — Project Ascension v0.1.0

**Status:** LOCKED
**Locked on:** 2026-04-22
**Locked by:** Surya Shetty
**Locks against:** the first measurement run of any phase that touches `runs.cost_usd` or writes to `experiments` / `interventions`. Once the first such run lands, this file becomes append-only — see *Amendment discipline* below.

This file is the binding pre-registration for the v0.1.0 ICERM sprint. It exists per `SCOPE.md §22.1` ("Pre-registered predictions live in a committed `predictions.md` file in the repo before the first measurement run") and `SCOPE.md §22.9` ("Pre-registered predictions protect the paper's integrity").

A pre-registered prediction that *fails* is a publishable finding when the architecture is principled. A prediction that is *quietly retconned* after the data lands is research misconduct. The discipline below exists to keep the line clean.

---

## Why pre-registration matters here

Project Ascension is making three claims that reviewers will challenge:

1. **The brain matters.** A persistent Agora across attempts beats a fresh-context baseline.
2. **The court matters.** Justice + Parmenides catches false-positive findings that a plain swarm would mark `confirmed`.
3. **In-context discovery beats raw model capability.** A diverse Council on the Alien Universe reaches discovery depth ≥ 2 (force law + correction term) when no single-model baseline does.

Each of these claims is a directional bet. Without pre-registration, "the brain matters" can be reframed post-hoc as "well, it didn't help on this metric but it helped on that one." Pre-registration removes that escape hatch by naming the metric, the comparator, the effect size, and the seed budget *before* the first run.

---

## The five predictions (SCOPE §11, verbatim)

These are the five predictions named in `SCOPE.md §11` as "the authoritative list" cited from `SCOPE.md §22.1`. They are **load-bearing for the paper**. Edits below this line require an Amendment block (see end of file).

### Prediction 1 — Coverage: Full > Baseline

**Claim:** The Full configuration (5 agents + Agora + Justice + Argus) achieves higher *coverage* (fraction of three Tier 1 benchmarks solved within the per-run call budget) than the Baseline configuration (single Gemini 3.1 Pro, no Agora, fresh context per attempt) at a fixed budget of 200 LLM calls per run.

**Quantitative threshold:** Effect size large enough to register at p < 0.05 with 8 seeds via Wilcoxon signed-rank test (one-sided, Holm-Bonferroni corrected across the family of pairwise tests in §11). Cliff's delta reported alongside.

**Comparator:** Baseline cell from the §11 Tier 1 ablation table.

**Falsification condition:** If `coverage(Full) ≤ coverage(Baseline)` at the same call budget, OR if the difference fails to register at p < 0.05 after correction, the prediction is **falsified**. Report honestly per `SCOPE.md §22.9`.

### Prediction 2 — Memory ablation: +Brain > Baseline

**Claim:** The +Brain configuration (single agent with persistent Agora across attempts) achieves higher coverage than the Baseline (single agent, fresh context).

**Quantitative threshold:** Effect size large enough to register at p < 0.05 with 8 seeds via Wilcoxon signed-rank test (one-sided, Holm-Bonferroni corrected). Cliff's delta reported.

**Comparator:** Baseline cell from §11 Tier 1 ablation table.

**Why this matters:** This isolates the *brain* contribution from the *swarm* contribution. If +Brain ≯ Baseline but Full > Baseline, the swarm is doing the work — interesting but a different paper.

**Falsification condition:** `coverage(+Brain) ≤ coverage(Baseline)` at the same call budget, or fails p < 0.05 after correction.

### Prediction 3 — False-positive rate: Full < +Swarm

**Claim:** The Full configuration achieves a *lower* false-positive rate (fraction of `confirmed` findings that fail held-out validation) than the +Swarm configuration (5 agents, no Agora, no Justice). Justice plus Parmenides catches errors that swarm-alone votes through.

**Quantitative threshold:** Same statistical bar as Prediction 1.

**Comparator:** +Swarm cell from §11 Tier 1 ablation table.

**Why this matters:** This is the operational claim of the *court*. If Full has the same false-positive rate as +Swarm, Justice is decorative — and the most novel architectural claim collapses.

**Falsification condition:** `false_positive_rate(Full) ≥ false_positive_rate(+Swarm)`, or fails p < 0.05 after correction.

### Prediction 4 — Tier 2 discovery depth ≥ 2

**Claim:** On the Alien Universe (Tier 2), the Council achieves discovery depth ≥ 2 — meaning at least the *force law* (depth 1) AND a *correction term* (depth 2) are correctly identified. Per `SCOPE.md §10` enumeration of the six discovery layers.

**Quantitative threshold:** Mean depth ≥ 2 over 5 seeds; 95% bootstrap confidence interval lower bound ≥ 1.5.

**Comparator:** Mean depth of Tier 2 single-model baselines (`SCOPE.md §11`, Tier 2 single-model baselines table). The `in-context novelty score` (Council depth minus best baseline depth) must be > 0.

**Why this matters:** This is the in-context discovery claim — that the architecture contributes beyond raw model capability. A novelty score of 0 means the architecture adds nothing on the harder problem.

**Falsification condition:** Mean Council depth < 2, OR baseline depth ≥ Council depth (novelty score ≤ 0).

### Prediction 5 — Hallucination defense ablation shows monotonic degradation

**Claim:** The hallucination defense ablation (`SCOPE.md §11`, Lotka-Volterra only, four configs: No-schema / No-execution-grounding / No-replication / Full-defense) shows **monotonic degradation** of false-positive rate as defense layers are removed:

```
fp_rate(Full-defense) < fp_rate(No-replication) < fp_rate(No-execution-grounding) < fp_rate(No-schema)
```

**Quantitative threshold:** All three pairwise inequalities hold over mean-of-5-seeds, with at least the (Full-defense vs. No-schema) endpoint registering at p < 0.05 via Wilcoxon signed-rank.

**Comparator:** The four cells of the §11 hallucination defense ablation table.

**Why this matters:** This is the value-of-defense-layers claim. If the layers don't compose monotonically, the architecture is a pile of independent fixes rather than a defense-in-depth design. A non-monotonic result is informative — it says some layers cancel each other — but it overturns the "stack the layers" framing.

**Falsification condition:** Any of the three pairwise inequalities fails at the mean level, OR the endpoint comparison fails p < 0.05.

---

## Predictions added 2026-05-03 (Amazon Research Day 2026 synthesis)

These four predictions are derived from `.planning/research/amazon-research-day-2026/SYNTHESIS.md` and bind under the same statistical machinery (§ below). Pre-registered before any further Council run.

### Prediction 6 — Active metabolism strictly dominates static-graph Agora

**Claim:** Council with full Agora metabolism (decay + cross-pollination + gap detection + synthesis) achieves strictly lower RMSE on Tier 1 ODE rediscovery than Council with metabolism disabled (static-graph Agora — the AtoZ AI Co-Scientist baseline shape, P4).

**Quantitative threshold:** Mean RMSE across 8 seeds, Wilcoxon signed-rank one-sided p < 0.05, Cliff's delta ≥ 0.33.

**Comparator:** Two-cell ablation: `metabolism=on` vs `metabolism=off` on Tier 1 (Lotka-Volterra, Lorenz, Van der Pol).

**Why this matters:** P4 (Amazon BRP "AtoZ" KG) is the static-graph empirical baseline. If our metabolism does not strictly dominate it on the same task class, the metabolism layer is not load-bearing.

**Falsification condition:** RMSE difference is not in the predicted direction at mean level, OR p ≥ 0.05, OR Cliff's δ < 0.33.

---

### Prediction 7 — Council outperforms AgentArk-distilled single-agent on Tier 2

**Claim:** Council outperforms an AgentArk-style PAD-distilled single-agent student (PRM + GRPO trained on Council's debate trajectories) on Tier 2 Alien Universe success rate, with Cliff's delta ≥ 0.4.

**Quantitative threshold:** Tier 2 success rate across 5 seeds, Cliff's δ ≥ 0.4, Wilcoxon signed-rank one-sided p < 0.05.

**Comparator:** Council (full architecture) vs PAD-distilled student (single-agent, same base model family available on Vertex).

**Why this matters:** P2 (AgentArk) is the cleanest direct threat to Council's architecture. The structural moats — preserved dissent, separated Argus, prompt-iterability without retraining, novel-domain transfer — should manifest most strongly on a domain the distillation training set never saw (Tier 2 Alien Universe with custom physics).

**Falsification condition:** Cliff's δ < 0.4 or p ≥ 0.05. A null or reversed result on Tier 2 forces a thesis pivot — the capability claim contracts and the paper has to rest on epistemic / safety arguments alone.

---

### Prediction 8 — Reliability-aware Bayesian aggregation reduces verdict error

**Claim:** Reliability-aware online Bayesian verdict aggregation (P6, Phase 13.05) reduces Justice verdict error rate vs majority-vote aggregation across heterogeneous Council members.

**Quantitative threshold:** Verdict error rate on a held-out contested-claim benchmark, Wilcoxon signed-rank one-sided p < 0.05, mean error reduction ≥ 10% relative.

**Comparator:** Two-cell ablation: `aggregation=majority_vote` vs `aggregation=bayesian_reliability` on the same Council outputs.

**Why this matters:** P6 (Wei/Marculescu/Faloutsos) shows the unification of reliability + online + temporal aggregation. D-10 Council diversity guarantees heterogeneous reliability profiles; voting them as equal is a known anti-pattern.

**Falsification condition:** Error reduction not in predicted direction OR p ≥ 0.05 OR mean reduction < 10% relative.

**Status (Phase 13.05, 2026-05-20):** Phase 13.05 shipped the CAPABILITY + the 2-cell ablation toggle (`ASCENSION_VERDICT_AGGREGATION ∈ {majority_vote, bayesian_reliability}`, D-06) — both aggregation paths are real and exposed. **EXP-053** validated the precondition on REAL stored posteriors (the EXP-052 Judge `VerdictPosterior` ∪ the EXP-003-011 Council posteriors, $0 LLM): the two ablation cells produce a measurably different, reliability-justified aggregate `P̂`, and preserved dissent holds at the aggregation layer (balanced input → ~2.0 bits). The MEASUREMENT itself (verdict-error reduction ≥10% relative, Wilcoxon p<0.05 on a held-out contested-claim benchmark) remains **Phase 16** — this prediction is NOT yet measured, only enabled.

---

### Prediction 9 — Recursive task decomposition reduces tokens-per-correct at depth

**Claim:** Council members using recursive task decomposition (P8, `decompose_then_solve` template) consume fewer tokens-per-correct-answer than flat CoT, with the gap widening at higher problem depth.

**Quantitative threshold:** Tokens-per-correct-answer reported as a difficulty curve (Tier 1 binned by problem complexity); gap at the top difficulty bin ≥ 25% relative reduction; monotonic non-increasing curve across bins (no inversion).

**Comparator:** Two-cell ablation: `reasoning=cot_flat` vs `reasoning=recursive_decomposition` on Tier 1 with depth-binning.

**Why this matters:** P8 (Recursive Task Decomposition) shows bounded-context recursive call trees hold token usage flat where CoT explodes. Within our API-only Gemini constraint, the call-graph primitive is implementable in prompt space — but it has to pay rent in efficiency.

**Falsification condition:** Top-bin gap < 25%, OR curve has an inversion (recursive worse at any bin), OR overall accuracy regresses.

---

## Pre-registered threshold added 2026-05-21 (Phase 13.1 preserved-dissent gate)

This is a LOCKED threshold registration, not a directional prediction — it binds under `SCOPE.md §22.1` (git-proven, like the EXP-053 τ-free aggregation precedent) and lands in its OWN commit BEFORE any 13.1 measured outcome (the measured D-08 EXP-NNN replay is Plan 04). Registering the threshold before the data is what makes the preserved-dissent claim falsifiable rather than fitted.

### Phase 13.1 preserved-dissent threshold τ = 0.5 bits (LOCKED)

**Claim:** Phase 13.1 preserved-dissent threshold τ is LOCKED at 0.5 bits for all measurement runs, registered before any 13.1 measured outcome. A dispute's aggregate verdict posterior (the persisted `aggregated_verdict_posteriors.entropy_bits`, Phase 13.05) with Shannon entropy H(P̂) > 0.5 bits is preserved as `contested` (the §22.7 no-forced-consensus rule, JSTC-02); H(P̂) ≤ 0.5 bits with dominant mass ≥ 0.7 may resolve.

**Threshold:** τ = 0.5 bits. This matches the lone-Judge `RESOLVE_ENTROPY_CEILING` (`court.py:80`) over the SAME 4-candidate `CANDIDATE_VERDICTS` space (max entropy log2(4) = 2.0 bits), so the in-court resolve and the aggregate gate share ONE coherent contested-definition. Stored as the module constant `dissent_gate.DISSENT_TAU` + the `Settings.ASCENSION_DISSENT_TAU` env knob (default 0.5). τ is FROZEN — any change requires a STATE.md decision-log entry AND a new pre-registration block.

**Decision-boundary anchors (pre-registered before the D-08 measured replay):** the real EXP-052 Judge posterior `{inconclusive:0.80, both_partial:0.20, ...}` at 0.7219 bits → `contested` (0.7219 > 0.5); a balanced `{0.25×4}` posterior at exactly 2.0 bits → `contested` (the §22.7 unresolvable-by-construction case); a sharp `{0.95, ...}` posterior < 0.5 bits → resolved. The contested-rate reporting invariant (D-05): a rate rounding to zero across a whole ablation cell is an architecture failure to investigate, not a clean result.

**Cross-reference (the measured replay):** the D-08 $0 live-replay over these boundary anchors is logged as **EXP-054** in `.planning/EXPERIMENTS.md` (Phase 13.1 Plan 04). Per §22.1, the EXP-054 replay predictions land in their OWN commit BEFORE the measured outcomes, and this τ pre-registration (Plan 01) precedes both. The measured 13.1 outcome is reported against THIS locked τ = 0.5.

**Why this matters:** Preserved dissent — Justice does NOT force consensus — is the single most novel architectural claim of Project Ascension (SCOPE §22.7 / RESEARCH_INTEGRITY C-07). The §22.7 PHASE-GATING invariant test locks "a balanced posterior is unresolvable" in CI; pre-registering τ before the measured replay removes the post-hoc escape hatch of tuning the threshold to the data.

**Falsification / amendment condition:** If a future run shows τ = 0.5 systematically over- or under-flags the AGGREGATE posterior (e.g. the aggregate's spread profile diverges materially from the lone Judge), τ is NOT silently retuned — it is changed only via a STATE.md decision-log entry + a new pre-registration block (an adaptive / posterior-mass-floor / calibrated-probability τ is a forward-dated SEED, not 13.1 work). The measured 13.1 outcome (the D-08 $0 replay, EXP-NNN) is reported against THIS locked τ.

---

## Pre-registered block added 2026-05-21 (Phase 14.0 Argus deterministic close-gate — ARGUS-01 / D-09)

This is a LOCKED, per-rule pre-registration for the Phase 14.0 Argus validation experiment (EXP-055). It binds under `SCOPE.md §22.1` (git-proven, like the EXP-053/EXP-054 deterministic-replay precedents) and lands in its OWN commit BEFORE any EXP-055 measured outcome (the measured D-09 $0-LLM replay is Plan 04 Task 2, a SEPARATE later commit — `git log` will prove the ordering). Phase 14.0 (Argus) generates NO LLM output (D-02 / SCOPE §22.6 — every detector is pure SQL/numpy over tables that already exist). So `validate_live_before_phase_close` here = running each of the 7 REAL `detect_<rule>` detectors over a real-or-realistically-seeded trace at $0 LLM, confirming each FIRES on its FIRE trace AND stays SILENT on its FALSE-POSITIVE trace, and recording the result against THESE predictions. Low false-positive is half the bar: a rule that fires on normal operation is as bad as one that misses.

### Phase 14.0 Argus close-gate — pre-registered thresholds

Four of the seven rules use SCOPE/§5/§10 LITERAL thresholds (documented as canonical, NOT new directional predictions): repeated-hypothesis cosine ≥ 0.95 over ≥ 3 consecutive emissions; cost ceiling at 90% of $2000 ($1800 trigger); schema-reject streak ≥ 5; stale-Justice timeout 300 s (5 min) [CITED: SCOPE.md §5/§10; ROADMAP.md]. These are stored as `ARGUS_REPEAT_COSINE_THRESHOLD=0.95`, `ARGUS_REPEAT_CONSECUTIVE_N=3`, `ASCENSION_HALT_FRACTION=0.9` × `ASCENSION_COST_CEILING_USD=2000.0`, `ARGUS_SCHEMA_REJECT_STREAK=5`, `ARGUS_STALE_JUSTICE_TIMEOUT_SECONDS=300`.

The remaining rules (6 PREMATURE_STOP, 7 RELIABILITY_DECAY, plus the rule-3 latency-cascade arm) are NEW rules with NO locked SCOPE literal (RESEARCH A1). Their threshold VALUES are pre-registered HERE before the measured run, so the detection claim is falsifiable rather than fitted:

| Threshold knob | Pre-registered value | Rule | Rationale |
|---|---|---|---|
| `ARGUS_PREMATURE_CONFIDENCE_THRESHOLD` | **0.40** | 6 | A council emission whose chosen-candidate max-mass < 0.40 is "low confidence"; two consecutive such emissions from one agent is the premature-convergence/stall proxy. 0.40 sits above a uniform 4-candidate prior (0.25) so a flat posterior is flagged, but below a decisive call (≥ 0.5). |
| `ARGUS_PREMATURE_CONSECUTIVE_N` | **2** | 6 | Two-in-a-row removes single-emission noise while staying sensitive (no `round_id`/turn-budget column exists — A3, scoped). |
| `ARGUS_RELIABILITY_DIAGONAL_FLOOR` | **0.40** | 7 | Mean diagonal mass of the row-normalized confusion matrix below 0.40 is a poor-accuracy/decayed agent (a perfectly-calibrated agent has diagonal mass → 1.0; chance for the 4-candidate verdict space is 0.25). |
| `ARGUS_RELIABILITY_STALE_WINDOW_SECONDS` | **604800** (7 days) | 7 | An agent_reliability matrix not updated in > 7 days is stale enough to warrant an operator look. |
| `ARGUS_TIMEOUT_LATENCY_MS` | **60000** (60 s) | 3 (latency arm) | The sandbox wall-clock timeout is 60 s (SCOPE §22 sandbox security); an LLM call exceeding 60 s latency is the derivable proxy for a stuck/cascading call. |
| `ARGUS_TIMEOUT_CASCADE_N` | **3** | 3 (latency arm) | Three consecutive >60 s calls from one agent is a cascade, not a single slow call. |

These NEW-rule thresholds are FROZEN for the EXP-055 run; any later change requires a STATE.md decision-log entry AND a new pre-registration block (per-rule empirical auto-tune from the false-positive corpus is a forward-dated SEED, not 14.0 work — RESEARCH §SEED).

### Phase 14.0 Argus close-gate — per-rule binary predictions (EXP-055)

14 predictions total — for each of the 7 rules, one FIRE prediction and one FALSE-POSITIVE prediction. All $0.00 LLM (no `live_llm` marker, no model client constructed in the harness — D-02), deterministic, and reproducible. Reuses real-artifact SHAPES where they exist (RESEARCH Q6): the EXP-052 stale-dispute shape (rule 5) + EXP-052 court schema-ok rows (rule 4 FALSE-POSITIVE class); the EXP-003-011 council hypothesis/posterior shapes (rules 1, 6); the 13.05 `agent_reliability` rows (rule 7). Each detection produced during the run is asserted to write `intervention_type='warning'` with a non-null `trigger_kind` and a non-empty replayable `context` JSONB (read back from the DB).

- **A-1 (rule 1 FIRES)** — REPEATED_HYPOTHESIS fires on 3 same-author hypothesis nodes with IDENTICAL embeddings (pairwise cosine 1.0 ≥ 0.95).
- **A-1' (rule 1 SILENT)** — REPEATED_HYPOTHESIS does NOT fire on 3 same-author hypotheses with DISTINCT (orthogonal) embeddings (pairwise cosine 0.0 < 0.95).
- **A-2 (rule 2 FIRES)** — COST_CEILING fires when the durable `SUM(llm_calls.cost_usd)` is ≥ $1800 (90% of the $2000 ceiling).
- **A-2' (rule 2 SILENT)** — COST_CEILING does NOT fire when the sprint-global cost SUM is < $1800 (the real EXP-052 court spent ~$0.18, well under).
- **A-3 (rule 3 FIRES)** — TIMEOUT_CASCADE fires on a persisted sandbox JSONL `_incomplete=true` envelope with `reason='timeout'` (sandbox arm) AND on ≥ 3 consecutive >60 s-latency `llm_calls` from one agent (latency arm).
- **A-3' (rule 3 SILENT)** — TIMEOUT_CASCADE does NOT fire on a clean finalize-only sandbox run (`_incomplete=false`, no violation reason) nor on normal-latency calls.
- **A-4 (rule 4 FIRES)** — SCHEMA_REJECT_CASCADE fires on 5 consecutive `request->>'schema_validation'='failed'` rows from one agent.
- **A-4' (rule 4 SILENT)** — SCHEMA_REJECT_CASCADE does NOT fire when a `'passed'` row breaks the most-recent-5 streak (the EXP-052 court rows were all schema-ok — the canonical no-fire class).
- **A-5 (rule 5 FIRES)** — STALE_JUSTICE_CASE fires on an `status='open'` dispute aged 10 min (> the 300 s timeout); detection-only (NEVER mutates `disputes.status` / NEVER closes contested — that is JSTC-03/Phase 14.1).
- **A-5' (rule 5 SILENT)** — STALE_JUSTICE_CASE does NOT fire on an `status='open'` dispute aged 1 min (within the 5-min timeout).
- **A-6 (rule 6 FIRES)** — PREMATURE_STOP fires on 2 consecutive council emissions from one agent with chosen-candidate max-mass < 0.40.
- **A-6' (rule 6 SILENT)** — PREMATURE_STOP does NOT fire on 2 high-max-mass (≈ 0.90) emissions.
- **A-7 (rule 7 FIRES)** — RELIABILITY_DECAY fires on a low-diagonal-mass (< 0.40) AND 30-day-stale `agent_reliability` matrix (evidence_count > 0).
- **A-7' (rule 7 SILENT)** — RELIABILITY_DECAY does NOT fire on a sharp (high-diagonal) freshly-updated matrix, and cold-start (evidence_count == 0) uniform priors are SKIPPED (false-positive guard).

**Cost prediction:** $0.00 LLM. The detection path imports NO model client, constructs NONE, and carries NO `live_llm` pytest marker (D-02 / SCOPE §22.6); the harness additionally asserts the replay created ZERO new `llm_calls` rows. The no-LLM grep gate (`argus_audit._check_no_llm` + `test_no_llm_in_argus.py`) independently guards the source.

**Known limitations (recorded loudly per `engineering_discipline_no_coverups` — the plan-check scoping warning):**
- **Rule 3 (TIMEOUT_CASCADE) is SCOPED.** There is no `executions` table in the schema yet (RESEARCH Pitfall 1 — VERIFIED), so the full sandbox-DB row with explicit `timed_out` / `exit_code` columns is FORWARD-DATED to whichever plan first persists sandbox runs to Postgres. Phase 14.0 detects the PERSISTED, replayable signals: (a) the sandbox JSONL `_incomplete=true` envelope + the classify `timeout`/`oom` reason codes (sink.py D-10 row shape, A2), and (b) the `llm_calls.latency_ms` cascade proxy. Both fire and both are replayable; the DB-row arm is the documented forward-dated extension.
- **Rule 6 (PREMATURE_STOP) is SCOPED.** No `round_id` / turn-budget column exists on `council_hypothesis_posteriors` (alembic 0014 — RESEARCH Pitfall 4 / A3), so the "no new query issued" / "turn budget remaining" sub-conditions are FORWARD-DATED. Phase 14.0 detects the persisted consecutive-low-confidence emission SEQUENCE per agent — deterministic and replayable now; the round-structure refinement is the documented forward-dated tightening.

**Falsification condition:** if any of the 14 binary predictions is wrong (a FIRE rule stays silent, or a FALSE-POSITIVE trace fires), the close-gate FAILS and is reported honestly — the root cause is fixed (never the test weakened). The EXP-055 measured outcomes (14 verdicts) are scored against THIS block in `.planning/EXPERIMENTS.md`, in a SEPARATE commit landing strictly AFTER this one (§22.1 git-proven).

**Cross-reference (the measured replay):** the D-09 $0-LLM deterministic replay over these 14 predictions is logged as **EXP-055** in `.planning/EXPERIMENTS.md` (Phase 14.0 Plan 04 Task 2). Per §22.1, this pre-registration commit precedes the EXP-055 outcome commit.

---

## Pre-registered block added 2026-05-21 (Phase 14.1 Argus intervention authorities — ARGUS-02 / JSTC-03 / D-09 close-gate, EXP-056)

This is a LOCKED pre-registration for the Phase 14.1 enactment close-gate (EXP-056). It binds under `SCOPE.md §22.1` (git-proven, like the EXP-053/EXP-054/EXP-055 deterministic-replay precedents) and lands in its OWN commit BEFORE any EXP-056 measured outcome (the measured D-09 $0-LLM replay is Plan 05 Task 2, a SEPARATE later commit — `git log` will prove the ordering: this predictions commit is a strict ancestor of the EXP-056 outcomes commit). Phase 14.1 (Argus intervention authorities) generates NO LLM output (D-02 / SCOPE §22.6 — enactment is deterministic SQL + asyncio over tables/tasks that already exist; Argus is the external read-and-log overseer that ISSUES authority records, never a research participant). So `validate_live_before_phase_close`, adapted for a deterministic phase exactly as EXP-055 was, here = running the REAL issue→poll→enact production path over synthetic kill/reset/halt/close scenarios at $0 LLM and recording the result against THESE predictions.

The headline 13.1 lesson binds this block: a piece-wise pass (the functions exist) is NOT a validation. The predictions below assert the FULL path RUNS end-to-end — Argus ISSUES (writes the pending `interventions` row) → the orchestrator/Justice poll READS + atomically CLAIMS it → the PHYSICAL action happens (agent task cancelled / `runs.status='halted'` / dispute `status='contested'`) → `enacted_at` is stamped — not just that the pieces exist in isolation.

### Phase 14.1 enactment close-gate — pre-registered values (no new SCOPE literal)

The action vocabulary (`kill`/`reset`/`halt`/`stale_case_close`) and the cross-process control channel (`interventions.enacted_at` NULL=pending) are mechanism, not thresholds. The two LITERAL values the gate keys on are documented as canonical (NOT new directional predictions): the halt naming convention `halt_reason='argus_halt:<rule>'` (D-04 / Pitfall 1) and the terminal `runs.status='halted'` (D-04 / ROADMAP success criterion). The poll cadence `ARGUS_ENACTMENT_POLL_SECONDS=30` (1 s in the fast-cadence test) and the conservative escalation toggle `ARGUS_ESCALATE_COST_CEILING_TO_HALT` are operator knobs, not falsifiable thresholds. No new threshold is fitted from the scenario data.

### Phase 14.1 enactment close-gate — the six binary scenario predictions (EXP-056)

Six scenarios (RESEARCH §"Deterministic validation scenarios" — D-09). All $0.00 LLM (no `live_llm` marker, no model client constructed or imported), deterministic, reproducible. Each runs the REAL production code (Argus `issue_intervention`; the orchestrator `enactment.poll_pending`/`claim`/`_enact_one` + the full `run()`→`shutdown()` halt path; Justice `_enact_all_pending_stale_case_closes`) — not mocks. Each enacted action's `enacted_at` is read back from the DB to confirm the issue→enact trail closed.

- **B-1 (KILL end-to-end)** — issuing a `kill` for a target agent, then driving the orchestrator poll→claim→enact, cancels the target's asyncio task (the agent OBSERVES `CancelledError` — the cancel is delivered, not swallowed), respawns a FRESH supervise task (restart clean, run continues N−1 in between), does NOT add the agent to `_halted_agents` (kill ≠ 3-strike halt — Pitfall 2), and stamps `enacted_at`. The bystander agent is untouched.
- **B-2 (RESET end-to-end)** — issuing a `reset`, then poll→claim→enact, calls `agent.reset()`: `system_prompt` is restored to the fresh canonical body, `_round_iteration_count==0`, `agent_id` is UNCHANGED (D-19 stable identity), and `enacted_at` is stamped. No DB write by reset; in-memory only.
- **B-3 (HALT end-to-end with audit chain)** — issuing a `halt` shortly after `run()` starts, the orchestrator's OWN poll-and-enact coroutine reads+claims+enacts it: `_stop_event` fires, ALL supervise loops exit, `shutdown()` writes `runs.status='halted'` + `halt_reason='argus_halt:cost_ceiling'` (NOT `'completed'` — Pitfall 1), the run result carries that halt_reason, and `enacted_at` is stamped. This is the FULL cross-process path through the real `run()` loop.
- **B-4 (JSTC-03 stale dispute CLOSED contested)** — issuing a `stale_case_close` for an `status='open'` dispute (Argus's authority record; the dispute id rides in `context->>'dispute_id'`), then the REAL Justice sweep poll→claim→`close_stale_case` UPDATEs the dispute `open`→`contested`, stamps `enacted_at`, and Argus NEVER wrote `disputes` (its row is the `stale_case_close` issue; Justice performed the UPDATE — the two are distinct authorities).
- **B-5 (boundary invariant holds)** — `tests/argus/test_argus_structural_separation.py` stays byte-unchanged green (the git-blob frozen + the 3 grep checks pass), the SEED-004 forbidden-table/verb set is unchanged, and the enactment-authority backing invariant is green.
- **B-6 (WR-02 green under live halt)** — a sink raising `CancelledError` mid-`_emit_row` propagates up through the queue (the D-06 emit-row regression), AND the kill/halt enactment path propagates `CancelledError` to the target agent (proven non-vacuously in B-1's `cancelled_observed`) — no swallowed cancellation, no silent-drop (a swallowed cancel would be a DoS-of-safety).

**Authority invariant prediction (D-04 / C-05):** over the seeded DB, every enacted Argus-authority action (an `interventions` row in {kill,reset,halt,stale_case_close} with non-NULL `enacted_at`) is backed by a logged authorizing intervention (non-NULL `trigger_kind`); the enactment-without-authorization population is exactly 0. The orchestrator's pre-existing 3-strike/budget halt is a DIFFERENT authority (writes NO `interventions` row, uses a non-`argus_halt:` `halt_reason`) and is asserted distinct — not gated by this invariant.

**Cost prediction:** $0.00 LLM. The enactment path imports NO model client, constructs NONE, and carries NO `live_llm` marker (D-02 / SCOPE §22.6); the EXP-056 harness asserts the replay created ZERO new `llm_calls` rows. The argus no-LLM grep gate independently guards the source.

**No new package / no new migration prediction:** `git diff pyproject.toml poetry.lock` is EMPTY (no new package — T-14.1-SC accepted); alembic head stays 0019 (no new migration beyond the Wave-1 0018 `enacted_at` add + 0019 `runs_halt_reason_check` widening).

**Falsification condition:** if any of the six scenarios fails (a kill that doesn't cancel, a halt that writes `'completed'`, a close that doesn't flip `contested`, an enacted action with no backing authority, a swallowed cancel, the boundary test edited, a new package/migration), the close-gate FAILS and is reported honestly — the root cause is fixed, never the test weakened or silently mocked. The EXP-056 measured outcomes are scored against THIS block in `.planning/EXPERIMENTS.md`, in a SEPARATE commit landing strictly AFTER this one (§22.1 git-proven).

**Cross-reference (the measured replay):** the D-09 $0-LLM deterministic end-to-end replay over these six scenarios is logged as **EXP-056** in `.planning/EXPERIMENTS.md` (Phase 14.1 Plan 05 Task 2). Per §22.1, this pre-registration commit precedes the EXP-056 outcome commit.

---

## Pre-registered block added 2026-05-23 (Phase 15.1 Alien Universe parameter-tuning sweep — TIER2-02 / D-11, EXP-071)

This is a LOCKED pre-registration for the Phase 15.1 $0-LLM Alien Universe parameter-tuning sweep (**EXP-071**). It binds under `SCOPE.md §22.1` (git-proven, like the EXP-053/EXP-054/EXP-055/EXP-056 deterministic-replay precedents) and lands in its OWN commit BEFORE any sweep verdict runs — the measured outcomes land in `.planning/EXPERIMENTS.md` as the **EXP-071 record** in a SEPARATE, strictly-later commit (Plan 02 SUMMARY / session-close), so `git log` proves the §22.1 ordering: this predictions commit is a strict ancestor of the EXP-071 outcomes commit.

Phase 15.1 generates **NO LLM output** (D-10 / SCOPE §22.6 — all tuning, residual-structure analysis, held-out RMSE, and the discoverability audit are deterministic numpy/scipy/sympy with no agent in any path; same explicit opt-out from `validate_live_before_phase_close` as Phase 15.0). So the "measurement" here is running the deterministic `simulator/tuning.py` sweep over `TIER2_TUNED_PARAMS` + the `TIER2_HELD_OUT_ICS` set and recording the verdict against THESE predictions. $0 LLM is still a real, falsifiable measurement.

**EXP-id allocation note.** EXP-071 is the next free id in the 15.0/15.1 "Alien Universe calibration" block (`EXP-071 – EXP-090`, `.planning/EXPERIMENTS.md:599`). `grep -c "EXP-071" .planning/EXPERIMENTS.md` returns `1` — and that single hit is the block-table *label* `| EXP-071 – EXP-090 | Alien Universe calibration |`, NOT a pre-existing record/entry; `grep -c "EXP-071" predictions.md` returned `0` before this block. So no prior EXP-071 record exists, consistent with the EXP-052..056 "highest pre-existing record + 1" precedent (highest pre-existing record is EXP-056). The OUTCOMES block lands in `.planning/EXPERIMENTS.md` in a LATER commit (Plan 02 SUMMARY / session-close) — predictions only here, no outcome numbers.

### Phase 15.1 tuning-sweep target band — the six binary predictions (EXP-071)

All $0.00 LLM (no `live_llm` marker, no model client constructed, no DB write, no new package — D-10 / SCOPE §22.6), deterministic, and reproducible. The "8 seeds per cell" §22.1 rigor has no literal meaning for RNG-free dynamics, so the **deterministic seed-analog is robustness across initial conditions**: every per-IC claim must hold on EVERY held-out IC, not as a single aggregate number (RESEARCH §the deterministic seed-analog).

- **P-1 (non-trivial / Newton provably fails)** — At `TIER2_TUNED_PARAMS`, across all ≥7 residual-structure held-out ICs, the nested-model F-test (Newton → +α → +hidden-charge, fit to the per-step radial acceleration recovered from observable positions) rejects the white-noise null at **p_alpha < 0.01 AND p_hidden < 0.01 on EVERY IC** (per-IC, the deterministic seed-analog). The Newton-only fit additionally leaves a relative residual ABOVE a 1e-3 effect-size floor (the falsifiability gate that distinguishes real structure from FD-truncation noise — see P-4).
- **P-2 (solvable / true law fits held-out)** — The true law reproduces every `TIER2_HELD_OUT_ICS` trajectory at **held-out RMSE < 1e-3** (the §10 numerical-match method, fresh `AlienUniverse` per held-out IC).
- **P-3 (15.0 conservation bar preserved)** — Every admissible cell holds **Q-drift < 1e-6 over 1000 steps** (read from `traj.Q`, the polar-Hamiltonian invariant, NOT `physics.conserved_Q`) and bit-identical reproducibility.
- **P-4 (negative control / the test is not vacuous)** — In the TRUE Newtonian limit (α=0, β=0, **charges=0** so the κ tangential-inertia coupling is inert) the Newton-only residual sits at the finite-difference-truncation FLOOR (relative residual < 1e-3) and the cell is **NOT in band** — proving the structure test is falsifiable rather than firing for any smooth deterministic residual (RESEARCH Pitfall 2: a bare F-test p-value is vacuous here; the effect-size gate is what makes the rejection meaningful).
- **P-5 (non-polynomial / beyond the bar)** — A polynomial-FORCE central-law baseline fit to the SEEN orbit FAILS held-out forward integration at **RMSE > 1e-2** (the honest D-11 guard: held-out forward integration, NOT a per-orbit polynomial-residual fit, which is the L-015 trap). The polynomial baseline lacks the hidden tangential-inertia DOF, so it cannot reproduce the held-out angular dynamics.
- **P-6 ($0 LLM)** — The entire sweep + audit incurs **$0.00 LLM cost**, constructs no model client, carries no `live_llm` marker, writes no DB row, and adds no new package (`git diff pyproject.toml poetry.lock` is EMPTY — numpy 2.4.4 / scipy 1.17.1 / sympy 1.14.0 already locked).

**Falsification condition:** if any of the six predictions is wrong (Newton fits the residual at the tuned point, the true law fails to reproduce a held-out trajectory, a cell violates the conservation bar, the Newtonian-limit negative control fires in-band, the polynomial baseline succeeds held-out, or any LLM cost / new package appears) the result is reported HONESTLY — the root cause is fixed, never the prediction retconned or the test weakened. The EXP-071 measured outcomes are scored against THIS block in `.planning/EXPERIMENTS.md`, in a SEPARATE commit landing strictly AFTER this one (§22.1 git-proven).

**Cross-reference (the measured sweep):** the D-11 $0-LLM deterministic sweep over `TIER2_TUNED_PARAMS` + `TIER2_HELD_OUT_ICS` is logged as **EXP-071** in `.planning/EXPERIMENTS.md` (Phase 15.1 Plan 02 SUMMARY / session-close). Per §22.1, this pre-registration commit precedes the EXP-071 outcome commit.

---

## Pre-registered block added 2026-05-23 (Phase 15.2 Alien Universe harness integration — the LIVE Council-vs-Alien discovery-depth runs, TIER2-03 / TIER2-04, EXP-072)

This is a LOCKED pre-registration for the Phase 15.2 **live** Council-vs-Alien measurement (**EXP-072**), approved by Surya at the D-12 gate (2026-05-23). It binds under `SCOPE.md §22.1` and lands in its OWN commit BEFORE any live run fires — the measured outcomes land in `.planning/EXPERIMENTS.md` as the **EXP-072 record** in a SEPARATE, strictly-later commit, so `git log` proves the §22.1 ordering (this predictions commit is a strict ancestor of the EXP-072 outcomes commit).

Unlike EXP-071 (a $0-LLM deterministic sweep), **EXP-072 spends real LLM** — the five-agent Council runs live (gemini-2.5-pro on [GCP project]; NVIDIA fallback for 429s) against the Alien Universe via the data-only `ObservationBundle`, with domain-blank Tier-2 prompts (no domain hints). Discovery depth is scored **POST-HOC** by the deterministic 6-layer `benchmarks/alien_depth.py` scorer (reads each hypothesis's `symbolic_form` via `_safe_parse_expr` — NO LLM judge, §22.6) over the run's persisted hypothesis nodes. In-loop selection ranks by held-out fit against the true law (never the ODE `score()` — the L-016 fix). The substrate (Phase 15.2 Plans 01-03) is green at $0 LLM; this is the live-validation step `validate_live_before_phase_close` requires.

**EXP-id allocation note.** EXP-072 is the next free id in the "Alien Universe calibration" block (`EXP-071 – EXP-090`); 15.1 used EXP-071. `grep -c "EXP-072"` returned `0` in both `predictions.md` and `.planning/EXPERIMENTS.md` before this block — no prior EXP-072 record exists ("highest pre-existing record + 1" precedent). Outcomes land in `.planning/EXPERIMENTS.md` in a LATER commit — predictions only here, no outcome numbers.

### Phase 15.2 live discovery-depth target band — seven binary predictions (EXP-072)

Cost ceilings RAISED for the live runs (per-run ~$30 for the 30-min TIER2-03 run, ~$75 for the depth-≥2-in-4hr TIER2-04 run); the hard rails STAY ($2000 sprint ceiling, Argus halt at 90%). Depth layers are §10's L1–L6.

- **P-15.2-1 (depth ≥ 2 — the floor)** — At least one live run reaches **discovery depth ≥ 2** within the 4-hour budget (L1 dominant inverse-power force + L2 correction term), and the L2 numerical gate (held-out acceleration-RSS reduction) passes. *[medium-high]*
- **P-15.2-2 (depth 3 — the differentiating bet / milestone target)** — At least one run reaches **depth 3** (hidden-charge EXISTENCE — the proposal posits an unobserved per-body property), the conceptual leap that distinguishes the Council from polynomial regression. *[medium]*
- **P-15.2-3 (ceiling)** — Depth 4 (hidden-force functional form) is reachable but NOT the median outcome; depth 5 (conservation law) / depth 6 (hidden symmetry) are NOT predicted within a single 4-hour run.
- **P-15.2-4 (numerical honesty)** — Any layer credited as a NUMERICAL match scores held-out **RMSE < 1e-3** on `TIER2_HELD_OUT_ICS` (the §10 numerical-match method) — no layer is credited on structural pattern-match alone.
- **P-15.2-5 (validity / no silent failure)** — All surviving Council findings pass `verifier.sign_off`; **zero** depth credits come from a silent confidence-fallback (the L-016 path WARNs loudly and is never silently credited).
- **P-15.2-6 (provenance / separation)** — Every alien run record carries `runs.config.benchmark_system='alien_universe'` AND `included_in_distillation_v1=False` (the audit-enforced guarantee Tier-2 trajectories never enter the AgentArk training set).
- **P-15.2-7 (no domain leak)** — The forbidden-token test holds on the live prompts: no domain name (lotka/volterra/newton/gravity/kepler/charge/alien/…) reaches the Council. The Council must DISCOVER the rules differ.

**Falsification condition:** if any prediction is wrong (no run reaches depth ≥ 2; a numerically-credited layer misses RMSE < 1e-3; a depth credit traces to a silent fallback; a provenance flag is absent; a domain hint leaks) it is reported HONESTLY — root cause fixed, never the prediction retconned, the depth rubric weakened, or a failed run re-rolled until it clears. EXP-072 outcomes are scored against THIS block in `.planning/EXPERIMENTS.md`, in a commit landing strictly AFTER this one (§22.1 git-proven).

**Cross-reference (the live runs):** the TIER2-03 30+min run + the TIER2-04 depth-≥2-in-4hr run are logged as **EXP-072** in `.planning/EXPERIMENTS.md` (Phase 15.2 session-close). Per §22.1, this pre-registration commit precedes the EXP-072 outcome commit.

### Amendment 2026-05-24 (EXP-072 prompt-method change — logged BEFORE the runs, §22.1)

Between the EXP-072 pre-registration (commit `e6b8852`, 2026-05-23) and the live measurement runs, the **alien task-description prompt was changed** (commit `31978d2`). This amendment records that change and its rationale on the record BEFORE any TIER2-03/04 run fires, so the §22.1 ordering is honoured and nothing is silently altered. **The seven predictions P-15.2-1..7 and the depth bands are UNCHANGED — they are NOT retconned.** Only the elicitation METHOD (the prompt wording) changed, for two reasons surfaced by the gate-approved live-smoke:

1. **Pinned observable variable names (a bug-fix).** The original domain-blank task did not PIN the observable variable names. The live-smoke (Surya-approved at the D-12 gate) caught a cascade: each run an agent invented a fresh component-naming convention (`vx0` → `x_0` → `p0x`/`p1y`) that tripped the agora write whitelist and discarded the hypothesis — so with the *old* prompt the measured depth would be a measurement artifact (≈0, no hypotheses surviving), not a discovery signal. The fix labels every data column with its exact pinned symbol (`r, x{j}, y{j}, vx{j}, vy{j}, v{j}, m{j}`) and tells the Council to reuse only those names. This is a root-cause fix of the elicitation pipeline, not a change to what counts as depth. The prompt stays **domain-blank** (the forbidden-token test still holds — P-15.2-7 intact).

2. **Radial framing clarified to match the locked scorer.** The ask was clarified from "the RHS of the equation of motion for the first coordinate of the first body" to "acceleration as a single expression in the inter-body distance `r`." Reason: the locked Plan-02 depth scorer (`benchmarks/alien_depth.py`) credits §10 layers from a **radial** law in `r`; a correct *Cartesian-componentwise* force renders as `~(x2-x1)/r**3` (radial exponent −3), which the radial scorer reads as depth 0. Without the radial framing the depth runs would measure a prompt/scorer-representation mismatch, not discovery quality. The pre-existing in-prompt example was already radial (`G*r - alpha/r**2`), so this aligns the prompt with the committed scorer rather than weakening the rubric. **The depth rubric (the six layer gates + numerical thresholds) is byte-unchanged.**

**Integrity note:** this amendment is additive and dated; it does not edit the original P-15.2-1..7 block above. The EXP-072 outcomes commit (in `.planning/EXPERIMENTS.md`) lands strictly AFTER both this pre-registration block and this amendment, and is scored against the unchanged predictions. If a depth prediction falsifies, it is reported honestly per the falsification condition above — the prompt change is not a license to re-roll.

---

## Pre-registered block added 2026-05-25 (Phase B — Autonomous Iterative Discovery Loop: residual feedback + novelty + memory; the EXP-073 control-vs-treatment ablation)

This is a LOCKED pre-registration for the **EXP-073** ablation, binding under `SCOPE.md §22.1`. It lands in its OWN commit BEFORE any EXP-073 measurement run fires; the measured outcomes land in `.planning/EXPERIMENTS.md` as the EXP-073 record in a SEPARATE, strictly-later commit, so `git log` proves the §22.1 ordering (this predictions commit is a strict ancestor of the EXP-073 outcomes commit).

**Motivation.** EXP-072 was an honest negative: across 375 live hypotheses the five-agent Council robustly rediscovers the inverse-square force (discovery depth 1) but never reaches the truth's `r**-3.5` correction (depth ≥ 2). The $0 spike (`scripts/spike_residual_feedback.py`, SEED-017) localized the cause: the agents reason BLIND — they never see how wrong their law is or what structure is left over, so they cannot iterate like a scientist (fit → inspect residual → refine). Phase B adds exactly that missing channel: a DETERMINISTIC, leak-safe residual-feedback signal (`benchmarks/residual_feedback.py`) computed at each round-close from the BEST law and surfaced into the next round's prompt, plus a deterministic novelty/motivation hint and a search-trail memory (all gated by the single A/B flag `COUNCIL_RESIDUAL_FEEDBACK_ENABLED`). EXP-073 measures whether closing that loop raises genuine discovery depth.

**EXP-id allocation note.** EXP-073 is the next free id in the "Alien Universe calibration" block (`EXP-071 – EXP-090`); 15.1 used EXP-071, 15.2 used EXP-072. `grep -c "EXP-073"` returned `0` in both `predictions.md` and `.planning/EXPERIMENTS.md` before this block — no prior EXP-073 record exists ("highest pre-existing record + 1" precedent). Outcomes land in `.planning/EXPERIMENTS.md` in a LATER commit — predictions only here, no outcome numbers.

### The ablation — arms, metric, statistic

- **Control arm:** `COUNCIL_RESIDUAL_FEEDBACK_ENABLED=False` — the EXP-072 blind loop, byte-for-byte unchanged behavior (the flag gates BOTH the round-close compute/persist AND the base.py read/render seam, so the control arm is a pure toggle).
- **Treatment arm:** `COUNCIL_RESIDUAL_FEEDBACK_ENABLED=True` — the iterative discovery loop (residual feedback + novelty hint + search-trail memory).
- **Rigor:** 8 seeds per arm, SAME wall-clock budget per run, SAME alien truth config (`TIER2_TUNED_PARAMS`) + `TIER2_HELD_OUT_ICS`, SAME domain-blank pinned-name Tier-2 prompt as EXP-072 (so the ONLY difference between arms is the feedback channel).
- **Primary metric:** `alien_depth.score_depth(...).max_depth`, computed **POST-HOC** over each run's persisted hypothesis nodes — the SEPARATE deterministic oracle (no LLM judge, §22.6), the SAME scorer EXP-072 used, byte-unchanged. The in-loop residual signal is NEVER the headline judge; a reviewer can delete every Phase B mechanism and recompute the identical depth score.
- **Statistic:** Wilcoxon signed-rank, paired by seed, one-sided (treatment ≥ control), per the pre-committed statistical-machinery table below. Secondary descriptive: per-arm count of hypotheses bearing an `r**-3.5` term (the spike's `has_alpha_exponent`) and a depth-progression-over-rounds curve.

### EXP-073 binary predictions

- **P-EXP-073-1 (the headline — depth lift)** — At least one TREATMENT run reaches `score_depth.max_depth ≥ 2` (genuine L2: the `r**-3.5` correction, passing the L2 held-out RMSE-reduction gate) within the per-run budget, while CONTROL runs stay at `max_depth = 1` (the EXP-072 result). *[medium]*
- **P-EXP-073-2 (distributional shift)** — Treatment `max_depth` stochastically dominates control across the 8 paired seeds: the Wilcoxon one-sided test rejects "treatment ≤ control" at α = 0.05. *[medium-low — discrete ordinal data; reported with the per-seed table regardless]*
- **P-EXP-073-3 (the mechanism fires)** — Treatment runs emit strictly MORE hypotheses bearing a second radial term (an `r**-p`, `p>2`, term) than control — direct evidence the residual feedback moved the search toward the leftover, not just noise. *[medium-high]*
- **P-EXP-073-4 (no LLM judge — §22.6)** — The headline depth metric is `alien_depth.score_depth` (deterministic sympy/scipy), unchanged from EXP-072; ZERO depth credit originates from the in-loop residual signal. The residual-feedback module makes ZERO model calls (the no-LLM grep gate holds). *[hard prediction — must hold]*
- **P-EXP-073-5 (anchor to ground truth — L-016)** — The residual feedback fits the OBSERVED acceleration (`_observed_accel` over `TIER2_HELD_OUT_ICS`), never self-reported confidence and never the truth-force oracle; identical `symbolic_form` → identical feedback regardless of any agent's confidence. *[hard prediction — must hold]*
- **P-EXP-073-6 (validation is not circular — no leak, §10)** — The rendered feedback block reveals only the leftover SHAPE (a coarse "steeper inverse-power" band + oscillation presence); the forbidden-token test holds on every persisted block (no `3.5`, no truth coupling value, no exponent, no oscillation wavelength, no domain word) — so a depth-2 discovery is a genuine discovery, not a copy of a revealed answer. *[hard prediction — must hold]*

**Falsification condition.** If P-EXP-073-1/2/3 are wrong — treatment does NOT raise depth above the EXP-072 control baseline — it is reported HONESTLY as a second negative, NOT retconned, the rubric NOT weakened, no run re-rolled. **A null result here is itself publishable and informative:** it would localize the depth-1 plateau as a deeper reasoning/credit-assignment limit (one feedback channel is insufficient) and point directly to the deferred champion hill-climb + Popperian-falsification phases as the next levers. If any integrity prediction (P-EXP-073-4/5/6) is violated, that is a STOP-and-fix defect, not a result. EXP-073 outcomes are scored against THIS block in `.planning/EXPERIMENTS.md`, in a commit landing strictly AFTER this one (§22.1 git-proven).

**Mechanism note (implementation, on the record before the run).** Component 3 ("search-trail memory") is implemented as a DETERMINISTIC trail composed in `round_close` from the cumulative exponent-signature histogram and appended to the persisted residual block — a deliberate refinement of the originally-discussed LLM-narrated `populate_criteria_from_round` path, chosen because a deterministic trail is strictly safer on the no-LLM-in-feedback (§22.6) and no-leak (§10) axes and carries the same "inherited search history" intent. This does not affect any prediction above; it is recorded here so the method is fixed before measurement.

### Amendment 2026-05-25 (EXP-073 elicitation-method change — integer-exponent anchoring fix, logged BEFORE the measurement runs, §22.1)

A cheap PILOT (1 replicate × 2 arms, 15-min each, $6.40 total) run AFTER this pre-registration but BEFORE the confirmatory measurement surfaced a method artifact. The treatment arm's residual loop engaged exactly as designed — the round-close persisted 9 leak-safe `council_residual_feedback` rows (control wrote 0), and the agents demonstrably read them: treatment proposals were saturated with steeper-power + oscillatory `cos` terms (and creative forms like `exp(-alpha*r)` screening and a variable exponent `r**(2+alpha*cos(...))`), whereas the control arm collapsed to the inverse-square plateau. BUT both arms scored `max_depth=1`: the treatment's second radial terms were consistently INTEGER powers (`r**-3`, `r**-4` — the endpoints the block named "between 1/r**3 and 1/r**4"), NEVER the half-integer `r**-3.5` the L2 gate requires. This is an anchoring ARTIFACT of the block wording, not a depth ceiling.

This amendment records the fix on the record BEFORE any measurement run: the residual block's steeper-power line now tells the Council the exponent need NOT be a whole number and to fit it as a CONTINUOUS value, rather than naming integer endpoints. The fix is LEAK-SAFE — it reveals NO exponent value (no `3.5`), no coupling, no domain word; the anti-leak regression test holds. It is a root-cause fix of an elicitation artifact, directly analogous to the EXP-072 pinned-name amendment. **Predictions P-EXP-073-1..6 and the depth gates are UNCHANGED — NOT retconned; only the elicitation wording changed to remove the integer anchor. The depth rubric (`score_depth`) is byte-unchanged.**

**Integrity note:** the pilot is DESCRIPTIVE (n=1/arm) and is NOT the pre-registered 8-seed measurement; its outcome is reported honestly regardless (a clean control-vs-treatment richness contrast at depth 1). If the refined block still fails to lift depth in the confirmatory run, that is reported as an honest negative per the falsification condition above — the fix is not a license to re-roll, and a continued depth-1 plateau points to the deferred champion-loop / Popperian-falsification phases.

**Cross-reference (the runs):** the EXP-073 control + treatment runs are logged as **EXP-073** in `.planning/EXPERIMENTS.md` (Phase B session-close). Per §22.1, this pre-registration commit precedes the EXP-073 outcome commit.

---

## Pre-registered block added 2026-05-26 (Phase C — Active experimentation: the EXP-074 observability-lift gate + the EXP-075 differentiating ablation)

This is a LOCKED pre-registration for the Phase C active-experimentation experiments (**EXP-074**, **EXP-075**), binding under `SCOPE.md §22.1`. It lands in its OWN commit BEFORE any measurement run fires; the measured outcomes land in `.planning/EXPERIMENTS.md` as the EXP-074 / EXP-075 records in a SEPARATE, strictly-later commit, so `git log` proves the §22.1 ordering (this predictions commit is a strict ancestor of the outcome commits). It is gated on the **2026-05-26 §10 active-experimentation amendment** (Surya signed off 2026-05-26), which authorizes (a) requesting new data-only `ObservationBundle`s from chosen initial conditions, and (b) a deterministic no-LLM instrument that fits the Council's OWN proposed structure to that clean data and returns the fitted value + RSS reduction. The headline depth metric is unchanged: the SEPARATE post-hoc `alien_depth.score_depth` oracle (§22.6), scored on `TIER2_HELD_OUT_ICS` the Council cannot choose.

**Motivation.** EXP-072 and the EXP-073 pilots are honest negatives at discovery depth 1; the $0 identifiability post-mortem (L-044 RESOLUTION) localized the cause to OBSERVABILITY — the `r**-3.5` correction is recoverable from a controlled radial-drop experiment (release from rest → v_perp=0 switches off the κ confound) but destroyed in passive orbit data. Phase C supplies the missing observability via active experimentation + the deterministic fitting instrument (`benchmarks/experiment_feedback.py`), gated by the single A/B flag `COUNCIL_EXPERIMENT_BATTERY_ENABLED`. EXP-074 (Stage 1, system-triggered) tests whether clean experimental data lifts depth; EXP-075 (Stage 2) tests whether the Council's engineered diversity adds value on top of the experimentation lever.

**EXP-id allocation note.** EXP-074/075 are the next free ids in the "Alien Universe calibration" block (`EXP-071 – EXP-090`); 15.1 used EXP-071, 15.2 used EXP-072, Phase B used EXP-073. `grep -c "EXP-074"` and `grep -c "EXP-075"` each returned `0` in both `predictions.md` and `.planning/EXPERIMENTS.md` before this block — no prior records ("highest pre-existing record + 1" precedent). Outcomes land in `.planning/EXPERIMENTS.md` in a LATER commit — predictions only here, no outcome numbers.

### EXP-074 — Stage 1 observability-lift (control vs treatment)

- **Control arm:** `COUNCIL_EXPERIMENT_BATTERY_ENABLED=False` — the Phase B residual loop on, no experiment battery (the EXP-073 treatment-arm behavior, byte-for-byte: the flag gates BOTH the round-close compute/persist AND the base.py read/render seam, so the control is a pure toggle).
- **Treatment arm:** `COUNCIL_EXPERIMENT_BATTERY_ENABLED=True` — the system-triggered clean-probe battery + fitting instrument, fired once a depth-1 plateau is detected.
- **Rigor:** 8 seeds per arm, SAME wall-clock budget per run, SAME alien truth config (`TIER2_TUNED_PARAMS`) + `TIER2_HELD_OUT_ICS`, SAME domain-blank pinned-name Tier-2 prompt as EXP-072/073 (so the ONLY difference between arms is the experiment battery).
- **Primary metric:** `alien_depth.score_depth(...).max_depth`, computed **POST-HOC** over each run's persisted hypothesis nodes — the SEPARATE deterministic oracle (no LLM judge, §22.6), byte-unchanged from EXP-072/073. The in-loop instrument signal is NEVER the headline judge; a reviewer can delete every Phase C mechanism and recompute the identical depth score.
- **Statistic:** Wilcoxon signed-rank, paired by seed, one-sided (treatment ≥ control), per the pre-committed statistical-machinery table below.

Binary predictions:

- **P-EXP-074-1 (headline — observability lift)** — TREATMENT reaches `score_depth.max_depth ≥ 2` (genuine L2: the `r**-3.5` correction, passing the L2 held-out RMSE-reduction gate) on **≥ 5 of 8 seeds**, while CONTROL stays at `max_depth = 1` (the EXP-072/073 result). *[medium]*
- **P-EXP-074-2 (distributional shift)** — Treatment `max_depth` stochastically dominates control across the 8 paired seeds; the Wilcoxon one-sided test rejects "treatment ≤ control" at α = 0.05. *[medium-low — discrete ordinal data; reported with the per-seed table regardless]*
- **P-EXP-074-3 (no LLM judge — §22.6)** — The headline depth metric is `alien_depth.score_depth` (deterministic sympy/scipy), unchanged; ZERO depth credit originates from the `experiment_feedback` instrument; the instrument makes ZERO model calls and the no-LLM grep gate holds on `experiment_feedback.py`. *[hard — must hold]*
- **P-EXP-074-4 (no leak — §10)** — The rendered experiment block names only the agent's OWN fitted structure (its exponent + RSS reduction); the forbidden-token test holds on every persisted block (no truth coupling value, no oscillation wavelength, no other-term value, no domain word; brace-free) — so a depth-2 discovery is a genuine discovery, not a copy of a revealed answer. *[hard — must hold]*
- **P-EXP-074-5 (data-only boundary — §10 amendment)** — Every serviced experiment returns an `ObservationBundle`-shaped `{t, positions, velocities, masses}`; no hidden invariant (charges *s* / α / β / γ / κ / Q / the law) is ever returned to the Council. *[hard — must hold]*
- **P-EXP-074-6 (anti-reward-hack)** — Requested / battery initial conditions are clamped to a legal band held strictly OFF `TIER2_HELD_OUT_ICS`; the held-out depth oracle scores ICs the Council cannot choose. *[hard — must hold]*

**Falsification condition.** If P-EXP-074-1/2 are wrong — treatment does NOT lift depth above the control baseline — it is reported HONESTLY as a third negative that relocates the depth-1 plateau back to reasoning/credit-assignment (NOT retconned, the rubric NOT weakened, no run re-rolled). A null here is publishable and informative. If any integrity prediction (P-EXP-074-3/4/5/6) is violated, that is a STOP-and-fix defect, not a result.

### EXP-075 — the differentiating ablation (Council-with-experimentation vs single-model-with-experimentation)

- **Arms:** full Council + experiment battery vs a single flagship model + the SAME experiment battery + the SAME fitting instrument, SAME budget, SAME metric.
- **Rigor:** 8 seeds per arm (deterministic seed-analog where applicable), Wilcoxon signed-rank one-sided.

Binary prediction:

- **P-EXP-075-1 (the headline differentiating claim)** — The full Council reaches strictly greater `score_depth.max_depth` than the single flagship model when both have the experiment battery — i.e. engineered diversity drives the deeper discovery on top of the experimentation lever, not the tool alone. *[medium]* **Pre-registered that the NULL is publishable:** if the single model with the tool matches the Council, the lever is the tool (experimental design + fitting), not the architecture, and that is reported honestly as a thesis-narrowing result.

**Cross-reference (the runs):** EXP-074 (Stage 1 control vs treatment) and EXP-075 (the Stage 2 differentiating ablation) are logged as **EXP-074** / **EXP-075** in `.planning/EXPERIMENTS.md` (Phase C session-close). Per §22.1, this pre-registration commit precedes both outcome commits.

### Amendment 2026-05-27 (EXP-074 outcome: FALSIFIED as-run; precondition unmet — infra confound, logged with the outcome)

This amendment records the EXP-074 outcome and its limitation per the amendment-discipline category "the data revealed a precondition was unmet" — NOT to reframe a falsified prediction as supported. **P-EXP-074-1/2 are FALSIFIED as run** (v1: treatment max_depth≥2 on 4/8, control reached 2 on 2/8, Wilcoxon p=0.50; v2: treatment 3/8, p=0.19) and are reported HONESTLY as such in EXPERIMENTS.md — the rubric was not weakened, no run was re-rolled, no seeds were cherry-picked. The integrity predictions **P-EXP-074-3/4/5/6 HELD** (no-LLM judge, no leak, data-only bundle, anti-reward-hack clamp).

The unmet precondition: a valid A/B requires **equal, gemini-served exploration per arm**. Both 8-seed batches violated it via this [GCP project] project's Vertex per-project quota exhaustion — v1's NVIDIA 429-fallback engaged on only 6/989 calls so back-half runs starved (L-045); v2's 429-cooldown-breaker fix routed to NVIDIA, but `llama-3.3-70b` cannot emit valid `CouncilHypothesis` JSON so early seeds yielded 0–4 hypotheses (depth 0, L-046). The mechanism itself is validated independently (the $0 end-to-end test + the $1.75 gemini-fresh smoke both reach `max_depth=2`; every gemini-healthy seed in v2 reached depth 2 on treatment). **P-EXP-074-1/2 are NOT retired or retuned** — they stand for a future CLEAN, gemini-served, powered measurement (quota relief or a structured-output-capable fallback; Surya's call). EXP-075 (Stage 2) remains gated on that clean EXP-074. Per §22.1, this amendment is additive and dated; it does not edit the EXP-074 block above, and the EXPERIMENTS.md outcome record is the authoritative honest report.

---

## Pre-registered block added 2026-05-27 (Phase C-depth3 — the L3+ observability-lift on the enriched multi-charge family)

This is a LOCKED pre-registration for the depth-3 experiment (**EXP-076**), binding under `SCOPE.md §22.1`. It lands in its OWN commit BEFORE any measurement run; the measured outcome lands in `.planning/EXPERIMENTS.md` as the EXP-076 record in a SEPARATE, strictly-later commit (`git log` proves the ordering). It is gated on the **2026-05-27 §10 observability amendment** (Surya signed off 2026-05-27), which provides the Council a FAMILY of systems governed by the SAME law but with DISTINCT per-body hidden charges — restoring the layer-3 "some configurations cannot be fit" signal the equal-charge TIER2 tuning removed.

**Motivation.** The $0 discoverability audit (`scripts/audit_depth_discoverability.py`, deterministic, no-LLM) PROVED that at the TIER2 equal unit charges (s₁=s₂=1.0), every hidden-charge factor that layers L3–L6 require is DEGENERATE: `s1*s2 ≡ 1` is byte-identical to a plain coefficient β (with/without-charge accel-fit RSS gap = 0; the fitted oscillatory amplitude tracks only the charge PRODUCT, so (1,1) and (2,0.5) are observationally identical), and `κ·s1·s2` collapses to a single constant (equal-product configs are bit-identical). The oscillatory force (γ̂=0.7) and the velocity coupling ARE identifiable as phenomena, but attributing them to a per-body hidden charge is not data-singled-out → L3–L6 are *verifiable but not discoverable* on the equal-charge benchmark (depth 2 is the honest ceiling there — an honest finding in its own right). The enrichment (distinct per-body charges across the observed family) makes the per-body charge identifiable from cross-configuration amplitude variation — exactly §10 layer 3.

**EXP-id allocation note.** EXP-076 is the next free id in the "Alien Universe calibration" block (`EXP-071 – EXP-090`): 071 (15.1), 072 (15.2), 073 (Phase B), 074/075 (Phase C). `grep -c "EXP-076"` returned 0 in both `predictions.md` and `.planning/EXPERIMENTS.md` before this block. Outcomes land in `.planning/EXPERIMENTS.md` in a LATER commit.

### EXP-076 — depth-3 observability-lift on the enriched multi-charge family (control vs treatment)

- **Setup:** the Council observes a family of 2-body systems sharing ONE law (same α/β/γ/κ, same functional form) with DISTINCT per-body hidden charges (products spanning e.g. {1, 2, 4, 6}); each system handed over as the standard data-only `ObservationBundle`. TREATMENT = the scientist's-escalation loop (the escalation-ladder methodology prompt + the cross-configuration probe diagnostics + the multi-parameter fitting instrument). CONTROL = single-charge family / escalation off (reproduces the depth-2 ceiling).
- **Primary metric:** `alien_depth.score_depth(...).max_depth`, computed POST-HOC over each run's persisted hypothesis nodes — the SEPARATE deterministic oracle (no LLM judge, §22.6), scored on the enriched held-out IC set the Council does not choose. The L3/L4 gates credit the hidden charge only when the per-config charge-product scaling (oracle-known) does ≥5% RSS work BEYOND a single global amplitude — so a global-β proposal cannot game L3.
- **P-EXP-076-1 (headline — depth-3 discovery)** — TREATMENT reaches `score_depth.max_depth ≥ 3` (writes `s1*s2` with the per-config scaling doing material held-out RSS work) on **≥ 5 of 8 seeds**, while CONTROL stays at `max_depth = 2`. *[medium]*
- **P-EXP-076-2 (distributional shift)** — Treatment `max_depth` stochastically dominates control across the 8 paired seeds; Wilcoxon one-sided rejects "treatment ≤ control" at α=0.05. *[medium-low — discrete ordinal; per-seed table reported regardless]*
- **P-EXP-076-3 (Council > single-model — the EXP-075 rail)** — The full Council reaches strictly greater `max_depth` than the single flagship model when both have the escalation machinery + the same family. *[medium]* **The NULL is publishable:** if the single model matches the Council, the lever is the tool, not the architecture — reported honestly as a thesis-narrowing result.
- **P-EXP-076-4 (no LLM judge — §22.6)** — headline depth is `score_depth` (deterministic sympy/scipy), unchanged; the escalation diagnostics + the fitter make ZERO model calls; the no-LLM grep gate holds. *[hard — must hold]*
- **P-EXP-076-5 (no leak — §10)** — the escalation diagnostics name only CATEGORY facts about the agent's OWN residual / measured cross-configuration differences (presence-only, per the blessed `residual_feedback.oscillation_present` precedent), and the fitter names only the agent's OWN proposed parameters; the forbidden-token test holds on every persisted block (no truth coupling value, no charge value, no γ, no domain word; brace-free). *[hard — must hold]*
- **P-EXP-076-6 (data-only boundary — §10)** — every observed system + serviced experiment returns an `ObservationBundle` `{t, positions, velocities, masses}`; the per-system charges / α / β / γ / κ / Q / the law are NEVER returned. *[hard — must hold]*
- **P-EXP-076-7 (anti-reward-hack)** — held-out scoring configurations (incl. their charges) are held strictly OFF the family the Council observes; the depth oracle scores configurations the Council cannot choose. *[hard — must hold]*

**Falsification condition.** If P-EXP-076-1/2 are wrong — the enriched family + escalation loop does NOT lift depth to ≥ 3 — it is reported HONESTLY as a finding that the barrier is reasoning/generative, NOT observability (not retconned, rubric not weakened, no run re-rolled). A null is publishable and informative: it would show that even with the hidden charge made identifiable, the Council cannot make the layer-3 conceptual leap. If any integrity prediction (P-EXP-076-4/5/6/7) is violated, that is a STOP-and-fix defect, not a result.

---

## Pre-registered block added 2026-06-02 (STRAT-01 — the call-matched same-compute ablation, **EXP-077**)

This is a LOCKED pre-registration binding under `SCOPE.md §22.1`. It lands in its OWN commit BEFORE any EXP-077 measurement run; the measured outcome lands in `.planning/EXPERIMENTS.md` as the EXP-077 record in a SEPARATE, strictly-later commit (`git log` proves the ordering).

**Motivation — closing the budget-vs-wall-clock gap in EXP-075.** EXP-075 confirmed the headline (full Council reaches strictly greater depth than the single flagship model; 8/8, Wilcoxon p=0.0039, Cliff δ=1.0). But that run matched **wall-clock, not compute**: in the shared 1200 s window the Council made ~90–170 LLM calls/run (5 agents) while the Solo made ~10–14 — roughly 10× more calls. The EXP-075 pre-registration above (line 392) says "SAME budget"; the run as fired matched wall-clock. So the cleanest reviewer kill is *"you gave the Council 10× the LLM calls — you measured budget, not architecture,"* and Cliff δ=1.0 is worthless against it because it measures the wrong contrast. The existing mitigant (the Solo missed L2 on 0/8 seeds → looks structural) is suggestive but NOT a controlled refutation: a starved Solo also fails consistently, so the 0/8 is consistent with BOTH a real architecture gap AND a throttling artifact. Only a call-matched run separates them.

**Design — match COMPUTE (total LLM calls), sweep the budget.** Both arms are re-fired under the STRAT-01 per-run call cap (`ASCENSION_PER_RUN_CALL_BUDGET` → the orchestrator call-budget watchdog, which drains the run at a clean agent/round boundary, never mid-deliberation, writing `runs.halt_reason='call_budget_exceeded'`), with a GENEROUS wall-clock backstop so the **call cap, not the clock, terminates the run**. We sweep three matched budgets **N ∈ {15, 40, 130}** to produce the depth-vs-equal-calls curve for each arm: N≈15 is the Solo's own natural budget (the Council is throttled hardest — ~3 calls/agent), N≈40 is a middle budget both arms can genuinely use, N≈130 is near the Council's natural budget (the Solo is boosted via a longer wall-clock). This makes the comparison fully compute-matched AND shows WHERE any architecture advantage holds, rather than betting the refutation on a single budget.

**EXP-id allocation note.** EXP-077 is the next free id in the "Alien Universe calibration" block (`EXP-071 – EXP-090`): 071–076 used. `grep -c "EXP-077"` returned 0 in both `predictions.md` and `.planning/EXPERIMENTS.md` before this block. Outcomes land in `.planning/EXPERIMENTS.md` in a LATER commit.

### EXP-077 — call-matched same-compute ablation, swept over the call budget (Council vs single-model at EQUAL total LLM calls)

- **Arms:** the EXP-075 configuration UNCHANGED (full Council vs single flagship model; SAME experiment battery + residual feedback + multi-charge family + fitting instrument), except BOTH arms are capped at the SAME total LLM-call budget N per run (the only changed variable). Sweep **N ∈ {15, 40, 130}**.
- **Rigor:** 8 seeds per (arm × budget) cell — 3 budgets × 2 arms × 8 = 48 runs. Wilcoxon signed-rank one-sided on the paired seeds within each budget; Holm-Bonferroni across the 3 budget comparisons; Cliff's delta per budget; 95% bootstrap CI. Generous wall-clock backstop per run (the call cap is the intended terminator; `halt_reason='call_budget_exceeded'` is the expected terminal reason, NOT `duration_exceeded`).
- **Primary metric:** the HONEST consecutive force-law depth from `alien_depth.score_depth` (L1→L4, stop at first False, per L-057), computed POST-HOC over each run's persisted hypothesis nodes by the SEPARATE deterministic oracle (no LLM judge, §22.6) — the SAME metric the EXP-075 outcome record cites, so the cells are directly comparable.

Binary predictions:

- **P-EXP-077-1 (headline — the architecture gap survives call-matching at usable budgets)** — At **N = 40 AND N = 130** (the budgets where neither arm is pathologically starved), the full Council reaches strictly greater consecutive `score_depth` than the single model on the paired seeds; Wilcoxon one-sided rejects "Council ≤ Solo" at α=0.05 after Holm-Bonferroni. *[medium]* **Pre-registered that the NULL is publishable:** if the Solo matches the Council at equal calls (N=40 and N=130), the EXP-075 depth gap was a COMPUTE artifact (the lever is the call budget / experiment battery, not the architecture), and EXP-075's claim is narrowed HONESTLY to "Council > Solo given equal wall-clock (more calls)" — the thesis-narrowing pre-committed at P-EXP-075-1 and PIVOT-001. EXP-075's outcome record is NOT edited or re-rolled; this is an additive correction.
- **P-EXP-077-2 (the curve — dominance across the sweep)** — The Council's depth-vs-call-budget curve lies at or above the Solo's at all three N; treatment `score_depth` stochastically dominates control within each budget. The full per-seed × per-budget table is reported regardless of significance. *[medium-low — discrete ordinal over 8 paired seeds]*
- **P-EXP-077-3 (the N=15 starvation floor — interpretation pre-committed BEFORE data)** — At N=15 the Council shares ~3 calls/agent and may be too starved to complete a deliberation. A **Council ≤ Solo result AT N=15 ONLY** (with Council wins at N=40 and N=130) is interpreted as a Council starvation floor and reported DESCRIPTIVELY — NOT as evidence against the architecture. Conversely a **Council WIN at N=15** is the single strongest cell (the architecture goes deeper even at the Solo's own call budget) and is reported as such. This interpretation is locked here so the harsh cell cannot be cherry-picked in either direction after seeing the data. *[medium]*
- **P-EXP-077-4 (call-matched validity — the contrast is genuinely compute-matched)** — Each counted run's realized `COUNT(*) llm_calls` lands within the watchdog's boundary overshoot of its target N (a few calls, from in-flight/concurrent agents at the stop boundary), and the two arms within a budget have statistically indistinguishable realized call counts. Any run that terminates BELOW N for a non-cap reason (agent 3-strike, natural completion, infra error → `halt_reason ≠ 'call_budget_exceeded'`) is FLAGGED and re-fired, never silently counted as an N-budget run. *[hard — must hold]*
- **P-EXP-077-5 (integrity rails inherited from EXP-075/076)** — The call cap changes ONLY when a run stops; it touches neither scoring nor the data boundary. Headline depth stays `score_depth` (deterministic sympy/scipy, §22.6, no LLM judge); the no-leak forbidden-token test and the data-only `ObservationBundle` boundary hold exactly as in EXP-075/076; held-out scoring ICs stay off the Council's choosable set. *[hard — must hold]*

**Falsification condition.** If P-EXP-077-1 is wrong — the Council does NOT exceed the Solo at the usable matched budgets (N=40, N=130) — the EXP-075 architecture claim is NARROWED honestly to an equal-wall-clock claim and the call budget is named as the operative lever (not retconned; EXP-075's record stands; rubric not weakened; no run re-rolled). That null is publishable and is exactly the thesis-narrowing PIVOT-001 pre-committed to. If P-EXP-077-1 HOLDS, the architecture claim is bulletproof against the compute-confound and the paper's central contrast survives the obvious attack. If any integrity prediction (P-EXP-077-4/5) is violated, that is a STOP-and-fix defect, not a result.

**Cross-reference (the runs):** EXP-077 (the 48-run call-matched sweep) is logged as **EXP-077** in `.planning/EXPERIMENTS.md` in a strictly-later commit. Per §22.1, this pre-registration commit precedes that outcome commit. The enforcement mechanism (per-run call cap + watchdog + `halt_reason='call_budget_exceeded'`, alembic 0023) landed in `6ceb7f5` BEFORE this pre-registration, which lands before the run — code-before-prereg-before-measurement, all separately committed.

---

## Pre-registered block added 2026-06-02 (STRAT-02 — the identifiability loop closed by the LIVE Council, **EXP-078**)

Binding pre-registration under `SCOPE.md §22.1`, in its OWN commit BEFORE the EXP-078 live run. This is a **capability smoke** (small n — one to a few live director calls, ~$0.01–0.25), not a powered statistical claim: it validates that the genuinely-novel contribution runs as ONE end-to-end artifact. The $0 deterministic core (`alien_council_loop.py`, the honesty boundary) and the figure renderer (`viz/loop_trace.py`) already shipped; EXP-078 swaps the stub proposer for the LIVE Heraclitus research-director (`council/director.py decide_next_move`).

**What it demonstrates.** On a degenerate single-configuration plateau (the `PlateauSignal` the diagnostic independently flags as unidentifiable), the LIVE Council director AUTONOMOUSLY chooses the next experiment; that choice is fed into the deterministic identifiability loop; the truth-blind OED ranker INDEPENDENTLY scores the designs the choice maps to; and the diagnostic re-checks — diagnose → design → re-diagnose, closed by a real agent decision, in one trace. The honesty boundary (decided in the open in `alien_council_loop.py`): the director's vocabulary is a FIXED two-move category, so the closure shown is that the live Council's CHOICE agrees with what the OED confirms is rank-restoring — not free-form product generation (that is future work and strengthens, not changes, the claim).

**EXP-id allocation note.** `grep -c "EXP-078"` returned 0 in both `predictions.md` and `.planning/EXPERIMENTS.md` before this block. The outcome lands in `.planning/EXPERIMENTS.md` in a strictly-later commit (the smoke script `scripts/run_strat02_live_smoke.py` is the run, not this lock).

### EXP-078 — live-Council identifiability-loop closure (capability smoke)

- **Setup:** `is_council_active()` True (real Council context); the live Heraclitus director is called once (or a few times) with the degenerate single-config `PlateauSignal`; the returned `NextMove` drives `run_alien_council_loop`; the deterministic OED ranker + diagnostic do the rest (no LLM). Per-run $2 dollar safety rail.
- **P-EXP-078-1 (the live closure)** — The director makes a REAL decision (`source='director'`, `fallback=False`); when that decision is `request_multi_charge_family`, the OED ranker INDEPENDENTLY selects a multi-distinct-product design (rejecting the status-quo single product) and the loop re-diagnoses to `outcome='solved'`. The diagnose→design→re-diagnose loop is closed by a live agent decision in one trace. *[medium]*
- **P-EXP-078-2 (load-bearing — the honest negative is pre-committed)** — The choice is load-bearing: if the live director instead chooses `request_new_initial_condition` (same single product), the OED cannot restore rank and the loop honestly fails to solve. A director IC-choice is reported as an HONEST NEGATIVE (a real, publishable result that the agent's decision determines closure), NOT retried-until-family and NOT a relabeled switch. *[medium]*
- **P-EXP-078-3 (no LLM in the loop/scoring — §22.6)** — Only the proposer (the director) calls an LLM. The identifiability diagnostic, the OED ranker, and the fit are deterministic (numpy/scipy/sympy); the loop/ranking/diagnosis make ZERO model calls. *[hard — must hold]*
- **P-EXP-078-4 (no leak — §10)** — The director decides from CATEGORY plateau facts only (`distinct_forms`, `dominant_share`, `rmse_flat`, the leak-audited `reason`); no truth value (per-body charges / α / β / γ / κ / the law) enters the director prompt or the decision. *[hard — must hold]*

**Falsification condition.** If the live director cannot make a real decision (always `fallback`/`substrate_noop` — an infra/model-availability failure), the smoke is INCONCLUSIVE and reported as such (infra, not a result), not dressed up as a closure. If the director makes a real decision but the loop does NOT close consistently with it (a family-choice that fails to solve, or an IC-choice that solves), that falsifies P-EXP-078-1/2 and is reported honestly. A small-n smoke does not claim a choice DISTRIBUTION — a powered version (K seeds, the director's family-vs-IC rate) is named as future work. If P-EXP-078-3/4 are violated, that is a STOP-and-fix defect, not a result.

**Cross-reference (the runs):** EXP-078 is logged as **EXP-078** in `.planning/EXPERIMENTS.md` in a strictly-later commit. Per §22.1, this pre-registration commit precedes that outcome commit; the $0 core (`23f6c34`) + figure (`6b4bf1c`) precede both.

---

## Pre-registered block added 2026-06-03 (STRAT-05 — the powered all-Gemini DEPTH-4 rate, **EXP-079**)

Binding pre-registration under `SCOPE.md §22.1`, in its OWN commit BEFORE any EXP-079 run; the outcome lands in `.planning/EXPERIMENTS.md` in a strictly-later commit.

**Motivation — promote depth-4 from an n=1 anecdote to a powered rate.** EXP-076 reached HONEST consecutive depth 4 on only **2/8** seeds, under the MIXED NVIDIA+Gemini roster. The ALL-Gemini config hit depth 4 in a SINGLE pilot (run `b9812937`, $5.71, EXPERIMENTS.md), explicitly n=1; "an all-Gemini powered run to measure the depth-4 rate" is a named, never-fired follow-up. Depth 4 = the whole force law (L1 + the r^-3.5 correction + the hidden charge doing real held-out RSS work + its functional form). "Rediscovers the FULL law structure reproducibly" is a strictly stronger headline than "depth ≥3". TWO-SIDED because we have no powered prior on the all-Gemini rate: a success bar AND the honest null are pre-registered with equal status, and the measured rate is reported as a number regardless of side.

**EXP-id allocation note.** `grep -c "EXP-079"` returned 0 in both `predictions.md` and `.planning/EXPERIMENTS.md` before this block (071–078 used). Outcomes land in EXPERIMENTS.md in a LATER commit.

### EXP-079 — powered all-Gemini depth-4 rate (single-arm rate measurement)

- **Arm (single):** the VALIDATED EXP-076 depth-4 stack, ALL-Gemini — full Council (`ASCENSION_COUNCIL_VARIANT=council`), 5 agents ALL on `gemini-2.5-pro` (`ASCENSION_MIXED_PROVIDER=0`), the multi-charge FAMILY feed + residual feedback + the EXP-074 exponent battery. The exact `b9812937` config, now at 8 seeds. No control arm: this is a RATE of a single known-winning config, not an ablation contrast.
- **Rigor:** 8 seeds (0–7), DISCRETE pinned single-episode runs (1200s each), same pinned prompts + isolated context as EXP-075/076. The rate is a binomial over 8 seeds; report the per-seed depth table + the exact-binomial (Clopper-Pearson) 95% CI on the depth-4 rate. No Wilcoxon (single arm); the comparison to EXP-076's mixed 2/8 is descriptive (provider is the only intended difference).
- **Primary metric:** HONEST consecutive force-law depth from `alien_depth.score_depth` (L1→L4, stop at first False, per L-057), POST-HOC, deterministic (no LLM judge, §22.6), scored `--family` on the held-out IC set — the SAME metric the EXP-076 record cites.

Rate predictions:

- **P-EXP-079-1 (headline — the depth-4 rate clears the powered bar)** — the all-Gemini Council reaches consecutive `score_depth = 4` on **≥ 5 of 8** seeds, with every seed at depth ≥ 3. *[medium]* **The two-sided NULL is pre-registered and publishable:** if depth 4 lands on **≤ 3 of 8** (the all-Gemini rate is indistinguishable from the mixed 2/8 — the exact-binomial 95% CI includes 0.25 and excludes 0.625), the honest finding is that **depth 3 is the reproducible ceiling and depth 4 is a tail event**, reported as the measured rate. The intermediate 4/8 is reported descriptively ("depth 4 on ~half of seeds, rate CI [0.16, 0.84]"), not rounded to the convenient side.
- **P-EXP-079-2 (the floor holds all-Gemini)** — consecutive depth ≥ 3 on **8/8** seeds; a regression below 3 would itself be a finding that the mixed roster (not all-Gemini) drove the floor, reported honestly. *[medium]*
- **P-EXP-079-3 (the depth-4 path is genuine — anti-L-057)** — on every seed scored at depth 4, the per-layer vector shows L2 AND L4 BOTH genuinely True on a real Council form (the `(s1*s2)*(α/r² + β/r^3.5)` structure), NOT the L-057 non-consecutive case. Depth-4 credits ONLY runs consecutively True through L4. *[hard — must hold]*
- **P-EXP-079-4/5/6 (integrity rails inherited)** — depth is `score_depth` (deterministic, §22.6, no LLM judge); the family/feedback/battery blocks pass the D-04 forbidden-token gates (no leak); every observed system is a data-only `ObservationBundle`, held-out scoring products {3,5} stay OFF the observed family {1,2,4,6}. *[hard — must hold]*

**Falsification condition.** If P-EXP-079-1's success bar is wrong (rate ≤ 3/8), it is reported HONESTLY as "depth 3 is the reproducible ceiling, depth 4 a tail event at rate r=[measured] (95% CI […])" — not retconned, rubric not weakened, no seed re-rolled, the depth-4 definition not loosened. That two-sided null bounds the ceiling the paper needs stated. Integrity-prediction violations are STOP-and-fix.

**Cross-reference (the run):** EXP-079 is logged as **EXP-079** in `.planning/EXPERIMENTS.md` in a strictly-later commit. The depth-4 config is the byte-identical EXP-076 treatment stack. Launch recipe (single council arm, all-Gemini, 8 seeds): `EXP075_CONFIRM=1 EXP075_ARMS="council" EXP075_MIXED_PROVIDER=0 EXP075_SEEDS="0 1 2 3 4 5 6 7" EXP075_CEILING_USD=10 EXP_LABEL=exp079 bash scripts/run_exp075_measured.sh` (~$40–50; fire off-peak/spaced — single-provider has no NVIDIA fallback to mask a Vertex 429, the EXP-074 lesson).

---

## Pre-registered block added 2026-06-03 (STRAT-03 — the SECOND hidden-physics regime, generalization vs overfit, **EXP-080**)

Binding pre-registration under `SCOPE.md §22.1`, in its OWN commit BEFORE any EXP-080 run. The V2 simulator config + the regime-aware scorer + the sum-confound diagnostic test landed in STRICTLY-EARLIER commits (the V2 build merge — code-before-prereg-before-measurement); the outcome lands in `.planning/EXPERIMENTS.md` in a strictly-later commit.

**Motivation — generalization is the antidote to "you tuned everything to your one toy law."** Every depth result to date (EXP-072→EXP-079) is on the SAME alien law `(s1*s2)*(α/r² + β/r^3.5)`. The two instruments that produced the depth lift are shaped to THIS law: the exponent battery fits exactly `r^-3.5`, the family feed restores exactly the `s1*s2` product. EXP-080 runs the EXP-076 stack UNCHANGED on a SECOND regime (V2) whose hidden structure is OFF both instruments' tuned bands, to test whether the machinery GENERALIZES or MEMORIZES (the reviewer attack PIVOT-001:16 names: "it only works on your one tuned law").

**The V2 regime (built + tested, V1 byte-identical):** the hidden charge enters as a SUM `s1+s2` (not a product — the family enrichment was not shaped for it; `structural.py` already detects the sum confound), and the correction term is `r^-2.5` (off the exponent battery's `r^-3.5` L2 band). The V2 simulator is Q-consistent (the §10 integrable invariant; Q-drift ~1e-8, dt-convergent) and the V2-regime scorer credits L2 only for `r^-2.5` and L3 only for genuine sum-confound charge work.

**EXP-id allocation note.** `grep -c "EXP-080"` returned 0 in both files before this block (071–079 used).

### EXP-080 — the EXP-076 stack run UNCHANGED on the V2 (sum-confound + r^-2.5) regime

- **Arm (single, transfer test):** the VALIDATED EXP-076 stack (full Council, all-Gemini, family feed + residual feedback + the exponent battery) run UNCHANGED on the V2 regime via `ASCENSION_BENCHMARK_SYSTEM=alien_family_v2`. We do NOT re-tune the instruments to V2 — that is the whole point. The SCORER is V2-regime-aware (the held-out oracle, not an instrument the agents use — making the oracle aware of the V2 truth is correct, not a leak).
- **Rigor:** 8 seeds (0–7), DISCRETE pinned single-episode runs (1200s), same pinned prompts + isolated context as EXP-075/076/079. Per-seed depth table + exact-binomial 95% CI on the "V2 depth ≥ 3" rate.
- **Primary metric:** HONEST consecutive depth from the V2-regime-aware `alien_depth.score_depth` (L1→L4), POST-HOC, deterministic (§22.6), on the V2 held-out IC/charge set.

Predictions (TWO-SIDED — transfer OR honest non-transfer, both publishable — either is the Burst B boundary finding PIVOT-001:73 wants):

- **P-EXP-080-1 (headline — the transfer test)** — running the UNCHANGED stack on V2, the Council reaches consecutive `score_depth ≥ 3` on **≥ 5 of 8** seeds (TRANSFER → the machinery is GENERAL). *[medium]* **The NON-TRANSFER branch is pre-registered + equally publishable:** if V2 depth ≥ 3 lands on **≤ 3 of 8**, the honest finding is the instruments are BESPOKE to V1, and the writeup DECOMPOSES which instrument failed (the L2 battery missing `r^-2.5` or the family enrichment being product-specific, read off the per-layer vectors). The intermediate 4/8 is reported descriptively. Neither branch retconned; the rate is the measured number.
- **P-EXP-080-2 (the free diagnostic win)** — `structural_identifiability` on `(s1+s2)·g(r)` returns `identifiable=False` with `confounded_groups` = `{s1,s2}` + a break-the-proportionality hint, WITHOUT new diagnostic logic (structural.py already handles the sum case) — a $0 deterministic generalization result reported regardless of the agent-loop outcome. *[medium — unit-test-level certainty; a failure is a STOP-and-fix defect].*
- **P-EXP-080-3 (decomposition — WHICH instrument transfers)** — report, across seeds, whether L2 (the `r^-2.5` correction) and L3 (the sum-confound charge) each fired, so the paper can state precisely which instrument is general vs bespoke. *[medium — descriptive, reported regardless].*
- **P-EXP-080-4/5/6 (integrity rails)** — depth is the V2-regime-aware `score_depth` (deterministic, §22.6, no LLM judge; the V2 sim/scorer add ZERO model calls to the scoring path); the V2 family/feedback/battery blocks pass the D-04 forbidden-token gates (no leak, no `s1+s2`/`2.5` literal); every observed V2 system is a data-only `ObservationBundle`, held-out V2 configs strictly OFF the observed family. *[hard — must hold].*

**Falsification condition.** If the TRANSFER bar (P-EXP-080-1) is wrong, it is reported HONESTLY as a NON-TRANSFER / boundary finding (not retconned, the V2 rubric not weakened, no seed re-rolled, the instruments NOT secretly re-tuned to V2 mid-experiment), with the per-layer decomposition naming the bespoke instrument. That null is the Burst B boundary finding and is publishable. If P-EXP-080-2 fails (structural.py's documented sum handling is broken) or any integrity prediction is violated, that is a STOP-and-fix defect, not a result.

**Cross-reference (the run):** EXP-080 is logged as **EXP-080** in `.planning/EXPERIMENTS.md` in a strictly-later commit. The V2 build (sim + scorer + diagnostic test, all V1-byte-identical + full-suite green) landed before this prereg. Launch: `EXP075_CONFIRM=1 EXP075_ARMS="council" EXP075_MIXED_PROVIDER=0 EXP075_SEEDS="0 1 2 3 4 5 6 7" EXP075_BENCHMARK_SYSTEM="alien_family_v2" EXP075_CEILING_USD=10 EXP_LABEL=exp080 bash scripts/run_exp075_measured.sh` (~$40; the override auto-selects the `--family-v2` scorer).

---

## Pre-registered block added 2026-06-09 (Session 039 — the V2 regime re-run with the REGIME-FAITHFUL instrument, **EXP-081**)

**Context (the L-063 bug, found by the Session 039 exponent-proximity analysis — see `.planning/research/session039_predictions.md`):** in EXP-080 the round-close experiment battery was regime-blind — `run_clean_probes` built its drop configs from the V1 TIER2 builder unconditionally, so the battery simulated the V1 universe and every feedback delivery told V2 agents their correction "best fits at about **3.4**" (V1's contaminated answer; V2 truth is **2.5**). Every feedback-receiving seed terminal-anchored at 3.4/3.0; the only seeds whose terminal hypothesis hit 2.5 exactly (seeds 3 and 7) are the only two with ZERO feedback deliveries. The fix (base_cfg threading; `held_out[0]` at the round-close call site) + 8 regression tests landed BEFORE this prereg. Known from the $0 regression work, stated for honesty: the regime-faithful V2 battery profiles to **≈2.1** (hidden-sum-term contamination, the V2 analog of V1's 3.4-vs-3.5 bias but 4× larger), which sits OUTSIDE the V2 L2 band [2.2, 2.8].

### EXP-081 — EXP-080 re-run, identical envelope, ONLY delta = the L-063 regime-faithful battery

Council-only, `alien_family_v2`, 8 pinned seeds (0–7), single-provider, 1200 s/run, $10/run ceiling, post-hoc out-of-band `--family-v2` scoring. Launch: `EXP081_CONFIRM=1 bash scripts/run_exp081_v2_regimefix.sh`.

- **P-EXP-081-1 (the bug signature is gone — hard):** ZERO `persisted experiment feedback` log lines report `fitted_exponent=3.4`; every delivery reports a value in **[2.0, 2.7]** (the V2 regime's own answer). A single 3.4 delivery = the fix regressed = STOP-and-fix defect, not a result. *[hard]*
- **P-EXP-081-2 (terminal structural accuracy improves):** the terminal-exponent error M1 (truth 2.5; metric defined in `scripts/analyze_exponent_proximity.py`, locked 9cc7466) has median across the 8 seeds **≤ 0.4**, and the count of seeds whose terminal correction exponent is V1-anchored at 3.4 is **0** (EXP-080: median 0.5, three seeds at 3.4). *[medium]*
- **P-EXP-081-3 (transfer does not regress):** consecutive `score_depth ≥ 3` on **≥ 5 of 8** seeds (EXP-080's measured rate with the sabotaged instrument). Fewer than 5/8 is reported honestly as "the corrected feedback did not preserve the transfer rate" — possible, since the corrected battery's 2.1 answer sits OUTSIDE the V2 L2 band and may steer terminal commitments off the band the scorer credits. *[medium — genuinely uncertain]*
- **P-EXP-081-4 (band interaction, descriptive, no threshold):** report the per-seed L2 band-pass count and where the feedback-delivered exponents land relative to the [2.2, 2.8] band. This measures the instrument-bias × gate-band interaction surfaced by the Session 039 attribution finding (neutralized battery recovers truth EXACTLY in both regimes; the bias is hidden-term contamination). Reported regardless of direction.

**Falsification condition.** P-EXP-081-2/3 failing is REPORTED, not retconned: no seed re-rolls, no battery re-tuning mid-experiment, no band widening. If P-EXP-081-1 fails the run is an infrastructure failure (L-062/L-063 class), renamed `*_CRASHED_*`-style and not counted as data.

---

## Pre-registered block added 2026-07-21 (Session 040 — the V2 re-run with BOTH agent-visible channels regime-faithful, **EXP-083**)

**Context.** The 2026-07-21 full audit (AUDIT-040 / L-064) found the L-063 fix corrected only ONE of the loop's two agent-visible instruments: in EXP-080 AND EXP-081 the residual-feedback channel's held-out ICs were V1-law configs, so that channel integrated V1 physics on both runs. Also fixed before this prereg: the battery plateau gate (V1 band hardcoded — on V2 the battery re-fired forever) and the post-hoc scorer's L2 Newton carve-out (effective band was [2.25, 2.8], stated [2.2, 2.8]). EXP-083 re-runs the exact EXP-080/081 envelope with the complete L-064 fix set (`b3d4674`, `6c9cf34`, DET-04) as the only deltas — the third point on the instrument-independence curve: poisoned (EXP-080) → battery-corrected (EXP-081) → both-channels-corrected (EXP-083).

### EXP-083 — EXP-081 re-run, identical envelope, ONLY delta = the L-064 fix set

- **P-EXP-083-1 (infra integrity — hard):** the run dir's `config.txt` records a git SHA containing the L-064 fixes; ZERO battery deliveries report `fitted_exponent=3.4`; every delivery reports a value in **[2.0, 2.7]**. A single 3.4 = regression = STOP-and-fix defect, renamed `*_CRASHED_*`-style, not data. *[hard]*
- **P-EXP-083-2 (the plateau gate behaves — the DET-02 behavioral signature):** in any seed where the Council persists a hypothesis whose second radial exponent lies in [2.2, 2.8] BEFORE a battery delivery would next fire, the battery goes QUIET (no further deliveries after that point in that seed's log). Pre-fix this was IMPOSSIBLE (the V1-band gate meant the battery could never see a V2 correction as "found"). Descriptive per-seed report; the prediction is that at least one seed exhibits the quiet-after-found pattern. *[medium]*
- **P-EXP-083-3 (instrument-independence of the load-bearing gate — the point of the curve):** the cross-config L3 sum-gate fires on **≥ 6 of 8** seeds (point prediction 7/8 — its value in BOTH prior runs, poisoned and half-corrected). Below 6/8 is reported honestly as "the residual channel was load-bearing for L3 after all" — that would be a real (surprising) finding, not a failure to hide. *[medium-hard]*
- **P-EXP-083-4 (depth + band, two-sided descriptive):** consecutive depth ≥3 rate and L2 rate reported with exact binomial CIs under BOTH scorers: primary = the corrected scorer (stated band [2.2, 2.8]); secondary = the legacy effective band ([2.25, 2.8]) for apples-to-apples comparison with EXP-080/081 as recorded. For the same purpose, EXP-080/081's STORED hypotheses are rescored under the corrected scorer ($0; secondary analysis — the primary records stand under their original scorer). No threshold; both directions publishable. *[descriptive]*
- **P-EXP-083-5 (instrument authority, the corrected-metric mechanism):** with the regime-aware proximity threshold (−2.05, per the DET-03 rescore), **≥ 6 of 8** seeds' terminal belief carries the battery's own delivered exponent (expected ≈2.1) — the both-directions instrument-authority pattern (EXP-081: 8/8 at exactly 2.1). Fewer means the corrected residual channel (now truly observing V2) weakened the battery's anchor — reported either way, with mean M1. *[medium]*

**Falsification condition.** P-EXP-083-2/3/5 failing is REPORTED, not retconned: no seed re-rolls, no re-tuning mid-experiment, no band widening, no threshold changes after data. If P-EXP-083-1 fails the run is an infrastructure failure (L-062/L-063/L-064 class) and is not counted as data.

**Cost envelope.** 8 seeds × $10/run worst-case ceiling; EXP-080 actual ~$17. Sprint spend at prereg time: $1,620 of the $2,000 ceiling (Argus halt at $1,800) — worst case stays under the halt line.

---

## Pre-registered block added 2026-07-21 (Session 040 cont. — EXP-083b, the clean-weather re-fire with the L-065 threaded dispatch)

**Context.** EXP-083 landed infra-confounded: a ~6h Vertex outage starved 4/8 seeds AND exposed L-065 (blocking sync SDK work stalls the event loop; the duration watchdog fired hours late, so the wall-clock envelope silently unbound). The L-065 fix (threaded provider dispatch, `0994721`) is live-validated (tier1 smoke PASS) and the provider weather has cleared (3.8s single-call latency at 09:44). EXP-083b re-fires the identical EXP-083 envelope. Deltas vs EXP-083: (1) the L-065 threaded dispatch (named, not silent), (2) nothing else.

### EXP-083b — EXP-083 re-fire, clean weather, watchdogs live

- **P-EXP-083b-0 (the L-065 fix's own falsifier — NEW, hard):** every seed's realized wall-clock is ≤ 1.5× the 1200s duration cap (i.e. ≤ 30 min including drain + scoring). A seed exceeding it means the loop-stall class survives the threaded dispatch — an L-065 regression, investigated before any science is read.
- **P-EXP-083b-1..5:** identical to P-EXP-083-1..5 (the locked `4ab7fe6` block applies verbatim — infra fingerprint hard-MET; plateau-gate quiet signature; L3 ≥6/8 with 7/8 point prediction; depth/L2 descriptive under the corrected scorer; anchoring reported with the delivery-scoped reading EXP-083 sharpened: seeds RECEIVING battery deliveries anchor on its value, no-delivery seeds report their own terminal).
- **Validity stratification pre-committed (the EXP-083 lesson):** any seed with realized calls < 100 is reported descriptively as starved (the N=15 starvation-floor rule), and the headline rates are stated BOTH raw and on the ≥100-call stratum, both directions publishable.

**Falsification condition.** As EXP-083's: failures REPORTED, never retconned; P-EXP-083b-0 or P1 failing makes the run an infrastructure datum, not science.

**Cost envelope.** 8 × $10 worst case; sprint spend at prereg $1,636 of $2,000 (Argus halt $1,800) — worst case stays under the line.

---

## Pre-registered block added 2026-08-25 (Session 041 — EXP-086 exploratory record + EXP-087 confirmatory cell for the design-class exhaustion certificate)

**Context.** A 2026-08-25 literature sweep found three uncited 2026 systems (Murphy's Model Discovery Agent, arXiv 2608.09696; LLM-ACES, 2606.25039; LLM-AutoSciLab, 2605.24043) that close an LLM-proposer + deterministic experiment-selection loop. None of them can say "no experiment in your design class can resolve this direction": each always picks a next experiment. The new module `src/ascension/diagnostics/exhaustion.py` certifies exactly that (a symbolic proof when a closed form exists; a stacked-sensitivity test over sampled designs for black-box/ODE models) and composes it with the diagnostic into a three-way plateau triage: `capability` / `design_within_class` / `change_class`.

**EXP-086 (exploratory, NOT pre-registered; reported as such).** Before this block was written, the following were observed on the unchanged engine and are pinned by tests (`tests/diagnostics/test_pk_bioavailability.py`, `tests/diagnostics/test_exhaustion.py`): (a) the oral one-compartment pharmacokinetic model (textbook non-identifiability: bioavailability F and volume V enter only as F/V) diagnoses rank 3/4 with the null direction on the (V, F) scaling axis, the exact symbolic checker reports the group {F, V}, the design ranker prefers an intravenous arm (rank 4/4) over denser or longer oral sampling (rank 3/4), and the ka<->ke flip-flop is shown to be a global limit the local check cannot see; (b) the F/V direction is certified class-level (symbolic and numeric routes) and triage returns `change_class`, while the IV arm returns `capability`; (c) on the Alien Universe product law at equal charges, the s1<->s2 swap is certified class-level for every design including a family (the honest report is the product), and the beta-vs-product scaling direction is certified for the single-configuration class but broken by a family; (d) the closed-form ansatz narrow-window collapse is NOT certified (a wider window in the same class resolves it), matching the ranker's earlier choice. These are exploratory: the predictions below are the held-out confirmatory test on a model none of that development touched.

### EXP-087 — Michaelis-Menten (kcat, E0, Km) with substrate observed: the black-box (numeric) certificate route on an ODE with no closed form

Model: dS/dt = -kcat*E0*S/(Km+S), S(0)=S0, substrate S observed on a sampling schedule. Operating point theta=(kcat, E0, Km)=(10, 0.1, 2.0) (Vmax = 1). Design class: any S0 in [0.2, 20], any schedule of 8 to 40 points in [0.05, 30]. Integration at rtol=atol=1e-10. Literature answer: only Vmax = kcat*E0 is identifiable from substrate (or product) time courses; kcat and E0 separately require an independent enzyme-concentration measurement.

- **P-EXP-087-1 (diagnosis):** on the standard design (S0=5, 24 points in [0.1, 20]) `practical_identifiability` returns rank 2/3, status `structurally_non_identifiable`. *[hard]*
- **P-EXP-087-2 (the named direction):** the null direction has |cos| > 0.99 with the scaling axis (kcat, -E0, 0) normalized at the operating point. *[hard]*
- **P-EXP-087-3 (class-level certificate + triage):** `certify_class_symmetry_numeric` over 32 random designs from the class returns `certified=True` (normalized stacked response <= 1e-7), and `triage_plateau` returns `change_class`. *[medium-hard; the point of the cell]*
- **P-EXP-087-4 (the class change vs a within-class design):** appending a direct E0 measurement (an independent assay observable) restores rank 3/3 and triage returns `capability`; a negative control that instead appends a second substrate-only run at a different S0 (a design inside the class) stays at rank 2/3. *[hard]*

**Falsification condition.** Any of P1-P4 failing is REPORTED as-is: no threshold changes, no re-sampling, no operating-point changes after data. P3 failing would mean the numeric certificate is not usable on black-box ODE models and the paper may claim only the symbolic route.

**Cost envelope.** $0 (deterministic scipy/numpy, no LLM, no DB).

---

## Statistical machinery (also pre-committed)

These choices are pre-committed alongside the predictions — they cannot be swapped for a different test family after seeing data.

| Choice | Value | Source |
|---|---|---|
| Test | Wilcoxon signed-rank (one-sided) | SCOPE §11, §22.1 |
| Multiple comparison correction | Holm-Bonferroni across the family of pairwise tests | SCOPE §11, §22.1 |
| Confidence interval | 95% bootstrap | SCOPE §11, §22.1 |
| Effect size | Cliff's delta | SCOPE §11, §22.1 |
| Seeds per Tier 1 cell | 8 | SCOPE §11 |
| Seeds per Tier 2 cell | 5 | SCOPE §11 |
| Seeds per brain-component / hallucination cell | 5 | SCOPE §11 |
| Per-run call budget (Tier 1) | 200 LLM calls | SCOPE §11 |
| External baseline | PySINDy on the same three Tier 1 systems | SCOPE §11, §22.1, REQUIREMENTS TIER1-03 |

---

## Amendment discipline

A pre-registered prediction file is append-only after the first measurement run. Edits proceed by appending an **Amendment** block, never by editing in place.

Amendments are allowed for two reasons only:

1. **Implementation of the prediction was wrong.** Example: "Prediction 1 said `coverage` but the implementation tracked `success_rate`. The amended prediction targets `coverage` per the original definition; runs against `success_rate` are reported separately as descriptive."
2. **The data revealed a precondition was unmet.** Example: "Prediction 4 assumed all 6 discovery layers were reachable; layer 5 (conservation laws) requires a simulator feature that is not implemented. Prediction 4 is amended to depth ≥ 2 over reachable layers only."

Amendments are NOT allowed for:

- Reframing a falsified prediction as supported (post-hoc rationalization).
- Changing the metric or comparator after seeing the data.
- Lowering an effect-size threshold to clear it.
- Splitting a single failed prediction into multiple sub-claims to find one that worked.

If the architecture is fundamentally rethought between locking and ICERM, the right move is to lock a *new* `predictions_v2.md` and report both — not edit this file.

### Amendment template

```
## Amendment N — YYYY-MM-DD

**Affected prediction(s):** [P1 / P2 / P3 / P4 / P5]
**Reason:** [implementation-bug / unmet-precondition]
**Change:** [exact diff to the prediction text]
**Discovered when:** [phase or run that surfaced the need]
**Discovered by:** [Surya / Claude]
```

(Amendments go below this line.)

## Amendment 1 — 2026-05-30 (EXP-076 depth metric → HONEST consecutive force-law depth; logged BEFORE the registered run, §22.1)

**Affected prediction(s):** P-EXP-076-1, P-EXP-076-2, P-EXP-076-3 (the depth metric)
**Reason:** implementation-bug (the scorer FIELD does not measure what the prediction MEANS)
**Change:** the headline metric "`score_depth.max_depth ≥ 3`" is amended to the HONEST CONSECUTIVE force-law depth — walk L1→L4 and stop at the first False; depth ≥ 3 means L1, L2, AND L3 all pass consecutively. `score_depth.max_depth` returns the highest-FIRING layer, NOT the consecutive run from L1. On the multi-charge family the L3 charge-credit plus the AUTOMATIC L6 parity credit (L-051: any `s1·s2` law is symbolically parity-invariant) make `max_depth ≥ 3` fire WITHOUT consecutive force-law discovery — verified on the Session-032 descriptive pilots, where a run with **L2=False** scored `max_depth=6`. Scoring the registered run against raw `max_depth` would manufacture a spurious "lift confirmed". The citeable depth is the consecutive L1→L4 value; L5/L6 are orthogonal (L-039) and reported SEPARATELY, never folded into the headline. `scripts/score_alien_run.py` now prints this consecutive value alongside `max_depth`.
**Discovered when:** the EXP-076 descriptive pilots (Session 032, runs `12bf216c` / `b9812937` / `d4c28858`) — the family made L3+L6 fire trivially, exposing `max_depth ≠ consecutive depth`.
**Discovered by:** Claude (surfaced + reported straight; L-057)

This is additive, dated, and logged BEFORE the registered 8-seed run (git proves the ordering). It does NOT reframe a falsified prediction, does NOT change the comparator (control vs treatment), and does NOT lower a threshold (≥3 stays ≥3) — it corrects the metric FIELD to match P-EXP-076-1's own stated meaning ("writes `s1*s2` with the per-config scaling doing material held-out RSS work" = genuine consecutive force-law discovery). The falsification condition is unchanged: a consecutive depth < 3 is reported honestly as a finding that the barrier is reasoning/generative, not observability.

**Provider-configuration note (§22.2 reproducibility, same amendment):** the registered run uses the PREVALENT mixed-provider roster (`ASCENSION_MIXED_PROVIDER=1`: plato+heraclitus on `gemini-2.5-pro`, aristotle/diogenes/parmenides on NVIDIA `meta/llama-3.3-70b-instruct`), applied IDENTICALLY to BOTH arms — so provider is held constant across treatment/control (NOT a confound between arms; it defines a logged, reproducible mixed-provider "Council"). Validated live (run `d4c28858`: 128 Google + 30 NVIDIA, $2.24, consecutive depth 3). Every call logs its provider. This relieves the Gemini rate-limit pressure that confounded EXP-074 (twice) over long batches. The all-Gemini config reached consecutive depth 4 (run `b9812937`); the mixed config trades ~1 depth for ~60% cost + quota safety — both reported.

## Pre-registered block added 2026-08-25 (Session 041, late — EXP-089: the 2x2 factorial that isolates observability from the instrument)

**Context.** EXP-076 changed three factors at once (multi-charge family, residual feedback, exponent battery) against a control with none. The paper (2026-08-25 draft, §7.2 and §8) reports the lift as a pilot for that reason. EXP-089 fills the two missing cells so the four arms form a 2x2: family × instrument (feedback + battery, on or off together, as in EXP-076).

**Design.** Same launcher, same 8 seeds (0 to 7), same 1200 s single-episode runs, same mixed-provider roster as EXP-076 (`ASCENSION_MIXED_PROVIDER=1`), scored post hoc by `score_alien_run.py` (no LLM). New arms: `family_only` (alien_family, feedback off, battery off, family L3 gate) and `instrument_only` (alien_universe, feedback on, battery on, single-config gate). Per-run ceiling $12; worst case $192; expected $30 to $60 (EXP-076 was $27 for 16 runs). The EXP-076 treatment and control cells are reused as the other two cells; they are NOT re-run.

**Predictions (locked before launch; falsifiers stated).**
- **P-EXP-089-1 (family delivers L3):** `family_only` passes the cross-configuration L3 charge gate on ≥5/8 seeds. Falsified if ≤3/8. This is the observability claim located where the paper puts it.
- **P-EXP-089-2 (instrument delivers L2):** `instrument_only` passes the L2 correction-exponent band on ≥5/8 seeds. Falsified if ≤3/8.
- **P-EXP-089-3 (no family, no L3):** `instrument_only` reaches consecutive depth ≥3 on ≤2/8 seeds (a single configuration cannot pass the cross-configuration gate; any credit comes from the looser single-config gate, as EXP-076 seed 2 showed). Falsified if ≥4/8.
- **P-EXP-089-4 (the honest expectation on depth):** `family_only` reaches consecutive depth ≥3 on ≤4/8 seeds, because consecutive depth requires L2 first and passive family data rarely surfaces the correction term (EXP-072). Interpretation committed now: if P-1 holds and P-4 holds, the family delivers L3 and the instrument delivers L2, exactly the split reading in the paper; if `family_only` reaches depth ≥3 on ≥5/8, the family alone suffices and the paper's confound caveat was too cautious (a stronger result, reported as such).
- **Pre-committed contrast:** across the four cells (32 runs), Fisher's exact test on L3 credit, family vs single configuration (two-sided), and the same on L2 credit, instrument vs none. Exact-binomial 95% CIs per cell. n=8 per cell; interpretation is descriptive beyond those two contrasts.
- **Validity rule:** a seed counts only if its run terminated on the duration cap or the natural end; provider-starved runs (fewer than 30 calls) are flagged and reported separately, per the EXP-083 stratification rule, and never silently dropped.

**Ordering.** This block is committed strictly before the launch; the outcome lands in a strictly later commit (SCOPE §22.1).

## Pre-registered block added 2026-09-08 (Session 042 — EXP-261: the IDEA diagnostic swept over ODEBench under the LLM-ACES protocol)

**Context.** Dr. Braga-Neto (2026-09-04, 2026-09-08) asked how the paper's evidence volume compares with LLM-ACES (122 ODE systems) and MDA. The answer is not more live Council seeds. The diagnostic is deterministic and free, so it is swept over the benchmark LLM-ACES itself uses, under the exact data LLM-ACES hands its own proposer. Committed BEFORE any full run; a code smoke on four systems (ids 2, 10, 58, 63; all identifiable) checked the pipeline only.

**Scope, stated before running.**
- Task set: the 63 ODEBench systems as shipped inside the pinned LLM-ACES release (`scripts/strogatz_ode.py`, commit `60d4df7cd23ad3e994138bc3c2a4f34b152c0cbe`, cloned to `.planning/research/external/LLM-ACES`). ODEBase (the other 59 of the 122) is NOT in the release: `generate_odebase.py` imports `scibench.data.equation_odes_odebase`, which the repository does not contain. So the sweep covers 63 of 122, and says so.
- Protocol: LLM-ACES round-0 data = one trajectory from IC-0 on t in [0, 1], 100 uniform points, every state AND its analytic derivative observed, noise-free (`generate_ode.py`). Acquisition class = one added trajectory from an initial condition drawn uniformly inside the per-system box in `llm-aces/ic_bounds.json`, 10 candidates (`--n_virtual 10`), plus the ODEBench held-out IC-1. Their OOD window (1, 10] is an evaluation split, included as a labelled 13th candidate only.
- Verdicts: exact symbolic structural check on the vector field; local sensitivity rank and condition number at the published constants (threshold 1e8, Section 3.1 of the paper); symbolic class-level certificate on each null direction (is the vector field itself invariant along it, for every state?); ranker over the 12 or 13 single added designs, each scored on round-0 data plus that design.
- Per-system LLM-ACES outcomes are NOT published (one run per system, aggregates only, Tables 2 and 3), so the prediction "non-identifiable tasks are where they plateau" is untestable from public data and is replaced by P1 to P4 below. Also noted before running: the LLM-ACES symbolic-accuracy metric strips constants, so it is blind by construction to parameter non-identifiability; MDA's interventional-forecast metric is not.
- Numerics: sensitivities by central differences at rtol = atol = 1e-10 (LLM-ACES generates data at 1e-5 / 1e-7; tolerance is an integrator setting, not the observation protocol).

**Predictions (locked; falsifiers stated).**
- **P-EXP-261-1 (most of the benchmark is identifiable at round 0):** at least 48 of 63 systems return `identifiable` under the round-0 protocol. Falsified if 40 or fewer. Rationale: full state plus true derivative on 100 points is a rich design; non-identifiability should come from parameter redundancy in the RHS or from trajectories that settle before t = 1.
- **P-EXP-261-2 (certified exhaustion exists on their benchmark):** at least one system is structurally non-identifiable by parameter redundancy in its vector field, and EVERY such null direction receives a symbolic class-level certificate: no initial condition in the LLM-ACES acquisition box, and no time window, can separate the parameters. Named in advance by inspection of the equation list: id 1, the RC circuit, `(c_0 - x_0/c_1)/c_2`, three constants entering only as `c_0/c_2` and `1/(c_1 c_2)`. Falsified if any structurally confounded direction is resolved by one of the candidate designs, or if no system is structurally non-identifiable.
- **P-EXP-261-3 (practical cases are resolvable inside the class):** among systems that are full-rank but practically non-identifiable at round 0 (condition number above 1e8), adding the ranker's best single design restores `identifiable` in at least half. Falsified if fewer than a quarter. If there are fewer than 4 such systems, P3 is reported descriptively and not scored.
- **P-EXP-261-4 (the LLM-ACES acquisition criterion has no signal on certified cases):** for every certified null direction, two models displaced by plus and minus 1e-3 along it produce rollouts from every candidate IC that differ by at most 1e-8 in relative sup norm. Falsified if any certified direction exceeds 1e-6. This is the analytic reason a disagreement-based acquisition cannot resolve a class-level symmetry, verified numerically on their own candidate class.

**Pre-committed reporting.** Counts per verdict with exact binomial 95% CIs; the certified systems listed by name with the invariant combinations; the practical systems listed with the best added design and its condition number before and after. No hypothesis test beyond the binomial CIs; n = 63 is the whole benchmark, not a sample.

**Ordering.** This block is committed strictly before the full run; the outcome lands in EXPERIMENTS.md as EXP-261 in a strictly later commit (SCOPE §22.1). EXP-261 is taken from the "post-hoc follow-ups" block (EXP-261 to EXP-270) of the EXPERIMENTS.md id table.

## Amendment 2 — 2026-09-08 (EXP-089 seeds 4 to 7: relaunched after an infrastructure abort, on the all-Gemini roster because the registered NVIDIA model reached end of life; logged BEFORE the relaunch, §22.1)

**What happened.** The 2026-08-25 batch (`.planning/research/exp076_20260825_171201/`) completed seeds 0 to 3 of both EXP-089 arms on the registered mixed roster. Seeds 4 to 7 of both arms never executed: the local Postgres container was down when the batch reached them (every log ends in a psycopg pool timeout with zero model calls), and `instrument_only` seed 3 completed its 1200 s window with 89 calls but could not write its halt reason or be scored until Postgres was restored (rescored 2026-09-08: single-configuration gate, per-layer (T,T,T,F,F,T), consecutive depth 3). None of this is a property of the arms.

**Why the roster changes.** A relaunch on 2026-09-08 with the registered mixed roster failed on the first iteration: NVIDIA Build returns HTTP 410 for `meta/llama-3.3-70b-instruct` ("reached its end of life on 2026-08-26"). The only NVIDIA chat model this account can still call is `nvidia/nemotron-3-super-120b-a12b`, a reasoning model whose content stream carries reasoning text, the failure mode that confounded EXP-074's fallback. That aborted relaunch (`exp076_20260908_121552`, one run, 0 valid calls) is discarded as infrastructure.

**Decision, before the relaunch.** Seeds 4 to 7 of BOTH arms run on the all-Gemini roster (MODEL_PLATO=gemini-2.5-pro MODEL_HERACLITUS=gemini-2.5-pro MODEL_ARISTOTLE=gemini-2.5-pro MODEL_DIOGENES=gemini-2.5-pro MODEL_PARMENIDES=gemini-2.5-pro ), the roster of EXP-075 and EXP-077. Within EXP-089 the roster is held identical across the two arms on every seed, so the pre-committed contrasts (family versus single configuration on L3 credit; instrument versus none on L2 credit) are unconfounded by provider within each seed. Across cells, seeds 0 to 3 are a mixed-roster stratum and seeds 4 to 7 an all-Gemini stratum; results are reported stratified and pooled, and the EXP-076 cells they are compared with are mixed-roster on all 8 seeds. The all-Gemini roster is the stronger one (consecutive depth 4 on run b9812937 versus 3 for the mixed run d4c28858), which biases `family_only` toward MORE depth, against P-EXP-089-4's ceiling and in favour of the paper's stated "stronger result" reading. Predictions P-EXP-089-1 to P-EXP-089-4 and their falsifiers are unchanged. Validity rule unchanged: a seed counts only on a duration-cap or natural termination with at least 30 calls.

## Amendment 3 — 2026-09-08 (EXP-261 numerics: exact forward sensitivities replace central differences; the structural checker gains an exact multi-parameter test; logged BEFORE the re-run, §22.1)

**What the first full run showed.** The first full sweep (2026-09-08, central differences at rtol = atol = 1e-10, as the block above specified) returned `identifiable` for id 1, the RC circuit, at rank 3 of 3 with condition number 2.35e7, just under the 1e8 practical threshold. That is the case P-EXP-261-2 named in advance as structurally non-identifiable. It is non-identifiable: the three constants enter only as `c_0/c_2` and `1/(c_1 c_2)`. Two things in the instrument, not in the benchmark, produced the miss. (a) A central difference has a round-off floor of about tolerance / step, here 1e-10 / 1e-6 = 1e-4 relative per column, which turns a true zero singular value into one of order 1e-8 relative, above the 1e-9 rank floor and inside the "well-conditioned" band. (b) The exact symbolic checker tested parameters pairwise, so a dependence that needs all three derivatives (the documented EXP-082 blind spot, Section 6.1 of the paper) was invisible to it; on this benchmark that blind spot is not a curiosity but the first system in the list. The first run also crashed on id 11 (no constants) and on a ranker field name; those are code errors, recorded in the log `.planning/research/exp261_sweep_20260908.log`, which is kept.

**What changes, before the re-run.**
1. Sensitivities are exact: the forward sensitivity equations x' = f(x, c), S' = J_x S + J_c, S(0) = 0, integrated at 1e-12, for every design. The central-difference verdict is still computed and stored per system as a cross-check field, and the summary lists every system where the two disagree.
2. The structural checker adds an exact multi-parameter dependence test (constant-coefficient linear dependence among the derivative functions, checked at 50 digits at generic rational observable and parameter points, required to hold at two independent parameter draws). The pairwise test is unchanged and the new test is unioned with it. Regression tests: `tests/diagnostics/test_structural_multiparam.py`; the EXP-082 boundary test now asserts the confound is found rather than missed.
3. Systems with no constants (id 11) are reported as their own category and excluded from the denominators.

**What does not change.** Task set, protocol, acquisition class, thresholds, predictions P-EXP-261-1 to P-EXP-261-4 and their falsifiers, and the reporting plan. The first run's per-system output is retained and will be reported alongside the exact run, so the reader can see what the finite-difference path would have said.


## Pre-registered block added 2026-09-28 (Session 047 — EXP-262: head-to-head of experiment-selection criteria on identical candidate sets)

**Context.** The paper's ranker orders candidates by sensitivity rank, then condition number. Reviewers will ask why not A-optimal (CARTOGRAPH, arXiv 2606.07576), Bayesian value of information (MDA, arXiv 2608.09696), or candidate disagreement (LLM-ACES, arXiv 2606.25039). This experiment runs all of them on the same candidates. It is committed BEFORE any criterion is computed on any task. $0, no LLM.

**Task suite (fixed now).**
- **T1**: the 62 ODEBench systems with free constants, under the EXP-261 LLM-ACES round-0 protocol (pinned LLM-ACES `60d4df7`). One trajectory from IC-0 on t in [0, 1], 100 points, full state plus analytic derivative.
- **T2**: the same 62 systems under a SPARSE protocol, defined now. Only the first state coordinate is observed, with no derivative, at 20 uniform points on [0, 1] from IC-0.
- **Candidate set** (T1 and T2): the 10 box initial conditions of EXP-261 plus the ODEBench held-out IC-1. Each candidate is one added trajectory under the same observation protocol as its task.
- **T3 (descriptive only)**: oral PK (candidates: oral dense, oral 72 h, oral plus IV); Michaelis-Menten (10 S0 values plus an E0 assay); the Alien Universe amplitude model (single product vs two products); the integer-sequence narrow window vs the geometric spread.

**Operating point and scaling.** Local design at the published constants θ*, as in EXP-261. Every criterion uses the same scaled sensitivity S~ = S·diag(max(|θ*|, 1e-3)), i.e. sensitivities with respect to log-parameters. Sensitivities are exact forward sensitivities (EXP-261 Amendment 3). Noise σ = 1% of the RMS of each observed channel over round 0.

**Criteria (all see round-0 data, the candidate list and θ*).**
- **RANK**: the paper's ranker (rank of [S0; Sc] first, condition number as tiebreak).
- **DOPT**: maximize log det(F0 + Fc + εI), where F = S~ᵀ S~/σ² and ε = 1e-10·tr(F0).
- **AOPT**: minimize tr((F0 + Fc + εI)⁻¹). This is the A-optimal rule; with the ridge, unresolved directions dominate, as in CARTOGRAPH-A.
- **VOI**: linear-Gaussian expected information gain, 0.5·log det(I + Sc Σ Scᵀ/σ²), with Σ = (F0 + I/τ²)⁻¹ and prior sd τ = 1 in log-parameters. This is MDA's mutual-information criterion in the linear-Gaussian limit.
- **DISAGREE**: draw K = 8 parameter vectors from N(log θ*, Σ). Roll each out on the candidate with the true simulator. Score = mean pairwise L2 distance of predictions / RMS of prediction (the LLM-ACES acquisition, applied to parameter uncertainty). Fixed seed 262.
- **RANDOM**: the expectation over a uniform pick, computed exactly as the mean over candidates.
- **HYBRID**: rank first, then DOPT as the tiebreak (the proposed replacement for RANK).

**Outcomes, computed for EVERY candidate so any criterion's pick can be scored.**
- **M1 (rank restoration)**: rank([S0; Sc]) at the 1e-9 relative floor, compared with the maximum over candidates.
- **M2 (empirical recovery)**:
  - Data: round 0 plus the candidate, with σ-Gaussian noise, 5 noise seeds (262-266).
  - Fit: log-parameters by Levenberg-Marquardt (scipy least_squares, method lm) from log θ* + N(0, 0.1²), with an identical start per seed across criteria.
  - Score: RMS log-parameter error over the parameters the candidate's design identifies (full-rank cases: all parameters); for rank-deficient candidates, over the orthogonal complement of the null space, after projecting the error onto it.
- **M3 (stop decision)**: on tasks the certificate marks change_class, experiments spent by each criterion. RANK plus the certificate spends 0; every pure selection criterion spends 1 per round.

**Predictions (locked).**
- **P-EXP-262-1 (rank restoration is easy for any Fisher criterion)**:
  - Setting: tasks where round 0 is rank-deficient and at least one candidate restores the maximum rank.
  - Prediction: RANK and HYBRID pick a max-rank candidate on 100% (by construction). DOPT, AOPT and VOI each pick one on ≥ 90% of such tasks. DISAGREE on ≥ 75%.
  - Falsified if any of DOPT, AOPT or VOI is below 75%, or DISAGREE is below 50%. If fewer than 5 such tasks exist across T1+T2, this is reported descriptively.
- **P-EXP-262-2 (our tiebreak is not the best precision criterion)**:
  - Setting: tasks where two or more candidates attain the maximum rank.
  - Prediction: median M2 error of RANK's pick ≥ that of DOPT's pick. One-sided Wilcoxon signed-rank that DOPT < RANK, paired by task, at p < 0.05.
  - Falsified (in RANK's favour) if RANK's median error is lower than DOPT's by more than 5%.
- **P-EXP-262-3 (the hybrid keeps both properties)**: HYBRID's median M2 error is within 5% of DOPT's, and its rank restoration is 100%. Falsified if its median error exceeds DOPT's by more than 10%.
- **P-EXP-262-4 (no selection criterion can see a certified direction)**:
  - Setting: every task with a certified change_class verdict (at minimum the RC circuit, T1 id 1).
  - Prediction: every criterion's pick leaves rank unchanged. DISAGREE's score contribution along the certified direction is ≤ 1e-8 relative. The certificate saves 1 experiment per round per task.
  - Falsified if any pick raises the rank on a certified task (that would falsify the certificate).
- **P-EXP-262-5 (disagreement is the weakest precision criterion)**: median M2 error of DISAGREE ≥ that of each of DOPT, AOPT and VOI. Falsified if DISAGREE is best on median.

**Pre-committed reporting.**
- Per protocol (T1, T2) and pooled: counts, rank-restoration rates with exact binomial CIs, median M2 with bootstrap CIs.
- Pairwise Wilcoxon (Holm-corrected across the 6 pairs with RANK), and the top-pick agreement matrix.
- T3 is reported case by case.
- The number of rank-deficient T2 tasks is NOT predicted (unknown; reported as found).

**Ordering.** This block is committed and pushed before any criterion is evaluated. Outcomes land as EXP-262 in a strictly later commit.

## Pre-registered block added 2026-09-28 (Session 047 — EXP-263: identifiability audit of the DiscoverPhysics benchmark against its published per-world outcomes)

**Disclosure first.** This is a RETROSPECTIVE test.
- **Outcomes already seen**: the per-world outcomes (13 models × 5 seeds × 11 public worlds: explanation score and geometric position error; leaderboard repo `8e9c858`) were read by the authors in Session 045 and again today while scoping this block. Also seen: the maximum explanation score per world across models; the private-world rows of two models (claude-fable-5, partially claude-opus-4-8).
- **What is blind**: every identifiability verdict below. No sensitivity, rank, certificate or equivalence computation has been run on any DiscoverPhysics world.
- **What this experiment tests**: whether a mechanical procedure, fixed here, reproduces and explains the outcome pattern. It is not a blind forecast of the outcomes.
- **Private worlds**: definitions are gated (HuggingFace, manual approval) and are out of scope, except for any private world whose definition is in the public code, which is reported separately and labelled contaminated.
- **Pinned**: DiscoverPhysics `33b7fa9df96de9c35744efd181ca7e5a8dd60ad5`, engine nbody, the benchmark's own JAX simulator (no reimplementation of the physics), noise_frac 5%.

**Procedure (mechanical, fixed now).**
1. **Graded claims.** For each public world, list every claim in the TOP band (score 10) of its explanation rubric in `scienceagent/worlds.py`, verbatim. Each claim is typed as:
   - a **PARAMETER** claim: a numeric constant of the true law with a stated tolerance, e.g. λ in [1, 4], R_c in [0.2, 1.5], H within ×2, T within ×2, coupling ratios within ×2, α in a stated range; or
   - a **MECHANISM** claim: a distinction between the true law and a named alternative that the rubric scores lower. Examples: fractional Laplacian vs "ordinary power law"; Kaluza-Klein crossover vs plain 1/r; time-modulated vs constant coupling; screened vs plain Laplacian; linear-in-r push vs constant outward push; hidden count vs coupling.
2. **Design class** = the agent interface documented in `PhysicsSchool/prompts` (2-particle: p1, p2 in [0.1, 10], pos2 in [-10, 10]², velocity2 in [-5, 5]², ≤ 10 times, duration in [5, 10]; probes: 5 probes in [-15, 15]², velocities in [-2, 2]², ≤ 10 times, duration ≥ 10; the oscillator's optional start_time). Two designs are scored:
   - (i) **DEFAULT**: the world's own `experiment_format` example experiment (or the 2-particle test case 1 where none is given);
   - (ii) **BEST**: the ranker's pick among 64 interface-legal designs sampled uniformly with seed 263, plus the DEFAULT.
3. **PARAMETER claims.**
   - Local sensitivity of the observed positions (all agent-visible particles, all measurement times) with respect to the world's constants, by central differences through the benchmark simulator.
   - Relative Cramér-Rao standard error of the graded quantity at 5% noise for ONE experiment, scaled by 1/sqrt(16) for the 16-round budget.
   - A claim is IDENTIFIABLE under a design if that CRLB is below half the rubric tolerance (e.g. relative SE < 0.25 for a factor-of-2 tolerance, where the ×2 tolerance means |log| < log 2 and relative SE is measured in log units).
4. **MECHANISM claims.**
   - Fit the named alternative family by least squares to the true world's noiseless predictions over the 64 sampled designs plus DEFAULT.
   - **EQUIVALENT (certified)** if a symbolic reparameterization maps one family onto the other exactly (checked with sympy on the kernel, e.g. riesz_2d_force ∝ r^-(3-2α) vs A·r^-n).
   - **EQUIVALENT (numeric)** if the best-fit alternative's normalized residual stays below 1% of the 5% noise level on every sampled design.
   - **DISTINGUISHABLE-BY-DESIGN** if the residual exceeds the 5% noise level on some sampled design but not on DEFAULT.
   - **DISTINGUISHABLE** if it exceeds it on DEFAULT.
5. **World verdict.**
   - **NONID**: some top-band claim is EQUIVALENT (certified or numeric), or a PARAMETER claim is non-identifiable even at BEST.
   - **RESOLVABLE**: not NONID, and some claim is identifiable or distinguishable only at BEST (not at DEFAULT).
   - **ID**: otherwise.

**Predicted verdicts (from the physics, before computing).**
- fractional: NONID. At α = 0.5 the benchmark's Riesz force is ∝ 1/r², exactly an ordinary inverse-square power law, so "fractional operator" vs "power law" is certified equivalent for every design.
- circle: NONID. At α = 0.75 the force is ∝ r^-1.5, again a pure power law; "fractional operator" vs "power law" is certified equivalent.
- extra_dimensions: RESOLVABLE. R_c and the crossover are invisible at DEFAULT separations (r of 3 to 5, image corrections exponentially small) and identifiable at r ≲ 1.
- dark_matter: RESOLVABLE or NONID. Hidden count vs per-particle coupling at fixed total source (50) is not distinguishable at DEFAULT (probes at radius 5 to 7, outside a cluster of radius about 1). Whether probes routed through the centre separate it is the one verdict we do not predict.
- gravity, yukawa, hubble, ether, oscillator, three_species, coulomb: ID.

**Predictions (locked).**
- **P-EXP-263-1 (verdicts)**: the computed verdicts match the predictions above on at least 9 of the 10 worlds with a stated prediction (dark_matter excluded). Falsified if fewer than 8 match.
- **P-EXP-263-2 (ceilings)**:
  - Prediction: every NONID world has a lower maximum explanation score across the 13 models than every ID world (perfect separation by the maximum), and RESOLVABLE worlds lie between or overlap. Statistic: Mann-Whitney U on the per-world maxima, NONID ∪ RESOLVABLE vs ID.
  - Disclosure: the maxima are already known to the authors (fractional 0.48, circle 0.70, extra_dimensions 0.54, dark_matter 0.56; the others ≥ 0.90), so this prediction is conditional on the verdicts coming out as predicted and is NOT independent evidence. It is reported as a consistency check.
- **P-EXP-263-3 (Proposition 3 signature; not yet computed by the authors)**:
  - A cell (model × world) is TRAJECTORY-GOOD if its geometric position error ≤ 0.1, and EXPLANATION-FAIL if its explanation score < 0.9.
  - Prediction: in worlds whose NONID verdict comes from a CERTIFIED equivalence, the fraction of trajectory-good cells that are explanation-fail is higher than in ID worlds. Fisher exact one-sided test, p < 0.05.
  - Falsified if that fraction is not higher.
- **P-EXP-263-4 (the ranker finds the resolving design)**: in every RESOLVABLE world, the ranker's BEST design identifies the graded quantity. For extra_dimensions it places the probe at r ≤ 1.5 at some measurement time. Falsified if BEST fails to identify it.

**Ordering.** This block is committed and pushed before any verdict is computed. Outcomes land as EXP-263 in a strictly later commit. Per-cell outcome data are copied from the pinned leaderboard commit into `.planning/research/exp263/` before analysis.


## Amendment 4 — 2026-09-30 (EXP-263: the run is stopped by the PI's decision; three three_species joint fits are NOT SCORED)

**What happened.**
- The EXP-263 computation ran for about 49 hours of wall clock (2026-09-28 20:00 to 2026-09-30 20:55 UTC) across five container restarts. The container is reclaimed whenever the session goes idle, which kills even detached processes.
- Ten of the 11 worlds completed, with every registered fit.
- For three_species, every PARAMETER claim and every registered joint fit completed EXCEPT three: TS-2 partition alternative `b5a9b6997cf4` (start 0 of its joint fit only), TS-5 `probes_B` (start 0 only) and TS-5 `probes_C` (no start). Each start of these 35-particle × 65-design fits takes 3 to 4 hours, and restarts repeatedly destroyed starts in flight.

**Decision (Surya, 2026-09-30: "Ok wrap it up").** The run is stopped. The three incomplete joint fits are reported NOT SCORED (compute and infrastructure), and the three_species world verdict is reported as not final.

**Disclosure.** Before this decision the authors had already seen the provisional 10-world aggregate and the single-design (solo) residuals of the three unfinished alternatives. The single-design residuals at DEFAULT are 0.73 (b5a9), 2.36 (probes_B) and 2.37 (probes_C), in noise units. They are reported as the solo variant, not as the registered joint fits.

**What does not change.** Methods, thresholds, predictions and their falsifiers. No prediction is rescored because of this stop.
