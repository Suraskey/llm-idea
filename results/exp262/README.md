# EXP-262 — head-to-head of experiment-selection criteria on identical candidate sets

Pre-registration: `predictions.md`, block "Pre-registered block added 2026-09-28 (Session 047 — EXP-262 ...)", commit 8f746eb. Script: `scripts/exp262_head_to_head.py`. $0, no LLM, no network. LLM-ACES pinned at 60d4df7.

Outputs in this directory: `per_task_candidate.tsv` (one row per protocol x system x candidate), `picks.tsv` (one row per protocol x task x criterion), `summary.json` (every number below), `units.jsonl` (raw per-task results), `run.log`.

## Predictions

| Prediction | Status | Measured |
|---|---|---|
| P-EXP-262-1 rank restoration | **NOT SCORED** | n = 1 tasks (T1+T2 pooled). RANK 1.000, HYBRID 1.000, DOPT 1.000, AOPT 1.000, VOI 1.000, DISAGREE 1.000, RANDOM 1.000 |
| P-EXP-262-2 RANK tiebreak vs DOPT | **NOT MET** | n = 122 tasks with >= 2 max-rank candidates; median M2 RANK 0.00186874 vs DOPT 0.00195508 (ratio 0.9558); Wilcoxon DOPT < RANK: 91 non-zero pairs, p_raw = 0.08059, p_Holm = 0.403 |
| P-EXP-262-3 HYBRID keeps both | **HELD** | all tasks pooled (n = 124): median M2 HYBRID 0.00195508 vs DOPT 0.00195508 (ratio 1.0000); HYBRID rank restoration on P1 setting 1/1 = 100.0% [2.5, 100.0] |
| P-EXP-262-4 certified directions | **HELD** | certified tasks: T1 id 1 (RC-circuit), T2 id 1 (RC-circuit); picks raising rank: 0; max DISAGREE contribution along the certified direction 6.90e-11; 1 per round per task on 2 tasks (RANK + certificate spends 0; every pure selection criterion spends 1 per round) |
| P-EXP-262-5 DISAGREE weakest | **HELD** | all tasks pooled: median M2 DOPT 0.00195508, AOPT 0.00160437, VOI 0.00195508, DISAGREE 0.00281426; lowest of all 7: AOPT |

Status vocabulary: HELD = the prediction as stated is met; FAILED = the registered falsifier fired; NOT MET = not met but the falsifier did not fire; NOT SCORED = the registered descriptive fallback applies.

## Counts

- Units run: 128 (0 errored). Scored tasks: T1 62, T2 62, T3 4.
- Round-0 status: T1 {'identifiable': 61, 'structurally_non_identifiable': 1, 'practically_non_identifiable': 0}; T2 {'identifiable': 59, 'structurally_non_identifiable': 2, 'practically_non_identifiable': 1}.
- Round-0 rank-deficient: T1 1, T2 2 (the T2 number was not predicted).
- Triage (certificate): T1 {'capability': 61, 'design_within_class': 0, 'change_class': 1}; T2 {'capability': 59, 'design_within_class': 2, 'change_class': 1}.
- P1 setting (rank-deficient round 0, some candidate raises rank): T1 0 (ids []), T2 1 (ids [53]); of these, full rank reachable: T1 0, T2 1.
- Rank-deficient with no candidate raising rank: T1 ids [1], T2 ids [1].
- P2 setting (>= 2 candidates at max rank): T1 61, T2 61, pooled 122.
- Single-candidate tasks: ['T1 id 63', 'T2 id 63'].
- Dropped candidates, T1 id 29: box IC 3: integration failed: A termination event occurred. (64/100 pts); box IC 4: integration failed: A termination event occurred. (29/100 pts)
- Dropped candidates, T2 id 29: box IC 3: integration failed: A termination event occurred. (13/20 pts); box IC 4: integration failed: A termination event occurred. (6/20 pts)
- Note, T1 id 63: ic_bounds.json malformed ((3, 2) vs dim 4); box class empty (as EXP-261)
- Note, T2 id 63: ic_bounds.json malformed ((3, 2) vs dim 4); box class empty (as EXP-261)
- Note, T3 id SEQ: round-0 RMS of log f is 0 at theta = (1, 0, 1); sigma = 0.01 absolute on log f
- Note, T3 id MM: E0-assay channel absent at round 0: sigma = 1% of E0*

### Errored units (excluded from every denominator)

- none

### M2 fit failures (NaN, counted; +inf in rankings)

- T1: 6 of 3350 fits failed; candidates with +inf mean M2: 5; reasons {'lm status 0': 6}; DISAGREE failed rollouts 1, NaN DISAGREE scores 0.
- T2: 32 of 3350 fits failed; candidates with +inf mean M2: 25; reasons {'lm status 0': 32}; DISAGREE failed rollouts 9, NaN DISAGREE scores 0.
- T3: 0 of 90 fits failed; candidates with +inf mean M2: 0; reasons {}; DISAGREE failed rollouts 0, NaN DISAGREE scores 0.

## Rank restoration (M1): pick attains the maximum rank over candidates

Exact binomial (Clopper-Pearson) 95% CIs; RANDOM is the exact expectation of a uniform pick.

**P1_T1**

| Criterion | Rate |
|---|---|
| RANK | NA (n = 0) |
| HYBRID | NA (n = 0) |
| DOPT | NA (n = 0) |
| AOPT | NA (n = 0) |
| VOI | NA (n = 0) |
| DISAGREE | NA (n = 0) |
| RANDOM | NA (expected, n = 0) |

**P1_T2**

| Criterion | Rate |
|---|---|
| RANK | 1/1 = 100.0% [2.5, 100.0] |
| HYBRID | 1/1 = 100.0% [2.5, 100.0] |
| DOPT | 1/1 = 100.0% [2.5, 100.0] |
| AOPT | 1/1 = 100.0% [2.5, 100.0] |
| VOI | 1/1 = 100.0% [2.5, 100.0] |
| DISAGREE | 1/1 = 100.0% [2.5, 100.0] |
| RANDOM | 1.000 (expected, n = 1) |

**P1_pooled**

| Criterion | Rate |
|---|---|
| RANK | 1/1 = 100.0% [2.5, 100.0] |
| HYBRID | 1/1 = 100.0% [2.5, 100.0] |
| DOPT | 1/1 = 100.0% [2.5, 100.0] |
| AOPT | 1/1 = 100.0% [2.5, 100.0] |
| VOI | 1/1 = 100.0% [2.5, 100.0] |
| DISAGREE | 1/1 = 100.0% [2.5, 100.0] |
| RANDOM | 1.000 (expected, n = 1) |

**ALL_pooled**

| Criterion | Rate |
|---|---|
| RANK | 124/124 = 100.0% [97.1, 100.0] |
| HYBRID | 124/124 = 100.0% [97.1, 100.0] |
| DOPT | 124/124 = 100.0% [97.1, 100.0] |
| AOPT | 124/124 = 100.0% [97.1, 100.0] |
| VOI | 124/124 = 100.0% [97.1, 100.0] |
| DISAGREE | 124/124 = 100.0% [97.1, 100.0] |
| RANDOM | 1.000 (expected, n = 124) |

## Median M2 (RMS log-parameter error of the pick; mean over 5 noise seeds)

Percentile bootstrap 95% CI, 10,000 resamples of tasks, seed 262. `n_inf` = tasks where the pick's M2 is +inf (some fit failed).

| Criterion | ALL_T1 (n=62) | ALL_T2 (n=62) | ALL_pooled (n=124) | P2_T1 (n=61) | P2_T2 (n=61) | P2_pooled (n=122) |
|---|---|---|---|---|---|---|
| RANK | 0.0007918 [0.0005573, 0.001159] | 0.009048 [0.005128, 0.01563] (inf: 2) | 0.001869 [0.001332, 0.004506] (inf: 2) | 0.0008032 [0.0006024, 0.001159] | 0.009047 [0.005128, 0.01563] (inf: 2) | 0.001869 [0.001332, 0.004506] (inf: 2) |
| HYBRID | 0.0005867 [0.0004472, 0.000871] | 0.009063 [0.003547, 0.02117] (inf: 2) | 0.001955 [0.001052, 0.003566] (inf: 2) | 0.0005892 [0.0004352, 0.000879] | 0.009047 [0.003547, 0.0191] (inf: 2) | 0.001955 [0.001052, 0.003638] (inf: 2) |
| DOPT | 0.0005867 [0.0004472, 0.000871] | 0.009063 [0.003547, 0.02117] (inf: 2) | 0.001955 [0.001052, 0.003566] (inf: 2) | 0.0005892 [0.0004352, 0.000879] | 0.009047 [0.003547, 0.0191] (inf: 2) | 0.001955 [0.001052, 0.003638] (inf: 2) |
| AOPT | 0.0005456 [0.0003475, 0.000823] | 0.006962 [0.003093, 0.01232] (inf: 2) | 0.001604 [0.0009311, 0.003179] (inf: 2) | 0.0005495 [0.0002859, 0.000823] | 0.006733 [0.003093, 0.01232] (inf: 2) | 0.001604 [0.0009281, 0.003179] (inf: 2) |
| VOI | 0.0005867 [0.0004472, 0.000871] | 0.009063 [0.003547, 0.02117] (inf: 2) | 0.001955 [0.001052, 0.003566] (inf: 2) | 0.0005892 [0.0004352, 0.000879] | 0.009047 [0.003547, 0.0191] (inf: 2) | 0.001955 [0.001052, 0.003638] (inf: 2) |
| DISAGREE | 0.0008349 [0.0005694, 0.001209] (inf: 1) | 0.009742 [0.006607, 0.02668] (inf: 3) | 0.002814 [0.001481, 0.006327] (inf: 4) | 0.0008469 [0.0005892, 0.001209] (inf: 1) | 0.009548 [0.006607, 0.0262] (inf: 3) | 0.002814 [0.001492, 0.006327] (inf: 4) |
| RANDOM | 0.001563 [0.0009774, 0.002426] (inf: 1) | 0.02088 [0.01206, 0.0671] (inf: 5) | 0.008761 [0.003506, 0.014] (inf: 6) | 0.001588 [0.001082, 0.002463] (inf: 1) | 0.02043 [0.01177, 0.05511] (inf: 5) | 0.008761 [0.00351, 0.014] (inf: 6) |

## Paired one-sided Wilcoxon signed-rank vs RANK (H1: criterion's M2 < RANK's M2), Holm across 6 pairs

**P2_pooled** — registered family (P2 setting, pooled)

| Criterion | n tasks | non-zero pairs | lower / higher than RANK | W | p raw | p Holm |
|---|---|---|---|---|---|---|
| DOPT | 122 | 91 | 58 / 33 | 1739 | 0.08059 | 0.403 |
| AOPT | 122 | 78 | 64 / 14 | 489 | 8.149e-08 | 4.889e-07 |
| VOI | 122 | 93 | 59 / 34 | 1834 | 0.08902 | 0.403 |
| DISAGREE | 122 | 91 | 36 / 55 | 2853 | 0.9987 | 1 |
| RANDOM | 122 | 120 | 34 / 86 | 6012 | 1 | 1 |
| HYBRID | 122 | 91 | 58 / 33 | 1739 | 0.08059 | 0.403 |

**P2_T1** — secondary

| Criterion | n tasks | non-zero pairs | lower / higher than RANK | W | p raw | p Holm |
|---|---|---|---|---|---|---|
| DOPT | 61 | 49 | 33 / 16 | 442 | 0.04494 | 0.2247 |
| AOPT | 61 | 42 | 34 / 8 | 164 | 0.0001623 | 0.0009739 |
| VOI | 61 | 49 | 33 / 16 | 442 | 0.04494 | 0.2247 |
| DISAGREE | 61 | 45 | 20 / 25 | 606 | 0.8411 | 1 |
| RANDOM | 61 | 61 | 21 / 40 | 1516 | 1 | 1 |
| HYBRID | 61 | 49 | 33 / 16 | 442 | 0.04494 | 0.2247 |

**P2_T2** — secondary

| Criterion | n tasks | non-zero pairs | lower / higher than RANK | W | p raw | p Holm |
|---|---|---|---|---|---|---|
| DOPT | 61 | 42 | 25 / 17 | 418 | 0.3377 | 1 |
| AOPT | 61 | 36 | 30 / 6 | 85 | 4.885e-05 | 0.0002931 |
| VOI | 61 | 44 | 26 / 18 | 463 | 0.3544 | 1 |
| DISAGREE | 61 | 46 | 16 / 30 | 802 | 0.9979 | 1 |
| RANDOM | 61 | 59 | 13 / 46 | 1495 | 1 | 1 |
| HYBRID | 61 | 42 | 25 / 17 | 418 | 0.3377 | 1 |

**ALL_pooled** — secondary

| Criterion | n tasks | non-zero pairs | lower / higher than RANK | W | p raw | p Holm |
|---|---|---|---|---|---|---|
| DOPT | 124 | 91 | 58 / 33 | 1739 | 0.08059 | 0.403 |
| AOPT | 124 | 78 | 64 / 14 | 489 | 8.149e-08 | 4.889e-07 |
| VOI | 124 | 93 | 59 / 34 | 1834 | 0.08902 | 0.403 |
| DISAGREE | 124 | 91 | 36 / 55 | 2853 | 0.9987 | 1 |
| RANDOM | 124 | 120 | 34 / 86 | 6012 | 1 | 1 |
| HYBRID | 124 | 91 | 58 / 33 | 1739 | 0.08059 | 0.403 |

## Top-pick agreement (fraction of tasks where two criteria pick the same candidate; ALL tasks pooled)

| | RANK | HYBRID | DOPT | AOPT | VOI | DISAGREE | RANDOM |
|---|---|---|---|---|---|---|---|
| RANK | 1.00 | 0.25 | 0.25 | 0.35 | 0.23 | 0.25 | 0.11 |
| HYBRID | 0.25 | 1.00 | 1.00 | 0.72 | 0.95 | 0.37 | 0.11 |
| DOPT | 0.25 | 1.00 | 1.00 | 0.72 | 0.95 | 0.37 | 0.11 |
| AOPT | 0.35 | 0.72 | 0.72 | 1.00 | 0.69 | 0.34 | 0.11 |
| VOI | 0.23 | 0.95 | 0.95 | 0.69 | 1.00 | 0.40 | 0.11 |
| DISAGREE | 0.25 | 0.37 | 0.37 | 0.34 | 0.40 | 1.00 | 0.11 |
| RANDOM | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 |

RANDOM entries are the expected agreement of a uniform pick (mean of 1/n_candidates).

## Certified cases (P4) and M3

- Certified change_class tasks: T1 id 1 (RC-circuit), T2 id 1 (RC-circuit).
- Picks that raised rank on a certified task: none.
- Max DISAGREE contribution along the certified direction (max over tasks, candidates, and the two registered-in-spirit measures): 6.901e-11.
- M3: 2 change_class tasks. Experiments spent per round: RANK 2, HYBRID 2, DOPT 2, AOPT 2, VOI 2, DISAGREE 2, RANDOM 2, RANK+certificate 0.

### Sanity checks

- RANK picks a max-rank candidate on every T1/T2 task: True. HYBRID: True.
- RC circuit (id 1) T1: round-0 rank 2/3; candidate ranks [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]; pick ranks {'RANK': 2, 'HYBRID': 2, 'DOPT': 2, 'AOPT': 2, 'VOI': 2, 'DISAGREE': 2}.
- RC circuit (id 1) T2: round-0 rank 2/3; candidate ranks [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]; pick ranks {'RANK': 2, 'HYBRID': 2, 'DOPT': 2, 'AOPT': 2, 'VOI': 2, 'DISAGREE': 2}.

## T3 (descriptive only)

| Case | round-0 rank | triage | candidate: rank, mean M2 | picks (criterion: candidate, rank, M2) |
|---|---|---|---|---|
| Alien amplitude A = beta * product^p | 1/2 (cond inf) | design_within_class (certified [False]) | single product [1]: 1/2, 0.005295<br>two distinct products [1, 2]: 2/2, 0.006556 | RANK: two distinct products [1, 2], 2, 0.006556<br>HYBRID: two distinct products [1, 2], 2, 0.006556<br>DOPT: two distinct products [1, 2], 2, 0.006556<br>AOPT: two distinct products [1, 2], 2, 0.006556<br>VOI: two distinct products [1, 2], 2, 0.006556<br>DISAGREE: two distinct products [1, 2], 2, 0.006556<br>RANDOM: uniform (expectation), -, 0.005926 |
| Michaelis-Menten (substrate observed) | 2/3 (cond inf) | design_within_class (certified [False]) | substrate run S0 = 0.2: 2/3, 0.01528<br>substrate run S0 = 0.334: 2/3, 0.01553<br>substrate run S0 = 0.557: 2/3, 0.01559<br>substrate run S0 = 0.928: 2/3, 0.01516<br>substrate run S0 = 1.55: 2/3, 0.01408<br>substrate run S0 = 2.58: 2/3, 0.01258<br>substrate run S0 = 4.31: 2/3, 0.01156<br>substrate run S0 = 7.19: 2/3, 0.004673<br>substrate run S0 = 12: 2/3, 0.003733<br>substrate run S0 = 20: 2/3, 0.002828<br>E0 assay (direct enzyme measurement): 3/3, 0.01684 | RANK: E0 assay (direct enzyme measurement), 3, 0.01684<br>HYBRID: E0 assay (direct enzyme measurement), 3, 0.01684<br>DOPT: E0 assay (direct enzyme measurement), 3, 0.01684<br>AOPT: E0 assay (direct enzyme measurement), 3, 0.01684<br>VOI: E0 assay (direct enzyme measurement), 3, 0.01684<br>DISAGREE: E0 assay (direct enzyme measurement), 3, 0.01684<br>RANDOM: uniform (expectation), -, 0.01162 |
| Oral one-compartment PK | 3/4 (cond inf) | change_class (certified [True]) | oral, 4x denser sampling (96 pts, 24 h): 3/4, 0.002414<br>oral, 72 h window (48 pts): 3/4, 0.00417<br>oral + intravenous arm (24 + 24 pts): 4/4, 0.002577 | RANK: oral + intravenous arm (24 + 24 pts), 4, 0.002577<br>HYBRID: oral + intravenous arm (24 + 24 pts), 4, 0.002577<br>DOPT: oral + intravenous arm (24 + 24 pts), 4, 0.002577<br>AOPT: oral + intravenous arm (24 + 24 pts), 4, 0.002577<br>VOI: oral + intravenous arm (24 + 24 pts), 4, 0.002577<br>DISAGREE: oral + intravenous arm (24 + 24 pts), 4, 0.002577<br>RANDOM: uniform (expectation), -, 0.003054 |
| Sequence ansatz log f = log th0 + th1 log n + n log th2 | 2/3 (cond inf) | design_within_class (certified [False]) | narrow window n = [400, 401, 402]: 2/3, 0.8405<br>geometric spread n = [2, 5, 20, 100, 400]: 3/3, 2.61 | RANK: geometric spread n = [2, 5, 20, 100, 400], 3, 2.61<br>HYBRID: geometric spread n = [2, 5, 20, 100, 400], 3, 2.61<br>DOPT: geometric spread n = [2, 5, 20, 100, 400], 3, 2.61<br>AOPT: geometric spread n = [2, 5, 20, 100, 400], 3, 2.61<br>VOI: geometric spread n = [2, 5, 20, 100, 400], 3, 2.61<br>DISAGREE: narrow window n = [400, 401, 402], 2, 0.8405<br>RANDOM: uniform (expectation), -, 1.725 |

- MM notes: E0-assay channel absent at round 0: sigma = 1% of E0*

- SEQ notes: round-0 RMS of log f is 0 at theta = (1, 0, 1); sigma = 0.01 absolute on log f

## Runtime

- Wall clock 244.3 s on 4 processes; per-unit wall times in `units.jsonl` (slowest: ['T2 53: 172.99 s', 'T2 52: 65.32 s', 'T2 8: 61.03 s', 'T2 16: 39.18 s', 'T2 18: 34.93 s']).

## Implementation choices (fixed before the first run; none changed after outcomes were seen)

1. Candidate order is EXP-261's: IC-1 (ODEBench held-out IC) first, then box IC 0..9 (np.random.default_rng(0).uniform over the ic_bounds.json box, exactly EXP-261). The EXP-261 OOD window is not a candidate (the registered set is 10 box ICs + IC-1). Every argmax/argmin breaks ties by this order (first wins); RANK's ties are broken by the library ranker's stable sort, which is the same order.
2. Scaled sensitivity is formed with SIGNED scales, S diag(sign(theta*) max(|theta*|, 1e-3)): the exact Jacobian with respect to the sign-preserving log-parameters delta that DISAGREE draws and LM fits (theta = theta* exp(delta)). It differs from the registered S diag(max(|theta*|, 1e-3)) only by column signs (ids 16, 57, 61 have negative constants), which leaves rank, condition number, DOPT, AOPT and VOI scores identical and makes the posterior covariance Sigma's correlations correct for the draws. Constants with |theta*| < 1e-3 use the additive coordinate theta = theta* + 1e-3 delta (none in ODEBench; theta1 = 0 in the T3 sequence ansatz).
3. Every criterion, M1 and the round-0 verdict use the same whitened matrix A = W S~, W = diag(1/sigma_row), so F = A^T A. RANK is the library ranker (rank_designs_by_identifiability) applied to [A0; Ac]; M1 is rank([A0; Ac]) at the library's 1e-9 relative floor (practical_identifiability), so RANK and M1 read the same number. The round-0 verdict (and so the certificate input) is practical_identifiability on A0 with the 1e8 condition threshold.
4. Certificate: certify_class_symmetry_symbolic on the vector field sum_k w_k f_k(x; c) with x and w free (exactly EXP-261), applied to each round-0 null direction mapped to theta-space. For T2 this field-level certificate is sound (a symmetry of the field is invisible to any observation) but conservative (an x0-only symmetry that is not a field symmetry would not be certified). Triage: capability if round 0 is identifiable; change_class if every round-0 direction is certified; design_within_class otherwise.
5. A candidate whose trajectory at theta* leaves the finite domain (|x| > 1e6) or fails to integrate at 1e-12 cannot be acquired; it is dropped from that task's candidate set (EXP-261 behaviour) and listed.
6. P1 setting 'at least one candidate restores the maximum rank' is read as: max over candidates of rank([A0; Ac]) > rank(A0). A pick 'is max-rank' iff its rank equals that maximum.
7. DISAGREE: the K = 8 draws delta_k = Sigma^(1/2) z_k (symmetric square root, z = default_rng(262).standard_normal((8, n))) are shared by every candidate of a task. Each draw is rolled out with the true simulator from the candidate IC and observed under the task protocol (T1: states + analytic derivatives on 100 points; T2: x0 on 20 points), at rtol = atol = 1e-10. Score = mean over pairs of ||y_i - y_j||_2 divided by the RMS of the mean prediction (LLM-ACES normalises by the mean trajectory). A draw whose rollout fails is excluded; with fewer than 2 valid rollouts the score is NaN and the candidate is never picked.
8. M2 noise and starts use common random numbers: for seed s, rng = default_rng(s) draws the start delta0 ~ N(0, 0.1^2 I), then round-0 noise, then candidate noise, so the start and the round-0 noise are identical across candidates (and criteria) for a seed, and the candidate noise vector is identical across candidates of equal length.
9. M2 fit: scipy least_squares(method='lm') on delta with residuals (model - data)/sigma_row and the exact Jacobian (forward sensitivities at the current theta, rtol = atol = 1e-9). A trial point whose integration fails returns a constant 1e6 residual so LM backs off. A fit FAILS (NaN, counted) if least_squares raises, status <= 0 (including the default max_nfev = 100 n reached), the final point does not integrate, or x/cost are non-finite. The per-candidate mean over seeds is +inf if any seed failed (the registered 'error = +inf in rankings').
10. M2 score: e = delta_hat (error in log-parameters; delta* = 0). For a rank-deficient combined design with null space N (the collapsed right singular vectors of [A0; Ac] at the 1e-9 floor), e is replaced by (I - N N^T) e and the RMS is ||e|| / sqrt(rank) (the RMS over the rank-dimensional orthogonal complement; full rank reduces to the plain RMS over all parameters).
11. RANDOM: per task, rank-restoration = fraction of candidates at max rank; M2 = mean of the candidates' M2 (so +inf if any candidate's fit failed); agreement with a criterion = 1/n_candidates.
12. P2's 'p < 0.05' is read with the registered Holm correction across the 6 pairs with RANK (each pair: one-sided Wilcoxon signed-rank, H1: criterion's M2 < RANK's M2, paired by task, zero differences dropped, scipy default method). Raw p is reported alongside.
13. Populations: P1 = tasks in the P1 setting; P2 = tasks where >= 2 candidates attain the maximum rank. P3 and P5 state no setting, so their medians are over ALL scored tasks (T1 and T2 pooled); P3's rank-restoration part is scored on the P1 setting. The other populations are reported as secondary. All predictions are scored on T1+T2 pooled; per-protocol numbers are descriptive. T3 is never pooled into a prediction.
14. 'Within 5%' (P3) means |median_HYBRID / median_DOPT - 1| <= 0.05. 'Best on median' (P5) means DISAGREE's median is strictly lower than each of DOPT, AOPT and VOI.
15. Status vocabulary: HELD (the prediction as stated is met); FAILED (the registered falsifier fired); NOT MET (the prediction as stated is not met but the registered falsifier did not fire); NOT SCORED (the registered descriptive fallback applies).
16. P4 DISAGREE contribution along a certified direction d (unit, in delta-space, from the certificate's snapped theta-direction): with the task's 8 draws, rollouts at 1e-12, (a) score of the draws projected onto d divided by the full score, and (b) |full score - score of the draws with the d-component removed| / full score; the prediction is scored on the max over candidates of both.
17. sigma floor: if a round-0 channel has RMS exactly 0, sigma = 1% of the RMS over all round-0 channels (and 0.01 absolute if that is 0 too); every use is listed in the notes.
18. Guard: 10 min wall clock per (system, protocol) unit (SIGALRM in the worker); an errored or timed-out unit is excluded from every denominator and listed with its reason.
19. Bootstrap: percentile 95% CI of the median, 10,000 resamples of tasks with default_rng(262), the same resampled task sets for every criterion; +inf is replaced by 1e300 (rank-preserving) and any bound >= 1e299 is reported as inf. Wilcoxon uses the same substitution (it only uses signs and ranks).
20. T3 definitions (fixed before running). PK: theta = (ka, ke, V, F) = (1.2, 0.25, 30, 0.6), dose 100; round 0 = oral, 24 points on [0.25, 24] h; candidates = the three follow-ups of scripts/diagnostic_pk_bioavailability.py as added data: oral 96 points on [0.25, 24], oral 48 points on [0.25, 72], oral 24 points + IV 24 points on [0.25, 24]; one channel (plasma concentration, the IV arm measures the same quantity) with sigma from round 0; certificate symbolic on the oral closed form (t free). Michaelis-Menten: theta = (kcat, E0, Km) = (10, 0.1, 2); round 0 = substrate from S0 = 5 at 24 points on [0.1, 20] (EXP-087 STD); candidates = substrate from S0 in geomspace(0.2, 20, 10) on the same grid, plus an E0 assay (one direct observation of E0, sigma = 1% of E0* since the channel is absent at round 0); certificate = EXP-087's numeric route (its sampler, seed 87, 32 designs). Alien amplitude: A(P) = beta P^p, theta = (1, 1); round 0 = P = [1]; candidates 'single product' P = [1] and 'two distinct products' P = [1, 2] (propose_charge_product_designs); certificate symbolic (P free). Sequence ansatz: log f(n) = log th0 + th1 log n + n log th2 at theta = (1, 0, 1) (test_exhaustion.py); round 0 = narrow window n = [400, 401, 402]; candidates 'narrow window' (the same window again) and 'geometric spread' n = [2, 5, 20, 100, 400]; round-0 RMS is 0 at this theta, so sigma = 0.01 absolute on log f (1% multiplicative on f); certificate symbolic (n free, snap_rational=False as in the test).

## Post-run observations (written AFTER the outcomes were seen; nothing above, and no definition or number, was changed)

1. **P1 is not scorable: the registered setting has 1 task.** Only T2 id 53 (Apoptosis, 10 parameters, rank 8/10 from x0 alone) is rank-deficient at round 0 with a candidate that raises rank; every criterion restores full rank there (1/1). T1's only rank-deficient task is the RC circuit, which no candidate can fix. EXP-261's T1 id 18 rank drop (3/4 on the raw S) does not survive the registered scaling: on the whitened log-scaled matrix id 18 is rank 4/4, cond 2.0e7. The raw-S deficiency was a column-scale artefact of the 1e-9 relative floor (c1 = 100 and c3 = 50 against c0 = 0.4 and c2 = 0.24). Under T2, id 18 is the one practically non-identifiable task (cond 1.33e8, not certified).
2. **P2 is close to its falsifier.** RANK's median M2 is 4.4% LOWER than DOPT's (ratio 0.956), just inside the 5% margin that would have falsified the prediction in RANK's favour. The one-sided Wilcoxon for DOPT < RANK is not significant (58 lower / 33 higher, raw p = 0.081, Holm p = 0.40). Medians and signed ranks point in opposite directions here: DOPT beats RANK on more tasks, but RANK's median is lower. AOPT is the only criterion that beats RANK significantly (64 lower / 14 higher, Holm p = 4.9e-7), and it has the lowest pooled median of all seven.
3. **P3 holds trivially.** DOPT picked a max-rank candidate on every task, so HYBRID and DOPT picked the same candidate on 124/124 tasks. VOI agrees with DOPT on 95% of tasks. RANK agrees with any Fisher criterion on only 23-35% of tasks: its condition-number tiebreak is a genuinely different rule.
4. **P4 positive control.** The certified-direction measure is not blind by construction. On T3 PK (certified F/V) it gives about 2e-14 for the two oral candidates and 1.00 / 0.99 for the oral + IV candidate, which leaves the oral class and breaks the symmetry. On the RC circuit (T1 and T2) the maximum is 6.9e-11. Every candidate and every pick stays at rank 2/3 in both protocols.
5. **M2 caveat (registered metric, flagged here).** For a rank-deficient pick, M2 counts only the identified directions, so a pick that identifies less can score better. The T3 rows show this: in the Alien case, 'single product' (rank 1/2) scores 0.0053 against 0.0066 for 'two products' (2/2); in SEQ, DISAGREE's narrow-window pick (2/3) scores 0.84 against 2.61. In T1/T2 this barely matters: only 3 tasks are rank-deficient at round 0 and every pick is max-rank. SEQ's M2 is also in additive units of 1e-3 for theta1 = 0, which inflates it.
6. **Fit failures** (lm status 0: max_nfev = 100 n reached) are concentrated in T2 ids 14, 18, 53, 60, 62 and T1 id 18: 38 of 6,700 T1/T2 fits in total. Picks with +inf M2 are counted in `n_inf` in the median table.
7. **Pre-existing test failure found, not caused by EXP-262.** The T3 Michaelis-Menten certificate (EXP-087's numeric route) is NOT certified here, so MM triages as design_within_class. It stays uncertified when given EXP-087's own finite-difference null direction: `tests/diagnostics/test_exp087_michaelis_menten.py::test_p3_class_level_certificate_and_change_class_triage` FAILS at HEAD with src/ and tests/ unmodified. The worst single-design response is 1.54e-6, against a null tolerance of 1e-7. This is a central-difference floor (tolerance / step), the same mechanism as EXP-261 Amendment 3, now in `certify_class_symmetry_numeric`. No registered prediction depends on it.
