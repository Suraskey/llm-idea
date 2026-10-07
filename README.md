# LLM-IDEA: code and run records

Code, run records, and pre-registration record for

> Surya Shetty and Ulisses Braga-Neto. *LLM-IDEA: Identifiability-Driven Experimental Agent for Autonomous Discovery of Mechanistic World Models.* arXiv, 2026.

Everything the paper measures with no language model in the loop can be re-run from this repository: the identifiability engine, the Alien Universe simulator, the scorers, the ODEBench sweep (EXP-261), the selection-criterion comparison (EXP-262), and the DiscoverPhysics audit (EXP-263). The live-agent experiments cannot be re-run without the proposer and a model provider, so their complete run records are included instead: every model call with the prompt that produced it, every hypothesis the scorer read, and the scorer's output for every seed.

Licensed under Apache-2.0 (see `LICENSE`). Cite with `CITATION.cff`.

## What is here

| Path | Contents | Paper |
|---|---|---|
| `src/ascension/diagnostics/` | The identifiability engine: sensitivity-matrix diagnostic, exact symbolic structural check, design ranker, exhaustion certificate (symbolic and numeric routes, with the regular-point check), and the loop that couples them to a proposer. No model client is imported anywhere in it; `tests/diagnostics` enforces that. | Sections 3.2 to 3.7, Algorithms 1 and 2 |
| `src/ascension/simulator/` | The Alien Universe: the two-body law with the hidden per-body charge, the symplectic integrator, the tuned operating point, the observation bundles agents see. | Section 4 |
| `src/ascension/benchmarks/` | The scorers (the depth ladder with the single-configuration and cross-configuration gates, the terminal-exponent metric), the textbook ODE systems, the amplitude-model loop the live director drove, the residual- and experiment-feedback blocks. | Sections 4.2 to 4.5, 5.2 to 5.6 |
| `src/ascension/common/` | The prompt registry (name, version, SHA-256), a settings shim, and the Postgres helper used only by the optional database scripts. | Appendix C |
| `scripts/` | One script per table, figure, or deterministic experiment (next section). | |
| `tests/` | The unit tests of the released packages. | |
| `prompts/` | Every registered system prompt, verbatim, one file each, with `index.tsv` giving its SHA-256. The `system_prompt_hash` on every released model call matches one of these. | Appendix C |
| `records/` | The run records of the live-agent experiments and the two recorded negatives (layout below). | Sections 5.1, 5.3, 5.4, 5.6, Appendix A |
| `results/` | The outputs of EXP-261, EXP-262, and EXP-263 as run for the paper, including the three tables shipped as the paper's ancillary files. | Sections 5.8 to 5.10 |
| `predictions.md` | The pre-registration record: every prediction, falsification condition, and amendment, in the order they were written. | Section 5.1 |
| `external/` | Where to clone LLM-ACES and DiscoverPhysics at the commits the paper used. Nothing from either is copied here. | |

The Python package keeps the project's working name, `ascension`, so that the hashes, checksums, and module paths in the run records match what was run.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[figures,dev]"
pytest            # the released unit tests; a few minutes, no network
```

Python 3.11 to 3.13. The engine needs only numpy, scipy, and sympy; `pysindy` is pulled in by the benchmarks package for its SINDy baseline. Add `.[db]` only for the two scripts that read a restored Postgres database.

## Reproducing each table and figure

Figures are written to `figures/`. Every command below is deterministic and makes no network call.

| Paper item | Command | Output |
|---|---|---|
| Table 1, textbook ODEs (Section 5.2) | `python scripts/diagnostic_classic_odes.py` | printed table: rank, verdict, condition number per system and design |
| Pharmacokinetic case and intravenous-arm ranking (Section 5.2) | `python scripts/diagnostic_pk_bioavailability.py` | printed |
| Constructed battery, EXP-082 (Section 5.2) | `python scripts/structural_boundary_battery.py` | printed |
| Michaelis-Menten certificate, EXP-087 (Section 5.4) | `pytest tests/diagnostics/test_exp087_michaelis_menten.py` | the registered bar is the test's assertion; see Section 5.4 on the platform dependence |
| Figure 2, observability lift (EXP-076); Figure 3, transfer (EXP-080/081/083b); Figure 4, loop trace (EXP-078); Figure 5, metric self-diagnosis (EXP-088); Figure 6, dose response (EXP-075/077) | `python scripts/make_paper_figures.py` | `figures/fig1-observability-lift`, `fig2-transfer-l3-stable`, `fig0-loop-closure-trace`, `fig3-oed-self-diagnosis`, `fig4-dose-response` (pdf and png) |
| The numbers behind Figure 5 (EXP-088) | `python scripts/diagnose_metric_identifiability.py` | printed: rank, condition number, and indifference band per design |
| Figure 7, selection criteria (EXP-262) | `python scripts/make_exp262_figure.py` | `figures/fig5-criterion-head-to-head` from `results/exp262/picks.tsv` |
| Table 2, DiscoverPhysics audit (EXP-263) | see below | `results/exp263/claims.tsv`, `leaderboard_cells.tsv`, `FINAL-SUMMARY.md` |
| Table 3, equal-wall-clock comparison (EXP-075, Appendix A.1) | `records/EXP-075/exp075_20260601_140400/results.tsv` | per-seed depth and per-layer vectors as scored; re-score any run with `score_from_records.py` (below) |
| Table 4, call-matched control (EXP-077, Appendix A.2) | `records/EXP-077/exp077_N40_20260603_143804/results.tsv` (and the N15, N130 batches) | same |
| Factorial contrasts (EXP-089, Section 5.3) | `python scripts/analyze_exp089_factorial.py` | printed: the four cells, Fisher tests, Clopper-Pearson intervals |
| Proposal-coupled rescore (Appendix A.3) | `python scripts/rescore_l2_proposal_coupled.py` | printed; reads the stored hypotheses from the run database (`.[db]`) |
| EXP-083 landing summary (Section 5.3) | `python scripts/summarize_exp083.py records/EXP-083/exp083_20260721_010654` | printed |
| ODEBench sweep, EXP-261 (Section 5.8) | `python scripts/sweep_odebench_identifiability.py` | `results/exp261/results.tsv`, `results.json`, `summary.json`; needs `external/LLM-ACES` |
| Selection-criterion head-to-head, EXP-262 (Section 5.9) | `python scripts/exp262_head_to_head.py` | `results/exp262/`; needs `external/LLM-ACES`; about four minutes on four processes |
| Table 5, prompt usage (Appendix C) | `python scripts/export_prompt_appendix.py` | regenerates the appendix; needs the run database (`.[db]`) |

**EXP-263** runs in stages against the DiscoverPhysics simulator and leaderboard (clone both as in `external/README.md`, install DiscoverPhysics's dependencies, JAX included):

```bash
python scripts/exp263_discoverphysics_audit.py --stage leaderboard
python scripts/exp263_discoverphysics_audit.py --stage worlds --jobs 4      # hours; checkpoints per fit
python scripts/exp263_discoverphysics_audit.py --stage aggregate
```

The paper's run stopped with one three_species joint fit unfinished (Amendment 4 in `predictions.md`); `results/exp263/FINAL-SUMMARY.md` is the state it stopped in.

## Run records

```
records/
  manifest.jsonl                       one line per exported run: experiment, batch, seed, arm, run_id, row counts
  EXP-076/exp076_20260530_155507/      one batch = one launch of a cell (8 seeds, one or two arms)
    results.tsv                        seed, arm, run_id, max_depth, per_layer vector, as scored for the paper
    score_<arm>_seed<k>.log            the scorer's full output per run (the per-layer verdicts and notes)
    <arm>_seed<k>_<run_id>.log         the orchestrator log of that run
    config.txt                         launch configuration and the development-repo commit it ran from
    prompt_versions.txt                prompt hashes at launch (empty for batches before 2026-07-21; see below)
    runs/<run_id>/
      run.json                         the run row: status, halt reason, budget, ceiling, benchmark flags
      llm_calls.jsonl.gz               every model call: request (system prompt and contents) and response verbatim,
                                       provider, model, provider_request_id, system_prompt_hash, tokens, cost, latency
      nodes.jsonl                      every knowledge-graph node the run wrote; type='hypothesis' rows are what the scorer reads
      edges.jsonl                      edges between those nodes
      hypothesis_posteriors.jsonl      each agent's posterior over candidates per round
      experiment_feedback.jsonl        the deterministic fitting-engine feedback blocks delivered to the agents
      residual_feedback.jsonl          the residual-feedback blocks
      disputes.jsonl, verdict_posteriors.jsonl, interventions.jsonl, agents.jsonl
```

| Directory | Paper | Batches |
|---|---|---|
| `EXP-072` | Section 5.1, first recorded negative (passive orbit data, two runs) | the two runs; the record is in `predictions.md` and the database export |
| `EXP-074` | Section 5.1, second recorded negative (both infrastructure-confounded attempts) | `attempt1`, `attempt2`, with the batch logs |
| `EXP-075` | Section 5.6 and Appendix A.1 | one batch |
| `EXP-076` | Section 5.3, the lift | one batch |
| `EXP-077` | Section 5.6 and Appendix A.2 | N15, N40, N130 (partial) |
| `EXP-078` | Section 5.4, the live loop | the trace the figure was drawn from and the director's six calls, from the on-disk call sink (they were not written to the database) |
| `EXP-080`, `EXP-081`, `EXP-083`, `EXP-083b` | Section 5.3, second-regime transfer | one batch each |
| `EXP-089` | Section 5.3, the factorial | seven batches (the cell was relaunched after a provider model reached end of life; Amendment 2) |

Not included: pilots (EXP-073, the EXP-076 pilot), smoke runs, and the April single-agent ODE-loop runs, none of which the paper reports. In the EXP-089 batch of 2026-08-25, seeds 4 to 7 never completed (their `results.tsv` rows read `ERR`) and the run ids on those rows have no database record; each such `runs/<run_id>/` holds only a `MISSING.txt`. The cell was relaunched on 2026-09-08 (the six later batches), which is where those seeds' records are.

To re-score a run from its record without a database:

```bash
python scripts/score_from_records.py records/EXP-076/exp076_20260530_155507/runs/<run_id> --family
```

The flag per batch is in its `config.txt` (`score_flag=`); the EXP-076 control arm, EXP-075, EXP-077, and the EXP-089 instrument_only arm take no flag. `scripts/score_alien_run.py` is the same scorer reading a restored database, and `scripts/dump_run_artifacts.py` is how the database rows were exported.

Two things to know when reading the records. First, `prompt_versions.txt` is empty for every batch before 2026-07-21: the launch scripts dumped the registry before it was populated. The authoritative link from a call to its prompt is the `system_prompt_hash` on the call, which matches `prompts/index.tsv`. Second, absolute paths from the machine the runs were made on were replaced by `<repo>` and `<home>` in the copied logs; nothing else in the records was edited.

## What is not released

The live proposer (the Agentic Council, its knowledge graph, the deliberation court, the overseer, and the orchestrator that ran them against a model provider) and the integer-sequence adapter of Section 5.7 are not part of this release. The Council's prompts, every call it made in the reported experiments, and every hypothesis it produced are in `prompts/` and `records/`.

## Pre-registration

`predictions.md` is the development repository's pre-registration file as of 2026-10-07, with two occurrences of a billing-project name replaced by `[GCP project]`. The paper's claim that each prediction was committed before its run refers to that repository's commit history, which is private; the ordering is available from the authors on request.
