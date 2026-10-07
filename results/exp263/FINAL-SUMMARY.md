# EXP-263 — final summary (stopped 2026-09-30 by PI decision; see predictions.md Amendment 4)

Identifiability audit of the DiscoverPhysics benchmark (11 public worlds, 13 models × 5 seeds on the pinned leaderboard `8e9c858`, simulator pinned `33b7fa9`).
- **Pre-registration:** `predictions.md` block of 2026-09-28 (`8f746eb`).
- **Retrospective:** the outcomes were seen before registration; the verdicts were blind.
- **Scope:** 10 worlds are complete. three_species is complete except three joint fits (Amendment 4), so its verdict is not final.
- **Detail:** full per-claim tables in `aggregate_provisional_10worlds.md`; checkpoints in `cache/ckpt/`.

## Per-world verdicts

**Columns:**
- *Explanation*: the leaderboard's explanation score, max and mean over the 13 models.
- *TG / TG & EF*: TG counts trajectory-good cells (geometric position error ≤ 0.1); TG & EF counts those that also fail the explanation (score < 0.9).

| world | predicted | computed | decisive claim(s) | explanation max | explanation mean | TG / TG & EF |
|---|---|---|---|---|---|---|
| fractional | NONID | **NONID** | FR-2: "fractional operator" vs power law, **certified equivalent** (exact reparameterization) | 0.48 | 0.39 | 11 / 11 |
| circle | NONID | **NONID** | CI-2: the same certified equivalence at α = 0.75 | 0.70 | 0.35 | 4 / 4 |
| extra_dimensions | RESOLVABLE | **RESOLVABLE** | ED-3: R_c has SE 1.19 at DEFAULT and 0.026 at BEST (tolerance 0.30); only 5 of 65 designs identify it | 0.54 | 0.25 | 6 / 6 |
| gravity | ID | RESOLVABLE | GR-2: the exponential alternative matches at DEFAULT and is separated only by other designs | 1.00 | 0.66 | 7 / 3 |
| yukawa | ID | RESOLVABLE | YU-3: λ identifiable only at BEST; YU-4 role swap separated only by design | 1.00 | 0.51 | 9 / 5 |
| hubble | ID | **ID** | — | 1.00 | 0.38 | 4 / 2 |
| ether | ID | **ID** | — | 0.90 | 0.39 | 0 / 0 |
| oscillator | ID | **ID** | — | 0.92 | 0.39 | 4 / 2 |
| coulomb | ID | **ID** | — | 0.98 | 0.52 | 1 / 1 |
| dark_matter | not predicted | **ID** | DM-3: hidden count vs coupling is distinguishable at DEFAULT (N = 5 residual 1.8σ; N = 20 also distinguishable) | 0.56 | 0.25 | 0 / 0 |
| three_species | ID | not final | see below | 0.94 | 0.31 | 1 / 0 |

**three_species, partial:**
- Coupling ratios TS-4a/b are identifiable at DEFAULT (SE ≈ 0.005 against a tolerance of 0.25).
- The joint fits for all_identical, merge_AB/AC/BC, nonneg and probes_A are all DISTINGUISHABLE at DEFAULT: residual 1.2 to 5.0σ, and above 1σ on 64/64 designs.
- Three joint fits are NOT SCORED:
  - b5a9 (slightly wrong partition): single-design DEFAULT residual 0.73σ. This alone rules out an ID verdict.
  - probes_B and probes_C: single-design DEFAULT residuals 2.36σ and 2.37σ.

## Predictions

**P-EXP-263-1 (verdicts): FALSIFIED.**
- 7 of the 9 scored worlds match (registered: pass at ≥ 9, falsified below 8).
- Misses: gravity and yukawa, predicted ID, are RESOLVABLE.
- three_species (predicted ID) cannot come out ID because of b5a9's 0.73σ, so the count cannot reach 8.
- Dark matter was not predicted. Session 045's guess ("non-identifiable decomposition") was wrong, as the Prony argument of Session 046 anticipated.

**P-EXP-263-2 (ceilings, consistency check): NOT SUPPORTED.**
- Mann–Whitney on per-world maxima, NONID ∪ RESOLVABLE vs ID: U = 10, one-sided p = 0.34.
- Dark matter (ID) has a max of 0.56, below circle's 0.70. Gravity and yukawa (RESOLVABLE) have a max of 1.00.
- Non-identifiability does not explain every low ceiling. Dark matter's is capability; its trajectory error is 251.

**P-EXP-263-3 (Proposition 3 signature): CONFIRMED.**
- In the certified-equivalence worlds (fractional, circle), 15 of 15 trajectory-good model cells fail the explanation.
- In the ID worlds, 5 of 9 do.
- Fisher exact, one-sided: p = 0.012. Counting three_species as ID gives p = 0.0047.
- Agents that predict the trajectories perfectly still fail the rubric exactly where the rubric grades a certified-equivalent distinction.

**P-EXP-263-4 (the ranker finds the resolving design): FALSIFIED as registered.**
- For extra_dimensions, the ranker's BEST design identifies R_c (SE 0.026), but its minimum probe separation is 1.79, not ≤ 1.5 as registered.
- For gravity, BEST = DEFAULT (a one-parameter world, so every design ties) and it does not resolve GR-2.

## Benchmark defects found (report to the DiscoverPhysics authors)

1. **Graded distinctions that no experiment can make.**
   - In 2D the benchmark's Riesz kernel is exactly a power law: `riesz_2d_force ∝ r^-(3−2α)`. Sympy verifies the reparameterization both ways, and the numeric maximum relative difference is 2e-16.
   - The fractional and circle rubrics award their top band for naming a fractional-Laplacian operator rather than an "ordinary power law". No experiment in the agent's interface can tell the two apart.
   - The benchmark's held-out trajectory test uses designs the agent could have run, so it is blind to the same distinction (Proposition 3).
2. **The fractional rubric's top band contradicts the simulated physics.**
   - The world is configured with α = 0.5 (`worlds.py:501`), so force ∝ 1/r². That decays faster than the 2D Laplacian's 1/r.
   - The top band and the "optimal explanation" say the force "decays more slowly … enhanced long-range" (`worlds.py:511`).
   - The displayed true-law string says α = 0.75 (`worlds.py:48`), while the rubric says the ground truth is α = 0.5.
   - The circle rubric's "intermediate between the logarithmic 2D Laplacian and pure long-range attraction" is likewise contradicted by its r^-1.5 force.
   - An agent that reports the physics correctly is scored down.

## Caveats

- **Local CRLBs.** They are computed at the true constants, and the fits may find local optima. Local optima bias the result toward DISTINGUISHABLE, not EQUIVALENT.
- **Dark matter derivatives.** The finite-difference and automatic-differentiation derivatives disagree for dark matter (median relative difference 1.0; the dynamics are chaotic), so its CRLB is unreliable. Its verdict rests on the count-vs-coupling fits.
- **Engine scope.** Time-evolving (wave/diffusion) alternatives were not scorable on the nbody engine.
- **Data granularity.** Leaderboard cells are published means over 5 seeds, not per-seed values.
- **Retrospective design.** The outcomes were known at registration.
- **Compute.** About 49 hours of wall clock and five container restarts. Execution notes are in `aggregate_provisional_10worlds.md` and `cache/resume*.log`.
