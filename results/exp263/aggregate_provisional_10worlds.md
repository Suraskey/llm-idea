> **PROVISIONAL: 10 of 11 worlds; missing ['three_species']. Prediction statistics below are computed on the available worlds only and are NOT the registered outcomes.**

# EXP-263 — Identifiability audit of the DiscoverPhysics benchmark

Generated 2026-09-30 07:54 UTC by `scripts/exp263_discoverphysics_audit.py`. Pre-registration: `predictions.md`, block "Pre-registered block added 2026-09-28 (Session 047 — EXP-263 ...)" (commit 8f746eb). Retrospective test: the per-world outcomes were seen before registration; the verdicts were not.

- DiscoverPhysics pinned `33b7fa9df96de9c35744efd181ca7e5a8dd60ad5` (engine `nbody`, yoshida4, dt = 0.005, softening 0.05); leaderboard pinned `8e9c858a95f9282717771e52bab30c45cf253259`.
- Step 1 was run first: `leaderboard_cells.tsv` (13 models x 11 worlds = 143 cells, per-cell means over 5 seeds as published) was copied from the pinned leaderboard before any verdict was computed.
- No LLM anywhere. Noise: sigma = 0.05 * sqrt(Var_GT[world]) (the benchmark's `noise_frac` definition in `scripts/run_benchmark.py`), i.i.d. Gaussian on every agent-visible position.

## Per-world results

| world | predicted | **verdict (registered)** | variant: solo-DEFAULT fits | BEST | claims unresolved at DEFAULT | unresolved at BEST | max expl | mean expl | median geom err | TG | EF | TG&EF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gravity | ID | **RESOLVABLE** | RESOLVABLE | DEFAULT | GR-2:exp | GR-2:exp | 1.00 | 0.66 | 0.010 | 7 | 9 | 3 |
| yukawa | ID | **RESOLVABLE** | RESOLVABLE | S13 | YU-2:laplace, YU-2:pow, YU-3, YU-4:swap | YU-2:pow | 1.00 | 0.51 | 0.058 | 9 | 9 | 5 |
| hubble | ID | **ID** | ID | S49 | — | — | 1.00 | 0.38 | 152.391 | 4 | 11 | 2 |
| ether | ID | **ID** | ID | S47 | — | — | 0.90 | 0.39 | 120.593 | 0 | 12 | 0 |
| oscillator | ID | **ID** | RESOLVABLE | DEFAULT | — | — | 0.92 | 0.39 | 0.664 | 4 | 11 | 2 |
| circle | NONID | **NONID** | NONID | S31 | CI-2:laplace, CI-2:pow, CI-3, CI-5:pow_fast | CI-2:pow, CI-5:pow_fast | 0.70 | 0.35 | 1.390 | 4 | 13 | 4 |
| extra_dimensions | RESOLVABLE | **RESOLVABLE** | RESOLVABLE | S13 | ED-1:crossover, ED-1:laplace, ED-1:pow, ED-1:yukawa, ED-2:crossover, ED-2:laplace, ED-2:pow, ED-2:yukawa, ED-3 | ED-1:crossover, ED-1:pow, ED-2:crossover, ED-2:pow | 0.54 | 0.25 | 0.154 | 6 | 13 | 6 |
| fractional | NONID | **NONID** | NONID | S13 | FR-2:laplace, FR-2:pow, FR-3, FR-4:pow_fast | FR-2:pow, FR-4:pow_fast | 0.48 | 0.39 | 0.009 | 11 | 13 | 11 |
| dark_matter | RES or NONID | **ID** | ID | S26 | — | DM-3 | 0.56 | 0.25 | 251.471 | 0 | 13 | 0 |
| coulomb | ID | **ID** | RESOLVABLE | DEFAULT | — | — | 0.98 | 0.52 | 4.923 | 1 | 11 | 1 |

TG = trajectory-good cells (geom_pos_err <= 0.1); EF = explanation-fail cells (score < 0.9); 13 model cells per world.

## Per-claim results

PARAMETER: relative CRLB standard error of the graded quantity (log units for multiplicative tolerances, absolute for alpha ranges) for ONE experiment divided by sqrt(16); identifiable if SE < tau. Step 1e-4 (registered) and 1e-3 (stability) shown. MECHANISM: RMS residual / sigma of the best-fit named alternative; `joint` = one fit over DEFAULT + 64 sampled designs (registered), `solo` = the alternative refitted to that design alone (diagnostic).

### gravity — verdict **RESOLVABLE** (BEST = DEFAULT)

sigma = 0.1035; ranker: 65 design(s) tied at the top score; 'DEFAULT' fully identifies all 1 parameters (rank 1/1, cond 1.00e+00); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| GR-1 | NOT_SCORED | — | named alternative is a time-evolving / wave-like field; the pinned nbody engine has only instantaneous pairwise forces, so it cannot be simulated here |
| GR-2 | MECHANISM | DISTINGUISHABLE-BY-DESIGN (solo variant: DISTINGUISHABLE-BY-DESIGN) | inv2: DISTINGUISHABLE [joint DEF 1.385, BEST 1.385, max 8.129, 19/64 >1; solo DEF 0.108, solo BEST 0.108, DEF+BEST 0.108]<br>exp: DISTINGUISHABLE-BY-DESIGN [joint DEF 0.515, BEST 0.515, max 2.400, 5/64 >1; solo DEF 0.001, solo BEST 0.001, DEF+BEST 0.001]<br>inv3: DISTINGUISHABLE [joint DEF 3.419, BEST 3.419, max 12.494, 25/64 >1; solo DEF 0.204, solo BEST 0.204, DEF+BEST 0.204]<br>const: DISTINGUISHABLE [joint DEF 2.905, BEST 2.905, max 13.780, 22/64 >1; solo DEF 0.119, solo BEST 0.119, DEF+BEST 0.119] |
| GR-3 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE-BY-DESIGN) | swap: DISTINGUISHABLE [joint DEF 4.621, BEST 4.621, max 30.616, 27/64 >1; solo DEF 0.000, solo BEST 0.000, DEF+BEST 0.000]<br>product: DISTINGUISHABLE [joint DEF 5.718, BEST 5.718, max 30.603, 18/64 >1; solo DEF 0.000, solo BEST 0.000, DEF+BEST 0.000] |

### yukawa — verdict **RESOLVABLE** (BEST = S13)

sigma = 0.1191; ranker: 1 design(s) tied at the top score; 'S13' fully identifies all 2 parameters (rank 2/2, cond 6.60e+00); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| YU-1 | NOT_SCORED | — | named alternative is a time-evolving / wave-like field; the pinned nbody engine has only instantaneous pairwise forces, so it cannot be simulated here |
| YU-2 | MECHANISM | INDETERMINATE (solo variant: INDETERMINATE) | laplace: DISTINGUISHABLE-BY-DESIGN [joint DEF 0.121, BEST 1.254, max 3.780, 6/64 >1; solo DEF 0.078, solo BEST 0.998, DEF+BEST 1.088]<br>pow: INDETERMINATE [joint DEF 0.194, BEST 0.678, max 0.678, 0/64 >1; solo DEF 0.002, solo BEST 0.022, DEF+BEST 0.118] |
| YU-3 | PARAMETER | IDENTIFIABLE only at BEST | SE DEFAULT 0.480 (1e-3: 0.480); SE BEST 0.071 (1e-3: 0.071); tau 0.250 (half-log 0.347); 4/65 designs identify; info-optimal S62 SE 0.017 |
| YU-4 | MECHANISM | DISTINGUISHABLE-BY-DESIGN (solo variant: DISTINGUISHABLE-BY-DESIGN) | swap: DISTINGUISHABLE-BY-DESIGN [joint DEF 0.595, BEST 2.023, max 12.356, 6/64 >1; solo DEF 0.000, solo BEST 2.51e-14, DEF+BEST 0.298] |

### hubble — verdict **ID** (BEST = S49)

sigma = 0.3209; ranker: 1 design(s) tied at the top score; 'S49' fully identifies all 2 parameters (rank 2/2, cond 1.39e+00); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| HU-1 | UNTYPED | — | lower band names only a generic 'wrong operator family' |
| HU-2 | UNTYPED | — | no lower band names an alternative |
| HU-3 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE) | const_push: DISTINGUISHABLE [joint DEF 13.192, BEST 30.729, max 30.729, 64/64 >1; solo DEF 12.598, solo BEST 9.255, DEF+BEST 30.882]<br>central_only: DISTINGUISHABLE [joint DEF 16.530, BEST 39.143, max 39.442, 64/64 >1; solo DEF 16.530, solo BEST 39.141, DEF+BEST 39.143] |
| HU-4 | UNTYPED | — | consequence of HU-3/HU-5; the lower band is a vagueness demotion, no alternative law |
| HU-5 | PARAMETER | IDENTIFIABLE at DEFAULT | SE DEFAULT 2.51e-06 (1e-3: 8.19e-06); SE BEST 2.14e-04 (1e-3: 2.14e-04); tau 0.250 (half-log 0.347); 65/65 designs identify; info-optimal DEFAULT SE 2.51e-06 |

### ether — verdict **ID** (BEST = S47)

sigma = 0.2305; ranker: 1 design(s) tied at the top score; 'S47' fully identifies all 2 parameters (rank 2/2, cond 1.26e+01); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| ET-1 | UNTYPED | — | lower bands only 'fails to identify which particle is the anchor' / generic 'wrong operator family' |
| ET-2 | UNTYPED | — | no lower band names an alternative |
| ET-3 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE) | mass_dep_drift: DISTINGUISHABLE [joint DEF 6.363, BEST 2.802, max 5.940, 64/64 >1; solo DEF 5.037, solo BEST 2.708, DEF+BEST 5.029]<br>repulsor: DISTINGUISHABLE [joint DEF 4.429, BEST 2.617, max 5.262, 64/64 >1; solo DEF 3.840, solo BEST 2.315, DEF+BEST 4.207] |
| ET-4 | PARAMETER | IDENTIFIABLE at DEFAULT | SE DEFAULT 0.003 (1e-3: 0.003); SE BEST 0.003 (1e-3: 0.003); tau 0.250 (half-log 0.347); 65/65 designs identify; info-optimal S11 SE 0.003 |
| ET-5 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE) | mass_dep_drift: DISTINGUISHABLE [joint DEF 6.363, BEST 2.802, max 5.940, 64/64 >1; solo DEF 5.037, solo BEST 2.708, DEF+BEST 5.029] |

### oscillator — verdict **ID** (BEST = DEFAULT)

sigma = 0.1258; ranker: 1 design(s) tied at the top score; 'DEFAULT' fully identifies all 3 parameters (rank 3/3, cond 7.07e+00); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| OS-1 | UNTYPED | — | no lower band names an alternative spatial form (the 4–6 band keeps the 1/r form) |
| OS-2 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE-BY-DESIGN) | const_coupling: DISTINGUISHABLE [joint DEF 7.389, BEST 7.389, max 35.900, 49/64 >1; solo DEF 0.739, solo BEST 0.739, DEF+BEST 0.739] |
| OS-3 | UNTYPED | — | lower band only says 'omits the sign-flipping behaviour' (an omission) |
| OS-4 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE-BY-DESIGN) | transient: DISTINGUISHABLE [joint DEF 9.716, BEST 9.716, max 25.349, 41/64 >1; solo DEF 0.348, solo BEST 0.348, DEF+BEST 0.348] |
| OS-5 | PARAMETER | IDENTIFIABLE at DEFAULT | SE DEFAULT 0.023 (1e-3: 0.023); SE BEST 0.023 (1e-3: 0.023); tau 0.250 (half-log 0.347); 54/65 designs identify; info-optimal S58 SE 0.004 |
| OS-6 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE-BY-DESIGN) | swap: DISTINGUISHABLE [joint DEF 1.079, BEST 1.079, max 134.980, 36/64 >1; solo DEF 0.000, solo BEST 0.000, DEF+BEST 0.000] |
| OS-N | NOTE | — | grader convention; no tolerance stated (CRLB of G0 reported for information) |

Information only (OS-N(G0), no tolerance registered): SE(log G0) DEFAULT 0.065, BEST 0.065.

### circle — verdict **NONID** (BEST = S31)

sigma = 0.1284; ranker: 1 design(s) tied at the top score; 'S31' fully identifies all 2 parameters (rank 2/2, cond 1.68e+00); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| CI-1 | NOT_SCORED | — | named alternative is a time-evolving / wave-like field; the pinned nbody engine has only instantaneous pairwise forces, so it cannot be simulated here |
| CI-2 | MECHANISM | EQUIVALENT (certified) (solo variant: EQUIVALENT (certified)) | pow: EQUIVALENT (certified) [joint DEF 0.000, BEST 3.92e-12, max 3.92e-12, 0/64 >1; solo DEF 0.000, solo BEST 0.000, DEF+BEST 0.000]<br>laplace: DISTINGUISHABLE-BY-DESIGN [joint DEF 0.381, BEST 7.333, max 7.754, 60/64 >1; solo DEF 0.011, solo BEST 4.817, DEF+BEST 4.913] |
| CI-3 | PARAMETER | IDENTIFIABLE only at BEST | SE DEFAULT 0.384 (1e-3: 0.384); SE BEST 6.69e-04 (1e-3: 6.69e-04); tau 0.125 (half-log 0.125); 48/65 designs identify; info-optimal S32 SE 5.57e-04 |
| CI-4 | UNTYPED | — | lower band only says 'omits the uniform-coupling claim' (an omission, no alternative law) |
| CI-5 | MECHANISM | EQUIVALENT (numeric) (solo variant: EQUIVALENT (numeric)) | pow_fast: EQUIVALENT (numeric) [joint DEF 0.000, BEST 9.81e-13, max 6.60e-12, 0/64 >1; solo DEF 4.57e-15, solo BEST 1.24e-12, DEF+BEST 1.17e-12]; top-band claim's own family (decay no faster than 1/r): CONTRADICTED by design (joint DEF 0.381, max 7.754) |

### extra_dimensions — verdict **RESOLVABLE** (BEST = S13)

sigma = 0.1031; ranker: 1 design(s) tied at the top score; 'S13' fully identifies all 2 parameters (rank 2/2, cond 5.81e+00); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| ED-1 | MECHANISM | INDETERMINATE (solo variant: INDETERMINATE) | crossover: INDETERMINATE [joint DEF 0.187, BEST 0.640, max 0.640, 0/64 >1; solo DEF 2.87e-04, solo BEST 0.026, DEF+BEST 0.151]<br>yukawa: DISTINGUISHABLE-BY-DESIGN [joint DEF 0.336, BEST 1.212, max 1.212, 1/64 >1; solo DEF 4.99e-04, solo BEST 0.138, DEF+BEST 0.732]<br>pow: INDETERMINATE [joint DEF 0.293, BEST 0.925, max 0.925, 0/64 >1; solo DEF 3.65e-04, solo BEST 0.057, DEF+BEST 0.309]<br>laplace: DISTINGUISHABLE-BY-DESIGN [joint DEF 0.243, BEST 1.318, max 1.318, 2/64 >1; solo DEF 0.007, solo BEST 0.862, DEF+BEST 1.093]<br>inv2: DISTINGUISHABLE [joint DEF 1.121, BEST 2.725, max 7.492, 18/64 >1; solo DEF 0.100, solo BEST 1.734, DEF+BEST 2.007] |
| ED-2 | MECHANISM | INDETERMINATE (solo variant: INDETERMINATE) | crossover: INDETERMINATE [joint DEF 0.187, BEST 0.640, max 0.640, 0/64 >1; solo DEF 2.87e-04, solo BEST 0.026, DEF+BEST 0.151]<br>yukawa: DISTINGUISHABLE-BY-DESIGN [joint DEF 0.336, BEST 1.212, max 1.212, 1/64 >1; solo DEF 4.99e-04, solo BEST 0.138, DEF+BEST 0.732]<br>pow: INDETERMINATE [joint DEF 0.293, BEST 0.925, max 0.925, 0/64 >1; solo DEF 3.65e-04, solo BEST 0.057, DEF+BEST 0.309]<br>laplace: DISTINGUISHABLE-BY-DESIGN [joint DEF 0.243, BEST 1.318, max 1.318, 2/64 >1; solo DEF 0.007, solo BEST 0.862, DEF+BEST 1.093]<br>inv2: DISTINGUISHABLE [joint DEF 1.121, BEST 2.725, max 7.492, 18/64 >1; solo DEF 0.100, solo BEST 1.734, DEF+BEST 2.007] |
| ED-3 | PARAMETER | IDENTIFIABLE only at BEST | SE DEFAULT 1.188 (1e-3: 1.188); SE BEST 0.026 (1e-3: 0.026); tau 0.300 (half-log 0.458); 5/65 designs identify; info-optimal S62 SE 0.011 |
| ED-4 | PARAMETER | IDENTIFIABLE at DEFAULT | SE DEFAULT 0.090 (1e-3: 0.090); SE BEST 0.014 (1e-3: 0.014); tau 0.150 (half-log 0.131); 22/65 designs identify; info-optimal S62 SE 0.003 |
| ED-5 | UNTYPED | — | no lower band of the extra_dimensions rubric names a p1/p2 alternative |

### fractional — verdict **NONID** (BEST = S13)

sigma = 0.1196; ranker: 1 design(s) tied at the top score; 'S13' fully identifies all 2 parameters (rank 2/2, cond 6.58e+00); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| FR-1 | NOT_SCORED | — | named alternative is a time-evolving / wave-like field; the pinned nbody engine has only instantaneous pairwise forces, so it cannot be simulated here |
| FR-2 | MECHANISM | EQUIVALENT (certified) (solo variant: EQUIVALENT (certified)) | laplace: DISTINGUISHABLE-BY-DESIGN [joint DEF 0.220, BEST 1.441, max 2.750, 3/64 >1; solo DEF 0.049, solo BEST 1.041, DEF+BEST 1.208]<br>pow: EQUIVALENT (certified) [joint DEF 1.24e-14, BEST 0.000, max 1.24e-14, 0/64 >1; solo DEF 1.24e-14, solo BEST 0.000, DEF+BEST 8.14e-15] |
| FR-3 | PARAMETER | IDENTIFIABLE only at BEST | SE DEFAULT 0.438 (1e-3: 0.438); SE BEST 0.032 (1e-3: 0.032); tau 0.100 (half-log 0.100); 3/65 designs identify; info-optimal S62 SE 0.010 |
| FR-4 | MECHANISM | EQUIVALENT (numeric) (solo variant: EQUIVALENT (numeric)) | pow_fast: EQUIVALENT (numeric) [joint DEF 1.24e-14, BEST 0.000, max 1.24e-14, 0/64 >1; solo DEF 1.24e-14, solo BEST 0.000, DEF+BEST 8.14e-15]; top-band claim's own family (decay no faster than 1/r): CONTRADICTED by design (joint DEF 0.220, max 2.750) |
| FR-5 | UNTYPED | — | no lower band of the fractional rubric names a p1/p2 alternative |

### dark_matter — verdict **ID** (BEST = S26)

sigma = 0.3978; ranker: 1 design(s) tied at the top score; 'S26' fully identifies all 2 parameters (rank 2/2, cond 2.13e+00); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| DM-1 | UNTYPED | — | lower band names only a generic 'wrong operator family' |
| DM-2 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE) | no_hidden: DISTINGUISHABLE [joint DEF 6.077, BEST 11.369, max 21.522, 64/64 >1; solo DEF 4.878, solo BEST 9.834, DEF+BEST 10.643] |
| DM-3 | MECHANISM (count vs coupling) | DISTINGUISHABLE | {"N=5": "DISTINGUISHABLE", "N=20": "DISTINGUISHABLE"}; residuals: {"5": {"solo_DEFAULT": 1.8255730258713867, "solo_S26": 0.8516777456780354, "solo_S33": 0.810010848256403, "solo_S21": 0.9188242495635731, "solo_S13": 0.8555073186285809, "joint_chosen": [1.9730945826917572, 1.1407729867612246, 1.3519059733100276, 1.1152646686389482, 1.443869838217114]}, "20": {"solo_DEFAULT": 1.073821138949178}} |
| DM-3b | PARAMETER | IDENTIFIABLE at DEFAULT | SE DEFAULT 5.61e-05 (1e-3: 3.64e-04); SE BEST 4.65e-05 (1e-3: 2.24e-04); tau 0.200 (half-log 0.235); 65/65 designs identify; info-optimal S53 SE 2.30e-05 |
| DM-4 | UNTYPED | — | lower band only says 'fails to characterise the probes as neutral' (an omission) |

### coulomb — verdict **ID** (BEST = DEFAULT)

sigma = 0.1030; ranker: 65 design(s) tied at the top score; 'DEFAULT' fully identifies all 1 parameters (rank 1/1, cond 1.00e+00); run it to disambiguate.

| claim | type | class | numbers |
|---|---|---|---|
| CO-1 | NOT_SCORED | — | named alternative (time-evolving, wave-like) ave-like field; the pinned nbody engine has only instantaneous pairwise forces, so it cannot be simulated here |
| CO-2 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE-BY-DESIGN) | inv1: DISTINGUISHABLE [joint DEF 2.070, BEST 2.070, max 141.102, 57/64 >1; solo DEF 0.023, solo BEST 0.023, DEF+BEST 0.023]<br>exp: DISTINGUISHABLE [joint DEF 3.293, BEST 3.293, max 79.103, 58/64 >1; solo DEF 2.13e-04, solo BEST 2.13e-04, DEF+BEST 2.13e-04]<br>const: DISTINGUISHABLE [joint DEF 3.390, BEST 3.390, max 173.556, 57/64 >1; solo DEF 0.046, solo BEST 0.046, DEF+BEST 0.046] |
| CO-3 | MECHANISM | DISTINGUISHABLE (solo variant: DISTINGUISHABLE-BY-DESIGN) | ratio_12: DISTINGUISHABLE [joint DEF 11.254, BEST 11.254, max 147.899, 56/64 >1; solo DEF 0.000, solo BEST 0.000, DEF+BEST 0.000]<br>ratio_21: DISTINGUISHABLE [joint DEF 10.931, BEST 10.931, max 185.491, 57/64 >1; solo DEF 0.000, solo BEST 0.000, DEF+BEST 0.000]<br>additive: DISTINGUISHABLE [joint DEF 10.783, BEST 10.783, max 75.753, 59/64 >1; solo DEF 1.44e-14, solo BEST 1.44e-14, DEF+BEST 1.44e-14] |
| CO-4 | UNTYPED | — | no lower band names an alternative |
| CO-5 | PARAMETER | IDENTIFIABLE at DEFAULT | SE DEFAULT 0.013 (1e-3: 0.013); SE BEST 0.013 (1e-3: 0.013); tau 0.250 (half-log 0.347); 63/65 designs identify; info-optimal S13 SE 1.73e-06 |

## Predictions

**P-EXP-263-1 (verdicts): FALSIFIED** — 7/9 match (pass >= 9, falsified < 8). Dark matter (not predicted): ID.

| world | predicted | computed | match | solo-DEFAULT variant |
|---|---|---|---|---|
| fractional | NONID | NONID | yes | NONID |
| circle | NONID | NONID | yes | NONID |
| extra_dimensions | RESOLVABLE | RESOLVABLE | yes | RESOLVABLE |
| gravity | ID | RESOLVABLE | NO | RESOLVABLE |
| yukawa | ID | RESOLVABLE | NO | RESOLVABLE |
| hubble | ID | ID | yes | ID |
| ether | ID | ID | yes | ID |
| oscillator | ID | ID | yes | RESOLVABLE |
| coulomb | ID | ID | yes | RESOLVABLE |

**P-EXP-263-2 (ceilings, consistency check only):** NONID∪RESOLVABLE maxima {"gravity": 1.0, "yukawa": 1.0, "circle": 0.7, "extra_dimensions": 0.54, "fractional": 0.48} vs ID maxima {"hubble": 1.0, "ether": 0.9, "oscillator": 0.92, "dark_matter": 0.56, "coulomb": 0.98}; Mann–Whitney U = 10.0, one-sided p = 0.3362; perfect separation of NONID below every ID world: False; condition 'verdicts as predicted' met: False.

**P-EXP-263-3 (Proposition 3 signature): CONFIRMED** — certified-NONID worlds ['circle', 'fractional'] vs ID worlds ['hubble', 'ether', 'oscillator', 'dark_matter', 'coulomb']; 2x2 [[TG&EF, TG&not-EF] certified; [.., ..] ID] = [[15, 0], [5, 4]]; fractions 1.000 vs 0.556; Fisher one-sided p = 0.01186.

**P-EXP-263-4 (ranker finds the resolving design): FALSIFIED**

- gravity: BEST = DEFAULT; design-dependent claims ['GR-2']; resolved at BEST {"GR-2": false}
- yukawa: BEST = S13; design-dependent claims ['YU-3', 'YU-4']; resolved at BEST {"YU-3": true, "YU-4": true}
- extra_dimensions: BEST = S13; design-dependent claims ['ED-3']; resolved at BEST {"ED-3": true}; min probe separation at BEST 1.787 (DEFAULT 3.015); r <= 1.5: False

## Symbolic certificate (sympy)

```
{
 "riesz_to_power(n=3-2a, A=G c_a (2-2a))": "0",
 "power_to_riesz(a=(3-n)/2)": "0",
 "certified": true,
 "domain": "alpha in (0,1) <-> n in (1,3); the same map applies to the softened radius r_eff = sqrt(r^2+0.05^2) because both kernels are evaluated on r_eff by make_acceleration_fn",
 "alpha=1/2": {
  "n": "2",
  "A_over_G": "1/(2*pi)",
  "A_over_G_float": 0.15915494309189535
 },
 "alpha=3/4": {
  "n": "3/2",
  "A_over_G": "sqrt(2)*gamma(1/4)/(8*pi*gamma(3/4))",
  "A_over_G_float": 0.16648396775085014
 },
 "numeric_max_rel_diff_alpha=0.5": 2.2162825087868426e-16,
 "numeric_max_rel_diff_alpha=0.75": 4.2032651937308854e-16
}
```

## Parity and finite-difference stability

| world | runner vs shipped executor (DEFAULT) | fast path vs executor (65 designs), max abs | / sigma | FD(1e-4) vs AD, median rel | max rel |
|---|---|---|---|---|---|
| gravity | 0.000 | 0.000 | 0.000 | 2.23e-09 | 4.92e-08 |
| yukawa | 0.000 | 1.78e-15 | 1.49e-14 | 5.31e-08 | 8.95e-05 |
| hubble | 0.000 | 4.57e-12 | 1.42e-11 | 1.45e-09 | 9.087 |
| ether | 0.000 | 5.07e-12 | 2.20e-11 | 1.20e-09 | 0.027 |
| oscillator | 0.000 | 0.000 | 0.000 | 1.35e-07 | 6.11e-07 |
| circle | 0.000 | 3.40e-12 | 2.65e-11 | 1.19e-09 | 1.15e-07 |
| extra_dimensions | 0.000 | 1.07e-14 | 1.03e-13 | 6.51e-09 | 2.68e-07 |
| fractional | 0.000 | 0.000 | 0.000 | 2.38e-08 | 7.20e-07 |
| dark_matter | 0.000 | 1.78e-06 | 4.48e-06 | 1.000 | 1.015 |
| coulomb | 0.000 | 0.000 | 0.000 | 7.54e-10 | 1.000 |

## Dark matter: hidden count vs per-particle coupling (total dark source 50)

Designs fitted: ['DEFAULT', 'S26', 'S33', 'S21', 'S13'] (DEFAULT, BEST, and the three sampled designs whose probes pass closest to the dark centroid). Min probe-to-dark-centroid distance: {"DEFAULT": 0.098, "S26": 0.217, "S33": 0.01, "S21": 0.016, "S13": 0.022}; median over sampled designs 0.240; dark cluster radius at t=0 2.288, max over the DEFAULT run 2.872. Growth of a 1e-9 perturbation of the dark positions (max probe/visible displacement / 1e-9): {"DEFAULT": 1980000.0, "S26": 10300000.0, "S33": 4680000.0, "S21": 858000.0, "S13": 349000.0}. Compute 3617 s. Full residuals in `dm_count_coupling.tsv`.

## Implementation choices (literal readings of ambiguous points)

1. Claim typing (fixed before any computation): PARAMETER = numeric constant with a stated tolerance; MECHANISM = top-band structure for which some lower band names a specific implementable alternative; every named alternative is fitted and the claim takes the LEAST distinguishable class. Claims whose lower-band counterpart is only an omission ('omits', 'fails to characterise') or a generic 'wrong operator family' are UNTYPED; 'static vs time-evolving/wave-like' claims are NOT SCORED (the pinned nbody engine cannot simulate a time-evolving field). Neither enters the verdict.
2. p1/p2 role claims are MECHANISM claims where the rubric's lower band names 'muddles the p1/p2 roles' (gravity, yukawa, oscillator: implemented as a role swap; gravity 4–6 'fails to distinguish' as a product; coulomb 7–9 as p1/p2, p2/p1 and additive). The fractional and extra_dimensions rubrics name no p1/p2 alternative, so there the role claim is UNTYPED.
3. fractional/circle operator claim: the alternatives are the rubric's 'ordinary power-law' / 'anomalous-decay operator' (A r^-n, both free) and 'standard Laplacian' (A/r). The 'direction of the anomaly' claims are tested twice: the named lower-band alternative (opposite direction, n in [1,4]) and the top-band claim's own family (n in [0,1]); a claim refuted by the true world is flagged CONTRADICTED (outside the registered rule; it does not change the verdict).
4. DEFAULT = the world's own `experiment_format` where the WORLDS entry defines one (coulomb, circle, dark_matter, three_species, ether, hubble); otherwise evaluator `_DEFAULT_TEST_CASES[0]` (p1 = p2 = 1, pos2 = [3,0], v2 = [0,0.5], times 1..10) for gravity, yukawa, fractional, oscillator (start_time 0) and extra_dimensions. (get_world() shows those agents a generic example with p1 = p2 = 1, pos2 = [3,0], times [0.5,1,2]; the registered text names test case 1.)
5. Design sampling (seed 263, numpy default_rng, one fresh stream per world, uniform): 2-particle p1, p2 U[0.1,10], pos2 U[-10,10]^2, v2 U[-5,5]^2, duration U[5,10], oscillator start_time U[0,10] (range not documented; >= 2.5 periods); probes (dark_matter, three_species) positions U[-15,15]^2, velocities U[-2,2]^2; ether/hubble use their own documented interface (probes_with_masses: U[-22,22]^2, U[-3,3]^2, masses drawn from {1,2,4}); circle ring_radius U[2,10], v_tang U[0,2]. Probe/circle durations fixed at 10 (the documented minimum). Always 10 evenly spaced measurement times ending at the duration.
6. World constants patched on the constructed executor (operator params, class attributes, a subclass for extra_dimensions). Background initial conditions computed in __init__ from the constants (hubble ring speeds, dark-matter visible speeds) are held at their true values: they are observed inputs, not model outputs.
7. Sensitivities: all agent-visible positions (2-particle: pos1 and pos2; dark_matter: the 25 agent indices; others: all particles) at every measurement time; velocities are also returned to agents but the registration names positions only. Relative central-difference step 1e-4 (registered) and 1e-3 (stability); for phi = 0 the steps are absolute (1e-4, 1e-3 rad). FIM = S^T S / sigma^2; CRLB SE = sqrt(g^T FIM^-1 g) / sqrt(16) with all world constants as joint unknowns. For three_species and dark_matter the Laplacian strength is not a separate constant (it is exactly confounded with the couplings).
8. Tolerance threshold tau: the registered example (x2 -> 0.25) is reproduced by tau = 0.5*min(1 - lo/t, hi/t - 1) for multiplicative ranges (SE in log units) and tau = half the distance to the nearer bound for additive alpha ranges. The alternative 'half of log tolerance' reading (0.347 for x2) is reported alongside; verdicts use tau.
9. Ranker = ascension.diagnostics.rank_designs_by_identifiability on per-design sensitivity matrices whitened by sigma and scaled to log-parameters (x theta_j; phi unscaled), DEFAULT first then S00..S63. Its score is rank + 1/(1+log10 cond); ties keep insertion order, so for one-constant worlds (gravity, coulomb) every design ties and BEST = DEFAULT.
10. MECHANISM fits (registered): ONE least-squares fit of the alternative's constants jointly over DEFAULT + 64 designs to the true world's noiseless executor predictions; per-design residual = RMS over all visible coordinates / sigma. EQUIVALENT (numeric) if every design < 0.01; DISTINGUISHABLE if DEFAULT > 1; DISTINGUISHABLE-BY-DESIGN if DEFAULT <= 1 and some sampled design > 1; otherwise INDETERMINATE (a gap in the registered classes). Because a joint fit can mispredict DEFAULT merely through the compromise with other designs, each alternative is ALSO refitted to DEFAULT alone, BEST alone and DEFAULT+BEST; the 'solo-DEFAULT variant' verdict (DISTINGUISHABLE only if DEFAULT alone refutes the alternative) is reported as a sensitivity analysis, not as the registered verdict.
11. Alternatives are simulated on the fast path (benchmark integrator, acceleration assembly and kernels with traced constants; exact Jacobians by forward-mode AD); multi-start trf least squares (<= 200 evaluations per start). Yukawa-family alternatives use a JAX K1 quadrature (max rel. error 3.6e-15 vs scipy) because the benchmark kernel's scipy callback has no JVP.
12. three_species 'slightly wrong partition': all 60 single-particle reassignments were scanned at the true couplings; the least distinguishable one (lowest overall RMS) was refitted with free couplings.
13. dark_matter count vs coupling: N in {5, 20} dark particles with coupling 50/N and zero initial velocity, 2N initial positions fitted (starts: random subset (N=5) or duplication (N=20) of the true positions + N(0, 0.1) jitter, 3 restarts, seed 263, <= 60 evaluations each) per design for DEFAULT, BEST and the three sampled designs whose probes pass closest to the dark centroid, plus one joint fit over those five. Joint fitting over all 65 designs was not attempted (40 chaotic nuisance parameters); the class uses per-design fits, which favour the alternative. DM-3b (coupling ratio) is a conditional CRLB with count and positions at truth.
14. Execution note (no method change): the first pooled run (4 worlds per process, sequential alternatives) failed for 6 light worlds with an XLA runtime error ('Failed to materialize symbols') in a worker that had accumulated thousands of compiled executables, and a later OOM kill broke the pool, losing the in-progress ether/dark_matter/three_species runs (hubble and yukawa had completed and were kept). All other worlds were re-run from scratch in fresh processes with jax.clear_caches() between executor batches and with a world's alternative fits distributed over worker processes (three_species 3, ether 2). Designs, seeds, starts, tolerances and budgets were unchanged; the rerun reproduced the first run's ranker BEST designs. First-run tracebacks are kept in cache/failed_run1/.
15. Execution note 2 (harness only): a container restart at ~01:30 UTC 2026-09-29 killed the in-flight ether/three_species/dark_matter reruns. They were re-run from scratch with the same method, seeds, starts, tolerances and the same 3600 s dark-matter count-fit budget (persisted across restarts), under a checkpointing driver (`--stage resume`): the base stage of each world (truth predictions, parity, FD, ranker, CRLBs) and every individual alternative fit (joint / solo DEFAULT / solo BEST / DEFAULT+BEST, and each dark-matter count fit) is written to cache/ckpt/ as soon as it finishes and skipped on restart. Parallelism: 4 long-lived worker processes, each pinned to one core, with BLAS/OpenMP and XLA intra-op threads set to 1 (the earlier load of ~16 on 4 cores came from oversubscribed threads).
16. Execution note 3 (harness only): on the second driver, pinning its two workers to cores 2-3 (05:13 UTC) starved the two first-driver workers already pinned there (runnable, ~0 CPU) until all workers were unpinned at 11:40 UTC; about 6 h of wall time lost, no effect on results. A second container restart (~16:00 UTC) killed both drivers; the run resumed from checkpoints with non-three_species fits first. From then on each multi-start of a fit was also checkpointed as soon as that start finished (x, cost, nfev, status); a restarted fit skips finished starts and re-evaluates the residual at the best start's saved x. Warm-restarting a start mid-way was NOT done: scipy's trf keeps internal state (trust radius, jac-based x_scale, evaluation count against max_nfev) that a restart from saved parameters would reset, which would change the registered optimizer behaviour.
17. Execution note 4: after the second restart the orchestrator briefly launched a duplicate driver (pid 637, ~16:31–16:32 UTC). Stopping it left its 4 pool workers orphaned; they were killed at ~16:34 UTC. That driver re-read cached stages and completed exactly one fit, ether ET-3 mass_dep_drift solo_default (checkpoint 16:31:59 UTC). It was produced by the same code (deterministic), parses cleanly and was kept; no other checkpoint came from it.
18. Execution note 5: a third container restart (~23:50 UTC 2026-09-29, the coordinator's count: fourth) killed the drivers again; per-start checkpoints (ether repulsor joint start 0, three_species nonneg joint start 0) survived. An orchestrator relaunch as a background shell command was killed by the harness time limit before any fit completed (nothing written). The agent then relaunched a single detached launcher (scripts run via setsid/nohup: ether with 1 worker followed by the provisional 10-world aggregate, three_species with 3 workers, then the final aggregate), skipping every checkpointed fit and start.
19. Scored cells for P-EXP-263-3: the 13 per-(model, world) means published on the leaderboard (not per-seed values). Mann–Whitney: scipy mannwhitneyu, alternative 'less'; Fisher: scipy fisher_exact, alternative 'greater'.

## Limitations

1. Local (linearised) CRLBs at the true constants; no global identifiability analysis for PARAMETER claims.
2. Static-vs-time-evolving claims are not scored: the pinned nbody engine cannot represent diffusion/wave fields and the FFT field engine was not used (engine difference; the leaderboard ran nbody).
3. Alternatives the rubric names but that are not deterministic force laws (noise, drag, 'wind', a directional Laplacian) were not implemented.
4. Least-squares fits can miss the global optimum; a missed optimum biases toward DISTINGUISHABLE. Multi-starts reduce but do not remove this, most of all for the chaotic dark-matter fits.
5. Close encounters (softening 0.05) in random designs make trajectories stiff; FD derivatives and residuals on those designs are dominated by near-singular dynamics. FD-vs-AD agreement is reported per world as a check on the registered finite differences.
6. The leaderboard reports per-cell means over 5 seeds; trajectory-good/explanation-fail are applied to those means, not to individual runs.
7. Private worlds were out of scope; none of the public-code worlds outside the 11 was scored.

## Files

- `leaderboard_cells.tsv` — per-(model, world) outcomes copied first from the pinned leaderboard.
- `claims.tsv` — every top-band claim, verbatim, typed, with named alternatives and tolerances.
- `crlb.tsv` — per-design CRLB SE for every PARAMETER claim at both FD steps.
- `mechanism_residuals.tsv` — per-design residuals for every alternative and fit type.
- `dm_count_coupling.tsv`, `parity.tsv`, `world_verdicts.tsv`, `summary.json`, `cache/world_*.json`.
