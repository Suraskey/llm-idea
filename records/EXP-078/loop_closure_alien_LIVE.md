<!-- LIVE EXP-078; 5/5 live director decisions; family=3 new_IC=2; total_cost_usd=0.0936 -->


# Identifiability-loop closure — LIVE Council move=request_multi_charge_family (n=3/5)

_diagnose → design-the-experiment → re-diagnose. No LLM judges the trace._

**Round 0** — rank 1/2 → **NON-IDENTIFIABLE (structurally_non_identifiable)**
  - proposed experiments: council→single_product (status quo), council→vary_family_2_distinct, council→vary_family_4_distinct
  - **OED selected:** `council→vary_family_2_distinct` — top-ranked by the rank-restoration criterion (whether it actually restored rank is shown by the re-diagnosis below)

**Round 1** — rank 2/2 → **IDENTIFIABLE**
  - re-diagnosed: full rank — the previously-confounded structure is now observable

**OUTCOME: SOLVED** (1 round(s))

> Identifiability RESTORED after 1 design-experiment round(s): the chosen experiment(s) made the previously-confounded structure observable. The data now determines it.


# Identifiability-loop closure — LIVE Council move=request_new_initial_condition (n=2/5)

_diagnose → design-the-experiment → re-diagnose. No LLM judges the trace._

**Round 0** — rank 1/2 → **NON-IDENTIFIABLE (structurally_non_identifiable)**
  - proposed experiments: council→new_initial_condition (still single product)
  - **OED selected:** `council→new_initial_condition (still single product)` — top-ranked by the rank-restoration criterion (whether it actually restored rank is shown by the re-diagnosis below)

**Round 1** — rank 1/2 → **NON-IDENTIFIABLE (structurally_non_identifiable)**
  - proposed experiments: council→new_initial_condition (still single product)
  - **OED selected:** `council→new_initial_condition (still single product)` — top-ranked by the rank-restoration criterion (whether it actually restored rank is shown by the re-diagnosis below)

**Round 2** — rank 1/2 → **NON-IDENTIFIABLE (structurally_non_identifiable)**
  - proposed experiments: council→new_initial_condition (still single product)
  - **OED selected:** `council→new_initial_condition (still single product)` — top-ranked by the rank-restoration criterion (whether it actually restored rank is shown by the re-diagnosis below)

**Round 3** — rank 1/2 → **NON-IDENTIFIABLE (structurally_non_identifiable)**

**OUTCOME: EXHAUSTED** (3 round(s))

> Still non-identifiable after 3 round(s) (structurally_non_identifiable): no proposed experiment broke the degeneracy within budget. Report the identifiable combination, not its factors, or widen the design space.
