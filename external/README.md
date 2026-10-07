# External benchmarks (not vendored)

The EXP-261, EXP-262, and EXP-263 scripts read two upstream benchmarks from this
directory (or from the environment variables named below). Clone them at the
pinned commits the paper used; nothing from either repository is copied here.

```bash
cd external
git clone https://github.com/scientific-discovery/LLM-ACES.git && git -C LLM-ACES checkout 60d4df7
git clone https://github.com/SampsonML/DiscoverPhysics.git && git -C DiscoverPhysics checkout 33b7fa9
git clone https://github.com/SampsonML/DiscoverPhysicsLeaderboard.git && git -C DiscoverPhysicsLeaderboard checkout 8e9c858
```

| Benchmark | Commit | Used by | Override |
|---|---|---|---|
| LLM-ACES (ships its copy of ODEBench) | `60d4df7` | `scripts/sweep_odebench_identifiability.py`, `scripts/exp262_head_to_head.py` | `LLM_ACES_DIR` |
| DiscoverPhysics | `33b7fa9` | `scripts/exp263_discoverphysics_audit.py` | `DISCOVERPHYSICS_DIR` |
| DiscoverPhysicsLeaderboard | `8e9c858` | `scripts/exp263_discoverphysics_audit.py` | `DISCOVERPHYSICS_LEADERBOARD_DIR` |

The EXP-263 script also needs DiscoverPhysics's own dependencies (JAX among them);
install them from that repository's instructions.
