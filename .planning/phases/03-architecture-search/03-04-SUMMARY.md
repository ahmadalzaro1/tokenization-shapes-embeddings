---
phase: 03-architecture-search
plan: 04
subsystem: training
tags: [mlx, arabic-nlp, architecture-search, autoresearch, d2-condition, control]

# Dependency graph
requires:
  - phase: 03-03
    provides: d1 and d3 entries in search_results.json; established autoresearch loop protocol
provides:
  - results_d2.tsv with 70 experiment rows on autoresearch/arabic-d2 branch
  - d2 entry in search_results.json (best_val_bpb=1.019569, commit=06e2864)
  - Complete Phase 3 comparison across d1, d2, d3
affects:
  - 04 (analysis and paper framing now has all three condition winners)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D2 search followed the same keep/discard loop as D3 and D1"
    - "Best D2 result extracted from branch-local TSV and merged into main search_results.json"

key-files:
  created:
    - results_d2.tsv (on autoresearch/arabic-d2 branch)
  modified:
    - search_results.json (on main)

key-decisions:
  - "D2 best config: DEPTH=4, WINDOW_PATTERN=SSS, HEAD_DIM=128, ASPECT_RATIO=24, TOTAL_BATCH_SIZE=2**15, val_bpb=1.019569"
  - "D2 improved substantially from baseline 1.596882, but remained worse than D3 (0.889682) and D1 (0.660090)"
  - "With all three conditions complete, the empirical story is D1 < D3 < D2, not D3 < D1"

requirements-completed: [SRCH-02]

# Metrics
duration: ~7hr
completed: 2026-03-13
---

# Phase 3 Plan 04: D2 Architecture Search Summary

**70-experiment D2 autoresearch loop completed the three-condition comparison — best D2 val_bpb=1.019569 using DEPTH=4, WINDOW_PATTERN=SSS, HEAD_DIM=128, ASPECT_RATIO=24, TOTAL_BATCH_SIZE=2**15**

## Performance

- **Duration:** ~7 hours
- **Tasks:** 3
- **Experiments run:** 70
- **Keep rows:** 4
- **Crash rows:** 0
- **Discard rows:** 66

## Accomplishments

- Completed the D2 control-condition architecture search on `autoresearch/arabic-d2`
- Produced `results_d2.tsv` with 70 data rows and valid Phase 3 schema
- Found D2 best config at commit `06e2864` with `val_bpb=1.019569`
- Added the D2 entry to `search_results.json` on `main`
- Verified `tests/test_search.py::test_search_d2` passes on `autoresearch/arabic-d2`
- Verified `tests/test_search.py` on `main` reaches the expected final state: `test_search_results_json` passes and the branch-local TSV tests skip

## Best D2 Progression (keep chain)

| Commit | val_bpb | Change |
|--------|---------|--------|
| a16bf04 | 1.200240 | baseline |
| 86f7eba | 1.108200 | halve batch size to `2**15` |
| c942262 | 1.024710 | apply D1 winning config |
| 64fe727 | 1.022476 | `WINDOW_PATTERN="SSS"` |
| 06e2864 | 1.019569 | `ASPECT_RATIO=24` |

## Final Phase 3 Comparison

- `d1`: baseline `1.190999` -> best `0.660090` (`-0.530909`)
- `d3`: baseline `1.075381` -> best `0.889682` (`-0.185699`)
- `d2`: baseline `1.596882` -> best `1.019569` (`-0.577313`)

## Interpretation

- D2 benefits from the same general architecture family that helped D1, but it still underperforms both D1 and D3
- The complete search result is `D1 < D3 < D2` on best `val_bpb`
- This strengthens the claim that stripping harakat imposes an ambiguity tax; D2 remains the hardest representation to model even after architecture search

---
*Phase: 03-architecture-search*
*Completed: 2026-03-13*
