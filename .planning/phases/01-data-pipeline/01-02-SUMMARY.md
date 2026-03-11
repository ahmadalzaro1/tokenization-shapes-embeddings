---
phase: 01-data-pipeline
plan: 02
subsystem: data-pipeline
tags: [tqdm, threading, arabic, homograph, collision-stats, json, huggingface]

# Dependency graph
requires:
  - phase: 01-data-pipeline
    plan: 01
    provides: "test infrastructure, pyproject.toml with pytest/tqdm deps, base build_dataset.py"
provides:
  - "load_dataset_with_progress() with tqdm heartbeat and MANUAL_INSTRUCTIONS fallback"
  - "MANUAL_INSTRUCTIONS constant for HuggingFace manual download guidance"
  - "context_window_collision_probability() standalone function (128-token window sampler)"
  - "Extended compute_collision_stats() writing JSON sidecar (collision_stats.json)"
  - "collision_stats.json with context_window_ambiguous_pct and top_50_ambiguous"
affects:
  - phase-04
  - any phase reading collision_stats.json for paper tables

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Daemon thread + tqdm heartbeat for long-running downloads with graceful failure"
    - "ensure_ascii=False for Arabic Unicode preservation in JSON sidecar"
    - "top_50 in JSON vs top-20 in txt (machine vs human output split)"
    - "context_window_collision_probability computed after full map build (no partial writes)"

key-files:
  created: []
  modified:
    - build_dataset.py
    - tests/test_pipeline.py

key-decisions:
  - "load_dataset_with_progress uses daemon thread so Ctrl+C always kills the process cleanly"
  - "context_window_collision_probability placed before compute_collision_stats (called inside it)"
  - "json import moved to top-level; threading and tqdm added to top-level imports"
  - "Short docs (< window_tokens words) skipped in sampling — bias documented in docstring"
  - "top_50_ambiguous in JSON (not top_20) to give Phase 4 paper tables richer data"

patterns-established:
  - "TDD for build_dataset.py: monkeypatch BASE_CACHE to tmp_path for isolation"
  - "Thread-based download wrapper: result[]/error[] lists capture daemon thread output"

requirements-completed:
  - DATA-01
  - DATA-05

# Metrics
duration: 3min
completed: 2026-03-12
---

# Phase 1 Plan 2: Download Progress and Context-Window Collision Metric Summary

**tqdm-wrapped HuggingFace download with MANUAL_INSTRUCTIONS fallback, and 128-token context-window collision metric written to collision_stats.json with ensure_ascii=False Arabic Unicode**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-11T23:11:56Z
- **Completed:** 2026-03-11T23:14:53Z
- **Tasks:** 2 (each TDD: RED commit + GREEN commit)
- **Files modified:** 2

## Accomplishments

- `load_dataset_with_progress()` wraps HuggingFace download in a daemon thread with a tqdm spinner bar; prints `MANUAL_INSTRUCTIONS` on KeyboardInterrupt or any thread exception
- `context_window_collision_probability()` samples 10,000 random 128-token windows from the vocalized corpus, returning the fraction of tokens that are homographically ambiguous
- `compute_collision_stats()` extended to call context-window metric after full map build and write `collision_stats.json` with `context_window_ambiguous_pct` and `top_50_ambiguous`; `collision_stats.txt` still written unchanged

## Task Commits

Each task was committed atomically with TDD RED then GREEN:

1. **Task 1 RED: load_dataset_with_progress tests** - `82f3126` (test)
2. **Task 1 GREEN: load_dataset_with_progress implementation** - `ddf249e` (feat)
3. **Task 2 RED: context-window metric + JSON sidecar tests** - `19d7ece` (test)
4. **Task 2 GREEN: context_window_collision_probability + extended compute_collision_stats** - `2e993cc` (feat)

_Note: TDD tasks have multiple commits (test RED → feat GREEN)_

## Files Created/Modified

- `build_dataset.py` - Added `MANUAL_INSTRUCTIONS`, `load_dataset_with_progress()`, `context_window_collision_probability()`, extended `compute_collision_stats()`; moved `json` import to top-level
- `tests/test_pipeline.py` - Added 11 new tests covering load progress, error handling, edge-case probabilities, and JSON sidecar content

## Decisions Made

- Daemon thread for `load_dataset_with_progress`: guarantees process exits cleanly on Ctrl+C without zombie threads blocking
- `context_window_collision_probability` placed as a standalone function above `compute_collision_stats` (called inside it); keeps single-responsibility and makes the function unit-testable in isolation
- Short docs skipped in sampling loop — documented bias in docstring; biases toward longer classical texts (Tashkeela/Shamela) which is the correct representational target
- `ensure_ascii=False` in `json.dump` is CRITICAL for Arabic character integrity; noted as such in a comment

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 1 Plan 3 (validate_dataset.py) can now call `compute_collision_stats()` and find both `.txt` and `.json` outputs written
- Phase 4 can read `~/.cache/autoresearch-arabic/collision_stats.json` for paper tables via `context_window_ambiguous_pct` and `top_50_ambiguous`
- All 14 tests pass (3 skipped — require live HF dataset or full build run)

---
*Phase: 01-data-pipeline*
*Completed: 2026-03-12*

## Self-Check: PASSED

- build_dataset.py: FOUND
- tests/test_pipeline.py: FOUND
- 01-02-SUMMARY.md: FOUND
- 82f3126 (test RED task1): FOUND
- ddf249e (feat GREEN task1): FOUND
- 19d7ece (test RED task2): FOUND
- 2e993cc (feat GREEN task2): FOUND
