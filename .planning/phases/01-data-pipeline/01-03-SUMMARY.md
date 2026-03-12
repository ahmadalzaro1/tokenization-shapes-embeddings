---
phase: 01-data-pipeline
plan: 03
subsystem: testing
tags: [validation, parquet, pyarrow, arabic, tdd, json]

# Dependency graph
requires:
  - phase: 01-data-pipeline plan 01
    provides: tests/test_pipeline.py, pyproject.toml, conftest.py, atomic encoding infrastructure
  - phase: 01-data-pipeline plan 02
    provides: build_dataset.py with process_condition(), metadata.txt format, collision stats

provides:
  - validate_condition() function in build_dataset.py — runs 4 mandatory checks per condition
  - write_validation_report() function in build_dataset.py — merges results into validation_report.json
  - _compute_char_distribution() helper — top-20 char frequency per condition (visual)
  - validate_dataset.py — standalone re-validator CLI with --condition flag
  - build_dataset.py main() now calls validation inline after each process_condition()
  - validation_report.json schema: {d1: {shards_loadable, row_count_matches, no_empty_texts, char_distribution}, ...}

affects:
  - Phase 2 (training) — consumes validation_report.json to confirm data integrity before tokenizer/train runs
  - Phase 4 (paper) — validation data serves as pipeline correctness evidence

# Tech tracking
tech-stack:
  added: []
  patterns:
    - TDD red-green cycle for each task (test commit then impl commit)
    - sys.exit(1) as hard-fail signal from validate_condition(); caught by validate_dataset.py for complete reporting
    - write_validation_report merges (not overwrites) — safe to call per-condition without losing other conditions
    - D3 integrity gate: 252-entry threshold enforced via sys.exit(1) before Phase 2 can proceed

key-files:
  created:
    - validate_dataset.py — standalone re-validator CLI; imports from build_dataset
  modified:
    - build_dataset.py — added import sys, validate_condition(), write_validation_report(), _compute_char_distribution(); updated main() loop
    - tests/test_pipeline.py — added import pyarrow as pa; 11 new tests (6 for Task 1, 5 for Task 2)

key-decisions:
  - "sys.exit(1) as hard-fail from validate_condition() — validate_dataset.py catches SystemExit to continue multi-condition reporting before final exit"
  - "write_validation_report merges via read-update-write pattern — safe for incremental per-condition writes in main() loop"
  - "char_distribution is visual-only (no hard-fail) — prevents blocking pipeline on cosmetic issues while still surfacing encoding problems"
  - "D3 gate threshold 252 entries — covers 26 letters x 9 harakat + shaddah combos + standalone harakat"

patterns-established:
  - "Validation pattern: validate_condition() + write_validation_report() called after each process_condition() in main()"
  - "Standalone re-validator imports from primary script — no logic duplication"
  - "SystemExit catch pattern in CLI loop — allows complete multi-item failure reporting before final exit"

requirements-completed: [DATA-02, DATA-03, DATA-04]

# Metrics
duration: 5min
completed: 2026-03-12
---

# Phase 1 Plan 3: Validation Suite Summary

**Inline validation checks (shards loadable, row count, no empties, char dist) + D3 PUA integrity gate added to build_dataset.py; validate_dataset.py CLI re-validates all conditions from cache**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-11T23:17:57Z
- **Completed:** 2026-03-11T23:22:29Z
- **Tasks:** 3 of 3 complete (human-verify checkpoint approved)
- **Files modified:** 3

## Accomplishments

- Added `validate_condition()` with 4 mandatory checks: shards loadable, row count matches metadata, no empty texts, character distribution (visual)
- Added D3 integrity gate: `atomic_mapping.json` must have >= 252 entries or pipeline exits
- Added `write_validation_report()` with merge-safe read-update-write pattern
- Created `validate_dataset.py` standalone CLI that imports from `build_dataset` (no duplication) and supports `--condition {d1,d2,d3,all}`
- Updated `build_dataset.py` `main()` to call validation inline after each `process_condition()` call
- Added 11 new unit tests in TDD fashion (6 for Task 1, 5 for Task 2); all 25 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Add failing tests for validate_condition** - `8b1c619` (test)
2. **Task 1 GREEN: Add validate_condition and write_validation_report** - `8df0799` (feat)
3. **Task 2 RED: Add failing tests for validate_dataset.py** - `b94352b` (test)
4. **Task 2 GREEN: Create validate_dataset.py standalone validator** - `65131a3` (feat)

_Note: TDD tasks have separate test and implementation commits_

## Files Created/Modified

- `/Users/ahmadalzaro/Desktop/Karpathy/autoresearch-arabic/build_dataset.py` — added `import sys`, `_compute_char_distribution()`, `validate_condition()`, `write_validation_report()`; updated `main()` conditions loop
- `/Users/ahmadalzaro/Desktop/Karpathy/autoresearch-arabic/validate_dataset.py` — new standalone CLI validator
- `/Users/ahmadalzaro/Desktop/Karpathy/autoresearch-arabic/tests/test_pipeline.py` — added `import pyarrow as pa`; 11 new unit tests

## Decisions Made

- `sys.exit(1)` as hard-fail from `validate_condition()`: consistent with shell conventions, catchable via `except SystemExit` in `validate_dataset.py` to continue multi-condition reporting
- `write_validation_report` uses read-update-write merge: calling it per-condition in a loop does not erase previous conditions
- `char_distribution` is visual-only: prevents blocking on non-critical encoding style differences while preserving signal for human review
- D3 gate threshold 252: covers the full combination space (letters × harakat variants + standalone harakat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing `import pyarrow as pa` in test file**
- **Found during:** Task 1 (TDD GREEN — first test run after implementation)
- **Issue:** New test `test_validate_condition_returns_required_keys` used `pa.table()` but `test_pipeline.py` only imported `pyarrow.parquet as pq`, not `pyarrow as pa`
- **Fix:** Added `import pyarrow as pa` to the import block
- **Files modified:** `tests/test_pipeline.py`
- **Verification:** `uv run pytest tests/test_pipeline.py -q` passes (20 → 25 tests)
- **Committed in:** `8df0799` (Task 1 GREEN commit)

**2. [Rule 1 - Bug] Fixed metadata fixture row count mismatch in test**
- **Found during:** Task 1 (TDD GREEN — second test run)
- **Issue:** Test fixture wrote 2 rows to `shard_00000.parquet` (the val shard) but declared `train_docs=1, val_docs=1` — `validate_condition()` correctly detected the mismatch and exited 1
- **Fix:** Changed metadata to `train_docs=0, val_docs=2` to match the actual shard contents
- **Files modified:** `tests/test_pipeline.py`
- **Verification:** `test_validate_condition_returns_required_keys` passes
- **Committed in:** `8df0799` (Task 1 GREEN commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — bugs in test fixtures)
**Impact on plan:** Both fixes in test code only; no production code changed. Row-count check working as designed.

## Issues Encountered

None in production code. Two test fixture issues auto-fixed inline (documented above).

## User Setup Required

None — no external service configuration required.

## Human Verification

Checkpoint approved 2026-03-12. User confirmed:
- D1 character distribution shows Arabic combining marks (harakat) in top-20 as expected
- D2 character distribution shows no harakat chars (stripping confirmed correct)
- D3 character distribution shows PUA chars prominently in top-20
- Full 1.46M row dataset built and validated — all mandatory checks PASSED for all 3 conditions
- `validation_report.json` confirmed with d1/d2/d3 keys, all checks `true`

## Next Phase Readiness

- Data pipeline fully validated: 4 mandatory checks + D3 gate enforced before Phase 2
- `validation_report.json` schema is Phase 2-ready: structured per-condition with bool checks
- Standalone `validate_dataset.py` allows re-validation at any time without rebuilding
- All three conditions verified at corpus scale (1.46M rows total)

---
*Phase: 01-data-pipeline*
*Completed: 2026-03-12*
