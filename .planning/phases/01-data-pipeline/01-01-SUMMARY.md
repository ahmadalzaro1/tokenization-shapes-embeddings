---
phase: 01-data-pipeline
plan: 01
subsystem: testing
tags: [pytest, pyarrow, parquet, arabic, harakat, pua-encoding, fixtures]

# Dependency graph
requires: []
provides:
  - pytest dev dependency installed via uv (pytest>=9.0.2)
  - tests/__init__.py package marker
  - tests/conftest.py with 100-row diacritized Arabic fixtures and d1/d2/d3 parquet shard fixtures
  - tests/test_pipeline.py with 6 stub tests covering DATA-01 through DATA-05 + validation report
affects:
  - 01-02 (build_dataset.py — makes test_d1_shards/test_d2_stripped/test_d3_pua green with real cache)
  - 01-03 (validate_dataset.py — makes test_validation_report green)
  - 01-04 (collision analysis — makes test_collision_stats_json green)

# Tech tracking
tech-stack:
  added:
    - pytest>=9.0.2 (dev dependency via uv add --dev)
    - pyarrow (already in dependencies, used in fixtures for parquet I/O)
  patterns:
    - TDD stub pattern: fixture-based tests pass immediately; cache-dependent tests skip with reason
    - Inline fixture data (no network): Arabic texts hardcoded in conftest.py, never fetched
    - Parquet shard fixture: _write_shard() helper creates tmp parquet files for each condition

key-files:
  created:
    - tests/__init__.py
    - tests/conftest.py
    - tests/test_pipeline.py
  modified:
    - pyproject.toml (added [dependency-groups] dev section)
    - uv.lock (updated with pytest, iniconfig, pluggy)

key-decisions:
  - "Used uv add --dev pytest (not pip install) to stay consistent with uv-managed venv"
  - "Hardcoded 10 Quran verses x10 as fixture data — guarantees harakat presence without network calls"
  - "d3_shard_path fixture replicates build_atomic_mapping() logic inline — avoids importing build_dataset.py (not yet testable)"
  - "Cache-dependent tests use pytest.mark.skip (not xfail) — they are not expected to fail, just deferred"

patterns-established:
  - "Fixture isolation: all fixtures write to tmp_path, never ~/.cache"
  - "Condition split naming: d1=harakat-present, d2=harakat-stripped, d3=PUA-encoded"
  - "uv run pytest tests/ -q is the canonical test invocation"

requirements-completed: [DATA-01, DATA-02, DATA-03, DATA-04, DATA-05]

# Metrics
duration: 2min
completed: 2026-03-12
---

# Phase 1 Plan 01: Test scaffold with pytest, Arabic fixtures, and six pipeline stubs

**pytest 9.0.2 installed, `tests/` scaffold with 100-row diacritized Arabic parquet fixtures and 6 pipeline stub tests (3 pass via fixtures, 3 skip awaiting Wave 2/3 scripts)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-11T23:07:07Z
- **Completed:** 2026-03-11T23:09:20Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- pytest 9.0.2 added as dev dependency; `uv run pytest --version` exits 0
- conftest.py provides 100 rows of inline diacritized Arabic (Quran verses), plus d1/d2/d3 parquet shard fixtures and cache_base — zero network calls
- test_pipeline.py has exactly 6 tests: test_d1_shards, test_d2_stripped, test_d3_pua pass; test_dataset_columns, test_collision_stats_json, test_validation_report skip cleanly

## Task Commits

Each task was committed atomically:

1. **Task 1: Install pytest and create test package skeleton** - `ede3837` (feat)
2. **Task 2: Create conftest.py with tiny Arabic text fixtures** - `d8087dd` (feat)
3. **Task 3: Create test_pipeline.py with six stub tests** - `29c5f0c` (feat)

## Files Created/Modified

- `tests/__init__.py` - Empty package marker
- `tests/conftest.py` - Shared fixtures: tiny_arabic_texts (100 rows), d1/d2/d3 shard paths, cache_base
- `tests/test_pipeline.py` - 6 stub tests for DATA-01 through DATA-05 + validation report
- `pyproject.toml` - Added [dependency-groups] dev with pytest>=9.0.2
- `uv.lock` - Updated lockfile (added pytest 9.0.2, iniconfig 2.3.0, pluggy 1.6.0)

## Decisions Made

- Used `uv add --dev pytest` (not pip/pip3) so pytest is tracked in uv's lockfile and dev dependency group
- Hardcoded 10 Quran-style diacritized verses repeated 10x as fixture data; guarantees harakat presence without any network calls
- Replicated `build_atomic_mapping()` inline in conftest.py for d3_shard_path fixture to avoid importing the not-yet-testable build_dataset.py
- Chose `pytest.mark.skip` (not `xfail`) for cache-dependent tests since skipping is the correct semantic (they are not expected to fail, just deferred until later plans run)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Test scaffold is complete; all subsequent plans can reference `uv run pytest tests/ -q` as their verification command
- Wave 2 plans (build_dataset.py) should target making test_d1_shards/test_d2_stripped/test_d3_pua green with real HF cache data
- Wave 2 JSON sidecar plan should target making test_collision_stats_json green
- Wave 3 validation plan should target making test_validation_report green
- Known blocker from STATE.md: HuggingFace downloads may stall on current network — may need Kaggle fallback for DATA-01

---
*Phase: 01-data-pipeline*
*Completed: 2026-03-12*

## Self-Check: PASSED

- tests/__init__.py: FOUND
- tests/conftest.py: FOUND
- tests/test_pipeline.py: FOUND
- 01-01-SUMMARY.md: FOUND
- Commit ede3837: FOUND
- Commit d8087dd: FOUND
- Commit 29c5f0c: FOUND
