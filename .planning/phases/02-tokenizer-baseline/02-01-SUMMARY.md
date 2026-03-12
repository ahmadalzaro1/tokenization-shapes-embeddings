---
phase: 02-tokenizer-baseline
plan: 01
subsystem: testing
tags: [pytest, tdd, nyquist, arabic, tokenizer, baseline, parquet, fixtures]

requires:
  - phase: 01-data-pipeline
    provides: tests/conftest.py with shared fixtures, constants, and _write_shard helper

provides:
  - Nyquist-compliant RED test stubs for all 7 Phase 2 requirements (TOK-01..TOK-04, BASE-01..BASE-03)
  - tiny_corpus_parquet factory fixture for d1/d2/d3 two-shard corpus generation
  - tests/test_tokenizer.py with 2 RED unit tests + 3 integration skips
  - tests/test_baseline.py with 1 RED unit test + 3 integration skips

affects:
  - 02-tokenizer-baseline/02-02 (must make test_vocab_size_flag and test_fertility_report_written go GREEN)
  - 02-tokenizer-baseline/02-03 (must make test_baseline_json_written go GREEN)

tech-stack:
  added: []
  patterns:
    - Factory fixture pattern (tiny_corpus_parquet returns make_corpus callable, not a fixture value)
    - RED stub pattern: unit tests assert on missing symbols/flags; integration tests use @pytest.mark.skip
    - Nyquist rule: every requirement has exactly one named automated test

key-files:
  created:
    - tests/test_tokenizer.py
    - tests/test_baseline.py
  modified:
    - tests/conftest.py

key-decisions:
  - "Factory fixture over pytest-parametrize: tiny_corpus_parquet yields callable make_corpus(condition) so test authors control when/which condition to create, enabling monkeypatching of BASE_CACHE before calling"
  - "Unit RED tests check source text / argparse stdout (not full execution) to stay fast and environment-independent"
  - "_encode_d3 helper extracted from d3_shard_path fixture to avoid duplication in make_corpus"

patterns-established:
  - "Integration skip pattern: @pytest.mark.skip(reason='integration: run after ...') with exact CLI command in reason string"
  - "RED assertion pattern: assert condition, f'Descriptive message. Plan N must fix this.' for self-documenting failures"

requirements-completed: [TOK-01, TOK-02, TOK-03, TOK-04, BASE-01, BASE-02, BASE-03]

duration: 3min
completed: 2026-03-12
---

# Phase 2 Plan 01: Tokenizer Baseline Test Stubs Summary

**Nyquist-compliant RED test stubs for all 7 Phase 2 requirements (TOK-01..TOK-04, BASE-01..BASE-03) plus a d1/d2/d3 factory fixture using two-shard parquet corpus generation**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-12T00:24:06Z
- **Completed:** 2026-03-12T00:26:17Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added `tiny_corpus_parquet` factory fixture to conftest.py — creates two-shard (train + val) parquet corpus for any of d1/d2/d3 conditions, writes metadata.txt with val_shard=1 for prepare.get_val_shard() compatibility
- Created tests/test_tokenizer.py with 5 tests: 2 RED unit stubs (test_vocab_size_flag, test_fertility_report_written) that immediately document what Plan 02 must implement, plus 3 integration skips
- Created tests/test_baseline.py with 4 tests: 1 RED unit stub (test_baseline_json_written) plus 3 integration skips for d2<d1 comparison, d3 plausibility check, and full schema validation

## Task Commits

Each task was committed atomically:

1. **Task 1: tiny_corpus_parquet fixture** - `2b13236` (feat)
2. **Task 2: test_tokenizer.py RED stubs** - `156acc1` (test)
3. **Task 3: test_baseline.py RED stubs** - `f256478` (test)

## Files Created/Modified

- `tests/conftest.py` - Added _encode_d3 helper and tiny_corpus_parquet factory fixture (74 lines added, no existing code modified)
- `tests/test_tokenizer.py` - New: 2 RED unit tests (TOK-01, TOK-02) + 3 integration skips (TOK-03, TOK-04, D3 roundtrip)
- `tests/test_baseline.py` - New: 1 RED unit test (BASE-01) + 3 integration skips (BASE-02, BASE-03, schema)

## Decisions Made

- Factory fixture over pytest-parametrize: `tiny_corpus_parquet` yields a callable `make_corpus(condition)` rather than parametrizing the fixture, so test authors control when and which condition is created and can monkeypatch `prepare.BASE_CACHE` before calling
- Unit RED tests inspect source text and argparse --help output rather than running full prepare.py/train.py — keeps tests fast, environment-independent, and unambiguous about what is missing
- `_encode_d3` helper extracted from the existing `d3_shard_path` fixture logic to avoid duplication in `make_corpus`, maintaining the DRY principle within conftest.py

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None — all 3 tasks completed cleanly on the first attempt.

## Test Results

Final suite (uv run pytest tests/ -q):
- 4 failed: 3 new RED stubs (expected) + 1 pre-existing pipeline test (out of scope)
- 24 passed: all previously passing tests still pass
- 9 skipped: 3 old integration skips + 6 new integration skips

## Next Phase Readiness

- Plan 02 can proceed immediately: test_vocab_size_flag, test_fertility_report_written, and test_baseline_json_written are the three failing targets
- tiny_corpus_parquet fixture is available for Plan 02 unit tests that test prepare.py against a temp corpus
- All 7 requirement IDs (TOK-01..TOK-04, BASE-01..BASE-03) have named tests for traceability

---
*Phase: 02-tokenizer-baseline*
*Completed: 2026-03-12*
