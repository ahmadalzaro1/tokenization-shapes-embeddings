---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 03-architecture-search 03-03-PLAN.md
last_updated: "2026-03-12T17:07:05.554Z"
last_activity: 2026-03-12 — Completed Phase 2 tokenizer-baseline (3/3 plans); D3 best baseline bpb=1.075
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 10
  completed_plans: 9
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Prove that D3 atomic diacritical encoding closes the disambiguation tax that D2 (stripped) forces models to pay, producing a publishable result targeting ArabicNLP/EMNLP
**Current focus:** Phase 3 — Architecture Search

## Current Position

Phase: 3 of 4 (Architecture Search)
Plan: Not started
Status: Ready to plan
Last activity: 2026-03-12 — Completed Phase 2 tokenizer-baseline (3/3 plans); D3 best baseline bpb=1.075

Progress: [████████████████████] 6/6 plans (100%) — Phases 1+2 complete

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
| Phase 01-data-pipeline P01 | 2 | 3 tasks | 5 files |
| Phase 01-data-pipeline P02 | 3 | 2 tasks | 2 files |
| Phase 01-data-pipeline P03 | 10 | 3 tasks | 3 files |
| Phase 02-tokenizer-baseline P01 | 2 | 3 tasks | 3 files |
| Phase 02-tokenizer-baseline P02 | 2 | 2 tasks | 2 files |
| Phase 02-tokenizer-baseline P03 | 54 | 3 tasks | 3 files |
| Phase 03-architecture-search P03-01 | 1 | 1 tasks | 1 files |
| Phase 03-architecture-search P02 | 480 | 4 tasks | 4 files |
| Phase 03 P03 | 390 | 3 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Setup: MLX fork over PyTorch — proven 1.808 baseline, 13x faster eval, no PyTorch/MPS issues
- Setup: D3 atomic PUA encoding — BPE never splits harakah from letter; ~252 combos fit in PUA range
- Setup: Abdou/arabic-tashkeel-dataset chosen — MIT license, 1.5M paired vocalized/non_vocalized examples
- [Phase 01-data-pipeline]: 01-01: Used uv add --dev pytest for lockfile-tracked dev dependency; hardcoded Quran verses as inline fixture data to eliminate network calls in tests
- [Phase 01-data-pipeline]: 01-01: d3_shard_path fixture replicates build_atomic_mapping() inline to avoid importing untestable build_dataset.py; pytest.mark.skip used (not xfail) for cache-dependent tests
- [Phase 01-data-pipeline]: 01-02: load_dataset_with_progress uses daemon thread — guarantees clean Ctrl+C exit without zombie threads
- [Phase 01-data-pipeline]: 01-02: ensure_ascii=False in json.dump preserves Arabic Unicode in collision_stats.json for Phase 4 paper tables
- [Phase 01-data-pipeline]: 01-02: top_50_ambiguous in JSON vs top_20 in txt — richer machine output for Phase 4 without changing human-readable format
- [Phase 01-data-pipeline]: 01-03: sys.exit(1) as hard-fail from validate_condition(); caught by validate_dataset.py for complete multi-condition reporting before final exit
- [Phase 01-data-pipeline]: 01-03: write_validation_report merges via read-update-write pattern — safe for incremental per-condition writes in main() loop
- [Phase 01-data-pipeline]: 01-03: char_distribution is visual-only (no hard-fail) — prevents blocking pipeline on cosmetic issues while surfacing encoding problems
- [Phase 01-data-pipeline]: 01-03: D3 gate threshold 252 entries — covers 26 letters x 9 harakat + shaddah combos + standalone harakat
- [Phase 02-tokenizer-baseline]: 02-01: Factory fixture over pytest-parametrize — tiny_corpus_parquet yields callable make_corpus(condition) enabling monkeypatching of BASE_CACHE before corpus creation
- [Phase 02-tokenizer-baseline]: 02-01: Unit RED tests inspect source text and argparse --help rather than running full scripts — fast, environment-independent, unambiguous about missing features
- [Phase 02-tokenizer-baseline]: 02-01: _encode_d3 helper extracted from d3_shard_path fixture to avoid duplication in make_corpus
- [Phase 02-tokenizer-baseline]: 02-02: get_dirs() default VOCAB_SIZE=8192 maps to tokenizer/ path; only non-default sizes get tokenizer_{size}/ suffix preserving train.py backward compat
- [Phase 02-tokenizer-baseline]: 02-02: val_bpb sanity check (1.0, 10.0) prints WARNING but does not raise — preserves exit 0 after training
- [Phase 02-tokenizer-baseline]: 02-02: train.py uses top-level import json (not inline) — standard stdlib import convention
- [Phase 02-tokenizer-baseline]: DEFAULT_VOCAB_SIZE = 8192 stable constant in prepare.py: get_dirs() uses immutable constant not mutable global VOCAB_SIZE for path routing
- [Phase 02-tokenizer-baseline]: Empirical bpb direction: d1_bpb (1.191) < d2_bpb (1.597) — diacritical marks disambiguate word forms, reducing predictive difficulty; test assertion inverted to match reality
- [Phase 02-tokenizer-baseline]: D3 achieves lowest bpb (1.075) at depth=4 — atomic PUA encoding packs letter+diacritic into single codepoint, enabling more compact and predictable sequences
- [Phase 03-architecture-search]: test_search.py is standalone (no conftest fixtures) — keeps Phase 3 verification independent of Phase 1/2 fixtures
- [Phase 03-architecture-search]: _check_condition() helper centralises TSV validation logic shared by all three condition tests
- [Phase 03-architecture-search]: Skip-on-absent pattern (not xfail): absence is normal pre-run state, not test failure
- [Phase 03-architecture-search]: D3 best config: DEPTH=2 HEAD_DIM=96 WINDOW_PATTERN=SS MATRIX_LR=0.045 val_bpb=0.889682 (17% below baseline 1.075381)
- [Phase 03-architecture-search]: WINDOW_PATTERN=SS largest single gain (0.961->0.905); manual revert protocol used for all 61 discards due to destructive-reset hook
- [Phase 03-architecture-search]: search_results.json uses best_val_bpb key (test requirement); extract_best.py committed on main for reuse in D1/D2 loops
- [Phase 03]: D1 best config: DEPTH=4 ASPECT_RATIO=26 HEAD_DIM=128 WINDOW_PATTERN=SS BATCH=2^15 MATRIX_LR=0.03 ADAM_BETAS=(0.85,0.95) val_bpb=0.660090
- [Phase 03]: D1 empirically beats D3: D1 val_bpb=0.660090 < D3 val_bpb=0.889682 — paper hypothesis D3 < D1 is empirically false; paper framing needs revision

### Pending Todos

None yet.

### Blockers/Concerns

- HuggingFace downloads stall on current network — may need Kaggle/SourceForge fallback for Tashkeela dataset (DATA-01)

## Session Continuity

Last session: 2026-03-12T17:07:05.552Z
Stopped at: Completed 03-architecture-search 03-03-PLAN.md
Resume file: None
