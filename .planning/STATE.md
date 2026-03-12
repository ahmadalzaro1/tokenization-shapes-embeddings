---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: Completed 02-tokenizer-baseline-02-03-PLAN.md
last_updated: "2026-03-12T01:54:01.223Z"
last_activity: 2026-03-12 — Completed 02-01 Nyquist test stubs; human-verify approved
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 6
  completed_plans: 6
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Prove that D3 atomic diacritical encoding closes the disambiguation tax that D2 (stripped) forces models to pay, producing a publishable result targeting ArabicNLP/EMNLP
**Current focus:** Phase 2 — Tokenizer Baseline

## Current Position

Phase: 2 of 4 (Tokenizer Baseline)
Plan: 1 of 3 in current phase — COMPLETE
Status: 02-01 done; ready for 02-02 (prepare.py extensions)
Last activity: 2026-03-12 — Completed 02-01 Nyquist test stubs; human-verify approved

Progress: [██████████] 100% (Phase 1) | [███░░░░░░░] 33% (Phase 2)

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

### Pending Todos

None yet.

### Blockers/Concerns

- HuggingFace downloads stall on current network — may need Kaggle/SourceForge fallback for Tashkeela dataset (DATA-01)

## Session Continuity

Last session: 2026-03-12T01:30:46.344Z
Stopped at: Completed 02-tokenizer-baseline-02-03-PLAN.md
Resume file: None
