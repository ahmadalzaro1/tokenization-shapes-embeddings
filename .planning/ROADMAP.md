# Roadmap: autoresearch-arabic

## Overview

Four phases take the experiment from raw dataset through publishable paper. Phase 1 builds the three dataset conditions and quantifies the homograph disambiguation tax. Phase 2 trains condition-specific tokenizers and establishes baselines. Phase 3 runs the overnight autoresearch loop across all three conditions. Phase 4 extracts the comparative story and writes the paper.

## Phases

- [ ] **Phase 1: Data Pipeline** - Download dataset, produce D1/D2/D3 parquet shards, compute homograph collision statistics
- [ ] **Phase 2: Tokenizer & Baseline** - Train BPE tokenizers per condition, measure fertility, run baseline val_bpb training
- [ ] **Phase 3: Architecture Search** - Run overnight autoresearch agent loop for D1, D2, D3 (70+ experiments each)
- [ ] **Phase 4: Analysis & Paper** - Compare winning architectures, run ablations, write paper draft

## Phase Details

### Phase 1: Data Pipeline
**Goal**: All three dataset conditions exist as validated parquet shards and homograph collision statistics quantify the disambiguation tax
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. D1, D2, and D3 parquet shards are on disk and load without error
  2. Token counts and character distributions are verified for each shard (no silent truncation or encoding corruption)
  3. Homograph collision rate is computed and logged — ألم-style ambiguities counted at corpus scale
  4. D3 PUA mapping table is complete (all letter+harakah combinations covered, ~252 combos)
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — Test scaffold: install pytest, create tests/ with conftest fixtures and six stub tests
- [ ] 01-02-PLAN.md — Extend build_dataset.py: tqdm download wrapper + context-window collision metric + JSON sidecar
- [ ] 01-03-PLAN.md — Add inline validation to build_dataset.py + create validate_dataset.py standalone validator

### Phase 2: Tokenizer & Baseline
**Goal**: Each condition has a trained BPE tokenizer and a measured baseline val_bpb on identical architecture, establishing the benchmark the search will beat
**Depends on**: Phase 1
**Requirements**: TOK-01, TOK-02, TOK-03, TOK-04, BASE-01, BASE-02, BASE-03
**Success Criteria** (what must be TRUE):
  1. Three BPE tokenizers trained (one per condition), each at multiple vocab sizes
  2. Fertility table (tokens/word × condition × vocab size) is computed and shows measurable difference between D1/D2/D3
  3. Baseline val_bpb recorded for D1, D2, D3 on fixed depth=4 architecture
  4. D2 baseline is lower than D1 baseline (stripping reduces surface complexity — expected sanity check)
**Plans**: TBD

### Phase 3: Architecture Search
**Goal**: Autoresearch has run 70+ experiments per condition overnight and best-performing architecture configs per condition are identified
**Depends on**: Phase 2
**Requirements**: SRCH-01, SRCH-02, SRCH-03
**Success Criteria** (what must be TRUE):
  1. D1, D2, and D3 overnight runs each complete with 70+ experiments logged
  2. Best val_bpb per condition is below the Phase 2 baseline (search found improvements)
  3. Winning architecture configs (depth, heads, window) are recorded per condition
  4. Search results are stored in condition-labeled output directories for reproducibility
**Plans**: TBD

### Phase 4: Analysis & Paper
**Goal**: A paper draft exists that makes the D3 encoding argument quantitatively, with comparison tables and ablation results
**Depends on**: Phase 3
**Requirements**: ANLZ-01, ANLZ-02, ANLZ-03
**Success Criteria** (what must be TRUE):
  1. Cross-condition comparison table shows D3 val_bpb vs D1 and D2 at matched parameter count
  2. Ablation results cover at least depth, vocab size, and window pattern
  3. Paper draft (LaTeX or Markdown) contains abstract, method, results section with figures, and conclusion
  4. Paper makes the disambiguation tax argument with the homograph collision statistic from Phase 1
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline | 1/3 | In Progress|  |
| 2. Tokenizer & Baseline | 0/TBD | Not started | - |
| 3. Architecture Search | 0/TBD | Not started | - |
| 4. Analysis & Paper | 0/TBD | Not started | - |
