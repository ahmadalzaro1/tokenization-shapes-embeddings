# Requirements: autoresearch-arabic

## Overview

Research pipeline studying how Arabic diacritical marks (harakat) should be encoded for small language model pretraining. Three dataset conditions (D1/D2/D3) are run through tokenizer training, baseline measurement, and overnight autoresearch architecture search to produce a publishable ML paper.

## v1 Requirements

### Data Pipeline

| ID | Requirement | Priority |
|----|-------------|----------|
| DATA-01 | Download and validate Abdou/arabic-tashkeel-dataset (1.5M examples) | Must have |
| DATA-02 | Build D1 parquet shards (diacritized, raw Unicode combining chars) | Must have |
| DATA-03 | Build D2 parquet shards (harakat stripped — control condition) | Must have |
| DATA-04 | Build D3 parquet shards (atomic PUA encoding — letter+harakah as single codepoint) | Must have |
| DATA-05 | Compute homograph collision statistics at corpus scale | Must have |

### Tokenizer & Baseline

| ID | Requirement | Priority |
|----|-------------|----------|
| TOK-01 | Train BPE tokenizer for D1 condition | Must have |
| TOK-02 | Train BPE tokenizer for D2 condition | Must have |
| TOK-03 | Train BPE tokenizer for D3 condition | Must have |
| TOK-04 | Measure tokenizer fertility (tokens/word) per condition × vocab size | Must have |
| BASE-01 | Run baseline val_bpb training for D1 | Must have |
| BASE-02 | Run baseline val_bpb training for D2 | Must have |
| BASE-03 | Run baseline val_bpb training for D3 | Must have |

### Architecture Search

| ID | Requirement | Priority |
|----|-------------|----------|
| SRCH-01 | Run autoresearch overnight for D1 (70+ experiments) | Must have |
| SRCH-02 | Run autoresearch overnight for D2 (70+ experiments) | Must have |
| SRCH-03 | Run autoresearch overnight for D3 (70+ experiments) | Must have |

### Analysis & Paper

| ID | Requirement | Priority |
|----|-------------|----------|
| ANLZ-01 | Compare winning architectures across D1/D2/D3 conditions | Must have |
| ANLZ-02 | Run targeted ablations (depth, vocab size, window pattern) | Must have |
| ANLZ-03 | Write paper draft with results | Must have |

## Out of Scope (v2+)

- Production Arabic LLM deployment
- Large models (>100M params)
- Dialect Arabic (experiment uses classical Arabic / Tashkeela)
- Fine-tuning or RLHF
- Multi-GPU / CUDA infrastructure

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| TOK-01 | Phase 2 | Pending |
| TOK-02 | Phase 2 | Pending |
| TOK-03 | Phase 2 | Pending |
| TOK-04 | Phase 2 | Pending |
| BASE-01 | Phase 2 | Pending |
| BASE-02 | Phase 2 | Pending |
| BASE-03 | Phase 2 | Pending |
| SRCH-01 | Phase 3 | Pending |
| SRCH-02 | Phase 3 | Pending |
| SRCH-03 | Phase 3 | Pending |
| ANLZ-01 | Phase 4 | Pending |
| ANLZ-02 | Phase 4 | Pending |
| ANLZ-03 | Phase 4 | Pending |
