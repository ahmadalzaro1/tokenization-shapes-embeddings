---
status: complete
phase: 05-definitive-d3-structural-advantage-experiments
source: [05-01-SUMMARY.md, 05-02-SUMMARY.md, 05-03-SUMMARY.md]
started: 2026-03-15T00:00:00Z
updated: 2026-03-16T12:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. BPBL Results File
expected: `experiments/results/bpbl_results.json` exists and shows D1 beating D3: D1 mean_bpbl < D3 mean_bpbl (lower is better). File has 3 seeds for each condition and a "conclusion" field.
result: pass

### 2. WTE Embedding Files
expected: 6 numpy files exist at `experiments/results/wte_{d1,d3}_seed{42,137,2024}.npy`, each saving the learned token embeddings for that training run.
result: pass

### 3. Embedding Similarity JSON
expected: `experiments/results/embedding_similarity.json` exists with D3 intra-group cosine similarity data for 20+ base letters and a D1 analysis section.
result: pass

### 4. D3 Heatmap Figure
expected: `experiments/results/d3_embedding_heatmap.png` exists and is a non-empty image (>50KB) showing cosine similarity heatmaps per Arabic base letter.
result: pass

### 5. Embedding Space Scatter Plot
expected: `experiments/results/embedding_space_comparison.png` exists (>50KB), showing PCA scatter plots comparing D1 and D3 embedding spaces side by side.
result: pass

### 6. Per-letter Bar Chart
expected: `experiments/results/d3_per_letter_similarity.png` exists (>50KB), showing a bar chart of mean intra-group cosine similarity per Arabic base letter.
result: pass

### 7. Iso-data Scaling Experiment (Plan 03)
expected: `experiments/results/iso_data_results.json` has 30 runs (2 conditions × 5 budgets × 3 seeds) and `experiments/results/iso_data_scaling_curves.png` exists.
result: pass

### 8. Iso-data Experiment Driver
expected: `experiments/exp5_iso_data.py` exists as a ~400+ line script with resumability (completed-set skip logic) and try/finally train.py restore.
result: pass

### 9. Training Logs
expected: `experiments/results/iso_data_logs/` contains exactly 30 log files named `{condition}_{budget}_seed{seed}.log`.
result: pass

### 10. Results JSON Structure
expected: `iso_data_results.json` has calibration section, 30 runs with all required fields, and budget-level summary with mean/std for both D1 and D3.
result: pass

### 11. train.py Restored
expected: train.py has env var patches from Plan 01 (AUTORESEARCH_CONDITION, AUTORESEARCH_SEED, AUTORESEARCH_MAX_STEPS) but hyperparams restored to D1 best config (ASPECT_RATIO=26, DEPTH=4, HEAD_DIM=128).
result: pass

## Summary

total: 11
passed: 11
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
