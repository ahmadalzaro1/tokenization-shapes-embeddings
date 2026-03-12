# Phase 3: Architecture Search - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the autoresearch autonomous experiment loop for all three conditions (D1, D2, D3), achieving 70+ experiments per condition with best-performing architecture configs identified. This phase is execution, not code — the search loop protocol is already defined in `autoresearch-mlx/program.md`. No new code is written; train.py is modified per-experiment by the agent.

</domain>

<decisions>
## Implementation Decisions

### Run order
- D3 first — it's the novel contribution and has the best baseline (1.075 bpb). Establish it works well under search before moving on.
- D1 second — reference condition (1.191 bpb), needed as the comparison anchor
- D2 last — control condition (1.597 bpb), highest bpb, most room to improve but least important for the paper's main claim

### Branch strategy
- Three separate autoresearch branches: `autoresearch/arabic-d3`, `autoresearch/arabic-d1`, `autoresearch/arabic-d2`
- Each branch is independent — results are not merged, just compared
- Branch off current `main` for each run

### Condition switching mechanism
- Use `AUTORESEARCH_CONDITION` env var (same as Phase 2 train.py invocation)
- Before each condition's run: verify that condition's tokenizer exists at `~/.cache/autoresearch-arabic/{condition}/tokenizer/`
- Baseline to beat (from baseline_results.json): D3=1.075, D1=1.191, D2=1.597

### Results storage
- Per-condition `results.tsv` at project root, following autoresearch-mlx program.md format (tab-separated, 5 cols: commit, val_bpb, memory_gb, status, description)
- Rename or scope per condition: `results_d3.tsv`, `results_d1.tsv`, `results_d2.tsv` — keeps runs isolated and avoids overwrite
- After each full run, commit results.tsv to the condition's branch

### Search scope
- Full autonomous loop per program.md: architecture, optimizer, hyperparameters, batch size, model size all in scope
- Only train.py is modified — prepare.py and the evaluation harness are read-only
- 70+ experiments per condition target (at ~7 min/experiment = ~8 hours per condition, ~24 hours total)
- Agent runs until manually stopped or 70+ experiments logged — NEVER pauses to ask

### Stopping and handoff
- Each condition run produces: condition branch + results.tsv with all experiments logged
- Final plan task reads best config from results.tsv and records it in a `search_results.json` summary
- search_results.json format: `{d1: {best_val_bpb, config, commit}, d2: {...}, d3: {...}}`
- Phase 3 complete when all three conditions have search_results.json entries and 70+ experiments each

### Claude's Discretion
- Specific architectural ideas to try (autoresearch agent decides autonomously)
- Exact experiment ordering within each condition run
- When to discard vs keep marginal improvements (per program.md guidelines)

</decisions>

<specifics>
## Specific Ideas

- The autoresearch protocol (program.md) is the ground truth for how the loop runs — agent must follow it exactly (NEVER STOP, git commit per experiment, results.tsv logging)
- D3's 18.1M tokens/5-min vs D1's 5.2M and D2's 4.5M suggests D3 trains much faster (PUA encoding packs more info per token) — search may converge differently
- D2's high baseline (1.597) leaves most room for improvement — but the paper cares about D3 < D1, so D3 and D1 are the critical comparison
- `baseline_results.json` already exists at `~/.cache/autoresearch-arabic/baseline_results.json` — agent should read it at start of each condition to know the target

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `train.py`: The file autoresearch modifies. Currently produces baseline; Phase 3 agent iterates it.
- `prepare.py`: Read-only. Contains `evaluate_bpb()`, data loading, tokenizer path resolution, time budget. `AUTORESEARCH_CONDITION` env var controls which condition's tokenizer/data is loaded.
- `autoresearch-mlx/program.md`: Complete protocol spec — the agent's "rules of the road"

### Established Patterns
- `AUTORESEARCH_CONDITION=d3 uv run train.py` — condition selection via env var
- `uv run train.py > run.log 2>&1` then `grep "^val_bpb:" run.log` — standard run + result extraction
- `git add train.py && git commit` per experiment (autoresearch-mlx/ prefix staging if in monorepo)
- results.tsv: tab-separated, commit/val_bpb/memory_gb/status/description columns

### Integration Points
- `~/.cache/autoresearch-arabic/{condition}/tokenizer/tokenizer.pkl` — tokenizer path prepare.py reads
- `~/.cache/autoresearch-arabic/baseline_results.json` — Phase 2 baseline values to beat
- `results.tsv` / `results_d{n}.tsv` — output artifact consumed by Phase 4 analysis

</code_context>

<deferred>
## Deferred Ideas

- Parallel runs across conditions — single Mac, not feasible
- Sharing best architecture across conditions as a starting point — Phase 4 analysis
- Automated stopping when improvement plateaus — not in program.md protocol, defer to Phase 4

</deferred>

---

*Phase: 03-architecture-search*
*Context gathered: 2026-03-12*
