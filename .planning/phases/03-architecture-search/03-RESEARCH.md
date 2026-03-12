# Phase 3: Architecture Search - Research

**Researched:** 2026-03-12
**Domain:** Autonomous experiment loop (autoresearch-mlx protocol), git branch management, TSV logging, search_results.json summary
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Run order:**
- D3 first (baseline 1.075 bpb) — novel contribution, establish search works here first
- D1 second (baseline 1.191 bpb) — reference condition / comparison anchor
- D2 last (baseline 1.597 bpb) — control, highest bpb, least critical for paper's main claim

**Branch strategy:**
- Three separate autoresearch branches: `autoresearch/arabic-d3`, `autoresearch/arabic-d1`, `autoresearch/arabic-d2`
- Each branch is independent — results are not merged, just compared
- Branch off current `main` for each run

**Condition switching mechanism:**
- Use `AUTORESEARCH_CONDITION` env var (same as Phase 2 train.py invocation)
- Before each condition's run: verify that condition's tokenizer exists at `~/.cache/autoresearch-arabic/{condition}/tokenizer/`
- Baseline to beat (from baseline_results.json): D3=1.075, D1=1.191, D2=1.597

**Results storage:**
- Per-condition `results.tsv` at project root, following autoresearch-mlx program.md format (tab-separated, 5 cols: commit, val_bpb, memory_gb, status, description)
- Rename or scope per condition: `results_d3.tsv`, `results_d1.tsv`, `results_d2.tsv` — keeps runs isolated and avoids overwrite
- After each full run, commit results.tsv to the condition's branch

**Search scope:**
- Full autonomous loop per program.md: architecture, optimizer, hyperparameters, batch size, model size all in scope
- Only train.py is modified — prepare.py and the evaluation harness are read-only
- 70+ experiments per condition target (at ~7 min/experiment = ~8 hours per condition, ~24 hours total)
- Agent runs until manually stopped or 70+ experiments logged — NEVER pauses to ask

**Stopping and handoff:**
- Each condition run produces: condition branch + results.tsv with all experiments logged
- Final plan task reads best config from results.tsv and records it in a `search_results.json` summary
- search_results.json format: `{d1: {best_val_bpb, config, commit}, d2: {...}, d3: {...}}`
- Phase 3 complete when all three conditions have search_results.json entries and 70+ experiments each

### Claude's Discretion
- Specific architectural ideas to try (autoresearch agent decides autonomously)
- Exact experiment ordering within each condition run
- When to discard vs keep marginal improvements (per program.md guidelines)

### Deferred Ideas (OUT OF SCOPE)
- Parallel runs across conditions — single Mac, not feasible
- Sharing best architecture across conditions as a starting point — Phase 4 analysis
- Automated stopping when improvement plateaus — not in program.md protocol, defer to Phase 4
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SRCH-01 | Run autoresearch overnight for D1 (70+ experiments) | Branch `autoresearch/arabic-d1` off main; `AUTORESEARCH_CONDITION=d1 uv run train.py`; log to `results_d1.tsv`; agent follows program.md loop autonomously until 70+ rows |
| SRCH-02 | Run autoresearch overnight for D2 (70+ experiments) | Branch `autoresearch/arabic-d2` off main; `AUTORESEARCH_CONDITION=d2 uv run train.py`; log to `results_d2.tsv`; same loop; target beat 1.597 baseline |
| SRCH-03 | Run autoresearch overnight for D3 (70+ experiments) | Branch `autoresearch/arabic-d3` off main; `AUTORESEARCH_CONDITION=d3 uv run train.py`; log to `results_d3.tsv`; target beat 1.075 baseline; run first per locked decision |
</phase_requirements>

---

## Summary

Phase 3 is pure execution — no new code is written. The autoresearch loop protocol is fully defined in `autoresearch-mlx/program.md` and the training infrastructure (train.py + prepare.py) is complete and verified from Phase 2. All three baseline values are recorded in `~/.cache/autoresearch-arabic/baseline_results.json` (D1=1.191, D2=1.597, D3=1.075). The agent's job is to set up three independent git branches, initialize `results_{condition}.tsv` files with a baseline entry, then run the autonomous loop until 70+ experiments are logged per condition.

The critical adaptation from the standard program.md setup is naming. The original protocol creates `results.tsv` at the project root and uses a generic run tag. This project uses condition-specific filenames (`results_d3.tsv`, etc.) to avoid overwriting across three separate runs. The git staging rule from program.md also adapts: `git add autoresearch-mlx/train.py` becomes `git add train.py` since `train.py` is at the project root, not inside the monorepo subdirectory. The `program.md` monorepo note applies here: "Never use blind `git add -A`."

After all three overnight runs complete, a final analysis step reads each `results_{condition}.tsv`, extracts the best `val_bpb` row, and writes `search_results.json` to the project root for Phase 4 consumption. The search_results.json captures `best_val_bpb`, the train.py config params (depth, window_pattern, etc.) and the commit hash for reproducibility.

**Primary recommendation:** Set up each condition branch immediately before its overnight run to minimize stale-state risk. Verify tokenizer exists at `~/.cache/autoresearch-arabic/{condition}/tokenizer/tokenizer.pkl` before starting. Run D3 first, sleep, then D1, sleep, then D2.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mlx | >=0.30.0 | Model training + eval in train.py | Already in use; entire model stack is MLX; Apple Silicon only; no alternatives |
| uv | (system) | Run train.py in isolated env | Already established; `uv run train.py` is the exact invocation from program.md |
| git | (system) | Per-experiment commit + discard via reset --hard | Integral to program.md loop; each kept experiment is a commit; discards revert to previous |
| json (stdlib) | stdlib | search_results.json output | No new dependency; established project JSON pattern |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python csv/tsv (stdlib) | stdlib | Read results_d{n}.tsv to find best row | Used at end of each condition run to extract best config |
| grep (shell) | (system) | Extract val_bpb from run.log | Exact command from program.md: `grep "^val_bpb:\|^peak_vram_mb:" run.log` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Condition-specific TSV names (`results_d3.tsv`) | Single `results.tsv` per branch | Single file risks overwrite if branches share workspace; condition-specific names make analysis unambiguous |
| `git reset --hard` for discards | `git stash` | program.md specifies reset --hard; stash adds complexity; reset is clean and deterministic |
| Manual best-row extraction | Automated plateau detection | Plateau detection is deferred to Phase 4; manual extraction is explicit and auditable |

**Installation:**
```bash
# No new packages needed — all existing
uv sync
```

---

## Architecture Patterns

### Recommended Project Structure (Phase 3 outputs)

```
(project root — main branch)
├── train.py                    # read-only during search; only agent modifies on its branches
├── prepare.py                  # read-only, never modified
├── search_results.json         # NEW: written after all 3 runs, consumed by Phase 4
│
(autoresearch/arabic-d3 branch)
├── train.py                    # agent's evolving modifications
├── results_d3.tsv              # experiment log for D3 condition
│
(autoresearch/arabic-d1 branch)
├── train.py
├── results_d1.tsv
│
(autoresearch/arabic-d2 branch)
├── train.py
├── results_d2.tsv
```

### Pattern 1: Branch Setup Per Condition

**What:** Create a fresh branch off main, initialize the TSV header + baseline row, then hand off to the autonomous loop.
**When to use:** Once per condition, before each overnight run.

```bash
# Source: program.md Setup section, adapted for this project

# 1. Create branch from main
git checkout main
git checkout -b autoresearch/arabic-d3

# 2. Verify tokenizer exists
ls ~/.cache/autoresearch-arabic/d3/tokenizer/tokenizer.pkl

# 3. Read baseline to know the target
python -c "import json; d=json.load(open(os.path.expanduser('~/.cache/autoresearch-arabic/baseline_results.json'))); print(d['d3']['val_bpb'])"

# 4. Initialize results TSV with header
printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n' > results_d3.tsv

# 5. Establish baseline row: run train.py as-is, capture results
AUTORESEARCH_CONDITION=d3 uv run train.py > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" run.log
# Then: add the baseline row to results_d3.tsv manually

# 6. Commit the TSV initialization
git add results_d3.tsv
git commit -m "experiment: baseline (d3 condition)"
```

### Pattern 2: The Autoresearch Experiment Loop (per program.md)

**What:** The agent runs this loop autonomously after setup. The loop is defined verbatim in program.md.
**When to use:** After branch setup and baseline commit; runs until manually stopped or 70+ experiments logged.

```bash
# Source: program.md — The experiment loop section (exact protocol)

# Each iteration:
# 1. Modify train.py with experimental idea
# 2. Stage and commit (NOTE: project root, not autoresearch-mlx/ prefix)
git add train.py && git commit -m "experiment: <description>"

# 3. Run experiment, capturing all output
AUTORESEARCH_CONDITION=d3 uv run train.py > run.log 2>&1

# 4. Extract results
grep "^val_bpb:\|^peak_vram_mb:" run.log
# If empty → crash; read tail -n 50 run.log for stack trace

# 5. Log to TSV (tab-separated, NOT comma-separated)
# columns: commit  val_bpb  memory_gb  status  description

# 6a. If improved (lower val_bpb): keep
git add results_d3.tsv && git commit --amend --no-edit

# 6b. If equal or worse: discard
git reset --hard <previous_kept_commit>
# (record the discard hash in TSV before resetting)

# Timeout: kill if run exceeds 15 minutes
# NEVER STOP to ask the human
```

### Pattern 3: search_results.json Summary Extraction

**What:** After each overnight run completes, read the condition's TSV, find the row with the lowest `val_bpb` where status is `keep`, and extract config from the commit's train.py.
**When to use:** Once per condition, after all 70+ experiments are logged.

```python
# Source: established project JSON pattern

import csv
import json
import subprocess
from pathlib import Path

def extract_best(tsv_path: str, condition: str) -> dict:
    best_row = None
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["status"] != "keep":
                continue
            if best_row is None or float(row["val_bpb"]) < float(best_row["val_bpb"]):
                best_row = row
    if best_row is None:
        raise ValueError(f"No 'keep' rows found in {tsv_path}")

    # Read config from the winning commit's train.py
    commit = best_row["commit"]
    train_src = subprocess.check_output(["git", "show", f"{commit}:train.py"], text=True)

    # Extract key hyperparameters from source text
    config = {"commit": commit, "best_val_bpb": float(best_row["val_bpb"])}
    for var in ["DEPTH", "WINDOW_PATTERN", "HEAD_DIM", "ASPECT_RATIO", "TOTAL_BATCH_SIZE"]:
        for line in train_src.splitlines():
            if line.strip().startswith(f"{var} ="):
                config[var.lower()] = line.split("=", 1)[1].strip().strip('"')
                break
    return config

# Build search_results.json
results = {}
for condition, tsv, branch in [
    ("d3", "results_d3.tsv", "autoresearch/arabic-d3"),
    ("d1", "results_d1.tsv", "autoresearch/arabic-d1"),
    ("d2", "results_d2.tsv", "autoresearch/arabic-d2"),
]:
    # Must be run while checked out on the correct branch (or use git show)
    results[condition] = extract_best(tsv, condition)

search_results_path = Path("search_results.json")
with open(search_results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Written: {search_results_path}")
```

### Pattern 4: Condition-Specific Staging Rule

**What:** When the agent stages files, it must stage `train.py` (project root) and `results_d{n}.tsv` — never `git add -A`. The `autoresearch-mlx/` prefix in program.md does NOT apply here; `train.py` lives at project root.
**When to use:** Every experiment commit and every TSV amend.

```bash
# CORRECT (project root files)
git add train.py && git commit -m "experiment: ..."
git add results_d3.tsv && git commit --amend --no-edit

# WRONG — do not use
git add -A                    # may stage .DS_Store, __pycache__, etc.
git add autoresearch-mlx/train.py  # wrong path — train.py is at root
```

### Anti-Patterns to Avoid

- **Staging with `git add -A`:** program.md explicitly warns against this in monorepo contexts. Stage only `train.py` and `results_d{n}.tsv` by name.
- **Modifying prepare.py:** The evaluation harness is read-only. Any changes to `evaluate_bpb()` would invalidate the comparison against the Phase 2 baseline.
- **Using a single `results.tsv`:** All three condition runs would overwrite each other. Use `results_d{n}.tsv` per the locked decision.
- **Carrying train.py edits across branches:** Each branch must start from main's current train.py. Never cherry-pick experiment commits across condition branches.
- **Running without verifying tokenizer:** If `~/.cache/autoresearch-arabic/{condition}/tokenizer/tokenizer.pkl` is missing, `train.py` fails silently with a confusing `FileNotFoundError`. Always verify before starting the loop.
- **Amending a commit that includes results_d{n}.tsv and then forgetting to re-add it after a discard:** After `git reset --hard`, the TSV reverts. Re-log the discarded row from memory/run.log before continuing.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Experiment logging | Custom database or SQLite | Tab-separated `results_d{n}.tsv` per program.md | program.md specifies TSV exactly; any deviation breaks the established format expected by downstream consumers |
| Hyperparameter search | Custom Bayesian optimizer | Autonomous agent judgment + manual iteration | program.md defines the loop as free-form agent judgment; structure would conflict with program.md autonomy |
| val_bpb evaluation | Custom metric | `prepare.evaluate_bpb()` in prepare.py | Fixed evaluation harness; must not be modified; any custom metric invalidates comparison with Phase 2 baseline |
| Discard / keep logic | Threshold-based automation | `git reset --hard` + manual TSV entry per program.md | program.md defines keep/discard; automated thresholding is a Phase 4 concern |
| Crash recovery | Watchdog daemon | `tail -n 50 run.log` + agent judgment per program.md | program.md specifies: read stack trace, fix if trivial, skip if fundamentally broken |

**Key insight:** program.md is the law. Anything not in program.md is out of scope for Phase 3. The loop structure must be followed exactly, not "improved."

---

## Common Pitfalls

### Pitfall 1: Wrong git add Path

**What goes wrong:** Agent runs `git add autoresearch-mlx/train.py` (the path from program.md's monorepo note). File not found error because `train.py` is at project root.
**Why it happens:** program.md's setup instructions reference the standard autoresearch-mlx monorepo layout where train.py is inside `autoresearch-mlx/`. This project has train.py at root.
**How to avoid:** All plans must specify `git add train.py` (no subdirectory prefix) and `git add results_d{condition}.tsv` (condition-specific filename).
**Warning signs:** `git add autoresearch-mlx/train.py` returns `fatal: pathspec did not match any files`.

### Pitfall 2: results.tsv Overwrite Across Conditions

**What goes wrong:** Running D1 and D3 both write to `results.tsv` (if the condition-specific naming is forgotten). D3 results overwrite D1 results or vice versa.
**Why it happens:** program.md uses a generic `results.tsv` name. This project needs condition isolation.
**How to avoid:** Every plan task must explicitly specify `results_d{condition}.tsv` as the output file. The branch setup step initializes the correct filename before any experiments run.
**Warning signs:** `results.tsv` exists on the branch instead of `results_d{n}.tsv`.

### Pitfall 3: Baseline Not Re-Established on the Branch

**What goes wrong:** The agent skips the baseline run and immediately starts experimenting. All improvement comparisons are made against the Phase 2 baseline_results.json value (1.075/1.191/1.597), but without a branch-local baseline to reset to, the discard logic (`git reset --hard <previous_kept_commit>`) has no safe anchor.
**Why it happens:** program.md Step 5 says "Run `uv run train.py` once to establish YOUR baseline on this hardware." The Phase 2 values are already the canonical hardware baseline, so the agent may skip this.
**How to avoid:** The first experiment on each branch must be the unmodified train.py (the same as Phase 2 baseline). This establishes the initial commit to reset to, and verifies the branch environment is working. The val_bpb from this first run should match baseline_results.json within 0.001 (same code, same hardware, same 5-minute budget).
**Warning signs:** First TSV row does not have status=`keep` or description=`baseline`.

### Pitfall 4: Token Budget Differences Across Conditions Mislead Architecture Search

**What goes wrong:** D3 trains 18.1M tokens in 5 minutes vs D1's 5.2M and D2's 4.5M (from baseline_results.json). An architecture change that increases throughput benefits D1/D2 more than D3 (D3 is already token-rich). The agent may explore different areas of architecture space for each condition without realizing why.
**Why it happens:** D3's PUA encoding packs more info per token; BPE at 8192 vocab merges PUA codepoints aggressively. D3 models see ~3.5x more tokens per second than D1.
**How to avoid:** Treat each condition independently. D3 may benefit more from model capacity (more tokens = faster learning per step). D1/D2 may benefit more from efficient architectures that extract more from each token. The agent makes these decisions autonomously per program.md; this is context for understanding results, not a constraint on search direction.
**Warning signs:** D3 search finds very deep models optimal (benefits from more compute per token) while D1/D2 find wider shallower models optimal — this is expected, not a bug.

### Pitfall 5: Git State Corruption from Interrupted Runs

**What goes wrong:** The user manually kills a training run mid-execution (Ctrl+C). The run.log is incomplete; grep extracts nothing. If the agent ran `git add train.py && git commit` before starting the run, there is now a dangling uncommitted change in the modified train.py (wait, actually: commit already happened before run) — so the last commit is an experiment commit with no results. The agent must handle this gracefully.
**Why it happens:** Machine interruptions during overnight runs are possible.
**How to avoid:** If grep on run.log returns nothing (crash/interrupt), treat it identically to a crash: log `crash` status with 0.0 for val_bpb and memory, then `git reset --hard` to the previous kept commit. The interrupted commit is discarded cleanly.
**Warning signs:** `grep "^val_bpb:" run.log` returns empty output.

### Pitfall 6: search_results.json Extraction from Wrong Branch

**What goes wrong:** When extracting the best config using `git show {commit}:train.py`, if the extraction script runs from the wrong branch (e.g., main instead of autoresearch/arabic-d3), the commit hash from results_d3.tsv may not be reachable.
**Why it happens:** Each condition's commits are on their own branch. `git show` requires the commit to be reachable from the current HEAD or by hash.
**How to avoid:** Run the extraction script from the condition's branch, or use commit hashes directly (git show always works with a full SHA). The extraction plan task must specify which branch to check out before running.
**Warning signs:** `git show abc1234:train.py` fails with `fatal: Path 'train.py' does not exist in 'abc1234'`.

---

## Code Examples

Verified patterns from direct code inspection:

### Exact Output Format from train.py

```bash
# Source: train.py lines 550–560
# train.py prints this block after evaluation:
---
val_bpb:          1.075381
training_seconds: 300.9
total_seconds:    401.2
peak_vram_mb:     3456.0
mfu_percent:      0.00
total_tokens_M:   18.1
num_steps:        47
num_params_M:     11.53
depth:            4
```

```bash
# Extraction command (from program.md):
grep "^val_bpb:\|^peak_vram_mb:" run.log
# Output: val_bpb:          1.075381
#         peak_vram_mb:     3456.0
```

### TSV Row Format

```
# Source: program.md Logging results section
commit	val_bpb	memory_gb	status	description
383abb4	1.075381	3.4	keep	baseline
909dd59	1.062000	3.4	keep	increase depth to 6
4161af3	1.063000	3.4	discard	add warmup ratio 0.1 — no improvement
```

Note: memory_gb = peak_vram_mb / 1024, rounded to .1f.

### Tokenizer Existence Verification

```bash
# Verify before starting each condition's run
ls ~/.cache/autoresearch-arabic/d3/tokenizer/tokenizer.pkl
ls ~/.cache/autoresearch-arabic/d3/tokenizer/token_bytes.npy
# If either is missing: run `uv run prepare.py --condition d3` first (should not be needed — Phase 2 complete)
```

### Reading Baseline Targets

```bash
# Read baseline targets from Phase 2 output
python -c "
import json, os
path = os.path.expanduser('~/.cache/autoresearch-arabic/baseline_results.json')
with open(path) as f:
    d = json.load(f)
for cond in ['d3', 'd1', 'd2']:
    print(f'{cond}: val_bpb={d[cond][\"val_bpb\"]:.6f}')
"
# Expected output:
# d3: val_bpb=1.075381
# d1: val_bpb=1.190999
# d2: val_bpb=1.596882
```

### search_results.json Expected Structure

```json
{
  "d3": {
    "best_val_bpb": 1.042000,
    "commit": "a3f7b2c",
    "depth": "6",
    "window_pattern": "SSSL",
    "head_dim": "128",
    "aspect_ratio": "64",
    "total_batch_size": "65536"
  },
  "d1": {
    "best_val_bpb": 1.155000,
    "commit": "b8c1d4e",
    "depth": "4",
    "window_pattern": "SSSL",
    "head_dim": "128",
    "aspect_ratio": "64",
    "total_batch_size": "65536"
  },
  "d2": {
    "best_val_bpb": 1.520000,
    "commit": "c2e5f9a",
    "depth": "8",
    "window_pattern": "SSLL",
    "head_dim": "128",
    "aspect_ratio": "64",
    "total_batch_size": "65536"
  }
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single `results.tsv` for one condition | Condition-specific `results_d{n}.tsv` | Phase 3 design | Keeps three overnight runs isolated; prevents overwrite |
| Generic autoresearch run tag (e.g. `mar5`) | Condition-based branch name (`autoresearch/arabic-d3`) | Phase 3 design | Branch names encode condition semantics; makes Phase 4 comparison explicit |
| train.py at `autoresearch-mlx/train.py` | train.py at project root | Project setup | Git staging path changes from `git add autoresearch-mlx/train.py` to `git add train.py` |

**Already correct (no changes needed):**
- `AUTORESEARCH_CONDITION` env var already wired in train.py and prepare.py
- `baseline_results.json` already exists at `~/.cache/autoresearch-arabic/baseline_results.json` with all three conditions
- All three tokenizers already exist at `~/.cache/autoresearch-arabic/{condition}/tokenizer/`
- train.py already prints all required output fields in program.md's expected format

---

## Open Questions

1. **Whether the first branch baseline run should overwrite baseline_results.json**
   - What we know: train.py always writes `baseline_results.json` at the end of each run. Running the baseline on the `autoresearch/arabic-d3` branch will overwrite the D3 entry in baseline_results.json with the same value (same code, same hardware).
   - What's unclear: If train.py is modified on the branch and then run, the baseline_results.json in `~/.cache` (which is not tracked by git) gets overwritten with experiment values — misleading for future Phase 4 reads.
   - Recommendation: The plan must ensure the first run on each branch uses unmodified train.py (identical to main). After the loop begins modifying train.py, the writes to baseline_results.json will reflect experimental configs, not the canonical baseline. The canonical baseline is preserved in git at the main branch's commit. This is acceptable because Phase 4 reads `search_results.json` (the best per condition), not `baseline_results.json` for comparisons.

2. **Experiment count tracking: 70+ means TSV rows with status=keep or total rows?**
   - What we know: program.md says "70 experiments" in the context of 8–9/hour over ~8 hours of sleep. The intent is 70 experiment iterations (each iteration = one train.py run), regardless of keep/discard.
   - What's unclear: Whether a crashed run counts toward the 70.
   - Recommendation: Count total rows in the TSV (including discard and crash rows). Each row = one experiment attempt. 70+ rows means 70+ iterations of the loop. This matches program.md's timing math.

3. **Memory pressure across a 70-experiment overnight run**
   - What we know: program.md documents that each experiment takes ~7 minutes (5 min training + ~1 min compile/eval overhead). After 70 experiments, no explicit cleanup is mentioned.
   - What's unclear: Whether MLX unified memory accumulates garbage across subprocess runs. Each `uv run train.py` is a fresh Python process, so this is not a risk — MLX memory is released when the process exits.
   - Recommendation: No action needed. Each `uv run train.py` is a fresh subprocess; memory resets between experiments. The `gc.collect(); gc.freeze(); gc.disable()` in train.py optimizes within a single run, not across runs.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 (installed) |
| Config file | none — uses pyproject.toml project root |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SRCH-01 | results_d1.tsv has 70+ rows with valid schema and at least one 'keep' row with val_bpb < 1.191 | smoke (post-run) | `uv run pytest tests/test_search.py::test_search_d1 -x` | Wave 0 |
| SRCH-02 | results_d2.tsv has 70+ rows with valid schema and at least one 'keep' row with val_bpb < 1.597 | smoke (post-run) | `uv run pytest tests/test_search.py::test_search_d2 -x` | Wave 0 |
| SRCH-03 | results_d3.tsv has 70+ rows with valid schema and at least one 'keep' row with val_bpb < 1.075 | smoke (post-run) | `uv run pytest tests/test_search.py::test_search_d3 -x` | Wave 0 |
| All | search_results.json exists with d1/d2/d3 keys, each with best_val_bpb and commit | smoke (post-run) | `uv run pytest tests/test_search.py::test_search_results_json -x` | Wave 0 |

**Test strategy:** These tests are smoke checks that run after each overnight run completes — they cannot run during the loop (experiments are still in progress). Tests read the TSV files and search_results.json from the filesystem (not `~/.cache`; TSV files live at project root on the branch). Tests must be run while the condition's branch is checked out, or must accept a path argument.

### Sampling Rate

- **Per task commit:** N/A — no code changes during the loop; tests run only after each condition's run completes
- **Per wave merge:** `uv run pytest tests/test_search.py -q` after the post-run summary task
- **Phase gate:** All three `test_search_d{n}` tests green + `test_search_results_json` green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_search.py` — covers SRCH-01 through SRCH-03 + search_results.json schema; reads TSV files at project root and JSON at project root; skips with helpful message if TSV not yet created (pre-run state)

*(Existing test files: `tests/test_baseline.py`, `tests/test_tokenizer.py`, `tests/test_pipeline.py`, `tests/conftest.py` — none cover Phase 3; all are Phase 1/2 artifacts)*

---

## Sources

### Primary (HIGH confidence)

- Direct read of `autoresearch-mlx/program.md` — full protocol spec; all loop mechanics, TSV format, git commands, timeout rules confirmed
- Direct read of `train.py` — all hyperparameter names (DEPTH, WINDOW_PATTERN, HEAD_DIM, ASPECT_RATIO, TOTAL_BATCH_SIZE, etc.), output format, `AUTORESEARCH_CONDITION` wiring confirmed
- Direct read of `prepare.py` — TIME_BUDGET=300, EVAL_TOKENS=3*524288, MAX_SEQ_LEN=2048, FINAL_EVAL_BATCH_SIZE=256 confirmed; read-only status confirmed
- `~/.cache/autoresearch-arabic/baseline_results.json` — D1=1.190999, D2=1.596882, D3=1.075381 confirmed; all three conditions present
- Phase 2 RESEARCH.md — established patterns for condition env var, JSON merge pattern, tokenizer path conventions
- Direct inspection of `tests/` directory — confirmed existing test files; `test_search.py` does not yet exist

### Secondary (MEDIUM confidence)

- Phase 3 CONTEXT.md — all locked decisions and constraints confirmed against code inspection

### Tertiary (LOW confidence — marked for validation)

- Expected val_bpb improvement from search: D3 < 1.075, D1 < 1.191, D2 < 1.597 — theoretically motivated (search should beat a fixed baseline) but not yet empirically confirmed for this corpus/hardware combination

---

## Metadata

**Confidence breakdown:**
- Protocol mechanics (program.md loop, git commands, TSV format): HIGH — source is authoritative and complete
- Condition adaptation (path changes, TSV naming): HIGH — derived from direct code inspection
- Expected search outcomes (val_bpb improvements): LOW — empirical question, answer unknown until runs complete

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (program.md is a fixed spec; MLX and git APIs are stable)
