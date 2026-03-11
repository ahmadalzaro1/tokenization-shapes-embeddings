# Phase 1: Data Pipeline - Research

**Researched:** 2026-03-12
**Domain:** Arabic NLP data pipeline — HuggingFace dataset download, parquet shard writing, Unicode/PUA encoding, homograph collision statistics, inline validation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Download fallback strategy**
- HuggingFace is the primary source — no hard fallback to Kaggle in code
- Add tqdm progress bar to the HF download so the user can see if it's stuck
- No timeout: user manually Ctrl+C if stalled; script must print clear manual download instructions on exit
- Auto-detect local HF cache first (`load_dataset` will reuse it); only attempts network download if cache is cold
- HF's built-in caching handles re-run optimization — no extra raw cache layer needed

**Collision statistics scope**
- Compute both word-level stats (already implemented) AND 128-token context-window collision probability
- Context-window metric: for a random 128-token window, what fraction of tokens are homographically ambiguous?
- Output: `collision_stats.txt` (human-readable, already exists) + `collision_stats.json` (machine-readable sidecar)
- JSON must include: aggregate stats + top-50 most ambiguous words with all diacritized variants (Phase 4 pulls this into paper tables)

**Validation suite**
- Validation runs inline after each condition in `build_dataset.py` (auto, no flag needed)
- Validation also available as standalone `validate_dataset.py` for re-checking without rebuilding
- Output: print pass/fail per check to stdout + write `~/.cache/autoresearch-arabic/validation_report.json`
- Mandatory checks (all must pass before Phase 2):
  1. All shards load without exception
  2. Row count matches `metadata.txt` (train_docs + val_docs)
  3. Zero empty or null texts in any shard
  4. Character distribution sample (top-20 most frequent chars per condition, printed — visual assertion that D1 has harakat, D2 doesn't, D3 has PUA)
- D3 PUA coverage: hard-fail if atomic mapping size < 252 entries; data integrity check, not a warning

### Claude's Discretion
- Exact tqdm integration approach and progress bar format
- JSON schema for `validation_report.json` (as long as it has per-condition pass/fail and the 4 check results)
- How to handle HF `load_dataset` network errors vs. stall (both print the manual instructions)
- Character distribution display format

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Download and validate Abdou/arabic-tashkeel-dataset (1.5M examples) | HF datasets>=3.0 `load_dataset` with built-in cache reuse; dataset confirmed at 1.46M train + 30K valid + 15K test rows; columns: `vocalized`, `non_vocalized`, `source` |
| DATA-02 | Build D1 parquet shards (diacritized, raw Unicode combining chars) | `process_condition("d1", vocalized)` already implemented in `build_dataset.py`; pyarrow>=21.0 `write_table` + `read_table` round-trip confirmed safe |
| DATA-03 | Build D2 parquet shards (harakat stripped — control condition) | `strip_harakat()` regex `[\u064B-\u0652\u0670]` already implemented; must strip from the `vocalized` column (not `non_vocalized`) to ensure same base texts as D1 |
| DATA-04 | Build D3 parquet shards (atomic PUA encoding) | `build_atomic_mapping()` + `apply_atomic_encoding()` already implemented; 252-entry hard-fail integrity check needed during validation |
| DATA-05 | Compute homograph collision statistics at corpus scale | Word-level stats already written; needs extension: 128-token context-window metric + JSON sidecar with top-50 ambiguous words |
</phase_requirements>

---

## Summary

Phase 1 has substantial existing implementation. `build_dataset.py` contains working code for all three conditions (D1/D2/D3), the atomic mapping builder, and a word-level collision stats function. The gap between current code and phase completion is small but specific: (1) the `compute_collision_stats()` function outputs only `collision_stats.txt` and only computes word-level ambiguity — it needs a 128-token context-window metric and a JSON sidecar with top-50 words; (2) there is no validation suite (neither inline in `build_dataset.py` nor a standalone `validate_dataset.py`); (3) the download lacks a tqdm progress indicator and exit-time manual instructions.

The dataset is confirmed: `Abdou/arabic-tashkeel-dataset` on HuggingFace has three columns (`vocalized`, `non_vocalized`, `source`), 1.46M train / 30.2K valid / 15.1K test rows, MIT license, ~4GB total. The `load_dataset("Abdou/arabic-tashkeel-dataset", split="train")` call in `build_dataset.py` already targets the right split. HF datasets library provides automatic cache reuse via `~/.cache/huggingface/datasets` — no extra cache layer needed.

**Primary recommendation:** Extend `build_dataset.py` with the JSON collision stats sidecar + context-window metric, add inline validation calls after each `process_condition()`, and create `validate_dataset.py` as a thin standalone wrapper. These are additive changes that do not touch the working parquet-writing core.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| datasets | >=3.0.0 | HF dataset download and caching | Only library with direct `Abdou/arabic-tashkeel-dataset` access; handles cache transparently |
| pyarrow | >=21.0.0 | Parquet shard read/write | Project-established; `pa.table({"text": batch})` + `pq.write_table()` already used throughout |
| tqdm | (bundled with datasets, or standalone) | Download progress bar | Standard Python progress library; HF datasets uses it internally |
| json (stdlib) | stdlib | collision_stats.json + validation_report.json | No dependency needed |
| collections.defaultdict (stdlib) | stdlib | Collision stat accumulation | Already used in `compute_collision_stats()` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| re (stdlib) | stdlib | HARAKAT_RE strip for D2 | Already used; regex pattern covers U+064B-U+0652 + U+0670 |
| pathlib.Path (stdlib) | stdlib | All cache path construction | Already established via `BASE_CACHE` convention |
| argparse (stdlib) | stdlib | CLI `--condition` flag | Already used; `validate_dataset.py` needs same pattern |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pq.write_table` per shard | `pq.ParquetWriter` streaming | Streaming writer avoids full-batch memory; but batch approach works for 50K doc shards and is already tested |
| HF `split="train"` only | Download all splits and merge | Current code ignores valid/test splits from HF — acceptable since 1.46M train rows is sufficient |

**Installation (already in pyproject.toml):**
```bash
uv sync
```

---

## Architecture Patterns

### Recommended Project Structure

```
build_dataset.py          # Extended: inline validation + JSON stats sidecar (existing file)
validate_dataset.py       # New standalone script; thin wrapper around validation logic
~/.cache/autoresearch-arabic/
├── atomic_mapping.json   # PUA mapping (already produced by build_dataset.py)
├── collision_stats.txt   # Human-readable stats (already produced)
├── collision_stats.json  # NEW: machine-readable sidecar for Phase 4
├── validation_report.json # NEW: per-condition pass/fail record
├── d1/
│   ├── metadata.txt
│   └── data/
│       ├── shard_00000.parquet
│       └── shard_NNNNN.parquet  # last = val shard
├── d2/
│   └── ...
└── d3/
    └── ...
```

### Pattern 1: Inline Validation After Each Condition

**What:** After `process_condition()` writes shards, immediately run validation checks on the result.
**When to use:** Always — no flag needed. If validation fails, abort with a clear error before proceeding to the next condition.

```python
# Source: CONTEXT.md locked decisions
def validate_condition(condition: str) -> dict:
    """
    Run all 4 mandatory checks for one condition.
    Returns per-check results dict. Raises SystemExit on hard-fail checks.
    """
    results = {}
    data_dir = BASE_CACHE / condition / "data"
    meta_path = BASE_CACHE / condition / "metadata.txt"

    # Check 1: all shards load without exception
    shards = sorted(data_dir.glob("shard_*.parquet"))
    load_ok = True
    for shard in shards:
        try:
            pq.read_table(shard)
        except Exception as e:
            load_ok = False
            print(f"  FAIL: {shard.name} failed to load: {e}")
    results["shards_loadable"] = load_ok

    # Check 2: row count matches metadata.txt
    meta = dict(line.strip().split("=") for line in meta_path.read_text().splitlines() if "=" in line)
    expected_train = int(meta["train_docs"])
    expected_val = int(meta["val_docs"])
    actual_train = sum(
        pq.read_table(s).num_rows for s in shards if s.name != meta["val_filename"]
    )
    actual_val = pq.read_table(data_dir / meta["val_filename"]).num_rows
    count_ok = (actual_train == expected_train) and (actual_val == expected_val)
    results["row_count_matches"] = count_ok

    # Check 3: zero empty or null texts
    empty_ok = True
    for shard in shards:
        col = pq.read_table(shard).column("text").to_pylist()
        if any(t is None or t == "" for t in col):
            empty_ok = False
    results["no_empty_texts"] = empty_ok

    # Check 4: character distribution (visual assertion only — not a hard fail)
    results["char_distribution"] = _compute_char_distribution(condition, data_dir)

    return results
```

### Pattern 2: Context-Window Collision Probability

**What:** Sample random 128-token windows from the corpus, tokenize naively (split on whitespace), check what fraction of tokens are homographically ambiguous (undiacritized form maps to >1 diacritized variant).
**When to use:** Called once inside `compute_collision_stats()` after building the `diacritized_forms` map.

```python
# Source: CONTEXT.md locked decisions
def context_window_collision_probability(
    diacritized_forms: dict,
    vocalized_texts: list[str],
    window_tokens: int = 128,
    n_samples: int = 10_000,
    seed: int = 42,
) -> float:
    """
    Estimate: in a random 128-token window, what fraction of tokens are ambiguous?
    Tokens defined as whitespace-split words (consistent with collision map build).
    """
    import random
    rng = random.Random(seed)
    ambiguous_token_count = 0
    total_token_count = 0

    for _ in range(n_samples):
        doc = rng.choice(vocalized_texts)
        words = doc.split()
        if len(words) < window_tokens:
            continue
        start = rng.randint(0, len(words) - window_tokens)
        window = words[start:start + window_tokens]
        for w in window:
            stripped = HARAKAT_RE.sub('', w)
            total_token_count += 1
            if len(diacritized_forms.get(stripped, set())) > 1:
                ambiguous_token_count += 1

    return ambiguous_token_count / total_token_count if total_token_count > 0 else 0.0
```

### Pattern 3: JSON Stats Sidecar

**What:** After `collision_stats.txt` is written, write a parallel `collision_stats.json` with structured data for Phase 4 paper tables.
**When to use:** Always, at end of `compute_collision_stats()`.

```python
# Source: CONTEXT.md locked decisions
collision_json = {
    "total_undiacritized_forms": len(diacritized_forms),
    "total_diacritized_forms": sum(collision_counts),
    "ambiguous_form_count": ambiguous,
    "ambiguous_form_pct": round(100 * ambiguous / len(diacritized_forms), 2),
    "avg_collision_rate": round(avg_collision, 4),
    "max_collision_rate": max_collision,
    "total_words_analyzed": total_words,
    "context_window_tokens": 128,
    "context_window_ambiguous_pct": round(100 * cw_prob, 2),
    "top_50_ambiguous": [
        {"word": word, "variants": sorted(variants)}
        for word, variants in top_ambiguous_50
    ],
}
json_path = BASE_CACHE / "collision_stats.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(collision_json, f, ensure_ascii=False, indent=2)
```

### Pattern 4: tqdm Download Wrapper

**What:** Wrap the `load_dataset` call with a tqdm spinner (indeterminate) to indicate liveness, since HF's internal progress bars can stall silently on slow networks. On any exception (including KeyboardInterrupt), print manual download instructions.
**When to use:** Around the `load_dataset("Abdou/arabic-tashkeel-dataset", split="train")` call.

```python
# Source: CONTEXT.md locked decisions + tqdm docs
import threading
from tqdm import tqdm

MANUAL_INSTRUCTIONS = """
Manual download instructions:
  1. Visit https://huggingface.co/datasets/Abdou/arabic-tashkeel-dataset
  2. Download the Parquet files from the 'Files and versions' tab
  3. Place them in ~/.cache/huggingface/datasets/Abdou___arabic-tashkeel-dataset/
  4. Re-run this script (load_dataset will detect the local files)
"""

def load_dataset_with_progress(name: str, split: str):
    result = [None]
    error = [None]

    def _load():
        try:
            from datasets import load_dataset
            result[0] = load_dataset(name, split=split)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    try:
        with tqdm(desc=f"Downloading {name}", unit="it", dynamic_ncols=True) as pbar:
            while t.is_alive():
                t.join(timeout=0.5)
                pbar.update(0)  # keep bar alive / show spinner
    except KeyboardInterrupt:
        print(MANUAL_INSTRUCTIONS)
        raise

    if error[0] is not None:
        print(MANUAL_INSTRUCTIONS)
        raise error[0]
    return result[0]
```

### Anti-Patterns to Avoid

- **Downloading non_vocalized column for D2**: D2 must be produced by stripping harakat from the `vocalized` column — NOT by using the `non_vocalized` column directly. The pre-stripped column may use different whitespace or normalization; deriving D2 from D1 ensures identical tokenization boundaries.
- **Building validation as an opt-in flag**: Validation must run automatically (no `--validate` flag). A skipped validation is an undetected corruption.
- **top-20 vs top-50 in JSON**: The `.txt` file shows top-20 for human readability. The `.json` file MUST have top-50 (Phase 4 needs examples for paper tables).
- **Soft warning for PUA < 252**: The 252-entry check is a hard `sys.exit(1)`, not a warning. It guards against silent encoding regression.
- **Writing `collision_stats.json` before `diacritized_forms` is fully populated**: The context-window probability function needs the full map. Compute everything first, then write both files.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parquet I/O | Custom binary format | `pyarrow.parquet` | Already in project stack; handles column typing, compression, schema evolution |
| Unicode normalization | Custom Arabic char tables | Python `unicodedata` + explicit regex ranges | U+064B-U+0652 range is stable Unicode standard; regex already correct in codebase |
| Dataset download + caching | Custom HTTP downloader with retry | `datasets.load_dataset()` | HF library handles partial downloads, cache fingerprinting, format conversion |
| JSON serialization | Custom text format for stats | `json.dump(..., ensure_ascii=False)` | Arabic chars must survive JSON round-trip; `ensure_ascii=False` is the one required flag |

**Key insight:** The parquet-writing core and PUA encoding are already working and tested. The only new code is in the stats/validation layer — stay additive, don't refactor what works.

---

## Common Pitfalls

### Pitfall 1: D2 Derived from Wrong Column

**What goes wrong:** Using `non_vocalized` (the HF dataset's pre-stripped column) for D2 instead of stripping from `vocalized`.
**Why it happens:** `non_vocalized` looks correct but may have different whitespace normalization or minor text differences from the source.
**How to avoid:** `build_dataset.py` already calls `process_condition("d2", vocalized)` — do not change this.
**Warning signs:** If D2 collision stats show 0% ambiguity — all forms already have 1 variant, meaning the word mapping was built from a different text than D1.

### Pitfall 2: JSON ensure_ascii=False Omission

**What goes wrong:** `json.dump()` without `ensure_ascii=False` outputs `\u0623\u0644\u0645` escape sequences instead of Arabic characters. Phase 4 paper generation reads raw strings from this file.
**Why it happens:** Python's `json.dump` defaults to `ensure_ascii=True`.
**How to avoid:** Always pass `ensure_ascii=False` when writing any Arabic-containing JSON in this project.
**Warning signs:** `collision_stats.json` opened in a text editor shows only `\uXXXX` sequences for the top-50 words.

### Pitfall 3: HF Download Stall With No Feedback

**What goes wrong:** `load_dataset()` hangs silently for minutes with no output. User cannot tell if it is downloading or deadlocked. Ctrl+C leaves no recovery instructions.
**Why it happens:** HF downloads can stall on slow/flaky networks; the library suppresses progress on redirects or S3 throttling.
**How to avoid:** Wrap download in a thread + tqdm heartbeat (see Pattern 4). Print `MANUAL_INSTRUCTIONS` in the `except (KeyboardInterrupt, Exception)` handler.
**Warning signs:** Script runs for >60 seconds with no stdout output.

### Pitfall 4: PUA Mapping Count Regression

**What goes wrong:** A change to `HARAKAT_CODEPOINTS` or the letter range in `build_atomic_mapping()` reduces the mapping below 252 entries, silently producing a broken D3.
**Why it happens:** The function is called at runtime and its output is not checked before encoding.
**How to avoid:** Hard-fail immediately after `build_atomic_mapping()` returns if `len(atomic_mapping) < 252`.
**Warning signs:** `atomic_mapping.json` has fewer keys than expected; D3 texts contain raw combining chars that weren't encoded.

### Pitfall 5: Validation Row Count vs Shard File Count

**What goes wrong:** `metadata.txt` says 1,470,000 train docs but the actual shards have 1,469,999 (off-by-one from integer division). The row count check fails spuriously.
**Why it happens:** `build_dataset.py` computes `n_val = max(1, int(n * VAL_RATIO))` — floating point. The count written to `metadata.txt` is the list length AFTER split, which is correct. Just verify the check reads `train_docs` from metadata (not recomputes from VAL_RATIO).
**How to avoid:** Validation check must sum actual rows across train shards and compare against the `train_docs` field written by `process_condition()` — not recompute from VAL_RATIO.

### Pitfall 6: context-window metric biased by short documents

**What goes wrong:** Short documents (< 128 words) are skipped in the sampling loop. If the corpus has many short docs, the sample is biased toward long classical texts (Tashkeela/Shamela), which may have different ambiguity rates.
**Why it happens:** Naive sampling skips shorts.
**How to avoid:** Accept the bias — it is documented in the paper (short docs excluded from context-window metric, n_samples=10,000 from docs with >=128 words). Add a comment in code noting the exclusion.

---

## Code Examples

Verified patterns from official sources and existing codebase:

### Read Parquet Shard (validation check)
```python
# Source: pyarrow docs (arrow.apache.org/docs/python/parquet.html)
import pyarrow.parquet as pq

table = pq.read_table("/path/to/shard_00000.parquet")
texts = table.column("text").to_pylist()
assert all(t is not None and t != "" for t in texts)
```

### Write JSON with Arabic Characters
```python
# Source: Python stdlib json docs
import json

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

### Detect HF Cache Warmth (No Extra Code Needed)
```python
# Source: HuggingFace datasets docs (huggingface.co/docs/datasets/cache)
# load_dataset reuses cache automatically — REUSE_DATASET_IF_EXISTS is the default.
# No explicit check needed. If cache is warm, the call returns immediately.
from datasets import load_dataset
ds = load_dataset("Abdou/arabic-tashkeel-dataset", split="train")
```

### CLI Pattern (consistent with existing scripts)
```python
# Source: existing build_dataset.py + prepare.py
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--condition", choices=["d1", "d2", "d3", "all"], default="all")
args = parser.parse_args()
```

### validate_dataset.py Entry Point
```python
# Source: CONTEXT.md decisions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate autoresearch-arabic dataset conditions")
    parser.add_argument("--condition", choices=["d1", "d2", "d3", "all"], default="all")
    args = parser.parse_args()
    conditions = ["d1", "d2", "d3"] if args.condition == "all" else [args.condition]
    for cond in conditions:
        results = validate_condition(cond)
        write_validation_report(cond, results)
    # Exit 1 if any condition failed any check
    sys.exit(0 if all_passed else 1)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom retry loop for HF downloads | Thread + tqdm heartbeat, print instructions on failure | This phase | User always sees progress; Ctrl+C is safe |
| top-20 most ambiguous words (txt only) | top-50 in JSON sidecar | This phase | Phase 4 paper generation can pull structured data |
| Word-level collision rate only | Word-level + 128-token context-window probability | This phase | Context-window metric is the paper's main disambiguation-tax quantifier |
| No validation after build | Inline validation after each condition + standalone `validate_dataset.py` | This phase | Phase 2 can gate on `validation_report.json` |

**Deprecated/outdated:**
- `--skip-stats` flag: still useful for rapid iteration during development, but validation must NOT have an equivalent skip — validation always runs.

---

## Open Questions

1. **HF download timeout on current network**
   - What we know: STATE.md documents "HuggingFace downloads stall on current network" as an active blocker. CONTEXT.md decision: no hard fallback in code, just print instructions.
   - What's unclear: whether the stall is a DNS issue, S3 throttle, or network-level block.
   - Recommendation: The tqdm heartbeat + manual instructions approach is the correct implementation. The actual download may require a VPN or Kaggle alternative outside the script — document this in the script's docstring.

2. **D3 exact PUA mapping count**
   - What we know: The current `build_atomic_mapping()` produces approximately 700+ entries (28 letters × 9 harakat + double combos + standalone harakat), which is well above 252. The 252 threshold is a minimum correctness gate.
   - What's unclear: The exact count was noted as "~252 combos" in requirements but the code actually maps many more combinations (letter+shaddah+harakah doubles). The 252 check guards against a near-zero regression, not a precise target.
   - Recommendation: Hard-fail at < 252 as specified. Document actual count in the build output (already printed: "Atomic mapping: X entries").

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (not yet installed — Wave 0 gap) |
| Config file | none — see Wave 0 |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | `load_dataset` returns dataset with `vocalized` + `non_vocalized` columns | smoke | `uv run pytest tests/test_pipeline.py::test_dataset_columns -x` | Wave 0 |
| DATA-02 | D1 parquet shards exist and load without error, contain harakat chars | unit | `uv run pytest tests/test_pipeline.py::test_d1_shards -x` | Wave 0 |
| DATA-03 | D2 shards have zero harakat codepoints (U+064B-U+0652, U+0670) | unit | `uv run pytest tests/test_pipeline.py::test_d2_stripped -x` | Wave 0 |
| DATA-04 | D3 shards contain PUA codepoints; atomic_mapping.json has >= 252 entries | unit | `uv run pytest tests/test_pipeline.py::test_d3_pua -x` | Wave 0 |
| DATA-05 | collision_stats.json exists with `top_50_ambiguous` list of >= 50 items and `context_window_ambiguous_pct` field | unit | `uv run pytest tests/test_pipeline.py::test_collision_stats_json -x` | Wave 0 |
| (validation) | validation_report.json exists; all per-condition checks pass | integration | `uv run pytest tests/test_pipeline.py::test_validation_report -x` | Wave 0 |

**Note:** Because the dataset itself is not in the repo (it lives in `~/.cache`), tests must either (a) use a small fixture built from a 100-row subset of D1/D2/D3 parquets, or (b) be marked `@pytest.mark.requires_cache` and skipped in CI. The planner should decide which approach to use; either is valid.

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_pipeline.py -x -q`
- **Per wave merge:** `uv run pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_pipeline.py` — covers DATA-01 through DATA-05 + validation report
- [ ] `tests/conftest.py` — shared fixtures: tiny parquet shard builder (100 rows of real Arabic text), path helpers
- [ ] `tests/__init__.py` — empty package marker
- [ ] Framework install: `uv add --dev pytest` — pytest not in pyproject.toml dependencies

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `build_dataset.py` — all function names, constants, and patterns confirmed from source
- Direct code inspection of `prepare.py` — confirms integration points (metadata.txt parsing, parquet path conventions)
- `01-CONTEXT.md` — locked decisions are authoritative for this phase
- HuggingFace dataset page (fetched): confirmed 3 columns, 1.46M train rows, MIT license

### Secondary (MEDIUM confidence)
- Apache Arrow docs (arrow.apache.org) — `pq.write_table` / `pq.read_table` API confirmed current
- HuggingFace datasets docs (huggingface.co/docs/datasets/cache) — cache reuse via REUSE_DATASET_IF_EXISTS confirmed

### Tertiary (LOW confidence — marked for validation)
- tqdm threading approach for download heartbeat: pattern is standard Python but not from official HF docs — validate that the thread approach doesn't conflict with HF's own internal tqdm usage

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in pyproject.toml, versions pinned, code running
- Architecture: HIGH — patterns derived from existing working code in `build_dataset.py`; only extensions needed
- Pitfalls: HIGH for D2-column and JSON issues (seen in practice); MEDIUM for context-window sampling bias (theoretical)

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (HF datasets API is stable; pyarrow parquet format is stable)
