"""
Build Arabic harakat experiment datasets from Abdou/arabic-tashkeel-dataset.
Creates 3 conditions in autoresearch-compatible parquet format:

  D1 (diacritized): Full Arabic text with all harakat preserved
  D2 (stripped):    Same text with all harakat removed
  D3 (atomic):     Letter+harakah pairs mapped to single PUA codepoints

Each condition gets its own cache directory with train/val shards.

Usage:
    uv run build_dataset.py              # build all 3 conditions
    uv run build_dataset.py --condition d1  # build only D1
"""

import argparse
import json
import os
import re
import sys
import threading
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_CACHE = Path.home() / ".cache" / "autoresearch-arabic"

# All Arabic combining marks (harakat + shadda + sukun + superscript alef)
HARAKAT_RANGE = r'[\u064B-\u0652\u0670]'
HARAKAT_RE = re.compile(HARAKAT_RANGE)

# Individual harakat codepoints
HARAKAT_CODEPOINTS = [
    '\u064B',  # fathatan
    '\u064C',  # dammatan
    '\u064D',  # kasratan
    '\u064E',  # fathah
    '\u064F',  # dammah
    '\u0650',  # kasrah
    '\u0651',  # shaddah
    '\u0652',  # sukun
    '\u0670',  # superscript alef
]

# Arabic base letter range (standard + extended)
ARABIC_LETTER_RE = re.compile(r'[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]')

# PUA (Private Use Area) base for atomic encoding
# We use U+E000-U+E8FF — enough for 28 letters × 9 harakat × combinations
PUA_BASE = 0xE000

def build_atomic_mapping() -> dict[str, str]:
    """
    Build mapping: (base_letter + harakah) -> single PUA codepoint.
    Also maps (base_letter + shaddah + harakah) -> single PUA codepoint.
    Standalone harakat (no preceding letter) map to their own PUA codepoint.
    """
    mapping = {}
    idx = 0

    # Standard Arabic letters
    letters = [chr(c) for c in range(0x0621, 0x063B)] + [chr(c) for c in range(0x0641, 0x064B)]

    # Single harakah combinations: letter + harakah
    for letter in letters:
        for harakah in HARAKAT_CODEPOINTS:
            key = letter + harakah
            mapping[key] = chr(PUA_BASE + idx)
            idx += 1

    # Double combinations: letter + shaddah + harakah
    shaddah = '\u0651'
    for letter in letters:
        for harakah in HARAKAT_CODEPOINTS:
            if harakah == shaddah:
                continue  # skip shaddah+shaddah
            key = letter + shaddah + harakah
            mapping[key] = chr(PUA_BASE + idx)
            idx += 1
        # letter + shaddah alone
        key = letter + shaddah
        mapping[key] = chr(PUA_BASE + idx)
        idx += 1

    # Standalone harakat (appear without preceding letter)
    for harakah in HARAKAT_CODEPOINTS:
        mapping[harakah] = chr(PUA_BASE + idx)
        idx += 1

    print(f"Atomic mapping: {idx} entries (PUA range U+E000-U+{PUA_BASE + idx - 1:04X})")
    return mapping


def apply_atomic_encoding(text: str, mapping: dict[str, str]) -> str:
    """
    Replace (letter+harakat) sequences with single PUA codepoints.
    Process longest matches first (letter+shaddah+harakah before letter+harakah).
    """
    result = []
    i = 0
    n = len(text)

    while i < n:
        # Try 3-char match (letter + shaddah + harakah)
        if i + 2 < n and text[i:i+3] in mapping:
            result.append(mapping[text[i:i+3]])
            i += 3
        # Try 2-char match (letter + harakah)
        elif i + 1 < n and text[i:i+2] in mapping:
            result.append(mapping[text[i:i+2]])
            i += 2
        # Try standalone harakah
        elif text[i] in mapping:
            result.append(mapping[text[i]])
            i += 1
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)


def build_reverse_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """Build PUA -> original sequence mapping for decoding."""
    return {v: k for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

MANUAL_INSTRUCTIONS = """
HuggingFace download stalled or failed.
Manual download instructions:
  1. Visit https://huggingface.co/datasets/Abdou/arabic-tashkeel-dataset
  2. Click 'Files and versions' tab and download all Parquet files
  3. Place them in: ~/.cache/huggingface/datasets/Abdou___arabic-tashkeel-dataset/
  4. Re-run this script (load_dataset will detect the local files automatically)

Alternative: use 'huggingface-cli download Abdou/arabic-tashkeel-dataset' on a stable connection.
"""


LOCAL_DATASET_DIR = Path(__file__).parent / "arabic-tashkeel-dataset"


def load_dataset_with_progress(name: str, split: str):
    """
    Load dataset from local directory if present, otherwise download from HuggingFace.

    Local path: arabic-tashkeel-dataset/data/{split}-*.parquet (next to this script).
    Falls back to HF download with a tqdm heartbeat spinner.
    """
    if LOCAL_DATASET_DIR.exists():
        data_files = sorted((LOCAL_DATASET_DIR / "data").glob(f"{split}-*.parquet"))
        if data_files:
            print(f"Loading {name} from local directory ({len(data_files)} shard(s))...")
            from datasets import load_dataset as _load
            return _load("parquet", data_files=[str(f) for f in data_files], split="train")

    result: list = [None]
    error: list = [None]

    def _load():
        try:
            from datasets import load_dataset
            result[0] = load_dataset(name, split=split)
        except Exception as e:  # noqa: BLE001
            error[0] = e

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    try:
        with tqdm(desc=f"Downloading {name}", unit="it", dynamic_ncols=True) as pbar:
            while t.is_alive():
                t.join(timeout=0.5)
                pbar.update(0)  # keep bar alive (spinner effect)
    except KeyboardInterrupt:
        print(MANUAL_INSTRUCTIONS)
        raise

    if error[0] is not None:
        print(MANUAL_INSTRUCTIONS)
        raise error[0]
    return result[0]


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

DOCS_PER_SHARD = 50_000  # ~50K documents per parquet shard
VAL_RATIO = 0.02         # 2% for validation


def strip_harakat(text: str) -> str:
    return HARAKAT_RE.sub('', text)


def process_condition(condition: str, texts: list[str], atomic_mapping: dict | None = None):
    """Process texts and write parquet shards for one condition."""
    data_dir = BASE_CACHE / condition / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Filter empty/None texts from source data
    texts = [t for t in texts if t and t.strip()]
    print(f"  {condition}: {len(texts):,} documents after filtering empties")

    # Transform texts based on condition
    if condition == "d1":
        processed = texts  # keep as-is (diacritized)
    elif condition == "d2":
        print(f"  Stripping harakat from {len(texts):,} documents...")
        processed = [strip_harakat(t) for t in texts]
        processed = [t for t in processed if t and t.strip()]
    elif condition == "d3":
        print(f"  Applying atomic encoding to {len(texts):,} documents...")
        processed = [apply_atomic_encoding(t, atomic_mapping) for t in texts]
        processed = [t for t in processed if t and t.strip()]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Split into train/val
    n = len(processed)
    n_val = max(1, int(n * VAL_RATIO))
    val_texts = processed[-n_val:]
    train_texts = processed[:-n_val]

    # Write train shards
    shard_idx = 0
    for start in range(0, len(train_texts), DOCS_PER_SHARD):
        batch = train_texts[start:start + DOCS_PER_SHARD]
        table = pa.table({"text": batch})
        path = data_dir / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(table, path)
        shard_idx += 1

    # Write val shard (always last shard index, matching autoresearch convention)
    val_shard_idx = shard_idx
    table = pa.table({"text": val_texts})
    val_path = data_dir / f"shard_{val_shard_idx:05d}.parquet"
    pq.write_table(table, val_path)

    print(f"  {condition}: {shard_idx} train shards + 1 val shard ({val_shard_idx})")
    print(f"  Train: {len(train_texts):,} docs | Val: {len(val_texts):,} docs")

    # Write metadata
    meta_path = BASE_CACHE / condition / "metadata.txt"
    meta_path.write_text(
        f"condition={condition}\n"
        f"train_docs={len(train_texts)}\n"
        f"val_docs={len(val_texts)}\n"
        f"train_shards={shard_idx}\n"
        f"val_shard={val_shard_idx}\n"
        f"val_filename=shard_{val_shard_idx:05d}.parquet\n"
    )

    return val_shard_idx


def context_window_collision_probability(
    diacritized_forms: dict,
    vocalized_texts: list[str],
    window_tokens: int = 128,
    n_samples: int = 10_000,
    seed: int = 42,
) -> float:
    """
    Estimate: in a random 128-token window, what fraction of tokens are homographically ambiguous?

    Tokens are whitespace-split words (consistent with diacritized_forms map construction).
    Short documents (< window_tokens words) are skipped — this biases the sample toward
    longer classical texts (Tashkeela/Shamela). Bias is documented; see paper methods section.

    Args:
        diacritized_forms: mapping from undiacritized word -> set of diacritized variants
        vocalized_texts: list of diacritized text strings (D1 source)
        window_tokens: context window size in tokens (128 matches typical small-model attention window)
        n_samples: number of windows to sample
        seed: random seed for reproducibility
    """
    import random
    rng = random.Random(seed)
    ambiguous_token_count = 0
    total_token_count = 0

    for _ in range(n_samples):
        doc = rng.choice(vocalized_texts)
        words = doc.split()
        if len(words) < window_tokens:
            continue  # skip short docs (see docstring bias note)
        start = rng.randint(0, len(words) - window_tokens)
        window = words[start:start + window_tokens]
        for w in window:
            stripped = HARAKAT_RE.sub('', w)
            total_token_count += 1
            if len(diacritized_forms.get(stripped, set())) > 1:
                ambiguous_token_count += 1

    return ambiguous_token_count / total_token_count if total_token_count > 0 else 0.0


def compute_collision_stats(vocalized: list[str], non_vocalized: list[str]):
    """Compute homograph collision rate between diacritized and stripped forms.

    Outputs:
      - collision_stats.txt: human-readable with top-20 most ambiguous words
      - collision_stats.json: machine-readable sidecar for Phase 4 paper tables;
        includes top-50 ambiguous words and 128-token context-window metric
    """
    from collections import defaultdict

    diacritized_forms = defaultdict(set)
    total_words = 0

    for voc, non_voc in zip(vocalized, non_vocalized):
        voc_words = voc.split()
        non_voc_words = non_voc.split()
        for vw, nw in zip(voc_words, non_voc_words):
            diacritized_forms[nw].add(vw)
            total_words += 1

    # --- Compute all stats after diacritized_forms is fully populated ---
    collision_counts = [len(v) for v in diacritized_forms.values()]
    ambiguous = sum(1 for c in collision_counts if c > 1)
    avg_collision = sum(collision_counts) / len(collision_counts) if collision_counts else 0
    max_collision = max(collision_counts) if collision_counts else 0

    # Context-window collision probability (must be after full map build)
    print("  Computing 128-token context-window collision probability (n=10,000 samples)...")
    cw_prob = context_window_collision_probability(diacritized_forms, vocalized)
    print(f"  Context-window ambiguous token rate: {100 * cw_prob:.2f}%")

    # Top ambiguous: top-50 for JSON, top-20 for txt
    top_ambiguous_all = sorted(
        diacritized_forms.items(), key=lambda x: len(x[1]), reverse=True
    )
    top_ambiguous_50 = top_ambiguous_all[:50]
    top_ambiguous_20 = top_ambiguous_all[:20]

    # Write human-readable txt (unchanged format)
    BASE_CACHE.mkdir(parents=True, exist_ok=True)
    stats_path = BASE_CACHE / "collision_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Total unique undiacritized forms: {len(diacritized_forms):,}\n")
        f.write(f"Total unique diacritized forms: {sum(collision_counts):,}\n")
        f.write(f"Ambiguous forms (>1 diacritized variant): {ambiguous:,} ({100*ambiguous/len(diacritized_forms):.1f}%)\n")
        f.write(f"Average collision rate: {avg_collision:.2f}\n")
        f.write(f"Max collision rate: {max_collision}\n")
        f.write(f"Total words analyzed: {total_words:,}\n")
        f.write(f"Context-window ambiguous token rate (128-tok): {100 * cw_prob:.2f}%\n")
        f.write(f"\nTop 20 most ambiguous words:\n")
        for word, variants in top_ambiguous_20:
            f.write(f"  {word} ({len(variants)} variants): {' | '.join(sorted(variants)[:10])}\n")

    # Write machine-readable JSON sidecar (Phase 4 paper tables)
    # CRITICAL: ensure_ascii=False so Arabic chars survive JSON round-trip
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
    print(f"  Saved collision stats to: {stats_path}")
    print(f"  Saved JSON sidecar to:    {json_path}")

    print(f"\nCollision statistics:")
    print(f"  Unique undiacritized forms: {len(diacritized_forms):,}")
    print(f"  Unique diacritized forms:   {sum(collision_counts):,}")
    print(f"  Ambiguous (>1 variant):     {ambiguous:,} ({100*ambiguous/len(diacritized_forms):.1f}%)")
    print(f"  Average collision rate:     {avg_collision:.2f}x")
    print(f"  Max collision:              {max_collision}x")
    print(f"  Context-window (128-tok):   {100 * cw_prob:.2f}% ambiguous")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _compute_char_distribution(condition: str, data_dir: Path) -> dict:
    """Count characters across all shards and return top-20 as repr(char) -> count."""
    import collections

    counter: collections.Counter = collections.Counter()
    shards = sorted(data_dir.glob("shard_*.parquet"))
    for shard in shards:
        table = pq.read_table(shard)
        for text in table.column("text").to_pylist():
            if text:
                counter.update(text)

    top_20 = counter.most_common(20)
    # Print a short preview of the top 10 chars for visual inspection
    preview = " ".join(repr(ch) for ch, _ in top_20[:10])
    print(f"  [{condition}] Top-20 chars: {preview} ...")

    return {repr(ch): cnt for ch, cnt in top_20}


def validate_condition(condition: str) -> dict:
    """Run 4 mandatory validation checks on a built condition.

    Checks:
      1. shards_loadable  — all shard files can be read by pyarrow
      2. row_count_matches — actual row count matches metadata.txt
      3. no_empty_texts   — no None or empty strings in any shard
      4. char_distribution — top-20 chars (visual, never hard-fails)

    D3 extra gate: atomic_mapping.json must have >= 252 entries.

    Calls sys.exit(1) on any hard-fail (checks 1-3, D3 gate).
    Returns a results dict with bool values for checks 1-3 and a dict for 4.
    """
    data_dir = BASE_CACHE / condition / "data"
    meta_path = BASE_CACHE / condition / "metadata.txt"

    if not data_dir.exists():
        print(f"  [{condition}] FAIL: data directory not found: {data_dir}")
        sys.exit(1)
    if not meta_path.exists():
        print(f"  [{condition}] FAIL: metadata.txt not found: {meta_path}")
        sys.exit(1)

    shards = sorted(data_dir.glob("shard_*.parquet"))
    if not shards:
        print(f"  [{condition}] FAIL: no shard files found in {data_dir}")
        sys.exit(1)

    # --- Parse metadata ---
    meta: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()

    val_filename = meta.get("val_filename", "")
    expected_train = int(meta.get("train_docs", 0))
    expected_val = int(meta.get("val_docs", 0))

    # --- Check 1: shards_loadable ---
    load_ok = True
    for shard in shards:
        try:
            pq.read_table(shard)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{condition}] FAIL shards_loadable: {shard.name} — {exc}")
            load_ok = False
    if not load_ok:
        sys.exit(1)
    print(f"  [{condition}] PASS shards_loadable")

    # --- Check 2: row_count_matches ---
    train_rows = 0
    val_rows = 0
    for shard in shards:
        table = pq.read_table(shard)
        if shard.name == val_filename:
            val_rows += table.num_rows
        else:
            train_rows += table.num_rows

    row_ok = (train_rows == expected_train) and (val_rows == expected_val)
    if not row_ok:
        print(
            f"  [{condition}] FAIL row_count_matches: "
            f"train expected {expected_train} got {train_rows}, "
            f"val expected {expected_val} got {val_rows}"
        )
        sys.exit(1)
    print(f"  [{condition}] PASS row_count_matches ({train_rows} train, {val_rows} val)")

    # --- Check 3: no_empty_texts ---
    empty_ok = True
    for shard in shards:
        table = pq.read_table(shard)
        texts = table.column("text").to_pylist()
        if any(t is None or t == "" for t in texts):
            print(f"  [{condition}] FAIL no_empty_texts: empty/None text found in {shard.name}")
            empty_ok = False
    if not empty_ok:
        sys.exit(1)
    print(f"  [{condition}] PASS no_empty_texts")

    # --- Check 4: char_distribution (visual only, no hard-fail) ---
    char_dist = _compute_char_distribution(condition, data_dir)

    # --- D3 integrity gate ---
    if condition == "d3":
        mapping_path = BASE_CACHE / "atomic_mapping.json"
        if not mapping_path.exists():
            print(f"  [d3] FAIL D3 gate: atomic_mapping.json not found at {mapping_path}")
            sys.exit(1)
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)
        if len(mapping) < 252:
            print(f"  [d3] FAIL D3 gate: atomic_mapping.json has {len(mapping)} entries (need >= 252)")
            sys.exit(1)
        print(f"  [d3] PASS D3 gate: atomic_mapping.json has {len(mapping)} entries")

    print(f"  [{condition}] All mandatory checks PASSED")

    return {
        "shards_loadable": True,
        "row_count_matches": True,
        "no_empty_texts": True,
        "char_distribution": char_dist,
    }


def write_validation_report(condition: str, results: dict) -> None:
    """Merge per-condition validation results into validation_report.json.

    Reads the existing report if present, updates the condition key, and
    writes back with ensure_ascii=False so Arabic chars survive round-trip.
    """
    report_path = BASE_CACHE / "validation_report.json"
    report: dict = {}
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            try:
                report = json.load(f)
            except json.JSONDecodeError:
                report = {}

    report[condition] = results

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Validation report updated: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Arabic harakat experiment datasets")
    parser.add_argument("--condition", choices=["d1", "d2", "d3", "all"], default="all",
                       help="Which condition(s) to build")
    parser.add_argument("--max-examples", type=int, default=-1,
                       help="Limit examples for testing (-1 = all)")
    parser.add_argument("--skip-stats", action="store_true",
                       help="Skip collision statistics computation")
    args = parser.parse_args()

    ds = load_dataset_with_progress("Abdou/arabic-tashkeel-dataset", split="train")
    print(f"Loaded {len(ds):,} examples")

    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
        print(f"Truncated to {len(ds):,} examples")

    # Extract columns
    vocalized = ds["vocalized"]
    non_vocalized = ds["non_vocalized"]

    # Collision stats (only need to compute once)
    if not args.skip_stats:
        print("\nComputing homograph collision statistics...")
        compute_collision_stats(vocalized, non_vocalized)

    # Build atomic mapping
    atomic_mapping = build_atomic_mapping()

    # Save atomic mapping for later decoding
    mapping_path = BASE_CACHE / "atomic_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: hex(ord(v)) for k, v in atomic_mapping.items()}
    with open(mapping_path, "w") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Saved atomic mapping to {mapping_path}")

    conditions = ["d1", "d2", "d3"] if args.condition == "all" else [args.condition]

    for condition in conditions:
        print(f"\nBuilding condition: {condition}")
        if condition == "d1":
            process_condition("d1", vocalized)
        elif condition == "d2":
            process_condition("d2", vocalized)  # strip from vocalized, not pre-stripped
        elif condition == "d3":
            process_condition("d3", vocalized, atomic_mapping)
        print(f"\nValidating condition: {condition}")
        val_results = validate_condition(condition)
        write_validation_report(condition, val_results)

    print("\nDone! Dataset conditions ready at:", BASE_CACHE)
    print("Next: run 'uv run prepare.py --condition d1' to train tokenizer for D1")


if __name__ == "__main__":
    main()
