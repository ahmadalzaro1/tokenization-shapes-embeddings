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


def load_dataset_with_progress(name: str, split: str):
    """
    Load a HuggingFace dataset in a background thread with a tqdm heartbeat.
    If the download stalls, the user sees the bar is alive vs. dead.
    On any exit (KeyboardInterrupt or exception), prints manual download instructions.

    Note: Uses load_dataset() which auto-reuses ~/.cache/huggingface/datasets — no extra
    cache layer needed. If cache is warm, returns immediately.
    """
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

    # Transform texts based on condition
    if condition == "d1":
        processed = texts  # keep as-is (diacritized)
    elif condition == "d2":
        print(f"  Stripping harakat from {len(texts):,} documents...")
        processed = [strip_harakat(t) for t in texts]
    elif condition == "d3":
        print(f"  Applying atomic encoding to {len(texts):,} documents...")
        processed = [apply_atomic_encoding(t, atomic_mapping) for t in texts]
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


def compute_collision_stats(vocalized: list[str], non_vocalized: list[str]):
    """Compute homograph collision rate between diacritized and stripped forms."""
    from collections import defaultdict

    diacritized_forms = defaultdict(set)
    total_words = 0

    for voc, non_voc in zip(vocalized, non_vocalized):
        voc_words = voc.split()
        non_voc_words = non_voc.split()
        for vw, nw in zip(voc_words, non_voc_words):
            diacritized_forms[nw].add(vw)
            total_words += 1

    collision_counts = [len(v) for v in diacritized_forms.values()]
    ambiguous = sum(1 for c in collision_counts if c > 1)
    avg_collision = sum(collision_counts) / len(collision_counts) if collision_counts else 0
    max_collision = max(collision_counts) if collision_counts else 0

    # Find the most ambiguous words
    top_ambiguous = sorted(diacritized_forms.items(), key=lambda x: len(x[1]), reverse=True)[:20]

    stats_path = BASE_CACHE / "collision_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Total unique undiacritized forms: {len(diacritized_forms):,}\n")
        f.write(f"Total unique diacritized forms: {sum(collision_counts):,}\n")
        f.write(f"Ambiguous forms (>1 diacritized variant): {ambiguous:,} ({100*ambiguous/len(diacritized_forms):.1f}%)\n")
        f.write(f"Average collision rate: {avg_collision:.2f}\n")
        f.write(f"Max collision rate: {max_collision}\n")
        f.write(f"Total words analyzed: {total_words:,}\n")
        f.write(f"\nTop 20 most ambiguous words:\n")
        for word, variants in top_ambiguous:
            f.write(f"  {word} ({len(variants)} variants): {' | '.join(sorted(variants)[:10])}\n")

    print(f"\nCollision statistics:")
    print(f"  Unique undiacritized forms: {len(diacritized_forms):,}")
    print(f"  Unique diacritized forms:   {sum(collision_counts):,}")
    print(f"  Ambiguous (>1 variant):     {ambiguous:,} ({100*ambiguous/len(diacritized_forms):.1f}%)")
    print(f"  Average collision rate:     {avg_collision:.2f}x")
    print(f"  Max collision:              {max_collision}x")
    print(f"  Saved to: {stats_path}")


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

    print("\nDone! Dataset conditions ready at:", BASE_CACHE)
    print("Next: run 'uv run prepare.py --condition d1' to train tokenizer for D1")


if __name__ == "__main__":
    main()
