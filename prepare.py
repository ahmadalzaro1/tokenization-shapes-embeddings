"""
Arabic harakat autoresearch data preparation.
Trains a BPE tokenizer on the selected condition (d1/d2/d3).

Usage:
    uv run prepare.py --condition d1   # train tokenizer for diacritized Arabic
    uv run prepare.py --condition d2   # train tokenizer for stripped Arabic
    uv run prepare.py --condition d3   # train tokenizer for atomic encoding

Data shards must exist at ~/.cache/autoresearch-arabic/<condition>/data/.
Run build_dataset.py first to create them.
"""

import argparse
import math
import os
import pickle
import sys
import time

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq
import rustbpe
import tiktoken

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 3 * 524288

# ---------------------------------------------------------------------------
# Configuration — set by --condition flag
# ---------------------------------------------------------------------------

CONDITION = os.environ.get("AUTORESEARCH_CONDITION", "d1")

BASE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-arabic")
VOCAB_SIZE = 8192
DEFAULT_VOCAB_SIZE = 8192  # stable constant — never mutated, used by get_dirs for path routing

# Arabic-aware BPE split pattern: handles Arabic letters, harakat, and PUA atomic tokens
SPLIT_PATTERN = r"""[\u0621-\u064A\u064B-\u0652\u0670\uE000-\uEFFF]+|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


def write_fertility_report(condition: str, vocab_size: int, fertility: float) -> None:
    """Merge fertility result into BASE_CACHE/fertility_report.json using read-update-write."""
    import json as _json
    report_path = os.path.join(BASE_CACHE, "fertility_report.json")
    report: dict = {}
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            try:
                report = _json.load(f)
            except _json.JSONDecodeError:
                report = {}
    if condition not in report:
        report[condition] = {}
    report[condition][str(vocab_size)] = round(fertility, 4)
    with open(report_path, "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Fertility report updated: {report_path}")


def get_dirs(condition: str, vocab_size: int = DEFAULT_VOCAB_SIZE) -> tuple[str, str, str]:
    cache_dir = os.path.join(BASE_CACHE, condition)
    data_dir = os.path.join(cache_dir, "data")
    if vocab_size == DEFAULT_VOCAB_SIZE:
        tokenizer_dir = os.path.join(cache_dir, "tokenizer")
    else:
        tokenizer_dir = os.path.join(cache_dir, f"tokenizer_{vocab_size}")
    return cache_dir, data_dir, tokenizer_dir


def get_val_shard(condition):
    """Read val shard index from metadata."""
    meta_path = os.path.join(BASE_CACHE, condition, "metadata.txt")
    if not os.path.exists(meta_path):
        # Fallback: highest shard index
        _, data_dir, _ = get_dirs(condition)
        shards = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
        return int(shards[-1].replace("shard_", "").replace(".parquet", ""))
    with open(meta_path) as f:
        for line in f:
            if line.startswith("val_shard="):
                return int(line.strip().split("=")[1])
    raise ValueError(f"val_shard not found in {meta_path}")


# Module-level globals set by init_condition()
CACHE_DIR = None
DATA_DIR = None
TOKENIZER_DIR = None
VAL_SHARD = None
VAL_FILENAME = None


def init_condition(condition: str, vocab_size: int = VOCAB_SIZE) -> None:
    global CACHE_DIR, DATA_DIR, TOKENIZER_DIR, VAL_SHARD, VAL_FILENAME, CONDITION, VOCAB_SIZE
    CONDITION = condition
    VOCAB_SIZE = vocab_size
    CACHE_DIR, DATA_DIR, TOKENIZER_DIR = get_dirs(condition, vocab_size)
    VAL_SHARD = get_val_shard(condition)
    VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"


def list_parquet_files():
    files = sorted(name for name in os.listdir(DATA_DIR) if name.endswith(".parquet") and not name.endswith(".tmp"))
    return [os.path.join(DATA_DIR, name) for name in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    parquet_paths = [path for path in list_parquet_files() if not path.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        parquet_file = pq.ParquetFile(filepath)
        for rg_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(rg_idx)
            for text in row_group.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards. Run build_dataset.py first.")
        sys.exit(1)

    print(f"Tokenizer: training BPE tokenizer for condition '{CONDITION}'...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(key): value for key, value in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name=f"rustbpe-arabic-{CONDITION}",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as handle:
        pickle.dump(enc, handle)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # Build token_bytes lookup for BPB evaluation
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes = np.array(token_bytes_list, dtype=np.int32)
    np.save(token_bytes_path, token_bytes)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check with Arabic text
    test = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ" if CONDITION == "d1" else "بسم الله الرحمن الرحيم"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

    # Fertility report
    sample_texts = list(text_iterator(max_chars=1_000_000))
    total_tokens = sum(len(enc.encode_ordinary(t)) for t in sample_texts)
    total_words = sum(len(t.split()) for t in sample_texts)
    fertility = total_tokens / total_words if total_words > 0 else 0
    print(f"Tokenizer: fertility = {fertility:.3f} tokens/word (on ~1M chars sample)")
    write_fertility_report(CONDITION, VOCAB_SIZE, fertility)


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=None):
        if tokenizer_dir is None:
            tokenizer_dir = TOKENIZER_DIR
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as handle:
            enc = pickle.load(handle)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes():
    path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing token_bytes lookup at {path}. Run prepare.py first.")
    token_bytes = np.load(path)
    return mx.array(token_bytes, dtype=mx.int32)


def _document_batches(split, tokenizer_batch_size=128):
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run build_dataset.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [path for path in parquet_paths if path != val_path]
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            parquet_file = pq.ParquetFile(filepath)
            for rg_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(rg_idx)
                batch = row_group.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, batch_size, seq_len, split, buffer_size=1000):
    assert split in ["train", "val"]
    row_capacity = seq_len + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    while True:
        all_rows = []
        for _ in range(batch_size):
            row = []
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for index, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = index
                        best_len = doc_len
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda index: len(doc_buffer[index]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])
                    pos += remaining
            all_rows.append(row[:row_capacity])

        row_array = mx.array(all_rows, dtype=mx.int32)
        inputs = row_array[:, :-1]
        targets = row_array[:, 1:]
        yield inputs, targets, epoch


def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes()
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    total_valid_tokens = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = mx.take(token_bytes, y_flat, axis=0)
        mask = nbytes > 0
        total_nats += mx.sum(loss_flat * mask).item()
        total_bytes += int(mx.sum(nbytes).item())
        total_valid_tokens += int(mx.sum(mask).item())
    if total_bytes == 0:
        return (float("inf"), 0.0, 0, 0)
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb, total_nats, total_bytes, total_valid_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare tokenizer for Arabic harakat experiment")
    parser.add_argument("--condition", choices=["d1", "d2", "d3"], required=True,
                       help="Dataset condition to prepare")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help=f"BPE vocabulary size including special tokens (default: {VOCAB_SIZE})")
    args = parser.parse_args()

    init_condition(args.condition, args.vocab_size)

    print(f"Condition: {args.condition}")
    print(f"Cache directory: {CACHE_DIR}")
    print()

    train_tokenizer()
    print()
    print(f"Done! Ready to train with condition '{args.condition}'.")
    print(f"Run: AUTORESEARCH_CONDITION={args.condition} uv run train.py")
