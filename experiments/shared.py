"""Shared utilities for Phase 5 experiments."""

import json
import os
import re
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Path constants (adapted from paper/run_fixed_arch_ablation.py)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "train.py"
SEARCH_RESULTS_PATH = ROOT / "search_results.json"
BASELINE_JSON = Path.home() / ".cache" / "autoresearch-arabic" / "baseline_results.json"

# ---------------------------------------------------------------------------
# Regex constant
# ---------------------------------------------------------------------------
ARABIC_LETTER_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]")

# ---------------------------------------------------------------------------
# Hyperparameter keys (from paper/run_fixed_arch_ablation.py lines 19-35)
# ---------------------------------------------------------------------------
PARAM_KEYS = [
    "ASPECT_RATIO",
    "HEAD_DIM",
    "WINDOW_PATTERN",
    "TOTAL_BATCH_SIZE",
    "EMBEDDING_LR",
    "UNEMBEDDING_LR",
    "MATRIX_LR",
    "SCALAR_LR",
    "WEIGHT_DECAY",
    "ADAM_BETAS",
    "WARMUP_RATIO",
    "WARMDOWN_RATIO",
    "FINAL_LR_FRAC",
    "DEPTH",
    "DEVICE_BATCH_SIZE",
]


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------
def sh(cmd, env=None):
    return subprocess.check_output(cmd, text=True, cwd=ROOT, env=env)


# ---------------------------------------------------------------------------
# Config extraction and patching
# ---------------------------------------------------------------------------
def extract_params_from_commit(commit: str) -> dict[str, str]:
    """Extract full hyperparameters from a git commit via regex."""
    source = sh(["git", "show", f"{commit}:train.py"])
    params: dict[str, str] = {}
    for key in PARAM_KEYS:
        match = re.search(rf"^{key}\s*=\s*(.+)$", source, flags=re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not find {key} in commit {commit}")
        params[key] = match.group(1).strip()
    return params


def patch_train(params: dict[str, str]) -> None:
    """Regex-replace hyperparameters in train.py."""
    text = TRAIN_PATH.read_text(encoding="utf-8")
    for key, value in params.items():
        text, count = re.subn(
            rf"^{key}\s*=\s*.+$",
            f"{key} = {value}",
            text,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(f"Failed to patch {key}")
    TRAIN_PATH.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Metric parsing (extended beyond run_fixed_arch_ablation.py's 5 patterns)
# ---------------------------------------------------------------------------
def parse_metrics(log_text: str) -> dict[str, float]:
    """Extract metrics from train.py stdout. Returns None for missing keys."""
    patterns = {
        "val_bpb": r"^val_bpb:\s+([0-9.]+)$",
        "peak_vram_mb": r"^peak_vram_mb:\s+([0-9.]+)$",
        "total_tokens_M": r"^total_tokens_M:\s+([0-9.]+)$",
        "num_params_M": r"^num_params_M:\s+([0-9.]+)$",
        "training_seconds": r"^training_seconds:\s+([0-9.]+)$",
        "total_eval_nats": r"^total_eval_nats:\s+([0-9.]+)$",
        "total_eval_bytes": r"^total_eval_bytes:\s+([0-9]+)$",
        "total_valid_tokens": r"^total_valid_tokens:\s+([0-9]+)$",
        "num_steps": r"^num_steps:\s+([0-9]+)$",
    }
    result: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text, flags=re.MULTILINE)
        if match:
            result[key] = float(match.group(1))
    return result


# ---------------------------------------------------------------------------
# Base letter counting
# ---------------------------------------------------------------------------
def count_val_base_letters(data_dir: str, val_filename: str) -> int:
    """Count Arabic base letters in the val parquet shard."""
    val_path = Path(data_dir) / val_filename
    table = pq.read_table(val_path)
    total = 0
    for text in table.column("text").to_pylist():
        total += len(ARABIC_LETTER_RE.findall(text))
    return total


# ---------------------------------------------------------------------------
# Best configs from search_results.json
# ---------------------------------------------------------------------------
def _load_best_configs() -> dict:
    with open(SEARCH_RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


BEST_CONFIGS = _load_best_configs()
