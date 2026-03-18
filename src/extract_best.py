"""
Extract best D3 config from results_d3.tsv and write search_results.json.
Run from the autoresearch/arabic-d3 branch so git show has access to commits.
"""

import json
import subprocess
import re
import sys

TSV_PATH = "results_d3.tsv"
OUTPUT_PATH = "search_results.json"

def parse_tsv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            row = dict(zip(header, parts))
            rows.append(row)
    return rows


def find_best_keep(rows: list[dict]) -> dict:
    keep_rows = [r for r in rows if r.get("status") == "keep" and float(r["val_bpb"]) > 0]
    if not keep_rows:
        raise ValueError("No keep rows found in TSV")
    return min(keep_rows, key=lambda r: float(r["val_bpb"]))


def extract_hyperparams_from_commit(commit: str) -> dict:
    try:
        source = subprocess.check_output(
            ["git", "show", f"{commit}:train.py"],
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not retrieve commit {commit}: {e}", file=sys.stderr)
        source = ""

    def extract(pattern: str, default=None):
        m = re.search(pattern, source, re.MULTILINE)
        if m:
            return m.group(1)
        return default

    params = {
        "ASPECT_RATIO": extract(r"^ASPECT_RATIO\s*=\s*(\d+)", "64"),
        "HEAD_DIM": extract(r"^HEAD_DIM\s*=\s*(\d+)", "96"),
        "WINDOW_PATTERN": extract(r'^WINDOW_PATTERN\s*=\s*"([^"]+)"', "SS"),
        "TOTAL_BATCH_SIZE_EXPR": extract(r"^TOTAL_BATCH_SIZE\s*=\s*(2\*\*\d+|\d+)", "2**15"),
        "EMBEDDING_LR": extract(r"^EMBEDDING_LR\s*=\s*([\d.]+)", "0.6"),
        "UNEMBEDDING_LR": extract(r"^UNEMBEDDING_LR\s*=\s*([\d.]+)", "0.004"),
        "MATRIX_LR": extract(r"^MATRIX_LR\s*=\s*([\d.]+)", "0.045"),
        "SCALAR_LR": extract(r"^SCALAR_LR\s*=\s*([\d.]+)", "0.5"),
        "WEIGHT_DECAY": extract(r"^WEIGHT_DECAY\s*=\s*([\d.]+)", "0.2"),
        "ADAM_BETAS": extract(r"^ADAM_BETAS\s*=\s*(\([\d.,\s]+\))", "(0.8, 0.95)"),
        "WARMUP_RATIO": extract(r"^WARMUP_RATIO\s*=\s*([\d.]+)", "0.0"),
        "WARMDOWN_RATIO": extract(r"^WARMDOWN_RATIO\s*=\s*([\d.]+)", "0.5"),
        "FINAL_LR_FRAC": extract(r"^FINAL_LR_FRAC\s*=\s*([\d.]+)", "0.0"),
        "DEPTH": extract(r"^DEPTH\s*=\s*(\d+)", "2"),
        "DEVICE_BATCH_SIZE": extract(r"^DEVICE_BATCH_SIZE\s*=\s*(\d+)", "16"),
    }
    # Convert numeric strings
    for key in ("ASPECT_RATIO", "HEAD_DIM", "DEPTH", "DEVICE_BATCH_SIZE"):
        if params[key] is not None:
            params[key] = int(params[key])
    for key in ("EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR",
                "WEIGHT_DECAY", "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC"):
        if params[key] is not None:
            params[key] = float(params[key])

    return params


def main():
    rows = parse_tsv(TSV_PATH)
    best = find_best_keep(rows)

    print(f"Best keep row:")
    print(f"  commit      : {best['commit']}")
    print(f"  val_bpb     : {best['val_bpb']}")
    print(f"  memory_gb   : {best['memory_gb']}")
    print(f"  description : {best['description']}")

    # Locate the actual git commit (TSV may store the pre-amend hash)
    best_commit = best["commit"]
    # Try to verify it exists
    try:
        subprocess.check_output(["git", "cat-file", "-e", best_commit], text=True)
        print(f"  git commit  : {best_commit} (verified)")
    except subprocess.CalledProcessError:
        # Fall back to HEAD of the keep chain
        result = subprocess.run(
            ["git", "log", "--all", "--oneline"],
            capture_output=True, text=True,
        )
        # Find by description keyword
        keyword = best["description"].split()[0]
        for line in result.stdout.splitlines():
            if keyword.lower() in line.lower():
                best_commit = line.split()[0]
                print(f"  git commit  : {best_commit} (resolved from log)")
                break

    params = extract_hyperparams_from_commit(best_commit)
    print(f"\nExtracted hyperparameters:")
    for k, v in params.items():
        print(f"  {k} = {v}")

    result = {
        "d3": {
            "condition": "d3",
            "val_bpb": float(best["val_bpb"]),
            "memory_gb": float(best["memory_gb"]),
            "commit": best_commit,
            "description": best["description"],
            "hyperparameters": params,
            "total_experiments": len(rows),
            "keep_count": sum(1 for r in rows if r.get("status") == "keep"),
            "crash_count": sum(1 for r in rows if r.get("status") == "crash"),
        }
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
