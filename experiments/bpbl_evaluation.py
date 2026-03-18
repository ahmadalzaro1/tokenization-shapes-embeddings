"""Experiment 3b: BPBL Robustness (5-seed evaluation).

Extends Experiment 3 by adding 2 new seeds (0, 1) to each condition.
Combines new results with the existing 3 seeds to produce robust statistics
(including the median) to mitigate the impact of catastrophic outliers.

Usage:
    cd /path/to/autoresearch-arabic
    uv run python experiments/exp3b_bpbl_robust.py
"""

import sys
from pathlib import Path

import json
import math
import os
import subprocess

import prepare
from shared import (
    BASELINE_JSON,
    ROOT,
    TRAIN_PATH,
    count_val_base_letters,
    extract_params_from_commit,
    parse_metrics,
    patch_train,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D1_COMMIT = "ab075a6"
D3_COMMIT = "532b0d1"
FERTILITIES = {"d1": 2.5189, "d3": 2.1934}
NEW_SEEDS = [0, 1, 3, 5, 7, 11, 13]

RESULTS_DIR = ROOT / "results"
EXISTING_RESULTS_JSON = RESULTS_DIR / "phase-05" / "bpbl_results.json"
ROBUST_RESULTS_JSON = RESULTS_DIR / "bpbl_results_robust.json"


def median(values: list[float]) -> float:
    """Compute the median of a list of floats (robust to single outliers)."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def main() -> None:
    print("============================================================", flush=True)
    print("Exp 3b: Robust BPBL Evaluation (Adding Seeds 0 and 1)", flush=True)
    print("============================================================", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Load existing results
    # ------------------------------------------------------------------
    if not EXISTING_RESULTS_JSON.exists():
        raise FileNotFoundError(f"Original results not found at {EXISTING_RESULTS_JSON}")

    print(f"Loading existing results from {EXISTING_RESULTS_JSON}...", flush=True)
    existing = json.loads(EXISTING_RESULTS_JSON.read_text(encoding="utf-8"))

    original_d1_bpbl = existing["d1"]["bpbl"]
    original_d3_bpbl = existing["d3"]["bpbl"]
    original_d1_seeds = existing["d1"]["seeds"]
    original_d3_seeds = existing["d3"]["seeds"]

    print(f"  Existing D1 seeds: {original_d1_seeds}, BPBL: {original_d1_bpbl}", flush=True)
    print(f"  Existing D3 seeds: {original_d3_seeds}, BPBL: {original_d3_bpbl}", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Pre-compute base letter count
    # ------------------------------------------------------------------
    print("\nInitializing D1 data paths for base letter counting...", flush=True)
    prepare.init_condition("d1")

    base_letter_count = count_val_base_letters(prepare.DATA_DIR, prepare.VAL_FILENAME)
    eval_words = 2_990_668
    base_letters_per_word = base_letter_count / eval_words

    base_letters_per_token = {
        cond: base_letters_per_word / fert for cond, fert in FERTILITIES.items()
    }

    # ------------------------------------------------------------------
    # Step 3: Extract hyperparameters and prepare backups
    # ------------------------------------------------------------------
    print("\nExtracting optimal hyperparameters...", flush=True)
    d1_params = extract_params_from_commit(D1_COMMIT)
    d3_params = extract_params_from_commit(D3_COMMIT)
    configs = {"d1": d1_params, "d3": d3_params}

    print("Backing up train.py and baseline_results.json...", flush=True)
    original_train = TRAIN_PATH.read_text(encoding="utf-8")
    baseline_backup = BASELINE_JSON.read_text(encoding="utf-8") if BASELINE_JSON.exists() else None

    new_results: dict = {
        "d1": {"seeds": [], "bpbl": []},
        "d3": {"seeds": [], "bpbl": []}
    }

    # ------------------------------------------------------------------
    # Step 4: Run new seeds
    # ------------------------------------------------------------------
    try:
        for condition in ["d1", "d3"]:
            for seed in NEW_SEEDS:
                print(f"\n{'='*60}", flush=True)
                print(f"Running new seed: condition={condition.upper()}, seed={seed}", flush=True)
                print(f"{'='*60}", flush=True)

                patch_train(configs[condition])

                env = os.environ.copy()
                env["AUTORESEARCH_CONDITION"] = condition
                env["AUTORESEARCH_SEED"] = str(seed)
                # AUTORESEARCH_SAVE_WTE intentionally omitted — don't overwrite Exp 4 embeddings

                run = subprocess.run(
                    ["uv", "run", "train.py"],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )

                log_text = run.stdout + run.stderr
                print(log_text[-3000:] if len(log_text) > 3000 else log_text, flush=True)

                if run.returncode != 0:
                    print(f"WARNING: Run failed for {condition} seed {seed}. Skipping.", flush=True)
                    continue

                metrics = parse_metrics(log_text)
                val_bpb = metrics.get("val_bpb")
                total_eval_nats = metrics.get("total_eval_nats")
                total_valid_tokens = metrics.get("total_valid_tokens")

                if val_bpb is None or total_eval_nats is None or total_valid_tokens is None:
                    print(f"WARNING: Missing metrics for {condition} seed {seed}. Skipping.", flush=True)
                    continue

                eval_base_letters = total_valid_tokens * base_letters_per_token[condition]
                bpbl = total_eval_nats / (eval_base_letters * math.log(2))

                print(f"  val_bpb={val_bpb:.6f}, total_eval_nats={total_eval_nats:.2f}, "
                      f"eval_base_letters={eval_base_letters:.0f}, BPBL={bpbl:.6f}", flush=True)

                new_results[condition]["seeds"].append(seed)
                new_results[condition]["bpbl"].append(round(bpbl, 6))

    finally:
        print("\nRestoring train.py and baseline_results.json...", flush=True)
        TRAIN_PATH.write_text(original_train, encoding="utf-8")
        if baseline_backup is None:
            if BASELINE_JSON.exists():
                BASELINE_JSON.unlink()
        else:
            BASELINE_JSON.write_text(baseline_backup, encoding="utf-8")
        print("Restored.", flush=True)

    # ------------------------------------------------------------------
    # Step 5: Merge and compute robust statistics
    # ------------------------------------------------------------------
    output: dict = {
        "calibration": {
            "base_letters_per_word": round(base_letters_per_word, 4),
            "d1_base_letters_per_token": round(base_letters_per_token["d1"], 4),
            "d3_base_letters_per_token": round(base_letters_per_token["d3"], 4),
        },
        "new_seeds": NEW_SEEDS,
    }

    for condition in ["d1", "d3"]:
        orig_seeds = existing[condition]["seeds"]
        orig_bpbl = existing[condition]["bpbl"]
        new_seeds_run = new_results[condition]["seeds"]
        new_bpbl_run = new_results[condition]["bpbl"]

        all_seeds = orig_seeds + new_seeds_run
        all_bpbl = orig_bpbl + new_bpbl_run

        n_count = len(all_bpbl)
        if n_count == 0:
            continue

        mean_bpbl = sum(all_bpbl) / n_count
        variance = sum((x - mean_bpbl) ** 2 for x in all_bpbl) / max(n_count - 1, 1)
        std_bpbl = math.sqrt(variance)
        med_bpbl = median(all_bpbl)

        highest_bpbl = max(all_bpbl)
        highest_bpbl_seed = all_seeds[all_bpbl.index(highest_bpbl)]

        output[condition] = {
            "original_seeds": orig_seeds,
            "original_bpbl": orig_bpbl,
            "new_seeds": new_seeds_run,
            "new_bpbl": new_bpbl_run,
            "all_seeds": all_seeds,
            "all_bpbl": all_bpbl,
            "mean_bpbl": round(mean_bpbl, 6),
            "median_bpbl": round(med_bpbl, 6),
            "std_bpbl": round(std_bpbl, 6),
            "highest_bpbl_seed": highest_bpbl_seed,
        }

    # Conclusion based on median (robust)
    d1_med = output.get("d1", {}).get("median_bpbl")
    d3_med = output.get("d3", {}).get("median_bpbl")
    if d1_med is not None and d3_med is not None:
        if d1_med < d3_med:
            output["conclusion"] = "D1 beats D3 on median BPBL (robust to outliers)"
        else:
            output["conclusion"] = f"Unexpected: D1 median BPBL ({d1_med:.4f}) >= D3 median BPBL ({d3_med:.4f})"
    else:
        output["conclusion"] = "Incomplete — runs missing"

    ROBUST_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    ROBUST_RESULTS_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nRobust 5-seed results written to {ROBUST_RESULTS_JSON}", flush=True)

    # Print Summary
    print("\n" + "=" * 60, flush=True)
    print("ROBUST 5-SEED SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for cond in ["d1", "d3"]:
        c_data = output.get(cond, {})
        if c_data:
            print(f"  {cond.upper()} (n={len(c_data['all_bpbl'])})", flush=True)
            print(f"    Mean:   {c_data['mean_bpbl']:.4f} ± {c_data['std_bpbl']:.4f}", flush=True)
            print(f"    Median: {c_data['median_bpbl']:.4f}", flush=True)
            print(f"    Worst:  Seed {c_data['highest_bpbl_seed']} with BPBL {max(c_data['all_bpbl']):.4f}", flush=True)
    print(f"  Conclusion: {output.get('conclusion', 'N/A')}", flush=True)


if __name__ == "__main__":
    main()
