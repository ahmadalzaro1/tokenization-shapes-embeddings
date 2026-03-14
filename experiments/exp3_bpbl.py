"""Experiment 3: BPBL (bits-per-base-letter) metric for D1 vs D3.

Runs 6 training jobs (3 seeds x D1 + 3 seeds x D3) with the best hyperparameters
from Phase 3, computes bits-per-base-letter, and writes bpbl_results.json.

Usage:
    cd /path/to/autoresearch-arabic
    uv run python experiments/exp3_bpbl.py
"""

import sys
from pathlib import Path

# Critical: fix sys.path so 'import prepare' and 'from experiments.shared import ...' work
# when executed as 'uv run python experiments/exp3_bpbl.py'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math
import os
import subprocess

import prepare
from experiments.shared import (
    BASELINE_JSON,
    ROOT,
    TRAIN_PATH,
    count_val_base_letters,
    extract_params_from_commit,
    parse_metrics,
    patch_train,
)

# ---------------------------------------------------------------------------
# Commit hashes for best D1 and D3 configs (from search_results.json)
# ---------------------------------------------------------------------------
D1_COMMIT = "ab075a6"
D3_COMMIT = "532b0d1"

# ---------------------------------------------------------------------------
# Fertility values from fertility_report.json (vocab_size=8192)
# ---------------------------------------------------------------------------
FERTILITIES = {"d1": 2.5189, "d3": 2.1934}

# ---------------------------------------------------------------------------
# Seeds for 3 independent runs per condition
# ---------------------------------------------------------------------------
SEEDS = [42, 137, 2024]


def main() -> None:
    # ------------------------------------------------------------------
    # Step 1: Pre-compute base letter count from D1 val shard.
    # Always use D1 val shard because D3 PUA codepoints are outside the
    # Arabic letter regex range.
    # ------------------------------------------------------------------
    print("Initializing D1 data paths for base letter counting...", flush=True)
    prepare.init_condition("d1")
    print(f"  DATA_DIR:     {prepare.DATA_DIR}", flush=True)
    print(f"  VAL_FILENAME: {prepare.VAL_FILENAME}", flush=True)

    print("Counting Arabic base letters in D1 val shard...", flush=True)
    base_letter_count = count_val_base_letters(prepare.DATA_DIR, prepare.VAL_FILENAME)
    print(f"  base_letter_count = {base_letter_count:,}", flush=True)

    # eval_words from Phase 2 baseline (d1 entry in baseline_results.json)
    eval_words = 2_990_668  # from baseline_results.json d1.eval_words
    base_letters_per_word = base_letter_count / eval_words
    print(f"  base_letters_per_word = {base_letters_per_word:.4f}", flush=True)

    base_letters_per_token = {
        cond: base_letters_per_word / fert
        for cond, fert in FERTILITIES.items()
    }
    print(f"  base_letters_per_token: {base_letters_per_token}", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Extract full hyperparameters from git commits.
    # Using commit source (not search_results.json) avoids _EXPR suffix issues.
    # ------------------------------------------------------------------
    print("\nExtracting D1 hyperparameters from commit ab075a6...", flush=True)
    d1_params = extract_params_from_commit(D1_COMMIT)
    print(f"  D1 params: {d1_params}", flush=True)

    print("Extracting D3 hyperparameters from commit 532b0d1...", flush=True)
    d3_params = extract_params_from_commit(D3_COMMIT)
    print(f"  D3 params: {d3_params}", flush=True)

    configs = {"d1": d1_params, "d3": d3_params}

    # ------------------------------------------------------------------
    # Step 3: Back up train.py content and baseline_results.json.
    # The backup captures train.py AFTER all structural patches (4-tuple
    # unpack, new print lines, env-var support) are already in place.
    # We only restore the hyperparameters, not the structural changes.
    # ------------------------------------------------------------------
    print("\nBacking up train.py and baseline_results.json...", flush=True)
    original_train = TRAIN_PATH.read_text(encoding="utf-8")
    baseline_backup = BASELINE_JSON.read_text(encoding="utf-8") if BASELINE_JSON.exists() else None

    # ------------------------------------------------------------------
    # Step 4: Run 6 training jobs (D1 x 3 seeds, D3 x 3 seeds).
    # ------------------------------------------------------------------
    results: dict = {"d1": {}, "d3": {}}
    for cond in ["d1", "d3"]:
        results[cond] = {
            "seeds": [],
            "val_bpb": [],
            "total_eval_nats": [],
            "total_valid_tokens": [],
            "eval_base_letters": [],
            "bpbl": [],
        }

    try:
        for condition in ["d1", "d3"]:
            for seed in SEEDS:
                print(
                    f"\n{'='*60}",
                    flush=True,
                )
                print(
                    f"Running: condition={condition.upper()}, seed={seed}",
                    flush=True,
                )
                print(f"{'='*60}", flush=True)

                # Patch train.py with this condition's best hyperparameters
                patch_train(configs[condition])

                # Set up environment
                env = os.environ.copy()
                env["AUTORESEARCH_CONDITION"] = condition
                env["AUTORESEARCH_SEED"] = str(seed)
                wte_path = str(
                    ROOT / "experiments" / "results" / f"wte_{condition}_seed{seed}.npy"
                )
                env["AUTORESEARCH_SAVE_WTE"] = wte_path

                # Run training
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
                    print(
                        f"WARNING: Run failed for {condition} seed {seed}, "
                        f"returncode={run.returncode}. Skipping.",
                        flush=True,
                    )
                    continue

                # Parse metrics from stdout
                metrics = parse_metrics(log_text)

                val_bpb = metrics.get("val_bpb")
                total_eval_nats = metrics.get("total_eval_nats")
                total_valid_tokens = metrics.get("total_valid_tokens")

                if val_bpb is None or total_eval_nats is None or total_valid_tokens is None:
                    print(
                        f"WARNING: Missing metrics for {condition} seed {seed}. "
                        f"Got: {metrics}. Skipping.",
                        flush=True,
                    )
                    continue

                # Compute BPBL
                # eval_base_letters = number of Arabic base letters in the eval window
                # (matched to the token window, not the full shard)
                eval_base_letters = total_valid_tokens * base_letters_per_token[condition]
                bpbl = total_eval_nats / (eval_base_letters * math.log(2))

                print(
                    f"  val_bpb={val_bpb:.6f}, "
                    f"total_eval_nats={total_eval_nats:.2f}, "
                    f"total_valid_tokens={int(total_valid_tokens):,}, "
                    f"eval_base_letters={eval_base_letters:.0f}, "
                    f"BPBL={bpbl:.6f}",
                    flush=True,
                )

                results[condition]["seeds"].append(seed)
                results[condition]["val_bpb"].append(round(val_bpb, 6))
                results[condition]["total_eval_nats"].append(round(total_eval_nats, 4))
                results[condition]["total_valid_tokens"].append(int(total_valid_tokens))
                results[condition]["eval_base_letters"].append(round(eval_base_letters, 1))
                results[condition]["bpbl"].append(round(bpbl, 6))

    finally:
        # Always restore train.py and baseline_results.json
        print("\nRestoring train.py and baseline_results.json...", flush=True)
        TRAIN_PATH.write_text(original_train, encoding="utf-8")
        if baseline_backup is None:
            if BASELINE_JSON.exists():
                BASELINE_JSON.unlink()
        else:
            BASELINE_JSON.write_text(baseline_backup, encoding="utf-8")
        print("Restored.", flush=True)

    # ------------------------------------------------------------------
    # Step 5: Compute per-condition statistics and write results JSON.
    # ------------------------------------------------------------------
    output: dict = {
        "full_shard_base_letter_count": base_letter_count,
        "base_letters_per_word": round(base_letters_per_word, 4),
        "fertilities": FERTILITIES,
        "base_letters_per_token": {k: round(v, 4) for k, v in base_letters_per_token.items()},
    }

    for condition in ["d1", "d3"]:
        bpbl_list = results[condition]["bpbl"]
        if len(bpbl_list) == 0:
            print(f"WARNING: No successful runs for {condition}. Skipping stats.", flush=True)
            output[condition] = results[condition]
            output[condition]["mean_bpbl"] = None
            output[condition]["std_bpbl"] = None
            continue

        n = len(bpbl_list)
        mean_bpbl = sum(bpbl_list) / n
        variance = sum((x - mean_bpbl) ** 2 for x in bpbl_list) / max(n - 1, 1)
        std_bpbl = math.sqrt(variance)

        output[condition] = results[condition]
        output[condition]["mean_bpbl"] = round(mean_bpbl, 6)
        output[condition]["std_bpbl"] = round(std_bpbl, 6)

    # Conclusion
    d1_mean = output["d1"].get("mean_bpbl")
    d3_mean = output["d3"].get("mean_bpbl")
    if d1_mean is not None and d3_mean is not None:
        if d1_mean < d3_mean:
            output["conclusion"] = "D1 beats D3 on BPBL (lower is better)"
        else:
            output["conclusion"] = f"Unexpected: D1 BPBL ({d1_mean:.4f}) >= D3 BPBL ({d3_mean:.4f})"
    else:
        output["conclusion"] = "Incomplete — some runs failed"

    out_path = ROOT / "experiments" / "results" / "bpbl_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT 3 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for condition in ["d1", "d3"]:
        mean = output[condition].get("mean_bpbl")
        std = output[condition].get("std_bpbl")
        n = len(output[condition].get("bpbl", []))
        if mean is not None:
            print(
                f"  {condition.upper()} BPBL: {mean:.6f} +/- {std:.6f} (n={n})",
                flush=True,
            )
        else:
            print(f"  {condition.upper()} BPBL: FAILED", flush=True)
    print(f"  Conclusion: {output.get('conclusion', 'N/A')}", flush=True)


if __name__ == "__main__":
    main()
