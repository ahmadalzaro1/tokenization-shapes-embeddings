"""Experiment 5: Iso-data scaling curves — BPBL vs base letters processed.

Trains D1 and D3 at 5 data budgets (5M, 15M, 30M, 50M, 100M base letters),
each with 3 seeds, using each condition's optimal architecture. Plots BPBL
vs base-letters-processed scaling curves.

Usage:
    cd /path/to/autoresearch-arabic
    uv run python experiments/exp5_iso_data.py
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
SEEDS = [42, 137, 2024]
BUDGETS = [5_000_000, 15_000_000, 30_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000]
TOTAL_BATCH_SIZE = 32768
EVAL_WORDS = 2_990_668

RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "iso_data_logs"
RESULTS_JSON = RESULTS_DIR / "iso_data_results.json"


def budget_label(budget: int) -> str:
    """Return human-readable budget label like '5M', '100M'."""
    return f"{budget // 1_000_000}M"


def compute_summary(runs: list[dict]) -> dict:
    """Compute mean/std BPBL per condition per budget from runs list."""
    summary: dict = {}
    for cond in ["d1", "d3"]:
        summary[cond] = {}
        for budget in BUDGETS:
            label = budget_label(budget)
            matching = [
                r["bpbl"]
                for r in runs
                if r["condition"] == cond and r["target_base_letters"] == budget
            ]
            if len(matching) == 0:
                summary[cond][label] = {"mean_bpbl": None, "std_bpbl": None, "n": 0}
                continue
            n = len(matching)
            mean_bpbl = sum(matching) / n
            variance = sum((x - mean_bpbl) ** 2 for x in matching) / max(n - 1, 1)
            std_bpbl = math.sqrt(variance)
            summary[cond][label] = {
                "mean_bpbl": round(mean_bpbl, 6),
                "std_bpbl": round(std_bpbl, 6),
                "n": n,
            }
    return summary


def plot_scaling_curves(results: dict) -> None:
    """Generate BPBL vs base letters processed plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots.", flush=True)
        return

    runs = results["runs"]
    summary = results["summary"]

    for log_scale, suffix in [(False, ""), (True, "_log")]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cond, color, marker, label in [
            ("d1", "blue", "o", "D1 (Compositional)"),
            ("d3", "red", "^", "D3 (Atomic)"),
        ]:
            x_means = []
            y_means = []
            y_stds = []

            for budget in BUDGETS:
                bl = budget_label(budget)
                stats = summary[cond].get(bl, {})
                if stats.get("mean_bpbl") is None:
                    continue

                # Compute mean actual_base_letters for this budget
                matching = [
                    r for r in runs
                    if r["condition"] == cond and r["target_base_letters"] == budget
                ]
                if not matching:
                    continue

                mean_actual = sum(r["actual_base_letters"] for r in matching) / len(matching)
                x_means.append(mean_actual)
                y_means.append(stats["mean_bpbl"])
                y_stds.append(stats["std_bpbl"])

            if not x_means:
                continue

            x_arr = np.array(x_means)
            y_arr = np.array(y_means)
            s_arr = np.array(y_stds)

            ax.plot(x_arr, y_arr, color=color, marker=marker, label=label,
                    linewidth=2, markersize=8)
            ax.fill_between(x_arr, y_arr - s_arr, y_arr + s_arr,
                            color=color, alpha=0.2)

        ax.set_xlabel("Base Letters Processed", fontsize=12)
        ax.set_ylabel("Bits per Base Letter (BPBL)", fontsize=12)
        ax.set_title("BPBL vs Training Data Budget", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_xscale("log")

        fig.tight_layout()
        out_path = ROOT / "assets" / f"iso_data_scaling_curves{suffix}.png"
        fig.savefig(str(out_path), dpi=300)
        plt.close(fig)
        print(f"Plot saved: {out_path}", flush=True)


def print_summary_table(summary: dict) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 80, flush=True)
    print("EXPERIMENT 5 SUMMARY: BPBL vs Data Budget", flush=True)
    print("=" * 80, flush=True)
    header = f"{'Budget (BL)':<14} | {'D1 BPBL (mean +/- std)':<26} | {'D3 BPBL (mean +/- std)':<26} | Winner"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for budget in BUDGETS:
        bl = budget_label(budget)
        d1 = summary["d1"].get(bl, {})
        d3 = summary["d3"].get(bl, {})

        d1_str = (
            f"{d1['mean_bpbl']:.4f} +/- {d1['std_bpbl']:.4f}"
            if d1.get("mean_bpbl") is not None
            else "N/A"
        )
        d3_str = (
            f"{d3['mean_bpbl']:.4f} +/- {d3['std_bpbl']:.4f}"
            if d3.get("mean_bpbl") is not None
            else "N/A"
        )

        winner = "N/A"
        if d1.get("mean_bpbl") is not None and d3.get("mean_bpbl") is not None:
            winner = "D1" if d1["mean_bpbl"] < d3["mean_bpbl"] else "D3"

        print(f"{bl:<14} | {d1_str:<26} | {d3_str:<26} | {winner}", flush=True)

    print("=" * 80, flush=True)


def main() -> None:
    # ------------------------------------------------------------------
    # Step 1: Pre-compute base_letters_per_token from D1 val shard.
    # ------------------------------------------------------------------
    print("Initializing D1 data paths for base letter counting...", flush=True)
    prepare.init_condition("d1")
    print(f"  DATA_DIR:     {prepare.DATA_DIR}", flush=True)
    print(f"  VAL_FILENAME: {prepare.VAL_FILENAME}", flush=True)

    print("Counting Arabic base letters in D1 val shard...", flush=True)
    base_letter_count = count_val_base_letters(prepare.DATA_DIR, prepare.VAL_FILENAME)
    print(f"  base_letter_count = {base_letter_count:,}", flush=True)

    base_letters_per_word = base_letter_count / EVAL_WORDS
    print(f"  base_letters_per_word = {base_letters_per_word:.4f}", flush=True)

    base_letters_per_token = {
        cond: base_letters_per_word / fert
        for cond, fert in FERTILITIES.items()
    }
    print(f"  base_letters_per_token: d1={base_letters_per_token['d1']:.4f}, "
          f"d3={base_letters_per_token['d3']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Calibrate max_steps per condition per budget.
    # ------------------------------------------------------------------
    print("\n--- Step Calibration Table ---", flush=True)
    print(f"{'Budget':<10} | {'Cond':<5} | {'BL/step':<12} | {'max_steps':<10}", flush=True)
    print("-" * 45, flush=True)

    step_table: dict = {}  # (condition, budget) -> max_steps
    for budget in BUDGETS:
        for cond in ["d1", "d3"]:
            bl_per_step = TOTAL_BATCH_SIZE * base_letters_per_token[cond]
            max_steps = int(budget / bl_per_step)
            step_table[(cond, budget)] = max_steps
            print(f"{budget_label(budget):<10} | {cond.upper():<5} | {bl_per_step:<12.1f} | {max_steps:<10}",
                  flush=True)

    # ------------------------------------------------------------------
    # Step 3: Extract hyperparameters from git commits.
    # ------------------------------------------------------------------
    print("\nExtracting D1 hyperparameters from commit ab075a6...", flush=True)
    d1_params = extract_params_from_commit(D1_COMMIT)
    print(f"  D1 params: {d1_params}", flush=True)

    print("Extracting D3 hyperparameters from commit 532b0d1...", flush=True)
    d3_params = extract_params_from_commit(D3_COMMIT)
    print(f"  D3 params: {d3_params}", flush=True)

    configs = {"d1": d1_params, "d3": d3_params}

    # ------------------------------------------------------------------
    # Step 4: Back up train.py content and baseline_results.json.
    # ------------------------------------------------------------------
    print("\nBacking up train.py and baseline_results.json...", flush=True)
    original_train = TRAIN_PATH.read_text(encoding="utf-8")
    baseline_backup = BASELINE_JSON.read_text(encoding="utf-8") if BASELINE_JSON.exists() else None

    # ------------------------------------------------------------------
    # Step 5: Create output directories and load any partial results.
    # ------------------------------------------------------------------
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load partial results if they exist (for resumability)
    runs: list[dict] = []
    if RESULTS_JSON.exists():
        try:
            existing = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
            runs = existing.get("runs", [])
            print(f"Loaded {len(runs)} existing runs from {RESULTS_JSON}", flush=True)
        except (json.JSONDecodeError, KeyError):
            runs = []

    # Build set of already-completed runs for skip logic
    completed = {
        (r["condition"], r["target_base_letters"], r["seed"])
        for r in runs
    }

    # ------------------------------------------------------------------
    # Step 6: Run training jobs.
    # ------------------------------------------------------------------
    calibration = {
        "base_letters_per_word": round(base_letters_per_word, 4),
        "d1_base_letters_per_token": round(base_letters_per_token["d1"], 4),
        "d3_base_letters_per_token": round(base_letters_per_token["d3"], 4),
    }

    total_jobs = len(["d1", "d3"]) * len(BUDGETS) * len(SEEDS)
    job_num = 0

    try:
        for condition in ["d1", "d3"]:
            for budget in BUDGETS:
                max_steps = step_table[(condition, budget)]
                for seed in SEEDS:
                    job_num += 1

                    # Skip already-completed runs
                    if (condition, budget, seed) in completed:
                        print(f"\n[{job_num}/{total_jobs}] SKIP (already done): "
                              f"{condition.upper()} {budget_label(budget)} seed={seed}",
                              flush=True)
                        continue

                    print(
                        f"\n{'='*60}",
                        flush=True,
                    )
                    print(
                        f"[{job_num}/{total_jobs}] Running: condition={condition.upper()}, "
                        f"budget={budget_label(budget)}, seed={seed}, max_steps={max_steps}",
                        flush=True,
                    )
                    print(f"{'='*60}", flush=True)

                    # Patch train.py with this condition's best hyperparameters
                    patch_train(configs[condition])

                    # Set up environment
                    env = os.environ.copy()
                    env["AUTORESEARCH_CONDITION"] = condition
                    env["AUTORESEARCH_SEED"] = str(seed)
                    env["AUTORESEARCH_MAX_STEPS"] = str(max_steps)

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

                    # Save log file
                    log_path = LOG_DIR / f"{condition}_{budget_label(budget)}_seed{seed}.log"
                    log_path.write_text(log_text, encoding="utf-8")
                    print(f"  Log saved: {log_path}", flush=True)

                    # Print tail of log
                    print(log_text[-2000:] if len(log_text) > 2000 else log_text, flush=True)

                    if run.returncode != 0:
                        print(
                            f"WARNING: Run failed for {condition} {budget_label(budget)} "
                            f"seed {seed}, returncode={run.returncode}. Skipping.",
                            flush=True,
                        )
                        continue

                    # Parse metrics from stdout
                    metrics = parse_metrics(log_text)

                    val_bpb = metrics.get("val_bpb")
                    total_eval_nats = metrics.get("total_eval_nats")
                    total_eval_bytes = metrics.get("total_eval_bytes")
                    total_valid_tokens = metrics.get("total_valid_tokens")
                    total_tokens_M = metrics.get("total_tokens_M")
                    num_steps = metrics.get("num_steps")

                    if (
                        val_bpb is None
                        or total_eval_nats is None
                        or total_valid_tokens is None
                        or num_steps is None
                    ):
                        print(
                            f"WARNING: Missing metrics for {condition} {budget_label(budget)} "
                            f"seed {seed}. Got: {metrics}. Skipping.",
                            flush=True,
                        )
                        continue

                    # Compute BPBL
                    eval_base_letters = total_valid_tokens * base_letters_per_token[condition]
                    bpbl = total_eval_nats / (eval_base_letters * math.log(2))

                    # Compute actual base letters processed (using parsed num_steps)
                    actual_base_letters = num_steps * TOTAL_BATCH_SIZE * base_letters_per_token[condition]

                    print(
                        f"  val_bpb={val_bpb:.6f}, "
                        f"num_steps={int(num_steps)}, "
                        f"actual_BL={actual_base_letters:.0f}, "
                        f"eval_BL={eval_base_letters:.0f}, "
                        f"BPBL={bpbl:.6f}",
                        flush=True,
                    )

                    result_entry = {
                        "condition": condition,
                        "target_base_letters": budget,
                        "actual_base_letters": round(actual_base_letters, 1),
                        "max_steps": max_steps,
                        "num_steps": int(num_steps),
                        "seed": seed,
                        "val_bpb": round(val_bpb, 6),
                        "total_eval_nats": round(total_eval_nats, 4),
                        "total_eval_bytes": int(total_eval_bytes) if total_eval_bytes else 0,
                        "total_valid_tokens": int(total_valid_tokens),
                        "eval_base_letters": round(eval_base_letters, 1),
                        "bpbl": round(bpbl, 6),
                    }
                    if total_tokens_M is not None:
                        result_entry["total_tokens_M"] = round(total_tokens_M, 1)

                    runs.append(result_entry)

                    # Write incremental JSON
                    output = {
                        "calibration": calibration,
                        "runs": runs,
                        "summary": compute_summary(runs),
                    }
                    RESULTS_JSON.write_text(
                        json.dumps(output, indent=2), encoding="utf-8"
                    )
                    print(f"  Incremental results written ({len(runs)} runs)", flush=True)

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
    # Step 7: Final summary and plots.
    # ------------------------------------------------------------------
    summary = compute_summary(runs)
    final_output = {
        "calibration": calibration,
        "runs": runs,
        "summary": summary,
    }
    RESULTS_JSON.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    print(f"\nFinal results written to {RESULTS_JSON} ({len(runs)} runs)", flush=True)

    # Print summary table
    print_summary_table(summary)

    # Generate plots
    plot_scaling_curves(final_output)

    print(f"\nDone. Total runs: {len(runs)}", flush=True)


if __name__ == "__main__":
    main()
