"""Experiment 6: Architecture Comparison Scaling Curves.

Three-condition iso-data scaling experiment to disentangle encoding effects
from architecture capacity effects:

  - d1_optimal:  D1 tokenizer + D1 optimal arch (depth=4)  [loaded from exp5]
  - d3_optimal:  D3 tokenizer + D3 optimal arch (depth=2)  [loaded from exp5]
  - d3_d1arch:   D3 tokenizer + D1 arch (depth=4)          [15 new runs]

If D3 with D1's architecture (d3_d1arch) still loses to D1_optimal by a
similar margin as D3_optimal, the architecture confound is ruled out and
the encoding effect is the sole cause. If d3_d1arch closes the gap to
d3_optimal or beyond, the architecture was a factor.

Only 15 new training runs needed — D1 and D3 optimal results reused from exp5.

Usage:
    cd /path/to/autoresearch-arabic
    uv run python experiments/exp6_shared_arch_iso_data.py
"""

import sys
from pathlib import Path

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
# Constants
# ---------------------------------------------------------------------------
D1_COMMIT = "ab075a6"   # D1 optimal arch: DEPTH=4, AR=26, HD=128, SS
D3_COMMIT = "532b0d1"   # D3 optimal arch: DEPTH=2, AR=64, HD=96, SS

FERTILITIES = {"d1": 2.5189, "d3": 2.1934}
SEEDS = [42, 137, 2024]
BUDGETS = [5_000_000, 15_000_000, 30_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000]
TOTAL_BATCH_SIZE = 32768
EVAL_WORDS = 2_990_668

RESULTS_DIR = ROOT / "experiments" / "results"
EXP5_RESULTS_JSON = RESULTS_DIR / "iso_data_results.json"         # source for d1_optimal + d3_optimal
LOG_DIR = RESULTS_DIR / "iso_data_logs_d3_d1arch"                 # logs for new d3_d1arch runs only
RESULTS_JSON = RESULTS_DIR / "iso_data_results_arch_comparison.json"


def budget_label(budget: int) -> str:
    return f"{budget // 1_000_000}M"


def compute_summary(runs: list[dict]) -> dict:
    """Compute mean/std BPBL per condition per budget."""
    conditions = ["d1_optimal", "d3_optimal", "d3_d1arch"]
    summary: dict = {}
    for cond in conditions:
        summary[cond] = {}
        for budget in BUDGETS:
            label = budget_label(budget)
            matching = [
                r["bpbl"] for r in runs
                if r["condition"] == cond and r["target_base_letters"] == budget
            ]
            if not matching:
                summary[cond][label] = {"mean_bpbl": None, "std_bpbl": None, "n": 0}
                continue
            n = len(matching)
            mean_bpbl = sum(matching) / n
            variance = sum((x - mean_bpbl) ** 2 for x in matching) / max(n - 1, 1)
            summary[cond][label] = {
                "mean_bpbl": round(mean_bpbl, 6),
                "std_bpbl": round(math.sqrt(variance), 6),
                "n": n,
            }
    return summary


def plot_arch_comparison(results: dict) -> None:
    """Plot all three conditions on one figure (linear + log scale)."""
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

    STYLES = {
        "d1_optimal": ("blue",  "o", "solid",  "D1 — Compositional (D1 arch)"),
        "d3_optimal": ("red",   "^", "solid",  "D3 — Atomic (D3 arch, optimal)"),
        "d3_d1arch":  ("red",   "^", "dashed", "D3 — Atomic (D1 arch, non-optimal)"),
    }

    for log_scale, suffix in [(False, ""), (True, "_log")]:
        fig, ax = plt.subplots(figsize=(11, 6))

        for cond, (color, marker, linestyle, label) in STYLES.items():
            x_means, y_means, y_stds = [], [], []

            for budget in BUDGETS:
                bl = budget_label(budget)
                stats = summary.get(cond, {}).get(bl, {})
                if not stats.get("mean_bpbl"):
                    continue
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

            ax.plot(x_arr, y_arr, color=color, marker=marker,
                    linestyle=linestyle, label=label, linewidth=2, markersize=8)
            ax.fill_between(x_arr, y_arr - s_arr, y_arr + s_arr,
                            color=color, alpha=0.12)

        ax.set_xlabel("Base Letters Processed", fontsize=12)
        ax.set_ylabel("Bits per Base Letter (BPBL)", fontsize=12)
        ax.set_title("Encoding vs Architecture: BPBL Scaling Curves", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_xscale("log")

        fig.tight_layout()
        out_path = RESULTS_DIR / f"iso_data_arch_comparison{suffix}.png"
        fig.savefig(str(out_path), dpi=300)
        plt.close(fig)
        print(f"Plot saved: {out_path}", flush=True)


def print_summary_table(summary: dict) -> None:
    print("\n" + "=" * 100, flush=True)
    print("EXPERIMENT 6 SUMMARY: Architecture Comparison", flush=True)
    print("=" * 100, flush=True)
    header = f"{'Budget':<10} | {'D1 optimal':<22} | {'D3 optimal':<22} | {'D3 w/ D1 arch':<22} | Encoding gap | Arch gap"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for budget in BUDGETS:
        bl = budget_label(budget)
        d1 = summary.get("d1_optimal", {}).get(bl, {})
        d3 = summary.get("d3_optimal", {}).get(bl, {})
        d3a = summary.get("d3_d1arch", {}).get(bl, {})

        d1_str  = f"{d1['mean_bpbl']:.4f}±{d1['std_bpbl']:.3f}"  if d1.get("mean_bpbl")  else "N/A"
        d3_str  = f"{d3['mean_bpbl']:.4f}±{d3['std_bpbl']:.3f}"  if d3.get("mean_bpbl")  else "N/A"
        d3a_str = f"{d3a['mean_bpbl']:.4f}±{d3a['std_bpbl']:.3f}" if d3a.get("mean_bpbl") else "N/A"

        enc_gap  = f"{d3['mean_bpbl'] - d1['mean_bpbl']:.4f}"   if d3.get("mean_bpbl") and d1.get("mean_bpbl") else "N/A"
        arch_gap = f"{d3a['mean_bpbl'] - d3['mean_bpbl']:.4f}"  if d3a.get("mean_bpbl") and d3.get("mean_bpbl") else "N/A"

        print(f"{bl:<10} | {d1_str:<22} | {d3_str:<22} | {d3a_str:<22} | {enc_gap:<12} | {arch_gap}", flush=True)

    print("=" * 100, flush=True)
    print("Encoding gap  = D3_optimal − D1_optimal  (pure encoding effect)", flush=True)
    print("Arch gap      = D3_d1arch  − D3_optimal  (architecture effect on D3)", flush=True)
    print("  Arch gap ≈ 0  → architecture confound ruled out", flush=True)
    print("  Arch gap > 0  → D1 arch hurts D3 (D3's shallow arch is better for it)", flush=True)
    print("  Arch gap < 0  → D1 arch helps D3 (D3 was under-architected)", flush=True)


def load_exp5_runs_as_condition(cond_name: str, exp5_condition: str) -> list[dict]:
    """Load exp5 runs for a condition and relabel them for this experiment."""
    if not EXP5_RESULTS_JSON.exists():
        raise FileNotFoundError(f"exp5 results not found at {EXP5_RESULTS_JSON}. Run exp5 first.")
    data = json.loads(EXP5_RESULTS_JSON.read_text(encoding="utf-8"))
    relabeled = []
    for r in data["runs"]:
        if r["condition"] == exp5_condition:
            entry = dict(r)
            entry["condition"] = cond_name
            entry["source"] = "exp5"
            relabeled.append(entry)
    return relabeled


def main() -> None:
    print("============================================================", flush=True)
    print("Exp 6: Architecture Comparison — 3 Conditions", flush=True)
    print("  d1_optimal : D1 tokenizer + D1 arch (depth=4)  [from exp5]", flush=True)
    print("  d3_optimal : D3 tokenizer + D3 arch (depth=2)  [from exp5]", flush=True)
    print("  d3_d1arch  : D3 tokenizer + D1 arch (depth=4)  [15 new runs]", flush=True)
    print("============================================================", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Load exp5 results for d1_optimal and d3_optimal
    # ------------------------------------------------------------------
    print("\nLoading exp5 results...", flush=True)
    d1_optimal_runs = load_exp5_runs_as_condition("d1_optimal", "d1")
    d3_optimal_runs = load_exp5_runs_as_condition("d3_optimal", "d3")
    print(f"  d1_optimal: {len(d1_optimal_runs)} runs loaded", flush=True)
    print(f"  d3_optimal: {len(d3_optimal_runs)} runs loaded", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Base letter calibration (needed for new d3_d1arch runs)
    # ------------------------------------------------------------------
    print("\nInitializing D1 data paths...", flush=True)
    prepare.init_condition("d1")
    base_letter_count = count_val_base_letters(prepare.DATA_DIR, prepare.VAL_FILENAME)
    base_letters_per_word = base_letter_count / EVAL_WORDS
    base_letters_per_token = {cond: base_letters_per_word / fert for cond, fert in FERTILITIES.items()}
    print(f"  base_letters_per_token: d1={base_letters_per_token['d1']:.4f}, d3={base_letters_per_token['d3']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Step calibration for d3_d1arch runs (D3 fertility, D1 arch)
    # ------------------------------------------------------------------
    print("\n--- Step Calibration for d3_d1arch ---", flush=True)
    print(f"{'Budget':<10} | {'BL/step (D3 fert)':<20} | {'max_steps':<10}", flush=True)
    print("-" * 45, flush=True)
    step_table: dict = {}
    for budget in BUDGETS:
        bl_per_step = TOTAL_BATCH_SIZE * base_letters_per_token["d3"]
        max_steps = int(budget / bl_per_step)
        step_table[budget] = max_steps
        print(f"{budget_label(budget):<10} | {bl_per_step:<20.1f} | {max_steps:<10}", flush=True)

    # ------------------------------------------------------------------
    # Step 4: Extract D1 arch hyperparameters (applied to D3 tokenizer)
    # ------------------------------------------------------------------
    print("\nExtracting D1 architecture hyperparameters...", flush=True)
    d1_params = extract_params_from_commit(D1_COMMIT)
    print(f"  D1 arch: DEPTH={d1_params['DEPTH']}, AR={d1_params['ASPECT_RATIO']}, "
          f"HD={d1_params['HEAD_DIM']}, WINDOW={d1_params['WINDOW_PATTERN']}", flush=True)

    # ------------------------------------------------------------------
    # Step 5: Backups and resumability for new runs
    # ------------------------------------------------------------------
    print("\nBacking up train.py and baseline_results.json...", flush=True)
    original_train = TRAIN_PATH.read_text(encoding="utf-8")
    baseline_backup = BASELINE_JSON.read_text(encoding="utf-8") if BASELINE_JSON.exists() else None

    os.makedirs(LOG_DIR, exist_ok=True)

    # Load any partial d3_d1arch results
    d3_d1arch_runs: list[dict] = []
    if RESULTS_JSON.exists():
        try:
            existing = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
            d3_d1arch_runs = [r for r in existing.get("runs", []) if r.get("condition") == "d3_d1arch"]
            print(f"  Loaded {len(d3_d1arch_runs)} existing d3_d1arch runs", flush=True)
        except (json.JSONDecodeError, KeyError):
            d3_d1arch_runs = []

    completed = {(r["target_base_letters"], r["seed"]) for r in d3_d1arch_runs}

    calibration = {
        "base_letters_per_word": round(base_letters_per_word, 4),
        "d1_base_letters_per_token": round(base_letters_per_token["d1"], 4),
        "d3_base_letters_per_token": round(base_letters_per_token["d3"], 4),
    }

    total_jobs = len(BUDGETS) * len(SEEDS)
    job_num = 0

    # ------------------------------------------------------------------
    # Step 6: Run 15 new d3_d1arch jobs
    # ------------------------------------------------------------------
    try:
        for budget in BUDGETS:
            max_steps = step_table[budget]
            for seed in SEEDS:
                job_num += 1

                if (budget, seed) in completed:
                    print(f"\n[{job_num}/{total_jobs}] SKIP: d3_d1arch {budget_label(budget)} seed={seed}", flush=True)
                    continue

                print(f"\n{'='*60}", flush=True)
                print(f"[{job_num}/{total_jobs}] d3_d1arch  {budget_label(budget)} seed={seed} max_steps={max_steps}", flush=True)
                print(f"  (D3 tokenizer + D1 architecture depth=4)", flush=True)
                print(f"{'='*60}", flush=True)

                patch_train(d1_params)  # D1 architecture

                env = os.environ.copy()
                env["AUTORESEARCH_CONDITION"] = "d3"    # D3 tokenizer + data
                env["AUTORESEARCH_SEED"] = str(seed)
                env["AUTORESEARCH_MAX_STEPS"] = str(max_steps)

                run = subprocess.run(
                    ["uv", "run", "train.py"],
                    cwd=ROOT, env=env, text=True, capture_output=True, check=False,
                )

                log_text = run.stdout + run.stderr
                log_path = LOG_DIR / f"d3_d1arch_{budget_label(budget)}_seed{seed}.log"
                log_path.write_text(log_text, encoding="utf-8")
                print(log_text[-2000:] if len(log_text) > 2000 else log_text, flush=True)

                if run.returncode != 0:
                    print(f"WARNING: Run failed for d3_d1arch {budget_label(budget)} seed {seed}. Skipping.", flush=True)
                    continue

                metrics = parse_metrics(log_text)
                val_bpb          = metrics.get("val_bpb")
                total_eval_nats  = metrics.get("total_eval_nats")
                total_eval_bytes = metrics.get("total_eval_bytes")
                total_valid_tokens = metrics.get("total_valid_tokens")
                total_tokens_M   = metrics.get("total_tokens_M")
                num_steps        = metrics.get("num_steps")

                if val_bpb is None or total_eval_nats is None or total_valid_tokens is None or num_steps is None:
                    print(f"WARNING: Missing metrics. Got: {metrics}. Skipping.", flush=True)
                    continue

                # D3 fertility for calibration (tokenizer property, unchanged)
                bl_per_token_d3  = base_letters_per_token["d3"]
                eval_base_letters = total_valid_tokens * bl_per_token_d3
                bpbl = total_eval_nats / (eval_base_letters * math.log(2))
                actual_base_letters = num_steps * TOTAL_BATCH_SIZE * bl_per_token_d3

                print(f"  val_bpb={val_bpb:.6f}, num_steps={int(num_steps)}, "
                      f"actual_BL={actual_base_letters:.0f}, BPBL={bpbl:.6f}", flush=True)

                result_entry = {
                    "condition": "d3_d1arch",
                    "tokenizer": "d3",
                    "architecture": "d1_optimal",
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

                d3_d1arch_runs.append(result_entry)

                # Write incremental — all 3 conditions
                all_runs = d1_optimal_runs + d3_optimal_runs + d3_d1arch_runs
                output = {
                    "experiment": "arch_comparison",
                    "conditions": {
                        "d1_optimal":  "D1 tokenizer + D1 arch (depth=4) — reused from exp5",
                        "d3_optimal":  "D3 tokenizer + D3 arch (depth=2) — reused from exp5",
                        "d3_d1arch":   "D3 tokenizer + D1 arch (depth=4) — new runs",
                    },
                    "calibration": calibration,
                    "runs": all_runs,
                    "summary": compute_summary(all_runs),
                }
                RESULTS_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
                print(f"  Incremental results written ({len(d3_d1arch_runs)}/15 new runs)", flush=True)

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
    # Step 7: Final output
    # ------------------------------------------------------------------
    all_runs = d1_optimal_runs + d3_optimal_runs + d3_d1arch_runs
    summary = compute_summary(all_runs)
    final_output = {
        "experiment": "arch_comparison",
        "conditions": {
            "d1_optimal":  "D1 tokenizer + D1 arch (depth=4) — reused from exp5",
            "d3_optimal":  "D3 tokenizer + D3 arch (depth=2) — reused from exp5",
            "d3_d1arch":   "D3 tokenizer + D1 arch (depth=4) — new runs",
        },
        "calibration": calibration,
        "runs": all_runs,
        "summary": summary,
    }
    RESULTS_JSON.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    print(f"\nFinal results written to {RESULTS_JSON} ({len(all_runs)} total runs)", flush=True)

    print_summary_table(summary)
    plot_arch_comparison(final_output)
    print(f"\nDone. {len(d3_d1arch_runs)}/15 new d3_d1arch runs completed.", flush=True)


if __name__ == "__main__":
    main()
