import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "train.py"
SEARCH_RESULTS_PATH = ROOT / "search_results.json"
OUTPUT_JSON = ROOT / "paper" / "fixed_architecture_ablation.json"
OUTPUT_MD = ROOT / "paper" / "fixed_architecture_ablation.md"
LOG_DIR = ROOT / "paper" / "ablation_logs"
RUN_LOG = ROOT / "run.log"
BASELINE_JSON = Path.home() / ".cache" / "autoresearch-arabic" / "baseline_results.json"

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


def sh(cmd, env=None):
    return subprocess.check_output(cmd, text=True, cwd=ROOT, env=env)


def extract_params_from_commit(commit: str) -> dict[str, str]:
    source = sh(["git", "show", f"{commit}:train.py"])
    params: dict[str, str] = {}
    for key in PARAM_KEYS:
        match = re.search(rf"^{key}\s*=\s*(.+)$", source, flags=re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not find {key} in commit {commit}")
        params[key] = match.group(1).strip()
    return params


def patch_train(params: dict[str, str]) -> None:
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


def parse_metrics(log_text: str) -> dict[str, float]:
    patterns = {
        "val_bpb": r"^val_bpb:\s+([0-9.]+)$",
        "peak_vram_mb": r"^peak_vram_mb:\s+([0-9.]+)$",
        "total_tokens_M": r"^total_tokens_M:\s+([0-9.]+)$",
        "num_params_M": r"^num_params_M:\s+([0-9.]+)$",
        "training_seconds": r"^training_seconds:\s+([0-9.]+)$",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text, flags=re.MULTILINE)
        if not match:
            raise RuntimeError(f"Missing metric {key}")
        result[key] = float(match.group(1))
    result["memory_gb"] = result["peak_vram_mb"] / 1024.0
    return result


def load_existing() -> dict:
    if OUTPUT_JSON.exists():
        return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    return {"architectures": {}, "summary": {}}


def save_outputs(data: dict) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

    arch_order = ["d1", "d2", "d3"]
    cond_order = ["d1", "d2", "d3"]
    lines = [
        "# Fixed-Architecture Ablation",
        "",
        "| Shared Architecture | D1 | D2 | D3 | Ordering |",
        "|---|---:|---:|---:|---|",
    ]

    summary = {}
    for arch in arch_order:
        row = data["architectures"].get(arch, {})
        vals = {cond: row[cond]["val_bpb"] for cond in cond_order if cond in row}
        ordering = "incomplete"
        if len(vals) == 3:
            ordering = " < ".join(sorted(vals, key=vals.get)).upper()
            summary[arch] = ordering
        d1 = f"{vals['d1']:.6f}" if "d1" in vals else "-"
        d2 = f"{vals['d2']:.6f}" if "d2" in vals else "-"
        d3 = f"{vals['d3']:.6f}" if "d3" in vals else "-"
        lines.append(f"| {arch.upper()} winner | {d1} | {d2} | {d3} | {ordering} |")

    lines.append("")
    lines.append("## Interpretation")
    for arch in arch_order:
        if arch not in summary:
            continue
        lines.append(f"- Under the shared {arch.upper()}-winner architecture, the ordering is `{summary[arch]}`.")

    data["summary"] = summary
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    original_train = TRAIN_PATH.read_text(encoding="utf-8")
    baseline_backup = BASELINE_JSON.read_text(encoding="utf-8") if BASELINE_JSON.exists() else None
    data = load_existing()
    results = json.loads(SEARCH_RESULTS_PATH.read_text(encoding="utf-8"))

    arch_params = {
        arch: extract_params_from_commit(entry["commit"])
        for arch, entry in results.items()
    }

    try:
        for arch in ["d1", "d2", "d3"]:
            data["architectures"].setdefault(arch, {})
            for cond in ["d1", "d2", "d3"]:
                if cond in data["architectures"][arch]:
                    continue
                print(f"Running shared {arch.upper()} winner on condition {cond.upper()}...", flush=True)
                patch_train(arch_params[arch])
                env = os.environ.copy()
                env["AUTORESEARCH_CONDITION"] = cond
                run = subprocess.run(
                    ["uv", "run", "train.py"],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                log_text = run.stdout + run.stderr
                log_path = LOG_DIR / f"{arch}_on_{cond}.log"
                log_path.write_text(log_text, encoding="utf-8")
                if run.returncode != 0:
                    raise RuntimeError(f"Run failed for {arch} on {cond}: see {log_path}")
                metrics = parse_metrics(log_text)
                metrics["log_path"] = str(log_path.relative_to(ROOT))
                metrics["shared_architecture"] = arch
                metrics["condition"] = cond
                data["architectures"][arch][cond] = metrics
                print(
                    f"Completed {arch.upper()} on {cond.upper()}: "
                    f"val_bpb={metrics['val_bpb']:.6f}, "
                    f"memory_gb={metrics['memory_gb']:.1f}",
                    flush=True,
                )
                save_outputs(data)
    finally:
        TRAIN_PATH.write_text(original_train, encoding="utf-8")
        if baseline_backup is None:
            if BASELINE_JSON.exists():
                BASELINE_JSON.unlink()
        else:
            BASELINE_JSON.write_text(baseline_backup, encoding="utf-8")

    save_outputs(data)


if __name__ == "__main__":
    main()
