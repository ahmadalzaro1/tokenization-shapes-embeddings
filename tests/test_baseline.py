"""
Baseline training tests for Phase 2 (BASE-01 through BASE-03).
Unit test fails RED until Plan 02 extends train.py.
Integration tests are skipped until baseline training runs complete (3 x 300s).
"""
import json
from pathlib import Path

import pytest

BASE_CACHE = Path.home() / ".cache" / "autoresearch-arabic"
PROJECT_ROOT = Path(__file__).parent.parent
REQUIRED_KEYS = [
    "val_bpb",
    "depth",
    "vocab_size",
    "num_params_M",
    "training_seconds",
    "total_tokens_M",
    "window_pattern",
    "timestamp",
]


# ---------------------------------------------------------------------------
# BASE-01: train.py writes baseline_results.json (RED until Plan 02 adds it)
# ---------------------------------------------------------------------------

def test_baseline_json_written() -> None:
    """Fails RED until Plan 02 adds baseline_results.json writer to train.py."""
    source: str = (PROJECT_ROOT / "train.py").read_text(encoding="utf-8")
    assert "baseline_results.json" in source, (
        "train.py does not contain 'baseline_results.json'. "
        "Plan 02 must add a JSON writer for baseline results."
    )


# ---------------------------------------------------------------------------
# BASE-02: D2 val_bpb < D1 val_bpb (SKIP — integration)
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="integration: run after AUTORESEARCH_CONDITION=d{1,2,3} uv run train.py completes"
)
def test_d2_lower_than_d1() -> None:
    """Checks that stripping harakat lowers val_bpb compared to diacritized input."""
    results_path = BASE_CACHE / "baseline_results.json"
    assert results_path.exists(), f"baseline_results.json not found at {results_path}"
    with results_path.open(encoding="utf-8") as f:
        results: dict = json.load(f)

    assert "d1" in results, "Missing 'd1' key in baseline_results.json"
    assert "d2" in results, "Missing 'd2' key in baseline_results.json"

    d2_bpb: float = results["d2"]["val_bpb"]
    d1_bpb: float = results["d1"]["val_bpb"]
    assert d2_bpb < d1_bpb, (
        f"D2 val_bpb ({d2_bpb}) must be lower than D1 val_bpb ({d1_bpb}) "
        "— stripping harakat reduces surface complexity"
    )


# ---------------------------------------------------------------------------
# BASE-03: D3 baseline exists with plausible val_bpb (SKIP — integration)
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="integration: run after AUTORESEARCH_CONDITION=d{1,2,3} uv run train.py completes"
)
def test_baseline_d3() -> None:
    """Checks that d3 entry exists in baseline_results.json with plausible val_bpb."""
    results_path = BASE_CACHE / "baseline_results.json"
    assert results_path.exists(), f"baseline_results.json not found at {results_path}"
    with results_path.open(encoding="utf-8") as f:
        results: dict = json.load(f)

    assert "d3" in results, "Missing 'd3' key in baseline_results.json"
    val_bpb: float = results["d3"]["val_bpb"]
    assert 1.0 < val_bpb < 10.0, (
        f"D3 val_bpb={val_bpb} is outside plausible range (1.0, 10.0)"
    )


# ---------------------------------------------------------------------------
# BASE-01 (schema): all required keys present with correct constant values
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="integration: run after AUTORESEARCH_CONDITION=d{1,2,3} uv run train.py completes"
)
def test_baseline_schema() -> None:
    """Checks that baseline_results.json has all required keys for each condition."""
    results_path = BASE_CACHE / "baseline_results.json"
    assert results_path.exists(), f"baseline_results.json not found at {results_path}"
    with results_path.open(encoding="utf-8") as f:
        results: dict = json.load(f)

    for condition in ["d1", "d2", "d3"]:
        assert condition in results, f"Missing condition {condition!r} in baseline_results.json"
        for key in REQUIRED_KEYS:
            assert key in results[condition], (
                f"Missing required key {key!r} for condition {condition!r}"
            )
        assert results[condition]["depth"] == 4, (
            f"Expected depth=4 for condition {condition!r}, "
            f"got {results[condition]['depth']}"
        )
        assert results[condition]["window_pattern"] == "SSSL", (
            f"Expected window_pattern='SSSL' for condition {condition!r}, "
            f"got {results[condition]['window_pattern']!r}"
        )
