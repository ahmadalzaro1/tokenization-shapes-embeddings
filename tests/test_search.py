"""
Phase 3 Architecture Search — Wave 0 smoke tests (03-01).

Four tests that verify overnight run artifacts:
  - test_search_d3 / test_search_d1 / test_search_d2  : one per condition TSV
  - test_search_results_json                           : the summary JSON

All tests skip gracefully when the artifact does not yet exist (pre-run
state), making the suite safe to run at any point in the project lifecycle.
Each condition's run is declared complete once its test turns green.

Baselines to beat:
  d3: 1.075381
  d1: 1.190999
  d2: 1.596882

TSV schema: commit, val_bpb, memory_gb, status, description (TAB-delimited)
"""

import csv
import json
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BASELINES: dict[str, float] = {
    "d3": 1.075381,
    "d1": 1.190999,
    "d2": 1.596882,
}

PROJECT_ROOT: Path = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"commit", "val_bpb", "memory_gb", "status", "description"}


def _read_tsv(path: Path) -> list[dict] | None:
    """Read a TAB-delimited file via csv.DictReader.

    Returns a list of row dicts, or None if the file does not exist.
    """
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _check_condition(condition: str) -> None:
    """Core validation logic shared by all three condition tests.

    Skips if the TSV is absent; otherwise asserts row count, schema, and
    that at least one "keep" row beats the condition's baseline bpb.
    """
    path = PROJECT_ROOT / f"results_{condition}.tsv"
    rows = _read_tsv(path)

    if rows is None:
        pytest.skip(
            f"results_{condition}.tsv not yet created — "
            f"run D{condition[-1].upper()} overnight search first"
        )

    # Row count (header excluded by DictReader)
    assert len(rows) >= 70, (
        f"Expected >= 70 rows in results_{condition}.tsv, got {len(rows)}"
    )

    # Schema check
    for i, row in enumerate(rows):
        missing = REQUIRED_COLUMNS - set(row.keys())
        assert not missing, (
            f"Row {i} in results_{condition}.tsv missing columns: {missing}"
        )

    # At least one "keep" row below baseline
    baseline = BASELINES[condition]
    has_improvement = any(
        row["status"] == "keep" and float(row["val_bpb"]) < baseline
        for row in rows
    )
    assert has_improvement, (
        f"No 'keep' row in results_{condition}.tsv beats the baseline bpb "
        f"of {baseline} for condition {condition!r}"
    )


# ---------------------------------------------------------------------------
# Condition tests
# ---------------------------------------------------------------------------

def test_search_d3() -> None:
    """Verify results_d3.tsv: 70+ rows, valid schema, one keep row below D3 baseline."""
    _check_condition("d3")


def test_search_d1() -> None:
    """Verify results_d1.tsv: 70+ rows, valid schema, one keep row below D1 baseline."""
    _check_condition("d1")


def test_search_d2() -> None:
    """Verify results_d2.tsv: 70+ rows, valid schema, one keep row below D2 baseline."""
    _check_condition("d2")


# ---------------------------------------------------------------------------
# Summary JSON test
# ---------------------------------------------------------------------------

def test_search_results_json() -> None:
    """Verify search_results.json has d1/d2/d3 keys with best_val_bpb and commit."""
    path = PROJECT_ROOT / "search_results.json"

    if not path.exists():
        pytest.skip("search_results.json not yet created")

    data: dict = json.loads(path.read_text(encoding="utf-8"))

    assert {"d1", "d2", "d3"} <= set(data.keys()), (
        f"search_results.json is missing one or more of d1/d2/d3 keys. "
        f"Found: {list(data.keys())}"
    )

    for cond in ["d1", "d2", "d3"]:
        assert "best_val_bpb" in data[cond], f"missing best_val_bpb for {cond}"
        assert isinstance(data[cond]["best_val_bpb"], float), (
            f"best_val_bpb not float for {cond}: {type(data[cond]['best_val_bpb'])}"
        )
        assert "commit" in data[cond], f"missing commit for {cond}"
        assert isinstance(data[cond]["commit"], str) and len(data[cond]["commit"]) >= 7, (
            f"commit for {cond!r} must be a str of length >= 7, "
            f"got {data[cond]['commit']!r}"
        )
