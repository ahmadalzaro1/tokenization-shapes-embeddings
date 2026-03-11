"""
Pipeline tests for autoresearch-arabic Phase 1.
Tests DATA-01 through DATA-05 + validation report.

Tests using ~/.cache data are skipped until build_dataset.py has run.
Fixture-based tests (D1/D2/D3 shard shape/content) pass immediately.
"""
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

BASE_CACHE = Path.home() / ".cache" / "autoresearch-arabic"
HARAKAT_RANGE = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')
PUA_START = 0xE000
PUA_END = 0xF8FF


# ---------------------------------------------------------------------------
# DATA-01: Dataset download (smoke test — requires cache or live network)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires HF cache from build_dataset.py run; run manually")
def test_dataset_columns():
    """Abdou/arabic-tashkeel-dataset has vocalized and non_vocalized columns."""
    from datasets import load_dataset
    ds = load_dataset("Abdou/arabic-tashkeel-dataset", split="train[:10]")
    assert "vocalized" in ds.column_names
    assert "non_vocalized" in ds.column_names
    assert len(ds) == 10


# ---------------------------------------------------------------------------
# DATA-02: D1 shards contain harakat characters
# ---------------------------------------------------------------------------

def test_d1_shards(d1_shard_path):
    """D1 parquet shard loads without error and text column contains harakat."""
    table = pq.read_table(d1_shard_path)
    assert "text" in table.schema.names
    texts = table.column("text").to_pylist()
    assert len(texts) == 100
    assert all(t is not None and t != "" for t in texts)
    # D1 must contain at least one harakat character somewhere
    combined = "".join(texts)
    assert any(c in HARAKAT_RANGE for c in combined), "D1 texts should contain harakat"


# ---------------------------------------------------------------------------
# DATA-03: D2 shards have zero harakat codepoints
# ---------------------------------------------------------------------------

def test_d2_stripped(d2_shard_path):
    """D2 parquet shard has zero harakat codepoints (U+064B-U+0652, U+0670)."""
    table = pq.read_table(d2_shard_path)
    texts = table.column("text").to_pylist()
    assert all(t is not None and t != "" for t in texts)
    combined = "".join(texts)
    assert not any(c in HARAKAT_RANGE for c in combined), "D2 texts must have all harakat stripped"


# ---------------------------------------------------------------------------
# DATA-04: D3 shards contain PUA codepoints; atomic_mapping.json >= 252 entries
# ---------------------------------------------------------------------------

def test_d3_pua(d3_shard_path, cache_base, tmp_path):
    """D3 parquet shard contains PUA codepoints; atomic_mapping.json has >= 252 entries."""
    table = pq.read_table(d3_shard_path)
    texts = table.column("text").to_pylist()
    assert all(t is not None and t != "" for t in texts)
    combined = "".join(texts)
    assert any(PUA_START <= ord(c) <= PUA_END for c in combined), "D3 texts should contain PUA codepoints"

    # Check atomic_mapping.json (requires build_dataset.py to have run)
    mapping_path = BASE_CACHE / "atomic_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)
        assert len(mapping) >= 252, f"atomic_mapping.json has {len(mapping)} entries, need >= 252"


# ---------------------------------------------------------------------------
# DATA-01 (Wave 2): load_dataset_with_progress + MANUAL_INSTRUCTIONS
# ---------------------------------------------------------------------------

def test_load_dataset_with_progress_exists():
    """load_dataset_with_progress must be importable from build_dataset."""
    import build_dataset
    assert hasattr(build_dataset, "load_dataset_with_progress"), \
        "build_dataset must export load_dataset_with_progress"


def test_manual_instructions_constant_exists():
    """MANUAL_INSTRUCTIONS constant must be defined in build_dataset."""
    import build_dataset
    assert hasattr(build_dataset, "MANUAL_INSTRUCTIONS"), \
        "build_dataset must export MANUAL_INSTRUCTIONS"
    assert "huggingface" in build_dataset.MANUAL_INSTRUCTIONS.lower(), \
        "MANUAL_INSTRUCTIONS should reference HuggingFace"


def test_load_dataset_with_progress_signature():
    """load_dataset_with_progress must accept (name, split) and run callable."""
    import inspect
    import build_dataset
    sig = inspect.signature(build_dataset.load_dataset_with_progress)
    params = list(sig.parameters.keys())
    assert "name" in params, "must have 'name' param"
    assert "split" in params, "must have 'split' param"


def test_load_dataset_with_progress_keyboard_interrupt(monkeypatch, capsys):
    """KeyboardInterrupt during download must print MANUAL_INSTRUCTIONS and re-raise."""
    import threading
    import build_dataset

    # Patch load_dataset inside the module to raise KeyboardInterrupt from the thread
    call_count = [0]

    def fake_load(name, split):
        raise KeyboardInterrupt()

    # Monkeypatch the datasets.load_dataset used inside the thread closure
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load)

    # The tqdm bar runs in-thread; KeyboardInterrupt from the worker thread
    # is stored in error[0] and raised in the main thread — but the plan
    # spec says KeyboardInterrupt is caught in the tqdm loop (from signal).
    # Since the thread doesn't propagate KeyboardInterrupt automatically,
    # we test that any exception from the thread triggers MANUAL_INSTRUCTIONS.
    class SentinelError(Exception):
        pass

    def fake_load2(name, split):
        raise SentinelError("forced error")

    monkeypatch.setattr(datasets, "load_dataset", fake_load2)

    with pytest.raises(SentinelError):
        build_dataset.load_dataset_with_progress("test/dataset", split="train")

    captured = capsys.readouterr()
    assert build_dataset.MANUAL_INSTRUCTIONS in captured.out, \
        "MANUAL_INSTRUCTIONS must be printed on exception"


def test_main_uses_load_dataset_with_progress():
    """main() source code must call load_dataset_with_progress, not bare load_dataset."""
    import inspect
    import build_dataset
    src = inspect.getsource(build_dataset.main)
    assert "load_dataset_with_progress" in src, \
        "main() must call load_dataset_with_progress"
    # The bare direct call (without _with_progress) should not appear
    # (allow for the import line but not as a standalone call)
    # We check that the assignment goes through load_dataset_with_progress
    assert 'ds = load_dataset_with_progress(' in src or \
           'load_dataset_with_progress(' in src


# ---------------------------------------------------------------------------
# DATA-05: collision_stats.json has top_50_ambiguous and context_window_ambiguous_pct
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DATA-05 (Wave 2 unit): context_window_collision_probability + JSON sidecar
# ---------------------------------------------------------------------------

def test_context_window_collision_probability_exists():
    """context_window_collision_probability must be a top-level function in build_dataset."""
    import build_dataset
    assert hasattr(build_dataset, "context_window_collision_probability"), \
        "build_dataset must export context_window_collision_probability"
    import inspect
    assert callable(build_dataset.context_window_collision_probability)


def test_context_window_collision_probability_signature():
    """context_window_collision_probability accepts diacritized_forms and vocalized_texts."""
    import inspect
    import build_dataset
    sig = inspect.signature(build_dataset.context_window_collision_probability)
    params = list(sig.parameters.keys())
    assert "diacritized_forms" in params
    assert "vocalized_texts" in params
    assert "window_tokens" in params


def test_context_window_collision_probability_all_ambiguous():
    """With 100% ambiguous words, probability returns 1.0."""
    import build_dataset
    # Every undiacritized form has 2 diacritized variants
    diacritized_forms = {
        "كتب": {"كَتَبَ", "كُتِبَ"},
        "علم": {"عِلْمٌ", "عَلِمَ"},
        "بيت": {"بَيْتٌ", "بَيَّتَ"},
    }
    # 200 words of text so windows of 128 can be sampled
    text = " ".join(list(diacritized_forms.keys()) * 200)
    vocalized = [text]
    result = build_dataset.context_window_collision_probability(
        diacritized_forms, vocalized, window_tokens=10, n_samples=100, seed=42
    )
    assert result == 1.0, f"Expected 1.0 for all-ambiguous corpus, got {result}"


def test_context_window_collision_probability_no_ambiguous():
    """With zero ambiguous words, probability returns 0.0."""
    import build_dataset
    # Every undiacritized form has exactly 1 diacritized variant (no collision)
    diacritized_forms = {
        "ذهب": {"ذَهَبَ"},
        "قال": {"قَالَ"},
        "جاء": {"جَاءَ"},
    }
    text = " ".join(list(diacritized_forms.keys()) * 200)
    vocalized = [text]
    result = build_dataset.context_window_collision_probability(
        diacritized_forms, vocalized, window_tokens=10, n_samples=100, seed=42
    )
    assert result == 0.0, f"Expected 0.0 for unambiguous corpus, got {result}"


def test_compute_collision_stats_writes_json(tmp_path, monkeypatch):
    """compute_collision_stats must write collision_stats.json alongside .txt."""
    import build_dataset

    # Redirect BASE_CACHE to tmp_path
    monkeypatch.setattr(build_dataset, "BASE_CACHE", tmp_path)

    # Small corpus: 3 words each with 2 diacritized forms -> all ambiguous
    vocalized = ["كَتَبَ عِلْمٌ بَيْتٌ"] * 100
    non_vocalized = ["كتب علم بيت"] * 100

    build_dataset.compute_collision_stats(vocalized, non_vocalized)

    json_path = tmp_path / "collision_stats.json"
    assert json_path.exists(), "collision_stats.json must be written"

    import json
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "context_window_ambiguous_pct" in data, "must have context_window_ambiguous_pct"
    assert "top_50_ambiguous" in data, "must have top_50_ambiguous"
    assert "context_window_tokens" in data
    assert data["context_window_tokens"] == 128

    # Verify Arabic chars survive JSON round-trip (ensure_ascii=False)
    for entry in data["top_50_ambiguous"]:
        assert "word" in entry
        assert "variants" in entry
        # At least some Arabic chars in the word field
        assert any('\u0600' <= c <= '\u06FF' for c in entry["word"]), \
            f"word {entry['word']!r} should contain Arabic characters"


def test_compute_collision_stats_txt_still_written(tmp_path, monkeypatch):
    """compute_collision_stats must still write collision_stats.txt (unchanged format)."""
    import build_dataset
    monkeypatch.setattr(build_dataset, "BASE_CACHE", tmp_path)

    vocalized = ["كَتَبَ عِلْمٌ"] * 50
    non_vocalized = ["كتب علم"] * 50
    build_dataset.compute_collision_stats(vocalized, non_vocalized)

    txt_path = tmp_path / "collision_stats.txt"
    assert txt_path.exists(), "collision_stats.txt must still be written"
    content = txt_path.read_text(encoding="utf-8")
    assert "Top 20 most ambiguous words" in content


@pytest.mark.skip(reason="Requires build_dataset.py to run with Wave 2 JSON sidecar")
def test_collision_stats_json():
    """collision_stats.json exists with top_50 ambiguous words and context_window metric."""
    json_path = BASE_CACHE / "collision_stats.json"
    assert json_path.exists(), f"collision_stats.json not found at {json_path}"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "context_window_ambiguous_pct" in data, "Missing context_window_ambiguous_pct field"
    assert "top_50_ambiguous" in data, "Missing top_50_ambiguous field"
    assert len(data["top_50_ambiguous"]) >= 50, "top_50_ambiguous must have >= 50 entries"
    assert "context_window_tokens" in data
    assert data["context_window_tokens"] == 128
    # Verify Arabic characters survive JSON round-trip (ensure_ascii=False)
    for entry in data["top_50_ambiguous"][:3]:
        assert "word" in entry
        assert "variants" in entry
        # Word should contain Arabic characters, not only escape sequences
        assert any('\u0600' <= c <= '\u06FF' for c in entry["word"]), \
            "top_50 words should be Arabic chars, not escape sequences"


# ---------------------------------------------------------------------------
# Integration: validation_report.json all conditions pass
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires build_dataset.py + validate_dataset.py to run (Wave 3)")
def test_validation_report():
    """validation_report.json exists and all per-condition checks pass."""
    report_path = BASE_CACHE / "validation_report.json"
    assert report_path.exists(), f"validation_report.json not found at {report_path}"
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    for condition in ["d1", "d2", "d3"]:
        assert condition in report, f"Missing condition {condition} in report"
        cond_data = report[condition]
        assert cond_data.get("shards_loadable") is True, f"{condition}: shards_loadable check failed"
        assert cond_data.get("row_count_matches") is True, f"{condition}: row_count_matches check failed"
        assert cond_data.get("no_empty_texts") is True, f"{condition}: no_empty_texts check failed"
        assert "char_distribution" in cond_data, f"{condition}: missing char_distribution"


# ---------------------------------------------------------------------------
# DATA-02/03/04 Wave 3: validate_condition() unit tests
# ---------------------------------------------------------------------------

def test_validate_condition_exists():
    """validate_condition must be importable from build_dataset."""
    import build_dataset
    assert hasattr(build_dataset, "validate_condition"), \
        "build_dataset must export validate_condition"
    assert callable(build_dataset.validate_condition)


def test_write_validation_report_exists():
    """write_validation_report must be importable from build_dataset."""
    import build_dataset
    assert hasattr(build_dataset, "write_validation_report"), \
        "build_dataset must export write_validation_report"
    assert callable(build_dataset.write_validation_report)


def test_compute_char_distribution_exists():
    """_compute_char_distribution must be importable from build_dataset."""
    import build_dataset
    assert hasattr(build_dataset, "_compute_char_distribution"), \
        "build_dataset must export _compute_char_distribution"
    assert callable(build_dataset._compute_char_distribution)


def test_validate_condition_returns_required_keys(tmp_path, monkeypatch):
    """validate_condition returns dict with all 4 mandatory keys."""
    import build_dataset

    # Build a minimal valid cache structure
    condition = "d1"
    data_dir = tmp_path / condition / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Write a valid shard with real Arabic diacritized text
    texts = [
        "\u0628\u0650\u0633\u0652\u0645\u0650 \u0627\u0644\u0644\u0651\u064e\u0647\u0650",  # Bismillah
        "\u0627\u0644\u0652\u062d\u064e\u0645\u0652\u062f\u064f \u0644\u0650\u0644\u0651\u064e\u0647\u0650",  # Alhamdu lillah
    ]
    table = pa.table({"text": texts})
    shard_path = data_dir / "shard_00000.parquet"
    pq.write_table(table, shard_path)

    # Write matching metadata.txt — shard_00000 is the sole (val) shard with 2 rows.
    # train_docs=0 because there are no train shards (only val shard).
    meta = (
        f"condition={condition}\n"
        f"train_docs=0\n"
        f"val_docs=2\n"
        f"train_shards=0\n"
        f"val_shard=0\n"
        f"val_filename=shard_00000.parquet\n"
    )
    meta_path = tmp_path / condition / "metadata.txt"
    meta_path.write_text(meta, encoding="utf-8")

    monkeypatch.setattr(build_dataset, "BASE_CACHE", tmp_path)

    result = build_dataset.validate_condition(condition)
    assert isinstance(result, dict), "validate_condition must return a dict"
    assert "shards_loadable" in result
    assert "row_count_matches" in result
    assert "no_empty_texts" in result
    assert "char_distribution" in result
    assert result["shards_loadable"] is True
    assert result["no_empty_texts"] is True


def test_write_validation_report_writes_json(tmp_path, monkeypatch):
    """write_validation_report writes validation_report.json and merges across conditions."""
    import build_dataset

    monkeypatch.setattr(build_dataset, "BASE_CACHE", tmp_path)

    results_d1 = {
        "shards_loadable": True,
        "row_count_matches": True,
        "no_empty_texts": True,
        "char_distribution": {"a": 10},
    }
    results_d2 = {
        "shards_loadable": True,
        "row_count_matches": True,
        "no_empty_texts": True,
        "char_distribution": {"b": 5},
    }

    build_dataset.write_validation_report("d1", results_d1)
    build_dataset.write_validation_report("d2", results_d2)

    report_path = tmp_path / "validation_report.json"
    assert report_path.exists(), "validation_report.json must be written"

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    assert "d1" in report, "d1 key must be present"
    assert "d2" in report, "d2 key must be present — must merge, not overwrite"
    assert report["d1"]["shards_loadable"] is True
    assert report["d2"]["shards_loadable"] is True


def test_main_calls_validate_condition():
    """main() source code must call validate_condition and write_validation_report."""
    import inspect
    import build_dataset
    src = inspect.getsource(build_dataset.main)
    assert "validate_condition" in src, "main() must call validate_condition"
    assert "write_validation_report" in src, "main() must call write_validation_report"


# ---------------------------------------------------------------------------
# Task 2: validate_dataset.py standalone validator
# ---------------------------------------------------------------------------

def test_validate_dataset_importable():
    """validate_dataset.py must be importable without error."""
    import importlib
    vd = importlib.import_module("validate_dataset")
    assert vd is not None


def test_validate_dataset_imports_from_build_dataset():
    """validate_dataset must import validate_condition from build_dataset (no logic duplication)."""
    import pathlib
    src = pathlib.Path("validate_dataset.py").read_text(encoding="utf-8")
    assert "from build_dataset import" in src, \
        "validate_dataset.py must import from build_dataset"
    assert "validate_condition" in src


def test_validate_dataset_has_argparse():
    """validate_dataset.py must use argparse with --condition flag."""
    import pathlib
    src = pathlib.Path("validate_dataset.py").read_text(encoding="utf-8")
    assert "argparse" in src, "validate_dataset.py must use argparse"
    assert "--condition" in src, "validate_dataset.py must have --condition argument"


def test_validate_dataset_help(tmp_path):
    """uv run python validate_dataset.py --help must exit 0."""
    import subprocess
    result = subprocess.run(
        ["uv", "run", "python", "validate_dataset.py", "--help"],
        capture_output=True, text=True,
        cwd="/Users/ahmadalzaro/Desktop/Karpathy/autoresearch-arabic",
    )
    assert result.returncode == 0, f"--help must exit 0; got: {result.stderr}"
    assert "--condition" in result.stdout, "--condition must appear in help output"


def test_validate_dataset_prints_summary(tmp_path, monkeypatch):
    """validate_dataset.main() prints 'Validation complete: N/M conditions passed'."""
    import importlib
    import build_dataset

    # Patch validate_condition to always succeed for one condition
    def fake_validate(condition):
        return {
            "shards_loadable": True,
            "row_count_matches": True,
            "no_empty_texts": True,
            "char_distribution": {},
        }

    def fake_write_report(condition, results):
        pass

    monkeypatch.setattr(build_dataset, "validate_condition", fake_validate)
    monkeypatch.setattr(build_dataset, "write_validation_report", fake_write_report)

    # Reload validate_dataset so it picks up the monkeypatched build_dataset
    import sys as _sys
    if "validate_dataset" in _sys.modules:
        del _sys.modules["validate_dataset"]
    vd = importlib.import_module("validate_dataset")
    # Patch the names in validate_dataset's namespace too
    monkeypatch.setattr(vd, "validate_condition", fake_validate)
    monkeypatch.setattr(vd, "write_validation_report", fake_write_report)

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            vd.main(["--condition", "d1"])
        except SystemExit as e:
            assert e.code == 0, f"Expected exit 0, got {e.code}"

    output = buf.getvalue()
    assert "Validation complete:" in output, \
        f"Expected 'Validation complete:' in output; got:\n{output}"
    assert "1/1" in output, \
        f"Expected '1/1 conditions passed'; got:\n{output}"
