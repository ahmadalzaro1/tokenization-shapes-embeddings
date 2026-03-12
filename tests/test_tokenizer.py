"""
Tokenizer tests for Phase 2 (TOK-01 through TOK-04).
Unit tests fail RED until Plan 02 extends prepare.py.
Integration tests are skipped until tokenizer training runs complete.
"""
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

BASE_CACHE = Path.home() / ".cache" / "autoresearch-arabic"
PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# TOK-01: --vocab-size CLI flag (RED until Plan 02 adds it)
# ---------------------------------------------------------------------------

def test_vocab_size_flag() -> None:
    """Fails RED until Plan 02 adds --vocab-size to prepare.py argparse."""
    result = subprocess.run(
        ["uv", "run", "prepare.py", "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert "--vocab-size" in result.stdout, (
        f"--vocab-size flag not found in prepare.py --help output.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# TOK-02: write_fertility_report() exists (RED until Plan 02 adds it)
# ---------------------------------------------------------------------------

def test_fertility_report_written() -> None:
    """Fails RED until Plan 02 adds write_fertility_report to prepare.py."""
    import prepare  # noqa: PLC0415
    assert hasattr(prepare, "write_fertility_report"), (
        "write_fertility_report not found in prepare module. "
        "Plan 02 must add this function."
    )
    assert callable(prepare.write_fertility_report), (
        "prepare.write_fertility_report is not callable."
    )


# ---------------------------------------------------------------------------
# TOK-03: tokenizer files exist after training (SKIP — integration)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="integration: run after uv run prepare.py --condition d1")
def test_tokenizer_files_exist() -> None:
    """Checks that tokenizer.pkl and token_bytes.npy exist for all conditions."""
    for condition in ["d1", "d2", "d3"]:
        tokenizer_pkl = BASE_CACHE / condition / "tokenizer" / "tokenizer.pkl"
        token_bytes_npy = BASE_CACHE / condition / "tokenizer" / "token_bytes.npy"
        assert tokenizer_pkl.exists(), f"Missing tokenizer.pkl for condition={condition}"
        assert token_bytes_npy.exists(), f"Missing token_bytes.npy for condition={condition}"


# ---------------------------------------------------------------------------
# TOK-04: fertility table has d1/d2/d3 keys (SKIP — integration)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="integration: run after tokenizer sweep completes")
def test_fertility_table_conditions() -> None:
    """Checks fertility_report.json has d1/d2/d3 keys with plausible float values."""
    import json  # noqa: PLC0415

    report_path = BASE_CACHE / "fertility_report.json"
    assert report_path.exists(), f"fertility_report.json not found at {report_path}"
    with report_path.open(encoding="utf-8") as f:
        fertility: dict = json.load(f)

    for condition in ["d1", "d2", "d3"]:
        assert condition in fertility, f"Missing condition {condition!r} in fertility_report.json"
        assert "8192" in fertility[condition], (
            f"Missing vocab_size key '8192' for condition {condition!r}"
        )
        value = fertility[condition]["8192"]
        assert isinstance(value, float), (
            f"fertility[{condition!r}]['8192'] is not a float: {value!r}"
        )
        assert 0.5 <= value <= 5.0, (
            f"fertility[{condition!r}]['8192'] = {value} is outside plausible range [0.5, 5.0]"
        )

    # D3 PUA encoding is more compact — should have lower fertility than D1
    assert fertility["d3"]["8192"] < fertility["d1"]["8192"], (
        f"D3 fertility ({fertility['d3']['8192']}) must be lower than "
        f"D1 fertility ({fertility['d1']['8192']}) — PUA encoding is more compact"
    )


# ---------------------------------------------------------------------------
# TOK-04 (extra): D3 tokenizer roundtrip (SKIP — integration)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="integration: run after uv run prepare.py --condition d3")
def test_d3_tokenizer_roundtrip() -> None:
    """Checks that D3 tokenizer encodes and decodes PUA strings without loss."""
    tokenizer_pkl = BASE_CACHE / "d3" / "tokenizer" / "tokenizer.pkl"
    assert tokenizer_pkl.exists(), f"Missing d3 tokenizer.pkl at {tokenizer_pkl}"

    with tokenizer_pkl.open("rb") as f:
        enc = pickle.load(f)

    # Hardcoded PUA codepoint sequence (representative D3-encoded token sequence)
    test_d3: str = chr(0xE000) + chr(0xE001) + " " + chr(0xE002)
    encoded = enc.encode_ordinary(test_d3)
    decoded = enc.decode(encoded)
    assert decoded == test_d3, (
        f"D3 tokenizer roundtrip failed: {test_d3!r} -> encoded={encoded} -> {decoded!r}"
    )
