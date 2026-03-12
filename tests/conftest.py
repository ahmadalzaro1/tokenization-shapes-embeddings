import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import re
import sys

# ---------------------------------------------------------------------------
# 100 rows of real classical Arabic sentences with harakat (diacritized).
# Using short Quran-style verses — guaranteed to contain harakat.
# ---------------------------------------------------------------------------
ARABIC_SAMPLE_TEXTS = [
    "\u0628\u0650\u0633\u0652\u0645\u0650 \u0627\u0644\u0644\u0651\u064e\u0647\u0650 \u0627\u0644\u0631\u0651\u064e\u062d\u0652\u0645\u064e\u0646\u0650 \u0627\u0644\u0631\u0651\u064e\u062d\u0650\u064a\u0645\u0650",
    "\u0627\u0644\u0652\u062d\u064e\u0645\u0652\u062f\u064f \u0644\u0650\u0644\u0651\u064e\u0647\u0650 \u0631\u064e\u0628\u0651\u0650 \u0627\u0644\u0652\u0639\u064e\u0627\u0644\u064e\u0645\u0650\u064a\u0646\u064e",
    "\u0627\u0644\u0631\u0651\u064e\u062d\u0652\u0645\u064e\u0646\u064f \u0627\u0644\u0631\u0651\u064e\u062d\u0650\u064a\u0645\u064f \u0645\u064e\u0627\u0644\u0650\u0643\u0650 \u064a\u064e\u0648\u0652\u0645\u0650 \u0627\u0644\u062f\u0651\u0650\u064a\u0646\u0650",
    "\u0625\u0650\u064a\u0651\u064e\u0627\u0643\u064e \u0646\u064e\u0639\u0652\u0628\u064f\u062f\u064f \u0648\u064e\u0625\u0650\u064a\u0651\u064e\u0627\u0643\u064e \u0646\u064e\u0633\u0652\u062a\u064e\u0639\u0650\u064a\u0646\u064f",
    "\u0627\u0647\u0652\u062f\u0650\u0646\u064e\u0627 \u0627\u0644\u0635\u0651\u0650\u0631\u064e\u0627\u0637\u064e \u0627\u0644\u0652\u0645\u064f\u0633\u0652\u062a\u064e\u0642\u0650\u064a\u0645\u064e",
    "\u0635\u0650\u0631\u064e\u0627\u0637\u064e \u0627\u0644\u0651\u064e\u0630\u0650\u064a\u0646\u064e \u0623\u064e\u0646\u0652\u0639\u064e\u0645\u0652\u062a\u064e \u0639\u064e\u0644\u064e\u064a\u0652\u0647\u0650\u0645\u0652",
    "\u063a\u064e\u064a\u0652\u0631\u0650 \u0627\u0644\u0652\u0645\u064e\u063a\u0652\u0636\u064f\u0648\u0628\u0650 \u0639\u064e\u0644\u064e\u064a\u0652\u0647\u0650\u0645\u0652 \u0648\u064e\u0644\u064e\u0627 \u0627\u0644\u0636\u0651\u064e\u0627\u0644\u0651\u0650\u064a\u0646\u064e",
    "\u0627\u0644\u0645 \u0630\u064e\u0644\u0650\u0643\u064e \u0627\u0644\u0652\u0643\u0650\u062a\u064e\u0627\u0628\u064f \u0644\u064e\u0627 \u0631\u064e\u064a\u0652\u0628\u064e \u0641\u0650\u064a\u0647\u0650",
    "\u0647\u064f\u062f\u064b\u0649 \u0644\u0650\u0651\u0644\u0652\u0645\u064f\u062a\u0651\u064e\u0642\u0650\u064a\u0646\u064e \u0627\u0644\u0651\u064e\u0630\u0650\u064a\u0646\u064e \u064a\u064f\u0624\u0652\u0645\u0650\u0646\u064f\u0648\u0646\u064e",
    "\u0648\u064e\u0645\u0650\u0645\u0651\u064e\u0627 \u0631\u064e\u0632\u064e\u0642\u0652\u0646\u064e\u0627\u0647\u064f\u0645\u0652 \u064a\u064f\u0646\u0652\u0641\u0650\u0642\u064f\u0648\u0646\u064e",
] * 10  # 100 rows total

HARAKAT_RE = re.compile(r'[\u064B-\u0652\u0670]')
HARAKAT_CODEPOINTS = [
    '\u064B', '\u064C', '\u064D', '\u064E', '\u064F',
    '\u0650', '\u0651', '\u0652', '\u0670',
]
PUA_BASE = 0xE000
ARABIC_LETTERS = [chr(c) for c in range(0x0621, 0x063B)] + [chr(c) for c in range(0x0641, 0x064B)]


@pytest.fixture
def tiny_arabic_texts():
    return list(ARABIC_SAMPLE_TEXTS)


@pytest.fixture
def cache_base(tmp_path):
    base = tmp_path / "autoresearch-arabic"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_shard(path: Path, texts: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, path)
    return path


def _build_atomic_mapping():
    """Minimal copy of build_atomic_mapping() for fixture use."""
    mapping = {}
    idx = 0
    shaddah = '\u0651'
    for letter in ARABIC_LETTERS:
        for harakah in HARAKAT_CODEPOINTS:
            mapping[letter + harakah] = chr(PUA_BASE + idx)
            idx += 1
    for letter in ARABIC_LETTERS:
        for harakah in HARAKAT_CODEPOINTS:
            if harakah == shaddah:
                continue
            mapping[letter + shaddah + harakah] = chr(PUA_BASE + idx)
            idx += 1
        mapping[letter + shaddah] = chr(PUA_BASE + idx)
        idx += 1
    for harakah in HARAKAT_CODEPOINTS:
        mapping[harakah] = chr(PUA_BASE + idx)
        idx += 1
    return mapping


@pytest.fixture
def d1_shard_path(tmp_path, tiny_arabic_texts):
    return _write_shard(tmp_path / "d1" / "data" / "shard_00000.parquet", tiny_arabic_texts)


@pytest.fixture
def d2_shard_path(tmp_path, tiny_arabic_texts):
    stripped = [HARAKAT_RE.sub('', t) for t in tiny_arabic_texts]
    return _write_shard(tmp_path / "d2" / "data" / "shard_00000.parquet", stripped)


@pytest.fixture
def d3_shard_path(tmp_path, tiny_arabic_texts):
    mapping = _build_atomic_mapping()
    pua_texts = []
    for text in tiny_arabic_texts:
        result = []
        i = 0
        n = len(text)
        while i < n:
            if i + 2 < n and text[i:i+3] in mapping:
                result.append(mapping[text[i:i+3]])
                i += 3
            elif i + 1 < n and text[i:i+2] in mapping:
                result.append(mapping[text[i:i+2]])
                i += 2
            elif text[i] in mapping:
                result.append(mapping[text[i]])
                i += 1
            else:
                result.append(text[i])
                i += 1
        pua_texts.append(''.join(result))
    return _write_shard(tmp_path / "d3" / "data" / "shard_00000.parquet", pua_texts)


def _encode_d3(texts: list[str]) -> list[str]:
    """Encode a list of Arabic texts to D3 PUA encoding."""
    mapping = _build_atomic_mapping()
    pua_texts: list[str] = []
    for text in texts:
        result: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            if i + 2 < n and text[i:i + 3] in mapping:
                result.append(mapping[text[i:i + 3]])
                i += 3
            elif i + 1 < n and text[i:i + 2] in mapping:
                result.append(mapping[text[i:i + 2]])
                i += 2
            elif text[i] in mapping:
                result.append(mapping[text[i]])
                i += 1
            else:
                result.append(text[i])
                i += 1
        pua_texts.append("".join(result))
    return pua_texts


@pytest.fixture
def tiny_corpus_parquet(tmp_path):
    """Factory fixture that creates a tiny two-shard corpus for a given condition.

    Usage in tests::

        def test_something(tiny_corpus_parquet):
            cache_root = tiny_corpus_parquet("d1")
            # monkeypatch prepare.BASE_CACHE = str(cache_root)

    Returns a callable ``make_corpus(condition: str) -> Path`` where the
    returned Path is *tmp_path* (the cache-base root), so callers can
    monkeypatch ``prepare.BASE_CACHE`` to point at it.
    """

    def make_corpus(condition: str) -> Path:
        base_texts: list[str] = list(ARABIC_SAMPLE_TEXTS[:50])

        if condition == "d1":
            shard_texts = base_texts
        elif condition == "d2":
            shard_texts = [HARAKAT_RE.sub("", t) for t in base_texts]
        elif condition == "d3":
            shard_texts = _encode_d3(base_texts)
        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        data_dir: Path = tmp_path / condition / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        _write_shard(data_dir / "shard_00000.parquet", shard_texts)  # train shard
        _write_shard(data_dir / "shard_00001.parquet", shard_texts)  # val shard

        meta: Path = tmp_path / condition / "metadata.txt"
        meta.write_text(
            f"condition={condition}\n"
            "train_docs=50\n"
            "val_docs=50\n"
            "train_shards=1\n"
            "val_shard=1\n"
            "val_filename=shard_00001.parquet\n",
            encoding="utf-8",
        )

        return tmp_path

    return make_corpus
