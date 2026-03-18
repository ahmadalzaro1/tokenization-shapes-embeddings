<div align="center">
<h2 align="center">
  <b>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
    Tokenization Shapes Embeddings
    <br/>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
  </b>
</h2>
<p><b>Ahmad Al-Zaro</b></p>
</div>

<p align="center">
  <a href="paper.pdf">Paper</a> &nbsp;|&nbsp;
  <a href="#overview">Overview</a> &nbsp;|&nbsp;
  <a href="#results">Results</a> &nbsp;|&nbsp;
  <a href="#reproduce">Reproduce</a> &nbsp;|&nbsp;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="assets/iso_data_scaling_curves.png" width="600" />
</p>
<p align="center"><em>
  Iso-data scaling curves: the D1 advantage over D3 shrinks from 25% at 5M base letters to 1.7% at 500M, but does not close at this model scale.
</em></p>

---

This is the official repository for **The Compositionality-Atomicity Tradeoff**, a controlled mechanistic study of how tokenization design shapes learned representations in language models. Using Arabic diacritical marks (harakat) as a natural test case, we show that "perfect" atomic tokenization destroys compositional structure in the embedding layer — and explain why mechanistically.

## Overview

Standard BPE on diacritized Arabic text preserves compositional structure: harakah (diacritical marks) share subword fragments with their base letters, allowing the model to learn relationships through shared representations. We designed an atomic encoding (D3) that maps each letter+diacritic pair to a single indivisible token — eliminating fragmentation entirely.

**D3 should win. It does not.**

Across 200+ architecture-search experiments, atomic encoding consistently yields higher loss than standard BPE. We explain this by analyzing the learned embedding matrices:

- In **D3** (atomic): diacritical variants of the same letter have near-random cosine similarity (mean **0.125**). The model treats *ba* and *bu* as unrelated tokens.
- In **D1** (compositional): harakah tokens cluster together (cosine **0.224**), well-separated from base letters. The model inherits structure from shared subwords.

We term this the **compositionality-atomicity tradeoff**: atomic tokenization eliminates fragmentation but forces the model to learn compositional relationships from scratch rather than inheriting them from shared subword structure.

### Three Encoding Strategies

| Condition | Strategy | Description |
|:----|:----|:----|
| **D1** | Compositional | Raw diacritized text, standard BPE. Harakah are Unicode combining characters. |
| **D2** | Lossy | Diacritics stripped entirely. What every Arabic LLM does today. |
| **D3** | Atomic | Each letter+diacritic pair mapped to a single unique token (PUA codepoints). |

## Results

### BPBL Comparison (10-seed robust evaluation)

We introduce **Bits Per Base Letter (BPBL)**, a tokenizer-fair metric that removes the fertility confound inherent in standard bits-per-byte comparisons.

| Condition | Median BPBL | Mean BPBL | Std |
|:---|:---:|:---:|:---:|
| **D1** (compositional) | **2.93** | 2.95 | 0.08 |
| **D3** (atomic) | 3.41 | 3.43 | 0.12 |

The ordering **D1 < D3 < D2** (lower is better) is preserved across all seeds, architectures, and data scales.

### Iso-Data Scaling

The D1 advantage is data-scale dependent — it shrinks with more data but does not close at this model scale.

| Base Letters | D1 BPBL | D3 BPBL | Gap |
|:---|:---:|:---:|:---:|
| 5M | 4.23 | 5.30 | 25.3% |
| 15M | 3.55 | 4.17 | 17.5% |
| 30M | 3.26 | 3.73 | 14.4% |
| 50M | 3.10 | 3.49 | 12.6% |
| 100M | 2.93 | 3.19 | 8.9% |
| 200M | 2.78 | 2.89 | 4.0% |
| 500M | 2.69 | 2.74 | 1.7% |

### Architecture Control

<p align="center">
  <img src="assets/iso_data_arch_comparison.png" width="600" />
</p>
<p align="center"><em>
  D3 with D1's architecture (depth-4) still loses to D1, ruling out architecture as a confound.
</em></p>

### Embedding Structure

| Metric | D1 | D3 |
|:---|:---:|:---:|
| Intra-group cosine similarity | 0.224 | 0.125 |
| Interpretation | Compositional clusters | Near-random |

In D3, the embedding layer does not organize diacritical variants into compositional clusters. Tokens that *should* be related are treated as independent.

## Reproduce

**Requirements**: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/ahmadalzaro1/tokenization-shapes-embeddings.git
cd tokenization-shapes-embeddings
uv sync

# Prepare data (downloads from HuggingFace, builds D1/D2/D3 shards)
uv run src/build_dataset.py
uv run src/prepare.py

# Train a single model
uv run src/train.py

# Run experiments
uv run experiments/iso_data_scaling.py      # Scaling curves
uv run experiments/bpbl_evaluation.py       # BPBL metric (10 seeds)
uv run experiments/embedding_analysis.py    # Embedding analysis
uv run experiments/architecture_control.py  # Architecture confound control
```

### Project Structure

```
src/                         Core training code
  train.py                   GPT training loop (MLX)
  prepare.py                 Tokenizer + data shard preparation
  build_dataset.py           D1/D2/D3 encoding pipeline
  validate_dataset.py        Dataset integrity checks
  extract_best.py            Extract best config from search results
  shared.py                  Shared utilities

experiments/                 Experiment scripts
  bpbl_evaluation.py         BPBL metric with 10-seed robust eval
  embedding_analysis.py      Embedding cosine similarity analysis
  iso_data_scaling.py        Iso-data scaling curves (D1 vs D3)
  architecture_control.py    Architecture confound control

results/                     Raw experiment data (JSONs)
assets/                      Figures
tests/                       pytest suite
```

### Dataset

We use the [arabic-tashkeel-dataset](https://huggingface.co/datasets/Abdou/arabic-tashkeel-dataset) (MIT license) — 1.5M examples of fully vocalized classical Arabic text from the Tashkeela corpus. Running `src/build_dataset.py` downloads it automatically.

### Built With

- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework
- [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — Karpathy's autoresearch ported to MLX
- 200+ training runs on a single Mac

## Citation

```bibtex
@article{alzaro2026compositionality,
  title   = {The Compositionality--Atomicity Tradeoff: How Tokenization Design Shapes Embedding Structure in Language Models},
  author  = {Al-Zaro, Ahmad},
  year    = {2026}
}
```

## License

MIT
