# Paper Outline

## Working Title Candidates

1. **The Compositionality-Atomicity Tradeoff: How Tokenization Design Shapes Embedding Structure in Language Models**
2. Tokenization Shapes Representations: A Mechanistic Study of Arabic Diacritical Encoding
3. Opening the Embedding Layer: Why Perfect Tokenization Produces Worse Language Models

## One-Sentence Contribution

We show through controlled experiments on Arabic that "perfect" atomic tokenization destroys compositional structure in the embedding layer (cosine similarity = 0.125), creating a data efficiency disadvantage that shrinks from 25% to 3% with scale — the first mechanistic proof of how tokenization design propagates into learned representations.

## What The Paper Claims

- Tokenization design directly shapes internal model representations (mechanistic interpretability finding)
- Atomic tokenization (D3) destroys compositionality in the embedding space — proved by near-random cosine similarity (0.125) between tokens that should be related
- The compositionality-atomicity tradeoff is data-scale dependent: 25% gap at 5M letters → 3% at 200M letters
- BPBL (Bits Per Base Letter) is a fair, tokenizer-agnostic metric for cross-tokenizer comparison
- For Arabic specifically: stripping harakat (D2) is worst, atomic encoding (D3) is middle, raw diacritized (D1) is best
- The tradeoff generalizes conceptually to any morphologically rich language

## What The Paper Must Not Claim

- D3 would win at larger scale (we show the TREND but can't prove the crossover)
- Results generalize to large models (billions of parameters)
- Results generalize to all Arabic domains (we tested classical/vocalized Arabic only)
- D3 is useless (it closes the gap rapidly — it may win at sufficient scale)

## Abstract Skeleton

We present a controlled mechanistic study of how tokenization design shapes learned representations in language models. Using Arabic diacritical marks as a natural test case, we compare three encoding strategies: standard BPE on diacritized text (compositional), harakat-stripped BPE (lossy), and a novel atomic encoding mapping letter+diacritic pairs to unique tokens (atomic). Across 200+ architecture search experiments and a tokenizer-fair metric (Bits Per Base Letter), we find that the atomic encoding — despite achieving perfect tokenization with zero fragmentation and 15% lower fertility — produces worse models than standard BPE. Mechanistic analysis of the embedding layer reveals why: atomic tokens have near-random cosine similarity (0.125) between diacritical variants of the same letter, proving the model fails to learn compositional structure. Iso-data scaling experiments show this disadvantage is data-dependent, shrinking from 25% at 5M base letters to 3% at 200M. We term this the compositionality-atomicity tradeoff: atomic tokenization eliminates fragmentation but forces the model to learn compositional relationships from scratch rather than inheriting them from shared subword structure. This tradeoff applies to any morphologically rich language and provides a mechanistic framework for tokenization design decisions.

## Key Numbers (from experiments)

| Metric | Value | What It Means |
|--------|-------|--------------|
| D3 intra-group cosine similarity | 0.125 | Embedding layer treats related tokens as strangers |
| D1 harakah cluster cosine | 0.224 | D1 learns structured, compositional representations |
| BPBL median D1 (10 seeds) | 2.933 | D1 needs fewer bits per Arabic letter (better) |
| BPBL median D3 (10 seeds) | 3.412 | D3 needs more bits (worse) |
| Scaling gap at 5M | 25% | Huge advantage for D1 with little data |
| Scaling gap at 200M | 3% | Advantage almost disappears with lots of data |
| Architecture search runs | 200+ | Rigorous, not cherry-picked |
| Unique PUA tokens in D3 | 327 | Each letter+diacritic combo is its own token |
| Arabic base letters analyzed | 33 | Full coverage of Arabic alphabet |

## Figure and Table Plan

### Figure 1: Encoding Comparison Diagram
Visual showing the same Arabic word encoded three ways (D1, D2, D3) and how BPE tokenizes each.

### Figure 2: Embedding Heatmap (THE CENTERPIECE)
D3 per-letter cosine similarity heatmaps showing near-random similarity. This is the mechanistic proof. Reader should immediately see "these should be similar but they're not."

### Figure 3: Embedding Space PCA
Side-by-side D1 vs D3 PCA scatter. D1 shows structured clusters. D3 shows uniform scatter. Visual proof of compositionality vs atomicity.

### Figure 4: Scaling Curves (THE MAIN RESULT)
BPBL vs base letters processed (5M → 500M). Two lines (D1 blue, D3 red) converging. Error bands. Log scale version. Shows the tradeoff is data-dependent.

### Figure 5: Architecture Comparison (from exp6)
Three lines: D1 optimal, D3 optimal, D3 with D1's architecture. Shows encoding effect vs architecture effect.

### Table 1: Dataset and Ambiguity Statistics
Corpus size, collision rate, ambiguous form percentage.

### Table 2: Architecture Search Results
Best val_bpb per condition, winning architecture, fertility.

### Table 3: BPBL Results (10 seeds)
Mean, median, std for D1 and D3. Identifies outlier seeds transparently.

### Table 4: Iso-data Scaling
Budget vs D1 BPBL vs D3 BPBL vs gap vs gap%.

### Table 5: Per-letter Cosine Similarity
Selected Arabic letters showing intra-group cosine similarity in D3.

## Section Outline

### 1. Introduction
- Language models learn through tokenization — but how does tokenizer design shape what the model learns internally?
- Arabic diacritics provide a natural controlled experiment: same text, three different tokenizations
- We discover the compositionality-atomicity tradeoff: perfect tokenization can make models worse
- Mechanistic proof via embedding analysis + scaling law characterization
- Contributions: BPBL metric, embedding structure analysis, scaling characterization, practical Arabic NLP guidance

### 2. Background and Related Work
- Arabic diacritical marks and their linguistic role
- BPE tokenization and its known limitations for morphologically rich languages
- Mechanistic interpretability: what it means to look inside the model
- Prior work on tokenization comparison (mostly benchmark-only, no mechanistic analysis)

### 3. Experimental Setup
- Tashkeela corpus and three conditions (D1, D2, D3)
- D3 encoding: how letter+harakah → PUA codepoint works
- Model architecture: small GPT on Apple Silicon MLX
- Architecture search: 70+ experiments per condition
- BPBL metric: definition, why it's needed, how it removes fertility confound

### 4. Results: D1 Beats D3 Despite Perfect Tokenization
- Architecture search results: D1=0.660, D3=0.890, D2=1.020
- Fixed-architecture transfer matrix: ordering survives
- BPBL confirmation: D1 median=2.93, D3 median=3.41 (10 seeds)
- This is surprising — D3 has zero fragmentation and 15% better fertility

### 5. Mechanistic Analysis: Why D3 Loses (KEY SECTION)
- Extract embedding matrices from trained models
- D3: compute intra-group cosine similarity for 33 base letters (327 PUA tokens)
- Result: mean cosine similarity = 0.125 (near-random)
- The model treats بَ and بُ as completely unrelated tokens
- D1: harakah tokens form coherent cluster (cosine = 0.224), well-separated from base letters
- D1 learns compositional structure. D3 doesn't.
- This is the mechanistic cause of D3's worse performance

### 6. Scaling Analysis: Is This Permanent?
- Iso-data scaling: 5M to 500M base letters, 3 seeds per point
- Gap: 25% at 5M → 3% at 200M
- The compositionality advantage shrinks with data but doesn't disappear
- Architecture comparison (exp6): isolates encoding effect from model capacity effect
- Interpretation: D3 CAN learn compositional relationships from co-occurrence, but it needs much more data than D1

### 7. Discussion
- The compositionality-atomicity tradeoff as a general principle
- Applies to any morphologically rich language (Turkish, Finnish, Hebrew, Hindi)
- Prediction: at large model scale with sufficient data, atomic encoding may eventually win
- Practical guidance: don't strip Arabic diacritics; don't use atomic encoding at small scale
- Connection to mechanistic interpretability: tokenization as a tool for probing what models learn

### 8. Limitations
- Small model scale (~5M parameters)
- Single corpus (Tashkeela, classical Arabic)
- No downstream task evaluation
- Architecture confound between conditions (addressed by exp6)
- Gradient step imbalance in iso-data comparison (acknowledged)
- AI-assisted code development

### 9. Conclusion
- Tokenization design shapes internal representations — we proved it mechanistically
- The compositionality-atomicity tradeoff is real, quantifiable, and data-scale dependent
- BPBL provides a fair metric for cross-tokenizer comparison
- For Arabic practitioners: keep your diacritics, skip the fancy encoding (for now)

## Target Venues

| Venue | Deadline (est.) | Fit |
|-------|----------------|-----|
| EMNLP 2026 Findings | June 2026 | Strong — mechanistic + practical |
| BlackboxNLP @ EMNLP | Sept 2026 | Perfect — interpretability workshop |
| COLM 2026 | TBD | Good — language modeling focus |
| ArabicNLP @ EMNLP | Sept 2026 | Fallback — Arabic NLP audience |

---
*Last updated: 2026-03-17 after Phase 5 experiments, 10-seed robust BPBL, extended scaling, and Gemini audit*
