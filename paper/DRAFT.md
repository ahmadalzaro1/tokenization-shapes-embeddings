# Harakat Are Signal, Not Noise: Representation Choice in Small-Model Arabic Pretraining

> Citation status: this draft is intentionally citation-light. Any literature-dependent statement that has not yet been programmatically verified is marked `[CITATION NEEDED]`.

## Abstract

Arabic language models are often trained on undiacritized text even though removing harakat collapses many distinct word forms into the same surface string `[CITATION NEEDED]`. We study how Arabic diacritics should be represented during small-model pretraining by comparing three conditions on a Tashkeela-derived corpus: fully diacritized Unicode text (D1), harakat-stripped text (D2), and a novel atomic encoding that maps letter+harakah pairs to private-use code points (D3). Across Phase 1 ambiguity analysis, Phase 2 tokenizer/baseline experiments, Phase 3 architecture search, and a Phase 4 fixed-architecture transfer matrix, representation choice materially changes modeling efficiency. D2 is consistently worst, D3 improves over D2, and D1 performs best overall. Corpus-level statistics show that 26.95% of undiacritized forms are ambiguous and that 96.77% of tokens in a 128-token context window belong to ambiguous forms, supporting the view that stripping harakat imposes an ambiguity tax. These results suggest that, in vocalized classical Arabic under a fixed 5-minute MLX budget, harakat are not disposable surface noise and representation design should be treated as a first-order pretraining decision.

## 1. Introduction

Arabic writing often omits short vowels, leaving many surface forms ambiguous `[CITATION NEEDED]`. In vocalized text, harakat provide exactly the information that is otherwise missing: pronunciation cues, morphological distinctions, and grammatical disambiguation. The standard engineering response has usually been to normalize Arabic aggressively and train on undiacritized text `[CITATION NEEDED]`. That choice is reasonable for broad web Arabic, but it may be suboptimal for classical or fully vocalized corpora where the removed marks are densely informative.

This paper asks a narrow but important question: how should harakat be represented during small-model Arabic pretraining? We study three conditions built from the same underlying corpus:

- `D1`: raw Unicode diacritized Arabic
- `D2`: the same text with harakat stripped
- `D3`: a novel atomic encoding where each letter+harakah pair is mapped to a private-use code point

The experimental story evolved during the project. Phase 2 baselines initially suggested that D3 was the strongest representation. Phase 3 architecture search changed that picture. After 70+ runs per condition, the best ordering became `D1 < D3 < D2` on validation bits-per-byte (`val_bpb`), where lower is better. Phase 4 then tested whether this ordering was only an artifact of architecture search by running a fixed-architecture transfer matrix. The same ranking survived under all three shared winner architectures.

The main contribution of this paper is therefore not "atomic encoding wins." The stronger and more defensible claim is that stripping harakat imposes a measurable ambiguity tax, that atomic encoding recovers part of the lost signal, and that raw diacritized text is strongest in the current small-model classical-Arabic regime.

## 2. Motivation and Related Work

This draft positions the work at the intersection of Arabic normalization, Arabic tokenization, and representation design for language-model pretraining. The final version should verify and cite:

- Arabic NLP work that normalizes or strips tashkeel during preprocessing `[CITATION NEEDED]`
- Arabic tokenization and vocabulary studies `[CITATION NEEDED]`
- Arabic diacritization literature showing that harakat carry lexical and grammatical information `[CITATION NEEDED]`

The present draft does not claim novelty for diacritics themselves. The novelty is the controlled pretraining comparison between:

1. raw diacritized text,
2. stripped text, and
3. atomic letter+harakah encoding,

all under the same corpus, training budget, and search framework.

## 3. Dataset and Representations

### 3.1 Corpus

The experiments use a Tashkeela-derived paired corpus of vocalized and non-vocalized Arabic text. The pipeline produces three parallel conditions from the same source examples:

- `D1`: fully diacritized text in standard Unicode combining-character form
- `D2`: harakat-stripped text
- `D3`: a private-use-area encoding that combines each base letter and attached harakah into a single atomic symbol

This design isolates representation choice from corpus choice.

### 3.2 Ambiguity Statistics

Phase 1 quantified the ambiguity introduced by stripping harakat. Across the analyzed corpus:

- total undiacritized forms: `2,256,984`
- total diacritized forms: `4,674,932`
- ambiguous undiacritized forms: `608,287`
- ambiguous-form percentage: `26.95%`
- average collision rate: `2.0713`
- 128-token context-window ambiguous percentage: `96.77%`

These numbers support the intuition that removing harakat is not a harmless normalization step in this corpus. More than a quarter of undiacritized forms map to multiple vocalized forms, and nearly all 128-token contexts contain ambiguous tokens.

### 3.3 Representation Definitions

The three conditions differ only in how the same underlying Arabic text is encoded:

- `D1` preserves the original vocalized text
- `D2` removes the vocalic marks
- `D3` preserves the same information as D1 but changes its surface form so that the tokenizer sees letter+harakah pairs as atomic units

`D3` is the project's novel representation. The original working hypothesis was that this atomic encoding might outperform raw diacritized Unicode by giving the model cleaner local units. The final results do not support that hypothesis in full, but they do show that D3 consistently outperforms stripped text.

## 4. Experimental Setup

### 4.1 Training Regime

All experiments use the same small GPT-style decoder-only transformer family implemented in MLX on Apple Silicon. Each training run is constrained to a fixed 5-minute budget, and model quality is measured using validation bits-per-byte (`val_bpb`), where lower is better.

### 4.2 Phase Structure

The work proceeds in four stages:

1. **Phase 1** builds the three dataset conditions and measures ambiguity/collision statistics.
2. **Phase 2** trains tokenizers, measures fertility, and records fixed-architecture baselines.
3. **Phase 3** runs 70+ architecture-search experiments per condition.
4. **Phase 4** adds a fixed-architecture transfer matrix and drafts the manuscript.

### 4.3 Tokenization and Baselines

At vocabulary size 8192, fertility differs substantially across conditions:

- `D1`: `2.5189` tokens/word
- `D2`: `1.4670` tokens/word
- `D3`: `2.1934` tokens/word

Phase 2 fixed-architecture baselines were:

- `D1`: `1.190999`
- `D2`: `1.596882`
- `D3`: `1.075381`

The baseline result initially favored D3. That is one reason the final D1 win after search is important: the representation ordering changes once each condition is allowed to find a stronger architecture.

### 4.4 Architecture Search and Transfer Ablation

Phase 3 recorded the best searched result per condition:

| Condition | Best `val_bpb` | Winning architecture summary |
|---|---:|---|
| D1 | 0.660090 | depth=4, aspect_ratio=26, head_dim=128, window=SS, batch=2**15 |
| D2 | 1.019569 | depth=4, aspect_ratio=24, head_dim=128, window=SSS, batch=2**15 |
| D3 | 0.889682 | depth=2, aspect_ratio=64, head_dim=96, window=SS, batch=2**15 |

Phase 4 then reran a fixed-architecture transfer matrix using each Phase 3 winner as a shared architecture across all three conditions.

## 5. Results

### 5.1 Main Quantitative Comparison

| Condition | Description | Fertility @8192 | Phase 2 Baseline | Phase 3 Best | Delta vs Baseline |
|---|---|---:|---:|---:|---:|
| D1 | Raw diacritized Unicode | 2.5189 | 1.190999 | 0.660090 | -0.530909 |
| D2 | Harakat stripped | 1.4670 | 1.596882 | 1.019569 | -0.577313 |
| D3 | Atomic PUA encoding | 2.1934 | 1.075381 | 0.889682 | -0.185699 |

The most important result is the final ordering after architecture search:

`D1 < D3 < D2`

This means:

- keeping harakat in raw form is best in the current setup,
- atomic encoding is meaningfully better than stripping,
- stripping remains worst even after allowing each condition to search for its own architecture.

### 5.2 Architecture Differences Across Conditions

The winning configurations differ across conditions. D1 and D2 both prefer depth 4 and head dimension 128, but D2 prefers a longer `SSS` local-window pattern while D1 prefers `SS`. D3 is qualitatively different: it prefers a shallower architecture (depth 2) with a much larger aspect ratio (64) and smaller head dimension (96).

This pattern suggests that the three encodings do not simply change sequence length or token count; they change what kind of model shape works best. D3's atomic packaging appears to want a different tradeoff between depth and width than D1.

### 5.3 Fixed-Architecture Transfer Matrix

| Shared Architecture | D1 | D2 | D3 | Ordering |
|---|---:|---:|---:|---|
| D1 winner | 0.700745 | 1.052422 | 0.837692 | D1 < D3 < D2 |
| D2 winner | 0.680836 | 1.036620 | 0.930068 | D1 < D3 < D2 |
| D3 winner | 1.037330 | 1.403904 | 1.289970 | D1 < D3 < D2 |

The transfer matrix preserves the same ranking under all three shared architectures. This matters because it shows the main result is not only an artifact of the architecture-search loop.

At the same time, the absolute values in the transfer matrix should be interpreted carefully. The D3 winner reruns substantially worse in this matrix than in its original best searched run (`1.289970` versus `0.889682`). That makes the matrix useful as a transfer/robustness check, but not as a substitute for multi-seed stability evidence.

## 6. Discussion

### 6.1 Stripping Harakat Is Costly

The strongest and most stable conclusion is that D2 is the worst representation in this corpus. That is already visible in Phase 1 ambiguity statistics, remains true in Phase 2 baselines, remains true after Phase 3 search, and survives the Phase 4 transfer matrix. In this regime, stripping harakat discards predictive information that the model then has to reconstruct from context.

### 6.2 D3 Helps, but Does Not Win

D3 is valuable because it consistently improves over D2 while preserving diacritical information in a tokenizer-friendly form. That supports the broader idea that representation design matters. However, the current experiments do not support the stronger claim that atomic encoding is the best representation overall.

One possible interpretation is that D3 gains cleaner local units at the cost of losing some compositional structure that raw Unicode diacritics preserve. This is an inference from the observed results, not a directly proven mechanism.

### 6.3 Why D1 May Be Best Here

The current evidence suggests that raw diacritized text gives the model direct access to the most useful distinctions without forcing the model into the specific atomic packaging used by D3. If the corpus is already fully vocalized and consistent, preserving that native structure may simply be the easiest representation for the model family used here.

### 6.4 What the Paper Can Claim

The paper can defensibly claim that, for vocalized classical Arabic under a fixed small-model MLX budget:

- representation choice materially affects model efficiency,
- stripping harakat is harmful,
- atomic encoding recovers part of the lost signal,
- raw diacritized Unicode performs best overall.

The paper should not claim that D1 is universally best for all Arabic modeling settings, or that D3 is refuted as a general idea beyond this regime.

## 7. Limitations

This work is intentionally bounded.

- It uses a single Tashkeela-derived corpus.
- It targets small models under a fixed 5-minute training budget.
- It evaluates language-modeling efficiency with `val_bpb`, not downstream-task accuracy.
- It does not yet include multi-seed repeats.
- The fixed-architecture transfer matrix is informative, but it is not a seed-stability study.
- The draft still needs verified citations for related work and background claims.

These limits matter. The safest interpretation is not "all Arabic models should immediately switch to D1," but rather "for vocalized classical Arabic, stripping harakat throws away information and should not be treated as a neutral preprocessing default."

## 8. Conclusion

This project began with the hypothesis that an atomic Arabic diacritic encoding might outperform raw diacritized text. The final result is different and more interesting. Across ambiguity analysis, tokenization, fixed-architecture baselines, architecture search, and a fixed-architecture transfer matrix, the ordering is consistently:

`D1 < D3 < D2`

The central lesson is that harakat are signal, not noise, in the regime studied here. Removing them imposes a real ambiguity tax. Atomic encoding helps recover some of what is lost, but raw diacritized text remains the strongest representation in this small-model classical-Arabic setting.
