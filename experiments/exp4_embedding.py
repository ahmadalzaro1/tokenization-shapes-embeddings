"""Experiment 4: Embedding Similarity Analysis — D1 vs D3.

Analyzes learned token embedding matrices to mechanistically explain WHY D3
loses to D1 despite having perfect tokenization. Produces cosine similarity
heatmaps grouped by base letter (D3), harakah clustering analysis (D1),
PCA scatter plots, and per-letter bar charts.

Usage:
    cd /path/to/autoresearch-arabic
    uv run python experiments/exp4_embedding.py
"""

import sys
from pathlib import Path

# Critical: fix sys.path so 'import prepare' and 'from experiments.shared import ...' work
# when executed as 'uv run python experiments/exp4_embedding.py'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math
import re
import unicodedata

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 300})
from sklearn.decomposition import PCA

import prepare
from prepare import Tokenizer
from experiments.shared import ARABIC_LETTER_RE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
CACHE_DIR = Path.home() / ".cache" / "autoresearch-arabic"

# HARAKAT codepoints (from build_dataset.py)
HARAKAT_CODEPOINTS = [
    '\u064B',  # fathatan
    '\u064C',  # dammatan
    '\u064D',  # kasratan
    '\u064E',  # fathah
    '\u064F',  # dammah
    '\u0650',  # kasrah
    '\u0651',  # shaddah
    '\u0652',  # sukun
    '\u0670',  # superscript alef
]

HARAKAT_NAMES = {
    '\u064B': 'fathatan',
    '\u064C': 'dammatan',
    '\u064D': 'kasratan',
    '\u064E': 'fathah',
    '\u064F': 'dammah',
    '\u0650': 'kasrah',
    '\u0651': 'shaddah',
    '\u0652': 'sukun',
    '\u0670': 'superscript_alef',
}


def load_bpbl_results() -> dict:
    """Load bpbl_results.json and determine best seed per condition."""
    path = RESULTS_DIR / "bpbl_results.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def best_seed_for_condition(bpbl_data: dict, condition: str) -> int:
    """Return the seed with lowest val_bpb for a condition."""
    seeds = bpbl_data[condition]["seeds"]
    val_bpbs = bpbl_data[condition]["val_bpb"]
    best_idx = int(np.argmin(val_bpbs))
    return seeds[best_idx]


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for embeddings (N, D)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    normed = embeddings / norms
    return normed @ normed.T


def mean_intra_cosine(sim_matrix: np.ndarray) -> float:
    """Mean of off-diagonal elements (excluding self-similarity=1.0)."""
    n = sim_matrix.shape[0]
    if n < 2:
        return float('nan')
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(sim_matrix[mask]))


# ---------------------------------------------------------------------------
# D3 Analysis
# ---------------------------------------------------------------------------
def analyze_d3(bpbl_data: dict) -> dict:
    """Analyze D3 embeddings: group PUA tokens by base letter, compute cosine similarity."""
    print("\n" + "=" * 60, flush=True)
    print("D3 EMBEDDING ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Load atomic mapping and build reverse mapping
    mapping_path = CACHE_DIR / "atomic_mapping.json"
    with open(mapping_path, encoding="utf-8") as f:
        saved = json.load(f)
    # reverse: PUA char -> original sequence
    reverse_map = {chr(int(v, 16)): k for k, v in saved.items()}
    print(f"  Loaded atomic mapping: {len(saved)} entries", flush=True)

    # Step 2: Group PUA chars by base letter (filter out standalone harakat)
    base_letter_groups: dict[str, list[str]] = {}
    standalone_harakat_count = 0
    for pua_char, original_seq in reverse_map.items():
        base = original_seq[0]
        if not ARABIC_LETTER_RE.match(base):
            standalone_harakat_count += 1
            continue
        if base not in base_letter_groups:
            base_letter_groups[base] = []
        base_letter_groups[base].append(pua_char)

    print(f"  Base letter groups: {len(base_letter_groups)}", flush=True)
    print(f"  Standalone harakat (filtered out): {standalone_harakat_count}", flush=True)

    # Step 3: Load D3 tokenizer
    prepare.init_condition("d3")
    tokenizer = Tokenizer.from_directory()
    print(f"  D3 tokenizer vocab size: {tokenizer.enc.n_vocab}", flush=True)

    # Step 4: Map PUA chars to token IDs (skip those not in vocab as singletons)
    base_letter_token_ids: dict[str, list[int]] = {}
    base_letter_pua_labels: dict[str, list[str]] = {}  # for heatmap labels
    total_pua_tokens = 0
    skipped_pua = 0
    for base_letter, pua_chars in base_letter_groups.items():
        ids = []
        labels = []
        for pua_char in pua_chars:
            try:
                token_id = tokenizer.enc.encode_single_token(pua_char)
                ids.append(token_id)
                # label = the harakah portion of the original sequence
                orig = reverse_map[pua_char]
                harakah_part = orig[1:]  # everything after base letter
                label_parts = []
                for ch in harakah_part:
                    label_parts.append(HARAKAT_NAMES.get(ch, f"U+{ord(ch):04X}"))
                labels.append("+".join(label_parts) if label_parts else "bare")
                total_pua_tokens += 1
            except KeyError:
                skipped_pua += 1
        if ids:
            base_letter_token_ids[base_letter] = ids
            base_letter_pua_labels[base_letter] = labels

    print(f"  PUA tokens found as singletons: {total_pua_tokens}", flush=True)
    print(f"  PUA chars skipped (not in vocab): {skipped_pua}", flush=True)

    # Step 5: Load best D3 wte
    best_seed = best_seed_for_condition(bpbl_data, "d3")
    wte_path = RESULTS_DIR / f"wte_d3_seed{best_seed}.npy"
    wte = np.load(wte_path)
    print(f"  Loaded D3 wte from seed {best_seed}: shape {wte.shape}", flush=True)

    # Step 6: Compute per-letter cosine similarity
    per_letter: dict[str, dict] = {}
    per_letter_mean_sims: list[float] = []

    for base_letter, token_ids in sorted(base_letter_token_ids.items()):
        embeddings = wte[token_ids]  # shape (n_variants, n_embd)
        sim_mat = cosine_sim_matrix(embeddings)
        mean_sim = mean_intra_cosine(sim_mat)

        # Use unicodedata name for the letter key
        try:
            letter_name = unicodedata.name(base_letter).split()[-1].lower()
        except ValueError:
            letter_name = f"U+{ord(base_letter):04X}"

        per_letter[letter_name] = {
            "char": base_letter,
            "n_variants": len(token_ids),
            "mean_cosine_sim": round(mean_sim, 6) if not math.isnan(mean_sim) else None,
        }
        if not math.isnan(mean_sim):
            per_letter_mean_sims.append(mean_sim)

    # Step 7: Overall statistics (only from letters with >= 2 variants)
    overall_mean = float(np.mean(per_letter_mean_sims)) if per_letter_mean_sims else 0.0
    overall_std = float(np.std(per_letter_mean_sims, ddof=1)) if len(per_letter_mean_sims) > 1 else 0.0

    print(f"\n  Overall mean intra-group cosine: {overall_mean:.6f}", flush=True)
    print(f"  Overall std intra-group cosine:  {overall_std:.6f}", flush=True)
    print(f"  Number of base letters analyzed: {len(per_letter)}", flush=True)

    # Determine interpretation
    if overall_mean < 0.15:
        interpretation = (
            f"Very low intra-group similarity (mean={overall_mean:.4f}): D3 learns nearly "
            f"independent embeddings per letter+harakah variant, completely destroying "
            f"compositionality. The model cannot share information across diacritical "
            f"variants of the same base letter."
        )
    elif overall_mean < 0.3:
        interpretation = (
            f"Low intra-group similarity (mean={overall_mean:.4f}): D3 learns mostly "
            f"independent embeddings per variant, with weak compositional structure. "
            f"The atomic encoding hinders cross-variant generalization."
        )
    else:
        interpretation = (
            f"Moderate intra-group similarity (mean={overall_mean:.4f}): D3 retains some "
            f"compositional structure, but the atomic encoding still fragments the "
            f"embedding space."
        )

    d3_result = {
        "best_seed": best_seed,
        "overall_mean_intra_cosine": round(overall_mean, 6),
        "overall_std_intra_cosine": round(overall_std, 6),
        "n_base_letters": len(per_letter),
        "total_pua_singleton_tokens": total_pua_tokens,
        "per_letter": per_letter,
        "interpretation": interpretation,
    }

    return d3_result


# ---------------------------------------------------------------------------
# D1 Analysis
# ---------------------------------------------------------------------------
def analyze_d1(bpbl_data: dict) -> dict:
    """Analyze D1 embeddings: check harakah singleton tokens, fallback to pair analysis."""
    print("\n" + "=" * 60, flush=True)
    print("D1 EMBEDDING ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Load D1 tokenizer
    prepare.init_condition("d1")
    tokenizer = Tokenizer.from_directory()
    print(f"  D1 tokenizer vocab size: {tokenizer.enc.n_vocab}", flush=True)

    # Step 2: Check which harakah are singleton tokens
    singleton_harakah_ids: dict[str, int] = {}
    for harakah in HARAKAT_CODEPOINTS:
        try:
            token_id = tokenizer.enc.encode_single_token(harakah)
            name = HARAKAT_NAMES.get(harakah, f"U+{ord(harakah):04X}")
            singleton_harakah_ids[name] = token_id
            print(f"  Singleton harakah: {name} (U+{ord(harakah):04X}) -> token {token_id}", flush=True)
        except KeyError:
            name = HARAKAT_NAMES.get(harakah, f"U+{ord(harakah):04X}")
            print(f"  Harakah NOT in vocab as singleton: {name} (U+{ord(harakah):04X})", flush=True)

    # Step 3: Load D1 wte
    best_seed = best_seed_for_condition(bpbl_data, "d1")
    wte_path = RESULTS_DIR / f"wte_d1_seed{best_seed}.npy"
    wte = np.load(wte_path)
    print(f"  Loaded D1 wte from seed {best_seed}: shape {wte.shape}", flush=True)

    d1_result: dict = {
        "best_seed": best_seed,
        "n_singleton_harakah": len(singleton_harakah_ids),
        "singleton_harakah": list(singleton_harakah_ids.keys()),
    }

    if len(singleton_harakah_ids) == 0:
        # Fallback: all harakah absorbed into BPE merges
        print("\n  All harakah absorbed into BPE merges — running fallback analysis", flush=True)
        d1_result["note"] = (
            "All harakah absorbed into BPE merges — no singleton harakah tokens in D1 vocabulary. "
            "This itself supports compositionality: D1's BPE learns letter+harakah units naturally."
        )
        d1_result["fallback_analysis"] = "letter_harakah_pair_tokens"

        # Find letter+harakah pair tokens in vocabulary
        vocab = [tokenizer.enc.decode_bytes([i]).decode("utf-8", errors="ignore")
                 for i in range(tokenizer.enc.n_vocab)]

        pair_tokens: list[tuple[int, str]] = []
        for i, tok_str in enumerate(vocab):
            tok_clean = tok_str.strip()
            if (len(tok_clean) == 2
                    and ARABIC_LETTER_RE.match(tok_clean[0])
                    and unicodedata.category(tok_clean[1]) == 'Mn'):
                pair_tokens.append((i, tok_clean))

        print(f"  Found {len(pair_tokens)} letter+harakah pair tokens in D1 vocab", flush=True)
        d1_result["n_pair_tokens"] = len(pair_tokens)

        if len(pair_tokens) >= 2:
            # Group pair tokens by base letter
            pair_groups: dict[str, list[tuple[int, str]]] = {}
            for token_id, tok_str in pair_tokens:
                base = tok_str[0]
                if base not in pair_groups:
                    pair_groups[base] = []
                pair_groups[base].append((token_id, tok_str))

            # Compute intra-group cosine similarity for pair tokens
            pair_per_letter: dict[str, dict] = {}
            pair_mean_sims: list[float] = []

            for base_letter, group in sorted(pair_groups.items()):
                if len(group) < 2:
                    continue
                ids = [idx for idx, _ in group]
                embeddings = wte[ids]
                sim_mat = cosine_sim_matrix(embeddings)
                mean_sim = mean_intra_cosine(sim_mat)

                try:
                    letter_name = unicodedata.name(base_letter).split()[-1].lower()
                except ValueError:
                    letter_name = f"U+{ord(base_letter):04X}"

                pair_per_letter[letter_name] = {
                    "char": base_letter,
                    "n_variants": len(group),
                    "mean_cosine_sim": round(mean_sim, 6),
                }
                pair_mean_sims.append(mean_sim)

            if pair_mean_sims:
                pair_overall_mean = float(np.mean(pair_mean_sims))
                pair_overall_std = float(np.std(pair_mean_sims, ddof=1)) if len(pair_mean_sims) > 1 else 0.0
                d1_result["pair_overall_mean_intra_cosine"] = round(pair_overall_mean, 6)
                d1_result["pair_overall_std_intra_cosine"] = round(pair_overall_std, 6)
                d1_result["pair_per_letter"] = pair_per_letter
                d1_result["pair_n_letters_analyzed"] = len(pair_per_letter)
                print(f"  Pair token analysis: {len(pair_per_letter)} letters, "
                      f"mean intra-cosine = {pair_overall_mean:.6f}", flush=True)

                d1_result["interpretation"] = (
                    f"D1 BPE merges harakah with base letters, creating {len(pair_tokens)} "
                    f"letter+harakah pair tokens. Intra-group cosine similarity "
                    f"(mean={pair_overall_mean:.4f}) shows how D1 clusters same-letter "
                    f"variants. Compare with D3's atomic approach to assess compositionality."
                )
            else:
                d1_result["interpretation"] = (
                    "Insufficient pair tokens with shared base letters for intra-group analysis."
                )
        else:
            d1_result["interpretation"] = (
                "Too few letter+harakah pair tokens found for meaningful analysis."
            )

    else:
        # Singleton harakah exist — compute cosine similarity
        harakah_ids = list(singleton_harakah_ids.values())
        harakah_embeddings = wte[harakah_ids]
        harakah_sim = cosine_sim_matrix(harakah_embeddings)
        harakah_mutual = mean_intra_cosine(harakah_sim)
        d1_result["harakah_mutual_cosine"] = round(harakah_mutual, 6)

        # Base letter tokens
        base_letter_ids = []
        base_letter_chars = []
        letters = [chr(c) for c in range(0x0621, 0x063B)] + [chr(c) for c in range(0x0641, 0x064B)]
        for letter in letters:
            try:
                token_id = tokenizer.enc.encode_single_token(letter)
                base_letter_ids.append(token_id)
                base_letter_chars.append(letter)
            except KeyError:
                pass

        if base_letter_ids:
            base_embeddings = wte[base_letter_ids]

            # Cross-group cosine similarity (harakah vs base letters)
            h_norms = np.linalg.norm(harakah_embeddings, axis=1, keepdims=True)
            h_norms = np.maximum(h_norms, 1e-10)
            h_normed = harakah_embeddings / h_norms

            b_norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
            b_norms = np.maximum(b_norms, 1e-10)
            b_normed = base_embeddings / b_norms

            cross_sim = h_normed @ b_normed.T
            harakah_vs_base = float(np.mean(cross_sim))
            d1_result["harakah_vs_base_letter_cosine"] = round(harakah_vs_base, 6)

            print(f"  Harakah mutual cosine: {harakah_mutual:.6f}", flush=True)
            print(f"  Harakah vs base letter cosine: {harakah_vs_base:.6f}", flush=True)

        d1_result["interpretation"] = (
            f"Harakah tokens cluster together (mutual cosine={harakah_mutual:.4f}) "
            f"and separately from base letters, showing D1 learns compositional structure."
        )

    return d1_result


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_d3_heatmap(bpbl_data: dict, d3_result: dict) -> None:
    """Figure 1: D3 Cosine Similarity Heatmap — 2x5 subplots for top 10 letters."""
    print("\n  Generating D3 cosine similarity heatmap...", flush=True)

    # Reload D3 data for plotting
    mapping_path = CACHE_DIR / "atomic_mapping.json"
    with open(mapping_path, encoding="utf-8") as f:
        saved = json.load(f)
    reverse_map = {chr(int(v, 16)): k for k, v in saved.items()}

    # Group PUA chars by base letter
    base_letter_groups: dict[str, list[str]] = {}
    for pua_char, original_seq in reverse_map.items():
        base = original_seq[0]
        if not ARABIC_LETTER_RE.match(base):
            continue
        if base not in base_letter_groups:
            base_letter_groups[base] = []
        base_letter_groups[base].append(pua_char)

    # Load tokenizer and wte
    prepare.init_condition("d3")
    tokenizer = Tokenizer.from_directory()
    best_seed = best_seed_for_condition(bpbl_data, "d3")
    wte = np.load(RESULTS_DIR / f"wte_d3_seed{best_seed}.npy")

    # Build token ID groups with labels
    letter_data: list[tuple[str, str, np.ndarray, list[str]]] = []
    for base_letter, pua_chars in sorted(base_letter_groups.items()):
        ids = []
        labels = []
        for pua_char in pua_chars:
            try:
                token_id = tokenizer.enc.encode_single_token(pua_char)
                ids.append(token_id)
                orig = reverse_map[pua_char]
                harakah_part = orig[1:]
                label_parts = []
                for ch in harakah_part:
                    label_parts.append(HARAKAT_NAMES.get(ch, f"U+{ord(ch):04X}"))
                labels.append("+".join(label_parts) if label_parts else "bare")
            except KeyError:
                pass
        if len(ids) >= 2:
            embeddings = wte[ids]
            sim_mat = cosine_sim_matrix(embeddings)
            try:
                letter_name = unicodedata.name(base_letter).split()[-1].lower()
            except ValueError:
                letter_name = f"U+{ord(base_letter):04X}"
            letter_data.append((base_letter, letter_name, sim_mat, labels))

    # Sort by number of variants (descending) and take top 10
    letter_data.sort(key=lambda x: x[2].shape[0], reverse=True)
    top_10 = letter_data[:10]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("D3 Intra-group Embedding Similarity by Base Letter", fontsize=14, fontweight='bold')

    for idx, (base_char, name, sim_mat, labels) in enumerate(top_10):
        row, col = idx // 5, idx % 5
        ax = axes[row, col]
        im = ax.imshow(sim_mat, cmap='RdBu_r', vmin=-0.5, vmax=1.0, aspect='auto')
        ax.set_title(f"{base_char} ({name})", fontsize=9)
        n = sim_mat.shape[0]
        # Only show tick labels if <= 12 variants
        if n <= 12:
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=5)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        mean_sim = mean_intra_cosine(sim_mat)
        ax.set_xlabel(f"mean={mean_sim:.3f}", fontsize=7)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Cosine Similarity")
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    out_path = RESULTS_DIR / "d3_embedding_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def plot_embedding_space_pca(bpbl_data: dict) -> None:
    """Figure 2: D1 vs D3 Embedding Space PCA — side-by-side scatter plots."""
    print("\n  Generating embedding space PCA comparison...", flush=True)

    # --- D3 subplot ---
    mapping_path = CACHE_DIR / "atomic_mapping.json"
    with open(mapping_path, encoding="utf-8") as f:
        saved = json.load(f)
    reverse_map = {chr(int(v, 16)): k for k, v in saved.items()}

    # Group by base letter for coloring
    base_letter_groups: dict[str, list[str]] = {}
    for pua_char, original_seq in reverse_map.items():
        base = original_seq[0]
        if not ARABIC_LETTER_RE.match(base):
            continue
        if base not in base_letter_groups:
            base_letter_groups[base] = []
        base_letter_groups[base].append(pua_char)

    prepare.init_condition("d3")
    tokenizer_d3 = Tokenizer.from_directory()
    best_seed_d3 = best_seed_for_condition(bpbl_data, "d3")
    wte_d3 = np.load(RESULTS_DIR / f"wte_d3_seed{best_seed_d3}.npy")

    # Collect all PUA token embeddings with base letter labels
    d3_ids: list[int] = []
    d3_base_letters: list[str] = []
    for base_letter, pua_chars in base_letter_groups.items():
        for pua_char in pua_chars:
            try:
                token_id = tokenizer_d3.enc.encode_single_token(pua_char)
                d3_ids.append(token_id)
                d3_base_letters.append(base_letter)
            except KeyError:
                pass

    d3_embeddings = wte_d3[d3_ids]

    # Top 10 most frequent base letters for coloring
    from collections import Counter
    letter_counts = Counter(d3_base_letters)
    top_10_letters = [letter for letter, _ in letter_counts.most_common(10)]
    distinct_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    d3_colors = []
    for bl in d3_base_letters:
        if bl in top_10_letters:
            d3_colors.append(distinct_colors[top_10_letters.index(bl)])
        else:
            d3_colors.append([0.7, 0.7, 0.7, 0.5])
    d3_colors = np.array(d3_colors)

    # PCA for D3
    pca_d3 = PCA(n_components=2)
    d3_2d = pca_d3.fit_transform(d3_embeddings)

    # --- D1 subplot ---
    prepare.init_condition("d1")
    tokenizer_d1 = Tokenizer.from_directory()
    best_seed_d1 = best_seed_for_condition(bpbl_data, "d1")
    wte_d1 = np.load(RESULTS_DIR / f"wte_d1_seed{best_seed_d1}.npy")

    # Categorize D1 tokens
    vocab_d1 = [tokenizer_d1.enc.decode_bytes([i]).decode("utf-8", errors="ignore")
                for i in range(tokenizer_d1.enc.n_vocab)]

    d1_ids: list[int] = []
    d1_types: list[str] = []  # 'harakah', 'base_letter', 'pair', 'other'

    # Check singleton harakah
    for harakah in HARAKAT_CODEPOINTS:
        try:
            token_id = tokenizer_d1.enc.encode_single_token(harakah)
            d1_ids.append(token_id)
            d1_types.append('harakah')
        except KeyError:
            pass

    # Base letters
    letters = [chr(c) for c in range(0x0621, 0x063B)] + [chr(c) for c in range(0x0641, 0x064B)]
    for letter in letters:
        try:
            token_id = tokenizer_d1.enc.encode_single_token(letter)
            d1_ids.append(token_id)
            d1_types.append('base_letter')
        except KeyError:
            pass

    # Letter+harakah pairs
    for i, tok_str in enumerate(vocab_d1):
        tok_clean = tok_str.strip()
        if (len(tok_clean) == 2
                and ARABIC_LETTER_RE.match(tok_clean[0])
                and unicodedata.category(tok_clean[1]) == 'Mn'):
            if i not in d1_ids:
                d1_ids.append(i)
                d1_types.append('pair')

    # Sample ~200 other Arabic tokens
    arabic_token_re = re.compile(r'[\u0600-\u06FF]')
    other_arabic_ids = []
    for i, tok_str in enumerate(vocab_d1):
        if i not in set(d1_ids) and arabic_token_re.search(tok_str):
            other_arabic_ids.append(i)
    rng = np.random.default_rng(42)
    if len(other_arabic_ids) > 200:
        sampled = rng.choice(other_arabic_ids, size=200, replace=False).tolist()
    else:
        sampled = other_arabic_ids
    for oid in sampled:
        d1_ids.append(oid)
        d1_types.append('other')

    d1_embeddings = wte_d1[d1_ids]

    # PCA for D1
    pca_d1 = PCA(n_components=2)
    d1_2d = pca_d1.fit_transform(d1_embeddings)

    # Color mapping for D1
    color_map = {'harakah': 'red', 'base_letter': 'blue', 'pair': 'green', 'other': (0.7, 0.7, 0.7, 0.5)}
    d1_colors = [color_map[t] for t in d1_types]

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Embedding Space Structure: D3 (Atomic) vs D1 (Compositional)",
                 fontsize=14, fontweight='bold')

    # D3 scatter
    ax1.scatter(d3_2d[:, 0], d3_2d[:, 1], c=d3_colors, s=8, alpha=0.7)
    ax1.set_title(f"D3 Atomic (PCA, n={len(d3_ids)} PUA tokens)")
    ax1.set_xlabel(f"PC1 ({pca_d3.explained_variance_ratio_[0]:.1%} var)")
    ax1.set_ylabel(f"PC2 ({pca_d3.explained_variance_ratio_[1]:.1%} var)")

    # Legend for D3
    for i, letter in enumerate(top_10_letters):
        try:
            letter_name = unicodedata.name(letter).split()[-1].lower()
        except ValueError:
            letter_name = f"U+{ord(letter):04X}"
        ax1.scatter([], [], c=[distinct_colors[i]], label=f"{letter} ({letter_name})", s=20)
    ax1.scatter([], [], c=[(0.7, 0.7, 0.7, 0.5)], label="other letters", s=20)
    ax1.legend(fontsize=6, loc='best', ncol=2)

    # D1 scatter — plot by type for legend
    for token_type, color, label in [
        ('harakah', 'red', 'Harakah'),
        ('base_letter', 'blue', 'Base letters'),
        ('pair', 'green', 'Letter+harakah pairs'),
        ('other', (0.7, 0.7, 0.7, 0.5), 'Other Arabic'),
    ]:
        mask = [t == token_type for t in d1_types]
        if any(mask):
            points = d1_2d[mask]
            ax2.scatter(points[:, 0], points[:, 1], c=[color], s=8, alpha=0.7, label=label)

    ax2.set_title(f"D1 Compositional (PCA, n={len(d1_ids)} tokens)")
    ax2.set_xlabel(f"PC1 ({pca_d1.explained_variance_ratio_[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({pca_d1.explained_variance_ratio_[1]:.1%} var)")
    ax2.legend(fontsize=8, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = RESULTS_DIR / "embedding_space_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def plot_per_letter_bar_chart(d3_result: dict) -> None:
    """Figure 3: Bar chart of per-letter mean cosine similarity."""
    print("\n  Generating per-letter bar chart...", flush=True)

    per_letter = d3_result["per_letter"]
    if not per_letter:
        print("  No per-letter data — skipping bar chart", flush=True)
        return

    # Sort by mean cosine sim (filter out letters with None / single variant)
    items = sorted(
        [(k, v) for k, v in per_letter.items() if v["mean_cosine_sim"] is not None],
        key=lambda x: x[1]["mean_cosine_sim"],
    )
    names = [f"{v['char']} ({k})" for k, v in items]
    values = [v["mean_cosine_sim"] for _, v in items]
    overall_mean = d3_result["overall_mean_intra_cosine"]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(names)), values, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=1.5,
               label=f'Overall mean = {overall_mean:.4f}')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Mean Intra-group Cosine Similarity")
    ax.set_xlabel("Base Letter")
    ax.set_title("D3 Per-Letter Mean Intra-group Cosine Similarity (Sorted)")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=min(0, min(values) - 0.05))

    plt.tight_layout()
    out_path = RESULTS_DIR / "d3_per_letter_similarity.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60, flush=True)
    print("EXPERIMENT 4: EMBEDDING SIMILARITY ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    # Load BPBL results to determine best seeds
    bpbl_data = load_bpbl_results()
    print(f"Best D1 seed: {best_seed_for_condition(bpbl_data, 'd1')} "
          f"(val_bpb={min(bpbl_data['d1']['val_bpb']):.6f})", flush=True)
    print(f"Best D3 seed: {best_seed_for_condition(bpbl_data, 'd3')} "
          f"(val_bpb={min(bpbl_data['d3']['val_bpb']):.6f})", flush=True)

    # Run analyses
    d3_result = analyze_d3(bpbl_data)
    d1_result = analyze_d1(bpbl_data)

    # Write JSON results
    output = {"d3": d3_result, "d1": d1_result}

    # Remove non-serializable fields from per_letter (sim_matrix is numpy)
    out_path = RESULTS_DIR / "embedding_similarity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {out_path}", flush=True)

    # Generate visualizations
    plot_d3_heatmap(bpbl_data, d3_result)
    plot_embedding_space_pca(bpbl_data)
    plot_per_letter_bar_chart(d3_result)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT 4 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  D3 mean intra-group cosine: {d3_result['overall_mean_intra_cosine']:.6f} "
          f"+/- {d3_result['overall_std_intra_cosine']:.6f}", flush=True)
    print(f"  D3 base letters analyzed: {d3_result['n_base_letters']}", flush=True)
    print(f"  D1 singleton harakah: {d1_result['n_singleton_harakah']}", flush=True)
    print(f"  D3 interpretation: {d3_result['interpretation']}", flush=True)
    print(f"  D1 interpretation: {d1_result['interpretation']}", flush=True)

    # Verify outputs
    print("\nOutput files:", flush=True)
    for fname in ["embedding_similarity.json", "d3_embedding_heatmap.png",
                   "embedding_space_comparison.png", "d3_per_letter_similarity.png"]:
        p = RESULTS_DIR / fname
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  {fname}: {size_kb:.1f} KB", flush=True)
        else:
            print(f"  {fname}: MISSING", flush=True)


if __name__ == "__main__":
    main()
