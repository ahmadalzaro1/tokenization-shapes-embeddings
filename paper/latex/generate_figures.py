"""Generate publication-quality figures for the paper."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "experiments" / "results"
OUTDIR = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# Colorblind-safe palette (Okabe-Ito)
C_D1 = '#0072B2'  # blue
C_D3 = '#D55E00'  # vermillion
C_D3_D1ARCH = '#009E73'  # green


def fig_scaling_curves():
    """Figure: Iso-data scaling curves (BPBL vs base letters)."""
    with open(RESULTS / "iso_data_results.json") as f:
        data = json.load(f)

    scales_str = ["5M", "15M", "30M", "50M", "100M", "200M", "500M"]
    scales_num = [5, 15, 30, 50, 100, 200, 500]

    d1_mean = [data["summary"]["d1"][s]["mean_bpbl"] for s in scales_str]
    d1_std = [data["summary"]["d1"][s]["std_bpbl"] for s in scales_str]
    d3_mean = [data["summary"]["d3"][s]["mean_bpbl"] for s in scales_str]
    d3_std = [data["summary"]["d3"][s]["std_bpbl"] for s in scales_str]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.errorbar(scales_num, d1_mean, yerr=d1_std, fmt='o-', color=C_D1,
                markersize=4, linewidth=1.5, capsize=3, label='D1 (compositional)')
    ax.errorbar(scales_num, d3_mean, yerr=d3_std, fmt='s-', color=C_D3,
                markersize=4, linewidth=1.5, capsize=3, label='D3 (atomic)')

    ax.set_xscale('log')
    ax.set_xlabel('Base Letters (millions)')
    ax.set_ylabel('BPBL ↓')
    ax.set_xticks(scales_num)
    ax.set_xticklabels(['5', '15', '30', '50', '100', '200', '500'])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Iso-Data Scaling: D1 vs D3')

    # Annotate gap at endpoints
    gap_5m = (d3_mean[0] - d1_mean[0]) / d1_mean[0] * 100
    gap_500m = (d3_mean[-1] - d1_mean[-1]) / d1_mean[-1] * 100
    ax.annotate(f'{gap_5m:.0f}% gap', xy=(5, (d1_mean[0]+d3_mean[0])/2),
                fontsize=7, ha='left', color='gray')
    ax.annotate(f'{gap_500m:.1f}%', xy=(500, (d1_mean[-1]+d3_mean[-1])/2),
                fontsize=7, ha='right', color='gray')

    fig.savefig(OUTDIR / "scaling_curves.pdf")
    plt.close()
    print(f"Saved {OUTDIR / 'scaling_curves.pdf'}")


def fig_scaling_with_arch_control():
    """Figure: Scaling curves with architecture confound control."""
    with open(RESULTS / "iso_data_results_arch_comparison.json") as f:
        data = json.load(f)

    scales_str = ["5M", "15M", "30M", "50M", "100M", "200M", "500M"]
    scales_num = [5, 15, 30, 50, 100, 200, 500]

    d1_mean = [data["summary"]["d1_optimal"][s]["mean_bpbl"] for s in scales_str]
    d3_mean = [data["summary"]["d3_optimal"][s]["mean_bpbl"] for s in scales_str]
    d3d1_mean = [data["summary"]["d3_d1arch"][s]["mean_bpbl"] for s in scales_str]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot(scales_num, d1_mean, 'o-', color=C_D1, markersize=4, linewidth=1.5,
            label='D1 (own arch)')
    ax.plot(scales_num, d3_mean, 's-', color=C_D3, markersize=4, linewidth=1.5,
            label='D3 (own arch)')
    ax.plot(scales_num, d3d1_mean, '^--', color=C_D3_D1ARCH, markersize=4,
            linewidth=1.5, label='D3 + D1 arch')

    ax.set_xscale('log')
    ax.set_xlabel('Base Letters (millions)')
    ax.set_ylabel('BPBL ↓')
    ax.set_xticks(scales_num)
    ax.set_xticklabels(['5', '15', '30', '50', '100', '200', '500'])
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title('Architecture Confound Control')

    fig.savefig(OUTDIR / "arch_control.pdf")
    plt.close()
    print(f"Saved {OUTDIR / 'arch_control.pdf'}")


def fig_embedding_heatmap():
    """Figure: D3 per-letter cosine similarity bar chart."""
    with open(RESULTS / "phase-05" / "embedding_similarity.json") as f:
        data = json.load(f)

    d3 = data["d3"]["per_letter"]
    # Filter letters with valid cosine similarity
    letters = []
    cosines = []
    for name, info in d3.items():
        if info["mean_cosine_sim"] is not None:
            letters.append(name)
            cosines.append(info["mean_cosine_sim"])

    # Sort by cosine similarity
    sorted_pairs = sorted(zip(cosines, letters))
    cosines = [c for c, _ in sorted_pairs]
    letters = [l for _, l in sorted_pairs]

    fig, ax = plt.subplots(figsize=(3.3, 4.0))
    colors = [C_D3 if c < 0.15 else '#E69F00' for c in cosines]
    ax.barh(range(len(letters)), cosines, color=colors, edgecolor='none', height=0.7)
    ax.set_yticks(range(len(letters)))
    ax.set_yticklabels(letters, fontsize=6)
    ax.set_xlabel('Mean Intra-Group Cosine Similarity')
    ax.axvline(x=0.125, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.annotate('overall mean\n(0.125)', xy=(0.125, len(letters)-1),
                fontsize=6, ha='left', va='top', color='gray')
    ax.set_title('D3: Per-Letter Embedding Similarity')
    ax.set_xlim(0, 0.35)

    fig.savefig(OUTDIR / "d3_embedding_similarity.pdf")
    plt.close()
    print(f"Saved {OUTDIR / 'd3_embedding_similarity.pdf'}")


def fig_gap_curve():
    """Figure: Gap percentage vs data scale."""
    with open(RESULTS / "iso_data_results.json") as f:
        data = json.load(f)

    scales_str = ["5M", "15M", "30M", "50M", "100M", "200M", "500M"]
    scales_num = [5, 15, 30, 50, 100, 200, 500]

    d1_mean = [data["summary"]["d1"][s]["mean_bpbl"] for s in scales_str]
    d3_mean = [data["summary"]["d3"][s]["mean_bpbl"] for s in scales_str]
    gaps = [(d3 - d1) / d1 * 100 for d1, d3 in zip(d1_mean, d3_mean)]

    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.plot(scales_num, gaps, 'o-', color='#CC79A7', markersize=5, linewidth=2)
    ax.fill_between(scales_num, gaps, alpha=0.15, color='#CC79A7')
    ax.set_xscale('log')
    ax.set_xlabel('Base Letters (millions)')
    ax.set_ylabel('BPBL Gap (%)')
    ax.set_xticks(scales_num)
    ax.set_xticklabels(['5', '15', '30', '50', '100', '200', '500'])
    ax.grid(True, alpha=0.3)
    ax.set_title('D3 Disadvantage Shrinks with Data')

    for i, (x, g) in enumerate(zip(scales_num, gaps)):
        if i in (0, 3, 6):
            ax.annotate(f'{g:.1f}%', xy=(x, g), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=7)

    fig.savefig(OUTDIR / "gap_curve.pdf")
    plt.close()
    print(f"Saved {OUTDIR / 'gap_curve.pdf'}")


if __name__ == "__main__":
    fig_scaling_curves()
    fig_scaling_with_arch_control()
    fig_embedding_heatmap()
    fig_gap_curve()
    print("All figures generated.")
