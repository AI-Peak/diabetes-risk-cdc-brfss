#!/usr/bin/env python
"""
Effect-size evidence by feature family Figure Generator
-------------------------------------------------------
Reads statistical results directly from CSV files and generates a two-panel 
publication-quality figure comparing effect sizes across feature families.

- Panel A: Cramér's V (Categorical/Ordinal predictors)
- Panel B: Absolute Rank-Biserial Correlation (Numerical predictors)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_STAT_DIR = BASE_DIR / "results" / "statistical_analysis"
DOCS_FIG_DIR = BASE_DIR / "docs" / "figures"

def generate_effect_size_figure():
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_STAT_DIR.mkdir(parents=True, exist_ok=True)

    chi2_path = RESULTS_STAT_DIR / "chi_square_results.csv"
    num_path = RESULTS_STAT_DIR / "numerical_results.csv"

    if not chi2_path.exists() or not num_path.exists():
        raise FileNotFoundError("Statistical CSV files not found. Run statistical_analysis.py first.")

    cat_df = pd.read_csv(chi2_path)
    cramers_col = [c for c in cat_df.columns if "Cram" in c][0]
    cat_df = cat_df.sort_values(by=cramers_col, ascending=True)

    num_df = pd.read_csv(num_path)
    rb_col = [c for c in num_df.columns if "Biserial" in c or "Effect_Size" in c or "Rank" in c][0]
    num_df[rb_col] = num_df[rb_col].abs()
    num_df = num_df.sort_values(by=rb_col, ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1.2, 1]})

    # Panel A: Cramér's V
    y_pos_cat = np.arange(len(cat_df))
    ax1.axvspan(0.00, 0.05, facecolor="#F8FAFC", alpha=0.9, zorder=1)
    ax1.axvspan(0.05, 0.10, facecolor="#EFF6FF", alpha=0.9, zorder=1)
    ax1.axvspan(0.10, 0.20, facecolor="#ECFDF5", alpha=0.9, zorder=1)
    ax1.axvspan(0.20, 0.35, facecolor="#FEF3C7", alpha=0.9, zorder=1)

    for border in [0.05, 0.10, 0.20]:
        ax1.axvline(border, color="#CBD5E1", linestyle="--", linewidth=0.8, zorder=2)

    ax1.hlines(y_pos_cat, xmin=0, xmax=cat_df[cramers_col], color="#64748B", linewidth=1.2, zorder=3)
    ax1.scatter(cat_df[cramers_col], y_pos_cat, color="#2563EB", s=60, zorder=4, edgecolor="#0F172A", linewidth=0.8)

    for y, (_, row) in zip(y_pos_cat, cat_df.iterrows()):
        val = row[cramers_col]
        ax1.text(val + 0.004, y, f"{val:.3f}", va="center", ha="left", fontsize=8.5, color="#1E293B", fontweight="medium")

    cat_labels = cat_df["Description"] if "Description" in cat_df.columns else cat_df["Variable"]
    ax1.set_yticks(y_pos_cat)
    ax1.set_yticklabels(cat_labels, fontsize=8.5, color="#1E293B")
    ax1.set_xlabel("Cramér's V (Categorical Association)", fontsize=9.5, fontweight="bold", color="#0F172A", labelpad=8)
    ax1.set_title("Panel A: Categorical Features (Cramér's V)", fontsize=11, fontweight="bold", color="#0F172A")
    ax1.set_xlim(0, max(cat_df[cramers_col]) * 1.15)
    ax1.grid(axis='x', linestyle=':', alpha=0.5)

    # Panel B: Absolute Rank-Biserial Correlation (|r_rb| < 0.10: Negligible, 0.10–0.30: Small, 0.30–0.50: Moderate, >= 0.50: Large)
    y_pos_num = np.arange(len(num_df))
    ax2.axvspan(0.00, 0.10, facecolor="#F8FAFC", alpha=0.9, zorder=1) # Negligible
    ax2.axvspan(0.10, 0.30, facecolor="#EFF6FF", alpha=0.9, zorder=1) # Small
    ax2.axvspan(0.30, 0.50, facecolor="#ECFDF5", alpha=0.9, zorder=1) # Moderate

    for border in [0.10, 0.30]:
        ax2.axvline(border, color="#CBD5E1", linestyle="--", linewidth=0.8, zorder=2)

    ax2.hlines(y_pos_num, xmin=0, xmax=num_df[rb_col], color="#64748B", linewidth=1.2, zorder=3)
    ax2.scatter(num_df[rb_col], y_pos_num, color="#D97706", marker="s", s=60, zorder=4, edgecolor="#0F172A", linewidth=0.8)

    for y, (_, row) in zip(y_pos_num, num_df.iterrows()):
        val = row[rb_col]
        ax2.text(val + 0.005, y, f"{val:.3f}", va="center", ha="left", fontsize=8.5, color="#1E293B", fontweight="medium")

    num_labels = num_df["Description"] if "Description" in num_df.columns else num_df["Variable"]
    ax2.set_yticks(y_pos_num)
    ax2.set_yticklabels(num_labels, fontsize=8.5, color="#1E293B")
    ax2.set_xlabel("Absolute Rank-Biserial Correlation", fontsize=9.5, fontweight="bold", color="#0F172A", labelpad=8)
    ax2.set_title("Panel B: Numerical Features (Rank-Biserial r)", fontsize=11, fontweight="bold", color="#0F172A")
    ax2.set_xlim(0, 0.45)
    ax2.grid(axis='x', linestyle=':', alpha=0.5)

    fig.suptitle("Effect-size evidence by feature family", fontsize=13, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = DOCS_FIG_DIR / "effect_size_ranking.png"
    results_png_path = RESULTS_STAT_DIR / "cramers_v_ranking.png"

    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(results_png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Generated 2-panel effect size figure: {png_path}")

if __name__ == "__main__":
    generate_effect_size_figure()
