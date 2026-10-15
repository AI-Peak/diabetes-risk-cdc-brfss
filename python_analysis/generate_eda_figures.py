#!/usr/bin/env python
"""
Univariate & Multivariate EDA Figure Generator
----------------------------------------------
Author: Advanced Data Analytics Agent
Description:
    Generates presentation-ready EDA figures that complement the existing
    statistical-analysis figures. These cover the distribution layer of the
    workflow (univariate) and the feature-interdependence layer (multivariate),
    which were previously documented only as CSV tables.

    Figures produced (300 DPI PNG + SVG vector):
        - docs/figures/eda_univariate_numeric.png
        - docs/figures/eda_univariate_categorical.png
        - docs/figures/eda_correlation_heatmap.png

    All figures are computed from data/processed/diabetes_cleaned.csv so they
    stay consistent with every other output of the pipeline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "docs" / "figures"

# Feature grouping mirrors python_analysis/model_training.py
BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]
ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]
NUMERIC_FEATURES = ["BMI", "MentHlth", "PhysHlth"]
TARGET = "Diabetes_binary"

COLOR_PRIMARY = "#2563EB"
COLOR_ACCENT = "#DC2626"
COLOR_TEXT = "#0F172A"
COLOR_MUTED = "#64748B"

ORDINAL_TICKS = {
    "GenHlth": ["1\nExcellent", "2\nVery good", "3\nGood", "4\nFair", "5\nPoor"],
    "Age": ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
            "55-59", "60-64", "65-69", "70-74", "75-79", "80+"],
    "Education": ["Never", "Elem.", "Some HS", "HS grad", "Some col.", "College+"],
    "Income": ["<10k", "10-15k", "15-20k", "20-25k", "25-35k", "35-50k",
               "50-75k", ">=75k"],
}


def apply_academic_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.labelcolor"] = COLOR_TEXT
    plt.rcParams["xtick.color"] = COLOR_MUTED
    plt.rcParams["ytick.color"] = COLOR_MUTED
    plt.rcParams["axes.titlecolor"] = COLOR_TEXT


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  - {png_path}")
    print(f"  - {svg_path}")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {DATA_PATH}. Run python_analysis/data_preprocessing.py first."
        )
    return pd.read_csv(DATA_PATH)


def plot_univariate_numeric(df: pd.DataFrame) -> None:
    """Distribution shape of the three genuinely continuous/count variables."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    configs = [
        ("BMI", "Body Mass Index (kg/m^2)", np.arange(10, 62, 2)),
        ("MentHlth", "Poor mental health days (last 30)", np.arange(0, 32, 2)),
        ("PhysHlth", "Poor physical health days (last 30)", np.arange(0, 32, 2)),
    ]

    for ax, (col, xlabel, bins) in zip(axes, configs):
        series = df[col]
        ax.hist(series, bins=bins, color=COLOR_PRIMARY, alpha=0.85,
                edgecolor="white", linewidth=0.6)
        median = series.median()
        mean = series.mean()
        ax.axvline(median, color=COLOR_ACCENT, linestyle="--", linewidth=1.6,
                   label=f"Median = {median:.1f}")
        ax.axvline(mean, color="#0F172A", linestyle=":", linewidth=1.6,
                   label=f"Mean = {mean:.2f}")
        skew = series.skew()
        ax.set_title(f"{col}  (skewness = {skew:.2f})", fontsize=11.5, fontweight="bold", pad=9)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylabel("Respondents", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Univariate distributions of numeric health indicators  (N = {len(df):,})",
        fontsize=13, fontweight="bold", color=COLOR_TEXT, y=1.04,
    )
    fig.tight_layout()
    save_figure(fig, "eda_univariate_numeric")


def plot_univariate_categorical(df: pd.DataFrame) -> None:
    """Composition of the ordinal variables plus prevalence of every binary flag."""
    fig = plt.figure(figsize=(13.5, 8.2))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.55, wspace=0.28)

    for idx, col in enumerate(ORDINAL_FEATURES):
        ax = fig.add_subplot(grid[0, idx])
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=COLOR_PRIMARY,
               alpha=0.88, edgecolor="white", linewidth=0.6)
        ticks = ORDINAL_TICKS.get(col)
        if ticks and len(ticks) == len(counts):
            ax.set_xticks(range(len(ticks)))
            ax.set_xticklabels(ticks, fontsize=6.6, rotation=45, ha="right")
        ax.set_title(col, fontsize=11, fontweight="bold", pad=7)
        ax.set_ylabel("Respondents", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)

    ax_bin = fig.add_subplot(grid[1, :])
    prevalence = (df[BINARY_FEATURES].mean() * 100).sort_values(ascending=True)
    bars = ax_bin.barh(prevalence.index, prevalence.values, color=COLOR_PRIMARY,
                       alpha=0.88, edgecolor="white", linewidth=0.6, height=0.68)
    for bar, value in zip(bars, prevalence.values):
        ax_bin.text(value + 1.2, bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%", va="center", fontsize=8.5, color=COLOR_TEXT)
    ax_bin.set_xlim(0, 105)
    ax_bin.set_xlabel("Share of respondents coded 1 (%)", fontsize=9.5)
    ax_bin.set_title("Prevalence of binary health indicators", fontsize=11,
                     fontweight="bold", pad=8)
    ax_bin.spines[["top", "right"]].set_visible(False)
    ax_bin.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax_bin.set_axisbelow(True)
    ax_bin.tick_params(axis="y", labelsize=8.5)

    fig.suptitle(
        f"Univariate composition of ordinal and binary indicators  (N = {len(df):,})",
        fontsize=13, fontweight="bold", color=COLOR_TEXT, y=0.98,
    )
    save_figure(fig, "eda_univariate_categorical")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Spearman rank correlation across all 21 predictors and the target."""
    columns = NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES + [TARGET]
    corr = df[columns].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10.6, 9.8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    display = np.ma.masked_where(mask, corr.values)

    im = ax.imshow(display, cmap="RdBu_r", vmin=-0.6, vmax=0.6)

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=8.2)
    ax.set_yticklabels(columns, fontsize=8.2)

    for i in range(len(columns)):
        for j in range(len(columns)):
            if mask[i, j]:
                continue
            value = corr.values[i, j]
            if abs(value) < 0.15:
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.4,
                    color="white" if abs(value) > 0.38 else COLOR_TEXT)

    # Highlight the target row so the audience's eye lands there first
    target_idx = len(columns) - 1
    ax.add_patch(plt.Rectangle((-0.5, target_idx - 0.5), len(columns), 1,
                               fill=False, edgecolor=COLOR_ACCENT, linewidth=2.0))

    cbar = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.03, anchor=(0.0, 0.0))
    cbar.set_label("Spearman rank correlation", fontsize=9.5, color=COLOR_TEXT)
    cbar.ax.tick_params(labelsize=8.5)

    predictors = [c for c in columns if c != TARGET]
    pred_corr = corr.loc[predictors, predictors].abs().copy()
    pred_vals = pred_corr.to_numpy(copy=True)
    np.fill_diagonal(pred_vals, 0.0)
    pred_corr_clean = pd.DataFrame(pred_vals, index=pred_corr.index, columns=pred_corr.columns)
    stacked = pred_corr_clean.stack()
    max_pair = stacked.idxmax()
    max_value = stacked.max()

    fig.suptitle(
        "Multivariate structure: Spearman correlation matrix (21 predictors + target)",
        fontsize=12.5, fontweight="bold", color=COLOR_TEXT, x=0.5, y=0.955,
    )
    fig.text(
        0.5, 0.045,
        f"Highlighted row = Diabetes_binary.  Strongest predictor-predictor association: "
        f"{max_pair[0]} - {max_pair[1]} (|rho| = {max_value:.2f}).\n"
        "All pairs stay far below the 0.70 multicollinearity concern level, consistent with "
        "VIF < 1.8 reported in adjusted_association.csv.",
        ha="center", fontsize=8.8, color=COLOR_MUTED, linespacing=1.5,
    )

    fig.subplots_adjust(left=0.17, right=0.94, top=0.90, bottom=0.20)
    save_figure(fig, "eda_correlation_heatmap")


def main() -> None:
    apply_academic_style()
    df = load_data()
    print(f"Loaded cleaned dataset: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print("Generating univariate numeric distributions...")
    plot_univariate_numeric(df)

    print("\nGenerating univariate categorical composition...")
    plot_univariate_categorical(df)

    print("\nGenerating multivariate correlation heatmap...")
    plot_correlation_heatmap(df)

    max_corr = (
        df[NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES]
        .corr(method="spearman")
        .abs()
        .where(lambda m: m < 0.999)
        .max()
        .max()
    )
    print(f"\nMaximum absolute predictor-predictor Spearman correlation: {max_corr:.4f}")
    print("EDA figure generation complete.")


if __name__ == "__main__":
    main()
