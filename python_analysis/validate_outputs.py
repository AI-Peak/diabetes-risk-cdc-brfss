"""
Automated Output Validation Module
-----------------------------------
Author: Senior Data Analytics Engineer / Team
Description: Validates pipeline artifacts for correctness, schema integrity, metric ranges,
             confusion matrix consistency, and documentation guidelines.
"""

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
DATA_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_cleaned.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

def validate_pipeline_outputs():
    print("=== Starting Pipeline Output Validation ===")
    errors = []

    # 1. Raw & Processed Data Checks
    if not DATA_RAW_PATH.exists():
        errors.append(f"Raw data file missing at: {DATA_RAW_PATH}")
    else:
        df_raw = pd.read_csv(DATA_RAW_PATH)
        if df_raw.shape != (253680, 22):
            errors.append(f"Raw dataset shape changed! Expected (253680, 22), got {df_raw.shape}")

    if not DATA_CLEAN_PATH.exists():
        errors.append(f"Processed data file missing at: {DATA_CLEAN_PATH}")
    else:
        df_clean = pd.read_csv(DATA_CLEAN_PATH)
        if df_clean.shape != (253680, 22):
            errors.append(f"Cleaned dataset shape invalid! Expected (253680, 22), got {df_clean.shape}")
        if df_clean.isnull().sum().sum() > 0:
            errors.append("Cleaned dataset contains unexpected missing values.")

    # 2. Required Results & Manuscript Artifacts Checks
    required_files = [
        # Data understanding & duplicate audit
        RESULTS_DIR / "data_understanding" / "dataset_audit.json",
        RESULTS_DIR / "data_understanding" / "dataset_audit.csv",
        RESULTS_DIR / "data_understanding" / "duplicate_summary.csv",
        # Statistical analysis
        RESULTS_DIR / "statistical_analysis" / "chi_square_results.csv",
        RESULTS_DIR / "statistical_analysis" / "numerical_results.csv",
        RESULTS_DIR / "statistical_analysis" / "adjusted_association.csv",
        RESULTS_DIR / "statistical_analysis" / "multivariable_associations.csv",
        RESULTS_DIR / "statistical_analysis" / "adjusted_feature_contributions.csv",
        # Modeling & holdout evaluation
        RESULTS_DIR / "modeling" / "model_selection.json",
        RESULTS_DIR / "modeling" / "final_test_metrics.csv",
        RESULTS_DIR / "modeling" / "threshold_analysis.csv",
        RESULTS_DIR / "modeling" / "calibration_metrics.csv",
        RESULTS_DIR / "modeling" / "model_configuration.json",
        RESULTS_DIR / "modeling" / "final_model.joblib",
        # XAI & evidence alignment
        RESULTS_DIR / "xai" / "explanation_consistency.csv",
        RESULTS_DIR / "xai" / "rank_sensitivity_analysis.csv",
        # Summaries
        RESULTS_DIR / "final_results_summary.json",
        RESULTS_DIR / "final_results_summary.md"
    ]
    for rf in required_files:
        if not rf.exists():
            errors.append(f"Required result artifact missing: {rf.relative_to(PROJECT_ROOT)}")

    # 3. Metric Range & Confusion Matrix Consistency Checks
    test_metrics_path = RESULTS_DIR / "modeling" / "final_test_metrics.csv"
    if test_metrics_path.exists():
        df_metrics = pd.read_csv(test_metrics_path)
        for _, row in df_metrics.iterrows():
            lbl = row["Operating_Threshold_Label"]
            for m in ["Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC", "PR-AUC"]:
                val = row[m]
                if not (0.0 <= val <= 1.0):
                    errors.append(f"Metric {m} out of range [0, 1] for {lbl}: {val}")
                    
            tp, fp, tn, fn = row["TP"], row["FP"], row["TN"], row["FN"]
            total_holdout = tp + fp + tn + fn
            if total_holdout != 50736:
                errors.append(f"Confusion matrix sum != 50736 for {lbl}: {total_holdout}")
                
            calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            calc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            calc_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            calc_acc = (tp + tn) / total_holdout
            
            if abs(calc_prec - row["Precision"]) > 1e-3:
                errors.append(f"Precision mismatch for {lbl}: reported {row['Precision']}, calculated {calc_prec:.4f}")
            if abs(calc_rec - row["Recall"]) > 1e-3:
                errors.append(f"Recall mismatch for {lbl}: reported {row['Recall']}, calculated {calc_rec:.4f}")
            if abs(calc_spec - row["Specificity"]) > 1e-3:
                errors.append(f"Specificity mismatch for {lbl}: reported {row['Specificity']}, calculated {calc_spec:.4f}")

    # 4. Calibration Assessment Check
    calib_path = RESULTS_DIR / "modeling" / "calibration_metrics.csv"
    if calib_path.exists():
        df_calib = pd.read_csv(calib_path)
        brier = df_calib.iloc[0]["Brier_Score"]
        if not (0.0 <= brier <= 1.0):
            errors.append(f"Brier score out of range [0, 1]: {brier}")

    # 5. Documentation Integrity Checks
    md_files = list(DOCS_DIR.glob("*.md")) + [PROJECT_ROOT / "README.md"]
    forbidden_terms = ["file:///", "C:\\Users", "/Users/"]
    for mdf in md_files:
        if mdf.exists():
            content = mdf.read_text(encoding="utf-8")
            for term in forbidden_terms:
                if term in content:
                    errors.append(f"Forbidden path pattern '{term}' found in: {mdf.relative_to(PROJECT_ROOT)}")

    if errors:
        print("\n[FAILED] VALIDATION FAILED WITH THE FOLLOWING ERRORS:")
        for err in errors:
            print(f"  - {err}")
        raise ValueError(f"Output validation failed with {len(errors)} error(s).")
    else:
        print("\n[SUCCESS] ALL PIPELINE OUTPUT VALIDATION CHECKS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    validate_pipeline_outputs()
