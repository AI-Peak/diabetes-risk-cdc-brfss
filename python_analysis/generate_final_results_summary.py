"""
Generates results/final_results_summary.json and final_results_summary.md automatically from canonical outputs.
Do NOT hardcode numbers; compute dynamically from dataset, modeling, statistical, and XAI artifacts.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_cleaned.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    n_total = len(df)
    class_counts = df["Diabetes_binary"].astype(int).value_counts()
    no_diabetes_count = int(class_counts.get(0, 0))
    prediabetes_or_diabetes_count = int(class_counts.get(1, 0))
    feature_cols = [c for c in df.columns if c != "Diabetes_binary"]
    exact_duplicate_surplus_rows = int(df.duplicated().sum())
    repeated_predictor_profile_surplus_rows = int(df.duplicated(subset=feature_cols).sum())

    # Predictor profile group metrics
    profile_sizes = df.groupby(feature_cols).size()
    multi_profiles = profile_sizes[profile_sizes > 1]
    multi_observation_predictor_profiles = int(len(multi_profiles))
    records_in_repeated_predictor_groups = int(multi_profiles.sum())

    profile_targets = df.groupby(feature_cols)["Diabetes_binary"].nunique()
    conflicting_profiles = profile_targets[profile_targets > 1]
    conflicting_label_profiles = int(len(conflicting_profiles))
    observations_in_conflicting_profiles = int(profile_sizes[conflicting_profiles.index].sum())

    model_sel_path = RESULTS_DIR / "modeling" / "model_selection.json"
    consistency_path = RESULTS_DIR / "xai" / "explanation_consistency.csv"
    sens_path = RESULTS_DIR / "xai" / "rank_sensitivity_analysis.csv"
    calib_path = RESULTS_DIR / "modeling" / "calibration_metrics.csv"
    adj_path = RESULTS_DIR / "statistical_analysis" / "adjusted_association.csv"
    
    if not model_sel_path.exists():
        raise FileNotFoundError(f"Model selection metadata not found at {model_sel_path}")
        
    with open(model_sel_path, "r", encoding="utf-8") as f:
        model_sel = json.load(f)

    consistency_df = pd.read_csv(consistency_path) if consistency_path.exists() else None
    sens_df = pd.read_csv(sens_path) if sens_path.exists() else None
    calib_df = pd.read_csv(calib_path) if calib_path.exists() else None
    adj_df = pd.read_csv(adj_path) if adj_path.exists() else None
    
    timestamp = datetime.now(timezone.utc).isoformat()

    summary = {
        "title": "Predicting Diabetes Risk Using CDC Health Indicators - Final Results Summary",
        "metadata": {
            "generated_at_utc": timestamp,
            "random_seed": 42,
            "target_variable": "Diabetes_binary",
            "class_definitions": {
                "0": "No reported diabetes",
                "1": "Prediabetes or diabetes"
            }
        },
        "dataset": {
            "n_records": n_total,
            "n_features": len(df.columns) - 1,
            "exact_duplicate_surplus_rows": exact_duplicate_surplus_rows,
            "repeated_predictor_profile_surplus_rows": repeated_predictor_profile_surplus_rows,
            "multi_observation_predictor_profiles": multi_observation_predictor_profiles,
            "records_in_repeated_predictor_groups": records_in_repeated_predictor_groups,
            "conflicting_label_profiles": conflicting_label_profiles,
            "observations_in_conflicting_profiles": observations_in_conflicting_profiles,
            "target_distribution": {
                "no_diabetes_count": no_diabetes_count,
                "no_diabetes_pct": round((no_diabetes_count / n_total) * 100, 2),
                "prediabetes_or_diabetes_count": prediabetes_or_diabetes_count,
                "prediabetes_or_diabetes_pct": round((prediabetes_or_diabetes_count / n_total) * 100, 2)
            },
            "split": {
                "development_size": model_sel["development_sample_size"],
                "holdout_test_size": model_sel["holdout_test_sample_size"],
                "cv_folds": model_sel["n_folds"]
            }
        },
        "model_selection": {
            "selected_model": model_sel["selected_model"],
            "selection_criterion": model_sel["selection_criterion"],
            "threshold_selection_criterion": model_sel.get("threshold_selection_rule", "Recall >= 0.80 target"),
            "cross_validation_results": model_sel["cross_validation_metrics"]
        },
        "holdout_test_evaluation": {
            "selected_threshold": model_sel["selected_threshold"],
            "metrics_at_selected_threshold": model_sel["final_holdout_test_metrics"]["selected_threshold"],
            "metrics_at_default_0.50": model_sel["final_holdout_test_metrics"]["default_0.50"],
            "bootstrap_95_ci": model_sel["final_holdout_test_metrics"]["bootstrap_95_ci_selected_threshold"]
        }
    }

    if calib_df is not None:
        summary["holdout_test_evaluation"]["calibration"] = calib_df.to_dict(orient="records")[0]

    if adj_df is not None:
        top5_adj = adj_df.head(5)[["Variable", "Odds_Ratio"]].to_dict(orient="records")
        summary["adjusted_association"] = {
            "top_5_odds_ratios": top5_adj,
            "max_vif": round(float(adj_df["VIF"].max()), 2)
        }

    if consistency_df is not None:
        groups = {}
        for group_name, df_g in consistency_df.groupby("Consistency_Group"):
            groups[group_name] = df_g["Variable"].tolist()
        summary["xai_consistency"] = {
            "evidence_groups": groups,
            "total_features": len(consistency_df)
        }
        
    if sens_df is not None:
        summary["xai_consistency"]["rank_sensitivity"] = sens_df.to_dict(orient="records")
        
    out_json_path = RESULTS_DIR / "final_results_summary.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Successfully generated {out_json_path}")

    # Generate Markdown Summary
    md_lines = [
        "# Final Results Summary",
        "",
        "## Dataset & Split",
        f"- **Total Records:** {summary['dataset']['n_records']:,}",
        f"- **Features:** {summary['dataset']['n_features']}",
        f"- **Exact Duplicate Surplus Rows:** {summary['dataset']['exact_duplicate_surplus_rows']:,}",
        f"- **Repeated Predictor Profile Surplus Rows:** {summary['dataset']['repeated_predictor_profile_surplus_rows']:,}",
        f"- **Multi-Observation Predictor Profiles:** {summary['dataset']['multi_observation_predictor_profiles']:,}",
        f"- **Records in Repeated Predictor Groups:** {summary['dataset']['records_in_repeated_predictor_groups']:,}",
        f"- **Conflicting Label Profiles:** {summary['dataset']['conflicting_label_profiles']:,}",
        f"- **Observations in Conflicting Profiles:** {summary['dataset']['observations_in_conflicting_profiles']:,}",
        f"- **Class 0 (No reported diabetes):** {summary['dataset']['target_distribution']['no_diabetes_count']:,} ({summary['dataset']['target_distribution']['no_diabetes_pct']}%)",
        f"- **Class 1 (Prediabetes or diabetes):** {summary['dataset']['target_distribution']['prediabetes_or_diabetes_count']:,} ({summary['dataset']['target_distribution']['prediabetes_or_diabetes_pct']}%)",
        f"- **Development Set:** {summary['dataset']['split']['development_size']:,}",
        f"- **Holdout Test Set:** {summary['dataset']['split']['holdout_test_size']:,}",
        "",
        "## Selected Model & Threshold",
        f"- **Selected Model:** {summary['model_selection']['selected_model']}",
        f"- **Selection Criterion:** {summary['model_selection']['selection_criterion']}",
        f"- **Selected Screening Threshold:** {summary['holdout_test_evaluation']['selected_threshold']:.2f}",
        "",
        "## Holdout Test Performance (Selected Threshold)",
        f"- **PR-AUC:** {summary['holdout_test_evaluation']['metrics_at_selected_threshold']['PR-AUC']:.4f}",
        f"- **ROC-AUC:** {summary['holdout_test_evaluation']['metrics_at_selected_threshold']['ROC-AUC']:.4f}",
        f"- **Recall:** {summary['holdout_test_evaluation']['metrics_at_selected_threshold']['Recall']:.4f}",
        f"- **Precision:** {summary['holdout_test_evaluation']['metrics_at_selected_threshold']['Precision']:.4f}",
        f"- **Specificity:** {summary['holdout_test_evaluation']['metrics_at_selected_threshold']['Specificity']:.4f}",
        f"- **F1-score:** {summary['holdout_test_evaluation']['metrics_at_selected_threshold']['F1-score']:.4f}",
        f"- **Accuracy:** {summary['holdout_test_evaluation']['metrics_at_selected_threshold']['Accuracy']:.4f}",
        ""
    ]
    out_md_path = RESULTS_DIR / "final_results_summary.md"
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Successfully generated {out_md_path}")

if __name__ == "__main__":
    main()

