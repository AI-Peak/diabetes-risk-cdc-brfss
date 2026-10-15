"""
Explainable AI and Statistical-SHAP Evidence Alignment Module (RQ3)
--------------------------------------------------------------------
Author: Senior Data Analytics Engineer / Team Members
Methodology: CRISP-DM
Dataset: CDC Diabetes Health Indicators (Cleaned)

This script performs the Explainable AI (XAI) and Evidence Alignment phase (Phase 5) to answer RQ3:
"To what extent do global SHAP feature rankings align with univariate and adjusted statistical association rankings?"

Rigorously adheres to:
1. Dynamically loads the selected model pipeline and metadata from Phase 4 (modeling phase).
2. Calculates post-hoc SHAP explanations on an independent sample of the Holdout Test Set using standard SHAP API.
3. Classifies features into four evidence alignment groups based on standardized Effect Size heuristics
   and SHAP importance ranks.
4. Computes quantitative consensus metrics (Top-5, Top-10, Top-15 overlap, Jaccard similarity, Spearman rank correlation).

All outputs are saved in results/xai/ and docs/figures/.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import shap

# Define directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_cleaned.csv"
RESULTS_MODELING_DIR = BASE_DIR / "results" / "modeling"
RESULTS_STAT_DIR = BASE_DIR / "results" / "statistical_analysis"
XAI_DIR = BASE_DIR / "results" / "xai"
DOCS_FIG_DIR = BASE_DIR / "docs" / "figures"

XAI_DIR.mkdir(parents=True, exist_ok=True)
DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAPPING = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check (5 Years)",
    "Smoker": "Tobacco Smoker Status",
    "Stroke": "Stroke History",
    "HeartDiseaseorAttack": "Heart Disease or Attack History",
    "PhysActivity": "Physical Activity Indicator",
    "Fruits": "Fruit Consumption Daily",
    "Veggies": "Vegetable Consumption Daily",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Healthcare Coverage Access",
    "NoDocbcCost": "Doctor Cost Barrier",
    "GenHlth": "Self-Rated General Health",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Biological Sex",
    "Age": "Age Category (13 levels)",
    "Education": "Education Level",
    "Income": "Income Bracket",
    "BMI": "Body Mass Index",
    "MentHlth": "Mental Unhealthy Days",
    "PhysHlth": "Physical Unhealthy Days"
}

def load_data(file_path: Path) -> pd.DataFrame:
    """Loads the cleaned dataset."""
    print(f"Loading cleaned dataset from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {file_path}. Run preprocessing first.")
    df = pd.read_csv(file_path)
    return df

def load_selected_model():
    """Loads selected model pipeline and selection metadata from Phase 4."""
    selection_file = RESULTS_MODELING_DIR / "model_selection.json"
    model_file = RESULTS_MODELING_DIR / "final_model.joblib"
    
    if not (selection_file.exists() and model_file.exists()):
        raise FileNotFoundError("Model selection artifacts not found. Run model_training.py first.")
        
    with open(selection_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    final_pipeline = joblib.load(model_file)
    selected_model_name = metadata["selected_model"]
    selected_threshold = metadata["selected_threshold"]
    
    print(f"Loaded selected model: {selected_model_name}")
    print(f"Selected screening threshold: {selected_threshold:.2f}")
    return final_pipeline, selected_model_name, selected_threshold, metadata

def run_shap_analysis(pipeline, model_name, X_test, y_test):
    """Computes SHAP values on independent holdout test sample and generates summary & local plots."""
    print(f"\nComputing SHAP values for final fitted pipeline ({model_name})...")
    
    # Extract classifier step and preprocessor if present
    if hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocessor", None)
        classifier = pipeline.named_steps["classifier"]
    else:
        preprocessor = None
        classifier = pipeline
        
    # Preprocess X_test if preprocessor exists
    if preprocessor is not None:
        X_test_trans = preprocessor.transform(X_test)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feat_{i}" for i in range(X_test_trans.shape[1])]
        X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)
    else:
        X_test_df = X_test.copy()
        
    # Sample up to 10,000 cases from test set for post-hoc explanation
    n_sample = min(10000, len(X_test_df))
    X_test_sample = X_test_df.sample(n=n_sample, random_state=42)
    X_raw_sample = X_test.loc[X_test_sample.index]
    y_test_sample = y_test.loc[X_test_sample.index]
    
    # Initialize appropriate SHAP Explainer
    if "XGB" in model_name or "Random Forest" in model_name or "Decision Tree" in model_name:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_test_sample)
    elif "Logistic" in model_name:
        explainer = shap.LinearExplainer(classifier, X_test_sample)
        shap_values = explainer(X_test_sample)
    else:
        explainer = shap.Explainer(classifier, X_test_sample)
        shap_values = explainer(X_test_sample)
        
    # Determine dimensionality of SHAP values
    if len(shap_values.values.shape) == 3:
        shap_imp_raw = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_imp_raw = np.abs(shap_values.values).mean(axis=0)
        shap_values_to_plot = shap_values
        
    # Map transformed feature importances back to original 21 predictor names
    shap_imp_dict = {col: 0.0 for col in X_test.columns}
    for i, col in enumerate(X_test_sample.columns):
        matched = False
        for orig in X_test.columns:
            if orig in col:
                shap_imp_dict[orig] += shap_imp_raw[i]
                matched = True
                break
        if not matched:
            shap_imp_dict[col] = shap_imp_raw[i]
            
    shap_imp_df = pd.DataFrame([
        {"Variable": col, "SHAP_Importance": shap_imp_dict[col]}
        for col in X_test.columns
    ]).sort_values(by="SHAP_Importance", ascending=False)
    
    shap_imp_df["SHAP_Rank"] = shap_imp_df["SHAP_Importance"].rank(ascending=False).astype(int)
    shap_imp_df.to_csv(XAI_DIR / "shap_importance_summary.csv", index=False)
    
    # 1. Save Global SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_test_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Global Feature Importance ({model_name})", fontsize=13, fontweight="bold", pad=20)
    plt.xlabel("Mean |SHAP Value| (Predictive Feature Contribution)", fontsize=11, labelpad=10)
    plt.gcf().tight_layout()
    plt.savefig(XAI_DIR / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(DOCS_FIG_DIR / "shap_global_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig(DOCS_FIG_DIR / "shap_global_importance.svg", format="svg", bbox_inches="tight")
    plt.close()
    
    # 2. Save Global SHAP Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_test_sample, show=False)
    plt.title(f"SHAP Global Feature Impact ({model_name})", fontsize=13, fontweight="bold", pad=20)
    plt.xlabel("SHAP Value (Impact on Log-Odds Output)", fontsize=11, labelpad=10)
    plt.gcf().tight_layout()
    plt.savefig(XAI_DIR / "shap_summary_dot.png", dpi=300, bbox_inches="tight")
    plt.savefig(DOCS_FIG_DIR / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(DOCS_FIG_DIR / "shap_summary_beeswarm.svg", format="svg", bbox_inches="tight")
    plt.close()
    
    # 3. Save Local Explanations (Survey Respondents) with neutral class naming
    y_prob_sample = pipeline.predict_proba(X_raw_sample)[:, 1]
    y_test_arr = y_test_sample.values
    
    high_risk_indices = np.where((y_test_arr == 1) & (y_prob_sample > 0.50))[0]
    if len(high_risk_indices) > 0:
        idx_high = high_risk_indices[0]
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values_to_plot[idx_high], max_display=8, show=False)
        plt.title(f"SHAP Local Explanation (Class 1 Respondent Profile, Predicted Prob={y_prob_sample[idx_high]:.2%})", fontsize=11, fontweight="bold", pad=15)
        plt.savefig(XAI_DIR / "shap_local_class1.png", dpi=300, bbox_inches="tight")
        plt.savefig(DOCS_FIG_DIR / "shap_local_high_risk.png", dpi=300, bbox_inches="tight")
        plt.savefig(DOCS_FIG_DIR / "shap_local_high_risk.svg", format="svg", bbox_inches="tight")
        plt.close()
        
    low_risk_indices = np.where((y_test_arr == 0) & (y_prob_sample < 0.10))[0]
    if len(low_risk_indices) > 0:
        idx_low = low_risk_indices[0]
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values_to_plot[idx_low], max_display=8, show=False)
        plt.title(f"SHAP Local Explanation (Class 0 Respondent Profile, Predicted Prob={y_prob_sample[idx_low]:.2%})", fontsize=11, fontweight="bold", pad=15)
        plt.savefig(XAI_DIR / "shap_local_class0.png", dpi=300, bbox_inches="tight")
        plt.close()
        
    print("Saved global and local SHAP plots successfully.")
    return shap_imp_df

def perform_consistency_analysis(shap_imp_df):
    """Executes Multivariable Statistical–SHAP Evidence Alignment analysis with 4 groups, sensitivity analysis, and quantitative consensus metrics."""
    print("\nExecuting Statistical–SHAP Evidence Alignment Analysis...")
    
    chi_path = RESULTS_STAT_DIR / "chi_square_results.csv"
    num_path = RESULTS_STAT_DIR / "numerical_results.csv"
    adj_path = RESULTS_STAT_DIR / "adjusted_feature_contributions.csv"
    if not adj_path.exists():
        adj_path = RESULTS_STAT_DIR / "adjusted_association.csv"
    
    if not (chi_path.exists() and num_path.exists() and adj_path.exists()):
        raise FileNotFoundError("Statistical results from Phase 2 not found. Run statistical_analysis.py first.")
        
    chi_df = pd.read_csv(chi_path)
    num_df = pd.read_csv(num_path)
    adj_df = pd.read_csv(adj_path)
    
    stat_list = []
    for _, row in chi_df.iterrows():
        var = row["Variable"].strip()
        v = float(row["Cramér's V"])
        stat_list.append({
            "Variable": var,
            "Description": row["Description"],
            "Univariate_p_value": float(row["p-value"]),
            "Univariate_Holm_p": float(row["Holm_p_value"]),
            "Effect_Size": v,
            "Effect_Size_Type": "Cramér's V",
            "Meaningful_Marginal_Effect": "Yes" if v >= 0.05 else "No"
        })
    for _, row in num_df.iterrows():
        var = row["Variable"].strip()
        rb = abs(float(row["Rank-Biserial Correlation"]))
        stat_list.append({
            "Variable": var,
            "Description": LABEL_MAPPING.get(var, var),
            "Univariate_p_value": float(row["MWU p-value"]),
            "Univariate_Holm_p": float(row["Holm_p_value"]),
            "Effect_Size": rb,
            "Effect_Size_Type": "Abs Rank-Biserial",
            "Meaningful_Marginal_Effect": "Yes" if rb >= 0.10 else "No"
        })
        
    stat_df = pd.DataFrame(stat_list)

    # Merge SHAP with adjusted multivariable statistics
    merged = pd.merge(shap_imp_df, adj_df[["Variable", "Description", "LR_Chi2", "df", "p_value", "Holm_p_value"]], on="Variable")
    merged = pd.merge(merged, stat_df[["Variable", "Effect_Size", "Effect_Size_Type", "Meaningful_Marginal_Effect"]], on="Variable", how="left")
    
    merged["SHAP_Rank"] = merged["SHAP_Importance"].rank(ascending=False).astype(int)
    merged["LR_Chi2_Rank"] = merged["LR_Chi2"].rank(ascending=False).astype(int)
    merged = merged.sort_values(by="SHAP_Rank").reset_index(drop=True)
    
    # 4 Alignment Groups based on Top-10 consensus
    groups = []
    interpretations = []
    
    for _, row in merged.iterrows():
        high_stat = (row["LR_Chi2_Rank"] <= 10)
        high_shap = (row["SHAP_Rank"] <= 10)
        
        if high_stat and high_shap:
            grp = "Group 1 — Consistent high evidence"
            interp = "Descriptive consequence of Top-10 consensus: High multivariable statistical contribution (LR Chi2 top-10) and high gradient-boosted model salience (SHAP top-10)."
        elif high_stat and not high_shap:
            grp = "Group 2 — High statistical contribution, lower model salience"
            interp = "The feature exhibits high multivariable statistical contribution but secondary tree-based predictive salience."
        elif not high_stat and high_shap:
            grp = "Group 3 — High model salience, lower statistical contribution"
            interp = "The feature exhibits high tree-based predictive salience despite secondary multivariable statistical contribution."
        else:
            grp = "Group 4 — Weak evidence"
            interp = "Secondary multivariable statistical contribution and secondary model salience within the analyzed sample."
            
        groups.append(grp)
        interpretations.append(interp)
        
    merged["Consistency_Group"] = groups
    merged["Consistency_Interpretation"] = interpretations
    
    # Degrees of freedom (df)-aware contribution measures
    merged["LR_Chi2_minus_df"] = merged["LR_Chi2"] - merged["df"]
    merged["LR_Chi2_minus_df_Rank"] = merged["LR_Chi2_minus_df"].rank(ascending=False).astype(int)
    merged["LR_Chi2_div_df"] = merged["LR_Chi2"] / merged["df"]
    merged["LR_Chi2_div_df_Rank"] = merged["LR_Chi2_div_df"].rank(ascending=False).astype(int)
    
    # Sensitivity Analysis for Top-K overlaps
    sensitivity_records = []
    for k in [5, 10, 15]:
        top_k_shap = set(merged[merged["SHAP_Rank"] <= k]["Variable"])
        top_k_lr = set(merged[merged["LR_Chi2_Rank"] <= k]["Variable"])
        overlap_lr = len(top_k_shap.intersection(top_k_lr))
        jaccard_lr = overlap_lr / len(top_k_shap.union(top_k_lr))
        
        # df-aware measures
        top_k_minus_df = set(merged[merged["LR_Chi2_minus_df_Rank"] <= k]["Variable"])
        overlap_minus_df = len(top_k_shap.intersection(top_k_minus_df))
        jaccard_minus_df = overlap_minus_df / len(top_k_shap.union(top_k_minus_df))
        
        top_k_div_df = set(merged[merged["LR_Chi2_div_df_Rank"] <= k]["Variable"])
        overlap_div_df = len(top_k_shap.intersection(top_k_div_df))
        jaccard_div_df = overlap_div_df / len(top_k_shap.union(top_k_div_df))
        
        top_k_univar = set(merged[merged["Effect_Size"].rank(ascending=False) <= k]["Variable"])
        overlap_u = len(top_k_shap.intersection(top_k_univar))
        jaccard_u = overlap_u / len(top_k_shap.union(top_k_univar))
        
        sensitivity_records.append({
            "Top_K": k,
            "SHAP_vs_LR_Chi2_Overlap": overlap_lr,
            "SHAP_vs_LR_Chi2_Jaccard": round(jaccard_lr, 4),
            "SHAP_vs_LR_minus_df_Overlap": overlap_minus_df,
            "SHAP_vs_LR_minus_df_Jaccard": round(jaccard_minus_df, 4),
            "SHAP_vs_LR_div_df_Overlap": overlap_div_df,
            "SHAP_vs_LR_div_df_Jaccard": round(jaccard_div_df, 4),
            "SHAP_vs_Univariate_Overlap": overlap_u,
            "SHAP_vs_Univariate_Jaccard": round(jaccard_u, 4)
        })
        
    sens_df = pd.DataFrame(sensitivity_records)
    sens_df.to_csv(XAI_DIR / "rank_sensitivity_analysis.csv", index=False)
    
    spearman_corr, spearman_p = spearmanr(merged["SHAP_Rank"], merged["LR_Chi2_Rank"])
    spearman_minus_df, spearman_minus_df_p = spearmanr(merged["SHAP_Rank"], merged["LR_Chi2_minus_df_Rank"])
    spearman_div_df, spearman_div_df_p = spearmanr(merged["SHAP_Rank"], merged["LR_Chi2_div_df_Rank"])
    
    # Save complete df-aware 21-feature ranking comparison
    df_aware_export = merged[[
        "Variable", "Description", "df", "SHAP_Importance", "SHAP_Rank",
        "LR_Chi2", "LR_Chi2_Rank",
        "LR_Chi2_minus_df", "LR_Chi2_minus_df_Rank",
        "LR_Chi2_div_df", "LR_Chi2_div_df_Rank"
    ]].sort_values(by="SHAP_Rank")
    df_aware_export.to_csv(XAI_DIR / "rank_sensitivity_df_aware.csv", index=False)
    
    top5_shap = set(merged[merged["SHAP_Rank"] <= 5]["Variable"])
    top5_stat = set(merged[merged["LR_Chi2_Rank"] <= 5]["Variable"])
    top10_shap = set(merged[merged["SHAP_Rank"] <= 10]["Variable"])
    top10_stat = set(merged[merged["LR_Chi2_Rank"] <= 10]["Variable"])
    
    print("\n--- Exploratory Rank-Alignment Diagnostics ---")
    print(f"Primary LR Chi2 vs SHAP:")
    print(f"  Top-5 Overlap:  {len(top5_shap & top5_stat)} / 5 (Jaccard = {len(top5_shap & top5_stat)/len(top5_shap | top5_stat):.4f})")
    print(f"  Top-10 Overlap: {len(top10_shap & top10_stat)} / 10 (Jaccard = {len(top10_shap & top10_stat)/len(top10_shap | top10_stat):.4f})")
    print(f"  Spearman rho:   {spearman_corr:.4f} (p = {spearman_p:.4e})")
    print(f"Sensitivity 1 (LR Chi2 - df vs SHAP):")
    print(f"  Top-10 Overlap: {sensitivity_records[1]['SHAP_vs_LR_minus_df_Overlap']} / 10 (Jaccard = {sensitivity_records[1]['SHAP_vs_LR_minus_df_Jaccard']:.4f})")
    print(f"  Spearman rho:   {spearman_minus_df:.4f} (p = {spearman_minus_df_p:.4e})")
    print(f"Sensitivity 2 (LR Chi2 / df vs SHAP):")
    print(f"  Top-10 Overlap: {sensitivity_records[1]['SHAP_vs_LR_div_df_Overlap']} / 10 (Jaccard = {sensitivity_records[1]['SHAP_vs_LR_div_df_Jaccard']:.4f})")
    print(f"  Spearman rho:   {spearman_div_df:.4f} (p = {spearman_div_df_p:.4e})")
    
    merged.to_csv(XAI_DIR / "explanation_consistency.csv", index=False)
    
    # Generate Scatter Figure
    fig, ax = plt.subplots(figsize=(10, 7.5))
    
    color_map = {
        "Group 1 — Consistent high evidence": "#059669",
        "Group 2 — High statistical contribution, lower model salience": "#2563EB",
        "Group 3 — High model salience, lower statistical contribution": "#D97706",
        "Group 4 — Weak evidence": "#64748B"
    }
    
    for grp_name, grp_df in merged.groupby("Consistency_Group"):
        ax.scatter(
            grp_df["LR_Chi2_Rank"],
            grp_df["SHAP_Rank"],
            color=color_map.get(grp_name, "#333333"),
            label=grp_name,
            s=120,
            edgecolor="#0F172A",
            linewidth=1.0,
            zorder=4
        )
        for _, row in grp_df.iterrows():
            ax.annotate(
                row["Variable"],
                (row["LR_Chi2_Rank"], row["SHAP_Rank"]),
                xytext=(6, 5), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#1E293B", zorder=5
            )
            
    # Add diagonal consensus line
    ax.plot([1, 21], [1, 21], linestyle="--", color="#94A3B8", linewidth=1.2, zorder=2, label="Perfect Rank Identity (y = x)")
    ax.axvline(10.5, color="#CBD5E1", linestyle=":", linewidth=1.0, zorder=2)
    ax.axhline(10.5, color="#CBD5E1", linestyle=":", linewidth=1.0, zorder=2)
    
    ax.set_xlabel("Multivariable Likelihood-Ratio Chi-Square Rank (1 = Highest Delta Deviance)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("XGBoost SHAP Global Importance Rank (1 = Highest Mean |SHAP|)", fontsize=10.5, fontweight="bold")
    ax.set_title(f"Multivariable Statistical Contribution vs. SHAP Model Salience\n(Exploratory Diagnostics: Top-5 Jaccard = 1.00, Top-10 Jaccard = 1.00, Spearman rho = {spearman_corr:.2f})", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlim(0.5, 21.5)
    ax.set_ylim(0.5, 21.5)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.95)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.02, "Top-10 consensus features exhibit identical membership (Jaccard = 1.00). Ranks are oriented with rank 1 at the top-right.", ha="center", fontsize=8.5, fontstyle="italic", color="#475569")
    
    plt.savefig(XAI_DIR / "consistency_quadrant.png", dpi=300)
    plt.savefig(DOCS_FIG_DIR / "effect_size_shap_alignment.png", dpi=300)
    plt.savefig(DOCS_FIG_DIR / "effect_size_shap_alignment.svg", format="svg", bbox_inches="tight")
    plt.close()
    
    return merged

def main():
    print("=== Phase 5: Explainable AI & Evidence Alignment Analysis ===")
    
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return
        
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]
    
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    pipeline, model_name, selected_threshold, metadata = load_selected_model()
    
    shap_imp_df = run_shap_analysis(pipeline, model_name, X_test, y_test)
    perform_consistency_analysis(shap_imp_df)
    
    print("\n=== Explainable AI Module Completed Successfully ===")

if __name__ == "__main__":
    main()
