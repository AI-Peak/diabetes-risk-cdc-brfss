"""
Statistical Hypothesis Testing & Adjusted Association Module (RQ1)
-------------------------------------------------------------------
Author: Senior Data Analytics Engineer / Team Members
Methodology: CRISP-DM
Dataset: CDC Diabetes Health Indicators (Cleaned)

This script performs the statistical analysis phase (Phase 2) to answer RQ1:
"Which demographic, lifestyle, and health-related factors are statistically
associated with diabetes status in the CDC BRFSS 2015 dataset?"

It applies:
- Chi-Square Test of Independence and Cramér's V for categorical/binary/ordinal variables.
- Mann-Whitney U Test (non-parametric) and Two-Sample t-Test (parametric) with
  Cohen's d and Rank-Biserial Correlation for numerical variables.
- Multivariable Logistic Regression with categorical indicator blocks for ordinal/multiclass
  predictors (GenHlth, Age, Education, Income), reporting dummy-level AORs, 95% CIs, VIFs,
  and feature-level nested Likelihood-Ratio tests (LR Chi2 / delta deviance).

All outputs are saved in results/statistical_analysis/ and docs/statistical_analysis.md.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Define directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results" / "statistical_analysis"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Define column categories
COLUMNS_CATEGORICAL = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", 
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", 
    "DiffWalk", "Sex", "Age", "Education", "Income"
]

COLUMNS_NUMERICAL = ["BMI", "MentHlth", "PhysHlth"]

# Labels for prettier reports
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
    "DiffWalk": "Difficulty Walking or Climbing Stairs",
    "Sex": "Biological Sex",
    "Age": "Age Category (13 levels)",
    "Education": "Education Level (6 levels)",
    "Income": "Income Bracket (8 levels)",
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

def interpret_cramers_v(v: float) -> str:
    """Provides standard interpretation of Cramér's V effect size."""
    if v < 0.05:
        return "Negligible"
    elif v < 0.10:
        return "Weak"
    elif v < 0.20:
        return "Small"
    elif v < 0.30:
        return "Moderate"
    else:
        return "Strong"

def interpret_rank_biserial(r: float) -> str:
    """Provides standard interpretation of absolute Rank-Biserial correlation effect size."""
    r_abs = abs(r)
    if r_abs < 0.10:
        return "Negligible"
    elif r_abs < 0.30:
        return "Small"
    elif r_abs < 0.50:
        return "Moderate"
    else:
        return "Large"

def perform_categorical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Performs Chi-square tests of independence and calculates Cramér's V."""
    print("Performing Chi-Square tests of independence...")
    results = []
    
    for col in COLUMNS_CATEGORICAL:
        if col not in df.columns:
            print(f"Warning: {col} not in dataset. Skipping.")
            continue
            
        contingency_table = pd.crosstab(df[col], df["Diabetes_binary"])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        
        n = contingency_table.sum().sum()
        v = np.sqrt(chi2 / n)
        
        rates = contingency_table.div(contingency_table.sum(axis=1), axis=0)[1] * 100
        rate_diff = rates.max() - rates.min()
        
        results.append({
            "Variable": col,
            "Description": LABEL_MAPPING.get(col, col),
            "Chi2 Statistic": chi2,
            "p-value": p_val,
            "Degrees of Freedom": dof,
            "Cramér's V": v,
            "Effect_Size": v,
            "Effect_Size_Type": "Cramér's V",
            "Effect Size Interpretation": interpret_cramers_v(v),
            "Min Diabetes Rate (%)": rates.min(),
            "Max Diabetes Rate (%)": rates.max(),
            "Max Difference (%)": rate_diff
        })
        
    results_df = pd.DataFrame(results).sort_values(by="Cramér's V", ascending=False)
    return results_df

def perform_numerical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Performs Mann-Whitney U test and Independent t-test on numerical variables."""
    print("Performing statistical tests on numerical variables...")
    results = []
    
    g_class_1 = df[df["Diabetes_binary"] == 1]
    g_class_0 = df[df["Diabetes_binary"] == 0]
    
    n_class_1 = len(g_class_1)
    n_class_0 = len(g_class_0)
    
    for col in COLUMNS_NUMERICAL:
        if col not in df.columns:
            print(f"Warning: {col} not in dataset. Skipping.")
            continue
            
        x_class_1 = g_class_1[col].values
        x_class_0 = g_class_0[col].values
        
        m_class_1, std_class_1 = x_class_1.mean(), x_class_1.std(ddof=1)
        m_class_0, std_class_0 = x_class_0.mean(), x_class_0.std(ddof=1)
        
        med_class_1 = np.median(x_class_1)
        med_class_0 = np.median(x_class_0)
        
        t_stat, t_pval = stats.ttest_ind(x_class_1, x_class_0, equal_var=False)
        
        pooled_std = np.sqrt(((n_class_1 - 1) * std_class_1**2 + (n_class_0 - 1) * std_class_0**2) / (n_class_1 + n_class_0 - 2))
        cohen_d = (m_class_1 - m_class_0) / pooled_std
        
        u_stat, mwu_pval = stats.mannwhitneyu(x_class_1, x_class_0, alternative="two-sided")
        
        cles = u_stat / (n_class_1 * n_class_0)
        rank_biserial = 2 * cles - 1
        abs_rb = abs(rank_biserial)
        
        results.append({
            "Variable": col,
            "No Reported Diabetes Mean": m_class_0,
            "Prediabetes/Diabetes Positive Mean": m_class_1,
            "Mean Difference": m_class_1 - m_class_0,
            "No Reported Diabetes Median": med_class_0,
            "Prediabetes/Diabetes Positive Median": med_class_1,
            "t-Statistic": t_stat,
            "t p-value": t_pval,
            "Cohen's d": cohen_d,
            "MWU U-Statistic": u_stat,
            "MWU p-value": mwu_pval,
            "CLES (Superiority)": cles,
            "Rank-Biserial Correlation": rank_biserial,
            "Effect_Size": abs_rb,
            "Effect_Size_Type": "Abs Rank-Biserial",
            "Effect Size Interpretation": interpret_rank_biserial(abs_rb)
        })
        
    results_df = pd.DataFrame(results)
    return results_df

def perform_adjusted_association_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Performs multivariable logistic regression with categorical indicator blocks for ordinal/categorical
    variables (GenHlth, Age, Education, Income), computes dummy-level AORs, 95% CIs, and VIFs,
    and conducts feature-level nested Likelihood-Ratio tests (LR Chi2 / delta deviance).
    """
    print("Performing multivariable adjusted association analysis (Logistic Regression)...")
    y = df["Diabetes_binary"]
    
    cat_cols = ["GenHlth", "Age", "Education", "Income"]
    binary_cols = [
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", 
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
    ]
    num_cols = ["BMI", "MentHlth", "PhysHlth"]

    # Construct design matrix with dummies
    X_parts = [df[binary_cols + num_cols]]
    col_map = {}
    for col in binary_cols + num_cols:
        col_map[col] = [col]

    for c in cat_cols:
        dummies = pd.get_dummies(df[c], prefix=c, drop_first=True, dtype=float)
        X_parts.append(dummies)
        col_map[c] = list(dummies.columns)

    X_full = pd.concat(X_parts, axis=1)
    X_full_const = sm.add_constant(X_full)

    # 1. Fit full model
    print(f"Fitting full multivariable logistic regression model ({X_full_const.shape[1]} parameters)...")
    full_model = sm.Logit(y, X_full_const).fit(disp=False)
    ll_full = full_model.llf
    params = full_model.params
    conf = full_model.conf_int()
    pvalues = full_model.pvalues
    bse = full_model.bse
    zvalues = full_model.tvalues

    # Compute term-level VIFs
    print("Computing Variance Inflation Factors (VIF)...")
    X_mat = X_full_const.values
    vif_dict = {}
    for i, col in enumerate(X_full_const.columns):
        if col != "const":
            vif_dict[col] = variance_inflation_factor(X_mat, i)

    # Build multivariable_associations.csv (term/dummy-level)
    dummy_records = []
    for term in X_full.columns:
        parent_var = term
        for orig, dummy_list in col_map.items():
            if term in dummy_list:
                parent_var = orig
                break
        
        coef = params[term]
        or_val = np.exp(coef)
        ci_lower = np.exp(conf.loc[term, 0])
        ci_upper = np.exp(conf.loc[term, 1])
        pval = pvalues[term]
        zstat = zvalues[term]
        vif = vif_dict[term]

        dummy_records.append({
            "Term": term,
            "Variable": parent_var,
            "Description": LABEL_MAPPING.get(parent_var, parent_var),
            "Coefficient": coef,
            "Std_Error": bse[term],
            "z_statistic": zstat,
            "p_value": pval,
            "Odds_Ratio": or_val,
            "OR_95_CI_Lower": ci_lower,
            "OR_95_CI_Upper": ci_upper,
            "VIF": vif
        })
    dummy_df = pd.DataFrame(dummy_records)
    reject_dummy, pvals_dummy_adj, _, _ = multipletests(dummy_df["p_value"], alpha=0.05, method="holm")
    dummy_df["Holm_p_value"] = pvals_dummy_adj
    dummy_df["Reject_Holm"] = reject_dummy
    dummy_df.to_csv(RESULTS_DIR / "multivariable_associations.csv", index=False)
    print("Saved multivariable_associations.csv (term-level coefficients).")

    # 2. Feature-level nested Likelihood-Ratio tests (delta deviance)
    print("Performing feature-level nested Likelihood-Ratio tests...")
    lr_results = []
    for orig_col, dummy_list in col_map.items():
        X_reduced = X_full_const.drop(columns=dummy_list)
        red_model = sm.Logit(y, X_reduced).fit(disp=False)
        ll_red = red_model.llf
        lr_stat = 2.0 * (ll_full - ll_red)
        df_diff = len(dummy_list)
        p_val = stats.chi2.sf(lr_stat, df_diff)
        
        if len(dummy_list) == 1:
            term = dummy_list[0]
            rep_or = np.exp(params[term])
            rep_ci_lower = np.exp(conf.loc[term, 0])
            rep_ci_upper = np.exp(conf.loc[term, 1])
            rep_vif = vif_dict[term]
        else:
            sub_df = dummy_df[dummy_df["Variable"] == orig_col]
            max_row = sub_df.loc[sub_df["Odds_Ratio"].idxmax()]
            rep_or = max_row["Odds_Ratio"]
            rep_ci_lower = max_row["OR_95_CI_Lower"]
            rep_ci_upper = max_row["OR_95_CI_Upper"]
            rep_vif = sub_df["VIF"].max()

        lr_results.append({
            "Variable": orig_col,
            "Description": LABEL_MAPPING.get(orig_col, orig_col),
            "LR_Chi2": lr_stat,
            "df": df_diff,
            "p_value": p_val,
            "Odds_Ratio": rep_or,
            "OR_95_CI_Lower": rep_ci_lower,
            "OR_95_CI_Upper": rep_ci_upper,
            "VIF": rep_vif
        })

    feature_contrib_df = pd.DataFrame(lr_results).sort_values(by="LR_Chi2", ascending=False).reset_index(drop=True)
    feature_contrib_df["Rank"] = range(1, len(feature_contrib_df) + 1)
    reject_feat, pvals_feat_adj, _, _ = multipletests(feature_contrib_df["p_value"], alpha=0.05, method="holm")
    feature_contrib_df["Holm_p_value"] = pvals_feat_adj
    feature_contrib_df["Reject_Holm"] = reject_feat

    cols_order = ["Rank", "Variable", "Description", "LR_Chi2", "df", "p_value", "Holm_p_value", "Reject_Holm"]
    feature_contrib_df[cols_order].to_csv(RESULTS_DIR / "adjusted_feature_contributions.csv", index=False)
    print("Saved adjusted_feature_contributions.csv (feature-level LR Chi2).")

    full_adj_order = ["Rank", "Variable", "Description", "LR_Chi2", "df", "p_value", "Holm_p_value", "Reject_Holm", "Odds_Ratio", "OR_95_CI_Lower", "OR_95_CI_Upper", "VIF"]
    feature_contrib_df[full_adj_order].to_csv(RESULTS_DIR / "adjusted_association.csv", index=False)
    print("Saved adjusted_association.csv.")

    return feature_contrib_df

def apply_holm_bonferroni_corrections(cat_df: pd.DataFrame, num_df: pd.DataFrame):
    """Applies Holm-Bonferroni p-value adjustment across prespecified primary association tests (Chi-square and Mann-Whitney U)."""
    all_pvals = list(cat_df["p-value"].values) + list(num_df["MWU p-value"].values)
    reject, pvals_corrected, _, _ = multipletests(all_pvals, alpha=0.05, method="holm")
    
    n_cat = len(cat_df)
    cat_df["Holm_p_value"] = pvals_corrected[:n_cat]
    cat_df["Reject_Holm"] = reject[:n_cat]
    
    num_df["Holm_p_value"] = pvals_corrected[n_cat:]
    num_df["Reject_Holm"] = reject[n_cat:]
    
    cat_df.to_csv(RESULTS_DIR / "chi_square_results.csv", index=False)
    num_df.to_csv(RESULTS_DIR / "numerical_results.csv", index=False)
    print("Saved chi_square_results.csv and numerical_results.csv with Holm-adjusted p-values.")

def generate_visualizations(df: pd.DataFrame, cat_results: pd.DataFrame, num_results: pd.DataFrame):
    """Generates and saves scientific visualizations of statistical findings."""
    print("Generating statistical visualizations...")
    sns.set_theme(style="whitegrid")
    
    # 1. Generate Two-Panel Effect Size Figure via generate_effect_size_figure module
    from generate_effect_size_figure import generate_effect_size_figure
    generate_effect_size_figure()

    # 2. Top Categorical Prevalence Bar Chart
    top_factors = cat_results.head(4)["Variable"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    
    for i, col in enumerate(top_factors):
        contingency = pd.crosstab(df[col], df["Diabetes_binary"])
        rates = contingency.div(contingency.sum(axis=1), axis=0)[1] * 100
        ax = axes[i]
        
        x_labels = [str(int(x)) for x in rates.index]
        if col == "HighBP":
            x_labels = ["No High BP (0)", "High BP (1)"]
        elif col == "HighChol":
            x_labels = ["No High Chol (0)", "High Chol (1)"]
        elif col == "DiffWalk":
            x_labels = ["No Difficulty (0)", "Has Difficulty (1)"]
        elif col == "GenHlth":
            x_labels = ["1: Excellent", "2: Very Good", "3: Good", "4: Fair", "5: Poor"]
            
        bars = ax.bar(x_labels, rates.values, color="#1D4ED8", alpha=0.85, width=0.45)
        ax.set_title(f"Diabetes Prevalence (%) by {LABEL_MAPPING.get(col, col)}", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Prevalence (%)", fontsize=10)
        ax.set_ylim(0, max(rates.values) * 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E293B")
            
    plt.suptitle("Diabetes Prevalence Across Key Risk Indicator Subgroups", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_categorical_prevalence.png", dpi=300)
    plt.close()
    
    # 3. BMI Boxplot
    plt.figure(figsize=(8, 5))
    plot_df = df[df["BMI"] <= 60].copy()
    sns.boxplot(
        data=plot_df,
        x="Diabetes_binary",
        y="BMI",
        palette=["#3B82F6", "#EF4444"],
        hue="Diabetes_binary",
        legend=False,
        width=0.35
    )
    plt.title("Body Mass Index (BMI) Distribution vs Diabetes Status", fontsize=12, fontweight="bold", pad=15)
    plt.xticks([0, 1], ["0: No reported diabetes", "1: Prediabetes/diabetes positive class"])
    plt.xlabel("")
    plt.ylabel("Body Mass Index (BMI)", fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bmi_boxplot.png", dpi=300)
    plt.close()
    
    # 4. Physical & Mental Unhealthy Days Plot
    df_melt = pd.melt(df, id_vars=["Diabetes_binary"], value_vars=["PhysHlth", "MentHlth"], var_name="Metric", value_name="Days")
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Metric", y="Days", hue="Diabetes_binary", data=df_melt, palette=["#2563EB", "#D97706"], errorbar=None)
    plt.title("Comparison of Unhealthy Days (Past 30 Days)", fontsize=12, fontweight="bold", pad=15)
    plt.xticks([0, 1], ["Physical Unhealthy Days (PhysHlth)", "Mental Unhealthy Days (MentHlth)"])
    plt.ylabel("Mean Unhealthy Days", fontsize=11)
    plt.legend(title="Class", labels=["0: No reported diabetes", "1: Prediabetes/diabetes positive class"])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "health_days_comparison.png", dpi=300)
    if (DOCS_DIR / "figures").exists():
        plt.savefig(DOCS_DIR / "figures/health_days_comparison.png", dpi=300)
    plt.close()
    print("Saved all diagnostic statistical plots successfully.")

def write_academic_report(df: pd.DataFrame, cat_results: pd.DataFrame, num_results: pd.DataFrame, adj_results: pd.DataFrame):
    """Generates docs/statistical_analysis.md."""
    print("Writing academic documentation to docs/statistical_analysis.md...")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    top_cat = cat_results.iloc[0]["Variable"]
    top_cat_desc = cat_results.iloc[0]["Description"]
    top_cat_v = cat_results.iloc[0]["Cramér's V"]
    top_cat_diff = cat_results.iloc[0]["Max Difference (%)"]
    
    bmi_class_1_mean = num_results.loc[num_results["Variable"] == "BMI", "Prediabetes/Diabetes Positive Mean"].values[0]
    bmi_class_0_mean = num_results.loc[num_results["Variable"] == "BMI", "No Reported Diabetes Mean"].values[0]
    bmi_cohen_d = num_results.loc[num_results["Variable"] == "BMI", "Cohen's d"].values[0]
    bmi_rb = num_results.loc[num_results["Variable"] == "BMI", "Rank-Biserial Correlation"].values[0]
    
    phys_class_1_mean = num_results.loc[num_results["Variable"] == "PhysHlth", "Prediabetes/Diabetes Positive Mean"].values[0]
    phys_class_0_mean = num_results.loc[num_results["Variable"] == "PhysHlth", "No Reported Diabetes Mean"].values[0]
    
    n_total = len(df)
    
    markdown_content = f"""# Statistical Hypothesis Testing & Adjusted Association Report
## CDC Diabetes Health Indicators (Cleaned Dataset)

### Methodology: CRISP-DM (Exploratory & Statistical Analysis)
**Author:** Senior Data Analytics Engineer & Team Members  
**Date:** 2026-07-21  
**Project:** Diabetes-Analytics  
**Objective:** Answer RQ1: *Which demographic, lifestyle, and health-related factors are statistically associated with diabetes status in the CDC BRFSS 2015 sample?*

---

## 1. Introduction
This report presents the quantitative statistical evaluation of the relationships between 21 health indicators and diabetes status (`Diabetes_binary`). The analysis is conducted on the full dataset of **{n_total:,}** survey respondents (retaining repeated feature profiles to preserve natural sample distribution).

To ensure statistical rigor, we apply:
1. **Chi-Square Test of Independence** for categorical, binary, and ordinal variables.
2. **Cramér's V** to measure effect size for categorical associations.
3. **Independent Two-Sample Welch t-Test** (parametric mean comparison) and **Mann-Whitney U Test** (non-parametric median/distribution comparison) for continuous numerical variables.
4. **Absolute Rank-Biserial Correlation** (primary) and **Cohen's d** (secondary) to measure numerical effect sizes.
5. **Holm–Bonferroni Multiple Testing Correction**: Holm adjustment was applied across the prespecified primary association tests: Chi-square tests for categorical features and Mann–Whitney U tests for numerical features. Welch’s t-tests were retained as complementary sensitivity analyses.
6. **Multivariable Adjusted Association Analysis**: Multivariable Logistic Regression with categorical indicator blocks for ordinal/multiclass predictors (`GenHlth`, `Age`, `Education`, `Income`), reporting term-level AORs, 95% Confidence Intervals, and feature-level nested Likelihood-Ratio tests ($\\Delta\\text{{deviance}}$ / LR $\\chi^2$) to assess conditional feature contributions while controlling for all co-occurring indicators.

> **Methodological Note on Large Sample Size:**  
> With *N* = {n_total:,}, statistical tests possess near-infinite power, causing p-values for almost all predictors to drop below $p < 0.05$. Therefore, p-values are reported alongside Holm-adjusted values for formal hypothesis testing, but **practical feature importance is evaluated by Effect Size within each feature family and feature-level Likelihood-Ratio $\\chi^2$ ($\\Delta\\text{{deviance}}$)**.

---

## 2. Categorical Variable Analysis (Chi-Square & Cramér's V)

### Contingency Table & Chi-Square Summary (with Holm-Bonferroni Correction)
| Variable Name | Attribute Description | Chi-Square ($\\chi^2$) | Raw p-value | Holm-Adjusted p | Reject $H_0$ | df | Cramér's V | Effect Size Interpretation | Max Prevalence Diff |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
"""
    for _, row in cat_results.iterrows():
        p_raw_str = f"{row['p-value']:.2e}" if row['p-value'] > 0 else "< 1.00e-300"
        p_holm_str = f"{row['Holm_p_value']:.2e}" if row['Holm_p_value'] > 0 else "< 1.00e-300"
        cramers_val = row["Cramér's V"]
        markdown_content += (
            f"| `{row['Variable']}` | {row['Description']} | {row['Chi2 Statistic']:.2f} | "
            f"`{p_raw_str}` | `{p_holm_str}` | {'Yes' if row['Reject_Holm'] else 'No'} | "
            f"{int(row['Degrees of Freedom'])} | {cramers_val:.4f} | "
            f"**{row['Effect Size Interpretation']}** | {row['Max Difference (%)']:.2f}% |\n"
        )

    markdown_content += f"""
### Key Findings from Categorical Analysis:
1. **Strongest Marginal Associated Feature**: **`{top_cat}`** ({top_cat_desc}) exhibits the strongest univariate association within the analyzed BRFSS sample with diabetes status (*V* = **{top_cat_v:.4f}**), showing a **{top_cat_diff:.2f}%** difference in prevalence across health levels.
2. **Survey-based Health Indicators**: General Health (`GenHlth`, *V* = {cat_results.loc[cat_results['Variable'] == 'GenHlth', "Cramér's V"].values[0]:.4f}), High Blood Pressure (`HighBP`, *V* = {cat_results.loc[cat_results['Variable'] == 'HighBP', "Cramér's V"].values[0]:.4f}), High Cholesterol (`HighChol`, *V* = {cat_results.loc[cat_results['Variable'] == 'HighChol', "Cramér's V"].values[0]:.4f}), and Difficulty Walking (`DiffWalk`, *V* = {cat_results.loc[cat_results['Variable'] == 'DiffWalk', "Cramér's V"].values[0]:.4f}) represent the most salient marginal indicators.
3. **Behavioral Features**: Physical activity (`PhysActivity`, *V* = {cat_results.loc[cat_results['Variable'] == 'PhysActivity', "Cramér's V"].values[0]:.4f}) and fruit/vegetable intake show weak direct correlations (*V* < 0.10).
4. **Demographics**: Biological sex (`Sex`, *V* = {cat_results.loc[cat_results['Variable'] == 'Sex', "Cramér's V"].values[0]:.4f}) exhibits minimal marginal association with diabetes prevalence.

---

## 3. Numerical Variable Analysis (t-Test & Mann-Whitney U)

### Numerical Tests Summary
| Variable | No Reported Diabetes Mean | Prediabetes/Diabetes Positive Mean | Mean Diff | No Reported Diabetes Median | Prediabetes/Diabetes Median | t-Stat | Raw t p-val | MWU p-val | Holm MWU p | Reject $H_0$ | Cohen's d | Abs Rank-Biserial | Effect Size Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for _, row in num_results.iterrows():
        t_p_str = f"{row['t p-value']:.2e}" if row['t p-value'] > 0 else "< 1.00e-300"
        mwu_p_str = f"{row['MWU p-value']:.2e}" if row['MWU p-value'] > 0 else "< 1.00e-300"
        holm_p_str = f"{row['Holm_p_value']:.2e}" if row['Holm_p_value'] > 0 else "< 1.00e-300"
        markdown_content += (
            f"| `{row['Variable']}` | {row['No Reported Diabetes Mean']:.2f} | {row['Prediabetes/Diabetes Positive Mean']:.2f} | "
            f"{row['Mean Difference']:.2f} | {row['No Reported Diabetes Median']:.1f} | {row['Prediabetes/Diabetes Positive Median']:.1f} | "
            f"{row['t-Statistic']:.2f} | `{t_p_str}` | `{mwu_p_str}` | `{holm_p_str}` | "
            f"{'Yes' if row['Reject_Holm'] else 'No'} | {row['Cohen\'s d']:.4f} | "
            f"{row['Effect_Size']:.4f} | **{row['Effect Size Interpretation']}** |\n"
        )
        
    markdown_content += f"""
### Key Findings from Numerical Analysis:
1. **Body Mass Index (BMI)**: Mean BMI for the group without reported diabetes is **{bmi_class_0_mean:.2f}** vs **{bmi_class_1_mean:.2f}** for the prediabetes/diabetes positive group. Absolute Rank-Biserial correlation is **{abs(bmi_rb):.4f}** (Cohen's d = **{bmi_cohen_d:.4f}**), confirming a moderate practical effect size within the sample.
2. **Physical Unhealthy Days (`PhysHlth`)**: Respondents in the prediabetes/diabetes positive class report an average of **{phys_class_1_mean:.2f}** unhealthy physical days in the past 30 days compared to **{phys_class_0_mean:.2f}** days for respondents without reported diabetes.

---

## 4. Multivariable Adjusted Association Analysis (Logistic Regression)

To evaluate adjusted associations while controlling for co-occurring indicators, a multivariable logistic regression model was estimated. Categorical and ordinal features (`GenHlth`, `Age`, `Education`, `Income`) were modeled using categorical indicator blocks with the lowest category as reference. Feature-level statistical contributions were evaluated via nested Likelihood-Ratio $\\chi^2$ tests ($\\Delta\\text{{deviance}}$).

### Feature-Level Likelihood-Ratio Contributions (Nested Model Comparison)
| Rank | Variable Name | Description | LR $\\chi^2$ ($\\Delta$deviance) | df | Raw p-value | Holm-Adjusted p | Reject $H_0$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in adj_results.iterrows():
        p_raw_str = f"{row['p_value']:.2e}" if row['p_value'] > 0 else "< 1.00e-300"
        p_holm_str = f"{row['Holm_p_value']:.2e}" if row['Holm_p_value'] > 0 else "< 1.00e-300"
        markdown_content += (
            f"| {int(row['Rank'])} | `{row['Variable']}` | {row['Description']} | {row['LR_Chi2']:.2f} | "
            f"{int(row['df'])} | `{p_raw_str}` | `{p_holm_str}` | {'Yes' if row['Reject_Holm'] else 'No'} |\n"
        )

    markdown_content += f"""
### Key Findings from Multivariable Analysis:
1. **Top Feature Contributions by $\\Delta\\text{{deviance}}$**: `GenHlth` (LR $\\chi^2$ = {adj_results.loc[adj_results['Variable']=='GenHlth', 'LR_Chi2'].values[0]:.2f}, df = 4), `BMI` (LR $\\chi^2$ = {adj_results.loc[adj_results['Variable']=='BMI', 'LR_Chi2'].values[0]:.2f}, df = 1), `Age` (LR $\\chi^2$ = {adj_results.loc[adj_results['Variable']=='Age', 'LR_Chi2'].values[0]:.2f}, df = 12), `HighBP` (LR $\\chi^2$ = {adj_results.loc[adj_results['Variable']=='HighBP', 'LR_Chi2'].values[0]:.2f}, df = 1), and `HighChol` (LR $\\chi^2$ = {adj_results.loc[adj_results['Variable']=='HighChol', 'LR_Chi2'].values[0]:.2f}, df = 1) contribute the largest likelihood gains to the multivariable model.
2. **Category-Level Odds Ratios**: Full term-level adjusted odds ratios and confidence intervals across all dummy categories are archived in `results/statistical_analysis/multivariable_associations.csv`.

---

## 5. Visualizations and Diagnostics
Saved under `results/statistical_analysis/` and `docs/figures/`:
* **Effect Size Ranking**: [effect_size_ranking.png](figures/effect_size_ranking.png) — Two-panel figure displaying separate effect-size rankings: Cramér's V for categorical features (Panel A) and Absolute Rank-Biserial correlation for numerical features (Panel B).
* **Subgroup Prevalence**: [top_categorical_prevalence.png](figures/top_categorical_prevalence.png) — Prediabetes/diabetes positive rate by key risk factors.
* **BMI Distribution Boxplot**: [bmi_boxplot.png](figures/bmi_boxplot.png) — BMI range comparison across target classes.
* **Unhealthy Days Comparison**: [health_days_comparison.png](figures/health_days_comparison.png) — Mental and physical unhealthy day comparisons.

---

## 6. Conclusions for Research Question 1 (RQ1)
1. **Primary Associated Features**: General Health (`GenHlth`), Body Mass Index (`BMI`), Age (`Age`), High Blood Pressure (`HighBP`), and High Cholesterol (`HighChol`) demonstrate the highest multivariable likelihood contributions and sample-level effect sizes.
2. **Multiple Testing Control**: All key relationships remain statistically significant after Holm–Bonferroni correction, but feature prioritization is governed by effect size and likelihood contribution rather than raw p-value magnitudes.
"""
    
    output_path = DOCS_DIR / "statistical_analysis.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Generated academic report at: {output_path}")

def main():
    print("=== Phase 2: Statistical Hypothesis Testing ===")
    
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return
        
    cat_results = perform_categorical_tests(df)
    num_results = perform_numerical_tests(df)
    
    apply_holm_bonferroni_corrections(cat_results, num_results)
    adj_results = perform_adjusted_association_tests(df)
    
    cat_results = pd.read_csv(RESULTS_DIR / "chi_square_results.csv")
    num_results = pd.read_csv(RESULTS_DIR / "numerical_results.csv")
    
    generate_visualizations(df, cat_results, num_results)
    write_academic_report(df, cat_results, num_results, adj_results)
    print("=== Statistical Analysis Module Completed Successfully ===")

if __name__ == "__main__":
    main()
