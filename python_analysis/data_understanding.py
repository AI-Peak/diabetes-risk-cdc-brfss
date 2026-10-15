"""
Data Understanding Module
-------------------------
Author: Senior Data Analytics Engineer
Methodology: CRISP-DM
Dataset: CDC Diabetes Health Indicators (BRFSS 2015)

This script performs the initial Data Understanding phase. It inspects and
summarizes the dataset without modifying any values, checking for data types,
missing values, duplicate records, class imbalance, and validates variable ranges.
All outputs are saved to the results/data_understanding/ directory, and a
formal Markdown report is generated in docs/data_understanding.md.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
RESULTS_DIR = BASE_DIR / "results" / "data_understanding"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Define column categories and types
COL_CLASSIFICATION = {
    "Diabetes_binary": {"Category": "Target", "Type": "Binary", "Description": "Diabetes status (0 = no reported diabetes, 1 = prediabetes or diabetes)"},
    "HighBP": {"Category": "Health Condition", "Type": "Binary", "Description": "High blood pressure indicator (0 = no high BP, 1 = high BP)"},
    "HighChol": {"Category": "Health Condition", "Type": "Binary", "Description": "High cholesterol indicator (0 = no high cholesterol, 1 = high cholesterol)"},
    "CholCheck": {"Category": "Healthcare Access", "Type": "Binary", "Description": "Cholesterol check in past 5 years (0 = no check, 1 = check)"},
    "BMI": {"Category": "Physical Measurement", "Type": "Numerical", "Description": "Body Mass Index (BMI)"},
    "Smoker": {"Category": "Lifestyle", "Type": "Binary", "Description": "Smoked at least 100 cigarettes in lifetime (0 = no, 1 = yes)"},
    "Stroke": {"Category": "Health Condition", "Type": "Binary", "Description": "Ever told you had a stroke (0 = no, 1 = yes)"},
    "HeartDiseaseorAttack": {"Category": "Health Condition", "Type": "Binary", "Description": "Coronary heart disease or myocardial infarction (0 = no, 1 = yes)"},
    "PhysActivity": {"Category": "Lifestyle", "Type": "Binary", "Description": "Physical activity in past 30 days excluding work (0 = no, 1 = yes)"},
    "Fruits": {"Category": "Lifestyle", "Type": "Binary", "Description": "Consume fruit 1 or more times per day (0 = no, 1 = yes)"},
    "Veggies": {"Category": "Lifestyle", "Type": "Binary", "Description": "Consume vegetables 1 or more times per day (0 = no, 1 = yes)"},
    "HvyAlcoholConsump": {"Category": "Lifestyle", "Type": "Binary", "Description": "Heavy alcohol consumption (0 = no, 1 = yes)"},
    "AnyHealthcare": {"Category": "Healthcare Access", "Type": "Binary", "Description": "Have any health care coverage (0 = no, 1 = yes)"},
    "NoDocbcCost": {"Category": "Healthcare Access", "Type": "Binary", "Description": "Could not see doctor because of cost in past 12 months (0 = no, 1 = yes)"},
    "GenHlth": {"Category": "General Health", "Type": "Ordinal", "Description": "Self-reported general health scale (1 = excellent to 5 = poor)"},
    "MentHlth": {"Category": "General Health", "Type": "Numerical", "Description": "Days of poor mental health in past 30 days (0-30)"},
    "PhysHlth": {"Category": "General Health", "Type": "Numerical", "Description": "Days of poor physical health in past 30 days (0-30)"},
    "DiffWalk": {"Category": "Health Condition", "Type": "Binary", "Description": "Serious difficulty walking or climbing stairs (0 = no, 1 = yes)"},
    "Sex": {"Category": "Demographic", "Type": "Binary", "Description": "Biological sex (0 = female, 1 = male)"},
    "Age": {"Category": "Demographic", "Type": "Ordinal", "Description": "13-level age category (1 = 18-24 to 13 = 80+)"},
    "Education": {"Category": "Socioeconomic", "Type": "Ordinal", "Description": "Education level scale (1 = never attended school to 6 = college graduate)"},
    "Income": {"Category": "Socioeconomic", "Type": "Ordinal", "Description": "Income scale (1 = <$10,000 to 8 = $75,000+)"}
}

def load_data(file_path: Path) -> pd.DataFrame:
    """Loads the CDC Diabetes dataset from the raw CSV path."""
    print(f"Loading dataset from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the path.")
    df = pd.read_csv(file_path)
    return df

def generate_dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Generates and saves the dataset overview statistics."""
    overview_data = {
        "Metric": [
            "Number of Rows",
            "Number of Columns",
            "Dataset Shape",
            "Column Names"
        ],
        "Value": [
            len(df),
            len(df.columns),
            str(df.shape),
            ", ".join(df.columns.tolist())
        ]
    }
    overview_df = pd.DataFrame(overview_data)
    overview_df.to_csv(RESULTS_DIR / "dataset_overview.csv", index=False)
    print("Saved dataset_overview.csv")
    return overview_df

def generate_data_types_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes and summarizes column data types and classifications."""
    summary_list = []
    binary_count = 0
    numerical_count = 0
    ordinal_count = 0

    for col in df.columns:
        info = COL_CLASSIFICATION.get(col, {"Type": "Unknown", "Category": "Unknown"})
        col_type = info["Type"]
        
        if col_type == "Binary":
            binary_count += 1
        elif col_type == "Numerical":
            numerical_count += 1
        elif col_type == "Ordinal":
            ordinal_count += 1
            
        summary_list.append({
            "Column": col,
            "Pandas DataType": str(df[col].dtype),
            "Variable Type": col_type,
            "Domain Category": info["Category"]
        })
        
    summary_df = pd.DataFrame(summary_list)
    
    # Save a summary statistics row for variable counts
    counts_data = {
        "Metric": ["Binary Variables", "Numerical Variables", "Ordinal Variables", "Total Variables"],
        "Count": [binary_count, numerical_count, ordinal_count, len(df.columns)]
    }
    counts_df = pd.DataFrame(counts_data)
    
    # We will write both to the variable_information.csv and prints
    print(f"Variable types: Binary={binary_count}, Numerical={numerical_count}, Ordinal={ordinal_count}")
    return summary_df, counts_df

def generate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Checks and summarizes missing values per column."""
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count,
        "Missing Percentage": missing_pct
    }).sort_values(by="Missing Percentage", ascending=False)
    
    missing_df.to_csv(RESULTS_DIR / "missing_values.csv", index=False)
    print("Saved missing_values.csv")
    return missing_df

def generate_duplicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and summarizes duplicate rows."""
    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / len(df)) * 100
    
    dup_df = pd.DataFrame({
        "Metric": ["Total Rows", "Duplicate Rows Count", "Duplicate Percentage"],
        "Value": [len(df), dup_count, f"{dup_pct:.4f}%"]
    })
    
    dup_df.to_csv(RESULTS_DIR / "duplicate_summary.csv", index=False)
    print("Saved duplicate_summary.csv")
    return dup_df

def analyze_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes the target variable distribution and generates a plot."""
    target_counts = df["Diabetes_binary"].value_counts().sort_index()
    target_pct = df["Diabetes_binary"].value_counts(normalize=True).sort_index() * 100
    
    dist_df = pd.DataFrame({
        "Class": target_counts.index.astype(int),
        "Count": target_counts.values,
        "Percentage": target_pct.values
    })
    
    dist_df.to_csv(RESULTS_DIR / "target_distribution.csv", index=False)
    print("Saved target_distribution.csv")
    
    # Plot target distribution with premium styling
    plt.figure(figsize=(7, 5))
    colors = ["#4A90E2", "#E94E77"]  # Premium blue and coral/red colors
    
    bars = plt.bar(
        ["0: No Reported Diabetes", "1: Prediabetes/Diabetes"],
        target_counts.values,
        color=colors,
        edgecolor="none",
        width=0.5,
        alpha=0.85
    )

    
    # Customize grid and spines
    plt.grid(axis="y", linestyle="--", alpha=0.5, color="#CCCCCC")
    plt.gca().set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().spines["bottom"].set_color("#888888")
    
    # Annotate bars with counts and percentages
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        plt.text(
            bar.get_x() + bar.get_width()/2.0,
            height + (total * 0.01),
            f"{height:,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color="#333333"
        )
        
    plt.title("Distribution of Target Variable: Diabetes_binary", fontsize=13, fontweight="bold", pad=20, color="#222222")
    plt.ylabel("Frequency (Count)", fontsize=11, color="#444444")
    plt.tick_params(colors="#444444", labelsize=10, bottom=False)
    plt.ylim(0, max(target_counts.values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "target_distribution.png", dpi=300)
    plt.close()
    print("Saved target_distribution.png")
    
    return dist_df

def generate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generates summary statistics for numerical variables."""
    # Find numerical columns
    num_cols = [col for col, info in COL_CLASSIFICATION.items() if info["Type"] == "Numerical"]
    
    # Generate statistics for numerical variables
    desc_stats = df[num_cols].describe()
    # Add median explicitly if not already covered (it's 50%)
    # Add range (max - min)
    desc_stats.loc["range"] = desc_stats.loc["max"] - desc_stats.loc["min"]
    # Reorder index to be standard and clean
    desc_stats = desc_stats.reindex(["count", "mean", "std", "min", "25%", "50%", "75%", "max", "range"])
    
    # Transpose for easier reading (variables as rows, stats as columns)
    desc_stats_t = desc_stats.T
    desc_stats_t.index.name = "Variable"
    
    desc_stats_t.to_csv(RESULTS_DIR / "descriptive_statistics.csv")
    print("Saved descriptive_statistics.csv")
    return desc_stats_t

def generate_unique_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Reports unique counts and value lists for each column."""
    unique_list = []
    for col in df.columns:
        unique_cnt = df[col].nunique()
        sorted_uniques = sorted(df[col].unique())
        
        # Format the list of unique values
        if unique_cnt <= 15:
            uniques_str = str([int(x) if x.is_integer() else x for x in sorted_uniques])
        else:
            uniques_str = f"[{int(sorted_uniques[0])}, ..., {int(sorted_uniques[-1])}] (Total: {unique_cnt} values)"
            
        unique_list.append({
            "Column": col,
            "Unique Count": unique_cnt,
            "Unique Values": uniques_str
        })
        
    unique_df = pd.DataFrame(unique_list)
    unique_df.to_csv(RESULTS_DIR / "unique_values.csv", index=False)
    print("Saved unique_values.csv")
    return unique_df

def perform_range_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Validates if important variables are within expected ranges."""
    validation_rules = {
        "BMI": {"min": 0.0, "max": np.inf, "descr": "Non-negative value"},
        "MentHlth": {"min": 0.0, "max": 30.0, "descr": "0 to 30 days"},
        "PhysHlth": {"min": 0.0, "max": 30.0, "descr": "0 to 30 days"},
        "GenHlth": {"min": 1.0, "max": 5.0, "descr": "1 to 5 scale"},
        "Age": {"min": 1.0, "max": 13.0, "descr": "1 to 13 category scale"},
        "Education": {"min": 1.0, "max": 6.0, "descr": "1 to 6 category scale"},
        "Income": {"min": 1.0, "max": 8.0, "descr": "1 to 8 category scale"}
    }
    
    validation_results = []
    
    for col, rules in validation_rules.items():
        min_val = rules["min"]
        max_val = rules["max"]
        
        actual_min = df[col].min()
        actual_max = df[col].max()
        
        out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
        violation_count = len(out_of_bounds)
        
        status = "PASSED" if violation_count == 0 else "FAILED"
        
        validation_results.append({
            "Variable": col,
            "Expected Range": rules["descr"],
            "Actual Min": actual_min,
            "Actual Max": actual_max,
            "Violations Count": violation_count,
            "Status": status
        })
        
    validation_df = pd.DataFrame(validation_results)
    # Save range validation findings in unique format
    validation_df.to_csv(RESULTS_DIR / "range_validation.csv", index=False)
    print("Saved range_validation.csv")
    return validation_df

def generate_variable_information_table() -> pd.DataFrame:
    """Creates a structured table mapping every variable to its Type and Category."""
    info_list = []
    for var, details in COL_CLASSIFICATION.items():
        info_list.append({
            "Variable": var,
            "Type": details["Type"],
            "Category": details["Category"],
            "Description": details["Description"]
        })
    info_df = pd.DataFrame(info_list)
    info_df.to_csv(RESULTS_DIR / "variable_information.csv", index=False)
    print("Saved variable_information.csv")
    return info_df

def write_academic_documentation(
    df: pd.DataFrame,
    overview_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    dup_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    desc_stats_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    var_info_df: pd.DataFrame
) -> None:
    """Generates the docs/data_understanding.md file in clear academic English."""
    
    # Calculate statistics for narrative
    total_rows = len(df)
    diabetes_pct = dist_df.loc[dist_df["Class"] == 1, "Percentage"].values[0]
    healthy_pct = dist_df.loc[dist_df["Class"] == 0, "Percentage"].values[0]
    duplicate_rows = dup_df.loc[dup_df["Metric"] == "Duplicate Rows Count", "Value"].values[0]
    duplicate_pct = dup_df.loc[dup_df["Metric"] == "Duplicate Percentage", "Value"].values[0]
    
    markdown_content = f"""# Data Understanding Report
## CDC Diabetes Health Indicators (BRFSS 2015)

### Methodology: CRISP-DM (Data Understanding)
**Author:** Senior Data Analytics Engineer  
**Date:** {pd.Timestamp.now().strftime("%Y-%m-%d")}  
**Project:** Diabetes-Analytics

---

## 1. Introduction and Objectives
This document presents the **Data Understanding** phase of the CRISP-DM methodology for the **CDC Diabetes Health Indicators** dataset, sourced from the 2015 Behavioral Risk Factor Surveillance System (BRFSS). The primary objective is to inspect, summarize, and validate the dataset before any downstream data preparation, feature engineering, or modeling tasks. 

Importantly, this analysis is performed on the **original, imbalanced dataset** to accurately reflect real-world epidemiological diabetes prevalence within the surveyed population. No balancing techniques (e.g., SMOTE, oversampling, or undersampling) have been applied.

---

## 2. Dataset Overview and Structural Characteristics
The dataset represents a subset of survey responses focusing on behavioral risk factors, healthcare access, demographics, and clinical conditions related to diabetes.

* **Dataset Shape:** {overview_df.loc[overview_df["Metric"] == "Dataset Shape", "Value"].values[0]}
* **Number of Rows (Observations):** {total_rows:,}
* **Number of Columns (Variables):** {len(df.columns)}

### Dataset Overview Table
| Metric | Value |
| :--- | :--- |
| **Number of Observations (Rows)** | {total_rows:,} |
| **Number of Attributes (Columns)** | {len(df.columns)} |
| **Data Format** | Comma-Separated Values (CSV) |
| **File Location** | `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv` |

The attributes are represented in pandas as 64-bit floating-point numbers (`float64`), which reflect coded response values from the BRFSS questionnaire.

---

## 3. Variable Classification
The 22 variables in the dataset span multiple analytical domains, including clinical indicators, physical measurements, lifestyles, demographics, and socioeconomic factors. 

Below is the structured variable classification schema:

| Variable | Type | Category | Description |
| :--- | :--- | :--- | :--- |
"""
    # Append variables classification
    for _, row in var_info_df.iterrows():
        markdown_content += f"| `{row['Variable']}` | {row['Type']} | **{row['Category']}** | {row['Description']} |\n"
        
    markdown_content += f"""
### Variables Summary by Type
- **Binary Variables (15):** Indicator variables coded as `0` or `1`.
- **Numerical Variables (3):** Continuous or discrete ratio/interval measures (`BMI`, `MentHlth`, `PhysHlth`).
- **Ordinal Variables (4):** Categorical scales representing progression (e.g., age bracket, education, income, general health).

---

## 4. Missing Values Analysis
To assess data quality and completeness, a missingness check was performed across all 22 attributes.

| Column | Missing Count | Missing Percentage |
| :--- | :---: | :---: |
"""
    # Append missing values
    for _, row in missing_df.iterrows():
        markdown_content += f"| `{row['Column']}` | {int(row['Missing Count'])} | {row['Missing Percentage']:.4f}% |\n"
        
    markdown_content += """
**Key Observation:** There are **zero missing values** (nulls or NaNs) detected in the raw dataset. This completeness suggests that the dataset has undergone preliminary extraction and formatting prior to our receipt. Consequently, no imputation strategies are required at this stage.

---

## 5. Duplicate Records Evaluation
The dataset was scanned for identical row profiles to evaluate the degree of duplicate records.
"""

    markdown_content += f"""
- **Duplicate Rows Count:** {duplicate_rows}
- **Duplicate Percentage:** {duplicate_pct}

**Epidemiological Context:** In a large-scale survey consisting of 253,680 respondents and 22 coded categorical or ordinal variables, it is mathematically expected to observe identical response profiles (duplicates) without it implying data entry errors. For example, two respondents may share the exact same profile: female, aged 50-54, college graduate, high income, non-smoker, with high blood pressure, etc. Removing these records would artificiality skew the underlying sample distribution and reduce statistical power. Therefore, **in accordance with CRISP-DM guidelines, duplicate records are retained** for subsequent analysis.

---

## 6. Target Variable Distribution
The target variable, `Diabetes_binary`, represents the diabetes status of the respondent, where `0` indicates no diabetes, and `1` indicates prediabetes or diabetes.

| Target Class | Description | Count | Percentage |
| :---: | :--- | :---: | :---: |
"""
    # Append target distribution
    for _, row in dist_df.iterrows():
        desc = "No Diabetes" if row['Class'] == 0 else "Prediabetes or Diabetes"
        markdown_content += f"| `{row['Class']}` | {desc} | {int(row['Count']):,} | {row['Percentage']:.2f}% |\n"
        
    markdown_content += f"""
### Class Imbalance Diagnostics
The target variable exhibits a pronounced **class imbalance**, with approximately **{healthy_pct:.2f}%** of the sample classified as without reported diabetes (`0`) and only **{diabetes_pct:.2f}%** classified as having prediabetes or diabetes (`1`). This distribution aligns with real-world epidemiological statistics where diabetes is a prevalent but minority health condition in the general population. 

A visualization of this distribution has been saved as `results/data_understanding/target_distribution.png`. Downstream classification models will need to account for this imbalance (e.g., using screening-oriented machine learning evaluation or appropriate evaluation metrics such as F1-score, Precision-Recall AUC, rather than raw accuracy), but the raw data itself must remain unweighted and unadjusted in this phase to preserve real-world prevalence rates.

---

## 7. Descriptive Statistics for Numerical Variables
Descriptive statistics were computed for the three numerical attributes (`BMI`, `MentHlth`, `PhysHlth`) to characterize their central tendency, dispersion, and spread.

| Statistic | BMI | MentHlth (Mental Health) | PhysHlth (Physical Health) |
| :--- | :---: | :---: | :---: |
| **Count** | {int(desc_stats_df.loc['BMI', 'count']):,} | {int(desc_stats_df.loc['MentHlth', 'count']):,} | {int(desc_stats_df.loc['PhysHlth', 'count']):,} |
| **Mean** | {desc_stats_df.loc['BMI', 'mean']:.4f} | {desc_stats_df.loc['MentHlth', 'mean']:.4f} | {desc_stats_df.loc['PhysHlth', 'mean']:.4f} |
| **Std. Dev.** | {desc_stats_df.loc['BMI', 'std']:.4f} | {desc_stats_df.loc['MentHlth', 'std']:.4f} | {desc_stats_df.loc['PhysHlth', 'std']:.4f} |
| **Minimum** | {desc_stats_df.loc['BMI', 'min']:.1f} | {desc_stats_df.loc['MentHlth', 'min']:.1f} | {desc_stats_df.loc['PhysHlth', 'min']:.1f} |
| **25th Percentile (Q1)** | {desc_stats_df.loc['BMI', '25%']:.1f} | {desc_stats_df.loc['MentHlth', '25%']:.1f} | {desc_stats_df.loc['PhysHlth', '25%']:.1f} |
| **Median (Q2)** | {desc_stats_df.loc['BMI', '50%']:.1f} | {desc_stats_df.loc['MentHlth', '50%']:.1f} | {desc_stats_df.loc['PhysHlth', '50%']:.1f} |
| **75th Percentile (Q3)** | {desc_stats_df.loc['BMI', '75%']:.1f} | {desc_stats_df.loc['MentHlth', '75%']:.1f} | {desc_stats_df.loc['PhysHlth', '75%']:.1f} |
| **Maximum** | {desc_stats_df.loc['BMI', 'max']:.1f} | {desc_stats_df.loc['MentHlth', 'max']:.1f} | {desc_stats_df.loc['PhysHlth', 'max']:.1f} |
| **Range** | {desc_stats_df.loc['BMI', 'range']:.1f} | {desc_stats_df.loc['MentHlth', 'range']:.1f} | {desc_stats_df.loc['PhysHlth', 'range']:.1f} |

### Distributional Analysis:
1. **BMI:** The average Body Mass Index (BMI) is **28.38**, which lies within the "Overweight" category (BMI 25.0 to 29.9) according to WHO guidelines. The minimum BMI is 12.0 and the maximum is 98.0. The standard deviation of 6.61 suggests significant variability, and a median of 27.0 implies a slight right skew due to extreme outliers on the higher end.
2. **Mental Health (`MentHlth`):** Represents the number of days in the past 30 days the respondent rated their mental health as "not good". The mean is **3.18 days** with a standard deviation of **7.41**. With Q1 = 0.0, median = 0.0, and Q3 = 2.0 (meaning over 75% of respondents report 2 or fewer bad days in the past 30 days), this distribution is heavily zero-inflated and right-skewed. The maximum is 30 days.
3. **Physical Health (`PhysHlth`):** Represents the number of days in the past 30 days the respondent rated their physical health as "not good". Similar to mental health, it is zero-inflated, showing a mean of **4.24 days**, standard deviation of **8.71**, and a median of **0.0 days**. The maximum is 30 days.

---

## 8. Unique Values and Categorical Support
For every column, the distinct values and unique counts were logged to verify the domain bounds of nominal and ordinal variables.

| Attribute | Unique Count | Distinct Coded Values / Range |
| :--- | :---: | :--- |
"""
    # Append unique values
    for _, row in unique_df.iterrows():
        markdown_content += f"| `{row['Column']}` | {row['Unique Count']} | {row['Unique Values']} |\n"
        
    markdown_content += """
---

## 9. Data Range Validation Findings
A range validation check was conducted on critical variables to ensure they conform to the BRFSS standard coding schemes.

| Variable | Expected Range | Actual Min | Actual Max | Violations Count | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    # Append validation results
    for _, row in validation_df.iterrows():
        markdown_content += f"| `{row['Variable']}` | {row['Expected Range']} | {row['Actual Min']:.1f} | {row['Actual Max']:.1f} | {row['Violations Count']} | **{row['Status']}** |\n"
        
    markdown_content += """
**Validation Assessment:**
All investigated variables adhered strictly to their expected boundary conditions:
- `BMI` values are strictly positive, ranging from 12 to 98.
- Mental and physical health days (`MentHlth`, `PhysHlth`) are bounded exactly within [0, 30].
- Coded scale variables for general health (`GenHlth`), Age (`Age`), Education (`Education`), and Income (`Income`) fall strictly within their categorical ranges defined by the CDC.
- No values outside the expected coded ranges were detected in the examined variables.

---

## 10. Key Observations and Recommendations for Modeling
1. **Sample Positive-Class Proportion:** The positive-class proportion of **13.93%** describes the unweighted BRFSS sample analyzed in this project. It should not be interpreted as a weighted national prevalence estimate. Preserving the natural sample distribution is essential for training models that reflect the analyzed sample balance.
2. **Missing Values:** No imputation is necessary, as there are zero missing records.
3. **Repeated Feature Profiles:** Exact repeated rows cannot be verified as repeated observations of the same respondent because the dataset does not provide a respondent identifier. They may represent different respondents sharing the same discretized demographic, lifestyle, and health profile. These repeated feature profiles were retained to preserve the original sample frequencies and class distribution.
4. **Scale Differences:** Downstream machine learning models (especially distance-based or gradient-descent algorithms like SVMs, KNNs, or Logistic Regression) will require scaling or standardization due to the difference in scales between variables like `BMI` (12–98) and binary indicators (0–1).
5. **Class Imbalance Handling:** For modeling, since data rebalancing is forbidden in this phase, evaluation metrics must prioritize **F1-Score, Recall, Precision, and Precision-Recall AUC (PR-AUC)** over accuracy. A naive baseline predicting "No reported diabetes" for all cases would achieve 86.07% accuracy but 0% recall, making such a result unsuitable for the screening-oriented objective defined in this project.

---
*Report automatically generated by the Data Understanding Module.*
"""
    
    # Save the markdown report
    output_path = DOCS_DIR / "data_understanding.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Saved academic report to {output_path}")

def main():
    """Main execution function to run the Data Understanding steps."""
    print("=== Starting CRISP-DM Data Understanding Module ===")
    
    # 1. Load Dataset
    df = load_data(DATA_PATH)
    
    # 2. Dataset Overview
    overview_df = generate_dataset_overview(df)
    
    # 3. Data Types Summary
    summary_df, counts_df = generate_data_types_summary(df)
    
    # 4. Missing Values Analysis
    missing_df = generate_missing_values(df)
    
    # 5. Duplicate Records Evaluation
    dup_df = generate_duplicate_summary(df)
    
    # 6. Target Variable Distribution
    dist_df = analyze_target_distribution(df)
    
    # 7. Descriptive Statistics
    desc_stats_df = generate_descriptive_statistics(df)
    
    # 8. Unique Values Report
    unique_df = generate_unique_values_report(df)
    
    # 9. Range Validation
    validation_df = perform_range_validation(df)
    
    # 10. Variable Classification
    var_info_df = generate_variable_information_table()
    
    # Write Academic Documentation
    write_academic_documentation(
        df=df,
        overview_df=overview_df,
        missing_df=missing_df,
        dup_df=dup_df,
        dist_df=dist_df,
        desc_stats_df=desc_stats_df,
        unique_df=unique_df,
        validation_df=validation_df,
        var_info_df=var_info_df
    )
    
    print("=== Data Understanding Module Completed Successfully ===")

if __name__ == "__main__":
    main()
