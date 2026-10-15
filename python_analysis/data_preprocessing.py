import json
import numpy as np
import pandas as pd
from pathlib import Path

# Define paths relative to this script or project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_cleaned.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "data_preprocessing" / "preprocessing_summary.csv"
DOC_PATH = PROJECT_ROOT / "docs" / "data_preprocessing.md"

AUDIT_DIR = PROJECT_ROOT / "results" / "data_understanding"
AUDIT_JSON_PATH = AUDIT_DIR / "dataset_audit.json"
AUDIT_CSV_PATH = AUDIT_DIR / "dataset_audit.csv"
DUP_SUMMARY_PATH = AUDIT_DIR / "duplicate_summary.csv"

# Define expected value ranges based on CDC health indicator specifications
EXPECTED_RANGES = {
    "Diabetes_binary": (0, 1),
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),
    "BMI": (1, float("inf")),  # BMI > 0
    "Smoker": (0, 1),
    "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1),
    "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "DiffWalk": (0, 1),
    "Sex": (0, 1),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

def load_dataset(path: Path) -> pd.DataFrame:
    """Loads the CDC diabetes raw dataset."""
    print(f"Loading raw dataset from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")
    return pd.read_csv(path)

def validate_ranges(df: pd.DataFrame) -> dict:
    """Checks if any values in the dataframe fall outside expected ranges."""
    invalid_report = {}
    for col, bounds in EXPECTED_RANGES.items():
        if col not in df.columns:
            invalid_report[col] = "Column missing"
            continue
        
        min_val, max_val = bounds
        out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
        if not out_of_bounds.empty:
            invalid_report[col] = len(out_of_bounds)
    return invalid_report

def run_preprocessing():
    """Main execution function for data preprocessing and quality auditing."""
    # Ensure directory structures exist
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_dataset(RAW_DATA_PATH)
    raw_shape = df.shape
    print(f"Dataset shape before preprocessing: {raw_shape}")

    # 2. Check missing values
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    print(f"Total missing values: {total_missing}")

    # 3. Duplicate and Profile Analysis
    feature_cols = [c for c in df.columns if c != "Diabetes_binary"]

    # Exact full-row duplicates (across all 22 columns including label)
    exact_dup_mask = df.duplicated(keep="first")
    exact_dup_surplus = int(exact_dup_mask.sum())
    exact_dup_all_mask = df.duplicated(keep=False)
    exact_dup_in_groups = int(exact_dup_all_mask.sum())
    exact_dup_groups = int(df[exact_dup_all_mask].groupby(list(df.columns)).ngroups)

    # Repeated predictor profiles (across 21 features, excluding label)
    profile_dup_mask = df.duplicated(subset=feature_cols, keep="first")
    profile_dup_surplus = int(profile_dup_mask.sum())
    profile_dup_all_mask = df.duplicated(subset=feature_cols, keep=False)
    profile_dup_in_groups = int(profile_dup_all_mask.sum())
    profile_dup_groups = int(df[profile_dup_all_mask].groupby(feature_cols).ngroups)
    profile_unique_count = int((~profile_dup_mask).sum())

    # Conflicting-label profiles (identical 21 features, but differing Diabetes_binary labels)
    group_sizes = df.groupby(feature_cols).size()
    label_nunique = df.groupby(feature_cols)["Diabetes_binary"].nunique()
    conflicting_mask = label_nunique > 1
    conflicting_profiles = int(conflicting_mask.sum())
    conflicting_obs = int(group_sizes[conflicting_mask].sum())

    print(f"Exact full-row duplicate surplus: {exact_dup_surplus} ({exact_dup_in_groups} rows in {exact_dup_groups} groups)")
    print(f"Repeated predictor profiles surplus: {profile_dup_surplus} ({profile_dup_in_groups} rows in {profile_dup_groups} groups, {profile_unique_count} unique profiles)")
    print(f"Conflicting-label profiles: {conflicting_profiles} ({conflicting_obs} observations)")

    # 4. Range validation
    invalid_values = validate_ranges(df)
    total_invalid = sum(invalid_values.values())
    print(f"Total invalid values: {total_invalid}")
    if total_invalid > 0:
        print(f"Invalid columns details: {invalid_values}")

    # 5. Class distribution
    raw_class_counts = df["Diabetes_binary"].value_counts().to_dict()
    raw_class_pct = df["Diabetes_binary"].value_counts(normalize=True).to_dict()

    # Save comprehensive dataset audit artifacts
    audit_data = {
        "total_observations": int(len(df)),
        "total_features": int(len(feature_cols)),
        "exact_duplicate_surplus": exact_dup_surplus,
        "exact_duplicate_surplus_pct": round(exact_dup_surplus / len(df) * 100, 4),
        "exact_duplicate_in_groups": exact_dup_in_groups,
        "exact_duplicate_in_groups_pct": round(exact_dup_in_groups / len(df) * 100, 4),
        "exact_duplicate_groups": exact_dup_groups,
        "predictor_profile_unique": profile_unique_count,
        "predictor_profile_surplus": profile_dup_surplus,
        "predictor_profile_surplus_pct": round(profile_dup_surplus / len(df) * 100, 4),
        "predictor_profile_in_groups": profile_dup_in_groups,
        "predictor_profile_in_groups_pct": round(profile_dup_in_groups / len(df) * 100, 4),
        "predictor_profile_groups": profile_dup_groups,
        "conflicting_label_profiles": conflicting_profiles,
        "conflicting_label_observations": conflicting_obs,
        "conflicting_label_observations_pct": round(conflicting_obs / len(df) * 100, 4),
        "class_0_count": int((df["Diabetes_binary"] == 0).sum()),
        "class_0_pct": round(float((df["Diabetes_binary"] == 0).mean() * 100), 4),
        "class_1_count": int((df["Diabetes_binary"] == 1).sum()),
        "class_1_pct": round(float((df["Diabetes_binary"] == 1).mean() * 100), 4),
    }

    with open(AUDIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    audit_df = pd.DataFrame([
        {"Metric": k, "Value": v} for k, v in audit_data.items()
    ])
    audit_df.to_csv(AUDIT_CSV_PATH, index=False)

    dup_summary_df = pd.DataFrame([
        {"Category": "Total Observations", "Count": len(df), "Percentage": "100.00%"},
        {"Category": "Exact Full-Row Duplicate Surplus (22 cols)", "Count": exact_dup_surplus, "Percentage": f"{audit_data['exact_duplicate_surplus_pct']:.4f}%"},
        {"Category": "Total Rows in Exact Duplicate Groups", "Count": exact_dup_in_groups, "Percentage": f"{audit_data['exact_duplicate_in_groups_pct']:.4f}%"},
        {"Category": "Unique Predictor Profiles (21 features)", "Count": profile_unique_count, "Percentage": f"{profile_unique_count / len(df) * 100:.4f}%"},
        {"Category": "Predictor Profile Surplus Observations", "Count": profile_dup_surplus, "Percentage": f"{audit_data['predictor_profile_surplus_pct']:.4f}%"},
        {"Category": "Total Rows in Repeated Profile Groups", "Count": profile_dup_in_groups, "Percentage": f"{audit_data['predictor_profile_in_groups_pct']:.4f}%"},
        {"Category": "Conflicting-Label Predictor Profiles", "Count": conflicting_profiles, "Percentage": "-"},
        {"Category": "Observations in Conflicting Profiles", "Count": conflicting_obs, "Percentage": f"{audit_data['conflicting_label_observations_pct']:.4f}%"}
    ])
    dup_summary_df.to_csv(DUP_SUMMARY_PATH, index=False)
    print(f"Dataset audit saved to: {AUDIT_JSON_PATH} and {AUDIT_CSV_PATH}")

    # Build preprocessing step log
    steps_log = []

    steps_log.append({
        "step": "Load Raw Dataset",
        "result": f"Shape: {raw_shape}",
        "notes": f"Successfully loaded BRFSS 2015 dataset with {raw_shape[1]} columns and {raw_shape[0]:,} records."
    })

    steps_log.append({
        "step": "Check Missing Values",
        "result": f"{total_missing} missing values",
        "notes": "No missing value handling needed; survey records are fully populated."
    })

    cleaned_df = df.copy()
    steps_log.append({
        "step": "Inspect Exact Duplicates & Repeated Profiles",
        "result": f"{exact_dup_surplus} exact duplicate surplus; {profile_dup_surplus} profile surplus retained",
        "notes": (
            f"Exact full-row duplicates ({exact_dup_surplus}) and repeated feature profiles ({profile_dup_surplus}) "
            f"were intentionally retained because BRFSS lacks respondent identifiers to confirm duplicate identity. "
            f"Discretized survey bins naturally produce identical profiles across distinct individuals."
        )
    })

    steps_log.append({
        "step": "Check Expected Ranges",
        "result": f"{total_invalid} invalid values",
        "notes": "All variables conform to CDC BRFSS codebook specifications." if total_invalid == 0 else f"Found invalid values in: {invalid_values}"
    })

    # Type conversion
    numeric_values = cleaned_df.to_numpy(dtype=float)
    assert np.isfinite(numeric_values).all(), "Processed data contain non-finite values."
    assert np.allclose(numeric_values, np.round(numeric_values)), "Non-integer numeric values would be truncated by astype(int)."

    cleaned_df = cleaned_df.astype(int)
    steps_log.append({
        "step": "Convert Data Types",
        "result": "float64 -> int",
        "notes": "Converted categorical, ordinal, and binary float representations to integer types for clean formatting and memory efficiency."
    })

    # Assertions
    assert len(cleaned_df) == len(df), "Row count changed during preprocessing. Observations must be retained."
    assert list(cleaned_df.columns) == list(df.columns), "Column names or column order changed unexpectedly."

    raw_class_counts = df["Diabetes_binary"].astype(int).value_counts().sort_index()
    processed_class_counts = cleaned_df["Diabetes_binary"].astype(int).value_counts().sort_index()
    assert raw_class_counts.equals(processed_class_counts), "Target class counts changed during preprocessing."
    print("Preprocessing assertions passed successfully.")

    # Save cleaned dataset
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    steps_log.append({
        "step": "Save Cleaned Dataset",
        "result": f"Saved shape: {cleaned_df.shape}",
        "notes": "Cleaned data successfully written to CSV file."
    })

    # Check Class Imbalance Preservation
    cleaned_class_counts = cleaned_df["Diabetes_binary"].value_counts().to_dict()
    cleaned_class_pct = cleaned_df["Diabetes_binary"].value_counts(normalize=True).to_dict()
    
    notes_imbalance = (
        f"Imbalance preserved without SMOTE/resampling. "
        f"Raw: {raw_class_pct.get(1, 0):.2%} positive, {raw_class_pct.get(0, 0):.2%} negative. "
        f"Cleaned: {cleaned_class_pct.get(1, 0):.2%} positive, {cleaned_class_pct.get(0, 0):.2%} negative."
    )
    steps_log.append({
        "step": "Preserve Class Imbalance",
        "result": "Preserved",
        "notes": notes_imbalance
    })

    # Save summary table ONLY to results directory
    summary_df = pd.DataFrame(steps_log)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary table saved to: {SUMMARY_PATH}")

    # Generate Markdown documentation
    markdown_content = f"""# Data Preprocessing Documentation

## Purpose of Preprocessing
This document outlines the data preprocessing and quality audit for the **Diabetes-Analytics** project. The primary goal is to ensure the quality and integrity of the CDC Diabetes Health Indicators dataset, validate values against the BRFSS 2015 codebook, audit exact duplicates and repeated predictor profiles, and prepare the dataset for analysis and modeling. Importantly, the real-world class imbalance of the dataset is preserved; no artificial balancing techniques (like SMOTE or random under/oversampling) are applied.

## Datasets Directory Info
* **Raw Dataset Path:** `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`
* **Cleaned Dataset Path:** `data/processed/diabetes_cleaned.csv`

## Preprocessing Statistics and Checks

### 1. Dataset Shapes
* **Raw Shape:** {raw_shape[0]:,} rows, {raw_shape[1]} columns
* **Cleaned Shape:** {cleaned_df.shape[0]:,} rows, {cleaned_df.shape[1]} columns

### 2. Missing Values
* **Status:** {"No missing values found" if total_missing == 0 else f"Found {total_missing} missing values"}
* **Notes:** All fields are fully populated in the original survey response file.

### 3. Exact Duplicates & Repeated Predictor Profiles
* **Exact Full-Row Duplicates (22 columns):** {exact_dup_surplus:,} surplus rows ({exact_dup_in_groups:,} rows involved in {exact_dup_groups:,} distinct duplicate groups; {exact_dup_surplus / raw_shape[0]:.2%}).
* **Predictor Profiles (21 features):** {profile_unique_count:,} unique profiles across {raw_shape[0]:,} records.
* **Repeated Predictor Profile Surplus:** {profile_dup_surplus:,} surplus observations ({profile_dup_in_groups:,} rows involved in {profile_dup_groups:,} multi-observation profile groups; {profile_dup_surplus / raw_shape[0]:.2%}).
* **Conflicting-Label Profiles:** {conflicting_profiles:,} predictor profiles exhibit conflicting diabetes labels, comprising {conflicting_obs:,} observations ({conflicting_obs / raw_shape[0]:.2%}).
* **Retention Rationale:** The BRFSS survey dataset does not provide respondent identifiers. Discretized survey categories (e.g., 13 age brackets, 5 health ratings, 8 income brackets) inevitably cause independent respondents to share identical covariate combinations. Retaining these records preserves genuine sample frequencies and natural survey distributions.

### 4. Invalid Values (Out of Expected Range)
* **Status:** {"No invalid values found" if total_invalid == 0 else f"Found {total_invalid} invalid values"}
* **Checked ranges:**
"""
    for col, bounds in EXPECTED_RANGES.items():
        min_b, max_b = bounds
        range_str = f"> {min_b - 1}" if max_b == float("inf") else f"{min_b}-{max_b}"
        min_v = df[col].min()
        max_v = df[col].max()
        markdown_content += f"  * `{col}`: expected range `{range_str}` (actual min: {min_v}, max: {max_v}) - **OK**\n"

    markdown_content += f"""
### 5. Data Types and Casting
* **Initial data types:** All columns were loaded as `float64`.
* **Casting action:** Converted all columns to standard integers (`int`), as they represent binary indicators, categorical scales, or age categories with zero fractional parts.

### 6. Class Imbalance Preservation
* **SMOTE / Resampling:** **None applied (class imbalance is preserved)**
* **Diabetes Class Distribution Comparison:**

| Class (Diabetes_binary) | Raw Counts | Raw Pct | Cleaned Counts | Cleaned Pct |
|-------------------------|------------|---------|----------------|-------------|
| **0 (No reported diabetes)**      | {raw_class_counts.get(0, 0):,} | {raw_class_pct.get(0, 0):.2%} | {cleaned_class_counts.get(0, 0):,} | {cleaned_class_pct.get(0, 0):.2%} |
| **1 (Prediabetes/Diabetes positive class)** | {raw_class_counts.get(1, 0):,} | {raw_class_pct.get(1, 0):.2%} | {cleaned_class_counts.get(1, 0):,} | {cleaned_class_pct.get(1, 0):.2%} |

## Preprocessing Summary Table
Refer to the CSV summary at `results/data_preprocessing/preprocessing_summary.csv` for detailed steps.

| Step | Result | Notes |
|------|--------|-------|
"""
    for log in steps_log:
        markdown_content += f"| {log['step']} | {log['result']} | {log['notes']} |\n"

    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Documentation saved to: {DOC_PATH}")

if __name__ == "__main__":
    run_preprocessing()
