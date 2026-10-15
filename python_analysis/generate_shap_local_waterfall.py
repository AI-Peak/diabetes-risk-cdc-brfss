#!/usr/bin/env python
"""
Local SHAP Explanation Waterfall Generator (Figure 8)
------------------------------------------------------
Author: Advanced Data Analytics Agent
Description:
    Generates a publication-quality SHAP waterfall plot explaining a specific 
    high-risk, True-Positive prediction from the XGBoost model.
    
    Includes:
        - Consistent replication of the model and SHAP calculation pipeline.
        - Automated candidate search to locate a representative True-Positive case
          (high risk, positive-class record, showing typical risk factors like high BMI,
          high blood pressure, etc.).
        - Remapping of technical feature names to descriptive academic labels.
        - Mapping of standardized z-scores to original, easy-to-read clinical values 
          (e.g., BMI = 34, HighBP = Yes) for display in the plot.
        - Annotation of model scale (log-odds) and predicted probability (%).
        - Exports in SVG, PNG, and text metadata format.
        
    Outputs:
        - docs/figures/shap_local_high_risk.svg
        - docs/figures/shap_local_high_risk.png
        - docs/figures/shap_local_high_risk_metadata.txt
"""

import os
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

# Resolve paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "docs" / "figures"

# ---------------------------------------------------------
# Descriptive Mapping for Feature Names & Value Labels
# ---------------------------------------------------------
FEATURE_MAPPING = {
    "GenHlth": "Self-rated general health",
    "HighBP": "High blood pressure",
    "Age": "Age category",
    "BMI": "Body mass index (BMI)",
    "HighChol": "High cholesterol",
    "DiffWalk": "Difficulty walking",
    "Income": "Income level",
    "PhysHlth": "Poor physical health days",
    "HeartDiseaseorAttack": "Heart disease/attack history",
    "PhysActivity": "Physical activity",
    "Education": "Education level",
    "MentHlth": "Poor mental health days",
    "CholCheck": "Cholesterol check",
    "Smoker": "Smoker status",
    "Stroke": "Stroke history",
    "Sex": "Sex",
    "Fruits": "Fruit consumption",
    "Veggies": "Vegetable consumption",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Healthcare coverage",
    "NoDocbcCost": "Doctor cost barrier"
}

GEN_HLTH_LABELS = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor"
}

def load_data(file_path: Path) -> pd.DataFrame:
    """Loads the cleaned dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {file_path}. Run preprocessing first.")
    return pd.read_csv(file_path)

def train_xgboost_model(X_train, y_train):
    """Trains the best-performing XGBoost model identified in modeling phase."""
    print("Training XGBoost model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train using DataFrame to retain feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train_scaled_df, y_train)
    return model, scaler

def search_true_positive_candidate(model, X_test, scaler, y_test, X_test_sample):
    """Programmatically finds a representative high-risk True-Positive candidate."""
    print("Searching for representative high-risk True-Positive candidate...")
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    y_prob_sample = model.predict_proba(X_test_sample)[:, 1]
    X_test_sample_unscaled = X_test.loc[X_test_sample.index]
    
    candidates = []
    for idx in range(len(X_test_sample)):
        original_idx = X_test_sample.index[idx]
        actual_label = y_test.loc[original_idx]
        prob = y_prob_sample[idx]
        
        # Select True Positives with predicted probability >= 0.70
        if actual_label == 1 and prob >= 0.70:
            row = X_test_sample_unscaled.iloc[idx]
            score = 0
            # Score how typical/representative the risk profile is
            if row["HighBP"] == 1: score += 1
            if row["HighChol"] == 1: score += 1
            if row["GenHlth"] >= 4: score += 1
            if row["BMI"] >= 30: score += 1
            if row["Age"] >= 9: score += 1
            if row["DiffWalk"] == 1: score += 1
            
            candidates.append((idx, prob, score, row))
            
    # Sort candidates by typicallity score (descending), then probability (descending)
    candidates.sort(key=lambda x: (-x[2], -x[1]))
    
    if not candidates:
        # Relax constraints if no perfect match (prob >= 0.60)
        print("Relaxing constraints to find candidate...")
        for idx in range(len(X_test_sample)):
            original_idx = X_test_sample.index[idx]
            actual_label = y_test.loc[original_idx]
            prob = y_prob_sample[idx]
            if actual_label == 1 and prob >= 0.60:
                row = X_test_sample_unscaled.iloc[idx]
                score = 0
                if row["HighBP"] == 1: score += 1
                if row["HighChol"] == 1: score += 1
                if row["GenHlth"] >= 3: score += 1
                if row["BMI"] >= 26: score += 1
                candidates.append((idx, prob, score, row))
        candidates.sort(key=lambda x: (-x[2], -x[1]))
        
    if not candidates:
        raise ValueError("Could not find a suitable True Positive candidate in the test sample.")
        
    best_candidate = candidates[0]
    print(f"Selected candidate at sample index: {best_candidate[0]} (Original Index: {X_test_sample.index[best_candidate[0]]})")
    print(f"  Predicted Probability: {best_candidate[1]:.2%}")
    print(f"  Representative Score: {best_candidate[2]}/6")
    return best_candidate[0], best_candidate[1], best_candidate[3]

def generate_local_waterfall():
    """Generates and saves the SHAP waterfall local explanation."""
    # 1. Load data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]
    
    # 2. Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model, scaler = train_xgboost_model(X_train, y_train)
    
    # 3. Scale test and sample 10,000 cases
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    X_test_sample = X_test_scaled_df.sample(n=10000, random_state=42)
    
    # 4. Find best True Positive candidate
    best_idx, predicted_prob, original_row = search_true_positive_candidate(
        model, X_test, scaler, y_test, X_test_sample
    )
    
    # 5. Compute SHAP Values
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_sample)
    
    # Extract class 1 (positive class) explanation
    if len(shap_values.values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values
        
    # Get a copy of the explanation object for the chosen individual
    single_exp = copy.deepcopy(shap_values_to_plot[best_idx])
    
    # 6. Format Feature Names & Values for Display
    print("Formatting feature names and values for plot...")
    
    # Apply descriptive labels to the features list
    single_exp.feature_names = [
        FEATURE_MAPPING.get(name, name) for name in single_exp.feature_names
    ]
    
    # Map the standardized z-scores inside single_exp.data to original readable values
    formatted_values = []
    for col in X_test.columns:
        val = original_row[col]
        # Binary variables mapping
        if col in ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"]:
            formatted_values.append("Yes" if val == 1.0 else "No")
        # Sex mapping
        elif col == "Sex":
            formatted_values.append("Male" if val == 1.0 else "Female")
        # General Health mapping
        elif col == "GenHlth":
            formatted_values.append(f"{val:.0f} ({GEN_HLTH_LABELS.get(int(val), '')})")
        # Continuous or ordinal variables
        else:
            # Format as integer if whole number, else float
            formatted_values.append(int(val) if val.is_integer() else val)
            
    # Overwrite the data attribute with descriptive strings
    single_exp.data = np.array(formatted_values, dtype=object)
    
    # 7. Render Waterfall Plot
    print("Plotting local explanation waterfall...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # Set canvas size
    plt.figure(figsize=(9.5, 6.0))
    
    # Plot waterfall (display top 9 contributors, group others)
    shap.plots.waterfall(single_exp, max_display=9, show=False)
    
    # Subtitle detail matching user requests
    plt.title(
        "SHAP Local Explanation (True-Positive Survey Respondent)\n"
        f"Actual Class: Diabetes/Prediabetes  |  Predicted Risk: {predicted_prob:.1%}",
        fontsize=12, fontweight='bold', pad=25, color='#2D3748'
    )
    
    plt.xlabel("SHAP Value (Impact in Log-Odds Scale)", fontsize=10.5, fontweight='bold', color='#2D3748', labelpad=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=10.0, colors='#2D3748')
    
    # Prevent cutting off feature labels on the left or values on the right
    plt.tight_layout()
    
    # File Paths
    svg_path = OUTPUT_DIR / "shap_local_high_risk.svg"
    png_path = OUTPUT_DIR / "shap_local_high_risk.png"
    metadata_path = OUTPUT_DIR / "shap_local_high_risk_metadata.txt"
    
    # Save figures
    plt.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 8. Write Metadata File
    print("Writing metadata file...")
    # Calculate top contributing features (by absolute SHAP value)
    top_indices = np.argsort(np.abs(single_exp.values))[::-1]
    top_features_meta = []
    for idx_f in top_indices[:5]:
        top_features_meta.append(
            f"  - {shap_values_to_plot.feature_names[idx_f]} "
            f"(Value: {formatted_values[idx_f]}, SHAP: {single_exp.values[idx_f]:.4f})"
        )
        
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("SHAP Local Explanation Metadata (Figure 8)\n")
        f.write("===========================================\n\n")
        f.write(f"Original Test Set Index: {X_test_sample.index[best_idx]}\n")
        f.write("Actual Class: Diabetes/Prediabetes (1)\n")
        f.write(f"Predicted Probability: {predicted_prob:.4%}\n")
        f.write(f"Predicted Class: Diabetes/Prediabetes (1)\n")
        f.write(f"Model Output (Log-Odds): {single_exp.base_values + np.sum(single_exp.values):.4f}\n")
        f.write(f"Base Value (Log-Odds): {single_exp.base_values:.4f}\n\n")
        f.write("Top 5 Contributing Features:\n")
        f.write("\n".join(top_features_meta) + "\n")
        
    print(f"\nSHAP local explanation figures successfully generated:")
    print(f"  - SVG: {svg_path}")
    print(f"  - PNG: {png_path} (300 DPI)")
    print(f"  - Metadata TXT: {metadata_path}")

if __name__ == "__main__":
    generate_local_waterfall()
