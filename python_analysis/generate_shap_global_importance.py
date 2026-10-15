#!/usr/bin/env python
"""
Global SHAP Feature Importance Figure Generator (Figure 6)
-----------------------------------------------------------
Author: Advanced Data Analytics Agent
Description:
    Generates a publication-quality horizontal bar chart illustrating the 
    overall feature importance of health indicators for the XGBoost model 
    based on Mean Absolute SHAP values.
    
    Includes:
        - Consistent replication of the model and SHAP calculation pipeline 
          (XGBoost, 10,000-sample test subset, random_state=42).
        - Renaming of features to descriptive, reader-friendly academic labels.
        - Two-tone color coding highlighting the top 5 features.
        - Explicit labeling of mean absolute SHAP values at the end of each bar.
        - SVG and PNG file exports.
        
    Outputs:
        - docs/figures/shap_global_importance.svg
        - docs/figures/shap_global_importance.png
"""

import os
from pathlib import Path
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
# Descriptive Health Indicators Mapping
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
    "Sex": "Sex (Male/Female)",
    "Fruits": "Fruit consumption",
    "Veggies": "Vegetable consumption",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Healthcare coverage",
    "NoDocbcCost": "Doctor cost barrier"
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

def generate_global_importance():
    """Computes SHAP values and saves a horizontal bar chart of global importances."""
    # 1. Load dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading cleaned dataset...")
    df = load_data(DATA_PATH)
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]
    
    # 2. Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    model, scaler = train_xgboost_model(X_train, y_train)
    
    # 3. Test scaling and sampling
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    X_test_sample = X_test_scaled_df.sample(n=10000, random_state=42)
    
    # 4. Compute SHAP Values
    print("Computing SHAP values (10,000 samples)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_sample)
    
    # Take positive class predictions if binary output is 3D
    if len(shap_values.values.shape) == 3:
        shap_imp_vals = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
    else:
        shap_imp_vals = np.abs(shap_values.values).mean(axis=0)
        
    # 5. Build sorted DataFrame
    shap_imp_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": shap_imp_vals
    }).sort_values(by="Importance", ascending=False)
    
    # Apply descriptive naming
    shap_imp_df["Descriptive"] = shap_imp_df["Feature"].map(FEATURE_MAPPING)
    
    # Take top 12 features (matches the vertical display of beeswarm plot)
    top_n = 12
    shap_imp_top = shap_imp_df.head(top_n).copy()
    
    # Reverse order so the most important feature is plotted on top
    shap_imp_top = shap_imp_top.iloc[::-1]
    
    # 6. Render Figure
    print("Plotting global SHAP importance...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    
    # Two-tone color scheme (Navy for top 5 features, Muted Slate Blue for bottom 7)
    colors = []
    # Note: Since the list was reversed, top features are at the end of the DataFrame
    for i in range(len(shap_imp_top)):
        rank_from_top = len(shap_imp_top) - 1 - i
        if rank_from_top < 5:
            colors.append('#1B365D') # Dark Navy Blue
        else:
            colors.append('#5D7B93') # Muted Slate Blue
            
    # Draw horizontal bars
    bars = ax.barh(
        shap_imp_top["Descriptive"], 
        shap_imp_top["Importance"], 
        color=colors, 
        height=0.60, 
        edgecolor='#2D3748', 
        linewidth=1.0, 
        zorder=3
    )
    
    # Add numerical labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.005, 
            bar.get_y() + bar.get_height()/2.0, 
            f'{width:.4f}', 
            va='center', ha='left', 
            fontsize=9.5, fontweight='bold', color='#2D3748',
            zorder=4
        )
        
    # Styling
    max_val = shap_imp_top["Importance"].max()
    ax.set_xlim(0, max_val * 1.15) # Leave room for the text labels
    
    ax.set_xlabel("Mean Absolute SHAP Value (Average Contribution Magnitude)", fontsize=10.5, fontweight='bold', color='#2D3748', labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10.0, colors='#2D3748')
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#718096')
    ax.spines['bottom'].set_color('#718096')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Add vertical gridlines behind the bars
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#CBD5E0', zorder=1)
    ax.set_axisbelow(True)
    
    # Prevent cutting labels
    plt.tight_layout()
    
    # Save Paths
    svg_path = OUTPUT_DIR / "shap_global_importance.svg"
    png_path = OUTPUT_DIR / "shap_global_importance.png"
    
    # Save files
    plt.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    print(f"\nSHAP global importance figures successfully generated:")
    print(f"  - SVG: {svg_path}")
    print(f"  - PNG: {png_path} (300 DPI)")
    
    # Print the top 5 features and their values for the report
    print("\nTop 5 Features and Mean Absolute SHAP values:")
    top_5 = shap_imp_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"  {row['Feature']} ({row['Descriptive']}): {row['Importance']:.4f}")

if __name__ == "__main__":
    generate_global_importance()
