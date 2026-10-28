# Experiment B5: SHAP Ranking Stability Across Folds and Resampling Seeds

## 1. Stability Across 5 Development Cross-Validation Folds
- **Evaluation:** 5 independent fold-specific XGBoost models evaluated on validation samples.
- **Top-10 Mean Pairwise Jaccard Similarity:** `0.9273`
- **Top-10 Minimum Pairwise Jaccard Similarity:** `0.8182`
- **Consensus Features (Present in 5/5 Folds):**
  [
  "CholCheck",
  "GenHlth",
  "Age",
  "HighChol",
  "Sex",
  "BMI",
  "HeartDiseaseorAttack",
  "HighBP",
  "Income"
]
- **Feature Frequencies in Top-10 Across Folds:**
  {
  "CholCheck": 5,
  "GenHlth": 5,
  "Age": 5,
  "Sex": 5,
  "HighBP": 5,
  "BMI": 5,
  "Income": 5,
  "HvyAlcoholConsump": 4,
  "HighChol": 5,
  "MentHlth": 1,
  "HeartDiseaseorAttack": 5
}

## 2. Stability Across 3 Random Sampling Seeds (N = 10,000 each)
- **Seeds Evaluated:** [42, 123, 999] on the Holdout Test set.
- **Top-10 Mean Pairwise Jaccard Similarity:** `1.0000`
- **Top-10 Minimum Pairwise Jaccard Similarity:** `1.0000`
- **Consensus Features (Present in 3/3 Seeds):**
  [
  "CholCheck",
  "HvyAlcoholConsump",
  "GenHlth",
  "Age",
  "HighChol",
  "Sex",
  "BMI",
  "HeartDiseaseorAttack",
  "HighBP",
  "Income"
]
- **Feature Frequencies in Top-10 Across Seeds:**
  {
  "CholCheck": 3,
  "GenHlth": 3,
  "Age": 3,
  "Sex": 3,
  "HighBP": 3,
  "BMI": 3,
  "Income": 3,
  "HvyAlcoholConsump": 3,
  "HighChol": 3,
  "HeartDiseaseorAttack": 3
}

## 3. Methodological Interpretation
SHAP global rankings demonstrate remarkable stability across both model partitioning (fold splits) and subsampling variations:
1. Core cardiometabolic drivers (`GenHlth`, `HighBP`, `Age`, `BMI`, `HighChol`, `Sex`, `CholCheck`, `Income`, `HeartDiseaseorAttack`) consistently anchor the top ranks.
2. Jaccard similarity remains extraordinarily high ($\ge 0.82$), demonstrating that SHAP-based feature importance reflects stable underlying structural risk signals rather than sampling noise or model training stochasticity.
