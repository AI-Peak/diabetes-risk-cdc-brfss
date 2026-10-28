# Experiment B1: Duplicate-Profile Leakage Audit & Grouped Split Sensitivity

## 1. Duplicate & Profile Distribution
- **Total Records:** 253,680 across 21 predictors and 1 target (`Diabetes_binary`).
- **Exact Full-Row Duplicates (22 columns):** 24,206 surplus rows (9.54%).
- **Repeated Predictor Profiles (21 features):** 25,772 surplus rows (10.16%); 227,908 unique feature vectors.
- **Conflicting-Label Profiles:** 1,566 profiles representing 5,218 observations (2.06%).

## 2. Original Split Profile Overlap
In the original stratified 80/20 train/test split (random_state=42):
- **Holdout Test Set Size:** 50,736 observations.
- **Holdout Rows with Identical Profile in Dev Set:** 6,836 (13.47%).
- **Non-Overlapping Holdout Rows:** 43,900 (86.53%).
- **Mutually Exclusive Overlap Breakdown:**
  - No matching profile in Dev: 43,900 (86.53%)
  - Same-label match only in Dev: 5,965 (11.76%)
  - Different-label match only in Dev: 461 (0.91%)
  - Both same and different label matches in Dev: 410 (0.81%)
  - **Sum Check:** 50,736 / 50,736 (100.00%)

## 3. Baseline Model Performance on Non-Overlapping Holdout Subset
Evaluated using the original baseline XGBoost model and threshold 0.13:
- **PR-AUC:** 0.4317 (Baseline full holdout: 0.4238)
- **ROC-AUC:** 0.8180 (Baseline full holdout: 0.8272)
- **Recall:** 0.8281 (Baseline full holdout: 0.8099)
- **Precision:** 0.3021 (Baseline full holdout: 0.2991)
- **F1-score:** 0.4427 (Baseline full holdout: 0.4369)
- *Note:* In the non-overlapping subset, positive prevalence is slightly higher (15.28% vs 13.93%).

## 4. Profile-Grouped Split & Group-Disjoint Workflow
- **Grouping Key:** All 21 predictors (target strictly excluded).
- **Algorithm:** StratifiedGroupKFold (n_splits=5, shuffle=True, random_state=42).
- **Shared Profile Groups between Dev and Holdout:** 0 (Verified zero leakage).
- **Dev Set:** 202,943 rows (80.00%), Prevalence = 13.93%.
- **Holdout Set:** 50,737 rows (20.00%), Prevalence = 13.93%.
- **Grouped CV Best Model:** XGBoost (Mean CV PR-AUC = 0.4344 ± 0.0061).
- **Grouped OOF Selected Threshold:** 0.13 (Recall constraint >= 0.80).
- **Grouped Holdout Test Results:**
  - **PR-AUC:** 0.4321 (95% CI: [0.4213, 0.4436])
  - **ROC-AUC:** 0.8279 (95% CI: [0.8231, 0.8326])
  - **Recall:** 0.8037 (95% CI: [0.7948, 0.8126])
  - **Precision:** 0.2974 (95% CI: [0.2935, 0.301])
  - **F1-score:** 0.4341 (95% CI: [0.4289, 0.439])

## 5. Comparative Evaluation Summary
| Protocol | Evaluation Split | Model | Threshold | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| Baseline | Full Holdout (N=50,736) | XGBoost | 0.13 | 0.4238 | 0.8272 | 0.8099 | 0.2991 | 0.4369 |
| Sensitivity 1 | Non-Overlap Holdout (N=43,900) | XGBoost | 0.13 | 0.4317 | 0.8180 | 0.8281 | 0.3021 | 0.4427 |
| Sensitivity 2 | Grouped Holdout (N=50,737) | XGBoost | 0.13 | 0.4321 | 0.8279 | 0.8037 | 0.2974 | 0.4341 |

## 6. Interpretation and Methodological Conclusion
Profile overlap in the original random split accounted for 13.47% of the holdout set. However, strictly eliminating profile overlap via either non-overlap subsetting or a completely group-disjoint split protocol demonstrates that model discrimination and recall-constrained screening performance remain robust and structurally unaltered:
- PR-AUC varies within a narrow band (0.4238 full baseline vs 0.4317 non-overlap vs 0.4321 grouped holdout).
- ROC-AUC remains at ~0.82-0.83.
- Recall at the screening operating point remains stable above 0.80.
Thus, empirical performance differences observed across split designs remain within nominal sampling variation. While grouped-split comparison represents a composite design shift rather than an isolated causal test of duplication, the sensitivity analyses confirm that screening performance benchmarks (PR-AUC ~0.41-0.42, ROC-AUC ~0.82-0.83, Recall ~0.80) are stable across evaluation protocols.
