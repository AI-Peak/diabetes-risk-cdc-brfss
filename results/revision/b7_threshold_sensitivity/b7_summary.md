# Experiment B7: Decision-Threshold Sensitivity Analysis

## 1. Development-Only Threshold Selection
Screening operating thresholds were selected strictly from 5-fold cross-validation out-of-fold predictions on the Development set, maximizing Precision subject to alternative clinical Recall constraints:

| Recall Constraint | Global Dev OOF Threshold | Dev OOF Recall | Dev OOF Precision | Dev OOF F1 |
|---|---|---|---|---|
| **Recall >= 0.75** | `0.16` | 0.7538 | 0.3261 | 0.4552 |
| **Recall >= 0.80 (Baseline)** | `0.13` | 0.8097 | 0.3000 | 0.4378 |
| **Recall >= 0.85** | `0.10` | 0.8653 | 0.2736 | 0.4157 |

## 2. Fold-Level Threshold Stability
Fold-level variation was measured across the 5 outer validation folds:
- **Recall >= 0.75:** Mean = `0.16` ± `0.0000` (Range: [0.16, 0.16])
- **Recall >= 0.80:** Mean = `0.13` ± `0.0000` (Range: [0.13, 0.13])
- **Recall >= 0.85:** Mean = `0.10` ± `0.0000` (Range: [0.10, 0.10])

## 3. Descriptive Holdout Test Performance
Evaluating the frozen thresholds on the untouched Holdout Test partition ($N = 50,736$):

| Constraint | Frozen Threshold | Recall | Precision | F1-score | Specificity | Accuracy | TP | FP |
|---|---|---|---|---|---|---|---|---|
| **Recall >= 0.75** | `0.16` | 0.7512 | 0.3252 | 0.4539 | 0.7477 | 0.7482 | 5,310 | 11,018 |
| **Recall >= 0.80 (Baseline)** | `0.13` | 0.8099 | 0.2991 | 0.4369 | 0.6928 | 0.7091 | 5,725 | 13,416 |
| **Recall >= 0.85** | `0.10` | 0.8660 | 0.2719 | 0.4138 | 0.6245 | 0.6582 | 6,122 | 16,395 |

## 4. Key Interpretation
1. Tight fold stability: Standard deviations of fold-specific thresholds remain minimal ($\le 0.01$), confirming that the optimization surface is smooth and reliable.
2. Clinical trade-off: Increasing the Recall constraint from 0.75 to 0.85 shifts the operating threshold from `0.16` down to `0.10`. This increases sensitivity (from 75.1% to 86.6%), capturing `812` additional at-risk individuals at the cost of `5,377` additional false-positive follow-ups.
