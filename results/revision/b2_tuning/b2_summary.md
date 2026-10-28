# Experiment B2: Controlled Hyperparameter Tuning and Paired Comparison

## 1. Experimental Protocol
- **Evaluation Design:** 5 Outer Folds $\times$ 3 Repetitions (15 paired evaluation iterations).
- **Partition Pairing:** XGBoost and Random Forest were trained and evaluated on **identical outer test splits**.
- **Inner Tuning:** 3-fold stratified cross-validation evaluated on a stratified subsample of 40,000 rows per outer training fold for computational tractability, followed by full refitting of the optimal hyperparameter combination on the entire outer training fold (~162k rows).
- **Search Spaces:**
  - XGBoost: `max_depth` $\in [3, 6, 9]$, `learning_rate` $\in [0.05, 0.1]$
  - Random Forest: `max_depth` $\in [8, 12, \text{None}]$, `min_samples_leaf` $\in [1, 5]$
- **Leakage Controls:** Holdout test data strictly excluded; all model selection decisions made on Development CV.

## 2. Paired Performance Comparison
| Model | Mean PR-AUC | Fold Std | Baseline CV PR-AUC |
|---|---|---|---|
| **Tuned XGBoost** | `0.4338` | `0.0066` | 0.4359 |
| **Tuned Random Forest** | `0.4270` | `0.0073` | 0.4301 |
| **Paired Difference (XGB - RF)** | `+0.0068` | `0.0036` | +0.0058 |

## 3. Uncertainty Estimation for Paired Differences
- **Descriptive 95% Confidence Interval:** `[+0.0048, +0.0087]`
- **Nadeau-Bengio Dependence-Corrected 95% CI:** `[+0.0025, +0.0111]`
  - Corrected $t$-statistic = `3.3656` ($p = 0.0046$).
  - *Methodological Note:* The Nadeau-Bengio correction accounts for the overlap and dependence of repeated cross-validation folds (with correction factor $\frac{1}{KR} + \frac{n_{\text{val}}}{n_{\text{train}}} = \frac{1}{15} + 0.25 = 0.3167$).

## 4. Hyperparameter Selection Frequencies
### XGBoost:
{
  "{'max_depth': 3, 'learning_rate': 0.1}": 15
}

### Random Forest:
{
  "{'max_depth': 8, 'min_samples_leaf': 5}": 8,
  "{'max_depth': 12, 'min_samples_leaf': 5}": 5,
  "{'max_depth': 8, 'min_samples_leaf': 1}": 2
}

## 5. Methodological Conclusion
1. **Numerical Leader in 14 of 15 Iterations:** Tuned XGBoost achieves a higher PR-AUC than Tuned Random Forest in 14 out of 15 paired evaluations (93.33% win rate; mean difference $+0.0068$), with Random Forest holding a marginal lead in iteration 14 (-0.0003).
2. **Controlled Search Space:** Hyperparameter tuning did not invert the ranking observed in the baseline paper; the relative advantage of gradient boosted decision trees over bagging remains robust under formal inner tuning.
3. **Statistical Qualification:** While XGBoost is the numerical leader in 14/15 iterations, the corrected confidence interval confirms that both ensemble families operate within a closely matched performance bracket (PR-AUC $\approx 0.43-0.44$).
