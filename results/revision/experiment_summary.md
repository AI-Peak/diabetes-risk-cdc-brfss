# Master Experimental Summary: Phase 1 Revisions (B1–B7)

**Research Project:** *A Reliability-Oriented and Interpretable Machine Learning Framework for Diabetes Analytics Using the CDC BRFSS Dataset*  
**Commit:** `69022a8a13ba72be3611dd3dc303ad24fc4265b4`  
**Dataset SHA-256:** `19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0`  
**Date:** 2026-10-27 18:44:07Z

---

## 1. Executive Summary of Revisions

| Experiment | Status | Key Baseline Metric | Revised Metric | Methodological Impact |
|---|---|---|---|---|
| **B1 — Duplicate-Profile Leakage** | **VERIFIED** | Full Holdout PR-AUC = 0.4238, Recall = 0.8099 | Non-Overlap PR-AUC = 0.4317; Grouped Holdout PR-AUC = 0.4321, Recall = 0.8037 | Profile overlap (13.47% in holdout) accounts for marginal shifts within nominal sampling variation. Grouped split sensitivity confirms robust screening performance. |
| **B2 — Paired Model Comparison** | **VERIFIED** | CV PR-AUC: XGB = 0.4359, RF = 0.4301 (Diff = +0.0058) | Tuned Nested CV: XGB = 0.4338, RF = 0.427 (Diff = 0.0068) | XGBoost remains consistent numerical leader; corrected CI confirms both models perform within a tight bracket. |
| **B3 — Calibration Uncertainty** | **VERIFIED** | Slope = 0.9589, Intercept = -0.0515, Brier = 0.0974 | Slope 95% CI: [0.9342, 0.9854], Intercept 95% CI: [-0.0861, -0.0148], Brier: [0.0964, 0.0983] | Exact replication verified with 1,000 stratified bootstrap samples (0 failures). Reliability figure generated. |
| **B4 — Decision Curve Analysis** | **VERIFIED** | Operating point $p_t = 0.13$ | Net Benefit = +0.0733 (Screen All = +0.0107, Screen None = 0.0) | Clear positive net benefit over default strategies across $p_t \in [0.08, 0.40]$. |
| **B5 — SHAP Stability** | **VERIFIED** | Top-10 agreement reported once | Fold Mean Jaccard = 0.9273; Seed Mean Jaccard = 1.0000 | SHAP attributions are exceptionally stable across training folds and 10k subsampling seeds. |
| **B6 — Rare Categories & VIF** | **VERIFIED** | Max VIF = 357.58 (`Education_6`), Top-10 LR = 10/10 | Revised Max VIF = 1.80 (`PhysHlth`), Top-10 LR = 10/10, Spearman = 0.9338 | Collinearity was entirely an artifact of sparse reference category (Education 1, N=174). Modal recoding reduces VIF by 99.5% while preserving ranking. |
| **B7 — Threshold Sensitivity** | **VERIFIED** | Selected Threshold = 0.13 at Recall >= 0.80 | Recall >= 0.75: $t=0.16$; Recall >= 0.80: $t=0.13$; Recall >= 0.85: $t=0.10$ | Fold SD $\le 0.004$; demonstrates smooth clinical trade-off between sensitivity and workup burden. |

---

## 2. Detailed Findings by Experiment

### Experiment B1: Duplicate-Profile Leakage Audit & Grouped Split
- **Surplus Records:** 24,206 exact 22-column duplicate rows; 25,772 repeated 21-predictor profile rows (38,000 records across 12,228 multi-observation groups).
- **Conflicting Profiles:** 1,566 profiles with discordant diabetic labels (5,218 records), reflecting genuine demographic stochasticity and unmeasured latent variables in public survey data.
- **Original Split Overlap:** In the original random stratified 80/20 split, 6,836 holdout rows (13.47%) share identical 21-predictor profiles with development records (5,965 same-label only, 461 different-label only, 410 both).
- **Non-Overlapping Holdout Subset (N = 43,900):** Evaluating the baseline model strictly on non-overlapping records yields PR-AUC = 0.4317, ROC-AUC = 0.8180, Recall = 0.8281, Precision = 0.3021.
- **Grouped Split & Grouped CV (Zero Leakage):** Partitioning by unique predictor profiles (zero shared groups between Dev and Holdout, and zero shared groups between CV folds) preserves model ranking (XGBoost selected with Mean CV PR-AUC = 0.4344) and holdout metrics: PR-AUC = 0.4321, ROC-AUC = 0.8279, Recall = 0.8037, Precision = 0.2974, F1 = 0.4341.
- **Conclusion:** Bounded sensitivity analyses indicate screening performance benchmarks remain stable across split designs, with observed variations within nominal sampling error.

### Experiment B2: Hyperparameter Tuning & Paired Comparison
- Nested 5-fold $\times$ 3-repeat cross-validation strictly on Development data.
- Tuned XGBoost achieves Mean PR-AUC of 0.4338, whereas Tuned Random Forest achieves 0.427.
- Paired difference favors XGBoost in 14 of 15 outer evaluations (93.33% win rate; Mean diff = 0.0068), with Random Forest holding a marginal lead in iteration 14 (-0.0003).
- The Nadeau-Bengio dependence-corrected confidence interval [0.0025, 0.0111] shows that while XGBoost is numerically superior, both architectures perform within a narrow, competitive performance band.

### Experiment B3: Calibration Analysis with Bootstrap Confidence Intervals
- Brier Score: 0.0974 (95% CI: [0.0964, 0.0983]).
- Calibration Slope: 0.9589 (95% CI: [0.9342, 0.9854]).
- Calibration Intercept: -0.0515 (95% CI: [-0.0861, -0.0148]).
- 1,000 stratified bootstrap resamples completed with 1,000/1,000 successful fits (0 failures).
- Publication-quality reliability diagram exported to `results/revision_part1/figures/b3_calibration_reliability_curve.png` and `.pdf`.

### Experiment B4: Decision Curve Analysis (DCA)
- Net benefit evaluated across thresholds 0.05 to 0.40.
- At the prespecified operating threshold ($p_t = 0.13$), XGBoost achieves Net Benefit = +0.0733, substantially outperforming Screen All (+0.0107) and Screen None (0.0).
- Across $p_t \in [0.08, 0.40]$, model-guided screening maintains superior net clinical benefit.

### Experiment B5: SHAP Ranking Stability
- Across 5 cross-validation folds, Top-10 mean pairwise Jaccard similarity is 0.9273 (min = 0.8182).
- Across 3 independent sampling seeds of 10,000 cases each, Top-10 mean pairwise Jaccard similarity is 1.0000.
- Key cardiometabolic predictors (`GenHlth`, `HighBP`, `Age`, `BMI`, `HighChol`, `Sex`, `CholCheck`, `Income`, `HeartDiseaseorAttack`) appear in 100% of folds and resampling seeds.

### Experiment B6: Rare Categories, Multivariable Logistic Regression, and VIF
- Education Level 1 ("Never attended school / kindergarten") accounts for only 174 observations (0.069%).
- In the original dummy coding (`drop_first=True`), Level 1 was the reference category, creating extreme multicollinearity and inflating `Education_6` VIF to 357.58.
- Merging Level 1 into Level 2 and assigning modal reference categories (Level 6 for Education, Level 8 for Income) reduced maximum VIF to **1.80** (on `PhysHlth`).
- Feature-level likelihood-ratio ranking remained identical in Top-10 overlap (10/10 with original LR, 10/10 with baseline SHAP) and maintained $r_s = 0.9338$.
- `CholCheck` AOR (~3.44) reflects routine clinical contact and screening opportunity, not biological causality.

### Experiment B7: Decision-Threshold Sensitivity Analysis
- Under Recall constraints $\ge 0.75$, $\ge 0.80$, and $\ge 0.85$, Development OOF thresholds were selected as $0.16$, $0.13$, and $0.10$ respectively.
- Fold-level threshold stability is exceptional (SD $\le 0.004$).
- Evaluated on holdout data, thresholds exhibit smooth trade-offs between sensitivity (75.1% to 86.6%) and precision (32.5% to 27.2%).

---

## 3. Manuscript Impact Map (For Phase 2 Preparation)
*Note: Phase 1 does NOT modify manuscript files. The table below guides Phase 2 updates.*

| Revision Experiment | Affected Manuscript Sections | Specific Artifacts / Numbers to Integrate |
|---|---|---|
| **B1 (Leakage Audit)** | Sections 3.1, 4.3, 5.4 | Report 13.47% profile overlap in random split; report grouped holdout PR-AUC (0.4321), ROC-AUC (0.8279), and Recall (0.8037) confirming robustness. |
| **B2 (Tuned Comparison)** | Sections 3.3, 4.2, Table 2 | Update Table 2 with nested CV results: XGBoost (0.4338) vs Random Forest (0.427); note Nadeau-Bengio corrected CI. |
| **B3 (Calibration)** | Sections 3.5, 4.3, Figure | Add bootstrap 95% CIs for Slope [0.9342, 0.9854] and Intercept [-0.0861, -0.0148]; embed reliability curve panel. |
| **B4 (Decision Curve)** | Section 5.1, Discussion | Incorporate Net Benefit at $p_t = 0.13$ (+0.0733) and discuss clinical utility across $p_t \in [0.08, 0.40]$. |
| **B5 (SHAP Stability)** | Section 4.4, Discussion | Report mean Jaccard stability (0.9273 across folds, 1.0000 across seeds) confirming robust XAI rankings. |
| **B6 (VIF Sensitivity)** | Sections 4.1, 4.4 | Explain that 357.58 VIF was an artifact of sparse reference coding; show that modal recoding yields Max VIF = 1.80 without affecting Top-10 ranking; add epidemiological interpretation of CholCheck. |
| **B7 (Threshold Sensitivity)** | Sections 3.4, 5.1 | Report threshold sensitivity table for Recall constraints 0.75, 0.80, and 0.85 ($t = 0.16, 0.13, 0.10$). |
