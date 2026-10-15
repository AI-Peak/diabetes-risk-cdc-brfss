# Statistical Hypothesis Testing & Adjusted Association Report
## CDC Diabetes Health Indicators (Cleaned Dataset)

### Methodology: CRISP-DM (Exploratory & Statistical Analysis)
**Author:** Senior Data Analytics Engineer & Team Members  
**Date:** 2026-07-21  
**Project:** Diabetes-Analytics  
**Objective:** Answer RQ1: *Which demographic, lifestyle, and health-related factors are statistically associated with diabetes status in the CDC BRFSS 2015 sample?*

---

## 1. Introduction
This report presents the quantitative statistical evaluation of the relationships between 21 health indicators and diabetes status (`Diabetes_binary`). The analysis is conducted on the full dataset of **253,680** survey respondents (retaining repeated feature profiles to preserve natural sample distribution).

To ensure statistical rigor, we apply:
1. **Chi-Square Test of Independence** for categorical, binary, and ordinal variables.
2. **Cramér's V** to measure effect size for categorical associations.
3. **Independent Two-Sample Welch t-Test** (parametric mean comparison) and **Mann-Whitney U Test** (non-parametric median/distribution comparison) for continuous numerical variables.
4. **Absolute Rank-Biserial Correlation** (primary) and **Cohen's d** (secondary) to measure numerical effect sizes.
5. **Holm–Bonferroni Multiple Testing Correction**: Holm adjustment was applied across the prespecified primary association tests: Chi-square tests for categorical features and Mann–Whitney U tests for numerical features. Welch’s t-tests were retained as complementary sensitivity analyses.
6. **Multivariable Adjusted Association Analysis**: Multivariable Logistic Regression with categorical indicator blocks for ordinal/multiclass predictors (`GenHlth`, `Age`, `Education`, `Income`), reporting term-level AORs, 95% Confidence Intervals, and feature-level nested Likelihood-Ratio tests ($\Delta\text{deviance}$ / LR $\chi^2$) to assess conditional feature contributions while controlling for all co-occurring indicators.

> **Methodological Note on Large Sample Size:**  
> With *N* = 253,680, statistical tests possess near-infinite power, causing p-values for almost all predictors to drop below $p < 0.05$. Therefore, p-values are reported alongside Holm-adjusted values for formal hypothesis testing, but **practical feature importance is evaluated by Effect Size within each feature family and feature-level Likelihood-Ratio $\chi^2$ ($\Delta\text{deviance}$)**.

---

## 2. Categorical Variable Analysis (Chi-Square & Cramér's V)

### Contingency Table & Chi-Square Summary (with Holm-Bonferroni Correction)
| Variable Name | Attribute Description | Chi-Square ($\chi^2$) | Raw p-value | Holm-Adjusted p | Reject $H_0$ | df | Cramér's V | Effect Size Interpretation | Max Prevalence Diff |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `GenHlth` | Self-Rated General Health | 22728.07 | `< 1.00e-300` | `< 1.00e-300` | Yes | 4 | 0.2993 | **Moderate** | 35.38% |
| `HighBP` | High Blood Pressure | 17562.45 | `< 1.00e-300` | `< 1.00e-300` | Yes | 1 | 0.2631 | **Moderate** | 18.41% |
| `DiffWalk` | Difficulty Walking or Climbing Stairs | 12092.32 | `< 1.00e-300` | `< 1.00e-300` | Yes | 1 | 0.2183 | **Moderate** | 20.21% |
| `HighChol` | High Cholesterol | 10174.07 | `< 1.00e-300` | `< 1.00e-300` | Yes | 1 | 0.2003 | **Moderate** | 14.03% |
| `Age` | Age Category (13 levels) | 8795.05 | `< 1.00e-300` | `< 1.00e-300` | Yes | 12 | 0.1862 | **Small** | 20.48% |
| `HeartDiseaseorAttack` | Heart Disease or Attack History | 7971.16 | `< 1.00e-300` | `< 1.00e-300` | Yes | 1 | 0.1773 | **Small** | 21.02% |
| `Income` | Income Bracket (8 levels) | 7003.72 | `< 1.00e-300` | `< 1.00e-300` | Yes | 7 | 0.1662 | **Small** | 18.23% |
| `Education` | Education Level (6 levels) | 4027.11 | `< 1.00e-300` | `< 1.00e-300` | Yes | 5 | 0.1260 | **Small** | 19.57% |
| `PhysActivity` | Physical Activity Indicator | 3539.42 | `< 1.00e-300` | `< 1.00e-300` | Yes | 1 | 0.1181 | **Small** | 9.53% |
| `Stroke` | Stroke History | 2838.92 | `< 1.00e-300` | `< 1.00e-300` | Yes | 1 | 0.1058 | **Small** | 18.57% |
| `CholCheck` | Cholesterol Check (5 Years) | 1062.94 | `3.75e-233` | `3.38e-232` | Yes | 1 | 0.0647 | **Weak** | 11.83% |
| `Smoker` | Tobacco Smoker Status | 937.06 | `8.64e-206` | `6.91e-205` | Yes | 1 | 0.0608 | **Weak** | 4.24% |
| `HvyAlcoholConsump` | Heavy Alcohol Consumption | 825.12 | `1.87e-181` | `1.31e-180` | Yes | 1 | 0.0570 | **Weak** | 8.58% |
| `Veggies` | Vegetable Consumption Daily | 811.81 | `1.46e-178` | `8.78e-178` | Yes | 1 | 0.0566 | **Weak** | 5.01% |
| `Fruits` | Fruit Consumption Daily | 421.61 | `1.09e-93` | `5.44e-93` | Yes | 1 | 0.0408 | **Negligible** | 2.93% |
| `Sex` | Biological Sex | 250.41 | `2.11e-56` | `6.33e-56` | Yes | 1 | 0.0314 | **Negligible** | 2.19% |
| `NoDocbcCost` | Doctor Cost Barrier | 250.31 | `2.22e-56` | `6.33e-56` | Yes | 1 | 0.0314 | **Negligible** | 3.92% |
| `AnyHealthcare` | Healthcare Coverage Access | 66.81 | `2.99e-16` | `2.99e-16` | Yes | 1 | 0.0162 | **Negligible** | 2.61% |

### Key Findings from Categorical Analysis:
1. **Strongest Marginal Associated Feature**: **`GenHlth`** (Self-Rated General Health) exhibits the strongest univariate association within the analyzed BRFSS sample with diabetes status (*V* = **0.2993**), showing a **35.38%** difference in prevalence across health levels.
2. **Survey-based Health Indicators**: General Health (`GenHlth`, *V* = 0.2993), High Blood Pressure (`HighBP`, *V* = 0.2631), High Cholesterol (`HighChol`, *V* = 0.2003), and Difficulty Walking (`DiffWalk`, *V* = 0.2183) represent the most salient marginal indicators.
3. **Behavioral Features**: Physical activity (`PhysActivity`, *V* = 0.1181) and fruit/vegetable intake show weak direct correlations (*V* < 0.10).
4. **Demographics**: Biological sex (`Sex`, *V* = 0.0314) exhibits minimal marginal association with diabetes prevalence.

---

## 3. Numerical Variable Analysis (t-Test & Mann-Whitney U)

### Numerical Tests Summary
| Variable | No Reported Diabetes Mean | Prediabetes/Diabetes Positive Mean | Mean Diff | No Reported Diabetes Median | Prediabetes/Diabetes Median | t-Stat | Raw t p-val | MWU p-val | Holm MWU p | Reject $H_0$ | Cohen's d | Abs Rank-Biserial | Effect Size Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BMI` | 27.81 | 31.94 | 4.14 | 27.0 | 31.0 | 99.92 | `< 1.00e-300` | `< 1.00e-300` | `< 1.00e-300` | Yes | 0.6414 | 0.3766 | **Moderate** |
| `MentHlth` | 2.98 | 4.46 | 1.48 | 0.0 | 0.0 | 29.69 | `7.93e-192` | `1.75e-90` | `7.00e-90` | Yes | 0.2006 | 0.0546 | **Negligible** |
| `PhysHlth` | 3.64 | 7.95 | 4.31 | 0.0 | 1.0 | 68.97 | `< 1.00e-300` | `< 1.00e-300` | `< 1.00e-300` | Yes | 0.5022 | 0.2260 | **Small** |

### Key Findings from Numerical Analysis:
1. **Body Mass Index (BMI)**: Mean BMI for the group without reported diabetes is **27.81** vs **31.94** for the prediabetes/diabetes positive group. Absolute Rank-Biserial correlation is **0.3766** (Cohen's d = **0.6414**), confirming a moderate practical effect size within the sample.
2. **Physical Unhealthy Days (`PhysHlth`)**: Respondents in the prediabetes/diabetes positive class report an average of **7.95** unhealthy physical days in the past 30 days compared to **3.64** days for respondents without reported diabetes.

---

## 4. Multivariable Adjusted Association Analysis (Logistic Regression)

To evaluate adjusted associations while controlling for co-occurring indicators, a multivariable logistic regression model was estimated. Categorical and ordinal features (`GenHlth`, `Age`, `Education`, `Income`) were modeled using categorical indicator blocks with the lowest category as reference. Feature-level statistical contributions were evaluated via nested Likelihood-Ratio $\chi^2$ tests ($\Delta\text{deviance}$).

### Feature-Level Likelihood-Ratio Contributions (Nested Model Comparison)
| Rank | Variable Name | Description | LR $\chi^2$ ($\Delta$deviance) | df | Raw p-value | Holm-Adjusted p | Reject $H_0$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `GenHlth` | Self-Rated General Health | 4941.50 | 4 | `< 1.00e-300` | `< 1.00e-300` | Yes |
| 2 | `BMI` | Body Mass Index | 3986.04 | 1 | `< 1.00e-300` | `< 1.00e-300` | Yes |
| 3 | `Age` | Age Category (13 levels) | 2869.38 | 12 | `< 1.00e-300` | `< 1.00e-300` | Yes |
| 4 | `HighBP` | High Blood Pressure | 2448.45 | 1 | `< 1.00e-300` | `< 1.00e-300` | Yes |
| 5 | `HighChol` | High Cholesterol | 1572.34 | 1 | `< 1.00e-300` | `< 1.00e-300` | Yes |
| 6 | `HvyAlcoholConsump` | Heavy Alcohol Consumption | 476.94 | 1 | `9.89e-106` | `1.58e-104` | Yes |
| 7 | `CholCheck` | Cholesterol Check (5 Years) | 452.13 | 1 | `2.48e-100` | `3.72e-99` | Yes |
| 8 | `Sex` | Biological Sex | 391.04 | 1 | `4.91e-87` | `6.88e-86` | Yes |
| 9 | `Income` | Income Bracket (8 levels) | 297.98 | 7 | `1.63e-60` | `2.12e-59` | Yes |
| 10 | `HeartDiseaseorAttack` | Heart Disease or Attack History | 201.68 | 1 | `9.00e-46` | `1.08e-44` | Yes |
| 11 | `DiffWalk` | Difficulty Walking or Climbing Stairs | 73.65 | 1 | `9.31e-18` | `1.02e-16` | Yes |
| 12 | `Education` | Education Level (6 levels) | 49.61 | 5 | `1.67e-09` | `1.50e-08` | Yes |
| 13 | `Stroke` | Stroke History | 41.07 | 1 | `1.47e-10` | `1.47e-09` | Yes |
| 14 | `PhysHlth` | Physical Unhealthy Days | 18.73 | 1 | `1.50e-05` | `1.20e-04` | Yes |
| 15 | `PhysActivity` | Physical Activity Indicator | 16.57 | 1 | `4.69e-05` | `3.29e-04` | Yes |
| 16 | `MentHlth` | Mental Unhealthy Days | 11.43 | 1 | `7.22e-04` | `4.33e-03` | Yes |
| 17 | `Smoker` | Tobacco Smoker Status | 9.25 | 1 | `2.36e-03` | `1.18e-02` | Yes |
| 18 | `AnyHealthcare` | Healthcare Coverage Access | 4.95 | 1 | `2.61e-02` | `1.05e-01` | No |
| 19 | `Veggies` | Vegetable Consumption Daily | 3.36 | 1 | `6.68e-02` | `2.00e-01` | No |
| 20 | `Fruits` | Fruit Consumption Daily | 2.86 | 1 | `9.06e-02` | `2.00e-01` | No |
| 21 | `NoDocbcCost` | Doctor Cost Barrier | 0.11 | 1 | `7.46e-01` | `7.46e-01` | No |

### Key Findings from Multivariable Analysis:
1. **Top Feature Contributions by $\Delta\text{deviance}$**: `GenHlth` (LR $\chi^2$ = 4941.50, df = 4), `BMI` (LR $\chi^2$ = 3986.04, df = 1), `Age` (LR $\chi^2$ = 2869.38, df = 12), `HighBP` (LR $\chi^2$ = 2448.45, df = 1), and `HighChol` (LR $\chi^2$ = 1572.34, df = 1) contribute the largest likelihood gains to the multivariable model.
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
