# Experiment B6: Rare Categories, Logistic Regression, and VIF Sensitivity

## 1. Categorical Distribution & Operational Sparsity Audit
- **Rarity Criterion:** A category is defined as sparse/rare if sample count $< 1,000$ or cohort proportion $< 0.5\%$.
- **Education:** Level 1 ("Never attended school or only kindergarten") represents only **174 out of 253,680** observations (0.069\%). It meets the sparsity threshold and was merged into Level 2 (forming an "Elementary or less" stratum with $N = 4,217 total observations: 174 from Level 1 + 4,043 from Level 2).
- **Income:** Lowest bracket (Level 1, $<\$10,000$) contains **9,811** observations (3.87\%); Level 2 contains **11,783** (4.64\%); both exceed the rarity threshold by nearly an order of magnitude. Thus, category merging is neither necessary nor justified.
- **Reference Allocation:** Modal categories were assigned as unpenalized reference strata:
  - `Education`: Modal category **Level 6** (College 4+ years, 107,325 rows, 42.31\%) set as reference.
  - `Income`: Modal category **Level 8** ($\$75,000+$, 90,385 rows, 35.63\%) set as reference.
  - `GenHlth`: Modal category **Level 2** (Very Good) set as reference.
  - `Age`: Modal category **Level 10** (Age 65-69) set as reference.

## 2. Impact on Multicollinearity (VIF)
- **Original Model Maximum VIF:** `357.58` (severe multicollinearity artifact due to sparse Education Level 1 reference).
- **Revised Model Maximum VIF:** `1.80` (on `PhysHlth`).
- **Outcome:** Shifting the reference baseline from the sparse tail (Level 1, N=174) to the substantial modal category (Level 6, N=107,325) completely resolved dummy collinearity, reducing maximum VIF by **over 99%** (from 357.58 down to 1.80).

## 3. Income Sensitivity Evaluation (Distinct vs Collapsed Levels 1 & 2)
- Collapsing Income Levels 1 & 2 yields a maximum VIF of `1.80` on `PhysHlth`.
- Top-10 predictor overlap between distinct and collapsed income models: **10/10**.
- Predictor deviance Spearman rank correlation: $r_s = 1.0000$.
- Preserving distinct income categories provides greater socioeconomic granularity without any statistical penalty.

## 4. Feature Ranking Stability & SHAP Alignment
- **Feature Contributions:** Measured by nested block likelihood-ratio tests ($\Delta \text{deviance}$).
- **Top-10 Overlap:**
  - Original LR vs Revised LR Top-10: **10/10** (Jaccard = 1.0000).
  - Revised LR vs Baseline SHAP Top-10: **10/10** (Jaccard = 1.0000).
- **21-Predictor Spearman Correlation:** $r_s = 0.9338$ ($p = 6.38e-10$).
- *Conclusion:* While dummy-level parameter interpretations shifted relative to the modal reference, feature-level statistical significance and concordance with machine learning SHAP attribution remained robust.

## 5. Epidemiological Context of CholCheck (AOR ~ 3.44)
`CholCheck` (AOR = 3.44, 95% CI: [3.01, 3.93]) exhibits a strong positive adjusted association. Rather than implying that cholesterol testing causes diabetes, this finding exemplifies **healthcare engagement / diagnostic detection bias**: adults undergoing routine preventative screenings are substantially more likely to receive an accurate diagnosis of chronic, often asymptomatic cardiometabolic conditions.
