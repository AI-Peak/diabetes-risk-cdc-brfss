# Experiment B4: Decision Curve Analysis (DCA)

## 1. Clinical Context and Methodology
Decision Curve Analysis evaluates the clinical utility of probabilistic screening models by quantifying **Net Benefit** across varying risk preferences ($p_t \in [0.05, 0.40]$):
$$\text{Net Benefit} = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left(\frac{p_t}{1 - p_t}\right)$$
Where $p_t$ represents the threshold probability at which an individual or clinician is indifferent between false-positive testing burden and false-negative missed detection.

## 2. Key Findings
- **Prevalence in Evaluation Set:** 13.93% (7,069 / 50,736).
- **At Prespecified Threshold ($p_t = 0.13$):**
  - **XGBoost Net Benefit:** `0.0733`
  - **Screen All Strategy:** `0.0107`
  - **Screen None Strategy:** `0.0000`
  - **Confusion Matrix at $p_t = 0.13$:** TP = 5,725, FP = 13,416, TN = 30,251, FN = 1,344 (Recall = 80.99%, Missed Positive Cases = 1,344 [19.01%]).
  - **Clinical Trade-off:** The model operates under a decision-analytic exchange rate where false positives are penalized by $rac{p_t}{1 - p_t} = rac{0.13}{0.87} pprox 0.1494$. The net benefit gain of `+0.0626` over *Screen All* equates mathematically to avoiding approximately `41.9` unnecessary workups per 100 individuals relative to mass screening. **Crucially, this does not imply zero missed cases**: 1,344 positive individuals remain undetected at this operating point.

## 3. Threshold Range Dynamics
- Across the entire evaluated grid from $p_t = 0.05$ to $p_t = 0.40$, the XGBoost model strictly dominates both the *Screen All* and *Screen None* default strategies.
- At low thresholds (e.g., $p_t = 0.05$), the model achieves a Net Benefit of `0.1072` compared to `0.0940` for *Screen All* (+0.0131 higher).
- As threshold increases toward population prevalence ($\sim 0.139$), the Net Benefit of *Screen All* diminishes rapidly and crosses below zero at $p_t \approx 0.14$. In contrast, XGBoost maintains positive Net Benefit throughout ($0.0733$ at $p_t = 0.13$, $0.0543$ at $p_t = 0.20$, and $0.0117$ at $p_t = 0.40$).

## 4. Methodological Qualifications and Limitations
This dataset relies on self-reported survey responses from CDC BRFSS without formal clinical biochemical validation (HbA1c / fasting glucose). Decision Curve Analysis here illustrates *decision-analytic value* rather than prospective clinical guideline utility.
