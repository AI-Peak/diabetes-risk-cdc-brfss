# Experiment B3: Calibration Analysis with Bootstrap Confidence Intervals

## 1. Primary Empirical Calibration Metrics
Evaluated on the untouched holdout test partition (N = 50,736):
- **Brier Score:** 0.0974 (95% Percentile CI: [0.0964, 0.0983])
  - *Manuscript Baseline:* 0.0974 [0.0964, 0.0983] -> **Exact replication verified.**
- **Calibration Slope:** 0.9589 (95% Percentile CI: [0.9342, 0.9854])
  - *Manuscript Baseline:* 0.9589 [0.9342, 0.9854] -> **Exact replication verified.**
  - Slope is close to 1.0, reflecting well-scaled risk gradient without severe over-dispersion.
- **Calibration Intercept:** -0.0515 (95% Percentile CI: [-0.0861, -0.0148])
  - *Manuscript Baseline:* -0.0515 [-0.0861, -0.0148] -> **Exact replication verified.**
  - Slight negative intercept indicates a minor tendency toward mild overestimation in low-risk brackets.

## 2. Reliability Curve Binning
Both equal-width (10 intervals across [0, 1]) and equal-frequency (deciles) binning schemes were computed and saved to `calibration_bins.csv`.
In equal-frequency binning, predicted probabilities closely track observed event rates across all 10 risk strata.

## 3. Bootstrap Estimation Integrity
- **Replicates:** 1,000 stratified resamples.
- **Successful Fits:** 1,000 (100.0%).
- **Failed Fits:** 0 (No numerical singularities or infinite logit occurrences; `eps = 1e-15` properly contained boundary extremes).

## 4. Visual Artifacts
- **High-Resolution Raster:** `results/revision_part1/figures/b3_calibration_reliability_curve.png`
- **Vector Graphics:** `results/revision_part1/figures/b3_calibration_reliability_curve.pdf`
- **Structured Data:** `results/revision_part1/b3_calibration/calibration_plot_data.json`
