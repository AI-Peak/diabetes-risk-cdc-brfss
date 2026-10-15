# Final paper output map

The user-approved final source is `paper/main.tex`, copied unchanged from `Project_2_DAP (4).zip`. All paper tables are embedded in this source; there is no external generated table or metric-macro dependency.

## Six final figures

| Final image in paper/images/ | Role | Related analysis |
|---|---|---|
| pipeline_method.png | Final workflow diagram | Preserved from the approved ZIP |
| effect_size_ranking.png | Univariate effect-size ranking | statistical_analysis.py and generate_effect_size_figure.py |
| cv_model_comparison.png | Four-model cross-validation comparison | model_training.py; results/modeling/cv_model_comparison.csv |
| holdout_roc_pr_curves.png | Holdout ROC and PR curves | model_training.py; results/modeling/ |
| shap_summary_beeswarm.png | SHAP summary | shap_analysis.py; results/xai/ |
| effect_size_shap_alignment.png | SHAP/statistical rank alignment | shap_analysis.py; results/xai/explanation_consistency.csv |

Final images are not automatically replaced by regenerated figures. In particular, the final CV plot has a different binary representation from the previous generated project plot.

## Retained evidence

- Data validation and repeated-record counts: results/data_understanding/ and results/data_preprocessing/.
- Statistical tests and adjusted associations: results/statistical_analysis/.
- Cross-validation, threshold selection, holdout evaluation and calibration: results/modeling/.
- SHAP importance, multivariable/univariate rank comparisons and rank sensitivity: results/xai/.
- Supporting exploratory plots remain useful for explaining the analysis even when not inserted into the final paper.

The old profile-grouped redevelopment experiment is not part of this final manuscript. Its dedicated outputs and old ten-table manuscript generation workflow were removed. Existing historical environment metadata describes the earlier run and is not evidence that the cleaned pipeline has been retrained.

## Verification boundary

The cleanup checks source-file hashes, paths and code syntax. The final manuscript's substantive claims are unchanged. Any numerical or methodological discrepancy between the manuscript and analysis outputs requires separate scientific review; deletion is not a correction of such discrepancies.
