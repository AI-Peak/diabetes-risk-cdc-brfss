# Diabetes Risk Analytics with CDC BRFSS

Interpretable machine learning and statistical analysis of **253,680 CDC BRFSS 2015 survey records**, with development-only threshold selection, internal holdout evaluation, calibration assessment and SHAP explanations.

**Repository maintainer:** [TunDec](https://github.com/BuildWithTunDec) · **Project:** DAP391M, FPT University

[Manuscript](paper/main.tex) · [Methods and outputs](docs/paper_ready_outputs.md) · [Dataset description](docs/dataset_description.md) · [Data dictionary](docs/data_dictionary.md)

## Research questions

1. Which health and demographic indicators are associated with reported prediabetes or diabetes?
2. How do Logistic Regression, Decision Tree, Random Forest and XGBoost compare, and how does a screening threshold change the precision–recall trade-off?
3. How closely do SHAP rankings align with multivariable likelihood-ratio contributions and univariate effect sizes?

## Recorded results

| Measure | Result |
|---|---:|
| Observations / predictors | 253,680 / 21 |
| Positive-class prevalence | 13.93% |
| Development / holdout records | 202,944 / 50,736 |
| Selected model | XGBoost |
| Mean development CV average precision | 0.4359 |
| Holdout average precision / ROC-AUC | 0.4238 / 0.8272 |
| Development-selected threshold | 0.13 |
| Holdout recall / precision at 0.13 | 80.99% / 29.91% |
| Holdout Brier score | 0.0974 |
| Calibration slope / intercept | 0.9589 / -0.0515 |
| SHAP–likelihood-ratio Spearman correlation | 0.9338 |

These are the recorded analysis outputs, not a claim of a new full training run during repository preparation. PR-AUC is reported as **average precision**. Class 0 means no reported diabetes; class 1 combines prediabetes and diabetes. Repeated response profiles are retained because matching survey responses cannot establish duplicate individuals.

## Repository structure

```text
paper/            Latest reviewed manuscript, bibliography and six final figures
python_analysis/  Data processing, statistics, model training, SHAP and validation
notebooks/        Interactive data-understanding notebook
data/raw/         Original BRFSS-derived indicator data
data/processed/   Processed analysis data
results/          Recorded metrics, trained model, SHAP outputs and summaries
docs/             Dataset documentation, RQ explanations and analysis figures
sql/              SQL Server import/validation/EDA scripts and exports
run_pipeline.py   Analysis pipeline entry point
```

## Run the analysis

Use **Python 3.13** (syntax requires Python 3.12 or later) and an isolated environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
```

`requirements-lock.txt` preserves previously recorded package versions. Its installation in a clean environment has not yet been verified. A full run requires XGBoost, SHAP and all other listed dependencies, and can take substantial time for cross-validation and bootstrap calculations.

To check existing outputs without retraining:

```bash
python python_analysis/validate_outputs.py
```

Validation checks data dimensions, required artifacts, metric consistency and manuscript checksums. It does not independently validate every scientific claim. The threshold is chosen from development out-of-fold predictions; the final holdout is used for evaluation. Effect-size plotting runs once inside statistical analysis.

SQL scripts target SQL Server. Set `<PATH_TO_CSV>` in `sql/scripts/00_import_data.sql` before importing.

## Manuscript

`paper/` contains the revised manuscript from **Project_2_DAP (4).zip**, with the supplied text and six figures preserved. Research authorship is listed in the manuscript; repository commit attribution is separate from paper authorship.

With Tectonic installed:

```bash
cd paper
tectonic main.tex
```

The analysis pipeline does not overwrite approved manuscript text or figures. Publication tables and selected figures are maintained separately from generated analysis outputs. `paper/final_source_manifest.json` verifies the supplied source bytes.

### Review status

- The revised manuscript compiled successfully to 13 pages during review.
- Revised odds ratios, VIF, calibration and holdout counts were checked against stored outputs.
- Full model retraining and fresh-environment reproduction remain unverified.
- Before submission, review the manuscript's statement about regenerating every figure/table, confirm its AI-use declaration, and update its older repository URL. The manuscript itself is preserved as supplied rather than silently edited.
- Check the target venue's template, author information and disclosure requirements.

## Data provenance and interpretation

The dataset is the BRFSS 2015-derived Diabetes Health Indicators dataset described in `docs/dataset_description.md` and the manuscript bibliography, attributed to CDC and the UCI/Kaggle distribution sources. No ownership of the underlying survey data is claimed. Consult the original source terms before redistribution or reuse.

This is an **unweighted, cross-sectional, self-reported survey analysis with internal validation**. It does not establish causality, prospective disease risk, clinical diagnostic validity or national prevalence. SHAP explains the fitted model; it does not identify causal effects. Calibration assessment does not itself recalibrate predictions. External validation remains necessary.

## Repository history

This repository starts from a new snapshot of the project. Its Git history records maintenance of this snapshot; earlier development history is not imported. Existing scientific authorship and third-party attribution are retained.
