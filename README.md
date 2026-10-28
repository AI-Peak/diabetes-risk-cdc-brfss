# Diabetes Risk Analytics with CDC BRFSS

Interpretable machine learning and statistical analysis of **253,680 CDC BRFSS 2015 survey records**, with development-only threshold selection, internal holdout evaluation, calibration assessment, SHAP explanations, and targeted sensitivity experiments.

**Repository maintainer:** [TunDec](https://github.com/BuildWithTunDec) · **Project:** DAP391M, FPT University

[Dataset description](docs/dataset_description.md) · [Data dictionary](docs/data_dictionary.md) · [Analysis approach](docs/analytic_approach.md)

## Research questions

1. Which health and demographic indicators are associated with reported prediabetes or diabetes?
2. How do Logistic Regression, Decision Tree, Random Forest and XGBoost compare, and how does a screening threshold change the precision–recall trade-off?
3. How closely do SHAP rankings align with multivariable likelihood-ratio contributions and univariate effect sizes?

## Recorded baseline results

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

These are recorded analysis outputs, not a claim of a new full training run during repository preparation. PR-AUC is reported as **average precision**. Class 0 means no reported diabetes; class 1 combines prediabetes and diabetes. Repeated response profiles are retained because matching survey responses cannot establish duplicate individuals.

## Revision experiments

The B1–B7 experiments are stored under `results/revision/` and cover duplicate-profile leakage, nested tuning, calibration uncertainty, decision-curve analysis, SHAP stability, VIF sensitivity, and threshold sensitivity. Each experiment contains its structured outputs and a short result summary. The corresponding source scripts are in `python_analysis/revision_*.py`, with the runner in `python_analysis/run_revision_experiments.py`.

The consolidated JSON, summary, environment manifest, and artifact hash manifest are in `results/revision/`. The manifests are provenance records for the stored outputs; they do not replace the underlying experiment artifacts.

## Repository structure

```text
python_analysis/  Data processing, statistics, modeling, SHAP and revision experiments
data/raw/         Original BRFSS-derived indicator data
data/processed/   Processed analysis data
results/          Baseline outputs and B1–B7 revision artifacts
docs/             Dataset documentation, research-question explanations and figures
sql/              SQL Server import, validation and EDA scripts and exports
tests/            Regression tests for the revision experiments
run_pipeline.py   Baseline analysis pipeline entry point
```

## Run the baseline analysis

Use **Python 3.13** (the project requires Python 3.12 or later) in an isolated environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
```

`requirements-lock.txt` preserves a previously recorded environment. A full run requires XGBoost, SHAP and the other listed dependencies and can take substantial time for cross-validation and bootstrap calculations.

To run the revision experiments:

```bash
python python_analysis/run_revision_experiments.py --experiment ALL
```

To run their regression tests:

```bash
pytest tests/test_revision_part1.py
```

SQL scripts target SQL Server. Set `<PATH_TO_CSV>` in `sql/scripts/00_import_data.sql` before importing.

## Data provenance and interpretation

The dataset is the BRFSS 2015-derived Diabetes Health Indicators dataset described in `docs/dataset_description.md`, attributed to CDC and the UCI/Kaggle distribution sources. No ownership of the underlying survey data is claimed. Consult the original source terms before redistribution or reuse.

This is an **unweighted, cross-sectional, self-reported survey analysis with internal validation**. It does not establish causality, prospective disease risk, clinical diagnostic validity or national prevalence. SHAP explains the fitted model; it does not identify causal effects. Calibration assessment does not itself recalibrate predictions. External validation remains necessary.

## Repository history

This repository starts from a new snapshot of the project. Its Git history records maintenance of this snapshot; earlier development history is not imported. Scientific authorship and third-party attribution belong to the underlying project and dataset sources.
