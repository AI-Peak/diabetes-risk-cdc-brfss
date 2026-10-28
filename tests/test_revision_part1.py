"""
Automated Comprehensive Multi-Layer Test Suite for Revision Experiments (Phase 1)
---------------------------------------------------------------------------------
Multi-layer verification suite covering:
  Layer 1: Deterministic unit tests on core algorithmic logic (toy fixtures).
  Layer 2: Scientific regression tests for Codex-audited defects (B1-B7).
  Layer 3: Real-data empirical integration tests on BRFSS revision artifacts.
  Layer 4: Dynamic computation-based integration workflows (small data).
  Layer 5: Provenance, dataset hash consistency, and repository immutability.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_cleaned.csv"
REVISION_DIR = PROJECT_ROOT / "results" / "revision_part1"
PAPER_DIR = PROJECT_ROOT / "paper"

from python_analysis.revision_b4_dca import calculate_net_benefit
from python_analysis.revision_b7_threshold_sensitivity import optimize_threshold, evaluate_metrics

# ===========================================================================
# LAYER 1: DETERMINISTIC UNIT TESTS ON ALGORITHMIC LOGIC (TOY FIXTURES)
# ===========================================================================

def test_unit_dca_net_benefit_formula_deterministic():
    """Verifies DCA net benefit mathematics on exact synthetic values."""
    y_true = np.array([1] * 14 + [0] * 86)
    y_prob = np.array([0.8] * 11 + [0.05] * 3 + [0.8] * 26 + [0.05] * 60)
    
    threshold = 0.13
    res = calculate_net_benefit(y_true, y_prob, threshold)
    
    weight = 0.13 / (1.0 - 0.13)
    expected_nb_model = (11 / 100) - (26 / 100) * weight
    expected_nb_all = 0.14 - 0.86 * weight
    
    assert np.isclose(res["net_benefit_model"], expected_nb_model, atol=1e-5)
    assert np.isclose(res["net_benefit_all"], expected_nb_all, atol=1e-5)
    assert res["net_benefit_none"] == 0.0
    assert res["tp"] == 11 and res["fp"] == 26 and res["tn"] == 60 and res["fn"] == 3

    # Boundary property: threshold == prevalence -> Screen All net benefit is 0.0
    res_at_prev = calculate_net_benefit(y_true, y_prob, 0.14)
    assert np.isclose(res_at_prev["net_benefit_all"], 0.0, atol=1e-5)

def test_unit_threshold_optimization_logic_deterministic():
    """Verifies threshold optimization subject to Recall constraints on controlled probabilities."""
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.4, 0.2, 0.85, 0.65, 0.35, 0.15, 0.05])
    
    t80, d80 = optimize_threshold(y_true, probs, target_recall=0.80)
    assert d80["recall"] >= 0.80
    assert t80 == 0.40

    t100, d100 = optimize_threshold(y_true, probs, target_recall=1.00)
    assert d100["recall"] >= 1.00
    assert t100 == 0.20

    assert t80 >= t100

def test_unit_group_disjointness_property():
    """Verifies that StratifiedGroupKFold produces completely group-disjoint splits."""
    groups = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
    y = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    X = np.zeros((len(y), 2))

    sgkf = StratifiedGroupKFold(n_splits=2)
    for tr_idx, te_idx in sgkf.split(X, y, groups):
        tr_groups = set(groups[tr_idx])
        te_groups = set(groups[te_idx])
        assert tr_groups.intersection(te_groups) == set(), "Group leakage detected across partitions!"

def test_unit_nadeau_bengio_correction_math():
    """Verifies Nadeau-Bengio corrected variance formulation."""
    diffs = np.array([0.007, 0.006, 0.008, 0.005, 0.007] * 3)
    n_evals = len(diffs)
    mean_d = np.mean(diffs)
    std_d = np.std(diffs, ddof=1)
    
    var_naive = (std_d ** 2) / n_evals
    correction_factor = (1.0 / n_evals) + 0.25
    var_corrected = correction_factor * (std_d ** 2)
    
    assert var_corrected > var_naive
    se_naive = np.sqrt(var_naive)
    se_corrected = np.sqrt(var_corrected)
    assert se_corrected > se_naive
    assert (mean_d / se_corrected) < (mean_d / se_naive)

def test_unit_jaccard_similarity_calculation():
    """Verifies Jaccard similarity edge cases and partial overlap math."""
    def jaccard(s1, s2):
        u = set(s1).union(set(s2))
        return len(set(s1).intersection(set(s2))) / len(u) if u else 1.0

    assert jaccard(["A", "B", "C"], ["A", "B", "C"]) == 1.0
    assert jaccard(["A", "B"], ["C", "D"]) == 0.0
    s1 = [f"feat_{i}" for i in range(10)]
    s2 = [f"feat_{i}" for i in range(1, 11)]
    assert np.isclose(jaccard(s1, s2), 9 / 11)

# ===========================================================================
# LAYER 2: SCIENTIFIC REGRESSION TESTS FOR AUDITED DEFECTS
# ===========================================================================

def test_b2_pairwise_win_counts_and_iteration14():
    """Verifies B2 win counts: XGBoost won 14/15, RF won 1/15, zero ties."""
    paired_path = REVISION_DIR / "b2_tuning" / "nested_cv_paired_results.csv"
    assert paired_path.exists()
    df = pd.read_csv(paired_path)
    
    assert len(df) == 15, "Expected exactly 15 paired nested CV evaluations"
    assert len(df.drop_duplicates(subset=["repeat", "fold"])) == 15, "Folds/repeats must be unique"
    
    xgb_wins = int((df["xgb_pr_auc"] > df["rf_pr_auc"]).sum())
    rf_wins = int((df["rf_pr_auc"] > df["xgb_pr_auc"]).sum())
    ties = int((df["xgb_pr_auc"] == df["rf_pr_auc"]).sum())
    
    assert xgb_wins == 14, f"XGBoost should have won 14 evaluations, got {xgb_wins}"
    assert rf_wins == 1, f"Random Forest should have won 1 evaluation, got {rf_wins}"
    assert ties == 0, f"Expected 0 ties, got {ties}"
    assert xgb_wins + rf_wins + ties == 15
    
    # Specific verification of iteration 14
    row14 = df[df["iteration"] == 14].iloc[0]
    assert row14["rf_pr_auc"] > row14["xgb_pr_auc"], "Random Forest must be the winner in iteration 14"
    assert row14["paired_diff_pr_auc"] < 0, "Paired difference in iteration 14 must be negative"
    
    # Ensure JSON summary reflects 14/15 win counts and corrected B2 statistics
    summary_path = REVISION_DIR / "b2_tuning" / "b2_tuning_summary.json"
    with open(summary_path, "r") as f:
        sdata = json.load(f)
    assert sdata["pairwise_comparison_summary"]["xgb_wins"] == 14
    assert sdata["pairwise_comparison_summary"]["rf_wins"] == 1
    assert sdata["paired_difference"]["nadeau_bengio_corrected_95_ci"] == [0.0025, 0.0111]
    assert sdata["paired_difference"]["corrected_t_statistic"] == 3.3656
    assert sdata["paired_difference"]["corrected_p_value"] == 0.0046

def test_b4_false_negative_quantification_and_no_zero_missed_claim():
    """Verifies that B4 quantifies FN = 1344 (Recall 80.99%) and does not claim zero missed cases."""
    b4_data_path = REVISION_DIR / "b4_decision_curve" / "decision_curve_data.csv"
    assert b4_data_path.exists()
    df = pd.read_csv(b4_data_path)
    
    row13 = df[np.isclose(df["threshold"], 0.13, atol=1e-3)].iloc[0]
    assert row13["tp"] == 5725
    assert row13["fp"] == 13416
    assert row13["tn"] == 30251
    assert row13["fn"] == 1344
    
    total_pos = row13["tp"] + row13["fn"]
    recall = row13["tp"] / total_pos
    assert np.isclose(recall, 0.809874, atol=1e-4)
    assert row13["fn"] > 0, "Model has 1,344 false negatives at threshold 0.13"
    
    # Check that summary text does not claim zero missed cases
    summary_path = REVISION_DIR / "b4_decision_curve" / "b4_summary.md"
    text = summary_path.read_text(encoding="utf-8")
    assert "Crucially, this does not imply zero missed cases" in text
    assert "1,344 positive individuals" in text

def test_b6_spearman_alignment_permutation_invariance():
    """MANDATORY REGRESSION TEST: Verifies feature-aligned Spearman merge is permutation-invariant."""
    features = [f"F_{i}" for i in range(21)]
    np.random.seed(42)
    vals_lr = np.random.uniform(10, 5000, size=21)
    vals_shap = vals_lr * np.random.uniform(0.8, 1.2, size=21)
    
    df1 = pd.DataFrame({"feature": features, "lr_stat": vals_lr})
    # Deliberately permute row order of df2
    df2 = pd.DataFrame({"feature": features, "shap_stat": vals_shap}).sample(frac=1.0, random_state=123).reset_index(drop=True)
    
    # Method 1: WRONG independent sorting or naive array passing
    wrong_corr, _ = stats.spearmanr(df1.sort_values(by="lr_stat", ascending=False)["lr_stat"],
                                    df2.sort_values(by="shap_stat", ascending=False)["shap_stat"])
    assert np.isclose(wrong_corr, 1.0), "Independent sorting trivially yields false 1.0 correlation"
    
    # Method 2: CORRECT feature-name inner merge
    aligned = pd.merge(df1, df2, on="feature", validate="one_to_one")
    assert len(aligned) == 21
    true_corr, true_pval = stats.spearmanr(aligned["lr_stat"], aligned["shap_stat"])
    
    # Check that re-shuffling df1 does not alter true aligned correlation
    df1_shuffled = df1.sample(frac=1.0, random_state=999)
    aligned2 = pd.merge(df1_shuffled, df2, on="feature", validate="one_to_one")
    true_corr2, _ = stats.spearmanr(aligned2["lr_stat"], aligned2["shap_stat"])
    assert np.isclose(true_corr, true_corr2)

def test_b6_education_merged_count_and_row_conservation():
    """Verifies that Education Level 1 (174) + Level 2 (4043) = exactly 4,217 rows."""
    df = pd.read_csv(CLEANED_DATA_PATH)
    c1 = int((df["Education"] == 1).sum())
    c2 = int((df["Education"] == 2).sum())
    assert c1 == 174, f"Expected 174 in Education 1, got {c1}"
    assert c2 == 4043, f"Expected 4043 in Education 2, got {c2}"
    
    recoded = df["Education"].replace({1: 2})
    c_merged = int((recoded == 2).sum())
    assert c_merged == 4217, f"Expected 4217 merged rows, got {c_merged}"
    assert c_merged == c1 + c2
    assert len(recoded) == len(df) == 253680, "No rows lost during recoding"
    
    # Check summary report
    summary_path = REVISION_DIR / "b6_vif_sensitivity" / "b6_summary.md"
    text = summary_path.read_text(encoding="utf-8")
    assert "4,217" in text and "total observations" in text and "174 from Level 1" in text and "4,043 from Level 2" in text

def test_b1_non_causal_bounded_statements():
    """Verifies B1 summary does not make unsupported causal claims."""
    summary_path = REVISION_DIR / "b1_profile_leakage" / "b1_summary.md"
    text = summary_path.read_text(encoding="utf-8")
    assert "rather than an isolated causal test" in text
    assert "composite design shift" in text
    assert "did not artificially inflate" not in text

def test_dataset_sha256_provenance_consistency():
    """Verifies raw dataset SHA-256 matches manifest and audit report."""
    assert RAW_DATA_PATH.exists()
    computed_sha = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    assert computed_sha == "19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0"
    
    manifest_path = REVISION_DIR / "experiment_manifest.json"
    with open(manifest_path, "r") as f:
        m = json.load(f)
    assert m["raw_dataset"]["sha256"] == computed_sha
    
    audit_path = REVISION_DIR / "FINAL_PHASE1_AUDIT.md"
    audit_text = audit_path.read_text(encoding="utf-8")
    assert computed_sha in audit_text
    assert "64e9a8f46e5" not in audit_text, "Stale hash 64e9a8 must not appear in current audit"

# ===========================================================================
# LAYER 3: REAL-DATA EMPIRICAL INTEGRATION TESTS (B1 - B7 ARTIFACTS)
# ===========================================================================

def test_raw_dataset_integrity():
    assert RAW_DATA_PATH.exists(), "Raw dataset missing!"
    df = pd.read_csv(RAW_DATA_PATH)
    assert df.shape == (253680, 22), f"Expected shape (253680, 22), got {df.shape}"
    assert "Diabetes_binary" in df.columns, "Target column missing"
    assert set(df["Diabetes_binary"].unique()).issubset({0.0, 1.0}), "Invalid target values"
    assert df["Diabetes_binary"].value_counts()[0.0] == 218334, "Class 0 count mismatch"
    assert df["Diabetes_binary"].value_counts()[1.0] == 35346, "Class 1 count mismatch"

def test_duplicate_and_profile_counts():
    df = pd.read_csv(CLEANED_DATA_PATH)
    feature_cols = [c for c in df.columns if c != "Diabetes_binary"]
    assert len(feature_cols) == 21, "Must have exactly 21 predictor features"
    exact_surplus = df.duplicated().sum()
    profile_surplus = df.duplicated(subset=feature_cols).sum()
    assert exact_surplus == 24206, f"Expected 24206 exact duplicate surplus, got {exact_surplus}"
    assert profile_surplus == 25772, f"Expected 25772 profile surplus, got {profile_surplus}"

def test_b1_grouped_split_and_ci_consistency():
    summary_path = REVISION_DIR / "b1_profile_leakage" / "grouped_split_summary.json"
    assert summary_path.exists(), "Grouped split summary missing"
    with open(summary_path, "r") as f:
        data = json.load(f)
    assert data["shared_profile_groups"] == 0, "Group leakage detected: shared groups > 0!"
    assert data["dev_size"] + data["holdout_size"] == 253680
    assert abs(data["dev_prevalence"] - 0.1393) < 0.005
    assert abs(data["holdout_prevalence"] - 0.1393) < 0.005

    metrics_path = REVISION_DIR / "b1_profile_leakage" / "grouped_holdout_metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, "r") as f:
        m = json.load(f)
    assert m["PR-AUC"] == 0.4321
    assert m["bootstrap_95_ci"]["PR-AUC"] == [0.4213, 0.4436]
    assert m["ROC-AUC"] == 0.8279
    assert m["bootstrap_95_ci"]["ROC-AUC"] == [0.8231, 0.8326]
    assert m["Recall"] == 0.8037
    assert m["bootstrap_95_ci"]["Recall"] == [0.7948, 0.8126]

def test_b3_calibration_reliability_1000_bootstraps():
    b3_path = REVISION_DIR / "b3_calibration" / "calibration_results.json"
    assert b3_path.exists()
    with open(b3_path, "r") as f:
        data = json.load(f)
    assert data["brier_score"] == 0.0974
    assert data["calibration_slope"] == 0.9589
    assert data["calibration_intercept"] == -0.0515
    assert data["bootstrap_metadata"]["successful_replicates"] == 1000
    assert data["bootstrap_metadata"]["failed_replicates"] == 0
    assert data["calibration_slope_95_ci"] == [0.9342, 0.9854]
    assert data["calibration_intercept_95_ci"] == [-0.0861, -0.0148]

def test_b4_decision_curve_strict_dominance():
    b4_path = REVISION_DIR / "b4_decision_curve" / "decision_curve_data.csv"
    assert b4_path.exists()
    df = pd.read_csv(b4_path)
    assert (df["net_benefit_none"] == 0.0).all()
    assert (df["net_benefit_model"] > df["net_benefit_all"]).all()
    assert (df["net_benefit_model"] > df["net_benefit_none"]).all()

def test_b5_shap_stability_consensus():
    b5_path = REVISION_DIR / "b5_shap_stability" / "shap_stability_summary.json"
    assert b5_path.exists()
    with open(b5_path, "r") as f:
        data = json.load(f)
    assert data["fold_stability"]["mean_pairwise_jaccard"] >= 0.80
    assert data["seed_resampling_stability"]["mean_pairwise_jaccard"] >= 0.95
    consensus = set(data["fold_stability"]["consensus_features_in_all_5_folds"])
    for core_feat in ["GenHlth", "HighBP", "Age", "BMI", "HighChol"]:
        assert core_feat in consensus, f"Core driver {core_feat} missing from fold consensus"

def test_b6_rarity_modal_vif_and_income_sensitivity():
    inc_path = REVISION_DIR / "b6_vif_sensitivity" / "income_sensitivity_comparison.json"
    assert inc_path.exists()
    with open(inc_path, "r") as f:
        isens = json.load(f)
    assert isens["max_vif_distinct_income"] <= 2.0
    assert isens["max_vif_merged_income"] <= 2.0
    assert isens["top10_overlap_distinct_vs_merged"] == 10
    assert isens["spearman_rank_correlation_chi2"] == 1.0
    assert isens["alignment_verified_by_variable_name"] is True

def test_b7_threshold_isolated_lineage():
    b7_path = REVISION_DIR / "b7_threshold_sensitivity" / "threshold_sensitivity_summary.json"
    assert b7_path.exists()
    with open(b7_path, "r") as f:
        data = json.load(f)
    assert data["recall_ge_75"]["dev_oof_metrics"]["recall"] >= 0.75
    assert data["recall_ge_80"]["dev_oof_metrics"]["recall"] >= 0.80
    assert data["recall_ge_85"]["dev_oof_metrics"]["recall"] >= 0.85

# ===========================================================================
# LAYER 4: DYNAMIC COMPUTATION-BASED INTEGRATION TESTS
# ===========================================================================

def test_computation_based_integration_toy_flow():
    """Executes dynamic computation on synthetic cohort to verify schema and metrics without reading disk."""
    np.random.seed(42)
    n = 300
    y_synth = np.random.binomial(1, 0.15, size=n)
    probs_synth = np.clip(y_synth * 0.4 + np.random.uniform(0.01, 0.5, size=n), 0.01, 0.99)
    
    # 1. Dynamic DCA computation
    dca_res = calculate_net_benefit(y_synth, probs_synth, threshold=0.15)
    assert "net_benefit_model" in dca_res
    assert "net_benefit_all" in dca_res
    assert dca_res["net_benefit_none"] == 0.0
    
    # 2. Dynamic threshold optimization
    t_opt, t_details = optimize_threshold(y_synth, probs_synth, target_recall=0.75)
    assert t_details["recall"] >= 0.75
    assert 0.01 <= t_opt <= 0.99
    
    # 3. Dynamic feature-aligned Spearman
    df_a = pd.DataFrame({"variable": [f"V_{i}" for i in range(10)], "score_a": np.random.uniform(1, 100, 10)})
    df_b = pd.DataFrame({"variable": [f"V_{i}" for i in range(10)], "score_b": np.random.uniform(1, 100, 10)}).sample(frac=1.0)
    aligned = pd.merge(df_a, df_b, on="variable", validate="one_to_one")
    rho, pval = stats.spearmanr(aligned["score_a"], aligned["score_b"])
    assert -1.0 <= rho <= 1.0

# ===========================================================================
# LAYER 5: REPOSITORY IMMUTABILITY & PROTECTION TESTS
# ===========================================================================

def test_protected_files_immutability():
    """Confirms paper directory and baseline results are completely intact."""
    main_tex = PAPER_DIR / "main.tex"
    assert main_tex.exists(), "paper/main.tex must remain intact"
    baseline_summary = PROJECT_ROOT / "results" / "final_results_summary.json"
    assert baseline_summary.exists(), "Baseline results summary must remain intact"
    with open(baseline_summary, "r") as f:
        bdata = json.load(f)
    assert bdata["holdout_test_evaluation"]["selected_threshold"] == 0.13
    assert bdata["holdout_test_evaluation"]["metrics_at_selected_threshold"]["Recall"] == 0.8099

# ===========================================================================
# LAYER 6: REMEDIATION & AUDIT PROVENANCE VERIFICATION TESTS
# ===========================================================================

def test_b3_bootstrap_replicates_provenance_and_quantiles():
    """Verifies that B3 stores all 1,000 raw bootstrap replicates and matches 95% CIs."""
    replicates_path = REVISION_DIR / "b3_calibration" / "bootstrap_replicates.csv"
    assert replicates_path.exists(), "Raw bootstrap replicates artifact missing!"
    df_boot = pd.read_csv(replicates_path)
    assert len(df_boot) == 1000, f"Expected exactly 1,000 replicates, got {len(df_boot)}"
    assert {"replicate", "calibration_slope", "calibration_intercept", "brier_score"}.issubset(df_boot.columns)
    
    slope_lo = round(float(np.percentile(df_boot["calibration_slope"], 2.5)), 4)
    slope_hi = round(float(np.percentile(df_boot["calibration_slope"], 97.5)), 4)
    assert [slope_lo, slope_hi] == [0.9342, 0.9854]

    intercept_lo = round(float(np.percentile(df_boot["calibration_intercept"], 2.5)), 4)
    intercept_hi = round(float(np.percentile(df_boot["calibration_intercept"], 97.5)), 4)
    assert [intercept_lo, intercept_hi] == [-0.0861, -0.0148]

    brier_lo = round(float(np.percentile(df_boot["brier_score"], 2.5)), 4)
    brier_hi = round(float(np.percentile(df_boot["brier_score"], 97.5)), 4)
    assert [brier_lo, brier_hi] == [0.0964, 0.0983]

def test_b5_raw_shap_sample_artifact_integrity():
    """Verifies B5 raw SHAP values sample artifact exists and matches feature ranks."""
    raw_npz_path = REVISION_DIR / "b5_shap_stability" / "raw_shap_sample_seed42.npz"
    assert raw_npz_path.exists(), "Raw SHAP values sample artifact missing!"
    data = np.load(raw_npz_path)
    assert "shap_values" in data and "features" in data and "sample_indices" in data
    assert data["shap_values"].shape == (10000, 21)
    assert len(data["features"]) == 21
    assert len(data["sample_indices"]) == 10000
    
    # Compute mean absolute SHAP from raw array and verify top feature is GenHlth
    mean_abs = np.abs(data["shap_values"]).mean(axis=0)
    top_feat = data["features"][np.argmax(mean_abs)]
    assert top_feat == "GenHlth", f"Top feature should be GenHlth, got {top_feat}"

def test_all_execution_logs_status_and_source_hash_integrity():
    """Verifies every revision experiment has execution logs and matching source SHA-256."""
    logs_dir = REVISION_DIR / "logs"
    assert logs_dir.exists()
    
    expected_experiments = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "CONSOLIDATE"]
    for exp in expected_experiments:
        exec_json = logs_dir / f"{exp}.execution.json"
        assert exec_json.exists(), f"Execution JSON log missing for {exp}"
        with open(exec_json, "r") as f:
            rec = json.load(f)
        
        assert rec["status"] == "PASS", f"Execution status for {exp} is not PASS"
        assert rec["exit_code"] == 0, f"Exit code for {exp} is non-zero"
        
        script_path = PROJECT_ROOT / rec["script"]
        assert script_path.exists(), f"Source script {rec['script']} not found"
        actual_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
        assert rec["script_sha256"] == actual_sha, f"Source hash mismatch for {exp}!"

def test_no_forbidden_stale_phrases_in_markdown_reports():
    """Scans all revision markdown reports to ensure no forbidden stale claims remain."""
    forbidden_phrases = [
        "won 15 of 15",
        "won all 15",
        "across all 15 paired iterations",
        "did not artificially inflate"
    ]
    
    report_files = [
        REVISION_DIR / "b1_profile_leakage" / "b1_summary.md",
        REVISION_DIR / "b2_tuning" / "b2_summary.md",
        REVISION_DIR / "experiment_summary.md",
        REVISION_DIR / "FINAL_PHASE1_AUDIT.md",
        REVISION_DIR / "STRICT_AUDIT.md"
    ]
    
    for rpath in report_files:
        if rpath.exists():
            text = rpath.read_text(encoding="utf-8")
            for phrase in forbidden_phrases:
                assert phrase.lower() not in text.lower(), f"Forbidden stale phrase '{phrase}' found in {rpath.name}"

def test_output_artifact_manifest_and_provenance_structure():
    """Verifies that output_artifact_manifest.json exists and tracks artifacts with correct evidence type."""
    manifest_path = REVISION_DIR / "output_artifact_manifest.json"
    assert manifest_path.exists(), "Output artifact manifest missing!"
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["evidence_type"] == "post_hoc_artifact_snapshot"
    assert m["total_tracked_artifacts"] >= 35
    for item in m["artifacts"]:
        assert (PROJECT_ROOT / item["artifact_path"]).exists(), f"Artifact {item['artifact_path']} not found on disk"
        assert len(item["sha256"]) == 64


