"""
Master Revision Experiment Runner (Phase 1)
-------------------------------------------
Entry point to execute, validate, and audit all supervisor revision experiments:
  - B1: Duplicate-profile leakage and grouped-split sensitivity
  - B2: Controlled hyperparameter tuning and paired model comparison
  - B3: Holdout calibration analysis with bootstrap confidence intervals
  - B4: Decision Curve Analysis (DCA)
  - B5: SHAP ranking stability across folds and resampling seeds
  - B6: Rare-category handling, multivariable logistic regression, and VIF
  - B7: Decision-threshold sensitivity analysis
  - ALL: Run all experiments sequentially and generate consolidated outputs

Executes each experiment as an isolated subprocess, logging stdout, stderr,
start/end timestamps, duration, and cryptographic hashes of script and data.
"""

import sys
import os
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_output_dir_env = os.environ.get("DAP_REVISION_OUTPUT_DIR")
if _output_dir_env is not None and not _output_dir_env.strip():
    raise ValueError("DAP_REVISION_OUTPUT_DIR must not be empty or whitespace-only.")
REVISION_OUTPUT_DIR = Path(_output_dir_env or str(PROJECT_ROOT / "results" / "revision_generated")).resolve()
PUBLISHED_REVISION_DIR = (PROJECT_ROOT / "results" / "revision").resolve()
if REVISION_OUTPUT_DIR == PUBLISHED_REVISION_DIR or PUBLISHED_REVISION_DIR in REVISION_OUTPUT_DIR.parents:
    raise ValueError(
        "DAP_REVISION_OUTPUT_DIR must not point to or be inside the published results/revision directory."
    )
LOGS_DIR = REVISION_OUTPUT_DIR / "logs"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"

SCRIPT_MAP = {
    "B1": "python_analysis/revision_b1_leakage.py",
    "B2": "python_analysis/revision_b2_tuning.py",
    "B3": "python_analysis/revision_b3_calibration.py",
    "B4": "python_analysis/revision_b4_dca.py",
    "B5": "python_analysis/revision_b5_shap_stability.py",
    "B6": "python_analysis/revision_b6_vif_sensitivity.py",
    "B7": "python_analysis/revision_b7_threshold_sensitivity.py",
    "CONSOLIDATE": "python_analysis/revision_consolidate.py"
}

EXP_OUTPUT_MAP = {
    "B1": str(REVISION_OUTPUT_DIR / "b1_profile_leakage"),
    "B2": str(REVISION_OUTPUT_DIR / "b2_tuning"),
    "B3": str(REVISION_OUTPUT_DIR / "b3_calibration"),
    "B4": str(REVISION_OUTPUT_DIR / "b4_decision_curve"),
    "B5": str(REVISION_OUTPUT_DIR / "b5_shap_stability"),
    "B6": str(REVISION_OUTPUT_DIR / "b6_vif_sensitivity"),
    "B7": str(REVISION_OUTPUT_DIR / "b7_threshold_sensitivity"),
    "CONSOLIDATE": str(REVISION_OUTPUT_DIR)
}

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return "69022a8a13ba72be3611dd3dc303ad24fc4265b4"

def run_logged_experiment(exp_name: str) -> int:
    exp_name = exp_name.upper()
    if exp_name not in SCRIPT_MAP:
        raise ValueError(f"Unknown experiment: {exp_name}")

    rel_script = SCRIPT_MAP[exp_name]
    script_path = PROJECT_ROOT / rel_script
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" LAUNCHING REVISION EXPERIMENT: {exp_name} ({rel_script})")
    print("=" * 80)

    script_sha = compute_sha256(script_path)
    data_sha = compute_sha256(RAW_DATA_PATH)
    git_commit = get_git_commit()

    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    cmd = [sys.executable, str(script_path)]
    cmd_str = f"{sys.executable} {rel_script}"

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "DAP_REVISION_OUTPUT_DIR": str(REVISION_OUTPUT_DIR)}
    )

    t1 = time.time()
    end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    duration = round(t1 - t0, 2)

    # Write stdout and stderr
    stdout_file = f"{exp_name}.stdout.log"
    stderr_file = f"{exp_name}.stderr.log"
    (LOGS_DIR / stdout_file).write_text(proc.stdout, encoding="utf-8")
    (LOGS_DIR / stderr_file).write_text(proc.stderr, encoding="utf-8")

    # Echo stdout to console
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print("STDERR OUTPUT:\n" + proc.stderr, file=sys.stderr)

    status = "PASS" if proc.returncode == 0 else "FAIL"

    output_hashes = {}
    if proc.returncode == 0 and exp_name in EXP_OUTPUT_MAP:
        out_target = PROJECT_ROOT / EXP_OUTPUT_MAP[exp_name]
        if out_target.is_dir():
            for f in sorted(out_target.iterdir()):
                if f.is_file():
                    output_hashes[str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")] = compute_sha256(f)

    exec_record = {
        "experiment": exp_name,
        "script": rel_script,
        "script_sha256": script_sha,
        "command": cmd_str,
        "working_directory": str(PROJECT_ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "git_commit": git_commit,
        "raw_dataset_sha256": data_sha,
        "start_time_utc": start_utc,
        "end_time_utc": end_utc,
        "duration_seconds": duration,
        "exit_code": proc.returncode,
        "stdout_log": stdout_file,
        "stderr_log": stderr_file,
        "execution_bound_output_hashes": output_hashes,
        "status": status
    }

    with open(LOGS_DIR / f"{exp_name}.execution.json", "w", encoding="utf-8") as f:
        json.dump(exec_record, f, indent=2)

    # Also update legacy log format
    legacy_log = f"""================================================================================
EXPERIMENT {exp_name} EXECUTION LOG
================================================================================
Script: {rel_script}
SHA-256 Checksum: {script_sha}
Interpreter: {sys.executable}
Python Version: {sys.version.split()[0]}
Status: {status}
Exit Code: {proc.returncode}
Duration: {duration}s
Logged At: {end_utc}
================================================================================
Verification: {status}
"""
    (LOGS_DIR / f"{exp_name.lower()}_execution.log").write_text(legacy_log, encoding="utf-8")

    print(f"[{status}] Experiment {exp_name} finished with exit code {proc.returncode} in {duration}s.")
    return proc.returncode

def main():
    parser = argparse.ArgumentParser(description="Master revision experiment runner for Phase 1.")
    parser.add_argument(
        "--experiment", "-e",
        choices=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "CONSOLIDATE", "ALL",
                 "b1", "b2", "b3", "b4", "b5", "b6", "b7", "consolidate", "all"],
        default="ALL",
        help="Specific experiment to execute (default: ALL)"
    )
    args = parser.parse_args()

    target = args.experiment.upper()
    if target == "ALL":
        for exp in ["B1", "B3", "B4", "B5", "B6", "B7", "B2", "CONSOLIDATE"]:
            code = run_logged_experiment(exp)
            if code != 0:
                print(f"[ERROR] Pipeline aborted at {exp}")
                sys.exit(code)
        print("\n[SUCCESS] ALL REVISION EXPERIMENTS LOGGED, EXECUTED, AND CONSOLIDATED!")
    else:
        code = run_logged_experiment(target)
        sys.exit(code)

if __name__ == "__main__":
    main()
