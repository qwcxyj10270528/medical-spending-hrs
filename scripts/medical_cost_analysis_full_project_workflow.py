from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "scripts"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data" / "homework 1"

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "mpl_cache"))
sys.path.insert(0, str(SCRIPT_DIR))

import modeling_optimization as opt


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def check_reproducible_project_structure() -> None:
    required_files = [
        DATA_DIR / "h18core" / "h18sta.zip",
        DATA_DIR / "h20core" / "h20sta.zip",
        DATA_DIR / "h22core" / "H22sta.zip",
        DATA_DIR / "trk2022v1" / "trk2022tr_r.csv",
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required raw data files:\n{missing_text}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project directory: {PROJECT_DIR}")
    print(f"Raw data directory: {DATA_DIR}")
    print("All required raw data files are available inside the project folder.")


def run_data_preparation() -> pd.DataFrame:
    print_section("1. Data Loading, Codebook-Based Cleaning, and Target Variable Construction")
    raw, clean, optimized, cleaning_log = opt.build_optimized_dataset()

    print(f"Raw selected respondent-wave rows: {len(raw):,}")
    print(f"Cleaned rows before adding income and insurance variables: {len(clean):,}")
    print(f"Final optimized modeling rows: {len(optimized):,}")

    target_audit = pd.read_csv(OUTPUT_DIR / "target_rule_audit.csv")
    gate_audit = pd.read_csv(OUTPUT_DIR / "spending_gate_zero_rule_audit.csv")

    print("\nTarget rule audit:")
    print(target_audit.to_string(index=False))

    print("\nSpending gate-zero rule audit:")
    print(gate_audit.to_string(index=False))

    print("\nCleaning log:")
    print(cleaning_log.to_string(index=False))

    return optimized


def run_added_variable_checks(optimized: pd.DataFrame) -> None:
    print_section("2. Income and Insurance Variable Verification, Missingness, and EDA")

    added_missingness, income_insurance_eda = opt.make_income_insurance_eda(optimized)
    codebook_check = pd.read_csv(OUTPUT_DIR / "codebook_verification_income_insurance.csv")

    print("\nCodebook verification for added variables:")
    print(codebook_check.to_string(index=False))

    print("\nMissingness for added variables:")
    print(added_missingness.round(4).to_string(index=False))

    print("\nIncome and insurance EDA spending summaries:")
    print(income_insurance_eda.round(2).to_string(index=False))

    print(f"\nEDA figures saved to: {FIG_DIR}")


def run_feature_schema_checks() -> None:
    print_section("3. Feature Engineering and Variable Type Schema")

    base_features, _, _, _, base_schema = opt.make_feature_schema(include_added_variables=False)
    plus_features, _, _, _, plus_schema = opt.make_feature_schema(include_added_variables=True)

    print(f"Base feature count: {len(base_features)}")
    print(f"Expanded feature count: {len(plus_features)}")

    print("\nBase variable type schema:")
    print(base_schema.to_string(index=False))

    print("\nExpanded variable type schema:")
    print(plus_schema.to_string(index=False))


def run_model_selection_and_evaluation(optimized: pd.DataFrame) -> tuple[str, object]:
    print_section("4. Temporal CV Model Selection and 2022 Test Evaluation")

    cv_all, test_all, ctx_base, ctx_plus = opt.compare_base_vs_plus(optimized)
    cv_chosen = (
        cv_all.query("feature_set == 'plus_income_insurance'")
        .sort_values("temporal_cv_rmse_log_mean")
        .iloc[0]["model"]
    )

    print("\nTemporal CV results:")
    print(cv_all.round(4).to_string(index=False))

    print("\n2022 held-out test results:")
    print(test_all.round(4).to_string(index=False))

    print(f"\nModel selected by temporal CV among expanded-feature models: {cv_chosen}")

    print_section("5. Extreme-Bill Robustness Check")
    robustness = opt.outlier_robustness_check(ctx_plus, cv_chosen)
    print(robustness.round(4).to_string(index=False))

    print_section("6. Hyperparameter Tuning with the Same Temporal CV")
    tuned_results, best_model, model_path = opt.tune_cv_chosen_model(optimized, cv_chosen)
    print(tuned_results.round(6).to_string(index=False))
    print(f"\nSaved fitted pipeline: {model_path}")

    return cv_chosen, best_model


def run_deployment_example(best_model: object) -> None:
    print_section("7. Saved Pipeline and Deployment-Style Prediction Example")

    features, *_ = opt.make_feature_schema(include_added_variables=True)
    example_prediction = opt.make_deployment_example(best_model, features)

    print("\nExample prediction using the fitted pipeline:")
    print(example_prediction.round(3).to_string(index=False))

    raw_person = example_prediction.drop(columns=["predicted_oop_dollars"]).iloc[0].to_dict()
    predicted_from_saved_model = opt.predict_oop_dollars_from_raw(raw_person)
    print(f"\nPrediction loaded from saved joblib pipeline: ${predicted_from_saved_model:,.2f}")


def main() -> None:
    check_reproducible_project_structure()
    optimized = run_data_preparation()
    run_added_variable_checks(optimized)
    run_feature_schema_checks()
    _, best_model = run_model_selection_and_evaluation(optimized)
    run_deployment_example(best_model)

    print_section("Workflow Complete")
    print("The full project workflow has been reproduced from raw project data to saved model output.")
    print(f"Main cleaned dataset: {OUTPUT_DIR / 'hrs_3wave_modeling_optimized_clean.csv'}")
    print(f"Final model file: {MODEL_DIR / 'hrs_oop_spending_final_optimized.joblib'}")


if __name__ == "__main__":
    main()
