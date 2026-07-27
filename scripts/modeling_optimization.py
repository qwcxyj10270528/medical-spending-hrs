from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import phase2_cleaning_eda as phase2


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

INSURANCE_INCOME_VARIABLES = {
    2018: {
        "prefix": "Q",
        "n_file": "h18n_r.dta",
        "q_file": "h18q_h.dta",
        "medicare_coverage": "QN001",
        "num_private_hi_plans": "QN023",
        "social_security_income_monthly": "QQ085",
    },
    2020: {
        "prefix": "R",
        "n_file": "H20N_R.dta",
        "q_file": "H20Q_H.dta",
        "medicare_coverage": "RN001",
        "num_private_hi_plans": "RN023",
        "social_security_income_monthly": "RQ085",
    },
    2022: {
        "prefix": "S",
        "n_file": "H22N_R.dta",
        "q_file": "H22Q_H.dta",
        "medicare_coverage": "SN001",
        "num_private_hi_plans": "SN023",
        "social_security_income_monthly": "SQ085",
    },
}

CODEBOOK_VERIFICATION = pd.DataFrame(
    [
        {
            "final_variable": "medicare_coverage",
            "source_variables": "QN001 / RN001 / SN001",
            "source_section": "HRS Core Interview Section N",
            "codebook_label": "MEDICARE COVERAGE",
            "model_type": "binary",
            "cleaning_rule": "1 is mapped to yes; 5 is mapped to no; negative and special missing values are set to missing.",
        },
        {
            "final_variable": "num_private_hi_plans",
            "source_variables": "QN023 / RN023 / SN023",
            "source_section": "HRS Core Interview Section N",
            "codebook_label": "NUM PRIVATE HEALTH INS PLANS",
            "model_type": "numeric count",
            "cleaning_rule": "0 and positive counts are kept; negative and 98/99-style special codes are set to missing.",
        },
        {
            "final_variable": "social_security_income_monthly",
            "source_variables": "QQ085 / RQ085 / SQ085",
            "source_section": "HRS Core Interview Section Q",
            "codebook_label": "R AMOUNT OF SS INCOME - LAST MONTH",
            "model_type": "continuous numeric",
            "cleaning_rule": "Valid dollar amounts are kept; negative and repeated 9 special missing values are set to missing.",
        },
        {
            "final_variable": "oop_rx_drugs_annualized",
            "source_variables": "QN180 / RN180 / SN180",
            "source_section": "HRS Core Interview Section N",
            "codebook_label": "AMT PAY O-O-P RX DRUGS PER MONTH",
            "model_type": "target component",
            "cleaning_rule": "The monthly prescription drug amount is multiplied by 12 before being added to annual OOP spending.",
        },
    ]
)


def normalize_hhid(df: pd.DataFrame) -> pd.DataFrame:
    rename = {col: "HHID" for col in df.columns if col.lower() == "hhid"}
    rename.update({col: "PN" for col in df.columns if col.lower() == "pn"})
    return df.rename(columns=rename)


def read_stata_member(zip_path: Path, member: str, columns: list[str] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory() as tmp_dir:
        archive.extract(member, tmp_dir)
        path = Path(tmp_dir) / member
        return pd.read_stata(path, convert_categoricals=False, columns=columns)


def clean_binary_yes_no(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[numeric.eq(1)] = 1.0
    out[numeric.eq(5)] = 0.0
    return out


def clean_count(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.mask(numeric < 0)
    numeric = numeric.mask(numeric.isin([98, 99, 998, 999]))
    return numeric


def clean_income(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.mask(numeric < 0)
    missing_codes = {
        9998,
        9999,
        99998,
        99999,
        999998,
        999999,
        9999998,
        9999999,
        99999998,
        99999999,
    }
    numeric = numeric.mask(numeric.isin(missing_codes))
    return numeric


def first_nonmissing(series: pd.Series) -> float | str | None:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def load_income_insurance_for_wave(year: int) -> pd.DataFrame:
    cfg = INSURANCE_INCOME_VARIABLES[year]
    zip_path = phase2.WAVES[year]["zip"]

    n_raw = normalize_hhid(read_stata_member(zip_path, cfg["n_file"]))
    n_raw = n_raw.rename(
        columns={
            cfg["medicare_coverage"]: "medicare_coverage_raw",
            cfg["num_private_hi_plans"]: "num_private_hi_plans_raw",
        }
    )
    n_raw["medicare_coverage"] = clean_binary_yes_no(n_raw["medicare_coverage_raw"])
    n_raw["num_private_hi_plans"] = clean_count(n_raw["num_private_hi_plans_raw"])

    q_raw = normalize_hhid(read_stata_member(zip_path, cfg["q_file"]))
    q_raw = q_raw.rename(columns={cfg["social_security_income_monthly"]: "social_security_income_monthly_raw"})
    q_raw["social_security_income_monthly"] = clean_income(q_raw["social_security_income_monthly_raw"])
    q_clean = q_raw.groupby("HHID", as_index=False).agg({"social_security_income_monthly": first_nonmissing})

    merged = n_raw[["HHID", "PN", "medicare_coverage", "num_private_hi_plans"]].merge(q_clean, on="HHID", how="left")
    merged["wave"] = year
    return merged


def add_income_insurance_variables(clean: pd.DataFrame) -> pd.DataFrame:
    additions = pd.concat([load_income_insurance_for_wave(year) for year in sorted(INSURANCE_INCOME_VARIABLES)], ignore_index=True)
    out = clean.merge(additions, on=["HHID", "PN", "wave"], how="left")
    return out


def build_optimized_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = phase2.build_raw_long()
    clean, cleaning_log = phase2.clean_analysis_data(raw)
    optimized = add_income_insurance_variables(clean)
    optimized.to_csv(OUTPUT_DIR / "hrs_3wave_modeling_optimized_clean.csv", index=False)
    raw.to_csv(OUTPUT_DIR / "hrs_3wave_selected_raw_long.csv", index=False)
    cleaning_log.to_csv(OUTPUT_DIR / "phase2_cleaning_log_optimization.csv", index=False)

    target_audit = make_target_rule_audit(raw, clean)
    target_audit.to_csv(OUTPUT_DIR / "target_rule_audit.csv", index=False)
    CODEBOOK_VERIFICATION.to_csv(OUTPUT_DIR / "codebook_verification_income_insurance.csv", index=False)
    return raw, clean, optimized, cleaning_log


def make_target_rule_audit(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    component_cols = [c for c in phase2.SPENDING_COLUMNS if c in raw.columns]
    cleaned_components = pd.DataFrame({c: phase2.clean_spending(raw[c], c) for c in component_cols})
    cleaned_components["oop_rx_drugs_annualized"] = cleaned_components["oop_rx_drugs_monthly"] * 12
    spending_for_total = [c for c in component_cols if c != "oop_rx_drugs_monthly"] + ["oop_rx_drugs_annualized"]
    present_count_before_gate = cleaned_components[spending_for_total].notna().sum(axis=1)
    true_zero_like_before_gate = cleaned_components[spending_for_total].fillna(0).sum(axis=1).eq(0) & present_count_before_gate.gt(0)
    all_missing_before_gate = present_count_before_gate.eq(0)

    gated = raw.copy()
    for col in component_cols:
        gated[col] = phase2.clean_spending(gated[col], col)
    gated, gate_audit = phase2.apply_spending_gate_zero_rules(gated)
    gated["oop_rx_drugs_annualized"] = gated["oop_rx_drugs_monthly"] * 12
    present_count_after_gate = gated[spending_for_total].notna().sum(axis=1)
    true_zero_like_after_gate = gated[spending_for_total].fillna(0).sum(axis=1).eq(0) & present_count_after_gate.gt(0)
    all_missing_after_gate = present_count_after_gate.eq(0)
    components_set_zero_by_gate = int(gate_audit["component_missing_set_to_zero_n"].sum()) if not gate_audit.empty else 0
    return pd.DataFrame(
        [
            {
                "target_rule_item": "raw respondent-wave rows",
                "n": len(raw),
                "interpretation": "All selected 2018, 2020, and 2022 respondent-wave rows before outcome filtering.",
            },
            {
                "target_rule_item": "rows with at least one usable spending component before gate-zero rules",
                "n": int(present_count_before_gate.gt(0).sum()),
                "interpretation": "Rows with at least one observed spending amount before using service-use gate variables.",
            },
            {
                "target_rule_item": "rows with no usable spending component before gate-zero rules",
                "n": int(all_missing_before_gate.sum()),
                "interpretation": "Rows that would be dropped under the earlier conservative rule.",
            },
            {
                "target_rule_item": "spending components set to 0 using codebook gate variables",
                "n": components_set_zero_by_gate,
                "interpretation": "Component-level missing values recoded to 0 when the codebook gate variable clearly indicates no service/use.",
            },
            {
                "target_rule_item": "rows with no usable spending component after gate-zero rules",
                "n": int(all_missing_after_gate.sum()),
                "interpretation": "Rows still lacking any usable spending component after codebook gate variables are applied.",
            },
            {
                "target_rule_item": "true zero-spending rows retained after gate-zero rules",
                "n": int(true_zero_like_after_gate.sum()),
                "interpretation": "Rows with usable spending-component information summing to zero after gate-zero recoding; these remain in the dataset and are modeled with log1p.",
            },
            {
                "target_rule_item": "clean modeling rows after outcome rule",
                "n": len(clean),
                "interpretation": "Rows available after dropping only remaining missing outcomes, not zero-dollar outcomes.",
            },
        ]
    )


def make_missingness_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "feature": features,
            "n": len(df),
            "nonmissing_n": df[features].notna().sum().values,
            "missing_n": df[features].isna().sum().values,
            "missing_rate": df[features].isna().mean().values,
        }
    )
    table.to_csv(OUTPUT_DIR / "optimized_feature_missingness.csv", index=False)
    return table


def save_fig(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_income_insurance_eda(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for var in ["medicare_coverage", "num_private_hi_plans"]:
        temp = df.copy()
        if var == "num_private_hi_plans":
            temp["group"] = temp[var].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
        else:
            temp["group"] = temp[var].map({0.0: "No", 1.0: "Yes"})
        summary = temp.groupby("group", dropna=False)["total_oop_medical_spending"].agg(
            n="size", median_spending="median", mean_spending="mean"
        ).reset_index()
        summary["variable"] = var
        rows.append(summary)

    income_temp = df.copy()
    income_temp["social_security_income_group"] = pd.qcut(
        income_temp["social_security_income_monthly"], q=4, duplicates="drop"
    )
    income_summary = income_temp.groupby("social_security_income_group", dropna=False)["total_oop_medical_spending"].agg(
        n="size", median_spending="median", mean_spending="mean"
    ).reset_index()
    income_summary["variable"] = "social_security_income_monthly"
    income_summary = income_summary.rename(columns={"social_security_income_group": "group"})
    rows.append(income_summary)

    eda = pd.concat(rows, ignore_index=True)
    eda["group"] = eda["group"].astype(str)
    eda.to_csv(OUTPUT_DIR / "income_insurance_eda_spending_groups.csv", index=False)

    missing = make_missingness_table(
        df,
        ["medicare_coverage", "num_private_hi_plans", "social_security_income_monthly"],
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot = missing.sort_values("missing_rate")
    bars = ax.bar(plot["feature"], plot["missing_rate"] * 100, color=["#4E79A7", "#59A14F", "#E15759"])
    ax.set_title("Missingness of Added Income and Insurance Variables")
    ax.set_ylabel("Missing rate (%)")
    ax.tick_params(axis="x", rotation=18)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    save_fig(fig, "optimized_added_variable_missingness.png")

    med_plot = eda[eda["variable"].eq("medicare_coverage") & eda["group"].isin(["No", "Yes"])]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(med_plot["group"], med_plot["median_spending"], color=["#4E79A7", "#F28E2B"])
    ax.set_title("Median OOP Spending by Medicare Coverage")
    ax.set_xlabel("Medicare coverage")
    ax.set_ylabel("Median total OOP spending")
    ax.bar_label(bars, fmt="$%.0f", padding=3)
    save_fig(fig, "optimized_medicare_median_spending.png")

    private_plot = eda[eda["variable"].eq("num_private_hi_plans")]
    private_plot = private_plot[private_plot["group"].isin(["0", "1", "2", "3+"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(private_plot["group"], private_plot["median_spending"], color="#76B7B2")
    ax.set_title("Median OOP Spending by Number of Private Insurance Plans")
    ax.set_xlabel("Number of private health insurance plans")
    ax.set_ylabel("Median total OOP spending")
    ax.bar_label(bars, fmt="$%.0f", padding=3)
    save_fig(fig, "optimized_private_plan_median_spending.png")

    inc_plot = eda[eda["variable"].eq("social_security_income_monthly") & ~eda["group"].eq("nan")].copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.bar(range(len(inc_plot)), inc_plot["median_spending"], color="#9C755F")
    ax.set_title("Median OOP Spending by Social Security Income Quartile")
    ax.set_xlabel("Social Security monthly income group")
    ax.set_ylabel("Median total OOP spending")
    ax.set_xticks(range(len(inc_plot)))
    ax.set_xticklabels([str(g).replace(", ", ",\n") for g in inc_plot["group"]], rotation=0, fontsize=8)
    ax.bar_label(bars, fmt="$%.0f", padding=3, fontsize=8)
    save_fig(fig, "optimized_social_security_income_median_spending.png")
    return missing, eda


def make_preprocess(numeric_features: list[str], binary_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("binary", binary_pipeline, binary_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def make_feature_schema(include_added_variables: bool = True) -> tuple[list[str], list[str], list[str], list[str], pd.DataFrame]:
    numeric = ["age", "bmi_self_reported", "cigarettes_per_day", "chronic_condition_count"]
    binary = ["obese", "current_smoker"]
    categorical = ["sex", "race", "education"]
    if include_added_variables:
        numeric += ["num_private_hi_plans", "social_security_income_monthly"]
        binary += ["medicare_coverage"]
    features = numeric + binary + categorical
    schema = pd.DataFrame(
        [{"feature": col, "model_type": "numeric"} for col in numeric]
        + [{"feature": col, "model_type": "binary"} for col in binary]
        + [{"feature": col, "model_type": "categorical"} for col in categorical]
    )
    schema.to_csv(OUTPUT_DIR / ("variable_type_schema_plus.csv" if include_added_variables else "variable_type_schema_base.csv"), index=False)
    return features, numeric, binary, categorical, schema


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(name: str, y_true_dollars: pd.Series, pred_dollars: np.ndarray, y_true_log: pd.Series | None = None) -> dict[str, Any]:
    pred_dollars = np.clip(np.asarray(pred_dollars), 0, None)
    pred_log = np.log1p(pred_dollars)
    if y_true_log is None:
        y_true_log = np.log1p(y_true_dollars)
    return {
        "model": name,
        "MAE_dollars": mean_absolute_error(y_true_dollars, pred_dollars),
        "RMSE_dollars": rmse(y_true_dollars, pred_dollars),
        "R2_dollars": r2_score(y_true_dollars, pred_dollars),
        "RMSE_log": rmse(y_true_log, pred_log),
        "R2_log": r2_score(y_true_log, pred_log),
    }


def fit_log_model(model, preprocess: ColumnTransformer, X_train: pd.DataFrame, y_train_log: pd.Series) -> Pipeline:
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train_log)
    return pipe


def predict_log_model_dollars(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.expm1(pipe.predict(X))


def fit_hurdle_model(preprocess: ColumnTransformer, X_train: pd.DataFrame, y_train_dollars: pd.Series) -> dict[str, Pipeline]:
    has_spend = y_train_dollars.gt(0).astype(int)
    classifier = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    classifier.fit(X_train, has_spend)

    positive = y_train_dollars.gt(0)
    regressor = Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=1.0))])
    regressor.fit(X_train.loc[positive], np.log1p(y_train_dollars.loc[positive]))
    return {"classifier": classifier, "regressor": regressor}


def predict_hurdle_dollars(model: dict[str, Pipeline], X: pd.DataFrame) -> np.ndarray:
    p_spend = model["classifier"].predict_proba(X)[:, 1]
    amount = np.expm1(model["regressor"].predict(X))
    return np.clip(p_spend * amount, 0, None)


def fit_tweedie_model(preprocess: ColumnTransformer, X_train: pd.DataFrame, y_train_dollars: pd.Series) -> Pipeline:
    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", TweedieRegressor(power=1.5, alpha=0.1, link="log", max_iter=1000)),
        ]
    )
    pipe.fit(X_train, y_train_dollars)
    return pipe


def temporal_train_validation_masks(model_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, PredefinedSplit]:
    train_mask = model_df["wave"].isin([2018, 2020])
    test_mask = model_df["wave"].eq(2022)
    validation_fold = np.where(model_df.loc[train_mask, "wave"].eq(2020), 0, -1)
    temporal_cv = PredefinedSplit(test_fold=validation_fold)
    return train_mask, test_mask, model_df.loc[train_mask, "wave"].eq(2018), temporal_cv


def compare_models(model_df: pd.DataFrame, use_added_variables: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features, numeric, binary, categorical, _ = make_feature_schema(include_added_variables=use_added_variables)
    data = model_df[["HHID", "PN", "wave", "total_oop_medical_spending", "log_total_oop_medical_spending"] + features].copy()
    train_mask, test_mask, _, temporal_cv = temporal_train_validation_masks(data)
    X_train = data.loc[train_mask, features]
    X_test = data.loc[test_mask, features]
    y_train_log = data.loc[train_mask, "log_total_oop_medical_spending"]
    y_test_log = data.loc[test_mask, "log_total_oop_medical_spending"]
    y_train_dollars = data.loc[train_mask, "total_oop_medical_spending"]
    y_test_dollars = data.loc[test_mask, "total_oop_medical_spending"]
    preprocess = make_preprocess(numeric, binary, categorical)

    log_models = {
        "Baseline mean": DummyRegressor(strategy="mean"),
        "Ridge regression": Ridge(alpha=1.0),
        "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=RANDOM_STATE, max_iter=5000),
        "Random forest": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=1),
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, random_state=RANDOM_STATE),
    }

    cv_rows: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    for name, model in log_models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        cv = cross_validate(
            pipe,
            X_train,
            y_train_log,
            cv=temporal_cv,
            scoring={"rmse_log": "neg_root_mean_squared_error", "mae_log": "neg_mean_absolute_error"},
            n_jobs=1,
        )
        cv_rows.append(
            {
                "model": name,
                "feature_set": "plus_income_insurance" if use_added_variables else "base",
                "temporal_cv_rmse_log_mean": -cv["test_rmse_log"].mean(),
                "temporal_cv_mae_log_mean": -cv["test_mae_log"].mean(),
            }
        )
        trained[name] = fit_log_model(model, preprocess, X_train, y_train_log)

    X_tr_2018 = data.loc[data["wave"].eq(2018), features]
    y_tr_2018 = data.loc[data["wave"].eq(2018), "total_oop_medical_spending"]
    X_val_2020 = data.loc[data["wave"].eq(2020), features]
    y_val_2020 = data.loc[data["wave"].eq(2020), "total_oop_medical_spending"]

    hurdle_cv_model = fit_hurdle_model(preprocess, X_tr_2018, y_tr_2018)
    hurdle_val_pred = predict_hurdle_dollars(hurdle_cv_model, X_val_2020)
    cv_rows.append(
        {
            "model": "Two-part hurdle",
            "feature_set": "plus_income_insurance" if use_added_variables else "base",
            "temporal_cv_rmse_log_mean": rmse(np.log1p(y_val_2020), np.log1p(hurdle_val_pred)),
            "temporal_cv_mae_log_mean": mean_absolute_error(np.log1p(y_val_2020), np.log1p(hurdle_val_pred)),
        }
    )
    trained["Two-part hurdle"] = fit_hurdle_model(preprocess, X_train, y_train_dollars)

    tweedie_cv_model = fit_tweedie_model(preprocess, X_tr_2018, y_tr_2018)
    tweedie_val_pred = np.clip(tweedie_cv_model.predict(X_val_2020), 0, None)
    cv_rows.append(
        {
            "model": "Tweedie regressor",
            "feature_set": "plus_income_insurance" if use_added_variables else "base",
            "temporal_cv_rmse_log_mean": rmse(np.log1p(y_val_2020), np.log1p(tweedie_val_pred)),
            "temporal_cv_mae_log_mean": mean_absolute_error(np.log1p(y_val_2020), np.log1p(tweedie_val_pred)),
        }
    )
    trained["Tweedie regressor"] = fit_tweedie_model(preprocess, X_train, y_train_dollars)

    cv_results = pd.DataFrame(cv_rows).sort_values("temporal_cv_rmse_log_mean")

    test_rows = []
    for name, model in trained.items():
        if name == "Two-part hurdle":
            pred = predict_hurdle_dollars(model, X_test)
        elif name == "Tweedie regressor":
            pred = np.clip(model.predict(X_test), 0, None)
        else:
            pred = predict_log_model_dollars(model, X_test)
        row = evaluate_predictions(name, y_test_dollars, pred, y_test_log)
        row["feature_set"] = "plus_income_insurance" if use_added_variables else "base"
        test_rows.append(row)

    test_results = pd.DataFrame(test_rows).sort_values("RMSE_log")
    suffix = "plus" if use_added_variables else "base"
    cv_results.to_csv(OUTPUT_DIR / f"model_comparison_cv_{suffix}.csv", index=False)
    test_results.to_csv(OUTPUT_DIR / f"model_comparison_test_{suffix}.csv", index=False)
    return cv_results, test_results, {"features": features, "preprocess": preprocess, "trained": trained, "data": data}


def compare_base_vs_plus(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    cv_base, test_base, ctx_base = compare_models(model_df, use_added_variables=False)
    cv_plus, test_plus, ctx_plus = compare_models(model_df, use_added_variables=True)
    cv_all = pd.concat([cv_base, cv_plus], ignore_index=True).sort_values("temporal_cv_rmse_log_mean")
    test_all = pd.concat([test_base, test_plus], ignore_index=True).sort_values("RMSE_log")
    cv_all.to_csv(OUTPUT_DIR / "model_comparison_cv_all.csv", index=False)
    test_all.to_csv(OUTPUT_DIR / "model_comparison_test_all.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot = cv_all.copy()
    plot["label"] = plot["model"] + "\n" + plot["feature_set"].str.replace("_", " ")
    bars = ax.bar(plot["label"], plot["temporal_cv_rmse_log_mean"], color="#4E79A7")
    ax.set_title("Temporal CV RMSE by Model and Feature Set")
    ax.set_ylabel("RMSE on log1p spending")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    save_fig(fig, "optimized_temporal_cv_model_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    plot = test_all.copy()
    plot["label"] = plot["model"] + "\n" + plot["feature_set"].str.replace("_", " ")
    bars = ax.bar(plot["label"], plot["MAE_dollars"], color="#59A14F")
    ax.set_title("2022 Test MAE by Model and Feature Set")
    ax.set_ylabel("MAE in dollars")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.bar_label(bars, fmt="$%.0f", padding=2, fontsize=8)
    save_fig(fig, "optimized_test_mae_model_comparison.png")
    return cv_all, test_all, ctx_base, ctx_plus


def outlier_robustness_check(ctx: dict[str, Any], chosen_model_name: str) -> pd.DataFrame:
    data = ctx["data"]
    features = ctx["features"]
    test_mask = data["wave"].eq(2022)
    X_test = data.loc[test_mask, features]
    y_test = data.loc[test_mask, "total_oop_medical_spending"]
    y_test_log = data.loc[test_mask, "log_total_oop_medical_spending"]
    model = ctx["trained"][chosen_model_name]
    if chosen_model_name == "Two-part hurdle":
        pred = predict_hurdle_dollars(model, X_test)
    elif chosen_model_name == "Tweedie regressor":
        pred = np.clip(model.predict(X_test), 0, None)
    else:
        pred = predict_log_model_dollars(model, X_test)

    threshold = y_test.quantile(0.99)
    keep = y_test.lt(threshold)
    rows = []
    full = evaluate_predictions(f"{chosen_model_name} full 2022", y_test, pred, y_test_log)
    full["n"] = len(y_test)
    full["rule"] = "Full 2022 test set"
    rows.append(full)
    trimmed = evaluate_predictions(
        f"{chosen_model_name} trimmed 2022",
        y_test.loc[keep],
        pred[keep.to_numpy()],
        y_test_log.loc[keep],
    )
    trimmed["n"] = int(keep.sum())
    trimmed["rule"] = "Drops top 1 percent of 2022 OOP spending for robustness check"
    trimmed["top_1_percent_threshold"] = threshold
    rows.append(trimmed)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "outlier_robustness_check.csv", index=False)
    return out


def tune_cv_chosen_model(model_df: pd.DataFrame, chosen_model_name: str) -> tuple[pd.DataFrame, Any, Path]:
    features, numeric, binary, categorical, _ = make_feature_schema(include_added_variables=True)
    data = model_df[["wave", "total_oop_medical_spending", "log_total_oop_medical_spending"] + features].copy()
    train_mask, test_mask, _, temporal_cv = temporal_train_validation_masks(data)
    X_train = data.loc[train_mask, features]
    X_test = data.loc[test_mask, features]
    y_train_log = data.loc[train_mask, "log_total_oop_medical_spending"]
    y_test = data.loc[test_mask, "total_oop_medical_spending"]
    y_test_log = data.loc[test_mask, "log_total_oop_medical_spending"]
    preprocess = make_preprocess(numeric, binary, categorical)

    if chosen_model_name == "Gradient boosting":
        pipe = Pipeline(
            [
                ("preprocess", preprocess),
                ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        )
        grid = {
            "model__max_iter": [120, 200],
            "model__learning_rate": [0.03, 0.06],
            "model__max_leaf_nodes": [15, 31],
        }
    elif chosen_model_name == "Ridge regression":
        pipe = Pipeline([("preprocess", preprocess), ("model", Ridge())])
        grid = {"model__alpha": [0.1, 1.0, 10.0, 30.0]}
    elif chosen_model_name == "Elastic Net":
        pipe = Pipeline([("preprocess", preprocess), ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=6000))])
        grid = {"model__alpha": [0.001, 0.01, 0.05], "model__l1_ratio": [0.1, 0.3, 0.6]}
    else:
        pipe = Pipeline(
            [
                ("preprocess", preprocess),
                ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
            ]
        )
        grid = {
            "model__n_estimators": [100, 180],
            "model__max_depth": [8, 12],
            "model__min_samples_leaf": [5, 15],
        }

    default_cv = cross_validate(
        pipe,
        X_train,
        y_train_log,
        cv=temporal_cv,
        scoring={"rmse_log": "neg_root_mean_squared_error", "mae_log": "neg_mean_absolute_error"},
        n_jobs=1,
    )
    default_cv_rmse_log = float(-default_cv["test_rmse_log"].mean())
    default_cv_mae_log = float(-default_cv["test_mae_log"].mean())

    search = GridSearchCV(pipe, param_grid=grid, cv=temporal_cv, scoring="neg_root_mean_squared_error", n_jobs=1)
    search.fit(X_train, y_train_log)
    best_model = search.best_estimator_
    pred = np.expm1(best_model.predict(X_test))
    result = evaluate_predictions(f"Tuned {chosen_model_name}", y_test, pred, y_test_log)
    result["best_params"] = str(search.best_params_)
    result["temporal_cv_default_rmse_log"] = default_cv_rmse_log
    result["temporal_cv_default_mae_log"] = default_cv_mae_log
    result["temporal_cv_best_rmse_log"] = float(-search.best_score_)
    result["temporal_cv_rmse_log_improvement"] = default_cv_rmse_log - result["temporal_cv_best_rmse_log"]
    result["temporal_cv_rmse_log_improvement_percent"] = (
        result["temporal_cv_rmse_log_improvement"] / default_cv_rmse_log * 100
    )
    result_df = pd.DataFrame([result])
    result_df.to_csv(OUTPUT_DIR / "tuned_final_model_results.csv", index=False)
    model_path = MODEL_DIR / "hrs_oop_spending_final_optimized.joblib"
    joblib.dump(
        {
            "model": best_model,
            "features": features,
            "numeric_features": numeric,
            "binary_features": binary,
            "categorical_features": categorical,
            "chosen_model_name": chosen_model_name,
            "best_params": search.best_params_,
        },
        model_path,
    )
    return result_df, best_model, model_path


def predict_oop_dollars(model: Pipeline, person: dict[str, Any]) -> float:
    row = pd.DataFrame([person])
    return float(max(np.expm1(model.predict(row)[0]), 0.0))


def predict_oop_dollars_from_raw(person: dict[str, Any], model_path: Path | None = None) -> float:
    if model_path is None:
        model_path = MODEL_DIR / "hrs_oop_spending_final_optimized.joblib"
    bundle = joblib.load(model_path)
    features = bundle["features"]
    row = {feature: person.get(feature, np.nan) for feature in features}
    return predict_oop_dollars(bundle["model"], row)


def make_deployment_example(best_model: Pipeline, features: list[str]) -> pd.DataFrame:
    example = {
        "age": 68,
        "bmi_self_reported": 31.2,
        "cigarettes_per_day": 0,
        "chronic_condition_count": 3,
        "num_private_hi_plans": 1,
        "social_security_income_monthly": 1600,
        "obese": 1,
        "current_smoker": 0,
        "medicare_coverage": 1,
        "sex": "female",
        "race": "white",
        "education": "degree_3",
    }
    example = {key: example[key] for key in features}
    prediction = predict_oop_dollars(best_model, example)
    out = pd.DataFrame([{**example, "predicted_oop_dollars": prediction}])
    out.to_csv(OUTPUT_DIR / "deployment_example_prediction.csv", index=False)
    return out
