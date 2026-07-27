from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
HOMEWORK1_DIR = PROJECT_DIR / "data" / "homework 1"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

H18_STA_ZIP = HOMEWORK1_DIR / "h18core" / "h18sta.zip"
H20_STA_ZIP = HOMEWORK1_DIR / "h20core" / "h20sta.zip"
H22_STA_ZIP = HOMEWORK1_DIR / "h22core" / "H22sta.zip"
TRACKER_FILE = HOMEWORK1_DIR / "trk2022v1" / "trk2022tr_r.csv"

WAVES: dict[int, dict[str, Any]] = {
    2018: {
        "prefix": "Q",
        "zip": H18_STA_ZIP,
        "health": "h18c_r.dta",
        "spending": "h18n_r.dta",
        "physical": "h18i_r.dta",
        "tracker_age": "QAGE",
        "tracker_iwyear": "QIWYEAR",
        "tracker_weight": "QWGTR",
        "tracker_insamp": "QINSAMP",
    },
    2020: {
        "prefix": "R",
        "zip": H20_STA_ZIP,
        "health": "H20C_R.dta",
        "spending": "H20N_R.dta",
        "physical": None,
        "tracker_age": "RAGE",
        "tracker_iwyear": "RIWYEAR",
        "tracker_weight": "RWGTR",
        "tracker_insamp": "RINSAMP",
    },
    2022: {
        "prefix": "S",
        "zip": H22_STA_ZIP,
        "health": "H22C_R.dta",
        "spending": "H22N_R.dta",
        "physical": "H22I_R.dta",
        "tracker_age": "SAGE",
        "tracker_iwyear": "SIWYEAR",
        "tracker_weight": "SWGTR",
        "tracker_insamp": "SINSAMP",
    },
}

SPENDING_SUFFIXES = {
    "106": "oop_hospital",
    "119": "oop_nursing_home",
    "139": "oop_outpatient_surgery",
    "156": "oop_doctor_visits",
    "168": "oop_dental",
    "180": "oop_rx_drugs_monthly",
    "194": "oop_home_health",
    "239": "oop_other_health_service",
    "333": "oop_other_medical",
}

SPENDING_GATE_SUFFIXES = {
    "099": "gate_hospital_overnight",
    "114": "gate_nursing_home_overnight",
    "134": "gate_outpatient_surgery",
    "147": "gate_doctor_visit_count",
    "164": "gate_dental_visit",
    "175": "gate_takes_rx_regularly",
    "189": "gate_home_health_service",
    "203": "gate_paid_other_health_service",
    "332": "gate_other_medical_expense",
}

HEALTH_SUFFIXES = {
    "005": "high_blood_pressure_raw",
    "010": "diabetes_raw",
    "018": "cancer_raw",
    "030": "lung_disease_raw",
    "036": "heart_condition_raw",
    "053": "stroke_raw",
    "070": "arthritis_raw",
    "117": "current_smoker_raw",
    "118": "cigarettes_per_day",
    "139": "weight_lbs_self_reported",
    "141": "height_feet_self_reported",
    "142": "height_inches_self_reported",
}

PHYSICAL_SUFFIXES = {
    "834": "height_measured",
    "841": "weight_lbs_measured",
    "907": "waist_measured",
}

TRACKER_BASE_COLUMNS = ["HHID", "PN", "SEX", "RACE", "DEGREE"]
# Revised cleaning rules: special missing codes are handled by variable type instead of one global list.
# This prevents valid values such as cigarettes_per_day = 8 or 9 from being erased.
SPENDING_SPECIAL_CODES_BY_COLUMN = {
    "oop_hospital": {999998, 999999, 9999998, 9999999},
    "oop_nursing_home": {999998, 999999},
    "oop_outpatient_surgery": {99998, 99999},
    "oop_doctor_visits": {99998, 99999},
    "oop_dental": {99998, 99999, 999998, 999999},
    "oop_rx_drugs_monthly": {9998, 9999, 99998, 99999},
    "oop_home_health": {99998, 99999, 999998, 999999},
    "oop_other_health_service": {999998, 999999},
    "oop_other_medical": {99998, 99999, 999998, 999999},
}
CONDITION_COLUMNS = ["high_blood_pressure", "diabetes", "cancer", "lung_disease", "heart_condition", "stroke", "arthritis"]
SPENDING_COLUMNS = list(SPENDING_SUFFIXES.values())
HEALTH_YES_CODES = {1}
HEALTH_NO_CODES = {4, 5, 6}
SMOKING_YES_CODES = {1}
SMOKING_NO_CODES = {5}
RX_DRUGS_PERIOD_NOTE = "HRS codebook labels QN180/RN180/SN180 as AMT PAY O-O-P RX DRUGS PER MONTH; HRS notes that values are converted to monthly amounts where possible, so this project annualizes it by multiplying by 12."


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col.lower() == "hhid":
            rename_map[col] = "HHID"
        elif col.lower() == "pn":
            rename_map[col] = "PN"
    return df.rename(columns=rename_map)


def read_stata_member(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory() as tmp_dir:
        archive.extract(member, tmp_dir)
        return pd.read_stata(Path(tmp_dir) / member, convert_categoricals=False)


def read_wave_section(year: int, section: str) -> pd.DataFrame | None:
    target = WAVES[year][section]
    if target is None:
        return None
    return normalize_key_columns(read_stata_member(WAVES[year]["zip"], target))


def wave_var(prefix: str, section: str, suffix: str) -> str:
    return f"{prefix}{section}{suffix}"


def select_and_rename(df: pd.DataFrame, prefix: str, section: str, suffix_map: dict[str, str]) -> pd.DataFrame:
    rename = {}
    columns = ["HHID", "PN"]
    for suffix, normalized in suffix_map.items():
        raw = wave_var(prefix, section, suffix)
        if raw in df.columns:
            columns.append(raw)
            rename[raw] = normalized
    return df[columns].rename(columns=rename).copy()


def build_raw_long() -> pd.DataFrame:
    tracker_cols = TRACKER_BASE_COLUMNS + [
        WAVES[y][key] for y in WAVES for key in ["tracker_age", "tracker_iwyear", "tracker_weight", "tracker_insamp"]
    ]
    tracker = pd.read_csv(TRACKER_FILE, usecols=list(dict.fromkeys(tracker_cols)), dtype=str, low_memory=False)
    frames = []
    for year, config in WAVES.items():
        prefix = config["prefix"]
        spending_suffixes = {**SPENDING_SUFFIXES, **SPENDING_GATE_SUFFIXES}
        spending = select_and_rename(read_wave_section(year, "spending"), prefix, "N", spending_suffixes)
        health = select_and_rename(read_wave_section(year, "health"), prefix, "C", HEALTH_SUFFIXES)
        frame = spending.merge(health, on=["HHID", "PN"], how="left")
        physical = read_wave_section(year, "physical")
        if physical is not None:
            frame = frame.merge(select_and_rename(physical, prefix, "I", PHYSICAL_SUFFIXES), on=["HHID", "PN"], how="left")
        else:
            for col in PHYSICAL_SUFFIXES.values():
                frame[col] = np.nan
        wave_tracker = tracker[
            ["HHID", "PN", "SEX", "RACE", "DEGREE", config["tracker_age"], config["tracker_iwyear"], config["tracker_weight"], config["tracker_insamp"]]
        ].copy()
        wave_tracker = wave_tracker.rename(
            columns={
                "SEX": "sex_raw",
                "RACE": "race_raw",
                "DEGREE": "education_raw",
                config["tracker_age"]: "age",
                config["tracker_iwyear"]: "interview_year",
                config["tracker_weight"]: "survey_weight",
                config["tracker_insamp"]: "in_sample_status",
            }
        )
        frame = frame.merge(wave_tracker, on=["HHID", "PN"], how="left")
        frame["wave"] = year
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_spending(series: pd.Series, column_name: str) -> pd.Series:
    numeric = to_numeric(series)
    numeric = numeric.mask(numeric < 0)
    numeric = numeric.mask(numeric.isin(SPENDING_SPECIAL_CODES_BY_COLUMN.get(column_name, set())))
    return numeric


def apply_spending_gate_zero_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_rules = [
        {
            "spending_component": "oop_hospital",
            "gate_variable": "gate_hospital_overnight",
            "zero_condition": lambda s: to_numeric(s).eq(5),
            "codebook_basis": "N099 asks whether the respondent had an overnight hospital stay; code 5 means No.",
        },
        {
            "spending_component": "oop_nursing_home",
            "gate_variable": "gate_nursing_home_overnight",
            "zero_condition": lambda s: to_numeric(s).eq(5),
            "codebook_basis": "N114 asks whether the respondent was overnight in a nursing home/skilled nursing facility; code 5 means No.",
        },
        {
            "spending_component": "oop_outpatient_surgery",
            "gate_variable": "gate_outpatient_surgery",
            "zero_condition": lambda s: to_numeric(s).eq(5),
            "codebook_basis": "N134 asks whether the respondent had outpatient surgery; code 5 means No.",
        },
        {
            "spending_component": "oop_doctor_visits",
            "gate_variable": "gate_doctor_visit_count",
            "zero_condition": lambda s: to_numeric(s).eq(0),
            "codebook_basis": "N147 records the number of doctor/clinic/ER/house-call visits; code 0 means none.",
        },
        {
            "spending_component": "oop_dental",
            "gate_variable": "gate_dental_visit",
            "zero_condition": lambda s: to_numeric(s).eq(5),
            "codebook_basis": "N164 asks whether the respondent saw a dentist; code 5 means No.",
        },
        {
            "spending_component": "oop_rx_drugs_monthly",
            "gate_variable": "gate_takes_rx_regularly",
            "zero_condition": lambda s: to_numeric(s).eq(5),
            "codebook_basis": "N175 asks whether the respondent regularly takes prescription medications; code 5 means No.",
        },
        {
            "spending_component": "oop_home_health",
            "gate_variable": "gate_home_health_service",
            "zero_condition": lambda s: to_numeric(s).eq(5),
            "codebook_basis": "N189 asks whether a medically trained person came to the home; code 5 means No.",
        },
        {
            "spending_component": "oop_other_health_service",
            "gate_variable": "gate_paid_other_health_service",
            "zero_condition": lambda s: to_numeric(s).eq(5),
            "codebook_basis": "N203 asks whether respondent/spouse/partner had to pay for other health services; code 5 means No.",
        },
        {
            "spending_component": "oop_other_medical",
            "gate_variable": "gate_other_medical_expense",
            "zero_condition": lambda s: to_numeric(s).eq(5),
            "codebook_basis": "N332 asks whether there were other out-of-pocket medical expenses; code 5 means No.",
        },
    ]
    audit_rows = []
    for rule in gate_rules:
        component = rule["spending_component"]
        gate = rule["gate_variable"]
        if component not in df.columns or gate not in df.columns:
            continue
        zero_mask = rule["zero_condition"](df[gate])
        filled_mask = zero_mask & df[component].isna()
        df.loc[filled_mask, component] = 0
        audit_rows.append({
            "spending_component": component,
            "gate_variable": gate,
            "component_missing_set_to_zero_n": int(filled_mask.sum()),
            "component_missing_set_to_zero_rate": float(filled_mask.mean()),
            "codebook_basis": rule["codebook_basis"],
        })
    return df, pd.DataFrame(audit_rows)


def clean_numeric_with_codes(series: pd.Series, missing_codes: set[int | float] | None = None) -> pd.Series:
    numeric = to_numeric(series)
    numeric = numeric.mask(numeric < 0)
    if missing_codes:
        numeric = numeric.mask(numeric.isin(missing_codes))
    return numeric


def binary_from_codes(series: pd.Series, yes_codes: set[int], no_codes: set[int]) -> pd.Series:
    numeric = to_numeric(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[numeric.isin(yes_codes)] = 1.0
    out[numeric.isin(no_codes)] = 0.0
    return out


def yes_no_to_binary(series: pd.Series) -> pd.Series:
    return binary_from_codes(series, HEALTH_YES_CODES, HEALTH_NO_CODES)


def smoking_to_binary(series: pd.Series) -> pd.Series:
    return binary_from_codes(series, SMOKING_YES_CODES, SMOKING_NO_CODES)


def decode_sex(series: pd.Series) -> pd.Series:
    numeric = clean_numeric_with_codes(series, {-8, 8, 9})
    return numeric.map({1: "male", 2: "female"})


def decode_race(series: pd.Series) -> pd.Series:
    numeric = to_numeric(series)
    return numeric.map({0: "unknown", 1: "white", 2: "black", 7: "other"}).astype("object")


def decode_education(series: pd.Series) -> pd.Series:
    numeric = to_numeric(series)
    return numeric.map({
        0: "degree_0", 1: "degree_1", 2: "degree_2", 3: "degree_3",
        4: "degree_4", 5: "degree_5", 6: "degree_6", 9: "unknown"
    }).astype("object")


def fill_smoking_within_person(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filled = df.sort_values(["HHID", "PN", "wave"]).copy()
    audit_rows = []
    smoking_cols = ["current_smoker", "cigarettes_per_day"]
    for col in smoking_cols:
        before_missing = filled[col].isna()
        person_filled = filled.groupby(["HHID", "PN"], sort=False)[col].transform(lambda s: s.ffill().bfill())
        filled[f"{col}_crosswave_filled"] = before_missing & person_filled.notna()
        filled[col] = person_filled
        audit_rows.append({
            "feature": col,
            "missing_rate_before_crosswave_fill": float(before_missing.mean()),
            "crosswave_filled_rate": float(filled[f"{col}_crosswave_filled"].mean()),
            "missing_rate_after_crosswave_fill": float(filled[col].isna().mean()),
        })
    nonsmoker_cig_missing = filled["current_smoker"].eq(0) & filled["cigarettes_per_day"].isna()
    filled.loc[nonsmoker_cig_missing, "cigarettes_per_day"] = 0
    filled["cigarettes_per_day_set_zero_for_nonsmoker"] = nonsmoker_cig_missing
    audit_rows.append({
        "feature": "cigarettes_per_day_zero_for_nonsmokers",
        "missing_rate_before_crosswave_fill": np.nan,
        "crosswave_filled_rate": float(nonsmoker_cig_missing.mean()),
        "missing_rate_after_crosswave_fill": float(filled["cigarettes_per_day"].isna().mean()),
    })
    filled = filled.sort_index()
    return filled, pd.DataFrame(audit_rows)


def clean_analysis_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    log_rows = [{"step": "raw_combined", "rows": len(df), "columns": df.shape[1], "note": "Combined 2018, 2020, and 2022 selected raw variables."}]
    for col in SPENDING_COLUMNS:
        df[col] = clean_spending(df[col], col)
    df, gate_audit = apply_spending_gate_zero_rules(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gate_audit.to_csv(OUTPUT_DIR / "spending_gate_zero_rule_audit.csv", index=False)
    df["oop_rx_drugs_annualized"] = df["oop_rx_drugs_monthly"] * 12
    spending_for_total = [c for c in SPENDING_COLUMNS if c != "oop_rx_drugs_monthly"] + ["oop_rx_drugs_annualized"]
    df["spending_components_present"] = df[spending_for_total].notna().sum(axis=1)
    df["total_oop_medical_spending"] = df[spending_for_total].sum(axis=1, min_count=1)
    log_rows.append({"step": "clean_spending", "rows": len(df), "columns": df.shape[1], "note": "Cleaned spending components with variable-specific missing codes and used codebook gate variables to set clear no-service components to 0. " + RX_DRUGS_PERIOD_NOTE})
    for raw_col, clean_col in [
        ("high_blood_pressure_raw", "high_blood_pressure"),
        ("diabetes_raw", "diabetes"),
        ("cancer_raw", "cancer"),
        ("lung_disease_raw", "lung_disease"),
        ("heart_condition_raw", "heart_condition"),
        ("stroke_raw", "stroke"),
        ("arthritis_raw", "arthritis"),
    ]:
        df[clean_col] = yes_no_to_binary(df[raw_col])
    df["current_smoker"] = smoking_to_binary(df["current_smoker_raw"])
    df["cigarettes_per_day"] = clean_numeric_with_codes(df["cigarettes_per_day"], {-8, 98, 99})
    df, smoking_audit = fill_smoking_within_person(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    smoking_audit.to_csv(OUTPUT_DIR / "smoking_crosswave_fill_audit.csv", index=False)
    df["chronic_condition_count"] = df[CONDITION_COLUMNS].sum(axis=1, min_count=1)
    log_rows.append({"step": "binary_health_variables", "rows": len(df), "columns": df.shape[1], "note": "Converted chronic disease indicators with revised yes/no codes and filled smoking variables within person across waves."})
    numeric_cleaning = {
        "age": {-8, 998, 999},
        "survey_weight": {-8, 9999998, 9999999},
        "weight_lbs_self_reported": {-8, 998, 999},
        "height_feet_self_reported": {-8, 8, 9},
        "height_inches_self_reported": {-8, 98, 99},
        "height_measured": {-8, 98, 99, 998, 999},
        "weight_lbs_measured": {-8, 998, 999},
        "waist_measured": {-8, 998, 999},
    }
    for col, missing_codes in numeric_cleaning.items():
        df[col] = clean_numeric_with_codes(df[col], missing_codes)
    df["height_total_inches_self_reported"] = df["height_feet_self_reported"] * 12 + df["height_inches_self_reported"]
    df["bmi_self_reported"] = df["weight_lbs_self_reported"] / (df["height_total_inches_self_reported"] ** 2) * 703
    df["bmi_self_reported"] = df["bmi_self_reported"].where(df["bmi_self_reported"].between(10, 80))
    df["bmi_category"] = pd.cut(df["bmi_self_reported"], bins=[0, 18.5, 25, 30, 80], labels=["underweight", "normal", "overweight", "obese"], right=False)
    df["obese"] = (df["bmi_self_reported"] >= 30).astype("float").where(df["bmi_self_reported"].notna())
    log_rows.append({"step": "bmi_engineering", "rows": len(df), "columns": df.shape[1], "note": "Calculated BMI and obesity indicator from self-reported height and weight."})
    df["sex"] = decode_sex(df["sex_raw"])
    df["race"] = decode_race(df["race_raw"])
    df["education"] = decode_education(df["education_raw"])
    df["log_total_oop_medical_spending"] = np.log1p(df["total_oop_medical_spending"])
    log_rows.append({"step": "demographic_and_log_target", "rows": len(df), "columns": df.shape[1], "note": "Decoded demographics as categorical variables and created log1p target so zero-spending respondents stay in the dataset."})
    before = len(df)
    df = df[df["total_oop_medical_spending"].notna()].copy()
    log_rows.append({"step": "drop_missing_outcome_only", "rows": len(df), "columns": df.shape[1], "note": f"Removed {before - len(df)} rows without usable spending components; zero-spending and low-spending respondents are retained."})
    return df, pd.DataFrame(log_rows)


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_eda_outputs(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    by_wave = df.groupby("wave").agg(
        n=("total_oop_medical_spending", "size"),
        mean_total_oop=("total_oop_medical_spending", "mean"),
        median_total_oop=("total_oop_medical_spending", "median"),
        p75_total_oop=("total_oop_medical_spending", lambda s: s.quantile(0.75)),
        max_total_oop=("total_oop_medical_spending", "max"),
        mean_bmi=("bmi_self_reported", "mean"),
        obesity_rate=("obese", "mean"),
        current_smoker_rate=("current_smoker", "mean"),
        mean_chronic_conditions=("chronic_condition_count", "mean"),
        zero_spending_n=("total_oop_medical_spending", lambda s: int((s == 0).sum())),
        zero_spending_rate=("total_oop_medical_spending", lambda s: float((s == 0).mean())),
    ).reset_index()
    by_wave.to_csv(OUTPUT_DIR / "eda_summary_by_wave.csv", index=False)
    by_condition = []
    for condition in CONDITION_COLUMNS + ["current_smoker", "obese"]:
        temp = df.groupby(["wave", condition], dropna=False).agg(
            n=("total_oop_medical_spending", "size"),
            median_total_oop=("total_oop_medical_spending", "median"),
            mean_total_oop=("total_oop_medical_spending", "mean"),
        ).reset_index().rename(columns={condition: "group_value"})
        temp["factor"] = condition
        by_condition.append(temp)
    pd.concat(by_condition, ignore_index=True).to_csv(OUTPUT_DIR / "eda_spending_by_health_factors.csv", index=False)
    missing = df.isna().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["variable", "missing_rate"]
    missing.to_csv(OUTPUT_DIR / "eda_missingness_after_cleaning.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="log_total_oop_medical_spending", by="wave", ax=ax)
    ax.set_title("Log Total Out-of-Pocket Medical Spending by Wave")
    ax.set_xlabel("Wave")
    ax.set_ylabel("log1p(total OOP spending)")
    fig.suptitle("")
    save_fig(fig, "eda_log_spending_by_wave.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    by_wave.plot(kind="bar", x="wave", y="median_total_oop", legend=False, ax=ax, color="#3B6EA8")
    ax.set_title("Median Total Out-of-Pocket Medical Spending by Wave")
    ax.set_xlabel("Wave")
    ax.set_ylabel("Median spending")
    save_fig(fig, "eda_median_spending_by_wave.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    by_wave.plot(kind="bar", x="wave", y="mean_chronic_conditions", legend=False, ax=ax, color="#4E8B57")
    ax.set_title("Mean Chronic Condition Count by Wave")
    ax.set_xlabel("Wave")
    ax.set_ylabel("Mean chronic condition count")
    save_fig(fig, "eda_chronic_count_by_wave.png")
    plot_df = df.dropna(subset=["bmi_self_reported", "total_oop_medical_spending"])
    sample = plot_df.sample(min(len(plot_df), 6000), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample["bmi_self_reported"], sample["log_total_oop_medical_spending"], s=8, alpha=0.25)
    ax.set_title("BMI vs. Log Medical Spending")
    ax.set_xlabel("BMI from self-reported height/weight")
    ax.set_ylabel("log1p(total OOP spending)")
    save_fig(fig, "eda_bmi_vs_log_spending.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    chronic_summary = df.groupby("chronic_condition_count")["total_oop_medical_spending"].median().reset_index().dropna()
    ax.plot(chronic_summary["chronic_condition_count"], chronic_summary["total_oop_medical_spending"], marker="o")
    ax.set_title("Median Spending by Number of Chronic Conditions")
    ax.set_xlabel("Number of chronic conditions")
    ax.set_ylabel("Median total OOP spending")
    save_fig(fig, "eda_spending_by_chronic_count.png")


def to_markdown_table(df: pd.DataFrame) -> str:
    text_df = df.fillna("").astype(str)
    headers = text_df.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in text_df.values.tolist():
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def write_markdown_summary(df: pd.DataFrame, cleaning_log: pd.DataFrame) -> None:
    by_wave = pd.read_csv(OUTPUT_DIR / "eda_summary_by_wave.csv")
    lines = [
        "# Phase 2 Cleaning and EDA Summary",
        "",
        "## Cleaning Steps",
        "",
        to_markdown_table(cleaning_log),
        "",
        "## Final Cleaned EDA Dataset",
        "",
        f"Rows: {len(df):,}",
        f"Columns: {df.shape[1]:,}",
        "",
        "## EDA Summary by Wave",
        "",
        to_markdown_table(by_wave),
    ]
    (OUTPUT_DIR / "phase2_cleaning_eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
