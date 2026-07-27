"""
Flask app that serves the tuned model from Assignment 6.

Run locally:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# --------------------------------------------------------------------------

















































# Load the model bundle ONCE at startup, not on every request.
# The .joblib file is a dictionary, not a bare pipeline. It contains:
#   "model"                -> the fitted sklearn Pipeline
#   "features"             -> the 12 column names, in the exact order used
#   "numeric_features" / "binary_features" / "categorical_features"
#   "chosen_model_name", "best_params"
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "hrs_oop_spending_final_optimized.joblib"

bundle = joblib.load(MODEL_PATH)
PIPELINE = bundle["model"]
FEATURES = bundle["features"]
NUMERIC_FEATURES = bundle["numeric_features"]
BINARY_FEATURES = bundle["binary_features"]
CATEGORICAL_FEATURES = bundle["categorical_features"]
MODEL_NAME = bundle.get("chosen_model_name", "model")

# Allowed values for the dropdowns. These must match the training data exactly,
# because a value the encoder never saw is ignored by handle_unknown="ignore".
CATEGORY_CHOICES = {
    "sex": ["female", "male"],
    "race": ["white", "black", "other", "unknown"],
    "education": [
        "degree_0", "degree_1", "degree_2", "degree_3",
        "degree_4", "degree_5", "degree_6", "unknown",
    ],
}
CATEGORY_DISPLAY_LABELS = {
    "education": {
        "degree_0": "No degree",
        "degree_1": "GED",
        "degree_2": "High school diploma",
        "degree_3": "Two-year college degree",
        "degree_4": "Four-year college degree",
        "degree_5": "Master degree",
        "degree_6": "Professional degree (Ph.D., M.D., J.D.)",
        "unknown": "Unknown / missing",
    }
}
# Sensible defaults so the form is never blank on first load.
DEFAULTS = {
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

app = Flask(__name__)


def build_input_row(payload):
    """Turn a dict of raw form/JSON values into a single-row DataFrame.

    The column order must match bundle["features"], and numeric fields must be
    numbers rather than strings, otherwise the scaler will fail.
    """
    row = {}
    for feature in FEATURES:
        value = payload.get(feature, DEFAULTS.get(feature))
        if feature in NUMERIC_FEATURES or feature in BINARY_FEATURES:
            row[feature] = float(value)
        else:
            row[feature] = str(value)
    # Reindex to guarantee the exact training column order.
    return pd.DataFrame([row])[FEATURES]


def predict_dollars(payload):
    """Predict out-of-pocket spending in dollars.

    The pipeline was trained on log1p(spending), so the raw prediction is on
    the log scale and must be converted back with expm1.
    """
    row = build_input_row(payload)
    log_prediction = PIPELINE.predict(row)[0]
    dollars = float(np.expm1(log_prediction))
    return max(dollars, 0.0)  # never show a negative dollar amount


@app.route("/", methods=["GET", "POST"])
def home():
    """Render the form, and show a prediction after it is submitted."""
    form_values = dict(DEFAULTS)
    prediction = None
    error = None

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        try:
            prediction = predict_dollars(form_values)
        except (ValueError, TypeError) as exc:
            error = f"Could not read those inputs: {exc}"

    return render_template(
        "index.html",
        features=FEATURES,
        numeric_features=NUMERIC_FEATURES,
        binary_features=BINARY_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        category_choices=CATEGORY_CHOICES,
        category_display_labels=CATEGORY_DISPLAY_LABELS,
        values=form_values,
        prediction=prediction,
        error=error,
        model_name=MODEL_NAME,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint so the model can be called from code, not just the form.

    Example:
        curl -X POST http://127.0.0.1:5000/api/predict \
             -H "Content-Type: application/json" \
             -d '{"age": 68, "sex": "female"}'
    """
    payload = request.get_json(silent=True) or {}
    try:
        dollars = predict_dollars(payload)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "predicted_oop_dollars": round(dollars, 2),
        "model": MODEL_NAME,
    })


@app.route("/health")
def health():
    """Tiny endpoint to confirm the app is running and the model loaded."""
    return jsonify({"status": "ok", "model": MODEL_NAME, "n_features": len(FEATURES)})


if __name__ == "__main__":
    # debug=True auto-reloads when you edit the file. Turn it off for deployment.
    app.run(debug=True, port=5000)
