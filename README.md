# Predicting Out-of-Pocket Medical Spending (HRS 2018 to 2022)

DATA 975 Capstone, Drew University. Author: Weicheng (Wilson) Qian.

## What this project does

This project predicts individual annual out-of-pocket medical spending using the
Health and Retirement Study (HRS) core interviews for 2018, 2020, and 2022, joined
to the Cross-Wave Tracker File. It covers the full pipeline: data acquisition,
cleaning, exploratory analysis, feature engineering, model comparison, tuning,
and a small Flask app that serves the final model.

## Data

Data comes from the Health and Retirement Study (University of Michigan).

**The HRS data files are not included in this repository.** HRS requires each user
to register and agree to its conditions of use, so the raw and derived data files
are excluded in `.gitignore`. To reproduce the analysis, register at
https://hrsdata.isr.umich.edu, download the 2018, 2020, and 2022 core interview
files plus the Cross-Wave Tracker, and place them under `data/`.

## Method summary

- Target: total annual out-of-pocket medical spending, modeled as `log1p(spending)`.
- Split: temporal. Train on the 2018 and 2020 waves, test on 2022, which avoids
  using future data to predict the past.
- Validation: temporal cross-validation (train on 2018, validate on 2020).
- Variable types are handled separately: numeric features are imputed and scaled,
  binary features are imputed, and categorical features are one-hot encoded.
- Models compared: mean baseline, Ridge, Elastic Net, Random Forest, Gradient
  Boosting, a two-part hurdle model, and a Tweedie regressor.

## Results

The tuned Gradient Boosting model is the final choice, with an R-squared of about
0.15 on the log scale and a mean absolute error near $1,812 on the 2022 test wave.
Adding income and insurance variables raised R-squared from about 0.10 to 0.15.
Health and demographic variables alone explain only part of spending, so the model
is best read as a group-level estimate rather than an individual forecast.

## Repository layout

```
.
├── notebooks/      Assignment 6 modeling notebook
├── scripts/        cleaning, EDA, and modeling modules
├── outputs/
│   ├── models/     saved model bundle (.joblib)
│   └── figures/    EDA and model comparison charts
├── app/            Flask app that serves the model
├── environment.yml conda environment
└── requirements.txt
```

## How to reproduce

```bash
conda env create -f environment.yml
conda activate data975-medical
jupyter notebook
```

## How to run the app

```bash
cd app
python app.py
```

Then open http://127.0.0.1:5000 in a browser.

## Limitations

This is a course project built on survey data. The model explains a modest share
of the variation in spending, and it is not medical or financial advice.
