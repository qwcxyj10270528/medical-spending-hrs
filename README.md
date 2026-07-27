# Predicting Out-of-Pocket Medical Spending Using HRS Data

DATA 975 Capstone, Drew University  
Author: Weicheng (Wilson) Qian

## Project Overview

This project studies factors associated with individual annual out-of-pocket medical spending using the Health and Retirement Study (HRS). The main research question is:

**Which demographic, health, behavioral, income, and insurance-related factors are associated with medical out-of-pocket spending among older adults?**

The project uses HRS Core Interview data from 2018, 2020, and 2022, together with the Cross-Wave Tracker File. The workflow covers data preparation, cleaning, exploratory data analysis, feature engineering, model comparison, model tuning, and a Flask web application that serves the final trained model.

## Data Source

The data come from the Health and Retirement Study (HRS), conducted by the University of Michigan. HRS is a longitudinal survey of older adults in the United States and includes information on demographics, health conditions, health behaviors, income, insurance, and medical spending.

The raw HRS data files are **not included** in this GitHub repository. HRS requires each user to register and agree to its data-use conditions, so the raw and derived data files are excluded by `.gitignore`.

To reproduce the analysis, users need to download the required files directly from HRS and place them in the local `data/` folder. The required files include:

- 2018 HRS Core Interview File
- 2020 HRS Core Interview File
- 2022 HRS Core Interview File
- HRS Cross-Wave Tracker File

## Repository Structure

```text
medical-spending-hrs/
├── notebooks/
│   └── assignment6_modeling.ipynb
├── scripts/
│   ├── phase2_cleaning_eda.py
│   ├── modeling_optimization.py
│   └── medical_cost_analysis_full_project_workflow.py
├── outputs/
│   ├── models/
│   │   └── hrs_oop_spending_final_optimized.joblib
│   └── figures/
├── app/
│   ├── app.py
│   ├── model/
│   │   └── hrs_oop_spending_final_optimized.joblib
│   └── templates/
│       └── index.html
├── data/
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```

The `data/` folder is intentionally excluded from GitHub because HRS data cannot be redistributed publicly.

## Methodology

The target variable is annual out-of-pocket medical spending. Because medical spending is highly right-skewed, the model predicts the log-transformed target:

```python
log1p(total_oop_medical_spending)
```

Predictions are converted back to dollar scale with:

```python
np.expm1(prediction)
```

The project uses a temporal split:

- 2018 and 2020 data are used for training and validation
- 2022 data are used as the final test set

This setup avoids using future data to predict earlier outcomes and makes the evaluation more realistic for a longitudinal survey project.

Variables are processed according to their type:

- Numeric variables: age, BMI, cigarettes per day, chronic condition count, private insurance plan count, and monthly Social Security income
- Binary variables: obesity status, current smoking status, and Medicare coverage
- Categorical variables: sex, race, and education

Categorical variables are handled with one-hot encoding. Numeric, binary, and categorical variables are processed separately inside the modeling pipeline so that training and prediction use the same transformations.

## Models Compared

The project compares several modeling approaches:

- Baseline mean model
- Ridge Regression
- Elastic Net
- Random Forest
- Gradient Boosting
- Two-part hurdle model
- Tweedie Regressor

The final selected model is the tuned Gradient Boosting model, based on temporal validation performance.

## Key Results

The final model performs better than the baseline model, but the overall predictive power remains modest. This means the model can capture some group-level spending patterns, but it should not be interpreted as a precise individual medical cost calculator.

Key findings include:

- Out-of-pocket medical spending is highly right-skewed.
- Chronic condition burden is associated with higher median out-of-pocket spending.
- Income and insurance variables improve model performance.
- The model performs better on the log scale than on the raw dollar scale.
- Extreme high-cost cases strongly affect dollar-scale RMSE.

## Flask App

This repository includes a Flask web application that serves the final trained model.

To run the app locally:

```bash
cd app
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The app allows users to enter demographic, health, income, and insurance information and returns a predicted annual out-of-pocket spending amount.

The Flask app uses the same fitted pipeline as the notebook. This means missing value handling, scaling, and one-hot encoding remain consistent between training and prediction.

The app also displays education categories using user-friendly labels based on the HRS education codebook, while keeping the model-facing encoded values unchanged.

## Reproducibility

To reproduce the project:

```bash
conda env create -f environment.yml
conda activate base
jupyter notebook
```

Then open:

```text
notebooks/assignment6_modeling.ipynb
```

Because HRS data cannot be redistributed, users must download the data separately and place it in the local `data/` folder.

## Limitations

This project is a course-based predictive modeling project and should not be interpreted as medical or financial advice.

The model has limited individual-level predictive power because medical spending is affected by many factors that may not be fully captured in the available survey variables. Insurance and income variables are interpreted as predictors associated with spending, not as causal effects.

The Flask app is intended for local demonstration and model-serving practice.
