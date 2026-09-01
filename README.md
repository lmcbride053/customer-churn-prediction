# Predicting Churn, From Signal to Retention Decision

Can churn be predicted accurately enough to support targeted retention, and under what economic conditions does acting on those predictions actually beat simpler alternatives? A model is only half the answer — this project treats the decision built on top of it as the real point.

**Full write-up with charts:** [case study on my portfolio](https://lmcbride053.github.io/project-churn.html)

## Two questions, not one

- **Data science question:** can customers at elevated churn risk be identified reliably?
- **Decision-science question:** can those predictions improve a retention strategy, and does that hold up once the assumptions behind it are stated honestly?

Most churn projects stop at the first question. This one spends most of its effort on the second.

## Dataset

7,043 Telco customers (11 dropped for missing billing data), leaving 7,032. Churn rate: 26.6%.

## Approach

- **EDA:** Mann-Whitney U for continuous variables, chi-square and Cramér's V for categorical associations — every relationship tested statistically, not eyeballed.
- **Feature engineering:** leakage-safe by construction — engineered features use fixed rules or per-row arithmetic computed before any split; the two features that do need a dataset statistic (spend-median threshold, one-hot encoding) are fit on training data only.
- **Modeling:** Logistic Regression, Random Forest, and XGBoost compared; best model (Logistic Regression) reaches 0.848 CV AUC-ROC.
- **Threshold selection:** locked via cross-validation on training data only, not tuned against the test set.
- **Calibration:** checked directly (Brier score 0.168).
- **Interpretation:** SHAP values and subgroup error analysis, including an honest look at where the model's blind spots are.
- **Retention strategy:** six targeting strategies compared, with an explicit sensitivity analysis over CLV, offer cost, and retention success rate.

## Key findings

- Tenure and monthly charges are the two strongest individual predictors of churn, both by a wide statistical margin (p = 6.0×10⁻²¹¹ and p = 8.5×10⁻⁵⁴).
- Three well-tuned models land within 1 point of each other on AUC-ROC — the simplest one is competitive, not clearly worse.
- The model's biggest blind spot (long-tenure, long-contract churners) is exactly where its strongest predictive signal also lives. Two-year contract holders have a 100% false-negative rate — stated plainly as a limitation, not hidden.
- Model-based retention targeting beats contacting every customer, but only above a **~39% retention success rate**. Below that, blanket outreach wins. The claim is conditional, not absolute.

## Stack

Python · pandas · scikit-learn · XGBoost · SHAP · Power BI

## Data

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data, widely used on Kaggle). The raw and processed data files aren't included in this repo since it's third-party data, not mine to redistribute — download it from the link above and drop it in `data/` to reproduce the notebooks.

## Structure

```
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_and_evaluation.ipynb
│   └── 04_retention_strategy.ipynb
├── data/                (not included — see Data section above)
│   ├── plots/            figures generated during EDA/modeling
│   ├── chosen_model.joblib
│   ├── fitted_preprocessor.joblib
│   ├── locked_threshold.json
│   ├── test_predictions.csv
│   └── retention_strategy_scenarios.csv
└── README.md
```

Run the notebooks in order (01 → 04); each one picks up the artifacts (processed features, fitted preprocessor, trained model, locked threshold) saved by the notebook before it.
