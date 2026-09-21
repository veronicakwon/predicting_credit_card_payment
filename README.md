# Predicting Credit Card Payment Default

A machine learning project that predicts whether a credit card client will default on
their payment next month, using the [UCI "Default of Credit Card Clients"
dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (30,000
clients from a Taiwanese bank).

The project walks through EDA, feature engineering, model comparison (logistic
regression vs. XGBoost), a subgroup analysis restricted to clients who are currently up
to date on payments, and a cost-based threshold analysis for turning model scores into
an actionable collections policy.

## Dataset

Each row is one credit card client, described by:

- **Demographics**: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- **Repayment status** for the last 6 months (`SEP_REPAY_STATUS` ... `APR_REPAY_STATUS`,
  originally `PAY_0`...`PAY_6`): -2/-1/0 = no balance / paid in full / paid minimum,
  1-8 = months delinquent
- **Bill amounts** for the last 6 months (`SEP_BILL_AMT` ... `APR_BILL_AMT`)
- **Payment amounts** for the last 6 months (`SEP_PAY_AMT` ... `APR_PAY_AMT`)
- **Target**: `default.payment.next.month` (1 = defaulted, 0 = did not)

The raw CSV is stored at `data/raw/UCI_Credit_Card.csv` (also zipped at the repo root as
`UCI_Credit_Card.csv.zip`).

## Project structure

```
.
├── data/
│   └── raw/
│       └── UCI_Credit_Card.csv       # raw dataset
├── notebooks/
│   ├── 01_eda.ipynb                  # cleaning, renaming, exploratory analysis
│   ├── 02_baseline.ipynb             # baseline logistic regression / XGBoost models
│   ├── 03_feature_engineering.ipynb  # payment ratios, utilization, aggregate features
│   ├── 04_modeling.ipynb             # feature-set ablation study
│   ├── 05_current_subgroup.ipynb     # modeling only clients current on payments
│   ├── 06_threshold_and_cost.ipynb   # cost-sensitive threshold / capacity analysis
│   ├── credit_card_cleaned.csv       # output of 01_eda.ipynb
│   └── credit_card_featured.csv      # output of 03_feature_engineering.ipynb
└── UCI_Credit_Card.csv.zip
```

## Workflow

1. **`01_eda.ipynb`** — loads the raw data, renames the `PAY_*`/`BILL_AMT*`/`PAY_AMT*`
   columns to month names, collapses undocumented `EDUCATION`/`MARRIAGE` categories,
   and explores default rates by age, education, marital status, credit limit, and
   repayment status. Saves `credit_card_cleaned.csv`.
2. **`02_baseline.ipynb`** — trains baseline logistic regression and XGBoost models on
   the raw features with 5-fold cross-validation, reporting ROC-AUC and PR-AUC.
3. **`03_feature_engineering.ipynb`** — engineers payment-to-prior-bill ratios,
   utilization rates (bill / credit limit), and aggregate features (average/max/min/std
   utilization, months with no balance, bill growth ratio, trend features). Saves
   `credit_card_featured.csv`.
4. **`04_modeling.ipynb`** — ablation study measuring the incremental value of each
   engineered feature group (payment ratios, utilization, aggregates) on top of the raw
   features.
5. **`05_current_subgroup.ipynb`** — repeats the modeling/ablation exercise restricted
   to clients whose current repayment status is not delinquent, to see how predictive
   power changes for a "currently in good standing" subpopulation.
6. **`06_threshold_and_cost.ipynb`** — turns predicted probabilities into a decision
   rule by sweeping classification thresholds against an assumed cost of a missed
   default vs. a false alarm, and reports precision/recall at fixed outreach-capacity
   levels (5%/10%/20% of the portfolio).

## Requirements

- Python 3
- `pandas`, `numpy`, `matplotlib`
- `scikit-learn`
- `xgboost`

No `requirements.txt`/environment file is currently checked in — install the packages
above (any reasonably recent version) to run the notebooks.

## Running

The notebooks currently reference the author's local file paths (e.g.
`/Users/veron/Desktop/predicting_credit_card_payment/...`) rather than relative paths.
Update the `pd.read_csv(...)` path at the top of each notebook to point at this repo's
`data/raw/UCI_Credit_Card.csv` (for `01_eda.ipynb`) or the corresponding
`notebooks/credit_card_*.csv` file, then run the notebooks in order (01 → 06), since
each downstream notebook depends on a CSV produced by an earlier one.
