# Responsible Credit Scoring: Predictive Risk Modeling & Ethical Fairness Auditing
**CIND820 — Big Data Analytics Capstone Project**
Toronto Metropolitan University | Carlos Elizondo | 2026

---

## Project Overview

Legacy credit scoring systems deny loans to a large population of creditworthy applicants — recent immigrants, young professionals, and low-income households — not because they are risky borrowers, but because they lack the credit history that traditional models require. This project builds an alternative credit risk pipeline using behavioral data from Home Credit Group's relational dataset, and audits that model for fairness under Canadian law.

The central deliverable is a **three-model audit**: an unconstrained baseline (Model A), a version with protected characteristics removed (Model B), and a version with SHAP-detected proxy features also removed (Model B+). Comparing these three across effectiveness, stability, fairness, and profitability answers the project's core question — what does responsible lending actually cost a bank? In this dataset, the answer is nothing: Model B+ matched Model A on every deciding metric and was the most profitable of the three.

**Full results are documented in the [Milestone 4 Final Report](Docs/Milestone_4.md).**

---

## Research Questions

**RQ1 — Exploratory (Feature Significance)**
For consumer loan applicants who lack a formal credit score, which combination of behavioral indicators (payment history, installment patterns, bureau records) alongside demographic features most reliably signals default risk, and what does that tell a lending team about how to evaluate thin-file applicants?

**RQ2 — Predictive (Risk Tiering)**
Using behavioral and demographic data available at the time of application, can we build a model that predicts each applicant's probability of default with enough precision to assign them to one of four lending tiers, each corresponding to a differentiated interest rate reflecting actual risk?

**RQ3 — Prescriptive (Responsible Credit Scoring)**
When the risk-tiering model is constrained to exclude features that act as proxies for protected characteristics under the Canadian Human Rights Act, how does predictive performance change, and can a 35% APR cap (Criminal Code, s.347) be applied so no approved applicant is placed in unsustainable debt?

---

## Key Findings

| Question | Answer | Evidence |
|---|---|---|
| RQ1 | Behavioral indicators (bureau utilization, payment history) outperform static demographics | Consensus feature selection kept 100% of financial ratios, only 17% of demographic features |
| RQ2 | Yes — calibrated tiers show a clean, monotonic default-rate progression (4.3% -> 15.2% -> 26.1% -> 41.4%) | Breakeven-rate analysis confirms the pricing is economically sound, not just statistically separated |
| RQ3 | Removing protected + proxy features cost nothing measurable — Model B+ matched or exceeded Model A on every deciding metric and was **$4.05M more profitable** on the test set | Paired t-test: only Accuracy significant (p=0.0295); AUC-ROC, Recall, F1, F2 not significant. Fairness metrics clear regulatory thresholds (Disparate Impact 0.9899) |

---

## Dataset

**Home Credit Default Risk** — Kaggle Competition Dataset
Source: https://www.kaggle.com/c/home-credit-default-risk
Owner: Home Credit Group | License: Non-commercial academic use

| Table | Rows (approx.) | Description |
|---|---|---|
| `application_train.csv` | 307,511 | Main application data — demographics, assets, financials, TARGET |
| `bureau.csv` | 1,716,428 | Credit history from external institutions |
| `bureau_balance.csv` | 27,299,925 | Monthly balances of bureau credits |
| `previous_application.csv` | 1,670,214 | Prior loan applications at Home Credit |
| `installments_payments.csv` | 13,605,401 | Payment history on prior loans |
| `POS_CASH_balance.csv` | 10,001,358 | Monthly POS and cash loan balances |
| `credit_card_balance.csv` | 3,840,312 | Monthly credit card snapshots |

> Raw data is not stored in this repository. Notebook 1 downloads it directly via the Kaggle API (`kagglehub`) — see **How to Run** below.

---

## Technical Pipeline

```
Stage 1 — Data Ingestion & Relational Aggregation                          [COMPLETE]
  |-- SQLite joins across 7 tables on SK_ID_CURR
  |-- GROUP BY aggregations -> behavioral features per applicant

Stage 2 — Data Cleaning & Feature Engineering                              [COMPLETE]
  |-- DAYS_EMPLOYED anomaly correction (365,243 -> NaN, then imputed)
  |-- Missing values treated by cause, not blanket-imputed: thin-file
      behavioral aggregates -> 0 + missingness flag, EXT_SOURCE_1/2/3 ->
      median + flag, OCCUPATION_TYPE -> explicit "Not_Employed" category
  |-- Outlier audit: 3 income data-entry errors removed (>99.9th pct.)
  |-- Categorical encoding: binary mapping, ordinal ranking for
      education, one-hot encoding for remaining categories -> 194 features

Stage 3 — Leakage-Safe Consensus Feature Selection & Baseline Modeling     [COMPLETE]
  |-- Train/test split performed BEFORE any preprocessing or feature
      selection (rebuilt after Milestone 3 instructor feedback flagged
      a leakage risk in the original pipeline)
  |-- 5-method voting table: Pearson Correlation, Chi-Squared,
      Decision Tree importance, Random Forest importance, Naive Bayes
      permutation importance — computed on training data only
  |-- Logistic Regression and Multiple Linear Regression excluded after
      a VIF test confirmed severe multicollinearity (11 of 14 tested
      features exceeded VIF > 10, two infinite-VIF pairs)
  |-- Feature set reduced from 194 -> 82 features (42.3% retained)
  |-- Baseline models compared via 5-fold stratified cross-validation;
      Random Forest selected as strongest (AUC-ROC 0.75 vs. 0.71 DT,
      0.60 NB) and most stable (std 0.0019 vs. 0.0057 DT)

Stage 4 — Model Refinement & Calibration                                   [COMPLETE]
  |-- GridSearchCV tested but rejected — tuned hyperparameters improved
      AUC-ROC marginally while dropping Recall from 0.65 to 0.51;
      default configuration kept instead
  |-- Further reduction to top-50 features tested and rejected (worse
      AND slower than the 82-feature set)
  |-- Probability calibration (CalibratedClassifierCV, sigmoid) applied
      to correct raw probabilities inflated by class_weight='balanced'

Stage 5 — Unsupervised Learning: Customer Segmentation                     [COMPLETE]
  |-- K-Means clustering, k=3 selected via Elbow Method + Silhouette
      Score (k=2's apparent peak is a degenerate single-applicant split)
  |-- Result: three substantial, overlapping clusters (not sharply
      distinct personas) differentiated mainly by credit card
      utilization — reported honestly as a risk continuum

Stage 6 — Dual-Model Fairness Audit                                        [COMPLETE]
  |-- Model A: unconstrained (82 features)
  |-- Model B: 6 protected features removed per Canadian Human Rights
      Act (gender, age x2, marital status, children count, family
      size) — 77 features
  |-- Model B+: 3 SHAP-detected proxy features also removed
      (FLAG_MISSING_DAYS_EMPLOYED, OCCUPATION_TYPE_Laborers,
      ORGANIZATION_TYPE_XNA) — 74 features
  |-- Fairness metrics (Fairlearn): Disparate Impact Ratio 0.9899,
      Demographic Parity Diff 0.0099, Equalized Odds Diff 0.0134 — all
      clear regulatory thresholds
  |-- Paired t-test (Model A vs. B, 5-fold): only Accuracy significant
      (p=0.0295); AUC-ROC, Recall, F1, F2 not significant
  |-- Business value: interest rate assignment (8.5%/18%/30%, capped at
      35% APR per Criminal Code s.347) and profitability analysis —
      Model B+ earned $4.05M more than Model A on the test set
```

---

## Tools & Environment

| Tool | Purpose |
|---|---|
| Google Colab | Primary compute environment |
| Google Drive | Persistent storage across sessions |
| SQLite | Relational aggregation of 50M+ rows |
| Pandas / NumPy | Feature engineering and data manipulation |
| Scikit-Learn | Model training, evaluation, calibration |
| Statsmodels | Variance Inflation Factor (multicollinearity testing) |
| SciPy | Paired t-tests, statistical comparisons |
| SHAP | Explainability and proxy leakage detection |
| Fairlearn | Fairness metrics (Demographic Parity, Equalized Odds, Disparate Impact) |

See `requirements.txt` for exact package versions.

---

## How to Run

1. Clone this repository or open the notebooks directly via the Colab links below.
2. Install dependencies: `pip install -r requirements.txt`
3. **Kaggle authentication:** Notebook 1 downloads the dataset automatically via `kagglehub`, using a Kaggle API token read from Colab Secrets (not hardcoded in the notebook). To run it yourself:
   - Generate a token at kaggle.com/settings -> API -> Create New Token
   - In Colab, click the key icon in the left sidebar -> add a secret named `KAGGLE_API_TOKEN` with your token value
4. Run the notebooks in order:
   - **01_Feature_Selection_&_Relational_Aggregation.ipynb** -> downloads the dataset, builds the SQLite relational aggregation, produces `master_feature_matrix.csv`
   - **02_EDA_Report.ipynb** -> produces the YData Profiling and SweetViz HTML reports
   - **03_Baseline_Modeling_Post_Instructor_Feedback.ipynb** -> leakage-safe preprocessing, consensus feature selection (194->82), baseline model tournament, 5-fold cross-validation
   - **04_Final_Analysis.ipynb** -> GridSearchCV tuning, Models B/B+, SHAP proxy audit, calibration, fairness metrics, business value/profitability, K-Means segmentation
5. Each notebook mounts Google Drive and expects intermediate files under `MyDrive/CIND820_Project/` — update the path in the first code cell if your Drive structure differs.

---

## Repository Structure

```
|-- Data/
|   |-- Processed/           # Link to master_feature_matrix.csv
|   |-- Kaggle_Link.md       # Link to the raw Kaggle dataset
|-- Docs/
|   |-- Milestone_1.md
|   |-- Milestone_2.md
|   |-- Milestone_3.md
|   |-- Milestone_4.md       # Final report
|   |-- Milestone 5/         # In progress — presentation
|   |-- Project Journal.md
|   |-- AI Journal.md
|   |-- Prof_Meeting.md
|-- Notebooks/
|   |-- 01_Feature_Selection_&_Relational_Aggregation/
|   |-- 02_EDA_Report/
|   |-- 03_Baseline_Modeling/
|   |   |-- 03_Baseline_Modeling_Post_Instructor_Feedback.ipynb
|   |   |-- Post_feedback Corrections.md
|   |-- 04_Final_Analysis/
|       |-- 04_Final_Analysis.ipynb
|-- Outputs/
|   |-- EDA_Reports/
|       |-- EDA_SweetViz_Report.html
|       |-- categorical_default_rates.png
|       |-- correlation_heatmap.png
|       |-- feature_summary_stats.csv
|       |-- home_credit_schema.png
|       |-- univariate_analysis.png
|-- requirements.txt
|-- .gitignore
|-- LICENSE
|-- README.md
```

---

## Figure Reference (Milestone 4 Report -> Source)

| Report Figure | Source Notebook | Section / Cell |
|---|---|---|
| Fig. 1 — Full vs. consensus feature metrics | Notebook 3 | Section 11 |
| Fig. 7 — Vote distribution | Notebook 3 | Section 10 |
| Fig. 8 — Risk tiers, baseline vs. optimized | Notebook 3 | Section 11 |
| Fig. 9 — Model B SHAP top-20 | Notebook 4 | Section 7, cell 34 |
| Fig. 10 — Confusion matrices, A vs. B | Notebook 4 | Section 6, cell 26 |
| Fig. 11 — Metrics comparison, A/B/B+ | Notebook 4 | Section 6, cell 27 |
| Fig. 12-15 — Clustering (elbow, scatterplots, radar) | Notebook 4 | Section 10 |
| Fig. 16 — Calibrated tier breakdown | Notebook 4 | Section 12 (bonus) |
| Fig. 17 — Model B+ profit by tier | Notebook 4 | Section 9, cell 61 |
| Fig. 18 — Three-way profit comparison, A/B/B+ | Notebook 4 | Section 12 (bonus), cell 93 |
| Fig. 19 — SHAP single-applicant breakdown | Notebook 4 | Section 12 (bonus), cell 94 |

---

## Notebooks

Notebooks are hosted on Google Colab (public access):
- 01 - Feature Selection & Relational Aggregation: https://colab.research.google.com/drive/1qXGVmgSG6hwApMIzogeSpWi_3wArz03L?usp=sharing
- 02 - EDA Report: https://colab.research.google.com/drive/1krCPjElX9ODE21Ygg3mYrwhHaa61sTUy?usp=sharing
- 03 - Baseline Modeling (Post-Feedback, Leakage-Safe): https://colab.research.google.com/drive/1SnDJ_Tv7wUszstMdAHpsSNQAsN8ppwex?usp=sharing
- 04 - Final Analysis: https://colab.research.google.com/drive/1gNBHM8wug1B_zcbZisi7FLx6tAvNWwel?usp=sharing

## EDA Reports

Due to file size limits, EDA reports are hosted externally:
- YData Profiling Report: https://drive.google.com/file/d/1xR8hVg6rj513P6TEtoJZGlzKfBRtgveo/view?usp=sharing
- SweetViz Comparison Report: https://drive.google.com/file/d/114bQMF8pew9wggJWO1e13_MUBsXHeHv3/view?usp=sharing

## GitHub Repository

Raw files and documentation: https://github.com/carloselizondo05/Responsible-Credit-Scoring-Predictive-Risk-Modeling

---

## Regulatory Context

This project is grounded in Canada's current AI governance environment. **OSFI Guideline E-23 (Model Risk Management)** requires federally regulated financial institutions to validate ML models for proxy discrimination, maintain explainability of automated decisions, and govern the full model lifecycle. The dual-model audit in this project simulates exactly the kind of fairness-accuracy analysis a Model Risk Management team would need to run before deploying any ML-assisted lending tool.

Protected characteristics under the **Canadian Human Rights Act**, cross-referenced against the full list of protected grounds:
- Age (`DAYS_BIRTH`, `AGE_YEARS`)
- Gender (`CODE_GENDER`)
- Marital status (`NAME_FAMILY_STATUS_Single / not married`)
- Family status (`CNT_CHILDREN`, `CNT_FAM_MEMBERS`)

Six features total, removed to build Model B. A SHAP-based proxy audit (mean rank-change threshold of mean + 2 standard deviations) then identified three additional features acting as stand-ins for the removed characteristics, which were removed to build Model B+.

The dataset is not Canadian; Canadian law is applied here as a design constraint to demonstrate the framework, with the intention that models would be retrained on Canadian applicant data before any real deployment.

---

## Status

| Milestone | Status |
|---|---|
| Milestone 1 — Project Design | Complete |
| Milestone 2 — Architecture & Data Audit | Complete |
| Milestone 3 — Initial Results & Coding | Complete |
| Milestone 4 — Final Results & Report | Complete |
| Milestone 5 — Final Presentation & Demo | In progress |

---

## Citation

If you reference this work:

```
Elizondo, C. (2026). Responsible Credit Scoring: Predictive Risk Modeling
& Ethical Fairness Auditing. CIND820 Capstone Project, Toronto Metropolitan University.
```

Dataset citation:
```
Montoya, A., Inversion, Odintsov, K., & Kotek, M. (2018).
Home Credit Default Risk [Kaggle competition].
https://kaggle.com/competitions/home-credit-default-risk
```

---

*This project is for academic purposes only. No real lending decisions are made or implied.*
