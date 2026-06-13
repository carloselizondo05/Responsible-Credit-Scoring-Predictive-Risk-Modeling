# Responsible Credit Scoring: Predictive Risk Modeling & Ethical Fairness Auditing

**CIND820 — Big Data Analytics Capstone Project**  
Toronto Metropolitan University | Carlos Elizondo | 2026

---

## Project Overview

Legacy credit scoring systems deny loans to a large population of creditworthy applicants — recent immigrants, young professionals, and low-income households — not because they are risky borrowers, but because they lack the credit history that traditional models require. This project builds an alternative credit risk pipeline using behavioral data from Home Credit Group's relational dataset and then goes a step further: it audits that model for fairness.

The central deliverable is not just a prediction — it is a **dual-model audit** that quantifies the accuracy cost of responsible lending. By building an unconstrained baseline (Model A) and an ethically constrained version (Model B), the project produces a measurable argument about what it actually costs a bank to comply with Canadian human rights and model risk regulations.

---

## Research Questions

**RQ1 — Exploratory (Feature Significance)**  
For consumer loan applicants who lack a formal credit score, which combination of behavioral indicators — such as payment history, installment patterns, and bureau records — alongside demographic features most reliably signals default risk, and what does that tell a lending team about how to evaluate thin-file applicants?

**RQ2 — Predictive (Consensus Feature Selection)**  
Using behavioral and demographic data available at the time of application, can we build a model that predicts each applicant's probability of default with enough precision to assign them to one of four lending tiers — low risk (0–10%), medium risk (10–25%), high risk (25–40%), or decline (40%+) — where each tier corresponds to a differentiated interest rate reflecting the actual risk the bank is absorbing?

**RQ3 — Prescriptive (Responsible Credit Scoring)**  
When the risk-tiering model is constrained to exclude features that act as proxies for protected characteristics under the Canadian Human Rights Act, how does its predictive performance change — and can an interest rate cap at 35% APR (Criminal Code, s.347) be applied so that no approved applicant receives a rate that would place them in unsustainable debt, even if that means reclassifying some high-risk applicants as declines?

---

## Dataset

**Home Credit Default Risk** — Kaggle Competition Dataset  
Source: [https://www.kaggle.com/c/home-credit-default-risk](https://www.kaggle.com/c/home-credit-default-risk)  
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

> **Note:** Raw data files are not included in this repository. Download directly from Kaggle and place in the `/data/raw/` directory before running any notebooks.

---

## Technical Pipeline

```
Stage 1 — Data Ingestion & Relational Aggregation
  └── SQLite joins across 7 tables on SK_ID_CURR
  └── GROUP BY aggregations → behavioral features per applicant
  └── Temporal leakage filter (records predating DAYS_DECISION only)

Stage 2 — Data Cleaning & Feature Engineering
  └── DAYS_EMPLOYED anomaly correction (365,243 placeholder)
  └── OWN_CAR_AGE / FLAG_OWN_CAR cross-reference imputation
  └── EXT_SOURCE two-stage imputation + missingness flags
  └── Composite features: Debt-to-Income Ratio, Payment-to-Annuity Ratio
  └── Proxy correlation audit (r > 0.70 threshold)

Stage 3 — Consensus Feature Selection
  └── 6-method voting table: Chi-Squared, Pearson, MLR, NB, DT, RF
  └── Features retained only on majority consensus

Stage 4 — Supervised Learning: Model Tournament
  └── Models: Naive Bayes, Decision Tree, Random Forest, Logistic Regression
  └── Full feature set vs. consensus feature set (before/after comparison)
  └── class_weight='balanced' for imbalanced classes
  └── Metrics: AUC-ROC, F1-Score, pAUC@FPR<0.1, Precision-Recall curves

Stage 5 — Unsupervised Learning: Customer Segmentation
  └── K-Means clustering on applicant pool
  └── Cluster count: Elbow Method + Silhouette Score
  └── Risk persona profiling and labeling

Stage 6 — Dual-Model Fairness Audit
  └── Model A: Unconstrained (full consensus feature set)
  └── Model B: Ethically constrained (proxies removed, threshold calibrated)
  └── Fairness metrics: Demographic Parity, Equalized Odds, Disparate Impact Ratio
  └── SHAP summary + force plots for both models
  └── Proxy leakage detection on high-SHAP remaining features
```

---

## Tools & Environment

| Tool | Purpose |
|---|---|
| Google Colab | Primary compute environment |
| Google Drive | Persistent storage across sessions |
| SQLite | Relational aggregation of 50M+ rows |
| Pandas / NumPy | Feature engineering and data manipulation |
| Scikit-Learn | Model training and evaluation |
| SHAP | Explainability and proxy leakage detection |
| Fairlearn / AI Fairness 360 | Fairness metrics computation |

---

## Repository Structure

```
├── data/
│   ├── raw/               # Place Kaggle CSVs here (not tracked by git)
│   └── processed/         # Aggregated flat table output from Stage 1
├── notebooks/
│   ├── 01_ingestion.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_model_tournament.ipynb
│   ├── 05_segmentation.ipynb
│   └── 06_fairness_audit.ipynb
├── outputs/
│   ├── figures/           # SHAP plots, confusion matrices, ROC curves
│   └── tables/            # Voting table, fairness metrics, model comparison
├── docs/
│   └── milestone_1_design.pdf
├── .gitignore
└── README.md
```

---

## Regulatory Context

This project is grounded in Canada's current AI governance environment. **OSFI Guideline E-23 (Model Risk Management)** requires federally regulated financial institutions to validate ML models for proxy discrimination, maintain explainability of automated decisions, and govern the full model lifecycle. The dual-model audit in this project simulates exactly the kind of fairness-accuracy analysis a Model Risk Management team would need to run before deploying any ML-assisted lending tool.

Protected characteristics under the **Canadian Human Rights Act** that guide the fairness audit:
- Age (derived from DAYS_BIRTH)
- Gender (CODE_GENDER)
- Marital status (NAME_FAMILY_STATUS)
- Family status (CNT_CHILDREN, CNT_FAM_MEMBERS)

---

## Status

| Milestone | Status |
|---|---|
| Milestone 1 — Project Design | ✅ Complete |
| Milestone 2 — Data Preparation & EDA | ✅ Complete |
| Milestone 3 — Modeling & Results | 🔄 In progress |
| Milestone 4 — Final Report | ⏳ Upcoming |

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
