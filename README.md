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

> **Note:** Raw data files are not included in this repository. Download directly from Kaggle and place in `Data/raw/` before running any notebooks, or use the Kaggle link stored in `Data/`.

---

## Technical Pipeline
```
Stage 1 — Data Ingestion & Relational Aggregation                          [COMPLETE]
  └── SQLite joins across 7 tables on SK_ID_CURR
  └── GROUP BY aggregations → behavioral features per applicant
  └── Deliberate feature exclusions: housing block (~40 cols),
      document flags (FLAG_DOCUMENT_2–21), contact flags dropped
 
Stage 2 — Data Cleaning & Feature Engineering                              [COMPLETE]
  └── DAYS_EMPLOYED anomaly correction (365,243 → NaN, then imputed)
  └── Missing values grouped and treated by cause, not blanket-imputed:
      thin-file behavioral aggregates → 0 + missingness flag,
      EXT_SOURCE_1/2/3 → median + flag, OCCUPATION_TYPE → explicit
      "Not_Employed" category or education-group mode
  └── Composite features: Debt-to-Income Ratio, Annuity-to-Income Ratio,
      Credit-to-Goods Ratio, AGE_YEARS, EMPLOYED_YEARS
  └── Outlier audit: 3 income data-entry errors (>10M) removed using
      99.9th-percentile evidence
  └── Categorical encoding: binary mapping, ordinal ranking for education,
      one-hot encoding for remaining multi-category columns
 
Stage 3 — Consensus Feature Selection & Baseline Modeling                  [COMPLETE]
  └── 5-method voting table: Pearson Correlation, Chi-Squared,
      Decision Tree importance, Random Forest importance,
      Naive Bayes permutation importance
  └── Logistic Regression tested and excluded as a 6th method after a
      Variance Inflation Factor (VIF) check confirmed severe
      multicollinearity (11 of 14 tested features exceeded VIF > 10,
      including infinite VIF between DAYS_BIRTH/AGE_YEARS and
      DAYS_EMPLOYED/EMPLOYED_YEARS)
  └── Top-10-per-method features protected from elimination regardless
      of vote count; remaining features kept at a 2-of-5 vote threshold
  └── Feature set reduced from 194 → 89 features (45.9% retained)
  └── Baseline models: Naive Bayes, Decision Tree, Random Forest, trained
      and compared on identical stratified 80/20 split, identical seed,
      class_weight='balanced' for the ~92:8 class imbalance
  └── Random Forest selected as best baseline (AUC-ROC 0.75 vs. 0.71 DT,
      0.60 NB) and retrained on the 89-feature consensus set — every
      metric held steady or improved after feature reduction
  └── Predicted probabilities binned into 4 risk tiers (Low/Medium/High/
      Decline) and validated against actual default rate per tier
 
Stage 4 — Model Refinement & Calibration                                   [PLANNED — Milestone 4]
  └── Probability calibration (CalibratedClassifierCV) — current raw
      probabilities are inflated by class_weight='balanced' and do not
      yet reflect true real-world default likelihood
  └── Interest rate cap: 35% APR maximum (Criminal Code, s.347)
  └── Hyperparameter tuning budget equalization (GridSearchCV)
 
Stage 5 — Unsupervised Learning: Customer Segmentation                     [PLANNED — Milestone 4]
  └── K-Means clustering on applicant pool
  └── Cluster count: Elbow Method + Silhouette Score
  └── Risk persona profiling and labeling
 
Stage 6 — Dual-Model Fairness Audit                                        [PLANNED — Milestone 4]
  └── Model A: Unconstrained (current consensus feature set)
  └── Model B: Ethically constrained (proxies removed per Canadian
      Human Rights Act — age, gender, marital status, family status)
  └── Fairness metrics: Demographic Parity, Equalized Odds,
      Disparate Impact Ratio (minimum 0.80 threshold)
  └── SHAP global summary + local force plots for both models
  └── Proxy leakage check: OCCUPATION_TYPE vs. CODE_GENDER (Cramér's V
      already computed in Milestone 3 as a baseline reference point)
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
| Statsmodels | Variance Inflation Factor (multicollinearity testing) |
| SciPy | Cramér's V and point-biserial association testing |
| SHAP | Explainability and proxy leakage detection (Milestone 4) |
| Fairlearn / AI Fairness 360 | Fairness metrics computation (Milestone 4) |
 
See `requirements.txt` for exact package list.

---

## How to Run
 
1. Clone this repository or open the notebooks directly via the Colab links below.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the Home Credit Default Risk dataset from Kaggle (link above) and place the raw CSVs in `Data/raw/`, or use the Kaggle link stored in `Data/`.
4. Run the notebooks in order:
   - `01_Feature_Selection_&_Relational_Aggregation.ipynb` → produces `master_feature_matrix.csv`
   - `02_EDA_Report.ipynb` → produces the YData Profiling and SweetViz HTML reports
   - `03_Baseline_Modeling.ipynb` → data audits, cleaning, encoding, consensus feature selection, baseline and optimized Random Forest models, 4-tier risk breakdown
5. Each notebook mounts Google Drive and expects `master_feature_matrix.csv` at `MyDrive/CIND820_Project/master_feature_matrix.csv` — update the path in the first code cell if your Drive structure differs.

---

## Repository Structure
 
```
├── Data/
│   ├── Processed/          # Link to master_feature_matrix.csv
│   └── Kaggle_Link.md      # Link to the raw Kaggle dataset
├── Notebooks/
│   ├── 01_Feature_Selection_&_Relational_Aggregation/
│   │   └── 01_Feature_Selection_&_Relational_Aggregation.ipynb
│   ├── 02_EDA_Report/
│   │   ├── 02_EDA_Report.ipynb
│   │   └── Colab_Link.md
│   ├── 03_Baseline_Modeling/
│   │   ├── 03_Baseline_Modeling.ipynb
│   │   └── 03_Baseline_Modeling.pdf     # compiled, no-code-required version
│   ├── 05_Segmentation/                 # planned — Milestone 4
│   └── 06_Fairness_Audit/               # planned — Milestone 4
├── Outputs/
│   ├── EDA_Reports/
│   │   ├── EDA_SweetViz_Report.html
│   │   ├── categorical_default_rates.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_summary_stats.csv
│   │   ├── home_credit_schema.png
│   │   └── univariate_analysis.png
│   ├── Figures/             # confusion matrices, ROC curves, tier breakdowns
│   └── Tables/               # voting scoreboard, VIF results, model comparison
├── Docs/
│   ├── Milestone_1.md
│   ├── Milestone_2.md
│   ├── Milestone_3.md
│   ├── Milestone_4/          # planned
│   ├── Milestone_5/          # planned
│   ├── Prof_Meeting.md
│   ├── Project_Journal.md
│   └── AI_Journal.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```
 
---

## EDA Reports
 
Due to file size limits, EDA reports are hosted externally:
- [YData Profiling Report](https://drive.google.com/file/d/1xR8hVg6rj513P6TEtoJZGlzKfBRtgveo/view?usp=sharing) — Full automated profiling (Google Drive)
- [SweetViz Comparison Report](https://drive.google.com/file/d/114bQMF8pew9wggJWO1e13_MUBsXHeHv3/view?usp=sharing) — Defaulter vs repaid comparison (Google Drive)

## Notebooks
 
Notebooks are hosted on Google Colab (public access):
- [01 - Feature Selection & Relational Aggregation](https://colab.research.google.com/drive/1qXGVmgSG6hwApMIzogeSpWi_3wArz03L?usp=sharing)
- [02 - EDA Report](https://colab.research.google.com/drive/1krCPjElX9ODE21Ygg3mYrwhHaa61sTUy?usp=sharing)
- [03 - Baseline Modeling](https://colab.research.google.com/drive/1wrD-oXeDEZZe-skFgcu3y88zXQXCflul?usp=sharing)

## GitHub Repository
 
Raw files and documentation: https://github.com/carloselizondo05/Responsible-Credit-Scoring-Predictive-Risk-Modeling
 
---
 
## Regulatory Context
 
This project is grounded in Canada's current AI governance environment. **OSFI Guideline E-23 (Model Risk Management)** requires federally regulated financial institutions to validate ML models for proxy discrimination, maintain explainability of automated decisions, and govern the full model lifecycle. The dual-model audit planned for Milestone 4 simulates exactly the kind of fairness-accuracy analysis a Model Risk Management team would need to run before deploying any ML-assisted lending tool.
 
Protected characteristics under the **Canadian Human Rights Act** that will guide the Milestone 4 fairness audit:
- Age (derived from DAYS_BIRTH)
- Gender (CODE_GENDER)
- Marital status (NAME_FAMILY_STATUS)
- Family status (CNT_CHILDREN, CNT_FAM_MEMBERS)
Both `CODE_GENDER` and `DAYS_BIRTH` are retained in the Milestone 3 baseline feature set for direct before/after comparison once Model B is built, and are not treated as protected in this milestone's models.
 
---
 
## Status
 
| Milestone | Status |
|---|---|
| Milestone 1 — Project Design | ✅ Complete |
| Milestone 2 — Architecture & Data Audit | ✅ Complete |
| Milestone 3 — Initial Results & Coding | ✅ Complete |
| Milestone 4 — Final Results & Report | ⏳ Upcoming |
| Milestone 5 — Final Presentation & Demo | ⏳ Upcoming |
 
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
