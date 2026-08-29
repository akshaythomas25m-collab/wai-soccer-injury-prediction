# ⚽ Soccer Injury Prediction & Workload Management

**WAI (Working with AI) — Individual Sports Analytics Project**

## 📋 Project Overview

Predicting severe injury risk in European football players using machine learning to help clubs manage squad rotation and reduce the financial impact of player injuries.

## 💰 Business Case

- Premier League injuries cost clubs £250m in 2024/25 (Howden Injury Index)
- £1.19bn spent on injured player wages over 5 seasons
- 22,596 injuries across Europe's Top 5 leagues (2020-2025), costing €3.45bn
- A model that prevents even 1-2 injuries pays for itself

## 🎯 Classification Target

**Binary: Severe Injury (Yes/No)** — where severe = 28+ days out

## 🤖 Models Used

- **Logistic Regression** — Interpretable baseline
- **Random Forest** — Nonlinear patterns + feature importance
- **XGBoost** — Best performer with hyperparameter tuning

## 📊 Datasets

| Dataset | Source | Records |
| --- | --- | --- |
| European Football Injuries 2020-2025 | Kaggle | 15,603 |
| Transfermarkt Player Datalake | GitHub | 93,000+ |
| PL Injuries & Team Performance | Kaggle | — |

## 🛠️ Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, Matplotlib, Seaborn, Tableau Public, Google Colab

## 📁 Repository Structure

```
├── data/                 # Raw and cleaned datasets
├── notebooks/            # Jupyter/Colab notebooks
├── reports/              # PDF deliverables (EDA, Assumptions, Modeling, Executive Summary)
├── dashboard/            # Tableau workbook or published link
├── presentation/         # Final slide deck + video link
├── prompt-logbook/       # AI prompt interaction log
└── README.md
```

## 📄 Deliverables

- [ ] Executive Summary (PDF)
- [ ] EDA Analysis (PDF)
- [ ] Assumptions Document (PDF)
- [ ] Modelling Techniques with Confusion Matrices & ROC (PDF)
- [ ] Tableau / Power BI Dashboard
- [ ] Final Presentation (8-10 slides)
- [ ] 3-Minute Video Presentation
- [ ] Prompt Logbook
- [ ] Dataset + Repository

## 📝 Status

- 🟢 **Phase 0 — Environment Setup:** COMPLETE
- ⬜ **Phase 1 — Data Acquisition:** IN PROGRESS
- ⬜ Phase 2 — Data Cleaning & Merging
- ⬜ Phase 3 — Exploratory Data Analysis
- ⬜ Phase 4 — Assumptions Document
- ⬜ Phase 5 — Predictive Modeling
- ⬜ Phase 6 — NLP Component (Optional)
- ⬜ Phase 7 — Dashboard
- ⬜ Phase 8 — Executive Summary
- ⬜ Phase 9 — Final Presentation & Video
