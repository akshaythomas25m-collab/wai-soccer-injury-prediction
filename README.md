# ⚽ Soccer Injury Prediction & Workload Management

**WAI (Working with AI) — Individual Sports Analytics Project**

### ▶️ Try it live — [severe-injury-risk-simulator.streamlit.app](https://severe-injury-risk-simulator.streamlit.app)

Set an injury's attributes and the model scores it, live, with SHAP explaining
that individual prediction. A second tab looks up any of the 4,081 players in
the dataset. Source in [`app/`](app/); first load may take ~30 seconds if the
app has been idle.

## 📋 Project Overview

Predicting severe injury risk in European football players using machine learning to help clubs manage squad rotation and reduce the financial impact of player injuries.

## 💰 Business Case

- Premier League injuries cost clubs **£250m** in 2024/25 (Howden Injury Index)
- **£1.19bn** spent on injured player wages over 5 seasons
- 22,596 injuries across Europe's Top 5 leagues (2020-2025), costing **€3.45bn**
- A model that prevents even 1-2 injuries pays for itself

## 🎯 Classification Target

Binary: **Severe Injury (Yes/No)** — where severe = 28+ days out

## 📊 Headline Result

| Metric | Value |
|---|---|
| Best model | **XGBoost** (`max_depth=3`, `learning_rate=0.05`) |
| Test ROC-AUC | **0.747** |
| Recall on severe injuries | **0.704** — catches 7 of every 10 |
| Precision on severe injuries | 0.526 — nearly half of all flags are false alarms |
| Test set | 3,906 injuries / 1,021 players, none seen in training |

All three models finish within **0.006 AUC** of each other. No model is meaningfully better than the others here, which is itself a result: the available signal is largely accessible to a linear model.

## 🔍 Principal finding — a target leak caught after modelling

An earlier run scored **0.801** AUC. It was wrong.

Five features summarised each player's injury history, built by aggregating their entire Transfermarkt record. That record spans 1973–2025 and therefore **contains the very injuries being predicted**:

- **50.9%** of all Transfermarkt injury records fall inside the 2020–25 modelling window
- For **54%** of players, their whole recorded history sits inside it
- The median player has **3** career injuries — one row is a third of its own feature value

Rebuilding the features *as-of* each injury date (only injuries starting strictly earlier) cost two-thirds to four-fifths of the apparent signal in every one:

| Feature | corr (leaked) | corr (as-of) | signal lost |
|---|---|---|---|
| `avg_days_missed` | 0.268 | 0.056 | 79% |
| `max_days_missed` | 0.151 | 0.052 | 65% |
| `total_games_missed` | 0.130 | 0.043 | 67% |
| `total_injuries` | −0.112 | −0.034 | 70% |
| `distinct_injury_types` | −0.118 | −0.041 | 65% |

Refitting on the identical split, grids and seed:

| Model | AUC (leaked) | AUC (corrected) | Change |
|---|---|---|---|
| XGBoost | 0.8012 | **0.7472** | −0.0539 |
| Random Forest | 0.7921 | 0.7435 | −0.0486 |
| Logistic Regression | 0.7758 | 0.7417 | −0.0341 |

**0.747 is the result. 0.801 is the finding.**

## 🧠 What actually predicts severity

Correcting the leak changed the answer, not just the score. The model is now led by **injury type** — `is_illness` (mean |SHAP| 0.48), `injury_category_other` (0.33), `is_knee` (0.14) — with the strongest history feature down at 0.10.

Corroborated by the chi-square tests: injury category scores **2,165.6** against severity, an order of magnitude above league (235.7) or position (24.7).

> What kind of injury a player sustains predicts its severity. Who the player is barely does.

A secondary finding runs against intuition: players with **longer** injury records suffer slightly **less** severe injuries (median 10 prior injuries for severe cases vs 12 for non-severe). Durable players who play often accumulate many minor knocks, so a long record signals availability rather than fragility.

## 🤖 Models Used

- **Logistic Regression** — interpretable baseline
- **Random Forest** — nonlinear patterns + feature importance
- **XGBoost** — best performer with hyperparameter tuning

All tuned with `GridSearchCV` over `StratifiedGroupKFold`, scored on ROC-AUC, `RANDOM_STATE = 42`.

## 📊 Datasets

| Dataset | Source | Records |
|---|---|---|
| European Football Injuries 2020-2025 | Kaggle | 15,603 |
| Transfermarkt Player Datalake | GitHub | 93,000+ profiles / 143,195 injury records |
| PL Injuries & Team Performance | Kaggle | context only |

Data files are **not committed** — see [`data/README.md`](data/README.md) for what you need and where to put it.

## 🛠️ Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, SHAP, Matplotlib, Seaborn, Google Colab

Note on imbalance handling: the class split is 35.5 / 64.5 (**1.8:1**) — mild. `class_weight='balanced'` and `scale_pos_weight` are used rather than SMOTE, which is designed for far more extreme imbalance and would manufacture noise here.

## 📁 Repository Structure

```
├── app/                  # Live Streamlit simulator (deployed from this repo)
├── data/                 # Sources and setup instructions (files not committed)
├── notebooks/            # Colab notebook + paste-ready phase cells
├── reports/              # PDF deliverables (EDA, Assumptions, Modelling, Exec Summary)
├── dashboard/            # Results dashboard
├── presentation/         # Final slide deck + video link
├── prompt-logbook/       # AI prompt interaction log
└── README.md
```

## ▶️ Running it

Written for Google Colab, loading data from Google Drive so uploads survive a runtime restart.

1. Open `notebooks/WAI_Soccer_Injury_Prediction.ipynb` in Colab
2. Create a `wai_data` folder in your Google Drive and put the four CSVs in it (see `data/README.md`)
3. **Runtime → Run all**, clicking through the Drive authorisation prompt

A full run takes ~15 minutes; the three grid searches in Phase 5 and the re-run in Phase 5b are the slow parts. To run locally, `pip install -r requirements.txt` and replace the Drive mount in the Phase 1 load cell with local paths.

## ⚠️ Known limitations

- **No exposure data.** Minutes played, training load, match congestion and travel are unavailable, and are among the strongest known injury predictors in the sports-science literature. This caps what any model on this data can achieve.
- **Only recorded injuries appear.** Knocks managed without a reported absence are absent, so the model learns the severity distribution of *reported* injuries.
- **One feature still looks ahead.** `injuries_in_dataset` counts a player's injuries across the whole window and should be made as-of or dropped.
- **Threshold tuning predates the leak correction** and needs repeating against the corrected model.
- The 28-day threshold is a modelling choice following clinical convention, not a property of the data. Every relationship reported is associational, not causal.

## 📄 Deliverables

- [x] Executive Summary (PDF) — `reports/`
- [x] Assumptions Document (PDF) — `reports/`, includes the leakage repair
- [x] EDA Analysis (PDF) — `reports/`
- [x] Modelling Techniques with Confusion Matrices & ROC (PDF) — `reports/`
- [x] Live prediction app — [severe-injury-risk-simulator.streamlit.app](https://severe-injury-risk-simulator.streamlit.app)
- [ ] Tableau / Power BI Dashboard
- [x] Interim results dashboard (HTML) — `dashboard/`
- [ ] Final Presentation (8-10 slides)
- [ ] 3-Minute Video Presentation
- [ ] Prompt Logbook
- [x] Dataset + Repository

## 📝 Status

- 🟢 **Phase 0** — Environment Setup: COMPLETE
- 🟢 **Phase 1** — Data Acquisition: COMPLETE
- 🟢 **Phase 2** — Data Cleaning & Merging: COMPLETE (0% → 97.2% match rate)
- 🟢 **Phase 3** — Exploratory Data Analysis: COMPLETE
- 🟢 **Phase 4** — Assumptions Document: COMPLETE
- 🟢 **Phase 5** — Predictive Modeling: COMPLETE
- 🟢 **Phase 5b** — Leakage Repair: COMPLETE
- ⏭️ **Phase 6** — NLP Component: SKIPPED (optional)
- 🟡 **Phase 7** — Dashboard: HTML built; Tableau version outstanding
- 🟢 **Phase 8** — Executive Summary: COMPLETE
- 🟢 **Phase 9** — Live Demo App: COMPLETE (Streamlit Community Cloud)
- ⬜ **Phase 10** — Final Presentation & Video
