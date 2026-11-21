<!-- AI-GENERATED (Claude via Antigravity) -->
# AI Declaration

## How I used AI

_TODO: describe in your own words how you used AI in this project._

## File and Notebook Log

| File or notebook section | What it does | Written by |
|---|---|---|
| `.gitignore` | Excludes `data/raw/`, `.venv/`, caches, `catboost_info/`, and `models/large/` from git | AI |
| `requirements.txt` | Lists and pins Python library versions for reproducibility and Streamlit Cloud | AI |
| `README.md` | Project overview, setup, problem setup, results table, run steps, and limitations | AI |
| `AI_DECLARATION.md` | This file-by-file AI declaration table | AI |
| `src/features.py` | Builds next-week target, case/climate lags, rolling stats, seasonality; drops leaky columns | AI |
| `src/split.py` | Time-based train/test split, district one-hot `ColumnTransformer`, `TimeSeriesSplit(5)` | AI |
| `src/evaluate.py` | MAE, MSE, RMSE, R², MedAE on the original case scale and `expm1` conversion helper | AI |
| `src/stacking.py` | Base model builders, `CatBoostDistrict` wrapper, and manual `TimeSeriesStacker` | AI |
| `src/train_final.py` | Trains winning tuned Pipeline and exports `best_model.pkl`, metadata, predictions, final table | AI |
| `notebooks/01_data_audit.ipynb` | Audits raw CSV: shape, gaps, skew, seasonal/climate plots, leakage-risk table | AI |
| `notebooks/02_features.ipynb` | Runs `features.py`, verifies `cases_zscore` leak, runs no-future-data asserts, saves `features.csv` | AI |
| `notebooks/03_split_and_baseline.ipynb` | Time split, `TimeSeriesSplit(5)` fold table, naive baselines, `log1p` demo | AI |
| `notebooks/04_models.ipynb` | Trains 6 default bagging/boosting models and compares against naive baselines | AI |
| `notebooks/05_stacking.ipynb` | `VotingRegressor` and manual `TimeSeriesStacker`, checks `StackingRegressor` limitation | AI |
| `notebooks/06_tuning.ipynb` | Tunes RF, XGBoost, and CatBoost with `GridSearchCV` + `TimeSeriesSplit(5)` | AI |
| `notebooks/07_evaluation.ipynb` | Final comparison, 4-district plots, residuals, outbreak/classification view, bootstrap CIs | AI |
| `notebooks/08_interpretation.ipynb` | Learning curves, built-in/permutation/SHAP importance, grouped importance, draft interpretation | AI |
| `notebooks/09_hybrid.ipynb` | Hybrid `ExponentialSmoothing` + RF on residuals vs. RF alone on top 5 districts | AI |
| `app/app.py` | 4-tab Streamlit dashboard loading `best_model.pkl`, metadata, and results | AI |
| `app/interpretation.md` | 3-bullet interpretation displayed in Tab 3 of the Streamlit app | AI |
