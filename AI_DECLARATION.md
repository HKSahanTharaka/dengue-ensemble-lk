<!-- AI-GENERATED (Claude via Antigravity) -->
# AI Declaration

| File or notebook section | What it does | Written by |
|---|---|---|
| .gitignore | Keeps raw data, .venv, caches and large models out of git | Claude via Antigravity |
| requirements.txt | Lists the Python libraries the project needs | Claude via Antigravity |
| AI_DECLARATION.md | This table of AI-written files | Claude via Antigravity |
| README.md | Project title, data source stub and setup instructions | Claude via Antigravity |
| notebooks/01_data_audit.ipynb | Audits the raw CSV: shape, gaps, plots, correlations, leakage-risk table (not yet run) | Claude via Antigravity |
| src/features.py | Builds next-week target, case/climate lags, past-only rolling stats, seasonality; drops leaky columns | Claude via Antigravity |
| notebooks/02_features.ipynb | Runs features.py, tests lag and no-leak (asserts), saves data/processed/features.csv | Claude via Antigravity |
| src/features.py (update) | Adds this-week cases and weather (lag 0) and rolling stats that include this week; still no future data | Claude via Antigravity |
| notebooks/02_features.ipynb (update) | Verifies cases_zscore uses all years (leak), stronger no-future-data tests, saves 51-feature table | Claude via Antigravity |
| src/split.py | Time-based train/test split, district one-hot ColumnTransformer, TimeSeriesSplit with fold date table | Claude via Antigravity |
| src/evaluate.py | MAE, MSE, RMSE, R2 and median absolute error on the original case scale; expm1 helper | Claude via Antigravity |
| notebooks/03_split_and_baseline.ipynb | Runs the split, CV folds, two naive baselines and a log1p demo | Claude via Antigravity |
| notebooks/04_models.ipynb | Trains 6 default ensemble models (bagging, RF, GBM, XGBoost, CatBoost, AdaBoost), scores them with the baselines | Claude via Antigravity |
| src/evaluate.py (update) | Adds fit_and_score: timed fit on log1p target, expm1 back, train/test scores | Claude via Antigravity |
| src/stacking.py | Base models, CatBoost wrapper, and TimeSeriesStacker (manual time-aware stacking with RidgeCV) | Claude via Antigravity |
| notebooks/05_stacking.ipynb | Voting and manual stacking models, StackingRegressor limitation check, meta-model weights, updated results table | Claude via Antigravity |
| notebooks/06_tuning.ipynb | Tunes Random Forest, XGBoost and CatBoost with GridSearchCV + TimeSeriesSplit(5), saves best_params.json and results_tuned.csv | Claude via Antigravity |
| notebooks/07_evaluation.ipynb | Final model comparison, 4-district plots, residual analysis, outbreak vs normal error, classification metrics, and 95% block bootstrap CIs | Claude via Antigravity |
| notebooks/08_interpretation.ipynb | Learning curves, built-in/permutation/SHAP feature importance, grouped importance, SHAP dependence plots, and draft interpretation | Claude via Antigravity |
