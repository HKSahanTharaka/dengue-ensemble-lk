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
