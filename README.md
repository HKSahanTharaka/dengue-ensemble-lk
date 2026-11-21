<!-- AI-GENERATED (Claude via Antigravity) -->
# Sri Lanka Weekly Dengue Case Forecasting (`dengue-ensemble-lk`)

This project predicts **next week's dengue cases per district in Sri Lanka** across all 25 districts using ensemble learning (bagging, boosting, voting, and stacking). On the 2022–2025 held-out test set, a tuned Random Forest (`test RMSE = 25.79`, `MAE = 11.95`, `R² = 0.831`) beats the naive persistence baseline (`RMSE = 27.68`, `MAE = 12.80`, `R² = 0.805`) by ~7% on RMSE. All models are trained on `log1p(next-week cases)` and converted back with `expm1` before scoring.

## Data
- **Dataset:** Kaggle `Dengue_Research_2006-2026` (`dengue_srilanka_full_2006_2025.csv`, 24,775 rows, 28 columns, 25 districts × 991 weeks).
- **Author:** [AUTHOR] | **License:** [LICENSE]
- The raw CSV is **not** committed to this repo (excluded in `.gitignore`). Place it in `data/raw/dengue_srilanka_full_2006_2025.csv`.

## Problem Setup
- **Target:** `target_next_cases` = next week's `cases` in the same district (`shift(-1)` inside `groupby("district")` after sorting by `start.date`), trained as `target_log = log1p(target_next_cases)`.
- **Features (51 total, after dropping 225 initial/final rows with missing lags or no next week):**
  - *Case lags (6):* `cases_lag_0`, `_1`, `_2`, `_3`, `_4`, `_8`
  - *Rolling cases (4):* `cases_roll_mean_4`, `cases_roll_max_4`, `cases_roll_mean_8`, `cases_roll_max_8`
  - *Climate (36):* `rainfall_mm`, `temp_max_c`, `temp_min_c`, `humidity_pct` at lags `0..8`
  - *Seasonality (2):* `week_sin`, `week_cos` (from `start.date` ISO week)
  - *District & size (3):* `district` (one-hot encoded inside `Pipeline`, native categorical in CatBoost), `population`, `population_density`
- **Dropped for leakage / redundancy:** `is_epidemic_year` (whole-year flag using future weeks), `cases_zscore` (uses district mean/std across all 2006–2025 years), `incidence_rate` (rescaled copy of `cases`), plus raw `year`/`week`/`cases_lag1`/`rainfall_lag2`/`rainfall_lag4` (rebuilt cleanly from `start.date`).
- **Time Split & CV:** Never shuffled. Train = `2007-02-17` to `2021-12-25` (19,400 rows); Test = `2022-01-01` to `2025-12-06` (5,150 rows). Tuning uses `TimeSeriesSplit(n_splits=5, gap=25)` on train only.

## Models
- **Bagging:** `BaggingRegressor` and `RandomForestRegressor` average deep trees on bootstrap samples to cut variance. Tuned Random Forest (`n_estimators=200`, `max_depth=8`, `min_samples_leaf=10`, `max_features=0.5`) eliminated the default forest's 2.41× train/test overfit gap.
- **Boosting:** `GradientBoostingRegressor`, `XGBRegressor`, `CatBoostRegressor` (native `district`), and `AdaBoostRegressor` fit trees sequentially to fix prior residuals. Tuned XGBoost (`n_estimators=200`, `max_depth=3`, `learning_rate=0.03`, `subsample=0.8`) and tuned CatBoost (`iterations=600`, `depth=4`, `learning_rate=0.03`, `l2_leaf_reg=5.0`) used shallow trees and slow learning to avoid chasing outbreak spikes.
- **Voting & Stacking:** `VotingRegressor` averaged default RF, XGBoost, and CatBoost (`26.80` RMSE). Because `StackingRegressor` fails with `TimeSeriesSplit` (`cross_val_predict` requires partitions), `TimeSeriesStacker` in `src/stacking.py` collects out-of-fold predictions manually and fits `RidgeCV` (`27.28` RMSE).

## Results (Test Set: 2022–2025)
| Model | Family | Test MAE | Test RMSE | Test MSE | Test R² | Train RMSE |
|---|---|---:|---:|---:|---:|---:|
| **Random Forest (tuned)** | bagging | 11.951 | **25.787** | 664.968 | **0.831** | 26.620 |
| CatBoost (tuned) | boosting | 12.094 | 25.921 | 671.882 | 0.829 | 27.975 |
| XGBoost (tuned) | boosting | **11.912** | 25.961 | 673.996 | 0.829 | 37.642 |
| Gradient Boosting | boosting | 12.035 | 26.054 | 678.814 | 0.827 | 27.213 |
| Voting (RF + XGB + CatBoost) | combined | 12.231 | 26.797 | 718.086 | 0.817 | 16.880 |
| Stacking (RF + XGB + CatBoost, RidgeCV) | combined | 12.509 | 27.282 | 744.324 | 0.811 | 14.795 |
| CatBoost | boosting | 12.645 | 27.613 | 762.495 | 0.806 | 21.955 |
| **Baseline: persistence (this week)** | baseline | 12.802 | **27.677** | 766.020 | **0.805** | 28.157 |
| Random Forest | bagging | 12.484 | 27.821 | 774.017 | 0.803 | 11.533 |
| Bagging (decision trees) | bagging | 12.525 | 27.878 | 777.170 | 0.802 | 11.746 |
| XGBoost | boosting | 12.825 | 28.338 | 803.057 | 0.796 | 19.025 |
| AdaBoost | boosting | 19.563 | 41.286 | 1704.539 | 0.567 | 61.763 |
| Baseline: district training average | baseline | 21.906 | 48.961 | 2397.207 | 0.390 | 76.893 |

Tuned Random Forest ranks first on test RMSE (`25.79` vs. persistence `27.68`), though 95% block-bootstrap intervals show the top three tuned models (`25.79–25.96`) are statistically tied. The improvement over persistence is clearest in normal weeks (`RMSE 18.42` vs. `21.76`), whereas `log1p` training makes models conservative on extreme outbreaks (`Recall 0.541` vs. `0.643`).

## How to Run
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Run notebooks in order (`notebooks/`):
1. `01_data_audit.ipynb` — audit raw CSV, gaps, skew, climate correlations, leakage table.
2. `02_features.ipynb` — build 51 leak-free features and save `data/processed/features.csv`.
3. `03_split_and_baseline.ipynb` — time split, `TimeSeriesSplit(5)`, naive baselines.
4. `04_models.ipynb` — 6 default bagging/boosting models vs. baselines.
5. `05_stacking.ipynb` — `VotingRegressor` and manual `TimeSeriesStacker`.
6. `06_tuning.ipynb` — `GridSearchCV` for RF, XGBoost, and CatBoost.
7. `07_evaluation.ipynb` — final table, residuals, outbreak/classification view, bootstrap CIs.
8. `08_interpretation.ipynb` — learning curves, 3-way & grouped importance, SHAP plots.
9. `09_hybrid.ipynb` — `ExponentialSmoothing` + RF residuals hybrid on top 5 districts.

```powershell
python src/train_final.py     # exports models/best_model.pkl, model_meta.json, test_predictions.csv
streamlit run app/app.py      # launches the 4-tab dashboard
```

## Repository Structure
```text
├── app/                 # Streamlit dashboard (app.py, interpretation.md, data/test_features.csv)
├── data/                # raw/ (gitignored) and processed/features.csv
├── models/              # best_model.pkl (5.5 MB), best_params.json, model_meta.json, test_predictions.csv
├── notebooks/           # 01_data_audit.ipynb .. 09_hybrid.ipynb
├── reports/             # results_*.csv and figures/
├── src/                 # features.py, split.py, evaluate.py, stacking.py, train_final.py
├── AI_DECLARATION.md
├── README.md
└── requirements.txt
```

## Limitations
1. **Reporting delays:** Real-world surveillance data arrives late and gets revised, whereas the model assumes `cases_lag_0` is finalised at week's end.
2. **Same-week weather inputs:** Using `lag_0` climate features assumes current-week weather is known when forecasting (live use requires weather forecasts or dropping `lag_0`).
3. **Single test period:** Evaluating only on `2022–2025` cannot rule out future shifts in dengue serotypes, post-2020 mobility, or reporting practices.
4. **Skewed counts & outbreak under-prediction:** `13%` zero weeks and rare spikes (max `2,631`) mean `log1p` training improves typical weeks but under-predicts sudden surges.
5. **Climate & case history only, district-level only:** No mosquito-density, serotype, or sub-district (MOH-area) data is available, so local urban hotspots (e.g., Kalutara, Colombo) show the largest errors.

## AI Usage
Claude via Antigravity assisted in writing the Python modules, Jupyter notebooks, Streamlit app, and documentation for this project. All generated code cells and files are tagged with `# AI-GENERATED (Claude via Antigravity)` and verified against the real dataset outputs. See [AI_DECLARATION.md](AI_DECLARATION.md) for the full file-by-file log.

## Links
- **Live App:** https://dengue-ensemble-lk-gcvc2nykbvvjanzje4gyql.streamlit.app/
- **Video Walkthrough:** [VIDEO LINK]
