<!-- AI-GENERATED (Claude via Antigravity) -->
- **Recent case memory dominates:** Across built-in, permutation, and SHAP importance, `cases_roll_mean_4` and `cases_lag_0` drive most of the model's predictions because dengue incidence changes smoothly from week to week.
- **Tuning cured Random Forest's overfitting:** Capping tree depth (`max_depth=8`) and requiring at least `10` weeks per leaf eliminated the default forest's 2.4x train/test gap, making it slightly better than tuned CatBoost and XGBoost.
- **Climate adds a secondary signal:** Rainfall 4 weeks earlier (`rainfall_mm_lag_4`) is the strongest weather feature, matching the ~4-week mosquito breeding-to-reporting cycle, though weather helps mainly in normal weeks rather than sudden outbreaks.
