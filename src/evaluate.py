# AI-GENERATED (Claude via Antigravity)
"""Scoring helpers. All metrics are on the ORIGINAL case-count scale (not log)."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def to_cases(pred_log):
    """Undo log1p: expm1, and never predict fewer than 0 cases."""
    return np.clip(np.expm1(pred_log), 0, None)


def regression_scores(y_true, y_pred):
    """MAE, MSE, RMSE, R2 and median absolute error, in 'cases per district per week'."""
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred), "MedAE": median_absolute_error(y_true, y_pred)}
