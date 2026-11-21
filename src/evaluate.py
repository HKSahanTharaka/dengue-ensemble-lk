# AI-GENERATED (Claude via Antigravity)
"""Scoring helpers. All metrics are on the ORIGINAL case-count scale (not log)."""
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def to_cases(pred_log):
    """Undo log1p: expm1, and never predict fewer than 0 cases."""
    return np.clip(np.expm1(pred_log), 0, None)


def fit_and_score(name, family, model, X_train, y_train_log, X_test, y_train_raw, y_test_raw):
    """Fit on log1p cases (timed), predict, convert back with expm1, score on the original scale."""
    start = time.perf_counter()
    model.fit(X_train, y_train_log)
    seconds = time.perf_counter() - start
    train = regression_scores(y_train_raw, to_cases(model.predict(X_train)))
    test = regression_scores(y_test_raw, to_cases(model.predict(X_test)))
    return {"model": name, "family": family, "train_RMSE": train["RMSE"], "test_RMSE": test["RMSE"],
            "test_MAE": test["MAE"], "test_R2": test["R2"], "train_seconds": seconds}


def regression_scores(y_true, y_pred):
    """MAE, MSE, RMSE, R2 and median absolute error, in 'cases per district per week'."""
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred), "MedAE": median_absolute_error(y_true, y_pred)}
