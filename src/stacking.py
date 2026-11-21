# AI-GENERATED (Claude via Antigravity)
"""Base models and a manual, time-aware stacking model (sklearn's StackingRegressor cannot use TimeSeriesSplit)."""
import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from split import make_preprocessor


class CatBoostDistrict(RegressorMixin, BaseEstimator):
    """Thin wrapper so scikit-learn can clone CatBoost; district is passed as a native categorical column."""

    def __init__(self, iterations=500, depth=6, learning_rate=0.1, random_state=42):
        self.iterations, self.depth, self.learning_rate, self.random_state = iterations, depth, learning_rate, random_state

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
                                        cat_features=["district"], random_state=self.random_state, verbose=0,
                                        thread_count=-1, allow_writing_files=False)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def make_base_models(features):
    """The three diverse base models (same default settings as notebook 04). All trained on log1p cases."""
    return {
        "Random Forest": Pipeline([("prep", make_preprocessor(features)),
                                   ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]),
        "XGBoost": Pipeline([("prep", make_preprocessor(features)),
                             ("model", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1))]),
        "CatBoost": CatBoostDistrict(),
    }


class TimeSeriesStacker(RegressorMixin, BaseEstimator):
    """Manual stacking:
    1. For each TimeSeriesSplit fold: fit every base model on the fold's PAST part, predict its FUTURE part.
    2. Collect those out-of-fold predictions (the earliest block never gets any, so it is left out).
    3. Fit a RidgeCV meta-model on them (it learns how much to trust each base model).
    4. Refit every base model on ALL training rows; at prediction time the meta-model combines them.
    Rows must be sorted by time (oldest first)."""

    def __init__(self, base_models, cv, alphas=tuple(np.logspace(-3, 3, 13))):
        self.base_models, self.cv, self.alphas = base_models, cv, alphas

    def fit(self, X, y):
        X, y = X.reset_index(drop=True), np.asarray(y)
        oof = np.full((len(X), len(self.base_models)), np.nan)         # out-of-fold predictions
        for train_idx, valid_idx in self.cv.split(X):
            for j, model in enumerate(self.base_models.values()):
                oof[valid_idx, j] = clone(model).fit(X.iloc[train_idx], y[train_idx]).predict(X.iloc[valid_idx])
        has_pred = ~np.isnan(oof).any(axis=1)                          # earliest block has no predictions
        self.n_meta_rows_ = int(has_pred.sum())
        self.meta_ = RidgeCV(alphas=self.alphas, cv=TimeSeriesSplit(n_splits=5)).fit(oof[has_pred], y[has_pred])
        self.fitted_ = {name: clone(m).fit(X, y) for name, m in self.base_models.items()}   # final refit, all training data
        return self

    def predict(self, X):
        base = np.column_stack([m.predict(X) for m in self.fitted_.values()])
        return self.meta_.predict(base)


