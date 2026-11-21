# AI-GENERATED (Claude via Antigravity)
"""Train the winning tuned Pipeline on the training set and export pickle, metadata, predictions, and final table."""
import json
import pickle
import sys
from pathlib import Path

import catboost
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import regression_scores, to_cases  # noqa: E402
from split import DATE, DISTRICT, TARGET_LOG, TARGET_RAW, feature_columns, load_features, make_preprocessor, time_split  # noqa: E402


def main():
    df = load_features(ROOT / "data" / "processed" / "features.csv")
    train, test = time_split(df)
    features = feature_columns(df)

    with open(ROOT / "models" / "best_params.json") as f:
        rf_params = {k.replace("model__", ""): v for k, v in json.load(f)["Random Forest"].items()}

    # Full Pipeline: takes raw 51 feature columns, one-hot encodes district, predicts log1p(next-week cases)
    pipe = Pipeline([
        ("prep", make_preprocessor(features)),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, **rf_params)),
    ])
    pipe.fit(train[features], train[TARGET_LOG])

    # Evaluate on held-out test set on the original case-count scale
    test_pred = to_cases(pipe.predict(test[features]))
    metrics = regression_scores(test[TARGET_RAW], test_pred)

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = models_dir / "best_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipe, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "model_name": "Random Forest (tuned)",
        "target": TARGET_RAW,
        "trained_on": "log1p(target_next_cases) -> invert predictions with np.clip(np.expm1(y), 0, None)",
        "features": features,
        "districts": sorted(df[DISTRICT].unique().tolist()),
        "train_date_range": [str(train[DATE].min().date()), str(train[DATE].max().date())],
        "test_date_range": [str(test[DATE].min().date()), str(test[DATE].max().date())],
        "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "catboost": catboost.__version__,
        },
    }
    with open(models_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    preds_df = pd.DataFrame({
        "district": test[DISTRICT].to_numpy(),
        "date": test[DATE].dt.strftime("%Y-%m-%d").to_numpy(),
        "actual": test[TARGET_RAW].to_numpy(),
        "predicted": test_pred,
    })
    preds_df.to_csv(models_dir / "test_predictions.csv", index=False)

    # Final comparison table with MSE included
    res = pd.read_csv(ROOT / "reports" / "results_tuned.csv")
    res["test_MSE"] = res["test_RMSE"] ** 2
    cols = ["model", "family", "test_MAE", "test_RMSE", "test_MSE", "test_R2", "train_RMSE", "train_seconds"]
    res[cols].sort_values("test_RMSE").to_csv(ROOT / "reports" / "results_final.csv", index=False)

    size_mb = pkl_path.stat().st_size / (1024 * 1024)
    print(f"Saved {pkl_path.name} ({size_mb:.2f} MB), model_meta.json, test_predictions.csv, reports/results_final.csv")
    print("Test metrics:", meta["test_metrics"])


if __name__ == "__main__":
    main()
