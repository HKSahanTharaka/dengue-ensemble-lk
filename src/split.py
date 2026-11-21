# AI-GENERATED (Claude via Antigravity)
"""Time-based train/test split, preprocessing and time-series cross-validation. Nothing is ever shuffled."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder

DATE, DISTRICT = "start.date", "district"
TARGET_RAW, TARGET_LOG = "target_next_cases", "target_log"
REFERENCE = [DATE, "cases", TARGET_RAW, TARGET_LOG]      # kept for splitting/scoring, never model inputs
LAST_TRAIN_YEAR = 2021


def load_features(path):
    """Load features.csv and sort by time (then district) so row order = time order."""
    df = pd.read_csv(path, parse_dates=[DATE])
    return df.sort_values([DATE, DISTRICT]).reset_index(drop=True)


def time_split(df, last_train_year=LAST_TRAIN_YEAR):
    """Train = all weeks up to and including last_train_year. Test = everything after."""
    year = df[DATE].dt.year
    return df[year <= last_train_year].copy(), df[year > last_train_year].copy()


def feature_columns(df):
    """Model inputs = every column that is not a reference/target column."""
    return [c for c in df.columns if c not in REFERENCE]


def make_preprocessor(features):
    """One-hot encode district; pass numbers through. Fitted on training data only (inside a Pipeline)."""
    numbers = [c for c in features if c != DISTRICT]
    return ColumnTransformer([
        ("district", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [DISTRICT]),
        ("numbers", "passthrough", numbers),
    ])


def make_cv(train, n_splits=5):
    """TimeSeriesSplit: validation always comes AFTER training.
    gap = one week of rows (25 districts): the label of the last training week is the first validation
    week's cases, so we skip that week to keep the two parts cleanly apart."""
    return TimeSeriesSplit(n_splits=n_splits, gap=train[DISTRICT].nunique())


def describe_folds(train, cv):
    """Table of each fold's train and validation date range (proves validation is in the future)."""
    rows = []
    for i, (tr, va) in enumerate(cv.split(train), start=1):
        d = train[DATE]
        rows.append({"fold": i,
                     "train from": d.iloc[tr].min().date(), "train to": d.iloc[tr].max().date(), "train rows": len(tr),
                     "valid from": d.iloc[va].min().date(), "valid to": d.iloc[va].max().date(), "valid rows": len(va)})
        assert d.iloc[tr].max() < d.iloc[va].min(), "validation must be after training"
    return pd.DataFrame(rows)
