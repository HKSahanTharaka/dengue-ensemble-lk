# AI-GENERATED (Claude via Antigravity)
"""Build model-ready features for next-week dengue forecasting.

Golden rule: a feature on row "week t" may only use information from weeks BEFORE t.
Everything is built inside groupby(district) so one district never leaks into another.
"""
import numpy as np
import pandas as pd

DISTRICT, DATE, CASES = "district", "start.date", "cases"
CLIMATE = ["rainfall_mm", "temp_max_c", "temp_min_c", "humidity_pct"]
CASE_LAGS = [1, 2, 3, 4, 8]          # memory of past cases (weeks back)
ROLL_WINDOWS = [4, 8]                # rolling mean / max windows (weeks)
CLIMATE_LAGS = range(1, 9)           # weather 1..8 weeks back
STATIC = ["population", "population_density"]
# Columns the audit marked as leakage risks (whole-year flag, or rescaled copies of cases)
DROP_HIGH_RISK = ["is_epidemic_year", "incidence_rate", "cases_zscore"]


def load_raw(path):
    """Read the raw CSV (never modified) and sort by district, then time."""
    df = pd.read_csv(path, parse_dates=[DATE])
    return df.sort_values([DISTRICT, DATE]).reset_index(drop=True)


def build_features(raw):
    """Return (table, groups). `groups` maps a type name to its feature columns."""
    df = raw.drop(columns=DROP_HIGH_RISK).sort_values([DISTRICT, DATE]).reset_index(drop=True)
    g = df.groupby(DISTRICT)
    parts, groups = {}, {}

    # Target: NEXT week's cases of the same district (shift -1 inside each district)
    parts["target_next_cases"] = g[CASES].shift(-1)
    parts["target_log"] = np.log1p(parts["target_next_cases"])   # train on log1p, convert back with expm1

    # 1. Lagged cases = the model's "memory" of recent weeks
    for k in CASE_LAGS:
        parts[f"cases_lag_{k}"] = g[CASES].shift(k)
    groups["case lags"] = [f"cases_lag_{k}" for k in CASE_LAGS]

    # 2. Rolling mean/max of PAST cases: shift(1) first so the current week is never in the window
    for w in ROLL_WINDOWS:
        past_g = g[CASES].shift(1).groupby(df[DISTRICT])
        parts[f"cases_roll_mean_{w}"] = past_g.transform(lambda s: s.rolling(w).mean())
        parts[f"cases_roll_max_{w}"] = past_g.transform(lambda s: s.rolling(w).max())
    groups["rolling past cases"] = [f"cases_roll_{f}_{w}" for w in ROLL_WINDOWS for f in ("mean", "max")]

    # 3. Lagged climate (weather is not derived from cases, lags 1..8 only)
    for c in CLIMATE:
        for k in CLIMATE_LAGS:
            parts[f"{c}_lag_{k}"] = g[c].shift(k)
    groups["lagged climate"] = [f"{c}_lag_{k}" for c in CLIMATE for k in CLIMATE_LAGS]

    # 4. Seasonality from the DATE (week numbers in the file are unreliable)
    week = df[DATE].dt.isocalendar().week.astype(float).clip(upper=52)
    parts["week_sin"] = np.sin(2 * np.pi * week / 52)
    parts["week_cos"] = np.cos(2 * np.pi * week / 52)
    groups["seasonality"] = ["week_sin", "week_cos"]

    # 5. Static columns (district stays a text category; encoded later inside the Pipeline)
    groups["district and size"] = [DISTRICT] + STATIC

    out = pd.concat([df[[DATE, CASES, DISTRICT] + STATIC], pd.DataFrame(parts)], axis=1)
    # Drop rows with missing lags (first 8 weeks per district) and the last week (no next week yet)
    features = [c for cols in groups.values() for c in cols]
    out = out.dropna(subset=["target_log"] + features).reset_index(drop=True)
    # Column order: reference columns (not model inputs), then the features
    return out[[DATE, CASES, "target_next_cases", "target_log"] + features], groups
