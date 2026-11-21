# AI-GENERATED (Claude via Antigravity)
"""Build model-ready features for next-week dengue forecasting.

Forecast setting: on row "week t" we know everything up to the END of week t (this week's cases and weather)
and predict week t+1. So a feature may use week t and earlier, but NEVER week t+1 or later.
Everything is built inside groupby(district) so one district never leaks into another.
"""
import numpy as np
import pandas as pd

DISTRICT, DATE, CASES = "district", "start.date", "cases"
CLIMATE = ["rainfall_mm", "temp_max_c", "temp_min_c", "humidity_pct"]
CASE_LAGS = [0, 1, 2, 3, 4, 8]       # memory of cases (0 = this week, 1 = last week, ...)
ROLL_WINDOWS = [4, 8]                # rolling mean / max windows (weeks)
CLIMATE_LAGS = range(0, 9)           # weather this week (0) and 1..8 weeks back
STATIC = ["population", "population_density"]
# Dropped: is_epidemic_year uses whole-year info; cases_zscore uses the district mean/std of ALL years (verified in notebook);
# incidence_rate is a rescaled copy of cases. Also dropped as redundant: cases_lag1, rainfall_lag2/4 (rebuilt below).
DROP_HIGH_RISK = ["is_epidemic_year", "incidence_rate", "cases_zscore"]
REDUNDANT = ["cases_lag1", "rainfall_lag2", "rainfall_lag4"]


def load_raw(path):
    """Read the raw CSV (never modified) and sort by district, then time."""
    df = pd.read_csv(path, parse_dates=[DATE])
    return df.sort_values([DISTRICT, DATE]).reset_index(drop=True)


def build_features(raw):
    """Return (table, groups). `groups` maps a type name to its feature columns."""
    df = raw.drop(columns=DROP_HIGH_RISK + REDUNDANT).sort_values([DISTRICT, DATE]).reset_index(drop=True)
    g = df.groupby(DISTRICT)
    parts, groups = {}, {}

    # Target: NEXT week's cases of the same district (shift -1 inside each district)
    parts["target_next_cases"] = g[CASES].shift(-1)
    parts["target_log"] = np.log1p(parts["target_next_cases"])   # train on log1p, convert back with expm1

    # 1. Lagged cases = the model's "memory" of recent weeks
    for k in CASE_LAGS:
        parts[f"cases_lag_{k}"] = g[CASES].shift(k)
    groups["case lags"] = [f"cases_lag_{k}" for k in CASE_LAGS]

    # 2. Rolling mean/max of the last w weeks INCLUDING this week (never future weeks)
    for w in ROLL_WINDOWS:
        parts[f"cases_roll_mean_{w}"] = g[CASES].transform(lambda s: s.rolling(w).mean())
        parts[f"cases_roll_max_{w}"] = g[CASES].transform(lambda s: s.rolling(w).max())
    groups["rolling cases"] = [f"cases_roll_{f}_{w}" for w in ROLL_WINDOWS for f in ("mean", "max")]

    # 3. Climate: this week and 8 weeks back (weather is not derived from cases)
    for c in CLIMATE:
        for k in CLIMATE_LAGS:
            parts[f"{c}_lag_{k}"] = g[c].shift(k)
    groups["climate"] = [f"{c}_lag_{k}" for c in CLIMATE for k in CLIMATE_LAGS]

    # 4. Seasonality from the DATE (week numbers in the file are unreliable)
    week = df[DATE].dt.isocalendar().week.astype(float).clip(upper=52)
    parts["week_sin"] = np.sin(2 * np.pi * week / 52)
    parts["week_cos"] = np.cos(2 * np.pi * week / 52)
    groups["seasonality"] = ["week_sin", "week_cos"]

    # 5. Static columns (district stays a text category; encoded later inside the Pipeline)
    groups["district and size"] = [DISTRICT] + STATIC

    out = pd.concat([df[[DATE, CASES, DISTRICT] + STATIC], pd.DataFrame(parts)], axis=1)
    # Drop rows with missing lags (first 7-8 weeks per district) and the last week (no next week yet)
    features = [c for cols in groups.values() for c in cols]
    out = out.dropna(subset=["target_log"] + features).reset_index(drop=True)
    # Column order: reference columns (not model inputs), then the features
    return out[[DATE, CASES, "target_next_cases", "target_log"] + features], groups

