"""Reusable preprocessing for the load-type project.

Kept in a module (not a notebook cell) so the saved joblib pipeline can be
loaded anywhere with:  import sys; sys.path.insert(0, "src"); joblib.load(...)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

POWER_FACTOR_COLUMNS = ["Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"]
TELEMETRY_FEATURES = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
]
WINTER_MONTHS = [1, 2, 11, 12]


def load_clean_timeline(path) -> pd.DataFrame:
    """Read the raw CSV and fix the timeline.

    * Readings stamped 00:00 close the day they are dated with (the row after
      '01-01-2018 23:45' is '01-01-2018 00:00'), so they are moved +1 day.
    * The duplicated timestamp is resolved by keeping the copy with fewer
      missing / invalid values.
    """
    df = pd.read_csv(path)
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], format="%d-%m-%Y %H:%M")
    midnight = (df["Date_Time"].dt.hour == 0) & (df["Date_Time"].dt.minute == 0)
    df.loc[midnight, "Date_Time"] += pd.Timedelta(days=1)

    quality_issues = df[TELEMETRY_FEATURES].isna().sum(axis=1)
    for col in POWER_FACTOR_COLUMNS:
        quality_issues += ((df[col] < 0) | (df[col] > 100)).astype(int)
    df = (df.assign(_issues=quality_issues)
            .sort_values(["Date_Time", "_issues"], kind="stable")
            .drop_duplicates(subset="Date_Time", keep="first")
            .drop(columns="_issues")
            .reset_index(drop=True))
    return df


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    t = data["Date_Time"]
    data["Hour"] = t.dt.hour
    data["Day_of_Week"] = t.dt.day_of_week
    data["Month"] = t.dt.month
    data["Time_Slot"] = t.dt.hour * 4 + t.dt.minute // 15
    data["Is_Weekend"] = (data["Day_of_Week"] >= 5).astype(int)
    # 0 = winter schedule (Nov-Feb), 1 = Mar-Oct schedule (see modeling notebook, section 5)
    data["Schedule_Regime"] = np.where(t.dt.month.isin(WINTER_MONTHS), 0, 1)
    return data


def has_extra_decimals(values: pd.Series) -> pd.Series:
    """Genuine readings have at most 2 decimals; corrupted ones have many."""
    values = pd.to_numeric(values, errors="coerce")
    return values.notna() & (np.abs(values * 100 - np.round(values * 100)) > 1e-6)


class PowerFactorCleaner(BaseEstimator, TransformerMixin):
    """Original cleaner: power factors outside [0, 100] -> NaN."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X.loc[(X[col] < 0) | (X[col] > 100), col] = np.nan
        return X


class SensorCleaner(BaseEstimator, TransformerMixin):
    """Power factors outside [0, 100] AND corrupted readings (more than two
    decimals) in every telemetry column -> NaN. Stateless, so no leakage."""

    def __init__(self, columns, power_factor_columns):
        self.columns = columns
        self.power_factor_columns = power_factor_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X.loc[has_extra_decimals(X[col]), col] = np.nan
        for col in self.power_factor_columns:
            X.loc[(X[col] < 0) | (X[col] > 100), col] = np.nan
        return X


class RegimeScheduleLookup(BaseEstimator, ClassifierMixin):
    """Predicts the most frequent load type for (regime, weekday, 15-min slot),
    falling back to (weekday, slot) and then to the overall majority class."""

    def fit(self, X, y):
        keys = ["Schedule_Regime", "Day_of_Week", "Time_Slot"]
        data = X[keys].reset_index(drop=True).copy()
        data["Target"] = np.asarray(y)
        mode = lambda s: s.value_counts().idxmax()
        self.regime_lookup_ = data.groupby(keys)["Target"].agg(mode)
        self.global_lookup_ = data.groupby(["Day_of_Week", "Time_Slot"])["Target"].agg(mode)
        self.default_class_ = data["Target"].mode().iloc[0]
        self.classes_ = np.unique(data["Target"])
        return self

    def predict(self, X):
        regime = self.regime_lookup_.reindex(
            pd.MultiIndex.from_frame(X[["Schedule_Regime", "Day_of_Week", "Time_Slot"]])).to_numpy()
        fallback = self.global_lookup_.reindex(
            pd.MultiIndex.from_frame(X[["Day_of_Week", "Time_Slot"]])).to_numpy()
        return pd.Series(regime).fillna(pd.Series(fallback)).fillna(self.default_class_).to_numpy()
