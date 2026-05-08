"""Shared loading, preprocessing, evaluation, and prediction helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    BEST_PARAMS_PATH,
    HOUR_COLUMNS,
    LOAD_CANDIDATES,
    MAPPING_PATH,
    PREDICTION_PATH,
    RANDOM_STATE,
    TARGET_DAYS,
    TARGET_MONTH,
    TARGET_YEAR,
    TEMP_CANDIDATES,
)


def find_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find a required CSV. Checked:\n{checked}")


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    load_path = find_existing_path(LOAD_CANDIDATES)
    temp_path = find_existing_path(TEMP_CANDIDATES)
    return pd.read_csv(load_path), pd.read_csv(temp_path)


def wide_to_long(df: pd.DataFrame, id_col: str, value_name: str) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=[id_col, "year", "month", "day"],
        value_vars=HOUR_COLUMNS,
        var_name="hour",
        value_name=value_name,
    )
    long_df["hour"] = long_df["hour"].str[1:].astype(int)
    return long_df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(dict(year=out["year"], month=out["month"], day=out["day"]))
    out["dayofweek"] = dt.dt.dayofweek
    out["dayofyear"] = dt.dt.dayofyear
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["sin_dayofyear"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    out["cos_dayofyear"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)
    return out


def target_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["year"] == TARGET_YEAR)
        & (df["month"] == TARGET_MONTH)
        & (df["day"].isin(TARGET_DAYS))
    )


def tuning_train_mask(df: pd.DataFrame) -> pd.Series:
    return (df["year"] <= 2006) & (~target_mask(df)) & (df["load"] > 0)


def validation_mask(df: pd.DataFrame) -> pd.Series:
    return (df["year"] == 2007) & (~target_mask(df)) & (df["load"] > 0)


def final_train_mask(df: pd.DataFrame) -> pd.Series:
    return (df["year"] <= 2007) & (~target_mask(df)) & (df["load"] > 0)


def heldout_test_mask(df: pd.DataFrame) -> pd.Series:
    return (df["year"] == 2008) & (~target_mask(df)) & (df["load"] > 0)


def final_prediction_train_mask(df: pd.DataFrame) -> pd.Series:
    return (df["year"] <= 2008) & (~target_mask(df)) & (df["load"] > 0)


def history_fit_mask(df: pd.DataFrame, max_year: int, value_col: str) -> pd.Series:
    return (df["year"] <= max_year) & (~target_mask(df)) & (df[value_col] > 0)


def _local_group_median(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    valid_mask: pd.Series,
) -> pd.Series:
    medians = (
        df.loc[valid_mask, group_cols + [value_col]]
        .groupby(group_cols)[value_col]
        .median()
        .rename("_local_median")
    )
    return df[group_cols].merge(medians, on=group_cols, how="left")["_local_median"]


def clean_load_values(load_long: pd.DataFrame, fit_year_max: int = 2007) -> pd.DataFrame:
    """Impute invalid known load zeros and mask robust local outliers."""
    out = load_long.copy()
    out["load"] = out["load"].astype(float)
    is_target = target_mask(out)

    known_positive = history_fit_mask(out, fit_year_max, "load")
    zone_month_hour = _local_group_median(
        out,
        "load",
        ["zone_id", "month", "hour"],
        known_positive,
    )
    zone_hour = _local_group_median(out, "load", ["zone_id", "hour"], known_positive)
    zone = _local_group_median(out, "load", ["zone_id"], known_positive)
    local_fill = zone_month_hour.fillna(zone_hour).fillna(zone)

    zero_defect = (out["load"] <= 0) & (~is_target)
    out["load_was_imputed"] = zero_defect.astype(int)
    out.loc[zero_defect, "load"] = local_fill.loc[zero_defect]

    known = out["load"].notna() & history_fit_mask(out, fit_year_max, "load")
    stats = (
        out.loc[known, ["zone_id", "month", "hour", "load"]]
        .groupby(["zone_id", "month", "hour"])["load"]
        .quantile([0.25, 0.50, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.50: "median", 0.75: "q3"})
        .reset_index()
    )
    out = out.merge(stats, on=["zone_id", "month", "hour"], how="left")
    iqr = out["q3"] - out["q1"]
    low = out["q1"] - 3.0 * iqr
    high = out["q3"] + 3.0 * iqr
    outlier = known & iqr.gt(0) & ((out["load"] < low) | (out["load"] > high))
    out["load_was_outlier"] = outlier.astype(int)
    out.loc[outlier, "load"] = out.loc[outlier, "median"]
    return out.drop(columns=["q1", "median", "q3"])


def clean_temperature_values(temp_long: pd.DataFrame, fit_year_max: int = 2007) -> pd.DataFrame:
    """Impute impossible zero temperature readings without clipping real extremes."""
    out = temp_long.copy()
    out["temperature"] = out["temperature"].astype(float)
    valid = history_fit_mask(out, fit_year_max, "temperature")
    station_month_hour = _local_group_median(
        out,
        "temperature",
        ["station_id", "month", "hour"],
        valid,
    )
    station_hour = _local_group_median(out, "temperature", ["station_id", "hour"], valid)
    station = _local_group_median(out, "temperature", ["station_id"], valid)
    zero_defect = out["temperature"] <= 0
    out["temperature_was_imputed"] = zero_defect.astype(int)
    out.loc[zero_defect, "temperature"] = (
        station_month_hour.fillna(station_hour).fillna(station).loc[zero_defect]
    )
    return out


def normalize_mapping(raw_mapping: dict) -> dict[str, list[int]]:
    """Return a zone -> station-list mapping, accepting the old integer schema."""
    mapping = {}
    for zone, stations in raw_mapping.items():
        if isinstance(stations, list):
            mapping[str(zone)] = sorted({int(station) for station in stations})
        elif stations is None:
            mapping[str(zone)] = []
        else:
            mapping[str(zone)] = [int(stations)]
    return mapping


def load_mapping(path: Path = MAPPING_PATH) -> dict[str, list[int]]:
    if not path.exists():
        return {str(zone): [1] for zone in range(1, 21)}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return normalize_mapping(raw)


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def add_lag_features(data: pd.DataFrame, history_year_max: int) -> pd.DataFrame:
    out = data.sort_values(["zone_id", "year", "month", "day", "hour"]).copy()
    history_load = out["load"].where(history_fit_mask(out, history_year_max, "load"))

    out["_history_load"] = history_load
    grouped = out.groupby(["zone_id", "hour"], sort=False)["_history_load"]
    out["prev_day_same_hour_load"] = grouped.shift(1)
    out["rolling_7d_same_hour_load"] = grouped.transform(
        lambda series: series.shift(1).rolling(7, min_periods=2).mean()
    )

    fallback = out["zone_hour_month_mean"].fillna(out["zone_hour_mean"]).fillna(out["zone_mean"])
    out["prev_day_same_hour_load"] = out["prev_day_same_hour_load"].fillna(fallback)
    out["rolling_7d_same_hour_load"] = out["rolling_7d_same_hour_load"].fillna(fallback)
    return out.drop(columns=["_history_load"])


def temp_station_column(station_id: int) -> str:
    return f"temp_station_{int(station_id)}"


def pivot_station_temperatures(temp_long: pd.DataFrame) -> pd.DataFrame:
    temp_wide = temp_long.pivot_table(
        index=["year", "month", "day", "hour"],
        columns="station_id",
        values="temperature",
        aggfunc="first",
    )
    temp_wide = temp_wide.rename(columns={col: temp_station_column(int(col)) for col in temp_wide.columns})
    temp_wide = temp_wide.reset_index()
    station_cols = [col for col in temp_wide.columns if col.startswith("temp_station_")]
    temp_wide["regional_temp_mean"] = temp_wide[station_cols].mean(axis=1)
    temp_wide["regional_temp_min"] = temp_wide[station_cols].min(axis=1)
    temp_wide["regional_temp_max"] = temp_wide[station_cols].max(axis=1)
    temp_wide["regional_temp_std"] = temp_wide[station_cols].std(axis=1).fillna(0.0)
    return temp_wide


def add_mapped_temperature_features(data: pd.DataFrame, mapping: dict[str, list[int]]) -> pd.DataFrame:
    out = data.copy()
    station_cols = [col for col in out.columns if col.startswith("temp_station_")]
    available_station_ids = {
        int(col.replace("temp_station_", "")): col for col in station_cols
    }

    out["mapped_temp_mean"] = np.nan
    out["mapped_temp_min"] = np.nan
    out["mapped_temp_max"] = np.nan
    out["mapped_temp_std"] = np.nan
    out["mapped_station_count"] = 0

    for zone_id in sorted(out["zone_id"].unique()):
        selected_cols = [
            available_station_ids[station_id]
            for station_id in mapping.get(str(int(zone_id)), [])
            if station_id in available_station_ids
        ]
        zone_mask = out["zone_id"] == zone_id
        out.loc[zone_mask, "mapped_station_count"] = len(selected_cols)
        if selected_cols:
            temps = out.loc[zone_mask, selected_cols]
            out.loc[zone_mask, "mapped_temp_mean"] = temps.mean(axis=1)
            out.loc[zone_mask, "mapped_temp_min"] = temps.min(axis=1)
            out.loc[zone_mask, "mapped_temp_max"] = temps.max(axis=1)
            out.loc[zone_mask, "mapped_temp_std"] = temps.std(axis=1).fillna(0.0)

    fallback = {
        "mapped_temp_mean": "regional_temp_mean",
        "mapped_temp_min": "regional_temp_min",
        "mapped_temp_max": "regional_temp_max",
        "mapped_temp_std": "regional_temp_std",
    }
    for mapped_col, regional_col in fallback.items():
        out[mapped_col] = out[mapped_col].fillna(out[regional_col])
    out["mapped_temp_sq"] = out["mapped_temp_mean"] ** 2
    out["mapped_cooling_degree"] = np.maximum(out["mapped_temp_mean"] - 65.0, 0.0)
    out["mapped_heating_degree"] = np.maximum(65.0 - out["mapped_temp_mean"], 0.0)
    return out


def prepare_model_frame(
    mapping: dict[str, list[int]] | None = None,
    stats_fit_year_max: int = 2007,
) -> pd.DataFrame:
    load_df, temp_df = load_raw_data()
    mapping = mapping or load_mapping()
    mapping = normalize_mapping(mapping)

    load_long = clean_load_values(wide_to_long(load_df, "zone_id", "load"), stats_fit_year_max)
    temp_long = clean_temperature_values(
        wide_to_long(temp_df, "station_id", "temperature"),
        stats_fit_year_max,
    )

    temp_wide = pivot_station_temperatures(temp_long)
    data = load_long.merge(
        temp_wide,
        on=["year", "month", "day", "hour"],
        how="left",
        validate="many_to_one",
    )
    data = add_mapped_temperature_features(data, mapping)
    data = add_calendar_features(data)

    known = data[history_fit_mask(data, stats_fit_year_max, "load")]
    zone_hour_month_mean = (
        known.groupby(["zone_id", "month", "hour"])["load"].mean().rename("zone_hour_month_mean")
    )
    zone_hour_mean = known.groupby(["zone_id", "hour"])["load"].mean().rename("zone_hour_mean")
    zone_mean = known.groupby("zone_id")["load"].mean().rename("zone_mean")
    last_year = known[["zone_id", "year", "month", "day", "hour", "load"]].copy()
    last_year["year"] += 1
    last_year = last_year.rename(columns={"load": "last_year_load"})

    data = data.merge(zone_hour_month_mean, on=["zone_id", "month", "hour"], how="left")
    data = data.merge(zone_hour_mean, on=["zone_id", "hour"], how="left")
    data = data.merge(zone_mean, on="zone_id", how="left")
    data = data.merge(last_year, on=["zone_id", "year", "month", "day", "hour"], how="left")
    data["last_year_load"] = data["last_year_load"].fillna(data["zone_hour_month_mean"])
    data["zone_hour_month_mean"] = data["zone_hour_month_mean"].fillna(data["zone_hour_mean"])
    data["zone_hour_mean"] = data["zone_hour_mean"].fillna(data["zone_mean"])
    data = add_lag_features(data, stats_fit_year_max)
    return data


FEATURE_COLUMNS = [
    "zone_id",
    "year",
    "month",
    "day",
    "hour",
    "dayofweek",
    "dayofyear",
    "is_weekend",
    "mapped_temp_mean",
    "mapped_temp_min",
    "mapped_temp_max",
    "mapped_temp_std",
    "mapped_temp_sq",
    "mapped_cooling_degree",
    "mapped_heating_degree",
    "mapped_station_count",
    "regional_temp_mean",
    "regional_temp_min",
    "regional_temp_max",
    "regional_temp_std",
    "sin_hour",
    "cos_hour",
    "sin_dayofyear",
    "cos_dayofyear",
    "zone_hour_month_mean",
    "zone_hour_mean",
    "zone_mean",
    "last_year_load",
    "prev_day_same_hour_load",
    "rolling_7d_same_hour_load",
]

CATEGORICAL_COLUMNS = ["zone_id", "month", "hour", "dayofweek", "is_weekend"]
NUMERIC_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]


def train_test_time_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = data[heldout_test_mask(data)].copy()
    train = data[final_train_mask(data)].sample(frac=1.0, random_state=RANDOM_STATE).copy()
    return train, test


def final_prediction_train_split(data: pd.DataFrame) -> pd.DataFrame:
    return data[final_prediction_train_mask(data)].sample(frac=1.0, random_state=RANDOM_STATE).copy()


def validation_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = data[tuning_train_mask(data)].sample(frac=1.0, random_state=RANDOM_STATE).copy()
    valid = data[validation_mask(data)].copy()
    test = data[heldout_test_mask(data)].copy()
    return train, valid, test


def build_linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )


def make_linear_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_linear_preprocessor()), ("model", model)])


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS], df["load"]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    positive = y_true > 0
    if positive.any():
        mape = np.mean(np.abs((y_true[positive] - y_pred[positive]) / y_true[positive])) * 100.0
    else:
        mape = np.nan
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape": float(mape),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"{label}: RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")


def fit_and_time(model, train_df: pd.DataFrame):
    x_train, y_train = get_xy(train_df)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    return model, time.perf_counter() - start


def evaluate_and_time(model, df: pd.DataFrame) -> tuple[dict[str, float], np.ndarray, float]:
    x, y = get_xy(df)
    start = time.perf_counter()
    pred = model.predict(x)
    elapsed = time.perf_counter() - start
    return regression_metrics(y, pred), pred, elapsed


def top_errors(df: pd.DataFrame, pred: np.ndarray, n: int = 10) -> pd.DataFrame:
    out = df[["zone_id", "year", "month", "day", "hour", "load"]].copy()
    out["predicted load"] = pred
    out["relative % error"] = 100.0 * (out["load"] - out["predicted load"]) / out["load"]
    out = out.rename(columns={"zone_id": "zone", "load": "true load"})
    return out.reindex(out["relative % error"].abs().sort_values(ascending=False).index).head(n)


def load_best_params() -> dict:
    if not BEST_PARAMS_PATH.exists():
        from config import DEFAULT_BEST_PARAMS

        return DEFAULT_BEST_PARAMS.copy()
    with BEST_PARAMS_PATH.open("r", encoding="utf-8") as f:
        params = json.load(f)
    return params.get("best_model", params)


def write_prediction_csv(model, data: pd.DataFrame, path: Path = PREDICTION_PATH) -> float:
    target = data[target_mask(data)].copy().sort_values(["zone_id", "year", "month", "day", "hour"])
    start = time.perf_counter()
    target["predicted_load"] = np.maximum(model.predict(target[FEATURE_COLUMNS]), 0.0)
    elapsed = time.perf_counter() - start

    wide = target.pivot_table(
        index=["zone_id", "year", "month", "day"],
        columns="hour",
        values="predicted_load",
        aggfunc="first",
    )
    wide = wide.reindex(columns=list(range(1, 25))).reset_index()
    wide.columns = ["zone_id", "year", "month", "day"] + HOUR_COLUMNS
    for col in HOUR_COLUMNS:
        wide[col] = wide[col].round().astype(int)
    wide.to_csv(path, index=False)
    return elapsed
