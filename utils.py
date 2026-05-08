"""Shared loading, preprocessing, evaluation, and prediction helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    BEST_PARAMS_PATH,
    HOUR_COLUMNS,
    LOAD_CANDIDATES,
    MAPPING_PATH,
    PREDICTION_PATH,
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


def load_mapping(path: Path = MAPPING_PATH) -> dict[str, int]:
    if not path.exists():
        return {str(zone): 1 for zone in range(1, 21)}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def prepare_model_frame(mapping: dict[str, int] | None = None) -> pd.DataFrame:
    load_df, temp_df = load_raw_data()
    mapping = mapping or load_mapping()

    load_long = wide_to_long(load_df, "zone_id", "load")
    temp_long = wide_to_long(temp_df, "station_id", "temperature")

    frames = []
    for zone_id, station_id in sorted((int(k), int(v)) for k, v in mapping.items()):
        z = load_long[load_long["zone_id"] == zone_id]
        s = temp_long[temp_long["station_id"] == station_id]
        merged = z.merge(
            s,
            on=["year", "month", "day", "hour"],
            how="left",
            validate="many_to_one",
        )
        frames.append(merged)

    data = pd.concat(frames, ignore_index=True)
    data["station_id"] = data["station_id"].astype(int)
    data = add_calendar_features(data)
    data["temp_sq"] = data["temperature"] ** 2
    data["cooling_degree"] = np.maximum(data["temperature"] - 65.0, 0.0)
    data["heating_degree"] = np.maximum(65.0 - data["temperature"], 0.0)

    known = data[(data["load"] > 0) & (~target_mask(data))]
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
    return data


FEATURE_COLUMNS = [
    "zone_id",
    "station_id",
    "year",
    "month",
    "day",
    "hour",
    "dayofweek",
    "dayofyear",
    "is_weekend",
    "temperature",
    "temp_sq",
    "cooling_degree",
    "heating_degree",
    "sin_hour",
    "cos_hour",
    "sin_dayofyear",
    "cos_dayofyear",
    "zone_hour_month_mean",
    "zone_hour_mean",
    "zone_mean",
    "last_year_load",
]

CATEGORICAL_COLUMNS = ["zone_id", "station_id", "month", "hour", "dayofweek", "is_weekend"]
NUMERIC_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]


def train_test_time_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    known = data[(data["load"] > 0) & (~target_mask(data))].copy()
    test = known[known["year"] == 2008].copy()
    train = known[known["year"] < 2008].copy()
    return train, test


def validation_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known = data[(data["load"] > 0) & (~target_mask(data))].copy()
    train = known[known["year"] <= 2006].copy()
    valid = known[known["year"] == 2007].copy()
    test = known[known["year"] == 2008].copy()
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
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"{label}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")


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
