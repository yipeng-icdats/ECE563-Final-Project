"""Mapping discovery and lightweight hyperparameter tuning.

This script is allowed to run longer than the final model scripts.  It writes:
- mapping.json
- best_params.json
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from config import BEST_PARAMS_PATH, DEFAULT_BEST_PARAMS, MAPPING_PATH, RANDOM_STATE
from utils import (
    FEATURE_COLUMNS,
    get_xy,
    load_raw_data,
    regression_metrics,
    save_json,
    validation_split,
    wide_to_long,
    prepare_model_frame,
    make_linear_pipeline,
)


def determine_mapping() -> dict[str, int]:
    load_df, temp_df = load_raw_data()
    load_long = wide_to_long(load_df, "zone_id", "load")
    temp_long = wide_to_long(temp_df, "station_id", "temperature")
    load_long = load_long[load_long["load"] > 0]

    mapping = {}
    scores = {}
    for zone in sorted(load_long["zone_id"].unique()):
        zone_load = load_long[load_long["zone_id"] == zone]
        best_station = None
        best_score = -np.inf
        station_scores = {}
        for station in sorted(temp_long["station_id"].unique()):
            station_temp = temp_long[temp_long["station_id"] == station]
            merged = zone_load.merge(
                station_temp,
                on=["year", "month", "day", "hour"],
                how="inner",
                validate="many_to_one",
            )
            corr = merged["load"].corr(merged["temperature"])
            score = abs(float(corr)) if np.isfinite(corr) else 0.0
            station_scores[str(int(station))] = score
            if score > best_score:
                best_score = score
                best_station = int(station)
        mapping[str(int(zone))] = int(best_station)
        scores[str(int(zone))] = station_scores
    save_json(mapping, MAPPING_PATH)
    return mapping


def tune_models(mapping: dict[str, int]) -> dict:
    data = prepare_model_frame(mapping)
    train, valid, _ = validation_split(data)

    ridge_grid = [0.1, 1.0, 10.0, 50.0, 100.0]
    best_ridge = None
    for alpha in ridge_grid:
        model = make_linear_pipeline(Ridge(alpha=alpha))
        model.fit(*get_xy(train))
        pred = model.predict(valid[FEATURE_COLUMNS])
        metrics = regression_metrics(valid["load"], pred)
        if best_ridge is None or metrics["rmse"] < best_ridge["metrics"]["rmse"]:
            best_ridge = {"alpha": alpha, "metrics": metrics}
        print(f"Ridge alpha={alpha}: valid RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")

    hgb_grid = [
        DEFAULT_BEST_PARAMS,
        {**DEFAULT_BEST_PARAMS, "learning_rate": 0.05, "max_iter": 300},
        {**DEFAULT_BEST_PARAMS, "learning_rate": 0.10, "max_iter": 180, "max_leaf_nodes": 45},
        {**DEFAULT_BEST_PARAMS, "l2_regularization": 0.10, "min_samples_leaf": 50},
    ]
    best_hgb = None
    for params in hgb_grid:
        model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        model.fit(train[FEATURE_COLUMNS], train["load"])
        pred = model.predict(valid[FEATURE_COLUMNS])
        metrics = regression_metrics(valid["load"], pred)
        if best_hgb is None or metrics["rmse"] < best_hgb["metrics"]["rmse"]:
            best_hgb = {"params": params, "metrics": metrics}
        print(f"HGB {params}: valid RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")

    output = {
        "split_strategy": "train <= 2006, validation = 2007, test = 2008 known nonzero rows",
        "random_state": RANDOM_STATE,
        "other_model": {"alpha": best_ridge["alpha"], "validation_metrics": best_ridge["metrics"]},
        "best_model": best_hgb["params"],
        "best_model_validation_metrics": best_hgb["metrics"],
    }
    save_json(output, BEST_PARAMS_PATH)
    return output


def main() -> None:
    start = time.perf_counter()
    mapping = determine_mapping()
    print("Zone to station mapping:")
    for zone, station in mapping.items():
        print(f"  Zone {zone}: Station {station}")
    params = tune_models(mapping)
    print(f"Wrote {MAPPING_PATH.name} and {BEST_PARAMS_PATH.name}")
    print(f"Selected parameters: {params}")
    print(f"Total tuning runtime: {time.perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    main()
