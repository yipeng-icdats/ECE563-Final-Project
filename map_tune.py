"""Mapping discovery and lightweight hyperparameter tuning.

This script is allowed to run longer than the final model scripts.  It writes:
- mapping.json
- mapping_diagnostics.json
- best_params.json
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

from config import (
    BEST_PARAMS_PATH,
    DEFAULT_BEST_PARAMS,
    MAPPING_DIAGNOSTICS_PATH,
    MAPPING_PATH,
    ML_IMPORTANCE_RELATIVE_THRESHOLD,
    PEARSON_MIN_THRESHOLD,
    PEARSON_RELATIVE_THRESHOLD,
    RANDOM_STATE,
)
from utils import (
    FEATURE_COLUMNS,
    add_calendar_features,
    clean_load_values,
    clean_temperature_values,
    get_xy,
    load_raw_data,
    make_linear_pipeline,
    pivot_station_temperatures,
    prepare_model_frame,
    regression_metrics,
    save_json,
    validation_split,
    wide_to_long,
)


def prepare_mapping_frame() -> tuple[pd.DataFrame, list[int], list[str]]:
    load_df, temp_df = load_raw_data()
    load_long = clean_load_values(wide_to_long(load_df, "zone_id", "load"), fit_year_max=2006)
    temp_long = clean_temperature_values(
        wide_to_long(temp_df, "station_id", "temperature"),
        fit_year_max=2006,
    )
    temp_wide = pivot_station_temperatures(temp_long)
    station_ids = sorted(temp_long["station_id"].astype(int).unique())
    station_cols = [f"temp_station_{station_id}" for station_id in station_ids]

    data = load_long.merge(
        temp_wide,
        on=["year", "month", "day", "hour"],
        how="left",
        validate="many_to_one",
    )
    data = add_calendar_features(data)
    data = data[(data["load"] > 0) & (data["year"] <= 2006)].copy()
    return data, station_ids, station_cols


def select_by_pearson(zone_df: pd.DataFrame, station_ids: list[int], station_cols: list[str]) -> tuple[list[int], dict[str, float]]:
    scores = {}
    for station_id, station_col in zip(station_ids, station_cols):
        corr = zone_df["load"].corr(zone_df[station_col])
        scores[str(station_id)] = abs(float(corr)) if np.isfinite(corr) else 0.0

    best_score = max(scores.values()) if scores else 0.0
    threshold = max(PEARSON_MIN_THRESHOLD, PEARSON_RELATIVE_THRESHOLD * best_score)
    selected = [int(station_id) for station_id, score in scores.items() if score >= threshold]
    return selected, scores


def select_by_ml(zone_df: pd.DataFrame, station_ids: list[int], station_cols: list[str]) -> tuple[list[int], dict[str, float]]:
    train = zone_df[zone_df["year"] <= 2005].copy()
    valid = zone_df[zone_df["year"] == 2006].copy()
    if train.empty or valid.empty:
        return [], {str(station_id): 0.0 for station_id in station_ids}

    calendar_cols = [
        "year",
        "month",
        "day",
        "hour",
        "dayofweek",
        "dayofyear",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_dayofyear",
        "cos_dayofyear",
    ]
    feature_cols = calendar_cols + station_cols
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=25,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(train[feature_cols], train["load"])
    result = permutation_importance(
        model,
        valid[feature_cols],
        valid["load"],
        scoring="neg_root_mean_squared_error",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importances = {
        str(station_id): max(0.0, float(result.importances_mean[feature_cols.index(station_col)]))
        for station_id, station_col in zip(station_ids, station_cols)
    }
    best_importance = max(importances.values()) if importances else 0.0
    threshold = ML_IMPORTANCE_RELATIVE_THRESHOLD * best_importance
    selected = [
        int(station_id)
        for station_id, importance in importances.items()
        if importance > 0 and importance >= threshold
    ]
    return selected, importances


def determine_mapping() -> tuple[dict[str, list[int]], dict]:
    data, station_ids, station_cols = prepare_mapping_frame()
    mapping = {}
    diagnostics = {
        "method": "union of Pearson-correlation and ML permutation-importance selections",
        "preprocessing": "clean_load_values and clean_temperature_values applied before mapping",
        "pearson_years": "2004-2006",
        "ml_train_years": "2004-2005",
        "ml_importance_year": "2006",
        "validation_year": "2007",
        "heldout_test_year": "2008 known non-target rows",
        "pearson_relative_threshold": PEARSON_RELATIVE_THRESHOLD,
        "pearson_min_threshold": PEARSON_MIN_THRESHOLD,
        "ml_importance_relative_threshold": ML_IMPORTANCE_RELATIVE_THRESHOLD,
        "zones": {},
    }

    for zone in sorted(data["zone_id"].astype(int).unique()):
        zone_df = data[data["zone_id"] == zone].copy()
        pearson_selected, pearson_scores = select_by_pearson(zone_df, station_ids, station_cols)
        ml_selected, ml_importances = select_by_ml(zone_df, station_ids, station_cols)
        final_selected = sorted(set(pearson_selected) | set(ml_selected))
        mapping[str(zone)] = final_selected
        diagnostics["zones"][str(zone)] = {
            "pearson_selected": pearson_selected,
            "ml_selected": ml_selected,
            "final_selected": final_selected,
            "pearson_scores": pearson_scores,
            "ml_permutation_importances": ml_importances,
        }

    save_json(mapping, MAPPING_PATH)
    save_json(diagnostics, MAPPING_DIAGNOSTICS_PATH)
    return mapping, diagnostics


def tune_models(mapping: dict[str, list[int]], mapping_diagnostics: dict) -> dict:
    data = prepare_model_frame(mapping, stats_fit_year_max=2006)
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
        print(f"Ridge alpha={alpha}: valid RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")

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
        print(f"HGB {params}: valid RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")

    output = {
        "split_strategy": "train <= 2006, validation = 2007, test = 2008 known nonzero rows",
        "random_state": RANDOM_STATE,
        "mapping_summary": {
            "schema": "zone_id -> list[station_id]",
            "combination_rule": "union",
            "diagnostics_file": MAPPING_DIAGNOSTICS_PATH.name,
            "station_count_by_zone": {zone: len(stations) for zone, stations in mapping.items()},
            "preprocessing": mapping_diagnostics["preprocessing"],
            "pearson_years": mapping_diagnostics["pearson_years"],
            "ml_train_years": mapping_diagnostics["ml_train_years"],
            "ml_importance_year": mapping_diagnostics["ml_importance_year"],
        },
        "other_model": {"alpha": best_ridge["alpha"], "validation_metrics": best_ridge["metrics"]},
        "best_model": best_hgb["params"],
        "best_model_validation_metrics": best_hgb["metrics"],
    }
    save_json(output, BEST_PARAMS_PATH)
    return output


def main() -> None:
    start = time.perf_counter()
    mapping, diagnostics = determine_mapping()
    print("Zone to station mapping:")
    for zone, stations in mapping.items():
        pearson = diagnostics["zones"][zone]["pearson_selected"]
        ml = diagnostics["zones"][zone]["ml_selected"]
        print(f"  Zone {zone}: final={stations}, pearson={pearson}, ml={ml}")
    params = tune_models(mapping, diagnostics)
    print(f"Wrote {MAPPING_PATH.name}, {MAPPING_DIAGNOSTICS_PATH.name}, and {BEST_PARAMS_PATH.name}")
    print(f"Selected parameters: {params}")
    print(f"Total tuning runtime: {time.perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    main()
