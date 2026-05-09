"""Mapping discovery and lightweight hyperparameter tuning.

This script is allowed to run longer than the final model scripts.  It writes:
- mapping.json
- mapping_diagnostics.json
- best_params.json
"""

from __future__ import annotations

import os
import time
from itertools import product

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
    TUNING_REPORT_PATH,
)
from utils import (
    FEATURE_COLUMNS,
    add_calendar_features,
    clean_load_values,
    clean_temperature_values,
    get_xy,
    load_raw_data,
    make_log_target_pipeline,
    pivot_station_temperatures,
    prepare_model_frame,
    regression_metrics,
    save_json,
    validation_split,
    wide_to_long,
)


SPLIT_STRATEGY = "train <= 2006, validation = 2007, test = 2008 known nonzero rows"

RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 50.0, 100.0, 300.0, 1000.0]

HGB_GRID_VALUES = {
    "learning_rate": [0.05, 0.08, 0.1],
    "max_iter": [180, 220, 300],
    "max_leaf_nodes": [31, 45],
    "l2_regularization": [0.01, 0.1],
    "min_samples_leaf": [30],
}


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


def rank_candidates(candidate_results: list[dict]) -> list[dict]:
    ranked = sorted(candidate_results, key=lambda item: item["validation_metrics"]["rmse"])
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
        result["selected"] = rank == 1
    return ranked


def evaluate_ridge_candidate(alpha: float, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    candidate_start = time.perf_counter()
    model = make_log_target_pipeline(Ridge(alpha=alpha))

    fit_start = time.perf_counter()
    model.fit(*get_xy(train))
    fit_seconds = time.perf_counter() - fit_start

    train_pred_start = time.perf_counter()
    train_pred = np.maximum(model.predict(train[FEATURE_COLUMNS]), 0.0)
    train_prediction_seconds = time.perf_counter() - train_pred_start

    valid_pred_start = time.perf_counter()
    valid_pred = np.maximum(model.predict(valid[FEATURE_COLUMNS]), 0.0)
    validation_prediction_seconds = time.perf_counter() - valid_pred_start

    return {
        "model": "Ridge Regression",
        "params": {"alpha": alpha},
        "train_metrics": regression_metrics(train["load"], train_pred),
        "validation_metrics": regression_metrics(valid["load"], valid_pred),
        "fit_seconds": fit_seconds,
        "train_prediction_seconds": train_prediction_seconds,
        "validation_prediction_seconds": validation_prediction_seconds,
        "total_seconds": time.perf_counter() - candidate_start,
    }


def build_hgb_grid() -> list[dict]:
    keys = list(HGB_GRID_VALUES)
    grid = []
    for values in product(*(HGB_GRID_VALUES[key] for key in keys)):
        params = dict(zip(keys, values))
        grid.append(params)
    return grid


def evaluate_hgb_candidate(params: dict, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    candidate_start = time.perf_counter()
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)

    fit_start = time.perf_counter()
    model.fit(train[FEATURE_COLUMNS], train["load"])
    fit_seconds = time.perf_counter() - fit_start

    train_pred_start = time.perf_counter()
    train_pred = np.maximum(model.predict(train[FEATURE_COLUMNS]), 0.0)
    train_prediction_seconds = time.perf_counter() - train_pred_start

    valid_pred_start = time.perf_counter()
    valid_pred = np.maximum(model.predict(valid[FEATURE_COLUMNS]), 0.0)
    validation_prediction_seconds = time.perf_counter() - valid_pred_start

    return {
        "model": "HistGradientBoostingRegressor",
        "params": params.copy(),
        "train_metrics": regression_metrics(train["load"], train_pred),
        "validation_metrics": regression_metrics(valid["load"], valid_pred),
        "fit_seconds": fit_seconds,
        "train_prediction_seconds": train_prediction_seconds,
        "validation_prediction_seconds": validation_prediction_seconds,
        "total_seconds": time.perf_counter() - candidate_start,
    }


def format_float(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def markdown_candidate_table(candidate_results: list[dict], model_label: str) -> list[str]:
    lines = [
        f"### {model_label} Candidate Results",
        "",
        "| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |",
        "|---:|:---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(candidate_results, key=lambda item: item["rank"]):
        params = ", ".join(f"{key}={value}" for key, value in result["params"].items())
        selected = "yes" if result["selected"] else ""
        lines.append(
            "| "
            f"{result['rank']} | {selected} | `{params}` | "
            f"{format_float(result['train_metrics']['rmse'])} | "
            f"{format_float(result['train_metrics']['mape'])} | "
            f"{format_float(result['validation_metrics']['rmse'])} | "
            f"{format_float(result['validation_metrics']['mape'])} | "
            f"{format_float(result['fit_seconds'])} | "
            f"{format_float(result['total_seconds'])} |"
        )
    lines.append("")
    return lines


def write_tuning_report(tuning_output: dict, path=TUNING_REPORT_PATH) -> None:
    other = tuning_output["other_model"]
    best = tuning_output["best_model"]
    lines = [
        "# Hyperparameter Tuning Report",
        "",
        "This report summarizes the validation-based hyperparameter tuning used for the final forecasting scripts.",
        "",
        "## Experimental Setup",
        "",
        f"- Split strategy: {tuning_output['split_strategy']}",
        f"- Random state: {tuning_output['random_state']}",
        "- Selection metric: validation RMSE",
        "- Tuning train years: 2004-2006 known load rows",
        "- Validation year: 2007 known load rows",
        "- Held-out 2008 known non-target rows are not used for tuning.",
        "- June 1-7, 2008 target rows remain prediction-only.",
        "",
        "Validation RMSE is used for model selection because it measures error in the same units as load and directly rewards lower forecast magnitude error on a future year that is not used for fitting.",
        "",
        "## Candidate Grids",
        "",
        f"- Ridge alpha values: `{RIDGE_ALPHA_GRID}`",
        f"- HGB learning_rate values: `{HGB_GRID_VALUES['learning_rate']}`",
        f"- HGB max_iter values: `{HGB_GRID_VALUES['max_iter']}`",
        f"- HGB max_leaf_nodes values: `{HGB_GRID_VALUES['max_leaf_nodes']}`",
        f"- HGB l2_regularization values: `{HGB_GRID_VALUES['l2_regularization']}`",
        f"- HGB min_samples_leaf values: `{HGB_GRID_VALUES['min_samples_leaf']}`",
        "",
        "## Selected Parameters",
        "",
        f"- Ridge selected alpha: `{other['alpha']}`",
        f"- Ridge validation RMSE: {format_float(other['validation_metrics']['rmse'])}",
        f"- Ridge validation MAPE: {format_float(other['validation_metrics']['mape'])}%",
        f"- HGB selected parameters: `{best}`",
        f"- HGB validation RMSE: {format_float(tuning_output['best_model_validation_metrics']['rmse'])}",
        f"- HGB validation MAPE: {format_float(tuning_output['best_model_validation_metrics']['mape'])}%",
        "",
        "## Runtime Summary",
        "",
        f"- Total tuning runtime: {format_float(tuning_output['tuning_runtime_seconds'])} seconds",
        f"- Ridge candidates tested: {tuning_output['other_model_candidate_count']}",
        f"- HGB candidates tested: {tuning_output['best_model_candidate_count']}",
        "",
    ]
    lines.extend(markdown_candidate_table(tuning_output["other_model_candidate_results"], "Ridge Regression"))
    lines.extend(markdown_candidate_table(tuning_output["best_model_candidate_results"], "HistGradientBoostingRegressor"))
    path.write_text("\n".join(lines), encoding="utf-8")


def tune_models(mapping: dict[str, list[int]], mapping_diagnostics: dict) -> dict:
    tuning_start = time.perf_counter()
    data = prepare_model_frame(mapping, stats_fit_year_max=2006)
    train, valid, _ = validation_split(data)

    ridge_results = []
    ridge_start = time.perf_counter()
    for alpha in RIDGE_ALPHA_GRID:
        result = evaluate_ridge_candidate(alpha, train, valid)
        ridge_results.append(result)
        metrics = result["validation_metrics"]
        print(f"Ridge alpha={alpha}: valid RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
    ridge_results = rank_candidates(ridge_results)
    ridge_runtime = time.perf_counter() - ridge_start
    best_ridge = ridge_results[0]

    hgb_results = []
    hgb_grid = build_hgb_grid()
    hgb_start = time.perf_counter()
    for params in hgb_grid:
        result = evaluate_hgb_candidate(params, train, valid)
        hgb_results.append(result)
        metrics = result["validation_metrics"]
        print(f"HGB {params}: valid RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
    hgb_results = rank_candidates(hgb_results)
    hgb_runtime = time.perf_counter() - hgb_start
    best_hgb = hgb_results[0]

    output = {
        "split_strategy": SPLIT_STRATEGY,
        "random_state": RANDOM_STATE,
        "selection_metric": "validation_rmse",
        "tuning_runtime_seconds": time.perf_counter() - tuning_start,
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
        "other_model": {
            "alpha": best_ridge["params"]["alpha"],
            "validation_metrics": best_ridge["validation_metrics"],
        },
        "other_model_candidate_count": len(ridge_results),
        "other_model_tuning_runtime_seconds": ridge_runtime,
        "other_model_candidate_results": ridge_results,
        "best_model": best_hgb["params"],
        "best_model_validation_metrics": best_hgb["validation_metrics"],
        "best_model_candidate_count": len(hgb_results),
        "best_model_tuning_runtime_seconds": hgb_runtime,
        "best_model_candidate_results": hgb_results,
        "tuning_report": TUNING_REPORT_PATH.name,
    }
    save_json(output, BEST_PARAMS_PATH)
    write_tuning_report(output)
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
