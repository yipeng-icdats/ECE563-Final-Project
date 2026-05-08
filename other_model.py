"""Simple baseline model for ECE 563 load forecasting."""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from sklearn.linear_model import Ridge

from config import BEST_PARAMS_PATH, DEFAULT_OTHER_PARAMS
from utils import (
    evaluate_and_time,
    fit_and_time,
    load_mapping,
    make_linear_pipeline,
    prepare_model_frame,
    print_metrics,
    top_errors,
    train_test_time_split,
)


def load_other_params() -> dict:
    if not BEST_PARAMS_PATH.exists():
        return DEFAULT_OTHER_PARAMS.copy()
    with BEST_PARAMS_PATH.open("r", encoding="utf-8") as f:
        params = json.load(f)
    return params.get("other_model", DEFAULT_OTHER_PARAMS.copy())


def main() -> None:
    total_start = time.perf_counter()
    mapping = load_mapping()
    params = load_other_params()
    alpha = float(params.get("alpha", DEFAULT_OTHER_PARAMS["alpha"]))

    data = prepare_model_frame(mapping, stats_fit_year_max=2007)
    train_df, test_df = train_test_time_split(data)
    model = make_linear_pipeline(Ridge(alpha=alpha))

    model, train_time = fit_and_time(model, train_df)
    train_metrics, _, train_pred_time = evaluate_and_time(model, train_df)
    test_metrics, test_pred, test_pred_time = evaluate_and_time(model, test_df)

    print("Model: Ridge Regression baseline")
    print(f"Hyperparameters: alpha={alpha}")
    print("Split: train = 2004-2007 known loads, held-out test = 2008 known non-target loads")
    print("Preprocessing statistics fit through 2007 only; June 1-7, 2008 target rows are prediction-only")
    print_metrics("Training", train_metrics)
    print_metrics("Testing", test_metrics)
    print(f"Final training time: {train_time:.2f} seconds")
    print(f"Training prediction time: {train_pred_time:.2f} seconds")
    print(f"Testing prediction time: {test_pred_time:.2f} seconds")
    print("Top 10 testing errors:")
    print(top_errors(test_df, test_pred).to_string(index=False))
    print(f"Total script runtime: {time.perf_counter() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
