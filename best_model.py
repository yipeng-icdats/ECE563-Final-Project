"""Best model script: train, evaluate, and generate Load_prediction.csv."""

from __future__ import annotations

import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from sklearn.ensemble import HistGradientBoostingRegressor

from config import RANDOM_STATE
from utils import (
    evaluate_and_time,
    fit_and_time,
    load_best_params,
    load_mapping,
    prepare_model_frame,
    print_metrics,
    top_errors,
    train_test_time_split,
    write_prediction_csv,
)


def main() -> None:
    total_start = time.perf_counter()
    mapping = load_mapping()
    params = load_best_params()

    data = prepare_model_frame(mapping, stats_fit_year_max=2007)
    train_df, test_df = train_test_time_split(data)
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)

    model, train_time = fit_and_time(model, train_df)
    train_metrics, _, train_pred_time = evaluate_and_time(model, train_df)
    test_metrics, test_pred, test_pred_time = evaluate_and_time(model, test_df)
    june_pred_time = write_prediction_csv(model, data)

    print("Model: HistGradientBoostingRegressor")
    print(f"Hyperparameters: {params}")
    print("Split: train = 2004-2007 known loads, held-out test = 2008 known non-target loads")
    print("Preprocessing statistics fit through 2007 only; June 1-7, 2008 target rows are prediction-only")
    print_metrics("Training", train_metrics)
    print_metrics("Testing", test_metrics)
    print(f"Final training time: {train_time:.2f} seconds")
    print(f"Training prediction time: {train_pred_time:.2f} seconds")
    print(f"Testing prediction time: {test_pred_time:.2f} seconds")
    print(f"June 1-7, 2008 prediction time: {june_pred_time:.2f} seconds")
    print("Top 10 testing errors:")
    print(top_errors(test_df, test_pred).to_string(index=False))
    print("Generated Load_prediction.csv")
    print(f"Total script runtime: {time.perf_counter() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
