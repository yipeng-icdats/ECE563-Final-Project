"""Best model script: train, evaluate, and generate Load_prediction.csv."""

from __future__ import annotations

import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from sklearn.ensemble import HistGradientBoostingRegressor

from config import RANDOM_STATE
from utils import (
    evaluate_and_time,
    final_prediction_train_split,
    fit_and_time,
    load_best_params,
    load_mapping,
    make_log_target_regressor,
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
    final_prediction_train_df = final_prediction_train_split(data)
    model = make_log_target_regressor(
        HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    )

    model, train_time = fit_and_time(model, train_df)
    train_metrics, _, train_pred_time = evaluate_and_time(model, train_df)
    test_metrics, test_pred, test_pred_time = evaluate_and_time(model, test_df)

    final_prediction_model = make_log_target_regressor(
        HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    )
    final_prediction_model, final_refit_time = fit_and_time(
        final_prediction_model,
        final_prediction_train_df,
    )
    june_pred_time = write_prediction_csv(final_prediction_model, data)

    print("Model: HistGradientBoostingRegressor")
    print(f"Hyperparameters: {params}")
    print("Split: train = 2004-2007 known loads, held-out test = 2008 known non-target loads")
    print("Final prediction refit: 2004-2008 known non-target loads")
    print("Preprocessing statistics fit through 2007 only; June 1-7, 2008 target rows are prediction-only")
    print_metrics("Training", train_metrics)
    print_metrics("Testing", test_metrics)
    print(f"Test-evaluation training time: {train_time:.2f} seconds")
    print(f"Training prediction time: {train_pred_time:.2f} seconds")
    print(f"Test prediction time: {test_pred_time:.2f} seconds")
    print(f"Final all-known refit time: {final_refit_time:.2f} seconds")
    print(f"Final June 1-7, 2008 prediction/CSV generation time: {june_pred_time:.2f} seconds")
    print("Top 10 testing errors:")
    print(top_errors(test_df, test_pred).to_string(index=False))
    print("Generated Load_prediction.csv")
    print(f"Total script runtime: {time.perf_counter() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
