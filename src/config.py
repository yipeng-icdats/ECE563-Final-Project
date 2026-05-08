"""Project-wide constants for the ECE 563 load forecasting prototype."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent

# The grading layout places the supplied CSV files in the parent directory above
# the submitted code folder.
LOAD_CANDIDATES = [
    ROOT.parent / "Load_history_final.csv",
]
TEMP_CANDIDATES = [
    ROOT.parent / "Temp_history_final.csv",
]

MAPPING_PATH = ROOT / "mapping.json"
MAPPING_DIAGNOSTICS_PATH = ROOT / "mapping_diagnostics.json"
BEST_PARAMS_PATH = ROOT / "best_params.json"
PREDICTION_PATH = ROOT / "Load_prediction.csv"
TUNING_REPORT_PATH = ROOT / "tuning_report.md"

HOUR_COLUMNS = [f"h{i}" for i in range(1, 25)]
TARGET_YEAR = 2008
TARGET_MONTH = 6
TARGET_DAYS = list(range(1, 8))

RANDOM_STATE = 563
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
PEARSON_RELATIVE_THRESHOLD = 0.90
PEARSON_MIN_THRESHOLD = 0.05
ML_IMPORTANCE_RELATIVE_THRESHOLD = 0.10

DEFAULT_OTHER_PARAMS = {
    "alpha": 10.0,
}

DEFAULT_BEST_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 220,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.01,
    "min_samples_leaf": 30,
}
