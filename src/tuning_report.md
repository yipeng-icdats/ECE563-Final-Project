# Hyperparameter Tuning Report

This report summarizes the validation-based hyperparameter tuning used for the final forecasting scripts.

## Experimental Setup

- Split strategy: train <= 2006, validation = 2007, test = 2008 known nonzero rows
- Random state: 563
- Selection metric: validation RMSE
- Tuning train years: 2004-2006 known load rows
- Validation year: 2007 known load rows
- Held-out 2008 known non-target rows are not used for tuning.
- June 1-7, 2008 target rows remain prediction-only.

Validation RMSE is used for model selection because it measures error in the same units as load and directly rewards lower forecast magnitude error on a future year that is not used for fitting.

## Candidate Grids

- Ridge alpha values: `[0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 50.0, 100.0, 300.0, 1000.0]`
- HGB learning_rate values: `[0.05, 0.08, 0.1]`
- HGB max_iter values: `[180, 220, 300]`
- HGB max_leaf_nodes values: `[31, 45]`
- HGB l2_regularization values: `[0.01, 0.1]`
- HGB min_samples_leaf values: `[30]`

## Selected Parameters

- Ridge selected alpha: `30.0`
- Ridge validation RMSE: 217075.74
- Ridge validation MAPE: 13.61%
- HGB selected parameters: `{'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 45, 'l2_regularization': 0.1, 'min_samples_leaf': 30}`
- HGB validation RMSE: 186127.90
- HGB validation MAPE: 9.92%

## Runtime Summary

- Total tuning runtime: 344.90 seconds
- Ridge candidates tested: 12
- HGB candidates tested: 36

### Ridge Regression Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `alpha=30.0` | 177020.27 | 10.85 | 217075.74 | 13.61 | 0.57 | 0.95 |
| 2 |  | `alpha=10.0` | 177001.32 | 10.85 | 217078.65 | 13.61 | 0.56 | 0.97 |
| 3 |  | `alpha=50.0` | 177048.54 | 10.85 | 217089.02 | 13.62 | 0.52 | 0.89 |
| 4 |  | `alpha=3.0` | 176993.03 | 10.85 | 217121.76 | 13.61 | 0.58 | 1.06 |
| 5 |  | `alpha=100.0` | 177160.28 | 10.86 | 217147.71 | 13.62 | 0.54 | 0.92 |
| 6 |  | `alpha=1.0` | 176981.41 | 10.84 | 217232.03 | 13.61 | 0.57 | 0.96 |
| 7 |  | `alpha=0.3` | 176960.65 | 10.84 | 217479.46 | 13.60 | 0.56 | 0.89 |
| 8 |  | `alpha=300.0` | 178079.83 | 10.92 | 217630.61 | 13.68 | 0.60 | 1.02 |
| 9 |  | `alpha=0.1` | 176942.69 | 10.84 | 217759.29 | 13.60 | 0.54 | 0.92 |
| 10 |  | `alpha=0.01` | 176928.81 | 10.84 | 218053.24 | 13.60 | 0.56 | 0.95 |
| 11 |  | `alpha=0.001` | 176927.14 | 10.84 | 218096.48 | 13.60 | 0.57 | 0.96 |
| 12 |  | `alpha=1000.0` | 184853.27 | 11.41 | 221364.06 | 14.11 | 0.58 | 1.00 |

### HistGradientBoostingRegressor Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 83622.75 | 5.16 | 186127.90 | 9.92 | 11.39 | 12.77 |
| 2 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 83363.88 | 5.18 | 186138.18 | 9.93 | 11.38 | 12.71 |
| 3 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 88303.98 | 5.45 | 189489.83 | 10.05 | 8.66 | 9.86 |
| 4 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 88794.42 | 5.46 | 189845.51 | 10.07 | 8.61 | 9.85 |
| 5 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 86977.45 | 5.33 | 190637.54 | 10.07 | 11.56 | 12.97 |
| 6 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 86159.49 | 5.33 | 190851.75 | 10.08 | 11.52 | 12.92 |
| 7 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 88356.64 | 5.41 | 191064.61 | 10.09 | 8.56 | 9.62 |
| 8 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 88524.50 | 5.42 | 191660.55 | 10.13 | 8.13 | 9.19 |
| 9 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 92001.53 | 5.62 | 194308.07 | 10.22 | 8.71 | 9.96 |
| 10 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 92499.08 | 5.62 | 194462.25 | 10.24 | 8.69 | 9.94 |
| 11 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 93392.67 | 5.71 | 194746.50 | 10.30 | 6.44 | 7.37 |
| 12 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 93262.39 | 5.69 | 194772.57 | 10.24 | 6.30 | 7.31 |
| 13 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 91752.02 | 5.58 | 194854.89 | 10.25 | 6.82 | 7.71 |
| 14 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 91958.61 | 5.58 | 195253.26 | 10.24 | 6.88 | 7.77 |
| 15 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 91740.94 | 5.59 | 195719.23 | 10.27 | 8.77 | 10.01 |
| 16 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 91837.20 | 5.58 | 196843.85 | 10.27 | 8.36 | 9.46 |
| 17 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 94316.51 | 5.70 | 198033.14 | 10.36 | 11.99 | 13.56 |
| 18 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 94354.34 | 5.71 | 198538.91 | 10.38 | 12.01 | 13.51 |
| 19 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 96443.61 | 5.86 | 198782.07 | 10.41 | 5.49 | 6.30 |
| 20 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 96658.94 | 5.88 | 198809.26 | 10.46 | 5.20 | 5.99 |
| 21 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 95508.14 | 5.76 | 199876.52 | 10.42 | 7.37 | 8.31 |
| 22 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 97672.00 | 5.87 | 200414.70 | 10.47 | 6.61 | 7.59 |
| 23 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 97520.28 | 5.87 | 200726.15 | 10.44 | 6.63 | 7.62 |
| 24 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 95518.95 | 5.76 | 200774.08 | 10.41 | 7.39 | 8.33 |
| 25 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 100283.56 | 6.00 | 204059.25 | 10.60 | 9.15 | 10.51 |
| 26 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 101486.83 | 6.05 | 204641.49 | 10.64 | 5.33 | 6.15 |
| 27 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 100549.62 | 6.00 | 204792.70 | 10.62 | 9.14 | 10.53 |
| 28 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 101424.21 | 6.06 | 204835.47 | 10.61 | 5.60 | 6.42 |
| 29 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 100589.15 | 5.98 | 205732.75 | 10.62 | 8.88 | 10.13 |
| 30 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 100804.09 | 5.98 | 206401.45 | 10.65 | 9.22 | 10.42 |
| 31 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 107188.12 | 6.29 | 212883.52 | 10.94 | 6.78 | 7.82 |
| 32 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 107680.91 | 6.30 | 213784.46 | 10.95 | 6.74 | 7.77 |
| 33 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 106930.92 | 6.21 | 213850.27 | 10.89 | 7.78 | 8.75 |
| 34 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 106975.11 | 6.22 | 214248.24 | 10.91 | 7.49 | 8.47 |
| 35 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 113348.49 | 6.53 | 220760.77 | 11.22 | 6.14 | 6.99 |
| 36 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 113748.69 | 6.53 | 221643.07 | 11.23 | 5.76 | 6.59 |
