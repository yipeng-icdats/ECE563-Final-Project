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

- Ridge selected alpha: `0.001`
- Ridge validation RMSE: 280766.30
- Ridge validation MAPE: 142.01%
- HGB selected parameters: `{'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 45, 'l2_regularization': 0.1, 'min_samples_leaf': 30}`
- HGB validation RMSE: 177694.97
- HGB validation MAPE: 31.18%

## Runtime Summary

- Total tuning runtime: 322.06 seconds
- Ridge candidates tested: 12
- HGB candidates tested: 36

### Ridge Regression Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `alpha=0.001` | 165650.68 | 127.14 | 280766.30 | 142.01 | 0.61 | 0.97 |
| 2 |  | `alpha=0.01` | 165650.68 | 127.14 | 280777.14 | 141.97 | 0.56 | 0.93 |
| 3 |  | `alpha=0.1` | 165650.84 | 127.15 | 280853.91 | 141.72 | 0.54 | 0.90 |
| 4 |  | `alpha=0.3` | 165651.24 | 127.15 | 280932.87 | 141.46 | 0.57 | 0.93 |
| 5 |  | `alpha=1.0` | 165651.86 | 127.16 | 281009.32 | 141.23 | 0.54 | 0.92 |
| 6 |  | `alpha=3.0` | 165652.24 | 127.17 | 281048.48 | 141.13 | 0.56 | 0.91 |
| 7 |  | `alpha=10.0` | 165652.51 | 127.19 | 281075.44 | 141.12 | 0.52 | 0.89 |
| 8 |  | `alpha=30.0` | 165653.46 | 127.24 | 281113.11 | 141.19 | 0.50 | 0.87 |
| 9 |  | `alpha=50.0` | 165655.17 | 127.30 | 281146.07 | 141.28 | 0.56 | 0.95 |
| 10 |  | `alpha=100.0` | 165662.46 | 127.44 | 281222.72 | 141.51 | 0.56 | 0.94 |
| 11 |  | `alpha=300.0` | 165721.28 | 128.07 | 281478.06 | 142.48 | 0.54 | 0.91 |
| 12 |  | `alpha=1000.0` | 166069.77 | 130.24 | 281998.30 | 145.50 | 0.54 | 0.95 |

### HistGradientBoostingRegressor Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 76367.62 | 19.62 | 177694.97 | 31.18 | 10.42 | 11.42 |
| 2 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 76593.04 | 17.85 | 179155.97 | 29.88 | 10.43 | 11.46 |
| 3 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 82740.47 | 19.24 | 179458.49 | 31.52 | 7.98 | 8.90 |
| 4 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 79644.21 | 18.25 | 179637.11 | 29.69 | 10.86 | 11.94 |
| 5 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 82043.17 | 20.88 | 180043.51 | 33.35 | 7.93 | 8.85 |
| 6 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 79537.64 | 18.70 | 180127.36 | 30.49 | 10.85 | 11.92 |
| 7 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 81424.29 | 19.20 | 180736.41 | 31.27 | 8.04 | 8.85 |
| 8 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 81264.19 | 18.38 | 181747.62 | 30.61 | 8.12 | 8.94 |
| 9 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 85432.23 | 20.30 | 182655.52 | 32.88 | 8.33 | 9.29 |
| 10 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 84520.78 | 18.63 | 183010.95 | 30.38 | 8.61 | 9.58 |
| 11 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 84466.42 | 19.43 | 183045.60 | 31.67 | 6.87 | 7.59 |
| 12 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 87368.82 | 19.83 | 183194.09 | 32.58 | 6.19 | 6.95 |
| 13 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 85458.20 | 19.20 | 183219.36 | 31.93 | 8.41 | 9.44 |
| 14 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 84467.99 | 19.22 | 183627.45 | 31.54 | 8.14 | 9.05 |
| 15 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 86903.60 | 21.53 | 183947.06 | 34.25 | 6.18 | 6.93 |
| 16 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 84515.32 | 18.71 | 184532.84 | 31.56 | 6.98 | 7.72 |
| 17 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 86730.75 | 19.58 | 185489.57 | 32.52 | 11.51 | 12.80 |
| 18 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 86720.87 | 19.16 | 185642.53 | 31.80 | 11.53 | 12.75 |
| 19 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 90329.16 | 20.83 | 185842.71 | 34.23 | 5.01 | 5.68 |
| 20 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 87795.43 | 18.84 | 186004.84 | 31.02 | 6.80 | 7.56 |
| 21 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 89842.06 | 22.24 | 186560.87 | 35.08 | 4.89 | 5.53 |
| 22 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 87744.69 | 19.43 | 186704.29 | 32.02 | 7.11 | 7.91 |
| 23 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 90132.21 | 19.37 | 187169.24 | 32.59 | 6.39 | 7.24 |
| 24 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 90085.39 | 21.30 | 187226.17 | 34.51 | 6.00 | 6.80 |
| 25 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 92559.25 | 20.71 | 189870.20 | 33.79 | 8.09 | 9.20 |
| 26 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 92585.45 | 20.92 | 190000.36 | 34.41 | 8.63 | 9.78 |
| 27 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 93309.44 | 21.98 | 190563.34 | 35.50 | 5.12 | 5.79 |
| 28 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 92055.40 | 20.73 | 190825.13 | 34.36 | 8.67 | 9.69 |
| 29 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 93654.80 | 20.09 | 190971.83 | 33.73 | 5.11 | 5.76 |
| 30 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 92254.52 | 20.30 | 191162.12 | 33.97 | 8.43 | 9.45 |
| 31 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 96055.91 | 21.88 | 195145.42 | 35.84 | 7.49 | 8.33 |
| 32 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 97836.50 | 22.22 | 195204.65 | 36.14 | 6.32 | 7.21 |
| 33 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 96170.56 | 21.52 | 195513.15 | 35.60 | 7.31 | 8.16 |
| 34 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 97946.25 | 22.23 | 195553.03 | 36.60 | 6.56 | 7.41 |
| 35 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 101859.37 | 24.03 | 200016.52 | 38.07 | 5.69 | 6.44 |
| 36 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 101998.86 | 24.35 | 200330.73 | 38.94 | 5.69 | 6.46 |
