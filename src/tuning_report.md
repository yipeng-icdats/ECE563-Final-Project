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
- HGB validation RMSE: 176712.65
- HGB validation MAPE: 21.16%

## Runtime Summary

- Total tuning runtime: 322.22 seconds
- Ridge candidates tested: 12
- HGB candidates tested: 36

### Ridge Regression Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `alpha=30.0` | 177020.27 | 10.85 | 217075.74 | 13.61 | 0.55 | 0.91 |
| 2 |  | `alpha=10.0` | 177001.32 | 10.85 | 217078.65 | 13.61 | 0.55 | 0.95 |
| 3 |  | `alpha=50.0` | 177048.54 | 10.85 | 217089.02 | 13.62 | 0.56 | 0.92 |
| 4 |  | `alpha=3.0` | 176993.03 | 10.85 | 217121.76 | 13.61 | 0.56 | 0.94 |
| 5 |  | `alpha=100.0` | 177160.28 | 10.86 | 217147.71 | 13.62 | 0.52 | 0.88 |
| 6 |  | `alpha=1.0` | 176981.41 | 10.84 | 217232.03 | 13.61 | 0.53 | 0.91 |
| 7 |  | `alpha=0.3` | 176960.65 | 10.84 | 217479.46 | 13.60 | 0.53 | 0.92 |
| 8 |  | `alpha=300.0` | 178079.83 | 10.92 | 217630.61 | 13.68 | 0.56 | 0.94 |
| 9 |  | `alpha=0.1` | 176942.69 | 10.84 | 217759.29 | 13.60 | 0.54 | 0.92 |
| 10 |  | `alpha=0.01` | 176928.81 | 10.84 | 218053.24 | 13.60 | 0.49 | 0.92 |
| 11 |  | `alpha=0.001` | 176927.14 | 10.84 | 218096.48 | 13.60 | 0.49 | 0.84 |
| 12 |  | `alpha=1000.0` | 184853.27 | 11.41 | 221364.06 | 14.11 | 0.56 | 0.93 |

### HistGradientBoostingRegressor Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 76277.96 | 15.77 | 176712.65 | 21.16 | 10.45 | 11.47 |
| 2 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 76002.09 | 15.40 | 176862.93 | 20.44 | 10.77 | 11.79 |
| 3 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 79262.50 | 15.26 | 178237.09 | 20.35 | 10.61 | 11.69 |
| 4 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 81908.02 | 16.05 | 178799.39 | 21.11 | 8.06 | 9.01 |
| 5 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 82108.59 | 16.62 | 178991.05 | 22.20 | 8.21 | 9.21 |
| 6 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 80694.32 | 16.03 | 179508.64 | 21.20 | 7.97 | 8.85 |
| 7 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 81109.15 | 16.21 | 179643.56 | 21.58 | 7.98 | 8.79 |
| 8 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 79239.25 | 15.48 | 179654.67 | 20.78 | 10.51 | 11.60 |
| 9 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 85119.96 | 17.22 | 181011.23 | 22.81 | 8.21 | 9.30 |
| 10 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 83788.95 | 16.63 | 181784.25 | 22.25 | 6.60 | 7.31 |
| 11 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 84131.06 | 16.35 | 181788.28 | 21.60 | 8.49 | 9.59 |
| 12 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 84323.46 | 16.70 | 182066.67 | 22.14 | 7.19 | 7.94 |
| 13 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 86556.10 | 17.28 | 182598.01 | 23.14 | 6.03 | 6.75 |
| 14 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 85132.97 | 17.29 | 182622.83 | 22.64 | 8.40 | 9.35 |
| 15 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 84121.15 | 16.30 | 182835.71 | 21.71 | 8.66 | 9.57 |
| 16 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 86543.68 | 16.68 | 182918.82 | 22.02 | 6.30 | 7.08 |
| 17 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 86269.46 | 17.23 | 184147.02 | 22.64 | 10.95 | 12.21 |
| 18 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 87252.00 | 17.09 | 184248.75 | 22.67 | 7.15 | 7.95 |
| 19 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 86576.77 | 17.47 | 184402.43 | 22.92 | 11.01 | 12.32 |
| 20 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 89748.16 | 17.79 | 185340.30 | 23.86 | 4.92 | 5.56 |
| 21 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 87227.84 | 17.04 | 185346.02 | 22.59 | 7.11 | 7.91 |
| 22 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 89937.45 | 18.75 | 185567.47 | 24.49 | 6.29 | 7.09 |
| 23 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 89648.60 | 17.17 | 185839.84 | 22.59 | 4.93 | 5.58 |
| 24 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 89749.29 | 18.52 | 186244.81 | 24.08 | 6.32 | 7.13 |
| 25 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 92351.16 | 18.47 | 188537.71 | 24.44 | 8.36 | 9.54 |
| 26 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 92231.84 | 18.67 | 188682.21 | 25.11 | 8.34 | 9.46 |
| 27 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 93153.59 | 19.72 | 188915.14 | 25.54 | 5.04 | 5.73 |
| 28 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 91690.52 | 18.92 | 189481.47 | 24.56 | 8.73 | 9.79 |
| 29 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 93221.45 | 19.54 | 189681.74 | 25.20 | 5.39 | 6.12 |
| 30 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 92017.42 | 19.14 | 190043.80 | 24.84 | 8.44 | 9.43 |
| 31 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 95648.43 | 20.56 | 193858.41 | 26.07 | 7.38 | 8.23 |
| 32 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 97940.41 | 20.82 | 194499.23 | 27.14 | 6.30 | 7.19 |
| 33 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 95970.28 | 20.88 | 194557.37 | 26.86 | 7.32 | 8.17 |
| 34 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 97806.08 | 20.82 | 194600.98 | 27.61 | 6.58 | 7.48 |
| 35 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 101849.85 | 23.39 | 199765.94 | 29.62 | 5.67 | 6.44 |
| 36 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 101895.97 | 23.77 | 199981.98 | 30.18 | 5.63 | 6.38 |
