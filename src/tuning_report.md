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
- Ridge validation RMSE: 279783.06
- Ridge validation MAPE: 101.94%
- HGB selected parameters: `{'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 45, 'l2_regularization': 0.1, 'min_samples_leaf': 30}`
- HGB validation RMSE: 176712.65
- HGB validation MAPE: 21.16%

## Runtime Summary

- Total tuning runtime: 325.02 seconds
- Ridge candidates tested: 12
- HGB candidates tested: 36

### Ridge Regression Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `alpha=0.001` | 164630.94 | 69.62 | 279783.06 | 101.94 | 0.55 | 0.94 |
| 2 |  | `alpha=0.01` | 164630.90 | 69.62 | 279793.05 | 101.87 | 0.54 | 0.93 |
| 3 |  | `alpha=0.1` | 164630.70 | 69.62 | 279863.83 | 101.40 | 0.53 | 0.88 |
| 4 |  | `alpha=0.3` | 164630.70 | 69.62 | 279936.59 | 100.91 | 0.56 | 0.93 |
| 5 |  | `alpha=1.0` | 164630.85 | 69.62 | 280007.01 | 100.46 | 0.57 | 0.93 |
| 6 |  | `alpha=3.0` | 164630.94 | 69.62 | 280043.26 | 100.24 | 0.56 | 0.93 |
| 7 |  | `alpha=10.0` | 164630.88 | 69.62 | 280068.81 | 100.12 | 0.55 | 0.93 |
| 8 |  | `alpha=30.0` | 164631.07 | 69.64 | 280105.54 | 100.02 | 0.54 | 0.94 |
| 9 |  | `alpha=50.0` | 164632.00 | 69.66 | 280137.81 | 99.95 | 0.57 | 0.97 |
| 10 |  | `alpha=100.0` | 164637.13 | 69.71 | 280212.62 | 99.79 | 0.55 | 0.91 |
| 11 |  | `alpha=300.0` | 164685.30 | 69.94 | 280458.88 | 99.36 | 0.53 | 0.88 |
| 12 |  | `alpha=1000.0` | 164988.92 | 70.76 | 280945.19 | 98.78 | 0.57 | 1.01 |

### HistGradientBoostingRegressor Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 76277.96 | 15.77 | 176712.65 | 21.16 | 11.17 | 12.29 |
| 2 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 76002.09 | 15.40 | 176862.93 | 20.44 | 10.79 | 11.90 |
| 3 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 79262.50 | 15.26 | 178237.09 | 20.35 | 10.97 | 12.08 |
| 4 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 81908.02 | 16.05 | 178799.39 | 21.11 | 7.76 | 8.72 |
| 5 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 82108.59 | 16.62 | 178991.05 | 22.20 | 8.11 | 9.05 |
| 6 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 80694.32 | 16.03 | 179508.64 | 21.20 | 8.19 | 9.04 |
| 7 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 81109.15 | 16.21 | 179643.56 | 21.58 | 8.20 | 9.04 |
| 8 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 79239.25 | 15.48 | 179654.67 | 20.78 | 11.12 | 12.23 |
| 9 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 85119.96 | 17.22 | 181011.23 | 22.81 | 8.10 | 9.09 |
| 10 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 83788.95 | 16.63 | 181784.25 | 22.25 | 6.85 | 7.58 |
| 11 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 84131.06 | 16.35 | 181788.28 | 21.60 | 8.28 | 9.16 |
| 12 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 84323.46 | 16.70 | 182066.67 | 22.14 | 6.84 | 7.57 |
| 13 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 86556.10 | 17.28 | 182598.01 | 23.14 | 5.91 | 6.66 |
| 14 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 85132.97 | 17.29 | 182622.83 | 22.64 | 8.42 | 9.41 |
| 15 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 84121.15 | 16.30 | 182835.71 | 21.71 | 8.47 | 9.42 |
| 16 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 86543.68 | 16.68 | 182918.82 | 22.02 | 6.21 | 6.99 |
| 17 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 86269.46 | 17.23 | 184147.02 | 22.64 | 11.67 | 12.91 |
| 18 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 87252.00 | 17.09 | 184248.75 | 22.67 | 7.00 | 7.77 |
| 19 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 86576.77 | 17.47 | 184402.43 | 22.92 | 11.01 | 12.30 |
| 20 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 89748.16 | 17.79 | 185340.30 | 23.86 | 5.16 | 5.82 |
| 21 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 87227.84 | 17.04 | 185346.02 | 22.59 | 6.73 | 7.49 |
| 22 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 89937.45 | 18.75 | 185567.47 | 24.49 | 6.32 | 7.11 |
| 23 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 89648.60 | 17.17 | 185839.84 | 22.59 | 5.18 | 5.83 |
| 24 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 89749.29 | 18.52 | 186244.81 | 24.08 | 6.28 | 7.08 |
| 25 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 92351.16 | 18.47 | 188537.71 | 24.44 | 8.35 | 9.49 |
| 26 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 92231.84 | 18.67 | 188682.21 | 25.11 | 8.61 | 9.76 |
| 27 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 93153.59 | 19.72 | 188915.14 | 25.54 | 5.31 | 6.00 |
| 28 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 91690.52 | 18.92 | 189481.47 | 24.56 | 8.64 | 9.67 |
| 29 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 93221.45 | 19.54 | 189681.74 | 25.20 | 5.25 | 5.93 |
| 30 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 92017.42 | 19.14 | 190043.80 | 24.84 | 8.71 | 9.74 |
| 31 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 95648.43 | 20.56 | 193858.41 | 26.07 | 7.03 | 7.88 |
| 32 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 97940.41 | 20.82 | 194499.23 | 27.14 | 6.63 | 7.57 |
| 33 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 95970.28 | 20.88 | 194557.37 | 26.86 | 7.42 | 8.34 |
| 34 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 97806.08 | 20.82 | 194600.98 | 27.61 | 7.05 | 7.95 |
| 35 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 101849.85 | 23.39 | 199765.94 | 29.62 | 5.90 | 6.66 |
| 36 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 101895.97 | 23.77 | 199981.98 | 30.18 | 5.33 | 6.07 |
