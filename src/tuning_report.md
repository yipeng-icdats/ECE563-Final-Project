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
- Ridge validation RMSE: 280586.65
- Ridge validation MAPE: 115.47%
- HGB selected parameters: `{'learning_rate': 0.1, 'max_iter': 300, 'max_leaf_nodes': 45, 'l2_regularization': 0.1, 'min_samples_leaf': 30}`
- HGB validation RMSE: 177691.18
- HGB validation MAPE: 28.64%

## Runtime Summary

- Total tuning runtime: 316.69 seconds
- Ridge candidates tested: 12
- HGB candidates tested: 36

### Ridge Regression Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `alpha=0.001` | 164792.17 | 69.83 | 280586.65 | 115.47 | 0.57 | 0.97 |
| 2 |  | `alpha=0.01` | 164792.13 | 69.83 | 280597.09 | 115.39 | 0.55 | 0.92 |
| 3 |  | `alpha=0.1` | 164791.95 | 69.83 | 280671.03 | 114.83 | 0.53 | 0.90 |
| 4 |  | `alpha=0.3` | 164791.97 | 69.83 | 280747.09 | 114.27 | 0.52 | 0.89 |
| 5 |  | `alpha=1.0` | 164792.18 | 69.83 | 280820.71 | 113.74 | 0.50 | 0.89 |
| 6 |  | `alpha=3.0` | 164792.30 | 69.83 | 280858.45 | 113.48 | 0.54 | 0.90 |
| 7 |  | `alpha=10.0` | 164792.25 | 69.84 | 280884.48 | 113.36 | 0.53 | 0.93 |
| 8 |  | `alpha=30.0` | 164792.44 | 69.86 | 280920.93 | 113.28 | 0.54 | 0.95 |
| 9 |  | `alpha=50.0` | 164793.34 | 69.87 | 280952.82 | 113.23 | 0.52 | 0.87 |
| 10 |  | `alpha=100.0` | 164798.42 | 69.92 | 281026.87 | 113.13 | 0.54 | 0.90 |
| 11 |  | `alpha=300.0` | 164846.55 | 70.15 | 281272.20 | 112.88 | 0.52 | 0.90 |
| 12 |  | `alpha=1000.0` | 165152.02 | 70.96 | 281762.38 | 112.68 | 0.54 | 0.89 |

### HistGradientBoostingRegressor Candidate Results

| Rank | Selected | Parameters | Train RMSE | Train MAPE (%) | Valid RMSE | Valid MAPE (%) | Fit Seconds | Total Seconds |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 1 | yes | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 76356.30 | 16.00 | 177691.18 | 28.64 | 10.46 | 11.48 |
| 2 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 76584.00 | 15.11 | 179151.88 | 27.68 | 10.50 | 11.53 |
| 3 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 82731.29 | 16.25 | 179454.95 | 29.38 | 8.01 | 8.93 |
| 4 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 79637.98 | 15.86 | 179635.12 | 28.09 | 10.74 | 11.83 |
| 5 |  | `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 82030.94 | 17.29 | 180038.05 | 30.47 | 7.90 | 8.82 |
| 6 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 79529.96 | 15.99 | 180124.63 | 28.60 | 10.79 | 11.86 |
| 7 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 81416.52 | 16.46 | 180733.61 | 29.38 | 7.95 | 8.76 |
| 8 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 81256.41 | 15.90 | 181743.60 | 28.47 | 7.99 | 8.80 |
| 9 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 85423.82 | 17.45 | 182651.97 | 30.71 | 8.15 | 9.28 |
| 10 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 84516.16 | 16.66 | 183009.48 | 29.11 | 8.32 | 9.27 |
| 11 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 84459.45 | 16.87 | 183042.93 | 29.86 | 6.62 | 7.31 |
| 12 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 87361.59 | 17.36 | 183191.06 | 30.72 | 6.00 | 6.73 |
| 13 |  | `learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 85451.93 | 16.82 | 183217.14 | 30.38 | 7.86 | 8.82 |
| 14 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 84461.87 | 16.91 | 183625.11 | 29.89 | 8.25 | 9.13 |
| 15 |  | `learning_rate=0.1, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 86892.88 | 18.24 | 183941.73 | 31.47 | 6.08 | 6.81 |
| 16 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 84508.73 | 16.51 | 184529.29 | 29.66 | 6.60 | 7.33 |
| 17 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 86725.41 | 17.54 | 185487.14 | 30.93 | 11.13 | 12.37 |
| 18 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 86716.25 | 17.20 | 185640.61 | 30.36 | 11.18 | 12.41 |
| 19 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 90322.92 | 18.67 | 185839.95 | 32.53 | 5.04 | 5.67 |
| 20 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 87791.97 | 17.23 | 186003.66 | 29.96 | 6.94 | 7.77 |
| 21 |  | `learning_rate=0.1, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 89832.20 | 19.09 | 186555.71 | 32.33 | 5.03 | 5.67 |
| 22 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 87739.56 | 17.39 | 186702.25 | 30.54 | 6.81 | 7.56 |
| 23 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 90127.95 | 17.58 | 187167.65 | 31.39 | 6.16 | 6.96 |
| 24 |  | `learning_rate=0.08, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 90078.38 | 18.80 | 187222.95 | 32.52 | 6.12 | 6.89 |
| 25 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 92554.68 | 18.80 | 189868.29 | 32.34 | 8.33 | 9.42 |
| 26 |  | `learning_rate=0.05, max_iter=300, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 92580.47 | 18.95 | 189997.89 | 32.74 | 8.42 | 9.53 |
| 27 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 93303.66 | 19.85 | 190560.51 | 33.71 | 5.12 | 5.84 |
| 28 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 92052.14 | 19.38 | 190823.38 | 33.19 | 8.47 | 9.47 |
| 29 |  | `learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 93651.77 | 18.72 | 190970.66 | 32.80 | 5.10 | 5.76 |
| 30 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 92251.91 | 19.09 | 191160.86 | 32.99 | 8.40 | 9.40 |
| 31 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.01, min_samples_leaf=30` | 96054.23 | 21.15 | 195144.32 | 35.04 | 7.00 | 7.82 |
| 32 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 97834.19 | 21.09 | 195203.64 | 35.23 | 6.58 | 7.47 |
| 33 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=45, l2_regularization=0.1, min_samples_leaf=30` | 96169.41 | 20.93 | 195512.46 | 35.02 | 7.01 | 7.85 |
| 34 |  | `learning_rate=0.05, max_iter=220, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 97943.82 | 21.08 | 195551.69 | 35.53 | 6.22 | 7.08 |
| 35 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.01, min_samples_leaf=30` | 101858.53 | 23.52 | 200016.12 | 37.63 | 5.20 | 5.95 |
| 36 |  | `learning_rate=0.05, max_iter=180, max_leaf_nodes=31, l2_regularization=0.1, min_samples_leaf=30` | 101997.73 | 23.74 | 200330.02 | 38.29 | 5.13 | 5.85 |
