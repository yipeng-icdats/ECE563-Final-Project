# ECE 563 Final Project — Structured Summary

**Course:** ECE 563 — AI in Smart Grid  
**Semester:** Spring 2026  
**Deadline:** May 8, 2026 @ 2:00 PM

---

# 1. Project Goal

The project focuses on electric load forecasting using:

* Historical hourly load data
* Temperature data from weather stations

You must:

1. Analyze relationships between:
    * 20 load zones
    * 11 temperature stations
2. Build two machine learning regression models to predict electrical load.
3. Use the best model to predict:
    * Hourly load values for June 1–7, 2008
    * For all 20 load zones

---

# 2. Core Tasks You Must Complete

## Task A — Data Understanding & Exploration

You must investigate:

* Patterns in hourly load data
* Correlation between:
    * load zones
    * temperature stations

### Main Objective

For each load zone:

* determine which temperature station best predicts load

If no obvious correlation exists:

* explain your methodology for selecting temperature stations

---

# 3. Required Workflow

---

## Step 1 — Data Preprocessing

You may perform:

* Data cleaning
* Outlier removal
* Feature engineering
* Data shuffling
* Dataset subset selection

### Important

You must explain:

* what preprocessing was used
* why it was used
* tradeoffs considered

---

## Step 2 — Temperature Station ↔ Load Zone Mapping

For each of the 20 load zones:

* assign one temperature station

### You Must Provide

A mapping table

Example:

| Load Zone | Temperature Station |
|---|---|
| Zone 1 | Station 4 |
| Zone 2 | Station 8 |

### Methodology explanation

Explain:

* correlation analysis
* feature importance
* ML-based selection
* geographical reasoning
* or any other strategy

---

## Step 3 — Dataset Splitting

You must split data into:

* Training set
* Validation set (optional but recommended)
* Test set

### Important Requirements

You must document:

* exact split strategy
* percentages
* `random_state` values

So the instructor can reproduce your results exactly.

---

## Step 4 — Build Two ML Models

You must implement:

---

### Model 1 — Simple/Fast Model

**Purpose:**

* understand forecasting problem
* establish baseline
* fast experimentation

**Examples:**

* Linear Regression
* Ridge Regression
* Decision Tree
* Random Forest (small)

---

### Model 2 — More Complex Model

**Purpose:**

* improve performance
* achieve better accuracy

**Examples:**

* `MLPRegressor`
* XGBoost
* Gradient Boosting
* Ensemble methods

---

## Step 5 — Hyperparameter Tuning

You must tune both models.

### Requirements

* Identify important hyperparameters
* Run validation experiments
* Select best hyperparameter values

### Important

Final submitted scripts:

* MUST already contain optimal hyperparameters
* MUST NOT perform expensive tuning during grading runtime

---

## Step 6 — Model Evaluation

You must evaluate:

---

### Accuracy Metrics

For:

* training set
* validation set
* test set

Possible metrics:

* `score()`
* RMSE
* MAE
* R²

---

### Runtime Measurements

Measure:

#### Validation/Tuning Time

How long tuning took overall.

#### Training Time

How long final training took.

#### Prediction Time

How long inference took.

#### June 2008 Prediction Time

Time required to generate final predictions.

---

## Step 7 — Error Analysis

You must provide:

### Top 10 Worst Prediction Errors

Required columns:

| zone | year | month | day | hour | predicted load | true load | relative % error |
|---|---|---|---|---|---|---|---|

Relative percentage error:

$$
100\times\frac{Y_{true}-Y_{pred}}{Y_{true}}
$$

Sort by:

$$
\left|100\times\frac{Y_{true}-Y_{pred}}{Y_{true}}\right|
$$

Descending order (worst first).

---

## Step 8 — Compare Both Models

You must compare:

| Aspect | Compare |
|---|---|
| Accuracy | Which predicts better |
| CPU Time | Faster/slower |
| Training Time | Efficiency |
| Prediction Time | Inference speed |
| Error Patterns | Similar mistakes? |
| Complexity | Simplicity vs performance |
| Advantages | Pros |
| Disadvantages | Cons |

---

## Step 9 — Final June 2008 Prediction

Choose the best model.

Use it to predict:

* all hourly loads
* all 20 zones
* June 1–7, 2008

Output:

* `Load_prediction.csv`

Must follow provided format exactly.

---

# 4. Critical Runtime Constraints

## STRICT LIMIT

Each model script must finish within:

**5 minutes**

INCLUDING:

* loading data
* preprocessing
* training
* prediction
* generating outputs

BUT EXCLUDING:

* separate validation/tuning script

---

## Important Grading Risk

If your final scripts exceed the runtime limit:

**You receive ZERO credit for model implementation.**

---

# 5. Required Python Scripts

You MUST submit exactly these files.

---

## Script 1 — `map_tune.py`

**No time limit**

### Purpose:

* load datasets
* determine zone ↔ station mapping
* tune hyperparameters

### Required Outputs

* mapping
* best hyperparameters

---

## Script 2 — `other_model.py`

**Must finish within 5 minutes**

### Purpose:

* load data
* preprocess
* load mapping
* train simple model
* evaluate model

### Must Display

* training performance
* testing performance

---

## Script 3 — `best_model.py`

**Must finish within 5 minutes**

### Purpose:

* load data
* preprocess
* load mapping
* train best model
* evaluate model
* predict June 2008 loads

### Must Generate

* `Load_prediction.csv`

---

# 6. Required Report Contents

Your report must include:

---

## Section 1 — Project Overview

Brief project summary.

---

## Section 2 — Data Processing

Explain:

* cleaning
* preprocessing
* feature engineering
* outlier handling

---

## Section 3 — Mapping Strategy

Explain:

* how stations were assigned to zones
* why the mapping makes sense

Include mapping table.

---

## Section 4 — Dataset Splits

Explain:

* train/validation/test split
* `random_state` values

---

## Section 5 — Model 1

Include:

* algorithm choice
* hyperparameters
* tuning process
* results
* observations
* strengths/weaknesses

---

## Section 6 — Model 2

Same analysis as Model 1.

---

## Section 7 — Comparison Table

Direct comparison between both models.

---

## Section 8 — Top 10 Errors

Required error table.

---

## Section 9 — Final Prediction Discussion

Explain:

* why final model was selected
* possible future improvements

---

## Section 10 — Conclusion

Brief project conclusion.

---

# 7. Appendix Requirements

You must include:

* ALL Python code
* fully commented
* pasted as text
* NOT screenshots

Each script:

* starts on its own appendix page

Use:

* small font

---

# 8. Submission Requirements

You must submit:

---

## A. Report File

### Format:

* PDF or Word

### Requirements:

* under 1 MB
* code pasted into appendix

---

## B. ZIP Archive

Contains:

* all Python scripts
* generated CSV output

### Do NOT include:

* provided dataset CSV files

---

# 9. Required File Paths

Your code MUST use:

```python
pd.read_csv("../Load_history_final.csv")
pd.read_csv("../Temp_history_final.csv")