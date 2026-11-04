Task A (Linear Regression / ADR Prediction) — Work Documentation 
**Course:** CSE382 Introduction to Machine Learning  
**Team:** 29  
**Target Variable:** Average Daily Rate (`ADR`)  
**Dataset:** City Hotel (`H2.csv` — 79,330 raw records)  
**Context Date:** November 2026

\---

## 1\. Summary of Contributions

As part of the Task A (Regression) subteam, the following engineering, cleaning, evaluation, and pipeline integration tasks were performed:

1. **Dataset Versioning \& Cleaning Scripts (`src/Preprocessing/Regression2.py`, `src/Preprocessing/Regression3.py`):**

   * Identified and fixed two data-structure issues present in the preliminary data pipeline:

     * **Constant Flag Columns:** In raw data, `Agent` and `Company` contained 11-character padded strings (e.g., `"       NULL"`). String replacement failed without prior stripping, causing `HasAgent` and `HasCompany` to become constant `1` flags (which normalized to constant `0` with zero variance).
     * **Trailing Whitespace in Category Headers:** Category labels contained trailing spaces (e.g., `Meal\_FB       `), creating messy column names.
   * Created `Regression2.py` to generate **Version 2 data splits** under `src/data/regression\_data/version 2/`.
   * Dropped the zero-variance `HasAgent` and `HasCompany` flags.
   * Stripped trailing whitespace across all column names.
   * Preserved Version 2 unchanged and created **Version 3** under `src/data/regression\_data/version 3/`.
   * Version 3 strips trailing whitespace from categorical values, then applies the fixed top-10 country grouping before duplicate removal and splitting. This removes exact duplicate rows created when different countries are grouped into `Other`, then creates new train/validation/test splits with clean one-hot feature names.
   * Version 3 removes 236 duplicate rows after country grouping and verifies zero duplicate rows within and across all three splits.
   * Preserved raw original CSVs and left the final test set **unopened and untouched**.
2. **Feature Specificity Analysis (`RequiredCarParkingSpaces`):**

   * Conducted an ablation study on `RequiredCarParkingSpaces` (48,944 zero rows vs. 1,843 non-zero rows).
   * Verified that `RequiredCarParkingSpaces` provides a small, legitimate validation performance gain ($R^2 +0.0017$) without introducing target leakage at booking time. Retained the feature for the baseline.
3. **Regression Baseline Pipeline (`src/regression/baseline.py`):**

   * Built the modular training and evaluation module for Ordinary Least Squares (OLS) Linear Regression.
   * Integrated 5-fold cross-validation on the training split only.
   * Generated validation prediction files, coefficient weights, and residual error diagnostic plots under `outputs/regression/`.
4. **CLI Integration (`main.py`):**

   * Updated `main.py` to route `--task regression` commands directly to the regression pipeline.
5. **PCA Dimensionality Reduction \& Analysis (`src/regression/pca\_analysis.py`):**

   * Implemented PCA fitted strictly on scaled training features.
   * Evaluated OLS Linear Regression performance across uncompressed features, 99%, 95%, 90%, 85%, 80% variance, and 2D/3D component projections.
6. **Unit \& Integration Test Suite (`tests/test\_regression\_baseline.py` \& `tests/test\_regression\_pca.py`):**

   * Added automated tests covering data shapes, CLI routing, metric calculations, Version 3 split uniqueness, clean categorical text, and strict test-set isolation. All 56 repository tests pass.

\---

## 2\. Model Performance Metrics (How Good is the Model?)

In regression tasks, **Accuracy** (which measures discrete correct/incorrect classifications) is not applicable. Instead, model performance is evaluated using continuous distance metrics:

* **$R^2$ (Coefficient of Determination):** Measures how much variance in ADR is explained by the model (closer to $1.0$ is better).
* **MAE (Mean Absolute Error):** Average prediction error in dollars ($$$).
* **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.

### Baseline OLS Linear Regression (Version 3 Data)

|Evaluation Stage|MAE ($$$)|MSE|RMSE ($$$)|$R^2$|
|-|:-:|:-:|:-:|:-:|
|**5-Fold Cross-Validation (Train Set)**|$18.9174 \\pm 0.1120$|—|$25.1875 \\pm 0.1733$|$0.5940 \\pm 0.0037$|
|**Validation Set Evaluation**|**19.1080**|**651.2322**|**25.5193**|**0.5986**|

* **Model Quality Assessment:** On average, the baseline model predicts room rates within **$\\approx $19.11$** of the true price, achieving an **$R^2$ of $0.5986$** ($59.86%$ variance explained).

### PCA Feature Compression vs. Original Features (Validation Set)

|Representation|Components ($k$)|Retained Variance|Val MAE ($$$)|Val RMSE ($$$)|Val $R^2$|
|-|:-:|:-:|:-:|:-:|:-:|
|**Full Original Features**|**50**|**100.0%**|**19.1080**|**25.5193**|**0.5986**|
|**PCA 99% Variance**|43|99.0%|19.1425|25.6033|0.5960|
|**PCA 95% Variance**|39|95.7%|20.2011|26.9518|0.5523|
|**PCA 90% Variance**|35|90.9%|21.1619|27.9092|0.5199|
|**PCA 85% Variance**|32|86.0%|21.5763|28.2953|0.5066|
|**PCA 80% Variance**|29|80.5%|21.9948|28.6502|0.4941|
|**PCA 3 Components**|3|18.1%|27.7945|36.4517|0.1811|
|**PCA 2 Components**|2|13.3%|28.6811|37.7180|0.1232|

* **PCA Takeaway:** Full original features yield the highest $R^2$ ($0.5986$). 2D PCA retains only $13.3%$ variance and is suitable for scatter plot visualizations rather than primary predictive modeling.

### Duplicate-Row Correction (Version 3)

Version 1 and Version 2 had the same duplicate counts:

|Check|Version 1|Version 2|
|-|:-:|:-:|
|Duplicate rows inside train|146|146|
|Duplicate rows inside validation|8|8|
|Duplicate rows inside test|5|5|
|Distinct rows shared by train and validation|38|38|
|Distinct rows shared by train and test|35|35|
|Distinct rows shared by validation and test|15|15|

The cause was the preprocessing order: exact duplicate rows were removed before splitting, then countries outside the top 10 were grouped into `Other` after splitting. Different bookings from different countries could therefore become identical after encoding. Version 2 inherited these rows because it only removed flag columns and renamed columns from the existing Version 1 splits.

Version 3 fixes the order:

1. Start from the Version 2 cleaned table.
2. Strip trailing whitespace from categorical values and column names.
3. Apply the fixed top-10 country grouping before splitting.
4. Remove exact duplicate rows after grouping: 236 rows removed.
5. Create new 70%/15%/15% splits with seed 42.
6. Verify zero duplicate rows within each split and zero overlap between splits.

Final Version 3 sizes:

|File|Rows|Columns|
|-|:-:|:-:|
|`regression_cleaned.csv`|50,551|23|
|`train_regression.csv`|35,384|51|
|`val_regression.csv`|7,584|51|
|`test_regression.csv`|7,583|51|

The test split remains unused by training, cross-validation, validation evaluation, and PCA.

\---

## 3\. How to Run and Check the Code

Follow these commands to execute the cleaning scripts, run tests, and inspect model outputs.

### Step 1: Generate Version 3 Cleaned Regression Splits

```bash
python -m src.Preprocessing.Regression3
```

* **What it does:** Reads `src/data/regression\_data/version 2/regression_cleaned.csv`, strips trailing whitespace from categorical values and column names, applies the fixed top-10 country grouping, removes duplicate rows after grouping, fits the encoder and scaler on the new training split only, and saves the duplicate-free matrices to `src/data/regression\_data/version 3/`.
* **Version history:** `src/data/regression\_data/version 2/` remains unchanged. `Regression2.py` is retained only to reproduce that earlier version.

### Step 2: Run the Automated Test Suite

```bash
python -m pytest -q
```

* **What it does:** Runs the full repository test suite, including test-set isolation, Version 3 split uniqueness, input shapes, and pipeline behavior.

### Step 3: Run Baseline Cross-Validation \& Validation Evaluation

```bash
# Run 5-fold CV on Training Set
python main.py baseline --task regression --model linear\_regression

# Run Validation Evaluation
python main.py evaluate --task regression --model linear\_regression
```

* **Where to check outputs:**

  * `outputs/regression/baseline\_cv\_metrics.csv`
  * `outputs/regression/baseline\_validation\_metrics.csv`
  * `outputs/regression/linear\_regression\_coefficients.csv`
  * `outputs/regression/linear\_regression\_predictions\_val.csv`
  * `outputs/regression/linear\_regression\_residuals.png`

### Step 4: Run PCA Dimensionality Reduction Experiments

```bash
python main.py pca --task regression
```

* **Where to check outputs:**

  * `outputs/regression/pca/pca\_model\_comparison.csv`
  * `outputs/regression/pca/pca\_scree.png`
  * `outputs/regression/pca/pca\_2d\_adr.png`

