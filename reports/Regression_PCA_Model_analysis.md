Task A (Linear Regression / ADR Prediction) — Work Documentation 
**Course:** CSE382 Introduction to Machine Learning  
**Team:** 29  
**Target Variable:** Average Daily Rate (`ADR`)  
**Dataset:** City Hotel (`H2.csv` — 79,330 raw records)  
**Context Date:** November 2026

\---

## 1\. Summary of Contributions

As part of the Task A (Regression) subteam, the following engineering, cleaning, evaluation, and pipeline integration tasks were performed:

1. **Dataset Versioning \& Cleaning Script (`src/Preprocessing/Regression2.py`):**

   * Identified and fixed two data-structure issues present in the preliminary data pipeline:

     * **Constant Flag Columns:** In raw data, `Agent` and `Company` contained 11-character padded strings (e.g., `"       NULL"`). String replacement failed without prior stripping, causing `HasAgent` and `HasCompany` to become constant `1` flags (which normalized to constant `0` with zero variance).
     * **Trailing Whitespace in Category Headers:** Category labels contained trailing spaces (e.g., `Meal\_FB       `), creating messy column names.
   * Created `Regression2.py` to generate clean **Version 2 data splits** under `src/data/regression\_data/version 2/`.
   * Dropped the zero-variance `HasAgent` and `HasCompany` flags.
   * Stripped trailing whitespace across all column names.
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

   * Authored 14 automated tests covering data shapes, CLI routing, metric calculations, and strict test-set isolation. All 46 repository tests pass.

\---

## 2\. Model Performance Metrics (How Good is the Model?)

In regression tasks, **Accuracy** (which measures discrete correct/incorrect classifications) is not applicable. Instead, model performance is evaluated using continuous distance metrics:

* **$R^2$ (Coefficient of Determination):** Measures how much variance in ADR is explained by the model (closer to $1.0$ is better).
* **MAE (Mean Absolute Error):** Average prediction error in dollars ($$$).
* **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.

### Baseline OLS Linear Regression (Version 2 Data)

|Evaluation Stage|MAE ($$$)|MSE|RMSE ($$$)|$R^2$|
|-|:-:|:-:|:-:|:-:|
|**5-Fold Cross-Validation (Train Set)**|$18.9258 \\pm 0.2093$|—|$25.1632 \\pm 0.4419$|$0.5932 \\pm 0.0090$|
|**Validation Set Evaluation**|**18.8522**|**649.7113**|**25.4894**|**0.5990**|

* **Model Quality Assessment:** On average, the baseline model predicts room rates within **$\\approx $18.85$** of the true price, achieving an **$R^2$ of $0.5990$** ($59.9%$ variance explained).

### PCA Feature Compression vs. Original Features (Validation Set)

|Representation|Components ($k$)|Retained Variance|Val MAE ($$$)|Val RMSE ($$$)|Val $R^2$|
|-|:-:|:-:|:-:|:-:|:-:|
|**Full Original Features**|**50**|**100.0%**|**18.8522**|**25.4894**|**0.5990**|
|**PCA 99% Variance**|43|99.0%|18.8516|25.5645|0.5967|
|**PCA 95% Variance**|39|96.0%|20.1619|27.0228|0.5493|
|**PCA 90% Variance**|35|91.2%|21.1551|27.9556|0.5177|
|**PCA 2 Components**|2|13.4%|28.8772|37.8370|0.1165|

* **PCA Takeaway:** Full original features yield the highest $R^2$ ($0.5990$). 2D PCA retains only $13.4%$ variance and is suitable for scatter plot visualizations rather than primary predictive modeling.

\---

## 3\. How to Run and Check the Code

Follow these commands to execute the cleaning scripts, run tests, and inspect model outputs.

### Step 1: Generate Version 2 Cleaned Regression Splits

```bash
python -m src.Preprocessing.Regression2
```

* **What it does:** Reads `src/data/regression\_data/`, drops constant flags (`HasAgent`, `HasCompany`), strips whitespace from category names, and saves the cleaned matrices to `src/data/regression\_data/version 2/`.

### Step 2: Run the Automated Test Suite

```bash
python -m pytest -q
```

* **What it does:** Verifies all 46 test cases across the repository, ensuring zero test-set access, correct input shapes, and proper pipeline behavior.

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

