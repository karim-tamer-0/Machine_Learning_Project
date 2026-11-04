# Milestone 1: Task A – Regression Data Preparation & Leakage Audit Report (Version 3)

**Team 29 | CSE382: Introduction to Machine Learning**
**Target Variable:** Average Daily Rate (`ADR`)
**Dataset:** City Hotel (`H2.csv` - 79,330 initial bookings)

## 1. Executive Summary

This report documents the data understanding, exploratory analysis, leakage prevention audit, row-level cleaning, and feature engineering pipeline implemented for **Task A (Linear Regression / ADR Prediction)** using the active **Version 3** data. Version 3 corrects the duplicate-row leakage present in Versions 1 and 2 by normalizing categorical text and applying the fixed country grouping **before** duplicate removal and splitting. Data-dependent transformations (`StandardScaler` and one-hot encoding) are fitted **strictly on the training partition**; the country grouping is a fixed, pre-declared rule rather than a statistic learned from the data.

## 2. Prediction Point & Feature Eligibility (Leakage Audit)

To prevent target and lifecycle leakage, we established the exact **prediction point**: **at the initial booking creation in the hotel Property Management System (PMS)**.
Any variable recorded *after* the initial request, or reflecting final stay outcomes, was systematically removed.

### Dropped Post-Booking Leakage & ID Variables:

1. **`IsCanceled`**: Target variable for Task B; cancellation status is finalized after booking.

2. **`ReservationStatus`**: Records the final disposition (`Check-Out`, `Canceled`, `No-Show`) at the conclusion of the stay.

3. **`ReservationStatusDate`**: Timestamp of the final reservation status update (occurs in the future relative to booking).

4. **`AssignedRoomType`**: Room type ultimately assigned at front desk check-in (often subject to operational upgrades; distinct from `ReservedRoomType`).

5. **`BookingChanges`**: Cumulative count of modifications made by the customer between booking creation and check-in/cancellation.

6. **`DaysInWaitingList`**: Duration the booking remained on the waiting list before confirmation.

7. **`Agent` & `Company`**: High-cardinality ID columns containing padded literal `"NULL"` strings (\~94% missing for `Company`). The preliminary `HasAgent`/`HasCompany` indicators became zero-variance because the padded strings prevented correct `NULL` detection; they were dropped in Version 2, and Version 3 excludes the raw ID columns entirely.

## 3. Data Understanding & Zero-Inflation Analysis

A raw-data inspection across the 79,330 records revealed the following zero-inflation distribution; it is descriptive only and does not determine the final Version 3 feature matrix:

| Feature | Zero Count (%) | Domain Interpretation & Handling Strategy | 
| ----- | ----- | ----- | 
| **`Babies`** | 99.55% | Rare family demographic; retained as a count feature. | 
| **`PreviousBookingsNotCanceled`** | 98.25% | Indicates first-time or returning guest history; preserved. | 
| **`IsRepeatedGuest`** | 79.91% | Binary flag for guest loyalty; preserved. | 
| **`RequiredCarParkingSpaces`** | 97.60% | Rare parking request; preserved as a behavioral indicator. | 
| **`DaysInWaitingList`** | 95.61% | Highly sparse waiting list duration; dropped via leakage audit. | 
| **`Children`** | 93.53% | Missing/zero values reliably indicate no children; imputed with `0`. | 
| **`PreviousCancellations`** | 93.21% | Critical behavioral risk signal; preserved. | 
| **`BookingChanges`** | 87.31% | Post-booking amendment count; dropped due to lifecycle leakage. | 
| **`TotalOfSpecialRequests`** | 60.51% | Customer engagement indicator; preserved. | 

> **Justification for Preserving Sparse Features:** While features like `PreviousCancellations` and `Babies` exhibit >93% zeros, they represent rare behavioral conditions rather than data collection artifacts. Dropping them would discard extreme-value signal. Instead of arbitrary threshold dropping, they were preserved and scaled safely.

## 4. Row-Level Cleaning Workflow

1. **Target Filtering (`ADR > 0`):** Removed complimentary rooms and data entry errors where `ADR <= 0`.

2. **Outlier Removal (`ADR <= 1000`):** Filtered out the single corrupted extreme outlier in `H2.csv` (`ADR = 5400`). Because Ordinary Least Squares (Linear Regression) minimizes squared error ($\text{MSE}$), retaining this single record would severely distort slope coefficients.

3. **Duplicate Correction (Version 3):** Exact duplicate rows are removed **after** trailing whitespace is stripped and countries outside the fixed Top 10 are grouped into `Other`, and **before** the train/validation/test split. This order prevents two originally different bookings from becoming identical after encoding and then leaking across splits. Version 3 removes **236** post-grouping duplicate rows.

Versions 1 and 2 removed duplicates before country grouping, so grouping could create new identical rows after the split. Version 2 inherited those rows because it only removed the constant flags and renamed columns from the existing Version 1 splits. Version 3 rebuilds the split from the cleaned Version 2 table with the corrected order.

**Row-count trail:**

* **Raw `H2.csv`:** 79,330 rows.
* **Shared upstream-cleaned input:** 78,092 rows.
* **Version 2 cleaned table:** 50,787 rows (preserved unchanged).
* **Version 3 cleaned table:** **50,551 rows** (`regression_cleaned.csv`, 23 columns).
* **Version 3 splits:** 35,384 train / 7,584 validation / 7,583 test rows, each with 51 columns.

## 5. Feature Engineering & Preprocessing Pipeline

* **Text Normalization:** Column names and all categorical values (`Meal`, `Country`, `MarketSegment`, `DistributionChannel`, `ReservedRoomType`, `DepositType`, `CustomerType`) are stripped of surrounding whitespace before grouping and encoding. This prevents categories such as `Meal_FB` and `Meal_FB       ` from becoming separate one-hot columns.

* **Temporal Cyclicality:** Mapped `ArrivalDateMonth` from text to integer (1–12).

* **High-Cardinality Categorical Handling (`Country`):** Before duplicate removal and splitting, countries outside a fixed Top 10 list (`PRT`, `FRA`, `GBR`, `DEU`, `ESP`, `ITA`, `BEL`, `BRA`, `NLD`, `USA`) are grouped into `Other` to prevent sparse dimensionality expansion. The list is fixed in `Regression3.py`; it is not recomputed per split.

* **Multicollinearity Prevention:** Applied `OneHotEncoder(drop='first', handle_unknown='ignore')` to all categorical features. Dropping the first dummy column prevents the **Dummy Variable Trap** (perfect multicollinearity) in Linear Regression.

* **Numerical Scaling:** Fitted `StandardScaler` strictly on `X_train` ($\mu = 0, \sigma = 1$) to ensure regularization penalties (Ridge/Lasso) operate on comparable feature scales.

## 6. Train / Validation / Test Split Strategy

To support robust model selection and hyperparameter tuning without contaminating the test evaluation, the Version 3 cleaned table was partitioned with `random_state=42` into:

* **Training Set (70%):** **35,384** rows (used to fit the encoder, scaler, and regression weights).

* **Validation Set (15%):** **7,584** rows (used for model selection and evaluation).

* **Test Set (15%):** **7,583** rows (**Locked away** until final model evaluation).

The split is constructed only after Version 3 duplicate removal. `Regression3.py` verifies zero duplicate rows within each split and zero overlap across train/validation/test.

All preprocessed split matrices are saved under `src/data/regression_data/version 3/`:

* `train_regression.csv` ($35,384 \times 51$)

* `val_regression.csv` ($7,584 \times 51$)

* `test_regression.csv` ($7,583 \times 51$)

* `regression_cleaned.csv` ($50,551 \times 23$)

Version 2 files remain under `src/data/regression_data/version 2/` for reproducibility and are not used by the active baseline.

## 7. Plan for Milestone 2

1. Train and evaluate **at least three regression models** (Linear Regression baseline, Ridge/Lasso regularized variants, and Random Forest / XGBoost Regressor).

2. Perform systematic hyperparameter tuning with cross-validation on the training set, using the validation set only for model selection; keep the test set locked until final evaluation.

3. Conduct PCA dimensionality reduction experiments comparing model performance with and without principal components.

4. Integrate the final trained regression pipeline into the application inference artifacts (`artifacts/regression_preprocessor.joblib`).