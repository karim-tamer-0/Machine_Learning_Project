# Milestone 1: Task A – Regression Data Preparation & Leakage Audit Report

**Team 29 | CSE382: Introduction to Machine Learning**
**Target Variable:** Average Daily Rate (`ADR`)
**Dataset:** City Hotel (`H2.csv` - 79,330 initial bookings)

## 1. Executive Summary

This report documents the data understanding, exploratory analysis, leakage prevention audit, row-level cleaning, and feature engineering pipeline implemented for **Task A (Linear Regression / ADR Prediction)**. All preprocessing steps adhere strictly to course requirements, ensuring that data-dependent operations (such as scaling, encoding, and frequency grouping) are fitted **strictly on the training partition** to prevent any form of indirect data leakage.

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

7. **`Agent` & `Company`**: High-cardinality ID columns containing literal `"NULL"` strings (\~94% missing for company). Replaced by robust binary indicator flags (`HasAgent`, `HasCompany`) rather than arbitrary numerical imputation.

## 3. Data Understanding & Zero-Inflation Analysis

An inspection of feature sparsity across the 79,330 records revealed the following zero-inflation distribution:

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

3. **Duplicate Removal:** Dropped exact duplicate rows to prevent identical tour/group bookings from leaking across Train and Test splits.

* **Initial Records:** 78,092 rows (after upstream cleaning)

* **Final Cleaned Rows:** **50,787 rows**

## 5. Feature Engineering & Preprocessing Pipeline

* **Temporal Cyclicality:** Mapped `ArrivalDateMonth` from text to integer (1–12).

* **High-Cardinality Categorical Handling (`Country`):** Grouped countries outside the Top 10 most frequent categories into an `"Other"` bucket to prevent sparse dimensionality expansion.

* **Multicollinearity Prevention:** Applied `OneHotEncoder(drop='first', handle_unknown='ignore')` to all categorical features. Dropping the first dummy column prevents the **Dummy Variable Trap** (perfect multicollinearity) in Linear Regression.

* **Numerical Scaling:** Fitted `StandardScaler` strictly on `X_train` ($\mu = 0, \sigma = 1$) to ensure regularization penalties (Ridge/Lasso) operate on comparable feature scales.

## 6. Train / Validation / Test Split Strategy

To support robust model selection and hyperparameter tuning without contaminating the test evaluation, data was partitioned into:

* **Training Set (70%):** $35,548$ rows (used to fit encoders, scalers, and regression weights).

* **Validation Set (15%):** $7,620$ rows (used for hyperparameter tuning).

* **Test Set (15%):** $7,619$ rows (**Locked away** until final model evaluation).

All preprocessed split matrices are saved under `src/data/regression_data/`:

* `train_regression.csv` ($35,548 \times 53$)

* `val_regression.csv` ($7,620 \times 53$)

* `test_regression.csv` ($7,619 \times 53$)

* `regression_cleaned.csv` ($50,787 \times 25$)

## 7. Plan for Milestone 2

1. Train and evaluate **at least three regression models** (Linear Regression baseline, Ridge/Lasso regularized variants, and Random Forest / XGBoost Regressor).

2. Perform systematic hyperparameter tuning using cross-validation on the validation set.

3. Conduct PCA dimensionality reduction experiments comparing model performance with and without principal components.

4. Integrate the final trained regression pipeline into the application inference artifacts (`artifacts/regression_preprocessor.joblib`).