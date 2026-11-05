# Classification Modeling Report: Milestone 1 (Member B)

Task: predict `IsCanceled` for City Hotel (H2) bookings.
Companion to `reports/classification_data_report.md` (Member A), which covers data understanding, EDA and the column-level leakage audit.
All results below can be regenerated with the commands in the README.

## 1. Split strategy

The data comes from Member A's `clean_classification()`: 52,005 bookings with a 30.3% cancel rate, after removing exact duplicates and extreme rows. **The split is made after cleaning**, so duplicate rows cannot end up in different parts.

| Part | Share | Rows | Cancel rate | Used for |
|---|---|---|---|---|
| Train | 70% | about 36,403 | 30.3% | Fitting models. 5-fold CV and tuning run here only. |
| Validation | 20% | 10,401 | 30.3% | Comparing baseline vs tuned models, PCA/LDA variants and model families, and choosing the deployed model. |
| Test | 10% | about 5,201 | 30.3% | Touched once, at the end, for the final reported performance. **Not used in Milestone 1.** |

- The split is stratified on `IsCanceled`, so every part keeps the same cancel rate.
- The seed is 42 and is stored in `config.yaml`. The row ids of each part are saved in `outputs/classification/splits/split_indices.csv`, so the split can be reproduced exactly.
- Cross-validation is 5-fold `StratifiedKFold` (shuffled, seed 42) on the training part only.
- Tests in `tests/test_split.py` check the proportions, that the parts do not overlap, stratification, reproducibility, and that no leaky column is in the feature lists.

## 2. Preprocessing design

Every step lives in one scikit-learn `Pipeline`: `BookingFeatures` then the preprocessing `ColumnTransformer`, then an optional PCA/LDA slot (Milestone 2), then the model. Cross-validation therefore refits every learned step (imputer, scaler, encoder) on the training part of each fold only.

| Step | What it does | Why |
|---|---|---|
| Feature set | Member A's "strict" list (booking-time columns only) | Consistent with the prediction point: booking creation time |
| `has_company` | Replaces the `Company` ID with a 1/0 flag | 95.7% of bookings have no company and the other 200 IDs are rare, so the ID would overfit. The flag keeps the signal: 12.3% cancel rate with a company vs 31.2% without. |
| `total_nights`, `total_guests` | Sums of the stay-length and guest-count columns | Simple, fixed rules that summarize related columns |
| Previous history | `PreviousCancellations` and `PreviousBookingsNotCanceled` are kept | Known before the booking, and strongly related to the target: 71.2% cancel rate with an earlier cancellation vs 29.5% without; 6.6% with earlier completed stays vs 30.9% without |
| `log1p` | Applied to the two columns above in the scaled variant | They are extremely skewed (skew 24.4 for `PreviousCancellations`). Not needed for trees. |
| Median imputation | Numeric columns | Robust to skew. Member A's cleaning already filled the known gaps, so this is a safeguard. |
| One-hot encoding | Categorical columns, with categories under 1% of rows grouped together | `Agent` has 223 values. Grouping prevents hundreds of near-empty columns. Unseen categories are ignored at prediction time. |
| Scaling | `StandardScaler` in the scaled variant only | Needed by Logistic Regression, KNN, SVM, MLP, PCA and LDA, not by trees |

Two variants are used: **scaled** (linear and distance-based models) and **unscaled** (Decision Tree, Random Forest, XGBoost). `BookingFeatures` is stateless: it applies fixed row-level rules and learns nothing from the data, so it cannot leak.

## 3. Baseline results (untuned)

Two baselines with default settings: Logistic Regression (scaled) and Random Forest (unscaled), both on the strict feature set.

**Cross-validation on the training set** (5 folds, mean with standard deviation):

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.726 (0.003) | 0.730 (0.005) |
| Precision | 0.634 (0.016) | 0.575 (0.013) |
| Recall | 0.232 (0.004) | 0.422 (0.009) |
| F1 | 0.340 (0.004) | 0.487 (0.010) |
| ROC-AUC | 0.734 (0.008) | 0.752 (0.007) |
| PR-AUC | 0.555 (0.006) | 0.571 (0.008) |
| Training F1 | 0.342 | 0.986 |
| Fit time per fold | about 0.9 s | about 1.9 s |

**Validation set** (models fitted on the training part):

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.733 | 0.734 |
| Precision | 0.651 | 0.581 |
| Recall | 0.256 | 0.443 |
| F1 | 0.367 | 0.503 |
| ROC-AUC | 0.747 | 0.762 |
| PR-AUC | 0.574 | 0.594 |

Confusion matrices on the validation set (rows are actual, columns are predicted; not cancelled, cancelled). Figures: `outputs/classification/confusion_matrix_baseline_*.png`.

| | Logistic Regression | Random Forest |
|---|---|---|
| Actually not cancelled | 6,813 and 433 | 6,238 and 1,008 |
| Actually cancelled | 2,348 and 807 | 1,756 and 1,399 |

**Interpretation.**
- Accuracy is misleading: always predicting "not cancelled" already gives 69.7%. F1, recall and PR-AUC (random guessing scores 0.303) describe the models better.
- Logistic Regression **underfits**. Its training and CV F1 are both about 0.34, and it finds only 26% of the cancellations on validation. The model is too simple for this data.
- Random Forest finds more cancellations (recall 0.44 vs 0.26, F1 0.50 vs 0.37) at the cost of more false alarms. It **overfits**: training F1 is 0.986 against 0.487 in CV, because default trees grow until they memorize the training rows. Tuning tree depth and leaf size should narrow this gap.
- Validation and CV scores agree for both models, so the results are stable.
- ROC-AUC is about 0.75 for both, with no sign of a leaked column (a leak would push it far higher). The scores are modest partly because the strict feature set excludes strong but timing-dependent columns such as `DepositType` and `TotalOfSpecialRequests`.

## 4. Preliminary leakage notes (process level)

Column-level decisions are in Member A's audit. The controls below protect the modeling process.

- The split happens after cleaning and before any fitted step. Duplicates are removed first, so identical rows cannot be in both train and validation or test.
- Imputation, scaling, encoding and (later) PCA and LDA are fitted inside the pipeline on training data only.
- `BookingFeatures` is stateless, so it learns nothing from the data.
- All new features use information that exists at booking time: the company flag, stay length, guest count and previous history.
- The test set has not been used for any decision.
- **Limitation:** the checks behind `has_company` and the previous-history columns were run on all cleaned rows, including rows that later form validation and test. They were descriptive, and the differences are large (for example 12.3% vs 31.2%), so the risk is small. Final scores must not be used to revise these choices.

## 5. Plan for Milestone 2

- **More models:** add XGBoost (boosting), plus Gaussian Naive Bayes and a Decision Tree or MLP. Every model gets a baseline and a tuned result.
- **Tuning:** randomized search or Optuna with stratified 5-fold CV on the training set, optimizing F1 or PR-AUC (to be fixed and documented), with the search spaces and best parameters saved.
- **Low recall:** test class weights and the decision threshold, since both baselines miss many cancellations.
- **PCA and LDA:** PCA after encoding and scaling with an explained-variance plot, at least two models with and without PCA, and original vs PCA vs LDA features for one classifier. LDA on two classes gives one component.
- **Strict vs relaxed features:** report the score gap as evidence for the leakage section.
- **Model selection:** a criterion stated in advance, applied on CV and validation results only. The test set is used once, for the final evaluation.
- **Application:** save the fitted pipeline, add the `predict` command and the classification part of the final interface, with input validation.

## 6. Member contributions (classification)

| Member | Contributions |
|---|---|
| Member A | Data loading and verification, cleaning rules, EDA tables and plots, leakage audit, `config.yaml` feature lists, data report, the `BookingFeatures` feature builder (`src/features/booking_features.py`) and the classification preprocessing (`src/Preprocessing/Classification.py`), with their tests (`tests/test_booking_features.py`, `tests/test_preprocessing.py`) |
| Member B (GitHub: karim-tamer-0) | Stratified 70/20/10 split with saved indices and CV folds, full pipeline builder, baseline CV and validation evaluation with confusion matrices, command-line interface (`baseline`, `evaluate`), the corresponding tests, README and requirements, this report |
