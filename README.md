# Machine-Learning-Project
Team 28's Machine Learning Project

## Classification data preparation

```bash
pip install -r requirements.txt
python -m src.data.cleaning   # print the rows removed by each cleaning rule
python -m src.data.eda        # write tables and plots to outputs/classification/
```

```python
from src.data.loader import load_all_models, get_features
from src.data.cleaning import clean_classification

df, report = clean_classification(load_all_models())  # run before the train/test split
numeric, categorical = get_features("strict")         # or "relaxed"
```

## Project links

- GitHub repository: https://github.com/karim-tamer-0/Machine_Learning_Project
- Submitted commit SHA (Milestone 1): `TO BE FILLED AT SUBMISSION`###########!!PLZ DONT FORGET ME!!###########

## Environment setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Tested with Python 3.13 and scikit-learn 1.2 or newer. Run every command from the repository root.

## Classification (IsCanceled): baselines, Milestone 1

- Prediction point: booking creation time. Features: the "strict" list in `config.yaml`.
- Data: the cleaned shared dataset, split 70% train / 20% validation / 10% test (stratified, seed 42). The test set is not used in Milestone 1.

```bash
python -m src.data.split                                    # check split sizes and save the split indices
python main.py baseline --task classification --model all   # untuned baselines, 5-fold CV on the training set
python main.py evaluate --task classification --model all   # validation metrics and confusion matrices
python -m pytest -q                                         # run all tests
```

Results are written to `outputs/classification/`: `baseline_cv_metrics.csv`, `baseline_validation_metrics.csv`, `confusion_matrix_baseline_*.png` and `splits/split_indices.csv`.

## Regression (Task A, ADR): baseline, Milestone 1

- Target: `ADR` (average daily rate). Prediction point: booking creation time, so post-booking columns (IsCanceled, ReservationStatus*, AssignedRoomType, BookingChanges, DaysInWaitingList) and the raw Agent/Company IDs are dropped.
- Cleaning: rows with `ADR <= 0` and the single `ADR = 5400` outlier are removed. Countries outside a fixed top 10 are grouped into `Other` **before** duplicates are removed and the data is split (Version 3, 236 extra duplicates removed). Versions 1 and 2 grouped countries after the split, which leaked identical rows across train/validation/test.
- Data: 50,551 cleaned rows, split 70% / 15% / 15% (seed 42) into 35,384 train / 7,584 validation / 7,583 test. The scaler and one-hot encoder are fitted on the training split only, and the test set is not used in Milestone 1.
- Baseline: untuned OLS Linear Regression. 5-fold CV on train gives R² 0.594 ± 0.004. On validation: MAE 19.11, RMSE 25.52, R² 0.599.
- PCA: the full 50 features score best. Keeping 99% of the variance (43 components) is almost identical (R² 0.596), but accuracy drops quickly below that (R² 0.552 at 95%, 0.494 at 80%). Two components (13% of the variance) are only good for plotting (R² 0.123).
- Reports: `reports/regression_data_preparation_report.md` and `reports/Regression_PCA_Model_analysis.md`.

```bash
python -m src.Preprocessing.Regression3                     # rebuild the Version 3 cleaned data and splits
python main.py baseline --task regression --model linear_regression   # 5-fold CV on the training set
python main.py evaluate --task regression --model linear_regression   # validation metrics and residual plot
python main.py pca --task regression                        # PCA experiments
python -m pytest -q tests/test_regression_baseline.py tests/test_regression_pca.py
```

Results go to `outputs/regression/` (and `outputs/regression/pca/`). Split data is in `src/data/regression_data/version 3/`; Version 2 is kept only for reproducibility.

## Clustering (Task C): baseline, Milestone 1

- Analysis point: booking creation time. Input: `outputs/Clustering/kmeans_cleaned_scaled.csv` (made by `src/Preprocessing/Clustering/clusteringprocessing.py`). Outcome columns (IsCanceled, ReservationStatus*, ADR, AssignedRoomType, BookingChanges, DaysInWaitingList) are not inputs.
- Report: `reports/clustering_milestone1.md`.

```bash
python main.py cluster --algorithm kmeans                 # untuned K-Means (k=8), k=2..12 sweep, PCA plot, cluster profile
python main.py cluster --algorithm kmeans --compare       # same baseline on 4 feature representations
python -m pytest -q tests/test_clustering.py
```

Results go to `outputs/Clustering/` (`*_v1.*` is the team matrix). The fitted model is saved in `artifacts/clustering/` (git-ignored).
