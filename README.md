# Machine-Learning-Project
Team 29's Machine Learning Project

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

## Regression (Task A - ADR Prediction): baselines, Milestone 1

- Prediction point: Initial booking creation in the hotel Property Management System (PMS). Post-booking outcome and lifecycle variables (`IsCanceled`, `ReservationStatus`, `ReservationStatusDate`, `AssignedRoomType`, `BookingChanges`, `DaysInWaitingList`) along with raw ID columns (`Agent`, `Company`) are excluded to eliminate target and lifecycle leakage.
- Data & Splits: Version 3 cleaned pipeline filters non-positive rates (`ADR <= 0`), removes extreme corrupted outliers (`ADR > 1000`), normalizes categorical whitespace, applies fixed Top-10 country grouping, and removes duplicate rows prior to splitting. Split 70% train (35,384 rows) / 15% validation (7,584 rows) / 15% test (7,583 rows, locked) with seed 42.
- Preprocessing: `StandardScaler` and `OneHotEncoder(drop='first', handle_unknown='ignore')` are fitted strictly on the training partition.
- Baseline & PCA: Ordinary Least Squares (OLS) baseline achieves validation MAE $19.11, RMSE $25.52, and $R^2 = 0.5986$. PCA dimensionality reduction sweeps (99% down to 80% variance and 2D/3D projections) are evaluated under `src/regression/pca_analysis.py`.
- Reports: `reports/regression_data_preparation_report.md` and `reports/Regression_PCA_Model_analysis.md`.

```bash
python -m src.Preprocessing.Regression3                             # generate Version 3 cleaned splits
python main.py baseline --task regression --model linear_regression # 5-fold CV on the training set
python main.py evaluate --task regression --model linear_regression # validation evaluation and residual plots
python main.py pca --task regression                                # PCA scree sweep and model comparison
python -m pytest -q tests/test_regression_baseline.py tests/test_regression_pca.py


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
## Clustering (Task C): baseline, Milestone 1

- Analysis point: booking creation time. Input: `outputs/Clustering/kmeans_cleaned_scaled.csv` (made by `src/Preprocessing/Clustering/clusteringprocessing.py`). Outcome columns (IsCanceled, ReservationStatus*, ADR, AssignedRoomType, BookingChanges, DaysInWaitingList) are not inputs.
- Report: `reports/clustering_milestone1.md`.

```bash
python main.py cluster --algorithm kmeans                 # untuned K-Means (k=8), k=2..12 sweep, PCA plot, cluster profile
python main.py cluster --algorithm kmeans --compare       # same baseline on 4 feature representations
python -m pytest -q tests/test_clustering.py
```

Results go to `outputs/Clustering/` (`*_v1.*` is the team matrix). The fitted model is saved in `artifacts/clustering/` (git-ignored).
