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