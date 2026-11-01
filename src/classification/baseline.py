"""Untuned baseline classifiers evaluated with cross-validation on the TRAINING set only
(Member B, Milestone 1). The validation and test sets are not used here.
 
Run from the repo root:   python -m src.classification.baseline
"""
import time
 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
 
from src.classification.pipeline import build_pipeline
from src.data.loader import PROJECT_ROOT, load_config
from src.data.split import get_cv, load_classification_split
 
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
 
 
def run_cv(name, pipeline, X, y, cv, feature_set="strict", scaled=True):
    """5-fold CV of one pipeline. Returns one row: mean and std of every metric,
    the training F1 (to spot overfitting) and the mean fit time per fold."""
    start = time.perf_counter()
    res = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING,
                         return_train_score=True, n_jobs=1)
    row = {"model": name, "config": "baseline (untuned)", "feature_set": feature_set,
           "preprocessing": "scaled" if scaled else "unscaled"}
    for metric in SCORING:
        row[f"cv_{metric}_mean"] = res[f"test_{metric}"].mean()
        row[f"cv_{metric}_std"] = res[f"test_{metric}"].std()
    row["train_f1_mean"] = res["train_f1"].mean()
    row["fit_time_per_fold_s"] = res["fit_time"].mean()
    row["total_time_s"] = time.perf_counter() - start
    return row
 
 
def baseline_models(seed):
    """name -> (untuned estimator, uses scaled preprocessing)."""
    return {
        "logistic_regression": (LogisticRegression(max_iter=1000, random_state=seed), True),
        "random_forest": (RandomForestClassifier(random_state=seed, n_jobs=-1), False),
    }
 
 
if __name__ == "__main__":
    config = load_config()
    X_train, _, _, y_train, _, _ = load_classification_split("strict", config)
    cv = get_cv(config)
 
    rows = []
    for name, (model, scaled) in baseline_models(config["seed"]).items():
        pipeline = build_pipeline(model, "strict", scale=scaled, config=config)
        rows.append(run_cv(name, pipeline, X_train, y_train, cv, scaled=scaled))
        print(f"done: {name}")
 
    results = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / config["classification"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baseline_cv_metrics.csv"
    results.to_csv(path, index=False)
    print(results.set_index("model").T.round(4).to_string())
    print("saved:", path)