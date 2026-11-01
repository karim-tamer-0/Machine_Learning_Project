"""Validation-set check of the untuned baseline, with confusion matrix (Member B, Milestone 1).
 
The model is fitted on the TRAINING set and scored on the VALIDATION set.
The test set is not used.
 
Run from the repo root:   python -m src.classification.evaluate_baseline
"""
import matplotlib
 
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
 
from src.classification.baseline import baseline_models
from src.classification.pipeline import build_pipeline
from src.data.loader import PROJECT_ROOT, load_config
from src.data.split import load_classification_split
 
 
def evaluate_on_validation(pipeline, X_val, y_val):
    """Return (metrics dict, 2x2 confusion matrix) for a fitted pipeline."""
    pred = pipeline.predict(X_val)
    proba = pipeline.predict_proba(X_val)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(y_val, pred),
        "recall": recall_score(y_val, pred),
        "f1": f1_score(y_val, pred),
        "roc_auc": roc_auc_score(y_val, proba),
        "pr_auc": average_precision_score(y_val, proba),
    }
    return metrics, confusion_matrix(y_val, pred)
 
 
def save_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Not canceled", "Canceled"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
if __name__ == "__main__":
    config = load_config()
    X_train, X_val, _, y_train, y_val, _ = load_classification_split("strict", config)
    out_dir = PROJECT_ROOT / config["classification"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
 
    rows = []
    for name, (model, scaled) in baseline_models(config["seed"]).items():
        pipeline = build_pipeline(model, "strict", scale=scaled, config=config)
        pipeline.fit(X_train, y_train)
        metrics, cm = evaluate_on_validation(pipeline, X_val, y_val)
        rows.append({"model": name, "config": "baseline (untuned)", **metrics})
        save_confusion_matrix(cm, f"{name} baseline (validation set)",
                              out_dir / f"confusion_matrix_baseline_{name}.png")
        print(name, {k: round(v, 4) for k, v in metrics.items()})
        print("confusion matrix [rows = actual, columns = predicted]:")
        print(cm)
 
    pd.DataFrame(rows).to_csv(out_dir / "baseline_validation_metrics.csv", index=False)
    print("saved to:", out_dir)