import numpy as np
from sklearn.linear_model import LogisticRegression
 
from src.classification.evaluate_baseline import evaluate_on_validation, save_confusion_matrix
from src.classification.pipeline import build_pipeline
from tests.test_pipeline import _data
 
 
def test_validation_metrics_and_confusion_matrix(tmp_path):
    X, y = _data(n=2000)
    X_tr, X_va, y_tr, y_va = X.iloc[:1500], X.iloc[1500:], y.iloc[:1500], y.iloc[1500:]
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    metrics, cm = evaluate_on_validation(pipe, X_va, y_va)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert cm.shape == (2, 2) and cm.sum() == len(y_va)
    save_confusion_matrix(cm, "test", tmp_path / "cm.png")
    assert (tmp_path / "cm.png").exists()