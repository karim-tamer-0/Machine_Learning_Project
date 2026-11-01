import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.classification.baseline import SCORING, run_cv
from src.classification.pipeline import build_pipeline
from tests.test_pipeline import _data


def test_run_cv_returns_every_required_metric():
    X, y = _data()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    row = run_cv("logistic_regression", pipe, X, y, cv)
    for metric in SCORING:
        assert np.isfinite(row[f"cv_{metric}_mean"])
        assert row[f"cv_{metric}_std"] >= 0
    assert row["config"] == "baseline (untuned)"
    assert row["fit_time_per_fold_s"] > 0


def test_baselines_include_a_linear_model_and_an_ensemble_and_both_fit():
    from src.classification.baseline import baseline_models
    X, y = _data()
    models = baseline_models(42)
    assert {"logistic_regression", "random_forest"} <= set(models)
    for name, (model, scaled) in models.items():
        pipe = build_pipeline(model, scale=scaled).fit(X, y)
        assert pipe.predict(X).shape == (len(X),)