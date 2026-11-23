import json

import numpy as np
import pytest
from scipy.stats import rv_continuous  # noqa: F401  (confirms scipy is available)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.classification.baseline import SCORING
from src.classification.pipeline import build_pipeline
from src.classification.tuning import (build_distributions, refit_key, save_tuning_result,
                                       to_distribution, tune_model)
from src.data.loader import load_config
from tests.test_pipeline import _data


def test_distribution_types():
    ints = to_distribution({"type": "int", "low": 2, "high": 5}).rvs(200, random_state=0)
    assert ints.min() >= 2 and ints.max() <= 5
    logs = to_distribution({"type": "float", "low": 1e-5, "high": 1e-1, "log": True}).rvs(200, random_state=0)
    assert logs.min() >= 1e-5 and logs.max() <= 1e-1
    flat = to_distribution({"type": "float", "low": 0.0, "high": 1.0}).rvs(200, random_state=0)
    assert 0.0 <= flat.min() and flat.max() <= 1.0
    choices = to_distribution({"type": "choice", "values": [[64], [64, 32], "sqrt"]})
    assert choices == [(64,), (64, 32), "sqrt"]


def test_unknown_space_type_raises():
    with pytest.raises(ValueError):
        to_distribution({"type": "banana"})


def test_every_search_space_in_config_is_valid():
    spaces = load_config()["classification"]["tuning"]["search_spaces"]
    assert spaces, "no search spaces in config.yaml"
    for name, space in spaces.items():
        assert all(p.startswith("model__") for p in space), name
        assert build_distributions(space)


def test_refit_key_maps_scoring_name():
    assert refit_key("average_precision") == "pr_auc"
    with pytest.raises(ValueError):
        refit_key("not_a_metric")


def test_tune_model_returns_row_params_and_results(tmp_path):
    X, y = _data(n=1200)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    space = {"model__C": {"type": "float", "low": 0.01, "high": 10, "log": True}}
    result = tune_model("logistic_regression", pipe, X, y, cv, space, n_iter=3)

    assert "model__C" in result["best_params"]
    assert np.isfinite(result["best_score"])
    assert len(result["cv_results"]) == 3
    row = result["row"]
    assert row["config"] == "tuned"
    for metric in SCORING:
        assert np.isfinite(row[f"cv_{metric}_mean"])
    assert row["cv_pr_auc_mean"] == pytest.approx(result["best_score"])

    path = save_tuning_result(result, "logistic_regression", tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["best_params"] == result["best_params"]
    assert (tmp_path / "logistic_regression_cv_results.csv").exists()