import numpy as np

from src.classification.tuning import build_distributions
from src.classification.xgboost_plateau import running_best, tuned_xgboost_pipeline
from src.data.loader import load_config
from tests.test_pipeline import _data


def test_running_best_never_decreases():
    assert list(running_best([0.5, 0.4, 0.6, 0.55, 0.7])) == [0.5, 0.5, 0.6, 0.6, 0.7]


def test_plateau_search_space_is_valid_and_kept_out_of_regular_tuning():
    config = load_config()
    space = config["classification"]["plateau_check"]["search_space"]
    assert all(p.startswith("model__") for p in space)
    assert build_distributions(space)
    assert "xgboost_wide" not in config["classification"]["tuning"]["search_spaces"]


def test_best_parameters_can_be_set_on_the_xgboost_pipeline():
    X, y = _data()
    params = {"model__n_estimators": 20, "model__max_depth": 3, "model__gamma": 0.5,
              "model__reg_alpha": 0.01}
    pipe = tuned_xgboost_pipeline(params, load_config()).fit(X, y)
    assert pipe.named_steps["model"].get_params()["gamma"] == 0.5
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(axis=1), 1)
