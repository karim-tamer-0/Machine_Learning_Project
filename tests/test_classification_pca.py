import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from src.classification.pca import (_pipeline, choose_n_components, explained_variance_table,
                                    fit_full_pca, loadings_table)
from src.data.loader import load_config
from tests.test_pipeline import _data


def test_choose_n_components_reaches_the_threshold():
    cumulative = [0.5, 0.8, 0.9, 0.96, 1.0]
    assert choose_n_components(cumulative, 0.80) == 2
    assert choose_n_components(cumulative, 0.95) == 4
    assert choose_n_components(cumulative, 1.0) == 5


def test_explained_variance_table_is_cumulative_and_sums_to_one():
    X, _ = _data()
    fitted = fit_full_pca(X)
    table = explained_variance_table(fitted.named_steps["model"])
    assert np.all(np.diff(table["cumulative"]) >= -1e-12)
    assert table["cumulative"].iloc[-1] == pytest.approx(1.0)


def test_loadings_name_the_encoded_columns():
    X, _ = _data()
    loadings = loadings_table(fit_full_pca(X), n_components=2, top_n=3)
    assert len(loadings) == 6
    assert set(loadings["component"]) == {1, 2}


def test_pca_pipelines_always_use_scaled_preprocessing():
    from src.classification.baseline import baseline_models
    config = load_config()
    model, scaled = baseline_models(42)["xgboost"]
    assert scaled is False
    pipeline, used_scaling = _pipeline(model, scaled, 5, 42, config)
    assert used_scaling is True
    assert [name for name, _ in pipeline.steps] == ["features", "prep", "reducer", "model"]
    original, used_scaling = _pipeline(model, scaled, None, 42, config)
    assert used_scaling is False and "reducer" not in original.named_steps
    every, _ = _pipeline(model, scaled, "all", 42, config)
    assert every.named_steps["reducer"].n_components is None


def test_pca_is_fitted_inside_each_cv_fold():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X, y = _data()
    pipeline, _ = _pipeline(LogisticRegression(max_iter=1000), True, 5, 42, load_config())
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="average_precision")
    assert len(scores) == 3 and np.isfinite(scores).all()
