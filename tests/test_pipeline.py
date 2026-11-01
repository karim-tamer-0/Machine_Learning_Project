import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
 
from src.classification.pipeline import build_pipeline
from tests.test_preprocessing import _toy
 
 
def _data(n=1500, seed=0):
    X = _toy(n, seed)
    y = (X["LeadTime"] > 25).astype(int)
    return X, y
 
 
def test_step_order_with_and_without_reducer():
    plain = build_pipeline(LogisticRegression())
    assert [name for name, _ in plain.steps] == ["features", "prep", "model"]
    reduced = build_pipeline(LogisticRegression(), reducer=PCA(n_components=5))
    assert [name for name, _ in reduced.steps] == ["features", "prep", "reducer", "model"]
 
 
def test_logistic_regression_pipeline_predicts_probabilities():
    X, y = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=42)).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)
 
 
def test_tree_pipeline_uses_unscaled_preprocessing():
    X, y = _data()
    pipe = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42),
                          scale=False).fit(X, y)
    assert pipe.predict(X).shape == (len(X),)
 
 
def test_pca_reducer_works_inside_pipeline():
    X, y = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000), reducer=PCA(n_components=5)).fit(X, y)
    assert pipe.named_steps["reducer"].n_components_ == 5
 
 
def test_cross_validation_runs_on_the_whole_pipeline():
    X, y = _data()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(build_pipeline(LogisticRegression(max_iter=1000)), X, y,
                             cv=cv, scoring="f1")
    assert len(scores) == 3 and np.isfinite(scores).all()
 
 
def test_pipeline_predicts_on_a_single_raw_row():
    X, y = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    assert pipe.predict(X.iloc[[0]]).shape == (1,)