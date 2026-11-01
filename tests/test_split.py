import numpy as np
import pandas as pd
 
from src.data.loader import get_features, load_config
from src.data.split import get_cv, make_split
 
LEAKY = [
    "ReservationStatus", "ReservationStatusDate", "AssignedRoomType",
    "BookingChanges", "DaysInWaitingList", "RequiredCarParkingSpaces",
]
 
 
def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "LeadTime": rng.integers(0, 300, n),
        "Adults": rng.integers(1, 4, n),
        "Meal": rng.choice(["BB", "HB", "SC"], n),
        "IsCanceled": (rng.random(n) < 0.3).astype(int),
    })
 
 
def _split(df, cols=("LeadTime", "Meal")):
    return make_split(df, list(cols), "IsCanceled", 0.2, 0.1, 42)
 
 
def test_split_proportions_are_70_20_10():
    df = _toy()
    X_tr, X_va, X_te, *_ = _split(df)
    assert abs(len(X_tr) - 1400) <= 2
    assert abs(len(X_va) - 400) <= 2
    assert abs(len(X_te) - 200) <= 2
    assert len(X_tr) + len(X_va) + len(X_te) == len(df)
 
 
def test_parts_do_not_overlap():
    X_tr, X_va, X_te, *_ = _split(_toy())
    assert set(X_tr.index).isdisjoint(X_va.index)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert set(X_va.index).isdisjoint(X_te.index)
 
 
def test_split_is_stratified():
    _, _, _, y_tr, y_va, y_te = _split(_toy())
    assert abs(y_tr.mean() - y_va.mean()) < 0.02
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
 
 
def test_split_is_reproducible():
    df = _toy()
    a, b = _split(df), _split(df)
    for part_a, part_b in zip(a, b):
        assert list(part_a.index) == list(part_b.index)
 
 
def test_only_requested_columns_reach_the_model():
    X_tr, X_va, X_te, *_ = _split(_toy())
    for X in (X_tr, X_va, X_te):
        assert list(X.columns) == ["LeadTime", "Meal"]
 
 
def test_feature_lists_exclude_leaky_columns_and_target():
    config = load_config()
    for feature_set in ("strict", "relaxed"):
        numeric, categorical = get_features(feature_set, config)
        used = set(numeric + categorical)
        assert used.isdisjoint(LEAKY)
        assert config["classification"]["target"] not in used
 
 
def test_cv_folds_are_stratified_and_seeded():
    config = load_config()
    cv = get_cv(config)
    assert cv.get_n_splits() == config["classification"]["split"]["cv_folds"]
    assert cv.shuffle is True
    assert cv.random_state == config["seed"]