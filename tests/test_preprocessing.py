import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
 
from src.Preprocessing.Classification import build_preprocessor
from src.data.loader import get_features
from src.features.booking_features import BookingFeatures
 
 
def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    numeric, categorical = get_features("strict")
    df = pd.DataFrame({c: rng.integers(0, 50, n) for c in numeric})
    df["PreviousCancellations"] = rng.choice([0, 0, 0, 0, 1, 5, 26], n)
    df["PreviousBookingsNotCanceled"] = rng.choice([0, 0, 0, 0, 1, 3, 40], n)
    for c in categorical:
        df[c] = rng.choice(["A", "B", "C"], n)
    df["Agent"] = rng.choice(
        ["a1", "a2", "a3"] + [f"x{i}" for i in range(50)], n, p=[0.3] * 3 + [0.1 / 50] * 50)
    df["Company"] = rng.choice(["NoCompany", "40", "223"], n, p=[0.9, 0.05, 0.05])
    return df
 
 
def _pipe(scale):
    return Pipeline([("features", BookingFeatures()), ("prep", build_preprocessor(scale=scale))])
 
 
def _names(pipe):
    return list(pipe.named_steps["prep"].get_feature_names_out())
 
 
def test_company_id_replaced_by_flag():
    pipe = _pipe(True).fit(_toy())
    names = _names(pipe)
    assert "has_company" in names
    assert not any(n.startswith("Company") for n in names)
 
 
def test_rare_agents_are_grouped():
    pipe = _pipe(True).fit(_toy())
    assert len([n for n in _names(pipe) if n.startswith("Agent_")]) <= 5
 
 
def test_unseen_category_and_missing_values_do_not_break_or_leave_nans():
    train, other = _toy(seed=0), _toy(seed=1)
    other.loc[0, "MarketSegment"] = "BrandNew"
    other.loc[1, "LeadTime"] = np.nan
    other.loc[2, "Children"] = np.nan
    out = _pipe(True).fit(train).transform(other)
    assert not np.isnan(out.astype(float)).any()
 
 
def test_scaler_is_fitted_on_training_data_only():
    train, other = _toy(seed=0), _toy(seed=1)
    pipe = _pipe(True).fit(train)
    i = _names(pipe).index("LeadTime")
    assert abs(pipe.transform(train)[:, i].mean()) < 1e-6
    assert abs(pipe.transform(other)[:, i].mean()) > 1e-6
 
 
def test_skewed_columns_are_log_transformed_then_scaled():
    train = _toy()
    pipe = _pipe(True).fit(train)
    i = _names(pipe).index("PreviousCancellations")
    logged = np.log1p(train["PreviousCancellations"].to_numpy())
    expected = (logged - logged.mean()) / logged.std()
    assert np.allclose(pipe.transform(train)[:, i], expected)
 
 
def test_tree_variant_leaves_numbers_unscaled():
    train = _toy()
    pipe = _pipe(False).fit(train)
    i = _names(pipe).index("LeadTime")
    assert np.allclose(pipe.transform(train)[:, i], train["LeadTime"])
    j = _names(pipe).index("PreviousCancellations")
    assert np.allclose(pipe.transform(train)[:, j], train["PreviousCancellations"])