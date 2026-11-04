import numpy as np
import pandas as pd
import pytest

from src.clustering import representations
from src.clustering.data import (FORBIDDEN, PROJECT_ROOT, load_aligned, load_config,
                                 load_scaled, verify_alignment)
from src.clustering.evaluation import evaluate, silhouette_sample_index

CFG = load_config()
SCALED = PROJECT_ROOT / CFG["clustering"]["scaled_path"]
RAW = PROJECT_ROOT / CFG["data"]["raw_path"]
needs_data = pytest.mark.skipif(not (SCALED.exists() and RAW.exists()),
                                reason="clustering CSV or H2.csv not present")


@needs_data
def test_no_forbidden_columns_and_no_nans():
    df, constant = load_scaled(CFG)
    assert not FORBIDDEN & set(df.columns)
    assert not df.isna().any().any()
    assert "ReservedRoomType_P" in constant          # all-zero column is dropped


@needs_data
def test_scaled_rows_match_h2_rows():
    scaled, raw, dup_mask, _ = load_aligned(CFG)
    assert len(scaled) == len(raw) == len(dup_mask) == 78_938
    assert verify_alignment(scaled, raw)
    assert (raw["Adults"] > 0).all() and (raw["Babies"] <= 2).all()


@needs_data
def test_one_hot_groups_sum_to_one():
    scaled, *_ = load_aligned(CFG)
    for pre in ("Meal_", "MarketSegment_", "DistributionChannel_", "CustomerType_", "Country_"):
        assert (scaled[[c for c in scaled if c.startswith(pre)]].sum(axis=1) == 1).all()


@needs_data
@pytest.mark.parametrize("name", representations.REPRESENTATIONS)
def test_representations_are_finite_and_consistent(name):
    scaled, raw, dup_mask, _ = load_aligned(CFG)
    X, rows, _ = representations.build(name, scaled, raw, dup_mask)
    assert len(X) == rows.sum() and np.isfinite(X.to_numpy(float)).all()
    if name.startswith("v2"):
        assert X["HasBabies"].isin([0, 1]).all() and "Babies" not in X.columns


def test_metrics_are_deterministic_with_fixed_seed():
    from sklearn.cluster import KMeans
    X = np.random.RandomState(0).normal(size=(600, 5))
    idx = silhouette_sample_index(len(X), 300, 42)
    a = KMeans(3, n_init=3, random_state=42).fit(X).labels_
    b = KMeans(3, n_init=3, random_state=42).fit(X).labels_
    assert (a == b).all() and evaluate(X, a, idx) == evaluate(X, b, idx)
