import pandas as pd
import pytest

from src.classification.feature_sets import ablation_table, borderline_columns, gap_table
from src.data.loader import get_features, load_config
from src.data.split import load_classification_split


def test_strict_plus_one_column_adds_exactly_that_column():
    strict_num, strict_cat = get_features("strict")
    num, cat = get_features("strict+Country")
    assert num == strict_num and cat == strict_cat + ["Country"]
    num, cat = get_features("strict+TotalOfSpecialRequests")
    assert num == strict_num + ["TotalOfSpecialRequests"] and cat == strict_cat


def test_strict_plus_rejects_columns_outside_relaxed_extra():
    with pytest.raises(ValueError):
        get_features("strict+ReservationStatus")
    with pytest.raises(ValueError):
        get_features("loose")


def test_relaxed_is_strict_plus_every_borderline_column():
    config = load_config()
    strict = sum(get_features("strict", config), [])
    relaxed = sum(get_features("relaxed", config), [])
    assert sorted(relaxed) == sorted(strict + borderline_columns(config))
    dropped = set(config["classification"]["features"]["drop"])
    assert not dropped & set(relaxed)


def test_every_feature_set_uses_the_same_rows():
    config = load_config()
    strict = load_classification_split("strict", config)
    for feature_set in ("relaxed", "strict+DepositType"):
        other = load_classification_split(feature_set, config)
        for a, b in zip(strict[:3], other[:3]):
            assert a.index.equals(b.index)


def test_gap_and_ablation_tables():
    cv_table = pd.DataFrame({
        "model": ["m1", "m1", "m2", "m2"],
        "feature_set": ["strict", "relaxed", "strict", "relaxed"],
        "cv_pr_auc_mean": [0.5, 0.7, 0.6, 0.65],
    })
    gaps = gap_table(cv_table)
    assert list(gaps.index) == ["m1", "m2"]
    assert gaps.loc["m1", "gain"] == pytest.approx(0.2)

    runs = pd.DataFrame({"feature_set": ["strict+A", "strict+B"], "cv_pr_auc_mean": [0.55, 0.62]})
    ablation = ablation_table(runs, strict_score=0.5)
    assert list(ablation["column"]) == ["B", "A"]
    assert ablation["gain"].iloc[0] == pytest.approx(0.12)
