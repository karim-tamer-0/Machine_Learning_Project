from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=PROJECT_ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(config=None):
    """Load H2.csv and check it matches the published 79,330 rows and 31 columns."""
    config = config or load_config()
    df = pd.read_csv(PROJECT_ROOT / config["data"]["raw_path"])
    expected = (config["data"]["expected_raw_rows"], config["data"]["expected_columns"])
    if df.shape != expected:
        raise ValueError(f"H2.csv has shape {df.shape}, expected {expected}")
    return df


def load_all_models(config=None):
    """Load the shared dataset produced by src/Preprocessing/All models.py."""
    config = config or load_config()
    df = pd.read_csv(PROJECT_ROOT / config["data"]["all_models_path"])
    if df.shape[1] != config["data"]["expected_columns"]:
        raise ValueError(f"Expected {config['data']['expected_columns']} columns, got {df.shape[1]}")
    return df


def get_features(feature_set="strict", config=None):
    """Return (numeric, categorical) column lists for 'strict' or 'relaxed'."""
    config = config or load_config()
    features = config["classification"]["features"]
    numeric = list(features["strict"]["numeric"])
    categorical = list(features["strict"]["categorical"])
    if feature_set == "relaxed":
        numeric += features["relaxed_extra"]["numeric"]
        categorical += features["relaxed_extra"]["categorical"]
    elif feature_set != "strict":
        raise ValueError("feature_set must be 'strict' or 'relaxed'")
    return numeric, categorical


def data_quality_summary(df):
    """Per-column dtype, missing count, cardinality and numeric range."""
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique": df.nunique(),
    })
    numeric = df.select_dtypes("number")
    summary["min"] = numeric.min()
    summary["max"] = numeric.max()
    summary["skew"] = numeric.skew().round(2)
    return summary
