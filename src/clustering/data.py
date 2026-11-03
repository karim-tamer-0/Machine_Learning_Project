"""Loading and verification of the clustering feature matrix (Task C).

The preprocessing teammate's script (src/Preprocessing/Clustering/clusteringprocessing.py)
writes outputs/Clustering/kmeans_cleaned_scaled.csv without row ids. To interpret clusters
with unscaled values we rebuild the same row filter from data/H2.csv and VERIFY that it
reproduces the CSV exactly, instead of trusting the row order.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NUMERIC = ["LeadTime", "Adults", "Children", "Babies", "TotalOfSpecialRequests",
           "StaysInWeekendNights", "StaysInWeekNights", "ArrivalDateDayOfMonth"]
ONE_HOT_PREFIXES = ["Meal_", "MarketSegment_", "DistributionChannel_",
                    "ReservedRoomType_", "CustomerType_", "Country_"]
# Outcome / post-booking columns that must never be an input of the clustering.
FORBIDDEN = {"IsCanceled", "ReservationStatus", "ReservationStatusDate", "ADR",
             "AssignedRoomType", "BookingChanges", "DaysInWaitingList"}


def load_config(path=PROJECT_ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scaled(config=None):
    """Load the teammate's scaled matrix, drop constant columns, check for leakage."""
    config = config or load_config()
    df = pd.read_csv(PROJECT_ROOT / config["clustering"]["scaled_path"])
    leaked = FORBIDDEN & set(df.columns)
    if leaked:
        raise ValueError(f"Leakage-prone columns in the clustering matrix: {sorted(leaked)}")
    constant = [c for c in df.columns if df[c].nunique() == 1]
    df = df.drop(columns=constant)
    if df.isna().any().any():
        raise ValueError("NaN values in the clustering matrix")
    return df, constant


def load_raw_aligned(n_expected, config=None):
    """Return (raw_kept, raw_all): H2 rows that survive the teammate's filters, in CSV order.

    Filters (same as the preprocessing script): Children NaN -> 0, Adults > 0, Babies <= 2.
    Text columns are stripped because H2.csv pads them with spaces.
    """
    config = config or load_config()
    raw = pd.read_csv(PROJECT_ROOT / config["data"]["raw_path"])
    for c in raw.select_dtypes(exclude="number"):
        raw[c] = raw[c].str.strip()
    raw["Children"] = raw["Children"].fillna(0)
    keep = (raw["Adults"] > 0) & (raw["Babies"] <= 2)
    kept = raw[keep].copy()
    if len(kept) != n_expected:
        raise ValueError(f"Rebuilt filter keeps {len(kept)} rows, CSV has {n_expected}")
    return kept.reset_index(drop=True), raw


def verify_alignment(scaled, raw_kept):
    """Check the rebuilt rows are exactly the rows of the scaled CSV (numeric + month)."""
    z = StandardScaler().fit_transform(raw_kept[NUMERIC].astype(float))
    if not np.allclose(z, scaled[NUMERIC].to_numpy(), atol=1e-6):
        raise ValueError("Scaled CSV does not match H2.csv rows (different filters or order)")
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    m = raw_kept["ArrivalDateMonth"].map({n: i for i, n in enumerate(months)}).to_numpy()
    if not np.allclose(np.sin(2 * np.pi * m / 12), scaled["Month_sin"].to_numpy(), atol=1e-9):
        raise ValueError("Month encoding does not match H2.csv rows")
    return True


def load_aligned(config=None):
    """(scaled, raw_kept, dup_mask, dropped_constant); dup_mask = exact duplicate raw rows."""
    config = config or load_config()
    scaled, constant = load_scaled(config)
    raw_kept, raw_all = load_raw_aligned(len(scaled), config)
    verify_alignment(scaled, raw_kept)
    dup_all = raw_all.duplicated(keep="first")
    keep = (raw_all["Adults"] > 0) & (raw_all["Babies"] <= 2)
    dup_mask = dup_all[keep].to_numpy()
    return scaled, raw_kept, dup_mask, constant
