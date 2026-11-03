"""Feature representations for clustering experiments.

v1            the preprocessing teammate's matrix, unchanged (z-scored counts + one-hot).
v1_dedup      v1 without exact duplicate H2 rows (the classification task also removes them).
v2            proposal: binary HasChildren/HasBabies, log1p on skewed columns, no
              ArrivalDateDayOfMonth, rescaled. One-hot blocks unchanged.
v2_no_country v2 without the Country dummies.
Scalers here are fitted on all rows: clustering is unsupervised, there is no target or test set.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import ONE_HOT_PREFIXES

REPRESENTATIONS = ["v1", "v1_dedup", "v2", "v2_no_country"]


def _v2(scaled, raw, drop_country):
    prefixes = [p for p in ONE_HOT_PREFIXES if not (drop_country and p == "Country_")]
    onehot = [c for c in scaled.columns if c.startswith(tuple(prefixes))]
    cont = pd.DataFrame({
        "LeadTime_log": np.log1p(raw["LeadTime"]),
        "Adults": raw["Adults"],
        "TotalOfSpecialRequests": raw["TotalOfSpecialRequests"],
        "StaysInWeekendNights_log": np.log1p(raw["StaysInWeekendNights"]),
        "StaysInWeekNights_log": np.log1p(raw["StaysInWeekNights"]),
    }).astype(float)
    scaler = StandardScaler().fit(cont)
    cont = pd.DataFrame(scaler.transform(cont), columns=cont.columns, index=cont.index)
    flags = pd.DataFrame({"HasChildren": (raw["Children"] > 0).astype(float),
                          "HasBabies": (raw["Babies"] > 0).astype(float)})
    X = pd.concat([cont, flags, scaled[["Month_sin", "Month_cos"]], scaled[onehot]], axis=1)
    return X, scaler


def build(name, scaled, raw, dup_mask):
    """Return (X, row_mask, scaler_or_None); row_mask selects the rows of `scaled` that are used."""
    all_rows = np.ones(len(scaled), dtype=bool)
    if name == "v1":
        return scaled, all_rows, None
    if name == "v1_dedup":
        keep = ~dup_mask
        return scaled[keep], keep, None
    if name in ("v2", "v2_no_country"):
        X, scaler = _v2(scaled, raw, drop_country=(name == "v2_no_country"))
        return X, all_rows, scaler
    raise ValueError(f"Unknown representation '{name}', choose from {REPRESENTATIONS}")
