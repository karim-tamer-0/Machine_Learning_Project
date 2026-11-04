"""Internal metrics, k sweep and cluster profiling for the clustering task."""
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from .data import ONE_HOT_PREFIXES


def silhouette_sample_index(n_rows, size, seed):
    """Fixed random sample: silhouette is O(n^2), so it is computed on a sample."""
    return np.random.RandomState(seed).choice(n_rows, min(size, n_rows), replace=False)


def evaluate(X, labels, sample_idx):
    counts = np.bincount(labels)
    return {"silhouette": float(silhouette_score(X[sample_idx], labels[sample_idx])),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "min_cluster": int(counts.min()), "max_cluster": int(counts.max())}


def k_sweep(X, k_values, n_init, seed, sample_idx):
    rows = []
    for k in k_values:
        t = time.time()
        km = KMeans(k, n_init=n_init, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     **evaluate(X, km.labels_, sample_idx),
                     "fit_seconds": round(time.time() - t, 1)})
    return pd.DataFrame(rows)


def profile(scaled, raw, labels):
    """Per-cluster booking profile from UNSCALED values (raw H2 rows aligned to the matrix).

    IsCanceled and ADR are added for interpretation only (post-hoc); they were never used to
    fit the clusters or to choose any parameter.
    """
    p = pd.DataFrame({
        "LeadTime": raw["LeadTime"], "Adults": raw["Adults"],
        "StaysInWeekendNights": raw["StaysInWeekendNights"],
        "StaysInWeekNights": raw["StaysInWeekNights"],
        "TotalOfSpecialRequests": raw["TotalOfSpecialRequests"],
        "HasChildren": (raw["Children"] > 0).astype(int),
        "HasBabies": (raw["Babies"] > 0).astype(int),
        "posthoc_cancel_rate": raw["IsCanceled"], "posthoc_mean_ADR": raw["ADR"]})
    num = p.groupby(labels).mean().round(3)
    cat = pd.DataFrame(index=num.index)
    for pre in ONE_HOT_PREFIXES:
        cols = [c for c in scaled.columns if c.startswith(pre)]
        if not cols:
            continue
        mode = scaled[cols].idxmax(axis=1).str.replace(pre, "", regex=False)
        cat[pre.rstrip("_")] = mode.groupby(labels).agg(
            lambda s: f"{s.value_counts().index[0]} ({s.value_counts(normalize=True).iloc[0]:.0%})")
    size = pd.Series(np.bincount(labels), name="size")
    out = pd.concat([size, (size / len(labels)).round(3).rename("share"), num, cat], axis=1)
    return out
