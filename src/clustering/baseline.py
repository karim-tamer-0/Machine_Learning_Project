"""Milestone 1 clustering experiments: untuned K-Means baseline (+ representation check).

    python main.py cluster --algorithm kmeans                  # baseline on the team matrix (v1)
    python main.py cluster --algorithm kmeans --compare        # v1 vs dedup vs v2 variants
Everything is seeded with config.yaml `seed`. No supervised label is used to fit or tune.
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import representations  # noqa: E402
from .data import PROJECT_ROOT, load_aligned, load_config  # noqa: E402
from .evaluation import evaluate, k_sweep, profile, silhouette_sample_index  # noqa: E402


def _dirs(config):
    out = PROJECT_ROOT / config["clustering"]["output_dir"]
    art = PROJECT_ROOT / config["clustering"]["artifact_dir"]
    for d in (out, art):
        if d.exists() and not d.is_dir():
            raise NotADirectoryError(
                f"'{d}' is a file (placeholder). Delete it with `git rm {d.name}` and rerun.")
        d.mkdir(parents=True, exist_ok=True)
    return out, art


def _fit_baseline(X, config, seed, sample_idx):
    b = config["clustering"]["baseline"]
    t = time.time()
    km = KMeans(b["n_clusters"], n_init=b["n_init"], random_state=seed).fit(X)
    seconds = round(time.time() - t, 2)
    return km, seconds, evaluate(X, km.labels_, sample_idx)


def run_baseline(representation="v1", config=None, sweep=True):
    config = config or load_config()
    cc, seed = config["clustering"], config["seed"]
    out, art = _dirs(config)
    scaled, raw, dup_mask, constant = load_aligned(config)
    Xdf, rows, scaler = representations.build(representation, scaled, raw, dup_mask)
    scaled_u, raw_u = scaled[rows].reset_index(drop=True), raw[rows].reset_index(drop=True)
    X = Xdf.to_numpy(dtype=float)
    idx = silhouette_sample_index(len(X), cc["silhouette_sample"], seed)
    tag = representation

    if sweep:
        k_sweep(X, range(*cc["k_range"]), cc["baseline"]["n_init"], seed, idx).to_csv(
            out / f"kmeans_k_sweep_{tag}.csv", index=False)
    km, seconds, metrics = _fit_baseline(X, config, seed, idx)
    pca = PCA(random_state=seed).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)

    result = {"algorithm": "KMeans (untuned baseline)", "representation": tag,
              "n_rows": len(X), "n_features": X.shape[1], "dropped_constant": constant,
              "params": {"n_clusters": cc["baseline"]["n_clusters"], "init": "k-means++",
                         "n_init": cc["baseline"]["n_init"], "random_state": seed},
              "silhouette_sample_size": len(idx), "fit_seconds": seconds,
              "pca_components_for_80_90_95": [int(np.searchsorted(cum, t)) + 1 for t in (.8, .9, .95)],
              "pca_pc1_pc2_variance": float(cum[1]), **metrics}
    (out / f"kmeans_baseline_metrics_{tag}.json").write_text(json.dumps(result, indent=2))
    profile(scaled_u, raw_u, km.labels_).to_csv(out / f"kmeans_baseline_profile_{tag}.csv")
    joblib.dump({"model": km, "scaler": scaler, "feature_names": list(Xdf.columns),
                 "representation": tag, "seed": seed}, art / f"kmeans_baseline_{tag}.joblib")
    _figures(out, tag, X, km, pca, cum, seed, sweep)
    return result


def _figures(out, tag, X, km, pca, cum, seed, sweep):
    if sweep:
        s = pd.read_csv(out / f"kmeans_k_sweep_{tag}.csv")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(s.k, s.inertia, "o-"); ax[0].set(title="Elbow (inertia)", xlabel="k")
        ax[1].plot(s.k, s.silhouette, "o-g"); ax[1].set(title="Silhouette (higher is better)", xlabel="k")
        ax[2].plot(s.k, s.davies_bouldin, "o-r"); ax[2].set(title="Davies-Bouldin (lower is better)", xlabel="k")
        fig.suptitle(f"K-Means k selection, representation {tag}")
        fig.tight_layout(); fig.savefig(out / f"kmeans_k_selection_{tag}.png", dpi=150); plt.close(fig)
    Z = pca.transform(X)[:, :2]
    pick = np.random.RandomState(seed).choice(len(X), min(15000, len(X)), replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(Z[pick, 0], Z[pick, 1], c=km.labels_[pick], cmap="tab10", s=4, alpha=.6)
    plt.title(f"Baseline K-Means (k={km.n_clusters}), {tag}: PCA 2-D ({cum[1]:.0%} variance)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    plt.savefig(out / f"kmeans_baseline_pca2d_{tag}.png", dpi=150); plt.close()
    pd.DataFrame({"component": range(1, len(cum) + 1), "cumulative_variance": cum}).to_csv(
        out / f"pca_explained_variance_{tag}.csv", index=False)


def compare_representations(config=None):
    """Same untuned K-Means on each representation (k sweep skipped)."""
    config = config or load_config()
    rows = [run_baseline(r, config, sweep=False) for r in representations.REPRESENTATIONS]
    table = pd.DataFrame([{k: r[k] for k in ("representation", "n_rows", "n_features", "silhouette",
                                             "davies_bouldin", "calinski_harabasz",
                                             "min_cluster", "max_cluster", "fit_seconds")}
                          for r in rows])
    out, _ = _dirs(config)
    table.round(4).to_csv(out / "representation_comparison.csv", index=False)
    return table
