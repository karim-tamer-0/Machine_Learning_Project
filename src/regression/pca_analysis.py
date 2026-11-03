"""PCA analysis for the ADR regression task on the version 2 data.

The projection is fitted on the training split only. The validation split is
used to inspect the projections and compare full-feature Linear Regression with
PCA-Linear Regression. The test split is never loaded.

Run from the repository root:

    python main.py pca --task regression
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.regression.baseline import (
    build_pipeline as build_baseline_pipeline,
    load_regression_splits,
    regression_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "regression" / "pca"
DEFAULT_VARIANCE_THRESHOLDS = (0.80, 0.85, 0.90, 0.95, 0.99)
DEFAULT_COMPONENT_COUNTS = (2, 3)
RANDOM_STATE = 42


def build_projection_pipeline():
    """Return the unfitted variance, scaling, and full PCA pipeline."""
    return Pipeline([
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=None, random_state=RANDOM_STATE)),
    ])


def build_pca_pipeline(n_components):
    """Return an unfitted PCA-Linear Regression pipeline."""
    return Pipeline([
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("model", LinearRegression()),
    ])


def explained_variance_table(pipeline):
    """Return per-component eigenvalues, variance ratios, and cumulative ratios."""
    pca = pipeline.named_steps["pca"]
    ratios = pca.explained_variance_ratio_
    return pd.DataFrame({
        "component": np.arange(1, len(ratios) + 1),
        "eigenvalue": pca.explained_variance_,
        "explained_variance_ratio": ratios,
        "cumulative_variance_ratio": np.cumsum(ratios),
    })


def component_loadings_table(pipeline, feature_names):
    """Return loadings with features as rows and components as columns."""
    support = pipeline.named_steps["variance"].get_support()
    names = pd.Index(feature_names)[support]
    pca = pipeline.named_steps["pca"]
    columns = [f"PC{index + 1}" for index in range(pca.n_components_)]
    return pd.DataFrame(pca.components_.T, index=names, columns=columns)


def top_loadings_table(loadings, explained, n_components=3, top_n=15):
    """Return the largest absolute loadings for the leading components."""
    rows = []
    for index in range(min(n_components, loadings.shape[1])):
        column = loadings.iloc[:, index]
        for rank, feature in enumerate(
            column.abs().sort_values(ascending=False).head(top_n).index, start=1
        ):
            rows.append({
                "component": loadings.columns[index],
                "rank": rank,
                "feature": feature,
                "loading": column[feature],
                "abs_loading": abs(column[feature]),
                "explained_variance_ratio": explained.iloc[index][
                    "explained_variance_ratio"
                ],
            })
    return pd.DataFrame(rows)


def projection_frame(y_val, projections):
    """Return validation projections with ADR and ADR quartiles."""
    y_val = pd.Series(y_val).reset_index(drop=True)
    quartile_numbers = pd.qcut(
        y_val, q=4, labels=False, duplicates="drop"
    ).astype(int)
    frame = pd.DataFrame({
        "ActualADR": y_val,
        "ADRQuartile": [f"Q{number + 1}" for number in quartile_numbers],
        "PC1": projections[:, 0],
        "PC2": projections[:, 1],
        "PC3": projections[:, 2],
    })
    return frame


def adr_quartile_summary(projections):
    """Summarize the projected validation bookings by ADR quartile."""
    return (
        projections.groupby("ADRQuartile", sort=True)
        .agg(
            n=("ActualADR", "size"),
            adr_min=("ActualADR", "min"),
            adr_mean=("ActualADR", "mean"),
            adr_max=("ActualADR", "max"),
            pc1_mean=("PC1", "mean"),
            pc2_mean=("PC2", "mean"),
            pc3_mean=("PC3", "mean"),
        )
        .reset_index()
    )


def comparison_table(
    X_train,
    y_train,
    X_val,
    y_val,
    variance_thresholds=DEFAULT_VARIANCE_THRESHOLDS,
    component_counts=DEFAULT_COMPONENT_COUNTS,
):
    """Compare full-feature and PCA Linear Regression on the validation split."""
    rows = []

    started = time.perf_counter()
    baseline = build_baseline_pipeline().fit(X_train, y_train)
    predictions = baseline.predict(X_val)
    metrics = regression_metrics(y_val, predictions)
    rows.append({
        "representation": "Full features",
        "n_components": int(baseline.named_steps["variance"].get_support().sum()),
        "cumulative_explained_variance": 1.0,
        "val_MAE": metrics["MAE"],
        "val_RMSE": metrics["RMSE"],
        "val_R2": metrics["R2"],
        "fit_time_s": time.perf_counter() - started,
    })

    for n_components in component_counts:
        started = time.perf_counter()
        pipeline = build_pca_pipeline(n_components).fit(X_train, y_train)
        predictions = pipeline.predict(X_val)
        metrics = regression_metrics(y_val, predictions)
        pca = pipeline.named_steps["pca"]
        rows.append({
            "representation": f"PCA {n_components} components",
            "n_components": int(pca.n_components_),
            "cumulative_explained_variance": pca.explained_variance_ratio_.sum(),
            "val_MAE": metrics["MAE"],
            "val_RMSE": metrics["RMSE"],
            "val_R2": metrics["R2"],
            "fit_time_s": time.perf_counter() - started,
        })

    for threshold in variance_thresholds:
        started = time.perf_counter()
        pipeline = build_pca_pipeline(threshold).fit(X_train, y_train)
        predictions = pipeline.predict(X_val)
        metrics = regression_metrics(y_val, predictions)
        pca = pipeline.named_steps["pca"]
        rows.append({
            "representation": f"PCA {threshold:.0%} variance",
            "n_components": int(pca.n_components_),
            "cumulative_explained_variance": pca.explained_variance_ratio_.sum(),
            "val_MAE": metrics["MAE"],
            "val_RMSE": metrics["RMSE"],
            "val_R2": metrics["R2"],
            "fit_time_s": time.perf_counter() - started,
        })

    return pd.DataFrame(rows)


def save_explained_variance_plot(explained, path):
    """Save the scree plot and cumulative explained-variance plot."""
    components = explained["component"]
    ratios = explained["explained_variance_ratio"] * 100.0
    cumulative = explained["cumulative_variance_ratio"] * 100.0

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(components, ratios, color="#2a78d6")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Scree plot")

    axes[1].plot(components, cumulative, color="#2a78d6", marker="o", markersize=3)
    for level in (80, 90, 95, 99):
        axes[1].axhline(level, color="0.65", linestyle="--", linewidth=0.7)
        axes[1].text(
            len(components), level + 0.8, f"{level}%",
            ha="right", va="bottom", fontsize=8, color="0.35",
        )
    axes[1].set_xlabel("Principal component")
    axes[1].set_ylabel("Cumulative explained variance (%)")
    axes[1].set_title("Cumulative variance")
    axes[1].set_ylim(0, 103)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_loading_plot(loadings, path, top_n=12):
    """Save the leading loadings for PC1 and PC2."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, axis in enumerate(axes):
        column = loadings.iloc[:, index]
        selected = column.abs().sort_values(ascending=False).head(top_n).index
        values = column.loc[selected].sort_values()
        colors = ["#d64545" if value < 0 else "#2a78d6" for value in values]
        axis.barh(values.index, values.values, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Loading")
        axis.set_title(f"{loadings.columns[index]} loadings")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_projection_plots(projections, explained, output_dir):
    """Save 2D and 3D ADR-colored projections plus the ADR-quartile plot."""
    labels = [
        f"PC{index + 1} ({explained.iloc[index]['explained_variance_ratio']:.1%} var)"
        for index in range(3)
    ]
    actual = projections["ActualADR"]
    quartile = projections["ADRQuartile"]

    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        projections["PC1"], projections["PC2"],
        c=actual, cmap="viridis", s=10, alpha=0.65, linewidths=0,
    )
    figure.colorbar(scatter, ax=axis, label="ADR")
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title("Validation bookings in PC1-PC2 space")
    figure.tight_layout()
    figure.savefig(output_dir / "pca_2d_adr.png", dpi=150)
    plt.close(figure)

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        projections["PC1"], projections["PC2"], projections["PC3"],
        c=actual, cmap="viridis", s=6, alpha=0.55, depthshade=False,
    )
    figure.colorbar(scatter, ax=axis, shrink=0.7, pad=0.1, label="ADR")
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_zlabel(labels[2])
    axis.set_title("Validation bookings in PC1-PC2-PC3 space")
    figure.tight_layout()
    figure.savefig(output_dir / "pca_3d_adr.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    colors = {"Q1": "#4575b4", "Q2": "#91bfdb", "Q3": "#fc8d59", "Q4": "#d73027"}
    for group in sorted(quartile.unique()):
        mask = quartile == group
        values = actual[mask]
        axis.scatter(
            projections.loc[mask, "PC1"],
            projections.loc[mask, "PC2"],
            color=colors.get(group, "0.5"),
            s=10,
            alpha=0.65,
            linewidths=0,
            label=f"{group}: ADR {values.min():.0f}-{values.max():.0f}",
        )
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title("Validation bookings by ADR quartile")
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "pca_2d_adr_quartiles.png", dpi=150)
    plt.close(figure)


def run_pca_analysis(
    data_dir=None,
    output_dir=None,
    variance_thresholds=DEFAULT_VARIANCE_THRESHOLDS,
    component_counts=DEFAULT_COMPONENT_COUNTS,
):
    """Fit PCA on train, analyze it, and save tables and figures."""
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val = load_regression_splits(data_dir)
    feature_names = X_train.columns

    projection_pipeline = build_projection_pipeline().fit(X_train)
    explained = explained_variance_table(projection_pipeline)
    loadings = component_loadings_table(projection_pipeline, feature_names)
    top_loadings = top_loadings_table(loadings, explained)

    projections_array = projection_pipeline.transform(X_val)
    projections = projection_frame(y_val, projections_array)

    comparison = comparison_table(
        X_train,
        y_train,
        X_val,
        y_val,
        variance_thresholds=variance_thresholds,
        component_counts=component_counts,
    )

    explained_path = output_dir / "pca_explained_variance.csv"
    loadings_path = output_dir / "pca_component_loadings.csv"
    top_loadings_path = output_dir / "pca_top_loadings.csv"
    projections_path = output_dir / "pca_validation_projections.csv"
    quartile_path = output_dir / "pca_adr_quartile_summary.csv"
    comparison_path = output_dir / "pca_model_comparison.csv"

    explained.to_csv(explained_path, index=False)
    loadings.rename_axis("feature").to_csv(loadings_path)
    top_loadings.to_csv(top_loadings_path, index=False)
    projections.to_csv(projections_path, index=False)
    adr_quartile_summary(projections).to_csv(quartile_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    save_explained_variance_plot(explained, output_dir / "pca_scree.png")
    save_loading_plot(loadings, output_dir / "pca_loadings_pc1_pc2.png")
    save_projection_plots(projections, explained, output_dir)

    print(comparison.round(4).to_string(index=False))
    print(f"\nPCA components for 95% variance: "
          f"{int((explained['cumulative_variance_ratio'] < 0.95).sum()) + 1}")
    for path in (
        explained_path,
        loadings_path,
        top_loadings_path,
        projections_path,
        quartile_path,
        comparison_path,
        output_dir / "pca_scree.png",
        output_dir / "pca_loadings_pc1_pc2.png",
        output_dir / "pca_2d_adr.png",
        output_dir / "pca_3d_adr.png",
        output_dir / "pca_2d_adr_quartiles.png",
    ):
        print("saved:", path)
    return comparison


if __name__ == "__main__":
    run_pca_analysis()
