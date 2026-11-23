"""PCA experiments for the classification task (Member A, Milestone 2, T5).

PCA runs on the scaled, one-hot encoded features, inside the model pipeline:

    BookingFeatures -> scaled preprocessing -> PCA -> model

Everything is fitted on the TRAINING set only. The number of components is chosen from
the training set's explained variance, cross-validation runs on the training set, and the
validation set is only used to compare models with and without PCA. The test set is
never loaded here.

Run from the repo root:   python main.py pca --task classification
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.classification.baseline import baseline_models, run_cv
from src.classification.evaluate_baseline import evaluate_on_validation
from src.classification.pipeline import build_pipeline
from src.data.eda import BLUE, GRAY, ORANGE, _save, plt
from src.data.loader import PROJECT_ROOT, load_config
from src.data.split import get_cv, load_classification_split

COMPARED_MODELS = ["logistic_regression", "gaussian_nb", "xgboost"]
SWEEP_MODEL = "logistic_regression"


def fit_full_pca(X_train, seed=42, config=None):
    """Fit preprocessing and a PCA that keeps every component, on the training set."""
    return build_pipeline(PCA(random_state=seed), "strict", scale=True, config=config).fit(X_train)


def explained_variance_table(pca):
    ratio = pca.explained_variance_ratio_
    return pd.DataFrame({
        "component": np.arange(1, len(ratio) + 1),
        "explained_variance_ratio": ratio,
        "cumulative": np.cumsum(ratio),
    })


def choose_n_components(cumulative, threshold):
    """Fewest components whose cumulative explained variance reaches the threshold."""
    cumulative = np.asarray(cumulative)
    if threshold >= 1.0:
        return len(cumulative)
    return int(np.searchsorted(cumulative, threshold) + 1)


def loadings_table(fitted, n_components=3, top_n=8):
    """Encoded columns with the largest absolute weight on each of the first components."""
    pca = fitted.named_steps["model"]
    names = fitted.named_steps["prep"].get_feature_names_out()
    rows = []
    for i in range(n_components):
        weights = pd.Series(pca.components_[i], index=names)
        for name in weights.abs().sort_values(ascending=False).index[:top_n]:
            rows.append({"component": i + 1, "feature": name, "loading": weights[name]})
    return pd.DataFrame(rows)


def _pipeline(model, scaled, n_components, seed, config):
    """PCA needs scaled inputs, so every PCA pipeline uses the scaled preprocessing.
    n_components=None means no PCA; "all" keeps every component, because a CV fold can
    have one encoded column fewer than the full training set when a rare category is absent."""
    if n_components is None:
        return build_pipeline(model, "strict", scale=scaled, config=config), scaled
    reducer = PCA(n_components=None if n_components == "all" else n_components, random_state=seed)
    return build_pipeline(model, "strict", scale=True, reducer=reducer, config=config), True


def compare_with_and_without_pca(n_components, X_train, y_train, X_val, y_val, cv, config):
    """Untuned models on the original features and on n_components PCA features.
    Returns (CV table on train, validation table)."""
    seed = config["seed"]
    models = baseline_models(seed)
    cv_rows, val_rows = [], []
    for name in COMPARED_MODELS:
        model, scaled = models[name]
        for label, k in (("original", None), (f"pca_{n_components}", n_components)):
            pipeline, used_scaling = _pipeline(model, scaled, k, seed, config)
            row = run_cv(name, pipeline, X_train, y_train, cv, scaled=used_scaling)
            row["representation"] = label
            cv_rows.append(row)
            metrics, _ = evaluate_on_validation(pipeline.fit(X_train, y_train), X_val, y_val)
            val_rows.append({"model": name, "representation": label, **metrics})
            print(f"done: {name} ({label})")
    return pd.DataFrame(cv_rows), pd.DataFrame(val_rows)


def component_sweep(ks, n_total, X_train, y_train, cv, config):
    """CV PR-AUC of one model across several component counts, on the training set.
    The largest count (n_total) is run as "keep every component"."""
    model, _ = baseline_models(config["seed"])[SWEEP_MODEL]
    rows = []
    for k in ks:
        pipeline, _ = _pipeline(model, True, "all" if k == n_total else k, config["seed"], config)
        row = run_cv(SWEEP_MODEL, pipeline, X_train, y_train, cv)
        rows.append({"n_components": k, "cv_pr_auc_mean": row["cv_pr_auc_mean"],
                     "cv_pr_auc_std": row["cv_pr_auc_std"],
                     "fit_time_per_fold_s": row["fit_time_per_fold_s"]})
    return pd.DataFrame(rows)


def plot_explained_variance(table, n_components, threshold, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(table["component"], table["explained_variance_ratio"], color=BLUE, width=0.8,
           label="Each component")
    ax.plot(table["component"], table["cumulative"], color=ORANGE, linewidth=2,
            marker="o", markersize=3, label="Cumulative")
    ax.axhline(threshold, color=GRAY, linestyle="--", linewidth=1)
    ax.axvline(n_components, color=GRAY, linestyle=":", linewidth=1)
    ax.annotate(f"{n_components} components reach {threshold:.0%}",
                xy=(n_components, threshold), xytext=(n_components + 3, threshold - 0.18),
                color=GRAY, fontsize=9, arrowprops={"arrowstyle": "-", "color": GRAY})
    ax.set_xlim(0, len(table) + 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Principal component (of {len(table)} encoded columns)")
    ax.set_ylabel("Share of variance explained")
    ax.legend(loc="center right", frameon=False)
    ax.set_title("PCA explained variance (training set)")
    _save(fig, out_dir, "explained_variance.png")


def plot_comparison(cv_table, n_components, out_dir):
    models = COMPARED_MODELS
    pca_label = f"pca_{n_components}"
    scores = cv_table.pivot(index="model", columns="representation", values="cv_pr_auc_mean")
    scores = scores.loc[models]
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for offset, label, color, text in ((-0.2, "original", BLUE, "Original features"),
                                       (0.2, pca_label, ORANGE, f"PCA ({n_components} components)")):
        ax.barh(y + offset, scores[label], height=0.36, color=color, label=text)
        for i, value in enumerate(scores[label]):
            ax.text(value + 0.005, y[i] + offset, f"{value:.3f}", va="center", color=GRAY, fontsize=9)
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("CV PR-AUC (5-fold, training set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.set_title("Untuned models with and without PCA")
    _save(fig, out_dir, "pca_comparison.png")


def plot_sweep(sweep, baseline_score, out_dir):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.errorbar(sweep["n_components"], sweep["cv_pr_auc_mean"], yerr=sweep["cv_pr_auc_std"],
                color=BLUE, marker="o", markersize=6, linewidth=2, capsize=3,
                label="With PCA")
    ax.axhline(baseline_score, color=GRAY, linestyle="--", linewidth=1,
               label=f"Original features ({baseline_score:.3f})")
    for k, score in zip(sweep["n_components"], sweep["cv_pr_auc_mean"]):
        ax.annotate(f"{score:.3f}", (k, score), textcoords="offset points", xytext=(8, -14),
                    ha="left", color=GRAY, fontsize=9)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("CV PR-AUC")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(f"{SWEEP_MODEL}: CV PR-AUC by number of components")
    _save(fig, out_dir, "pca_component_sweep.png")


def run_pca_analysis():
    config = load_config()
    settings = config["classification"]["pca"]
    out_dir = PROJECT_ROOT / config["classification"]["output_dir"] / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, _, y_train, y_val, _ = load_classification_split("strict", config)
    cv = get_cv(config)

    fitted = fit_full_pca(X_train, config["seed"], config)
    table = explained_variance_table(fitted.named_steps["model"])
    threshold = settings["variance_threshold"]
    n_components = choose_n_components(table["cumulative"], threshold)
    table.to_csv(out_dir / "explained_variance.csv", index=False)
    loadings_table(fitted).to_csv(out_dir / "top_loadings.csv", index=False)
    plot_explained_variance(table, n_components, threshold, out_dir)
    print(f"{len(table)} encoded columns; {n_components} components explain "
          f"{table['cumulative'].iloc[n_components - 1]:.1%} of the variance")

    ks = sorted({choose_n_components(table["cumulative"], t) for t in settings["sweep_thresholds"]})
    sweep = component_sweep(ks, len(table), X_train, y_train, cv, config)
    sweep.to_csv(out_dir / "component_sweep_cv.csv", index=False)

    cv_table, val_table = compare_with_and_without_pca(
        n_components, X_train, y_train, X_val, y_val, cv, config)
    cv_table.to_csv(out_dir / "pca_comparison_cv.csv", index=False)
    val_table.to_csv(out_dir / "pca_comparison_validation.csv", index=False)

    original = cv_table.query("model == @SWEEP_MODEL and representation == 'original'")
    plot_sweep(sweep, original["cv_pr_auc_mean"].iloc[0], out_dir)
    plot_comparison(cv_table, n_components, out_dir)

    columns = ["model", "representation", "cv_pr_auc_mean", "cv_roc_auc_mean", "cv_f1_mean",
               "cv_recall_mean", "fit_time_per_fold_s"]
    print(sweep.round(4).to_string(index=False))
    print(cv_table[columns].round(4).to_string(index=False))
    print(val_table.round(4).to_string(index=False))
    print("saved:", out_dir)
    return table, sweep, cv_table, val_table


if __name__ == "__main__":
    run_pca_analysis()
