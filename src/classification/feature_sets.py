"""Strict vs relaxed feature sets for the classification task (Member A, Milestone 2, T7).

The strict list holds only columns known when the booking is created. The relaxed list adds
four borderline columns (DepositType, TotalOfSpecialRequests, Country, ArrivalDateYear).
Running the same untuned models (Logistic Regression, Gaussian Naive Bayes, XGBoost) on
both lists shows how much the borderline columns raise
the scores. Each borderline column is also added to strict on its own, to show which one
is responsible. A large gain is evidence of leakage, not of a better model.

Both lists use the same rows in train, validation and test, because the split depends only
on the seed and the target. Models are fitted on the training set (5-fold CV) and checked
on the validation set. The test set is never loaded here.

Run from the repo root:   python -m src.classification.feature_sets
"""
import numpy as np
import pandas as pd

from src.classification.baseline import baseline_models, run_cv
from src.classification.evaluate_baseline import evaluate_on_validation
from src.classification.pipeline import build_pipeline
from src.data.eda import BLUE, GRAY, ORANGE, _save, plt
from src.data.loader import PROJECT_ROOT, load_config
from src.data.split import get_cv, load_classification_split

COMPARED_MODELS = ["logistic_regression", "gaussian_nb", "xgboost"]
ABLATION_MODEL = "xgboost"


def borderline_columns(config):
    extra = config["classification"]["features"]["relaxed_extra"]
    return extra["numeric"] + extra["categorical"]


def evaluate_feature_set(feature_set, models, cv, config):
    """CV on train and validation scores for every model on one feature set."""
    X_train, X_val, _, y_train, y_val, _ = load_classification_split(feature_set, config)
    cv_rows, val_rows = [], []
    for name, (model, scaled) in models.items():
        pipeline = build_pipeline(model, feature_set, scale=scaled, config=config)
        cv_rows.append(run_cv(name, pipeline, X_train, y_train, cv,
                              feature_set=feature_set, scaled=scaled))
        metrics, _ = evaluate_on_validation(pipeline.fit(X_train, y_train), X_val, y_val)
        val_rows.append({"model": name, "feature_set": feature_set, **metrics})
        print(f"done: {name} ({feature_set})")
    return cv_rows, val_rows


def gap_table(cv_table, metric="cv_pr_auc_mean"):
    """Relaxed minus strict for every model, on one metric."""
    scores = cv_table.pivot(index="model", columns="feature_set", values=metric)
    scores["gain"] = scores["relaxed"] - scores["strict"]
    return scores[["strict", "relaxed", "gain"]].sort_values("gain", ascending=False)


def ablation_table(cv_table, strict_score, metric="cv_pr_auc_mean"):
    """Gain of each strict+<Column> run over strict alone."""
    table = cv_table[["feature_set", metric]].copy()
    table["column"] = table["feature_set"].str.split("+", n=1).str[1]
    table["gain"] = table[metric] - strict_score
    return table[["column", metric, "gain"]].sort_values("gain", ascending=False)


def plot_strict_vs_relaxed(gaps, out_dir):
    models = list(gaps.index)
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(models) + 1.6))
    for offset, column, color, label in ((-0.2, "strict", BLUE, "Strict (20 features)"),
                                         (0.2, "relaxed", ORANGE, "Relaxed (24 features)")):
        ax.barh(y + offset, gaps[column], height=0.36, color=color, label=label)
        for i, value in enumerate(gaps[column]):
            ax.text(value + 0.005, y[i] + offset, f"{value:.3f}", va="center", color=GRAY, fontsize=9)
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("CV PR-AUC (5-fold, training set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    ax.set_title("Untuned models on strict vs relaxed features")
    _save(fig, out_dir, "strict_vs_relaxed.png")


def plot_ablation(ablation, out_dir):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(ablation["column"], ablation["gain"], color=ORANGE, height=0.6)
    for i, gain in enumerate(ablation["gain"]):
        ax.text(gain + 0.003, i, f"+{gain:.3f}", va="center", color=GRAY, fontsize=9)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.invert_yaxis()
    ax.set_xlim(min(0, ablation["gain"].min()) - 0.01, ablation["gain"].max() + 0.06)
    ax.set_xlabel(f"Gain in CV PR-AUC over strict ({ABLATION_MODEL}, untuned)")
    ax.set_title("Gain from adding each borderline column to strict")
    _save(fig, out_dir, "borderline_column_gain.png")


def run_feature_set_comparison():
    config = load_config()
    out_dir = PROJECT_ROOT / config["classification"]["output_dir"] / "feature_sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = get_cv(config)
    registry = baseline_models(config["seed"])
    models = {name: registry[name] for name in COMPARED_MODELS}

    cv_rows, val_rows = [], []
    for feature_set in ("strict", "relaxed"):
        rows = evaluate_feature_set(feature_set, models, cv, config)
        cv_rows += rows[0]
        val_rows += rows[1]
    cv_table, val_table = pd.DataFrame(cv_rows), pd.DataFrame(val_rows)
    cv_table.to_csv(out_dir / "strict_vs_relaxed_cv.csv", index=False)
    val_table.to_csv(out_dir / "strict_vs_relaxed_validation.csv", index=False)

    ablation_rows = []
    single = {ABLATION_MODEL: models[ABLATION_MODEL]}
    for column in borderline_columns(config):
        ablation_rows += evaluate_feature_set(f"strict+{column}", single, cv, config)[0]
    strict_score = cv_table.query("model == @ABLATION_MODEL and feature_set == 'strict'")
    ablation = ablation_table(pd.DataFrame(ablation_rows),
                              strict_score["cv_pr_auc_mean"].iloc[0])
    ablation.to_csv(out_dir / "borderline_column_gain_cv.csv", index=False)

    gaps = gap_table(cv_table)
    gaps.to_csv(out_dir / "strict_vs_relaxed_gain.csv")
    plot_strict_vs_relaxed(gaps, out_dir)
    plot_ablation(ablation, out_dir)

    print(gaps.round(4).to_string())
    print(ablation.round(4).to_string(index=False))
    print(val_table.round(4).to_string(index=False))
    print("saved:", out_dir)
    return cv_table, val_table, ablation


if __name__ == "__main__":
    run_feature_set_comparison()
