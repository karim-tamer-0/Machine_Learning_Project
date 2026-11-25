"""Plateau check for XGBoost tuning (Member A, Milestone 2).

The regular search tried 25 settings. This one tries 60 in a wider space (more trees,
deeper trees, lower learning rates, and two extra regularisation settings: reg_alpha and
gamma). If the best score barely moves, tuning has reached a plateau: the model is limited
by the information in the features, not by its settings.

Same protocol as the regular search: 5-fold CV on the TRAINING set, optimising PR-AUC.
Both best settings are then refitted on train and scored once on the VALIDATION set.
The test set is never loaded here.

Run from the repo root:   python -m src.classification.xgboost_plateau
"""
import json

import numpy as np
import pandas as pd

from src.classification.baseline import baseline_models
from src.classification.evaluate_baseline import evaluate_on_validation
from src.classification.pipeline import build_pipeline
from src.classification.tuning import save_tuning_result, tune_model
from src.data.eda import BLUE, GRAY, _save, plt
from src.data.loader import PROJECT_ROOT, load_config
from src.data.split import get_cv, load_classification_split

NAME = "xgboost_wide"


def running_best(scores):
    """Best score found so far after each trial."""
    return np.maximum.accumulate(np.asarray(scores, dtype=float))


def tuned_xgboost_pipeline(params, config):
    """Untuned XGBoost from the registry, set to the given best parameters."""
    model, scaled = baseline_models(config["seed"])["xgboost"]
    return build_pipeline(model, "strict", scale=scaled, config=config).set_params(**params)


def plot_plateau(trial_scores, original_best, original_std, n_original, out_dir):
    trials = np.arange(1, len(trial_scores) + 1)
    best = running_best(trial_scores)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axhspan(original_best - original_std, original_best + original_std, color=GRAY, alpha=0.12, lw=0)
    ax.axhline(original_best, color=GRAY, linestyle="--", linewidth=1)
    ax.text(len(trials) + 0.5, original_best, f"first search: {original_best:.3f}\n"
            f"(band = ± 1 fold std)", va="center", color=GRAY, fontsize=9)
    ax.scatter(trials, trial_scores, s=14, color=GRAY, alpha=0.6, label="Each setting tried")
    ax.plot(trials, best, color=BLUE, linewidth=2, label="Best so far")
    ax.text(len(trials), best[-1] - 0.004, f"best: {best[-1]:.3f}", ha="right", va="top",
            color=BLUE, fontsize=9)
    ax.set_xlim(0, len(trials) + 14)
    ax.set_ylim(max(0, min(trial_scores) - 0.02), max(best[-1], original_best + original_std) + 0.02)
    ax.set_xlabel(f"Setting tried (wider search, {len(trials)} trials; first search had {n_original})")
    ax.set_ylabel("CV PR-AUC (5-fold, training set)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("XGBoost tuning has plateaued: a wider, longer search finds the same score")
    _save(fig, out_dir, "xgboost_plateau.png")


def run_plateau_check():
    config = load_config()
    settings = config["classification"]["plateau_check"]
    out_dir = PROJECT_ROOT / config["classification"]["output_dir"] / "tuning"
    seed = config["seed"]

    X_train, X_val, _, y_train, y_val, _ = load_classification_split("strict", config)
    model, scaled = baseline_models(seed)["xgboost"]
    pipeline = build_pipeline(model, "strict", scale=scaled, config=config)
    result = tune_model(NAME, pipeline, X_train, y_train, get_cv(config), settings["search_space"],
                        settings["n_iter"], config["classification"]["tuning"]["scoring"],
                        seed, scaled=scaled)
    save_tuning_result(result, NAME, out_dir, seed)

    original = json.loads((out_dir / "xgboost_best.json").read_text(encoding="utf-8"))
    original_cv = pd.read_csv(out_dir / "xgboost_cv_results.csv")
    rows = []
    for label, params, cv_mean, cv_std, n in (
            ("first search", original["best_params"], original["metrics"]["cv_pr_auc_mean"],
             original["metrics"]["cv_pr_auc_std"], original["n_iter"]),
            ("wider search", result["best_params"], result["row"]["cv_pr_auc_mean"],
             result["row"]["cv_pr_auc_std"], settings["n_iter"])):
        fitted = tuned_xgboost_pipeline(params, config).fit(X_train, y_train)
        metrics, _ = evaluate_on_validation(fitted, X_val, y_val)
        rows.append({"search": label, "n_trials": n, "cv_pr_auc_mean": cv_mean, "cv_pr_auc_std": cv_std,
                     **{f"val_{k}": v for k, v in metrics.items()}})
    summary = pd.DataFrame(rows)
    summary["cv_gain_over_first"] = summary["cv_pr_auc_mean"] - summary["cv_pr_auc_mean"].iloc[0]
    summary.to_csv(out_dir / "xgboost_plateau_summary.csv", index=False)

    trial_scores = result["cv_results"]["mean_test_pr_auc"].to_numpy()
    plot_plateau(trial_scores, original["metrics"]["cv_pr_auc_mean"],
                 original["metrics"]["cv_pr_auc_std"], len(original_cv), out_dir)

    print("best params:", result["best_params"])
    print(summary.round(4).T.to_string())
    print("saved:", out_dir)
    return summary


if __name__ == "__main__":
    run_plateau_check()
