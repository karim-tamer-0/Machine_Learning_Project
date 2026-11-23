"""Hyperparameter tuning for the classification models (Member B, Phase 2).

RandomizedSearchCV with 5-fold stratified CV on the TRAINING set only. The search is run
on the full pipeline, so preprocessing is refitted inside every fold. Every metric is
scored for every candidate, and the search is refitted on the optimisation metric
(PR-AUC by default, set in config.yaml: classification -> tuning -> scoring).

Search spaces live in config.yaml (classification -> tuning -> search_spaces). Parameter
names start with "model__" because the classifier is the pipeline step called "model".

Run from the repo root:
    python -m src.classification.tuning --model random_forest
    python -m src.classification.tuning --model random_forest --n-iter 2   (quick smoke test)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV

from src.classification.baseline import SCORING, baseline_models, select_models
from src.classification.pipeline import build_pipeline
from src.data.loader import PROJECT_ROOT, load_config
from src.data.split import get_cv, load_classification_split


def to_distribution(spec):
    """Turn one search-space entry from config.yaml into something RandomizedSearchCV can sample."""
    kind = spec["type"]
    if kind == "int":
        return randint(spec["low"], spec["high"] + 1)
    if kind == "float":
        if spec.get("log"):
            return loguniform(spec["low"], spec["high"])
        return uniform(spec["low"], spec["high"] - spec["low"])
    if kind == "choice":
        return [tuple(v) if isinstance(v, list) else v for v in spec["values"]]
    raise ValueError(f"Unknown search-space type '{kind}'")


def build_distributions(space):
    return {param: to_distribution(spec) for param, spec in space.items()}


def refit_key(scoring_name):
    """Name used in SCORING for the optimisation metric, e.g. 'average_precision' -> 'pr_auc'."""
    for key, scorer in SCORING.items():
        if scorer == scoring_name:
            return key
    raise ValueError(f"Scoring '{scoring_name}' is not one of {list(SCORING.values())}")


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def tune_model(name, pipeline, X, y, cv, space, n_iter, scoring="average_precision",
               seed=42, feature_set="strict", scaled=True):
    """Random search over `space`. Returns the metrics row (same columns as the baseline
    table), the best parameters, the full CV results and the refitted best pipeline."""
    key = refit_key(scoring)
    search = RandomizedSearchCV(
        pipeline, build_distributions(space), n_iter=n_iter, scoring=SCORING, refit=key,
        cv=cv, random_state=seed, return_train_score=True, n_jobs=1, error_score="raise")
    start = time.perf_counter()
    search.fit(X, y)
    total = time.perf_counter() - start

    i = search.best_index_
    res = search.cv_results_
    row = {"model": name, "config": "tuned", "feature_set": feature_set,
           "preprocessing": "scaled" if scaled else "unscaled"}
    for metric in SCORING:
        row[f"cv_{metric}_mean"] = float(res[f"mean_test_{metric}"][i])
        row[f"cv_{metric}_std"] = float(res[f"std_test_{metric}"][i])
    row["train_f1_mean"] = float(res["mean_train_f1"][i])
    row["fit_time_per_fold_s"] = float(res["mean_fit_time"][i])
    row["total_time_s"] = total
    return {
        "row": row,
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "best_score": float(search.best_score_),
        "scoring": scoring,
        "n_iter": n_iter,
        "search_space": space,
        "cv_results": pd.DataFrame(res),
        "best_estimator": search.best_estimator_,
    }


def save_tuning_result(result, name, out_dir, seed=42):
    """Write <name>_best.json (best parameters, metrics, search space) and
    <name>_cv_results.csv (every candidate tried) to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": name,
        "optimisation_metric": result["scoring"],
        "n_iter": result["n_iter"],
        "seed": seed,
        "best_cv_score": result["best_score"],
        "best_params": result["best_params"],
        "metrics": {k: _jsonable(v) for k, v in result["row"].items()},
        "search_space": result["search_space"],
    }
    json_path = out_dir / f"{name}_best.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result["cv_results"].to_csv(out_dir / f"{name}_cv_results.csv", index=False)
    return json_path


def run_tuning(model_name="all", n_iter=None):
    config = load_config()
    tcfg = config["classification"]["tuning"]
    spaces = tcfg["search_spaces"]
    n_iter = n_iter or tcfg["n_iter"]

    X_train, _, _, y_train, _, _ = load_classification_split("strict", config)
    cv = get_cv(config)
    out_dir = PROJECT_ROOT / config["classification"]["output_dir"] / "tuning"

    # only models that exist in the registry AND have a search space
    available = {n: m for n, m in baseline_models(config["seed"]).items() if n in spaces}
    rows = []
    for name, (model, scaled) in select_models(available, model_name).items():
        pipeline = build_pipeline(model, "strict", scale=scaled, config=config)
        result = tune_model(name, pipeline, X_train, y_train, cv, spaces[name], n_iter,
                            tcfg["scoring"], config["seed"], scaled=scaled)
        path = save_tuning_result(result, name, out_dir, config["seed"])
        rows.append(result["row"])
        print(f"{name}: best {tcfg['scoring']} = {result['best_score']:.4f}")
        print("  best params:", result["best_params"])
        print("  saved:", path)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune classification models")
    parser.add_argument("--model", default="all")
    parser.add_argument("--n-iter", type=int, default=None, help="override n_iter from config.yaml")
    args = parser.parse_args()
    results = run_tuning(args.model, args.n_iter)
    print(results.set_index("model").T.round(4).to_string())