"""Command-line entry point for the CSE382 hotel booking project.

Milestone 1 commands:
    python main.py baseline --task classification --model all
    python main.py evaluate --task classification --model logistic_regression
    python main.py baseline --task regression --model linear_regression
    python main.py evaluate --task regression --model linear_regression
    python main.py pca --task regression

    python main.py cluster --algorithm kmeans
    python main.py cluster --algorithm kmeans --compare

Agglomerative / GMM / DBSCAN clustering and the train / tune / predict commands are added in
later milestones.
"""
import argparse

TASKS = ["regression", "classification", "clustering"]


def _require_task(args, supported):
    if args.task not in supported:
        supported_text = ", ".join(supported)
        raise SystemExit(
            f"Task '{args.task}' is not implemented for '{args.command}'. "
            f"Supported tasks: {supported_text}."
        )


def cmd_baseline(args):
    """Untuned baselines with cross-validation on the training set."""
    _require_task(args, ["regression", "classification"])
    if args.task == "classification":
        from src.classification.baseline import run_baseline_cv
        run_baseline_cv(args.model)
    else:
        from src.regression.baseline import run_baseline_cv
        run_baseline_cv(args.model)


def cmd_evaluate(args):
    """Fit baselines on train and score them on the validation set."""
    _require_task(args, ["regression", "classification"])
    if args.task == "classification":
        from src.classification.evaluate_baseline import run_validation_report
        run_validation_report(args.model)
    else:
        from src.regression.baseline import run_validation_report
        run_validation_report(args.model)


def cmd_pca(args):
    """Run PCA analysis on the training and validation splits."""
    _require_task(args, ["regression"])
    from src.regression.pca_analysis import run_pca_analysis
    run_pca_analysis()


def cmd_cluster(args):
    """Untuned clustering baseline (Milestone 1: K-Means only)."""
    if args.algorithm != "kmeans":
        raise SystemExit(f"Algorithm '{args.algorithm}' is not implemented yet. Supported: kmeans.")
    from src.clustering.baseline import compare_representations, run_baseline
    if args.compare:
        print(compare_representations().round(4).to_string(index=False))
    else:
        result = run_baseline(args.representation)
        print({k: result[k] for k in ("representation", "n_rows", "n_features",
                                      "silhouette", "davies_bouldin", "min_cluster")})


COMMANDS = {
    "baseline": (cmd_baseline, "untuned baselines with cross-validation (training set)"),
    "evaluate": (cmd_evaluate, "validation-set metrics and figures"),
    "pca": (cmd_pca, "PCA explained variance, projections, and regression comparison"),
}


def build_parser():
    parser = argparse.ArgumentParser(description="CSE382 City Hotel (H2) ML project")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (func, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--task", choices=TASKS, default="classification")
        p.add_argument(
            "--model",
            default="all",
            help=(
                "model name or 'all' "
                "(classification: logistic_regression, random_forest; "
                "regression: linear_regression)"
            ),
        )
        p.set_defaults(func=func)
    cl = sub.add_parser("cluster", help="clustering baseline (K-Means, unsupervised)")
    cl.add_argument("--algorithm", default="kmeans", help="clustering algorithm (kmeans)")
    cl.add_argument("--representation", default="v1",
                    choices=["v1", "v1_dedup", "v2", "v2_no_country"],
                    help="feature matrix (v1 = preprocessing teammate's matrix)")
    cl.add_argument("--compare", action="store_true",
                    help="run the baseline on every representation and compare")
    cl.set_defaults(func=cmd_cluster)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
