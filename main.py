"""Command-line entry point for the CSE382 hotel booking project.

Milestone 1 commands:
    python main.py baseline --task classification --model all
    python main.py evaluate --task classification --model logistic_regression
    python main.py baseline --task regression --model linear_regression
    python main.py evaluate --task regression --model linear_regression
    python main.py pca --task regression

Clustering and the train / tune / predict commands are added in later milestones.
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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
