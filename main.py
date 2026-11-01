"""Command-line entry point for the CSE382 hotel booking project.

Milestone 1 commands (classification):
    python main.py baseline --task classification --model all
    python main.py evaluate --task classification --model logistic_regression

Other tasks and the train / tune / predict commands are added in later milestones.
"""
import argparse

TASKS = ["regression", "classification", "clustering"]


def _require_classification(args):
    if args.task != "classification":
        raise SystemExit(f"Task '{args.task}' is not implemented yet.")


def cmd_baseline(args):
    """Untuned baselines with 5-fold CV on the training set."""
    _require_classification(args)
    from src.classification.baseline import run_baseline_cv
    run_baseline_cv(args.model)


def cmd_evaluate(args):
    """Fit baselines on train, score them on the validation set, plot confusion matrices."""
    _require_classification(args)
    from src.classification.evaluate_baseline import run_validation_report
    run_validation_report(args.model)


COMMANDS = {
    "baseline": (cmd_baseline, "untuned baselines with cross-validation (training set)"),
    "evaluate": (cmd_evaluate, "validation-set metrics and confusion matrices"),
}


def build_parser():
    parser = argparse.ArgumentParser(description="CSE382 City Hotel (H2) ML project")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (func, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--task", choices=TASKS, default="classification")
        p.add_argument("--model", default="all",
                       help="model name or 'all' (classification: logistic_regression, random_forest)")
        p.set_defaults(func=func)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()