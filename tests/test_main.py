import pytest

from main import build_parser, main
from src.classification.baseline import baseline_models, select_models


def test_parser_reads_command_task_and_model():
    args = build_parser().parse_args(
        ["baseline", "--task", "classification", "--model", "random_forest"])
    assert args.command == "baseline"
    assert args.task == "classification"
    assert args.model == "random_forest"


def test_parser_reads_regression_task_and_model():
    args = build_parser().parse_args(
        ["evaluate", "--task", "regression", "--model", "linear_regression"])
    assert args.command == "evaluate"
    assert args.task == "regression"
    assert args.model == "linear_regression"


def test_defaults_are_classification_and_all_models():
    args = build_parser().parse_args(["evaluate"])
    assert args.task == "classification" and args.model == "all"


def test_unimplemented_task_exits_with_a_message():
    with pytest.raises(SystemExit):
        main(["baseline", "--task", "clustering"])


def test_regression_baseline_is_dispatched(monkeypatch):
    import src.regression.baseline as baseline

    called = {}
    monkeypatch.setattr(baseline, "run_baseline_cv", lambda model: called.update(model=model))
    main(["baseline", "--task", "regression", "--model", "linear_regression"])
    assert called["model"] == "linear_regression"


def test_regression_evaluate_is_dispatched(monkeypatch):
    import src.regression.baseline as baseline

    called = {}
    monkeypatch.setattr(
        baseline, "run_validation_report", lambda model: called.update(model=model))
    main(["evaluate", "--task", "regression", "--model", "linear_regression"])
    assert called["model"] == "linear_regression"


def test_regression_pca_is_dispatched(monkeypatch):
    import src.regression.pca_analysis as pca_analysis

    called = {}
    monkeypatch.setattr(
        pca_analysis, "run_pca_analysis", lambda: called.update(ran=True))
    main(["pca", "--task", "regression"])
    assert called["ran"]


def test_pca_rejects_classification():
    with pytest.raises(SystemExit):
        main(["pca", "--task", "classification"])


def test_unknown_model_exits_with_a_message():
    with pytest.raises(SystemExit):
        select_models(baseline_models(42), "not_a_model")


def test_select_models_all_and_single():
    models = baseline_models(42)
    assert set(select_models(models, "all")) == set(models)
    assert list(select_models(models, "random_forest")) == ["random_forest"]