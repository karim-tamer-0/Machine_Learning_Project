import pytest

from main import build_parser, main
from src.classification.baseline import baseline_models, select_models


def test_parser_reads_command_task_and_model():
    args = build_parser().parse_args(
        ["baseline", "--task", "classification", "--model", "random_forest"])
    assert args.command == "baseline"
    assert args.task == "classification"
    assert args.model == "random_forest"


def test_defaults_are_classification_and_all_models():
    args = build_parser().parse_args(["evaluate"])
    assert args.task == "classification" and args.model == "all"


def test_unimplemented_task_exits_with_a_message():
    with pytest.raises(SystemExit):
        main(["baseline", "--task", "regression"])


def test_unknown_model_exits_with_a_message():
    with pytest.raises(SystemExit):
        select_models(baseline_models(42), "not_a_model")


def test_select_models_all_and_single():
    models = baseline_models(42)
    assert set(select_models(models, "all")) == set(models)
    assert list(select_models(models, "random_forest")) == ["random_forest"]