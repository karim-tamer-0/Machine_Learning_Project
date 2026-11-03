from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.regression.baseline as baseline


def test_regression_metrics_are_exact_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    metrics = baseline.regression_metrics(y, y)
    assert metrics["MAE"] == 0.0
    assert metrics["MSE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0


def test_version_2_has_no_flags_or_padded_names():
    X_train, X_val, y_train, y_val = baseline.load_regression_splits()
    assert X_train.shape == (35548, 50)
    assert X_val.shape == (7620, 50)
    assert len(y_train) == 35548 and len(y_val) == 7620
    assert "HasAgent" not in X_train.columns
    assert "HasCompany" not in X_train.columns
    assert "RequiredCarParkingSpaces" in X_train.columns
    assert list(X_train.columns) == list(X_val.columns)
    assert all(column == column.rstrip() for column in X_train.columns)
    assert not X_train.isna().any().any()
    assert not X_val.isna().any().any()


def test_loader_does_not_read_the_test_split(monkeypatch):
    real_read_csv = pd.read_csv
    seen = []

    def guarded_read_csv(path, *args, **kwargs):
        seen.append(Path(path).name)
        if Path(path).name == "test_regression.csv":
            raise AssertionError("The baseline must not read the test split")
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(baseline.pd, "read_csv", guarded_read_csv)
    baseline.load_regression_splits()
    assert "test_regression.csv" not in seen


def test_model_selection_accepts_the_baseline_only():
    assert baseline.select_model("all") == [baseline.MODEL_NAME]
    assert baseline.select_model("linear_regression") == [baseline.MODEL_NAME]
    with pytest.raises(SystemExit):
        baseline.select_model("random_forest")


def test_pipeline_fits_and_predicts_finite_values():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "a": rng.normal(size=30),
        "b": rng.normal(size=30),
        "c": rng.integers(0, 2, size=30),
    })
    y = 3.0 * X["a"] - 2.0 * X["b"] + 5.0
    pipeline = baseline.build_pipeline().fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == (30,)
    assert np.isfinite(predictions).all()


def test_validation_report_writes_its_outputs(tmp_path):
    rng = np.random.default_rng(7)
    train = pd.DataFrame({
        "a": rng.normal(size=60),
        "b": rng.normal(size=60),
        "ADR": rng.normal(100, 20, size=60),
    })
    val = pd.DataFrame({
        "a": rng.normal(size=15),
        "b": rng.normal(size=15),
        "ADR": rng.normal(100, 20, size=15),
    })
    train.to_csv(tmp_path / "train_regression.csv", index=False)
    val.to_csv(tmp_path / "val_regression.csv", index=False)
    output_dir = tmp_path / "outputs"

    baseline.run_validation_report(data_dir=tmp_path, output_dir=output_dir)

    assert (output_dir / "baseline_validation_metrics.csv").exists()
    assert (output_dir / "linear_regression_predictions_val.csv").exists()
    assert (output_dir / "linear_regression_coefficients.csv").exists()
    assert (output_dir / "linear_regression_residuals.png").exists()
