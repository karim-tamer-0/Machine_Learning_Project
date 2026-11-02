"""Untuned Linear Regression baseline on the version 2 regression splits.

The baseline is fitted on the training split and evaluated with five-fold
cross-validation on the training split and on the validation split. The final
test split is not loaded here.

Run from the repository root:

    python main.py baseline --task regression --model linear_regression
    python main.py evaluate --task regression --model linear_regression
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "data" / "regression_data" / "version 2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "regression"
MODEL_NAME = "linear_regression"
SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}


def load_regression_splits(data_dir=None):
    """Load the version 2 train and validation splits. The test split is not read."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    train_path = data_dir / "train_regression.csv"
    val_path = data_dir / "val_regression.csv"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            f"Version 2 regression splits not found in {data_dir}. "
            "Run: python -m src.Preprocessing.Regression2"
        )

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    for name, frame in (("train", train), ("validation", val)):
        if "ADR" not in frame.columns:
            raise SystemExit(f"ADR is missing from the {name} split.")
        if frame.isna().any().any():
            raise SystemExit(f"The {name} split contains missing values.")

    train = train.rename(columns=lambda column: str(column).strip())
    val = val.rename(columns=lambda column: str(column).strip())
    X_train = train.drop(columns="ADR")
    y_train = train["ADR"]
    X_val = val.drop(columns="ADR")
    y_val = val["ADR"]

    if list(X_train.columns) != list(X_val.columns):
        raise SystemExit("Train and validation feature columns do not match.")
    return X_train, X_val, y_train, y_val


def build_pipeline():
    """Return the unfitted baseline pipeline."""
    return Pipeline([
        ("variance", VarianceThreshold(threshold=0.0)),
        ("model", LinearRegression()),
    ])


def regression_metrics(y_true, y_pred):
    """Return MAE, MSE, RMSE and R-squared."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def select_model(model_name):
    """Return the requested regression model names."""
    if model_name == "all":
        return [MODEL_NAME]
    if model_name != MODEL_NAME:
        raise SystemExit(
            f"Unknown regression model '{model_name}'. "
            f"Choose from: all, {MODEL_NAME}"
        )
    return [MODEL_NAME]


def run_baseline_cv(model_name="all", data_dir=None, output_dir=None):
    """Run five-fold CV on the training split and save the metrics table."""
    select_model(model_name)
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    X_train, _, y_train, _ = load_regression_splits(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    started = time.perf_counter()
    result = cross_validate(
        build_pipeline(),
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=1,
    )
    row = {
        "model": MODEL_NAME,
        "config": "baseline (untuned)",
        "feature_set": "version 2 (flags removed, names stripped)",
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "cv_folds": cv.get_n_splits(),
    }
    for metric in SCORING:
        values = result[f"test_{metric}"]
        if metric != "R2":
            values = -values
        row[f"cv_{metric}_mean"] = values.mean()
        row[f"cv_{metric}_std"] = values.std()
    row["train_R2_mean"] = result["train_R2"].mean()
    row["fit_time_per_fold_s"] = result["fit_time"].mean()
    row["total_time_s"] = time.perf_counter() - started

    table = pd.DataFrame([row])
    path = output_dir / "baseline_cv_metrics.csv"
    table.to_csv(path, index=False)
    print(table.set_index("model").T.round(4).to_string())
    print("saved:", path)
    return table


def run_validation_report(model_name="all", data_dir=None, output_dir=None):
    """Fit on train, evaluate on validation, and save predictions and diagnostics."""
    select_model(model_name)
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    X_train, X_val, y_train, y_val = load_regression_splits(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline()
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    train_metrics = regression_metrics(y_train, train_pred)
    val_metrics = regression_metrics(y_val, val_pred)
    row = {
        "model": MODEL_NAME,
        "config": "baseline (untuned)",
        "feature_set": "version 2 (flags removed, names stripped)",
        "n_features": int(pipeline.named_steps["variance"].get_support().sum()),
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "fit_time_s": fit_time,
    }
    row.update({f"train_{metric}": value for metric, value in train_metrics.items()})
    row.update({f"val_{metric}": value for metric, value in val_metrics.items()})

    table = pd.DataFrame([row])
    metrics_path = output_dir / "baseline_validation_metrics.csv"
    table.to_csv(metrics_path, index=False)

    predictions = pd.DataFrame({
        "ActualADR": y_val.to_numpy(),
        "PredictedADR": val_pred,
        "Residual": y_val.to_numpy() - val_pred,
    })
    predictions_path = output_dir / "linear_regression_predictions_val.csv"
    predictions.to_csv(predictions_path, index=False)

    support = pipeline.named_steps["variance"].get_support()
    coefficients = pd.Series(
        pipeline.named_steps["model"].coef_,
        index=X_train.columns[support],
        name="coefficient",
    ).sort_values(key=lambda values: values.abs(), ascending=False)
    coefficients_path = output_dir / "linear_regression_coefficients.csv"
    coefficients.rename_axis("feature").reset_index().to_csv(coefficients_path, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(val_pred, predictions["Residual"], s=8, alpha=0.35)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Predicted ADR")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Validation residuals")
    axes[1].hist(predictions["Residual"], bins=50, color="#2a78d6")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Bookings")
    axes[1].set_title("Residual distribution")
    figure.tight_layout()
    plot_path = output_dir / "linear_regression_residuals.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(table.set_index("model").T.round(4).to_string())
    print("saved:", metrics_path)
    print("saved:", predictions_path)
    print("saved:", coefficients_path)
    print("saved:", plot_path)
    return table


if __name__ == "__main__":
    run_baseline_cv("all")
