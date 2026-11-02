from pathlib import Path

import numpy as np
import pandas as pd

import src.regression.baseline as baseline
import src.regression.pca_analysis as pca_analysis


def _synthetic_split(n_rows=120, seed=42):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        rng.normal(size=(n_rows, 8)),
        columns=[f"feature_{index}" for index in range(8)],
    )
    target = (
        3.0 * features["feature_0"]
        - 2.0 * features["feature_1"]
        + 0.5 * features["feature_2"]
        + rng.normal(scale=0.1, size=n_rows)
    )
    features["ADR"] = target
    return features


def test_projection_pipeline_reports_all_components():
    train = _synthetic_split()
    X_train = train.drop(columns="ADR")

    pipeline = pca_analysis.build_projection_pipeline().fit(X_train)
    explained = pca_analysis.explained_variance_table(pipeline)
    loadings = pca_analysis.component_loadings_table(pipeline, X_train.columns)

    assert len(explained) == X_train.shape[1]
    assert np.isclose(explained["cumulative_variance_ratio"].iloc[-1], 1.0)
    assert loadings.shape == (X_train.shape[1], X_train.shape[1])
    assert list(loadings.columns[:2]) == ["PC1", "PC2"]


def test_pca_pipeline_fits_and_predicts_finite_values():
    train = _synthetic_split()
    val = _synthetic_split(seed=7)
    X_train = train.drop(columns="ADR")
    y_train = train["ADR"]
    X_val = val.drop(columns="ADR")

    pipeline = pca_analysis.build_pca_pipeline(0.90).fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    assert predictions.shape == (len(X_val),)
    assert np.isfinite(predictions).all()
    assert 0 < pipeline.named_steps["pca"].n_components_ <= X_train.shape[1]


def test_run_pca_analysis_writes_tables_and_figures_without_test_split(
    tmp_path, monkeypatch
):
    train = _synthetic_split()
    val = _synthetic_split(seed=7)
    train.to_csv(tmp_path / "train_regression.csv", index=False)
    val.to_csv(tmp_path / "val_regression.csv", index=False)
    output_dir = tmp_path / "outputs"

    real_read_csv = pd.read_csv
    seen = []

    def guarded_read_csv(path, *args, **kwargs):
        seen.append(Path(path).name)
        if Path(path).name == "test_regression.csv":
            raise AssertionError("PCA analysis must not read the test split")
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(baseline.pd, "read_csv", guarded_read_csv)

    comparison = pca_analysis.run_pca_analysis(
        data_dir=tmp_path,
        output_dir=output_dir,
        variance_thresholds=(0.80, 0.95),
        component_counts=(2, 3),
    )

    assert "test_regression.csv" not in seen
    assert len(comparison) == 5
    assert comparison["representation"].iloc[0] == "Full features"
    assert (output_dir / "pca_explained_variance.csv").exists()
    assert (output_dir / "pca_component_loadings.csv").exists()
    assert (output_dir / "pca_top_loadings.csv").exists()
    assert (output_dir / "pca_validation_projections.csv").exists()
    assert (output_dir / "pca_adr_quartile_summary.csv").exists()
    assert (output_dir / "pca_model_comparison.csv").exists()
    assert (output_dir / "pca_scree.png").exists()
    assert (output_dir / "pca_loadings_pc1_pc2.png").exists()
    assert (output_dir / "pca_2d_adr.png").exists()
    assert (output_dir / "pca_3d_adr.png").exists()
    assert (output_dir / "pca_2d_adr_quartiles.png").exists()
