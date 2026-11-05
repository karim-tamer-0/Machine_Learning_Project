"""Create the duplicate-free version 3 regression data.

Version 3 starts from the version 2 cleaned table, normalizes categorical text,
applies a fixed country grouping, removes exact duplicate modeling rows after
grouping, and only then creates the train/validation/test split. Version 2 is
preserved unchanged.

Run from the repository root:

    python -m src.Preprocessing.Regression3
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    PROJECT_ROOT / "src" / "data" / "regression_data" / "version 2"
    / "regression_cleaned.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "regression_data" / "version 3"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "regression_v3_preprocessor.joblib"
TOP_COUNTRIES = (
    "PRT", "FRA", "GBR", "DEU", "ESP",
    "ITA", "BEL", "BRA", "NLD", "USA",
)
CATEGORICAL_COLUMNS = (
    "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "DepositType", "CustomerType",
)
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.1765


def load_source():
    if not SOURCE_PATH.exists():
        raise SystemExit(
            f"Version 2 cleaned regression data not found: {SOURCE_PATH}. "
            "Run: python -m src.Preprocessing.Regression2"
        )
    frame = pd.read_csv(SOURCE_PATH)
    if "ADR" not in frame.columns:
        raise SystemExit("ADR is missing from the version 2 cleaned table.")
    if "HasAgent" in frame.columns or "HasCompany" in frame.columns:
        raise SystemExit("Version 2 cleaned table still contains flag columns.")
    if frame.isna().any().any():
        raise SystemExit("Version 2 cleaned table contains missing values.")
    return frame


def normalize_categories(frame):
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    for column in CATEGORICAL_COLUMNS:
        normalized[column] = normalized[column].str.strip()
    return normalized


def group_countries(frame):
    grouped = frame.copy()
    grouped["Country"] = grouped["Country"].where(
        grouped["Country"].isin(TOP_COUNTRIES), "Other"
    )
    return grouped


def split_frame(frame):
    X = frame.drop(columns="ADR")
    y = frame["ADR"]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_preprocessor(X_train):
    categorical = list(CATEGORICAL_COLUMNS)
    numerical = [column for column in X_train.columns if column not in categorical]
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[categorical])
    scaler = StandardScaler()
    scaler.fit(X_train[numerical])
    return encoder, scaler, numerical, categorical


def transform_split(X_split, y_split, encoder, scaler, numerical, categorical):
    numeric_frame = pd.DataFrame(
        scaler.transform(X_split[numerical]),
        columns=numerical,
        index=X_split.index,
    )
    categorical_frame = pd.DataFrame(
        encoder.transform(X_split[categorical]),
        columns=encoder.get_feature_names_out(categorical),
        index=X_split.index,
    )
    result = pd.concat([numeric_frame, categorical_frame], axis=1)
    result["ADR"] = y_split.to_numpy()
    return result


def assert_no_duplicates(train, val, test):
    frames = {"train": train, "validation": val, "test": test}
    for left_name, right_name in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = frames[left_name].merge(frames[right_name], how="inner")
        if not overlap.empty:
            raise SystemExit(
                f"Duplicate rows found between {left_name} and {right_name}: "
                f"{len(overlap)}"
            )
    for name, frame in frames.items():
        duplicate_count = int(frame.duplicated().sum())
        if duplicate_count:
            raise SystemExit(f"{name} contains {duplicate_count} duplicate rows.")


def main():
    source = normalize_categories(load_source())
    grouped = group_countries(source)
    deduplicated = grouped.drop_duplicates().reset_index(drop=True)
    removed_duplicates = len(grouped) - len(deduplicated)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deduplicated.to_csv(OUTPUT_DIR / "regression_cleaned.csv", index=False)

    X_train, X_val, X_test, y_train, y_val, y_test = split_frame(deduplicated)
    encoder, scaler, numerical, categorical = fit_preprocessor(X_train)

    train = transform_split(X_train, y_train, encoder, scaler, numerical, categorical)
    val = transform_split(X_val, y_val, encoder, scaler, numerical, categorical)
    test = transform_split(X_test, y_test, encoder, scaler, numerical, categorical)

    assert_no_duplicates(train, val, test)

    train.to_csv(OUTPUT_DIR / "train_regression.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val_regression.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_regression.csv", index=False)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "encoder": encoder,
            "top_10_countries": list(TOP_COUNTRIES),
        },
        ARTIFACT_PATH,
    )

    manifest = pd.DataFrame([{
        "source_file": str(SOURCE_PATH.relative_to(PROJECT_ROOT)),
        "source_rows": len(source),
        "grouped_rows_before_deduplication": len(grouped),
        "rows_after_deduplication": len(deduplicated),
        "removed_duplicate_rows": removed_duplicates,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "columns": len(train.columns),
        "top_countries": ", ".join(TOP_COUNTRIES),
        "random_state": RANDOM_STATE,
        "artifact": str(ARTIFACT_PATH.relative_to(PROJECT_ROOT)),
    }])
    manifest_path = OUTPUT_DIR / "regression3_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("Version 3 shapes:")
    print("  cleaned:", deduplicated.shape)
    print("  train:", train.shape)
    print("  validation:", val.shape)
    print("  test:", test.shape)
    print("  removed duplicate rows:", removed_duplicates)
    print("saved:", OUTPUT_DIR)
    print("saved:", manifest_path)
    print("saved:", ARTIFACT_PATH)


if __name__ == "__main__":
    main()
