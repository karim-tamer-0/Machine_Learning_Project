"""Create the version 2 regression data.

This script reads every top-level CSV in src/data/regression_data, removes the
constant HasAgent and HasCompany columns, strips trailing whitespace from the
column names, and writes the results to src/data/regression_data/version 2.
The original files are not modified.

Run from the repository root:

    python -m src.Preprocessing.Regression2
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "src" / "data" / "regression_data"
OUTPUT_DIR = INPUT_DIR / "version 2"
FLAG_COLUMNS = ["HasAgent", "HasCompany"]


def clean_columns(df):
    """Drop flag columns and strip trailing whitespace from column names."""
    removed = [column for column in FLAG_COLUMNS if column in df.columns]
    cleaned = df.drop(columns=removed).copy()
    renamed = {column: str(column).rstrip() for column in cleaned.columns}
    cleaned.columns = [renamed[column] for column in cleaned.columns]
    if cleaned.columns.duplicated().any():
        duplicates = cleaned.columns[cleaned.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names after stripping: {duplicates}")
    return cleaned, removed, renamed


def process_file(path):
    """Process one CSV and return a manifest row."""
    df = pd.read_csv(path)
    original_columns = list(df.columns)
    cleaned, removed, renamed = clean_columns(df)
    output_path = OUTPUT_DIR / path.name
    cleaned.to_csv(output_path, index=False)
    stripped = [
        f"{before!r}->{after!r}"
        for before, after in renamed.items()
        if before != after
    ]
    return {
        "input_file": path.name,
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        "rows": len(cleaned),
        "original_columns": len(original_columns),
        "output_columns": len(cleaned.columns),
        "removed_columns": ", ".join(removed),
        "renamed_columns": "; ".join(stripped),
    }


def main():
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Input directory not found: {INPUT_DIR}")

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in csv_files:
        row = process_file(path)
        rows.append(row)
        print(
            f"{row['input_file']}: {row['rows']:,} rows, "
            f"{row['original_columns']} -> {row['output_columns']} columns, "
            f"removed [{row['removed_columns']}]"
        )

    manifest_path = OUTPUT_DIR / "regression2_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Saved: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    main()
