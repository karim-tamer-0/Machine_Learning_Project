import pandas as pd

from src.data.loader import load_all_models, load_config


def _standardize_text(df):
    """Strip the PMS padding and name the NULL category, which means 'not applicable'."""
    df = df.copy()
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip()
    df["Agent"] = df["Agent"].replace("NULL", "NoAgent")
    df["Company"] = df["Company"].replace("NULL", "NoCompany")
    return df


def clean_classification(df, config=None):
    """Clean the shared dataset for classification. Run this before the train/test split.

    Every rule is row-level with fixed limits, so nothing is learned from the data.
    Returns the cleaned DataFrame and a table of the rows removed by each rule.
    """
    config = config or load_config()
    rules = config["classification"]["cleaning"]
    limits = rules["outlier_limits"]

    df = _standardize_text(df)
    steps = [("start: All_models_data.csv", 0, len(df))]

    def apply(name, mask):
        nonlocal df
        before = len(df)
        df = df[mask]
        steps.append((name, before - len(df), len(df)))

    if rules["drop_duplicates"]:
        apply("exact duplicate rows", ~df.duplicated())
    apply("no adult on the booking", df["Adults"] > 0)
    apply(f"ADR > {limits['ADR']}", df["ADR"] <= limits["ADR"])
    apply(f"LeadTime > {limits['LeadTime']}", df["LeadTime"] <= limits["LeadTime"])
    total_nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    apply(f"total nights > {limits['TotalNights']}", total_nights <= limits["TotalNights"])
    apply(f"Babies > {limits['Babies']}", df["Babies"] <= limits["Babies"])

    report = pd.DataFrame(steps, columns=["rule", "rows_removed", "rows_left"])
    return df.reset_index(drop=True), report


if __name__ == "__main__":
    cleaned, report = clean_classification(load_all_models())
    print(report.to_string(index=False))
    print(f"Cancel rate: {cleaned['IsCanceled'].mean():.3f}")
