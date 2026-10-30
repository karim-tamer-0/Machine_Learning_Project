from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.data.cleaning import _standardize_text, clean_classification
from src.data.loader import (PROJECT_ROOT, data_quality_summary, load_all_models,
                             load_config, load_raw)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#52514e"
SURFACE = "#fcfcfb"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": GRAY, "text.color": "#0b0b0b",
    "xtick.color": GRAY, "ytick.color": GRAY, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=150)
    plt.close(fig)


def _rate_bars(rates, overall, title, xlabel, out_dir, name):
    """Horizontal cancel-rate bars, labelled with the rate and the group size."""
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rates) + 1.6))
    labels = rates.index.astype(str)
    ax.barh(labels, rates["mean"] * 100, color=BLUE, height=0.65)
    for i, (rate, size) in enumerate(zip(rates["mean"], rates["size"])):
        ax.text(rate * 100 + 1, i, f"{rate:.0%}  (n={size:,})", va="center", color=GRAY, fontsize=9,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1}, zorder=3)
    ax.axvline(overall * 100, color=GRAY, linestyle="--", linewidth=1, zorder=2)
    ax.set_xlim(0, 125)
    ax.invert_yaxis()
    ax.set_xlabel("Cancel rate (%)")
    ax.set_ylabel(xlabel)
    ax.set_title(f"{title} (dashed line: overall {overall:.0%})")
    _save(fig, out_dir, name)


def plot_class_balance(before, after, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3))
    labels = [f"Shared data\n(n={len(before):,})", f"Cleaned data\n(n={len(after):,})"]
    rates = [before["IsCanceled"].mean(), after["IsCanceled"].mean()]
    ax.barh(labels, [1 - r for r in rates], color=BLUE, height=0.6, label="Not canceled")
    ax.barh(labels, rates, left=[1 - r for r in rates], color=ORANGE, height=0.6, label="Canceled")
    for i, r in enumerate(rates):
        ax.text(1 - r / 2, i, f"{r:.1%}", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    ax.set_title("Class balance before and after cleaning")
    _save(fig, out_dir, "01_class_balance.png")


def plot_cancel_rates(df, out_dir):
    overall = df["IsCanceled"].mean()
    bins = [-1, 7, 30, 90, 180, 365, np.inf]
    names = ["0-7", "8-30", "31-90", "91-180", "181-365", "366+"]
    lead = pd.cut(df["LeadTime"], bins=bins, labels=names)
    groups = [
        (lead, "LeadTime (days)", "Cancel rate rises with lead time", "02_cancel_by_leadtime.png", None),
        (df["ArrivalDateMonth"], "Arrival month", "Cancel rate by arrival month", "03_cancel_by_month.png", MONTHS),
        (df["DepositType"], "DepositType", "Cancel rate by deposit type", "04_cancel_by_deposit.png", None),
        (df["MarketSegment"], "MarketSegment", "Cancel rate by market segment", "05_cancel_by_segment.png", None),
    ]
    for key, label, title, name, order in groups:
        rates = df.groupby(key, observed=True)["IsCanceled"].agg(["mean", "size"])
        rates = rates.reindex(order) if order else rates
        if key is not lead and order is None:
            rates = rates.sort_values("mean", ascending=False)
        _rate_bars(rates, overall, title, label, out_dir, name)


def plot_distributions(df, out_dir):
    columns = ["LeadTime", "ADR", "StaysInWeekNights", "StaysInWeekendNights",
               "PreviousCancellations", "PreviousBookingsNotCanceled"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.flat, columns):
        ax.hist(df[col], bins=40, color=BLUE, log=col.startswith("Previous"))
        ax.set_title(f"{col} (skew {df[col].skew():.1f})", fontsize=9)
    fig.supxlabel("Value")
    fig.supylabel("Bookings (log scale for Previous*)")
    _save(fig, out_dir, "06_distributions.png")


def plot_outliers(before, out_dir, limits):
    """Show the raw spread of each outlier rule against its fixed limit."""
    total_nights = before["StaysInWeekendNights"] + before["StaysInWeekNights"]
    series = {"ADR": before["ADR"], "LeadTime": before["LeadTime"],
              "TotalNights": total_nights, "Babies": before["Babies"]}
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (name, values) in zip(axes, series.items()):
        ax.boxplot(values, vert=False, widths=0.5, flierprops={"markersize": 3, "markeredgecolor": GRAY},
                   medianprops={"color": ORANGE})
        ax.axvline(limits[name], color=ORANGE, linestyle="--", linewidth=1)
        if name == "ADR":
            ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_title(f"{name} (limit {limits[name]})", fontsize=10)
    _save(fig, out_dir, "07_outliers.png")


def plot_correlation(df, out_dir):
    corr = df.select_dtypes("number").corr()
    cmap = LinearSegmentedColormap.from_list("diverging", [BLUE, "#f0efec", ORANGE])
    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i != j and abs(corr.iloc[i, j]) >= 0.2:
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.spines[:].set_visible(False)
    fig.colorbar(image, shrink=0.8, label="Pearson correlation")
    ax.set_title("Correlation of numeric columns")
    _save(fig, out_dir, "08_correlation.png")


def leakage_evidence(df):
    """Cancel rate on each side of the columns reviewed in the leakage audit."""
    flags = {
        "RequiredCarParkingSpaces > 0": df["RequiredCarParkingSpaces"] > 0,
        "AssignedRoomType != ReservedRoomType": df["AssignedRoomType"] != df["ReservedRoomType"],
        "BookingChanges > 0": df["BookingChanges"] > 0,
        "DaysInWaitingList > 0": df["DaysInWaitingList"] > 0,
        "TotalOfSpecialRequests > 0": df["TotalOfSpecialRequests"] > 0,
        "DepositType = Non Refund": df["DepositType"] == "Non Refund",
    }
    rows = []
    for name, flag in flags.items():
        rows.append({"condition": name, "bookings_true": int(flag.sum()),
                     "cancel_rate_true": round(df.loc[flag, "IsCanceled"].mean(), 3),
                     "cancel_rate_false": round(df.loc[~flag, "IsCanceled"].mean(), 3)})
    return pd.DataFrame(rows)


def main():
    config = load_config()
    settings = config["classification"]
    out_dir = PROJECT_ROOT / settings["output_dir"]
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(config)
    shared = load_all_models(config)
    cleaned, report = clean_classification(shared, config)

    data_quality_summary(raw).to_csv(out_dir / "raw_data_quality.csv")
    data_quality_summary(cleaned).to_csv(out_dir / "cleaned_data_quality.csv")
    report.to_csv(out_dir / "cleaning_report.csv", index=False)
    evidence = leakage_evidence(cleaned)
    evidence.to_csv(out_dir / "leakage_evidence.csv", index=False)

    plot_class_balance(shared, cleaned, eda_dir)
    plot_cancel_rates(cleaned, eda_dir)
    plot_distributions(cleaned, eda_dir)
    plot_outliers(_standardize_text(shared).drop_duplicates(), eda_dir,
                  settings["cleaning"]["outlier_limits"])
    plot_correlation(cleaned, eda_dir)

    print(f"Raw: {raw.shape}, duplicates {raw.duplicated().sum():,}")
    print(report.to_string(index=False))
    print(f"Cancel rate: {shared['IsCanceled'].mean():.3f} -> {cleaned['IsCanceled'].mean():.3f}")
    print(evidence.to_string(index=False))


if __name__ == "__main__":
    main()
