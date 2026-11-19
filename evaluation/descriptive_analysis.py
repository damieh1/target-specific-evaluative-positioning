"""Calculate target and outlet distributions, model agreement and monthly positioning."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LABELS = ["Favorable", "Neutral / Descriptive", "Unfavorable"]
TARGETS = ["Israel", "Palestine", "Hamas"]
FAST = ["qwen_orientation", "gemma_orientation", "mistral_orientation"]


def read_measurement(path):
    df = pd.read_csv(path, dtype={"occurrence_id": "string", "video_id": "string"}, low_memory=False)
    required = {"occurrence_id", "video_id", "outlet", "posted_at", "core_family", "final_orientation"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != 22540 or df["occurrence_id"].isna().any() or not df["occurrence_id"].is_unique:
        raise ValueError("Expected 22,540 unique target occurrences")
    if not df["final_orientation"].isin(LABELS).all():
        raise ValueError("Invalid final orientations")
    df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce", utc=True)
    df["month"] = df["posted_at"].dt.tz_localize(None).dt.to_period("M").astype("string")
    return df


def positioning_table(df, groups):
    counts = df.groupby(groups + ["final_orientation"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=LABELS, fill_value=0)
    totals = counts.sum(axis=1)
    return counts.div(totals, axis=0).mul(100).assign(total_n=totals).round(1)


def run(measurement, output_dir):
    df = read_measurement(measurement)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Instances: {len(df)}; videos: {df['video_id'].nunique()}; months: {df['month'].nunique()}")
    print("Date range:", df["posted_at"].min(), df["posted_at"].max())
    print("Missing dates/outlets/families:", df[["posted_at", "outlet", "core_family"]].isna().sum().to_dict())
    for column in ["outlet", "core_family", "final_orientation", "agreement_structure"]:
        df[column].value_counts(dropna=False).to_csv(output_dir / f"counts_{column}.csv", header=["n"])

    counts = pd.crosstab(df["core_family"], df["outlet"])
    shares = pd.crosstab(df["core_family"], df["outlet"], normalize="columns").mul(100)
    counts.to_csv(output_dir / "target_family_counts_by_outlet.csv")
    shares.to_csv(output_dir / "target_family_shares_by_outlet.csv")
    coverage = df.groupby("core_family").agg(instances=("occurrence_id", "size"), videos=("video_id", "nunique"),
                    outlets=("outlet", "nunique"), months=("month", "nunique"))
    coverage["corpus_share_percent"] = 100 * coverage["instances"] / len(df)
    coverage["outlets_with_30_or_more"] = counts.ge(30).sum(axis=1)
    coverage["outlets_with_50_or_more"] = counts.ge(50).sum(axis=1)
    coverage.sort_values("instances", ascending=False).to_csv(output_dir / "target_family_coverage.csv")

    family = positioning_table(df, ["core_family"]).sort_values("total_n", ascending=False)
    by_outlet = positioning_table(df, ["outlet", "core_family"]).reset_index()
    by_outlet["support"] = np.select([by_outlet["total_n"] >= 100, by_outlet["total_n"] >= 30],
                                     ["high", "moderate"], default="limited")
    family.to_csv(output_dir / "target_family_positioning_overall.csv")
    by_outlet.sort_values(["core_family", "total_n"], ascending=[True, False]).to_csv(
        output_dir / "target_family_positioning_by_outlet.csv", index=False)

    # Percentage-point balance: favorable minus unfavorable, divided by all labels.
    monthly = df.groupby(["core_family", "month", "final_orientation"]).size().unstack(fill_value=0)
    monthly = monthly.reindex(columns=LABELS, fill_value=0)
    monthly["n"] = monthly[LABELS].sum(axis=1)
    monthly["balance_pp"] = 100 * (monthly["Favorable"] - monthly["Unfavorable"]) / monthly["n"]
    monthly["direction"] = np.select([monthly["balance_pp"] > 0, monthly["balance_pp"] < 0],
                                     ["favorable", "unfavorable"], default="balanced")
    monthly.to_csv(output_dir / "monthly_positioning.csv")
    dominant = monthly.reset_index().loc[lambda x: x["core_family"].isin(TARGETS)]
    summary = dominant.groupby("core_family").agg(
        months=("month", "nunique"), favorable_months=("direction", lambda x: (x == "favorable").sum()),
        unfavorable_months=("direction", lambda x: (x == "unfavorable").sum()),
        balanced_months=("direction", lambda x: (x == "balanced").sum()),
        minimum_balance=("balance_pp", "min"), maximum_balance=("balance_pp", "max"))
    summary.round(1).to_csv(output_dir / "dominant_monthly_summary.csv")
    print(summary.round(1).to_string())

    # Pairwise agreement is calculated across all evaluable cases, not only 2:1 cases.
    alignment = pd.DataFrame({"Family": df["core_family"],
        "Qwen - Gemma": df[FAST[0]].eq(df[FAST[1]]) * 100,
        "Qwen - Mistral": df[FAST[0]].eq(df[FAST[2]]) * 100,
        "Gemma - Mistral": df[FAST[1]].eq(df[FAST[2]]) * 100,
        "Unanimous": df["agreement_structure"].eq("unanimous") * 100})
    rates = alignment.groupby("Family").mean()
    rates = rates.loc[rates.mean(axis=1).sort_values().index]
    rates.to_csv(output_dir / "model_alignment_by_family.csv")
    agreement = df.groupby(["core_family", "month", "agreement_structure"]).size().rename("n").reset_index()
    agreement["percentage"] = 100 * agreement["n"] / agreement.groupby(["core_family", "month"])["n"].transform("sum")
    agreement.to_csv(output_dir / "monthly_model_agreement.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.scatterplot(data=rates.reset_index().melt(id_vars="Family", var_name="Agreement type", value_name="Alignment (%)"),
                    x="Alignment (%)", y="Family", hue="Agreement type", style="Agreement type", ax=ax)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "model_misalignment.pdf", bbox_inches="tight")
    plt.close(fig)

    # Recreate the heatmaps from the calculated tables. The supplied notebook lacked these plotting cells.
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, label in zip(axes, LABELS):
        values = by_outlet.pivot(index="core_family", columns="outlet", values=label)
        sns.heatmap(values, ax=ax, vmin=0, vmax=100, cmap="Blues", annot=True, fmt=".0f", cbar=False)
        ax.set(title=label, xlabel="", ylabel="")
        ax.tick_params(axis="x", rotation=80)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_outlet_evaluative_positioning.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(monthly["balance_pp"].unstack("month"), cmap="RdBu", center=0,
                vmin=-100, vmax=100, ax=ax, cbar_kws={"label": "Balance (percentage points)"})
    ax.set(xlabel="Month", ylabel="")
    fig.tight_layout()
    fig.savefig(output_dir / "longitudinal_balance.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.measurement, args.output_dir)


if __name__ == "__main__":
    main()
