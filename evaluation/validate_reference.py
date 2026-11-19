"""Evaluate the 550-case expert reference and estimate corpus accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, norm
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, cohen_kappa_score, f1_score,
    precision_recall_fscore_support,
)

EXPECTED_SHA256 = "ad5a375d7af89adfec27b511789c86a8838a4196ea667bcd884b4fc5c5b09558"
EXPERT = "expert_orientation_3way"
FAST_COLUMNS = ["qwen_3way", "gemma_3way", "mistral_3way"]
LABELS = ["Favorable", "Neutral / Descriptive", "Unfavorable"]
STRATUM = "current_agreement_structure"
EXPECTED_AGREEMENT = {"unanimous": 250, "two_to_one": 250, "all_three_different": 50}
EXPECTED_SOURCES = {
    "unchanged_reused": 441, "changed_context_reannotation": 33,
    "new_blind_replacement": 21, "new_blind_case": 15, "new_blind_extension": 40,
}
CORPUS_COUNTS = {"unanimous": 17000, "two_to_one": 5235, "all_three_different": 305}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_fast_fields(row):
    counts = row[FAST_COLUMNS].value_counts()
    if len(counts) == 1:
        return "unanimous", counts.index[0]
    if counts.iloc[0] == 2:
        return "two_to_one", counts.index[0]
    return "all_three_different", pd.NA


def check_reference(df):
    required = ["validation_id", "occurrence_id", "reference_source", EXPERT,
                *FAST_COLUMNS, STRATUM, "fast_majority_orientation",
                "fast_consensus_orientation", "slow_orientation", "final_orientation"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != 550:
        raise ValueError(f"Expected 550 cases, found {len(df)}")
    for column in ["validation_id", "occurrence_id"]:
        if df[column].isna().any() or df[column].duplicated().any():
            raise ValueError(f"Missing or duplicate {column}")
    for column in [EXPERT, *FAST_COLUMNS, "final_orientation"]:
        if not df[column].isin(LABELS).all():
            raise ValueError(f"Invalid or missing labels in {column}")
    for column, expected in [(STRATUM, EXPECTED_AGREEMENT), ("reference_source", EXPECTED_SOURCES)]:
        if df[column].value_counts(dropna=False).to_dict() != expected:
            raise ValueError(f"Unexpected counts in {column}")

    derived = df.apply(derive_fast_fields, axis=1, result_type="expand")
    if not derived[0].eq(df[STRATUM]).all():
        raise ValueError("Stored agreement structure differs from model labels")
    unanimous = derived[0].eq("unanimous")
    majority = derived[0].eq("two_to_one")
    for mask, column in [(unanimous, "fast_consensus_orientation"), (majority, "fast_majority_orientation")]:
        if not derived.loc[mask, 1].eq(df.loc[mask, column]).fillna(False).all():
            raise ValueError(f"Inconsistent {column}")
    if not df.loc[~unanimous, "slow_orientation"].isin(LABELS).all():
        raise ValueError("A disagreement case lacks a valid Slow orientation")
    for mask, column in [(unanimous, "fast_consensus_orientation"), (~unanimous, "slow_orientation")]:
        if not df.loc[mask, "final_orientation"].eq(df.loc[mask, column]).fillna(False).all():
            raise ValueError("Final labels do not follow Fast-Slow routing")


def evaluate(data, prediction, system, sample):
    evaluated = data.dropna(subset=[EXPERT, prediction])
    n = len(evaluated)
    n_correct = int(evaluated[EXPERT].eq(evaluated[prediction]).sum())
    # Wilson score confidence interval.
    z = norm.ppf(0.975)
    p = n_correct / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return {
        "system": system, "sample": sample, "n": n,
        "accuracy": accuracy_score(evaluated[EXPERT], evaluated[prediction]),
        "accuracy_ci_low": center - margin, "accuracy_ci_high": center + margin,
        "macro_f1": f1_score(evaluated[EXPERT], evaluated[prediction], labels=LABELS,
                             average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(evaluated[EXPERT], evaluated[prediction], labels=LABELS),
    }


def run(reference, output_dir):
    observed_sha256 = sha256_file(reference)
    if observed_sha256 != EXPECTED_SHA256:
        raise ValueError(f"Reference SHA-256 mismatch: {observed_sha256}")
    df = pd.read_csv(reference, dtype={"validation_id": "string", "occurrence_id": "string"}, low_memory=False)
    check_reference(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    unanimous = df.loc[df[STRATUM].eq("unanimous")]
    two_to_one = df.loc[df[STRATUM].eq("two_to_one")]
    all_different = df.loc[df[STRATUM].eq("all_three_different")]
    disagreement = df.loc[df[STRATUM].ne("unanimous")]
    specs = [
        (df, "qwen_3way", "Qwen3-14B", "all validation cases"),
        (df, "gemma_3way", "Gemma3-12B", "all validation cases"),
        (df, "mistral_3way", "Mistral Small 3.1 24B", "all validation cases"),
        (unanimous, "fast_consensus_orientation", "Fast consensus", "unanimous"),
        (two_to_one, "fast_majority_orientation", "Fast majority", "2:1 disagreement"),
        (two_to_one, "slow_orientation", "Slow adjudicator", "same 2:1 cases"),
        (disagreement, "slow_orientation", "Slow adjudicator", "all disagreement"),
        (all_different, "slow_orientation", "Slow adjudicator", "all three different"),
        (df, "final_orientation", "Complete Fast--Slow measurement", "all validation cases"),
    ]
    metrics = pd.DataFrame([evaluate(*spec) for spec in specs])
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))

    # Exact McNemar test on the same 250 cases.
    majority_correct = two_to_one["fast_majority_orientation"].eq(two_to_one[EXPERT])
    slow_correct = two_to_one["slow_orientation"].eq(two_to_one[EXPERT])
    majority_only = int((majority_correct & ~slow_correct).sum())
    slow_only = int((~majority_correct & slow_correct).sum())
    discordant = majority_only + slow_only
    mcnemar = {
        "n_paired": len(two_to_one), "majority_correct_slow_wrong": majority_only,
        "majority_wrong_slow_correct": slow_only, "discordant_pairs": discordant,
        "exact_two_sided_p": float(binomtest(slow_only, discordant, p=0.5).pvalue) if discordant else 1.0,
        "fast_majority_accuracy": float(majority_correct.mean()),
        "slow_accuracy": float(slow_correct.mean()),
        "accuracy_difference_slow_minus_majority": float(slow_correct.mean() - majority_correct.mean()),
    }

    # Fisher's exact test: unanimous consensus versus 2:1 majority.
    unanimous_correct = unanimous["fast_consensus_orientation"].eq(unanimous[EXPERT])
    table = [[int(unanimous_correct.sum()), int((~unanimous_correct).sum())],
             [int(majority_correct.sum()), int((~majority_correct).sum())]]
    odds_ratio, fisher_p = fisher_exact(table, alternative="two-sided")
    fisher = {"unanimous_n": len(unanimous), "unanimous_correct": table[0][0],
              "two_to_one_n": len(two_to_one), "two_to_one_majority_correct": table[1][0],
              "odds_ratio": float(odds_ratio), "exact_two_sided_p": float(fisher_p)}

    # Weight validation accuracy by the final corpus agreement structure.
    stratum_accuracy = df.assign(correct=df[EXPERT].eq(df["final_orientation"])).groupby(STRATUM)["correct"].mean()
    poststrata = pd.DataFrame({"corpus_n": pd.Series(CORPUS_COUNTS), "accuracy": stratum_accuracy})
    poststrata["weight"] = poststrata["corpus_n"] / poststrata["corpus_n"].sum()
    poststrata["weighted_accuracy"] = poststrata["weight"] * poststrata["accuracy"]
    poststrata.to_csv(output_dir / "poststratified_accuracy.csv")

    precision, recall, f1, support = precision_recall_fscore_support(
        df[EXPERT], df["final_orientation"], labels=LABELS, zero_division=0)
    pd.DataFrame({"label": LABELS, "precision": precision, "recall": recall,
                  "f1": f1, "support": support}).to_csv(output_dir / "final_class_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ConfusionMatrixDisplay.from_predictions(df[EXPERT], df["final_orientation"], labels=LABELS,
        display_labels=["Favorable", "Neutral /\nDescriptive", "Unfavorable"], cmap="Blues", colorbar=False, ax=ax)
    ax.set(xlabel="Predicted orientation", ylabel="Expert reference", title="Complete Fast--Slow measurement")
    fig.tight_layout()
    fig.savefig(output_dir / "final_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    error_columns = ["validation_id", "occurrence_id", "target", "reference_source", STRATUM,
                     EXPERT, *FAST_COLUMNS, "fast_majority_orientation", "slow_orientation", "final_orientation"]
    errors = df.loc[df[EXPERT].ne(df["final_orientation"]), error_columns]
    errors.to_csv(output_dir / "final_validation_errors.csv", index=False)
    results = {"reference_file": str(reference), "reference_sha256": observed_sha256,
               "n_cases": len(df), "mcnemar": mcnemar, "fisher_exact": fisher,
               "poststratified_accuracy": float(poststrata["weighted_accuracy"].sum()), "n_final_errors": len(errors)}
    (output_dir / "statistical_tests.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.output_dir)


if __name__ == "__main__":
    main()
