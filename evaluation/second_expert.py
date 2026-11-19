"""Draw the 150-case sample and compare the two expert annotations."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from validate_reference import EXPECTED_SHA256, sha256_file

KEYS = ["validation_id", "occurrence_id"]
LABEL = "expert_orientation_3way"
LABELS = ["Favorable", "Neutral / Descriptive", "Unfavorable"]
STRATA = {"unanimous": 50, "two_to_one": 75, "all_three_different": 25}
TEXT = ["target", "video_description", "context_before", "focal_occurrence", "context_after"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--second-expert", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.reference) != EXPECTED_SHA256:
        raise RuntimeError("Reference SHA-256 mismatch")
    ref = pd.read_csv(args.reference, dtype={key: "string" for key in KEYS})
    sample = pd.concat([
        ref.loc[ref["current_agreement_structure"].eq(stratum)].sample(n=n, random_state=42)
        for stratum, n in STRATA.items()
    ]).sample(frac=1.0, random_state=42)

    # Do not give the second expert the first expert's labels, predictions or agreement strata.
    sample[KEYS + TEXT].to_csv(args.output_dir / "second_expert_blind_sample.csv", index=False)
    if args.second_expert is None:
        return

    second = pd.read_csv(args.second_expert, dtype={key: "string" for key in KEYS})
    merged = sample[KEYS + [LABEL]].merge(
        second[KEYS + [LABEL]], on=KEYS, how="outer", validate="one_to_one",
        suffixes=("_expert1", "_expert2"), indicator=True,
    )
    if len(merged) != 150 or not merged["_merge"].eq("both").all():
        raise ValueError("The returned annotations do not match the 150 sampled IDs")
    y_true, y_pred = merged[LABEL + "_expert1"], merged[LABEL + "_expert2"]
    if not y_true.isin(LABELS).all() or not y_pred.isin(LABELS).all():
        raise ValueError("Invalid or missing expert orientation")
    overall = pd.Series({"n": len(merged), "agreement": accuracy_score(y_true, y_pred),
                         "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=LABELS)})
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    metrics = pd.DataFrame({"class": LABELS, "precision": precision, "recall": recall,
                            "f1": f1, "support": support})
    overall.to_csv(args.output_dir / "expert_agreement.csv", header=["value"])
    metrics.to_csv(args.output_dir / "expert_class_metrics.csv", index=False)
    print(overall.to_string())
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
