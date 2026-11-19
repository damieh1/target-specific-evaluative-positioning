"""Compare three Fast-model outputs for the expanded contextual corpus.

Retain raw four-way labels and map Mixed to Neutral / Descriptive.
Mark a valid null orientation as a Fast abstention; the Slow input script
excludes abstentions before adjudicating disagreements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


RAW_TO_FINAL = {
    "Favorable": "Favorable",
    "Unfavorable": "Unfavorable",
    "Neutral / Descriptive": "Neutral / Descriptive",
    "Mixed": "Neutral / Descriptive",
}
MODEL_NAMES = ("qwen", "gemma", "mistral")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--mistral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=22541)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_value(value) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    return None


def clean_raw_orientation(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_model(path: Path, model: str, expected_ids: set[str], expected_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"occurrence_id": str}, low_memory=False)
    if len(df) != expected_rows:
        raise RuntimeError(f"{model}: expected {expected_rows} rows, found {len(df)}")
    if "occurrence_id" not in df.columns or "orientation" not in df.columns:
        raise RuntimeError(f"{model}: required columns are absent")
    if df["occurrence_id"].duplicated().any():
        raise RuntimeError(f"{model}: duplicate occurrence_id values")
    if set(df["occurrence_id"]) != expected_ids:
        raise RuntimeError(f"{model}: occurrence-ID set differs from contexts")
    if "parse_error" in df.columns:
        errors = df["parse_error"].fillna("").astype(str).str.strip().ne("")
        if errors.any():
            raise RuntimeError(f"{model}: {int(errors.sum())} parse errors remain")

    raw = df["orientation"].map(clean_raw_orientation)
    invalid = sorted({value for value in raw.dropna().unique() if value not in RAW_TO_FINAL})
    if invalid:
        raise RuntimeError(f"{model}: invalid raw orientations: {invalid}")

    review = (
        df["needs_human_review"].map(bool_value)
        if "needs_human_review" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    invalid_abstention = raw.isna() & review.ne(True)
    if invalid_abstention.any():
        bad = df.loc[invalid_abstention, "occurrence_id"].tolist()
        raise RuntimeError(
            f"{model}: {len(bad)} null orientations are not valid review abstentions: {bad[:10]}"
        )

    selected = pd.DataFrame({
        "occurrence_id": df["occurrence_id"],
        f"{model}_orientation_raw": raw,
        f"{model}_orientation": raw.map(RAW_TO_FINAL),
        f"{model}_ambiguity": (
            df["ambiguity_detected"] if "ambiguity_detected" in df.columns else pd.NA
        ),
        f"{model}_review": review,
        f"{model}_reason": df["reason"] if "reason" in df.columns else pd.NA,
    })
    return selected


def classify_row(row: pd.Series) -> tuple[str, str | None]:
    values = [row[f"{model}_orientation"] for model in MODEL_NAMES]
    if any(pd.isna(value) for value in values):
        return "fast_abstention", None
    counts = pd.Series(values).value_counts()
    if len(counts) == 1:
        return "unanimous", str(counts.index[0])
    if len(counts) == 2:
        return "two_to_one", str(counts.index[0])
    return "all_three_different", None


def pairwise_stats(df: pd.DataFrame, left: str, right: str) -> dict:
    a = df[f"{left}_orientation"]
    b = df[f"{right}_orientation"]
    complete = a.notna() & b.notna()
    if not complete.any():
        raise RuntimeError(f"No complete ratings for {left}--{right}")
    return {
        "n_complete": int(complete.sum()),
        "agreement_percent": float((a[complete] == b[complete]).mean() * 100),
        "cohen_kappa": float(cohen_kappa_score(a[complete], b[complete])),
    }


def krippendorff_alpha_nominal(df: pd.DataFrame) -> tuple[float, int]:
    """Nominal alpha using coincidence accounting with missing ratings."""
    categories = sorted(RAW_TO_FINAL.values())
    categories = list(dict.fromkeys(categories))
    coincidence = pd.DataFrame(0.0, index=categories, columns=categories)
    usable_units = 0

    for _, row in df.iterrows():
        values = [row[f"{model}_orientation"] for model in MODEL_NAMES]
        values = [value for value in values if not pd.isna(value)]
        m = len(values)
        if m < 2:
            continue
        usable_units += 1
        counts = pd.Series(values).value_counts()
        for left in categories:
            for right in categories:
                if left == right:
                    contribution = counts.get(left, 0) * (counts.get(left, 0) - 1)
                else:
                    contribution = counts.get(left, 0) * counts.get(right, 0)
                coincidence.loc[left, right] += contribution / (m - 1)

    n = float(coincidence.to_numpy().sum())
    if n <= 1:
        raise RuntimeError("Insufficient ratings for Krippendorff alpha")
    observed = (
        coincidence.to_numpy().sum() - np.trace(coincidence.to_numpy())
    ) / n
    marginals = coincidence.sum(axis=1).to_numpy()
    expected = (n * n - np.square(marginals).sum()) / (n * (n - 1))
    alpha = 1.0 if expected == 0 else 1.0 - observed / expected
    return float(alpha), usable_units


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contexts = pd.read_csv(args.contexts, dtype={"occurrence_id": str}, low_memory=False)
    if len(contexts) != args.expected_rows:
        raise RuntimeError(
            f"contexts: expected {args.expected_rows} rows, found {len(contexts)}"
        )
    if contexts["occurrence_id"].duplicated().any():
        raise RuntimeError("contexts: duplicate occurrence_id values")
    expected_ids = set(contexts["occurrence_id"])

    paths = {"qwen": args.qwen, "gemma": args.gemma, "mistral": args.mistral}
    joined = contexts.copy()
    for model in MODEL_NAMES:
        selected = load_model(paths[model], model, expected_ids, args.expected_rows)
        joined = joined.merge(selected, on="occurrence_id", how="left", validate="one_to_one")

    classifications = joined.apply(classify_row, axis=1, result_type="expand")
    joined["agreement_structure"] = classifications[0]
    joined["fast_majority_orientation"] = classifications[1]
    joined["route_to_slow"] = joined["agreement_structure"].ne("unanimous")

    structure_order = [
        "unanimous", "two_to_one", "all_three_different", "fast_abstention"
    ]
    structure = joined["agreement_structure"].value_counts().reindex(
        structure_order, fill_value=0
    )

    alpha, alpha_units = krippendorff_alpha_nominal(joined)
    summary = {
        "n_instances": len(joined),
        "raw_label_counts": {
            model: {
                str(label): int(count)
                for label, count in joined[f"{model}_orientation_raw"]
                .fillna("<abstention>").value_counts().items()
            }
            for model in MODEL_NAMES
        },
        "final_label_counts": {
            model: {
                str(label): int(count)
                for label, count in joined[f"{model}_orientation"]
                .fillna("<abstention>").value_counts().items()
            }
            for model in MODEL_NAMES
        },
        "agreement_structure": {
            category: {"n": int(count), "percent": float(count / len(joined) * 100)}
            for category, count in structure.items()
        },
        "routed_to_slow": int(joined["route_to_slow"].sum()),
        "pairwise": {
            "qwen_gemma": pairwise_stats(joined, "qwen", "gemma"),
            "qwen_mistral": pairwise_stats(joined, "qwen", "mistral"),
            "gemma_mistral": pairwise_stats(joined, "gemma", "mistral"),
        },
        "krippendorff_alpha_nominal": alpha,
        "krippendorff_alpha_usable_units": alpha_units,
        "files": {
            "contexts": {"path": str(args.contexts), "sha256": sha256_file(args.contexts)},
            **{
                model: {"path": str(path), "sha256": sha256_file(path)}
                for model, path in paths.items()
            },
        },
    }

    full_path = args.output_dir / "fast_agreement_full_22541.csv"
    queue_path = args.output_dir / "slow_routing_queue_22541.csv"
    pattern_path = args.output_dir / "agreement_structure.csv"
    summary_path = args.output_dir / "fast_agreement_summary.json"

    joined.to_csv(full_path, index=False)
    joined.loc[joined["route_to_slow"]].to_csv(queue_path, index=False)
    pd.DataFrame({
        "agreement_structure": structure.index,
        "n": structure.values,
        "percent": structure.values / len(joined) * 100,
    }).to_csv(pattern_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    for path in (full_path, queue_path, pattern_path, summary_path):
        print(path)


if __name__ == "__main__":
    main()
