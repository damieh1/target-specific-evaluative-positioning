""" With this script, we verify the final Fast--Slow measurement dataset.

The script derives routing from the three normalized Fast labels, retains
unanimous Fast classifications, substitutes Slow adjudications for Fast
disagreements, and archives incomplete Fast rows.  An optional expert-reference
audit tests whether previously annotated target-conditioned inputs are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


FINAL_LABELS = {
    "Favorable",
    "Unfavorable",
    "Neutral / Descriptive",
}

IDENTITY_FIELDS = (
    "annotation_target",
    "video_description",
    "context_before_4",
    "focal_sentence_original",
    "context_after_4",
)

CONTEXT_FIELDS = (
    "video_id",
    "outlet",
    "posted_at",
    "sentence_id",
    "sentence_index",
    "annotation_target",
    "aspect_surface",
    "aspect_norm",
    "aspect",
    "core_family",
    "entity_type",
    "political_actor",
    "video_description",
    "focal_sentence_original",
    "focal_sentence_marked",
    "context_before_4",
    "context_after_4",
    "input_sha256",
    "transcript_text_sha256",
    "transcript_file_sha256",
    "metadata_sha256",
    "lexicon_sha256",
    "context_version",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a verified final Fast--Slow measurement dataset."
    )
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--slow", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--fast-columns",
        nargs=3,
        default=("qwen_orientation", "gemma_orientation", "mistral_orientation"),
        metavar=("QWEN_COLUMN", "GEMMA_COLUMN", "MISTRAL_COLUMN"),
        help="Normalized Fast-label columns in the agreement file.",
    )
    parser.add_argument(
        "--expert-reference",
        type=Path,
        help="Optional old expert-reference CSV to audit for exact reuse.",
    )
    parser.add_argument(
        "--expert-id-column",
        default="occurrence_id",
        help="Stable identifier in the expert-reference CSV (default: occurrence_id).",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"video_id": "string"}, low_memory=False)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def require_unique(frame: pd.DataFrame, column: str, source: str) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{source}.{column} contains missing values")
    duplicates = frame[column].duplicated(keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, column].astype(str).head(5).tolist()
        raise ValueError(f"{source}.{column} is not unique; examples: {examples}")


def normalize_label(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def parse_boolean(series: pd.Series) -> pd.Series:
    true_values = {"true", "1", "yes"}
    return series.astype("string").str.strip().str.lower().isin(true_values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def identity_hash(frame: pd.DataFrame) -> pd.Series:
    require_columns(frame, IDENTITY_FIELDS, "identity input")

    def hash_row(row: pd.Series) -> str:
        payload = json.dumps(
            [normalized_text(row[field]) for field in IDENTITY_FIELDS],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return frame.loc[:, IDENTITY_FIELDS].apply(hash_row, axis=1)


def build_measurement(
    contexts: pd.DataFrame,
    fast: pd.DataFrame,
    slow: pd.DataFrame,
    fast_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    require_columns(contexts, ["occurrence_id"], "contexts")
    require_columns(fast, ["occurrence_id", *fast_columns], "fast agreement")
    require_columns(slow, ["occurrence_id", "orientation"], "slow output")
    require_unique(contexts, "occurrence_id", "contexts")
    require_unique(fast, "occurrence_id", "fast agreement")
    require_unique(slow, "occurrence_id", "slow output")

    contexts = contexts.copy()
    fast = fast.copy()
    slow = slow.copy()

    for column in fast_columns:
        fast[column] = normalize_label(fast[column])
        invalid = fast[column].dropna().loc[lambda values: ~values.isin(FINAL_LABELS)]
        if not invalid.empty:
            raise ValueError(
                f"{column} contains invalid normalized labels: "
                f"{sorted(invalid.unique().tolist())}"
            )

    slow["orientation"] = normalize_label(slow["orientation"])
    invalid_slow = slow.loc[~slow["orientation"].isin(FINAL_LABELS), "orientation"]
    if not invalid_slow.empty:
        invalid_slow_counts = (
            invalid_slow.fillna("<missing>").value_counts().to_dict()
        )
        raise ValueError(
            "Slow output contains missing or invalid orientations: "
            f"{invalid_slow_counts}"
        )

    fast_complete = fast[fast_columns].notna().all(axis=1)
    distinct_labels = fast[fast_columns].nunique(axis=1, dropna=True)
    unanimous = fast_complete & distinct_labels.eq(1)
    disagreement = fast_complete & distinct_labels.gt(1)
    excluded = ~fast_complete

    fast["measurement_stage"] = pd.NA
    fast.loc[unanimous, "measurement_stage"] = "fast_unanimous"
    fast.loc[disagreement, "measurement_stage"] = "slow_adjudicated"
    fast.loc[excluded, "measurement_stage"] = "fast_incomplete_excluded"
    fast["final_orientation"] = pd.NA
    fast.loc[unanimous, "final_orientation"] = fast.loc[unanimous, fast_columns[0]]

    optional_slow = [
        column
        for column in (
            "selected_candidate",
            "partial_output_recovered",
            "recovery_note",
            "uncertainty",
            "primary_instability_source",
        )
        if column in slow.columns
    ]
    slow_keep = slow[["occurrence_id", "orientation", *optional_slow]].copy()
    slow_keep = slow_keep.rename(
        columns={
            "orientation": "slow_orientation",
            **{column: f"slow_{column}" for column in optional_slow},
        }
    )

    combined = fast.merge(slow_keep, on="occurrence_id", how="left", validate="one_to_one")
    slow_mask = combined["measurement_stage"].eq("slow_adjudicated")
    combined.loc[slow_mask, "final_orientation"] = combined.loc[
        slow_mask, "slow_orientation"
    ]

    routed_ids = set(combined.loc[slow_mask, "occurrence_id"])
    slow_ids = set(slow["occurrence_id"])
    if routed_ids != slow_ids:
        raise ValueError(
            "Slow rows do not exactly match Fast disagreement rows: "
            f"missing_from_slow={len(routed_ids - slow_ids)}, "
            f"unexpected_in_slow={len(slow_ids - routed_ids)}"
        )

    context_keep = ["occurrence_id"] + [
        column
        for column in CONTEXT_FIELDS
        if column in contexts.columns and column not in combined.columns
    ]
    combined = combined.merge(
        contexts[context_keep], on="occurrence_id", how="left", validate="one_to_one"
    )

    if set(combined["occurrence_id"]) != set(contexts["occurrence_id"]):
        raise ValueError("Fast agreement and context datasets do not contain identical IDs")

    excluded_rows = combined.loc[excluded].copy()
    evaluable = combined.loc[~excluded].copy()
    if evaluable["final_orientation"].isna().any():
        raise ValueError("At least one evaluable row lacks a final orientation")
    if not evaluable["final_orientation"].isin(FINAL_LABELS).all():
        raise ValueError("At least one evaluable row has an invalid final orientation")

    if "slow_partial_output_recovered" in evaluable.columns:
        partial_count = int(parse_boolean(evaluable["slow_partial_output_recovered"]).sum())
    else:
        partial_count = 0

    summary = {
        "source_instances": int(len(combined)),
        "evaluable_instances": int(len(evaluable)),
        "fast_unanimous": int(unanimous.sum()),
        "slow_adjudicated": int(disagreement.sum()),
        "fast_incomplete_excluded": int(excluded.sum()),
        "slow_partial_outputs_recovered": partial_count,
        "final_label_counts": {
            str(label): int(count)
            for label, count in evaluable["final_orientation"].value_counts().items()
        },
    }
    return evaluable, excluded_rows, summary


def audit_expert_reference(
    expert: pd.DataFrame,
    contexts: pd.DataFrame,
    id_column: str,
) -> tuple[pd.DataFrame, dict]:
    require_columns(expert, [id_column], "expert reference")
    require_unique(expert, id_column, "expert reference")
    require_columns(contexts, ["occurrence_id", *IDENTITY_FIELDS], "contexts")

    reference = expert.copy()
    if id_column != "occurrence_id":
        reference = reference.rename(columns={id_column: "occurrence_id"})

    # Exact content hashes are computed from the fields visible to the annotator.
    require_columns(reference, IDENTITY_FIELDS, "expert reference")
    reference["expert_context_sha256"] = identity_hash(reference)
    current = contexts[["occurrence_id", *IDENTITY_FIELDS]].copy()
    current["current_context_sha256"] = identity_hash(current)

    audit = reference[["occurrence_id", "expert_context_sha256"]].merge(
        current[["occurrence_id", "current_context_sha256"]],
        on="occurrence_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    audit["present_in_rebuilt_corpus"] = audit["_merge"].eq("both")
    audit["context_identical"] = (
        audit["present_in_rebuilt_corpus"]
        & audit["expert_context_sha256"].eq(audit["current_context_sha256"])
    )
    audit = audit.drop(columns="_merge")

    summary = {
        "expert_rows": int(len(audit)),
        "present_in_rebuilt_corpus": int(audit["present_in_rebuilt_corpus"].sum()),
        "context_identical": int(audit["context_identical"].sum()),
        "missing_from_rebuilt_corpus": int((~audit["present_in_rebuilt_corpus"]).sum()),
        "changed_context": int(
            (audit["present_in_rebuilt_corpus"] & ~audit["context_identical"]).sum()
        ),
        "all_expert_labels_reusable": bool(audit["context_identical"].all()),
    }
    return audit, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contexts = read_csv(args.contexts)
    fast = read_csv(args.fast)
    slow = read_csv(args.slow)
    evaluable, excluded, summary = build_measurement(
        contexts, fast, slow, list(args.fast_columns)
    )

    evaluable_path = args.output_dir / "fast_slow_evaluable.csv"
    excluded_path = args.output_dir / "fast_incomplete_excluded.csv"
    summary_path = args.output_dir / "measurement_manifest.json"
    evaluable.to_csv(evaluable_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    manifest = {
        **summary,
        "inputs": {
            "contexts": {"path": str(args.contexts), "sha256": sha256_file(args.contexts)},
            "fast": {"path": str(args.fast), "sha256": sha256_file(args.fast)},
            "slow": {"path": str(args.slow), "sha256": sha256_file(args.slow)},
        },
        "outputs": {
            "evaluable": {
                "path": str(evaluable_path),
                "sha256": sha256_file(evaluable_path),
            },
            "excluded": {
                "path": str(excluded_path),
                "sha256": sha256_file(excluded_path),
            },
        },
    }

    if args.expert_reference:
        expert = read_csv(args.expert_reference)
        audit, audit_summary = audit_expert_reference(
            expert, contexts, args.expert_id_column
        )
        audit_path = args.output_dir / "expert_reference_reuse_audit.csv"
        audit.to_csv(audit_path, index=False)
        manifest["expert_reference_audit"] = {
            **audit_summary,
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
        }

    summary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
