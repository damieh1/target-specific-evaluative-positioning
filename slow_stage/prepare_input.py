from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import pandas as pd


ORIENTATIONS = {
    "Favorable",
    "Unfavorable",
    "Neutral / Descriptive",
}

# We use the following columns from the dataset for the slow stage adjudication.
CONTEXT_COLUMNS = [
    "corpus_row_id",
    "occurrence_id",
    "video_id",
    "outlet",
    "annotation_target",
    "aspect_surface",
    "aspect_norm",
    "core_family",
    "video_description",
    "context_before_4",
    "focal_sentence_original",
    "focal_sentence_marked",
    "context_after_4",
    "input_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=22541)
    parser.add_argument("--expected-abstentions", type=int, default=1)
    parser.add_argument("--expected-slow", type=int, default=5540)
    parser.add_argument("--shuffle-seed", default="slow-v12-expanded-20260922") # Frozen for reproducibility.
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_shuffle(values: list[str], occurrence_id: str, seed: str) -> list[str]:
    material = f"{seed}\0{occurrence_id}".encode("utf-8")
    integer_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    shuffled = list(values)
    random.Random(integer_seed).shuffle(shuffled)
    return shuffled


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        args.agreement,
        dtype={"occurrence_id": str, "video_id": str},
        low_memory=False,
    )
    if len(df) != args.expected_total:
        raise RuntimeError(f"Expected {args.expected_total} rows, found {len(df)}")
    if df["occurrence_id"].duplicated().any():
        raise RuntimeError("Duplicate occurrence_id values")

    # The expanded agreement file preserves canonical contexts.csv order but
    # contexts.csv itself does not necessarily store a positional row ID.
    # Derive it once here for resume-safe processing and output ordering.
    if "corpus_row_id" not in df.columns:
        df.insert(0, "corpus_row_id", range(len(df)))
    else:
        numeric_ids = pd.to_numeric(df["corpus_row_id"], errors="coerce")
        if numeric_ids.isna().any() or numeric_ids.duplicated().any():
            raise RuntimeError("Existing corpus_row_id values are invalid or duplicated")
        df["corpus_row_id"] = numeric_ids.astype(int)

    required = set(CONTEXT_COLUMNS) | {
        "agreement_structure",
        "qwen_orientation",
        "gemma_orientation",
        "mistral_orientation",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    excluded = df[df["agreement_structure"].eq("fast_abstention")].copy()
    if len(excluded) != args.expected_abstentions:
        raise RuntimeError(
            f"Expected {args.expected_abstentions} abstentions, found {len(excluded)}"
        )

    evaluable = df[~df["agreement_structure"].eq("fast_abstention")].copy()
    unanimous = evaluable[evaluable["agreement_structure"].eq("unanimous")].copy()
    routed = evaluable[
        evaluable["agreement_structure"].isin(["two_to_one", "all_three_different"])
    ].copy()
    if len(routed) != args.expected_slow:
        raise RuntimeError(f"Expected {args.expected_slow} Slow rows, found {len(routed)}")

    slow_records = []
    audit_records = []
    for _, row in routed.iterrows():
        model_values = {
            "qwen": row["qwen_orientation"],
            "gemma": row["gemma_orientation"],
            "mistral": row["mistral_orientation"],
        }
        if any(pd.isna(value) for value in model_values.values()):
            raise RuntimeError(f"Missing routed label for {row['occurrence_id']}")
        if set(model_values.values()) - ORIENTATIONS:
            raise RuntimeError(f"Invalid routed label for {row['occurrence_id']}")

        # We remove duplicate orientations so the Slow adjudicator cannot infer how many Fast models supported an interpretation.
        unique = list(dict.fromkeys(model_values.values()))
        expected_candidates = 2 if row["agreement_structure"] == "two_to_one" else 3
        if len(unique) != expected_candidates:
            raise RuntimeError(
                f"Unexpected candidate count for {row['occurrence_id']}: {len(unique)}"
            )
        candidates = deterministic_shuffle(unique, row["occurrence_id"], args.shuffle_seed)
        candidate_map = {
            "candidate_a": candidates[0],
            "candidate_b": candidates[1],
            "candidate_c": candidates[2] if len(candidates) == 3 else pd.NA,
        }

        record = {column: row[column] for column in CONTEXT_COLUMNS}
        record.update(candidate_map)
        slow_records.append(record)

        audit_records.append({
            "occurrence_id": row["occurrence_id"],
            "corpus_row_id": row["corpus_row_id"],
            "agreement_structure": row["agreement_structure"],
            **{f"{model}_orientation": value for model, value in model_values.items()},
            **candidate_map,
            "candidate_count": len(candidates),
        })

    slow = pd.DataFrame(slow_records)
    audit = pd.DataFrame(audit_records)
    if slow["occurrence_id"].duplicated().any():
        raise RuntimeError("Duplicate Slow occurrence IDs")
    if any(column in slow.columns for column in ["qwen_orientation", "gemma_orientation", "mistral_orientation"]):
        raise RuntimeError("Model identities leaked into Slow input")

    excluded_path = args.output_dir / "excluded_fast_abstention_1.csv"
    evaluable_path = args.output_dir / "fast_evaluable_22540.csv"
    unanimous_path = args.output_dir / "unanimous_fast_17000.csv"
    slow_path = args.output_dir / "slow_stage_input_5540.csv"
    audit_path = args.output_dir / "slow_candidate_audit_5540.csv"
    manifest_path = args.output_dir / "slow_input_manifest.json"

    excluded.to_csv(excluded_path, index=False)
    evaluable.to_csv(evaluable_path, index=False)
    unanimous.to_csv(unanimous_path, index=False)
    slow.to_csv(slow_path, index=False)
    audit.to_csv(audit_path, index=False)

    manifest = {
        "source_file": str(args.agreement),
        "source_sha256": sha256_file(args.agreement),
        "source_occurrences": len(df),
        "excluded_fast_abstentions": len(excluded),
        "evaluable_occurrences": len(evaluable),
        "unanimous_fast_retained": len(unanimous),
        "slow_routed": len(slow),
        "two_to_one": int((audit["agreement_structure"] == "two_to_one").sum()),
        "all_three_different": int(
            (audit["agreement_structure"] == "all_three_different").sum()
        ),
        "candidate_count_distribution": {
            str(key): int(value)
            for key, value in audit["candidate_count"].value_counts().sort_index().items()
        },
        "candidate_shuffle_seed": args.shuffle_seed,
        "slow_input": str(slow_path),
        "slow_input_sha256": sha256_file(slow_path),
        "model_identity_columns_in_slow_input": False,
        "duplicate_orientations_removed_before_anonymization": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("\nSaved:")
    for path in (excluded_path, evaluable_path, unanimous_path, slow_path, audit_path, manifest_path):
        print(path)


if __name__ == "__main__":
    main()