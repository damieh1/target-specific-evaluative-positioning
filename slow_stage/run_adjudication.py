import argparse
import re
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from openai import OpenAI

from protocol import (
    PROMPT_VERSION,
    RESPONSE_FORMAT,
    SYSTEM_PROMPT,
    build_user_prompt,
    extract_json,
    validate_result,
)


MODEL = (
    "RedHatAI/"
    "Magistral-Small-2506-FP8"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Adjudicate the Fast disagreement cases.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    return parser.parse_args()


TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 7000
SEED = 20260831


MAGISTRAL_SYSTEM = f"""
You are performing a deliberate Slow-stage reasoning task.

{SYSTEM_PROMPT}

Reason carefully, but keep the deliberation bounded.

Work through the competing interpretations once. Focus on:

1. target binding;
2. the focal occurrence;
3. relevant contextual evidence;
4. event valence versus target evaluation;
5. agency or responsibility;
6. the strongest evidence for each candidate;
7. the strongest limitation of each candidate;
8. the final comparison.

Do not repeatedly reconsider an argument after it has already been assessed.

If two interpretations remain plausible after the evidence audit, select the
better-supported interpretation and express the remaining ambiguity through
the uncertainty field.

After completing the reasoning, output exactly one valid JSON object
matching the required schema and no additional prose.
""".strip()


def parse_final(raw):
    text = str(raw).strip()

    if "</think>" in text:
        text = (
            text
            .split("</think>", 1)[1]
            .strip()
        )

    return extract_json(text)

def recover_partial_decision(raw, row):
    if not isinstance(raw, str):
        return None

    orientation_match = re.search(
        r'"orientation"\s*:\s*"([^"]+)"',
        raw,
    )

    candidate_match = re.search(
        r'"selected_candidate"\s*:\s*"([^"]+)"',
        raw,
    )

    if not orientation_match or not candidate_match:
        return None

    orientation = (
        orientation_match
        .group(1)
        .strip()
    )

    candidate = (
        candidate_match
        .group(1)
        .strip()
    )

    candidate_col = {
        "A": "candidate_a",
        "B": "candidate_b",
        "C": "candidate_c",
    }.get(candidate)

    if candidate_col is None:
        return None

    if (
        candidate_col not in row
        or pd.isna(row[candidate_col])
    ):
        return None

    expected = str(
        row[candidate_col]
    ).strip()

    if orientation != expected:
        return None

    return {
        "orientation": orientation,
        "selected_candidate": candidate,
        "selected_candidate_repaired": False,
        "candidate_assessments": None,
        "candidate_assessment_complete": False,
        "candidate_assessments_missing": None,
        "instability_sources": None,
        "primary_instability_source": None,
        "uncertainty": None,
        "evidence_summary": None,
    }


def main():
    args = parse_args()
    INPUT = args.input
    OUTPUT = args.output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    assert len(df) == 5540
    assert df["corpus_row_id"].is_unique
    assert df["occurrence_id"].is_unique


    # Resume
    if OUTPUT.exists():
        old = pd.read_csv(
            OUTPUT,
            low_memory=False,
        )

        completed = set(
            old.loc[
                old["parse_error"].isna()
                & old["orientation"].notna(),
                "corpus_row_id",
            ]
            .astype(int)
            .tolist()
        )

        records = old.to_dict(
            orient="records"
        )

    else:
        completed = set()
        records = []


    remaining = df.loc[
        ~df[
            "corpus_row_id"
        ]
        .astype(int)
        .isin(completed)
    ].copy()


    print("Slow-stage adjudication")
    print("Rows:", len(df))
    print("Already complete:", len(completed))
    print("Remaining:", len(remaining))
    print()


    # Save after every case

    def save_records():
        out = pd.DataFrame(
            records
        )

        if len(out) == 0:
            return

        out["corpus_row_id"] = (
            out["corpus_row_id"]
            .astype(int)
        )

        sort_cols = []

        if "benchmark_id" in out.columns:
            sort_cols.append(
                "benchmark_id"
            )

        sort_cols.append(
            "corpus_row_id"
        )

        # Preserve insertion order when replacing an earlier failed attempt.
        # The newest appended record must win before any sorting occurs.
        out = out.drop_duplicates(
            subset="corpus_row_id",
            keep="last",
        )

        out = out.sort_values(
            sort_cols,
            kind="stable",
        )

        out.to_csv(
            OUTPUT,
            index=False,
        )


    # Inference

    for i, (_, row) in enumerate(
        remaining.iterrows(),
        start=1,
    ):

        row_id = int(
            row["corpus_row_id"]
        )

        occurrence_id = str(
            row["occurrence_id"]
        )

        benchmark_id = (
            int(row["benchmark_id"])
            if (
                "benchmark_id" in row
                and pd.notna(
                    row["benchmark_id"]
                )
            )
            else None
        )

        raw = None
        result = {}
        error = None
        partial_output_recovered = False
        recovery_note = None

        try:
            response = (
                client
                .chat
                .completions
                .create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                MAGISTRAL_SYSTEM,
                        },
                        {
                            "role": "user",
                            "content":
                                build_user_prompt(row),
                        },
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                    seed=SEED,
                    response_format=RESPONSE_FORMAT,
                )
            )

            raw = (
                response
                .choices[0]
                .message
                .content
            )

            result = parse_final(
                raw
            )

            result = validate_result(
                result,
                row,
            )

        except Exception as exc:
            recovered = recover_partial_decision(
                raw,
                row,
            )

            if recovered is not None:
                result = recovered
                error = None
                partial_output_recovered = True
                recovery_note = (
                    "Orientation and selected candidate "
                    "recovered verbatim from truncated JSON."
                )

            else:
                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        record = {
            "benchmark_id":
                benchmark_id,

            "corpus_row_id":
                row_id,

            "occurrence_id":
                occurrence_id,

            "model":
                MODEL,

            "prompt_version":
                PROMPT_VERSION,

            "orientation":
                result.get(
                    "orientation"
                ),

            "selected_candidate":
                result.get(
                    "selected_candidate"
                ),

            "selected_candidate_repaired":
                result.get(
                    "selected_candidate_repaired"
                ),

            "candidate_assessments": (
                json.dumps(
                    result.get(
                        "candidate_assessments"
                    ),
                    ensure_ascii=False,
                )
                if result.get(
                    "candidate_assessments"
                ) is not None
                else None
            ),

            "candidate_assessment_complete":
                result.get(
                    "candidate_assessment_complete"
                ),

            "candidate_assessments_missing": (
                json.dumps(
                    result.get(
                        "candidate_assessments_missing"
                    ),
                    ensure_ascii=False,
                )
                if result.get(
                    "candidate_assessments_missing"
                ) is not None
                else None
            ),

            "instability_sources": (
                json.dumps(
                    result.get(
                        "instability_sources"
                    ),
                    ensure_ascii=False,
                )
                if result.get(
                    "instability_sources"
                ) is not None
                else None
            ),

            "primary_instability_source":
                result.get(
                    "primary_instability_source"
                ),

            "uncertainty":
                result.get(
                    "uncertainty"
                ),

            "evidence_summary":
                result.get(
                    "evidence_summary"
                ),

            "partial_output_recovered":
                partial_output_recovered,

            "recovery_note":
                recovery_note,

            "raw_output":
                raw,

            "parse_error":
                error,

            "timestamp_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        records.append(
            record
        )

        save_records()


        done_now = (
            len(completed)
            + i
        )

        status = (
            result.get(
                "orientation"
            )
            if error is None
            else f"ERROR: {error}"
        )

        print(
            f"{done_now:04d}/5540 "
            f"row={row_id:05d} "
            f"{status}"
        )


    # Summary
    final = pd.read_csv(
        OUTPUT,
        low_memory=False,
    )

    final = (
        final
        .drop_duplicates(
            subset="corpus_row_id",
            keep="last",
        )
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "MAGISTRAL SLOW-STAGE EXPANDED CORPUS 5540 "
        "— v1.2 MARKED FROZEN COMPLETE"
    )

    print("=" * 72)

    print(
        "Rows:",
        len(final),
    )

    print(
        "Errors:",
        final[
            "parse_error"
        ]
        .notna()
        .sum(),
    )


    print("\nOrientations:")
    print(
        final[
            "orientation"
        ]
        .value_counts(
            dropna=False
        )
    )


    print("\nUncertainty:")
    print(
        final[
            "uncertainty"
        ]
        .value_counts(
            dropna=False
        )
    )


    print("\nPrimary instability:")
    print(
        final[
            "primary_instability_source"
        ]
        .value_counts(
            dropna=False
        )
    )


    complete = (
        final[
            "candidate_assessment_complete"
        ]
        .astype("boolean")
    )

    repaired = (
        final[
            "selected_candidate_repaired"
        ]
        .astype("boolean")
    )


    print(
        "\nComplete candidate assessments:",
        int(
            complete
            .fillna(False)
            .sum()
        ),
    )

    print(
        "Incomplete candidate assessments:",
        int(
            (
                complete == False
            ).sum()
        ),
    )

    print(
        "Selected-candidate repairs:",
        int(
            repaired
            .fillna(False)
            .sum()
        ),
    )


    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
