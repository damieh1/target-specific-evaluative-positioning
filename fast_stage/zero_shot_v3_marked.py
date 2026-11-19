from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time

import pandas as pd
from openai import OpenAI


PROMPT_VERSION = "contextual_target_evaluation_zero_shot_v2"

ORIENTATIONS = [
    "Favorable",
    "Unfavorable",
    "Neutral / Descriptive",
    "Mixed",
]


SYSTEM_PROMPT = """
You are performing a research annotation task called contextual target
evaluation.

Your task is NOT conventional sentiment analysis.

Determine the evaluative positioning conveyed toward ONE specified
target at ONE focal occurrence, using the supplied local discourse
context to interpret that occurrence.

There are four substantive orientations:

Favorable:
The target is supported, legitimized, defended, sympathized with,
protected, recognized, praised, or otherwise positioned favorably.
Victimhood or unjust suffering may position the target sympathetically.

Unfavorable:
The target is criticized, blamed, condemned, delegitimized, associated
with wrongdoing, portrayed as harmful or threatening, obstructive, or
otherwise positioned unfavorably.

Neutral / Descriptive:
No meaningful evaluative positioning toward the specified target is
conveyed. Do not choose Neutral merely because evaluation is implicit
or no sentiment words appear.

Mixed:
Both favorable and unfavorable positioning clearly apply to the SAME
specified target.

You must additionally determine whether meaningful interpretive
ambiguity is present.

Meaningful ambiguity can involve sarcasm, irony, competing literal and
implied readings, unclear attribution, unclear referential scope, or
other genuinely competing interpretations.

Ambiguity is NOT an orientation.

Ambiguity does NOT automatically require human review.

If ambiguity is present but the supplied context resolves the target
orientation reliably, assign the resolved orientation,
set ambiguity_detected=true, and needs_human_review=false.

If the target orientation cannot be resolved reliably even after using
the supplied context, set orientation=null and
needs_human_review=true.

Do not use human review merely because the case is indirect, sarcastic,
complex, or requires inference. Use it only when the available evidence
does not support a sufficiently reliable orientation.

Important rules:

1. Evaluate only the SPECIFIED TARGET.

2. The focal occurrence is the anchor. Context helps interpret it.
Do not classify the overall political stance of the video.

3. Different actors in the same passage may receive different
evaluations.

4. Event valence is not target evaluation. A negative event affecting
a target may position that target sympathetically.

5. Political framing involving victimhood, responsibility, legitimacy,
recognition, credibility, threat, protection, exclusion, success,
failure, or wrongdoing can convey evaluation without sentiment words.

6. Quoted or attributed evaluation retains its target-specific
orientation. The journalist or outlet does not need to endorse it.

7. Distinguish evaluation toward the target from evaluation of the
speaker and from editorial endorsement by the outlet.

8. For sarcasm and irony, classify the implied orientation rather than
literal lexical polarity when the pragmatic reading can be resolved
from context.

9. Political disagreement alone does not automatically imply an
unfavorable evaluation.

Return only the requested structured result. Give a short
target-specific justification, not an extended reasoning trace.
""".strip()


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "contextual_target_evaluation_v2",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "orientation": {
                    "anyOf": [
                        {
                            "type": "string",
                            "enum": ORIENTATIONS,
                        },
                        {
                            "type": "null"
                        },
                    ]
                },
                "ambiguity_detected": {
                    "type": "boolean"
                },
                "needs_human_review": {
                    "type": "boolean"
                },
                "reason": {
                    "type": "string"
                },
            },
            "required": [
                "orientation",
                "ambiguity_detected",
                "needs_human_review",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def atomic_save(df, path):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(path)


def build_user_prompt(row):

    return f"""
SPECIFIED TARGET:
{clean(row["annotation_target"])}

VIDEO DESCRIPTION:
{clean(row["video_description"])}

CONTEXT BEFORE:
{clean(row["context_before_4"])}

FOCAL OCCURRENCE:
{clean(row["focal_sentence_marked"])}

CONTEXT AFTER:
{clean(row["context_after_4"])}

Determine:

1. the evaluative orientation toward the specified target;
2. whether meaningful interpretive ambiguity is present;
3. whether the orientation nevertheless can be resolved reliably.

If it cannot be resolved reliably, return orientation=null and
needs_human_review=true.
""".strip()


def validate_output(parsed):

    orientation = parsed.get(
        "orientation"
    )

    ambiguity = parsed.get(
        "ambiguity_detected"
    )

    review = parsed.get(
        "needs_human_review"
    )

    if ambiguity not in {
        True,
        False,
    }:
        raise ValueError(
            "Invalid ambiguity_detected"
        )

    if review not in {
        True,
        False,
    }:
        raise ValueError(
            "Invalid needs_human_review"
        )

    if review:

        if orientation is not None:
            raise ValueError(
                "Human-review case must have "
                "orientation=null"
            )

    else:

        if orientation not in ORIENTATIONS:
            raise ValueError(
                f"Invalid orientation: {orientation}"
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--model-tag",
        required=True,
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        input_path,
        dtype={"video_id": str},
        low_memory=False,
    )

    assert len(df) == 120

    forbidden = [
        c for c in df.columns
        if any(
            token in c.lower()
            for token in [
                "gold",
                "label",
                "prediction",
                "sentiment",
            ]
        )
    ]

    if forbidden:
        raise RuntimeError(
            f"Input is not blind: {forbidden}"
        )

    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
    )

    models = client.models.list().data

    if not models:
        raise RuntimeError(
            "No model found."
        )

    served_model = models[0].id

    print("=" * 72)
    print("CONTEXTUAL ZERO-SHOT V2")
    print("=" * 72)
    print("Model:", served_model)
    print("Rows:", len(df))
    print("Prompt:", PROMPT_VERSION)

    if output_path.exists():

        results = pd.read_csv(
            output_path,
            dtype={"video_id": str},
            low_memory=False,
        )

        done = set(
            results["pilot_id"].astype(int)
        )

    else:

        results = pd.DataFrame()
        done = set()

    for _, row in df.iterrows():

        pilot_id = int(
            row["pilot_id"]
        )

        if pilot_id in done:
            continue

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(row),
            },
        ]

        parsed = None
        raw_output = ""
        error = ""
        latency = None

        for attempt in range(
            1,
            args.max_retries + 1,
        ):

            try:

                start = time.time()

                response = (
                    client.chat.completions.create(
                        model=served_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=220,
                        response_format=RESPONSE_FORMAT,
                    )
                )

                latency = (
                    time.time() - start
                )

                raw_output = (
                    response
                    .choices[0]
                    .message
                    .content
                    or ""
                )

                parsed = json.loads(
                    raw_output
                )

                validate_output(
                    parsed
                )

                error = ""
                break

            except Exception as exc:

                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                time.sleep(
                    2 * attempt
                )

        if parsed is None:

            orientation = None
            ambiguity = None
            review = None
            reason = ""

        else:

            orientation = parsed[
                "orientation"
            ]

            ambiguity = parsed[
                "ambiguity_detected"
            ]

            review = parsed[
                "needs_human_review"
            ]

            reason = clean(
                parsed["reason"]
            )

        result = {
            "pilot_id":
                pilot_id,

            "video_id":
                clean(row["video_id"]),

            "annotation_target":
                clean(
                    row["annotation_target"]
                ),

            "model_tag":
                args.model_tag,

            "served_model":
                served_model,

            "prompt_version":
                PROMPT_VERSION,

            "orientation":
                orientation,

            "ambiguity_detected":
                ambiguity,

            "needs_human_review":
                review,

            "reason":
                reason,

            "raw_output":
                raw_output,

            "parse_error":
                error,

            "latency_sec":
                latency,

            "timestamp_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        results = pd.concat(
            [
                results,
                pd.DataFrame([result]),
            ],
            ignore_index=True,
        )

        results = (
            results
            .drop_duplicates(
                subset=["pilot_id"],
                keep="last",
            )
            .sort_values("pilot_id")
            .reset_index(drop=True)
        )

        atomic_save(
            results,
            output_path,
        )

        # Deliberately hide predictions
        print(
            f"[{len(results):3d}/120] completed"
        )

    manifest = {
        "model_tag":
            args.model_tag,

        "served_model":
            served_model,

        "prompt_version":
            PROMPT_VERSION,

        "prompt_sha256":
            sha256_text(
                SYSTEM_PROMPT
            ),

        "input_file":
            str(input_path),

        "input_sha256":
            sha256_file(
                input_path
            ),

        "output_file":
            str(output_path),

        "temperature":
            0.0,

        "max_tokens":
            220,

        "orientations":
            ORIENTATIONS,

        "n_input":
            len(df),

        "n_output":
            len(results),

        "n_errors":
            int(
                results["parse_error"]
                .fillna("")
                .astype(str)
                .str.len()
                .gt(0)
                .sum()
            ),

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    manifest_file = (
        output_path
        .with_suffix(
            ".manifest.json"
        )
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print("Rows:", len(results))
    print("Errors:", manifest["n_errors"])
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
