from pathlib import Path
from datetime import datetime, timezone
import argparse
import asyncio
import hashlib
import importlib.util
import json

import pandas as pd
from openai import AsyncOpenAI


# Load the original prompt and response schema.
PROTOCOL_PATH = (
    Path(__file__).resolve().parent
    / "zero_shot_v3_marked.py"
)

spec = importlib.util.spec_from_file_location(
    "zero_shot_v2_protocol",
    PROTOCOL_PATH,
)

v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


# Keep the original prompt and schema.
SYSTEM_PROMPT = v2.SYSTEM_PROMPT
RESPONSE_FORMAT = v2.RESPONSE_FORMAT
PROMPT_VERSION = "v3_marked_focal_occurrence_compact_retry"
_base_build_user_prompt = v2.build_user_prompt

def build_user_prompt(row):
    return (
        _base_build_user_prompt(row)
        + "\n\nOUTPUT FORMAT REQUIREMENT: "
        + "Return one compact JSON object without blank lines "
        + "or trailing whitespace. Include every required key "
        + "and close the object with }."
    )

validate_output_v2 = v2.validate_output


def validate_output(parsed):
    """Validate corpus output while retaining provisional review labels.

    The frozen v2 validator requires orientation=null whenever
    needs_human_review=True. For full-corpus inference we retain any
    provisional orientation returned by the model and treat the review
    flag separately during downstream adjudication.

    All other v2 validation rules remain unchanged.
    """

    check = dict(parsed)

    if (
        check.get("needs_human_review") is True
        and check.get("orientation") is not None
    ):
        check["orientation"] = None

    return validate_output_v2(check)
clean = v2.clean



def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sha256_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def atomic_save(df, path):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(path)




async def infer_one(
    client,
    served_model,
    row,
    semaphore,
    max_retries,
):

    corpus_row_id = int(
        row["corpus_row_id"]
    )

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

    async with semaphore:

        for attempt in range(
            1,
            max_retries + 1,
        ):

            try:

                response = (
                    await client
                    .chat
                    .completions
                    .create(
                        model=served_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=512,
                        response_format=RESPONSE_FORMAT,
                    )
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

                parsed = None

                await asyncio.sleep(
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


    return {
        "corpus_row_id":
            corpus_row_id,

        "occurrence_id":
            clean(row["occurrence_id"]),

        "video_id":
            clean(row["video_id"]),

        "annotation_target":
            clean(
                row["annotation_target"]
            ),

        "aspect":
            clean(
                row.get("aspect", "")
            ),

        "aspect_surface":
            clean(
                row.get(
                    "aspect_surface",
                    "",
                )
            ),

        "core_family":
            clean(
                row.get(
                    "core_family",
                    "",
                )
            ),

        "model_tag":
            row["_model_tag"],

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

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }




async def run(args):

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    df = pd.read_csv(
        input_path,
        dtype={
            "video_id": str,
        },
        low_memory=False,
    )


    print("Input rows:", len(df))


    required = [
        "occurrence_id",
        "video_id",
        "annotation_target",
        "video_description",
        "context_before_4",
        "focal_sentence_original",
        "focal_sentence_marked",
        "context_after_4",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )


    # Stable row identifier for this exact input file
    df = df.reset_index(
        drop=True
    )

    df["corpus_row_id"] = (
        df.index.astype(int)
    )


    # Pass the target, not earlier predictions.
    df["annotation_target"] = (
        df["annotation_target"]
        .fillna("")
        .astype(str)
        .str.strip()
    )



    if (
        df["annotation_target"]
        .eq("")
        .any()
    ):

        bad = int(
            df["annotation_target"]
            .eq("")
            .sum()
        )

        raise RuntimeError(
            f"{bad} rows have empty targets"
        )


    df["_model_tag"] = (
        args.model_tag
    )

    # Client / served model

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
    )

    models = (
        await client.models.list()
    ).data

    if not models:

        raise RuntimeError(
            "No model found."
        )

    served_model = (
        models[0].id
    )


    print("Model:", served_model)
    print("Target column: annotation_target")
    print("Prompt:", PROMPT_VERSION)
    print("Concurrency:", args.concurrency)

    # Resume
    if output_path.exists():

        results = pd.read_csv(
            output_path,
            dtype={
                "video_id": str,
            },
            low_memory=False,
        )

        successful = (
            results["parse_error"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        done = set(
            results.loc[
                successful,
                "corpus_row_id",
            ].astype(int)
        )

        print(
            "Already completed:",
            len(done),
        )

    else:

        results = pd.DataFrame()
        done = set()


    remaining = df[
        ~df["corpus_row_id"]
        .isin(done)
    ].copy()


    print(
        "Remaining:",
        len(remaining),
    )


    semaphore = asyncio.Semaphore(
        args.concurrency
    )

    
    # Process in save batches
    
    rows = [
        row
        for _, row
        in remaining.iterrows()
    ]


    for start in range(
        0,
        len(rows),
        args.save_batch_size,
    ):

        batch = rows[
            start:
            start + args.save_batch_size
        ]


        tasks = [
            infer_one(
                client=client,
                served_model=served_model,
                row=row,
                semaphore=semaphore,
                max_retries=args.max_retries,
            )
            for row in batch
        ]


        batch_results = (
            await asyncio.gather(
                *tasks
            )
        )


        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    batch_results
                ),
            ],
            ignore_index=True,
        )


        results = (
            results
            .drop_duplicates(
                subset=[
                    "corpus_row_id"
                ],
                keep="last",
            )
            .sort_values(
                "corpus_row_id"
            )
            .reset_index(
                drop=True
            )
        )


        atomic_save(
            results,
            output_path,
        )


        errors = int(
            results[
                "parse_error"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )


        print(
            f"[{len(results):5d}/{len(df)}] "
            f"saved | errors={errors}"
        )

    # Manifest

    errors = int(
        results[
            "parse_error"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )


    manifest = {
        "model_tag":
            args.model_tag,

        "served_model":
            served_model,

        "protocol":
            "v2_direct_orientation",

        "prompt_version":
            PROMPT_VERSION,

        "protocol_file":
            str(PROTOCOL_PATH),

        "protocol_file_sha256":
            sha256_file(PROTOCOL_PATH),

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

        "target_column": "annotation_target",

        "output_file":
            str(output_path),

        "temperature":
            0.0,

        "concurrency":
            args.concurrency,

        "n_input":
            len(df),

        "n_output":
            len(results),

        "n_errors":
            errors,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


    manifest_path = (
        output_path
        .with_suffix(
            ".manifest.json"
        )
    )


    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )


    print("\n" + "=" * 72)
    print("COMPLETE")
    print("=" * 72)

    print(
        "Rows:",
        len(results),
    )

    print(
        "Errors:",
        errors,
    )

    print(
        "Saved:",
        output_path,
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
        default="http://127.0.0.1:8000/v1",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--save-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    asyncio.run(
        run(args)
    )


if __name__ == "__main__":
    main()
