import json
import re
from json_repair import repair_json


# Final orientation labels.
ORIENTATIONS = [
    "Favorable",
    "Unfavorable",
    "Neutral / Descriptive",
]

INSTABILITY_SOURCES = [
    "target_binding",
    "responsibility_attribution",
    "victimhood_or_suffering",
    "event_vs_target_valence",
    "quotation_or_attribution",
    "multiple_actors",
    "contextual_framing",
    "competing_readings",
    "other",
]

PROMPT_VERSION = (
    "contextual_target_evaluation_"
    "slow_adjudicator_magistral_v1_2_marked_bounded_frozen"
)


SYSTEM_PROMPT = """
You are the Slow-stage adjudicator for a contextual target-evaluation task.

Several independent language models have produced different interpretations
of the evaluative positioning toward ONE specified target.

Your task is not to vote.

You do not know which models produced the candidate interpretations, how
many models supported each interpretation, or whether one interpretation
represented a majority.

Compare the candidate interpretations against the supplied discourse
evidence and determine which target orientation is best supported.

The possible orientations are:

Favorable:
The target is supported, legitimized, defended, sympathized with,
protected, recognized, praised, or otherwise positioned favorably.
Victimhood or unjust suffering can position the target sympathetically.

Unfavorable:
The target is criticized, blamed, condemned, delegitimized, associated
with wrongdoing, portrayed as harmful or threatening, obstructive, or
otherwise positioned unfavorably.

Neutral / Descriptive:
The supplied discourse does not convey meaningful evaluative positioning
toward the specified target. Do not choose Neutral merely because the
evaluation is implicit or because explicit sentiment words are absent.

Evidence audit:

1. Evaluate only the SPECIFIED TARGET.

2. Keep the focal occurrence as the anchor. Use surrounding context to
interpret that occurrence rather than classify the overall political
position of the video.

3. Treat candidate orientations as hypotheses, not evidence.

4. Check which evaluative cues actually attach to the specified target.

5. Distinguish event valence from target evaluation. A negative event
affecting a target can position that target sympathetically.

6. Examine agency and responsibility. Determine who causes, enables,
prevents, condemns, threatens, protects, recognizes, or legitimizes whom.

7. Consider victimhood and suffering when they position the specified
target sympathetically.

8. Distinguish quoted or attributed evaluation from editorial endorsement.
Quoted evaluation can still carry a target-specific orientation.

9. Different actors in the same passage can receive different evaluations.

10. Use the surrounding discourse when it changes or resolves the
interpretation of the focal occurrence.

11. Compare competing readings and prefer the interpretation that receives
the strongest contextual support with the fewest unsupported assumptions.

12. You may select an orientation that does not appear among the candidate
interpretations if the discourse evidence supports it more strongly.

The candidate order is arbitrary.

Return a concise structured assessment. Do not produce an extended reasoning
trace. Record only the evidence needed to justify the adjudication.
""".strip()


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "slow_stage_adjudication_v12_marked_frozen",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "orientation": {
                    "type": "string",
                    "enum": ORIENTATIONS,
                },
                "selected_candidate": {
                    "type": "string",
                    "enum": ["A", "B", "C", "none"],
                },
                "candidate_assessments": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate": {
                                "type": "string",
                                "enum": ["A", "B", "C"],
                            },
                            "fit": {
                                "type": "string",
                                "enum": [
                                    "strong",
                                    "moderate",
                                    "weak",
                                ],
                            },
                            "evidence": {
                                "type": "string",
                            },
                            "limitation": {
                                "type": "string",
                            },
                        },
                        "required": [
                            "candidate",
                            "fit",
                            "evidence",
                            "limitation",
                        ],
                        "additionalProperties": False,
                    },
                },
                "instability_sources": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "string",
                        "enum": INSTABILITY_SOURCES,
                    },
                },
                "primary_instability_source": {
                    "type": "string",
                    "enum": INSTABILITY_SOURCES,
                },
                "uncertainty": {
                    "type": "string",
                    "enum": [
                        "low",
                        "moderate",
                        "high",
                    ],
                },
                "evidence_summary": {
                    "type": "string",
                },
            },
            "required": [
                "orientation",
                "selected_candidate",
                "candidate_assessments",
                "instability_sources",
                "primary_instability_source",
                "uncertainty",
                "evidence_summary",
            ],
            "additionalProperties": False,
        },
    },
}


def clean_text(value):
    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def build_user_prompt(row):
    candidates = []

    for letter, col in [
        ("A", "candidate_a"),
        ("B", "candidate_b"),
        ("C", "candidate_c"),
    ]:
        value = clean_text(
            row.get(col, "")
        )

        if value:
            candidates.append(
                f"Candidate {letter}: {value}"
            )

    return f"""
SPECIFIED TARGET:
{clean_text(row["annotation_target"])}

VIDEO DESCRIPTION:
{clean_text(row.get("video_description", "")) or "[none]"}

CONTEXT BEFORE:
{clean_text(row.get("context_before_4", "")) or "[none]"}

FOCAL OCCURRENCE:
{clean_text(row["focal_sentence_marked"])}

CONTEXT AFTER:
{clean_text(row.get("context_after_4", "")) or "[none]"}

COMPETING INTERPRETATIONS:
{chr(10).join(candidates)}

Compare the candidate interpretations against the discourse evidence.

Return:
1. the best-supported final orientation;
2. which candidate it corresponds to, or "none" if another orientation is
   better supported;
3. a brief evidence assessment for each candidate;
4. the source or sources of interpretive instability;
5. the primary source of instability;
6. residual uncertainty;
7. a short target-specific evidence summary.
""".strip()


def extract_json(text):
    text = str(text).strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\\s*",
            "",
            text,
        )
        text = re.sub(
            r"\\s*```$",
            "",
            text,
        )

    # First try strict JSON.
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # Extract the JSON object if the model added surrounding text.
    match = re.search(
        r"\\{.*\\}",
        text,
        flags=re.S,
    )

    if not match:
        raise json.JSONDecodeError(
            "No JSON object found",
            text,
            0,
        )

    candidate = match.group(0)

    # Try strict parsing again on the isolated object.
    try:
        return json.loads(candidate)

    except json.JSONDecodeError:
        pass

    # Final fallback: deterministic syntax repair only.
    # The repaired object still passes through validate_result(),
    # so labels and diagnostic fields must "satisfy" our frozen schema.
    repaired = repair_json(
        candidate,
        return_objects=False,
    )

    return json.loads(repaired)


def normalize_instability_sources(value):
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    normalized = []

    for item in value:
        if item is None:
            continue

        parts = re.split(
            r"\s*\|\s*",
            str(item).strip(),
        )

        for part in parts:
            part = part.strip()

            if (
                part
                and part not in normalized
            ):
                normalized.append(part)

    return normalized


def validate_result(result, row):
    result["instability_sources"] = (
        normalize_instability_sources(
            result.get(
                "instability_sources"
            )
        )
    )

    orientation = str(
        result.get(
            "orientation",
            "",
        )
    ).strip()

    if orientation not in ORIENTATIONS:
        raise ValueError(
            f"Invalid orientation: {orientation}"
        )

    result["orientation"] = orientation


    candidates = {
        "A": clean_text(
            row.get("candidate_a", "")
        ),
        "B": clean_text(
            row.get("candidate_b", "")
        ),
        "C": clean_text(
            row.get("candidate_c", "")
        ),
    }

    candidates = {
        key: value
        for key, value in candidates.items()
        if value
    }


    selected = str(
        result.get(
            "selected_candidate",
            "none",
        )
    ).strip()

    result[
        "selected_candidate_repaired"
    ] = False


    # Technical repair only.
    if (
        selected == "none"
        and orientation in candidates.values()
    ):
        selected = next(
            key
            for key, value in candidates.items()
            if value == orientation
        )

        result[
            "selected_candidate"
        ] = selected

        result[
            "selected_candidate_repaired"
        ] = True


    if selected != "none":
        if selected not in candidates:
            raise ValueError(
                f"Selected unavailable candidate: {selected}"
            )

        if candidates[selected] != orientation:
            raise ValueError(
                "selected_candidate does not match "
                "final orientation"
            )

    elif orientation in candidates.values():
        raise ValueError(
            'selected_candidate="none" but orientation '
            "is an available candidate"
        )


    assessments = result.get(
        "candidate_assessments",
        [],
    )

    assessed = [
        x.get("candidate")
        for x in assessments
        if x.get("candidate")
    ]

    unknown = (
        set(assessed)
        - set(candidates)
    )

    if unknown:
        raise ValueError(
            "Assessment contains unavailable "
            f"candidate(s): {sorted(unknown)}"
        )

    missing = sorted(
        set(candidates)
        - set(assessed)
    )

    result[
        "candidate_assessment_complete"
    ] = len(missing) == 0

    result[
        "candidate_assessments_missing"
    ] = missing


    sources = result[
        "instability_sources"
    ]

    bad_sources = (
        set(sources)
        - set(INSTABILITY_SOURCES)
    )

    if bad_sources:
        raise ValueError(
            "Invalid instability source(s): "
            f"{sorted(bad_sources)}"
        )

    if not sources:
        raise ValueError(
            "No instability source returned"
        )


    primary = str(
        result.get(
            "primary_instability_source",
            "",
        )
    ).strip()

    if primary not in INSTABILITY_SOURCES:
        raise ValueError(
            "Invalid primary instability source: "
            f"{primary}"
        )

    if primary not in sources:
        raise ValueError(
            "Primary instability source is not "
            "included in instability_sources"
        )

    result[
        "primary_instability_source"
    ] = primary


    uncertainty = str(
        result.get(
            "uncertainty",
            "",
        )
    ).strip()

    if uncertainty not in [
        "low",
        "moderate",
        "high",
    ]:
        raise ValueError(
            f"Invalid uncertainty: {uncertainty}"
        )

    result[
        "uncertainty"
    ] = uncertainty

    return result
