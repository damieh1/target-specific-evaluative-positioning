# TikTok Contextual Target Evaluation

This repository contains the code we use to examine target-conditioned evaluative positioning in international news coverage on TikTok. We build contextual inputs from English-language transcripts and video descriptions, parse the focal sentences with SuPar, and classify the positioning of individual targets using three Fast models and a Slow adjudicator.

The scripts cover the expanded September 2026 measurement run and the subsequent statistical analysis. The final measurement contains 22,540 evaluable target occurrences across 3,955 videos from eight news outlets. The source dataset contains 22,541 occurrences; we excluded one unresolved Fast abstention. We retained 17,000 unanimous Fast classifications and adjudicated 5,540 disagreements.

## Repository

| Directory | Contents |
|---|---|
| `context_rebuild/` | Input inventory, target extraction, context construction and dependency parsing. The directory also contains the earlier integrated inference implementation; see its [README](context_rebuild/README.md). |
| `fast_stage/` | Fast prompts, inference, retry of invalid outputs, and model agreement. |
| `slow_stage/` | Preparation of anonymized candidate orientations and Slow adjudication. |
| `evaluation/` | Final measurement, expert validation, descriptive statistics, regression models and syntactic analysis. |
| `provenance/` | Aggregate counts and file hashes from the recorded measurement run. |

## Setup

Use Python 3.10 or newer. Install the packages for inference and analysis:

```bash
python -m pip install -r requirements.txt
```

For dependency parsing, use a compatible SuPar and PyTorch installation with the `biaffine-dep-en` checkpoint. Run the LLMs on a compatible OpenAI-style inference server (the original workflows use local model servers). The requirements list names the dependencies; it does not record the exact versions of the original GPU environment.

## Data preparation

The code starts with existing transcripts, language-screening files and video metadata. It does not download TikTok videos or run Whisper. Place the private inputs under `data/`, edit `context_rebuild/config_vm.example.json` (or a local copy), and follow the [context construction steps](context_rebuild/README.md). The lexicon and model prompts remain in the repository.

The following commands assume you run them from the repository root and store private inputs and intermediate results in the ignored `data/` and `outputs/` directories.

## Fast stage

Run each model independently against the same `parsed_contexts.csv` or `contexts.csv` input. Use the same input file for all three models. We use Qwen3-14B, Gemma3-12B and Mistral Small 3.1 24B. The model tag must match the model loaded on the server.

```bash
python fast_stage/run_fast_inference.py --input outputs/rebuild/parsing/parsed_contexts.csv --output outputs/qwen.csv --model-tag Qwen/Qwen3-14B
python fast_stage/run_fast_inference.py --input outputs/rebuild/parsing/parsed_contexts.csv --output outputs/gemma.csv --model-tag google/gemma-3-12b-it
python fast_stage/run_fast_inference.py --input outputs/rebuild/parsing/parsed_contexts.csv --output outputs/mistral.csv --model-tag RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8
```

Load the corresponding model before each command and check that the served model matches the model tag. The runners resume using saved row IDs; only resume when the input, its row order, and the served model are unchanged. If a run contains invalid outputs, use `retry_invalid_fast_outputs.py` with the same input and model tag and the existing output file. The compact retry uses a different output budget and prompt suffix; keep that distinction. Do not use the integrated `context_rebuild` inference commands as substitutes for these runners.

Compare the Fast outputs:

```bash
python fast_stage/compare_fast_models.py \
  --contexts outputs/rebuild/dataset/contexts.csv \
  --qwen outputs/qwen.csv --gemma outputs/gemma.csv \
  --mistral outputs/mistral.csv --output-dir outputs/agreement
```

We retain the raw Fast labels and map Mixed to Neutral / Descriptive for the final three-way classification. The comparison script also calculates pairwise agreement, Cohen's kappa and nominal Krippendorff's alpha.

## Slow stage and final measurement

Prepare the disagreement cases. We remove the unresolved abstention and shuffle the distinct candidate orientations without identifying the Fast models or their vote counts.

```bash
python slow_stage/prepare_input.py \
  --agreement outputs/agreement/fast_agreement_full_22541.csv \
  --output-dir outputs/slow_input

python slow_stage/run_adjudication.py \
  --input outputs/slow_input/slow_stage_input_5540.csv \
  --output outputs/slow.csv

python evaluation/build_measurement.py \
  --contexts outputs/rebuild/dataset/contexts.csv \
  --fast outputs/agreement/fast_agreement_full_22541.csv \
  --slow outputs/slow.csv --output-dir outputs/measurement
```

The Slow runner uses `RedHatAI/Magistral-Small-2506-FP8`, a temperature of 0.7, top-p of 0.95, 7,000 output tokens and seed 20260831. It preserves the original JSON repair and partial-output recovery rules. The recorded measurement manifest includes six recovered partial outputs. Use the same Slow input file when resuming. The final assembly checks the identities and routing of the inputs, and writes `fast_slow_evaluable.csv` and the excluded occurrence separately.

## Statistical analysis

The scripts below use the frozen final measurement and the expert-reference files. They save the results in the specified output directory.

```bash
python evaluation/validate_reference.py \
  --reference data/expert_reference_550_final.csv \
  --output-dir outputs/validation

python evaluation/second_expert.py \
  --reference data/expert_reference_550_final.csv \
  --second-expert data/blind_second_expert_eval_sample_150.csv \
  --output-dir outputs/expert_agreement

python evaluation/descriptive_analysis.py \
  --measurement outputs/measurement/fast_slow_evaluable.csv \
  --output-dir outputs/descriptive

python evaluation/outlet_models.py \
  --measurement outputs/measurement/fast_slow_evaluable.csv \
  --output-dir outputs/regression

python evaluation/syntactic_analysis.py \
  --measurement outputs/measurement/fast_slow_evaluable.csv \
  --parses outputs/rebuild/parsing/parsed_contexts.csv \
  --output-dir outputs/syntax
```

The validation script checks the SHA-256 hash of the 550-case reference and calculates accuracy, Wilson confidence intervals, macro-F1, Cohen's kappa, the exact McNemar and Fisher tests, and post-stratified corpus accuracy. The second script reproduces the 150-case sampling design and assesses agreement between the experts. Omit `--second-expert` to generate only the blind sample.

The descriptive scripts calculate distributions and monthly evaluative balance. The regression script fits separate binomial models for favorable and unfavorable positioning and uses video-clustered standard errors. The syntactic script counts governing forms by target family, syntactic role and orientation. It uses surface forms when no lemmas are available and preserves the stopword list from the supplied statistical notebook.

## Data and reproducibility

We do not include videos, transcripts, metadata, annotation files, model outputs or generated results. The [measurement manifest](provenance/expanded_20260921/measurement_manifest.json) records counts and SHA-256 hashes for the original files. Use the original input files and model environment to reproduce the reported results. Rebuilding or re-running inference can change outputs; the manifest identifies the recorded run, not a guarantee of bit-identical regeneration.
