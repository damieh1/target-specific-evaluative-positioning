# Context construction and dependency parsing

We build a single sentence index from each selected transcript, match occurrences of the 31 terms in the lexicon, and mark the focal target. Each input contains the video description, focal sentence, and up to four preceding and four following sentences. We link the exact target span to its dependency parse.

## Inputs

Edit `config_vm.example.json` to point to the transcript directories, language-screening files, metadata and lexicon. Paths are relative to the configuration file. The script reads JSON, TSV, SRT, VTT and TXT transcripts. Set `tsv_unit` to match the timestamps in the source files. The example uses milliseconds.

The inventory assumes language screening has already taken place. It reads the recorded eligibility decisions instead of rerunning ASR or language identification. Supply the final screening outputs used in the study. The metadata must identify the video, outlet, posting date and description. Inspect the inventory for missing files and conflicting decisions before building the contexts.

## Build and parse

Run the commands from the repository root. Use a new run directory when the source data or configuration changes.

```bash
python context_rebuild/pipeline.py inventory --config context_rebuild/config_vm.example.json --run outputs/rebuild
python context_rebuild/pipeline.py build --run outputs/rebuild
python context_rebuild/pipeline.py verify --run outputs/rebuild
python context_rebuild/pipeline.py parse --run outputs/rebuild --model biaffine-dep-en --device cuda:0 --limit 20
python context_rebuild/pipeline.py parse --run outputs/rebuild --model biaffine-dep-en --device cuda:0
```

The limited parsing run checks the parser and alignment before processing the remaining sentences. The full run resumes from the saved parses. The code does not assign a whole-token dependency to a target embedded within a larger token. It records the containing token separately.

| Output (under the run directory) | Contents |
|---|---|
| `inventory/` | Source selection, eligibility, issues and source hashes. |
| `dataset/contexts.csv` | Exact focal occurrences and their contextual inputs. |
| `dataset/` | Sentence index, corpus inventory and verification records. |
| `parsing/parsed_contexts.csv` | Context rows with dependency information. |
| `parsing/parse_review.csv` | Parsing and alignment failures. |
| `parsing/contained_target_audit.csv` | Targets embedded within larger parser tokens. |

## Inference implementations

For the recorded expanded measurement, use the separate `fast_stage/` and `slow_stage/` scripts described in the main README.

This directory also retains an integrated inference workflow from the rebuild implementation. It uses the same target-conditioned concept, but its prompt wrappers, token budgets, handling of review/null labels, response validation and routing differ from the separate runners. We preserve it so the original rebuild protocol remains inspectable. Do not mix its outputs with the recorded expanded measurement without checking the correspondence of inputs and labels.

To use the integrated workflow independently, serve the corresponding model and run `pipeline.py infer --help`. Perform a preflight check (`--preflight-only`) and a limited inference run (`--limit`) before processing the full slot. Run all three Fast slots before `route`, then run the Slow slot before `finalize`. The default output budgets are 400 tokens for Fast and 6,000 for Slow. Use a server context length that accommodates the complete input and output budgets. Keep the deployment ID fixed only for the same actual model deployment.

The integrated workflow writes its own logs, routing tables, and completed and pending occurrence files. It does not silently shorten the contextual evidence or treat incomplete runs as complete.
