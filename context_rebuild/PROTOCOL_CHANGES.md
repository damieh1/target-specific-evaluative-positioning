# Explicit protocol decisions for the rebuild

This document records the integrated rebuild protocol. It does not describe the final expanded inference run in every detail. The reference prompts are in `tcr/protocol_reference.json`; the separate Fast and Slow runners remain in the repository. The earlier `reference/` directory and test suite were not included in the supplied repository.

## Preserved

- Target-conditioned contextual evaluation and evaluative positioning, including implicit evaluation, victimhood, responsibility, quotation, and distinctions from editorial endorsement.
- The supplied core lexicon, including separate Muslims, Israeli_Military, and politician families. There is no added Gaza entry and no broad Israel/Palestine camp merger.
- Case-insensitive surface matching, one row per occurrence, and dependency enrichment using the specified SuPar checkpoint.
- Up to four preceding and four following sentences, bounded by the video transcript. Sentences without a core target remain in the index and context.
- The actual target surface is supplied as `annotation_target`. `core_family`, `entity_type`, and `political_actor` are metadata, not substituted for the focal target in prompts.
- Fast model identities from the supplied runner and commands: Qwen3-14B, Gemma3-12B, and the RedHatAI W8A8 Mistral Small 3.1 24B checkpoint. Slow uses the RedHatAI FP8 Magistral Small 2506 checkpoint.
- Qwen thinking disabled; Slow temperature, top-p and seed retained. The integrated runner uses its own output budget.
- Unanimous valid F/U/N results retained directly; disagreement adjudicated using anonymized distinct candidate orientations. Slow may choose an orientation absent from the candidates.

## Changed deliberately

1. **One sentence index.** Previously the parser and context builders independently stitched/reconstructed sentences, using different continuation lists and alignment fallbacks. The rebuild joins the chosen transcript's normalized segment text once, then applies punctuation-based segmentation with abbreviation guards. Caption line breaks alone are not sentence boundaries. The single index is reused for occurrence extraction and context. This removes reliance on the linker-list versions, old group IDs, nearest timestamps, and duplicate-sentence heuristics.
2. **Sentence segmentation is explicit, not claimed perfect.** It preserves basic punctuation-based segmentation while protecting a documented list such as `Mr.` and initials/acronyms. Abbreviations at a true sentence boundary and unpunctuated speech remain possible errors. Long sentences and a deterministic review sample are exported. No automatic sentence-length cap discards text. Normalization only collapses whitespace; transcript wording is retained.
3. **Exact mentions.** Matching is done against the canonical sentence's character spans before dependency parsing. Longer lexicon forms take precedence, so `Israel's` is not counted again as `Israel`. Straight/curly apostrophes are equivalent for lookup, but original surfaces are retained. The matched span is marked `<FOCAL_TARGET>...</FOCAL_TARGET>` in both stages. Repeated mentions in the same sentence therefore have different inputs.
4. **Stable identity.** Occurrence IDs depend on video ID, normalized transcript hash, absolute character positions, and matched lexicon form. They do not depend on CSV order. Sentence IDs additionally record the segmentation version. Model reuse depends on the exact marked input and protocol/deployment contract. Lexicon-family changes do not change occurrence identity; the changed lexicon hash and metadata remain recorded.
5. **Source selection.** JSON, TSV, SRT, VTT, and TXT are supported recursively. Equivalent normalized texts use priority JSON, TSV, SRT, VTT, TXT. Divergent valid texts are held for documented selection. All candidates, invalid alternatives, source hashes, original language records, and metadata source rows are retained. The selected file hash is checked again before building. Untimed transcripts are not assigned synthetic times.
6. **Eligibility.** Explicit audit decisions are preferred. English-only lists can document eligibility when audit rows are absent; these retain a separate status. Conflicts or invalid recorded decisions are held. Absence of evidence is not an exclusion decision. This permits the recovered AJ list-only transcripts to enter transparently without fabricating scores.
7. **F/U/N recoding.** Fast retains the original four raw substantive orientations and review/null option. Mixed is collapsed to Neutral / Descriptive before agreement and routing in this rebuild. Both raw and collapsed patterns are saved, including the number of rows recoded. A null response is held as unresolved and never collapsed. This defines the new run's policy; it does not establish when historical scripts performed their recoding.
8. **Uniform Fast generation.** The old pilot/corpus runner used 220 output tokens; Mistral used 250, with a 400-token JSON retry. The rebuild uses a fixed 400-token budget for all Fast calls and retries, temperature 0, top-p 1, and seed 20260831. This is recorded as a new protocol. Original instructions requiring null when review is requested are enforced consistently.
9. **Minimal prompt additions.** The original substantive system text is retained. A short shared instruction explains the focal delimiters and that source text is evidence, not executable instructions. Empty evidence fields use `[none]` consistently for Fast and Slow. Slow's bounded-reasoning wrapper from the runner is retained. Candidate construction deduplicates orientations and uses reproducible hash-based ordering, rather than assuming input columns already conceal vote support.
10. **Complete outputs only.** Fenced/plain JSON is parsed correctly; reasoning before a closed `</think>` marker is removed. Truncated objects, duplicate keys, multiple decisions, wrong types, inconsistent candidate selections, and missing candidate assessments are rejected. No regex fallback salvages an orientation from a partial reasoning response. No syntax repair fills absent diagnostic fields. The original runner's recovery flags and results remain untouched.
11. **Context capacity.** Input tokens are checked using the served tokenizer with the same messages and template options. The full output budget is reserved. Overlength cases stay pending; the evidence window is not silently shortened. Inference records the token count and full response.
12. **No fixed historical row counts.** The rebuilt corpus determines its counts. Zero-target videos remain in the video table. Failed parsing and incomplete classifications remain explicit, with the complete occurrence denominator retained.

## Implications for the paper

The pipeline measures positioning conveyed in supplied discourse; attributed or quoted evaluation need not indicate an outlet's endorsement. Lexicon matching does not establish referential meaning beyond the supplied term/context and cannot recover unlisted paraphrases or targets omitted by ASR.

The final N class combines neutral/descriptive and the few recoded Mixed Fast orientations. The raw distinction remains available. Disagreement is classification instability, not automatically error. Slow's uncertainty and instability categories are model assessments, not independently validated causes of disagreement.

The fixed four-sentence radius is reproducible but does not guarantee that every relevant antecedent or attribution is inside the window. Context QA and blind expert validation address different questions. Changed evidence should not silently inherit an expert label assigned under materially different input. Do not report the older validation scores as measurements of the revised implementation before checking that correspondence.

## Verification scope

The original documentation described tests for the rebuild workflow. The test files are not part of this repository snapshot.

SuPar model loading, tokenizer packages, GPU memory behavior, vLLM model execution, real corpus counts, real contextual adequacy, and final scientific validity must be checked on the actual VMs/data. No test fixture output should be interpreted as a research finding.
