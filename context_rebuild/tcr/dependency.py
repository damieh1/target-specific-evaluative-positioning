import json
import shutil
from collections import Counter
from pathlib import Path
from .common import append_json, digest, file_hash, read_csv, read_jsonl, write_csv, write_json
from .dataset import verify


def align_tokens(text, words):
    aliases = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}
    cursor, spans = 0, []
    for word in words:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        options = [str(word), aliases.get(str(word), str(word))]
        if word in {"``", "''", '"'}:
            options += ['"', "“", "”"]
        if "'" in str(word):
            options += [str(word).replace("'", "’"), str(word).replace("'", "‘")]
        match = next((w for w in options if text.startswith(w, cursor)), None)
        if match is None:
            raise ValueError(f"Cannot align parser token {word!r} at character {cursor}")
        spans.append((cursor, cursor+len(match)))
        cursor += len(match)
    if text[cursor:].strip():
        raise ValueError("Parser did not cover the entire canonical sentence")
    return spans


def enrich(row, parsed):
    a, b = int(row["mention_start"]), int(row["mention_end"])
    spans, arcs, words, rels = parsed["spans"], parsed["arcs"], parsed["words"], parsed["rels"]
    hits = [i for i, (x, y) in enumerate(spans) if x < b and y > a]
    details = {"token_alignment": "whole_token_span", "container_token_index": "", "container_token_form": "",
               "container_dep_rel": "", "container_head_form": "", "container_start": "", "container_end": "",
               "token_prefix": "", "token_suffix": ""}
    if not hits or spans[hits[0]][0] != a or spans[hits[-1]][1] != b:
        if len(hits) != 1:
            raise ValueError("Non-exact target span is not contained in a single parser token")
        i = hits[0]
        x, y = spans[i]
        sentence = row["sentence"]
        if not 0 <= x <= a < b <= y <= len(sentence):
            raise ValueError("Target is outside its containing parser token")
        if sentence[a:b] != row["aspect_surface"]:
            raise ValueError("Target surface differs from canonical character span")
        # Accept punctuation-bounded subparts (pro-Israel, Hezbollah's), never
        # arbitrary substrings such as Israel inside Israelis.
        if (a > x and (sentence[a-1].isalnum() or sentence[a-1] == "_")) or (
            b < y and (sentence[b].isalnum() or sentence[b] == "_")
        ):
            raise ValueError("Target boundary cuts through an alphanumeric token component")
        details.update(token_alignment="within_single_token", container_token_index=i+1,
                       container_token_form=words[i], container_dep_rel=rels[i],
                       container_head_form=words[arcs[i]-1] if arcs[i] else "ROOT",
                       container_start=x, container_end=y, token_prefix=sentence[x:a], token_suffix=sentence[b:y])
        # The parser's dependency belongs to the whole token, not necessarily
        # to this embedded target. Keep target-specific dependency fields blank.
        return {"target_token_indices": [], "target_head_token": "", "dep_rel": "", "head_form": "", "dep_form": "", **details}
    heads = [i for i in hits if arcs[i] == 0 or arcs[i]-1 not in hits]
    if len(heads) != 1:
        raise ValueError("Target span has no unique dependency head")
    i = heads[0]
    return {"target_token_indices": [j+1 for j in hits], "target_head_token": i+1,
            "dep_rel": rels[i], "head_form": words[arcs[i]-1] if arcs[i] else "ROOT", "dep_form": words[i], **details}


def parse_dataset(run, model, device, limit=None, predictor=None, cached_only=False):
    run = Path(run)
    verify(run)
    rows = list(read_csv(run/"dataset/contexts.csv"))
    out = run/"parsing"
    out.mkdir(parents=True, exist_ok=True)
    log = out/"sentence_parses.jsonl"
    contract = {"model": model, "device": device, "adapter_version": "supar_exact_alignment_v1"}
    if (out/"contract.json").exists() and json.loads((out/"contract.json").read_text()) != contract:
        raise ValueError("Parser contract changed; use a new run/parsing directory")
    write_json(out/"contract.json", contract)
    previous = {}
    for r in read_jsonl(log):
        if r["status"] == "ok":
            previous[r["sentence_id"]] = r
    todo = {r["sentence_id"]: r["sentence"] for r in rows if r["sentence_id"] not in previous}
    if cached_only and todo:
        raise ValueError(f"Cached-only mapping requires all sentence parses; {len(todo)} are missing. No GPU calls made.")
    if todo and predictor is None:
        import torch
        from supar import Parser
        # Match the user's existing loader; never silently substitute a checkpoint.
        parser = Parser.load(model, weights_only=True)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        parser.model = parser.model.to(device)
        def predictor(sentence):
            result = parser.predict([sentence], lang="en", prob=False, verbose=False, device=device)
            if len(result) != 1:
                raise ValueError("SuPar split a canonical sentence; inspect segmentation")
            p = result[0]
            return list(p.words), [int(x) for x in p.arcs], list(p.rels)
        versions = {"torch": torch.__version__}
        import importlib.metadata
        for name in ("supar", "stanza", "transformers"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not_installed"
        write_json(out/"environment.json", versions)
    for i, (sid, sentence) in enumerate(todo.items()):
        if limit is not None and i >= limit:
            break
        rec = {"sentence_id": sid, "sentence_sha256": digest(sentence), "model": model}
        try:
            words, arcs, rels = predictor(sentence)
            if not len(words) == len(arcs) == len(rels) or not words:
                raise ValueError("Malformed SuPar output lengths")
            if not all(0 <= x <= len(words) for x in arcs):
                raise ValueError("Invalid dependency head index")
            rec.update(status="ok", words=words, arcs=arcs, rels=rels, spans=align_tokens(sentence, words))
            previous[sid] = rec
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
        append_json(log, rec)
        if (i+1) % 100 == 0:
            print(f"Parsed {i+1}/{len(todo)} remaining sentences", flush=True)
    enriched = []
    for row in rows:
        rec = previous.get(row["sentence_id"])
        extra = {"parse_status": "pending", "parse_error": "", "target_token_indices": [],
                 "target_head_token": "", "dep_rel": "", "head_form": "", "dep_form": "",
                 "token_alignment": "", "container_token_index": "", "container_token_form": "",
                 "container_dep_rel": "", "container_head_form": "", "container_start": "", "container_end": "",
                 "token_prefix": "", "token_suffix": ""}
        if rec:
            if rec["sentence_sha256"] != digest(row["sentence"]):
                raise ValueError("Cached parser sentence changed")
            try:
                extra.update(enrich(row, rec), parse_status="ok")
            except ValueError as e:
                extra.update(parse_status="alignment_error", parse_error=str(e))
        enriched.append({**row, **extra})
    # Preserve the pre-fix enrichment once. Sentence parses and dataset remain untouched.
    backup = out/"before_token_mapping_v1"
    for name in ("parsed_contexts.csv", "parse_review.csv", "summary.json"):
        if (out/name).exists() and not (backup/name).exists():
            backup.mkdir(exist_ok=True)
            shutil.copy2(out/name, backup/name)
    write_csv(out/"parsed_contexts.csv", enriched, list(rows[0]) + list(extra) if rows else ["occurrence_id", "parse_status"])
    write_csv(out/"parse_review.csv", [r for r in enriched if r["parse_status"] != "ok"], list(enriched[0]) if enriched else ["occurrence_id"])
    write_csv(out/"contained_target_audit.csv", [r for r in enriched if r["token_alignment"] == "within_single_token"],
              list(enriched[0]) if enriched else ["occurrence_id"])
    write_json(out/"enrichment_contract.json", {"version": "exact_character_target_with_container_v1",
               "module_sha256": file_hash(__file__), "sentence_parser_contract_unchanged": contract,
               "rule": "Whole-token dependencies stay target-specific; embedded targets retain container dependencies separately."})
    report = {"n_occurrences": len(rows), "ok": sum(r["parse_status"] == "ok" for r in enriched),
              "pending_or_alignment_error": sum(r["parse_status"] != "ok" for r in enriched),
              "token_alignment_counts": dict(Counter(r["token_alignment"] for r in enriched if r["parse_status"] == "ok")),
              "new_sentence_parse_attempts": min(len(todo), limit) if limit is not None else len(todo),
              "parsed_contexts_sha256": file_hash(out/"parsed_contexts.csv")}
    write_json(out/"summary.json", report)
    return report
