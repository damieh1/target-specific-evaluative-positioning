"""Extract surface occurrences once; context is indexed, never approximately matched."""
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from .common import boolean, digest, ensure_unique, file_hash, read_csv, write_csv, write_json
from .transcripts import SEGMENTATION_VERSION, index_sentences, read_transcript

CONTEXT_VERSION = "exact_mention_four_sentences_v1"
MARK_OPEN, MARK_CLOSE = "<FOCAL_TARGET>", "</FOCAL_TARGET>"
BASE_COLUMNS = ["occurrence_id", "video_id", "outlet", "posted_at", "sentence_id", "sentence_index",
                "transcript_text_sha256", "transcript_file_sha256", "transcript_path", "language_status",
                "sentence", "focal_sentence_original", "focal_sentence_marked", "annotation_target",
                "aspect_surface", "aspect_norm", "aspect", "core_family", "entity_type", "political_actor",
                "mention_start", "mention_end", "transcript_mention_start", "transcript_mention_end",
                "segment_ids", "start", "end", "timestamp_precision", "video_description", "metadata_source",
                "metadata_sha256", "context_before_4", "context_after_4", "before_sentence_ids", "after_sentence_ids",
                "context_version", "lexicon_sha256", "input_sha256"]


def norm_term(s):
    return s.replace("’", "'").replace("‘", "'").lower()


def lexicon(path):
    terms = {}
    for r in read_csv(path):
        if not boolean(r["include_core_absa"]):
            continue
        term = norm_term(r["aspect_norm"].strip())
        if not term or term in terms:
            raise ValueError(f"Empty or duplicate lexicon term: {term}")
        if not r["core_family"].strip() or not r["entity_type"].strip():
            raise ValueError("Lexicon family/type missing")
        r["political_actor"] = boolean(r["political_actor"])
        r["aspect_norm"] = term
        terms[term] = r
    if not terms:
        raise ValueError("No included core lexicon terms")
    # Longest alternative first: Israel's yields one mention, not Israel plus Israel's.
    alternatives = [re.escape(t).replace("'", "['’‘]") for t in sorted(terms, key=lambda x: (-len(x), x))]
    pattern = re.compile(r"(?<!\w)(?ai:" + "|".join(alternatives) + r")(?!\w)")
    return terms, pattern


def input_hash(row):
    return digest({k: row[k] for k in ("occurrence_id", "annotation_target", "video_description",
                                      "focal_sentence_original", "focal_sentence_marked", "context_before_4",
                                      "context_after_4", "context_version")})


def build(run):
    run = Path(run)
    cfg = json.loads((run/"inventory/resolved_config.json").read_text())
    selected_path = run/"inventory/selected_sources.json"
    selected = json.loads(selected_path.read_text())
    if not selected:
        raise ValueError("No eligible transcripts with metadata; inspect inventory/video_manifest.csv")
    out = run/"dataset"
    if (out/"manifest.json").exists():
        raise FileExistsError("Dataset is frozen. Use a new run directory to rebuild.")
    terms, pattern = lexicon(cfg["lexicon"])
    lex_hash = file_hash(cfg["lexicon"])
    rows, sentences, videos, warnings = [], [], [], []
    for source in selected:
        vid = source["video_id"]
        p = Path(source["transcript_path"])
        if file_hash(p) != source["transcript"]["file_sha256"]:
            raise ValueError(f"Transcript changed since inventory: {p}")
        full, segs = read_transcript(p, cfg["tsv_unit"])
        if MARK_OPEN in full or MARK_CLOSE in full:
            raise ValueError(f"Reserved focal delimiters occur in transcript {vid}")
        sents = index_sentences(vid, full, segs)
        sentences.extend(sents)
        before_n = len(rows)
        meta = source["metadata"]
        for ix, sent in enumerate(sents):
            sentence = sent["sentence"]
            if len(sentence.split()) > 120:
                warnings.append({"video_id": vid, "sentence_id": sent["sentence_id"], "issue": "long_sentence", "words": len(sentence.split())})
            for match in pattern.finditer(sentence):
                a, b = match.span()
                term = norm_term(match.group())
                entry = terms[term]
                before, after = sents[max(0, ix-4):ix], sents[ix+1:ix+5]
                absolute_a, absolute_b = sent["char_start"]+a, sent["char_start"]+b
                oid = digest(["occurrence_v1", vid, digest(full), absolute_a, absolute_b, term])
                row = {
                    "occurrence_id": oid, "video_id": vid, "outlet": source["outlet"], "posted_at": meta["posted_at"],
                    "sentence_id": sent["sentence_id"], "sentence_index": ix, "transcript_text_sha256": digest(full),
                    "transcript_file_sha256": source["transcript"]["file_sha256"], "transcript_path": str(p),
                    "language_status": source["language_status"], "sentence": sentence, "focal_sentence_original": sentence,
                    "focal_sentence_marked": sentence[:a]+MARK_OPEN+sentence[a:b]+MARK_CLOSE+sentence[b:],
                    "annotation_target": match.group(), "aspect_surface": match.group(), "aspect_norm": term,
                    "aspect": entry["core_family"], "core_family": entry["core_family"], "entity_type": entry["entity_type"],
                    "political_actor": entry["political_actor"], "mention_start": a, "mention_end": b,
                    "transcript_mention_start": absolute_a, "transcript_mention_end": absolute_b,
                    "segment_ids": sent["segment_ids"], "start": sent["start"], "end": sent["end"],
                    "timestamp_precision": sent["timestamp_precision"], "video_description": meta["video_description"],
                    "metadata_source": meta["source"], "metadata_sha256": meta["source_sha256"],
                    "context_before_4": " ".join(s["sentence"] for s in before),
                    "context_after_4": " ".join(s["sentence"] for s in after),
                    "before_sentence_ids": [s["sentence_id"] for s in before], "after_sentence_ids": [s["sentence_id"] for s in after],
                    "context_version": CONTEXT_VERSION, "lexicon_sha256": lex_hash,
                }
                row["input_sha256"] = input_hash(row)
                rows.append(row)
        videos.append({"video_id": vid, "outlet": source["outlet"], "n_sentences": len(sents),
                       "n_occurrences": len(rows)-before_n, "language_status": source["language_status"],
                       "transcript_text_sha256": digest(full), "transcript_path": str(p)})
    ensure_unique(rows, "occurrence_id")
    ensure_unique(sentences, "sentence_id")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg["lexicon"], out/"frozen_lexicon.csv")
    write_csv(out/"contexts.csv", rows, BASE_COLUMNS)
    write_csv(out/"sentences.csv", sentences)
    write_csv(out/"videos.csv", videos)
    write_csv(out/"segmentation_review.csv", warnings, ["video_id", "sentence_id", "issue", "words"])
    # Deterministic, outlet-balanced review sample. This is development QA, not validation.
    sample = []
    for outlet in sorted({r["outlet"] for r in rows}):
        pool = sorted((r for r in rows if r["outlet"] == outlet), key=lambda r: digest(["qa_v1", r["occurrence_id"]]))
        sample.extend(pool[:10])
    write_csv(out/"context_review_sample.csv", sample, BASE_COLUMNS)
    manifest = {"context_version": CONTEXT_VERSION, "segmentation_version": SEGMENTATION_VERSION,
                "selected_sources_sha256": file_hash(selected_path), "lexicon_sha256": lex_hash,
                "n_videos": len(videos), "n_sentences": len(sentences), "n_occurrences": len(rows),
                "zero_target_videos": sum(r["n_occurrences"] == 0 for r in videos),
                "language_status_counts": dict(Counter(v["language_status"] for v in videos)),
                "builder_source_hashes": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "files": {name: file_hash(out/name) for name in ("contexts.csv", "sentences.csv", "videos.csv", "frozen_lexicon.csv")}}
    write_json(out/"manifest.json", manifest)
    return verify(run)


def verify(run):
    run = Path(run)
    out = run/"dataset"
    manifest = json.loads((out/"manifest.json").read_text())
    if file_hash(run/"inventory/selected_sources.json") != manifest["selected_sources_sha256"]:
        raise ValueError("Frozen source selection changed")
    for name, expected in manifest["files"].items():
        if file_hash(out/name) != expected:
            raise ValueError(f"Frozen dataset changed: {name}")
    rows, sents = list(read_csv(out/"contexts.csv")), list(read_csv(out/"sentences.csv"))
    ensure_unique(rows, "occurrence_id")
    ensure_unique(sents, "sentence_id")
    by_video = defaultdict(list)
    for s in sents:
        by_video[s["video_id"]].append(s)
    for vid, seq in by_video.items():
        if [int(s["sentence_index"]) for s in seq] != list(range(len(seq))):
            raise ValueError(f"Invalid sentence order for {vid}")
    for r in rows:
        def require(condition, detail):
            if not condition:
                raise ValueError(f"Occurrence {r['occurrence_id']}: {detail}")
        seq = by_video[r["video_id"]]
        ix = int(r["sentence_index"])
        s = seq[ix]
        require(s["sentence_id"] == r["sentence_id"], "wrong sentence ID")
        require(r["sentence"] == r["focal_sentence_original"] == s["sentence"], "focal sentence differs from index")
        a, b = int(r["mention_start"]), int(r["mention_end"])
        require(0 <= a < b <= len(r["sentence"]), "invalid mention offsets")
        require(r["sentence"][a:b] == r["aspect_surface"] == r["annotation_target"], "target does not equal exact span")
        require(r["focal_sentence_marked"] == r["sentence"][:a]+MARK_OPEN+r["sentence"][a:b]+MARK_CLOSE+r["sentence"][b:], "wrong focal marking")
        before, after = seq[max(0, ix-4):ix], seq[ix+1:ix+5]
        require(r["context_before_4"] == " ".join(s["sentence"] for s in before), "wrong preceding context")
        require(r["context_after_4"] == " ".join(s["sentence"] for s in after), "wrong following context")
        require(json.loads(r["before_sentence_ids"]) == [s["sentence_id"] for s in before], "preceding sentence IDs differ")
        require(json.loads(r["after_sentence_ids"]) == [s["sentence_id"] for s in after], "following sentence IDs differ")
        require(r["input_sha256"] == input_hash(r), "wrong input hash")
    report = {"status": "STRUCTURAL_CHECKS_PASSED", "n_videos": len(by_video), "n_occurrences": len(rows),
              "note": "Structural checks do not establish semantic adequacy of segmentation, ASR, or evaluation."}
    write_json(out/"verification.json", report)
    return report
