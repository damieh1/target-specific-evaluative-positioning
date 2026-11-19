"""Read actual transcript formats and make one deterministic sentence index.

Offsets refer to the whitespace-normalized transcript, never to audio frames.
Segment times bound sentences; they do not locate the target word precisely.
"""
import csv
import html
import json
import math
import re
from pathlib import Path
from .common import digest, norm_text, video_id

SEGMENTATION_VERSION = "punctuation_abbreviation_guards_v1"
FORMATS = {".json": 0, ".tsv": 1, ".srt": 2, ".vtt": 3, ".txt": 4}
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "gen.",
                 "lt.", "col.", "sgt.", "rev.", "sen.", "rep.", "st.", "vs.",
                 "e.g.", "i.e.", "no."}
END = re.compile(r'''[.!?…]+["'”’\)\]]*(?=\s|$)''')


def seconds(s):
    parts = s.replace(",", ".").split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid caption timestamp {s}")
    return sum(float(v) * (60 ** i) for i, v in enumerate(reversed(parts)))


def read_transcript(path, tsv_unit="milliseconds"):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("Empty transcript")
    suffix = path.suffix.lower()
    segments = []
    if suffix == ".json":
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Expected Whisper JSON object")
        if obj.get("video_id") and video_id(obj["video_id"]) != video_id(path.stem):
            raise ValueError("JSON and filename video IDs disagree")
        if isinstance(obj.get("segments"), list) and obj["segments"]:
            for i, s in enumerate(obj["segments"]):
                segments.append({"id": str(s.get("id", i)), "start": s.get("start"),
                                 "end": s.get("end"), "text": norm_text(s.get("text"))})
        elif isinstance(obj.get("text"), str):
            segments = [{"id": "0", "start": None, "end": None, "text": norm_text(obj["text"])}]
        else:
            raise ValueError("No Whisper segments or text")
    elif suffix == ".tsv":
        if tsv_unit not in {"milliseconds", "seconds"}:
            raise ValueError("TSV timestamp unit must be explicitly milliseconds or seconds")
        scale = 1000 if tsv_unit == "milliseconds" else 1
        rows = csv.DictReader(text.splitlines(), delimiter="\t")
        if not {"start", "end", "text"}.issubset(rows.fieldnames or []):
            raise ValueError("TSV requires start/end/text columns")
        for i, r in enumerate(rows):
            segments.append({"id": str(i), "start": float(r["start"])/scale,
                             "end": float(r["end"])/scale, "text": norm_text(r["text"])})
    elif suffix in {".srt", ".vtt"}:
        for block in re.split(r"\n\s*\n", text.replace("\r", "")):
            lines = block.strip().splitlines()
            ti = next((i for i, x in enumerate(lines) if "-->" in x), None)
            if ti is None:
                if suffix == ".vtt" and lines and lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
                    continue
                if lines:
                    raise ValueError("Caption block without timestamp")
                continue
            match = re.fullmatch(r"\s*(\S+)\s+-->\s+(\S+)(?:\s+.*)?", lines[ti])
            if not match:
                raise ValueError("Invalid caption timestamp line")
            content = " ".join(lines[ti+1:])
            # Remove recognized subtitle markup, not arbitrary angle-bracketed speech.
            content = re.sub(r"</?(?:i|b|u|c(?:\.[\w.-]+)?|v(?: [^>]+)?)>", "", content)
            segments.append({"id": str(len(segments)), "start": seconds(match[1]),
                             "end": seconds(match[2]), "text": norm_text(html.unescape(content))})
    elif suffix == ".txt":
        segments = [{"id": "0", "start": None, "end": None, "text": norm_text(text)}]
    else:
        raise ValueError(f"Unsupported format {suffix}")
    segments = [s for s in segments if s["text"]]
    if not segments:
        raise ValueError("No nonempty speech text")
    timed = []
    for s in segments:
        if (s["start"] is None) != (s["end"] is None):
            raise ValueError("Incomplete timestamp pair")
        if s["start"] is not None:
            s["start"], s["end"] = float(s["start"]), float(s["end"])
            if not all(math.isfinite(s[k]) for k in ("start", "end")) or not 0 <= s["start"] <= s["end"]:
                raise ValueError("Invalid segment time range")
            timed.append(s)
    if timed and len(timed) != len(segments):
        raise ValueError("Mixed timed and untimed segments")
    if any(a["start"] > b["start"] for a, b in zip(timed, timed[1:])):
        raise ValueError("Out-of-order segment times; inspect source instead of silently sorting")
    if len({s["id"] for s in segments}) != len(segments):
        raise ValueError("Duplicate segment IDs")
    full = ""
    for s in segments:
        if full:
            full += " "
        s["char_start"] = len(full)
        full += s["text"]
        s["char_end"] = len(full)
    return full, segments


def sentence_spans(text):
    """Punctuation-based segmentation with explicit abbreviation guards.

    All segment text is joined before segmentation. A caption break alone is
    not a sentence boundary. No temporal or fuzzy anchor reconstruction.
    """
    start = 0
    for m in END.finditer(text):
        punct = text[m.start():m.end()].rstrip('"\'”’)]')
        left = text[:m.start()+1]
        token = left.split()[-1].lower() if left.split() else ""
        if punct == "." and (token in ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.){2,}", token)
                             or re.fullmatch(r"[A-Z]\.", left.split()[-1])):
            continue
        end = m.end()
        if text[start:end].strip():
            a = start + len(text[start:end]) - len(text[start:end].lstrip())
            yield a, end
        start = end
    if text[start:].strip():
        yield start + len(text[start:]) - len(text[start:].lstrip()), len(text)


def index_sentences(vid, full, segments):
    transcript_hash = digest(full)
    rows = []
    for ix, (a, b) in enumerate(sentence_spans(full)):
        segs = [s for s in segments if s["char_start"] < b and s["char_end"] > a]
        times = [s for s in segs if s["start"] is not None]
        rows.append({"video_id": vid, "transcript_text_sha256": transcript_hash,
                     "sentence_id": digest([vid, transcript_hash, SEGMENTATION_VERSION, a, b]),
                     "sentence_index": ix, "char_start": a, "char_end": b,
                     "sentence": full[a:b], "segment_ids": [s["id"] for s in segs],
                     "start": min(s["start"] for s in times) if times else None,
                     "end": max(s["end"] for s in times) if times else None,
                     "timestamp_precision": "segment_bounds" if times else "unavailable"})
    if " ".join(s["sentence"] for s in rows) != full:
        raise ValueError("Sentence index does not preserve the complete normalized transcript")
    return rows
