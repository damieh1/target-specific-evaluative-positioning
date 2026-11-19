"""Resolve source files and eligibility without treating missing records as exclusions."""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from .common import boolean, digest, file_hash, norm_text, read_csv, row_id, utc_date, video_id, write_csv, write_json
from .transcripts import FORMATS, read_transcript


def configure(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    def absolute(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else path.parent/p)
    for name in ("lexicon", "eligibility_overrides", "transcript_choices"):
        if cfg.get(name):
            cfg[name] = absolute(cfg[name])
    for source in cfg["transcript_roots"]:
        source["path"] = absolute(source["path"])
    for key in ("audit_sources", "english_lists", "metadata_sources"):
        for source in cfg.get(key, []):
            source["glob"] = absolute(source["glob"])
    if cfg.get("tsv_unit") not in {"milliseconds", "seconds"}:
        raise ValueError("Set tsv_unit to milliseconds (Whisper default) or seconds")
    if cfg.get("context_radius") != 4:
        raise ValueError("This protocol fixes context_radius=4; changing it requires a new protocol version")
    if not isinstance(cfg.get("accept_english_lists"), bool):
        raise ValueError("Explicitly set accept_english_lists")
    for key in ("start_date", "end_date"):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cfg[key]):
            raise ValueError(f"Use ISO YYYY-MM-DD for {key}")
    if cfg["start_date"] > cfg["end_date"]:
        raise ValueError("Reversed date interval")
    return cfg


def inventory(cfg, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    issues, provenance, evidence, candidates = [], [], defaultdict(list), defaultdict(list)
    outlets = set(s["outlet"] for s in cfg["transcript_roots"])
    def issue(kind, source, detail, vid=""):
        issues.append({"kind": kind, "source": str(source), "video_id": vid, "detail": detail})
    def files(spec):
        paths = sorted({str(Path(p).resolve()) for p in glob.glob(spec["glob"], recursive=True) if Path(p).is_file()})
        if not paths:
            issue("no_source_files", spec["glob"], "Pattern matched no files")
            if spec.get("required", False):
                raise FileNotFoundError(spec["glob"])
        return paths
    def register(path, kind):
        h = file_hash(path)
        provenance.append({"path": str(path), "kind": kind, "sha256": h})
        return h

    for spec in cfg.get("audit_sources", []):
        for path in files(spec):
            h = register(path, "language_audit")
            for n, r in enumerate(read_csv(path), 2):
                vid = ""
                try:
                    vid = row_id(r)
                    flag = boolean(r[spec.get("decision_column", "keep_english_only")])
                    evidence[vid].append({"kind": "audit", "decision": flag, "source": path,
                                          "row": n, "sha256": h, "record": r})
                except (ValueError, KeyError) as e:
                    issue("invalid_audit_row", path, f"Row {n}: {e}", vid)
                    if vid:
                        evidence[vid].append({"kind": "invalid_audit", "decision": None, "source": path,
                                              "row": n, "sha256": h, "record": r})
    for spec in cfg.get("english_lists", []):
        for path in files(spec):
            h = register(path, "english_list")
            for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    vid = video_id(line)
                    evidence[vid].append({"kind": "list", "decision": True, "source": path,
                                          "row": n, "sha256": h, "record": line})
                except ValueError as e:
                    issue("invalid_english_list_line", path, f"Line {n}: {e}")

    overrides = {}
    if cfg.get("eligibility_overrides"):
        path = cfg["eligibility_overrides"]
        register(path, "eligibility_overrides")
        for r in read_csv(path):
            vid = row_id(r)
            if vid in overrides or not r.get("reason", "").strip() or not r.get("evidence", "").strip():
                raise ValueError("Eligibility overrides need unique IDs, a reason, and evidence")
            overrides[vid] = {**r, "decision": boolean(r["keep_english_only"])}

    # All readable transcript alternatives are retained in the inventory.
    seen = {}
    for spec in cfg["transcript_roots"]:
        root = Path(spec["path"])
        if not root.is_dir():
            if spec.get("required", True):
                raise FileNotFoundError(root)
            issue("missing_transcript_root", root, "Optional outlet root absent")
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in FORMATS:
                continue
            if "_video_" not in p.stem and not re.fullmatch(r"\d{18,20}", p.stem):
                continue
            p = p.resolve()
            if str(p) in seen:
                if seen[str(p)] != spec["outlet"]:
                    raise ValueError(f"Same transcript under two outlet roots: {p}")
                continue
            seen[str(p)] = spec["outlet"]
            try:
                vid = video_id(p.stem)
            except ValueError as e:
                issue("invalid_transcript_id", p, str(e))
                continue
            entry = {"video_id": vid, "outlet": spec["outlet"], "path": str(p),
                     "format": p.suffix.lower(), "file_sha256": register(p, "transcript"),
                     "status": "ok", "error": ""}
            try:
                m = re.search(r"@([^/\\]+)_video_\d+", p.stem)
                if m and m[1].casefold() != spec["outlet"].casefold():
                    raise ValueError("Filename outlet disagrees with configured outlet")
                full, segs = read_transcript(p, cfg["tsv_unit"])
                entry.update(text_sha256=digest(full), n_segments=len(segs),
                             timed=segs[0]["start"] is not None, text_length=len(full))
            except Exception as e:
                entry.update(status="invalid", error=f"{type(e).__name__}: {e}")
                issue("invalid_transcript", p, entry["error"], vid)
            candidates[vid].append(entry)

    choices = {}
    if cfg.get("transcript_choices"):
        register(cfg["transcript_choices"], "transcript_choices")
        for r in read_csv(cfg["transcript_choices"]):
            vid = row_id(r)
            if vid in choices or not r.get("reason", "").strip():
                raise ValueError("Transcript choices need unique IDs and a reason")
            r["path"] = str(Path(r["path"]).expanduser().resolve())
            choices[vid] = r

    metadata = defaultdict(list)
    for priority, spec in enumerate(cfg["metadata_sources"]):
        for path in files(spec):
            h = register(path, "metadata")
            for n, r in enumerate(read_csv(path), 2):
                try:
                    vid = row_id(r)
                    outlet = r.get("username", "").strip().lstrip("@").lower()
                    if not outlet:
                        outlet = r.get("account_filter", "").strip().lstrip("@").lower()
                    if outlet not in outlets:
                        continue
                    posted = next((r[k] for k in ("create_time", "posted_at", "createTime") if r.get(k, "").strip()), "")
                    if not posted:
                        raise ValueError("No publication date")
                    posted = utc_date(posted)
                    metadata[vid].append({"video_id": vid, "outlet": outlet, "posted_at": posted,
                                          "video_description": norm_text(r.get("video_description", "")),
                                          "source": path, "source_sha256": h, "source_row": n,
                                          "priority": priority, "original_record": r})
                except (ValueError, OverflowError, OSError) as e:
                    issue("invalid_metadata_row", path, f"Row {n}: {e}")

    manifest, selected = [], []
    for vid in sorted(set(candidates) | set(evidence) | set(metadata) | set(overrides)):
        ev = evidence[vid]
        flags = {r["decision"] for r in ev if r["decision"] is not None}
        if vid in overrides:
            keep, lang = overrides[vid]["decision"], "documented_override"
        elif any(r["kind"] == "invalid_audit" for r in ev):
            keep, lang = False, "invalid_language_evidence"
        elif flags == {False, True}:
            keep, lang = False, "conflicting_language_evidence"
        elif flags == {False}:
            keep, lang = False, "excluded_by_audit"
        elif flags == {True} and any(r["kind"] == "audit" for r in ev):
            keep, lang = True, "passed_audit"
        elif flags == {True} and cfg["accept_english_lists"]:
            keep, lang = True, "passed_list_only"
        else:
            keep, lang = False, "eligibility_not_established"
        valid = [r for r in candidates[vid] if r["status"] == "ok"]
        chosen, source_status = None, "no_readable_transcript"
        if valid:
            if len({r["outlet"] for r in valid}) != 1:
                source_status = "conflicting_transcript_outlets"
            elif vid in choices:
                chosen = next((r for r in valid if r["path"] == choices[vid]["path"]), None)
                if not chosen:
                    raise ValueError(f"Chosen transcript not valid/discovered: {vid}")
                source_status = "documented_choice"
            elif len({r["text_sha256"] for r in valid}) > 1:
                source_status = "conflicting_transcript_texts"
            else:
                chosen = sorted(valid, key=lambda r: (FORMATS[r["format"]], r["path"]))[0]
                source_status = "format_priority_equivalent_text"
        records = metadata[vid]
        meta, meta_status = None, "missing_metadata"
        if records:
            highest = min(r["priority"] for r in records)
            top = [r for r in records if r["priority"] == highest]
            variants = {(r["outlet"], r["posted_at"], r["video_description"]) for r in top}
            if len(variants) != 1:
                meta_status = "conflicting_metadata_at_same_priority"
            else:
                meta = sorted(top, key=lambda r: (r["source"], r["source_row"]))[0]
                meta_status = "matched"
                if not cfg["start_date"] <= meta["posted_at"][:10] <= cfg["end_date"]:
                    meta_status = "outside_collection_period"
                if chosen and meta["outlet"] != chosen["outlet"]:
                    meta_status = "metadata_transcript_outlet_mismatch"
        ready = bool(keep and chosen and meta_status == "matched")
        row = {"video_id": vid, "outlet": (meta or chosen or {}).get("outlet", ""),
               "language_status": lang, "language_keep": keep, "transcript_status": source_status,
               "metadata_status": meta_status, "ready": ready, "transcript_path": (chosen or {}).get("path", ""),
               "n_transcript_files": len(candidates[vid]), "n_language_records": len(ev)}
        manifest.append(row)
        if ready:
            selected.append({**row, "transcript": chosen, "language_evidence": ev,
                             "eligibility_override": overrides.get(vid), "transcript_choice": choices.get(vid),
                             "metadata": meta})
    write_csv(out/"video_manifest.csv", manifest)
    write_csv(out/"transcript_inventory.csv", [r for rows in candidates.values() for r in rows],
              ["video_id", "outlet", "path", "format", "file_sha256", "status", "error", "text_sha256", "n_segments", "timed", "text_length"])
    write_csv(out/"source_issues.csv", issues, ["kind", "source", "video_id", "detail"])
    write_json(out/"source_files.json", provenance)
    write_json(out/"language_evidence.json", dict(evidence))
    write_json(out/"metadata_records.json", dict(metadata))
    write_json(out/"selected_sources.json", selected)
    write_json(out/"resolved_config.json", cfg)
    return selected, manifest, issues
