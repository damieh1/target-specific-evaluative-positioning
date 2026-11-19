"""Collapse once, route by complete Fast results, anonymize unique orientations."""
import json
from collections import Counter
from pathlib import Path
from .common import digest, file_hash, read_csv, write_csv, write_json
from .dataset import BASE_COLUMNS, verify
from .inference import successful_results, verify_saved
from .protocol import collapse

SLOTS = ("qwen", "gemma", "mistral")
SEED = "candidate_order_v1_20260914"


def route(run):
    run = Path(run)
    verify(run)
    rows = list(read_csv(run/"dataset/contexts.csv"))
    results = {s: successful_results(run, s) for s in SLOTS}
    routing, slow, unanimous = [], [], []
    for row in rows:
        oid = row["occurrence_id"]
        rec = {"occurrence_id": oid, "input_sha256": row["input_sha256"], "route": "pending_fast",
               "final_label": "", "raw_pattern": "", "collapsed_pattern": "", "mixed_recoded": False}
        if all(oid in results[s] for s in SLOTS):
            objs = [verify_saved(results[s][oid], row, run, s) for s in SLOTS]
            raw = [x["orientation"] for x in objs]
            rec.update({s+"_raw": raw[i] for i, s in enumerate(SLOTS)})
            if any(v is None for v in raw):
                rec["route"] = "unresolved_fast_review"
            else:
                labels = [collapse(v) for v in raw]
                rec.update({s+"_label": labels[i] for i, s in enumerate(SLOTS)})
                rec["mixed_recoded"] = "Mixed" in raw
                rec["raw_pattern"] = pattern(raw)
                rec["collapsed_pattern"] = pattern(labels)
                if len(set(labels)) == 1:
                    rec.update(route="unanimous", final_label=labels[0])
                    unanimous.append({**row, "final_label": labels[0], "measurement_stage": "fast_unanimous"})
                else:
                    rec["route"] = "slow"
                    # Ordering depends on occurrence ID, not model, class frequency, or votes.
                    candidates = sorted(set(labels), key=lambda label: digest([SEED, oid, label]))
                    slow.append({**row, **{"candidate_"+letter.lower(): candidates[i] if i < len(candidates) else ""
                                           for i, letter in enumerate("ABC")}})
        routing.append(rec)
    out = run/"routing"
    out.mkdir(parents=True, exist_ok=True)
    # Route regeneration is permitted only before Slow inference, or if identical.
    if (run/"inference/slow.jsonl").exists() and (out/"slow_inputs.csv").exists():
        existing = list(read_csv(out/"slow_inputs.csv"))
        if existing != slow:
            raise ValueError("Slow inputs would change after adjudication began. Use a new run.")
    write_csv(out/"routing.csv", routing)
    write_csv(out/"slow_inputs.csv", slow, BASE_COLUMNS+["candidate_a", "candidate_b", "candidate_c"])
    write_csv(out/"unanimous.csv", unanimous, BASE_COLUMNS+["final_label", "measurement_stage"])
    report = {"n_occurrences": len(rows), "routes": dict(Counter(r["route"] for r in routing)),
              "mixed_recoded_rows": sum(r["mixed_recoded"] for r in routing),
              "policy": "Mixed collapsed into Neutral / Descriptive before agreement and routing; raw labels preserved",
              "candidate_order_seed": SEED, "slow_inputs_sha256": file_hash(out/"slow_inputs.csv"),
              "routing_sha256": file_hash(out/"routing.csv")}
    write_json(out/"manifest.json", report)
    return report


def pattern(labels):
    return {1: "unanimous", 2: "two_to_one", 3: "all_different"}[len(set(labels))]


def finalize(run):
    run = Path(run)
    verify(run)
    # Regenerate from validated Fast results, not editable routing labels.
    route(run)
    routing = {r["occurrence_id"]: r for r in read_csv(run/"routing/routing.csv")}
    slow_inputs = {r["occurrence_id"]: r for r in read_csv(run/"routing/slow_inputs.csv")}
    results = successful_results(run, "slow")
    all_rows = []
    for row in read_csv(run/"dataset/contexts.csv"):
        rec = routing[row["occurrence_id"]]
        label, stage = "", rec["route"]
        if stage == "unanimous":
            label, stage = rec["final_label"], "fast_unanimous"
        elif stage == "slow":
            saved = results.get(row["occurrence_id"])
            if saved:
                obj = verify_saved(saved, slow_inputs[row["occurrence_id"]], run, "slow")
                label, stage = obj["orientation"], "slow_adjudicated"
            else:
                stage = "pending_slow"
        all_rows.append({**row, "final_label": label, "measurement_stage": stage,
                         "raw_fast_pattern": rec["raw_pattern"], "collapsed_fast_pattern": rec["collapsed_pattern"],
                         "mixed_recoded": rec["mixed_recoded"]})
    out = run/"final"
    out.mkdir(parents=True, exist_ok=True)
    complete = [r for r in all_rows if r["final_label"]]
    cols = BASE_COLUMNS+["final_label", "measurement_stage", "raw_fast_pattern", "collapsed_fast_pattern", "mixed_recoded"]
    write_csv(out/"all_occurrences.csv", all_rows, cols)
    write_csv(out/"completed_classifications.csv", complete, cols)
    write_csv(out/"pending_occurrences.csv", [r for r in all_rows if not r["final_label"]], cols)
    report = {"status": "COMPLETE" if len(complete) == len(all_rows) else "INCOMPLETE",
              "n_occurrences": len(all_rows), "completed": len(complete), "pending": len(all_rows)-len(complete),
              "labels": dict(Counter(r["final_label"] for r in complete)),
              "stages": dict(Counter(r["measurement_stage"] for r in all_rows)),
              "completed_sha256": file_hash(out/"completed_classifications.csv")}
    write_json(out/"summary.json", report)
    return report
