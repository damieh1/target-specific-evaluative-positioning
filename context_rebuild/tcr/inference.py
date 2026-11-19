import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from .common import append_json, digest, ensure_unique, file_hash, now, read_csv, read_jsonl, write_csv, write_json
from .dataset import input_hash, verify
from .protocol import messages, protocol_hash, response_format, strict_json, validate_result

MODELS = {
    "qwen": "Qwen/Qwen3-14B",
    "gemma": "google/gemma-3-12b-it",
    "mistral": "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8",
    "slow": "RedHatAI/Magistral-Small-2506-FP8",
}


def http_json(base, route, payload=None, timeout=180):
    req = urllib.request.Request(base.rstrip("/")+route,
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def get_rows(run, slot):
    run = Path(run)
    verify(run)
    base = list(read_csv(run/"dataset/contexts.csv"))
    if slot == "slow":
        path = run/"routing/slow_inputs.csv"
        seal = json.loads((run/"routing/manifest.json").read_text())
        if file_hash(path) != seal["slow_inputs_sha256"]:
            raise ValueError("Slow input changed since routing")
        rows = list(read_csv(path))
    else:
        path = run/"parsing/parsed_contexts.csv"
        parsed = list(read_csv(path))
        ensure_unique(parsed, "occurrence_id")
        if {r["occurrence_id"] for r in parsed} != {r["occurrence_id"] for r in base}:
            raise ValueError("Parser row set differs from frozen dataset")
        original = {r["occurrence_id"]: r for r in base}
        for r in parsed:
            if any(r.get(k) != v for k, v in original[r["occurrence_id"]].items()):
                raise ValueError("Canonical context changed during parser enrichment")
        # Preserve the full measurement denominator; do not silently drop parser failures.
        if any(r["parse_status"] != "ok" for r in parsed):
            raise ValueError("Unfinished parsing/alignment: inspect parsing/parse_review.csv and resume parse")
        rows = parsed
    ensure_unique(rows, "occurrence_id")
    for r in rows:
        if r["input_sha256"] != input_hash(r):
            raise ValueError("Context input hash mismatch")
    return rows


def infer(run, slot, deployment_id, base_url="http://127.0.0.1:8000", max_model_len=8192,
          max_output_tokens=None, limit=None, preflight_only=False, reuse_results=(),
          schema_mode="response_format", transport=http_json):
    run = Path(run)
    stage = "slow" if slot == "slow" else "fast"
    model = MODELS[slot]
    budget = max_output_tokens or (6000 if stage == "slow" else 400)
    if budget < 1 or max_model_len <= budget:
        raise ValueError("Output budget must fit within a positive total context length")
    rows = get_rows(run, slot)
    out = run/"inference"
    out.mkdir(parents=True, exist_ok=True)
    log = out/(slot+".jsonl")
    contract_path = out/(slot+"_contract.json")
    params = {"temperature": 0.7 if stage == "slow" else 0.0, "top_p": 0.95 if stage == "slow" else 1.0,
              "max_tokens": budget, "seed": 20260831}
    template = {"chat_template_kwargs": {"enable_thinking": False}} if slot == "qwen" else {}
    contract = {"model": model, "slot": slot, "stage": stage, "deployment_id": deployment_id,
                "protocol_sha256": protocol_hash(), "parameters": params, "template": template,
                "max_model_len": max_model_len, "schema_mode": schema_mode}
    contract_id = digest(contract)
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Inference configuration changed. Preserve prior results and use a new run.")
    write_json(contract_path, contract)
    served = transport(base_url, "/v1/models")
    if model not in {r["id"] for r in served.get("data", [])}:
        raise ValueError(f"Server does not expose expected model {model}")
    write_json(out/(slot+"_server.json"), {"observed_at": now(), "models": served})
    all_saved = read_jsonl(log)
    for path in reuse_results:
        all_saved.extend(read_jsonl(path))
    successes = {}
    for rec in all_saved:
        if rec.get("status") == "ok" and rec.get("contract_sha256") == contract_id:
            if rec.get("slot") != slot:
                raise ValueError("Result slot does not match contract")
            old = successes.get(rec["request_sha256"])
            if old and old["result"] != rec["result"]:
                raise ValueError("Conflicting successful cached results for the same request")
            successes[rec["request_sha256"]] = rec
    own = {r["request_sha256"] for r in read_jsonl(log) if r.get("status") == "ok"}
    statuses, attempts, consecutive_failures = [], 0, 0
    token_cache = {}
    for row in rows:
        msg = messages(row, stage)
        request_key = digest([contract_id, row["input_sha256"], msg])
        recbase = {"occurrence_id": row["occurrence_id"], "input_sha256": row["input_sha256"],
                   "request_sha256": request_key, "contract_sha256": contract_id, "slot": slot,
                   "model": model, "protocol_sha256": protocol_hash(), "timestamp_utc": now()}
        if request_key in successes:
            validate_result(successes[request_key]["result"], row, stage)
            if request_key not in own and not preflight_only:
                append_json(log, {**successes[request_key], "reused_at": now()})
                own.add(request_key)
            statuses.append({**recbase, "status": "cached"})
            continue
        if limit is not None and attempts >= limit and not preflight_only:
            statuses.append({**recbase, "status": "not_attempted_run_limit"})
            continue
        if request_key not in token_cache:
            # Use the served tokenizer/chat template, not a character estimate.
            tok = transport(base_url, "/tokenize", {"model": model, "messages": msg,
                            "add_generation_prompt": True, **template})
            if type(tok.get("count")) is not int or tok["count"] <= 0:
                raise ValueError("/tokenize must return a positive integer count; no estimate fallback")
            token_cache[request_key] = tok
        tok = token_cache[request_key]
        declared = tok.get("max_model_len")
        actual_limit = min(max_model_len, int(declared)) if declared else max_model_len
        if tok["count"] + budget > actual_limit:
            status = {**recbase, "status": "overlength", "prompt_tokens": tok["count"], "model_limit": actual_limit}
            statuses.append(status)
            if not preflight_only:
                append_json(log, status)
            continue
        if preflight_only:
            statuses.append({**recbase, "status": "fits", "prompt_tokens": tok["count"]})
            continue
        payload = {"model": model, "messages": msg, **params, **template}
        if schema_mode == "response_format":
            payload["response_format"] = response_format(stage)
        else:
            payload["structured_outputs"] = {"json": response_format(stage)["json_schema"]["schema"]}
        success = False
        for retry in range(3):
            response = None
            attempts += 1
            if limit is not None and attempts > limit:
                attempts -= 1
                break
            rec = {**recbase, "timestamp_utc": now(), "prompt_tokens_preflight": tok["count"], "attempt_in_case": retry+1}
            try:
                response = transport(base_url, "/v1/chat/completions", payload, timeout=900)
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError(f"Incomplete/unexpected finish_reason={choice.get('finish_reason')}")
                obj = validate_result(strict_json(choice["message"].get("content")), row, stage)
                used = response.get("usage", {}).get("prompt_tokens")
                if used is not None and used != tok["count"]:
                    raise ValueError("Completion/tokenize prompt counts differ; inspect chat template configuration")
                rec.update(status="ok", result=obj, response=response)
                success = True
            except urllib.error.HTTPError as e:
                rec.update(status="error", error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}", response=response)
                append_json(log, rec)
                if e.code in {400, 401, 403, 404, 422}:
                    raise RuntimeError("Server rejected the request configuration; see inference log. No automatic protocol changes.") from e
                if retry < 2:
                    time.sleep(2*(retry+1))
                continue
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}", response=response)
            append_json(log, rec)
            if success:
                break
            if retry < 2:
                time.sleep(2*(retry+1))
        statuses.append({**recbase, "status": "ok" if success else "error"})
        consecutive_failures = 0 if success else consecutive_failures+1
        if attempts % 25 == 0 or not success:
            print(f"{slot}: {attempts} calls; occurrence {row['occurrence_id'][:10]} {'ok' if success else 'error'}", flush=True)
        if consecutive_failures >= 3:
            break
    report = {"slot": slot, "input_rows": len(rows), "rows_checked": len(statuses),
              "statuses": dict(Counter(r["status"] for r in statuses)), "actual_inference_calls": attempts,
              "all_rows_checked": len(statuses) == len(rows), "preflight_only": preflight_only}
    name = "preflight" if preflight_only else "run"
    write_csv(out/f"{slot}_{name}_status.csv", statuses)
    write_json(out/f"{slot}_{name}_summary.json", report)
    return report


def successful_results(run, slot):
    """Verify cache content against its contract; context checks happen at join time."""
    run = Path(run)
    path = run/"inference"/(slot+"_contract.json")
    if not path.exists():
        return {}
    contract = json.loads(path.read_text())
    if contract["slot"] != slot or contract["model"] != MODELS[slot] or contract["protocol_sha256"] != protocol_hash():
        raise ValueError("Incompatible saved inference contract")
    contract_id = digest(contract)
    results = {}
    for rec in read_jsonl(run/"inference"/(slot+".jsonl")):
        if rec.get("status") != "ok":
            continue
        if rec["contract_sha256"] != contract_id or rec["slot"] != slot:
            raise ValueError("Mixed inference contracts in result log")
        oid = rec["occurrence_id"]
        if oid in results and results[oid]["result"] != rec["result"]:
            raise ValueError("Conflicting successful predictions")
        results[oid] = rec
    return results


def verify_saved(rec, row, run, slot):
    contract = json.loads((Path(run)/"inference"/(slot+"_contract.json")).read_text())
    stage = "slow" if slot == "slow" else "fast"
    expected = digest([digest(contract), row["input_sha256"], messages(row, stage)])
    if rec["input_sha256"] != row["input_sha256"] or rec["request_sha256"] != expected:
        raise ValueError("Saved result belongs to different evidence/candidates")
    return validate_result(rec["result"], row, stage)
