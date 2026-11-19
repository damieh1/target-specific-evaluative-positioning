import argparse
import json
from collections import Counter
from pathlib import Path
from tcr.common import lock, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory", help="Inventory transcripts, language evidence and metadata")
    inv.add_argument("--config", required=True)
    inv.add_argument("--run", required=True)
    for name in ("build", "verify", "route", "finalize"):
        s = commands.add_parser(name)
        s.add_argument("--run", required=True)
    s = commands.add_parser("parse", help="Enrich exact target rows using the existing SuPar checkpoint")
    s.add_argument("--run", required=True)
    s.add_argument("--model", default="biaffine-dep-en")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--limit", type=int, help="Maximum new sentences, for the first smoke test")
    s = commands.add_parser("infer")
    s.add_argument("--run", required=True)
    s.add_argument("--slot", choices=["qwen", "gemma", "mistral", "slow"], required=True)
    s.add_argument("--deployment-id", required=True, help="Your fixed identifier for this checkpoint/environment")
    s.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server root, without /v1")
    s.add_argument("--max-model-len", type=int, default=8192)
    s.add_argument("--max-output-tokens", type=int)
    s.add_argument("--limit", type=int, help="Maximum actual inference calls, including retries")
    s.add_argument("--preflight-only", action="store_true")
    s.add_argument("--reuse-results", action="append", default=[], help="Prior JSONL from this bundle, same slot/protocol")
    s.add_argument("--schema-mode", choices=["response_format", "structured_outputs"], default="response_format")
    args = p.parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        p.error("--limit must be positive")
    run = Path(args.run).expanduser().resolve()
    # Model slots can run independently, each with its own result log.
    # Do not mutate the dataset or routing while inference is running.
    name = args.slot if args.command == "infer" else "dataset"
    with lock(run/("."+name+".lock")):
        if args.command == "inventory":
            from tcr.sources import configure, inventory
            if (run/"inventory/selected_sources.json").exists():
                raise FileExistsError("Inventory already exists. Choose a fresh run directory for a new snapshot.")
            cfg = configure(args.config)
            selected, manifest, issues = inventory(cfg, run/"inventory")
            result = {"videos_in_manifest": len(manifest), "eligible_ready_videos": len(selected), "source_issues": len(issues)}
            result["ready_by_outlet"] = dict(Counter(r["outlet"] for r in selected))
            result["language_statuses"] = dict(Counter(r["language_status"] for r in manifest))
            result["transcript_statuses"] = dict(Counter(r["transcript_status"] for r in manifest))
            result["metadata_statuses"] = dict(Counter(r["metadata_status"] for r in manifest))
            write_json(run/"inventory/summary.json", result)
        elif args.command in {"build", "verify"}:
            from tcr import dataset
            result = getattr(dataset, args.command)(run)
        elif args.command == "parse":
            from tcr.dependency import parse_dataset
            result = parse_dataset(run, args.model, args.device, args.limit)
        elif args.command in {"route", "finalize"}:
            from tcr import routing
            result = getattr(routing, args.command)(run)
        else:
            from tcr.inference import infer
            result = infer(run, args.slot, args.deployment_id, args.base_url, args.max_model_len,
                           args.max_output_tokens, args.limit, args.preflight_only, args.reuse_results, args.schema_mode)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
