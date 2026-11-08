#!/usr/bin/env python3
"""Durable bookkeeping for blinded source review; never calls a model."""
import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
from review_reporting import build, write_reports, source_decisions, history_order

ROOT = Path(__file__).resolve().parent
KINDS = {"primary": ("SOURCE_REVIEW.jsonl", 40), "independent": ("INDEPENDENT_CHALLENGE.jsonl", 40),
         "adjudication": ("ADJUDICATION.jsonl", 10), "history": ("CANDIDATE_GATE_A.jsonl", 12)}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def now():
    return dt.datetime.now(dt.timezone.utc)

def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def update_budget_ledger():
    path=ROOT/'BUDGET_LEDGER.json'
    budget=read(path)
    names={'primary':'primary_root_reviews','independent':'independent_checks',
           'adjudication':'adjudications','history':'history_deep_reviews'}
    for k,name in names.items(): budget[name]['used']=len(records(ROOT/KINDS[k][0]))
    path.write_text(json.dumps(budget,indent=2)+'\n')

def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="command", required=True)
    s.add_parser("start")
    a = s.add_parser("append"); a.add_argument("kind", choices=KINDS); a.add_argument("input")
    s.add_parser("status"); s.add_parser("finalize")
    args = p.parse_args()
    state_path = ROOT / "REVIEW_CLOCK.json"
    manifest = ROOT / "SAMPLE_MANIFEST.json"
    mapping = ROOT / "BLIND_MAPPING_PRIVATE.json"
    if not manifest.exists() or not mapping.exists():
        raise SystemExit("frozen sample and private mapping are required")
    if args.command == "start":
        if state_path.exists(): raise SystemExit("clock already started")
        # Validate the frozen allocation before starting the irreversible clock.
        # This keeps a malformed or partial freeze from silently changing the
        # denominator used at finalization.
        frozen = read(manifest).get("allocations", [])
        private = read(mapping).get("mapping", [])
        frozen_ids = [x.get("sample_id") for x in frozen]
        private_ids = [x.get("sample_id") for x in private]
        if len(frozen) > 40 or len(set(frozen_ids)) != len(frozen):
            raise SystemExit("frozen allocation must contain at most 40 unique samples")
        if len(private) != len(frozen) or len(set(private_ids)) != len(private) or set(frozen_ids) != set(private_ids):
            raise SystemExit("private mapping must bijectively cover the frozen allocation")
        sampler_counts = collections.Counter(x.get("sampler") for x in private)
        if set(sampler_counts)-{"SCORE","RANDOM_FROM_REMAINDER"} or any(v>20 for v in sampler_counts.values()):
            raise SystemExit("frozen allocation exceeds sampler limits")
        strata = collections.Counter((x.get("stratum"),x.get("sampler")) for x in private)
        if len({x.get('stratum') for x in private})>10 or any(v>2 for v in strata.values()):
            raise SystemExit("frozen allocation exceeds stratum limits")
        started = now()
        clock = {"status": "RUNNING", "started_utc": started.isoformat(),
                 "deadline_utc": (started + dt.timedelta(hours=8)).isoformat(),
                 "sample_sha256": digest(manifest), "mapping_sha256": digest(mapping)}
        with state_path.open("x") as f: json.dump(clock, f, indent=2); f.write("\n")
        print(json.dumps(clock)); return
    clock = read(state_path)
    if digest(manifest) != clock["sample_sha256"] or digest(mapping) != clock["mapping_sha256"]:
        raise SystemExit("frozen sampling artifacts changed")
    deadline = dt.datetime.fromisoformat(clock["deadline_utc"])
    if args.command == "append":
        if clock["status"] != "RUNNING" or now() >= deadline: raise SystemExit("review deadline reached or run closed")
        item = read(Path(args.input)); sample = item.get("sample_id")
        allocations = read(manifest)["allocations"]
        ids = {x["sample_id"] for x in allocations}
        if sample not in ids: raise SystemExit("unknown sample")
        mapping_rows = [x for x in read(mapping)["mapping"] if x.get("sample_id") == sample]
        if len(mapping_rows) != 1: raise SystemExit("sample is missing or duplicated in private mapping")
        allowed = {"PASS", "HOLD", "UNKNOWN"} if args.kind == "history" else {"PASS", "FAIL", "UNKNOWN"}
        if item.get("status") not in allowed: raise SystemExit("invalid status")
        if not item.get("reviewer_id") or not item.get("evidence"): raise SystemExit("reviewer_id and evidence are required")
        if len(item.get('certificates',[]))>2: raise SystemExit('at most two certificates per source')
        if args.kind!='history':
            packet=next(x for x in allocations if x['sample_id']==sample)
            if item.get('packet_sha256')!=packet['anonymous_packet_sha256'] or digest(Path(packet['anonymous_packet']))!=packet['anonymous_packet_sha256']:
                raise SystemExit('review packet hash mismatch or missing binding')
        filename, budget = KINDS[args.kind]; path = ROOT / filename; prior = records(path)
        if len(prior) >= budget or any(x["sample_id"] == sample for x in prior): raise SystemExit("budget exhausted or duplicate review")
        primary = {x["sample_id"]: x for x in records(ROOT / KINDS["primary"][0])}
        independent = {x["sample_id"]: x for x in records(ROOT / KINDS["independent"][0])}
        if args.kind == "independent":
            if sample not in primary or primary[sample]["reviewer_id"] == item["reviewer_id"]:
                raise SystemExit("independent reviewer and prior primary required")
        if args.kind == "adjudication":
            if sample not in primary or sample not in independent or primary[sample]["status"] == independent[sample]["status"]:
                raise SystemExit("adjudication requires two conflicting reviews")
            if item["reviewer_id"] in {primary[sample]["reviewer_id"], independent[sample]["reviewer_id"]}:
                raise SystemExit("adjudicator must be independent")
        if args.kind == "history":
            adjudicated = {x["sample_id"]: x for x in records(ROOT / KINDS["adjudication"][0])}
            # Freeze qualification priority only after the equal source-review
            # chain is settled. Six per group first; unused capacity transfers.
            private=read(mapping)['mapping']
            decisions=source_decisions(private,list(primary.values()),list(independent.values()),list(adjudicated.values()))
            if any(not x['double_reviewed'] for x in decisions.values()):
                raise SystemExit("settle all available source double reviews before history selection")
            order=history_order(private,decisions,'overnight-source-diagnostic-20261108-v1')
            order_path=ROOT/'HISTORY_SELECTION.json'
            if not order_path.exists():
                with order_path.open('x') as f: json.dump({'sample_ids':order,'budget':12,'initial_per_sampler':6,'transfer_unused':True},f,indent=2); f.write('\n')
            if read(order_path)['sample_ids']!=order: raise SystemExit('source decisions changed after history freeze')
            if sample not in order: raise SystemExit('sample outside frozen history allocation')
            if item.get('elapsed_seconds',601)>600: raise SystemExit('history review exceeds ten-minute budget')
            if sample not in primary or sample not in independent:
                raise SystemExit("history requires completed double review")
            status = primary[sample]["status"] if primary[sample]["status"] == independent[sample]["status"] else adjudicated.get(sample, {}).get("status")
            if status != "PASS": raise SystemExit("history is limited to source-pass candidates")
        item["recorded_utc"] = now().isoformat()
        with path.open("a") as f: f.write(json.dumps(item, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
        update_budget_ledger()
        print(json.dumps({"appended": args.kind, "sample_id": sample})); return
    if args.command == "status":
        print(json.dumps({"clock": clock, "deadline_reached": now() >= deadline,
                          "counts": {k: len(records(ROOT / v[0])) for k, v in KINDS.items()}})); return
    if clock["status"] != "RUNNING": raise SystemExit("run already closed")
    if now() < deadline: raise SystemExit("cannot finalize before deadline; do not stop early")
    # A missing or conflicting review is UNKNOWN; this never manufactures a pass.
    ledgers={k:records(ROOT/v[0]) for k,v in KINDS.items()}
    results = build(read(mapping)['mapping'],ledgers)
    clock["status"] = "CLOSED"; clock["closed_utc"] = now().isoformat()
    write_reports(ROOT,clock,results)
    update_budget_ledger()
    with (ROOT/'REVIEW_OUTPUT_BINDINGS.json').open('x') as f:
        json.dump({'files':[{'path':str(ROOT/v[0]),'sha256':digest(ROOT/v[0])} for v in KINDS.values() if (ROOT/v[0]).exists()]},f,indent=2); f.write('\n')
    state_path.write_text(json.dumps(clock, indent=2) + "\n")
    print(json.dumps(results))

if __name__ == "__main__":
    with (ROOT / ".review_controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
