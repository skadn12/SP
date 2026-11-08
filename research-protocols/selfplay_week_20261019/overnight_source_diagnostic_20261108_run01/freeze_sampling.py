#!/usr/bin/env python3
"""Freeze the 40-root source diagnostic sample and blind packets.

This is preparation only.  It performs no authoring, Solver call, model call,
or human review.  Existing artifacts are never overwritten.  The score is a
read-only copy of the frozen metadata scan formula; it is used only to define
the SCORE arm and is hidden from the blind packets.
"""
from __future__ import annotations
import hashlib, json, math, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = Path("/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl")
REGISTRY = ROOT / "REVIEWED_ROOT_REGISTRY.json"
SEED = "overnight-source-diagnostic-20261108-v1"
RELATIONS = ["Classmates", "Neighbors", "Co-workers", "Mentee and Mentor", "Parent and Child", "Patient and Doctor", "Husband and Wife", "Student and Teacher", "Athlete and Coach", "Employee and Boss"]
STOP = set("the a an and or but if then to of in on for with from by is are was were be been being i you he she it they we them this that these those have has had do did does can could should would will just really very about what when where how why there here their our your my his her its as at into out up down so than too also not no yes all any some more one two three first second third fourth fifth".split())
MARKERS = re.compile(r"\b(remember|earlier|previous|last time|few weeks|a while ago|mentioned|again|before|since then|as you said|we talked|we discussed|reminds me|already|still|continue|instead|if|because|therefore|should|could|try|suggest|advice|plan|need to|have to|must|goal|decided|agreed)\b", re.I)
TEMP = re.compile(r"\b(before|after|later|then|first|next|finally|already|still|again|last|previous|earlier|week|month|year|tomorrow|yesterday)\b", re.I)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def toks(v):
    return set(re.findall(r"[a-zA-Z][a-zA-Z']+", " ".join(map(str, v)).lower())) - STOP

def rootnum(v):
    m = re.search(r"(?:episode[-_])?(\d+)$", str(v))
    return m.group(1) if m else str(v)

def score_record(d):
    ss = [d.get(k, []) for k in ("first_session_dialogue", "second_session_dialogue", "third_session_dialogue", "fourth_session_dialogue", "fifth_session_dialogue")]
    ts = [toks(x) for x in ss]
    refs = sum(len(MARKERS.findall(" ".join(map(str, ss[j])))) for j in range(1, 5))
    temp = sum(len(TEMP.findall(" ".join(map(str, ss[j])))) for j in range(1, 5))
    overlaps = [len(ts[j] & ts[j - 1]) for j in range(1, 5)]
    cross = sum(overlaps); distinct = sum(1 for x in overlaps if x >= 4)
    nums = sum(bool(re.search(r"\b\d+(?:\.\d+)?\b", " ".join(map(str, x)))) for x in ss)
    strategy = sum(len(re.findall(r"\b(should|could|try|suggest|advice|recommend|plan|need to|must|goal|decided|agreed)\b", " ".join(map(str, x)), re.I)) for x in ss)
    score = 2 * distinct + .15 * cross + .25 * refs + .15 * temp + .1 * strategy + .2 * nums
    return score, {"cross_adjacent_token_overlap": overlaps, "reference_markers_later": refs, "temporal_markers_later": temp, "strategy_markers": strategy, "numeric_sessions": nums}

def anon_record(d):
    # Keep complete dialogue evidence, remove identity and sampler metadata.
    return {k: v for k, v in d.items() if k not in {"dataID", "relationship", "summary"}}

def main():
    for name in ("SAMPLING_SPEC.json", "SAMPLE_MANIFEST.json", "BLIND_MAPPING_PRIVATE.json"):
        if (ROOT / name).exists(): raise SystemExit(f"frozen artifact exists: {name}")
    if not SRC.exists() or not REGISTRY.exists(): raise SystemExit("run prepare_inputs.py first")
    registry = json.loads(REGISTRY.read_text())
    excluded = set().union(*(set(v) for v in registry["excluded_by_reason"].values()))
    by_rel = defaultdict(list); source_count = 0
    with SRC.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            d = json.loads(line); source_count += 1
            rid = rootnum(d.get("dataID", "")); rel = d.get("relationship")
            if rid in excluded or rel not in RELATIONS: continue
            sc, feats = score_record(d)
            by_rel[rel].append({"root_id": rid, "data_id": d.get("dataID"), "source_line_index": line_no, "record_sha256": hashlib.sha256(line.encode()).hexdigest(), "relationship": rel, "score": round(sc, 8), "features": feats, "record": d})
    if set(by_rel) != set(RELATIONS): raise SystemExit("stratum missing: " + repr(sorted(set(RELATIONS) - set(by_rel))))
    allocations = []; mapping = []; anon_dir = ROOT / "blind_packets"; anon_dir.mkdir()
    for rel in RELATIONS:
        items = sorted(by_rel[rel], key=lambda x: (-x["score"], x["root_id"], x["record_sha256"]))
        top = items[:2]; top_ids = {x["root_id"] for x in top}
        rem = [x for x in items if x["root_id"] not in top_ids]
        rem.sort(key=lambda x: hashlib.sha256(f"{SEED}:{x['root_id']}".encode()).hexdigest())
        randoms = rem[:2]
        for arm, x in [("SCORE", z) for z in top] + [("RANDOM_FROM_REMAINDER", z) for z in randoms]:
            sid = f"S{RELATIONS.index(rel)+1:02d}_{arm[:1]}_{len(allocations)+1:03d}"
            p = anon_dir / f"{sid}.json"
            p.write_text(json.dumps(anon_record(x["record"]), ensure_ascii=False, indent=2) + "\n")
            a = {"sample_id": sid, "sampler": arm, "stratum": rel, "anonymous_packet": str(p), "anonymous_packet_sha256": digest(p), "source_line_index": x["source_line_index"], "record_sha256": x["record_sha256"]}
            allocations.append(a)
            mapping.append({**a, "root_id": x["root_id"], "data_id": x["data_id"], "score": x["score"], "features": x["features"], "source_path": str(SRC)})
    spec = {"schema_version":"overnight_sampling_spec_v1","status":"FROZEN_PRE_REVIEW","source_path":str(SRC),"source_sha256":digest(SRC),"seed":SEED,"strata":RELATIONS,"per_stratum":{"SCORE":2,"RANDOM_FROM_REMAINDER":2},"total":len(allocations),"score_formula":"frozen operation_oriented_scan.py formula; score hidden from blind reviewers","exclusion_registry":str(REGISTRY),"exclusion_registry_sha256":digest(REGISTRY),"replace_failed_samples":False,"stop_early_when_eight_found":False}
    (ROOT / "SAMPLING_SPEC.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "SAMPLE_MANIFEST.json").write_text(json.dumps({"schema_version":"overnight_sample_manifest_v1","status":"FROZEN_PRE_REVIEW","source_sha256":digest(SRC),"sample_count":len(allocations),"allocations":allocations}, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "BLIND_MAPPING_PRIVATE.json").write_text(json.dumps({"schema_version":"overnight_blind_mapping_private_v1","status":"CONTROLLER_ONLY","source_sha256":digest(SRC),"mapping":mapping}, ensure_ascii=False, indent=2) + "\n")
    # Empty durable ledgers are part of preparation; review starts only when a
    # controller explicitly advances the state after the training health gate.
    (ROOT / "SOURCE_REVIEW.jsonl").touch(); (ROOT / "INDEPENDENT_CHALLENGE.jsonl").touch(); (ROOT / "ADJUDICATION.jsonl").touch()
    (ROOT / "BUDGET_LEDGER.json").write_text(json.dumps({"schema_version":"overnight_budget_ledger_v1","status":"FROZEN_PRE_REVIEW","primary_root_reviews":{"budget":40,"used":0},"independent_checks":{"budget":40,"used":0},"adjudications":{"budget":10,"used":0},"history_deep_reviews":{"budget":12,"used":0},"solver_requests":0,"training_runs":0,"wall_clock_limit_hours":8},indent=2)+"\n")
    (ROOT / "CONTROLLER_STATE.json").write_text(json.dumps({"schema_version":"overnight_controller_state_v1","status":"PREPARED_NOT_STARTED","stage":"await_training_health_gate","sample_manifest":str(ROOT/"SAMPLE_MANIFEST.json"),"resume_rule":"never rerun frozen records; append one durable record per completed root; stop on hash/identity/leakage mismatch"},indent=2)+"\n")
    print(json.dumps({"status":"FROZEN_PRE_REVIEW","source_records_scanned":source_count,"sample_count":len(allocations),"by_sampler":{"SCORE":sum(x['sampler']=='SCORE' for x in allocations),"RANDOM_FROM_REMAINDER":sum(x['sampler']=='RANDOM_FROM_REMAINDER' for x in allocations)}}, ensure_ascii=False))

if __name__ == "__main__": main()
