#!/usr/bin/env python3
"""Build an immutable F/G candidate manifest from canonical review resolution.

Inputs are treated as frozen: this script only reads them and writes a new
manifest directory. It aligns records by review_packet_id, verifies author
file hashes, and embeds each single candidate for downstream Solver gating.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from collections import Counter


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    canonical_path = run / "quality_source_review_20261108_v1/CANONICAL_REVIEW_RESOLUTION_20261108_v1.json"
    bindings_path = run / "private_review_bindings.json"
    canonical = read_json(canonical_path)
    bindings = read_json(bindings_path)
    resolved = canonical["resolved"]
    if len(resolved) != 140 or len(bindings) != len(resolved):
        raise SystemExit(f"unexpected counts resolved={len(resolved)} bindings={len(bindings)}")
    by_packet = {x["review_packet_id"]: x for x in bindings}
    if len(by_packet) != len(bindings):
        raise SystemExit("duplicate review_packet_id in bindings")
    records=[]; errors=[]
    for i, rr in enumerate(resolved):
        packet = rr["packet_id"]
        b = by_packet.get(packet)
        if b is None: errors.append((i, packet, "missing_binding")); continue
        draft = Path(b["author_draft_path"]).resolve()
        if not draft.exists(): errors.append((i, packet, "missing_draft")); continue
        raw = draft.read_bytes(); got_hash=sha256_bytes(raw)
        if got_hash != b["author_draft_sha256"]: errors.append((i, packet, f"draft_hash:{got_hash}")); continue
        doc = json.loads(raw.decode("utf-8"))
        cands = doc.get("candidates")
        if not isinstance(cands,list) or len(cands) != 1:
            errors.append((i,packet,f"candidate_count:{len(cands) if isinstance(cands,list) else 'invalid'}")); continue
        c = cands[0]
        if c.get("candidate_id") != b["candidate_id"]: errors.append((i,packet,"candidate_id_mismatch")); continue
        if c.get("root_id") != b["root_id"]: errors.append((i,packet,"root_id_mismatch")); continue
        if c.get("source_sha256") is None: errors.append((i,packet,"missing_source_sha256")); continue
        if rr.get("root_id") != b["root_id"]: errors.append((i,packet,"canonical_root_mismatch")); continue
        if rr.get("type") != c.get("question_type"): errors.append((i,packet,"canonical_type_mismatch")); continue
        status=rr.get("effective_status")
        rec={
            "index": i,
            "packet_id": packet,
            "root_id": b["root_id"],
            "arm": b["arm"],
            "candidate_id": b["candidate_id"],
            "question_type": c.get("question_type"),
            "canonical_status": status,
            "canonical_resolution_reason": rr.get("resolution_reason"),
            "evidence_complete": bool(rr.get("chosen",{}).get("evidence_complete",False)),
            "review_source": rr.get("chosen"),
            "author_draft_path": str(draft),
            "author_draft_sha256": got_hash,
            "source_path": c.get("source_path"),
            "source_sha256": c.get("source_sha256"),
            "source_turn_count": c.get("source_turn_count"),
            "gold_option_ids": c.get("gold_option_ids"),
            "option_count": len(c.get("options",[])) if isinstance(c.get("options"),list) else None,
            "pre_solver_eligible": status == "PASS",
            "candidate": c,
        }
        records.append(rec)
    if errors:
        (out/"BUILD_ERRORS.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        raise SystemExit(f"validation errors: {len(errors)} (see {out/'BUILD_ERRORS.json'})")
    # Preserve canonical input evidence and hashes without copying or modifying it.
    canon_raw=canonical_path.read_bytes(); bind_raw=bindings_path.read_bytes()
    manifest={
      "schema":"fg_candidate_manifest_v1",
      "created_from":"training_set_rebuild_20261105_FG_exploration_run01",
      "canonical_review_path":str(canonical_path),
      "canonical_review_sha256":sha256_bytes(canon_raw),
      "bindings_path":str(bindings_path),
      "bindings_sha256":sha256_bytes(bind_raw),
      "denominator":len(records),
      "status_counts":dict(Counter(x["canonical_status"] for x in records)),
      "arm_counts":dict(Counter(x["arm"] for x in records)),
      "type_counts":dict(Counter(x["question_type"] for x in records)),
      "pre_solver_pass_count":sum(x["pre_solver_eligible"] for x in records),
      "hold_count":sum(not x["pre_solver_eligible"] for x in records),
      "solver_status":"NOT_STARTED",
      "training_status":"NOT_STARTED",
      "records":records,
    }
    (out/"CANDIDATE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (out/"CANDIDATE_MANIFEST.jsonl").open("w",encoding="utf-8") as f:
      for rec in records: f.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n")
    summary={k:manifest[k] for k in ["schema","canonical_review_sha256","bindings_sha256","denominator","status_counts","arm_counts","type_counts","pre_solver_pass_count","hold_count","solver_status","training_status"]}
    (out/"BUILD_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text("""# F/G candidate manifest (2026-11-08)\n\nThis directory is a new immutable downstream manifest. Frozen authoring, packet, binding, and canonical-review inputs were read and hash-checked; none were edited. `CANDIDATE_MANIFEST.json` embeds one candidate per review packet and records canonical PASS/HOLD status. `pre_solver_eligible` is only a review gate; Solver, lineage/semantic/exposure gates, and training admission remain pending.\n\nThe two canonical HOLD records (both root `cc-train-episode-171871`) are retained for audit but have `pre_solver_eligible=false`.\n""",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
