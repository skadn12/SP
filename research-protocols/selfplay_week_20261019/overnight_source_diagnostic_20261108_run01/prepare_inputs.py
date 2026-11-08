#!/usr/bin/env python3
"""Create read-only, hash-bound preparation inputs for the overnight plan.

The script deliberately does not sample, review, call a model, or alter any
existing artifact.  It scans the source ledger and current historical indexes
only to freeze exclusions and reproducible input hashes.
"""
from __future__ import annotations
import hashlib, json, os, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
PROTO = ROOT.parent / "structure_weakness_next_step_proposal_20261107_v1"
SRC = Path("/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl")
LEDGER = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/source_admission_cc_train_full_20261106_v1/SOURCE_ADMISSION_LEDGER.jsonl")
CANON = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl")
RESERV = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/source_admission_cc_train_full_20261106_v1/SOURCE_RESERVATIONS.json")
STATUS = PROTO / "NEXT_ROOT_ELIGIBILITY_STATUS_20261108_v3.json"
PROGRESS = PROTO / "NEXT_ROOT_ELIGIBILITY_PROGRESS_20261108_v1.json"
HISTORY = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/history_projection_assessment_20261107_v1/HISTORICAL_PROJECTION_ASSESSMENT.json")
QUALITY140 = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261105_FG_exploration_run01/quality_source_review_20261108_v1/MANIFEST.json")
REFERENCE_DIR = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261105_FG_exploration_run01/private_semantic_reference")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def ids_from_json(v):
    out = set()
    if isinstance(v, dict):
        for k, x in v.items():
            if k in {"root_id", "data_id", "candidate_id"} and isinstance(x, str):
                m = re.search(r"(?:episode[-_])?(\d+)$", x)
                if m: out.add(m.group(1))
            out |= ids_from_json(x)
    elif isinstance(v, list):
        for x in v: out |= ids_from_json(x)
    return out

def rel_root(x: str) -> str:
    m = re.search(r"(?:episode[-_])?(\d+)$", str(x))
    return m.group(1) if m else str(x)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "INPUT_MANIFEST.json").exists() or (ROOT / "REVIEWED_ROOT_REGISTRY.json").exists():
        raise SystemExit("frozen preparation artifacts already exist; refusing to overwrite")
    required = [SRC, LEDGER, CANON, RESERV, STATUS, PROGRESS, HISTORY, QUALITY140]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing required input(s): " + ", ".join(missing))

    excluded = defaultdict(set)
    # Canonical split is an explicit isolation boundary.
    canonical_rows = []
    with CANON.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); canonical_rows.append(d)
            if d.get("root_id"): excluded["canonical_split"].add(rel_root(d["root_id"]))
    # Historical reference filenames are the stable 70-root inventory key.
    for p in REFERENCE_DIR.glob("REFERENCE_cc-train-episode-*.json"):
        excluded["historical_reference_inventory"].add(rel_root(p.stem))
    # Current 140 packets are reserved for quality review and must not enter
    # this source diagnostic's denominator.
    q = load(QUALITY140)
    for e in q.get("entries", []):
        if e.get("root_id"): excluded["current_140_review_reservation"].add(rel_root(e["root_id"]))
    # Existing status/progress/reservation artifacts are all preserved as
    # exclusion evidence, even when their labels conflict.
    for p, reason in [(STATUS, "eligibility_status"), (PROGRESS, "eligibility_progress")]:
        for x in ids_from_json(load(p)): excluded[reason].add(x)
    r = load(RESERV)
    for x in r.get("reserved_roots", []):
        if x.get("root_id"): excluded["source_reservation"].add(rel_root(x["root_id"]))
    # The historical assessment's inventory count is evidence only; its raw
    # sidecars are not treated as a positive novelty claim.
    hist = load(HISTORY)
    for x in ids_from_json(hist): excluded["historical_assessment_references"].add(x)

    source_hash = sha256(SRC)
    expected_source_hash = "f0fccf01ec59eeeab68c0a300b0f1a24bacd8eea8719e77bdbd003c242f1407c"
    if source_hash != expected_source_hash:
        raise SystemExit(f"source hash mismatch: expected {expected_source_hash}, got {source_hash}")
    registry = {
        "schema_version": "overnight_reviewed_root_registry_v1",
        "status": "FROZEN_PREPARATION_INPUT",
        "source_sha256": source_hash,
        "generated_by": str(Path(__file__).resolve()),
        "policy": "Exclude canonical, historical-reference, current-140, reserved and previously adjudicated roots; no exclusion is a new rejection/admission decision.",
        "counts": {k: len(v) for k, v in sorted(excluded.items())},
        "union_count": len(set().union(*excluded.values()) if excluded else set()),
        "excluded_by_reason": {k: sorted(v, key=lambda z: (len(z), z)) for k, v in sorted(excluded.items())},
    }
    (ROOT / "REVIEWED_ROOT_REGISTRY.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    bindings = []
    for p, role in [(SRC,"conversation_chronicles_source"),(LEDGER,"source_admission_ledger"),(CANON,"canonical_split"),(RESERV,"source_reservations"),(STATUS,"eligibility_status_v3"),(PROGRESS,"eligibility_progress_v1"),(HISTORY,"historical_projection_assessment"),(QUALITY140,"quality_140_manifest")]:
        bindings.append({"role":role,"path":str(p),"bytes":p.stat().st_size,"sha256":sha256(p)})
    manifest = {
        "schema_version":"overnight_input_manifest_v1",
        "status":"FROZEN_PREPARATION_NOT_STARTED",
        "plan_path":str(PROTO / "OVERNIGHT_PLAN_20261108_v1.json"),
        "source":{"path":str(SRC),"records":160000,"sha256":source_hash,"mutate":False},
        "bindings":bindings,
        "sampling":{"seed":"overnight-source-diagnostic-20261108-v1","strata":10,"score_per_stratum":2,"random_per_stratum":2,"maximum_unique_roots":40,"replace_failed_samples":False},
        "exclusion_registry":str(ROOT / "REVIEWED_ROOT_REGISTRY.json"),
        "exclusion_registry_sha256":sha256(ROOT / "REVIEWED_ROOT_REGISTRY.json"),
        "review_limits":{"primary_root_reviews":40,"independent_root_checks":40,"adjudications_total":10,"history_deep_reviews":12,"solver_requests":0,"training_runs":0},
        "next_action":"wait_for_observed_healthy_training_then_freeze_sample_manifest",
    }
    (ROOT / "INPUT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status":manifest["status"],"source_sha256":manifest["source"]["sha256"],"exclusion_union_count":registry["union_count"],"quality_140_status_counts":q.get("status_counts")}, ensure_ascii=False))

if __name__ == "__main__": main()
