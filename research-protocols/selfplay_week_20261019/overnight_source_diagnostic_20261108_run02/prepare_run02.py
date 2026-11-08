#!/usr/bin/env python3
"""Create a hash-bound, preparation-only overnight diagnostic run.

This script freezes references and exclusion sets only. It does not sample,
create blind packets, call models, review roots, or start background work.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
SOURCE=Path("/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl")
PLAN=BASE/"structure_weakness_next_step_proposal_20261107_v1/OVERNIGHT_PLAN_20261108_v1.json"
QDIR=BASE/"training_set_rebuild_20261105_FG_exploration_run01/quality_source_review_20261108_v1"
QPARENT=QDIR/"MANIFEST.json"
QCANON=QDIR/"CANONICAL_REVIEW_RESOLUTION_20261108_v1.json"
CANDIDATES=BASE/"training_set_rebuild_20261105_FG_exploration_run01/inputs"
OLD=BASE/"training_set_rebuild_20261105_F_supplement_run01/repaired_training_export_v1"
OLDTRAIN=OLD/"TRAIN.jsonl"
OLDLINEAGE=OLD/"QUESTION_LINEAGE.jsonl"
OLDSUMMARY=OLD/"SUMMARY.json"
CANON=Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl")
RES=BASE/"training_set_structure_pilot_plan_20261106_v1/source_admission_cc_train_full_20261106_v1/SOURCE_RESERVATIONS.json"
HIST=BASE/"training_set_structure_pilot_plan_20261106_v1/execution_20261107_v1/HISTORICAL_REFERENCE_MANIFEST.json"
SEED="overnight-source-diagnostic-20261108-v1"

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
def rid(v):
 m=re.search(r"(?:cc-train-episode-|episode[-_])(\d+)$",str(v))
 return m.group(1) if m else str(v)
def roots(obj):
 out=set()
 if isinstance(obj,dict):
  for k,v in obj.items():
   if k in {"root_id","data_id","candidate_id","source_id"} and isinstance(v,str):
    m=re.search(r"(?:cc-train-episode-|episode[-_])(\d+)$",v)
    if m: out.add(m.group(1))
   out |= roots(v)
 elif isinstance(obj,list):
  for v in obj: out |= roots(v)
 return out
def dump(name,obj):
 p=ROOT/name
 if p.exists(): raise SystemExit("refusing to overwrite "+str(p))
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 return sha(p)
def bind(p,role):
 return {"role":role,"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)}
def main():
 required=[SOURCE,PLAN,QPARENT,QCANON,CANDIDATES,OLDTRAIN,OLDLINEAGE,OLDSUMMARY,CANON,RES,HIST]
 missing=[str(p) for p in required if not p.exists()]
 if missing: raise SystemExit("missing inputs: "+", ".join(missing))
 q=json.loads(QCANON.read_text()); parent=json.loads(QPARENT.read_text())
 if q.get("manifest_sha256")!=sha(QPARENT): raise SystemExit("quality canonical resolution does not bind parent manifest")
 if q.get("denominator")!=140 or len(q.get("resolved",[]))!=140: raise SystemExit("quality review denominator mismatch")
 all_roots={rid(x["root_id"]) for x in q["resolved"] if x.get("root_id")}
 pass_roots={rid(x["root_id"]) for x in q["resolved"] if x.get("effective_status")=="PASS"}
 hold_roots={rid(x["root_id"]) for x in q["resolved"] if x.get("effective_status")=="HOLD"}
 if len(all_roots)!=70: raise SystemExit("expected 70 candidate roots")
 candidate_files={}
 for p in sorted(CANDIDATES.glob("cc-train-episode-*")):
  n=rid(p.name)
  if n not in all_roots: continue
  candidate_files[n]=[{"path":str(f.relative_to(p)),"bytes":f.stat().st_size,"sha256":sha(f)} for f in sorted(p.rglob("*")) if f.is_file()]
 if set(candidate_files)!=all_roots: raise SystemExit("candidate input root directory mismatch")
 old_lineage_roots=set(); lineage_n=0
 for line in OLDLINEAGE.open(encoding="utf-8"):
  lineage_n+=1; old_lineage_roots |= roots(json.loads(line))
 old_train_roots=set(); train_n=0
 for line in OLDTRAIN.open(encoding="utf-8"):
  train_n+=1; old_train_roots |= roots(json.loads(line))
 if (train_n,lineage_n,len(old_train_roots),len(old_lineage_roots))!=(135,135,65,65):
  raise SystemExit(f"old-135 mismatch: train={train_n}, lineage={lineage_n}, roots={len(old_train_roots)}/{len(old_lineage_roots)}")
 canonical_roots=set()
 for line in CANON.open(encoding="utf-8"): canonical_roots |= roots(json.loads(line))
 reservation_roots=roots(json.loads(RES.read_text()))
 hist_roots=roots(json.loads(HIST.read_text()))
 ex={
  "current_140_candidate_roots":sorted(all_roots,key=int),
  "current_140_pass_roots":sorted(pass_roots,key=int),
  "current_140_hold_roots":sorted(hold_roots,key=int),
  "old135_train_roots":sorted(old_train_roots,key=int),
  "old135_lineage_roots":sorted(old_lineage_roots,key=int),
  "canonical_split_roots":sorted(canonical_roots,key=int),
  "source_reservation_roots":sorted(reservation_roots,key=int),
  "historical_reference_roots":sorted(hist_roots,key=int)}
 reg={"schema_version":"overnight_exclusion_registry_v2","status":"FROZEN_PREPARATION_ONLY","policy":"Exclusion references only; this does not create or revoke source admission.","source_sha256":sha(SOURCE),
  "quality_binding":{"parent_manifest":str(QPARENT),"parent_sha256":sha(QPARENT),"canonical_resolution":str(QCANON),"canonical_sha256":sha(QCANON),"denominator":140,"resolved":140,"unique_roots":len(all_roots),"pass_unique_roots":len(pass_roots),"hold_unique_roots":len(hold_roots)},
  "counts":{k:len(v) for k,v in ex.items()},"union_count":len(set().union(*(set(v) for v in ex.values()))),"excluded_by_reason":ex,"candidate_root_file_hashes":candidate_files,"old135_train_records":train_n,"old135_lineage_records":lineage_n}
 reg_hash=dump("EXCLUSION_REGISTRY.json",reg)
 bindings=[bind(SOURCE,"conversation_chronicles_source"),bind(PLAN,"overnight_plan"),bind(QPARENT,"quality_140_parent_manifest"),bind(QCANON,"quality_140_canonical_resolution"),bind(OLDTRAIN,"old135_train"),bind(OLDLINEAGE,"old135_question_lineage"),bind(OLDSUMMARY,"old135_summary"),bind(CANON,"canonical_split"),bind(RES,"source_reservations"),bind(HIST,"historical_reference_manifest")]
 manifest={"schema_version":"overnight_input_manifest_v2","status":"FROZEN_PREPARATION_NOT_STARTED","run_policy":"prepare_and_validate_only_until_observed_healthy_training_gate","plan_path":str(PLAN),"plan_sha256":sha(PLAN),
  "source":{"path":str(SOURCE),"records":160000,"sha256":sha(SOURCE),"mutate":False},
  "quality_binding":{"parent_manifest_path":str(QPARENT),"parent_manifest_sha256":sha(QPARENT),"canonical_resolution_path":str(QCANON),"canonical_resolution_sha256":sha(QCANON),"denominator":140,"effective_status_counts":q.get("effective_status_counts"),"unique_candidate_roots":len(all_roots)},
  "bindings":bindings,"candidate_root_reference":{"path":str(CANDIDATES),"root_count":len(candidate_files),"registry_sha256":reg_hash},
  "old135_reference":{"train_path":str(OLDTRAIN),"lineage_path":str(OLDLINEAGE),"train_sha256":sha(OLDTRAIN),"lineage_sha256":sha(OLDLINEAGE),"root_count":len(old_lineage_roots)},
  "sampling":{"seed":SEED,"strata":10,"per_stratum":{"SCORE":2,"RANDOM_FROM_REMAINDER":2},"maximum_unique_roots":40,"replace_failed_samples":False},
  "review_limits":{"primary_root_reviews":40,"independent_root_checks":40,"adjudications_total":10,"history_deep_reviews":12,"solver_requests":0,"training_runs":0,"wall_clock_limit_hours":8},
  "exclusion_registry_path":str(ROOT/"EXCLUSION_REGISTRY.json"),"exclusion_registry_sha256":reg_hash,"next_action":"wait_for_explicit_healthy_training_gate"}
 man_hash=dump("INPUT_MANIFEST.json",manifest)
 audit={"schema_version":"overnight_preparation_audit_v2","status":"PREPARED_NOT_STARTED","run_directory":str(ROOT),"writes_to_old_run_directories":False,"sampling_started":False,"blind_packets_created":False,"solver_requests":0,"training_runs":0,
  "quality_checks":{"parent_hash_matches_canonical":True,"denominator":140,"resolved":140,"candidate_roots":len(all_roots),"pass_roots":len(pass_roots),"hold_roots":len(hold_roots)},
  "old135_checks":{"train_records":train_n,"lineage_records":lineage_n,"train_roots":len(old_train_roots),"lineage_roots":len(old_lineage_roots)},
  "isolated_script_defects":["run01 sample IDs exposed SCORE/RANDOM arm in the first character; run02 has no sampling code in preparation stage and later sample IDs must be opaque.","run01 preparation registry used narrow recursive root key extraction and did not bind the canonical quality resolution plus parent manifest; run02 binds both and freezes explicit root sets.","run01 already contained an obsolete quality binding; run02 is a new immutable directory."],
  "manifest_sha256":man_hash,"registry_sha256":reg_hash,"sampling_gate":"explicit training-controller message required"}
 dump("PREPARATION_AUDIT.json",audit)
 (ROOT/"README.md").write_text("# Overnight source diagnostic run 02\n\nPreparation-only immutable references for the frozen plan. Bound to both the quality parent manifest and canonical resolution, all 140 candidate root inputs, and old-135 TRAIN plus QUESTION_LINEAGE.\n\nNo sampling, blind packets, model calls, Solver calls, or training have started. Wait for an explicit observed-healthy training gate. Do not reuse or modify run01.\n",encoding="utf-8")
 print(json.dumps({"status":"PREPARED_NOT_STARTED","run":str(ROOT),"manifest_sha256":man_hash,"registry_sha256":reg_hash,"candidate_roots":len(all_roots),"old135_roots":len(old_lineage_roots),"sampling_started":False},ensure_ascii=False))
if __name__=="__main__": main()

