#!/usr/bin/env python3
"""Read-only validation for run02 preparation; never samples or mutates."""
from __future__ import annotations
import hashlib,json,re,sys
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def rid(v):
 m=re.search(r"(?:cc-train-episode-|episode[-_])(\d+)$",str(v))
 return m.group(1) if m else str(v)
def main():
 m=json.loads((R/"INPUT_MANIFEST.json").read_text()); a=json.loads((R/"PREPARATION_AUDIT.json").read_text()); e=json.loads((R/"EXCLUSION_REGISTRY.json").read_text())
 assert m["status"]=="FROZEN_PREPARATION_NOT_STARTED"
 assert a["status"]=="PREPARED_NOT_STARTED" and not a["sampling_started"] and not a["blind_packets_created"]
 assert a["solver_requests"]==0 and a["training_runs"]==0
 assert m["source"]["sha256"]==sha(Path(m["source"]["path"]))
 assert m["quality_binding"]["parent_manifest_sha256"]==sha(Path(m["quality_binding"]["parent_manifest_path"]))
 assert m["quality_binding"]["canonical_resolution_sha256"]==sha(Path(m["quality_binding"]["canonical_resolution_path"]))
 assert e["source_sha256"]==m["source"]["sha256"]
 for b in m["bindings"]:
  p=Path(b["path"]); assert p.exists(), p
  assert p.stat().st_size==b["bytes"], b
  assert sha(p)==b["sha256"], b
 assert m["candidate_root_reference"]["root_count"]==70
 assert m["old135_reference"]["root_count"]==65
 assert e["counts"]["current_140_candidate_roots"]==70
 assert e["counts"]["current_140_pass_roots"]==69
 assert e["counts"]["current_140_hold_roots"]==1
 forbidden=[]
 for p in R.rglob("*"):
  if p.is_file() and p.name not in {"prepare_run02.py","validate_preparation.py","README.md","INPUT_MANIFEST.json","EXCLUSION_REGISTRY.json","PREPARATION_AUDIT.json"}:
   forbidden.append(str(p))
 if forbidden: raise AssertionError("unexpected execution artifacts: "+repr(forbidden))
 print(json.dumps({"validation":"PASS","sampling_started":False,"blind_packets":False,"solver_requests":0,"training_runs":0,"bound_source_sha256":m["source"]["sha256"],"bound_quality_parent_sha256":m["quality_binding"]["parent_manifest_sha256"],"bound_quality_canonical_sha256":m["quality_binding"]["canonical_resolution_sha256"]},ensure_ascii=False))
if __name__=="__main__":main()

