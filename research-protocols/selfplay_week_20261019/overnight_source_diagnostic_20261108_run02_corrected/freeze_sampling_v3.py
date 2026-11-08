#!/usr/bin/env python3
"""Freeze 40 diagnostic samples only after an explicit healthy-training gate.

Arm labels and root identity stay in a controller-private mapping. Blind packet
IDs are opaque and packet content has source IDs redacted.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path("/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl")
RELATIONS=["Classmates","Neighbors","Co-workers","Mentee and Mentor","Parent and Child","Patient and Doctor","Husband and Wife","Student and Teacher","Athlete and Coach","Employee and Boss"]
STOP=set("the a an and or but if then to of in on for with from by is are was were be been being i you he she it they we them this that these those have has had do did does can could should would will just really very about what when where how why there here their our your my his her its as at into out up down so than too also not no yes all any some more one two three first second third fourth fifth".split())
MARKERS=re.compile(r"\b(remember|earlier|previous|last time|few weeks|a while ago|mentioned|again|before|since then|as you said|we talked|we discussed|reminds me|already|still|continue|instead|if|because|therefore|should|could|try|suggest|advice|plan|need to|have to|must|goal|decided|agreed)\b",re.I)
TEMP=re.compile(r"\b(before|after|later|then|first|next|finally|already|still|again|last|previous|earlier|week|month|year|tomorrow|yesterday)\b",re.I)
ROOT_TOKEN=re.compile(r"cc-train-episode-\d+|episode[-_]\d+",re.I)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def rid(v):
 m=re.search(r"(?:cc-train-episode-|episode[-_])(\d+)$",str(v))
 return m.group(1) if m else str(v)
def toks(v):
 return set(re.findall(r"[a-zA-Z][a-zA-Z']+"," ".join(map(str,v)).lower()))-STOP
def score(d):
 ss=[d.get(k,[]) for k in ("first_session_dialogue","second_session_dialogue","third_session_dialogue","fourth_session_dialogue","fifth_session_dialogue")]
 ts=[toks(x) for x in ss]
 refs=sum(len(MARKERS.findall(" ".join(map(str,ss[j])))) for j in range(1,5))
 temp=sum(len(TEMP.findall(" ".join(map(str,ss[j])))) for j in range(1,5))
 overlaps=[len(ts[j]&ts[j-1]) for j in range(1,5)]
 cross=sum(overlaps); distinct=sum(1 for x in overlaps if x>=4)
 nums=sum(bool(re.search(r"\b\d+(?:\.\d+)?\b"," ".join(map(str,x)))) for x in ss)
 strategy=sum(len(re.findall(r"\b(should|could|try|suggest|advice|recommend|plan|need to|must|goal|decided|agreed)\b"," ".join(map(str,x)),re.I)) for x in ss)
 return 2*distinct+.15*cross+.25*refs+.15*temp+.1*strategy+.2*nums,{"cross_adjacent_token_overlap":overlaps,"reference_markers_later":refs,"temporal_markers_later":temp,"strategy_markers":strategy,"numeric_sessions":nums}
def redact(v):
 if isinstance(v,str): return ROOT_TOKEN.sub("SOURCE_TOKEN",v)
 if isinstance(v,list): return [redact(x) for x in v]
 if isinstance(v,dict): return {k:redact(x) for k,x in v.items()}
 return v
def anon(d):
 hidden={"dataID","relationship","summary","root_id","source_id","candidate_id"}
 return {k:redact(v) for k,v in d.items() if k not in hidden}
def blind_order(selected):
 # Separate the presentation order from the arm and stratum allocation order.
 return sorted(selected,key=lambda item: hashlib.sha256(
  ("overnight-blind-order-v1:"+item[1]["root_id"]+":"+item[1]["record_sha256"]).encode()).digest())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--health-marker",required=True); args=ap.parse_args()
 marker=Path(args.health_marker)
 if not marker.exists(): raise SystemExit("training health marker missing; sampling remains gated")
 gate=json.loads(marker.read_text())
 if gate.get("status")!="HEALTHY" or gate.get("explicit_controller_attestation") is not True:
  raise SystemExit("health marker must have status=HEALTHY and explicit_controller_attestation=true")
 for name in ("SAMPLING_SPEC.json","SAMPLE_MANIFEST.json","BLIND_MAPPING_PRIVATE.json","blind_packets"):
  if (ROOT/name).exists(): raise SystemExit("frozen sampling artifact exists; refusing overwrite")
 m=json.loads((ROOT/"INPUT_MANIFEST.json").read_text()); reg=json.loads((ROOT/"EXCLUSION_REGISTRY.json").read_text())
 if sha(SOURCE)!=m["source"]["sha256"]: raise SystemExit("source hash changed")
 if sha(ROOT/"EXCLUSION_REGISTRY.json")!=m["exclusion_registry_sha256"]: raise SystemExit("exclusion registry changed")
 excluded=set().union(*(set(v) for v in reg["excluded_by_reason"].values()))
 by=defaultdict(list); n=0
 with SOURCE.open(encoding="utf-8") as f:
  for line_no,line in enumerate(f,1):
   n+=1; d=json.loads(line); r=rid(d.get("dataID","")); rel=d.get("relationship")
   if r in excluded or rel not in RELATIONS: continue
   sc,feat=score(d)
   by[rel].append({"root_id":r,"data_id":d.get("dataID"),"source_line_index":line_no,"record_sha256":hashlib.sha256(line.encode()).hexdigest(),"relationship":rel,"score":round(sc,8),"features":feat,"record":d})
 out=ROOT/"blind_packets"; out.mkdir()
 allocations=[]; private=[]; selected=[]
 for si,rel in enumerate(RELATIONS,1):
  items=sorted(by[rel],key=lambda x:(-x["score"],x["root_id"],x["record_sha256"]))
  top=items[:2]; top_ids={x["root_id"] for x in top}
  rem=[x for x in items if x["root_id"] not in top_ids]
  rem.sort(key=lambda x:hashlib.sha256(f"overnight-source-diagnostic-20261108-v1:{x['root_id']}".encode()).hexdigest())
  chosen=[("SCORE",x) for x in top]+[("RANDOM_FROM_REMAINDER",x) for x in rem[:2]]
  selected.extend((arm,x) for arm,x in chosen)
 for arm,x in blind_order(selected):
  pid=f"P{len(allocations)+1:03d}"
  p=out/(pid+".json"); p.write_text(json.dumps(anon(x["record"]),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
  entry={"sample_id":pid,"anonymous_packet":str(p),"anonymous_packet_sha256":sha(p),"packet_content_redacted":True}
  allocations.append(entry)
  private.append({**entry,"sampler":arm,"stratum":x["relationship"],"root_id":x["root_id"],"data_id":x["data_id"],"score":x["score"],"features":x["features"],"source_path":str(SOURCE),"source_line_index":x["source_line_index"],"record_sha256":x["record_sha256"]})
 spec={"schema_version":"overnight_sampling_spec_v2","status":"FROZEN_PRE_REVIEW","source_path":str(SOURCE),"source_sha256":sha(SOURCE),"seed":"overnight-source-diagnostic-20261108-v1","strata":RELATIONS,"per_stratum":{"SCORE":2,"RANDOM_FROM_REMAINDER":2},"total":len(allocations),"opaque_packet_ids":True,"arm_hidden_from_blind_review":True,"replace_failed_samples":False,"stop_early_when_eight_found":False,"health_marker":str(marker),"health_marker_sha256":sha(marker),"exclusion_registry":str(ROOT/"EXCLUSION_REGISTRY.json"),"exclusion_registry_sha256":sha(ROOT/"EXCLUSION_REGISTRY.json"),"history_priority":"SHA256(seed + ':history:' + sample_id), first six source-pass per sampler, unused quota transferred in same order"}
 (ROOT/"SAMPLING_SPEC.json").write_text(json.dumps(spec,ensure_ascii=False,indent=2)+"\n")
 (ROOT/"SAMPLE_MANIFEST.json").write_text(json.dumps({"schema_version":"overnight_sample_manifest_v2","status":"FROZEN_PRE_REVIEW","source_sha256":sha(SOURCE),"sample_count":len(allocations),"allocations":allocations},ensure_ascii=False,indent=2)+"\n")
 (ROOT/"BLIND_MAPPING_PRIVATE.json").write_text(json.dumps({"schema_version":"overnight_blind_mapping_private_v2","status":"CONTROLLER_ONLY","source_sha256":sha(SOURCE),"mapping":private},ensure_ascii=False,indent=2)+"\n")
 for name in ("SOURCE_REVIEW.jsonl","INDEPENDENT_CHALLENGE.jsonl","ADJUDICATION.jsonl"):
  (ROOT/name).touch()
 (ROOT/"BUDGET_LEDGER.json").write_text(json.dumps({"schema_version":"overnight_budget_ledger_v2","status":"FROZEN_PRE_REVIEW","primary_root_reviews":{"budget":40,"used":0},"independent_checks":{"budget":40,"used":0},"adjudications":{"budget":10,"used":0},"history_deep_reviews":{"budget":12,"used":0},"solver_requests":0,"training_runs":0,"wall_clock_limit_hours":8},indent=2)+"\n")
 (ROOT/"CONTROLLER_STATE.json").write_text(json.dumps({"schema_version":"overnight_controller_state_v2","status":"FROZEN_PRE_REVIEW","stage":"ready_for_blind_review","sample_manifest":str(ROOT/"SAMPLE_MANIFEST.json"),"resume_rule":"never rerun frozen records; append one durable record per completed root; stop on hash/identity/leakage mismatch"},indent=2)+"\n")
 print(json.dumps({"status":"FROZEN_PRE_REVIEW","sample_count":len(allocations),"sampler_counts":dict(Counter(x['sampler'] for x in private)),"opaque_ids":True},ensure_ascii=False))
if __name__=="__main__": main()
