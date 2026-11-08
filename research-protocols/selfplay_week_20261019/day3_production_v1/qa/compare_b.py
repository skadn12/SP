import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
P=Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/day3_production_v1")
def bind(p):
 b=p.read_bytes();return {"path":str(p),"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
def write(p,j):
 assert not p.exists(),p;p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+"\n")
def now():return datetime.now(timezone.utc).isoformat()
rows=[];checks=0
for recpath in sorted((P/"qa/B").rglob("IMMUTABLE_RECONSTRUCTION.json")):
 rec=json.loads(recpath.read_text());fp=recpath.parent/"RECONSTRUCTION_FREEZE.json";fr=json.loads(fp.read_text())
 assert bind(recpath)==fr["reconstruction"];assert rec["author_private_read"] is False
 root=rec["candidate_id"].split("stage1-")[1].rsplit("-multi_select-",1)[0].rsplit("-ordering-",1)[0]
 pubpath=Path(rec["public_binding"]["path"]);privatepath=pubpath.parent.parent/"private"/pubpath.name
 private=json.loads(privatepath.read_text())
 source=json.loads(Path(rec["source_binding"]["path"]).read_text());ti={t["turn_ref"]:(i,t) for i,t in enumerate(source["turns"])}
 errors=[];info=[];checks+=3
 if private["candidate_id"]!=rec["candidate_id"]:errors.append("CANDIDATE_ID_MISMATCH")
 if private["gold"]!=rec["independent_gold"]:errors.append("GOLD_MISMATCH")
 if private["source_sha256"]!=rec["source_binding"]["sha256"]:errors.append("SOURCE_HASH_MISMATCH")
 for label,e in private["evidence"].items():
  for x in [e["selected_anchor"],*e.get("plausible_mentions",[])]:
   checks+=4
   if x["turn_ref"] not in ti:errors.append(label+":BAD_REF");continue
   i,t=ti[x["turn_ref"]]
   if t["text"]!=x["quote"]:errors.append(label+":QUOTE_MISMATCH")
   if t["scene_ref"]!=x["scene_ref"]:errors.append(label+":SCENE_MISMATCH")
   if t["speaker"]!=x["speaker"]:errors.append(label+":SPEAKER_MISMATCH")
   if x["global_index"]!=i+1:errors.append(label+":ONE_BASED_INDEX_MISMATCH")
  checks+=1
  if "item_checks" in rec and e["selected_anchor"]["turn_ref"]!=rec["item_checks"][label]["selected_turn_ref"]:
   errors.append(label+":ANCHOR_MISMATCH")
 if "pair_checks" in rec:
  pp=private["ordering"]["pairs"];checks+=2
  if len(pp)!=15:errors.append("PAIR_COUNT")
  actual={(x["earlier"],x["later"],x["within_scene"]) for x in pp}
  expected={(x["earlier"],x["later"],x["same_scene"]) for x in rec["pair_checks"]}
  if actual!=expected:errors.append("PAIR_MISMATCH")
 if rec["relation_target"]["status"]=="FAIL":
  state="HOLD";issue="MC_BINDING target not realized: no sufficiently matched true/false binding contrast; false police-station/code row permits a type shortcut. A new contrast would change factual option content, so it is not eligible for wording-only repair."
  clar=None
 else:
  state="CLARIFY_ONCE";issue="B initial public text is Chinese while the matched A arm is English. Language is an uncontrolled arm difference. The content, independent Gold, source positions, and declared profile otherwise pass."
  clar={"kind":"WORDING_ONLY_LANGUAGE_TRANSLATION","request":"Translate this candidate's question and six options from Chinese into English. Preserve every label, entity, fact, negation, modality, requested-vs-completed distinction, ordering first-report policy, independent Gold, selected anchors, and scene profile. Do not add/drop a fact, change evidence, alter Gold, or substitute an item. This is the sole permitted content clarification for this inherited slot.","frozen_gold":rec["independent_gold"],"frozen_anchor_refs":{k:v["selected_turn_ref"] for k,v in rec.get("item_checks",{}).items()},"max_clarification_calls":1,"requested_output_tokens":3072,"actual_usage":None,"actual_cap_support":"UNKNOWN"}
 out={"schema_version":"day3_initial_private_comparison_v1","reviewer":"/root/day3_source_inventory","created_at_utc":now(),
      "candidate_id":rec["candidate_id"],"phase":"PRIVATE_COMPARISON_AFTER_IMMUTABLE_RECONSTRUCTION_FREEZE",
      "reconstruction_freeze_binding":bind(fp),"immutable_reconstruction_binding":bind(recpath),
      "author_private_binding":bind(privatepath),"public_binding":rec["public_binding"],"source_binding":rec["source_binding"],
      "private_gold_matches_independent":private["gold"]==rec["independent_gold"],"private_evidence_mechanical_errors":errors,
      "content_truth_status":rec["content_truth_status"],"relation_target":rec["relation_target"],
      "shortcut_status":"FAIL_TYPE_SHORTCUT" if rec["relation_target"]["status"]=="FAIL" else "PASS_WITH_RECORDED_LOCATING_CUES",
      "ordering_profile":rec.get("profile_check"),"pair_count":len(rec.get("pair_checks",[])),
      "read_boundary":{"b_feedback_card_content_read":False,"solver_scores_read":False,"private_ordering_lineage_field_incidentally_present":True if "pair_checks" in rec else False,"lineage_scope":"Only private record field naming the prior card is visible after reconstruction. Underlying B feedback card content was not opened; feedback lineage audit remains with main agent."},
      "initial_disposition":state,"issue":issue,"content_clarification_request":clar,
      "author_mention_documentation":"Author mention lists are source-valid but sometimes omit related acknowledgements/prospective predecessors. QA independent reconstruction retains a broader inclusion/exclusion search; use it in admitted evidence instead of calling author's list exhaustive.",
      "gold_or_anchor_repair_by_reviewer":False}
 assert not errors,(rec["candidate_id"],errors)
 write(recpath.parent/"INITIAL_REVIEW.json",out)
 rows.append({"candidate_id":rec["candidate_id"],"content":rec["content_truth_status"],"target":rec["relation_target"]["status"],"disposition":state,"review_binding":bind(recpath.parent/"INITIAL_REVIEW.json"),"public_binding":rec["public_binding"]})
summary={"schema_version":"day3_b_initial_qa_summary_v1","created_at_utc":now(),"status":"INITIAL_REVIEW_COMPLETE_TRANSLATION_PENDING",
 "reviewer":"/root/day3_source_inventory","planned_slots":12,"public_candidates_reviewed":10,"author_shortfalls":2,
 "independent_gold_agreement":10,"evidence_mechanical_check_count":checks,"evidence_mechanical_errors":0,
 "initial_dispositions":{"CLARIFY_ONCE_TRANSLATE_ONLY":9,"HOLD_TARGET_FAILURE":1},
 "accepted_candidates":0,"all_clarifications_require_post_review":True,
 "source_inventory_pass_is_not_question_pass":True,"solver_calls":0,"questions":rows}
write(P/"qa/B_INITIAL_REVIEW_SUMMARY.json",summary)
print(json.dumps({"initial_reviews":10,"checks":checks,"errors":0,"translation_clarifications":9,"hold":1}))
