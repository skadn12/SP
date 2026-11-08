"""Merge signed content reviews with batch semantic deduplication.

Run once after all reviewer artifacts are complete. This does not make a
training pool, generate examples, or call a model.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import json
import hashlib
import sys

D=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write("\n")
def binding(p):return {"path":str(p),"sha256":sha(p)}
dedup_path=Path(sys.argv[1]).resolve()
dedup=read(dedup_path)
# The signed reviewer may wrap rows in a named list; no semantic inference here.
rows=dedup.get("candidate_rows",dedup.get("candidates",dedup.get("rows",dedup.get("candidate_recommendations"))))
if rows is None:raise RuntimeError("Supply a per-candidate deduplication file.")
bycid={r["candidate_id"]:r for r in rows}
assert len(bycid)==20
freeze=read(D/"EXECUTION_FREEZE.json")
files=sorted((D/"records/A").glob("A-week1-*.json"))+sorted((D/"records/B").glob("*/*.json"))
assert len(files)==20
final=[];public=[];private=[];held=[];provenance=[]
now=datetime.now(timezone.utc).isoformat()
for path in files:
    r=read(path);d=bycid[r["candidate_id"]]
    old=json.loads(json.dumps(r))
    for k in ["family_id","atomic_relation_keys","related_candidate_ids","reuse_class","duplicate_review"]:
        r["fact_family"][k]=d[k]
    r["fact_family"]["reuse_class"]={
        "OVERLAPPING_FACT_DIFFERENT_RELATION":"OVERLAPPING_FACTS_DIFFERENT_RELATION",
        "SAME_FACT_PARAPHRASE":"SAME_FACTS_PARAPHRASE",
        "NEW_RELATION_WITHIN_BATCH":"NEW_RELATIONS",
    }.get(r["fact_family"]["reuse_class"],r["fact_family"]["reuse_class"])
    eligible=(r["qa_review"]["status"]=="PASS" and r["target_realization"]["target_implemented"]=="YES"
              and d["duplicate_review"]=="PASS" and not d["within_arm_duplicate"])
    h=r["generation_history"][-1]
    content_path=Path(h["raw_response_path"])
    assert sha(content_path)==h["candidate_content_sha256"]
    if eligible:
        state="ACCEPT_AFTER_CLARIFICATION" if r["budget"]["content_clarification_used"] else "ACCEPT_DIRECT"
        r["disposition"]["state"]=state
        r["disposition"]["accepted_content_sha256"]=sha(content_path)
        r["disposition"]["reason_codes"]=["SOURCE_AND_INDEPENDENT_CONTENT_TARGET_PASS","BATCH_SEMANTIC_DUPLICATE_REVIEW_PASS","HISTORICAL_FACT_REUSE_DISCLOSED"]
    else:
        r["disposition"]["state"]="HOLD"
        r["disposition"]["accepted_content_sha256"]=None
        r["disposition"]["reason_codes"]=["TARGET_OR_DUPLICATE_REVIEW_HOLD"]
    dest=D/"final_records"/(r["candidate_id"]+".json")
    dump(dest,r);final.append(r)
    gold=(r.get("multi_select") or r.get("ordering"))
    gold=gold.get("gold_option_ids",gold.get("gold_item_ids"))
    entry={"candidate_id":r["candidate_id"],"public":binding(content_path),
           "record":binding(dest),"source":{"path":r["source_binding"]["source_path"],"sha256":r["source_binding"]["source_sha256"]},
           "arm":r["arm"],"root_id":r["root_id"],"question_type":r["question_type"],
           "slot_id":r["inherited_slot_id"],"state":r["disposition"]["state"]}
    if eligible:
        public.append(read(content_path))
        private.append({**entry,"gold":gold,"fact_family_id":r["fact_family"]["family_id"],"use":"CONTENT_AUDITED_CANDIDATE_NOT_TRAINING_POOL_OR_HELD_OUT_EVAL"})
    else:held.append(entry)
    provenance.append({"candidate_id":r["candidate_id"],"original_schema_record":binding(path),"final_schema_record":binding(dest),
                       "deduplication_record":binding(dedup_path),"changed_sections":["fact_family","disposition"],
                       "dedup_reuse_class":d["reuse_class"],"schema_reuse_class":r["fact_family"]["reuse_class"],
                       "question_gold_anchors_changed":False,"content_QA_decision_unchanged":True})
assert len(private)==19 and len(held)==1
for name,records in [("content_accepted.public.jsonl",public),("content_accepted.private.jsonl",private),("held_candidates.jsonl",held)]:
    p=D/"exports"/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:
        for x in records:f.write(json.dumps(x,ensure_ascii=False)+"\n")
dump(D/"FINAL_RECORD_PROVENANCE.json",{"created_at_utc":now,"rows":provenance,
    "dedup_scope":"This actual 20-candidate batch only. Historical legacy question-pool semantic independence remains unestablished. All source roots have historical Train exposure.",
    "not_training_pool_freeze":True})
pairs=[]
for slot in sorted({r["slot_id"] for r in private}):
    a=next((r for r in private if r["arm"]=="A" and r["slot_id"]==slot),None)
    b=next((r for r in private if r["arm"]=="B" and r["slot_id"]==slot),None)
    if a and b:pairs.append({"slot_id":slot,"A":a["candidate_id"],"B":b["candidate_id"]})
dump(D/"MATCHED_SLOT_PAIRS.json",{"created_at_utc":now,"pairs":pairs,"matched_slots":len(pairs),"candidate_count":len(pairs)*2,
    "unmatched_content_pass":[r["candidate_id"] for r in private if r["slot_id"] not in {p["slot_id"] for p in pairs}],
    "not_training_selection_or_training_pool_freeze":True,"selection_uses_solver_scores":False})
shortfalls=[]
for arm in ["A","B"]:
    for root in freeze["first_batch_roots"]:
        s=next(s for s in freeze["slots"] if s["arm"]==arm and s["root_id"]==root and s["question_type"]=="multi_select" and s["local_slot"]==3)
        shortfalls.append({"candidate_slot_id":s["candidate_slot_id"],"slot_id":s["slot_id"],"arm":arm,"root_id":root,
                          "question_type":"multi_select","target_gold_count":s["target_gold_count"],"state":"SHORTFALL",
                          "reason":"SOURCE_LACKS_VERIFIED_NONREDUNDANT_CONDITIONAL_JOINT_RELATION","question_authored":False})
slot_ledger=private+held+shortfalls
assert len(slot_ledger)==24
dump(D/"exports/SLOT_LEDGER.json",{"created_at_utc":now,"rows":slot_ledger,"planned_first_batch_slots":24,"generation_scope":"Only first two inherited Train roots."})
arms={}
for arm in ["A","B"]:
    accepted=[r for r in final if r["arm"]==arm and r["disposition"]["state"].startswith("ACCEPT")]
    arms[arm]={"planned_slots":12,"authored_candidates":10,"accepted":len(accepted),
               "held":sum(r["arm"]==arm for r in held),"shortfall":2,
               "accepted_by_type":dict(Counter(r["question_type"] for r in accepted)),
               "accepted_MC_gold_counts":dict(Counter(str(r["multi_select"]["target_gold_count_inherited"]) for r in accepted if r["question_type"]=="multi_select")),
               "accepted_relation_coverage":dict(Counter(r["target_realization"]["primary_relation_id"] for r in accepted if r["question_type"]=="multi_select")),
               "accepted_ordering_profiles":dict(Counter(r["target_realization"]["ordering_profile_id"] for r in accepted if r["question_type"]=="ordering"))}
summary={"schema_version":"DAY3_FIRST_BATCH_SUMMARY_V1","created_at_utc":now,
         "status":"FIRST_BATCH_CONTENT_AND_STRUCTURE_COMPLETE_DAY3_FULL_PRODUCTION_PENDING",
         "stage1_planned_slots":240,"executed_slot_decisions":24,"remaining_unexecuted_slots":216,
         "roots_available":20,"full_history_semantically_admitted_roots":2,"remaining_full_history_reviews":18,
         "authored_questions":20,"accepted_questions":19,"held_questions":1,"shortfall_slots":4,
         "independent_gold_agreement":{"authored":20,"matches":20,"does_not_establish_target_implementation":True},
         "arms":arms,"matched_slot_pairs":len(pairs),"source_dialogue_extensions_created":0,
         "ordering_accepted":12,"ordering_pairs_verified":180,"initial_drafts":20,"wording_clarifications":14,
         "solver_calls":0,"training_runs":0,"actual_provider_cap":None,"actual_tokens":None,
         "feedback_scope":"Prospective B revision from84/120 targeted slots to at most60/120 narrow-ordering slots; actual batch B6 ordering content passes with read-plan-candidate lineage. No MC diagnosed target or within-scene reproduced weakness claim.",
         "causal_claim":"Process pilot only; no feedback utility, model challenge, training benefit, unseen-root or equal actual compute conclusion.",
         "history_semantic_independence":"UNKNOWN; do not count paraphrases or reused source facts as new independent knowledge.",
         "common_evaluation":{"original_dev_roots":15,"original_test_roots":16,"source_hashes_verified":31,"abstract_slots":186,
                              "actual_questions_locked":0,"actual_gold_request_freezes":0,"ineligible_by_explicit_scene_count":13,
                              "status":"SOURCE_AND_ABSTRACT_SLOTS_ONLY_CONTENT_PENDING"},
         "training_pool_frozen":False,"day3_complete":False,
         "next_action":"Use COMMON_INPUT_ADDENDUM_V2; complete full-history source review and frozen-slot production for the remaining18 Train roots; retain relation shortfalls. Separately material-admit and freeze actual common evaluation facts/content/Gold/request before Day4 training.",
         "files":{"accepted_public":binding(D/"exports/content_accepted.public.jsonl"),"accepted_labels_and_records":binding(D/"exports/content_accepted.private.jsonl"),
                  "all_slots":binding(D/"exports/SLOT_LEDGER.json"),"dedup":binding(dedup_path),
                  "execution_freeze":binding(D/"EXECUTION_FREEZE.json")}}
dump(D/"SUMMARY.json",summary)
print(json.dumps({"accepted":19,"held":1,"shortfall":4,"matched_pairs":len(pairs),"summary":str(D/"SUMMARY.json")}))
