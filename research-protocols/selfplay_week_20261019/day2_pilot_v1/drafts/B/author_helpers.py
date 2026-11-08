"""B author artifact writer; never opens A drafts or scores, never calls Solver."""
import json,hashlib,datetime
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def bind(p):
 p=Path(p).resolve();raw=p.read_bytes();return {"path":str(p),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
def save(p,d):
 with Path(p).open("x") as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write(chr(10))
def log(d):
 with (OUT/"INSTRUCTION_LOG.jsonl").open("a") as f:f.write(json.dumps({"at_utc":now(),**d},ensure_ascii=False)+chr(10))
def packet(root):return json.loads((BASE/"packets/B"/(root+".json")).read_text())
def plan(root,mc,order,targets,refs):
 p=packet(root);s=Path(p["source_binding"]["path"]);src=json.loads(s.read_text());feedback=Path(p["private_feedback"]["path"]);f=json.loads(feedback.read_text())
 assert bind(s)["sha256"]==p["source_binding"]["sha256"]
 assert bind(feedback)["sha256"]==p["private_feedback"]["sha256"]
 d={"schema_version":"day2_B_author_plan_v1","root_id":root,"arm":"B","planned_at_utc":now(),"actor":"day2_pilot_runtime","nonfresh_actor":True,"prior_exposure":"Runtime implementation; protocol; assigned B feedback only. No A draft/new score read.","source_binding":bind(s),"feedback_binding":bind(feedback),"source_semantic_read":f"All {len(src['turns'])} source turns rendered/read without truncation including full R02 and history","cards_read":[c["card_id"] for c in f["cards"]],"feedback_case_lineage":{"MC":["D2FB-56f498e8859613","D2FB-6dc0cdb527ab89"],"ordering":["D2FB-d5bcba101eb8bd","D2FB-b09d9ed6776506"]},"MC_plan":{"candidate_id":p["slots"][0]["candidate_id"],"gold_count":p["slots"][0]["gold_count"],"coarse_format":p["slots"][0]["coarse_format"],"source_material_and_transform":mc,"target_relation":"Single explicit attribute swaps/numeral/location mutations; ATTRIBUTE hypothesis only","target_labels":targets,"case_to_material":"Peg pocket and five-v-ten-pound cases motivate keeping entity+attribute+scope together, changing only stated target attribute; record item FP separately"},"ordering_plan":{"candidate_id":p["slots"][1]["candidate_id"],"coarse_format":p["slots"][1]["coarse_format"],"source_material_and_transform":order,"primary_refs":refs,"target_relation":"Narrow operational cross-scene first textual target occurrence, not real-world chronology","case_to_material":"Assigned fully supported citation pair inversion cases motivate six private literal anchors and pairwise first-occurrence checking","repeats_checked":"Full R02 scope; distinguish plans/requests and explicit target completion or report","structural_diversity_limitation":"Varied predicates/focus do not establish structural genre diversity"},"falsification":"Reject if true counterpart not entailed, false attribute not directly contradicted, or order ambiguous. No score-based replacement.","budget":{"requested_output_cap_tokens_per_question":3072,"backend_enforced":None,"actual_usage_tokens":None,"unknown_not_zero":True,"initial_drafts_per_slot":1,"max_content_clarifications":1},"solver_calls":0}
 save(OUT/(root+".PLAN.json"),d);log({"event":"SOURCE_SEMANTIC_READ_AND_PLAN_FROZEN","root_id":root,"source_binding":bind(s),"feedback_binding":bind(feedback),"bounded_feedback_read_event":"64b2bc precedes this PLAN; full assigned feedback JSON also parsed","plan_binding":bind(OUT/(root+".PLAN.json")),"no_A_or_new_scores_read":True});print("PLAN frozen",root)
def draft(root,mc_question,mc_rows,order_question,order_rows):
 p=packet(root);src=json.loads(Path(p["source_binding"]["path"]).read_text());turns={t["turn_ref"]:t for t in src["turns"]};questions=[]
 for index,rows in enumerate((mc_rows,order_rows)):
  slot=p["slots"][index];field="options" if index==0 else "items";labels=list("ABCDEF") if index==0 else ["I"+str(i) for i in range(1,7)];assert len(rows)==6
  q={k:slot[k] for k in ["slot_id","candidate_id","root_id","arm","question_type"]};q.update(source_path=p["source_binding"]["path"],source_sha256=p["source_binding"]["sha256"],question=mc_question if index==0 else order_question);q[field]={label:row[0] for label,row in zip(labels,rows)};ev={};gold=[]
  for label,row in zip(labels,rows):
   text,truth,ref,quote,reason=row[:5];assert quote in turns[ref]["text"],(root,label,ref,quote);e={"judgment":"supported" if truth else "contradicted","citations":[{"turn_ref":ref,"quote":quote}],"reason_zh":reason}
   if len(row)>5:
    for extra_ref,extra_quote in row[5]:assert extra_quote in turns[extra_ref]["text"];e["citations"].append({"turn_ref":extra_ref,"quote":extra_quote})
   if index==1:e["primary_turn_ref"]=ref
   if truth:gold.append(label)
   ev[label]=e
  if index==1:gold=sorted(labels,key=lambda label:next(i for i,t in enumerate(src["turns"]) if t["turn_ref"]==ev[label]["primary_turn_ref"]))
  assert len(gold)==slot["gold_count"],(root,len(gold),slot["gold_count"])
  q.update(gold=gold,evidence=ev,design={"coarse_format":slot["coarse_format"],"source_material":"See source-bound prior PLAN and each exact evidence span","target_relation":"Single explicit entity-attribute binding mutation" if index==0 else "Cross-scene first textual target occurrence","retrieved_card_ids":["D2-MC-ATTRIBUTE" if index==0 else "D2-ORDER-FIXED-CITATIONS"],"feedback_case_ids":["D2FB-56f498e8859613","D2FB-6dc0cdb527ab89"] if index==0 else ["D2FB-d5bcba101eb8bd","D2FB-b09d9ed6776506"],"target_labels":[label for label,row in zip(labels,rows) if not row[1]] if index==0 else labels,"true_control_labels":[label for label,row in zip(labels,rows) if row[1]] if index==0 else [],"falsification":"If target item correctly rejected/no pair inversion, record non-replication; no replacement. Wrong/ambiguous source support rejects content independently.","shortcuts_checked":["Names only, no guessed gender","No Gold-count disclosure","No scene ordinals/sequence words in items","Independent MC propositions; no duplicate counterparts","Non-monotonic ordering labels; first target occurrence excludes earlier mere intention"],"status":"ATTRIBUTE hypothesis only" if index==0 else "Narrow historical operational direction; new content untested","structural_diversity_limitation":"Predicate variety does not establish genre diversity"});questions.append(q)
 d={"schema_version":"day2_challenger_draft_v1","root_id":root,"arm":"B","draft_version":"initial_1","created_at_utc":now(),"plan_binding":bind(OUT/(root+".PLAN.json")),"questions":questions,"requested_output_cap_tokens_per_question":3072,"backend_cap_enforced":None,"actual_usage_tokens":None,"unknown_not_zero":True,"solver_scores_read":False,"A_drafts_read":False,"nonfresh_actor":True};save(OUT/(root+".DRAFT.json"),d);log({"event":"INITIAL_DRAFT_WRITTEN","root_id":root,"candidate_ids":[q["candidate_id"] for q in questions],"draft_binding":bind(OUT/(root+".DRAFT.json")),"solver_calls":0});print(json.dumps(bind(OUT/(root+".DRAFT.json"))))
