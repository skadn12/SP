"""Validate frozen day-3 artifacts; no generation, Solver, or training calls."""
import json
import hashlib
from pathlib import Path
from collections import Counter
from itertools import combinations
import sys
import jsonschema

D = Path(__file__).resolve().parent
W = D.parent
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
errors = []
checks = 0
def check(condition, message):
    global checks
    checks += 1
    if not condition: errors.append(message)
freeze = read(D/"EXECUTION_FREEZE.json")
old = {s["candidate_slot_id"]:s for s in read(W/"day1/SLOT_PLAN.json")["slots"] if s["stage"]==1}
check(len(freeze["slots"])==240, "Expected 240 inherited abstract slots.")
for s in freeze["slots"]:
    o=old[s["candidate_slot_id"]]
    for k in ["slot_id","root_id","arm","question_type","local_slot","target_gold_count","options_or_items"]:
        check(s[k]==o[k], f"Changed inherited slot {s['candidate_slot_id']}:{k}")
    check(sha(Path(s["source_binding"]["path"]))==s["source_binding"]["sha256"], "Source hash changed.")
for arm in ["A","B"]:
    quota=Counter(s["target_gold_count"] for s in freeze["slots"] if s["arm"]==arm and s["question_type"]=="multi_select")
    check(quota=={2:15,3:15,4:15,5:15}, f"{arm}: quota changed")
schema=read(D/"design/PUBLIC_DESIGN_LIBRARY.json")["schemas"]["candidate_record"]
validator=jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
records=sorted((D/"final_records").glob("*.json")) if (D/"final_records").exists() else sorted((D/"records/A").rglob("*.json"))+sorted((D/"records/B").rglob("*.json"))
record_count=0
for p in records:
    r=read(p)
    if "inherited_slot_id" not in r or "generation_history" not in r:
        continue
    record_count+=1
    for err in validator.iter_errors(r):
        check(False, f"{p.name} schema {list(err.path)}: {err.message}")
    check(r["inherited_slot_id"] in {s["slot_id"] for s in freeze["slots"]},"Unknown inherited slot")
    check(r["public_library_sha256"]==sha(D/"design/PUBLIC_DESIGN_LIBRARY.json"),"Public design hash drift")
    check(r["source_review"]["status"]=="PASS","Source admission incomplete")
    b=r["budget"]
    check(b["initial_used"]==1 and b["content_clarification_used"]<=1,"Generation budget exceeded")
    check(b["solver_calls"]==0,"Unexpected Solver call")
    refs={e["ref_id"]:e for e in r["evidence_span"]["evidence_refs"]}
    src=read(Path(r["source_binding"]["source_path"]))
    turns={t["turn_ref"]:(i,t) for i,t in enumerate(src["turns"])}
    for e in refs.values():
        i,t=turns[e["turn_id"]]
        pos=e["source_position"]
        check(i==pos["global_turn_index"],"Zero-based global index mismatch")
        check(t["text"][pos["span_start"]:pos["span_end"]]==e["quote"],"Evidence span quote mismatch")
    for h in r["generation_history"]:
        check(h["actual_output_tokens"] is None and h["actual_cap_support"]=="UNKNOWN","Unsupported provider metadata claim")
        check(h["request_id"] is None,"Interactive generation request ID invented")
        if h["raw_response_path"]:
            check(sha(Path(h["raw_response_path"]))==h["candidate_content_sha256"],"Author artifact changed")
    if r["question_type"]=="multi_select":
        m=r["multi_select"]
        check(len(m["options"])==6,"MC not six options")
        gold=[o["option_id"] for o in m["options"] if o["source_verdict"]=="SUPPORTED"]
        check(set(gold)==set(m["gold_option_ids"]),"MC truth and Gold disagree")
        check(len(gold)==m["target_gold_count_inherited"],"MC Gold quota differs")
    else:
        o=r["ordering"];items=o["items"]
        anchors={x["item_id"]:refs[x["selected_ref_id"]] for x in items}
        positions={k:(e["source_position"]["global_turn_index"],e["source_position"]["span_start"]) for k,e in anchors.items()}
        check(sorted(positions,key=positions.get)==o["gold_item_ids"],"Ordering Gold not source positions")
        check(len(set(positions.values()))==6,"Ordering position collision")
        counts=Counter(e["scene_id"] for e in anchors.values())
        check(sorted(counts.values(),reverse=True)==o["scene_event_counts"],"Ordering scene profile differs")
        within=sum(anchors[a]["scene_id"]==anchors[b]["scene_id"] for a,b in combinations(anchors,2))
        check(within==o["within_scene_pair_count"],"Within-scene count differs")
        for item in items:
            occurrences=item["all_occurrences"]
            check(all(x["ref_id"] in refs for x in occurrences),"Occurrence reference unresolved")
            inc=[refs[x["ref_id"]] for x in occurrences if x["included_for_event_key"]]
            if item["mention_policy"]=="FIRST_IN_DECLARED_SCOPE":
                check(item["selected_ref_id"] in {x["ref_id"] for x in occurrences if x["included_for_event_key"]},"Selected occurrence not included")
                check(positions[item["item_id"]]==min((e["source_position"]["global_turn_index"],e["source_position"]["span_start"]) for e in inc),"Not first included semantic occurrence")
    accepted=r["disposition"]["state"] in ["ACCEPT_DIRECT","ACCEPT_AFTER_CLARIFICATION"]
    if accepted:
        check(r["qa_review"]["status"]=="PASS","Accepted without QA PASS")
        check(r["target_realization"]["target_implemented"]=="YES","Accepted without target")
        check(all(c["result"]=="PASS" for c in r["target_realization"]["criteria"]),"Target criterion incomplete")
        check(r["fact_family"]["duplicate_review"]=="PASS","Accepted with pending duplicate review")
        check(r["disposition"]["accepted_content_sha256"] is not None,"Accepted without frozen content")
    if r["arm"]=="A":
        check(r["disposition"]["feedback_binding_location"] is None,"A feedback binding leak")
report={"status":"PASS" if not errors else "FAIL","checks":checks,"records":record_count,"errors":errors,
        "scope":"Schema, immutable source/artifact bytes, quotas, explicit evidence spans and occurrence/pair arithmetic. Source semantic judgements are the signed independent QA records, not proved by this script."}
print(json.dumps(report,ensure_ascii=False,indent=2))
if "--output" in sys.argv:
    p=Path(sys.argv[sys.argv.index("--output")+1])
    with p.open("x") as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write("\n")
sys.exit(bool(errors))
