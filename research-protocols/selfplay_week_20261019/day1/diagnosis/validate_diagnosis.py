"""Schema + semantic admission checks. Self-test fixtures are synthetic, not evidence."""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

BASE = Path(__file__).resolve().parent

def read(name):
    return json.loads((BASE / name).read_text())

SCHEMA = read("MEMORY_SCHEMA.json")
FACT_SCHEMA = read("FACT_MEMORY_SCHEMA.json")

def walk(value):
    if isinstance(value, dict):
        yield value
        for x in value.values(): yield from walk(x)
    elif isinstance(value, list):
        for x in value: yield from walk(x)

def validate(doc, allowed_roots=None):
    schema = FACT_SCHEMA if doc.get("document_type") == "prefix_fact_memory" else SCHEMA
    errors = [f"schema: {e.json_path}: {e.message}" for e in Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(doc)]
    if errors: return errors
    for value in walk(doc):
        if "turn_start" in value and value["turn_end"] < value["turn_start"]:
            errors.append("evidence turn_end precedes turn_start")
        if value.get("namespace") == "train.feedback" and "root_id" in value:
            if allowed_roots is not None and value["root_id"] not in allowed_roots:
                errors.append("provenance root not in allowed Train root manifest")
    kind = doc["document_type"]
    if kind == "solver_diagnostic_trace":
        if doc["actual_output_tokens"] > doc["output_budget_tokens"]:
            errors.append("diagnostic output exceeds declared token budget")
        ids = [x["content_id"] for x in doc["per_item"]]
        if len(ids) != len(set(ids)): errors.append("duplicate per_item content ID")
        if not set(doc["final_prediction_content_ids"]).issubset(ids): errors.append("prediction contains unknown item")
        if doc["task_type"] == "ordering" and set(doc["final_prediction_content_ids"]) != set(ids):
            errors.append("ordering prediction is not a complete permutation")
    elif kind == "postgold_diagnostic_evaluation":
        for item in doc["per_item"]:
            expected = {(True,True):"TP", (False,True):"FP", (True,False):"FN", (False,False):"TN"}[(item["gold_selected"], item["predicted_selected"])]
            if item["classification"] != expected: errors.append("wrong TP/FP/FN/TN classification")
        if doc["task_type"] == "ordering" and doc["per_item"]: errors.append("ordering uses order_pairs, not binary item labels")
    elif kind == "challenger_weakness_memory":
        cases = {x["failure_id"]:x for x in doc["failure_cases"]}
        probes = {x["probe_id"]:x for x in doc["probes"]}
        if len(cases) != len(doc["failure_cases"]): errors.append("duplicate failure_id")
        if len(probes) != len(doc["probes"]): errors.append("duplicate probe_id")
        cause_ids = [c["cause_id"] for c in doc["causes"]]
        if len(cause_ids) != len(set(cause_ids)): errors.append("duplicate cause_id")
        for case in cases.values():
            r = case["repeat_aggregation"]
            if r["content_failures"] > r["valid_calls"] or r["failed_display_mappings"] > r["total_display_mappings"]:
                errors.append("failure count exceeds denominator")
            if case["model_id"] != doc["model_id"]: errors.append("cross-model failure used in current snapshot")
        for cause in doc["causes"]:
            if cause["model_version"] != doc["model_id"]: errors.append("cause model differs from snapshot")
            missing = set(cause["supporting_failure_ids"]) - cases.keys()
            if missing:
                errors.append("cause cites missing failure_id"); continue
            selected = [cases[i] for i in cause["supporting_failure_ids"]]
            roots = {c["provenance"]["root_id"] for c in selected}
            families = {c["provenance"]["root_id"]+"::"+c["provenance"]["family_id"] for c in selected}
            if roots != set(cause["supporting_root_ids"]): errors.append("supporting_root_ids not derived from failures")
            if families != set(cause["supporting_family_keys"]): errors.append("supporting_family_keys not derived from failures")
            if set(cause["probe_ids"]) - probes.keys(): errors.append("cause cites missing probe_id")
            reproduced = cause["confidence_status"] in {"reproduced", "improving", "recurred"}
            supporting_probes = [probes[x] for x in cause["probe_ids"] if x in probes and probes[x]["outcome"] == "supports_hypothesis" and probes[x]["compared_model_id"] == doc["model_id"]]
            if reproduced and (len(roots) < 3 or len(families) < 3 or not supporting_probes): errors.append("reproduced requires 3 distinct roots/families and supporting same-model probe")
            if cause["state_history"][-1]["to"] != cause["confidence_status"]: errors.append("latest state does not match confidence_status")
            for attempt in cause["attempts"]:
                if attempt["candidate_count"] != sum(attempt[k] for k in ["admitted_count", "rejected_count", "held_count"]): errors.append("candidate ledger counts do not sum")
            for result in cause["new_question_results"]:
                key = result["provenance"]["root_id"]+"::"+result["provenance"]["family_id"]
                if key in families: errors.append("improvement probe reuses supporting failure family")
    elif kind == "prefix_fact_memory":
        if allowed_roots is not None and doc["root_id"] not in allowed_roots: errors.append("fact root not in allowed Train roots")
        facts = {f["fact_id"]:f for f in doc["facts"]}
        if len(facts) != len(doc["facts"]): errors.append("duplicate fact_id")
        for fact in doc["facts"]:
            if fact["valid_from_turn"] > doc["prefix_last_turn"]: errors.append("fact starts after accepted prefix")
            end = fact["valid_until_turn"]
            if end is not None and (end > doc["prefix_last_turn"] or end < fact["valid_from_turn"]): errors.append("fact valid_until outside legal prefix interval")
            for evidence in fact["evidence_refs"]:
                if evidence["source_id"] != doc["source_id"] or evidence["source_sha256"] != doc["source_sha256"]: errors.append("fact cites another source/prefix")
                if evidence["turn_end"] > doc["prefix_last_turn"]: errors.append("fact cites future turn")
            for old_id in fact["supersedes_fact_ids"]:
                if old_id not in facts or old_id == fact["fact_id"]: errors.append("missing/self superseded fact")
                elif facts[old_id]["valid_from_turn"] > fact["valid_from_turn"]: errors.append("supersedes a future fact")
                elif facts[old_id]["validity"] not in {"superseded", "cancelled"}: errors.append("superseded predecessor still marked active/uncertain")
        def cycle(node, path):
            if node in path: return True
            return any(cycle(x, path|{node}) for x in facts[node]["supersedes_fact_ids"] if x in facts)
        if any(cycle(x, set()) for x in facts): errors.append("supersedes cycle")
    return errors

def fixture():
    h = "0"*64
    artifact = {"path":"synthetic/fixture.json", "sha256":h}
    evidence = {"source_id":"synthetic-source", "source_sha256":h, "scene_id":"s1", "turn_start":0, "turn_end":0, "speaker":"Synthetic", "quote":"Synthetic fixture; no actual model data."}
    provenance = lambda i: {"split":"train", "namespace":"train.feedback", "root_id":f"r{i}", "family_id":f"f{i}", "source_id":"synthetic-source", "source_sha256":h}
    case = lambda i: {"failure_id":f"failure{i}", "question_id":f"q{i}", "model_id":"S0", "provenance":provenance(i), "content_gate":{"decision":"PASS", "score_used":False, "completed_before_solver_scoring":True, "record":artifact}, "format_valid":True, "full_evidence_visible":True, "observed_errors":["obs.mc.false_positive"], "evidence_refs":[evidence], "direct_answer_record":artifact, "diagnostic_trace":None, "postgold_evaluation":artifact, "repeat_aggregation":{"protocol":artifact,"valid_calls":18,"content_failures":3,"failed_display_mappings":1,"total_display_mappings":6}}
    doc = {"document_type":"challenger_weakness_memory", "schema_version":"1.0", "snapshot_id":"synthetic-memory", "created_at":"2026-10-19T00:00:00Z", "model_id":"S0", "namespace":"train.feedback", "audience":"challenger", "taxonomy":artifact, "allowed_train_roots_manifest":artifact, "parent_snapshot":None, "frequency_unit":"unique(root_id,family_id)", "reproduction_min_distinct_roots":3, "failure_cases":[case(i) for i in range(3)], "probes":[{"probe_id":"probe0", "type":"evidence_hint", "provenance":provenance(0),"compared_model_id":"S0","frozen_protocol":artifact,"record":artifact,"outcome":"supports_hypothesis","confounds":["Synthetic test only"]}], "causes":[{"cause_id":"cause0", "mechanism_tag":"hyp.evidence.retrieval", "mechanism":"Synthetic retrieval hypothesis", "confidence_status":"reproduced", "model_version":"S0", "supporting_failure_ids":[f"failure{i}" for i in range(3)], "supporting_root_ids":[f"r{i}" for i in range(3)], "supporting_family_keys":[f"r{i}::f{i}" for i in range(3)], "evidence_refs":[evidence], "competing_explanations":[{"explanation":"Synthetic competing mechanism", "status":"unresolved", "probe_ids":[], "interpretation_limit":"No causal claim"}], "probe_ids":["probe0"], "targeted_designs":[{"design_id":"d0","target_relation":"synthetic relation","needed_evidence":["synthetic"],"distractor_or_order_mechanism":"synthetic","new_fact_or_root_requirement":"different fact","ordering_definition":"not_applicable","avoid_shortcuts":["direct quotes"],"content_gate_before_score":True}], "attempts":[], "new_question_results":[], "state_history":[{"at":"2026-10-19T00:00:00Z","model_id":"S0","from":"hypothesis","to":"reproduced","reason":"synthetic fixture","record":artifact}], "limitations":["Synthetic fixture; not an experimental finding"]}], "eval_error_feedback_included":False, "known_limits":["Synthetic fixture"]}
    fact = {"document_type":"prefix_fact_memory", "schema_version":"1.0", "snapshot_id":"synthetic-facts", "created_at":"2026-10-19T00:00:00Z", "root_id":"r0", "namespace":"train.expansion.fact_memory", "audience":"expansion_planner_realizer", "source_id":"synthetic-source", "source_sha256":h, "accepted_prefix":artifact, "prefix_last_turn":2, "prefix_content_gate":artifact, "gold_or_question_conditioned":False, "future_turns_read":False, "parent_snapshot":None, "facts":[{"fact_id":"fact0","fact_type":"state","subject_ids":["p0"],"claim":"synthetic state","validity":"active","valid_from_turn":0,"valid_until_turn":None,"supersedes_fact_ids":[],"evidence_refs":[evidence],"uncertainty":""}], "known_limits":["Synthetic fixture"]}
    return doc, fact

def self_test():
    tests=[]
    for filename in ["WEAKNESS_TAXONOMY.json","MEMORY_SCHEMA.json","FACT_MEMORY_SCHEMA.json"]:
        Draft202012Validator.check_schema(read(filename)); tests.append({"name":"schema_valid:"+filename,"passed":True})
    base, fact = fixture()
    def check(name, doc, expect_valid):
        errors = validate(doc, {"r0","r1","r2"})
        tests.append({"name":name,"passed":bool(not errors)==expect_valid,"expected_valid":expect_valid,"error_count":len(errors)})
    check("three_root_reproduced_valid",base,True)
    check("prefix_fact_valid",fact,True)
    a = base["taxonomy"]
    evidence = base["failure_cases"][0]["evidence_refs"][0]
    trace={"document_type":"solver_diagnostic_trace","schema_version":"1.0","trace_id":"synthetic-trace","request_id":"synthetic-request","created_at":"2026-10-19T00:00:00Z","model_id":"S0","provenance":base["failure_cases"][0]["provenance"],"question_id":"q0","task_type":"multi_select","public_question":a,"source_input":a,"display_mapping":a,"protocol":a,"gold_visible":False,"author_evidence_visible":False,"output_budget_tokens":256,"actual_output_tokens":32,"per_item":[{"content_id":"i0","judgment":"supported","evidence_refs":[evidence],"brief_basis":"Synthetic fixture"}],"order_pairs":[],"final_prediction_content_ids":["i0"],"format_valid":True,"raw_response":a}
    check("pregold_solver_trace_valid",trace,True)
    d=copy.deepcopy(trace); d["gold_visible"]=True; check("pregold_trace_cannot_see_gold",d,False)
    d=copy.deepcopy(trace); d["author_evidence_visible"]=True; check("pregold_trace_cannot_see_author_evidence",d,False)
    d=copy.deepcopy(trace); d["actual_output_tokens"]=257; check("diagnostic_token_budget_enforced",d,False)
    post={"document_type":"postgold_diagnostic_evaluation","schema_version":"1.0","evaluation_id":"synthetic-eval","created_at":"2026-10-19T00:00:01Z","evaluator_id":"synthetic-reviewer","model_id":"S0","provenance":base["failure_cases"][0]["provenance"],"question_id":"q0","task_type":"multi_select","content_gate":base["failure_cases"][0]["content_gate"],"frozen_solver_trace":a,"solver_trace_committed_before_gold_read":True,"gold_reference":a,"direct_answer_record":a,"per_item":[{"content_id":"i0","gold_selected":False,"predicted_selected":True,"classification":"FP","reference_evidence":[evidence]}],"order_pairs":[],"observed_errors":["obs.mc.false_positive"],"cause_hypotheses":[],"competing_explanations":[],"explanation_faithfulness_assumed":False,"limitations":["Synthetic fixture"]}
    check("postgold_evaluation_valid",post,True)
    d=copy.deepcopy(post); d["per_item"][0]["classification"]="FN"; check("postgold_false_classification_rejected",d,False)
    mutations = [
        ("dev_provenance_rejected", lambda d:d["failure_cases"][0]["provenance"].update(split="dev")),
        ("unknown_train_root_rejected",lambda d:d["failure_cases"][0]["provenance"].update(root_id="test-root")),
        ("repeated_calls_do_not_create_roots",lambda d:[x["provenance"].update(root_id="r0") for x in d["failure_cases"]]),
        ("invented_family_keys_rejected",lambda d:d["causes"][0]["supporting_family_keys"].append("r99::f99")),
        ("non_supporting_probe_rejected",lambda d:d["probes"][0].update(outcome="inconclusive")),
        ("format_failure_not_weakness",lambda d:d["failure_cases"][0].update(format_valid=False)),
        ("score_driven_content_gate_rejected",lambda d:d["failure_cases"][0]["content_gate"].update(score_used=True)),
        ("cross_model_failure_rejected",lambda d:d["failure_cases"][0].update(model_id="old-model")),
        ("missing_failure_reference_rejected",lambda d:d["causes"][0]["supporting_failure_ids"].append("missing")),
        ("improving_without_new_probe_rejected",lambda d:d["causes"][0].update(confidence_status="improving")),
    ]
    for name, mutate in mutations:
        d=copy.deepcopy(base); mutate(d); check(name,d,False)
    d=copy.deepcopy(base); d["failure_cases"]=[]; d["probes"]=[]; d["causes"]=[]; check("empty_snapshot_allowed_not_fake_discovery",d,True)
    for name, mutate in [
        ("future_fact_evidence_rejected",lambda d:d["facts"][0]["evidence_refs"][0].update(turn_end=3)),
        ("gold_conditioned_fact_rejected",lambda d:d.update(gold_or_question_conditioned=True)),
        ("self_supersedes_rejected",lambda d:d["facts"][0]["supersedes_fact_ids"].append("fact0")),
        ("wrong_source_hash_rejected",lambda d:d["facts"][0]["evidence_refs"][0].update(source_sha256="1"*64)),
    ]:
        d=copy.deepcopy(fact); mutate(d); check(name,d,False)
    tax = Draft202012Validator(read("WEAKNESS_TAXONOMY.json"))
    tests.append({"name":"label_ambiguity_is_exclusion_not_mechanism","passed":tax.is_valid({"kind":"excluded","tag_id":"excluded.ambiguous_item","note":"Synthetic"}) and not tax.is_valid({"kind":"mechanism_hypothesis","tag_id":"excluded.ambiguous_item","note":"Synthetic"})})
    report={"artifact_type":"schema_boundary_validation_not_experiment_results","status":"PASS" if all(x["passed"] for x in tests) else "FAIL","checked_at_utc":datetime.now(timezone.utc).isoformat(),"synthetic_fixtures_only":True,"model_calls":0,"training_runs":0,"test_count":len(tests),"passed_count":sum(x["passed"] for x in tests),"checks":tests,"files":{name:{"sha256":hashlib.sha256((BASE/name).read_bytes()).hexdigest(),"bytes":(BASE/name).stat().st_size} for name in ["WEAKNESS_TAXONOMY.json","MEMORY_SCHEMA.json","FACT_MEMORY_SCHEMA.json","DIAGNOSIS_PROTOCOL_ZH.md","validate_diagnosis.py","build_schemas.py"]},"limits":["Does not prove real source isolation, content correctness, causal attribution or actual weak points.","Reference file byte bindings and collection chronology must be checked by parent experiment manifest."]}
    (BASE/"VALIDATION.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ["status","test_count","passed_count","model_calls","training_runs"]}))
    return report["status"]=="PASS"

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--self-test",action="store_true"); parser.add_argument("--artifact",type=Path); parser.add_argument("--allowed-roots",type=Path); args=parser.parse_args()
    if args.self_test: raise SystemExit(0 if self_test() else 1)
    if not args.artifact or not args.allowed_roots: parser.error("require --self-test or both --artifact and --allowed-roots")
    allowed=set(json.loads(args.allowed_roots.read_text())["root_ids"])
    errors=validate(json.loads(args.artifact.read_text()),allowed)
    print(json.dumps({"status":"FAIL" if errors else "PASS","errors":errors},ensure_ascii=False,indent=2)); raise SystemExit(bool(errors))
