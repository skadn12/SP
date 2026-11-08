"""Check final feedback provenance, admitted support, projections, and frozen inputs."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

P = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    checks, failures = [], []

    def check(name, ok, detail=None):
        row = {"check": name, "status": "PASS" if ok else "FAIL"}
        if detail is not None:
            row["detail"] = detail
        checks.append(row)
        if not ok:
            failures.append(row)

    def check_binding(name, binding):
        try:
            check(name, bind(binding["path"]) == binding)
        except (OSError, KeyError) as e:
            check(name, False, str(e))

    selection = read(P / "SELECTION_FREEZE.json")
    audit = read(P / "PRIVATE_AUDIT_RECORDS.json")["records"]
    cards_bundle = read(P / "DUAL_FEEDBACK_CARDS_PRIVATE.json")
    cards = cards_bundle["cards"]
    public = read(P / "CHALLENGER_PUBLIC_CARDS.json")["cards"]
    source_projection = read(P / "SOURCE_AUTHOR_EXTENSION_TASKS.json")
    contract = read(P / "REVIEW_CONTRACT.json")
    schema = read(P / "FEEDBACK_TEMPLATE.schema.json")
    interface = read(P / "EXTENSION_TASK_INTERFACE_DRAFT.json")
    summary = read(P / "SUMMARY.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    check("template_schema_valid", True)
    card_audit = {row["feedback_id"]: row for row in cards_bundle["card_audit"]}
    selected_ids = {row["candidate_id"] for row in selection["records"]}
    check("coverage_30_semantic_questions", len(audit) == 30 and {row["candidate_id"] for row in audit} == selected_ids)
    check("coverage_168_items", sum(len(row["author_case"]["item_checks"]) for row in audit) == 168)
    check("unique_19_cards", len(cards) == len({row["feedback_id"] for row in cards}) == 19)
    check("public_same_19_ids", {row["feedback_id"] for row in cards} == {row["feedback_id"] for row in public})
    by_public = {row["feedback_id"]: row for row in public}
    observations, observation_cases = {}, {}
    for case in audit:
        cid = case["candidate_id"]
        root = case["root_id"]
        content = read(P / "inputs" / (root + "_CONTENT_FIRST.json"))
        private = read(P / "inputs" / (root + "_SCORING_AND_RESPONSES_PRIVATE.json"))
        origin = next(row for row in private["records"] if row["candidate_id"] == cid)
        question = next(row for row in content["questions"] if row["candidate_id"] == cid)
        turns = {row["turn_ref"]: row for row in content["source"]["turns"]}
        check(cid + ":unaltered_question_raw_scoring", case["question"] == question and case["original_final_text"] == origin["final_text"] and case["frozen_scoring"] == origin["scoring"])
        check(cid + ":source_gold_content_pass", case["author_case"]["content_review"]["frozen_gold_matches"] is True and case["author_case"]["content_review"]["independently_derived_display_gold"] == origin["scoring"]["display_gold"])
        check(cid + ":single_fixed_selection", case["solver_selection"]["thinking"] is False and case["solver_selection"]["condition"] == "FULL" and case["solver_selection"]["display"] == "perm_00" and case["solver_selection"]["rollout_count"] == 1 and case["solver_selection"]["model"] == private["solver_model"])
        for key in ("source", "raw_response", "prompt", "schema", "content_input", "response_input"):
            check_binding(cid + ":" + key + "_binding", case["bindings"][key])
        check(cid + ":final_text_hash", hashlib.sha256(case["original_final_text"].encode()).hexdigest() == case["bindings"]["final_text_sha256"])
        check(cid + ":all_display_labels", {x["label"] for x in case["author_case"]["item_checks"]} == set(question["display"]))
        for item in case["author_case"]["item_checks"]:
            check(cid + ":item_" + item["label"] + "_source_quotes", all(x["turn_ref"] in turns and x["quote"] in turns[x["turn_ref"]]["text"] for x in item["source_quotes"]))
        for row in case["effective_observations"]:
            err = row["observation"]
            eid = err["error_id"]
            observations[eid] = row
            observation_cases[eid] = cid
            check(eid + ":exact_solver_quote", err["solver_quote"] in case["original_final_text"])
            check(eid + ":exact_source_quote", all(e["turn_ref"] in turns and e["quote"] in turns[e["turn_ref"]]["text"] for e in err["source_evidence"]))
            check(eid + ":admission_status", row["admitted_to_candidate_support"] == (row["review_status"] == "PASS"))
        for overlay in case["item_dimension_overlays"]:
            check(cid + ":overlay_" + overlay["label"] + "_binding", overlay["solver_quote"] in case["original_final_text"] and all(e["quote"] in turns[e["turn_ref"]]["text"] for e in overlay["source_evidence"]))
            check(cid + ":overlay_reason_true_judgment_false", overlay["dimension_overlay"]["solver_reason_content_supported"] is True and overlay["dimension_overlay"]["solver_judgment_correct"] is False and overlay["dimension_overlay"]["new_error_added"] is False)
    counts = Counter(row["review_status"] for row in observations.values())
    check("observations_53_pass_3_unknown", len(observations) == 56 and counts == {"PASS": 53, "UNKNOWN": 3})
    check("22_answer_error_31_reason_units", Counter(row["observation"]["status"] for row in observations.values() if row["review_status"] == "PASS") == {"OBSERVED_ERROR": 22, "OBSERVED_REASON_ERROR": 31})
    check("frozen_final_16_correct_14_wrong", Counter(row["frozen_scoring"]["exact"] for row in audit) == {True: 16, False: 14})
    check("gold_count_control_2_to_6", Counter(row["frozen_scoring"]["gold_count"] for row in audit if row["question"]["question_type"] == "multi_select") == {2: 4, 3: 4, 4: 4, 5: 4, 6: 4})
    revised = []
    for card in cards:
        fid, pc = card["feedback_id"], by_public[card["feedback_id"]]
        errors = list(validator.iter_errors(card))
        check(fid + ":schema", not errors, [e.message for e in errors] if errors else None)
        check(fid + ":only_PASS_support", bool(card["supporting_error_ids"]) and all(observations[eid]["review_status"] == "PASS" for eid in card["supporting_error_ids"]))
        check(fid + ":support_cases_exact", set(card["supporting_candidate_ids"]) == {observation_cases[eid] for eid in card["supporting_error_ids"]})
        check(fid + ":candidate_only", card["status"] == "HYPOTHESIS" and card_audit[fid]["maturity"] == "HYPOTHESIS" and not card_audit[fid]["formal_weakness_admitted"])
        ext = card["extension_task"]
        check(fid + ":no_fabricated_source_shortfall", ext["needed"] is False and ext["material_targets"] == [] and ext["source_shortfall_issue_ids"] == [])
        check(fid + ":public_allowed_fields", set(pc).issubset(contract["public_fields"]))
        check(fid + ":public_private_semantics_match", all(pc[k] == v for k, v in card["solver_feedback"].items()) and all(pc[k] == card[k] for k in ("questioner_feedback", "question_task", "extension_task", "limitations")))
        txt = json.dumps(pc, ensure_ascii=False)
        check(fid + ":no_original_private_data", all(x not in txt for x in ("PDF1-", "<box>", "display_gold", "final_answer", "supporting_error_ids", "display_prediction")) and all(eid not in txt for eid in observations))
        for key in ("effective_card_binding", "effective_public_binding", "review_binding"):
            check_binding(fid + ":audit_" + key, card_audit[fid][key])
        if card_audit[fid]["effective_version"] == "review_revision_v2":
            revised.append(fid)
            check(fid + ":original_revise_retained", card_audit[fid]["initial_decision"] == "REVISE")
    check("four_explicit_revisions", len(revised) == 4)
    check("theme_index_19_cards_8_descriptive_themes", len(read(P / "MECHANISM_THEME_INDEX.json")["themes"]) == 8 and len({x for t in read(P / "MECHANISM_THEME_INDEX.json")["themes"].values() for x in t["card_ids"]}) == 19)
    whitelist = interface["role_projections"]["material_author"]["allowed_fields"]
    forbidden = interface["role_projections"]["material_author"]["forbidden_fields"]
    for task in source_projection["tasks"]:
        check(task["task_id"] + ":source_author_allowed_fields", set(task).issubset(whitelist))
        check(task["task_id"] + ":neutral_projection", not (set(task) & set(forbidden)) and "PDF1-" not in json.dumps(task, ensure_ascii=False) and "<box>" not in json.dumps(task, ensure_ascii=False))
        check(task["task_id"] + ":draft_test_not_shortfall", task["purpose"] == "TEST_HYPOTHESIS" and task["scene_sequence"] == ["A1", "A2", "A3", "B1", "B2", "B3"])
    check("no_actual_consumption", read(P / "CHALLENGER_PUBLIC_CARDS.json")["consumed"] is False and source_projection["consumed"] is False)
    check("interface_remains_design_only", interface["legacy_pipeline_modified"] is False and interface["new_experiments"] is False and "NOT_IMPLEMENTED" in interface["status"])
    check("no_new_solver_or_training", summary["new_execution"] == {"solver_calls": 0, "extensions": 0, "question_generation": 0, "training_runs": 0, "challenger_consumption": False, "runtime_modified": False})
    qf = read(P / "QUESTIONER_FEEDBACK.json")
    check("no_confirmed_question_gold_or_material_defects", not qf["confirmed_question_or_gold_defects"] and not qf["confirmed_source_shortfalls"])
    case = next(r for r in audit if r["candidate_id"] == "PDF1-229686-MC3")
    check("original_confirmed_finding_preserved_but_effectively_demoted", case["author_case"]["questioner_findings"][0]["status"] == "CONFIRMED" and case["effective_questioner_findings"][0]["status"] == "PROSPECTIVE_CONTROL_SUGGESTION")
    for binding in read(P / "AUTHORING_MECHANICAL_CHECK.json")["core_bindings"]:
        check_binding("original_author_core_unchanged:" + binding["path"], binding)
    source_manifest = P.parent / "thinking_switch_20261024_structured_v2" / "FINAL_MANIFEST.json"
    old = read(source_manifest)
    old_failures = []
    for group in ("artifacts", "dependencies"):
        for b in old[group]:
            if bind(b["path"]) != b:
                old_failures.append(b["path"])
    check("old_experiment_424_artifacts_13_dependencies_unchanged", len(old["artifacts"]) == 424 and len(old["dependencies"]) == 13 and not old_failures, old_failures)
    old_result = {"status": "PASS" if not old_failures else "FAIL", "manifest_binding": bind(source_manifest), "artifacts_checked": len(old["artifacts"]), "dependencies_checked": len(old["dependencies"]), "errors": old_failures, "scope": "只读重算旧实验列举文件hash/bytes；不代表语义或因果保证。"}
    (P / "SOURCE_EXPERIMENT_INTEGRITY_CHECK.json").write_text(json.dumps(old_result, ensure_ascii=False, indent=2) + "\n")
    result = {"status": "PASS" if not failures else "FAIL", "created_at_utc": datetime.now(timezone.utc).isoformat(), "checks_count": len(checks), "checks": checks, "failures": failures,
              "summary_binding": bind(P / "SUMMARY.json"), "scope": "最终聚合完整性、原始证据精确性、经审核PASS支持、投影与既有来源不变；语义审核依赖绑定人工式Codex复核，不证明内部思维或反馈收益。"}
    (P / "DELIVERY_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_count", "failures")}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
