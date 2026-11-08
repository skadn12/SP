"""Lossless Day2 direct-answer export; no causal upgrade and no new inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAY2 = HERE.parent
DAY1 = DAY2.parent / "day1"
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("bound_day2_controller", DAY2 / "diagnosis/controller.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def read(path):
    return json.loads(Path(path).read_text())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def ref(path):
    return controller.reference(path)


def checkref(value):
    p = Path(value["path"])
    need(p.is_file() and controller.sha(p) == value["sha256"], "original artifact binding changed: " + str(p))
    return p


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return ref(path)


def onlyref(value):
    return {k: value[k] for k in ["path", "sha256"]}


def normalized_evidence(private, source, provenance):
    field = "option_evidence" if private["question_type"] == "multi_select" else "item_evidence"
    indices = {t["turn_ref"]: (i, t) for i, t in enumerate(source["turns"])}
    evidence = []
    for item in private[field].values():
        for citation in item.get("citations") or [item.get("primary_citation") or item]:
            i, turn = indices[citation["turn_ref"]]
            need(citation["speaker"] == turn["speaker"] and citation["quote"] in turn["text"], "author/source citation mismatch")
            evidence.append({"source_id": provenance["source_id"], "source_sha256": provenance["source_sha256"], "scene_id": turn["scene_ref"], "turn_start": i, "turn_end": i, "speaker": citation["speaker"], "quote": citation["quote"]})
    unique = {json.dumps(e, sort_keys=True): e for e in evidence}
    return list(unique.values())


def error_tags(rows, question_type):
    tags = []
    if question_type == "multi_select":
        if any(r["metrics"]["fp"] for r in rows): tags.append("obs.mc.false_positive")
        if any(r["metrics"]["fn"] for r in rows): tags.append("obs.mc.false_negative")
        if any(r["metrics"]["selected_count"] > r["metrics"]["gold_count"] for r in rows): tags.append("obs.mc.over_selection")
        if any(r["metrics"]["selected_count"] < r["metrics"]["gold_count"] for r in rows): tags.append("obs.mc.under_selection")
    else:
        for role, tag in [("within_scene", "obs.order.within_scene_pair"), ("across_scene", "obs.order.across_scene_pair")]:
            if any(r["metrics"][role]["correct_pairs"] < r["metrics"][role]["total_pairs"] for r in rows): tags.append(tag)
    need(tags, "selected question has no observable content error")
    return tags


def export(target):
    target = Path(target).resolve()
    need(not target.exists(), "export target already exists; choose a new version")
    release = target.name.rsplit("_", 1)[-1]
    need(re.fullmatch(r"v[0-9]+", release), "export directory must end in _vNNN")
    direct = DAY2 / "solver/direct"
    hint = DAY2 / "probes/evidence_hint"
    manifest = read(direct / "manifest.json")
    scores = read(direct / "SCORES.json")
    scoring = read(direct / "SCORING_STARTED.json")
    freeze = read(checkref(scores["raw_response_freeze_binding"]))
    need(scores["status"] == "COMPLETE" and scores["denominator_calls"] == 582, "direct is not the bound complete 582-call run")
    need(scoring["all_raw_responses_committed_before_gold"] is True, "original scoring chronology is not frozen")
    checkref(scoring["raw_response_freeze_binding"])
    private_path = checkref(manifest["private_answer_binding"])
    private_all = {q["candidate_id"]: q for q in read(private_path)["questions"]}
    public_all = {q["candidate_id"]: q for q in read(direct / "PUBLIC_QUESTIONS.json")["questions"]}
    mappings = read(direct / "private/DISPLAY_MAPPINGS.json")
    admission_path = checkref(manifest["content_admission_binding"])
    admission = read(admission_path)
    need(admission["status"] == "PASS", "content admission is not PASS")
    gates = {q["candidate_id"]: q for q in admission["questions"]}
    selected = read(hint / "PUBLIC_QUESTIONS.json")["questions"]
    need(len(selected) == len({q["root_id"] for q in selected}) == 6, "expected all six distinct-root hint-selected questions")
    need(Counter(q["question_type"] for q in selected) == {"multi_select": 3, "ordering": 3}, "expected 3 multi-select and 3 ordering")
    frozen_files = {Path(f["path"]).resolve(): f for f in freeze["files"]}
    model_binding = ref(DAY1 / "MODEL_BINDINGS.json")
    token_record = next(f for f in read(DAY1 / "MODEL_BINDINGS.json")["models"]["S0_M_clean_merged"]["files"] if Path(f["path"]).name == "tokenizer.json")
    tokenizer_binding = onlyref(token_record)
    request_meta = {r["request_id"]: r for r in manifest["requests"]}
    raw_origins, cases, failure_bindings, observations, hypotheses = [], [], [], [], []
    total_calls, wrong_calls = 0, 0
    created_at = datetime.now(timezone.utc).isoformat()
    target.mkdir(parents=True)
    for selection_index, selected_public in enumerate(selected):
        qid = selected_public["candidate_id"]
        public, private, gate_original = public_all[qid], private_all[qid], gates[qid]
        rows = [r for r in scores["requests"] if r["candidate_id"] == qid]
        expected = 18 if public["question_type"] == "multi_select" else 12
        need(len(rows) == expected and all(r["valid"] for r in rows), "selected direct question must retain all planned valid calls; invalid calls require explicit extended export")
        need(any(not r["exact"] for r in rows), "selected question has no wrong answer")
        source_path = checkref(gate_original["source_binding"])
        source = read(source_path)
        need(controller.sha(source_path) == public["source_sha256"], "source differs from exact public source version")
        # The original ledger family is root|type. ':' is the schema-safe normalization of this same dependence group.
        family_id = public["root_id"] + "::" + public["question_type"]
        provenance = {"split": "train", "namespace": "train.feedback", "root_id": public["root_id"], "family_id": family_id, "source_id": source["source_id"], "source_sha256": public["source_sha256"]}
        unit = target / "questions" / qid
        public_ref = write(unit / "PUBLIC_QUESTION.json", public)
        canonical_ids = mappings["public_to_canonical"][qid]
        gold_ids = [canonical_ids[label] for label in private["gold"]]
        private_ref = write(unit / "PRIVATE_GOLD.json", {"record_type": "private_gold", "question_id": qid, "task_type": public["question_type"], "gold_content_ids": gold_ids})
        evidence_ref = write(unit / "AUTHOR_EVIDENCE.json", {"record_type": "prior_author_source_evidence_export", "question_id": qid, "provenance": provenance, "evidence": private["option_evidence" if public["question_type"] == "multi_select" else "item_evidence"], "origin_private_questions": ref(private_path), "origin_content_review": onlyref(gate_original["content_review_record"]), "is_solver_trace": False, "new_semantic_review_performed": False})
        gate_ref = write(unit / "CONTENT_ADMISSION.json", {"record_type": "content_admission", "question_id": qid, "provenance": provenance, "decision": "PASS", "score_used": False, "completed_at": admission["created_at_utc"], "public_question": public_ref, "source_input": ref(source_path), "gold_reference": private_ref, "review_record": onlyref(gate_original["content_review_record"])})
        runs, evaluated, mapping_hashes, failed_mapping_hashes = [], [], set(), set()
        for row in rows:
            rid = row["request_id"]
            meta = request_meta[rid]
            raw_path = checkref(row["response_binding"])
            need(raw_path in frozen_files, "raw response absent from original pre-Gold freeze")
            checkref(frozen_files[raw_path])
            raw = read(raw_path)
            need(raw["status"] == "RETURNED" and raw["gold_read_by_solver"] is False, "original Solver record is not a returned public-only answer")
            prompt_path = direct / "prompts" / (rid + ".txt")
            prompt = prompt_path.read_text()
            need(hashlib.sha256(prompt.encode()).hexdigest() == meta["prompt_sha256"] == raw["prompt_sha256"], "actual rendered prompt hash mismatch")
            mapping_ref = write(unit / "mappings" / (rid + ".json"), mappings["requests"][rid])
            parsed = re.fullmatch(r"\s*<box>\s*(.*?)\s*</box>\s*", raw["raw_output"], re.DOTALL)
            need(parsed is not None, "valid raw response has no strict answer box")
            raw_labels = [label.strip().upper() for label in parsed.group(1).split(",")]
            raw_prediction = [mappings["requests"][rid]["display_to_content"][label] for label in raw_labels]
            scored_prediction = row["score"]["content_prediction"]
            need((set(raw_prediction) == set(scored_prediction)) if public["question_type"] == "multi_select" else (raw_prediction == scored_prediction), "raw answer and original score prediction disagree")
            mapping_hashes.add(mapping_ref["sha256"])
            if not row["exact"]: failed_mapping_hashes.add(mapping_ref["sha256"])
            request = {"record_type": "solver_request", "request_id": rid, "model_id": "S0_M_clean", "model_binding": model_binding, "question_id": qid, "provenance": provenance, "task_type": public["question_type"], "mode": "direct", "max_output_tokens": meta["max_output_tokens"], "public_question": public_ref, "source_input": ref(source_path), "display_mapping": mapping_ref, "prompt": prompt, "gold_visible": False, "author_evidence_visible": False, "started_at": raw["request_submitted_at_utc"]}
            request_ref = write(unit / "requests" / (rid + ".json"), request)
            response_ref = write(unit / "responses" / (rid + ".json"), {"record_type": "solver_response", "request_id": rid, "model_id": "S0_M_clean", "question_id": qid, "text": raw["raw_output"], "output_token_ids": raw["completion_token_ids"], "committed_at": raw["response_committed_at_utc"], "format_valid": row["valid"], "full_evidence_visible": len(source["turns"]) == meta["source_turns"] and manifest["source_truncation"] is False, "prediction_content_ids": raw_prediction})
            run_ref = write(unit / "runs" / (rid + ".json"), {"record_type": "solver_run", "request": request_ref, "response": response_ref, "tokenizer": tokenizer_binding})
            evaluation_ref = write(unit / "evaluations" / (rid + ".json"), {"record_type": "gold_evaluation", "solver_run": run_ref, "model_id": "S0_M_clean", "question_id": qid, "provenance": provenance, "gold_reference": private_ref, "content_gate": gate_ref, "exact_correct": row["exact"], "gold_first_read_at": scores["gold_first_read_at_utc"], "gold_read_scope": "this_scoring_run", "gold_read_scope_id": str((direct / "SCORING_STARTED.json").resolve()), "completed_at": scores["scored_at_utc"]})
            runs.append(run_ref)
            evaluated.append(evaluation_ref)
            raw_origins.append({"request_id": rid, "original_manifest": ref(direct / "manifest.json"), "original_raw_response": ref(raw_path), "original_prompt": ref(prompt_path), "original_scores": ref(direct / "SCORES.json"), "original_scoring_event": ref(direct / "SCORING_STARTED.json"), "original_preGold_response_freeze": ref(direct / "RAW_RESPONSE_FREEZE.json"), "original_mapping_collection": ref(direct / "private/DISPLAY_MAPPINGS.json"), "exported_request": request_ref, "exported_response": response_ref, "exported_run": run_ref, "exported_evaluation": evaluation_ref, "event_timestamps_copied_not_inferred": True})
        direct_group = write(unit / "DIRECT_ANSWER_GROUP.json", {"record_type": "solver_run_group", "runs": runs})
        evaluation_group = write(unit / "POSTGOLD_EVALUATION_GROUP.json", {"record_type": "evaluation_group", "observations": evaluated})
        tags = error_tags(rows, public["question_type"])
        failure_id = "failure:" + qid
        counts = {"valid_calls": len(rows), "content_failures": sum(not r["exact"] for r in rows), "failed_display_mappings": len(failed_mapping_hashes), "total_display_mappings": len(mapping_hashes)}
        cases.append({"failure_id": failure_id, "question_id": qid, "model_id": "S0_M_clean", "provenance": provenance, "content_gate": {"decision": "PASS", "score_used": False, "completed_before_solver_scoring": True, "record": gate_ref}, "format_valid": True, "full_evidence_visible": True, "observed_errors": tags, "evidence_refs": normalized_evidence(private, source, provenance), "direct_answer_record": direct_group, "diagnostic_trace": None, "postgold_evaluation": evaluation_group, "repeat_aggregation": {"protocol": ref(DAY1 / "PROTOCOL_V1.json"), **counts}})
        failure_bindings.append({"failure_id": failure_id, "direct_group": direct_group})
        observations.append({"question_id": qid, "root_id": public["root_id"], "question_type": public["question_type"], "original_family_id": public["root_id"] + "|" + public["question_type"], "normalized_family_id": family_id, "all_planned_direct_calls_retained": True, "planned_calls": expected, **counts, "exact_calls": sum(r["exact"] for r in rows), "exact_rate": sum(r["exact"] for r in rows) / len(rows), "observed_errors": tags, "author_evidence": evidence_ref, "author_evidence_is_solver_trace": False, "diagnostic_trace": None, "new_semantic_review_performed": False, "within_scene_pair_denominator": sum(r["metrics"].get("within_scene", {}).get("total_pairs", 0) for r in rows), "across_scene_pair_denominator": sum(r["metrics"].get("across_scene", {}).get("total_pairs", 0) for r in rows)})
        hypotheses.append({"question_id": qid, "failure_id": failure_id, "status": "OBSERVED_ERROR_MECHANISM_UNDETERMINED", "candidate_mechanisms": [], "review_required": ["Recheck exact statement/speaker/modality/scope and unique ordering occurrence against bound complete source", "Read the matched hint contrast and assess competing retrieval, relation-binding, order-definition and prompt effects", "Keep evidence-salience failure separate from a proved internal mechanism"], "may_generate_targeted_B_slots": False})
        total_calls += len(rows)
        wrong_calls += counts["content_failures"]
    need(total_calls == 90 and len(cases) == 6, "six-question full-denominator coverage changed")
    snapshot = {"document_type": "challenger_weakness_memory", "schema_version": "1.0", "snapshot_id": "B-S0-observed-six-" + release, "created_at": created_at, "model_id": "S0_M_clean", "namespace": "train.feedback", "audience": "challenger", "taxonomy": ref(DAY1 / "diagnosis/WEAKNESS_TAXONOMY.json"), "allowed_train_roots_manifest": ref(DAY1 / "TRAIN_ROOTS.json"), "parent_snapshot": ref(DAY1 / "WEAKNESS_MEMORY_B_v000.json"), "frequency_unit": "unique(root_id,family_id)", "reproduction_min_distinct_roots": 3, "failure_cases": cases, "probes": [], "causes": [], "eval_error_feedback_included": False, "known_limits": ["Six questions are the pre-frozen hash-selected hint subset conditional on at least one valid direct content failure, not a representative difficulty sample.", "All 90 matching direct calls, correct and wrong, are retained; repeated calls do not create extra roots/families.", "Prior source-grounded content admission is reused; no new semantic reviewer or reviewed Solver CoT is invented.", "diagnostic_trace is null because this export uses direct answers only; author evidence is explicitly not Solver evidence.", "Zero causes and zero reproduced mechanisms; this memory cannot satisfy targeted B slots.", "Gold chronology is scoped to the original scoring process, not the whole parent session."]}
    target_ref = write(target / ("MEMORY_B_S0_OBSERVED_" + release + ".json"), snapshot)
    sidecar_ref = write(target / ("CONTROLLER_B_S0_OBSERVED_" + release + ".json"), {"record_type": "diagnosis_controller_manifest", "schema_version": 1, "target": target_ref, "frozen_protocol": ref(DAY1 / "PROTOCOL_V1.json"), "models": [{"model_id": "S0_M_clean", "canonical_model_id": "S0_M_clean_merged", "parent_model_id": None, "binding": model_binding, "training_record": None}], "failure_bindings": failure_bindings, "probe_bindings": [], "result_bindings": []})
    write(target / "EXPORT_BINDINGS.json", {"record_type": "lossless_export_origin_bindings", "created_at": created_at, "export_tool": ref(Path(__file__)), "target": target_ref, "controller": sidecar_ref, "original_selected_public_questions": ref(hint / "PUBLIC_QUESTIONS.json"), "selection_policy": ref(DAY2 / "PROBE_SELECTION_POLICY.json"), "selection_policy_freeze": ref(DAY2 / "PROBE_POLICY_FREEZE.json"), "original_content_admission": ref(admission_path), "original_private_questions": ref(private_path), "original_public_questions": ref(direct / "PUBLIC_QUESTIONS.json"), "requests": raw_origins, "known_normalizations": ["root|type family dependence group is represented as root::type to satisfy immutable Day1 identifier regex; the original is retained in OBSERVATIONS.json.", "Each original raw response/prompt and collection-level mapping/score remains byte-bound. Normalized request/response/event records are exports, not new historical events.", "MC score predictions use set order; exported content prediction retains original raw box display order and is checked against the original score as a set. Ordering remains an exact sequence.", "content_admission.completed_at copies the actual pre-inference CONTENT_ADMISSION package timestamp and binds its prior semantic-review record; no new semantic pass is claimed."]})
    write(target / "OBSERVATIONS.json", {"record_type": "direct_error_observation_inventory", "status": "OBSERVATIONS_ONLY_NO_CAUSES", "created_at": created_at, "question_count": 6, "root_count": 6, "direct_calls": total_calls, "content_failures": wrong_calls, "probes_published": 0, "causes_published": 0, "questions": observations})
    write(target / "HYPOTHESIS_REVIEW_QUEUE.json", {"record_type": "unadjudicated_mechanism_review_queue", "status": "PENDING_PARENT_SOURCE_AND_MECHANISM_REVIEW", "items": hypotheses, "no_reproduced_cause": True})
    if (hint / "SCORES.json").exists():
        hint_scores = read(hint / "SCORES.json")
        need(hint_scores["status"] == "COMPLETE", "hint scoring incomplete")
        hq = {q["candidate_id"]: q for q in hint_scores["questions"]}
        comparisons = []
        for observation in observations:
            other = hq[observation["question_id"]]
            need(other["denominator_calls"] == observation["planned_calls"] and other["valid_calls"] == observation["valid_calls"], "hint denominator differs")
            comparisons.append({"question_id": observation["question_id"], "root_id": observation["root_id"], "question_type": observation["question_type"], "calls_per_condition": observation["planned_calls"], "baseline_exact": observation["exact_calls"], "hint_exact": other["exact_calls"], "exact_count_change": other["exact_calls"] - observation["exact_calls"]})
        positive = sum(x["exact_count_change"] > 0 for x in comparisons)
        write(target / "HINT_CONTRAST_CONTEXT.json", {"record_type": "unadjudicated_observed_hint_comparison", "status": "NO_REPRODUCED_MECHANISM", "original_hint_scores": ref(hint / "SCORES.json"), "original_matched_pairs": ref(hint / "MATCHED_PAIRS.json"), "original_policy": ref(DAY2 / "PROBE_SELECTION_POLICY.json"), "questions": comparisons, "strict_positive_distinct_roots": positive, "frozen_min_positive_roots": 3, "threshold_met": positive >= 3, "semantic_mechanism_review_performed": False, "published_as_memory_probe": False, "limits": ["Hint uses prior reviewed source evidence; it is an evidence-guided intervention, not Gold-blind benchmark.", "This reference context is not a reviewed supporting controlled-probe record and is not published in snapshot.probes.", "No training happened and improving state is prohibited for this contrast."]})
    report = controller.validate_file(Path(target_ref["path"]), Path(sidecar_ref["path"]))
    write(target / "CONTROLLER_VALIDATION.json", report)
    print(json.dumps({"status": report["status"], "failure_cases": len(cases), "roots": 6, "direct_calls": total_calls, "content_failures": wrong_calls, "causes": 0, "probes": 0, "controller": report}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "observed_six_v002")
    args = parser.parse_args()
    try:
        result = export(args.output)
    except (ValueError, KeyError, OSError) as e:
        failure = {"status": "FAIL", "error": str(e), "created_at": datetime.now(timezone.utc).isoformat(), "no_artifacts_removed": True}
        if args.output.exists() and not (args.output / "EXPORT_FAILURE.json").exists(): write(args.output / "EXPORT_FAILURE.json", failure)
        print(json.dumps(failure, ensure_ascii=False))
        raise SystemExit(1)
    raise SystemExit(result["status"] != "PASS")
