"""Read-only replay and bounded export/seal of the expansion-only v91 batch.

This tool never calls a model, materializes a source, changes a model verdict,
or writes into the engine/run. A seal covers an exact inventory, including
original failed attempts, evidence, code, logs and exports. Use Python -B.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
HOME = Path(__file__).resolve().parents[1]
RUN_NAME = "ten_roots_B_scene01"
SEAL_NAME = "COMPLETE_MANIFEST.json"
COHORTS = ("batch_other_roots", "prior_planner_failure_regression")
USABLE = {"USABLE", "USABLE_WITH_WARNINGS"}
VERDICTS = USABLE | {"HOLD", "NO_EXPANSION"}
EXPORTS = ("SUCCESSFUL_EXPANSIONS.json", "FAILURE_INDEX.json",
           "COMPLETE_BATCH_RESULTS_ZH.md", "ARTIFACT_INDEX.md", "evidence/EXPORT_AUDIT.json")


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def verify_binding(row, reason="binding_changed"):
    require(isinstance(row, dict) and set(("path", "sha256")) <= set(row),
            reason + ":invalid_binding")
    require(bind(row["path"]) == {"path": row["path"], "sha256": row["sha256"]},
            reason + ":" + str(row["path"]))


def inventory(home, exclude=()):
    home = Path(home).resolve()
    excluded = set(exclude)
    rows = []
    for path in sorted(home.rglob("*")):
        require(not path.is_symlink(), "inventory_symlink_not_allowed:" + str(path))
        if path.is_file():
            rel = path.relative_to(home).as_posix()
            if rel in excluded:
                continue
            data = path.read_bytes()
            rows.append({"path": rel, "sha256": digest(data), "bytes": len(data)})
    return rows


def verify_preservation(path):
    before = read(path)
    expected = before["files"]
    require(len(expected) == before["protected_count"], "protected_inventory_count_invalid")
    expected_paths = [row["path"] for row in expected]
    require(len(set(expected_paths)) == len(expected_paths), "duplicate_protected_path")
    actual = set()
    for directory in before["directories"]:
        folder = Path(directory)
        require(folder.is_dir(), "protected_directory_missing:" + directory)
        actual.update(str(p.resolve()) for p in folder.rglob("*") if p.is_file())
    required = set(expected_paths)
    require(actual == required, "protected_inventory_changed:" + json.dumps({
        "added": sorted(actual - required), "missing": sorted(required - actual)}, ensure_ascii=False))
    for row in expected:
        verify_binding(row, "protected_sha_changed")
    return {"status": "PASS", "protected_count": len(expected),
            "protected_directory_count": len(before["directories"]),
            "exact_file_inventory_verified": True, "all_sha256_verified": True,
            "before": bind(path)}


def verify_seal_inventory(home):
    manifest = read(home / SEAL_NAME)
    require(manifest.get("version") == "v91_exact_delivery_seal_v1", "seal_version_invalid")
    require(manifest.get("status") == "COMPLETE", "delivery_not_complete")
    require(manifest["files"] == inventory(home, (SEAL_NAME,)),
            "sealed_inventory_or_sha_changed")
    require(manifest["file_count"] == len(manifest["files"]), "seal_count_invalid")
    return manifest


def immutable_write(path, data):
    path = Path(path)
    require(not any((parent / SEAL_NAME).exists() for parent in path.parents), "write_after_seal")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, "immutable_export_changed:" + str(path))
        return
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module_unavailable:" + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_engine_audit(home, run, saved):
    """Capture the engine audit's sole write in memory; preserve original audit."""
    engine = home / "expansion_engine"
    sys.path.insert(0, str(engine))
    module = load_module(engine / "audit_run.py", "v91_export_readonly_audit")
    captured = []
    original = module.r.base.write

    def capture(path, value):
        require(Path(path).resolve() == run / "FINAL_AUDIT.json", "unexpected_audit_write:" + str(path))
        captured.append(value)

    module.r.base.write = capture
    try:
        recalculated = module.audit_run(run)
    finally:
        module.r.base.write = original
    require(len(captured) == 1 and captured[0] == recalculated, "audit_write_capture_invalid")
    require(recalculated["status"] == "PASS" and not recalculated["failures"],
            "engine_audit_replay_failed:" + json.dumps(recalculated["failures"], ensure_ascii=False))
    stable = lambda obj: {key: value for key, value in obj.items() if key != "audit_time_utc"}
    require(stable(saved) == stable(recalculated), "saved_final_audit_differs_from_replay")
    return module.r


def quality_records(home, inputs, outputs, results):
    records = defaultdict(list)
    bindings = []
    root_order = list(inputs)
    for shard_number, name in enumerate(("EXPANSION_QUALITY_REVIEW_A.json", "EXPANSION_QUALITY_REVIEW_B.json")):
        path = home / "evidence" / name
        document = read(path)
        require(isinstance(document.get("roots"), list), "quality_roots_missing:" + name)
        local_roots = set()
        bindings.append(bind(path))
        for row in document["roots"]:
            root = row["root_id"]
            require(root in inputs and root not in local_roots, "quality_root_unknown_or_duplicate:" + root)
            local_roots.add(root)
            require(row["verdict"] in VERDICTS, "quality_verdict_invalid:" + root)
            verify_binding(row["input_source"], "quality_input_changed")
            require(row["input_source"]["sha256"] == inputs[root]["sha256"], "quality_wrong_input:" + root)
            expected_output = outputs[root]
            if row.get("output_source") is not None:
                verify_binding(row["output_source"], "quality_output_changed")
                require(row["output_source"]["sha256"] == expected_output["sha256"], "quality_wrong_output:" + root)
            require(row["new_turn_count"] == results[root]["added_turns"], "quality_new_turn_count_changed:" + root)
            accepted = results[root]["accepted_scenes"] > 0
            if accepted:
                require(row.get("output_source") is not None, "quality_accepted_output_unbound:" + root)
                require(row.get("read_complete_source") is True, "quality_complete_source_not_read:" + root)
                require(row["verdict"] != "NO_EXPANSION", "accepted_root_marked_no_expansion:" + root)
            else:
                require(row["verdict"] == "NO_EXPANSION" and row["new_turn_count"] == 0,
                        "unexpanded_root_marked_usable:" + root)
            require(isinstance(row.get("findings"), list), "quality_findings_missing:" + root)
            records[root].append({"review_file": bind(path), "record": row})
        require(local_roots == set(root_order[shard_number * 5:(shard_number + 1) * 5]),
                "quality_shard_not_assigned_five_roots:" + name)
    require(set(records) == set(inputs), "quality_union_does_not_cover_exact_roots")
    require(all(len(rows) == 1 for rows in records.values()), "quality_root_not_covered_exactly_once")
    return records, bindings


def final_quality(rows, accepted):
    if not accepted:
        return "NO_EXPANSION"
    verdicts = {item["record"]["verdict"] for item in rows}
    if "HOLD" in verdicts:
        return "HOLD"
    return "USABLE_WITH_WARNINGS" if "USABLE_WITH_WARNINGS" in verdicts else "USABLE"


def body_tokenizer(manifest):
    """Fast tokenizer only; disable special tokens and all chat wrapping."""
    from tokenizers import Tokenizer
    tokenizer_home = Path("/root/autodl-tmp/Qwen3-4B")
    paths = [tokenizer_home / name for name in
             ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")]
    tokenizer = Tokenizer.from_file(str(tokenizer_home / "tokenizer.json"))
    config_path = tokenizer_home / "config.json"
    config = read(config_path)
    require(config.get("model_type") == "qwen3" and config.get("hidden_size") == 2560,
            "body_tokenizer_not_qwen3_4b")
    return tokenizer, {"model": "Qwen/Qwen3-4B", "definition":
        "sum(len(tokenizer.encode(turn['text'], add_special_tokens=False).ids)) for added turns only; "
        "no speaker names, labels, separators, JSON or chat wrappers",
        "files": [bind(path) for path in paths], "bound_at": "export audit, separately from prepare manifest",
        "prepare_capacity_tokenizer_model": "Qwen/Qwen3-14B",
        "prepare_capacity_checks_certify_qwen3_4b": False,
        "prepare_capacity_files": [bind(path) for path in map(Path, manifest["capacity_tokenizer_paths"])]}


def error_codes(verdict):
    if not isinstance(verdict, dict):
        return []
    codes = [error.get("code") for error in verdict.get("errors", []) if isinstance(error, dict)]
    if verdict.get("reason"):
        codes.append(verdict["reason"])
    return [code for code in codes if code]


def stage_analysis(stages, runtime):
    planner_rows = []
    repairs = Counter()
    groups = Counter()
    rejected = []
    for item in stages:
        request = Path(item["request_directory"])
        proof = read(request / "PROOF.json")
        role = item["role"]
        group = (role, proof.get("root_id"), proof.get("source_id"), proof.get("scene_id"))
        if role == "event_state_extractor":
            group += (proof.get("plan_attempt"),)
        elif role == "scene_reviewer":
            group += (proof.get("realization_sha256"),)
        groups[group] += 1
        if item["status"] != "PASS":
            rejected.append(item)
        if role == "planner":
            parsed = None
            response_path = request / "RESPONSE.json"
            if response_path.exists():
                try:
                    parsed = runtime.parse_object(read(response_path)["text"])
                except (ValueError, TypeError, KeyError):
                    pass
            scenes = parsed.get("scenes", []) if isinstance(parsed, dict) else []
            scenes = scenes if isinstance(scenes, list) else []
            counts = [len(s.get("developments", [])) if isinstance(s, dict)
                      and isinstance(s.get("developments"), list) else None for s in scenes]
            codes = error_codes(item["verdict"])
            planner_rows.append({"request": str(request), "status": item["status"],
                "development_counts": counts, "errors": codes,
                "cardinality_rejected": "scene_developments_count_invalid" in codes})
    for group, count in groups.items():
        repairs[group[0]] += max(0, count - 1)
    count_errors = sum(row["cardinality_rejected"] for row in planner_rows)
    valid_last = bool(planner_rows and planner_rows[-1]["status"] == "PASS"
                      and planner_rows[-1]["development_counts"]
                      and all(type(n) is int and 2 <= n <= 5 for n in planner_rows[-1]["development_counts"]))
    cardinality = ("REPAIRED" if count_errors else "PASS_WITHOUT_COUNT_ERROR") if valid_last else (
        "UNRESOLVED_COUNT_ERROR" if count_errors else "NO_VALID_FINAL_PLAN")
    return {"planner_attempts": planner_rows, "planner_count_error_attempts": count_errors,
            "planner_cardinality_status": cardinality,
            "logical_repeat_requests_by_role": dict(repairs),
            "logical_repeat_requests": sum(repairs.values()), "rejected_stages": rejected}


def collect(home, run):
    require(home == HOME, "exporter_home_must_be_current_v91")
    require(run == home / "expansion_engine" / "runs" / RUN_NAME, "unexpected_run")
    sealed = (home / SEAL_NAME).exists()
    if sealed:
        verify_seal_inventory(home)
    preservation = verify_preservation(home / "PRESERVATION_BEFORE.json")
    require(preservation["protected_count"] == 17651, "v91_protected_count_changed")
    plan = read(home / "PILOT_PLAN.json")
    registry = read(home / "expansion_engine" / "SOURCE_REGISTRY.json")
    manifest = read(run / "MANIFEST.json")
    roots = manifest["roots"]
    require(roots == [row["root_id"] for row in plan["roots"]]
            == [row["root_id"] for row in registry["roots"]], "root_order_or_selection_changed")
    require(len(set(roots)) == len(roots) == 10, "v91_root_count_changed")
    cohorts = {row["root_id"]: row["cohort"] for row in plan["roots"]}
    require(Counter(cohorts.values()) == Counter({COHORTS[0]: 8, COHORTS[1]: 2}), "cohorts_changed")
    require(manifest["root_cohorts"] == cohorts, "manifest_cohorts_changed")
    require(manifest["selected_routes"] == ["dataset_new_event"]
            and manifest["scenes_per_route_target"] == 1, "expansion_scope_changed")
    require(plan["question_generation"] is False and plan["solver_evaluation"] is False
            and plan["training_allowed"] is False and plan["formal_training_admission"] is False,
            "pilot_allows_downstream_activity")
    require(plan["selection_is_random"] is False and plan["all_roots_are_historical_train"] is True
            and plan["semantic_acceptance_rules_changed"] is False, "pilot_limitations_changed")
    require(manifest["generation_calls"] == manifest["solver_calls"] == 0
            and manifest["training_started"] is False, "manifest_downstream_activity")
    for row in manifest["files"]:
        verify_binding(row, "frozen_input_or_runtime_changed")
    require(len({row["path"] for row in manifest["files"]}) == len(manifest["files"]),
            "duplicate_manifest_binding")
    snapshots = read(run / "EXECUTION_CODE_SNAPSHOT.json")["files"]
    frozen_code = {row["path"]: row["sha256"] for row in manifest["files"] if Path(row["path"]).suffix == ".py"}
    require({row["original_path"] for row in snapshots} == set(frozen_code)
            and len(snapshots) == len(frozen_code), "execution_snapshot_inventory_changed")
    for row in snapshots:
        require(row["sha256"] == frozen_code[row["original_path"]]
                == digest(Path(row["snapshot_path"]).read_bytes()), "execution_snapshot_sha_changed")
    report = read(run / "REPORT.json")
    saved_audit = read(run / "FINAL_AUDIT.json")
    require(report["status"] in ("COMPLETE", "COMPLETE_WITH_SHORTFALL"), "run_incomplete")
    require(report["question_calls"] == report["solver_calls"] == 0 and report["training_started"] is False,
            "report_downstream_activity")
    require(saved_audit["status"] == "PASS" and not saved_audit["failures"], "final_audit_not_pass")
    require(saved_audit["semantic_certified"] is False and report["semantic_certified"] is False,
            "engine_audit_falsely_semantic_certified")
    runtime = replay_engine_audit(home, run, saved_audit)
    results = {row["root_id"]: row for row in report["routes"]}
    require(set(results) == set(roots) and len(report["routes"]) == len(roots), "route_report_roots_changed")
    inputs, outputs = {}, {}
    for source in registry["roots"]:
        root = source["root_id"]
        verify_binding({"path": source["source_path"], "sha256": source["source_sha256"]}, "registry_input_changed")
        require(source["cohort"] == cohorts[root] and source["historical_split"] == "train"
                and source["first_round_feedback"]["has_root_specific_feedback"] is True,
                "source_cohort_or_feedback_changed")
        input_path = run / "sources" / (root + ".json")
        require(input_path.read_bytes() == Path(source["source_path"]).read_bytes(), "frozen_source_not_exact_input")
        inputs[root] = bind(input_path)
        outputs[root] = bind(run / "roots" / root / "B" / "cumulative_dialogue.json")
    reviews, review_bindings = quality_records(home, inputs, outputs, results)
    completion_path = home / "evidence/REVIEW_COMPLETION_RECEIPT.json"
    completion = read(completion_path)
    require(completion["quality_A"] == review_bindings[0]
            and completion["quality_B"] == review_bindings[1], "review_completion_sources_changed")
    for key in ("quality_A", "quality_B", "parent_diagnostics"):
        verify_binding(completion[key], "review_completion_binding_changed")
    diagnostics = read(completion["parent_diagnostics"]["path"])
    for item in diagnostics["diagnostics"]:
        for binding in item["evidence"].values():
            verify_binding(binding, "parent_diagnostic_evidence_changed")
    review_protocol = {
        "completion_receipt": bind(completion_path),
        "completion_note": bind(home / "evidence/REVIEW_COMPLETION_NOTE.md"),
        "parent_diagnostics": completion["parent_diagnostics"],
        "next_minfix_diagnosis": bind(home / "evidence/NEXT_MINFIX_DIAGNOSIS_ZH.md"),
        "not_fully_blind_end_to_end": completion["not_fully_blind_end_to_end"],
        "independent_second_final_review_for_every_root": completion["independent_second_final_review_for_every_root"],
        "quality_B_final_report_written_by": completion["quality_B_final_report_written_by"],
    }
    material_index_path = home / "evidence" / "EXPANSION_MATERIALIZATION_COMPLETE.json"
    material = read(material_index_path)
    require(material["status"] == "COMPLETE" and material["report"] == bind(run / "REPORT.json"),
            "materialization_incomplete_or_report_changed")
    material_roots = {row["root_id"]: row for row in material["roots"]}
    require(set(material_roots) == set(roots) and len(material["roots"]) == len(roots), "materialization_roots_changed")
    tokenizer, token_definition = body_tokenizer(manifest)
    cards = read(run / "dataset_cards_result.json")["accepted_cards"]
    by_source = {card["source_id"]: card for card in cards}
    require(len(by_source) == len(cards), "multiple_cards_per_source")
    allocation = read(run / "UNIQUE_EVENT_ALLOCATION.json")
    rows = []
    for root in roots:
        result = results[root]
        folder = run / "roots" / root / "B"
        before = read(inputs[root]["path"])["turns"]
        after = read(outputs[root]["path"])["turns"]
        suffix = after[len(before):]
        require(after[:len(before)] == before and len(suffix) == result["added_turns"], "historical_prefix_or_added_count_changed")
        require(read(folder / "RESULT.json") == result, "root_result_not_reported_result")
        accepted = result["accepted_scenes"] > 0
        require(accepted == bool(suffix), "mechanical_acceptance_without_new_turns")
        short = root.rsplit("-", 1)[-1]
        record_path = home / "evidence" / ("expansion_provenance" if accepted else "expansion_failures") / (
            "root" + short + ("_PROVENANCE.json" if accepted else "_FAILURE.json"))
        record = read(record_path)
        require(record["root_id"] == root and record["input_source"] == inputs[root]
                and record["accepted_output"] == outputs[root], "materialized_provenance_source_changed")
        for item in record["bound_files"]:
            verify_binding(item, "materialization_binding_changed")
        require(record["new_turn_count"] == len(suffix) and record["historical_prefix_preserved"] is True,
                "materialized_added_turn_count_changed")
        require(material_roots[root]["new_turn_count"] == len(suffix)
                and material_roots[root]["source"] == record["source"], "materialization_index_changed")
        if accepted:
            verify_binding(record["source"], "materialized_source_changed")
            require(record["source"]["sha256"] == outputs[root]["sha256"], "materialized_source_not_exact_output")
        else:
            require(record["source"] is None and not suffix, "failure_materialized_source")
        stage_path = Path(record["stage_index"]["path"])
        require(material_roots[root]["stage_index"] == bind(stage_path), "materialized_stage_index_changed")
        stage = read(stage_path)
        analysis = stage_analysis(stage["stages"], runtime)
        source_ids = [item["source_id"] for item in allocation["assignments"][root]]
        selected_cards = [by_source[source_id] for source_id in source_ids if source_id in by_source]
        plan_path = folder / "S01.PLAN.json"
        final_plan = read(plan_path) if plan_path.exists() else None
        final_scenes = final_plan.get("scenes", []) if final_plan else []
        development_counts = [len(scene["developments"]) for scene in final_scenes]
        quality = final_quality(reviews[root], accepted)
        row = {"root_id": root, "cohort": cohorts[root], "input_source": inputs[root],
               "raw_output": outputs[root], "source": record["source"],
               "provenance_or_failure": bind(record_path), "stage_index": bind(stage_path),
               "mechanically_accepted": accepted, "accepted_scenes": result["accepted_scenes"],
               "input_turns": len(before), "new_turn_count": len(suffix), "output_turns": len(after),
               "new_body_tokens_qwen3_4b": sum(len(tokenizer.encode(turn["text"], add_special_tokens=False).ids) for turn in suffix),
               "source_quality_verdict": quality, "source_usable_after_review": quality in USABLE,
               "quality_reviews": reviews[root], "question_answer_certified": False, "training_allowed": False,
               "card_step_counts": [len(card["steps"]) for card in selected_cards],
               "six_step_card_hit": any(len(card["steps"]) == 6 for card in selected_cards),
               "final_development_counts": development_counts,
               "final_plan": bind(plan_path) if plan_path.exists() else None,
               "final_failures": result["failures"], "no_available_guidance": result["no_available_guidance"],
               "scene_metadata": result["scene_metadata"], **analysis}
        if row["cohort"] == COHORTS[1]:
            old_run = home.parent / "h4_end_to_end_expansion_mixed_v90" / "expansion_engine/runs/four_roots_B_scene01"
            prior_result = old_run / "roots" / root / "B/RESULT.json"
            prior = read(prior_result)
            require(prior["accepted_scenes"] == 0 and any(
                entry.get("reason") == "plan_rejected_after_retry" for entry in prior["failures"]),
                "regression_baseline_not_planner_failure")
            prior_errors = []
            prior_planner_bindings = []
            for proof_path in sorted((old_run / "requests").glob("*/PROOF.json")):
                proof = read(proof_path)
                if proof.get("root_id") == root and proof.get("role") == "planner":
                    path = proof_path.parent / "VALIDATION.json"
                    prior_errors.extend(error_codes(read(path)))
                    prior_planner_bindings.extend((bind(proof_path), bind(path)))
            require("scene_developments_count_invalid" in prior_errors, "regression_baseline_not_count_error")
            row["regression_baseline"] = {"result": bind(prior_result), "error_codes": prior_errors,
                "planner_bindings": prior_planner_bindings,
                "source_sha256_identical": bind(old_run / "sources" / (root + ".json"))["sha256"] == inputs[root]["sha256"],
                "same_new_dataset_card": False,
                "limitation": "Same-root new-card regression; cannot isolate the prompt fix from assigned-card differences."}
        rows.append(row)
    require(material["accepted_roots"] == sum(row["mechanically_accepted"] for row in rows)
            and material["failed_roots"] == sum(not row["mechanically_accepted"] for row in rows), "materialized_totals_changed")
    summaries = []
    for cohort in COHORTS:
        selected = [row for row in rows if row["cohort"] == cohort]
        summaries.append({"cohort": cohort, "total_roots": len(selected),
            "accepted_roots": sum(row["mechanically_accepted"] for row in selected),
            "usable_roots": sum(row["source_usable_after_review"] for row in selected),
            "hold_roots": sum(row["source_quality_verdict"] == "HOLD" for row in selected),
            "added_turns": sum(row["new_turn_count"] for row in selected),
            "added_body_tokens_qwen3_4b": sum(row["new_body_tokens_qwen3_4b"] for row in selected),
            "usable_added_turns": sum(row["new_turn_count"] for row in selected if row["source_usable_after_review"]),
            "usable_added_body_tokens_qwen3_4b": sum(row["new_body_tokens_qwen3_4b"] for row in selected if row["source_usable_after_review"]),
            "six_step_card_roots": sum(row["six_step_card_hit"] for row in selected),
            "planner_count_error_attempts": sum(row["planner_count_error_attempts"] for row in selected),
            "planner_cardinality_statuses": dict(Counter(row["planner_cardinality_status"] for row in selected)),
            "logical_repeat_requests": sum(row["logical_repeat_requests"] for row in selected),
            "final_failure_reasons": dict(Counter(entry.get("reason", "UNKNOWN") for row in selected for entry in row["final_failures"]))})
    return {"version": "v91_expansion_only_export_audit_v1", "status": "PASS", "run": str(run),
            "preservation": preservation, "frozen_input_runtime_files": len(manifest["files"]),
            "execution_code_snapshots": len(snapshots), "engine_audit": bind(run / "FINAL_AUDIT.json"),
            "raw_report": bind(run / "REPORT.json"), "materialization": bind(material_index_path),
            "quality_review_files": review_bindings, "quality_review_union_covers_exactly_all_roots": True,
            "quality_review_protocol": review_protocol,
            "source_registry": bind(home / "expansion_engine/SOURCE_REGISTRY.json"),
            "exporter": bind(Path(__file__)), "body_token_definition": token_definition,
            "model": manifest["model"], "http_attempts": report["http_attempts"],
            "logical_requests_by_role": saved_audit["logical_requests_by_role"],
            "question_calls": 0, "solver_calls": 0, "training_started": False,
            "audit_or_export_network_requests": 0, "historical_train_roots": True,
            "selection_is_random": False, "semantic_acceptance_rules_changed": False,
            "source_quality_is_manual_review_verdict_not_formal_semantic_certification": True,
            "question_answer_certified": False, "training_allowed": False,
            "logical_repeat_definition": "Requests beyond first per role/root/scene; dataset per source, event state per plan attempt, scene reviewer per realization SHA. HTTP retries excluded.",
            "cohorts": summaries, "roots": rows}


def rendered_exports(home, audit):
    rows = audit["roots"]
    accepted = [row for row in rows if row["mechanically_accepted"]]
    successes = {"version": "v91_mechanically_accepted_expansions_v1",
        "mechanically_accepted_count": len(accepted),
        "reviewed_usable_count": sum(row["source_usable_after_review"] for row in accepted),
        "quality_review_protocol": audit["quality_review_protocol"],
        "meaning": "Mechanical expansion acceptance does not imply semantic READY, certified answers or training admission.",
        "question_answer_certified": False, "training_allowed": False,
        "body_token_definition": audit["body_token_definition"], "roots": accepted}
    failures = {"version": "v91_all_expansion_failures_and_repairs_v1",
        "final_shortfall_roots": [row for row in rows if not row["mechanically_accepted"]],
        "quality_hold_roots": [row for row in rows if row["source_quality_verdict"] == "HOLD"],
        "all_roots_with_rejected_attempts": [{"root_id": row["root_id"], "cohort": row["cohort"],
            "final_mechanically_accepted": row["mechanically_accepted"], "quality_verdict": row["source_quality_verdict"],
            "rejected_stages": row["rejected_stages"], "planner_attempts": row["planner_attempts"],
            "logical_repeat_requests_by_role": row["logical_repeat_requests_by_role"],
            "final_failures": row["final_failures"], "stage_index": row["stage_index"]}
            for row in rows if row["rejected_stages"]],
        "all_original_responses_retained": True, "requests_reexecuted_by_exporter": False,
        "quality_review_protocol": audit["quality_review_protocol"],
        "network_requests": 0, "question_answer_certified": False, "training_allowed": False}
    text = ["# v91：修复 planner 项数问题后的扩充批测", "",
        "本轮仅测试完整扩充流程。流程接受与整段质量复核分别报告；题目答案尚未认证，也未进入训练。", "",
        "审核 A 保留分工代理原稿；审核 B 由父代理在分工代理 403 中断后，根据其已完成阅读和反馈及父代理直接复核完成。A 后续曾意外接触部分模型摘要，父代理也已看到运行判定，因此不能称为全程盲审，也没有每根独立第二最终审核者。限制见 [审核完成说明](evidence/REVIEW_COMPLETION_NOTE.md)。", "",
        "| 组别 | 流程接受 / 总根 | 复核可用 | HOLD | 流程新增 turns / tokens | 复核可用新增 turns / tokens | 6-step 卡片根数 | planner 项数错误 attempts | 重复逻辑请求 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {COHORTS[0]: "其他 roots 批测", COHORTS[1]: "旧 planner 失败回归"}
    for c in audit["cohorts"]:
        text.append(f"| {labels[c['cohort']]} | {c['accepted_roots']} / {c['total_roots']} | {c['usable_roots']} | {c['hold_roots']} | {c['added_turns']} / {c['added_body_tokens_qwen3_4b']} | {c['usable_added_turns']} / {c['usable_added_body_tokens_qwen3_4b']} | {c['six_step_card_roots']} | {c['planner_count_error_attempts']} | {c['logical_repeat_requests']} |")
    text += ["", "| root | 组别 | 流程结果 | 复核质量 | 卡片 steps → 最终 developments | 新增 turns / tokens | planner 项数状态 | 重复逻辑请求 | 最终失败原因 |",
             "|---|---|---|---|---|---:|---|---:|---|"]
    for row in rows:
        count = ",".join(map(str, row["card_step_counts"])) or "无卡片"
        dev = ",".join(map(str, row["final_development_counts"])) or "未达最终计划"
        reasons = "; ".join(entry.get("reason", "UNKNOWN") for entry in row["final_failures"]) or "无"
        text.append(f"| {row['root_id'].rsplit('-', 1)[-1]} | {labels[row['cohort']]} | {'接受' if row['mechanically_accepted'] else '短缺'} | {row['source_quality_verdict']} | {count} → {dev} | {row['new_turn_count']} / {row['new_body_tokens_qwen3_4b']} | {row['planner_cardinality_status']} | {row['logical_repeat_requests']} | {reasons.replace('|', '/')} |")
    text += ["", "本轮把每个新场景限定为 2–5 个 developments；6 个卡片步骤可以映射到同一个 development，但仍须完整保留每个步骤、依赖、行动主体及请求/同意/尝试/报告等状态。原卡片、历史对话及每次模型原响应均保留，未通过删步骤或放宽语义审核实现接受。", "",
        "planner 项数状态：PASS_WITHOUT_COUNT_ERROR 表示最后一次 planner 响应通过结构验证且满足 2–5 项，本轮未出现项数拒绝；REPAIRED 表示项数拒绝后最后一次 planner 响应结构通过；UNRESOLVED_COUNT_ERROR 表示项数拒绝仍未解决；NO_VALID_FINAL_PLAN 表示没有通过结构验证的最终 planner 响应。planner 结构通过仍可能在事件历史、实现或质量审核失败。", "",
        "旧失败回归使用同一 root 的原历史输入及新分配卡片，因此只能证明修复后同根流程表现，不能把全部改善归因于提示词。两组成功率分别统计，避免旧回归根混入其他 roots 的批测分母。", "",
        "重复逻辑请求统计同阶段超过首次的请求（包括重规划、结构或引用修复）；event state 按 plan attempt 分组，reviewer 按 realization SHA 分组，不计 HTTP 传输重试。失败索引保留所有初次拒绝和最终短缺，成功根的早期失败同样保留。", "",
        "质量复核 findings：", ""]
    for row in rows:
        findings = [finding for review in row["quality_reviews"] for finding in review["record"]["findings"]]
        text.append(f"- root {row['root_id'].rsplit('-', 1)[-1]}（{row['source_quality_verdict']}）：" +
                    (json.dumps(findings, ensure_ascii=False) if findings else "无新增质量发现。"))
    text += ["", f"扩充 HTTP attempts：{audit['http_attempts']}；新增 QA 调用 = 0，Solver 调用 = 0，训练 = 0。离线审核/导出网络请求 = 0。", "",
        "父代理补充诊断：20432 的事件检查四项均仅因旧历史没有这些大学访问新事件而拒绝，且 constraint_ids 全为空；这一拒绝理由不符合现有允许新事件建立新事实的规则。104814 的第二批测试则存在新完成报告与历史计划的区分问题，时间间隔仍需明确。保留模型原判，不把可能误拒绝补算为成功。见 [补充诊断](evidence/PARENT_REVIEW_DIAGNOSTICS.json)。", "",
        "两条流程接受 source 全部 HOLD，复核可用新增量为零，当前流程尚不支持稳定的大规模扩充。下一项最小修复是提前校验卡片语义与主体/状态、区分新事件和历史矛盾，并核对新对话人物与行程；本轮未实施这些后续修改。见 [下一项修复建议](evidence/NEXT_MINFIX_DIAGNOSIS_ZH.md)。", "",
        "正文 tokens 用真实本地 Qwen3-4B tokenizer（已额外 SHA 绑定到导出审核）对每个新增 turn.text 单独编码，无特殊 token、角色名字、标签、JSON 或聊天包装。prepare 的容量 tokenizer 实际来自 Qwen3-14B；该离线容量检查不能描述为 Qwen3-4B 认证。本轮没有 Solver 调用。机械新增量含 HOLD，审核可用新增量仅含 USABLE/USABLE_WITH_WARNINGS，二者均不含历史原对话。", "",
        "这 10 根均为历史 TRAIN 根，8 根其他 roots 来自已有可用来源，并非随机或陌生测试集。此次单场景小批及其失败证据不能证明大规模稳定、题目可出、Gold 答案可靠或训练准确率提高。", "",
        f"旧交付 {audit['preservation']['protected_count']:,} 个文件及精确目录清单逐 SHA 复核通过。输入/runtime binding、执行代码快照、原请求/原响应与接受链已复算，保存的 FINAL_AUDIT 与复算一致。", "",
        "文件入口：[完整成功对话及 provenance](SUCCESSFUL_EXPANSIONS.json)、[所有失败和修复](FAILURE_INDEX.json)、[产物索引](ARTIFACT_INDEX.md)。", ""]
    index = ["# v91 产物索引", "", "所有路径相对本交付目录。成功表示流程已接受；整段 source 复核质量及训练限制见各根记录。", "",
        "- [中文完整结果](COMPLETE_BATCH_RESULTS_ZH.md)", "- [流程成功及复核质量状态](SUCCESSFUL_EXPANSIONS.json)",
        "- [失败、修复、HOLD 索引](FAILURE_INDEX.json)", "- [导出一致性审核](evidence/EXPORT_AUDIT.json)",
        "- [原始运行报告](expansion_engine/runs/ten_roots_B_scene01/REPORT.json)",
        "- [机械审核](expansion_engine/runs/ten_roots_B_scene01/FINAL_AUDIT.json)",
        "- [来源物化索引](evidence/EXPANSION_MATERIALIZATION_COMPLETE.json)",
        "- [整段审核 A（分工代理原稿）](evidence/EXPANSION_QUALITY_REVIEW_A.json)", "- [整段审核 B（父代理完成）](evidence/EXPANSION_QUALITY_REVIEW_B.json)",
        "- [审核完成说明与盲审限制](evidence/REVIEW_COMPLETION_NOTE.md)",
        "- [审核完成机器记录](evidence/REVIEW_COMPLETION_RECEIPT.json)",
        "- [父代理事件审核补充诊断](evidence/PARENT_REVIEW_DIAGNOSTICS.json)",
        "- [下一项最小修复建议（未实施）](evidence/NEXT_MINFIX_DIAGNOSIS_ZH.md)",
        "- [原请求、所有 HTTP 响应及失败 attempts](expansion_engine/runs/ten_roots_B_scene01/requests/)",
        "- [输入与 runtime SHA 绑定](expansion_engine/runs/ten_roots_B_scene01/MANIFEST.json)",
        "- [执行代码快照](expansion_engine/runs/ten_roots_B_scene01/EXECUTION_CODE_SNAPSHOT.json)",
        "- [旧交付保护清单](PRESERVATION_BEFORE.json)",
        "- [最终精确封存清单（seal 最后生成）](COMPLETE_MANIFEST.json)", "",
        "| root | 完整成功 source | 原始累计 output（含失败） | provenance / failure | 全阶段索引 |",
        "|---|---|---|---|---|"]
    def link(binding):
        if binding is None:
            return "无新增 source"
        rel = Path(binding["path"]).relative_to(home).as_posix()
        return f"[{Path(rel).name}]({rel})"
    for row in rows:
        index.append(f"| {row['root_id'].rsplit('-', 1)[-1]} | {link(row['source'])} | {link(row['raw_output'])} | {link(row['provenance_or_failure'])} | {link(row['stage_index'])} |")
    index += ["", "封存后运行 verify-complete：精确校验全部文件增删改、旧文件 SHA 和当前输入/runtime，再复算审核并确认导出与已审核原稿一致。", ""]
    return {"SUCCESSFUL_EXPANSIONS.json": json_bytes(successes), "FAILURE_INDEX.json": json_bytes(failures),
            "COMPLETE_BATCH_RESULTS_ZH.md": "\n".join(text).encode("utf-8"),
            "ARTIFACT_INDEX.md": "\n".join(index).encode("utf-8"),
            "evidence/EXPORT_AUDIT.json": json_bytes(audit)}


def verify_exports(home, audit):
    for relative, data in rendered_exports(home, audit).items():
        require((home / relative).is_file() and (home / relative).read_bytes() == data,
                "export_missing_or_not_current_audited_data:" + relative)


def operate(home, run, action):
    if action == "verify-complete":
        require((home / SEAL_NAME).is_file(), "seal_missing")
        sealed = verify_seal_inventory(home)
        audit = collect(home, run)
        verify_exports(home, audit)
        require(sealed["audit"] == bind(home / "evidence/EXPORT_AUDIT.json"), "seal_audit_binding_changed")
        return {"status": "PASS", "action": action, "file_count": sealed["file_count"],
                "protected_old_file_count": audit["preservation"]["protected_count"],
                "network_requests": 0, "filesystem_writes": 0, "cohorts": audit["cohorts"]}
    if action in ("export", "seal"):
        require(not (home / SEAL_NAME).exists(), "sealed_delivery_refuses_write")
    audit = collect(home, run)
    if action == "export":
        for relative, data in rendered_exports(home, audit).items():
            immutable_write(home / relative, data)
        verify_exports(home, audit)
    elif action == "seal":
        verify_exports(home, audit)
        files = inventory(home, (SEAL_NAME,))
        require(files == inventory(home, (SEAL_NAME,)), "delivery_changed_during_seal")
        document = {"version": "v91_exact_delivery_seal_v1", "status": "COMPLETE",
            "home": str(home), "run": str(run), "files": files, "file_count": len(files),
            "excluded_self": SEAL_NAME, "audit": bind(home / "evidence/EXPORT_AUDIT.json"),
            "protected_old_file_count": audit["preservation"]["protected_count"],
            "all_original_responses_retained": True, "question_calls": 0, "solver_calls": 0,
            "training_started": False, "question_answer_certified": False, "training_allowed": False}
        immutable_write(home / SEAL_NAME, json_bytes(document))
        verify_seal_inventory(home)
    return {"status": "PASS", "action": action, "cohorts": audit["cohorts"],
            "protected_old_file_count": audit["preservation"]["protected_count"],
            "question_calls": 0, "solver_calls": 0, "training_started": False,
            "network_requests": 0, "seal_created": action == "seal"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("audit", "export", "seal", "verify-complete"))
    parser.add_argument("--home", type=Path, default=HOME)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    home = args.home.resolve()
    run = args.run.resolve() if args.run else home / "expansion_engine" / "runs" / RUN_NAME
    result = operate(home, run, args.action)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
