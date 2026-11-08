"""Freeze screened roots and unused native-event partitions before matching.

This bootstrap does not match/reserve events, author scenes, call a model,
write RUN_CONFIG, or freeze the eventual execution/QA tool inputs.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DATA = PROJECT / "SP-820-publish/first_training_pipeline/data"
SCREEN = HERE.parent / "h4_codex_baseline_screen_v116"
PREVIOUS = HERE.parent / "h4_codex_newroots_ab_validation_v115"
EXPECTED_ROOTS = ["cc-train-episode-57679", "cc-train-episode-68549",
                  "cc-train-episode-75223", "cc-train-episode-77882"]
POOL_SHA = "2f3ea771d4e01cb2780357738cff7f7496b87d34c221cc240df74501312438d3"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify(binding):
    assert sha(binding["path"]) == binding["sha256"], binding["path"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def copy(path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(Path(path).read_bytes())


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    # Other agents may prepare only these tools before bootstrap. No generated
    # output or directory may already exist; every output uses exclusive creation.
    preparation_tools = {"bootstrap.py", "evaluate.py", "audit_solver.py", "deliver.py",
                         "source_review.py", "prepare_path_adapter_v2.py"}
    startup_files = list(HERE.iterdir())
    assert {p.name for p in startup_files} <= preparation_tools
    assert all(p.is_file() for p in startup_files)
    assert (HERE / "bootstrap.py").is_file()
    created = now()
    selected = read(SCREEN / "SELECTED_ROOTS.json")
    screen_plan = read(SCREEN / "SCREEN_PLAN.json")
    handoff = read(SCREEN / "NEXT_BATCH_HANDOFF.json")
    audit = read(SCREEN / "DELIVERY_AUDIT.json")
    final = read(SCREEN / "FINAL_STATUS.json")
    screen_manifest = read(SCREEN / "COMPLETE_MANIFEST.json")
    assert [row["root_id"] for row in selected["roots"]] == EXPECTED_ROOTS
    assert final["selected_root_ids"] == EXPECTED_ROOTS
    assert audit["status"] == "PASS" and audit["errors"] == []
    assert handoff["generation_root_replacement_allowed"] is False
    assert all(row["selected"] and row["eligible"] and row["decision"] != "HOLD"
               for row in selected["roots"])
    for binding in final["bindings"] + audit["input_bindings"] + handoff["screen_bindings"]:
        verify(binding)
    screen_files = sorted(path for path in SCREEN.rglob("*") if path.is_file())
    assert {str(path) for path in screen_files} == {
        row["path"] for row in screen_manifest["files"]} | {str(SCREEN / "COMPLETE_MANIFEST.json")}
    for binding in screen_manifest["files"]:
        verify(binding)

    startup_history_path = HERE / "evidence/BOOTSTRAP_STARTUP_HISTORY.json"
    write(startup_history_path, {
        "recorded_at_utc": created,
        "previous_attempt": {
            "attempt_timestamp_utc": None,
            "timestamp_unavailable_reason": "The failed tool result did not record an absolute wall-clock timestamp.",
            "command": "python -B offline-deliveries/h4_codex_screened_ab_validation_v117/bootstrap.py",
            "bootstrap_sha256_at_attempt": "a8f0b011ae628c6c5a5a46a96233732806a84c7075f1d41142937255f6a9eb3f",
            "exit_code": 1,
            "stderr": "Traceback (most recent call last):\n  File \"/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_screened_ab_validation_v117/bootstrap.py\", line 381, in <module>\n    main()\n  File \"/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_screened_ab_validation_v117/bootstrap.py\", line 73, in main\n    assert sorted(p.name for p in HERE.iterdir()) == [\"bootstrap.py\"]\nAssertionError\n",
            "failure_stage": "initial_directory_guard_before_any_output_write",
            "bootstrap_outputs_written": False,
            "observed_directory_entries_after_failure": ["audit_solver.py", "bootstrap.py", "deliver.py", "evaluate.py"],
        },
        "correction": "Permit only the five authorized preparation tools alongside bootstrap; continue to reject every preexisting bootstrap output.",
        "current_startup_entries": sorted(p.name for p in startup_files),
        "preparation_tool_bindings": [bind(p) for p in sorted(startup_files)],
    })

    # Preserve all prior protected content and every file in the sealed v116 delivery.
    inherited_path = SCREEN / "HISTORICAL_PROTECTION.json"
    inherited = read(inherited_path)["files"]
    assert len(inherited) == 9567
    protection = {}
    for binding in inherited:
        verify(binding)
        protection[binding["path"]] = binding["sha256"]
    for path in screen_files:
        protection[str(path)] = sha(path)
    historical_path = HERE / "evidence/HISTORICAL_PROTECTION.json"
    write(historical_path, {
        "created_at_utc": created,
        "scope": "Exact transitive v116 historical protection plus all files in sealed v116, including its complete manifest.",
        "inherited_protection": bind(inherited_path),
        "inherited_file_count": len(inherited), "v116_file_count": len(screen_files),
        "protected_file_count": len(protection),
        "files": [{"path": path, "sha256": digest} for path, digest in sorted(protection.items())],
    })

    authority = bind(SCREEN / "SELECTED_ROOTS.json")
    gate_names = ["SCREEN_PLAN.json", "BASELINE_POLICY.md", "SCREEN_RESULTS.json",
                  "SELECTED_ROOTS.json", "REVIEW_COMPARISON.json", "DELIVERY_AUDIT.json",
                  "FINAL_STATUS.json", "NEXT_BATCH_HANDOFF.json", "SEAL_RECEIPT.json",
                  "COMPLETE_MANIFEST.json", "evidence/final_report_review.json"]
    gate_path = HERE / "BASELINE_GATE.json"
    write(gate_path, {
        "schema_version": "codex_screened_baseline_gate_v1", "screen_version": "v116",
        "created_at_utc": created, "selection_authority": authority,
        "fixed_root_ids": EXPECTED_ROOTS, "screen_bindings": [bind(SCREEN / name) for name in gate_names],
        "screen_candidate_denominator": 8, "screen_eligible_root_count": 6,
        "screen_selected_root_count": 4, "generation_root_replacement_allowed": False,
        "all_continuation_constraints_binding_required": True,
        "screen_is_external_semantic_certification": False,
        "selection_is_not_random_four_root_sampling": True,
    })

    # Provenance and three-level split isolation, using original source metadata only.
    originals_path = DATA / "selection_manifest_h4_unexpanded122.json"
    originals = {row["root_id"]: row for row in read(originals_path)["records"]}
    manifest_paths = {
        "train": DATA / "matched_training/m_clean_policy_training_manifest.jsonl",
        "dev": DATA / "matched_training/dev_full_eval_manifest.jsonl",
        "test": DATA / "matched_training/test_full_eval_manifest.jsonl",
    }
    split_rows = {split: jsonl(path) for split, path in manifest_paths.items()}
    split_roots = {split: {row["root_id"] for row in rows} for split, rows in split_rows.items()}
    metadata = {}
    original_bindings = []
    for rid in sorted(set.union(*split_roots.values())):
        item = originals[rid]
        assert sha(item["input_path"]) == item["input_sha256"]
        source = read(item["input_path"])["source"]
        assert source["source_record_sha256"] == item["source_record_sha256"]
        metadata[rid] = {"original_source_id": source["source_id"],
                         "original_source_record_sha256": source["source_record_sha256"]}
        original_bindings.append(bind(item["input_path"]))
    split_sets = {
        split: {"root_id": roots,
                "source_id": {metadata[rid]["original_source_id"] for rid in roots},
                "source_record_sha256": {metadata[rid]["original_source_record_sha256"] for rid in roots}}
        for split, roots in split_roots.items()
    }
    isolation = []
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = {level: sorted(split_sets[left][level] & split_sets[right][level])
                   for level in ("root_id", "source_id", "source_record_sha256")}
        assert not any(overlap.values())
        isolation.append({"left": left, "right": right, "overlap": overlap})
    assert set(EXPECTED_ROOTS).issubset(split_roots["train"])
    candidates = {row["root_id"]: row for row in screen_plan["candidates"]}
    roots = []
    for row in selected["roots"]:
        rid = row["root_id"]
        candidate = candidates[rid]
        for key, value in metadata[rid].items():
            assert candidate[key] == value
        baseline_source = Path(row["baseline_path"])
        assert sha(baseline_source) == row["baseline_sha256"]
        source = read(baseline_source)
        speakers = sorted({turn["speaker"] for turn in source["turns"]})
        scene_refs = {turn["scene_ref"] for turn in source["turns"]
                      if isinstance(turn.get("scene_ref"), str) and turn["scene_ref"].startswith("R01:")}
        assert len(speakers) == 2 and scene_refs == {"R01:S01", "R01:S02", "R01:S03"}
        baseline_target = HERE / "inputs" / ("root" + rid.rsplit("-", 1)[1] + "_BASELINE.json")
        copy(baseline_source, baseline_target)
        assert sha(baseline_target) == row["baseline_sha256"]
        constraints = row["continuation_constraints"]
        assert constraints == list(dict.fromkeys(c for review in row["reviews"]
                                                for c in review["continuation_constraints"]))
        constraints_path = HERE / "inputs" / ("root" + rid.rsplit("-", 1)[1] + "_CONTINUATION_CONSTRAINTS.json")
        write(constraints_path, {
            "schema_version": "codex_screened_root_constraints_v1", "root_id": rid,
            "baseline_gate_version": "v116", "baseline_gate_decision": row["decision"],
            "selection_authority": authority, "baseline_binding": bind(baseline_target),
            "review_bindings": row["reviews"], "continuation_constraints": constraints,
        })
        roots.append({
            "root_id": rid, "baseline_path": str(baseline_target), "baseline_sha256": sha(baseline_target),
            "upstream_path": str(baseline_source), "upstream_sha256": sha(baseline_source),
            "baseline_turn_count": len(source["turns"]), "baseline_speakers": speakers,
            **metadata[rid], "baseline_gate_decision": row["decision"],
            "continuation_constraints": constraints,
            "continuation_constraints_path": str(constraints_path),
            "continuation_constraints_sha256": sha(constraints_path),
            "v116_review_bindings": row["reviews"],
        })

    exposed = {}
    exposure_sources = []
    for directory in sorted(HERE.parent.iterdir()):
        version = re.search(r"_v(\d+)$", directory.name)
        if not directory.is_dir() or not version or not 104 <= int(version.group(1)) <= 115:
            continue
        if not directory.name.startswith(("h4_batch_codex_", "h4_codex_", "h4_unseen_root_")):
            continue
        path = next((directory / name for name in ("RUN_CONFIG.json", "FINAL_BATCH_ACCOUNTING.json", "PRECALL_PLAN.json")
                     if (directory / name).exists()), None)
        if path is None:
            continue
        binding = bind(path)
        exposure_sources.append(binding)
        for row in read(path).get("roots", []):
            rid = row["root_id"] if isinstance(row, dict) else row
            exposed.setdefault(rid, []).append(binding)
    assert all(exposed.get(rid) for rid in EXPECTED_ROOTS)
    exposure_path = HERE / "EXPOSURE_AUDIT.json"
    write(exposure_path, {
        "status": "PASS", "created_at_utc": created,
        "purpose": "TRAINING_SIDE_SCREENED_A3B3_WORKFLOW_VALIDATION",
        "selected_root_ids": EXPECTED_ROOTS, "selected_roots": roots,
        "already_exposed_training_roots": True,
        "prior_codex_authoring": {rid: exposed[rid] for rid in EXPECTED_ROOTS},
        "prior_codex_baseline_screening": bind(SCREEN / "SCREEN_PLAN.json"),
        "unexposed_clean_training_roots_remaining": len(split_roots["train"] - set(exposed)),
        "new_unexposed_generalization_claimed": False, "unseen_holdout_claimed": False,
        "baseline_has_C1": False, "prior_Codex_scene_text_reused": False,
        "clean_training_manifest": bind(manifest_paths["train"]),
        "holdout_manifests": [bind(manifest_paths[name]) for name in ("dev", "test")],
        "split_counts": {name: {"rows": len(split_rows[name]), "roots": len(split_roots[name])}
                         for name in split_rows},
        "split_pairwise_three_level_isolation": isolation,
        "selected_vs_holdout": {name: {
            "root_id_overlap": sorted(set(EXPECTED_ROOTS) & split_roots[name]),
            "source_id_overlap": sorted({r["original_source_id"] for r in roots} & split_sets[name]["source_id"]),
            "source_record_sha256_overlap": sorted({r["original_source_record_sha256"] for r in roots} & split_sets[name]["source_record_sha256"]),
        } for name in ("dev", "test")},
        "holdout_root_overlap": [], "holdout_source_id_overlap": [], "holdout_source_record_overlap": [],
        "original_provenance_manifest": bind(originals_path), "original_input_bindings": original_bindings,
        "exposure_scan_bindings": exposure_sources, "new_training_started": False,
        "external_semantic_certification": False,
    })

    # Metadata-only global consumption scan. Native turns remain opaque here.
    consumed = set()
    scan = []
    for path in sorted(PROJECT.rglob("SOURCE_RESERVATIONS.json")):
        if HERE in path.parents:
            continue
        value = read(path)
        ids = sorted({item for item in strings(value) if item.startswith("multiwoz22:train:")})
        consumed.update(ids)
        scan.append({**bind(path), "source_ids": ids,
                     "method": "All exact native MultiWOZ train source-id strings recursively; permanent including failed/unexecuted reservations."})
    assert any(row["path"] == str(PREVIOUS / "SOURCE_RESERVATIONS.json") for row in scan)
    legacy_path = HERE.parent / "h4_codex_controlled_batch_v105/evidence/SOURCE_SELECTION.json"
    legacy_ids = read(legacy_path)["consumed_source_ids"]
    consumed.update(legacy_ids)
    scan.append({**bind(legacy_path), "source_ids": sorted(set(legacy_ids)), "method": "Legacy explicit consumed_source_ids supplement."})
    pool_path = HERE.parent / "h4_external_event_source_refresh_v1/multiwoz_train_sample.json"
    assert sha(pool_path) == POOL_SHA
    native_pool = read(pool_path)
    native_ids = ["multiwoz22:train:" + event["dialogue_id"] for event in native_pool]
    assert len(native_ids) == len(set(native_ids)) == 512
    available = sorted((event for event in native_pool
                        if "multiwoz22:train:" + event["dialogue_id"] not in consumed),
                       key=lambda event: event["dialogue_id"])
    available_ids = ["multiwoz22:train:" + event["dialogue_id"] for event in available]
    assert not set(available_ids) & consumed
    groups = []
    for index, root in enumerate(roots):
        rid = root["root_id"]
        target = HERE / "pre_inputs" / ("root" + rid.rsplit("-", 1)[1] + "_EVENT_POOL.json")
        events = available[index::4]
        write(target, {
            "root_id": rid, "partition_index_zero_based": index, "partition_count": 4,
            "native_pool_binding": bind(pool_path), "events": events,
            "events_selected_or_reserved": False, "native_turns_semantically_read_by_bootstrap": False,
        })
        groups.append({"root_id": rid, "event_pool_path": str(target), "event_pool_sha256": sha(target),
                       "event_count": len(events), "partition_index_zero_based": index})
    all_partition_ids = ["multiwoz22:train:" + event["dialogue_id"]
                         for index in range(4) for event in available[index::4]]
    assert len(all_partition_ids) == len(set(all_partition_ids)) == len(available)
    assert set(all_partition_ids) == set(available_ids)
    scan_path = HERE / "evidence/SOURCE_CONSUMPTION_SCAN.json"
    write(scan_path, {
        "status": "PASS", "created_at_utc": created, "project_scan_root": str(PROJECT),
        "source_reservation_file_count": len(scan) - 1, "legacy_supplement": bind(legacy_path),
        "consumption_scan": scan, "consumed_source_ids": sorted(consumed),
        "source_pool": bind(pool_path), "native_pool_count": len(native_pool),
        "consumed_in_native_pool_count": len(set(native_ids) & consumed),
        "available_event_count": len(available), "available_source_ids": available_ids,
        "partition_method": "Sort unused native events by dialogue_id, then index modulo four in fixed v116 selected-root order.",
        "groups": groups, "partitions_disjoint_and_complete": True,
        "events_selected_or_reserved": False, "event_content_semantically_read": False,
    })

    contract_source = PREVIOUS / "evidence/CONTENT_CONTRACT.md"
    contract = contract_source.read_text().replace("内容合同 v115", "内容合同 v117").replace("## v115", "## v117")
    contract += "\n## v117 扩充前基线筛查与全部约束\n"
    contract += "- root 仅取已封存 v116 SELECTED_ROOTS.json 的四个既定条目；本批固定尝试分母为4 root、24个新scene，不换root、不补候选。v116筛查分母8不冒充本批生成分母。\n"
    contract += "- 每位事件选择者、作者和审核者须逐条读取本root的全部 continuation_constraints，并绑定 inputs/root<ID>_CONTINUATION_CONSTRAINTS.json；不得删掉任一读者限制、借新增事实救活旧矛盾，或补设既有年龄/性别/亲属/职业/完成状态。\n"
    contract += "- 新增明确当前行动可推进尚未完成的计划；不得回退原文已完成的阶段。WARNING持续继承，程序通过不等于逐句事实认证。\n"
    contract += "- 本bootstrap只冻结基线、约束、规则和不相交未用事件分区；须待四root全文事件匹配、永久预留和最终输入/工具冻结后才允许作者首稿。\n"
    contract_path = HERE / "evidence/CONTENT_CONTRACT.md"
    with contract_path.open("x", encoding="utf-8") as stream:
        stream.write(contract)
    rules = handoff["required_rules"]
    plan = {
        "schema_version": "codex_screened_ab_precall_plan_v1", "batch_id": HERE.name,
        "created_at_utc": created, "status": "FROZEN_BEFORE_EVENT_MATCHING_AND_AUTHORING",
        "roots": roots, "groups": groups, "baseline_gate": bind(gate_path),
        "root_selection_authority": authority,
        "root_selection": "Exactly the four entries already selected by sealed v116; no new root choice, replacement or replenishment.",
        "fixed_root_denominator": 4, "planned_scene_denominator": 24,
        "prior_screen_candidate_denominator": 8, "scene_sequence": rules["sequence"],
        "required_rules": rules,
        "source_selection": "Future full-baseline/full-native-event compatibility matching: one restaurant, attraction and train per root only within its frozen disjoint partition; reserve permanently before authoring.",
        "source_pool_path": str(pool_path), "source_pool_sha256": sha(pool_path),
        "available_event_count": len(available), "consumed_source_ids": sorted(consumed),
        "consumption_scan": scan, "source_consumption_scan": bind(scan_path),
        "new_turn_count": rules["turns_per_scene"], "author_attempts_per_scene": 1,
        "independent_full_source_ready_before_next_scene": True,
        "semantic_repairs": 0, "source_replacements": 0,
        "scene_payload_writer": "scripts/codex_scene_payload.py; authors supply only speaker/text. Metadata serialization is not semantic repair.",
        "qa_author_attempts_per_complete_root": 1,
        "slots_per_ready_root": {"gold2": 1, "gold3": 1, "gold4": 1, "gold5": 1, "gold6_control": 1, "ordering": 2},
        "planned_new_questions_max": 24, "planned_controls_max": 4,
        "solver_protocol": rules["solver_protocol"],
        "solver_models": {"trained": "M_clean_policy", "base": "M0"},
        "solver_calls_per_model_max": 456, "solver_calls_both_models_max": 912,
        "solver_automatic_retry": False,
        "selection": "Complete-valid 0 < equal-display partial mean < 1; Gold6 excluded from training candidates.",
        "historical_protection": bind(historical_path), "exposure_audit": bind(exposure_path),
        "content_contract": bind(contract_path), "content_contract_inherited_from": bind(contract_source),
        "bootstrap": bind(Path(__file__)),
        "bootstrap_startup_history": bind(startup_history_path),
        "source_matching_started": False, "events_reserved": False,
        "run_config_written": False, "final_execution_inputs_frozen": False,
        "generation_started": False, "solver_started": False, "training_started": False,
        "pool_merge_started": False, "training_or_pool_merge_authorized": False,
        "backend": "CURRENT_CODEX_SESSION", "exact_model_id": None, "sampling_parameters": None,
        "usage_tokens": None, "billed_cost": None,
        "existing_training_exposure": True, "unseen_holdout_claimed": False,
        "external_semantic_certification": False, "statistical_unit": "root/family",
        "formal_backend_comparison": False,
        "primary_metrics": ["6READY roots / all4 frozen roots", "READY scenes /24 and attempted scene count",
                            "actor_identity failures among reviewed scenes", "content pass and Solver-selected counts by root/type/Gold",
                            "wall time and failure categories"],
    }
    # Recheck all protected historical files immediately before the final plan write.
    for path, expected_sha in protection.items():
        assert sha(path) == expected_sha, path
    write(HERE / "PRECALL_PLAN.json", plan)
    write(HERE / "evidence/BOOTSTRAP_AUDIT.json", {
        "status": "PASS", "errors": [], "created_at_utc": now(),
        "fixed_root_ids": EXPECTED_ROOTS, "root_denominator": 4, "scene_denominator": 24,
        "all_continuation_constraints_preserved": True,
        "constraint_counts": {r["root_id"]: len(r["continuation_constraints"]) for r in roots},
        "available_events": len(available), "partition_sizes": [g["event_count"] for g in groups],
        "source_reservation_files_scanned": len(scan) - 1,
        "inherited_historical_files": len(inherited), "v116_files": len(screen_files),
        "protected_files": len(protection), "history_verified_after_bootstrap": True,
        "three_level_split_isolation": True,
        "plan": bind(HERE / "PRECALL_PLAN.json"),
        "source_matching_started": False, "events_reserved": False,
        "run_config_written": False, "final_execution_inputs_frozen": False,
        "generation_started": False, "solver_started": False, "training_started": False, "pool_merge_started": False,
    })
    print(json.dumps({"status": "BOOTSTRAPPED_NOT_EXECUTION_FROZEN", "roots": EXPECTED_ROOTS,
                      "available_events": len(available), "partition_sizes": [g["event_count"] for g in groups],
                      "source_reservation_files_scanned": len(scan) - 1,
                      "protected_files": len(protection), "precall_sha256": sha(HERE / "PRECALL_PLAN.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
