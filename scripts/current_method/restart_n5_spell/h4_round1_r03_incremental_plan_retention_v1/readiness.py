"""Versioned T1-T7 offline delivery. All model transports are mocked."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from copy import deepcopy
import difflib
import inspect
import io
import subprocess
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[4]
VERSION = "incremental_run_readiness_v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(output, name, value):
    path = Path(output) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))


def baseline(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from .preflight import protected_inventory
    protected = protected_inventory()
    runs = PROJECT / "restart-n5-runs"
    for name in ("h4_round1_r03_incremental_plan_retention_v1_offline_preflight_live_4",
                 "h4_round1_r03_incremental_retention_single_root_budget_check_live_1"):
        protected.update({str(p): digest(p) for p in (runs / name).rglob("*") if p.is_file()})
    save(output, "protected_before.json", protected)
    old = json.loads((runs / "h4_round1_r03_incremental_plan_retention_v1_offline_preflight_live_4/implementation_sha256.json").read_text())
    identities = {}
    for name, expected in old.items():
        source = Path(name)
        if source.parent == PACKAGE:
            actual = digest(source)
            identities[name] = {"live_4_sha256": expected, "captured_sha256": actual,
                                "matches_live_4": actual == expected}
            destination = output / "implementation_before" / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
    save(output, "implementation_before_identity.json", identities)


def run(output):
    from . import POLICY_ID
    from .bundle import build_failed_bundle, validate_bundle_feedback
    from .capacity import local_counter, proposed_budget, measure, MODEL_TOKENIZER_SOURCE
    from .input_views import project_supplement_inputs, reconstruct_supplement_inputs
    from .planning import prepare_original, sha
    from .prompts import render_supplement, package, SHARED_FEEDBACK_RULES_V3, TARGET_RULES, SUPPLEMENT_TEMPLATE
    from .run import main, implementation_manifest, offline_network_guard, production_registry
    from .test_readiness import exercise_http_cli
    from .transport_config import model_config_from_historical
    output = Path(output)
    if not (output / "protected_before.json").exists():
        raise ValueError("separate_prechange_baseline_required")
    if (output / "final_report.json").exists():
        raise ValueError("never_overwrite_delivery")
    real_registry_before = production_registry().exists()
    if real_registry_before:
        raise ValueError("production_state_requires_separate_readonly_identity_audit")
    counter = local_counter()
    runs = PROJECT / "restart-n5-runs"
    old_check = runs / "h4_round1_r03_incremental_retention_single_root_budget_check_live_1"
    bundle = build_failed_bundle(branch_id="r03-incremental-readiness-v2-122711-pending-authorization")
    feedback = validate_bundle_feedback(bundle)
    budget = proposed_budget()
    old_model = json.loads((runs / "h4_round1_r03_expanded_single_plan_fullflow_debug_live_3/execution_plan.json").read_text())
    model = model_config_from_historical(old_model, tokenizer_path=counter.identity["path"],
                                        tokenizer_sha256=counter.identity["tokenizer_sha256"])
    save(output, "bundle_PENDING_AUTHORIZATION.json", bundle)
    save(output, "budget_proposal_NOT_AUTHORIZED.json", budget)
    save(output, "model_config_PENDING_AUTHORIZATION.json", model)
    save(output, "feedback_source_to_bundle.json", feedback)
    save(output, "tokenizer_identity.json", counter.identity)
    capacity_source = {"user_supplied_provider_page_description": {"provider": "SiliconFlow", "model": model["model"], "context_display": "128K"},
        "screenshot_bitmap_available": False, "local_conservative_working_limit": 128000,
        "exact_provider_integer": "UNKNOWN", "server_context_or_8192_acceptance": "NOT_TESTED",
        "official_public_page_review": "No exact provider/model capacity integer established; no authenticated API or download/probe used.",
        "tokenizer_official_public_SHA_source": MODEL_TOKENIZER_SOURCE,
        "role_parameters_source": str(runs / "h4_round1_r03_expanded_single_plan_fullflow_debug_live_3/execution_plan.json"),
        "parameter_changes": "model/temperature/output/thinking unchanged; capacity accounting only",
        "roles": budget["role_capacity"], "thinking": False, "thinking_budget_parameter": "not sent, unchanged",
        "cumulative_dialogue_limit": {"value": 100000, "unit": budget["cumulative_length_measure"],
             "note": "pending proposal, not approved; prior 100000 byte bound was not actual tokenizer tokens"},
        "solver_training_limits": "UNCHANGED; no downstream limits written or widened"}
    save(output, "T1_role_capacity_sources.json", capacity_source)
    prepared = prepare_original(bundle["original_plan"], parent=bundle["parent"], candidates=bundle["candidates"], verified_tracks=bundle["verified_tracks"])
    old_budget = json.loads((old_check / "budget_proposal_NOT_AUTHORIZED.json").read_text())
    data = {"identity": {**bundle["identity"], "branch_id": "PROPOSED_NOT_AUTHORIZED_incremental_retention_122711"},
        "original_plan_version": "original_v1", "original_plan_sha256": prepared["original_sha256"],
        "original_plan": prepared["original"], "stable_candidates": prepared["candidate_registry"],
        "affected_candidate_ids": prepared["affected_candidate_ids"], "goal_diagnostics": prepared["goal_diagnostics"],
        "authoritative_history": bundle["parent"]["dialogue"]["turns"], "retained_actual_r03_dialogue": [],
        "budget": old_budget, "shared_feedback_view": {}}
    fixed_rules = SHARED_FEEDBACK_RULES_V3 + "\n" + TARGET_RULES + SUPPLEMENT_TEMPLATE
    a = package("supplement", fixed_rules, data)
    old_count = json.loads((old_check / "read_only_checks_result.json").read_text())["budget_measurements"]
    if sha(a["messages"]) != old_count["supplement_messages_sha256"]:
        raise ValueError("historical_A_reconstruction_identity_mismatch")
    data["shared_feedback_view"] = bundle["shared_feedback_view"]
    b = package("supplement", fixed_rules, data)
    c = render_supplement(data)
    mapping = c["input_projection_mapping"]
    save(output, "T2_field_retention_mapping.json", mapping)
    projected, _ = project_supplement_inputs(data)
    assert reconstruct_supplement_inputs(projected, mapping) == data
    table = ["| 原字段 | 重复 | 新视图唯一位置 |", "| --- | --- | --- |"]
    table.extend(f"| `{r['original_field']}` | {r['duplicate']} | `{r['unique_value_location']}` |" for r in mapping["fields"])
    save(output, "T2_field_retention_table.md", "\n".join(table) + "\n")
    comparisons = []
    for label, built in (("A_historical_empty", a), ("B_old_wrapper_real_feedback", b), ("C_lossless_wrapper_same_real_feedback", c)):
        save(output, "token_comparison/" + label + ".messages.json", built["messages"])
        comparisons.append({"case": label, "local_final_messages_tokens": counter.count_tokens(built["messages"]),
                            "messages_sha256": sha(built["messages"]),
                            "http_json_bytes_DIAGNOSTIC_ONLY": len(json.dumps(built["messages"], ensure_ascii=False).encode())})
    save(output, "T4_token_comparison.json", {"same_identity_and_budget_in_A_B_C": True,
         "B_C_same_nonempty_feedback": True, "records": comparisons,
         "A_reconstructed_matches_historical_SHA": True,
         "not_quality_or_semantic_improvement_evidence": True})
    raw_history = json.loads(Path(bundle["frozen_sources"]["history"]["path"]).read_text())["turns"]
    normalized = bundle["parent"]["dialogue"]["turns"]
    source_turns = []
    for raw, normal in zip(raw_history, normalized):
        same = raw["text"] == normal["text"] and raw["speaker"] == normal["speaker"] and raw["turn_ref"] == normal["turn_id"]
        scene_encoded = not raw.get("scene_ref") or raw["scene_ref"] in normal["turn_id"]
        source_turns.append({"raw_turn_ref": raw["turn_ref"], "normalized_turn_id": normal["turn_id"],
            "raw_scene_ref": raw.get("scene_ref"), "raw_session_ref": raw.get("session_ref"),
            "normalized_session_id": normal.get("session_id"), "raw_origin": raw.get("origin"),
            "same_ref_speaker_text": same, "scene_identity_preserved_in_ref": scene_encoded})
    assert len(raw_history) == len(normalized) == 143 and all(r["same_ref_speaker_text"] and r["scene_identity_preserved_in_ref"] for r in source_turns)
    save(output, "T2_all_143_source_turn_correspondence.json", source_turns)
    with offline_network_guard(), patch("sys.stdout", new=io.StringIO()):
        main(["--bundle", str(output / "bundle_PENDING_AUTHORIZATION.json"),
              "--budget", str(output / "budget_proposal_NOT_AUTHORIZED.json"),
              "--run-dir", str(output / "offline_cli_full_path")])
    actual = json.loads((output / "offline_cli_full_path/requests/001_supplement.json").read_text())
    actual_data = json.loads(actual["messages"][0]["content"].rsplit("\n\nINPUT DATA\n", 1)[1])
    assert actual_data["shared_feedback_view"] == bundle["shared_feedback_view"]
    save(output, "final_supplement_messages_PENDING_AUTHORIZATION.json", actual["messages"])
    blocks = [{"block": name, "standalone_text_tokens_NOT_ADDITIVE": counter.count_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)),
               "serialized_bytes_DIAGNOSTIC_ONLY": len(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode())}
              for name, value in actual_data.items()]
    save(output, "T4_final_supplement_blocks.json", {"whole_messages": measure(actual["messages"], counter=counter, budget=budget, role="supplement"),
         "blocks_explanatory_not_sum": blocks,
         "fixed_rules_standalone_tokens": counter.count_text(fixed_rules), "no_token_reduction_quota": True})
    for case in ("normal", "400", "429", "500", "non_json", "business_parse", "empty", "timeout"):
        exercise_http_cli(output / "http_mock_cases" / case, case, model_config=model)
    prefix = "scripts.current_method.restart_n5_spell."
    policy_prefix = prefix + PACKAGE.name + "."
    legacy = prefix + "simple_append_memorax_v1_memory_event_continuation_probe_v1."
    modules = [policy_prefix + name for name in ("test_policy", "test_bundle", "test_input_views", "test_http_evidence", "test_readiness")]
    modules += [legacy + name for name in ("test_memory_event_continuation", "test_planner_state_contract", "test_promotion", "test_scene_continuation", "test_scene_replacement", "test_realizer_adherence")]
    modules += [prefix + "h4_round1_r03_feedback_view_v2.test_feedback_view"]
    stream = io.StringIO()
    with offline_network_guard():
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
        from ..h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight import test_revision
        schemas = {}
        schema_root = runs / "h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight_live_7"
        schema_ids = {}
        for path in schema_root.rglob("*.json"):
            obj = json.loads(path.read_text())
            ident = obj.get("$id", "") if isinstance(obj, dict) else ""
            if ident == "h4.round1.r03.qa_feedback_output.v1.1.revision_1":
                schemas["schema_targeted"] = obj
                schema_ids[str(path)] = digest(path)
            elif ".fresh." in ident:
                schemas["schema_fresh"] = obj
                schema_ids[str(path)] = digest(path)
        qa_tests = []
        for name, func in inspect.getmembers(test_revision, inspect.isfunction):
            if name.startswith("test_"):
                try:
                    func(**{k: schemas[k] for k in inspect.signature(func).parameters})
                    qa_tests.append({"name": name, "status": "PASS"})
                except Exception as exc:
                    qa_tests.append({"name": name, "status": "FAIL", "exception": type(exc).__name__})
    save(output, "tests.log", stream.getvalue())
    save(output, "qa_revision_1_regression.json", {"tests": qa_tests, "schema_SHA": schema_ids})
    before = json.loads((output / "protected_before.json").read_text())
    after = {name: digest(name) for name in before}
    changed_protected = [name for name in before if before[name] != after[name]]
    save(output, "protected_after.json", after)
    changes = []
    for path in sorted(PACKAGE.glob("*.py")):
        previous = output / "implementation_before" / path.name
        text_before = previous.read_text() if previous.exists() else ""
        if text_before != path.read_text():
            changes.append({"file": str(path), "change": "modified" if previous.exists() else "new",
                            "sha256": digest(path), "before_sha256": digest(previous) if previous.exists() else None})
            diff = "".join(difflib.unified_diff(text_before.splitlines(True), path.read_text().splitlines(True),
                                               fromfile="before/" + path.name, tofile="after/" + path.name))
            save(output, "diffs/" + path.name + ".diff", diff)
    save(output, "changed_files.json", changes)
    compile_results = []
    for path in PACKAGE.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
        compile_results.append({"path": str(path), "in_memory_compile": "PASS"})
    save(output, "compile_checks.json", compile_results)
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=PROJECT / "current_method_github", capture_output=True, text=True)
    save(output, "git_diff_check.txt", diff_check.stdout + diff_check.stderr)
    manifest = implementation_manifest()
    save(output, "implementation_sha256.json", manifest)
    workflow = json.loads((output / "offline_cli_full_path/final_report.json").read_text())
    save(output, "T4_role_and_batch_capacity_checks.json", workflow["capacity_checks"])
    passed = result.wasSuccessful() and all(t["status"] == "PASS" for t in qa_tests) and not changed_protected and diff_check.returncode == 0
    passed &= not production_registry().exists() and workflow["external_call_count"] == 0 and workflow["mock_call_count"] == 17 and workflow["retained_scene_count"] == 8
    statuses = {f"T{i}": "已完成" if passed else "未完成" for i in range(1, 7)}
    statuses["T7"] = "等待预算授权"
    report = {"version": VERSION, "policy_id": POLICY_ID, "statuses": statuses,
        "offline_ready": bool(passed), "unittest_count": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped), "qa_groups": qa_tests,
        "protected_count": len(before), "protected_changes": changed_protected,
        "final_supplement_tokens": counter.count_tokens(actual["messages"]), "token_A_B_C": comparisons,
        "historical_initial_planner_calls": 1, "maximum_new_model_requests_PENDING_AUTHORIZATION": 17,
        "local_closure_reservation_slots": 1, "generated_token_cap_excluding_local_slot": 73728,
        "production_registry_before": real_registry_before, "production_registry_after": production_registry().exists(),
        "incremental_opportunity_consumed": False, "all_real_model_and_memory_calls": 0,
        "authenticated_API_and_tokenizer_downloads": 0, "public_document_reading_only": True,
        "NOT_TESTED": ["server 128000 / 8192 support and provider chat-template identity", "real model quality or semantics", "all 606 repository tests"],
        "implementation_changed_only_in_incremental_package": True,
        "old_QA_protected_manifest_provenance": "unresolved, not changed or rebaselined",
        "pending": ["separate full budget and execution authorization", "single root closure before other roots", "Composition/Fresh requests only after new R03 freeze; old14 paused", "QA mechanical duplicate historical coverage acceptance", "Targeted eligibility and compatibility before restart", "later Solver and training freeze/evaluation"]}
    save(output, "final_report.json", report)
    documentation = (PACKAGE / "READINESS_V2.md").read_text()
    save(output, "T1_T7_implementation_and_budget.md", documentation)
    command = "PYTHONDONTWRITEBYTECODE=1 python -m " + policy_prefix + "run --execute \\\n  --bundle " + str(output / "bundle_PENDING_AUTHORIZATION.json") + " \\\n  --budget " + str(output / "budget_proposal_NOT_AUTHORIZED.json") + " \\\n  --model-config " + str(output / "model_config_PENDING_AUTHORIZATION.json") + " \\\n  --authorization /ABS/SEPARATELY_AUTHORIZED_CURRENT_IMPLEMENTATION.json \\\n  --run-dir /ABS/NEW_AUTHORIZED_REAL_RUN_DIRECTORY\n"
    save(output, "UNEXECUTED_COMMAND_REQUIRES_AUTHORIZATION.md", "# 未执行，需另行授权\n\n未生成正式 authorization；本提案不能自动执行。\n\n```bash\n" + command + "```\n")
    save(output, "final_report.md", "# T1-T7 增量扩充准备修复\n\n" +
        "\n".join(f"- {key}: {value}" for key, value in statuses.items()) +
        f"\n\n本次 unittest {result.testsRun} 项，失败 {len(result.failures)}，错误 {len(result.errors)}；QA revision-1 {len(qa_tests)} 组。保护文件 {len(before)} 项，变化 {len(changed_protected)}。\n\n" +
        f"完整143 turns、3 continuation、8 new candidates及必要绑定保留；registry仅精确重复值改为引用。原反馈经过正式bundle/workflow/最终HTTP构造逐字段一致。\n\n最终待授权增量messages：{report['final_supplement_tokens']} tokens；输出8192、思考0、安全余量256；本地工作窗口128000，服务器NOT_TESTED。\n\n" +
        "A/B/C完整messages与分块计数见T4文件；逐批审计已知材料/未来原始响应/安全余量分列，生成后再校验。\n\n" +
        "预算提案：增量Planner≤1、Gate≤8、Realizer≤4、adherence≤4，最多17个真实请求；另1个本地closure预留槽，不是模型调用。历史正常Planner1次单列。额度不是产出保证。\n\n" +
        "实现与预算解释见T1_T7_implementation_and_budget.md；逐文件diff、完整字段映射、HTTP错误样例、保护SHA及未执行命令均已保存。\n\n" +
        "所有真实模型/记忆/训练调用0，生产registry未创建、唯一补充机会未消费。官方公开页面查阅不涉及认证API或容量探测。旧QA provenance仍是单独未决项；不声称本次恢复旧QA版本。\n\n" +
        "停止，等待预算授权。\n")
    save(output, "artifact_sha256.json", {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()})
    print(json.dumps({"offline_ready": bool(passed), "tests": result.testsRun, "external_model_calls": 0,
                      "supplement_tokens": report["final_supplement_tokens"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-baseline", action="store_true")
    args = parser.parse_args()
    if args.capture_baseline:
        baseline(args.output)
    else:
        raise SystemExit(run(args.output))
