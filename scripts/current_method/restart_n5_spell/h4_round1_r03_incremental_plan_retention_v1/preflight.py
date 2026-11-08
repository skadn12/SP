"""Generate new, explicitly OFFLINE diagnostic delivery; never overwrite runs."""
import argparse
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from . import POLICY_ID
from .fixtures import PROJECT, FAILED, PRIOR, failed_bundle, example_budget, FixtureBackend
from .planning import PROFILE, sha
from .prompts import SUPPLEMENT_TEMPLATE, TARGET_RULES, REALIZER_TEMPLATE, SHARED_FEEDBACK_RULES_V3, render_initial_planner
from .retention import ADHERENCE_TEMPLATE
from .run import implementation_manifest, validate_frozen_sources, offline_network_guard
from .workflow import execute_policy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_inventory():
    base = Path(__file__).resolve().parent.parent
    paths = []
    for name in ("simple_append_memorax_v1_memory_event_continuation_probe_v1",
                 "h4_round1_r03_feedback_view_v2", "h4_round1_r03_qa_output_contract_v1_1_revision_1"):
        paths.extend((base / name).glob("*.py"))
    paths.append(PRIOR)
    paths.append(base / "simple_append_v1/assembly.py")
    for directory in (FAILED, PROJECT / "restart-n5-runs/h4_round1_r03_feedback_view_v2_offline_preflight_live_27"):
        paths.extend(p for p in directory.rglob("*") if p.is_file())
    return {str(p): digest(p) for p in sorted(set(paths))}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        p = output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("x", encoding="utf-8") as handle:
            handle.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
    before = protected_inventory()
    save("protected_files_before.json", before)
    bundle = failed_bundle()
    validate_frozen_sources(bundle)
    save("profile.json", PROFILE)
    save("implementation_rules_and_unexecuted_commands.md", (Path(__file__).resolve().parent / "README.md").read_text())
    save("budget_EXAMPLE_NOT_AUTHORIZED.json", example_budget())
    save("frozen_input_OFFLINE_ONLY.json", bundle)
    save("templates/supplement_planner_v1.md", SHARED_FEEDBACK_RULES_V3 + "\n" + TARGET_RULES + SUPPLEMENT_TEMPLATE)
    save("templates/realizer_progress_retention_v1.md", TARGET_RULES + REALIZER_TEMPLATE)
    save("templates/adherence_retention_v1.md", TARGET_RULES + ADHERENCE_TEMPLATE)
    prefix = "scripts.current_method.restart_n5_spell."
    expansion = prefix + "simple_append_memorax_v1_memory_event_continuation_probe_v1."
    modules = [prefix + "h4_round1_r03_incremental_plan_retention_v1.test_policy"]
    modules += [expansion + name for name in ("test_memory_event_continuation", "test_planner_state_contract",
                "test_promotion", "test_scene_continuation", "test_scene_replacement", "test_realizer_adherence")]
    modules += [prefix + "h4_round1_r03_feedback_view_v2.test_feedback_view"]
    stream = io.StringIO()
    with offline_network_guard():
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        # The legacy QA revision tests are plain functions, not unittest cases.
        from ..h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight import test_revision
        schema_root = PROJECT / "restart-n5-runs/h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight_live_7"
        schema_files = sorted(schema_root.rglob("*.json"))
        schemas = {}
        for path in schema_files:
            obj = json.loads(path.read_text())
            if isinstance(obj, dict) and "$id" in obj:
                ident = obj["$id"]
                if ident == "h4.round1.r03.qa_feedback_output.v1.1.revision_1":
                    schemas["schema_targeted"] = obj
                elif ".fresh." in ident:
                    schemas["schema_fresh"] = obj
        qa_tests = []
        for name, func in inspect.getmembers(test_revision, inspect.isfunction):
            if name.startswith("test_"):
                try:
                    func(**{k: schemas[k] for k in inspect.signature(func).parameters})
                    qa_tests.append({"name": name, "status": "PASS"})
                except Exception as exc:
                    qa_tests.append({"name": name, "status": "FAIL", "error": type(exc).__name__})
        mock_report = execute_policy(bundle=bundle, budget=example_budget(), backend=FixtureBackend(bundle),
            state_registry=output / "mock_state_registry", artifact_dir=output / "offline_failed_plan_path")
        # Initial Planner rendering is an interface regression only; this entry
        # consumes the frozen plan and never schedules another initial Planner.
        from ..h4_round1_r03_feedback_view_v2.compiler import load_frozen_inputs
        loaded = load_frozen_inputs()
        record = next(r for r in loaded["records"] if r["root_id"] == bundle["identity"]["root_id"])
        dialogue = loaded["dialogues"][record["root_id"]]
        initial = render_initial_planner(record=record, dialogue=dialogue,
            memory_candidates=bundle["candidates"], verified_tracks=bundle["verified_tracks"])
        save("rendered/initial_planner_INTERFACE_ONLY_NOT_SCHEDULED.json", initial)
    save("tests.log", stream.getvalue())
    save("qa_revision_1_regression.json", qa_tests)
    after = protected_inventory()
    changed = [p for p in before if before[p] != after.get(p)]
    save("protected_files_after.json", after)
    identities = implementation_manifest()
    save("implementation_sha256.json", identities)
    checks = []
    for path in Path(__file__).resolve().parent.glob("*.py"):
        try:
            compile(path.read_text(), str(path), "exec")
            checks.append({"file": str(path), "in_memory_compile": "PASS"})
        except Exception as exc:
            checks.append({"file": str(path), "in_memory_compile": "FAIL", "error": type(exc).__name__})
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=PROJECT / "current_method_github", capture_output=True, text=True)
    save("git_diff_check.txt", diff_check.stdout + diff_check.stderr)
    save("compile_checks.json", checks)
    save("new_files.json", [str(p) for p in sorted(Path(__file__).resolve().parent.glob("*")) if p.is_file()])
    passed = result.wasSuccessful() and all(t["status"] == "PASS" for t in qa_tests) and not changed and mock_report["retained_scene_count"] == 8 and all(c["in_memory_compile"] == "PASS" for c in checks) and diff_check.returncode == 0
    report = {"status": "INCREMENTAL_PLAN_RETENTION_OFFLINE_PASS" if passed else "OFFLINE_CHECKS_WITH_GAPS",
              "policy_id": POLICY_ID, "unittest_count": result.testsRun,
              "unittest_failures": len(result.failures), "unittest_errors": len(result.errors),
              "unittest_skipped": len(result.skipped), "qa_test_groups": qa_tests,
              "protected_file_count": len(before), "protected_changes": changed,
              "mock_path": mock_report, "git_diff_check": diff_check.returncode,
              "all_external_calls": {k: 0 for k in ("Planner", "supplement_Planner", "Realizer", "Gate", "adherence", "QA", "Solver", "semantic_auditor", "training", "Memorax_Search", "Memorax_Status", "Memorax_Add", "Memorax_Mutation")},
              "project_existing_implementation_config_history_files_modified": 0,
              "new_package_only": True, "old_failed_plan_remains_failed": True,
              "NOT_TESTED": ["real endpoint / 8192 acceptance", "live HTTP transport with actual service",
                             "real incremental plan usefulness", "semantic correctness of audit judgments",
                             "new R03 quality or new facts", "all 606 repository tests", "previous protected-manifest failures not rebaselined"],
              "limitations": ["Current new entry consumes a frozen initial plan; initial Planner render is regression-only.",
                              "No new-policy repair/continuation/replacement channel: target-only shortfall never retries; non-target failures conservatively stop/quarantine.",
                              "Reservations are conservative, not refunded after a rejected Gate; this can defer more scenes.",
                              "Budget example uses generous mock limits, not the real server capacity or spending authorization."]}
    save("final_offline_report.json", report)
    save("final_offline_report.md", "# 增量计划与未完成目标保留：纯离线交付\n\n"
         + f"状态：`{report['status']}`\n\n"
         + f"受影响 unittest：{result.testsRun}，失败 {len(result.failures)}，错误 {len(result.errors)}；QA revision-1 独立测试组：{len(qa_tests)}。\n\n"
         + f"历史/旧实现逐文件 SHA 对照：{len(before)} 项，变化 {len(changed)}。全部真实外部调用 0。\n\n"
         + "模拟路径：旧计划依然保留 future-only 失败判定；新 policy 分离目标诊断，追加 2 个同事件阶段；初始 6 场 + 补充 2 场，4 次 mock Realizer 批次，目标仍为 partial。仅证明接口，不是真实 R03。\n\n"
         + "模板见 templates/；实际 mock 入口 messages、原始 mock 响应、校验与完整账本见 offline_failed_plan_path/。\n\n"
         + "当前入口只从冻结初始计划继续；不增加非目标类自动恢复。预算预留保守，未用额度不退回。实时服务、8192 接受情况、真实内容与审计语义准确性均 NOT_TESTED。\n\n"
         + "真实运行前仍需单独授权，填写完整预算、独立 branch、模型参数、本地 tokenizer 身份及全套 SHA；不能把示例预算/fixture 当授权。旧 SHA 保护基线不改。\n\n"
         + "停止，等待后续授权。\n")
    save("artifact_sha256.json", {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps({"status": report["status"], "tests": result.testsRun, "qa_groups": len(qa_tests), "external_calls": 0}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
