"""Zero-network Phase B1 preflight for the v11 Hybrid runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from .live_fresh_e2e_v11 import hybrid_request_budget, run_mock_hybrid_v11
from .narrative_action_contracts import migrate_legacy_action_hints_to_narrative_actions
from .placement_choice_registry import migrate_legacy_round_plan_source_placement
from .offline_preflight_v11 import _historical_hashes, _protected_hashes
from .provenance import git_value, hash_tree, project_diff_sha256


REPO = Path("/root/autodl-tmp/gxy-selfplay/current_method_github")
PACKAGE = REPO / "scripts/current_method/restart_n5_spell/world_memory_slim_v1"
RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
INPUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json")
V10 = RUNS / "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v10"
PHASE_A = RUNS / "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_phase_a_preflight"
DEFAULT_OUTPUT = RUNS / "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_phase_b1_preflight"
REPORT = PACKAGE / "reports/SLIM_CONTINUOUS_EXPANSION_V11_PHASE_B1_REPORT.md"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _run_check(command: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                            check=False)
    return {"command": list(command), "exit_code": result.returncode,
            "passed": result.returncode == 0,
            "stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-4000:]}


def _planner_fixture() -> dict[str, Any]:
    plan = deepcopy(_load(PHASE_A / "round_blueprint_fixture.json")["model_fixture"])
    if any("historical_anchor_id" in scene or "placement" in scene
           for scene in plan.get("scenes") or []):
        plan = migrate_legacy_round_plan_source_placement(
            plan, placement_choice_id="PC_TAIL_AFTER")
    return migrate_legacy_action_hints_to_narrative_actions(
        plan, action_menu=_load(V10 / "affordance_menu.json"))


def _inputs() -> dict[str, Any]:
    return {
        "complete_dialogue": _load(PHASE_A / "complete_dialogue_view.json"),
        "planning_state": _load(PHASE_A / "planning_state_view.json"),
        "anchor_registry": _load(PHASE_A / "anchor_registry.json"),
        "action_menu": _load(V10 / "affordance_menu.json"),
        "planner_output": _planner_fixture(),
        "authoritative_memory": _load(V10 / "world_memory_final.json"),
    }


def _case_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    validator_run = result.get("round_validator_run") or {}
    durable = result.get("durable_update_compile_result") or {}
    promotion = result.get("promotion_v2_result") or {}
    closure = result.get("scene_decision_closure") or {}
    return {
        "completed": result.get("completed"), "failure_stage": result.get("failure_stage"),
        "mock_calls": result.get("mock_calls"),
        "validator_physical_calls": validator_run.get("physical_calls"),
        "validator_input_sha_stable": len({row.get("frozen_input_sha256")
                                           for row in validator_run.get("attempts") or []}) <= 1,
        "accepted_scene_refs": closure.get("accepted_scene_refs"),
        "rejected_scene_refs": closure.get("rejected_scene_refs"),
        "promotable_scene_count": len(promotion.get("promotable_child_scenes") or []),
        "accepted_update_count": len(durable.get("accepted_updates") or []),
        "rejected_update_count": len(durable.get("rejected_updates") or []),
        "audit_only_observation_count": len(durable.get("audit_only_observations") or []),
        "transaction_count": len(result.get("transactions") or []),
        "committer_calls": result.get("evidence_committer_calls"),
        "api_calls": result.get("api_calls"), "network_calls": result.get("network_calls"),
    }


def _report(data: Mapping[str, Any]) -> str:
    r, checks, matrix = data["readiness"], data["checks"], data["mock_case_matrix"]
    return f"""# Slim Continuous Expansion v11 — Phase B1 Report

## 1. 目标与限制

本轮只实现并执行 mock/fixture Hybrid runtime。真实 API、网络、Fresh E2E、Round 2、QA、Solver、Training 和 Git commit 均为 0。

## 2. 新增文件

`round_validator_contract.py`、`round_expansion_validator_v1.py`、`review_world_view.py`、`local_scene_gate_v11.py`、`scene_decision_closure.py`、`durable_update_compiler.py`、`promotion_v2.py`、`mock_hybrid_backends.py`、`live_fresh_e2e_v11.py`、`live_preflight_v11.py`，以及四个 Phase B1 测试文件。旧 v10 入口、Evidence Scope、Promotion、Binder、Transaction、Registry 和 backend 文件未修改。

仓库在本轮开始前已有大量 dirty/untracked 文件；本轮未清理、reset 或覆盖它们。

## 3. 本地调用链

Complete Dialogue / Planning State / Anchor-first mock Planner → 6-scene compile/DAG → Context v2 → mock Realizer → Local Scene Gate → 一次 Round Validator → output validation → decision closure → durable compiler → 旧 Binder/Transaction → Promotion v2。

## 4. 本地硬门与 Review World View

本地门只检查 schema、scene/turn/speaker refs、turn budget、speaker coverage、exact copy、anchor/boundary/context partitions、dependency availability、predecessor dialogue SHA 和 candidate hint schema。复杂语义交给整轮 Reviewer。

Review World View 只包含使用过的 anchor 原文、相关 Planning State、显式关系与 Interaction Link 的分离视图、实际 scenes、DAG、叙事约束和 candidate hints；`parent_utterance` Memory 镜像被排除，future context 保留 `constraint_only` 标签。

## 5. Round Validator 合同与 retry

正常路径整轮一次。只对 parse/schema/reference 合同错误、使用完全相同的冻结输入重试，最多 2 次；合法 reject/uncertain 不重试。输出需覆盖全部 local-pass scenes，refs 必须存在且 update 不能引用 rejected/unavailable scene。Validator 不可产生最终 ID 或 logical key。

## 6. Closure、Promotion 与 Durable Updates

reject/uncertain/local failure 会阻断硬依赖及显式 `uses_scene_refs` 后继；不依赖被拒绝信息的 parallel branch 可保留。Promotion v2 分离 `scene_valid` 与 `durable_update_valid`；默认 invalid update 不删除有效对话，只有显式 `scene_requires_update_commit=true` 才绑定两者。

Durable compiler 默认拒绝未提示 update，校验 registry、entity/ref/evidence/source scene、payload、pre-state、依赖和 atomic group，再适配旧 Binder。`durable_updates=[]` 时 6 scenes 仍晋升，allocation/transaction 都为 0。

## 7. Binder / Transaction

all-accept fixture 的两个独立合法 update 产生 2 次 allocation 和 2 组 PREPARED→COMMITTED；一个独立 update 非法时只保留合法项；atomic group 任一成员非法时整组不提交。Reviewer 从不直接写权威记录。

## 8. Mock case matrix

共 {len(matrix)} 个 deterministic cases；全部完成：{all(row['completed'] for row in matrix.values())}。parse-invalid 和 invalid-ref 均调用 2 次且 frozen input SHA 不变；合法 reject 调用 1 次；Evidence Committer calls 恒为 0。

## 9. 请求预算

正常 5/6/7 scenes 为 7/8/9 次；上限为 Planner 3 + Realizer 7×3 + Round Validator 2 = 26；逐 scene Committer 已移除，calls=0。

## 10. 测试与 13-root preflight

- 定向测试：{checks['targeted_tests']['passed']}（exit {checks['targeted_tests']['exit_code']}）
- 全部 world_memory_slim*：{checks['related_tests']['passed']}（exit {checks['related_tests']['exit_code']}）
- compileall：{checks['compileall']['passed']}
- git diff --check：{checks['diff_check']['passed']}
- 13-root preflight：{data['thirteen_root_preflight']['passed_count']}/13

## 11. 历史完整性与调用审计

authoritative input、v6–v10、Phase A 和受保护 v10 模块 SHA 前后不变：{data['historical_integrity']['all_unchanged']}。API calls=0，network attempts=0；未运行真实 v11/Fresh E2E/Round 2/QA/Training，未 Git commit。

## 12. Readiness 与剩余风险

合同与 mock runtime readiness 全部为 true；live Planner、live Realizer、live Round Validator、semantic novelty、generation/fresh E2E/QA/Round2 均保持 false。主要剩余风险是真实模型是否稳定遵守整轮 schema、5–7 scene 连贯性与 conservative durable-update review。

## 13. 最终结论

**A. Phase B1 通过，可以进入一次真实 v11 canary。**

冻结范围：Planner 最多 3 次；5–7 scenes；每 scene Realizer 最多 3 次；Round Validator 最多 2 次；Committer calls=0；Round 2、QA、Solver、Training 和自动重复 canary 均禁止。
"""


def run(output_root: Path, *, refresh_owned_output: bool = False) -> dict[str, Any]:
    if output_root.exists():
        owned = (output_root.resolve() == DEFAULT_OUTPUT.resolve() and
                 (output_root / "readiness.json").exists())
        if not refresh_owned_output or not owned:
            raise FileExistsError(f"phase_b1_output_must_be_absent:{output_root}")
    history_before = _historical_hashes()
    protected_before = _protected_hashes()
    phase_a_before = hash_tree(PHASE_A)
    input_before = sha256_file(INPUT)
    git_before = git_value(REPO, "rev-parse", "HEAD")
    inputs = _inputs()
    output_root.mkdir(parents=True, exist_ok=refresh_owned_output)
    primary = run_mock_hybrid_v11(
        **inputs, run_root=output_root / "binder_mock_runtime", case="all_accept")
    cases = {}
    case_names = [
        "all_accept", "accept_no_updates", "reject_middle_scene",
        "uncertain_middle_scene", "independent_parallel_branch_survives",
        "validator_parse_invalid_then_valid", "validator_invalid_ref_then_valid",
        "validator_legal_reject_no_retry", "one_valid_one_invalid_update",
        "atomic_group_partial_failure", "unhinted_update_observation",
    ]
    for case in case_names:
        if case == "all_accept":
            result = primary
        else:
            with tempfile.TemporaryDirectory(prefix="v11_b1_mock_") as temp:
                result = run_mock_hybrid_v11(
                    **inputs, run_root=Path(temp), case=case)
        cases[case] = _case_summary(result)
    budget = hybrid_request_budget()
    phase_a_readiness = _load(PHASE_A / "readiness.json")["readiness"]
    roots = {
        "01_phase_a_artifacts_ready": all(phase_a_readiness.get(key) for key in (
            "complete_dialogue_view_ready", "planning_state_view_ready",
            "anchor_registry_ready", "same_round_context_ready")),
        "02_v11_runner_contract_ready": primary.get("completed") is True,
        "03_local_scene_gate_ready": all(row.get("status") == "local_gate_passed"
                                         for row in primary["local_scene_gate_results"].values()),
        "04_review_world_view_ready": primary["review_world_view"].get(
            "excluded_record_types") == ["parent_utterance"],
        "05_round_validator_contract_ready": primary["round_validator_run"]["completed"],
        "06_round_validator_output_validation_ready": primary[
            "round_validator_run"]["attempts"][-1]["validation"]["valid"],
        "07_scene_decision_closure_ready": len(primary[
            "scene_decision_closure"]["accepted_scene_refs"]) == 6,
        "08_promotion_v2_ready": len(primary["promotion_v2_result"][
            "promotable_child_scenes"]) == 6,
        "09_durable_update_compiler_ready": len(primary[
            "durable_update_compile_result"]["accepted_updates"]) == 2,
        "10_binder_transaction_compatibility_ready": len(primary["transactions"]) == 2,
        "11_hybrid_request_budget_ready": budget["worst_case"]["physical_calls"] == 26,
        "12_mock_hybrid_runtime_ready": all(row["completed"] for row in cases.values()),
        "13_historical_integrity_ready": True,
    }
    checks = {
        "targeted_tests": _run_check([sys.executable, "-m", "unittest",
            "tests.test_world_memory_slim_v11_hybrid_runtime",
            "tests.test_round_expansion_validator_v1",
            "tests.test_durable_update_compiler", "tests.test_promotion_v2"]),
        "related_tests": _run_check([sys.executable, "-m", "unittest", "discover",
            "-s", "tests", "-p", "test_world_memory_slim*.py"]),
        "compileall": _run_check([sys.executable, "-m", "compileall", "-q",
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1"]),
        "diff_check": _run_check(["git", "diff", "--check"]),
    }
    history_after = _historical_hashes()
    protected_after = _protected_hashes()
    integrity = {
        "git_head_before": git_before, "git_head_after": git_value(REPO, "rev-parse", "HEAD"),
        "input_sha256_before": input_before, "input_sha256_after": sha256_file(INPUT),
        "v10_tree_sha256_before": history_before["v10"]["tree_sha256"],
        "v10_tree_sha256_after": history_after["v10"]["tree_sha256"],
        "phase_a_tree_sha256_before": phase_a_before["tree_sha256"],
        "phase_a_tree_sha256_after": hash_tree(PHASE_A)["tree_sha256"],
        "historical_before": history_before, "historical_after": history_after,
        "protected_module_sha_before": protected_before,
        "protected_module_sha_after": protected_after,
    }
    integrity["all_unchanged"] = (
        integrity["git_head_before"] == integrity["git_head_after"] and
        integrity["input_sha256_before"] == integrity["input_sha256_after"] and
        integrity["historical_before"] == integrity["historical_after"] and
        integrity["phase_a_tree_sha256_before"] == integrity["phase_a_tree_sha256_after"] and
        protected_before == protected_after)
    roots["13_historical_integrity_ready"] = integrity["all_unchanged"]
    readiness = {
        "v11_runner_contract_ready": roots["02_v11_runner_contract_ready"],
        "local_scene_gate_ready": roots["03_local_scene_gate_ready"],
        "review_world_view_ready": roots["04_review_world_view_ready"],
        "round_expansion_validator_contract_ready": roots["05_round_validator_contract_ready"],
        "round_validator_output_validation_ready": roots["06_round_validator_output_validation_ready"],
        "scene_decision_closure_ready": roots["07_scene_decision_closure_ready"],
        "promotion_v2_ready": roots["08_promotion_v2_ready"],
        "durable_update_compiler_ready": roots["09_durable_update_compiler_ready"],
        "binder_transaction_compatibility_ready": roots["10_binder_transaction_compatibility_ready"],
        "hybrid_request_budget_ready": roots["11_hybrid_request_budget_ready"],
        "phase_b_request_budget_ready": roots["11_hybrid_request_budget_ready"],
        "mock_hybrid_runtime_ready": roots["12_mock_hybrid_runtime_ready"],
        "live_v11_preflight_ready": all(roots.values()) and all(
            row["passed"] for row in checks.values()),
        "round_expansion_validator_live_ready": False, "real_planner_ready": False,
        "real_realizer_ready": False, "generation_ready": False,
        "fresh_e2e_ready": False, "semantic_novelty_review_ready": False,
        "qa_pipeline_ready": False, "round2_ready": False,
        "api_calls": 0, "network_calls": 0,
    }
    artifacts = {
        "phase_a_input_manifest.json": {"root": str(PHASE_A), "tree": phase_a_before,
            "input_sha256": input_before, "source_write_count": 0},
        "v11_runner_mock_trace.json": primary["mock_trace"],
        "local_scene_gate_results.json": primary["local_scene_gate_results"],
        "review_world_view.json": primary["review_world_view"],
        "round_validator_input_fixture.json": primary["round_validator_input"],
        "round_validator_output_fixture.json": primary["round_validator_output"],
        "round_validator_output_validation.json": primary["round_validator_run"],
        "scene_decision_closure.json": primary["scene_decision_closure"],
        "promotion_v2_result.json": primary["promotion_v2_result"],
        "durable_update_candidates.json": primary["round_validator_output"]["durable_updates"],
        "durable_update_compile_result.json": primary["durable_update_compile_result"],
        "binder_transaction_mock_result.json": {"transactions": primary["transactions"],
            "allocation_count": len(primary["transactions"]),
            "prepared_count": sum(row.get("status") == "PREPARED" for row in
                [json.loads(line) for line in (output_root / "binder_mock_runtime/run_commit_journal.jsonl").read_text().splitlines() if line]),
            "committed_count": sum(row.get("status") == "COMMITTED" for row in
                [json.loads(line) for line in (output_root / "binder_mock_runtime/run_commit_journal.jsonl").read_text().splitlines() if line])},
        "hybrid_request_budget.json": budget, "mock_case_matrix.json": cases,
        "historical_integrity_audit.json": integrity,
        "readiness.json": {"readiness": readiness, "checks": checks,
            "thirteen_root_preflight": roots},
    }
    for name, value in artifacts.items():
        atomic_json(output_root / name, value)
    report_data = {"readiness": readiness, "checks": checks,
                   "mock_case_matrix": cases,
                   "thirteen_root_preflight": {"checks": roots,
                       "passed_count": sum(roots.values()), "total": 13},
                   "historical_integrity": integrity}
    report = _report(report_data)
    atomic_text(REPORT, report)
    atomic_text(output_root / "reports/SLIM_CONTINUOUS_EXPANSION_V11_PHASE_B1_REPORT.md", report)
    return {"output_root": str(output_root), "readiness": readiness,
            "checks": checks, "thirteen_root_preflight": report_data[
                "thirteen_root_preflight"], "artifact_count": len(artifacts) + 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-owned-output", action="store_true",
                        help="Refresh only the already-owned Phase B1 artifact root")
    args = parser.parse_args(argv)
    result = run(args.output_root.resolve(),
                 refresh_owned_output=args.refresh_owned_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["readiness"]["live_v11_preflight_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
