"""Prepare the two-root Planner repair from frozen R1 inputs, with zero APIs.

Runs the SAME execute_round up to its first model callback. No live transport
is constructed, and every other model/Memory mutation callback fails closed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .incremental_round import load_incremental_round_input
from .live import _load_feedback_context
from .profiles import DEVELOPMENT_GAIN_PROFILE
from .workflow import execute_round

DEFAULT_SOURCE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_r1_aligned_feedback_r02_generation_corrected_live_1")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_development_gain_v2_prepare_2")
ROOTS = ("cc-train-episode-198480", "cc-train-episode-122711")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class PreparedBoundary(Exception):
    """Intentional local stop immediately before the first model request."""


class FrozenMemory:
    def __init__(self, snapshot):
        self.snapshot = deepcopy(snapshot)
        self.calls = []  # Actual Memory API calls; local reads are separate.
        self.local_read_count = 0

    def search_buckets(self, *args, **kwargs):
        self.local_read_count += 1
        return {"success": True, "data": {"buckets": deepcopy(self.snapshot["buckets"])}}

    def add(self, *args, **kwargs):
        raise RuntimeError("offline_preparation_forbids_memory_write")

    def add_status(self, *args, **kwargs):
        raise RuntimeError("offline_preparation_forbids_memory_poll")


def prepare(*, source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError("prepare_output_exists; use_a_new_directory")
    output.mkdir(parents=True)
    manifests = []
    profile = DEVELOPMENT_GAIN_PROFILE
    for root in ROOTS:
        src = Path(source) / root / "attempt-02"
        old_freeze = json.loads((src / "run_freeze_manifest.json").read_text())
        parent_path = Path(old_freeze["input_path"])
        if _sha(parent_path) != old_freeze["input_sha256"]:
            raise ValueError(f"{root}:historical_R1_input_sha_mismatch")
        snapshot_path = src / "memory/recovery_pre_generation_bucket_snapshot.json"
        snapshot = json.loads(snapshot_path.read_text())
        if snapshot["root_id"] != root:
            raise ValueError("frozen_memory_root_mismatch")
        parent, prior_expanded, _ = load_incremental_round_input(parent_path, round_index=2)
        feedback_path = Path(old_freeze["feedback_input"]["path"])
        if _sha(feedback_path) != old_freeze["feedback_input"]["sha256"]:
            raise ValueError(f"{root}:historical_feedback_sha_mismatch")
        feedback_record, feedback_dialogue = _load_feedback_context(
            parent, feedback_input=feedback_path)
        artifacts = {}

        def sink(name, value):
            artifacts[name] = deepcopy(value)

        def at_planner(prompt):
            raise PreparedBoundary()

        def forbidden(*args, **kwargs):
            raise AssertionError("offline_preparation_must_not_call_a_model")

        gateway = FrozenMemory(snapshot)
        try:
            execute_round(parent=parent, prior_expanded=prior_expanded,
                memorax=gateway, verifier_call=forbidden, planner_call=at_planner,
                semantic_gate_call=forbidden, outline_semantic_call=forbidden,
                realizer_call=forbidden, adherence_call=forbidden,
                reuse_existing_namespace=True, no_memorax_writes=True,
                memorax_user_namespace=snapshot["user_id"].rsplit(":" + root, 1)[0], round_index=2,
                expose_complete_memory_to_planner=True,
                feedback_record=feedback_record, feedback_dialogue=feedback_dialogue,
                execution_profile=profile, artifact_sink=sink)
        except PreparedBoundary:
            pass
        prompt = artifacts["planner/prompt"]
        original_prompt = (src / "planner/prompt.txt").read_text()
        if prompt["prompt"].split("\n\nDEVELOPMENT QUALITY POLICY v2")[0] != original_prompt:
            raise ValueError(f"{root}:base_prompt_changed_beyond_policy_appendix")
        target = output / root
        _write(target / "input_R1.json", json.loads(parent_path.read_text()))
        _write(target / "frozen_memory.json", snapshot)
        _write(target / "profile.json", profile.to_dict())
        _write(target / "feedback_input.json", json.loads(feedback_path.read_text()))
        (target / "prompt.txt").write_text(prompt["prompt"])
        _write(target / "prompt_manifest.json", {k: v for k, v in prompt.items() if k != "prompt"})
        for name in ("planner/speaker_registry", "memory/event_candidates",
                     "event_verifier/materialized_tracks", "memory/global_state_view"):
            _write(target / (name.replace("/", "_") + ".json"), artifacts[name])
        source_files = [parent_path, snapshot_path, feedback_path, src / "run_freeze_manifest.json",
                        src / "planner/prompt.txt"]
        manifests.append({"root_id": root,
            "input_rounds": "P plus R01; before historical R02",
            "parent_turn_count": len(parent["dialogue"]["turns"]),
            "frozen_feedback_retained": True,
            "base_prompt_identical_to_historical": True,
            "sources": [{"path": str(p), "sha256": _sha(p)} for p in source_files],
            "new_prompt_sha256": _sha(target / "prompt.txt"),
            "old_prompt_sha256": _sha(src / "planner/prompt.txt"),
            "prompt_characters": len(prompt["prompt"]),
            "additional_memory_fact_count": prompt.get("additional_memory_fact_count", 0),
            "local_memory_reads": gateway.local_read_count,
            "external_model_calls": 0, "external_memory_calls": 0})
    module_dir = Path(__file__).parent
    manifest = {"status": "OFFLINE_PREPARED_NOT_AUTHORIZED_FOR_LIVE",
        "profile": profile.to_dict(), "roots": manifests,
        "model_calls_per_root_normal": 5,
        "model_calls_per_root_maximum": profile.worst_case_requests()["total_model_requests"],
        "model_calls_two_roots_maximum": 2 * profile.worst_case_requests()["total_model_requests"],
        "budget_components": profile.worst_case_requests(),
        "external_calls": 0,
        "code": [{"path": str(p), "sha256": _sha(p)} for p in sorted(module_dir.glob("*.py"))]}
    _write(output / "manifest.json", manifest)
    (output / "README.md").write_text("""# Planner development_gain_v2 离线准备

本目录仅复用两个 root 的真实 R02 之前 P+R01 和冻结 R1 Memory，外部调用 0。
请求通过正式 `execute_round` 路径准备，在首个 Planner callback 前停止。
未重复使用加日期R02排序试验全文，也未把历史Plan当成已发生事实。

独立 Planner 修复实验，不能混入 Writer 12 primary+12 repair 预算。
每 root 正常 5 次模型调用（Planner、Gate、outline、Realizer、adherence），
最多一次场景补全及一次复核，故最多 7；两 root 最多 14。
本profile显式关闭为凑场景而做的Planner repair和跨候选replacement。
没有额外模型阶段；不足三场不补采。真实HTTP上限和transport重试策略
尚未在本次离线准备授权，不能把这个逻辑预算当作已授权HTTP预算。

这是修复组合：提示与输入视图已变化，不声称与旧路径messages相同。
Planner保留历史真实可见输入：完整P+R01原文、既有Memory视图及冻结反馈。
198480为96条、122711为101条；不添加新Memory事实、不重复补整表。
Gate仅看同一冻结Memory；Realizer仍为text-free。配置名memory_only_fast
是跳过verifier的历史开关，不能用来断言Planner没有看原文。
Gate输出预算2048→8192、outline4096→8192；Planner保持4096。
Memory-only语义判断仍可能出错，精确引文验证不等于内容蕴含验证。
Gate UNKNOWN延后交给信息增量/必要前提检查，不自动认定完成或未完成。

离线准备入口（现有输出不会覆盖）：
`python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_prepare --output NEW_OUTPUT_DIR`

同流程mock与CLI参数回归：
`python -m unittest scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.test_development_policy -v`

正式原入口已支持 `live --profile development_gain_v2`。当前仅交付离线产物，
真实运行尚未授权。新profile的live入口有显式guard，在创建run或transport前
阻止执行，直到冻结Memory适配和HTTP上限得到实现及授权；不得用旧远端Memory
和可能自动重试的transport冒充这份冻结准备。
""")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = prepare(source=args.source, output=args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output), "external_calls": 0,
                      "max_model_calls_if_later_authorized": result["model_calls_two_roots_maximum"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
