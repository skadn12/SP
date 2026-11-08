"""Frozen previous-training-round feedback for the R02 expansion pilot.

This module is offline. It never modifies V4, discovers a latest run, takes
current-round answers, or infers the cause of a Solver error. The complete
cumulative transcript must still accompany every expansion request.

compile_feedback(outdir, roots) produces four route/role projections per root.
append_feedback_to_request returns a copy plus a receipt proving the exact
projection is present in the outgoing user text. Its expected target identity
must come from the caller's frozen experiment manifest.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


VERSION = "h4_v91_expansion_feedback_projection"
PROJECT = Path("/root/autodl-tmp/gxy-selfplay")
HERE = Path(__file__).resolve().parent
DEFAULT_V4_DIR = HERE / "compiled_history_v91"
TARGET = "M_clean_policy"
ROUTES = ("old_event_continuation", "dataset_new_event")
ROLES = ("planner", "realizer")
MAX_SUFFIX_CHARS = 7000
V4_MANIFEST_SHA256 = "b856c128819c4c51762d7884714b7166f162d0118fa11b9a940111fdcbd2eb72"
V4_SOURCE_LOCK_SHA256 = "ef171b19562fb7ece44388f079cb132233e008c24a849b271450e00340f8506c"
# These historical source transcripts supplement V4's IDs with quoted evidence.
TRANSCRIPT_SHA256 = {'cc-train-episode-14304': 'fc727c95e3203a0a8e16cf4c3d562835d63538c8dab564d97410816310f5eef0', 'cc-train-episode-19053': 'd594cf3cf945b2e520c32c8e41f4fa1b369f9fb91f5b80ab41646a575bd576aa', 'cc-train-episode-19708': '8d9d0041693f4081a630bd23506bc78ffc9923b136cfa88217c77302612b5754', 'cc-train-episode-20432': '843e66466840d95649d9e7bf01067c0a0b9f869ad9b65760cc98aa602b11f293', 'cc-train-episode-24719': '1ef0b70c4e7315b4ce08e752d72822310e965960fc65933973ab59361fb2d29b', 'cc-train-episode-25652': 'd8d771741a135001a53d11869090f268a297a615bdf1e4c1840ca33a3acb114b', 'cc-train-episode-38601': '433a1c37ce9f35ba5c9ba582c03ca171dee0583b79fabfa309845174a1639f31', 'cc-train-episode-9557': '8cec9c86885849ee90f68eda19461bf24b0b9eb2e128412df1658ac526a0fba7', 'cc-train-episode-104814': '9e442762faa7b70e1058395da5f5a2a8e48b0030606deb53149bf508fc019702', 'cc-train-episode-194259': '032b46863500a89945e69b364a95b47385fa312bc44674af2d1ba5b66753914f'}
SOURCE_REGISTRY_SHA256 = "2e217792d368f8a54d772be274246926d6bc0e28afd22ac4d0685b12a1e1e39a"
HISTORICAL_REGISTRY_PATH = (PROJECT / "offline-deliveries" /
    "h4_planner_cardinality_expansion_batch_v91/expansion_engine/SOURCE_REGISTRY.json")



def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_file(path: Path, expected: str) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file(), f"missing_source:{path}")
    require(sha256_file(path) == expected, f"source_hash_mismatch:{path}")
    return {"path": str(path), "sha256": expected}


def _verify_artifact(artifact: Mapping[str, str]) -> dict[str, str]:
    return verify_file(Path(artifact["path"]), artifact["sha256"])


def _target_identity(lock: Mapping[str, Any], model: str) -> dict[str, Any]:
    require(model == TARGET, "target_solver_not_frozen_M_clean_policy")
    report = read_json(Path(lock["model_identity_source"]["path"]))
    contract = read_json(Path(lock["solver_checkpoint_source"]["path"]))
    training = read_json(Path(lock["formal_training_checkpoint_source"]["path"]))
    row = report["models"][model]
    require(training.get("status") == "FORMAL_TRAINING_COMPLETE", "training_not_complete")
    require(model in training["policy_reports"], "model_not_in_training_report")
    require(row.get("model_name") == model and row.get("adapter_enabled") is True,
            "trained_solver_identity_invalid")
    require(row["base_model_identity"] == contract["checkpoint_identity"],
            "base_checkpoint_identity_disagrees_with_contract")
    require(report["solver_contract_sha256"] == lock["solver_checkpoint_source"]["sha256"],
            "solver_contract_binding_mismatch")
    adapter = Path(row["adapter_path"])
    actual_adapter = {
        "config": verify_file(adapter / "adapter_config.json", row["adapter_config_sha256"]),
        "state": verify_file(adapter / "adapter_model.pt", row["adapter_state_sha256"]),
    }
    require(row["inference_model_path"] == row["merged_model_identity"]["path"],
            "merged_checkpoint_path_mismatch")
    # The complete recorded file identities are bound here, not merely a label.
    # Expansion does not load a Solver; large merged weights are not reread.
    return {
        "model_name": model,
        "model_fingerprint_sha256": row["model_fingerprint_sha256"],
        "inference_model_path": row["inference_model_path"],
        "adapter_composite_sha256": row["adapter_composite_sha256"],
        "adapter_files_verified": actual_adapter,
        "base_model_identity": row["base_model_identity"],
        "merged_model_identity": row["merged_model_identity"],
        "identity_report_source": lock["model_identity_source"],
        "formal_training_run_id": lock["formal_training_run_id"],
        "verification_scope": "locked_report_and_adapter_bytes; merged_weight_identities_bound_from_report; no_solver_loaded",
    }


def _load_v4(compiled_dir: Path, roots: list[str], model: str) -> dict[str, Any]:
    require(model == TARGET, "target_solver_not_frozen_M_clean_policy")
    compiled_dir = Path(compiled_dir).resolve()
    verify_file(compiled_dir / "COMPILED_FEEDBACK.json", V4_MANIFEST_SHA256)
    verify_file(compiled_dir / "source_lock.json", V4_SOURCE_LOCK_SHA256)
    manifest = read_json(compiled_dir / "COMPILED_FEEDBACK.json")
    lock = read_json(compiled_dir / "source_lock.json")
    require(roots and len(roots) == len(set(roots)), "root_scope_empty_or_duplicate")
    require(set(roots) <= set(manifest["root_order"]), "unknown_feedback_root")
    require(lock["merge_current_generation_feedback"] is False and
            lock["automatic_latest_run_selection"] is False, "feedback_not_previous_round_only")
    for key in ("model_identity_source", "solver_checkpoint_source", "formal_training_checkpoint_source"):
        _verify_artifact(lock[key])
    for artifact in lock["evaluation_sources"].values():
        _verify_artifact(artifact)
    verify_file(HERE / "SOURCE_REGISTRY.json", SOURCE_REGISTRY_SHA256)
    # The archived v91 feedback is retained byte for byte. Its registry path
    # stays bound to that historical delivery; the local registry must match
    # the same frozen contents rather than relabel the archive's provenance.
    verify_file(HISTORICAL_REGISTRY_PATH, SOURCE_REGISTRY_SHA256)
    require(lock["source_registry"] == {"path": str(HISTORICAL_REGISTRY_PATH.resolve()),
            "sha256": SOURCE_REGISTRY_SHA256}, "feedback_registry_binding_mismatch")
    target = _target_identity(lock, model)
    records = {row["root_id"]: row for row in manifest["records"]}
    contexts = {}
    for root in roots:
        record = records[root]
        root_dir = compiled_dir / root
        verify_file(root_dir / "candidate_history.json", record["candidate_history_sha256"])
        verify_file(root_dir / "aggregate.json", record["aggregate_sha256"])
        for role, variants in record["suffix_sha256"].items():
            for variant, expected in variants.items():
                verify_file(root_dir / f"{role}__{variant}_feedback.txt", expected)
        pool_artifact = _verify_artifact(lock["candidate_pools"][root])
        transcript_artifact = _verify_artifact(lock["historical_transcripts"][root])
        transcript_path = Path(transcript_artifact["path"])
        transcript_artifact = verify_file(transcript_path, TRANSCRIPT_SHA256[root])
        transcript = read_json(transcript_path)
        require(transcript["root_id"] == root, "transcript_root_mismatch")
        history = read_json(root_dir / "candidate_history.json")
        require(len(history) == record["candidate_count"], "history_count_mismatch")
        require(len({row["candidate_id"] for row in history}) == len(history), "duplicate_feedback_candidate")
        require(all(row["root_id"] == root and row["historical_question"]["root_id"] == root and model in row["models"] for row in history),
                "history_root_or_model_mismatch")
        contexts[root] = {
            "history": history, "aggregate": read_json(root_dir / "aggregate.json"),
            "transcript": transcript, "transcript_source": transcript_artifact,
            "candidate_history_source": {"path": str(root_dir / "candidate_history.json"),
                                         "sha256": record["candidate_history_sha256"]},
            "candidate_pool_source": pool_artifact,
        }
    return {"contexts": contexts, "source_lock": lock, "target_identity": target,
            "target_solver_identity_sha256": digest(target), "v4_dir": str(compiled_dir)}


def _clipped(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [excerpt truncated]"


def _examples(context: Mapping[str, Any], model: str) -> list[dict[str, Any]]:
    turns = {row["turn_ref"]: row for row in context["transcript"]["turns"]}
    selected = []
    for qtype in ("multi_select", "ordering"):
        rows = [r for r in context["history"] if r["question_type"] == qtype]
        # Prioritize lower historical agreement only as a deterministic sample
        # of feedback topics. No current admission gate or causal claim follows.
        rows.sort(key=lambda r: (r["models"][model]["full_context"]["accuracy"]
                                 if r["models"][model]["full_context"]["accuracy"] is not None else 2,
                                 r["candidate_id"]))
        seen = set()
        for row in rows:
            q = row["historical_question"]
            refs = list(dict.fromkeys(q.get("evidence_refs") or []))
            signature = tuple(sorted(refs))
            if signature in seen:
                continue
            seen.add(signature)
            resolved = [ref for ref in refs if ref in turns]
            quote_refs = list(dict.fromkeys(resolved[:1] + resolved[-1:]))
            history = row["models"][model]
            selected.append({
                "candidate_id": row["candidate_id"], "question_type": qtype,
                "historical_stem_unverified": _clipped(str(q.get("question", "")), 300),
                "historical_evidence_refs": refs[:4],
                "evidence_ref_count": len(refs),
                "unresolved_historical_refs": [ref for ref in refs if ref not in turns],
                "source_turn_excerpts": [
                    {"ref": ref, "speaker": turns[ref]["speaker"],
                     "text": _clipped(turns[ref]["text"], 220)} for ref in quote_refs
                ],
                "full_exact_agreement": {k: history["full_context"][k]
                                         for k in ("correct_count", "total_count")},
                "no_context_exact_agreement": {k: history["no_context"][k]
                                               for k in ("correct_count", "total_count")},
                "technical_failure_count": history["technical_failure_count"],
                "parse_failure_count": history["parse_failure_count"],
            })
            if len(seen) == 2:
                break
    require(selected, "no_MS_or_ordering_feedback_examples")
    return selected


COMMON = (
    "以下只来自冻结的上一轮正式训练后复测，绑定指定训练后 Solver；不会读取本轮生成或回答结果。",
    "历史题干、关系标签及答案未获语义认证；引用原文只证明说过这些话。full/no 是历史整题答案一致次数，不能推出错误原因、难度或训练收益。",
    "完整累计对话才是当前人物、事件和状态的事实来源；先核实历史引用与当前结尾，不能从题干、统计或外部数据集补出人物经历。",
    "以新增行动、条件、状态和后续结果增加可区分信息；保持计划、建议、尝试、完成的区别，清楚写出人物及对象。不要仅改写已知事实，也不要禁止继续已讨论过的主题。",
    "为后续六选多选与六项对话出现顺序题保留自然的可定位回合；不要生成题目、选项或答案，不照搬旧四项排序模板。",
    "历史 no-context、full_zero、上下文收益、整题全对率及换位差异均不形成准入门槛；本阶段不判断题目是否足够难。",
)
ROUTE_GUIDANCE = {
    "old_event_continuation": (
        "延续旧事件：从累计对话中选仍可发展的真实事件，写出原有状态与具体锚点，再发展新的触发、行动、条件变化和结果。",
        "历史低一致题及引用仅提示可检查哪些人物/状态/先后关系；不能断言模型混淆了其中某一关系。若事件已经结束，可以写合理后续，不能重置或篡改既有结果。",
    ),
    "dataset_new_event": (
        "数据集新事件：从历史样例抽象出可探索的关系形式，例如人物与行动对应、计划/执行/结果区分、相似尝试的条件变化、消息更新和不同回合间的关联；这些是生成方向，不是对历史失误原因的诊断。",
        "结合当前 root 生成独立的新事件。外部事件有明确多阶段依赖时按 scaffold 保留提纲关系；不足以支撑提纲时降为 seed 并自行发展。记录来源结构与新编部分，不能把来源人物、对白或结局当作本 root 事实。",
    ),
}
ROLE_GUIDANCE = {
    "planner": (
        "Planner：明确每场景的人物、初始状态、触发、行动、条件变化与预期结果；列出依据累计对话的锚点，声明哪些是新发展。场景之间要有状态接续。",
        "在计划里标注使用了哪些反馈样例或关系方向，以及具体改变了哪一场景；无法适配的样例可跳过并说明，不为凑数量强接无关事件。",
    ),
    "realizer": (
        "对话生成：落实已给计划与累计对话的当前状态；让触发、采取行动和后续结果在自然对话中实际出现。不能只反复承诺以后会做，也不能把建议写成已完成事实。",
        "人物、对象、条件和状态变化要能从真实回合区分，避免靠含糊代词制造复杂度。记录本场景实际落实的计划与反馈方向，不把尚未写入对话的计划算作证据。",
    ),
}


def _suffix(root: str, context: Mapping[str, Any], loaded: Mapping[str, Any],
            route: str, role: str, model: str) -> str:
    types = context["aggregate"]["by_question_type_and_model"]
    payload = {
        "version": VERSION, "root_id": root, "route": route, "role": role,
        "target_solver_model": model,
        "target_solver_identity_sha256": loaded["target_solver_identity_sha256"],
        "evaluation_run_id": loaded["source_lock"]["evaluation_run_id"],
        "selection_of_examples": "up_to_two_per_type; lower_historical_full_exact_first; distinct_ref_sets; diagnostic_sampling_only",
        "historical_type_counts": {q: {"candidate_count": types[q][model]["candidate_count"],
                                       "full_exact": types[q][model]["full_context"],
                                       "no_exact": types[q][model]["no_context"]}
                                   for q in ("multi_select", "ordering") if q in types},
        "guidance": [*COMMON, *ROUTE_GUIDANCE[route], *ROLE_GUIDANCE[role]],
        "historical_examples": _examples(context, model),
    }
    suffix = "\n[PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]\n" + canonical(payload) + "\n[/PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]\n"
    require(len(suffix) <= MAX_SUFFIX_CHARS, f"feedback_char_budget_exceeded:{root}:{route}:{role}:{len(suffix)}")
    return suffix


def compile_feedback(outdir: Path, roots: list[str], target_solver_model: str = TARGET,
                     compiled_v4_dir: Path = DEFAULT_V4_DIR) -> dict[str, Any]:
    """Compile a new immutable directory; never overwrite any existing output."""
    outdir = Path(outdir).resolve()
    require(not outdir.exists(), "compiled_output_exists_no_overwrite")
    roots = list(roots)
    loaded = _load_v4(compiled_v4_dir, roots, target_solver_model)
    generated = {}
    for root, context in loaded["contexts"].items():
        for route in ROUTES:
            for role in ROLES:
                generated[f"{root}/{route}__{role}.txt"] = _suffix(
                    root, context, loaded, route, role, target_solver_model)
    source_lock = {
        "version": VERSION, "compiled_v4_dir": loaded["v4_dir"],
        "v4_manifest_sha256": V4_MANIFEST_SHA256, "v4_source_lock_sha256": V4_SOURCE_LOCK_SHA256,
        "historical_source_lock": loaded["source_lock"],
        "merge_current_generation_feedback": False, "automatic_latest_run_selection": False,
        "root_sources": {root: {key: value[key] for key in
                         ("candidate_history_source", "candidate_pool_source", "transcript_source")}
                         for root, value in loaded["contexts"].items()},
    }
    lock_text = json.dumps(source_lock, ensure_ascii=False, indent=2) + "\n"
    manifest = {
        "version": VERSION, "status": "COMPILED_OFFLINE", "external_calls": 0,
        "roots": roots, "target_solver_model": target_solver_model,
        "target_solver_identity": loaded["target_identity"],
        "target_solver_identity_sha256": loaded["target_solver_identity_sha256"],
        "source_lock_sha256": sha256_bytes(lock_text.encode()),
        "records": [{"path": path, "sha256": sha256_bytes(text.encode()), "chars": len(text)}
                    for path, text in generated.items()],
    }
    outdir.mkdir(parents=True)
    (outdir / "source_lock.json").write_text(lock_text, encoding="utf-8")
    for relative, text in generated.items():
        path = outdir / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (outdir / "COMPILED_FEEDBACK.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_feedback_context(outdir: Path, root_id: str, target_solver_model: str = TARGET,
                          expected_target_identity_sha256: str | None = None) -> dict[str, Any]:
    outdir = Path(outdir).resolve()
    manifest = read_json(outdir / "COMPILED_FEEDBACK.json")
    require(manifest["version"] == VERSION, "wrong_feedback_version")
    require(root_id in manifest["roots"], "unknown_feedback_root")
    require(target_solver_model == TARGET == manifest["target_solver_model"],
            "target_solver_model_mismatch")
    verify_file(outdir / "source_lock.json", manifest["source_lock_sha256"])
    lock = read_json(outdir / "source_lock.json")
    loaded = _load_v4(Path(lock["compiled_v4_dir"]), [root_id], target_solver_model)
    require(lock["historical_source_lock"] == loaded["source_lock"], "historical_lock_changed")
    require(lock["v4_manifest_sha256"] == V4_MANIFEST_SHA256 and
            lock["v4_source_lock_sha256"] == V4_SOURCE_LOCK_SHA256 and
            lock["merge_current_generation_feedback"] is False and
            lock["automatic_latest_run_selection"] is False, "projection_source_lock_changed")
    for key in ("candidate_history_source", "candidate_pool_source", "transcript_source"):
        require(lock["root_sources"][root_id][key] == loaded["contexts"][root_id][key],
                "projection_root_source_changed")
    identity_sha = loaded["target_solver_identity_sha256"]
    require(identity_sha == manifest["target_solver_identity_sha256"] ==
            digest(manifest["target_solver_identity"]), "target_solver_identity_mismatch")
    if expected_target_identity_sha256 is not None:
        require(identity_sha == expected_target_identity_sha256, "caller_target_identity_mismatch")
    suffixes = {}
    records = {row["path"]: row for row in manifest["records"]}
    for route in ROUTES:
        for role in ROLES:
            relative = f"{root_id}/{route}__{role}.txt"
            verify_file(outdir / relative, records[relative]["sha256"])
            text = (outdir / relative).read_text(encoding="utf-8")
            expected = _suffix(root_id, loaded["contexts"][root_id], loaded, route, role, target_solver_model)
            require(text == expected, "projection_not_from_frozen_sources")
            suffixes[(route, role)] = text
    projections = {key: json.loads(text.splitlines()[2]) for key, text in suffixes.items()}
    return {"manifest": manifest, "source_lock": lock, "suffixes": suffixes, "projections": projections,
            "target_solver_identity_sha256": identity_sha}


def append_feedback_to_request(payload: Any, outdir: Path, root_id: str, route: str, role: str,
                               target_solver_model: str = TARGET,
                               expected_target_identity_sha256: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Copy payload and append a verified frozen suffix to its final user text."""
    require(route in ROUTES and role in ROLES, "unknown_route_or_role")
    require(expected_target_identity_sha256 is not None, "caller_target_identity_required")
    _assert_request_root(payload, root_id)
    context = load_feedback_context(outdir, root_id, target_solver_model, expected_target_identity_sha256)
    suffix = context["suffixes"][(route, role)]
    result = copy.deepcopy(payload)
    messages = result if isinstance(result, list) else result.get("messages") if isinstance(result, dict) else None
    if isinstance(messages, list):
        users = [message for message in messages if message.get("role") == "user"]
        require(users and isinstance(users[-1].get("content"), str), "request_has_no_text_user_message")
        owner, field = users[-1], "content"
    elif isinstance(result, dict) and isinstance(result.get("prompt"), str):
        owner, field = result, "prompt"
    elif isinstance(result, dict) and isinstance(result.get("user_prompt"), str):
        owner, field = result, "user_prompt"
    else:
        raise ValueError("unsupported_request_shape")
    require("[PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]" not in owner[field], "feedback_already_appended")
    before_sha = sha256_bytes(owner[field].encode())
    owner[field] += suffix
    require(owner[field].endswith(suffix), "feedback_missing_from_request")
    manifest = context["manifest"]
    receipt = {
        "version": VERSION, "root_id": root_id, "route": route, "role": role,
        "target_solver_model": target_solver_model,
        "target_solver_identity_sha256": context["target_solver_identity_sha256"],
        "evaluation_run_id": context["source_lock"]["historical_source_lock"]["evaluation_run_id"],
        "source_lock_sha256": manifest["source_lock_sha256"],
        "suffix_sha256": sha256_bytes(suffix.encode()), "suffix_chars": len(suffix),
        "before_user_text_sha256": before_sha,
        "request_user_text_sha256": sha256_bytes(owner[field].encode()),
        "request_payload_sha256": digest(result), "feedback_appended": True,
        "current_generation_feedback_used": False,
    }
    return result, receipt


def _assert_request_root(payload: Any, expected_root: str) -> None:
    """Reject a known root mismatch before appending another root's feedback."""
    if isinstance(payload, dict) and "root_id" in payload:
        require(payload["root_id"] == expected_root, "request_feedback_root_mismatch")
    messages = payload if isinstance(payload, list) else payload.get("messages") if isinstance(payload, dict) else None
    if isinstance(messages, list):
        users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        text = users[-1].get("content") if users else None
    elif isinstance(payload, dict):
        text = payload.get("prompt", payload.get("user_prompt"))
    else:
        text = None
    if isinstance(text, str) and "\nINPUT_JSON\n" in text:
        try:
            context = json.JSONDecoder().raw_decode(text.split("\nINPUT_JSON\n", 1)[1])[0]
        except (ValueError, TypeError) as error:
            raise ValueError("request_feedback_context_invalid") from error
        require(isinstance(context, dict) and context.get("root_id") == expected_root,
                "request_feedback_root_mismatch")


def input_paths() -> list[Path]:
    """All source and compiled bytes that must be bound before a live run."""
    _load_v4(DEFAULT_V4_DIR, list(TRANSCRIPT_SHA256), TARGET)
    lock = read_json(DEFAULT_V4_DIR / "source_lock.json")
    paths = {HERE / "feedback_adapter.py", HERE / "feedback_compiler_v91.py", HERE / "SOURCE_REGISTRY.json"}
    paths.update(p for p in DEFAULT_V4_DIR.rglob("*") if p.is_file())
    for key in ("model_identity_source", "solver_checkpoint_source", "formal_training_checkpoint_source"):
        paths.add(Path(lock[key]["path"]))
    for key in ("evaluation_sources", "candidate_pools", "historical_transcripts"):
        paths.update(Path(row["path"]) for row in lock[key].values())
    target = _target_identity(lock, TARGET)
    paths.update(Path(row["path"]) for row in target["adapter_files_verified"].values())
    paths.update(p for p in (HERE.parent / "evidence" / "feedback_adapter").glob("*.py"))
    return sorted(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(compile_feedback(args.outdir, args.roots), ensure_ascii=False, indent=2))
