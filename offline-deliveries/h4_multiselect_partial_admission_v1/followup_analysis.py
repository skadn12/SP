"""Offline analysis of the 48 authorized follow-up answers and 24 baseline answers.

The legacy position diagnostic is executed unchanged from its source AST, so
analysis does not import Solver/model runtime modules or initialize a backend.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

from admission import EPS, VERSION, candidate_decision
from reanalyze import checked_audit, question_sha256, sha256, verify_pair
from scoring import score_response


HERE = Path(__file__).resolve().parent
PLAN = HERE / "followup_48_requests_revision_2.json"
RUN = HERE / "followup_k3_revision_2"
LEGACY_ROOT = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1")
LEGACY = LEGACY_ROOT / "scripts/current_method/restart_n5_spell/h4_round0_b0_k3_difficulty_v1"
MODES = ("no_context", "full_context")
PERMUTATION_DECISION_POLICY = {
    "version": "k3_dual_scope_frozen_before_corrected_run_results",
    "legacy_exclusion": "Full-context position_only blocks; none/sampling_only/mixed only pass that exclusion.",
    "positive_content_stability_contract_found": False,
    "positive_check_when_legacy_exclusion_passes": "PENDING_CONTENT_STABILITY_CRITERION",
    "basis": "Original task section 4.3 requires an explicit applicable passing criterion; the exact-only legacy diagnostic does not supply one, especially at full_exact=0.",
    "new_numeric_thresholds_introduced": False,
}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fraction(value: Fraction) -> dict:
    return {"fraction": str(value), "value": float(value)}


def legacy_position_function() -> tuple[Any, dict]:
    """Load only the exact two pure functions and constants needed for diagnosis."""
    source_path = LEGACY / "difficulty.py"
    constants_path = LEGACY / "__init__.py"
    source = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)
             and node.name in {"_prediction_key", "position_diagnostics"}]
    if {node.name for node in nodes} != {"_prediction_key", "position_diagnostics"}:
        raise ValueError("legacy_position_functions_missing")
    constants = {}
    for node in ast.parse(constants_path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"ROLLOUTS", "QUESTION_KEYS"}:
                    constants[target.id] = ast.literal_eval(node.value)
    if constants.get("ROLLOUTS") != 3:
        raise ValueError("legacy_rollouts_changed")
    namespace = {"Any": Any, "Iterable": Iterable, "Mapping": Mapping,
                 "defaultdict": defaultdict, **constants}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["position_diagnostics"], {
        "path": str(source_path), "sha256": sha256(source_path),
        "constants_path": str(constants_path), "constants_sha256": sha256(constants_path),
        "loading": "Unmodified pure function AST; no Solver/runtime import",
        "rollouts": constants["ROLLOUTS"], "position_count": 6,
        "between_definition": "Variation in exact correct counts across displays",
        "within_definition": "Canonical answer variation across slots within a display",
        "none_does_not_mean_no_position_effect": True,
    }


def score_j(score: dict) -> Fraction:
    return Fraction(score["tp"], score["g"]) - Fraction(score["fp"], score["n"] - score["g"])


def aggregate(rows: list[dict], question: dict) -> dict:
    valid = [r for r in rows if r["score"]["scoring_status"] == "SCORED"]
    complete = bool(rows) and len(valid) == len(rows) and not any(r["issues"] for r in rows)
    gold, labels = set(question["gold"]), sorted(question["options"])
    canonical = Counter(tuple(r["score"]["canonical_prediction"]) for r in valid)
    display = Counter(tuple(sorted(r["score"]["display_prediction"])) for r in valid)
    partial = sum((score_j(r["score"]) for r in valid), Fraction())
    return {
        "expected_count": len(rows), "scored_count": len(valid), "complete": complete,
        "exact_correct": sum(r["score"]["exact_match"] for r in valid),
        "exact_rate": sum(r["score"]["exact_match"] for r in valid) / len(rows) if complete else None,
        "partial_mean": fraction(partial / len(rows)) if complete else None,
        "canonical_set_counts": {",".join(k): v for k, v in sorted(canonical.items())},
        "display_set_counts": {",".join(k): v for k, v in sorted(display.items())},
        "canonical_selected_counts": {key: sum(key in r["score"]["canonical_prediction"] for r in valid)
                                      for key in labels},
        "display_letter_selected_counts": {key: sum(key in r["score"]["display_prediction"] for r in valid)
                                           for key in labels},
        "false_positive_counts": {key: sum(key in r["score"]["canonical_prediction"] for r in valid)
                                  for key in labels if key not in gold},
        "false_negative_counts": {key: sum(key not in r["score"]["canonical_prediction"] for r in valid)
                                  for key in labels if key in gold},
        "canonical_selected_by_display_letter": {
            option: {letter: {
                "observed": sum(r["request"]["variant"]["display_to_canonical"][letter] == option for r in valid),
                "selected": sum(r["request"]["variant"]["display_to_canonical"][letter] == option
                                and option in r["score"]["canonical_prediction"] for r in valid),
            } for letter in labels} for option in labels},
    }


def comparison(rows: list[dict], question: dict) -> dict:
    modes = {mode: aggregate([r for r in rows if r["request"]["mode"] == mode], question) for mode in MODES}
    no, full = (modes[mode] for mode in MODES)
    complete = no["complete"] and full["complete"]
    gain = (Fraction(full["partial_mean"]["fraction"]) - Fraction(no["partial_mean"]["fraction"])) if complete else None
    return {"modes": modes,
            "exact_context_gain": full["exact_rate"] - no["exact_rate"] if complete else None,
            "partial_context_gain": fraction(gain) if gain is not None else None}


def analyze(plan_path: Path = PLAN, run: Path = RUN) -> dict:
    plan = read(plan_path)
    complete_path = run / "COMPLETE.json"
    execution_path = run / "execution_manifest.json"
    if not complete_path.is_file() or read(complete_path).get("response_count") != 48:
        raise ValueError("followup_not_complete")
    execution = read(execution_path)
    contract_path = run / "ADMISSION_CONTRACT.json"
    admission_contract = read(contract_path)
    if (admission_contract.get("policy_version") != VERSION or admission_contract.get("eps") != EPS
            or admission_contract.get("new_thresholds_introduced") is not False
            or "PENDING_CONTENT_STABILITY_CRITERION" not in admission_contract.get("positive_permutation_contract", "")):
        raise ValueError("frozen_admission_contract_mismatch")
    if execution.get("baseline_merge_authorized") is not True or execution.get("plan_sha256") != sha256(plan_path):
        raise ValueError("execution_plan_or_merge_authorization_mismatch")
    identity_ref = execution["model_identity_evidence"]
    identity_path = Path(identity_ref["path"])
    if sha256(identity_path) != identity_ref["sha256"]:
        raise ValueError("model_identity_evidence_changed")
    identity = read(identity_path)
    if identity.get("baseline_merge_supported") is not True:
        raise ValueError("baseline_model_identity_not_supported")
    if execution.get("requests") != plan["requests"] or execution.get("request_count") != 48:
        raise ValueError("execution_requests_mismatch")
    if execution.get("model") != plan["model"] or execution.get("input_sha256") != plan["source_input_sha256"]:
        raise ValueError("execution_model_or_input_mismatch")
    if any(execution.get("generation_config", {}).get(k) != v for k, v in plan["decoding"].items()):
        raise ValueError("execution_decode_config_mismatch")
    baseline_path = Path(plan["baseline_manifest"]["path"])
    if sha256(baseline_path) != plan["baseline_manifest"]["sha256"] or execution.get("baseline_manifest_sha256") != sha256(baseline_path):
        raise ValueError("baseline_manifest_changed")
    baseline = read(baseline_path)
    baseline_run = baseline_path.parent
    hashes = {str(path): sha256(path) for path in (plan_path, baseline_path, complete_path, execution_path, identity_path,
                                                  contract_path, Path(__file__), HERE / "admission.py", HERE / "scoring.py",
                                                  HERE / "reanalyze.py")}
    for raw_path, expected_hash in execution.get("code_sha256", {}).items():
        path = Path(raw_path)
        if sha256(path) != expected_hash:
            raise ValueError("execution_code_changed:" + raw_path)
        hashes[raw_path] = expected_hash
    for name, expected in plan["source_input_sha256"].items():
        path = baseline_run / name
        if sha256(path) != expected or baseline["input_sha256"].get(name) != expected:
            raise ValueError("frozen_input_changed:" + name)
        hashes[str(path)] = expected
    frozen = read(baseline_run / "questions.json")["candidates"]
    gold = read(baseline_run / "gold.json")
    questions = {q["candidate_id"]: {**q, "gold": gold[q["candidate_id"]]}
                 for q in frozen if q["question_type"] == "multi_select"}
    if len(questions) != 2 or len(plan["requests"]) != 48 or plan.get("request_count") != 48:
        raise ValueError("followup_scope_mismatch")
    baseline_by_id = {r["request_id"]: r for r in baseline["requests"]}
    requests = []
    reused = {}
    shifts = {}
    for request in plan["requests"]:
        old = baseline_by_id[request["baseline_request_id"]]
        qid = request["candidate_id"]
        if request["slot"] not in {1, 2} or old["slot"] != 0:
            raise ValueError("invalid_followup_slot")
        if request["question_sha256"] != question_sha256(questions[qid]):
            raise ValueError("followup_question_version_mismatch")
        if any(request.get(k) != old.get(k) for k in ("candidate_id", "question_type", "variant", "mode", "prompt_sha256", "prompt_tokens")):
            raise ValueError("followup_baseline_request_mismatch:" + request["request_id"])
        vid = request["variant"]["variant_id"]
        if vid not in {f"perm_{i:02d}" for i in range(6)}:
            raise ValueError("followup_variant_out_of_scope")
        shift = request["legacy_diagnostic_variant_id"]
        if shift != f"shift_{int(vid[-2:])}":
            raise ValueError("followup_shift_mapping_mismatch")
        shifts[vid] = shift
        reused[old["request_id"]] = old
        requests.append((request, run / "responses", "new_slots_1_2"))
    if len(reused) != 24:
        raise ValueError("baseline_reuse_count_mismatch")
    requests += [(request, baseline_run / "responses", "baseline_slot_0") for request in reused.values()]
    expected_keys = {(qid, f"perm_{i:02d}", mode, slot) for qid in questions for i in range(6)
                     for mode in MODES for slot in range(3)}
    keys = [(r["candidate_id"], r["variant"]["variant_id"], r["mode"], r["slot"]) for r, _, _ in requests]
    if len(keys) != 72 or len(set(keys)) != 72 or set(keys) != expected_keys:
        raise ValueError("followup_grid_not_exactly_72")
    if {path.name for path in (run / "responses").glob("*.json")} != {r["request_id"] + ".json" for r in plan["requests"]}:
        raise ValueError("unexpected_or_missing_followup_response_files")
    if plan["model"] != baseline["model"] or any(plan["decoding"].get(k) != baseline.get(k)
                                                for k in ("temperature", "top_p", "max_tokens", "enable_thinking")):
        raise ValueError("followup_baseline_model_or_decode_mismatch")
    position_diagnostics, legacy_info = legacy_position_function()
    hashes[legacy_info["path"]] = legacy_info["sha256"]
    hashes[legacy_info["constants_path"]] = legacy_info["constants_sha256"]
    rows = []
    response_objects = {}
    for request, directory, origin in requests:
        path = directory / (request["request_id"] + ".json")
        if not path.is_file():
            raise ValueError("response_missing:" + str(path))
        response = read(path)
        hashes[str(path)] = sha256(path)
        response_objects[request["request_id"]] = response
        issues = verify_pair(request, request, response, response)
        if response.get("model_identifier") != plan["model"]:
            issues.append("MODEL_MISMATCH")
        if response.get("metadata", {}).get("finish_reason") not in {"stop", "eos"}:
            issues.append("INCOMPLETE_GENERATION")
        for field in ("temperature", "top_p", "max_tokens"):
            if response.get("metadata", {}).get(field) != plan["decoding"][field]:
                issues.append("DECODE_CONFIG_MISMATCH:" + field)
        score = score_response(questions[request["candidate_id"]], request, response)
        if score["scoring_status"] != "SCORED":
            issues.append("UNSCORED:" + score["scoring_status"])
        rows.append({"request": request, "origin": origin, "response_path": str(path),
                     "issues": sorted(set(issues)), "score": score})
    results = {}
    for qid, question in questions.items():
        selected = sorted([r for r in rows if r["request"]["candidate_id"] == qid],
                          key=lambda r: (r["request"]["variant"]["variant_id"], r["request"]["slot"], r["request"]["mode"]))
        by_key = {(r["request"]["variant"]["variant_id"], r["request"]["slot"], r["request"]["mode"]): r
                  for r in selected}
        paired = []
        for variant in sorted(shifts):
            for slot in range(3):
                no, full = (by_key[variant, slot, mode] for mode in MODES)
                issues = verify_pair(no["request"], full["request"], response_objects[no["request"]["request_id"]],
                                     response_objects[full["request"]["request_id"]])
                # The baseline and follow-up structured-output implementations must also agree.
                for row in (no, full):
                    base_request = reused[next(rid for rid, r in reused.items() if r["candidate_id"] == qid
                                               and r["variant"]["variant_id"] == variant and r["mode"] == row["request"]["mode"])]
                    original = response_objects[base_request["request_id"]]
                    current = response_objects[row["request"]["request_id"]]
                    for field in ("structured_output_api", "structured_output_version"):
                        if original["metadata"].get(field) != current["metadata"].get(field):
                            issues.append("BASELINE_STRUCTURED_CONFIG_MISMATCH:" + field)
                    row["issues"] = sorted(set(row["issues"] + issues))
                valid = not no["issues"] and not full["issues"]
                gain = score_j(full["score"]) - score_j(no["score"]) if valid else None
                paired.append({"variant_id": variant, "slot": slot, "issues": sorted(set(issues)),
                               "seed_equal": no["request"]["seed"] == full["request"]["seed"],
                               "partial_context_gain": fraction(gain) if gain is not None else None,
                               "exact_context_gain": int(full["score"]["exact_match"]) - int(no["score"]["exact_match"]) if valid else None})
        legacy_rows = [{"context_mode": r["request"]["mode"],
                        "variant_id": shifts[r["request"]["variant"]["variant_id"]], "slot_id": r["request"]["slot"],
                        "scored": not r["issues"], "correct": r["score"]["exact_match"],
                        "canonical_prediction": r["score"]["canonical_prediction"]} for r in selected]
        diagnostics = {mode: position_diagnostics(legacy_rows, question_type="multi_select", context_mode=mode,
                                                position_count=6) for mode in MODES}
        combined = comparison(selected, question)
        complete = all(d["technical_complete"] for d in diagnostics.values()) and not any(r["issues"] for r in selected)
        mechanism = diagnostics["full_context"]["position_mechanism"]
        legacy_exclusion_status = ("INCOMPLETE" if not complete else "POSITION_ONLY_DIAGNOSTIC" if mechanism == "position_only"
                                   else "PASS" if mechanism in {"none", "sampling_only", "mixed"} else "UNKNOWN_MECHANISM")
        permutation_status = ("PENDING_CONTENT_STABILITY_CRITERION" if legacy_exclusion_status == "PASS"
                              else legacy_exclusion_status)
        audits = {kind: checked_audit(HERE, kind, question, hashes) for kind in ("quality", "split")}
        no, full = (combined["modes"][mode] for mode in MODES)
        metrics = {"no_exact_rate": no["exact_rate"], "full_exact_rate": full["exact_rate"],
                   "no_partial_mean": no["partial_mean"]["value"] if no["partial_mean"] else None,
                   "full_partial_mean": full["partial_mean"]["value"] if full["partial_mean"] else None}
        decision = candidate_decision(metrics, quality_check_status=audits["quality"]["status"],
                                      split_check_status=audits["split"]["status"],
                                      pairing_check_status="PASS" if complete else "INCOMPLETE",
                                      permutation_check_status=permutation_status,
                                      confirmed_quarantine=any(a.get("confirmed_quarantine") is True for a in audits.values()))
        results[qid] = {"question_sha256": question_sha256(question), "root_id": question["root_id"],
                        "gold": question["gold"], "combined_k3": combined,
                        "new_slots_only": comparison([r for r in selected if r["request"]["slot"] in {1, 2}], question),
                        "per_slot": {str(slot): comparison([r for r in selected if r["request"]["slot"] == slot], question)
                                     for slot in range(3)},
                        "per_variant": {v: comparison([r for r in selected if r["request"]["variant"]["variant_id"] == v], question)
                                        for v in sorted(shifts)},
                        "position_diagnostics": diagnostics,
                        "legacy_position_exclusion_status": legacy_exclusion_status,
                        "permutation_check_scope": "Original explicit passing-criterion requirement remains separate from the full-context position_only exclusion.",
                        "review_evidence": audits, "per_pair": paired, "responses": selected, **decision}
    prior_path = HERE / "permutation_audit_revision_2.json"
    prior = read(prior_path)
    hashes[str(prior_path)] = sha256(prior_path)
    status_id = next(qid for qid in questions if qid.endswith("status-01"))
    results[status_id]["prior_temp0_evidence"] = {
        "source": str(prior_path), "gate_status_reused": False,
        "evidence": [e for e in prior["questions"][status_id]["evidence"] if "temperature0" in e],
        "qualification": "Prior position sensitivity remains relevant even if legacy exact-count mechanism is none/sampling_only.",
    }
    excluded = []
    wrong_runtime = run.with_name(run.name + "_runtime_mismatch_diagnostic")
    if wrong_runtime.is_dir():
        excluded.append({"path": str(wrong_runtime),
                         "response_count": len(list((wrong_runtime / "responses").glob("*.json"))),
                         "reason": "Initial execution used a different Python/vLLM runtime and structured-output API; retained only as diagnostic, excluded from all K3 scores and admission.",
                         "included_in_analysis": False})
        for name in ("COMPLETE.json", "execution_manifest.json"):
            path = wrong_runtime / name
            if path.is_file():
                hashes[str(path)] = sha256(path)
    return {"analysis_version": "followup_k3_revision_2", "policy_version": VERSION,
            "source_plan": str(plan_path), "hashes": hashes, "legacy_diagnostic": legacy_info,
            "model_identity_scope": execution.get("model_identity_scope"),
            "permutation_decision_policy": PERMUTATION_DECISION_POLICY,
            "admission_contract_path": str(contract_path), "admission_contract": admission_contract,
            "variant_adapter": shifts, "total_response_count": len(rows), "new_response_count": 48,
            "baseline_response_count": 24, "old_k1_pending_status_reused_as_gate": False,
            "excluded_runs": excluded,
            "training_promotion_executed": False, "training_promoted_count": 0,
            "counts": {"numeric_candidates": sum(r["numeric_candidate"] for r in results.values()),
                       "training_candidates": sum(r["new_policy_candidate_status"] == "TRAINING_CANDIDATE" for r in results.values())},
            "questions": results}


def report_markdown(result: dict) -> str:
    lines = ["# 两题 K=3 补评估结果", "",
             "合并6个正向循环展示的旧slot0与新增slot1/2：两题共72条回答，其中新增48条。反向排列与温度0对照仍作为补充证据，不混入同配置K=3重复。", "",
             "本次每模式18次是6种排列×3个槽位，上一份报告每模式12次是12种排列×1个槽位，两批排列构成不同。因此不能把均值变化都归因于增加重复次数；表中的上下文收益只在本批配对内比较，EPS也不表示统计显著性。", "",
             "| 题目 | no → full exact | no → full J | J收益 | full旧换位机制 | 数值候选 | 训练候选状态 |",
             "|---|---|---|---|---|---|---|"]
    for qid, q in result["questions"].items():
        c = q["combined_k3"]
        no, full = (c["modes"][m] for m in MODES)
        j = lambda x: x["partial_mean"]["fraction"] if x["partial_mean"] else "未完成"
        gain = c["partial_context_gain"]["fraction"] if c["partial_context_gain"] else "未完成"
        lines.append(f"| {qid} | {no['exact_correct']}/18 → {full['exact_correct']}/18 | {j(no)} → {j(full)} | {gain} | "
                     f"{q['position_diagnostics']['full_context']['position_mechanism']} | {q['numeric_candidate']} | {q['new_policy_candidate_status']} |")
    lines += ["", "## 新回答与旧slot0分开对照", "",
              "| 题目/槽位 | no → full exact | no → full J | J收益 |", "|---|---|---|---|"]
    for qid, q in result["questions"].items():
        groups = [("旧slot0", q["per_slot"]["0"]), ("新增slot1", q["per_slot"]["1"]),
                  ("新增slot2", q["per_slot"]["2"]), ("新增合计", q["new_slots_only"])]
        for name, c in groups:
            no, full = (c["modes"][m] for m in MODES)
            gain = c["partial_context_gain"]["fraction"] if c["partial_context_gain"] else "未完成"
            lines.append(f"| {qid}/{name} | {no['exact_correct']}/{no['expected_count']} → {full['exact_correct']}/{full['expected_count']} | "
                         f"{j(no)} → {j(full)} | {gain} |")
    lines += ["", "## 换位机制及选项统计", "",
              "旧函数直接执行原有规则：within表示同展示不同槽位的canonical回答有变化；between仅表示不同展示的exact正确数不同。完整K=3且full机制为position_only时继续诊断；其余none/sampling_only/mixed仅使legacy_position_exclusion_status=PASS，不能解释为无位置影响、选项内容完全稳定或因果证据。尤其full全零时between必为false，必须同时查看部分分与canonical/字母统计。", "",
              "最终换位检查与旧排除检查分开：原任务4.3本就要求有明确且适用的正向通过标准。旧exact-only机制算法没有提供内容稳定性正向标准，因此未触发旧排除时仍为PENDING_CONTENT_STABILITY_CRITERION；这是保留原合同，不是看结果后增加阈值。此双字段规则在修正环境的新结果产生前已固定。", ""]
    lines += ["事前合同与原要求引句保存在ADMISSION_CONTRACT.json，其SHA-256随分析一并记录。", ""]
    for qid, q in result["questions"].items():
        for mode in MODES:
            d = q["position_diagnostics"][mode]
            s = q["combined_k3"]["modes"][mode]
            lines.append(f"- {qid} / {mode}: within={d['within_position_uncertainty']}, between={d['between_position_sensitivity']}, "
                         f"mechanism={d['position_mechanism']}; canonical A..F={list(s['canonical_selected_counts'].values())}; "
                         f"显示字母 A..F={list(s['display_letter_selected_counts'].values())}。")
        lines.append(f"- {qid}：质量={q['quality_check_status']}，split={q['split_check_status']}，配对={q['pairing_check_status']}，旧位置排除={q['legacy_position_exclusion_status']}，最终换位={q['permutation_check_status']}；blockers={q['blockers']}。")
        lines.append(f"- {qid}：数值判定原因={q['new_policy_reason']}。")
    lines += ["", "此前状态题温度0对照显示B在前两显示位置0/4、后四位置8/8；变邻接后仍有位置差异。该事实保留，不因旧exact机制标签改变而抹去。完整每排列/槽位J、FP/FN、canonical集合和显示集合、选项随位置变化计数均在analysis.json。", "",
              "质量与split沿用绑定当前题目版本和证据哈希的审核；历史隔离信息保留。旧K1待诊断状态不继续作为新K3的强制门槛，seed不相同也不单独否决；全零exact若净部分分受上下文帮助，仍按新版数值规则判断。", "",
              f"数值候选{result['counts']['numeric_candidates']}题，训练候选{result['counts']['training_candidates']}题。训练候选是离线准入结果，本脚本没有导出SFT样本、写入生产池或执行训练，实际入训仍为0。", ""]
    if result["excluded_runs"]:
        lines += ["## 排除的首次执行", "",
                  "首次48条回答使用了不同Python/vLLM环境与结构化输出API，未与旧slot0混算；完整保存在followup_k3_revision_2_runtime_mismatch_diagnostic。正式结果只使用修正为原环境后的48条与24条旧回答。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    result = analyze(args.plan, args.run)
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.run / "REPORT.md").write_text(report_markdown(result), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
