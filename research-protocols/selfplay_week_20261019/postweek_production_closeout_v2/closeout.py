"""Verify the production seals and publish a separate navigation closeout."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    assert not (HERE / "FINAL_MANIFEST.json").exists(), "ALREADY_CLOSED"
    inputs, verified = {}, 0
    for name in ("postweek_production_batch_v1", "postweek_production_batch_v2",
                 "postweek_production_feedback_audit_v1", "postweek_production_pool_v2"):
        path = WEEK / name / "FINAL_MANIFEST.json"
        doc = read(path)
        for key in ("artifact_bindings", "external_reference_bindings", "input_bindings"):
            for item in doc.get(key, []):
                assert binding(item["path"]) == item, "SEAL_DRIFT:" + item["path"]
                verified += 1
        inputs[name] = binding(path)
    pool = read(WEEK / "postweek_production_pool_v2/POOL_MANIFEST.json")
    audit = read(WEEK / "postweek_production_feedback_audit_v1/SUMMARY.json")
    batch = read(WEEK / "postweek_production_batch_v2/FINAL_MANIFEST.json")
    pointer = read(WEEK / "PRODUCTION_BATCH_LATEST.json")
    assert pointer["manifest"] == inputs["postweek_production_batch_v2"]
    old_status = read(WEEK / "CURRENT_STATUS.json")
    save(HERE / "CURRENT_STATUS_BEFORE_CLOSEOUT.json", old_status)
    now = datetime.now(timezone.utc).isoformat()
    next_action = (
        "Run a separately frozen Train-only live dialogue-extension test that consumes "
        "source-material feedback, retaining the two-route three-scene structure and "
        "complete failed attempts. Then independently audit sources and jointly produce "
        "MC/ordering questions. Prospective weakness reproduction and matched feedback "
        "training remain separate pending experiments."
    )
    closeout = {
        "schema_version": "production_and_reason_audit_closeout_v2",
        "created_at_utc": now,
        "status": "BATCH2_AND_BATCH1_REASON_AUDIT_COMPLETE_POOL_INDEXED",
        "manifest_inputs": inputs,
        "verified_binding_checks": verified,
        "batch2_questions": batch["content_approved_unique_questions"],
        "batch2_question_types": batch["question_types"],
        "batch2_initial_pass_rate_planned": batch["initial_joint_pass_rate_planned"],
        "batch2_solver_requests": batch["solver_requests"],
        "batch2_solver_valid": batch["solver_valid"],
        "batch2_solver_conditions": batch["solver_conditions"],
        "cumulative_candidate_questions": pool["exported_unique_questions"],
        "cumulative_candidate_roots": pool["roots"],
        "cumulative_candidate_question_types": pool["question_types"],
        "later_content_holds_excluded": pool["excluded_later_content_holds"],
        "reason_audit_questions": audit["questions"],
        "reason_audit_full_responses": audit["responses"],
        "reason_audit_exact_answers": audit["exact_responses"],
        "reason_audit_all_dimensions_pass": audit["all_review_dimensions_pass"],
        "exploratory_weakness_cards": audit["exploratory_cards"],
        "new_dialogue_turns": 0,
        "new_training_runs": 0,
        "final_training_mix_built": False,
        "formal_weakness_admission": False,
        "feedback_causal_gain_proven": False,
        "next_action": next_action,
    }
    save(HERE / "SUMMARY.json", closeout)
    text = (
        "# 本次生产与理由审核交接\n\n"
        f"第二批10个Train roots已完成出题、独立内容审核、solver实测及封存："
        f"{batch['content_approved_unique_questions']}题，{batch['solver_requests']}次请求。"
        "具体题型、缺额、初审与修复、准确率及无效响应见 ../postweek_production_batch_v2/REPORT_ZH.md。\n\n"
        f"两批新生产候选合计{pool['exported_unique_questions']}题、{pool['roots']}个roots；"
        f"后续内容疑点暂缓{pool['excluded_later_content_holds']}题。"
        "完整索引见 ../postweek_production_pool_v2/README_ZH.md。该池没有纳入全部历史训练题，也不是最终训练混合。\n\n"
        f"首批{audit['responses']}条FULL响应已独立完成逐项语义审核。最终答案正确"
        f"{audit['exact_responses']}/{audit['responses']}；全部适用理由、引用和一致性检查通过"
        f"{audit['all_review_dimensions_pass']}/{audit['responses']}。二者衡量不同对象。"
        f"{audit['exploratory_cards']}张卡仅为探索性候选，真实示例见 "
        "../postweek_production_feedback_audit_v1/EXAMPLES_ZH.md。\n\n"
        "本次没有新对话扩充、训练、正式弱点准入或反馈训练收益证明。下一步单独冻结并实测"
        "Train-only来源素材反馈驱动的真实扩充，保留两条路线各三场及失败记录，再独立审核并联合出多选/排序题。"
        "候选弱点仍须前瞻受控复现；后续训练须另冻结混合、选项置换、固定验证集与W&B/每20步验证配置。\n"
    )
    with (HERE / "README_ZH.md").open("x", encoding="utf-8") as stream:
        stream.write(text)
    save(HERE / "FINAL_MANIFEST.json", {
        "schema_version": "production_closeout_seal_v2",
        "artifact_bindings": [binding(p) for p in sorted(HERE.iterdir()) if p.is_file()],
        "input_bindings": list(inputs.values()),
    })
    updated = {**old_status, "updated_at_utc": now, "status": closeout["status"],
               "production_closeout_latest": binding(HERE / "SUMMARY.json"),
               "production_feedback_audit_latest": read(WEEK / "PRODUCTION_FEEDBACK_AUDIT_LATEST.json"),
               "production_pool_latest": read(WEEK / "PRODUCTION_POOL_LATEST.json"),
               "next_action": next_action,
               "status_snapshot_before_closeout": binding(HERE / "CURRENT_STATUS_BEFORE_CLOSEOUT.json")}
    with (WEEK / "CURRENT_STATUS.json").open("w", encoding="utf-8") as stream:
        json.dump(updated, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(closeout, ensure_ascii=False))


if __name__ == "__main__":
    main()
