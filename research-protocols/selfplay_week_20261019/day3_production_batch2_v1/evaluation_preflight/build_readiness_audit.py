"""Read-only readiness audit of frozen common-evaluation metadata.

Writes only this new batch2 preflight directory. It does not author evaluation
facts/questions, inspect Solver outputs, or alter any frozen day3-v1 artifact.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json

HERE = Path(__file__).resolve().parent
D = HERE.parent
W = D.parent
P = W / "day3_production_v1"


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_new(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    lock_path = P / "evaluation" / "EVAL_SOURCE_FAMILY_LOCK.json"
    lock = read(lock_path)
    source_rows = []
    for record in lock["source_bindings"]:
        actual = bind(record["source_binding"]["path"])
        obj = read(actual["path"])
        turns = obj["turns"]
        scenes = Counter(t["scene_ref"] for t in turns if t.get("scene_ref"))
        checks = {
            "sha256_matches_frozen_lock": actual["sha256"] == record["source_binding"]["sha256"],
            "bytes_match_frozen_lock": actual["bytes"] == record["source_binding"]["bytes"],
            "turn_count_matches_frozen_lock": len(turns) == record["turn_count"],
            "turn_refs_unique": len({t["turn_ref"] for t in turns}) == len(turns),
            "explicit_scene_refs_match": sorted(scenes) == sorted(record["explicit_scene_refs"]),
        }
        assert all(checks.values()), (record["root_id"], checks)
        source_rows.append({
            "source_id": record["source_id"],
            "root_id": record["root_id"],
            "split": record["original_split"],
            "role": record["current_role"],
            "actual_source_binding": actual,
            "full_history_turn_count": len(turns),
            "explicit_scene_count": len(scenes),
            "explicit_scene_turn_counts": dict(scenes),
            "checks": checks,
            "semantic_source_admission": "NOT_DONE_BY_THIS_METADATA_AUDIT",
        })
    slots = lock["itemized_slot_family_ledger"]
    shortfalls = [{
        "slot_id": r["slot_id"], "root_id": r["root_id"], "split": r["split"],
        "question_type": r["question_type"], "local_slot": r["local_slot"],
        "explicit_scene_count": r["explicit_scene_count"],
        "minimum_explicit_scene_count": r["design_target"]["minimum_explicit_scene_count"],
        "design_target": r["design_target"],
        "reason": r["structural_scene_check"],
    } for r in slots if r["structural_scene_check"] == "SHORTFALL_EXPLICIT_SCENE_COUNT"]
    family_checks = [value for row in slots for value in row["family_checks"].values()]
    assert len(source_rows) == 31 and len(slots) == 186 and len(shortfalls) == 13
    assert len(family_checks) == 930
    assert all(value.startswith("PENDING") for value in family_checks)
    assert all(row["fact_family_id"] is None for row in slots)
    assert all(not row["question_generated"] and row["question_sha256"] is None
               and row["gold_sha256"] is None and row["request_manifest_sha256"] is None
               for row in slots)
    split_binding = bind(lock["authoritative_split_actual_verification"]["path"])
    assert split_binding["sha256"] == lock["authoritative_split_actual_verification"]["sha256"]

    # Inventory by names, file existence, byte count and digest only.
    # Do not expose diagnostic scores/answers/feedback in a future evaluator input.
    inventory_specs = [
        ("day1/reuse/INVENTORY.json", "Historical reusable-question inventory; may contain outcome-selected/answer-related fields. Boundary auditor input only, not an evaluator-safe exclusion index."),
        ("day1/reuse/SOURCE_INDEX.json", "Historical source index; source provenance input, not proof of complete semantic-family exclusion."),
        ("day1/reuse/INPUT_BINDINGS.json", "Historical input bindings; metadata only, no semantic-family clearance implied."),
        ("day1/diagnosis/FACT_MEMORY_SCHEMA.json", "A schema, not an inventory of all historical facts."),
        ("day1/diagnosis/MEMORY_SCHEMA.json", "A schema, not a no-score export of all diagnostic fact families."),
        ("day2/fact_memory/SOURCE_BINDINGS.json", "Fact-memory source bindings; useful provenance only."),
        ("day2/fact_memory/PUBLIC_INPUT_PLAN.json", "Fact-memory input plan; requires boundary-auditor filtering before evaluator use."),
        ("day3_production_v1/dedup/FACT_CATALOG.json", "Existing sealed first-batch Train fact catalog; narrow first-batch coverage, not a full historical/diagnostic/heldout exclusion index."),
        ("day3_production_v1/dedup/CANDIDATE_ADMISSION.json", "Existing sealed first-batch candidate admission; not complete historical family exclusion."),
        ("day3_production_v1/dedup/PAIR_LINKS.json", "Existing first-batch pairwise semantic links; narrow comparison scope."),
        ("day3_production_batch2_v1/dedup/FACT_CATALOG.json", "Potential current-batch catalog; if present, requires complete scope and explicit semantic review, not merely file existence."),
    ]
    inventory = []
    for relative, purpose in inventory_specs:
        path = W / relative
        inventory.append({
            "relative_path": relative, "exists_at_audit": path.is_file(),
            "binding": bind(path) if path.is_file() else None,
            "purpose_and_limit": purpose,
            "contents_read_for_this_inventory": False,
            "evaluator_safe_complete_exclusion_index_certified": False,
        })
    matching_paths = sorted(str(p.relative_to(W)) for p in W.rglob("*")
                            if p.is_file() and any(x in p.name.lower() for x in [
                                "exclusion", "family_index", "family_inventory", "eval_ready", "eval_freeze"
                            ]) and HERE not in p.parents)
    obj = {
        "schema_version": "day3_common_eval_metadata_readiness_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "/root/day3_source_inventory",
        "status": "SOURCE_BYTES_VERIFIED_FULL_COMMON_EVALUATION_NOT_READY",
        "frozen_source_slot_lock_binding": bind(lock_path),
        "frozen_lock_readme_binding": bind(P / "evaluation" / "README_ZH.md"),
        "day1_evaluation_policy_binding": bind(W / "day1" / "evaluation" / "EVALUATION_POLICY.json"),
        "authoritative_split_actual_binding": split_binding,
        "frozen_day3_final_manifest_binding": bind(P / "FINAL_MANIFEST.json"),
        "source_checks": {
            "sources": 31, "dev": 15, "test": 16,
            "actual_source_hashes_matched": 31,
            "turn_counts_matched": 31, "unique_turn_ref_checks_passed": 31,
            "explicit_scene_count_distribution": dict(sorted(Counter(r["explicit_scene_count"] for r in source_rows).items())),
            "full_semantic_source_review_completed_here": 0,
        },
        "sources": source_rows,
        "slot_readiness": {
            "abstract_slots": 186, "dev": 90, "test": 96,
            "multi_select": 93, "ordering": 93,
            "source_material_admissions_pending": 186,
            "known_structural_scene_shortfalls": 13,
            "remaining_minimum_scene_count_only_not_admitted": 173,
            "exact_fact_family_assignments_pending": 186,
            "semantic_family_checks_total": 930,
            "semantic_family_checks_pending": 930,
            "semantic_family_checks_passed": 0,
            "actual_generated_questions": 0, "actual_accepted_questions": 0,
            "actual_locked_fact_families": 0, "actual_gold_content_frozen": 0,
            "actual_request_manifests_frozen": 0,
            "source_material_status_counts": dict(Counter(r["source_material_admission"] for r in slots)),
        },
        "structural_shortfall_slots": shortfalls,
        "existing_inventory_path_metadata": inventory,
        "dedicated_exclusion_index_filename_scan": {
            "scope": str(W),
            "filename_terms": ["exclusion", "family_index", "family_inventory", "eval_ready", "eval_freeze"],
            "matching_paths_excluding_this_new_audit_directory": matching_paths,
            "limit": "Filename discovery is not proof that no equivalent information exists elsewhere. No complete no-score semantic exclusion index is certified by this audit.",
        },
        "remaining_actions_in_dependency_order": [
            "Boundary auditor exports a complete no-score exclusion index covering all relevant Train new/replay/basic and diagnostic families plus historical DEV/TEST/v140 families; include canonical facts, source hashes, evidence refs, aliases and related-family edges, never Solver answers/scores/error summaries/weakness cards.",
            "An evaluation author who has not read new A/B candidate content or B feedback reviews exact heldout sources under the frozen deterministic slots and performs prospective canonical-family inventory; preserve all structural/material shortfalls.",
            "Complete all five itemized semantic-family comparisons for each actual candidate using explicit exclusion-record IDs and reasons. Root/split/hash disjointness alone cannot pass this gate.",
            "Generate only source-admitted, family-cleared evaluation items and independently reconstruct answers from full source before private comparison; resolve mention scope, atomicity, all15pairs and public-design criteria.",
            "Freeze question/Gold/source/prompt/parser/permutations/seeds/sampling/request/model identities and zero-truncation checks before formal training or relevant Solver scores. Predeclare DEV aggregate selection; TEST final-only.",
        ],
        "reviewer_exposure_disclosure": {
            "has_read_new_A_B_candidate_content": True,
            "has_read_B_author_private_for_assigned_train_QA": True,
            "scope_of_this_audit": "Metadata readiness/boundary auditor, not a clean common-evaluation author.",
            "cannot_claim_blind_eval_authorship": True,
            "evaluative_question_content_generated_here": False,
            "solver_outputs_or_scores_read_for_this_audit": False,
            "historical_exposure": "Existing historically exposed DEV/TEST roots remain REGRESSION_ONLY; no pristine unseen-root claim.",
            "filesystem_isolation": "Shared filesystem with process disclosure, no technical access-control blinding.",
        },
        "execution": {
            "new_eval_questions_generated": 0, "new_eval_gold_created": 0,
            "solver_calls": 0, "training_calls": 0,
            "frozen_day3_v1_files_modified": False,
            "write_scope": str(HERE),
        },
    }
    write_new(HERE / "READINESS_AUDIT.json", obj)
    report = """# 共同评估准备情况核对

31 个既有来源（15 DEV、16 TEST）的实际 SHA256、turn 数、turn_ref 唯一性及显式场景均与已封存预锁一致。它们仍是历史暴露来源，只能按既定 REGRESSION_ONLY 边界报告。

当前仍只有 **186 个抽象题位**：186 项素材语义审核待完成，186 项事实题族分配待完成，930 项事实族比较全部 pending。实际生成题、接受题、锁定题族、冻结 Gold、冻结请求均为 **0**，不能据此启动正式共同评估或训练。

已有 **13 个显式场景数量短缺槽**：DEV 229160 的排序 1/2；TEST 101421 的多选 2/3 和排序 1/2/3；TEST 124804 的排序 2/3；TEST 1456 的排序 3；TEST 237841 的排序 1；TEST 35167 的排序 1/2。其余 173 槽只是最低场景数满足，尚未证明有合格素材。逐槽 ID、来源与目标见 READINESS_AUDIT.json。

下一步先由边界审核者整理**不含成绩/回答/错误摘要/弱点卡的完整事实排除索引**，覆盖训练新题、回放、基础题、诊断及历史 DEV/TEST/v140 事实；现有 source index、复用题清单和首批事实目录只能作为输入，不能代替完整排除审核。随后由未读新 A/B 题及反馈的评估作者进行来源审核、固定题位的事实族选择、逐项去重、出题与独立答案重建，最后冻结完整请求与模型身份。短缺保留，不按成绩换 root 或换设计。

本审核者已阅读新 A/B 题及其负责的 B 训练题私有标注，因此本次只作准备情况与边界核对，不能冒充未曝光 A/B 的盲评估作者。本次未生成评估题、未运行 Solver、未训练，未修改已封存 day3_production_v1。详细路径、散列与限制均在 READINESS_AUDIT.json。
"""
    with (HERE / "REPORT_ZH.md").open("x") as f:
        f.write(report)
    print("READINESS_AUDIT.json: 31 sources verified; 186 semantic admissions / 930 family checks pending; 13 structural shortfalls; actual questions/Gold/requests 0")


if __name__ == "__main__":
    main()
