"""Exclusive, descriptive QPR1 report; run only after sealed solver analysis.

This reporter carries frozen question reviews, scores final answers, and compares
the actually shared parent requests. It never certifies explanation semantics.
All reads, assertions, aggregates, and serializations precede any output open.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PARENT = BASE.parent / "postweek_feedback_control_4root_v1"
EXEC = BASE / "solver_runtime/execution"
OLD_EXEC = PARENT / "solver_runtime/execution"
BRANCHES = ("direct", "reason")
GROUPS = ("main_mc2_5", "ordering", "mc6_control")
SLOTS = ("MC2", "MC3", "MC4", "MC5", "MC6", "ORDER1", "ORDER2")
OUTPUTS = ("FINAL_REPORT.json", "FINAL_REPORT_ZH.md", "PAIRED_ROOT_RESULTS.csv", "FINAL_ARTIFACT_MANIFEST.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify(value):
    assert binding(value["path"]) == value, "BINDING_DRIFT:" + value["path"]


def serialized(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def unique(rows, key):
    result = {key(row): row for row in rows}
    assert len(result) == len(rows), "DUPLICATE_RECORD_ID"
    return result


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def group(slot):
    if slot in ("MC2", "MC3", "MC4", "MC5"):
        return "main_mc2_5"
    return "ordering" if slot.startswith("ORDER") else "mc6_control"


def metrics(rows):
    pair_correct = sum(row.get("pairwise_correct") or 0 for row in rows)
    pair_total = sum(row.get("pairwise_total") or 0 for row in rows)
    correct = sum(row["exact"] is True for row in rows)
    valid = sum(row["valid"] is True for row in rows)
    return {"n": len(rows), "valid": valid, "invalid": len(rows) - valid,
            "correct": correct, "accuracy": rate(correct, len(rows)),
            "false_positive": sum(row.get("false_positive") or 0 for row in rows),
            "false_negative": sum(row.get("false_negative") or 0 for row in rows),
            "ordering_pairwise_correct": pair_correct, "ordering_pairwise_total": pair_total,
            "ordering_pairwise_accuracy": rate(pair_correct, pair_total)}


def keyed_decisions(freeze):
    return unique(freeze["decisions"], lambda row: (row["root_id"], row["condition"], row["slot_id"]))


def admission_table(decisions):
    return {name: {"planned": sum(group(key[2]) == name for key in decisions),
                   "admitted": sum(row["status"] == "ADMIT" and group(key[2]) == name for key, row in decisions.items())}
            for name in GROUPS}


def compare_metrics(rows):
    result = {"requests": len(rows), "old": metrics([row["old_score"] for row in rows]),
              "new": metrics([row["new_score"] for row in rows]),
              "transitions": dict(Counter(("correct" if row["old_score"]["exact"] else "wrong") + "_to_" +
                                           ("correct" if row["new_score"]["exact"] else "wrong") for row in rows))}
    result["accuracy_difference_new_minus_old"] = (result["new"]["accuracy"] - result["old"]["accuracy"]) if rows else None
    return result


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def fraction(value):
    return f"{value['correct']}/{value['n']}（{pct(value['accuracy'])}）"


def main():
    # Reject a rerun before reading any model outputs. Never replace a report.
    for name in OUTPUTS:
        assert not (BASE / name).exists(), "EXCLUSIVE_OUTPUT_EXISTS:" + name

    required = {
        "protocol": BASE / "REPAIR_PROTOCOL.json",
        "receipt": BASE / "TRANSFORM_RECEIPT.json",
        "delta_review": BASE / "PRESENTATION_DELTA_REVIEW.json",
        "gate_regression": BASE / "PUBLIC_GATE_REGRESSION_V2.json",
        "question_freeze": BASE / "QUESTION_FREEZE.json",
        "export_freeze": BASE / "inputs/EXPORT_FREEZE.json",
        "public": BASE / "inputs/PUBLIC_QUESTIONS.json",
        "private": BASE / "inputs/PRIVATE_SCORING.json",
        "seeds": BASE / "inputs/PUBLIC_SAMPLING_SEEDS.json",
        "preparation": EXEC / "PREPARATION.json",
        "preflight": BASE / "solver_runtime/PREFLIGHT_REVIEW.json",
        "presentation_check": EXEC / "PUBLIC_PRESENTATION_CHECK.json",
        "started": EXEC / "STARTED.json",
        "raw_seal": EXEC / "RAW_RESPONSE_FREEZE.json",
        "scores": EXEC / "SCORES.json",
        "structures": EXEC / "ALL_STRUCTURAL_RECORDS.json",
        "runtime_summary": EXEC / "ANALYSIS_SUMMARY.json",
        "requests": EXEC / "PUBLIC_REQUESTS.json",
        "old_report": PARENT / "FINAL_REPORT.json",
        "old_original_freeze": PARENT / "ORIGINAL_QUESTION_QUALITY_FREEZE.json",
        "old_supplement_freeze": PARENT / "QUESTION_FREEZE.json",
        "old_scores": OLD_EXEC / "SCORES.json",
        "old_requests": OLD_EXEC / "PUBLIC_REQUESTS.json",
        "old_preparation": OLD_EXEC / "PREPARATION.json",
        "old_started": OLD_EXEC / "STARTED.json",
        "old_seal": OLD_EXEC / "RAW_RESPONSE_FREEZE.json",
    }
    assert all(path.is_file() for path in required.values()), "REPORT_INPUTS_NOT_COMPLETE"
    data = {key: read(path) for key, path in required.items()}
    protocol, receipt, delta = data["protocol"], data["receipt"], data["delta_review"]
    freeze, export = data["question_freeze"], data["export_freeze"]
    prep, preflight, seal, score_payload = data["preparation"], data["preflight"], data["raw_seal"], data["scores"]
    roots = protocol["roots"]
    conditions = protocol["conditions"]
    assert len(roots) == 4 and len(set(roots)) == 4 and conditions == ["GENERIC", "TARGETED"]
    assert protocol["source_expansion"] is protocol["author_regeneration"] is False
    assert delta["status"] == "PASS_DELTA_REVIEW_ONLY"
    assert data["gate_regression"]["status"] == "PASS_REAL_DRAFT_PRESENTATION_REGRESSION"
    assert freeze["status"] == export["status"] == "FROZEN_BEFORE_SOLVER"
    assert freeze["solver_scores_read_for_admission"] is False and export["solver_score_filtering"] is False
    assert preflight["status"] == "PASS_CPU_INTERFACE_ONLY", "INDEPENDENT_PREFLIGHT_REQUIRED"
    assert data["presentation_check"]["status"] == "PASS_RENDERED_PROMPTS_ONLY"
    assert seal["status"] == "ALL_REQUESTS_TERMINAL_RAW_SEALED" and seal["gold_read"] is False
    assert seal["automatic_retry"] is False and data["started"]["automatic_retry"] is False
    assert data["started"]["gold_read"] is False and score_payload["gold_read_after_all_raw_sealed"] is True
    assert score_payload["raw_freeze_binding"] == binding(required["raw_seal"])
    assert seal["started_binding"] == binding(required["started"])
    assert data["started"]["preflight_binding"] == binding(required["preflight"])
    assert data["started"]["preparation"] == binding(required["preparation"])
    assert data["old_seal"]["started_binding"] == binding(required["old_started"])
    assert data["old_started"]["preparation"] == binding(required["old_preparation"])
    assert export["public_binding"] == binding(required["public"])
    assert export["private_binding"] == binding(required["private"])
    assert export["question_freeze_binding"] == binding(required["question_freeze"])
    assert receipt["protocol_binding"] == binding(required["protocol"])
    assert delta["transform_receipt_binding"] == binding(required["receipt"])
    for row in protocol.get("bindings", []) + export.get("bindings", []):
        verify(row)
    for key in ("protocol_binding", "script_binding"):
        verify(receipt[key])
    for row in seal["bindings"] + seal.get("validation_bindings", []):
        verify(row)

    decisions = keyed_decisions(freeze)
    original = keyed_decisions(data["old_original_freeze"])
    supplement = keyed_decisions(data["old_supplement_freeze"])
    expected_slots = {(root, arm, slot) for root in roots for arm in conditions for slot in SLOTS}
    assert set(decisions) == set(original) == set(supplement) == expected_slots
    assert freeze["planned_slots"] == len(expected_slots) == len(receipt["records"])
    delta_by = unique(delta["questions"], lambda row: row["candidate_id"])
    receipt_by = unique(receipt["records"], lambda row: row["candidate_id"])
    all_questions = {}
    for row in decisions.values():
        candidate = row["candidate_id"]
        assert candidate in receipt_by and candidate in delta_by
        verify(row["question_binding"])
        verify(row["review_binding"])
        rr, dr = receipt_by[candidate], delta_by[candidate]
        assert row["question_binding"] == rr["repaired"] == dr["question_binding"]
        assert dr["parent_question_binding"] == rr["original"]
        assert dr["decision"] == ("FAIL" if row["status"] == "FAIL" else "ADMIT")
        assert dr["gold_evidence_source_display_unchanged"] is True
        for key in ("original", "original_content_review"):
            verify(rr[key])
        question = read(row["question_binding"]["path"])
        parent_question = read(rr["original"]["path"])
        for field in ("options", "items", "gold", "evidence", "source_binding", "display_order", "display_seed"):
            assert question[field] == parent_question[field], "REPAIR_CONTENT_DRIFT"
        assert row["parent_candidate_id"] == parent_question["candidate_id"]
        if rr["carried_operation_status"] != "PASS":
            assert row["status"] == "FAIL" and dr["carried_actual_operation"] == "FAIL"
        all_questions[candidate] = question
    retained_failure = decisions[("cc-train-episode-19053", "TARGETED", "MC5")]
    assert retained_failure["status"] == "FAIL", "KNOWN_OPERATION_FAILURE_MUST_REMAIN_FAILED"
    admitted = {row["candidate_id"]: row for row in decisions.values() if row["status"] == "ADMIT"}
    public = unique(data["public"]["questions"], lambda row: row["candidate_id"])
    private = unique(data["private"]["questions"], lambda row: row["candidate_id"])
    assert set(public) == set(private) == set(admitted)
    assert freeze["admitted_questions"] == len(admitted) == prep["questions"]

    requests = unique(data["requests"]["requests"], lambda row: row["request_id"])
    scores = unique(score_payload["records"], lambda row: row["request_id"])
    structures = unique(data["structures"]["records"], lambda row: row["request_id"])
    assert set(requests) == set(scores) == set(structures) == set(seal["request_ids"])
    prepared_bindings = unique(prep["bindings"], lambda row: row["path"])
    old_prepared_bindings = unique(data["old_preparation"]["bindings"], lambda row: row["path"])
    for request in requests.values():
        for field in ("prompt_path", "schema_path"):
            path = str(Path(request[field]).resolve())
            verify(prepared_bindings[path])
    assert len(requests) == len(admitted) * 2 == prep["requests"] == seal["requests"]
    assert data["presentation_check"]["requests"] == preflight["requests"] == len(requests)
    assert set(data["started"]["request_ids"]) == set(requests)
    assert {(row["candidate_id"], row["branch"]) for row in requests.values()} == {(cid, branch) for cid in admitted for branch in BRANCHES}
    sealed_raw = {row["path"]: row for row in seal["bindings"]}
    for rid, score in scores.items():
        request = requests[rid]
        candidate = request["candidate_id"]
        assert score["candidate_id"] == candidate and score["branch"] == request["branch"]
        assert score["raw_binding"] == sealed_raw[score["raw_binding"]["path"]]
        assert structures[rid]["raw_binding"] == score["raw_binding"]
        assert request["root_id"] == score["root_id"] == admitted[candidate]["root_id"]
        assert request["condition"] == score["condition"] == admitted[candidate]["condition"]
        assert request["display"] == public[candidate]["display"]
        assert request["source_binding"] == public[candidate]["source_binding"]
        assert request["sampling_seed"] == data["seeds"]["seeds"][candidate][score["branch"]]
        verify(score["raw_binding"])
        expected_gold = private[candidate]["display_gold"]
        assert (set(score["display_gold"]) == set(expected_gold) if request["question_type"] == "multi_select" else score["display_gold"] == expected_gold)
        score["private_slot_id"] = private[candidate]["slot_id"]
        score["question_group"] = group(score["private_slot_id"])
        assert score["mc6_control"] == (score["question_group"] == "mc6_control")

    source_bindings = {row["source_binding"]["path"]: row["source_binding"] for row in public.values()}
    for value in source_bindings.values():
        verify(value)
    original_n = sum(row["status"] == "ADMIT" for row in original.values())
    supplement_n = sum(row["status"] == "ADMIT" for row in supplement.values())
    assert original_n == data["old_original_freeze"]["admitted_before_presentation_supplement"] == data["old_report"]["original_first_pass_admitted"]
    assert supplement_n == data["old_supplement_freeze"]["admitted_questions"] == data["old_report"]["supplement_admitted"]
    admission_stages = {
        "original_first_pass": {"admitted": original_n, "planned": len(decisions), "groups": admission_table(original)},
        "old_shared_supplement": {"admitted": supplement_n, "planned": len(decisions), "groups": admission_table(supplement)},
        "current_stem_repair": {"admitted": len(admitted), "planned": len(decisions), "groups": admission_table(decisions)},
    }
    score_rows = list(scores.values())
    pooled = {branch: {name: metrics([row for row in score_rows if row["branch"] == branch and (name == "all" or row["question_group"] == name)])
                       for name in ("all",) + GROUPS} for branch in BRANCHES}
    condition_cells = [{"condition": arm, "branch": branch, **{name: metrics([row for row in score_rows if row["condition"] == arm and row["branch"] == branch and (name == "all" or row["question_group"] == name)])
                        for name in ("all",) + GROUPS}} for arm in conditions for branch in BRANCHES]
    units = []
    for root in roots:
        for arm in conditions:
            keys = [(root, arm, slot) for slot in SLOTS]
            unit = {"root_id": root, "condition": arm, "planned_slots": len(keys),
                    "original_admitted": sum(original[key]["status"] == "ADMIT" for key in keys),
                    "old_supplement_admitted": sum(supplement[key]["status"] == "ADMIT" for key in keys),
                    "repaired_admitted": sum(decisions[key]["status"] == "ADMIT" for key in keys),
                    "admitted_slots": [key[2] for key in keys if decisions[key]["status"] == "ADMIT"]}
            for branch in BRANCHES:
                unit[branch] = {name: metrics([row for row in score_rows if row["root_id"] == root and row["condition"] == arm and row["branch"] == branch and (name == "all" or row["question_group"] == name)]) for name in ("all",) + GROUPS}
            units.append(unit)
    paired = []
    for root in roots:
        common = [slot for slot in SLOTS if all(decisions[(root, arm, slot)]["status"] == "ADMIT" for arm in conditions)]
        row = {"root_id": root, "common_slots": common, "common_slot_count": len(common), "common_by_group": {}, "branch_group_cells": {}}
        for name in ("all",) + GROUPS:
            slots = [slot for slot in common if name == "all" or group(slot) == name]
            row["common_by_group"][name] = {"slots": slots, "pairs": len(slots), "requests_per_branch_across_conditions": 2 * len(slots)}
            for branch in BRANCHES:
                cells = {arm: metrics([s for s in score_rows if s["root_id"] == root and s["condition"] == arm and s["branch"] == branch and s["private_slot_id"] in slots]) for arm in conditions}
                row["branch_group_cells"][branch + "_" + name] = {"common_slots": slots, "conditions": cells,
                    "accuracy_targeted_minus_generic": cells["TARGETED"]["accuracy"] - cells["GENERIC"]["accuracy"] if slots else None}
        paired.append(row)
    effective_roots = {name: {"with_common_slots": sum(bool(row["common_by_group"][name]["pairs"]) for row in paired),
                              "planned_paired_roots": len(roots), "common_slot_pairs": sum(row["common_by_group"][name]["pairs"] for row in paired)} for name in GROUPS}
    gold_coverage = [{"gold_count": count, "planned": sum(key[2] == "MC" + str(count) for key in decisions),
                      "admitted": sum(row["status"] == "ADMIT" and key[2] == "MC" + str(count) for key, row in decisions.items()),
                      "all_correct_control": count == 6,
                      "branches": {branch: metrics([row for row in score_rows if row["branch"] == branch and row["question_type"] == "multi_select" and row["gold_count"] == count]) for branch in BRANCHES}} for count in range(2, 7)]

    # Compare only the shared parent+branch requests; changing cohort sizes are
    # never used as an estimate of repair benefit or feedback causality.
    old_scores = unique(data["old_scores"]["records"], lambda row: (row["candidate_id"], row["branch"]))
    old_requests = unique(data["old_requests"]["requests"], lambda row: (row["candidate_id"], row["branch"]))
    assert data["old_scores"]["raw_freeze_binding"] == binding(required["old_seal"])
    assert data["old_scores"]["gold_read_after_all_raw_sealed"] is True
    new_index = unique(score_rows, lambda row: (row["candidate_id"], row["branch"]))
    new_request_index = unique(list(requests.values()), lambda row: (row["candidate_id"], row["branch"]))
    parent_to_new = {row["parent_candidate_id"]: candidate for candidate, row in admitted.items()}
    old_parent_ids = {candidate for candidate, branch in old_scores}
    common_parent_ids = sorted(set(parent_to_new) & old_parent_ids)
    assert len(common_parent_ids) == supplement_n, "OLD_ADMITTED_PARENT_LOST"
    comparisons = []
    for parent in common_parent_ids:
        candidate = parent_to_new[parent]
        for branch in BRANCHES:
            old_score, new_score = old_scores[(parent, branch)], new_index[(candidate, branch)]
            old_request, new_request = old_requests[(parent, branch)], new_request_index[(candidate, branch)]
            same_seed = old_request["sampling_seed"] == new_request["sampling_seed"]
            same_display = old_request["display"] == new_request["display"]
            same_source = old_request["source_binding"] == new_request["source_binding"]
            same_gold = (set(old_score["display_gold"]) == set(new_score["display_gold"]) if new_score["question_type"] == "multi_select" else old_score["display_gold"] == new_score["display_gold"])
            assert same_seed and same_display and same_source and same_gold, "COMMON_PARENT_COMPARISON_DRIFT"
            old_prompt, new_prompt = binding(old_request["prompt_path"]), binding(new_request["prompt_path"])
            assert old_prompt == old_prepared_bindings[old_prompt["path"]], "OLD_PROMPT_PREPARATION_DRIFT"
            assert new_prompt == prepared_bindings[new_prompt["path"]], "NEW_PROMPT_PREPARATION_DRIFT"
            verify(old_score["raw_binding"])
            comparisons.append({"parent_candidate_id": parent, "candidate_id": candidate, "root_id": new_score["root_id"],
                "condition": new_score["condition"], "slot_id": new_score["private_slot_id"], "question_group": new_score["question_group"], "branch": branch,
                "same_sampling_seed": same_seed, "sampling_seed": new_request["sampling_seed"], "same_display": same_display,
                "same_source_binding": same_source, "source_binding": new_request["source_binding"], "same_display_gold": same_gold,
                "old_prompt_binding": old_prompt, "new_prompt_binding": new_prompt, "prompt_bytes_identical": old_prompt["sha256"] == new_prompt["sha256"],
                "old_raw_binding": old_score["raw_binding"], "new_raw_binding": new_score["raw_binding"],
                "old_score": {key: old_score.get(key) for key in ("exact", "valid", "prediction", "display_prediction", "display_gold", "false_positive", "false_negative", "pairwise_correct", "pairwise_total")},
                "new_score": {key: new_score.get(key) for key in ("exact", "valid", "prediction", "display_prediction", "display_gold", "false_positive", "false_negative", "pairwise_correct", "pairwise_total")}})
    common_cells = []
    for branch in BRANCHES:
        for name in ("all",) + GROUPS:
            chosen = [row for row in comparisons if row["branch"] == branch and (name == "all" or row["question_group"] == name)]
            common_cells.append({"branch": branch, "question_group": name, **compare_metrics(chosen),
                "identical_prompt": compare_metrics([row for row in chosen if row["prompt_bytes_identical"]]),
                "changed_prompt": compare_metrics([row for row in chosen if not row["prompt_bytes_identical"]])})

    reason_scores = [row for row in score_rows if row["branch"] == "reason"]
    reason_structures = [row for row in structures.values() if row["branch"] == "reason"]
    reason_review = {"scope": "STRUCTURAL_ONLY_NOT_NEW_FULL_SEMANTIC_REVIEW", "records": len(reason_scores),
        "reason_structure_valid": sum(row.get("reason_structure_valid") is True for row in reason_scores),
        "reason_structure_invalid": sum(row.get("reason_structure_valid") is False for row in reason_scores),
        "reason_structure_status_missing": sum(row.get("reason_structure_valid") is None for row in reason_scores),
        "items_with_any_reference": sum(row.get("mechanical_consistency", {}).get("items_with_any_reference", 0) for row in reason_structures),
        "supported_items_without_reference": sum(row.get("mechanical_consistency", {}).get("supported_items_without_reference", 0) for row in reason_structures),
        "final_exact": metrics(reason_scores), "citation_reliability": "NOT_SEMANTICALLY_REVIEWED_THIS_RUN",
        "actor_attribution": "NOT_SEMANTICALLY_REVIEWED_THIS_RUN", "relation_reasoning": "NOT_SEMANTICALLY_REVIEWED_THIS_RUN",
        "raw_saved_or_legal_reference_ids_do_not_establish_semantic_support": True,
        "old_fbc1_reason_reviews_not_transferred_to_new_responses": True}
    execution = {"requests": len(requests), "submitted": seal["submitted_requests"], "returned": seal["returned_requests"],
                 "request_errors": seal["request_errors"], "failure": seal["failure"], "wall_seconds": seal["wall_seconds"],
                 "automatic_retry": False, "gold_read_after_all_raw_sealed": True,
                 "max_prompt_tokens": prep["max_prompt_tokens"], "min_context_headroom": prep["min_context_headroom"],
                 "temperature": prep["temperature"], "top_p": prep["top_p"], "native_hidden_thinking": prep["native_hidden_thinking"],
                 "max_model_len": prep["max_model_len"], "model_role": prep["model_role"], "model_directory": prep["model_directory"]}
    assert execution["submitted"] <= execution["requests"] and execution["returned"] <= execution["submitted"]
    assert prep["source_truncation"] is False and prep["generation_reads_gold"] is False
    assert len({row["root_id"] for row in admitted.values()}) <= len(roots)
    integration_path = BASE / "MAINLINE_INTEGRATION_REVIEW.json"
    optional_bindings = []
    integration = None
    if integration_path.exists():
        integration = read(integration_path)
        assert integration["status"].startswith("PASS"), "MAINLINE_INTEGRATION_REVIEW_NOT_PASS"
        for value in integration.get("bindings", []):
            verify(value)
        optional_bindings.append(binding(integration_path))
    integration_zh = BASE / "MAINLINE_INTEGRATION_REVIEW_ZH.md"
    if integration_zh.exists():
        optional_bindings.append(binding(integration_zh))
    limitations = [
        "The independent units are four paired Train roots, not the number of questions, requests, options or citations.",
        "The same eight frozen source units are reused; no new source expansion or author generation occurred.",
        "The repair was designed after observing FBC1 defects and aggregate outcomes; this is an exploratory presentation retest.",
        "Original full-source key, grounding and operation reviews are carried with exact field invariance; delta review is not fresh full semantic adjudication.",
        "The known 19053 TARGETED MC5 operation failure remains failed; six-select-five has only one distractor and cannot prove strong joint-constraint discrimination.",
        "MC6 is privately an all-correct control. Always selecting all receives final credit there and does not prove distractor quality.",
        "Visible explanations were stored and structurally checked but were not semantically re-reviewed in full; legality or presence of citations is not reliability.",
        "Changed admission denominators cannot estimate performance improvement. The same-parent common subset is reported separately.",
        "Common-parent seeds, display and source are fixed, but actual prompt equality is measured. One sampled response per branch is descriptive, not causal repair or feedback evidence.",
        "Different branch seeds and one sample prohibit attributing direct-versus-reason differences solely to explanation generation.",
        "Initial sources were historically seen by M-clean; no unseen-root generalization claim is made.",
        "No training, formal stable weakness admission, training-pool admission, fresh-author stability, self-play causal closure or paper-completion claim is made.",
    ]
    inputs = [binding(path) for path in required.values()] + optional_bindings + [binding(__file__)]
    report = {"schema_version": "qpr1_final_presentation_retest_report_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_WITH_EXECUTION_ERRORS" if seal["failure"] or seal["request_errors"] else "COMPLETED",
        "scope": "Frozen-draft presentation repair retest; no fresh authoring or source expansion", "independent_units": len(roots),
        "independent_unit": "paired root", "roots": roots, "conditions": conditions,
        "planned_question_slots": len(decisions), "admission_stages": admission_stages,
        "transform_counts": {key: receipt[key] for key in ("candidate_count", "changed_stems", "which_count_removed", "exact_count_removed", "neutral_mc6_stems")},
        "retained_failures": [row for row in decisions.values() if row["status"] != "ADMIT"],
        "units": units, "paired_root_contrasts": paired, "paired_roots_with_common_slots_by_group": effective_roots,
        "mc_gold_coverage": gold_coverage, "pooled_solver_cells": pooled, "condition_solver_cells": condition_cells,
        "execution": execution, "reason_review": reason_review,
        "common_parent_comparison": {"questions": len(common_parent_ids), "requests": len(comparisons),
            "identical_prompt_requests": sum(row["prompt_bytes_identical"] for row in comparisons),
            "changed_prompt_requests": sum(not row["prompt_bytes_identical"] for row in comparisons),
            "all_same_seed_display_source_gold": all(row["same_sampling_seed"] and row["same_display"] and row["same_source_binding"] and row["same_display_gold"] for row in comparisons),
            "cells": common_cells, "records": comparisons, "interpretation": "Descriptive only; no mixed-denominator repair or feedback effect estimate"},
        "mainline_integration_review_status": integration["status"] if integration else "NOT_PRESENT",
        "limitations": limitations, "source_expansion": False, "author_regeneration": False, "fresh_author_stability_claim": False,
        "training_performed": False, "training_pool_admission": False, "formal_weakness_claim": False,
        "causal_feedback_benefit_claim": False, "bindings": inputs}

    labels = {"main_mc2_5": "多选2–5个正确项", "ordering": "排序整题完全正确", "mc6_control": "MC6全对控制"}
    lines = ["# 冻结题稿呈现修复复验报告", "", "本轮复用FBC1的4个Train root、8组冻结对话，不再扩充来源或重新出题。独立实验单位为4对root。", "",
        f"准入：原始 {original_n}/{len(decisions)} → 旧探索性补充 {supplement_n}/{len(decisions)} → 本次固定呈现修复 {len(admitted)}/{len(decisions)}。这是旧题回收率，不是新作者稳定出题率。", "",
        f"变换：删除 {receipt['which_count_removed']} 处Which数量提示、{receipt['exact_count_removed']} 处Select exactly数量提示；{receipt['neutral_mc6_stems']} 道MC6使用中性题干，实际改变{receipt['changed_stems']}个题干。Gold、选项/条目、证据、来源和显示顺序保持不变。", "",
        "| 题型 | 原始准入 | 旧补充准入 | 本次准入 |", "|---|---:|---:|---:|"]
    for name in GROUPS:
        columns = [admission_stages[stage]["groups"][name] for stage in ("original_first_pass", "old_shared_supplement", "current_stem_repair")]
        lines.append("| " + labels[name] + " | " + " | ".join(f"{x['admitted']}/{x['planned']}" for x in columns) + " |")
    lines += ["", "19053 TARGETED MC5仍因实际操作退化为单一train绑定而失败；呈现修复没有恢复这道题。所有原稿和失败记录保留。", "",
        f"solver共计划{execution['requests']}次、提交{execution['submitted']}次、返回{execution['returned']}次；运行{execution['wall_seconds']:.1f}秒。重试关闭，原始响应全部封存后才读取私有评分。", "",
        "| 最终答案指标 | 直接作答 | 可见理由分支 |", "|---|---:|---:|"]
    for name in GROUPS:
        lines.append(f"| {labels[name]} | {fraction(pooled['direct'][name])} | {fraction(pooled['reason'][name])} |")
    for branch, title in (("direct", "直接作答"), ("reason", "可见理由")):
        value = pooled[branch]["ordering"]
        lines.append(f"\n{title}排序pairwise：{value['ordering_pairwise_correct']}/{value['ordering_pairwise_total']}（{pct(value['ordering_pairwise_accuracy'])}）。")
    lines += ["", "| 条件 | 分支 | 多选2–5 | 排序整题 | MC6控制 | 排序pairwise |", "|---|---|---:|---:|---:|---:|"]
    for cell in condition_cells:
        order = cell["ordering"]
        lines.append(f"| {cell['condition']} | {cell['branch']} | {fraction(cell['main_mc2_5'])} | {fraction(order)} | {fraction(cell['mc6_control'])} | {order['ordering_pairwise_correct']}/{order['ordering_pairwise_total']} |")
    lines += ["", "| root | GENERIC准入 | TARGETED准入 | 共同多选槽位对 | 共同排序槽位对 | 共同MC6槽位对 |", "|---|---:|---:|---:|---:|---:|"]
    for pair in paired:
        counts = {u["condition"]: u["repaired_admitted"] for u in units if u["root_id"] == pair["root_id"]}
        lines.append(f"| {pair['root_id']} | {counts['GENERIC']}/7 | {counts['TARGETED']}/7 | {pair['common_by_group']['main_mc2_5']['pairs']} | {pair['common_by_group']['ordering']['pairs']} | {pair['common_by_group']['mc6_control']['pairs']} |")
    lines += ["", "每root、每条件、每分支及共同槽位的完整正确数、分母和pairwise详见JSON与CSV；条件差异仅描述，不视为反馈因果收益。", "",
        f"旧新共同子集：{len(common_parent_ids)}道题、{len(comparisons)}个题目×分支请求。全部保持相同parent seed、display、source与Gold；实际{sum(r['prompt_bytes_identical'] for r in comparisons)}个prompt字节相同，{sum(not r['prompt_bytes_identical'] for r in comparisons)}个发生改变。逐请求新旧prompt哈希与最终结果见JSON。", "",
        "| 共同题最终正确率（相同题分母） | 分支 | 旧 | 新 |", "|---|---|---:|---:|"]
    for cell in common_cells:
        if cell["question_group"] in GROUPS:
            lines.append(f"| {labels[cell['question_group']]} | {cell['branch']} | {fraction(cell['old'])} | {fraction(cell['new'])} |")
    lines += ["", "相同prompt/变更prompt分别统计了正确→正确、正确→错误、错误→正确和错误→错误；完整表位于JSON。相同种子与单次重测不构成因果证据，也不能把33题和本轮全部题混合分母后的准确率差称为修复收益。", "",
        f"本轮reason仅做结构校验：{reason_review['reason_structure_valid']}/{reason_review['records']}条理由结构有效，{reason_review['reason_structure_invalid']}条无效，{reason_review['reason_structure_status_missing']}条缺状态。可见解释和原始响应已保存，但没有进行新的全量引用、归因与关系语义审核。合法引用ID或保存raw不等于引用可靠，旧FBC1理由审核结论也不转移给新响应。", "",
        "本轮未训练、未正式入池、未建立新作者稳定性或反馈因果收益，也不代表整体self-play方案或论文实验完成。", "",
        f"公共门禁回归状态：{data['gate_regression']['status']}；运行时呈现检查：{data['presentation_check']['status']}；独立CPU preflight：{preflight['status']}。", "",
        f"主流程接入检查：{report['mainline_integration_review_status']}。范围仅为已审核的版本化公共导出入口；历史流程不据此宣称全部迁移。", "",
        "结果文件：FINAL_REPORT.json、PAIRED_ROOT_RESULTS.csv、FINAL_ARTIFACT_MANIFEST.json。所有输入与代码绑定纳入manifest，所有输出独占写入，不覆盖冻结材料。", ""]
    md = "\n".join(lines)
    csv_file = io.StringIO(newline="")
    fields = ["root_id", "condition", "branch", "question_group", "planned_slots", "original_admitted", "old_supplement_admitted", "repaired_admitted", "common_slots_in_root_group", "n", "valid", "invalid", "correct", "accuracy", "false_positive", "false_negative", "ordering_pairwise_correct", "ordering_pairwise_total", "ordering_pairwise_accuracy"]
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    paired_by = {row["root_id"]: row for row in paired}
    for unit in units:
        for branch in BRANCHES:
            for name in ("all",) + GROUPS:
                writer.writerow({"root_id": unit["root_id"], "condition": unit["condition"], "branch": branch, "question_group": name,
                    "planned_slots": sum(name == "all" or group(slot) == name for slot in SLOTS),
                    "original_admitted": sum(original[(unit['root_id'], unit['condition'], slot)]["status"] == "ADMIT" for slot in SLOTS if name == "all" or group(slot) == name),
                    "old_supplement_admitted": sum(supplement[(unit['root_id'], unit['condition'], slot)]["status"] == "ADMIT" for slot in SLOTS if name == "all" or group(slot) == name),
                    "repaired_admitted": unit[branch][name]["n"],
                    "common_slots_in_root_group": paired_by[unit["root_id"]]["common_by_group"][name]["pairs"], **unit[branch][name]})
    payloads = {"FINAL_REPORT.json": serialized(report), "FINAL_REPORT_ZH.md": md,
                "PAIRED_ROOT_RESULTS.csv": csv_file.getvalue()}
    future_bindings = []
    for name, content in payloads.items():
        encoded = content.encode("utf-8")
        future_bindings.append({"path": str(BASE / name), "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)})
    manifest = {"schema_version": "qpr1_final_artifact_manifest_v1", "created_at_utc": report["created_at_utc"],
                "status": "SEALED_REPORT_OUTPUTS", "output_bindings": future_bindings,
                "input_bindings": inputs, "self_hash_not_embedded": True, "no_overwrite": True}
    payloads["FINAL_ARTIFACT_MANIFEST.json"] = serialized(manifest)
    assert set(payloads) == set(OUTPUTS) and all(not (BASE / name).exists() for name in payloads)
    # Every calculation, source check, validation and serialization is complete.
    # Open all exclusive targets before writing; never use replacement mode.
    with ExitStack() as stack:
        handles = {name: stack.enter_context((BASE / name).open("x", encoding="utf-8", newline="")) for name in OUTPUTS}
        for name in OUTPUTS:
            handles[name].write(payloads[name])
    print(json.dumps({"status": report["status"], "admitted": len(admitted), "requests": len(requests),
                      "common_questions": len(common_parent_ids), "reports": [binding(BASE / name) for name in OUTPUTS]}))


if __name__ == "__main__":
    main()
