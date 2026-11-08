#!/usr/bin/env python3
"""Append exact saved F136 five-MC verification facts to the frozen report.

The frozen observer, renderer, report writer and hash interface are reused.
Only the additional engineering disclosure is inserted in memory. Statistics,
scores, methods, semantic decisions and pre-existing artifacts are untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import question_driver as q
import content_assembly_delivery as content
import finalize_report as frozen

DISCLOSURE = BASE / "F136202_REPORT_DISCLOSURE.json"
RAW_DISCLOSURE = BASE / "UNFROZEN_RAW_REPORT_DISCLOSURE.json"
FIELD = "f136202_saved_five_MC_first_verifier"
RAW_FIELD = "persisted_unfrozen_solver_raw"
ORIGINAL_ENGINEERING = frozen.engineering_disclosures
ORIGINAL_RENDER = frozen.render_markdown


def _slot_decisions(candidates: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = content.continuation.by_id(reviews, "report-delivery-reviews")
    result = {}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        if cid not in lookup:
            raise RuntimeError("REPORT_F136_FINAL_REVIEW_MISSING:" + cid)
        result[candidate["slot_id"]] = lookup[cid].get("decision")
    return result


def factual_disclosure(strict: bool = True) -> dict[str, Any]:
    """Read actual artifacts only; a missing semantic decision stays missing."""
    old_candidates = q.read(content.INPUT / "CANDIDATES_FINAL.json")
    old_reviews = q.read(content.INPUT / "CONTENT_REVIEW_FINAL.json")["reviews"]
    old_decisions = _slot_decisions(old_candidates, old_reviews)
    if old_decisions != {slot: "HOLD" if slot in content.SLOTS else "PASS"
                         for slot in q.SLOTS}:
        raise RuntimeError("REPORT_F136_BOUND_PRIOR_THREE_ORDER_FIVE_MC_STATUS_DRIFT")
    paths = [Path(__file__).resolve(), Path(content.__file__).resolve(), Path(frozen.__file__).resolve(),
             content.INPUT / "CANDIDATES_FINAL.json", content.INPUT / "CONTENT_REVIEW_FINAL.json",
             content.INPUT / "CONTENT_VERIFIER_CALL_revision_01.json",
             BASE / "mechanical_audits/F136202_SAVED_FINAL_GOLD_EXACT_FIVE_FIRST_VERIFIER_INSPECTION.json"]
    result: dict[str, Any] = {
        "schema_version": "fgr20_f136202_saved_MC_five_first_verifier_disclosure_v1",
        "root_id": content.ROOT, "arm": content.ARM,
        "serialization_variant": content.VARIANT,
        "report_extension_binding": q.binding(Path(__file__).resolve()),
        "content_extension_binding": q.binding(Path(content.__file__).resolve()),
        "frozen_report_assembler_binding": q.binding(Path(frozen.__file__).resolve()),
        "prior_final_slot_decisions": old_decisions,
        "only_new_verifier_call_id": content.NEW_CALL,
        "new_verifier_exact_slot_ids": list(content.SLOTS),
        "retained_existing_review_slot_ids": list(content.RETAIN),
        "prior_ORDER_verification_repeated": False,
        "prior_final_candidate_count": len(old_candidates),
        "new_plan_author_Gold_or_revision_calls_authorized": 0,
        "initial_statistics_replaced": False,
        "report_metrics_analysis_or_scores_changed": False,
        "semantic_admission_asserted_by_engineering": False,
        "terminal_observed": content.TERMINAL.is_file(),
    }
    if content.TERMINAL.is_file():
        checked = content.validate_f136_terminal_ready()
        row = checked["record"]
        candidates = q.read(content.OUT / "CANDIDATES_FINAL.json")
        reviews = q.read(content.OUT / "CONTENT_REVIEW_FINAL.json")["reviews"]
        normalizer = q.read(content.OUT / "MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json")
        gate = q.read(content.OUT / "FINAL_REVIEW_GATE.json")
        call = content.OUT / "CONTENT_VERIFIER_CALL_revision_01.json"
        result.update({
            "status": "OBSERVED_TERMINAL_EXACT_FIVE_FIRST_VERIFIER",
            "terminal_ready_binding": q.binding(content.TERMINAL),
            "actual_new_verifier_return_receipt_binding": q.binding(call) if call.is_file() else None,
            "actual_new_verifier_response_received": call.is_file(),
            "terminal_review_slot_decisions": _slot_decisions(candidates, reviews),
            "published_final_candidate_count": len(candidates),
            "published_final_review_count": len(reviews),
            "candidate_bytes_preserved": checked["candidate_bytes_preserved"],
            "existing_ORDER_review_objects_preserved": checked["existing_ORDER_review_objects_preserved"],
            "prior_final_reviews_archive_binding": row["prior_final_reviews_archive_binding"],
            "retained_ORDER_reviews_binding": row["retained_ORDER_reviews_binding"],
            "existing_ORDER_verifier_receipt_binding": row["existing_ORDER_verifier_receipt_binding"],
            "saved_Gold_normalization_binding": q.binding(content.OUT / "MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json"),
            "recorded_additional_list_keyed_transformations": len(normalizer.get("changes", [])),
            "recorded_round_trip_verified": normalizer.get("exact_round_trip_verified"),
            "terminal_gate_admitted_slot_ids": [c["slot_id"] for c in candidates
                if next((r.get("admitted") for r in gate.get("results", [])
                         if r.get("candidate_id") == c["candidate_id"]), False) is True],
            "source_binding": candidates[0]["source_binding"],
        })
        paths.extend(Path(binding["path"]) for binding in checked["input_bindings"])
    else:
        disposition = content.missingness_disposition() if strict else None
        result.update({"status": "EXPLICIT_PARENT_TERMINAL_MISSINGNESS" if disposition else "PREPARATION_OR_NOT_RUN_ONLY",
                       "terminal_ready_binding": None, "actual_new_verifier_return_receipt_binding": None,
                       "actual_new_verifier_response_received": False,
                       "terminal_review_slot_decisions": None,
                       "published_final_candidate_count": None,
                       "published_final_review_count": None,
                       "candidate_bytes_preserved": None,
                       "existing_ORDER_review_objects_preserved": None,
                       "terminal_gate_admitted_slot_ids": None,
                       "missingness_disposition": disposition,
                       "source_binding": old_candidates[0]["source_binding"]})
        if disposition:
            paths.append(Path(disposition["binding"]["path"]))
            paths.extend(Path(row["path"]) for row in disposition["record"]["evidence_bindings"])
        for path in sorted(content.OUT.glob("*.json")):
            paths.append(path)
    result["input_bindings"] = [q.binding(path) for path in sorted(set(paths), key=str)]
    for binding in result["input_bindings"]:
        q.verify_binding(binding)
    return result


def markdown_section(facts: dict[str, Any]) -> str:
    lines = ["## F136 五道 MC 首次核验的保存事实", "",
        "F136/F 的既有最终题共有八道。原 verifier 的实际请求仅覆盖三道 ORDER；五道 MC 的最终独立 Gold 已保存，但未曾进入该 verifier 请求。专项工程包装仅允许把已保存的五份完整六项理由列表按 option_id 可逆改为字典，并向一个 fresh verifier 发送这五道 MC；实际运行及返回由以下保存状态披露。原八题字节和三份 ORDER 判断须保持原值；不新增计划、作者、Gold 或实质修订。", "",
        f"记录状态：`{facts.get('status')}`；实际新 verifier 返回回执：`{facts.get('actual_new_verifier_response_received')}`。",
        "初稿的失败和产出率保持既有口径，专项核验不会回填初审统计。工程绑定与结构检查不构成语义 PASS。", "",
        "| 槽 | 原保存审核 | 专项最终保存审核 |", "| --- | --- | --- |"]
    current = facts.get("terminal_review_slot_decisions")
    for slot in q.SLOTS:
        value = current.get(slot) if isinstance(current, dict) else "未观察到专项终态"
        lines.append(f"| {slot} | {facts['prior_final_slot_decisions'].get(slot)} | {value} |")
    lines.extend(["", f"唯一授权新 call：`{facts.get('only_new_verifier_call_id')}`。",
        f"专项终态 READY：`{facts.get('terminal_ready_binding')}`。",
        f"新 verifier 回执：`{facts.get('actual_new_verifier_return_receipt_binding')}`。",
        f"逐文件事实披露回执：`{facts.get('disclosure_receipt_binding')}`。", ""])
    if facts.get("missingness_disposition") is not None:
        lines.extend([f"父级明确缺额处置：`{facts['missingness_disposition']}`。五道 MC 不因缺少新终态而获得准入。", ""])
    return "\n".join(lines)


def unfrozen_raw_disclosure(base: Path = BASE) -> dict[str, Any]:
    """Count persisted RETURNED records only when no complete raw freeze exists."""
    freeze = base / "execution/RAW_RESPONSE_FREEZE.json"
    if freeze.is_file():
        return {"schema_version": "fgr20_persisted_unfrozen_raw_disclosure_v1",
                "raw_response_freeze_present": True, "raw_response_freeze_binding": q.binding(freeze),
                "persisted_RETURNED_unfrozen_count": None, "observed_raw_record_count": None,
                "records": [], "input_bindings": [q.binding(freeze)],
                "interpretation": "Complete raw freeze exists; the frozen runner/report denominators remain authoritative.",
                "strict_denominators_changed": False, "scores_imputed": False,
                "analyze_or_model_call_launched": False}
    records = []
    for path in sorted((base / "execution/raw").glob("*.json")):
        row = q.read(path)
        if not isinstance(row, dict):
            raise RuntimeError("UNFROZEN_RAW_REPORT_RECORD_NOT_OBJECT:" + str(path))
        records.append({"binding": q.binding(path), "request_id": row.get("request_id"),
                        "saved_status": row.get("status"),
                        "persisted_RETURNED": row.get("status") == "RETURNED"})
    return {"schema_version": "fgr20_persisted_unfrozen_raw_disclosure_v1",
            "raw_response_freeze_present": False, "raw_response_freeze_binding": None,
            "persisted_RETURNED_unfrozen_count": sum(row["persisted_RETURNED"] for row in records),
            "observed_raw_record_count": len(records), "records": records,
            "input_bindings": [row["binding"] for row in records],
            "interpretation": "Persisted RETURNED raw records without a complete RAW_RESPONSE_FREEZE are disclosed as not fully sealed and unscored. They are not added to strict denominators and are never replayed or imputed as zero.",
            "strict_denominators_changed": False, "scores_imputed": False,
            "analyze_or_model_call_launched": False}


def install_disclosure(facts: dict[str, Any], raw_facts: dict[str, Any]) -> None:
    def engineering(base: Path, issues: list[str]) -> dict[str, Any]:
        if base.resolve() != BASE.resolve():
            raise RuntimeError("REPORT_DELIVERY_ONLY_EXACT_TASK_DIRECTORY")
        value = ORIGINAL_ENGINEERING(base, issues)
        value[FIELD] = copy.deepcopy(facts)
        value[RAW_FIELD] = copy.deepcopy(raw_facts)
        value["associated_artifact_bindings"] += (facts["input_bindings"] + [facts["disclosure_receipt_binding"]]
            + raw_facts["input_bindings"] + [raw_facts["disclosure_receipt_binding"]])
        return value
    def render(report: dict[str, Any], base: Path, failures: dict[str, Any], usage: dict[str, Any],
               issues: list[str], after: Any) -> str:
        markdown = ORIGINAL_RENDER(report, base, failures, usage, issues, after)
        section = markdown_section(report["engineering_disclosures"][FIELD])
        observed = report["engineering_disclosures"][RAW_FIELD]
        if not observed["raw_response_freeze_present"]:
            section += ("\n### 未完整封存的 Solver 原始返回\n\n"
                + f"逐条已持久化但未完整封存/未评分的 `RETURNED` raw 记录数：**{observed['persisted_RETURNED_unfrozen_count']}**；"
                + f"raw 目录记录总数：{observed['observed_raw_record_count']}。无 `RAW_RESPONSE_FREEZE.json` 时不运行 analyze、不重跑，也不补 0；主冻结/strict 分母保持原口径。"
                + f"逐条路径与 SHA256 保存在 `{observed['disclosure_receipt_binding']}`，并由最终 manifest 绑定。\n")
        marker = "## 复现结果\n"
        if markdown.count(marker) != 1:
            raise RuntimeError("FROZEN_REPORT_RESULT_SECTION_STRUCTURE_DRIFT")
        return markdown.replace(marker, section + "\n" + marker, 1)
    frozen.engineering_disclosures = engineering
    frozen.render_markdown = render


def write_report() -> dict[str, Any]:
    approved = content.require_approval()
    if approved.get("report_wrapper_binding") != q.binding(Path(__file__).resolve()):
        raise RuntimeError("REPORT_DELIVERY_WRAPPER_APPROVAL_DRIFT")
    review = q.read(approved["independent_review_binding"]["path"])
    if review.get("report_script_binding") != approved["report_wrapper_binding"]:
        raise RuntimeError("REPORT_DELIVERY_INDEPENDENT_REVIEW_BINDING_DRIFT")
    facts = factual_disclosure(strict=True)
    receipt = frozen.publish_immutable(DISCLOSURE, facts)
    facts = {**facts, "disclosure_receipt_binding": receipt}
    raw_facts = unfrozen_raw_disclosure()
    raw_receipt = frozen.publish_immutable(RAW_DISCLOSURE, raw_facts)
    raw_facts = {**raw_facts, "disclosure_receipt_binding": raw_receipt}
    build_path = BASE / "FINAL_REPORT_BUILD.json"
    if build_path.is_file():
        existing = q.read(build_path).get("report", {}).get("engineering_disclosures", {}).get(FIELD)
        if existing != facts:
            raise RuntimeError("FROZEN_REPORT_DELIVERY_EXTENSION_DRIFT")
        if q.read(build_path).get("report", {}).get("engineering_disclosures", {}).get(RAW_FIELD) != raw_facts:
            raise RuntimeError("FROZEN_REPORT_UNFROZEN_RAW_DISCLOSURE_DRIFT")
    install_disclosure(facts, raw_facts)
    return frozen.write_report(BASE)


def self_check() -> dict[str, Any]:
    fixture = {"prior_final_slot_decisions": {slot: "HOLD" for slot in q.SLOTS},
               "terminal_review_slot_decisions": None, "status": "ENGINEERING_FIXTURE",
               "actual_new_verifier_response_received": False}
    section = markdown_section(fixture)
    assert "未观察到专项终态" in section and "工程绑定与结构检查不构成语义 PASS" in section
    assert len([line for line in section.splitlines() if line.startswith("| MC") or line.startswith("| ORDER")]) == 8
    fixture["terminal_review_slot_decisions"] = {slot: "HOLD" for slot in q.SLOTS}
    section = markdown_section(fixture)
    assert "未观察到专项终态" not in section
    cache = BASE / "runtime_cache/finalize_report_delivery_selfcheck"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fgr20_unfrozen_raw_report_", dir=cache) as directory:
        base = Path(directory); raw = base / "execution/raw"; raw.mkdir(parents=True)
        for index, status in enumerate(("RETURNED", "RETURNED", "BACKEND_ERROR")):
            (raw / f"fixture_{index}.json").write_text(json.dumps({"request_id": f"fixture_{index}", "status": status}))
        observed = unfrozen_raw_disclosure(base)
        assert observed["persisted_RETURNED_unfrozen_count"] == 2 and observed["observed_raw_record_count"] == 3
        assert observed["strict_denominators_changed"] is False and observed["scores_imputed"] is False
        (base / "execution/RAW_RESPONSE_FREEZE.json").write_text("{}")
        assert unfrozen_raw_disclosure(base)["persisted_RETURNED_unfrozen_count"] is None
    return {"status": "PASS_ENGINEERING_ONLY", "model_calls": 0, "semantic_calls_started": 0,
            "statistics_or_scores_changed": False, "missing_terminal_not_filled_as_PASS": True,
            "eight_slot_disclosure_uses_only_supplied_saved_decisions": True,
            "frozen_observer_renderer_writer_reused": True, "actual_report_written": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-check", action="store_true")
    group.add_argument("--inspect", action="store_true")
    group.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = self_check() if args.self_check else factual_disclosure(strict=False) if args.inspect else write_report()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
