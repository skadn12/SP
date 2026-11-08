"""Count untouched-model yield through independent automatic review stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
ROOTS = ("210046", "231899", "194259", "96544")
MULTI_RAW = DELIVERIES / "h4_crossroot_multiselect_batch_v28_newroots_quota4to8"
MULTI_REVIEW = DELIVERIES / "h4_crossroot_multiselect_auto_review_v2_deepseek"
ORDER_REVIEW = DELIVERIES / "h4_crossroot_ordering_no_codex_pilot_v1"
TYPES = ("multi_select", "ordering")


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def candidate(root: str, kind: str, qid: str, mechanical: bool, *, raw: bool) -> dict:
    return {"root_id": root, "question_type": kind, "candidate_id": qid,
            "raw_mechanical_pass": raw, "mechanical_pass": mechanical,
            "semantic": "PENDING" if mechanical else "SKIPPED_MECHANICAL",
            "solver": "PENDING" if mechanical else "SKIPPED_MECHANICAL",
            "dedup": "PENDING" if mechanical else "SKIPPED_MECHANICAL"}


def raw_inputs() -> tuple[dict[str, dict], dict[str, str]]:
    rows: dict[str, dict] = {}
    inputs: dict[str, str] = {}
    for suffix in ROOTS:
        root = "cc-train-episode-" + suffix
        directory = MULTI_RAW / root / "batch-01"
        draft_path = directory / "MODEL_DRAFT.json"
        checks_path = directory / "MECHANICAL_CHECKS.json"
        draft, checks = read(draft_path), read(checks_path)
        require(draft["root_id"] == checks["root_id"] == root, "multi_root_mismatch:" + root)
        require(not checks["fatal_errors"], "multi_fatal_mechanical_error:" + root)
        questions = draft["questions"]
        decisions = checks["question_checks"]
        require(len(questions) == len(decisions) == 4, "multi_draft_count_mismatch:" + root)
        for question, check in zip(questions, decisions):
            qid = question["candidate_id"]
            require(qid == check["candidate_id"] and qid not in rows,
                    "multi_candidate_binding_invalid:" + qid)
            passed = check["mechanically_valid"] is True and not check["errors"]
            rows[qid] = candidate(root, "multi_select", qid, passed, raw=passed)
        require(sum(row["mechanical_pass"] for row in rows.values()
                    if row["root_id"] == root and row["question_type"] == "multi_select")
                == checks["mechanically_valid_count"], "multi_mechanical_count_mismatch:" + root)
        inputs[str(draft_path)] = sha(draft_path)
        inputs[str(checks_path)] = sha(checks_path)

    report_path = ORDER_REVIEW / "REPORT.json"
    rebound_path = ORDER_REVIEW / "LABEL_REBOUND_UNREVIEWED.json"
    report, rebound = read(report_path), read(rebound_path)
    rebound_ids = {q["candidate_id"] for q in rebound}
    require(len(rebound_ids) == len(rebound), "ordering_rebound_duplicate_ids")
    require(report["generator_model"] == "Qwen/Qwen3.6-35B-A3B", "ordering_generator_mismatch")
    require({row["root_id"] for row in report["roots"]}
            == {"cc-train-episode-" + root for root in ROOTS}, "ordering_root_set_mismatch")
    for group in report["roots"]:
        root = group["root_id"]
        require(group["generated_count"] == len(group["checks"]) == 4,
                "ordering_draft_count_mismatch:" + root)
        for check in group["checks"]:
            qid = check["candidate_id"]
            require(qid not in rows, "ordering_duplicate_id:" + qid)
            raw_pass = not check["raw_mechanical_errors"]
            label_pass = check["label_only_mechanical_candidate"] is True
            require(label_pass == (qid in rebound_ids), "ordering_rebound_id_mismatch:" + qid)
            rows[qid] = candidate(root, "ordering", qid, label_pass, raw=raw_pass)
    inputs[str(report_path)] = sha(report_path)
    inputs[str(rebound_path)] = sha(rebound_path)
    return rows, inputs


def multi_semantic(rows: dict[str, dict], inputs: dict[str, str], directory: Path) -> None:
    for suffix in ROOTS:
        root = "cc-train-episode-" + suffix
        candidates = {qid for qid, row in rows.items() if row["root_id"] == root
                      and row["question_type"] == "multi_select" and row["mechanical_pass"]}
        prepared_path = directory / root / "PREPARED.json"
        result_path = directory / root / "REVIEW_RESULT_POLICY_V3.json"
        if not prepared_path.exists() or not result_path.exists():
            continue
        prepared, result = read(prepared_path), read(result_path)
        inputs[str(prepared_path)] = sha(prepared_path)
        inputs[str(result_path)] = sha(result_path)
        require(prepared["root_id"] == result["root_id"] == root,
                "multi_review_root_mismatch:" + root)
        for name, expected in prepared["source_hashes"].items():
            source = MULTI_RAW / root / "batch-01" / name
            require(sha(source) == expected, "multi_review_source_changed:" + str(source))
        if not candidates:
            continue
        if result.get("status") != "REVIEWED_FAIL_CLOSED":
            continue
        response_path = directory / root / "RESPONSE.json"
        if (not response_path.exists()
                or result.get("response_sha256") != sha(response_path)):
            continue
        decisions = result.get("candidate_decisions")
        if not isinstance(decisions, dict) or set(decisions) != candidates:
            continue
        if result.get("returned_model") != prepared["review_model"]:
            continue
        approved = {qid for qid, value in decisions.items() if value.get("approved") is True}
        rejected = {qid for qid, value in decisions.items() if value.get("approved") is False}
        if (approved | rejected != candidates or approved & rejected
                or approved != set(result.get("approved_ids", []))
                or rejected != set(result.get("rejected_ids", []))):
            continue
        for qid in approved:
            rows[qid]["semantic"] = "PASS"
        for qid in rejected:
            rows[qid]["semantic"] = "REJECT"
            rows[qid]["solver"] = "SKIPPED_SEMANTIC"
            rows[qid]["dedup"] = "SKIPPED_SEMANTIC"


def ordering_semantic(rows: dict[str, dict], inputs: dict[str, str], directory: Path) -> None:
    first_path = directory / "semantic_review_deepseek_v2" / "SEMANTIC_REVIEW.json"
    challenge_path = directory / "semantic_challenge_deepseek_v3" / "CHALLENGE_REVIEW.json"
    independent_path = directory / "semantic_review_glm_v4" / "SEMANTIC_REVIEW.json"
    if not all(path.exists() for path in (first_path, challenge_path, independent_path)):
        return
    first, challenge, independent = (read(first_path), read(challenge_path),
                                     read(independent_path))
    for path in (first_path, challenge_path, independent_path):
        inputs[str(path)] = sha(path)
    require(first.get("status") == "AUTOMATIC_SEMANTIC_REVIEW_COMPLETED"
            and independent.get("status") == "DIFFERENT_CHECKPOINT_SEMANTIC_REVIEW_COMPLETED"
            and challenge.get("status") == "ADVERSARIAL_SEMANTIC_CHALLENGE_COMPLETED"
            and all(review.get("no_codex_content_review") is True
                    for review in (first, challenge, independent))
            and challenge.get("first_pass_review_sha256") == sha(first_path)
            and first.get("reviewer_model_requested")
            == challenge.get("reviewer_model_requested") == "deepseek-ai/DeepSeek-V3.2"
            and independent.get("reviewer_model_requested") == "zai-org/GLM-5.2",
            "ordering_semantic_artifact_invalid")
    roots = {"cc-train-episode-" + root for root in ROOTS}
    first_roots = {item["root_id"]: item for item in first["roots"]}
    challenge_roots = {item["root_id"]: item for item in challenge["roots"]}
    independent_roots = {item["root_id"]: item for item in independent["roots"]}
    require(set(first_roots) == set(independent_roots) == roots
            and set(challenge_roots) == {root for root, item in first_roots.items()
                                         if any(check["semantic_pass"] for check in item["checks"])},
            "ordering_semantic_roots_invalid")
    rebound_path = directory / "LABEL_REBOUND_UNREVIEWED.json"
    for root, item in first_roots.items():
        challenge_item = challenge_roots.get(root)
        independent_item = independent_roots[root]
        expected = {qid for qid, row in rows.items() if row["root_id"] == root
                    and row["question_type"] == "ordering" and row["mechanical_pass"]}
        checks = item["checks"]
        challenge_checks = {row["candidate_id"]: row for row in challenge_item["checks"]} if challenge_item else {}
        independent_checks = {row["candidate_id"]: row for row in independent_item["checks"]}
        require({row["candidate_id"] for row in checks} == set(independent_checks) == expected
                and len(checks) == len(expected)
                and len(independent_item["checks"]) == len(expected)
                and set(challenge_checks) == {row["candidate_id"] for row in checks
                                              if row["semantic_pass"] is True},
                "ordering_semantic_candidate_set_invalid:" + root)
        require(item["candidate_file_sha256"] == sha(rebound_path),
                "ordering_semantic_candidate_file_changed:" + root)
        source_path = directory.parent / "h4_crossroot_ordering_batch_v5_fresh_roots" / root / "source.json"
        reviews = [(item, "semantic_review_deepseek_v2", "deepseek-ai/DeepSeek-V3.2"),
                   (independent_item, "semantic_review_glm_v4", "zai-org/GLM-5.2")]
        if challenge_item:
            reviews.append((challenge_item, "semantic_challenge_deepseek_v3",
                            "deepseek-ai/DeepSeek-V3.2"))
        if any(group.get("schema_or_binding_errors")
               or group.get("reviewer_model_returned") != model
               or group.get("source_sha256") != sha(source_path)
               or group.get("response_file_sha256")
               != sha(directory / subdir / root / "RESPONSE.json")
               for group, subdir, model in reviews):
            continue
        for check in checks:
            qid = check["candidate_id"]
            independent_check = independent_checks[qid]
            challenge_check = challenge_checks.get(qid)
            if (check.get("review") is None or type(check.get("semantic_pass")) is not bool
                    or independent_check.get("review") is None
                    or type(independent_check.get("semantic_pass")) is not bool
                    or check["semantic_pass"] is True and
                    (challenge_check is None or challenge_check.get("review") is None
                     or type(challenge_check.get("challenge_pass")) is not bool)):
                continue
            passed = (check["semantic_pass"] and independent_check["semantic_pass"]
                      and challenge_check["challenge_pass"])
            rows[qid]["semantic"] = "PASS" if passed else "REJECT"
            if not passed:
                rows[qid]["solver"] = "SKIPPED_SEMANTIC"
                rows[qid]["dedup"] = "SKIPPED_SEMANTIC"


def original_content_matches(kind: str, qid: str, question: dict, root: str) -> bool:
    if kind == "multi_select":
        draft = read(MULTI_RAW / root / "batch-01" / "MODEL_DRAFT.json")
        originals = [row for row in draft["questions"] if row["candidate_id"] == qid]
        if len(originals) != 1:
            return False
        original = originals[0]
        if (question.get("question") != original["question"]
                or question.get("topic") != original["topic"]):
            return False
        actual = sorted((value, question["option_evidence"][label]["turn_ref"],
                         question["option_evidence"][label]["quote"],
                         question["option_evidence"][label]["relation"])
                        for label, value in question["options"].items())
        expected = sorted((option["proposition"], option["turn_ref"], option["quote"],
                           option["relation"]) for option in original["options"])
        expected_gold = {label for label, evidence in question["option_evidence"].items()
                         if evidence["relation"] == "supports"}
        return (actual == expected and len(question["options"]) == 6
                and set(question["option_evidence"]) == set(question["options"])
                and set(question["gold"]) == expected_gold
                and len(question["gold"]) == len(expected_gold))
    rebound = read(ORDER_REVIEW / "LABEL_REBOUND_UNREVIEWED.json")
    originals = [row for row in rebound if row["candidate_id"] == qid]
    if len(originals) != 1:
        return False
    original = originals[0]
    if any(question.get(key) != original[key] for key in
           ("question", "topic", "items", "gold")):
        return False
    return all(question["item_evidence"][label].get(key) == value.get(key)
               for label, value in original["item_evidence"].items()
               for key in ("turn_ref", "quote", "speaker"))


def solver_stage(rows: dict[str, dict], inputs: dict[str, str], kind: str,
                 path: Path | None) -> None:
    passed = {qid for qid, row in rows.items() if row["question_type"] == kind
              and row["semantic"] == "PASS"}
    if not passed or path is None or not path.exists():
        return
    report = read(path)
    inputs[str(path)] = sha(path)
    require(report.get("evaluation_data_status") == "VALID", "solver_report_invalid:" + kind)
    provenance = json.dumps(report.get("generator_provenance", {}), ensure_ascii=False).casefold()
    require("codex" not in provenance and "local_content_repair" not in provenance,
            "codex_repaired_solver_report_forbidden:" + kind)
    results = report.get("questions", {})
    require(set(results) == passed, "solver_question_set_not_auto_semantic_pass:" + kind)
    for qid, value in results.items():
        question = value.get("canonical_question", {})
        require(question.get("candidate_id") == qid,
                "solver_candidate_binding_invalid:" + qid)
        require(value.get("root_id") == rows[qid]["root_id"],
                "solver_root_binding_invalid:" + qid)
        require(original_content_matches(kind, qid, question, rows[qid]["root_id"]),
                "solver_question_differs_from_original:" + qid)
        if value.get("data_errors") or type(value.get("selected_by_policy")) is not bool:
            continue
        rows[qid]["solver"] = "SELECTED" if value["selected_by_policy"] else "FILTERED"
        if not value["selected_by_policy"]:
            rows[qid]["dedup"] = "SKIPPED_SOLVER"


def dedup_stage(rows: dict[str, dict], inputs: dict[str, str], path: Path | None,
                solver_paths: dict[str, Path | None]) -> None:
    selected = {qid for qid, row in rows.items() if row["solver"] == "SELECTED"}
    if not selected or path is None or not path.exists():
        return
    decision = read(path)
    inputs[str(path)] = sha(path)
    require(decision.get("version") == "no_codex_auto_dedup_v1", "dedup_version_invalid")
    for kind, solver_path in solver_paths.items():
        if any(row["question_type"] == kind and row["solver"] == "SELECTED"
               for row in rows.values()):
            require(solver_path is not None and
                    decision.get("solver_result_sha256", {}).get(kind) == sha(solver_path),
                    "dedup_solver_binding_invalid:" + kind)
    retained = set(decision.get("retained_ids", []))
    held = set(decision.get("held_ids", []))
    require(retained | held == selected and not retained & held,
            "dedup_decisions_not_complete")
    for qid in retained:
        rows[qid]["dedup"] = "RETAINED"
    for qid in held:
        rows[qid]["dedup"] = "HELD"


def summarize(rows: dict[str, dict], inputs: dict[str, str]) -> dict:
    root_rows = []
    for suffix in ROOTS:
        root = "cc-train-episode-" + suffix
        by_type = {}
        for kind in TYPES:
            subset = [row for row in rows.values()
                      if row["root_id"] == root and row["question_type"] == kind]
            count = lambda key, value: sum(row[key] == value for row in subset)
            unresolved = any(row["mechanical_pass"] and row["semantic"] == "PENDING"
                             or row["semantic"] == "PASS" and row["solver"] == "PENDING"
                             or row["solver"] == "SELECTED" and row["dedup"] == "PENDING"
                             for row in subset)
            by_type[kind] = {
                "generated": len(subset),
                "raw_mechanical_pass": count("raw_mechanical_pass", True),
                "mechanical_pass": count("mechanical_pass", True),
                "semantic_pass": count("semantic", "PASS"),
                "semantic_reject": count("semantic", "REJECT"),
                "semantic_pending": count("semantic", "PENDING"),
                "solver_selected": count("solver", "SELECTED"),
                "solver_filtered": count("solver", "FILTERED"),
                "solver_pending": sum(row["semantic"] == "PASS" and row["solver"] == "PENDING"
                                      for row in subset),
                "dedup_retained": count("dedup", "RETAINED"),
                "dedup_held": count("dedup", "HELD"),
                "dedup_pending": sum(row["solver"] == "SELECTED" and row["dedup"] == "PENDING"
                                     for row in subset),
                "gap_to_four_confirmed": max(0, 4 - count("dedup", "RETAINED"))
                if not unresolved else None,
            }
        root_rows.append({"root_id": root, "types": by_type})
    return {"version": HERE.name, "status": "COMPLETE" if all(
                not any(row["types"][kind][key] for key in
                        ("semantic_pending", "solver_pending", "dedup_pending"))
                for row in root_rows for kind in TYPES) else "INCOMPLETE_FAIL_CLOSED",
            "generator_model": "Qwen/Qwen3.6-35B-A3B",
            "definition": "Original model content only; ordering permits deterministic label-only rebind.",
            "target_confirmed_unique_per_root_per_type": [4, 6],
            "inputs_sha256": inputs, "roots": root_rows,
            "candidates": list(rows.values()),
            "training_started": False, "training_exported": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-review", type=Path, default=MULTI_REVIEW)
    parser.add_argument("--ordering-review", type=Path, default=ORDER_REVIEW)
    parser.add_argument("--multi-solver-results", type=Path)
    parser.add_argument("--ordering-solver-results", type=Path)
    parser.add_argument("--dedup-decision", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "REPORT.json")
    args = parser.parse_args()
    rows, inputs = raw_inputs()
    multi_semantic(rows, inputs, args.multi_review)
    ordering_semantic(rows, inputs, args.ordering_review)
    solver_paths = {"multi_select": args.multi_solver_results,
                    "ordering": args.ordering_solver_results}
    for kind, path in solver_paths.items():
        solver_stage(rows, inputs, kind, path)
    dedup_stage(rows, inputs, args.dedup_decision, solver_paths)
    result = summarize(rows, inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "roots": result["roots"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
