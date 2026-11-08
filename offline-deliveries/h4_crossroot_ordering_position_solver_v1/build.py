"""Bind prior independent content approvals to the new ordering display labels."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
BATCH = DELIVERIES / "h4_crossroot_ordering_batch_v5_fresh_roots"
PILOT = DELIVERIES / "h4_crossroot_ordering_no_codex_pilot_v1"
PRIOR = DELIVERIES / "h4_crossroot_ordering_no_codex_solver_v1"
BINDING = DELIVERIES / "h4_option_position_binding_v1"
SOURCE_POOL = BINDING / "v5_ordering_rebound" / "POOL.json"
SOURCE_REPORT = BINDING / "v5_ordering_rebound" / "REPORT.json"
LABELS = tuple(f"I{i}" for i in range(1, 7))
REVIEW_FILES = {
    "deepseek_first": PILOT / "semantic_review_deepseek_v2" / "SEMANTIC_REVIEW.json",
    "deepseek_challenge": PILOT / "semantic_challenge_deepseek_v3" / "CHALLENGE_REVIEW.json",
    "glm_independent": PILOT / "semantic_review_glm_v4" / "SEMANTIC_REVIEW.json",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_module("prior_auto_review_build", PRIOR / "build.py")
binder = load_module("position_binding_ordering", BINDING / "ordering.py")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def write_once(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def content_label_map(old: dict, new: dict) -> dict[str, str]:
    require(old["candidate_id"] == new["candidate_id"], "candidate_id_changed")
    fixed = set(old) - {"items", "item_evidence", "gold", "order_reason"}
    require(set(old) == set(new) and all(old[key] == new[key] for key in fixed),
            "non_label_question_content_changed:" + old["candidate_id"])
    old_pairs = {label: binder.canonical([old["items"][label], old["item_evidence"][label]])
                 for label in LABELS}
    new_pairs = {label: binder.canonical([new["items"][label], new["item_evidence"][label]])
                 for label in LABELS}
    require(len(set(old_pairs.values())) == len(set(new_pairs.values())) == 6
            and set(old_pairs.values()) == set(new_pairs.values()),
            "item_or_evidence_content_changed:" + old["candidate_id"])
    return {old_label: next(new_label for new_label in LABELS if pair == new_pairs[new_label])
            for old_label, pair in old_pairs.items()}


def reviewed_items(qid: str, root: str, mapping: dict[str, str]) -> dict:
    checks = {}
    for name, path in REVIEW_FILES.items():
        report = read(path)
        group = next((group for group in report["roots"] if group["root_id"] == root), None)
        require(group is not None and group["source_sha256"] == sha(BATCH / root / "source.json")
                and not group["schema_or_binding_errors"], "review_group_invalid:" + name)
        check = next((row for row in group["checks"] if row["candidate_id"] == qid), None)
        require(check is not None and not check["schema_or_binding_errors"]
                and set(check["review"]["items"]) == set(LABELS),
                "review_item_set_invalid:" + name)
        verdict_field = "challenge_pass" if name == "deepseek_challenge" else "semantic_pass"
        require(check[verdict_field] is True, "review_not_approved:" + name)
        checks[name] = {
            "model": group["reviewer_model_returned"],
            "source_review_sha256": sha(path), "pass": True,
            "items_by_new_label": {mapping[old]: value for old, value in check["review"]["items"].items()},
        }
    return checks


def make_pool() -> tuple[dict, dict]:
    approved_pool, prior_audit = prior.make_pool()
    require(read(PRIOR / "READY_FOR_EVALUATION.json") == approved_pool
            and read(PRIOR / "AUTOMATIC_REVIEW_AUDIT.json") == prior_audit,
            "prior_approved_pool_changed")
    expected_pool, expected_report = binder.build_result(PILOT / "LABEL_REBOUND_UNREVIEWED.json", BATCH)
    rebound_pool, rebound_report = read(SOURCE_POOL), read(SOURCE_REPORT)
    require((rebound_pool, rebound_report) == (expected_pool, expected_report)
            and rebound_report["input_count"] == 10
            and rebound_report["bound_count"] == 10
            and rebound_report["rejected_count"] == 0,
            "position_binding_replay_or_coverage_invalid")
    approved = approved_pool["questions"]
    rebound = {row["candidate_id"]: row for row in rebound_pool["questions"]}
    require(len(rebound) == len(rebound_pool["questions"]), "duplicate_rebound_id")
    source_audits = {row["candidate_id"]: row for row in rebound_report["audits"]}
    prior_audits = {row["derived_candidate_id"]: row for row in prior_audit["candidates"]}
    questions, audits = [], []
    for old in approved:
        qid, root = old["candidate_id"], old["root_id"]
        require(qid in rebound and qid in source_audits and qid in prior_audits,
                "approved_candidate_missing_from_rebound:" + qid)
        new = rebound[qid]
        mapping = content_label_map(old, new)
        source_path = BATCH / root / "source.json"
        source = read(source_path)
        require(new["source_path"] == str(source_path.resolve())
                and new["source_sha256"] == sha(source_path)
                and new["source_sha256"] == prior_audits[qid]["source_sha256"]
                and binder.source_order(new, source) == new["gold"]
                and [mapping[label] for label in old["gold"]] == new["gold"]
                and source_audits[qid]["old_item_evidence_multiset_sha256"]
                    == source_audits[qid]["new_item_evidence_multiset_sha256"]
                and source_audits[qid]["new_question_sha256"] == binder.sha(binder.canonical(new)),
                "source_or_gold_mapping_invalid:" + qid)
        checks = reviewed_items(qid, root, mapping)
        questions.append(new)
        protected = {**prior_audits[qid]["original_files_sha256"]}
        paths = (SOURCE_POOL, SOURCE_REPORT, PRIOR / "READY_FOR_EVALUATION.json",
                 PRIOR / "AUTOMATIC_REVIEW_AUDIT.json", BINDING / "ordering.py",
                 HERE / "build.py", HERE / "run_solver.py", HERE / "report_results.py")
        protected.update({str(path): sha(path) for path in paths})
        audits.append({
            "derived_candidate_id": qid, "root_id": root, "source_sha256": sha(source_path),
            "source_turn_count": len(source["turns"]),
            "original_model_provenance": {"returned_model": "Qwen/Qwen3.6-35B-A3B"},
            "original_files_sha256": protected,
            "old_to_new_label_by_content": mapping,
            "prior_review_items_rebound": checks,
        })
    require(questions, "no_content_approved_rebound_candidates")
    provenance = {**approved_pool["provenance"],
                  "normalization": "source_bound_position_binding_v1",
                  "prior_review_binding": "exact_item_and_evidence_content"}
    pool = {"status": "AUTOMATIC_REVIEWED_POSITION_BOUND_READY_FOR_SOLVER",
            "candidate_count": len(questions), "provenance": provenance,
            "questions": questions, "training_started": False}
    audit = {"status": "PRIOR_INDEPENDENT_REVIEW_REBOUND_BY_CONTENT",
             "candidate_count": len(audits), "candidates": audits,
             "position_pool_sha256": sha(SOURCE_POOL),
             "position_report_sha256": sha(SOURCE_REPORT),
             "prior_reviewed_pool_sha256": sha(PRIOR / "READY_FOR_EVALUATION.json")}
    return pool, audit


if __name__ == "__main__":
    pool, audit = make_pool()
    write_once(HERE / "READY_FOR_EVALUATION.json", pool)
    write_once(HERE / "REVIEW_REBIND_AUDIT.json", audit)
    print(json.dumps({"status": pool["status"], "candidate_count": len(pool["questions"]),
                      "candidate_ids": [row["candidate_id"] for row in pool["questions"]]}))
