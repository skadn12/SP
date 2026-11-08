"""Offline Codex-review copies; never mutate live or send model requests."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "live_review"
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-formal-feedback-routing-v1")
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from scripts.current_method.restart_n5_spell.h4_r02_ordering_temporal_contract_v1.contract import register_runtime_context, verify_candidate

POLICY = {"version": "ordering_grounded_options_v2", "node_count": 4}
ROOTS = ["cc-train-episode-259192", "cc-train-episode-96462", "cc-train-episode-112812"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def claim_signature(raw):
    """Only references, exact evidence, and date metadata may change."""
    raw = copy.deepcopy(raw)
    assert raw.get("gold") is None
    raw.pop("gold", None)
    raw.pop("evidence_groups", None)
    for entity in raw["event_nodes"] + raw["temporal_relations"]:
        for key in ("evidence_refs", "evidence_spans", "date_anchors"):
            entity.pop(key, None)
    return raw


def reviews():
    result = {}
    definitions = [
        ("fresh", "ordering_259192/review.json", "rows"),
        ("fresh", "ordering_96462/AUDIT.json", "candidate_reviews"),
        ("fresh", "ordering_112812/AUDIT.json", "rows"),
        ("evolution", "evolution_diagnosis/AUDIT.json", "ordering_reviews"),
    ]
    for branch, name, field in definitions:
        path = OUT / name
        for row in read(path)[field]:
            verdict = row.get("content_verdict", row.get("content_decision"))
            result[(branch, row["candidate_id"])] = (row, verdict, path)
    return result


def main():
    review_map = reviews()
    live_files = [p for branch in ("fresh", "evolution")
                  for p in (BASE / f"{branch}_prepared/live").rglob("*") if p.is_file()]
    live_before = {str(p): sha(p) for p in live_files}
    checks, choices, supported, held, summary = [], [], [], [], []
    for branch in ("fresh", "evolution"):
        for root in ROOTS:
            live = BASE / f"{branch}_prepared/live/roots" / root
            if not live.exists():
                continue
            source = read(live / "qa_source.json")
            original_pool = read(live / "routed_candidates.json")
            branch_choices, branch_ordering, branch_held = [], [], []
            for original in original_pool:
                cid = original["candidate_id"]
                key = f"{branch}/{cid}"
                if original["question_type"] != "ordering":
                    entry = {"key": key, "branch": branch, "root_id": root,
                             "policy": "PROGRAM_ONLY_NO_CONTENT_OR_EVIDENCE_GATE",
                             "semantic_correctness": "NOT_REVIEWED", "candidate": original}
                    branch_choices.append(entry)
                    continue
                row, verdict, audit_path = review_map[(branch, cid)]
                if not verdict.startswith("PASS"):
                    branch_held.append({"key": key, "branch": branch, "root_id": root,
                        "original_candidate": original, "content_decision": verdict,
                        "audit_path": str(audit_path), "release": False,
                        "reason": "UNIQUE_FULL_SOURCE_SUPPORTED_GOLD_NOT_ESTABLISHED",
                        "corrected_gold": None})
                    continue
                meta = original["ordering_derivation"]
                old_record_path = live / meta["path"]
                assert sha(old_record_path) == meta["sha256"]
                old_record = read(old_record_path)
                raw_original = old_record["raw_row"]
                raw_corrected = row["corrected_raw_candidate"]
                assert claim_signature(raw_corrected) == claim_signature(raw_original), cid
                old_identity = old_record["identity"]
                request_id = old_identity["request_id"]
                raw_response = live / "raw/responses" / f"{request_id}.json.txt"
                assert sha(raw_response) == old_identity["raw_sha256"]
                batch = read(raw_response)
                index = old_identity["raw_row_index"] - 1
                assert batch["candidates"][index] == raw_original
                batch["candidates"][index] = raw_corrected
                review_dir = OUT / "corrected" / branch / root / request_id
                corrected_response = review_dir / "raw/responses" / f"{request_id}.json.txt"
                write(corrected_response, batch)
                identity = {**old_identity, "run_id": "codex_offline_review_copies_v1",
                            "arm_id": branch.upper() + "_CODEX_EVIDENCE_COPY",
                            "raw_sha256": sha(corrected_response)}
                registry = {k: identity[k] for k in ("run_id", "arm_id", "root_id")}
                registry["requests"] = {request_id: {k: identity[k] for k in
                    ("intent_id", "source", "stage", "raw_sha256")}}
                register_runtime_context(review_dir, POLICY, registry)
                normalized = normalize_batch(batch, intent_id=original["intent_id"],
                    source=original["generation_source"], contract={}, expanded=source,
                    ordering_policy=POLICY,
                    ordering_context={**identity, "root_dir": review_dir},
                    choice_evidence_policy="optional_diagnostic_v1")
                candidate = normalized["candidates"][index]
                candidate["candidate_id"] = identity["candidate_id"]
                candidate["root_id"] = root
                assert candidate["schema_valid"], (cid, candidate["schema_errors"])
                for field in ("question", "items", "gold"):
                    assert candidate[field] == original[field], (cid, field)
                result = verify_candidate(candidate, expanded=source, policy=POLICY,
                                          context={**identity, "root_dir": review_dir})
                assert result["status"] == "PASS", (cid, result)
                write(review_dir / "reviewed_candidate.json", candidate)
                write(review_dir / "PROVENANCE.json", {
                    "artifact_kind": "CODEX_OFFLINE_REVIEW_COPY_NOT_MODEL_RESPONSE",
                    "source_candidate_id": cid, "source_branch": branch,
                    "export_key": key,
                    "original_routed_candidate_path": str(live / "routed_candidates.json"),
                    "original_routed_candidate_sha256": sha(live / "routed_candidates.json"),
                    "question_lineage": original.get("question_lineage"),
                    "source_raw_response": str(raw_response),
                    "source_raw_sha256": old_identity["raw_sha256"],
                    "source_derivation": str(old_record_path),
                    "source_qa_source": str(live / "qa_source.json"),
                    "source_qa_source_sha256": sha(live / "qa_source.json"),
                    "audit_path": str(audit_path), "audit_sha256": sha(audit_path),
                    "raw_original": raw_original, "raw_corrected": raw_corrected,
                    "claims_unchanged": True, "gold_unchanged": True,
                    "correction": "NONE" if raw_corrected == raw_original else "EVIDENCE_METADATA_ONLY",
                    "batch_note": "Whole batch copied only to preserve original row index; only reviewed_candidate is released. Other copied rows are not promoted.",
                    "external_calls": 0, "independent_human_certification": False,
                    "program_verification": result,
                })
                entry = {"key": key, "branch": branch, "root_id": root,
                         "source_candidate_id": cid,
                         "question_lineage": original.get("question_lineage"),
                         "original_routed_candidate_path": str(live / "routed_candidates.json"),
                         "original_routed_candidate_sha256": sha(live / "routed_candidates.json"),
                         "identity_scope": "Use key plus candidate_context_root; native candidate_id is local to its derivation context and may match a historical ID. Do not detach candidate from this wrapper.",
                         "content_decision": "PASS_CODEX_OFFLINE_DIAGNOSIS",
                         "original_binding_decision": row.get("original_binding_decision", row.get("submitted_binding_status", row.get("original_binding_assessment", "SEE_AUDIT"))),
                         "correction": "NONE" if raw_corrected == raw_original else "EVIDENCE_METADATA_ONLY",
                         "candidate": candidate,
                         "candidate_context_root": str(review_dir),
                         "qa_source_path": str(live / "qa_source.json"),
                         "audit_path": str(audit_path),
                         "independent_human_certification": False,
                         "difficulty": "UNKNOWN_NOT_EVALUATED",
                         "semantic_newness": "OLD_CHAIN_REFORMULATION" if root.endswith("-96462") else "SUPPORTED_EXTENSION_OR_DIFFERENT_EVENT_COMBINATION"}
                branch_ordering.append(entry)
                checks.append({"key": key, "native_candidate_schema": "PASS",
                    "native_quote_reference_and_dag_rederivation": "PASS",
                    "native_persisted_candidate_verification": "PASS",
                    "question_items_gold_unchanged": True,
                    "actor_state_direction_unchanged": True,
                    "program_semantic_support_status": candidate["semantic_support_status"]})
            choices.extend(branch_choices)
            supported.extend(branch_ordering)
            held.extend(branch_held)
            export = OUT / "exports" / branch / root
            write(export / "choices_program_accepted_unchanged.json", [x["candidate"] for x in branch_choices])
            write(export / "ordering_supported_review_copies.json", branch_ordering)
            write(export / "ordering_undetermined_hold.json", branch_held)
            summary.append({"branch": branch, "root_id": root,
                "routed_program_counts": dict(Counter(x["question_type"] for x in original_pool)),
                "choices_program_only_retained": len(branch_choices),
                "ordering_full_source_supported": len(branch_ordering),
                "ordering_undetermined": len(branch_held)})

    duplicate_pairs = [
        ("fresh/cc-train-episode-96462:I003-C-Q04:primary", "fresh/cc-train-episode-96462:I001-C-Q04:repair",
         "Same four specific event instances, same task and chronology; wording differs."),
        ("evolution/cc-train-episode-112812:I001-C-Q03:primary", "fresh/cc-train-episode-112812:I003-C-Q04:repair",
         "Same four office-progress stages; drafting vs drafting/sending gives no additional necessary relation."),
    ]
    all_keys = {x["key"] for x in supported}
    assert all(a in all_keys and b in all_keys for a,b,_ in duplicate_pairs)
    excluded = {a for a,_,_ in duplicate_pairs}
    distinct = [x for x in supported if x["key"] not in excluded]
    assert len(choices) == 25 and len(supported) == 7 and len(held) == 3 and len(distinct) == 5
    assert all(sha(p) == expected for p,expected in live_before.items())
    for root in ROOTS:
        root_choices = [x for x in choices if x["root_id"] == root]
        root_ordering = [x for x in distinct if x["root_id"] == root]
        write(OUT / "exports/combined" / root / "candidate_pool.json", {
            "artifact_kind": "REVIEW_EXPORT_NOT_TRAINING_PROMOTION",
            "root_id": root, "choice_policy": "PROGRAM_ONLY_NO_SEMANTIC_GATE",
            "ordering_policy": "CODEX_OFFLINE_SOURCE_REVIEW_WITH_NATIVE_PROGRAM_REVALIDATION",
            "choices": root_choices, "ordering": root_ordering,
            "counts": {"single_choice": sum(x["candidate"]["question_type"] == "single_choice" for x in root_choices),
                       "multi_select": sum(x["candidate"]["question_type"] == "multi_select" for x in root_choices),
                       "ordering": len(root_ordering)},
            "difficulty": "UNKNOWN_NOT_EVALUATED", "automatically_promoted": False,
        })
    write(OUT / "exports/ALL_CANDIDATES.json", {"choices_program_only": choices,
        "ordering_supported_distinct": distinct, "ordering_supported_before_same_batch_dedup": supported,
        "ordering_held": held, "combined_count": len(choices) + len(distinct),
        "combined_count_is_not_verified_new_or_hard_count": True})
    write(OUT / "ORDERING_DUPLICATES.json", {"scope": "CURRENT_BATCH_ONLY",
        "policy": "Same complete event task, not same topic or partial overlap",
        "groups": [{"excluded_from_distinct_export": a, "representative": b, "reason": why}
                   for a,b,why in duplicate_pairs],
        "supported_rows": len(supported), "distinct_supported_tasks": len(distinct),
        "old_formal_chain_reformulations": 1,
        "distinct_supported_extensions_or_combinations": 4,
        "difficulty": "UNKNOWN_NOT_EVALUATED", "choice_filtering_applied": False})
    write(OUT / "EXPORT_VERIFICATION.json", {
        "status": "PASS", "mode": "OFFLINE_ONLY", "external_calls": 0,
        "checked_original_live_files": len(live_before), "original_live_files_unchanged": True,
        "checks": checks, "per_branch_root": summary,
        "choices_copied_verbatim": len(choices), "supported_orderings_revalidated": len(supported),
        "distinct_supported_orderings": len(distinct), "ordering_held": len(held),
        "gold_changes": 0, "item_actor_state_direction_changes": 0,
        "live_source_manifest": [{"path": p, "sha256": digest} for p,digest in live_before.items()],
        "unaccepted_diagnostic_recoveries_promoted": 0, "solver_started": False,
        "independent_human_certification": False})
    print(json.dumps({"status": "PASS", "choices": len(choices), "supported_orderings": len(supported),
                      "distinct_orderings": len(distinct), "hold": len(held), "gold_changes": 0,
                      "original_live_files_unchanged": len(live_before)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
