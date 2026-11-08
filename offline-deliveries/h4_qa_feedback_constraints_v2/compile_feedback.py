"""Compile imperative QA feedback from frozen local artifacts; make no requests.

Run once with the existing worktree Python environment. Generated artifacts are
never overwritten. This compiler deliberately does not prepare or execute a run.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys


BASE = Path("/root/autodl-tmp/gxy-selfplay")
WORKTREE = BASE / "worktrees/h4-gate-downstream-v1"
DELIVERY = Path(__file__).resolve().parent
PREVIOUS = BASE / "offline-deliveries/h4_qa_difficulty_feedback_v1"
ORIGINAL = BASE / "offline-deliveries/h4_gate_downstream_v1"
sys.path.insert(0, str(WORKTREE))

from scripts.current_method.restart_n5_spell.core import (  # noqa: E402
    atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.qa_difficulty_feedback_run import (  # noqa: E402
    ROOTS, _compiled_feedback_files, validate_context,
)


VERSION = "qa_feedback_constraints_v2"
EXPECTED_LATEST = (8, 10, 11)
EXPECTED_PRIOR = (3, 7, 3)
EXPECTED_FOLDED = (10, 14, 11)
INDEX_OPEN = "[HISTORICAL_QUESTION_AVOIDANCE_INDEX]"
INDEX_CLOSE = "[/HISTORICAL_QUESTION_AVOIDANCE_INDEX]"
INTENT_GUIDANCE = (
    "Use the complete supplied transcript as the sole factual source. Select "
    "exactly three distinct transcript-grounded reasoning intents whose required "
    "judgments do not restate the historical question index below or each other. "
    "Older facts may support a genuinely different supported comparison or "
    "relation; do not force a particular round or topic. Historical questions "
    "and options are not evidence or required topics. Verify every proposed "
    "relation in the transcript, and do not invent temporal chains when the "
    "evidence cannot establish an order. Keep the existing intent schema and "
    "output contract."
)
INDEX_PURPOSE = (
    "Avoid reproducing these previously generated judgments, including by "
    "paraphrase, relabeling choices, or changing question type. Older events "
    "remain available for genuinely different supported judgments."
)
CLAIMS_STATUS = (
    "Historical question wording and options may contain unsupported claims. "
    "Do not treat them as transcript evidence, an answer key, or required topics. "
    "Verify all new claims against the complete supplied transcript."
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def question_fields(row):
    value = {key: deepcopy(row.get(key, {})) for key in
             ("question_type", "question", "options", "items")}
    require(value["question_type"] in {"single_choice", "multi_select", "ordering"},
            "unexpected_question_type")
    require(isinstance(value["question"], str) and bool(value["question"]),
            "missing_complete_question")
    for key in ("options", "items"):
        require(isinstance(value[key], dict)
                and all(isinstance(k, str) and isinstance(v, str)
                        for k, v in value[key].items()), "invalid_choice_mapping")
    return value


def fold_key(row):
    # Choice labels/order and historical answers do not define the judgment.
    # Retain exact strings, their multiplicity, and options/items separation.
    value = question_fields(row)
    return canonical_json({
        "question_type": value["question_type"], "question": value["question"],
        "option_content": sorted(value["options"].values()),
        "item_content": sorted(value["items"].values()),
    })


def main():
    feedback_root = DELIVERY / "feedback"
    manifest_path = DELIVERY / "COMPILED_FEEDBACK.json"
    require(not feedback_root.exists() and not manifest_path.exists(),
            "generated_output_exists_no_overwrite")
    rules_path = DELIVERY / "writer_constraints.txt"
    rules = rules_path.read_text(encoding="utf-8")
    require(bool(rules.strip()), "writer_constraints_empty")
    review_path = PREVIOUS / "reports/CONTENT_ALL_ROWS.json"
    reviews = read(review_path)
    require(isinstance(reviews, list) and len(reviews) == 40, "raw_review_scope_changed")
    require({row["root_id"] for row in reviews} == set(ROOTS), "review_roots_changed")
    require(sum(row["in_final_pool"] is True for row in reviews) == 29,
            "latest_final_count_changed")

    # Every output, its digest, and the complete manifest are constructed before
    # any output directory/file is created. JSON formatting matches atomic_json.
    pending = {}
    source_artifacts = {}

    def source(path):
        result = artifact(path)
        source_artifacts[result["path"]] = result
        return result

    def output(path, value, *, verbatim_text=False):
        require(path not in pending and not path.exists(), "output_exists_or_repeated")
        serialized = value if verbatim_text else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        pending[path] = (value, verbatim_text, serialized)
        return {"path": str(path), "sha256": sha256_bytes(serialized.encode("utf-8"))}

    records = []
    for root, latest_count, prior_count, folded_count in zip(
            ROOTS, EXPECTED_LATEST, EXPECTED_PRIOR, EXPECTED_FOLDED):
        root_out = feedback_root / root
        original_source_path = ORIGINAL / "prepared_run/qa/roots" / root / "qa_source.json"
        live_source_path = PREVIOUS / "prepared_run/live/roots" / root / "qa_source.json"
        frozen_source_path = PREVIOUS / "prepared_run/frozen" / root / "qa_source.json"
        original_source = read(original_source_path)
        require(original_source == read(live_source_path) == read(frozen_source_path),
                "full_transcript_date_or_source_identity_changed:" + root)
        require(original_source["root_id"] == root, "source_root_mismatch")

        latest_pool_path = PREVIOUS / "prepared_run/live/roots" / root / "candidate_pool.json"
        old_pool_path = ORIGINAL / "prepared_run/qa/roots" / root / "candidate_pool.json"
        prior_projection_path = PREVIOUS / "feedback" / root / "prompt_projection.json"
        prior_difficulty_path = PREVIOUS / "feedback" / root / "difficulty_feedback.json"
        correction_path = ORIGINAL / "feedback" / root / "feedback_input.json"
        previous_correction_path = PREVIOUS / "feedback" / root / "fact_feedback_preserved.json"
        require(correction_path.read_bytes() == previous_correction_path.read_bytes(),
                "historical_correction_copy_changed:" + root)

        latest_pool = read(latest_pool_path)
        final_reviews = [row for row in reviews
                         if row["root_id"] == root and row["in_final_pool"] is True]
        latest_by_id = {row["candidate_id"]: row for row in latest_pool}
        reviews_by_id = {row["candidate_id"]: row for row in final_reviews}
        require(len(latest_pool) == len(latest_by_id) == latest_count
                and len(final_reviews) == len(reviews_by_id) == latest_count
                and set(latest_by_id) == set(reviews_by_id), "latest_final_identity_mismatch:" + root)
        require(all(reviews_by_id[cid]["final_candidate"] == candidate
                    for cid, candidate in latest_by_id.items()), "final_candidate_changed:" + root)

        previous_projection = read(prior_projection_path)
        prior_rows = previous_projection["measured_difficulty"]["eligible_questions"]
        require(previous_projection["root_id"] == root and len(prior_rows) == prior_count,
                "prior_eligible_scope_changed:" + root)
        old_pool = read(old_pool_path)
        old_by_id = {row["candidate_id"]: row for row in old_pool}
        require(len(old_pool) == len(old_by_id), "duplicate_old_pool_id")
        require(len({row["candidate_id"] for row in prior_rows}) == prior_count,
                "duplicate_prior_eligible_id")
        for row in prior_rows:
            require(row["candidate_id"] in old_by_id
                    and question_fields(row) == question_fields(old_by_id[row["candidate_id"]]),
                    "prior_eligible_question_changed:" + root)
            require(row.get("historical_gold") == old_by_id[row["candidate_id"]].get("gold"),
                    "prior_historical_gold_provenance_changed:" + root)

        entries_by_key = {}
        audit_occurrences = []
        for batch, rows in ((ORIGINAL.name, prior_rows), (PREVIOUS.name, latest_pool)):
            for row in rows:
                occurrence = {"generation_batch": batch, "candidate_id": row["candidate_id"]}
                key = fold_key(row)
                if key not in entries_by_key:
                    entries_by_key[key] = {**question_fields(row), "prior_occurrences": []}
                entries_by_key[key]["prior_occurrences"].append(occurrence)
                audit_occurrences.append({
                    **occurrence,
                    "historical_gold_status": "HISTORICAL_NOT_INDEPENDENTLY_VERIFIED",
                    "historical_gold": deepcopy(row.get("historical_gold", row.get("gold"))),
                    "source_row": deepcopy(row),
                    "source_pool_candidate": deepcopy(
                        old_by_id[row["candidate_id"]] if batch == ORIGINAL.name else row),
                    "latest_content_review": deepcopy(
                        reviews_by_id[row["candidate_id"]] if batch == PREVIOUS.name else None),
                })
        entries = list(entries_by_key.values())
        require(len(entries) == folded_count, "exact_fold_count_changed:" + root)
        require(sum(len(e["prior_occurrences"]) for e in entries) == latest_count + prior_count,
                "historical_occurrence_lost")
        require(all(set(e) == {"question_type", "question", "options", "items", "prior_occurrences"}
                    and all(set(o) == {"generation_batch", "candidate_id"}
                            for o in e["prior_occurrences"]) for e in entries),
                "audit_fields_leaked_into_prompt_entries")
        prompt_index = {"root_id": root, "purpose": INDEX_PURPOSE,
                        "claims_status": CLAIMS_STATUS, "entries": entries}
        rendered_index = INDEX_OPEN + "\n" + canonical_json(prompt_index) + "\n" + INDEX_CLOSE
        writer_suffix = rules + "\n\n" + rendered_index
        context = validate_context({"intent_suffix": INTENT_GUIDANCE + "\n\n" + rendered_index,
                                    "slot_suffixes": [writer_suffix] * 3})
        require(context["slot_suffixes"][0][:len(rules)] == rules,
                "writer_rules_not_verbatim")

        lengths = {
            "writer_constraints_chars": len(rules),
            "historical_index_chars": len(rendered_index),
            "intent_suffix_chars": len(context["intent_suffix"]),
            "each_slot_suffix_chars": len(writer_suffix),
            "context_json_bytes": len((json.dumps(context, ensure_ascii=False, indent=2) + "\n").encode()),
        }
        counts = {"latest_final_candidates": latest_count, "prior_eligible_questions": prior_count,
                  "historical_occurrences": latest_count + prior_count,
                  "exact_folded_index_entries": len(entries),
                  "folded_occurrences": latest_count + prior_count - len(entries),
                  "latest_final_by_type": dict(Counter(r["question_type"] for r in latest_pool)),
                  "index_entries_by_type": dict(Counter(e["question_type"] for e in entries))}
        projection = {
            "version": VERSION, "root_id": root,
            "prompt_index": prompt_index,
            "prompt_projection_policy": {
                "model_visible": "Imperative guidance plus prompt_index only; no audit fields.",
                "fold_key": "Exact question_type + question text + sorted option text multiset + sorted item text multiset; labels/order and Gold excluded.",
                "representative": "First chronological occurrence retains its complete original question/options/items; all original rows remain in audit.",
                "source_identity": "generation_batch + candidate_id",
                "new_solver_measurements": False,
            },
            "counts": counts, "feedback_lengths": lengths,
            "audit_only": {
                "model_visible": False,
                "status": "Historical source records and diagnoses preserved for audit; not new facts, answer certification, or newly measured difficulty.",
                "transcript_complete_sha256": original_source["complete_dialogue_sha256"],
                "source_occurrences": audit_occurrences,
            },
        }
        context_artifact = output(root_out / "qa_feedback_context.json", context)
        projection_artifact = output(root_out / "prompt_projection.json", projection)
        correction_artifact = output(root_out / "fact_feedback_preserved.json",
                                     correction_path.read_text(encoding="utf-8"), verbatim_text=True)
        difficulty_artifact = output(root_out / "difficulty_feedback_preserved.json",
                                     prior_difficulty_path.read_text(encoding="utf-8"), verbatim_text=True)
        require(correction_artifact["sha256"] == sha256_file(correction_path)
                and difficulty_artifact["sha256"] == sha256_file(prior_difficulty_path),
                "audit_copy_not_byte_identical")
        sources = {
            "compiler": source(Path(__file__)), "writer_constraints": source(rules_path),
            "content_report": source(review_path), "latest_candidate_pool": source(latest_pool_path),
            "prior_candidate_pool": source(old_pool_path),
            "prior_projection": source(prior_projection_path),
            "prior_difficulty_feedback": source(prior_difficulty_path),
            "existing_correction": source(correction_path),
            "previous_preserved_correction": source(previous_correction_path),
            "prior_compiled_manifest": source(PREVIOUS / "COMPILED_FEEDBACK.json"),
            "qa_source": source(original_source_path),
            "latest_qa_source": source(live_source_path),
            "latest_frozen_qa_source": source(frozen_source_path),
            "preserved_difficulty": difficulty_artifact,
        }
        records.append({
            "root_id": root, "sources": sources,
            "context": context_artifact, "projection": projection_artifact,
            "preserved_correction": correction_artifact,
            "context_object_sha256": sha256_bytes(canonical_json(context).encode()),
            "transcript_complete_sha256": original_source["complete_dialogue_sha256"],
            "full_qa_source_object_identity_verified": True,
            "same_root_guidance_for_all_three_slots": True,
            "counts": counts, "feedback_lengths": lengths,
        })

    manifest = {
        "version": VERSION, "status": "OFFLINE_COMPILED_NOT_SENT",
        "root_order": list(ROOTS), "external_calls": 0,
        "planner_automatically_activated": False,
        "new_solver_measurements": False,
        "total_latest_final_candidates": sum(EXPECTED_LATEST),
        "total_prior_eligible_questions": sum(EXPECTED_PRIOR),
        "total_historical_occurrences": sum(EXPECTED_LATEST) + sum(EXPECTED_PRIOR),
        "total_exact_folded_index_entries": sum(EXPECTED_FOLDED),
        "records": records,
    }
    output(manifest_path, manifest)
    for src in source_artifacts.values():
        require(sha256_file(Path(src["path"])) == src["sha256"], "source_changed_during_compile")
    require(all(not p.exists() for p in pending), "generated_output_exists_no_overwrite")
    feedback_root.mkdir(parents=False, exist_ok=False)
    for path, (value, is_text, serialized) in pending.items():
        require(not path.exists(), "generated_output_exists_no_overwrite")
        if is_text:
            atomic_text(path, value)
        else:
            atomic_json(path, value)
        require(path.read_bytes() == serialized.encode("utf-8"), "written_output_mismatch")
    _, checked = _compiled_feedback_files(feedback_root, ORIGINAL / "prepared_run")
    require(checked == manifest, "formal_feedback_manifest_validation_failed")
    print(json.dumps({"status": manifest["status"], "external_calls": 0,
                      "manifest": str(manifest_path),
                      "roots": [{"root_id": r["root_id"], "counts": r["counts"],
                                 "feedback_lengths": r["feedback_lengths"]} for r in records]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
