"""Offline constructor proof for prepared merged-source question generation.

Writes request previews, never API request records. Review candidates are clearly
marked synthetic interface fixtures, never generated questions or training data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v17_source_flow_generator", HERE / "generate.py")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)
base, legacy = generation.base, generation.legacy
require, read, write = base.require, base.read, base.write


def parse_input(prompt, marker):
    data, end = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])
    return data


def review_fixture(root):
    return {"candidate_id": root + "-OFFLINE-CONSTRUCTOR-FIXTURE-NOT-A-QUESTION",
            "root_id": root, "question_type": "ordering", "chronology_kind": "conversation_order",
            "question": "OFFLINE CONSTRUCTOR FIXTURE ONLY; do not evaluate or train.",
            "items": {f"I{i}": "Placeholder interface item " + str(i) for i in range(1, 7)},
            "gold": ["SECRET_WRITER_GOLD"], "item_evidence": {"SECRET_WRITER_EVIDENCE": True},
            "qwen_review": {"SECRET_PRIOR_VERDICT": True}}


def review_payload(run, root, source):
    candidate = review_fixture(root)
    payload = {"model": generation.MODEL, "temperature": .25, "max_tokens": 4096,
               "enable_thinking": False, "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": legacy.ordering_gate.build_review_prompt(candidate, source)}]}
    payload, receipt = legacy.feedback.append_feedback_to_request(
        payload, run / "feedback", root, "writer", generation.TARGET)
    return payload, receipt


def check(run):
    manifest = generation.verify(run)
    require(not (run / "STARTED.json").exists(), "generation_already_started")
    require(not (run / "requests").exists(), "provider_request_records_exist")
    require(not (run / "SOURCE_FLOW_CHECK.json").exists(), "source_flow_check_already_exists")
    merge = read(Path(manifest["merge_manifest_path"]))
    reports, preview_paths = [], []
    for row in manifest["source_rows"]:
        root = row["root_id"]
        source = read(Path(row["source_path"]))
        prior = read(run / "prior" / (root + ".json"))
        inventory = generation.prompt_prior(prior)
        require(all(set(item) == {"candidate_id", "question_type", "question", "facts", "dedup_action"}
                    for item in inventory), "prior_inventory_unexpected_field")
        suffix = legacy.feedback.build_feedback_suffix(run / "feedback", root, "writer", generation.TARGET)
        paths, receipts = [], []
        for qtype in generation.TYPES:
            code = "multi" if qtype == "multi_select" else "order"
            slots = manifest["candidate_slots_per_type_per_root"]
            ids = [f"{root}-v17-{run.name}-merged-{code}-b01-{i:02d}" for i in range(1, slots + 1)]
            context = {"root_id": root, "context_mode": "checked_merged_root", "candidate_ids": ids,
                       "source_turns": source["turns"], "previous_candidates_dedup_only": inventory,
                       "priority_new_scene_refs": sorted({t.get("scene_ref", t["turn_ref"].rsplit(":", 1)[0])
                           for t in source["turns"] if t["turn_ref"].startswith("R02:")}),
                       "new_scene_evidence_required": False, "paraphrase_augmentation_allowed": True}
            payload, receipt = generation.make_payload(run, root, qtype, context)
            prompt = payload["messages"][0]["content"]
            parsed = parse_input(prompt, "\nINPUT_JSON\n")
            require(parsed == context and parsed["source_turns"] == source["turns"], "generator_source_truncated_or_changed")
            require(prompt.endswith(suffix) and receipt["feedback_appended"] is True, "generator_feedback_not_appended")
            require(payload["model"] == generation.MODEL, "generator_model_changed")
            require("Do not create another copy" not in prompt, "paraphrase_ban_present")
            target = run / "source_flow_previews" / f"{root}.{qtype}.GENERATOR_NOT_SENT.json"
            write(target, {"offline_preview_only": True, "sent_to_provider": False, "payload": payload,
                           "feedback_receipt": receipt, "source_sha256": row["source_sha256"]})
            paths.append(target); receipts.append(receipt)
        payload, receipt = review_payload(run, root, source)
        prompt = payload["messages"][0]["content"]
        parsed = parse_input(prompt, "\nDATA:\n")
        require(parsed["source_turns"] == source["turns"], "review_source_truncated_or_changed")
        require(prompt.endswith(suffix) and receipt["feedback_appended"] is True, "review_feedback_not_appended")
        require(all(secret not in prompt for secret in ("SECRET_WRITER_GOLD", "SECRET_WRITER_EVIDENCE", "SECRET_PRIOR_VERDICT")),
                "review_anchored_by_writer_or_prior_verdict")
        target = run / "source_flow_previews" / f"{root}.ORDERING_REVIEW_FIXTURE_NOT_SENT.json"
        write(target, {"offline_preview_only": True, "synthetic_review_interface_fixture": True,
                       "not_a_generated_question": True, "sent_to_provider": False,
                       "payload": payload, "feedback_receipt": receipt, "source_sha256": row["source_sha256"]})
        paths.append(target); receipts.append(receipt)
        routes = sorted({t["turn_ref"].split(":")[1] for t in source["turns"] if t["turn_ref"].startswith("R02:")})
        reports.append({"root_id": root, "source_path": row["source_path"], "source_sha256": row["source_sha256"],
                        "source_turns": len(source["turns"]), "accepted_scenes": row["accepted_scenes"],
                        "accepted_r02_routes": routes, "all_three_constructor_sources_exact": True,
                        "feedback_in_all_three_constructors": True, "prior_contains_current_gold_scores_answers": False,
                        "review_contains_writer_gold_evidence_prior_verdict": False,
                        "preview_files": base.bindings(paths), "feedback_receipts": receipts})
        preview_paths += paths
    # A separate synthetic source demonstrates the interface can carry both routes;
    # it is neither a real merge acceptance result nor a generated question.
    root = manifest["roots"][0]
    synthetic = {"root_id": root, "turns": [
        {"turn_ref": "R02:A:S01:T001", "speaker": "FixtureA", "text": "SYNTHETIC ROUTE A SENTINEL"},
        {"turn_ref": "R02:B:S01:T001", "speaker": "FixtureB", "text": "SYNTHETIC ROUTE B SENTINEL"}]}
    for qtype in generation.TYPES:
        payload, _ = generation.make_payload(run, root, qtype, {"root_id": root,
            "candidate_ids": ["SYNTHETIC-CONSTRUCTOR-ONLY"], "source_turns": synthetic["turns"],
            "previous_candidates_dedup_only": []})
        require(parse_input(payload["messages"][0]["content"], "\nINPUT_JSON\n")["source_turns"] == synthetic["turns"],
                "synthetic_both_route_generator_truncation")
    payload, _ = review_payload(run, root, synthetic)
    require(parse_input(payload["messages"][0]["content"], "\nDATA:\n")["source_turns"] == synthetic["turns"],
            "synthetic_both_route_reviewer_truncation")
    generation.verify(run)
    result = {"status": "PASS", "checked_at_utc": generation.v15.now(),
              "scope": "OFFLINE_ACTUAL_SOURCE_CONSTRUCTOR_AUDIT_NOT_LIVE_GENERATION",
              "prepared_manifest_path": str(run / "MANIFEST.json"), "merge_manifest_path": manifest["merge_manifest_path"],
              "roots": reports, "root_count": len(reports), "source_turn_count": sum(r["source_turns"] for r in reports),
              "accepted_scenes": sum(r["accepted_scenes"] for r in reports),
              "actual_roots_with_both_A_B_accepted": sum(r["accepted_r02_routes"] == ["A", "B"] for r in reports),
              "both_route_transport_test": {"status": "PASS", "synthetic_fixture_only": True,
                  "real_both_route_merge_success_claimed": False},
              "held_source_scenes": merge.get("excluded_scenes", []),
              "constructor_preview_count": len(preview_paths), "generator_constructor_previews": 2 * len(reports),
              "ordering_review_constructor_previews": len(reports),
              "prepared_candidate_slots": manifest["budget"]["candidate_slots"], "budget": manifest["budget"],
              "actual_generation_api_calls": 0, "actual_solver_calls": 0, "actual_training_calls": 0,
              "actual_new_questions": 0, "training_started": False,
              "future_generation_gold_rules": {
                  "multi_select": "Labels preserve the model's raw option order; Gold is labels with relation=supports; explicit inconsistent raw Gold rejects.",
                  "ordering": "Inherited exact-clause conversation-order mapping recomputes Gold from writer anchors, then independent Qwen reviewer anchors; original_gold preserved.",
                  "current_migration": "This audit and generator preparation do not edit, migrate, or recompute any existing question or Gold."},
              "files": base.bindings([Path(__file__), HERE / "generate.py", run / "MANIFEST.json",
                                       Path(manifest["merge_manifest_path"]), *preview_paths])}
    write(run / "SOURCE_FLOW_CHECK.json", result)
    return {key: result[key] for key in ("status", "root_count", "source_turn_count", "accepted_scenes",
        "actual_roots_with_both_A_B_accepted", "constructor_preview_count", "prepared_candidate_slots",
        "actual_generation_api_calls", "actual_solver_calls", "actual_new_questions")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.run.resolve()), indent=2))


if __name__ == "__main__":
    main()
