#!/usr/bin/env python3
"""One prospective source scene, with reviewer-selected atomic citations.

This independent diagnostic never admits a question or edits earlier results.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
from completion_guard_v3 import validate_completion
from controlled_anchors import expand_review, make_catalog
from production_core_v3 import (CHECKS, SCENES, bind, build_cumulative, make_turns, now,
                             read, require, validate_review, write)
from run_production_v3 import CodexCallBackend, HERE, OET1, ProductionPipeline, owned_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--root", required=True)
    parser.add_argument("--arm", required=True, choices=("GENERIC", "TARGETED"))
    parser.add_argument("--execute-model-calls", action="store_true")
    args = parser.parse_args()
    require(args.execute_model_calls, "MODEL_EXECUTION_FLAG_REQUIRED")
    run_dir = owned_output(args.run_dir)
    require(not run_dir.exists(), "NEW_PROSPECTIVE_PROBE_DIRECTORY_REQUIRED")
    run_dir.mkdir(parents=True)
    started = now()
    scene = SCENES[0]
    try:
        backend = CodexCallBackend(run_dir)
        helper = ProductionPipeline(run_dir, backend)
        write(run_dir / "PROTOCOL.json", {
            "schema_version": "controlled_citation_prospective_probe_v1", "started_at_utc": started,
            "root_id": args.root, "arm": args.arm, "scene": scene,
            "planned_scenes": 1, "max_semantic_attempts_per_role": 1,
            "new_author_and_reviewer": True, "old_failures_reclassified": False,
            "diagnostic_only_no_training_admission": True,
            "code_bindings": [bind(Path(__file__)), bind(HERE / "controlled_anchors.py"),
                              bind(HERE / "production_core_v3.py"), bind(HERE / "completion_guard_v3.py"),
                              bind(HERE / "run_production_v3.py"), bind(OET1 / "scripts/production_driver.py"),
                              bind(OET1 / "scripts/production_integrity.py")],
            "semantic_checks": CHECKS})
        packet, _, packet_path = helper._prepare_packet(args.root, args.arm)
        previous_path = Path(packet["source_binding"]["path"])
        previous = read(previous_path)

        def call(role, prompt):
            call_id = f"{args.root}__{args.arm}__{scene.replace(':', '_')}__{role}"
            response, result = helper._checked_call(call_id, prompt)
            _, events = helper.integrity.validate_call_result(run_dir / "cli_calls" / call_id / "RESULT.json", prompt)
            validate_completion(result, events)
            write(run_dir / f"{role}_RESULT.json", result)
            return response, result

        raw_author, author = call("AUTHOR", helper.prompt_driver.source_author(packet, previous, scene))
        turns = make_turns(raw_author, packet, previous, scene)
        draft = {"schema_version": "controlled_citation_scene_draft_v1", "root_id": args.root,
                 "scene_ref": scene, "author_id": author["backend"]["thread_id"],
                 "created_at_utc": author["finished_at_utc"], "input_source_binding": bind(previous_path),
                 "author_input_binding": bind(packet_path), "turns": turns,
                 "cli_raw_binding": author["raw_response"], "controller_metadata_only": True}
        draft_binding = write(run_dir / "DRAFT.json", draft)
        source, proof = build_cumulative(packet, previous, draft, scene, bind(previous_path), draft_binding)
        write(run_dir / "CUMULATIVE_SOURCE.json", source)
        write(run_dir / "PREFIX_PROOF.json", proof)
        catalog = make_catalog(packet, source, scene)
        write(run_dir / "EVIDENCE_CATALOG.json", catalog)
        prompt = helper.prompt_driver.prompt_prefix("independent source reviewer") + """
Read EVERY cumulative source turn and the new 14-turn append. Independently judge
time, location, actor_identity, plan_vs_completion, cross_scene_continuity,
semantic_repetition, naturalness, event_fidelity. Check prefix preservation,
all new clauses and factual/actor/state continuity. Inherited uncertainty may
be WARNING only if the append does not depend on or worsen it. New ambiguity,
unsupported service replies or source factual contradictions are blocking.
Domestic fictional actions are allowed. Never rewrite the source or request retries.

Atomic citation protocol: the catalog contains exact original full utterances
with immutable anchor IDs. For EVERY dimension select specific relevant evidence
yourself: evidence:[{"anchor_id":"..."}]. Explain which clause of the utterance
supports your reasoning; the catalog does not establish semantic correctness.
Do NOT output a turn_ref or copy a quote. The controller only mechanically expands
your selected anchor; it will never repair, replace or choose your evidence.
For findings, use anchor_ids:[...] instead of turn_refs. A finding must select at
least one relevant anchor. Read every turn even if you cite a small subset.

Return JSON {"catalog_id":"exact given catalog_id", "decision":"READY/FAIL/HOLD",
"full_source_read":true,"source_turn_count":N,"independent_of_source_author":true,
"source_or_gold_ambiguity_failure":boolean,"checks":{"time":{"status":"PASS/WARNING/FAIL/HOLD",
"reason":"specific reasoning about cited clauses","evidence":[{"anchor_id":"..."}]},
...all eight dimensions...},"findings":[{"severity":"WARNING/BLOCKING","dimension":"...",
"anchor_ids":["..."],"reason":"..."}],"failure_codes":[]}.
READY only when no blocking finding. Source factual/actor/status/time ambiguity
must set source_or_gold_ambiguity_failure true and explicit failure_codes.
Do not base your verdict on production yield or downstream scores.
""" + json.dumps({"packet_scope": {"root_id": args.root, "condition": args.arm, "scene": scene},
                        "expected_source_turn_count": len(source["turns"]), "source_metadata": {
                            key: value for key, value in source.items() if key != "turns"},
                        "draft": draft, "prefix_proof": proof, "complete_source_in_citation_catalog": catalog,
                        "assigned_native_event": None}, ensure_ascii=False)
        raw_review, reviewer = call("REVIEW", prompt)
        require(reviewer["backend"]["thread_id"] != author["backend"]["thread_id"], "SOURCE_SELF_REVIEW")
        write(run_dir / "RAW_REVIEW.json", raw_review)
        expanded, mapping = expand_review(raw_review, catalog, packet, source, scene)
        write(run_dir / "EXPANDED_REVIEW.json", expanded)
        write(run_dir / "MECHANICAL_EXPANSION.json", mapping)
        evidence = validate_review(expanded, packet, source, scene, require_ready=False)
        write(run_dir / "REVIEW_VALIDATION.json", evidence)
        validate_review(expanded, packet, source, scene)
        result = {"status": "PASS_SINGLE_SCENE_DIAGNOSTIC_ONLY", "started_at_utc": started,
                  "finished_at_utc": now(), "new_turns": len(turns), "source_turn_count": len(source["turns"]),
                  "model_calls": 2, "semantic_retries": 0, "review_decision": raw_review["decision"],
                  "checked_citations": len(evidence["checked_quotes"]),
                  "old_failures_reclassified": False, "training_pool_admission": False,
                  "multi_scene_or_multi_root_stability_established": False,
                  "protocol": bind(run_dir / "PROTOCOL.json"), "source": bind(run_dir / "CUMULATIVE_SOURCE.json"),
                  "raw_review": bind(run_dir / "RAW_REVIEW.json"),
                  "validation": bind(run_dir / "REVIEW_VALIDATION.json")}
        write(run_dir / "RESULT.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as error:
        result = {"status": "FAIL_PROSPECTIVE_DIAGNOSTIC_NO_RETRY", "started_at_utc": started,
                  "finished_at_utc": now(), "error_code": getattr(error, "code", type(error).__name__),
                  "error": str(error), "traceback": traceback.format_exc(), "training_pool_admission": False,
                  "old_failures_reclassified": False}
        write(run_dir / "RESULT.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
