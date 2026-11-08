"""Independent offline audit for the v17 checked-root migration.

This file deliberately does not call a provider or Solver.  It checks the
immutable source/merge/review/migration artifacts and, when a completed Solver
directory exists, delegates raw-call and score recomputation to the existing
v15 independent auditor.  It is kept separate from merge.py so a merge bug
cannot also make its own audit pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
QA = HERE.parent / "h4_feedback_training_pool_v15_qa"
OLD_AUDIT = QA / "audit_run.py"


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()


def binding(path: Path | str, expected: str) -> None:
    require(Path(path).is_file() and sha(path) == expected, "file_hash_changed:" + str(path))


def require(value: Any, reason: str) -> None:
    if not value:
        raise RuntimeError(reason)


def public_turn(turn: dict) -> tuple[str, str, str]:
    return (turn.get("turn_ref", ""), turn.get("speaker", ""), turn.get("text", ""))


def public_question(q: dict) -> tuple:
    field = "options" if q.get("question_type") == "multi_select" else "items"
    return (q.get("candidate_id"), q.get("root_id"), q.get("question_type"),
            q.get("question"), tuple((q.get(field) or {}).items()), tuple(q.get("gold", [])))


def ref_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(ref_values(item))
        return out
    if isinstance(value, dict):
        if isinstance(value.get("turn_ref"), str):
            return [value["turn_ref"]]
        out = []
        for item in value.values():
            out.extend(ref_values(item))
        return out
    return []


def aliases(ref: str, root: str) -> set[str]:
    result = {ref}
    if ref.startswith(root + ":"):
        result.add(ref[len(root) + 1:])
    else:
        result.add(root + ":" + ref)
    return result


def matching_ref(ref: str, old_turns: dict[str, dict], new_turns: dict[str, dict], root: str) -> tuple[str | None, str | None]:
    old = [key for key in aliases(ref, root) if key in old_turns]
    new = [key for key in aliases(ref, root) if key in new_turns]
    return (old[0] if len(old) == 1 else None, new[0] if len(new) == 1 else None)


def check_review(row: dict, verdict: dict, all_turns: dict[str, dict]) -> None:
    scenes = verdict.get("scenes")
    require(isinstance(scenes, list), "review_scenes_not_list")
    expected = set(row["scene_blocks"])
    require(len(scenes) == len(expected) and {s.get("scene_id") for s in scenes} == expected, "review_scene_set_mismatch")
    for scene in scenes:
        require(scene.get("status") in {"PASS", "HOLD"} and isinstance(scene.get("reason"), str)
                and scene["reason"].strip(), "review_verdict_shape")
        evidence = scene.get("evidence")
        require(isinstance(evidence, list), "review_evidence_shape")
        if scene["status"] == "HOLD":
            require(evidence, "hold_without_exact_evidence")
        for cite in evidence:
            ref, quote = cite.get("turn_ref"), cite.get("quote")
            require(isinstance(ref, str) and ref in all_turns and isinstance(quote, str) and quote
                    and quote in all_turns[ref]["text"], "review_quote_not_exact")
    order = verdict.get("accepted_scene_order")
    passed = {s["scene_id"] for s in scenes if s["status"] == "PASS"}
    require(isinstance(order, list) and len(order) == len(set(order)) and set(order) == passed,
            "review_accepted_order_mismatch")
    if verdict.get("review_basis") == "same_qwen_joint_source_compatibility":
        request, response = read(verdict["request_path"]), read(verdict["response_path"])
        model = "Qwen/Qwen3.6-35B-A3B"
        require(request["model"] == verdict["requested_model"] == verdict["returned_model"] == model,
                "review_wrong_model")
        require(response["metadata"]["returned_model"] == model and response["metadata"]["finish_reason"] == "stop",
                "review_response_model_or_completion")
        raw = read(verdict["response_path"])["text"]
        parsed = json.loads(raw)
        require(all(parsed.get(k) == verdict.get(k) for k in ("root_id", "scenes", "accepted_scene_order", "joint_reason")),
                "review_decision_not_raw_model_output")
        messages = request.get("messages")
        require(len(messages) == 1 and messages[0]["role"] == "user", "review_prompt_shape")
        request_data = json.loads(messages[0]["content"].split("\nINPUT:\n", 1)[1])
        expected_data = {"root_id": row["root_id"], "scene_ids_to_review_exactly": list(row["scene_blocks"]),
                         "frozen_base": [{k: t[k] for k in ("turn_ref", "speaker", "text")} for t in row["base"]["turns"]],
                         "independently_generated_scenes": {scene: [{k: t[k] for k in ("turn_ref", "speaker", "text")} for t in turns]
                                                            for scene, turns in row["scene_blocks"].items()}}
        require(request_data == expected_data, "review_context_wrong_or_question_gold_leak")
        directory = Path(verdict["request_path"]).parent
        proof = read(directory / "PROOF.json")
        attempt = read(directory / "HTTP_ATTEMPT.json")
        require(proof["payload_sha256"] == attempt["payload_sha256"] == digest(request), "review_request_digest_mismatch")
        require(attempt["actual_payload_equals_frozen"] is True, "review_actual_request_changed")
    else:
        require(verdict.get("review_basis") == "historical_accepted_source_unchanged",
                "review_unrecognized_basis")
        branches = {key.split(":")[1] for key in row["scene_blocks"]}
        require(len(branches) < 2 and verdict["new_model_calls"] == 0,
                "multiple_branches_without_joint_review")


def check_sources(run: Path, manifest: dict, merge_manifest: dict) -> dict[str, dict]:
    inputs = {root: read(run / "inputs" / (root + ".json")) for root in manifest["roots"]}
    sources: dict[str, dict] = {}
    for row in merge_manifest["source_rows"]:
        root = row["root_id"]
        path = Path(row["path"])
        require(path.is_file() and sha(path) == row["sha256"], "merged_source_hash:" + root)
        source = read(path); inp = inputs[root]
        for bound in [inp["base_binding"], *inp["input_bindings"]]:
            binding(bound["path"], bound["sha256"])
        require(read(inp["base_binding"]["path"]) == inp["base"], "input_base_not_bound_source")
        base = inp["base"]["turns"]
        original_blocks = {}
        for bound in inp["input_bindings"]:
            if Path(bound["path"]).name == "cumulative_dialogue.json":
                branch = read(bound["path"])
                require(branch["turns"][:len(base)] == base and branch["branch_independent"] is True,
                        "original_branch_prefix_or_independence_changed")
                result = read(Path(bound["path"]).with_name("RESULT.json"))
                require(result["other_branch_in_generator_context"] is False,
                        "branch_generation_not_independent")
                suffix = branch["turns"][len(base):]
                require(len(suffix) == result["added_turns"], "original_branch_turn_count")
                for turn in suffix:
                    original_blocks.setdefault(turn["turn_ref"].rsplit(":", 1)[0], []).append(turn)
        require(inp["scene_blocks"] == original_blocks, "input_scene_blocks_changed")
        turns = source["turns"]
        require(source["root_id"] == root and turns[:len(base)] == base,
                "base_prefix_changed_or_missing:" + root)
        require(len({t["turn_ref"] for t in turns}) == len(turns), "merged_turn_ref_collision:" + root)
        require(source.get("downstream_context_merged") is True and source.get("branch_generation_independent") is True,
                "merged_source_flags_missing:" + root)
        verdict_path = run / "review" / (root + ".retry01.json")
        if not verdict_path.exists():
            verdict_path = run / "review" / (root + ".json")
        verdict = read(verdict_path)
        require(read(run / "checks" / (root + ".json"))["joint_review"] == verdict,
                "recorded_check_review_differs")
        all_input_turns = {t["turn_ref"]: t for t in base}
        for block in inp["scene_blocks"].values():
            for turn in block:
                all_input_turns.setdefault(turn["turn_ref"], turn)
        check_review(inp, verdict, all_input_turns)
        passed = set(verdict["accepted_scene_order"])
        expected = list(base)
        duplicate_module = load_module("v17_audit_duplicate", HERE.parent / "h4_r02_independent_routes_v4/duplicate_check.py")
        checks = []
        for scene_id in verdict["accepted_scene_order"]:
            diagnostic = duplicate_module.inspect_duplicates(inp["scene_blocks"][scene_id], {"accepted_root_context": expected})
            checks.append({"scene_id": scene_id, **diagnostic})
            require(diagnostic["status"] == "PASS", "accepted_scene_fails_exact_duplicate_check")
            expected.extend(inp["scene_blocks"][scene_id])
        require(checks == read(run / "checks" / (root + ".json"))["exact_duplicate_checks"], "exact_duplicate_diagnostics_changed")
        require(turns == expected, "merged_turns_do_not_match_review_order:" + root)
        require(source["accepted_scene_order"] == verdict["accepted_scene_order"],
                "source_review_order_mismatch:" + root)
        require(row["added_turns"] == len(turns) - len(base) and row["base_turns"] == len(base)
                and row["accepted_scenes"] == len(verdict["accepted_scene_order"]), "source_row_counts_wrong")
        for scene_id in inp["scene_blocks"]:
            if scene_id not in passed:
                require(all(t not in turns for t in inp["scene_blocks"][scene_id]),
                        "held_scene_leaked_into_source:" + root + ":" + scene_id)
        sources[root] = {"path": path, "sha256": row["sha256"], "source": source}
    require(set(sources) == set(manifest["roots"]), "source_root_set_mismatch")
    return sources


def check_migration(run: Path, sources: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    original = [json.loads(line) for line in (run / "ORIGINAL_TRAINING_POOL.jsonl").read_text().splitlines() if line.strip()]
    migration = read(run / "MIGRATION.json")
    pool = read(run / "M/generation/SOLVER_POOL.json")
    ready = pool["questions"]
    held = migration["held"]
    old_by_id = {row["candidate_id"]: row["original_selected_question"] for row in original}
    require(len(original) == migration["historical_pool_count"], "historical_count_mismatch")
    require(len(ready) == migration["ready_for_solver"], "migration_ready_count_mismatch")
    require(len(ready) + len(held) == len(original), "migration_partition_not_complete")
    ready_ids = {q["candidate_id"] for q in ready}
    held_ids = {q["candidate_id"] for q in held}
    require(not ready_ids & held_ids and ready_ids | held_ids == set(old_by_id), "migration_ids_not_partitioned")
    counts = Counter(q["question_type"] for q in ready)
    require(dict(counts) == migration["counts_by_type"], "migration_type_counts_mismatch")
    for q in ready:
        old = old_by_id[q["candidate_id"]]
        require(public_question(q) == public_question(old), "public_or_gold_changed:" + q["candidate_id"])
        root = q["root_id"]; target = sources[root]
        require(Path(q["source_path"]) == target["path"] and q["source_sha256"] == target["sha256"],
                "question_source_binding_mismatch:" + q["candidate_id"])
        old_source = read(old["source_path"])
        require(sha(old["source_path"]) == old["source_sha256"], "historical_source_changed:" + q["candidate_id"])
        old_turns = {t["turn_ref"]: t for t in old_source["turns"]}
        new_turns = {t["turn_ref"]: t for t in target["source"]["turns"]}
        field = "option_evidence" if q["question_type"] == "multi_select" else "item_evidence"
        refs = []
        for entry in (q.get(field) or {}).values():
            refs.extend(ref_values(entry.get("turn_ref") if isinstance(entry, dict) else None))
        if q["question_type"] == "ordering":
            for key in ("order_anchor_spans", "generated_order_anchor_spans", "order_anchor", "support_refs",
                        "alternative_matching_anchors", "allowed_matching_anchor_spans", "alternative_matching_turn_refs"):
                refs.extend(ref_values(q.get(key)))
        for ref in refs:
            oldref, newref = matching_ref(ref, old_turns, new_turns, root)
            require(oldref is not None and newref is not None, "question_reference_unresolved:" + q["candidate_id"])
            require(public_turn(old_turns[oldref])[1:] == public_turn(new_turns[newref])[1:],
                    "question_reference_text_changed:" + q["candidate_id"])
        if q["question_type"] == "ordering":
            spans = q.get("order_anchor_spans")
            require(isinstance(spans, dict) and set(spans) == {f"I{i}" for i in range(1, 7)},
                    "migrated_order_spans_missing:" + q["candidate_id"])
    return original, ready


def synthetic_regression() -> dict:
    base = [{"turn_ref": "P:T001", "speaker": "A", "text": "base one"},
            {"turn_ref": "P:T002", "speaker": "B", "text": "base two"}]
    a = [{"turn_ref": "R02:A:S01:T001", "speaker": "A", "text": "new A"}]
    b = [{"turn_ref": "R02:B:S01:T001", "speaker": "B", "text": "new B"}]
    sys.path.insert(0, str(QA))
    implementation = load_module("v17_audit_actual_merge", HERE / "merge.py")
    generation = load_module("v17_audit_actual_generation", HERE / "generate.py")
    with tempfile.TemporaryDirectory(prefix="v17-audit-regression-") as folder:
        root = Path(folder); baseline = {"root_id": "synthetic", "turns": base}
        basefile = root / "base.json"; basefile.write_text(json.dumps(baseline))
        bound = {"path": str(basefile), "sha256": sha(basefile)}
        row = {"root_id": "synthetic", "base": baseline, "base_binding": bound,
               "input_bindings": [], "scene_blocks": {"R02:A:S01": a, "R02:B:S01": b}}
        verdict = {"root_id": "synthetic", "scenes": [{"scene_id": k, "status": "PASS", "reason": "fixture", "evidence": []}
                                                                  for k in row["scene_blocks"]],
                   "accepted_scene_order": list(row["scene_blocks"]), "review_basis": "synthetic_fixture_no_model"}
        def fixture(path, data, decision):
            (path / "inputs").mkdir(parents=True); (path / "review").mkdir()
            (path / "inputs/synthetic.json").write_text(json.dumps(data))
            (path / "review/synthetic.json").write_text(json.dumps(decision))
            (path / "MANIFEST.json").write_text(json.dumps({"roots": ["synthetic"]}))
        accepted = root / "accepted"; fixture(accepted, row, verdict)
        implementation.merge(accepted)
        source = read(accepted / "sources/synthetic.json")
        merged = source["turns"]
        require(merged == base + a + b and len({t["turn_ref"] for t in merged}) == 4,
                "actual_merge_dropped_or_duplicated_a_branch")
        _, checked, excluded = generation.checked_sources(accepted / "MERGE_MANIFEST.json")
        require(len(checked) == 1 and not excluded and checked[0]["source"] == source,
                "generation_does_not_consume_entire_root")
        original_payload = generation.legacy.payload_with_feedback
        try:
            generation.legacy.payload_with_feedback = lambda run, rid, instructions, context, budget: {
                "instructions": instructions, "context": deepcopy(context)}
            payload = generation.make_payload(root, "synthetic", "multi_select",
                                              {"candidate_ids": ["synthetic-multi"], "source_turns": merged})
            require(payload["context"]["source_turns"] == base + a + b,
                    "generation_payload_loses_a_branch")
        finally:
            generation.legacy.payload_with_feedback = original_payload
        collision = deepcopy(row); collision["scene_blocks"]["R02:B:S01"][0]["turn_ref"] = a[0]["turn_ref"]
        collrun = root / "collision"; fixture(collrun, collision, verdict)
        try:
            implementation.merge(collrun)
        except (ValueError, RuntimeError) as error:
            require("merged_ref_collision" in str(error), "collision_rejected_for_unexpected_reason")
        else:
            raise RuntimeError("actual_merge_accepted_reference_collision")
        copied = deepcopy(row)
        long_a = [{"turn_ref": f"R02:A:S01:T{i:03d}", "speaker": "A",
                   "text": f"Substantive unique copied clause number {i} containing more than forty characters."} for i in range(1, 4)]
        long_b = [dict(t, turn_ref=t["turn_ref"].replace(":A:", ":B:")) for t in long_a]
        copied["scene_blocks"] = {"R02:A:S01": long_a, "R02:B:S01": long_b}
        duprun = root / "duplicate"; fixture(duprun, copied, verdict)
        dupresult = implementation.merge(duprun)
        require(read(duprun / "sources/synthetic.json")["turns"] == base + long_a,
                "actual_merge_accepts_copied_dialogue")
        require(dupresult["excluded_scenes"][0]["reason"] == "exact_dialogue_duplicate",
                "copied_scene_exclusion_missing")
    return {"status": "PASS", "base_turns": 2, "accepted_scene_turns": 2,
            "merged_turns": 4, "actual_merge_code_exercised": True,
            "generation_adapter_and_payload_receive_both_branches": True,
            "repeated_dialogue_rejected": True, "reference_collision_rejected": True}


def independent_solver_audit(run: Path, pool: dict, sources: dict[str, dict]) -> dict:
    solver_dir = run / "M/solver"
    complete = solver_dir / "COMPLETE.json"
    if not complete.exists():
        return {"status": "PENDING_SOLVER_COMPLETE"}
    spec = importlib.util.spec_from_file_location("v17_old_audit", OLD_AUDIT)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    audit = module.Audit()
    audit.solver(solver_dir, pool, {root: row["source"] for root, row in sources.items()})
    return {"status": "PASS", "full_calls_recomputed": audit.counts.get("full_solver_calls_independently_recomputed", 0),
            "selected_ids": [r["candidate_id"] for r in audit.question_results if r["selected_by_policy"]]}


def check_final_pool(run: Path, original: list[dict], ready: list[dict], sources: dict, solver: dict) -> dict:
    directory = run / "training_pool"
    rows = [json.loads(line) for line in (directory / "TRAINING_POOL.jsonl").read_text().splitlines() if line.strip()]
    report = read(directory / "REPORT.json")
    results = read(run / "M/solver/RESULTS.json")
    by_id = {q["candidate_id"]: q for q in ready}
    old = {q["candidate_id"]: q for q in original}
    result_by_id = {q["candidate_id"]: q for q in results["questions"]}
    require([q["candidate_id"] for q in rows] == solver["selected_ids"], "final_selected_ids_mismatch")
    root_counts = {}
    for q in ready:
        root_counts.setdefault(q["root_id"], Counter())["evaluated"] += 1
    for row in rows:
        qid, root = row["candidate_id"], row["root_id"]
        question = row["original_selected_question"]
        source = sources[root]
        require(public_question(row) == public_question(old[qid]) == public_question(question)
                == public_question(by_id[qid]), "final_public_or_gold_rewritten:" + qid)
        require(row["source_path"] == question["source_path"] == str(source["path"])
                and row["source_sha256"] == question["source_sha256"] == source["sha256"],
                "final_source_binding_mismatch:" + qid)
        require(isinstance(row["full_context"], dict) and row["full_context"] == source["source"],
                "final_full_context_not_complete_source_object:" + qid)
        require(question["training_selection"] == result_by_id[qid], "final_selection_not_current_solver:" + qid)
        require(row["historical_pool_row"] == old[qid], "historical_pool_row_rewritten:" + qid)
        require(row["provenance_chain"] == old[qid]["provenance_chain"], "historical_provenance_rewritten:" + qid)
        require(row["origin"] == "v17_checked_root_merged_reevaluation" and row["canonical_exposure_units"] == 1
                and row["evaluated_permutations_are_separate_training_examples"] is False,
                "final_exposure_or_origin_wrong:" + qid)
        for field in ("canonical_question_id", "canonical_content_id_by_label", "complete_gold_content_ids"):
            require(row[field] == old[qid][field], "canonical_training_target_changed:" + qid + ":" + field)
        require(row["training_target"] == question["training_target"] == "complete_canonical_gold"
                and row["loss_weight_from_partial_score"] is False and question["loss_weight_from_partial_score"] is False,
                "partial_answer_or_score_weight_used:" + qid)
        root_counts[root]["selected"] += 1
        root_counts[root][row["question_type"]] += 1
    require(read(directory / "SELECTED_QUESTIONS.json")["questions"] == [r["original_selected_question"] for r in rows],
            "final_question_export_mismatch")
    require(report["selected"] == len(rows) and report["evaluated"] == len(ready)
            and report["old_pool_count"] == len(original), "final_counts_mismatch")
    require(report["by_type"] == dict(Counter(q["question_type"] for q in rows))
            and report["by_root"] == {root: dict(counts) for root, counts in root_counts.items()}, "final_breakdown_mismatch")
    require(report["source_held_count"] == len(original) - len(ready)
            and report["solver_rejected_count"] == len(ready) - len(rows)
            and report["full_solver_calls"] == solver["full_calls_recomputed"]
            and report["ground_calls"] == report["training_calls"] == 0
            and report["old_results_are_historical_only"] is True, "final_scope_counts_mismatch")
    rejected = [{k: q[k] for k in ("candidate_id", "question_type", "full_partial_mean", "selection_reason")}
                for q in results["questions"] if not q["selected_by_policy"]]
    return {"status": "PASS", "selected": len(rows), "by_type": report["by_type"], "rejected": rejected,
            "raw_question_gold_unchanged": True, "full_context_source_object": True,
            "current_selection_is_solver_results": True,
            "provenance_note": "Per-row provenance_chain is preserved historical provenance; current basis is source_migration plus current training_selection and independently bound Solver artifacts."}


def audit(run: Path, output: Path | None = None) -> dict:
    manifest = read(run / "MANIFEST.json")
    merge_manifest = read(run / "MERGE_MANIFEST.json")
    binding(manifest["original_pool"], manifest["original_pool_sha256"])
    binding(run / "ORIGINAL_TRAINING_POOL.jsonl", manifest["original_pool_sha256"])
    for bound in merge_manifest["files"]:
        binding(bound["path"], bound["sha256"])
    initial_code = [{"path": path, "prepared_sha256": expected, "current_sha256": sha(path),
                     "matches_current": sha(path) == expected} for path, expected in manifest["code_sha256"].items()]
    require(manifest["training_started"] is False and manifest["ground_calls"] == 0, "scope_flags_changed")
    require(merge_manifest["training_started"] is False and merge_manifest["original_base_included_once"] is True,
            "merge_scope_flags_changed")
    sources = check_sources(run, manifest, merge_manifest)
    original, ready = check_migration(run, sources)
    pool = read(run / "M/generation/SOLVER_POOL.json")
    solver_audit = independent_solver_audit(run, pool, sources)
    require(solver_audit["status"] == "PASS", "final_audit_requires_completed_solver")
    final_pool = check_final_pool(run, original, ready, sources, solver_audit)
    result = {"status": "PASS", "version": "h4_root_merged_context_v17_audit_v2",
              "run": str(run), "historical_pool_count": len(original), "ready_pool_count": len(ready),
              "ready_by_type": dict(Counter(q["question_type"] for q in ready)),
              "source_count": len(sources), "synthetic_regression": synthetic_regression(),
              "solver_audit": solver_audit,
              "final_training_pool": final_pool,
              "code_provenance": {"effective_merge_stage_bindings_verified": True,
                 "initial_prepare_code": initial_code,
                 "note": "Initial MANIFEST is the historical prepare-stage development record and is not claimed to bind the final code. MERGE_MANIFEST freezes the effective merge/migrate implementation; those bindings all match."},
              "checks": {"base_prefix_once": True, "held_scene_not_leaked": True,
                          "review_quotes_exact": True, "question_public_gold_unchanged": True,
                          "all_referenced_turns_unchanged": True, "ordering_span_sets_present": True,
                          "historical_pool_immutable": True, "training_not_started": True,
                          "ground_calls": 0, "semantic_multi_select_gate": False}}
    if output is None:
        output = run / "V17_AUDIT.json"
    output = Path(output)
    result["auditor"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    result["result_bindings"] = [{"path": str(path), "sha256": sha(path)} for path in (
        run / "MERGE_MANIFEST.json", run / "MIGRATION.json", run / "M/generation/SOLVER_POOL.json",
        run / "M/solver/manifest.json", run / "M/solver/RESULTS.json", run / "training_pool/TRAINING_POOL.jsonl",
        run / "training_pool/REPORT.json", run / "training_pool/SELECTED_QUESTIONS.json")]
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run.resolve(), args.output.resolve() if args.output else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
