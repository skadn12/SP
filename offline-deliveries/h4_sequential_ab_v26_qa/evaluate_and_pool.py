"""Evaluate new sequential-root questions and preserve an immutable new pool.

Reuses the v15 full-only evaluator, delivery, independent selection verifier,
and exact statement-set deduplicator. Only ``live`` performs Solver inference;
no phase generates questions, changes historical pools, or starts training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_original_search_path = sys.path[:]
generation = _load("v26_sequential_generation_for_pool", HERE / "generate.py")
sys.path[:] = _original_search_path
assembler = _load("v26_reused_v15_assembler", HERE.parent / "h4_feedback_training_pool_v15/assemble_pool.py")
v15, base = generation.v15, generation.base
evaluator, delivery = v15.evaluator, v15.delivery
read, write, require = base.read, base.write, base.require
VERSION = "h4_sequential_ab_v26_evaluation_pool_v1"


def _input_json(payload, marker):
    return json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split(marker, 1)[1])[0]


def verify_generation(run, inspector=None):
    """Replay source/Gold bindings and independent ordering review, offline."""
    inspector = inspector or assembler.Inspector()
    manifest = generation.verify(run)
    report = read(run / "REPORT.json")
    pool = read(run / "SOLVER_POOL.json")
    mechanical = read(run / "MECHANICAL_POOL.json")["questions"]
    require(report["version"] == pool["version"] == generation.VERSION, "generation_version_mismatch")
    require(report["ground_calls"] == report["training_calls"] == pool["ground_calls"] == 0
            and pool["training_started"] is False, "generation_scope_invalid")
    questions = pool["questions"]
    require(len(questions) == report["ready_for_solver"] == pool["candidate_count"], "generation_count_mismatch")
    by_id = {q["candidate_id"]: q for q in mechanical}
    require(len(by_id) == len(mechanical) == report["mechanical_pass"], "mechanical_pool_duplicate_or_count")
    require(len({q["candidate_id"] for q in questions}) == len(questions), "solver_pool_duplicate_id")
    source_rows = {row["root_id"]: row for row in manifest["source_rows"]}
    require(len(source_rows) == len(manifest["source_rows"]), "generation_multiple_sources_per_root")
    inspector.manifest(run / "MANIFEST.json")
    for path in (run / "REPORT.json", run / "SOLVER_POOL.json", run / "MECHANICAL_POOL.json",
                 Path(manifest["merge_manifest_path"])):
        inspector.bind(path)
    provenance = {}
    for question in questions:
        qid, root = question["candidate_id"], question["root_id"]
        require(qid in by_id and root in source_rows, "question_not_in_mechanical_or_root_pool")
        original = by_id[qid]
        source_path = Path(source_rows[root]["source_path"])
        source = read(source_path)
        require(Path(question["source_path"]).resolve() == source_path.resolve()
                and question["source_sha256"] == source_rows[root]["source_sha256"], "question_full_source_binding_wrong")
        inspector.bind(source_path, question["source_sha256"])
        request_path = Path(question["generation_request_path"])
        response_path = Path(question["generation_response_path"])
        require(request_path.parent.parent.resolve() == (run / "requests").resolve()
                and response_path == request_path.with_name("RESPONSE.json"), "generation_artifact_outside_run")
        request, response = read(request_path), read(response_path)
        context = _input_json(request, "\nINPUT_JSON\n")
        require(context["root_id"] == root and context["source_turns"] == source["turns"],
                "actual_generator_context_not_full_cumulative_root")
        metadata = response["metadata"]
        require(metadata["requested_model"] == metadata["returned_model"] == generation.MODEL
                and metadata["finish_reason"] == "stop", "generator_response_model_or_completion_invalid")
        drafts, _, _, error = generation.legacy.validate_draft(response["text"], context["candidate_ids"],
            root, question["question_type"], source, source_path)
        matches = [row for row in drafts if row["candidate_id"] == qid]
        require(error is None and len(matches) == 1, "raw_generation_candidate_not_valid")
        require(all(original.get(key) == value for key, value in matches[0].items()), "mechanical_question_changed_from_raw_Qwen")
        require(original.get("generator_model") == generation.MODEL
                and original.get("raw_question_content_preserved") is True, "raw_generator_provenance_missing")
        request_chain = inspector.feedback_request(request_path, run, root)
        request_chain["response"] = inspector.bind(response_path)
        review_chain = None
        if question["question_type"] == "ordering":
            review = question["qwen_review"]
            review_request, review_response = Path(review["request_path"]), Path(review["response_path"])
            require(review_request.parent.parent.resolve() == (run / "requests").resolve(), "ordering_review_outside_run")
            raw_review = read(review_response)
            replay = generation.legacy.ordering_gate.review_response(raw_review["text"], [original], source,
                review_response, review_request, raw_review["metadata"])
            require(replay["accepted"] == [question] and not replay["rejected"], "ordering_review_or_gold_replay_changed")
            review_chain = inspector.feedback_request(review_request, run, root)
            review_chain["response"] = inspector.bind(review_response, review["response_sha256"])
        else:
            require(question == original, "multiselect_changed_after_raw_validation")
        provenance[qid] = {"generation": request_chain, "ordering_review": review_chain,
                           "source": inspector.bind(source_path, question["source_sha256"])}
    # Enforce the established exact entire statement-set policy, without a
    # semantic paraphrase gate. Frozen prior entries never import current scores.
    for root in source_rows:
        current = []
        prior = read(run / "prior" / (root + ".json"))
        inspector.bind(run / "prior" / (root + ".json"))
        for kind in generation.TYPES:
            rows = [q for q in mechanical if q["root_id"] == root and q["question_type"] == kind]
            kept, duplicates = v15.dedup.deduplicate(rows, prior, current)
            require(not duplicates and kept == rows, "mechanical_pool_contains_exact_prior_or_batch_duplicate")
            current += kept
    return manifest, questions, provenance, inspector


def _scope(manifest):
    require(manifest.get("solver_model") == assembler.TARGET
            and manifest.get("model") == str(assembler.CHECKPOINT), "exact_current_m_clean_checkpoint_required")
    require(manifest.get("policy_version") == v15.POLICY
            and manifest.get("execution") == "full_only_no_ground_v1"
            and manifest.get("training_started") is False
            and manifest.get("training_promotion") is False, "solver_policy_or_training_scope_invalid")
    require(all(row.get("mode") == "full_context" for row in manifest["requests"]), "ground_call_not_allowed")


def preflight(generation_run, run):
    require(not run.exists(), "immutable_evaluation_run_already_exists")
    manifest, questions, _, inspector = verify_generation(generation_run)
    solver = run / "solver"
    if questions:
        evaluator.preflight(generation_run / "SOLVER_POOL.json", solver,
                            require_ordering_review=True, require_model=True, solver_model=assembler.TARGET)
        _scope(read(solver / "manifest.json"))
    else:
        write(solver / "SKIPPED.json", {"status": "EMPTY_POOL_NO_SOLVER", "candidate_count": 0,
                                      "ground_calls": 0, "training_calls": 0})
    for path in (Path(__file__), Path(assembler.__file__), Path(generation.__file__),
                 Path(v15.__file__), Path(evaluator.__file__), Path(delivery.__file__)):
        inspector.bind(path, snapshot=False)
    result = {"version": VERSION, "status": "PREFLIGHT_PASS", "generation_run": str(generation_run),
        "generation_manifest_sha256": assembler.sha(generation_run / "MANIFEST.json"),
        "roots": manifest["roots"], "solver_model": assembler.TARGET, "checkpoint": str(assembler.CHECKPOINT),
        "policy_version": v15.POLICY, "candidate_count": len(questions),
        "full_calls_planned": sum(18 if q["question_type"] == "multi_select" else 12 for q in questions),
        "ground_calls": 0, "training_started": False, "training_calls": 0,
        "files": [{"path": str(path), "sha256": sha} for path, sha in sorted(inspector.files.items())]}
    write(run / "MANIFEST.json", result)
    return result


def verify(run):
    manifest = read(run / "MANIFEST.json")
    require(manifest["version"] == VERSION and manifest["solver_model"] == assembler.TARGET
            and manifest["checkpoint"] == str(assembler.CHECKPOINT)
            and manifest["policy_version"] == v15.POLICY, "pipeline_scope_changed")
    base.verify_bindings(manifest["files"])
    return manifest


def live(run):
    manifest = verify(run)
    if manifest["candidate_count"] == 0:
        return read(run / "solver/SKIPPED.json")
    solver_manifest = read(run / "solver/manifest.json")
    _scope(solver_manifest)
    # This guard disallows base-only or skipped model identities before live's
    # own audited checkpoint/weight verification and exact venv check.
    assembler.Inspector().model(solver_manifest["solver_identity"])
    evaluator.live(run / "solver")
    return read(run / "solver/COMPLETE.json")


def verify_scores(questions, manifest, records, results):
    """Replay every answer, including nonselections, before exporting any pool."""
    _scope(manifest)
    require(not evaluator.audit_complete_call_set(manifest, records), "solver_call_set_incomplete_or_invalid")
    by_id = {row["candidate_id"]: row for row in results["questions"]}
    require(len(by_id) == len(results["questions"]) == len(questions)
            and set(by_id) == {q["candidate_id"] for q in questions}, "result_candidate_set_changed")
    selected = []
    for question in questions:
        qid, kind = question["candidate_id"], question["question_type"]
        planned = [row for row in manifest["requests"] if row["candidate_id"] == qid]
        variants, slots = (6, 3) if kind == "multi_select" else (12, 1)
        expected = {(f"perm_{i:02d}", slot) for i in range(variants) for slot in range(slots)}
        require(len(planned) == variants * slots and {(r["variant"]["variant_id"], r["slot"]) for r in planned} == expected,
                "wrong_full_variant_budget")
        canonical_variants = {v["variant_id"]: v for v in evaluator._variants_for(question)[0]}
        require(all(row["variant"] == canonical_variants[row["variant"]["variant_id"]] for row in planned),
                "random_gold_blind_permutation_policy_changed")
        require(all(row["seed"] == evaluator.SEED + index for index, row in enumerate(manifest["requests"])),
                "solver_sampling_seed_policy_changed")
        requests = {row["request_id"]: row for row in planned}
        scores, exact = {f"perm_{i:02d}": [] for i in range(variants)}, 0
        for record in (row for row in records if row["candidate_id"] == qid):
            request = requests[record["request_id"]]
            require(record["solver_model"] == assembler.TARGET and record["model_path"] == str(assembler.CHECKPOINT),
                    "response_solver_checkpoint_mismatch")
            require(record["data_errors"] == [] and record["metadata"]["finish_reason"] in {"stop", "eos"},
                    "invalid_solver_answer")
            for key in ("temperature", "top_p", "max_tokens", "structured_output_api", "structured_output_version"):
                require(record["metadata"][key] == manifest["decoder_contract"][key], "response_decoder_mismatch")
            value, is_exact = assembler.score(question, request["variant"], record["raw_output"])
            require(record["score"]["scored"] is True
                    and Fraction(record["score"]["partial_score_fraction"]) == value
                    and record["score"]["exact_match"] is is_exact, "stored_score_mismatch")
            scores[record["variant_id"]].append(value)
            exact += is_exact
        expected_result = evaluator.select(scores, list(scores), exact_rate=Fraction(exact, variants * slots), data_errors=[])
        expected_result.update(candidate_id=qid, root_id=question["root_id"], question_type=kind,
                               expected_full_calls=variants * slots, actual_full_calls=variants * slots)
        require(by_id[qid] == expected_result, "solver_selection_replay_changed")
        if expected_result["selected_by_policy"]:
            q = {**deepcopy(question), "training_selection": deepcopy(expected_result),
                 "training_target": "complete_canonical_gold", "loss_weight_from_partial_score": False}
            assembler.question_contract(q)
            assembler.verify_selection(q, manifest, records, expected_result)
            selected.append(q)
    require(results["selected_questions"] == [q["candidate_id"] for q in selected], "selected_id_list_changed")
    return selected


def pool_rows(entries, output):
    """Use canonical dedup while preserving raw questions, full Gold and source."""
    kept, duplicates = assembler.deduplicate(entries)
    rows = []
    for entry in kept:
        q = entry["question"]
        field = "options" if q["question_type"] == "multi_select" else "items"
        ids = {label: assembler.content_id(text) for label, text in q[field].items()}
        rows.append({"canonical_question_id": assembler.canonical_question_id(q), "candidate_id": q["candidate_id"],
            "root_id": q["root_id"], "question_type": q["question_type"], "question": q["question"],
            field: deepcopy(q[field]), "gold": deepcopy(q["gold"]),
            "canonical_content_id_by_label": ids, "complete_gold_content_ids": [ids[label] for label in q["gold"]],
            "source_sha256": q["source_sha256"], "source_path": str(output / "contexts" / (q["source_sha256"] + ".json")),
            "full_context": deepcopy(entry["source"]), "original_selected_question": deepcopy(q),
            "origin": "new", "provenance_chain": entry["provenance_chain"],
            "training_target": "complete_canonical_gold", "canonical_exposure_units": 1,
            "loss_weight_from_partial_score": False, "evaluated_permutations_are_separate_training_examples": False})
    return rows, duplicates


def finalize(run):
    outer = verify(run)
    target = run / "training_pool"
    require(not target.exists() and not (run / "delivery").exists(), "immutable_pool_or_delivery_already_exists")
    generation_run = Path(outer["generation_run"])
    _, questions, provenance, inspector = verify_generation(generation_run)
    selected, solver_calls = [], 0
    if questions:
        solver = run / "solver"
        require((solver / "COMPLETE.json").is_file(), "solver_not_complete")
        if not (solver / "RESULTS.json").exists():
            evaluator.report(solver)
        solver_manifest, records, _ = inspector.solver(solver)
        results = read(solver / "RESULTS.json")
        selected = verify_scores(questions, solver_manifest, records, results)
        solver_calls = len(records)
        delivery.deliver(generation_run, solver, run / "delivery")
        require(read(run / "delivery/SELECTED_QUESTIONS.json")["questions"] == selected, "delivery_changed_original_question_or_gold")
    entries = [{"question": q, "source": read(q["source_path"]), "is_new": True,
                "provenance": {**provenance[q["candidate_id"]], "input_role": "new",
                    "solver_manifest": inspector.bind(run / "solver/manifest.json"),
                    "solver_results": inspector.bind(run / "solver/RESULTS.json"),
                    "solver_identity": inspector.bind(run / "solver/solver_identity.json")}} for q in selected]
    rows, duplicates = pool_rows(entries, target)
    # All provenance and selection checks finish before writing pool artifacts.
    target.mkdir(parents=True, exist_ok=False)
    for q in selected:
        destination = target / "contexts" / (q["source_sha256"] + ".json")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(q["source_path"], destination)
        require(assembler.sha(destination) == q["source_sha256"], "pool_context_copy_changed")
    for path in sorted(inspector.snapshots):
        destination = target / "provenance" / (inspector.files[path] + "-" + path.name)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        require(assembler.sha(destination) == inspector.files[path], "pool_provenance_copy_changed")
    with (target / "TRAINING_POOL.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write(target / "SELECTED_QUESTIONS.json", {"questions": [r["original_selected_question"] for r in rows],
        "training_started": False, "training_target": "complete_canonical_gold", "loss_weight_from_partial_score": False})
    report = {"version": VERSION, "status": "IMMUTABLE_NEW_POOL_PREPARED_NOT_TRAINED", "evaluated": len(questions),
        "selected_before_exact_dedup": len(selected), "new_unique_added": len(rows), "duplicates": duplicates,
        "by_type": dict(Counter(row["question_type"] for row in rows)),
        "by_root": dict(Counter(row["root_id"] for row in rows)), "full_solver_calls": solver_calls,
        "ground_calls": 0, "training_calls": 0, "training_started": False,
        "solver_model": assembler.TARGET, "solver_checkpoint": str(assembler.CHECKPOINT),
        "selection_rule": "complete_valid_full_set and 0 < full_partial_mean < 1",
        "full_slots": {"multi_select": "6 random permutations x 3 full rollouts", "ordering": "12 random permutations x 1 full rollout"},
        "full_context_in_every_row": True, "complete_gold_in_every_row": True,
        "historical_pools_modified": False, "semantic_gold_certified": False, "semantic_novelty_certified": False}
    write(target / "REPORT.json", report)
    verify(run)
    for path in inspector.snapshots:
        require(assembler.sha(path) == inspector.files[path], "input_changed_during_pool_assembly")
    write(target / "MANIFEST.json", {"version": VERSION, "status": "VERIFIED_IMMUTABLE_NEW_POOL",
        "training_started": False, "pipeline_manifest": inspector.bind(run / "MANIFEST.json"),
        "files": [{"path": str(p.relative_to(target)), "sha256": assembler.sha(p)}
                  for p in sorted(target.rglob("*")) if p.is_file()],
        "original_bindings": [{"path": str(p), "sha256": sha} for p, sha in sorted(inspector.files.items())],
        "verification_scope": "generation_raw_content_and_full_source_replay_ordering_review_and_exact_checkpoint_complete_solver_scores"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "verify", "live", "finalize"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--generation", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.is_relative_to(HERE / "runs"), "evaluation_run_must_be_in_v26_qa_runs")
    if args.action == "preflight":
        require(args.generation is not None, "generation_run_required")
        result = preflight(args.generation.resolve(), run)
    else:
        result = {"verify": verify, "live": live, "finalize": finalize}[args.action](run)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
