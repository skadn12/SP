"""Independently audit the fixed-source supplementary question experiment.

No authoring, model inference, repairs, or edits to experiment inputs occur.
Use --stage prepared before calls, generation after API calls, or complete
after delivery. Complete also recomputes all v15 scores and provenance checks.
Reports go to stdout unless --output is given (exclusive creation only).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
QA = D / "h4_feedback_training_pool_v15_qa"
VERSION = "h4_r02_existing_source_yield_v16"
EXPANSION = D / "h4_feedback_training_pool_v15_expansion/runs/end_to_end_5roots_01"
PARENTS = [QA / "runs/end_to_end_5roots_01", QA / "runs/end_to_end_5roots_01_429_retry"]
POOL = D / "h4_feedback_training_pool_v15/runs/end_to_end_5roots_01_final_training_pool"
SELECTED = [D / "h4_r02_yield_pilot_v12/runs/contract_smoke_2roots_01_qa/SELECTED_QUESTIONS.json",
            D / "h4_r02_question_contract_v14/runs/frozen_v12_2roots_01/SELECTED_QUESTIONS.json",
            *[p / "SELECTED_QUESTIONS.json" for p in PARENTS]]
PUBLIC_PRIOR_KEYS = {"candidate_id", "question_type", "question", "facts", "dedup_action"}
ACTIVE = {"A": ["cc-train-episode-210046", "cc-train-episode-138343"],
          "B": ["cc-train-episode-104814", "cc-train-episode-135466", "cc-train-episode-138343"]}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def facts(row):
    values = row.get("facts") or row.get("options") or row.get("items")
    if isinstance(values, dict):
        values = list(values.values())
    require(isinstance(values, list) and values and all(isinstance(x, str) for x in values),
            "missing_statement_text:" + str(row.get("candidate_id")))
    return values


def signature(row):
    return tuple(sorted(" ".join(text.casefold().split()) for text in facts(row)))


def key(row):
    return row["root_id"], row["question_type"], signature(row)


def project_prior(rows):
    """Independent reconstruction of the five-field public dedup inventory."""
    result = {}
    for row in rows:
        if not row.get("facts"):
            continue
        k = row.get("question_type"), signature(row)
        public = {name: row.get(name) for name in ("candidate_id", "question_type", "question", "facts")}
        public["dedup_action"] = "avoid_already_selected" if row.get("hard_dedup") else "warning_only"
        if k not in result or row.get("hard_dedup"):
            result[k] = public
    return list(result.values())


def parse_generator_input(text, suffix):
    """Strip the exact frozen feedback suffix before parsing INPUT_JSON."""
    require(bool(suffix) and text.endswith(suffix), "request_missing_exact_frozen_feedback")
    prefix = text[:-len(suffix)]
    require("\nINPUT_JSON\n" in prefix, "generator_input_marker_missing")
    instructions, serialized = prefix.split("\nINPUT_JSON\n", 1)
    data, end = json.JSONDecoder().raw_decode(serialized.lstrip())
    require(not serialized.lstrip()[end:].strip(), "unexpected_text_after_input_json")
    require(isinstance(data, dict), "generator_input_not_object")
    return instructions, data


class Audit:
    def __init__(self):
        self.checked = {}
        self.counts = Counter()

    def binding(self, path, expected):
        path = Path(path).resolve()
        if path not in self.checked:
            self.checked[path] = sha(path)
        require(self.checked[path] == expected, "changed_frozen_file:" + str(path))

    def bindings(self, rows):
        for row in rows:
            self.binding(row["path"], row["sha256"])

    def existing_pool(self):
        manifest, report = read(POOL / "MANIFEST.json"), read(POOL / "REPORT.json")
        require(manifest["immutable_write_once"] is True and manifest["training_started"] is False
                and report["training_started"] is False and manifest["report"] == report,
                "existing_pool_manifest_report_mismatch")
        require(report["total_unique"] == 25, "existing_pool_not_25")
        listed = set()
        for row in manifest["files"]:
            path = (POOL / row["path"]).resolve()
            require(path.is_relative_to(POOL.resolve()) and row["path"] not in listed,
                    "invalid_or_duplicate_existing_pool_path")
            listed.add(row["path"])
            self.binding(path, row["sha256"])
        actual = {str(p.relative_to(POOL)) for p in POOL.rglob("*") if p.is_file()}
        require(actual == listed | {"MANIFEST.json"}, "existing_pool_unbound_or_missing_file")
        require([row["path"] for row in manifest["inputs"]] == list(map(str, SELECTED)),
                "existing_pool_inputs_changed")
        old = []
        for item in manifest["inputs"]:
            self.binding(item["path"], item["sha256"])
            questions = read(item["path"])["questions"]
            require(len(questions) == item["selected_count"], "existing_input_count_changed")
            old.extend(questions)
        require(len(old) == len({q["candidate_id"] for q in old}) == len({key(q) for q in old}) == 25,
                "existing_selected_ids_or_statement_sets_not_25_unique")
        rows = [json.loads(line) for line in (POOL / "TRAINING_POOL.jsonl").read_text().splitlines() if line.strip()]
        indexed = {q["candidate_id"]: q for q in old}
        require(len(rows) == 25 and {r["candidate_id"] for r in rows} == set(indexed), "existing_pool_row_ids_changed")
        for row in rows:
            require(row["original_selected_question"] == indexed[row["candidate_id"]], "existing_original_question_changed")
            require(row["canonical_exposure_units"] == 1
                    and row["evaluated_permutations_are_separate_training_examples"] is False
                    and row["loss_weight_from_partial_score"] is False,
                    "existing_pool_weight_or_target_changed")
        self.counts["existing_pool_files_verified"] = len(listed)
        self.counts["existing_selected_questions"] = len(old)
        return old

    def prepared(self, run):
        spec, manifest = read(run / "SUPPLEMENT_SPEC.json"), read(run / "MANIFEST.json")
        require(spec["version"] == manifest["supplement_version"] == VERSION, "wrong_supplement_version")
        expected = {"source_reuse_only": True, "new_dialogue_turns": 0, "new_dataset_events": 0,
                    "existing_selected_questions": 25, "active_branches": 5,
                    "additional_slots_per_active_branch_type": 4, "active_candidate_slots": 40,
                    "active_http_max": 40, "active_solver_full_max": 600,
                    "minimum_seconds_after_call_completion": 35, "automatic_transport_retries": 0,
                    "generator_model": "Qwen/Qwen3.6-35B-A3B", "reviewer_model": "Qwen/Qwen3.6-35B-A3B",
                    "solver_model": "M_clean_policy", "policy_version": "v15_full_only_random_mcq6x3_order12x1_v1",
                    "feedback_round": "previous_training_round_only", "current_solver_outputs_in_prompts": False,
                    "semantic_novelty_certified": False, "training_started": False}
        for name, value in expected.items():
            require(spec[name] == value, "supplement_scope_changed:" + name)
        require(spec["active_roots_by_branch"] == ACTIVE and spec["expansion_run"] == str(EXPANSION)
                and manifest["expansion_run"] == str(EXPANSION) and spec["existing_pool"] == str(POOL)
                and spec["existing_selected_exports"] == list(map(str, SELECTED))
                and spec["parent_question_runs"] == list(map(str, PARENTS)), "supplement_sources_changed")
        self.bindings(manifest["files"])
        bound = {str(Path(x["path"]).resolve()): x["sha256"] for x in manifest["files"]}
        for path in (HERE / "supplement.py", HERE / "POLICY.md", run / "SUPPLEMENT_SPEC.json",
                     POOL / "MANIFEST.json", POOL / "REPORT.json"):
            require(str(path.resolve()) in bound, "required_supplement_binding_missing:" + str(path))
        historical_code = {}
        for parent in PARENTS:
            require(read(parent / "AUDIT.json")["status"] == "PASS", "parent_audit_not_pass")
            for x in read(parent / "MANIFEST.json")["files"]:
                path = Path(x["path"])
                if path.suffix == ".py":
                    self.binding(path, x["sha256"])
                    historical_code[str(path.resolve())] = x["sha256"]
        for path in (QA / "qa.py", QA / "generate.py", QA / "dedup.py", QA / "deliver.py",
                     QA / "multiselect_gate.py", QA / "ordering_gate.py"):
            name = str(path.resolve())
            require(name in historical_code and bound.get(name) == historical_code[name],
                    "legacy_code_not_bound_to_historical_version:" + name)
        self.counts["historical_python_files_verified"] = len(historical_code)
        old = self.existing_pool()
        expansion_lock = read(EXPANSION / "feedback/source_lock.json")
        for branch in "AB":
            generation = run / branch / "generation"
            gm = read(generation / "MANIFEST.json")
            self.bindings(gm["files"])
            require(gm["candidate_slots_per_type_per_root"] == 4 and gm["automatic_transport_retries"] == 0,
                    "generation_quota_or_retry_changed")
            require([r["root_id"] for r in gm["source_rows"] if r["accepted_scenes"]] == ACTIVE[branch],
                    "active_source_inventory_changed")
            self.binding(generation / "feedback/source_lock.json", expansion_lock["v4_source_lock_sha256"])
            self.binding(generation / "feedback/COMPILED_FEEDBACK.json", expansion_lock["v4_manifest_sha256"])
            lock = read(generation / "feedback/source_lock.json")
            require(lock == expansion_lock["historical_source_lock"]
                    and lock["merge_current_generation_feedback"] is False
                    and lock["automatic_latest_run_selection"] is False, "feedback_not_previous_frozen_round")
            for row in gm["source_rows"]:
                root = row["root_id"]
                original = EXPANSION / "roots" / root / branch / "cumulative_dialogue.json"
                self.binding(original, row["source_sha256"])
                self.binding(row["source_path"], row["source_sha256"])
                for parent in PARENTS:
                    parent_path = parent / branch / "generation/sources" / (root + ".json")
                    require(Path(row["source_path"]).read_bytes() == parent_path.read_bytes(),
                            "source_not_identical_to_previous_question_batch")
                    suffix = root + "/writer__M_clean_policy_feedback.txt"
                    require((generation / "feedback" / suffix).read_bytes()
                            == (parent / branch / "generation/feedback" / suffix).read_bytes(),
                            "writer_feedback_differs_from_previous_batch")
                prior = read(generation / "prior" / (root + ".json"))
                hard = {q["candidate_id"]: q for q in prior if q.get("hard_dedup")}
                for question in old:
                    if question["root_id"] == root:
                        require(question["candidate_id"] in hard
                                and facts(hard[question["candidate_id"]]) == facts(question),
                                "selected_question_missing_or_changed_in_frozen_prior")
                self.counts["identical_branch_sources_verified"] += 1
        return spec, old

    def generation(self, run, spec, old):
        done = read(run / "GENERATION_COMPLETE.json")
        require(done["version"] == VERSION and done["new_dialogue_turns"] == 0,
                "generation_completion_scope_changed")
        timings, initial = [], Counter()
        for branch in "AB":
            generation = run / branch / "generation"
            gm = read(generation / "MANIFEST.json")
            directories = sorted((generation / "requests").iterdir())
            for directory in directories:
                require(directory.is_dir(), "unexpected_request_entry")
                request, proof = read(directory / "REQUEST.json"), read(directory / "PROOF.json")
                root = proof["root_id"]
                require(root in ACTIVE[branch], "api_called_for_failed_source_branch")
                suffix = (generation / "feedback" / root / "writer__M_clean_policy_feedback.txt").read_text()
                text = request["messages"][0]["content"]
                require(text.endswith(suffix) and proof["payload_sha256"] == digest(request),
                        "request_or_feedback_hash_changed")
                require(request["model"] == spec["generator_model"] and request["enable_thinking"] is False,
                        "provider_model_or_thinking_changed")
                receipt = proof["feedback_receipt"]
                require(receipt["feedback_appended"] is True and receipt["suffix_sha256"] == digest(suffix.encode())
                        and receipt["request_user_text_sha256"] == digest(text.encode())
                        and receipt["source_lock_sha256"] == sha(generation / "feedback/source_lock.json"),
                        "request_feedback_receipt_changed")
                if proof["role"] == "question_generator":
                    instructions, data = parse_generator_input(text, suffix)
                    require(spec["novelty_instruction"] in instructions, "supplement_instruction_missing_from_actual_request")
                    require(data["root_id"] == root and data["branch"] == branch
                            and data["source_turns"] == read(generation / "sources" / (root + ".json"))["turns"],
                            "actual_generation_source_changed")
                    allowed_context = {"root_id", "route", "branch", "candidate_ids", "source_turns",
                                       "previous_candidates_dedup_only", "priority_new_scene_refs",
                                       "prioritize_new_branch_content", "must_use_at_least_one_new_branch_evidence_turn",
                                       "mechanical_repair"}
                    require(set(data) <= allowed_context, "unexpected_generation_context_fields")
                    prior = data["previous_candidates_dedup_only"]
                    require(isinstance(prior, list) and all(isinstance(q, dict) and set(q) == PUBLIC_PRIOR_KEYS for q in prior),
                            "prior_contains_scores_gold_solver_answers_or_nonpublic_fields")
                    public_by_key = {(q["question_type"], signature(q)): q for q in prior}
                    require(len(public_by_key) == len(prior), "public_prior_contains_duplicate_fact_sets")
                    frozen = project_prior(read(generation / "prior" / (root + ".json")))
                    for row in frozen:
                        require(public_by_key.get((row["question_type"], signature(row))) == row,
                                "frozen_prior_missing_or_changed_in_actual_request")
                    for question in old:
                        if question["root_id"] != root:
                            continue
                        row = public_by_key.get((question["question_type"], signature(question)))
                        require(row is not None and row["candidate_id"] == question["candidate_id"]
                                and row["question"] == question["question"] and row["facts"] == facts(question)
                                and row["dedup_action"] == "avoid_already_selected",
                                "selected_question_not_covered_by_actual_public_prior")
                    require(data["candidate_ids"] == proof["candidate_ids"]
                            and 1 <= len(data["candidate_ids"]) <= 4, "actual_candidate_slots_changed")
                    if "mechanical_repair" not in data:
                        initial[(branch, root, proof["question_type"])] += 1
                        require(len(data["candidate_ids"]) == 4, "initial_request_not_four_slots")
                    self.counts["actual_generator_requests_with_verified_prior"] += 1
                else:
                    require(proof["role"] == "ordering_reviewer", "unexpected_api_role")
                    public = json.loads(text[:-len(suffix)].split("\nDATA:\n", 1)[1])
                    require(set(public) == {"root_id", "source_turns", "candidates"}, "review_context_nonpublic_fields")
                    require(all(set(q) == {"candidate_id", "root_id", "question_type", "chronology_kind", "question", "items"}
                                for q in public["candidates"]), "ordering_review_gold_or_evidence_leak")
                timing = read(directory / "TIMING.json")
                require(Path(timing["request_directory"]).resolve() == directory.resolve()
                        and timing["payload_sha256"] == digest(request)
                        and timing["http_attempted"] == (directory / "HTTP_ATTEMPT.json").exists(),
                        "timing_request_binding_changed")
                timings.append(timing)
                if timing["http_attempted"]:
                    attempt = read(directory / "HTTP_ATTEMPT.json")
                    require(attempt["transport_retries"] == 0 and attempt["payload_sha256"] == digest(request),
                            "http_payload_or_automatic_retry_changed")
                    self.counts["http_attempts"] += 1
        expected_initial = {(branch, root, kind) for branch, roots in ACTIVE.items() for root in roots
                            for kind in ("multi_select", "ordering")}
        require(set(initial) == expected_initial and all(n == 1 for n in initial.values()),
                "initial_request_repeated_or_missing")
        timings.sort(key=lambda row: row["call_sequence"])
        require(done["timings"] == timings and done["requests"] == len(timings) <= 40,
                "generation_global_budget_or_timing_inventory_changed")
        for index, row in enumerate(timings):
            require(row["call_sequence"] == index + 1 and row["elapsed_seconds"] >= 0,
                    "timing_sequence_or_duration_invalid")
            start = datetime.fromisoformat(row["started_at_utc"])
            end = datetime.fromisoformat(row["finished_at_utc"])
            require(end >= start and abs((end - start).total_seconds() - row["elapsed_seconds"]) < 1,
                    "timing_wallclock_and_monotonic_disagree")
            if index == 0:
                require(row["interval_since_previous_call_completion_seconds"] is None, "first_call_interval_not_null")
            else:
                elapsed = row["interval_since_previous_call_completion_seconds"]
                previous_end = datetime.fromisoformat(timings[index - 1]["finished_at_utc"])
                require(elapsed >= 35 and (start - previous_end).total_seconds() >= 34.99
                        and abs((start - previous_end).total_seconds() - elapsed) < 1,
                        "calls_not_sequential_or_35_second_gap_missing")
        mechanical, ready = [], []
        for branch in "AB":
            generation = run / branch / "generation"
            mechanical.extend(read(generation / "MECHANICAL_POOL.json")["questions"])
            ready.extend(read(generation / "SOLVER_POOL.json")["questions"])
        require(len(mechanical) <= 40 and len({q["candidate_id"] for q in mechanical}) == len(mechanical),
                "new_candidate_budget_or_id_uniqueness_changed")
        planned_full = sum(18 if q["question_type"] == "multi_select" else 12 for q in ready)
        require(planned_full <= 600, "actual_ready_pool_exceeds_full_budget")
        old_keys = {key(q) for q in old}
        require(all(key(q) not in old_keys for q in mechanical), "old_selected_exact_duplicate_reached_mechanical_pool")
        self.counts.update({"initial_generator_requests": len(initial), "paced_api_requests": len(timings),
                            "mechanical_candidates": len(mechanical), "ready_candidates": len(ready),
                            "planned_full_calls": planned_full})

    def complete(self, run, old):
        module_spec = importlib.util.spec_from_file_location("v16_independent_v15_auditor", QA / "audit_run.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        result = module.Audit().run(run)
        require(result["status"] == "PASS", "base_independent_v15_audit_failed")
        selected = read(run / "SELECTED_QUESTIONS.json")["questions"]
        old_keys, new_keys = {key(q) for q in old}, [key(q) for q in selected]
        require(len(set(new_keys)) == len(new_keys) and not old_keys.intersection(new_keys),
                "new_selected_exact_duplicates_with_old_pool_or_same_batch")
        require(result["counts"].get("full_solver_calls_independently_recomputed", 0) <= 600,
                "completed_solver_budget_exceeded")
        self.counts["new_selected_exact_unique"] = len(selected)
        self.counts["combined_existing_plus_new_exact_unique"] = 25 + len(selected)
        self.counts["full_solver_calls_independently_recomputed"] = result["counts"].get("full_solver_calls_independently_recomputed", 0)
        return {"status": result["status"], "audit_script_sha256": sha(QA / "audit_run.py"),
                "frozen_files_checked": result["frozen_files_checked"], "counts": result["counts"],
                "selected_complete_gold_count": result["selected_complete_gold_count"],
                "all_scores_and_means_independently_recomputed": True,
                "complete_planned_call_set": True}

    def run(self, run, stage):
        spec, old = self.prepared(run)
        if stage in {"generation", "complete"}:
            self.generation(run, spec, old)
        base_audit = self.complete(run, old) if stage == "complete" else None
        outputs = [run / "MANIFEST.json", run / "SUPPLEMENT_SPEC.json"]
        if stage in {"generation", "complete"}:
            outputs += [run / "GENERATION_COMPLETE.json"]
            for branch in "AB":
                outputs += [run / branch / "generation" / name for name in
                            ("MANIFEST.json", "REPORT.json", "MECHANICAL_POOL.json", "SOLVER_POOL.json")]
        if stage == "complete":
            outputs += [run / "REPORT.json", run / "SELECTED_QUESTIONS.json"]
            for branch in "AB":
                solver = run / branch / "solver"
                names = ("SKIPPED.json",) if (solver / "SKIPPED.json").exists() else (
                    "manifest.json", "RESULTS.json", "COMPLETE.json")
                outputs += [solver / name for name in names]
                outputs += [run / branch / "delivery" / name for name in ("REPORT.json", "SELECTED_QUESTIONS.json")]
        return {"status": "PASS", "stage": stage, "version": VERSION, "run": str(run),
                "audited_at_utc": datetime.now(timezone.utc).isoformat(), "audit_script_sha256": sha(Path(__file__)),
                "result_bindings": [{"path": str(path.resolve()), "sha256": sha(path)} for path in outputs],
                "bound_files_independently_checked": len(self.checked), "counts": dict(self.counts),
                "unchanged_dialogue_sources": True, "frozen_previous_round_feedback": True,
                "prior_payloads_checked": stage != "prepared", "existing_pool_25_frozen": True,
                "new_dialogue_turns": 0, "new_dataset_events": 0, "ground_calls": 0, "training_calls": 0,
                "semantic_gold_certified": False, "semantic_novelty_certified": False,
                "base_v15_independent_audit": base_audit,
                "limitation": "Exact text-set novelty and execution integrity only; no semantic Gold or novelty certification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepared", "generation", "complete"), default="complete")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        require(not args.output.exists(), "audit_output_already_exists")
    audit = Audit()
    try:
        result = audit.run(args.run.resolve(), args.stage)
    except Exception as error:
        result = {"status": "FAIL", "stage": args.stage, "version": VERSION, "run": str(args.run.resolve()),
                  "audit_script_sha256": sha(Path(__file__)), "error": type(error).__name__ + ":" + str(error),
                  "bound_files_independently_checked": len(audit.checked), "counts": dict(audit.counts),
                  "model_calls": 0, "training_calls": 0}
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output)
    print(output, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
