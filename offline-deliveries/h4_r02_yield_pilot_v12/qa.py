"""Bounded question generation and full-only evaluation of independent v12 branches.

The CLI keeps A/B in separate pools because the existing evaluation contract
requires exactly one frozen source per root. No phase starts training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
OLD = DELIVERIES / "h4_r02_qa_batch_v1"
VERSION = "h4_r02_yield_pilot_v12_qa"
POLICY = "v12_full_only_random_mcq6x3_order12x1_v1"
MODEL = "Qwen/Qwen3.6-35B-A3B"
TARGET = "M_clean_policy"
ROUTES = {"A": "old_event_continuation", "B": "dataset_new_event"}
TYPES = ("multi_select", "ordering")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load("v12_qa_legacy_generation", OLD / "generate.py")
base = legacy.base
sys.path.insert(0, str(OLD / "evaluation"))
evaluator = load("v12_qa_legacy_evaluator", OLD / "evaluation/run_solver.py")
delivery = load("v12_qa_legacy_delivery", OLD / "deliver.py")


def now():
    return evaluator.now()


def require(value, reason):
    base.require(value, reason)


def source_for_branch(expansion, root, letter):
    """Verify a completed branch against its frozen prefix and accepted scene count."""
    require(letter in ROUTES, "unknown_branch")
    frozen = base.read(expansion / "sources" / f"{root}.json")
    path = expansion / "roots" / root / letter / "cumulative_dialogue.json"
    source = base.read(path)
    result_path = path.with_name("RESULT.json")
    result = base.read(result_path)
    require(source.get("root_id") == root and source.get("route") == ROUTES[letter], "branch_source_identity")
    require(source.get("branch_independent") is True, "branch_not_independent")
    require(source["turns"][:len(frozen["turns"])] == frozen["turns"], "branch_frozen_prefix_changed")
    suffix = source["turns"][len(frozen["turns"]):]
    require(all(t.get("turn_ref", "").startswith(f"R02:{letter}:") for t in suffix), "other_branch_in_qa_source")
    require(len({t["turn_ref"] for t in source["turns"]}) == len(source["turns"]), "duplicate_source_ref")
    require(result["root_id"] == root and result["route"] == ROUTES[letter], "branch_result_identity")
    require(result["added_turns"] == len(suffix), "branch_result_turn_count")
    require(result["accepted_scenes"] == len(result["scene_metadata"]), "branch_result_scene_count")
    require(result["output_history_sha256"] == base.digest(source["turns"]), "branch_history_hash_mismatch")
    require(bool(suffix) == (result["accepted_scenes"] > 0), "accepted_scene_history_mismatch")
    return path, source, result_path, result


def prior_candidates(root):
    """Freeze explicit historical warning inputs; failed claims are not forever banned."""
    history_path = legacy.FEEDBACK / root / "candidate_history.json"
    histories = base.read(history_path)
    rows = []
    for row in histories:
        q = row.get("historical_question") or {}
        texts = q.get("options") or q.get("items") or {}
        rows.append({"candidate_id": row["candidate_id"], "question_type": q.get("question_type"),
                     "question": q.get("question"), "facts": list(texts.values())})
    files = [history_path]
    for pool_path in (
        DELIVERIES / "h4_groundless_random_feedback_v1/evaluation/SOLVER_POOL_V3.json",
        OLD / "runs/pilot_02/SOLVER_POOL.json",
        OLD / "runs/pilot_02_transport_retry_01/SOLVER_POOL.json",
    ):
        if pool_path.exists():
            pool = base.read(pool_path)
            candidates = [q for q in pool["questions"] if q["root_id"] == root]
            rows += legacy.compact_questions(candidates)
            files.append(pool_path)
    return rows, files


def budget(root_count, slots):
    require(type(slots) is int and 1 <= slots <= 8, "slots_outside_1_8")
    return {"candidate_slots": root_count * len(TYPES) * slots,
            "initial_generation_requests": root_count * len(TYPES),
            "mechanical_repair_requests_max": root_count * len(TYPES),
            "ordering_review_requests_max": root_count * slots,
            "http_attempts_max": root_count * (4 + slots),
            "solver_full_calls_max": root_count * slots * (18 + 12)}


def prepare(expansion, run, *, slots=4):
    require(not run.exists(), "qa_run_already_exists")
    em = base.read(expansion / "MANIFEST.json")
    require((expansion / "REPORT.json").is_file(), "expansion_not_completed")
    require(em["target_solver_model"] == TARGET, "expansion_solver_target_mismatch")
    expansion_feedback = base.read(expansion / "feedback/COMPILED_FEEDBACK.json")
    expansion_lock_path = expansion / "feedback/source_lock.json"
    expansion_lock = base.read(expansion_lock_path)
    require(expansion_feedback["target_solver_model"] == TARGET, "expansion_feedback_target_mismatch")
    require(base.digest(expansion_lock_path.read_bytes()) == expansion_feedback["source_lock_sha256"],
            "expansion_feedback_lock_changed")
    require(base.digest((legacy.FEEDBACK / "source_lock.json").read_bytes()) == expansion_lock["v4_source_lock_sha256"],
            "qa_and_expansion_feedback_rounds_differ")
    require(expansion_lock["historical_source_lock"]["model_identity_source"]["sha256"] == evaluator.TRAINED_AUDIT_SHA256,
            "qa_feedback_and_solver_checkpoint_differ")
    roots = em["roots"]
    require(roots and len(roots) == len(set(roots)), "invalid_roots")
    require(set(em["selected_routes"]) == set(ROUTES.values()), "both_routes_required")
    branch_inputs = {}
    for letter in ROUTES:
        branch_inputs[letter] = [source_for_branch(expansion, root, letter) for root in roots]
    profile, _, _, _, _ = base.transport()
    provider = profile()
    frozen = [Path(__file__), OLD / "generate.py", OLD / "ordering_gate.py", OLD / "deliver.py",
              *sorted((OLD / "evaluation").glob("*.py")), Path(legacy.feedback.__file__),
              Path(base.__file__), Path(legacy.multi.__file__), legacy.ordering_gate.PROMPTS_PATH,
              expansion / "MANIFEST.json", expansion / "REPORT.json",
              expansion / "feedback/COMPILED_FEEDBACK.json", expansion_lock_path]
    for letter, route in ROUTES.items():
        generation = run / letter / "generation"
        for path in sorted(legacy.FEEDBACK.rglob("*")):
            if path.is_file():
                target = generation / "feedback" / path.relative_to(legacy.FEEDBACK)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                frozen += [path, target]
        source_rows = []
        for root, (source_path, source, result_path, result) in zip(roots, branch_inputs[letter]):
            target = generation / "sources" / f"{root}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            prior, prior_files = prior_candidates(root)
            base.write(generation / "prior" / f"{root}.json", prior)
            legacy.feedback.build_feedback_suffix(generation / "feedback", root, "writer", TARGET)
            frozen += [source_path, result_path, target, generation / "prior" / f"{root}.json", *prior_files]
            source_rows.append({"root_id": root, "source_path": str(target.resolve()),
                "source_sha256": base.digest(target.read_bytes()), "accepted_scenes": result["accepted_scenes"],
                "added_turns": result["added_turns"], "source_round": "P+R01+R02_" + letter,
                "route": route, "branch": letter,
                "qa_status": "READY" if result["accepted_scenes"] else "EXPANSION_SHORTFALL_NO_QA"})
        base.write(generation / "provider_profile.json", provider)
        frozen.append(generation / "provider_profile.json")
        manifest = {"version": VERSION, "roots": roots, "route": route, "branch": letter,
            "solver_model": TARGET, "generator_model": MODEL, "reviewer_model": MODEL,
            "question_types": list(TYPES), "candidate_slots_per_type_per_root": slots,
            "budget": budget(len(roots), slots), "generation_output_budget": 8192,
            "ordering_review_output_budget": 4096, "automatic_transport_retries": 0,
            "semantic_multi_review": False, "ordering_independent_qwen_review": True,
            "codex_question_edits": False, "feedback_round": "previous_training_round_only",
            "source_rows": source_rows, "ground_calls": 0, "training_calls": 0,
            "new_branch_evidence_required": True,
            "zero_scene_branch_policy": "EXPANSION_SHORTFALL_NO_QA",
            "files": base.bindings(frozen)}
        base.write(generation / "MANIFEST.json", manifest)
    experiment = {"version": VERSION, "expansion_run": str(expansion), "roots": roots,
        "branches": ROUTES, "solver_model": TARGET, "generator_model": MODEL,
        "slots_per_root_route_type": slots, "budget_per_route": budget(len(roots), slots),
        "candidate_slots_total": 2 * budget(len(roots), slots)["candidate_slots"],
        "max_http_attempts": 2 * budget(len(roots), slots)["http_attempts_max"],
        "max_solver_full_calls": 2 * budget(len(roots), slots)["solver_full_calls_max"],
        "full_slots": {"multi_select": "6 random permutations x 3 full rollouts",
                       "ordering": "12 random permutations x 1 full rollout"},
        "selection_rule": "complete_valid_full_set and 0 < full_partial_mean < 1",
        "no_cross_branch_context": True, "selected_target_per_root": [4, 8],
        "selected_target_is_observation_not_filter_or_guarantee": True,
        "semantic_dedup_certified": False, "training_started": False,
        "prepared_at": now(), "files": base.bindings(frozen)}
    base.write(run / "MANIFEST.json", experiment)
    return experiment


def signature(candidate):
    values = candidate.get("options") or candidate.get("items") or {}
    return tuple(sorted(" ".join(text.casefold().split()) for text in values.values()))


def make_generation_payload(generation, root, qtype, context):
    instructions = legacy.COMMON.replace(
        "It includes P, R01, and recovered R02 A/B turns.",
        "It includes P, R01, and only this independent R02 branch's accepted turns.")
    return legacy.payload_with_feedback(generation, root, instructions +
        (legacy.MULTI if qtype == "multi_select" else legacy.ORDERING), context, 8192)


def generate_route(run, letter):
    generation = run / letter / "generation"
    manifest = base.read(generation / "MANIFEST.json")
    base.verify_bindings(manifest["files"])
    require(not (generation / "STARTED.json").exists(), "generation_already_started_no_replay")
    legacy.MAX_HTTP = manifest["budget"]["http_attempts_max"]
    base.write(generation / "STARTED.json", {"started_at": now(), "http_max": legacy.MAX_HTTP})
    caller = legacy.Caller(generation)
    ready, mechanical, outcomes = [], [], []
    for root in manifest["roots"]:
        source_row = next(row for row in manifest["source_rows"] if row["root_id"] == root)
        if not source_row["accepted_scenes"]:
            for qtype in TYPES:
                outcome = {"root_id": root, "route": manifest["route"], "branch": letter,
                    "question_type": qtype, "candidate_slots": manifest["candidate_slots_per_type_per_root"],
                    "mechanical_pass": 0, "mechanical_rejected": [], "abstentions": [], "review_rejected": [],
                    "ready_for_solver": 0, "qa_status": "EXPANSION_SHORTFALL_NO_QA", "actual_generation_requests": 0}
                outcomes.append(outcome)
                base.write(generation / "outcomes" / f"{root}.{qtype}.json", outcome)
            continue
        source_path = (generation / "sources" / f"{root}.json").resolve()
        source = base.read(source_path)
        prior = base.read(generation / "prior" / f"{root}.json")
        current = []
        for qtype in TYPES:
            code = "multi" if qtype == "multi_select" else "order"
            slots = manifest["candidate_slots_per_type_per_root"]
            ids = [f"{root}-v12-{letter}-{code}-b01-{index:02d}" for index in range(1, slots + 1)]
            context = {"root_id": root, "route": manifest["route"], "branch": letter,
                "candidate_ids": ids, "source_turns": source["turns"],
                "previous_candidates_dedup_only": prior + legacy.compact_questions(current),
                "priority_new_scene_refs": sorted({t.get("scene_ref", t["turn_ref"].rsplit(":", 1)[0])
                    for t in source["turns"] if t["turn_ref"].startswith(f"R02:{letter}:")}),
                "must_use_at_least_one_new_branch_evidence_turn": True}
            accepted, rejected, abstentions = [], [], []
            pending = ids[:]
            for attempt in (1, 2):
                payload, receipt = make_generation_payload(generation, root, qtype, context)
                raw, directory, metadata = caller.call(f"{root.split('-')[-1]}-{letter}-{code}-generate-{attempt}", payload,
                    {"root_id": root, "route": manifest["route"], "branch": letter,
                     "role": "question_generator", "question_type": qtype, "candidate_ids": pending,
                     "source_sha256": base.digest(source_path.read_bytes()), "feedback_receipt": receipt})
                if raw is None:
                    rejected = [{"candidate_id": qid, "errors": [base.read(directory / "ERROR.json")["error_type"]]} for qid in pending]
                    break
                valid, rejected, abstained, envelope_error = legacy.validate_draft(raw, pending, root, qtype, source, source_path)
                evidence_key = "option_evidence" if qtype == "multi_select" else "item_evidence"
                branch_valid = []
                for q in valid:
                    if not any(e["turn_ref"].startswith(f"R02:{letter}:") for e in q[evidence_key].values()):
                        rejected.append({"candidate_id": q["candidate_id"], "errors": ["no_new_branch_evidence"], "draft": q})
                    else:
                        q.update(generator_model=MODEL, generation_request_path=str(directory / "REQUEST.json"),
                            generation_response_path=str(directory / "RESPONSE.json"), raw_question_content_preserved=True,
                            expansion_route=manifest["route"], expansion_branch=letter)
                        branch_valid.append(q)
                accepted += branch_valid
                abstentions += abstained
                base.write(directory / "VALIDATION.json", {"accepted_ids": [q["candidate_id"] for q in branch_valid],
                    "rejected": rejected, "abstentions": abstained, "envelope_error": envelope_error,
                    "semantic_review": False, "new_branch_evidence_check": "source_ref_prefix_only"})
                if not rejected:
                    break
                pending = [row["candidate_id"] for row in rejected]
                repair = {"instruction": "Repair only the listed mechanical/source-binding errors; preserve accepted slots.", "failures": rejected}
                if envelope_error:
                    repair["previous_response"] = raw
                context = {**context, "candidate_ids": pending, "mechanical_repair": repair,
                    "previous_candidates_dedup_only": prior + legacy.compact_questions(current + accepted)}
            signatures = {tuple(sorted(" ".join(f.casefold().split()) for f in row["facts"])) for row in prior + legacy.compact_questions(current) if row.get("facts")}
            distinct = []
            for q in accepted:
                key = signature(q)
                if key in signatures:
                    rejected.append({"candidate_id": q["candidate_id"], "errors": ["duplicate_full_statement_set"]})
                else:
                    signatures.add(key)
                    distinct.append(q)
            accepted = distinct
            current += accepted
            mechanical += accepted
            outcome = {"root_id": root, "route": manifest["route"], "branch": letter,
                "question_type": qtype, "candidate_slots": slots, "mechanical_pass": len(accepted),
                "mechanical_rejected": rejected, "abstentions": abstentions, "review_rejected": [],
                "qa_status": "GENERATED", "actual_generation_requests": attempt}
            reviewed = []
            if qtype == "ordering":
                for candidate in accepted:
                    payload = {"model": MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
                        "response_format": {"type": "json_object"}, "messages": [{"role": "user",
                        "content": legacy.ordering_gate.build_review_prompt(candidate, source)}]}
                    payload, receipt = legacy.feedback.append_feedback_to_request(payload, generation / "feedback", root, "writer", TARGET)
                    raw, directory, metadata = caller.call(f"{root.split('-')[-1]}-{letter}-review-{candidate['candidate_id'].split('-')[-1]}", payload,
                        {"root_id": root, "route": manifest["route"], "branch": letter, "role": "ordering_reviewer",
                         "candidate_id": candidate["candidate_id"], "feedback_receipt": receipt,
                         "independent_request_without_prior_verdicts": True,
                         "source_sha256": base.digest(source_path.read_bytes())})
                    if raw is None:
                        outcome["review_rejected"].append({"candidate_id": candidate["candidate_id"], "reasons": ["review_request_failed"]})
                        continue
                    checked = legacy.ordering_gate.review_response(raw, [candidate], source,
                        directory / "RESPONSE.json", directory / "REQUEST.json", metadata)
                    base.write(directory / "REVIEW.json", checked)
                    reviewed += checked["accepted"]
                    outcome["review_rejected"] += checked["rejected"]
            else:
                reviewed = accepted
            ready += reviewed
            outcome["ready_for_solver"] = len(reviewed)
            outcomes.append(outcome)
            base.write(generation / "outcomes" / f"{root}.{qtype}.json", outcome)
            print(json.dumps({"event": "qa_type_done", "branch": letter, "root": root, "question_type": qtype,
                              "ready_for_solver": len(reviewed)}), flush=True)
    base.verify_bindings(manifest["files"])
    for directory in sorted((generation / "requests").glob("*")):
        payload, proof = base.read(directory / "REQUEST.json"), base.read(directory / "PROOF.json")
        suffix = legacy.feedback.build_feedback_suffix(generation / "feedback", proof["root_id"], "writer", TARGET)
        require(payload["messages"][0]["content"].endswith(suffix), "feedback_missing_from_actual_payload")
        require(proof["payload_sha256"] == base.digest(payload), "frozen_payload_changed")
        require(proof["feedback_receipt"]["feedback_appended"] is True, "feedback_receipt_missing")
    base.write(generation / "MECHANICAL_POOL.json", {"questions": mechanical})
    base.write(generation / "SOLVER_POOL.json", {"version": VERSION, "candidate_count": len(ready), "questions": ready,
        "provenance": {"requested_model": MODEL, "returned_model": MODEL, "codex_question_edits": False,
            "multi_select_semantic_review": False, "ordering_review_owner": MODEL,
            "source_round": "P+R01+R02_" + letter, "expansion_route": manifest["route"], "feedback_target_solver": TARGET},
        "ground_calls": 0, "training_started": False})
    report = {"version": VERSION, "branch": letter, "route": manifest["route"],
        "http_attempts": len(list((generation / "requests").glob("*/HTTP_ATTEMPT.json"))),
        "requests_prepared": caller.counter, "candidate_slots": manifest["budget"]["candidate_slots"],
        "mechanical_pass": len(mechanical), "ready_for_solver": len(ready), "roots": outcomes,
        "feedback_in_every_request": True, "semantic_dedup_certified": False,
        "expansion_shortfall_no_qa_roots": [row["root_id"] for row in manifest["source_rows"] if not row["accepted_scenes"]],
        "no_current_solver_feedback_in_generation": True, "ground_calls": 0, "training_calls": 0}
    base.write(generation / "REPORT.json", report)
    return report


def plan_requests(questions, sources, probe, tokenizer):
    """Reuse the existing MCQ 6 x 3 / ordering 12 x 1 full-only call budgets."""
    requests, prompts, bindings, public_questions, gold = [], {}, [], [], {}
    for original in questions:
        q = deepcopy(original)
        variants, _ = evaluator._variants_for(q)
        field = "options" if q["question_type"] == "multi_select" else "items"
        text_to_content = {text: content for content, text in variants[0]["content_to_text"].items()}
        labels = {label: text_to_content[text] for label, text in q[field].items()}
        gold[q["candidate_id"]] = [labels[label] for label in q["gold"]]
        public_questions.append({k: v for k, v in q.items() if k not in {"gold", "gold_content_ids"}})
        evidence = q["option_evidence" if field == "options" else "item_evidence"]
        bindings.append({"candidate_id": q["candidate_id"], "root_id": q["root_id"],
            "content_id_by_label": labels, "gold_content_ids": gold[q["candidate_id"]],
            "evidence_by_content_id": {labels[label]: item for label, item in evidence.items()}})
        source = sources[q["root_id"]]["source"]
        for variant in variants:
            choices = probe.contract.structured_choices(q["question_type"], variant)
            require(len(choices) == (63 if field == "options" else 720), "structured_choices_mismatch")
            if tokenizer is not None:
                dialogue = probe.contract.dialogue_rows(source)
                text = probe.contract.build_prompt(tokenizer, q, variant, dialogue, context_mode="full_context")
                public = {key: q[key] for key in ("question_type", "question")}
                require(text == probe.contract.build_prompt(tokenizer, public, {field: variant[field]}, dialogue,
                    context_mode="full_context"), "prompt_metadata_leak")
                tokens = len(tokenizer.encode(text, add_special_tokens=False))
                require(tokens + 32 <= 16384, "solver_prompt_exceeds_context")
            else:
                text, tokens = evaluator.prompt(q, variant, source), None
            for slot in range(3 if field == "options" else 1):
                rid = f"{q['candidate_id']}__{variant['variant_id']}__full_context__s{slot:02d}"
                requests.append({"request_id": rid, "candidate_id": q["candidate_id"], "root_id": q["root_id"],
                    "question_type": q["question_type"], "variant": variant, "mode": "full_context", "slot": slot,
                    "seed": evaluator.SEED + len(requests), "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "prompt_tokens": tokens})
                prompts[rid] = text
    require(len(requests) == sum(18 if q["question_type"] == "multi_select" else 12 for q in questions),
            "full_plan_count_mismatch")
    return requests, prompts, bindings, public_questions, gold


def solver_preflight(run, letter, *, require_model=True):
    generation, target = run / letter / "generation", run / letter / "solver"
    require(not target.exists(), "solver_run_already_exists")
    if require_model:
        require(Path(sys.prefix) == evaluator.EXPECTED_VENV, "USE_PROVEN_LOCAL4B_VENV")
    pool_path = generation / "SOLVER_POOL.json"
    if not base.read(pool_path)["questions"]:
        result = {"branch": letter, "status": "EMPTY_POOL_NO_SOLVER", "candidate_count": 0,
                  "request_count": 0, "ground_calls": 0, "training_calls": 0}
        base.write(target / "SKIPPED.json", result)
        return result
    pool, questions, sources = evaluator.validate_pool(pool_path, require_ordering_review=True)
    identity = evaluator.model_identity(TARGET) if require_model else {
        "model_path": str(evaluator.MODEL_PATHS[TARGET]), "model_identifier": TARGET,
        "tokenizer_path": str(evaluator.TOKENIZER), "skipped": True}
    probe = evaluator.load_probe("v12_qa_preflight_probe")
    decoder = probe.solver.generation_config("multi_select")
    expected = {"temperature": .8, "top_p": .9, "max_tokens": 32, "enable_thinking": False,
                "structured_output_version": "finite_choice_set_v1"}
    require(all(decoder.get(k) == v for k, v in expected.items()), "decoder_contract_mismatch")
    tokenizer = None
    if require_model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(evaluator.TOKENIZER), local_files_only=True)
    requests, prompts, bindings, public_questions, gold = plan_requests(questions, sources, probe, tokenizer)
    if require_model:
        evaluator._validate_model(identity)
    base.write(target / "questions.json", {"candidates": public_questions})
    base.write(target / "gold.json", gold)
    base.write(target / "canonical_bindings.json", {"candidates": bindings})
    frozen_sources = {}
    for root, row in sources.items():
        source = target / "sources" / f"{root}.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], source)
        frozen_sources[str(Path("sources") / source.name)] = row["sha256"]
    for rid, text in prompts.items():
        path = target / "prompts" / f"{rid}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(text)
    base.write(target / "solver_identity.json", identity)
    from scripts.current_method.restart_n5_spell import backends, vllm_backends, structured_qa_box_protocol
    code_paths = [Path(__file__), OLD / "evaluation/run_solver.py", *sorted((OLD / "evaluation").glob("*.py")),
        evaluator.BASE_RUNNER, Path(probe.solver.__file__), Path(probe.contract.__file__),
        Path(backends.__file__), Path(vllm_backends.__file__), Path(structured_qa_box_protocol.__file__)]
    manifest = {"status": "PREFLIGHT_PASS", "prepared_at_utc": now(), "policy_version": POLICY,
        "execution": "full_only_no_ground_v1", "model": identity["model_path"], "solver_model": TARGET,
        "tokenizer": str(evaluator.TOKENIZER), "candidate_count": len(questions), "request_count": len(requests),
        "full_slots": {"multi_select": "6 permutations x 3 rollouts", "ordering": "12 permutations x 1 rollout"},
        "random_display": {"version": "independent_seeded_permutation_v2", "seed": evaluator.SEED,
            "gold_blind": True, "cyclic_rebinding": False, "shape_filter": False},
        "generator_provenance": pool["provenance"], "solver_identity": identity, "decoder_contract": decoder,
        "input_sha256": {"pool": evaluator.sha(pool_path), **frozen_sources,
            **{name: evaluator.sha(target / name) for name in ("questions.json", "gold.json", "canonical_bindings.json")}},
        "pool_path": str(pool_path.resolve()), "pool_sha256": evaluator.sha(pool_path),
        "code_sha256": {str(path): evaluator.sha(path) for path in code_paths}, "requests": requests,
        "training_started": False, "training_promotion": False, "ground_calls": 0}
    base.write(target / "manifest.json", manifest)
    return {"branch": letter, "request_count": len(requests), "candidate_count": len(questions), "policy_version": POLICY}


def solver_report(run, letter):
    folder = run / letter / "solver"
    if (folder / "SKIPPED.json").exists():
        return base.read(folder / "SKIPPED.json")
    manifest = base.read(folder / "manifest.json")
    require(manifest["policy_version"] == POLICY, "wrong_v12_policy")
    records = [base.read(path) for path in sorted((folder / "responses").glob("*.json"))]
    questions = base.read(folder / "questions.json")["candidates"]
    results = []
    for q in questions:
        rows = [row for row in records if row.get("candidate_id") == q["candidate_id"]]
        expected_calls = 18 if q["question_type"] == "multi_select" else 12
        variants = [f"perm_{index:02d}" for index in range(6 if q["question_type"] == "multi_select" else 12)]
        scores = {variant: [Fraction(row["score"]["partial_score_fraction"]) for row in rows
            if row["variant_id"] == variant and row.get("evaluation_valid")] for variant in variants}
        valid = sum(len(values) for values in scores.values())
        errors = evaluator.audit_complete_call_set(manifest, records, q["candidate_id"])
        if valid != expected_calls or len(rows) != expected_calls:
            errors.append("PARSE_OR_CALL_FAILURE")
        exact = sum(bool(row.get("evaluation_valid") and row["score"].get("exact_match")) for row in rows)
        selection = evaluator.select(scores, variants, exact_rate=Fraction(exact, valid) if valid else None, data_errors=errors)
        selection.update(policy_version=POLICY, candidate_id=q["candidate_id"], root_id=q["root_id"],
            question_type=q["question_type"], expected_full_calls=expected_calls, actual_full_calls=len(rows))
        results.append(selection)
    result = {"policy_version": POLICY, "solver_model": TARGET, "model_path": manifest["model"],
        "ground_calls": 0, "questions": results, "call_set_errors": evaluator.audit_complete_call_set(manifest, records),
        "selected_questions": [row["candidate_id"] for row in results if row["selected_by_policy"] is True]}
    base.write(folder / "RESULTS.json", result)
    return result


def deliver(run):
    experiment = base.read(run / "MANIFEST.json")
    selected, routes = [], {}
    for letter in ROUTES:
        generation = run / letter / "generation"
        gen_report = base.read(generation / "REPORT.json")
        if gen_report["ready_for_solver"]:
            report = delivery.deliver(generation, run / letter / "solver", run / letter / "delivery")
        else:
            require(not (run / letter / "delivery").exists(), "delivery_already_exists")
            counts = {}
            for row in gen_report["roots"]:
                counts.setdefault(row["root_id"], {})[row["question_type"]] = {
                    "candidate_slots": row["candidate_slots"], "mechanical_pass": row["mechanical_pass"],
                    "ready_for_solver": 0, "selected": 0, "generator_abstentions": len(row["abstentions"]),
                    "mechanical_rejection_reasons": dict(Counter(e for r in row["mechanical_rejected"] for e in r["errors"])),
                    "qwen_review_rejected": len(row["review_rejected"])}
            for root in counts:
                counts[root].update(selected_total=0, trial_target_4_to_8_met=False)
            report = {"version": VERSION, "status": "EMPTY_POOL_NO_SOLVER", "per_root": counts,
                "full_solver_calls": 0, "generation_http_attempts_including_parent": gen_report["http_attempts"],
                "feedback_in_every_generation_request": gen_report["feedback_in_every_request"]}
            base.write(run / letter / "delivery/REPORT.json", report)
            base.write(run / letter / "delivery/SELECTED_QUESTIONS.json", {"questions": [], "training_started": False})
        routes[letter] = report
        selected += base.read(run / letter / "delivery/SELECTED_QUESTIONS.json")["questions"]
    # Exact duplicates across branches are counted once; no semantic model is added.
    unique, duplicates, seen = [], [], set()
    for q in selected:
        key = (q["root_id"], q["question_type"], signature(q))
        if key in seen:
            duplicates.append({"candidate_id": q["candidate_id"], "reason": "same_root_type_full_statement_set_duplicate"})
        else:
            seen.add(key)
            unique.append(q)
    per_root = {}
    for root in experiment["roots"]:
        by_route = {letter: routes[letter]["per_root"][root] for letter in ROUTES}
        for letter, counts in by_route.items():
            source = next(row for row in base.read(run / letter / "generation/MANIFEST.json")["source_rows"] if row["root_id"] == root)
            counts.update(expansion_accepted_scenes=source["accepted_scenes"], expansion_added_turns=source["added_turns"],
                qa_status="EVALUATED" if source["accepted_scenes"] else "EXPANSION_SHORTFALL_NO_QA")
        count = sum(q["root_id"] == root for q in unique)
        per_root[root] = {"routes": by_route, "selected_unique_exact": count,
            "trial_target_4_to_8_met": 4 <= count <= 8,
            "selected_multi_select": sum(q["root_id"] == root and q["question_type"] == "multi_select" for q in unique),
            "selected_ordering": sum(q["root_id"] == root and q["question_type"] == "ordering" for q in unique)}
    report = {"version": VERSION, "status": "EVALUATION_COMPLETE", "per_root": per_root,
        "generator_model": MODEL, "ordering_reviewer_model": MODEL, "solver_model": TARGET,
        "policy_version": POLICY, "selection_rule": experiment["selection_rule"],
        "selected_total_before_cross_branch_exact_dedup": len(selected), "selected_unique_exact": len(unique),
        "selected_multi_select": sum(q["question_type"] == "multi_select" for q in unique),
        "selected_ordering": sum(q["question_type"] == "ordering" for q in unique),
        "candidate_slots": experiment["candidate_slots_total"],
        "generation_http_attempts": sum(row["generation_http_attempts_including_parent"] for row in routes.values()),
        "full_solver_calls": sum(row["full_solver_calls"] for row in routes.values()),
        "feedback_in_every_generation_and_review_request": all(row["feedback_in_every_generation_request"] for row in routes.values()),
        "full_slots": experiment["full_slots"], "ground_calls": 0, "training_calls": 0,
        "codex_question_edits": False, "semantic_multi_review": False, "semantic_dedup_certified": False,
        "cross_branch_exact_duplicates": duplicates, "branches_independent": True,
        "route_deliveries": {letter: str(run / letter / "delivery/REPORT.json") for letter in ROUTES},
        "selected_wording": "Passed this training-selection policy; permutation behavior recorded separately."}
    base.write(run / "REPORT.json", report)
    base.write(run / "SELECTED_QUESTIONS.json", {"questions": unique, "training_started": False,
        "training_target": "complete_canonical_gold", "loss_weight_from_partial_score": False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generate", "solver-preflight", "solver-live", "solver-report", "deliver"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expansion-run", type=Path)
    parser.add_argument("--branch", choices=("A", "B", "all"), default="all")
    parser.add_argument("--slots", type=int, default=4)
    args = parser.parse_args()
    run = args.run.resolve()
    letters = tuple(ROUTES) if args.branch == "all" else (args.branch,)
    if args.action == "prepare":
        require(args.expansion_run is not None, "expansion_run_required")
        result = prepare(args.expansion_run.resolve(), run, slots=args.slots)
    elif args.action == "deliver":
        result = deliver(run)
    elif args.action == "generate":
        result = [generate_route(run, letter) for letter in letters]
    elif args.action == "solver-preflight":
        result = [solver_preflight(run, letter) for letter in letters]
    elif args.action == "solver-live":
        evaluator.POLICY_VERSION = POLICY
        result = []
        for letter in letters:
            if (run / letter / "solver/SKIPPED.json").exists():
                result.append(base.read(run / letter / "solver/SKIPPED.json"))
            else:
                evaluator.live(run / letter / "solver")
                result.append({"branch": letter, "status": "COMPLETE"})
    else:
        result = [solver_report(run, letter) for letter in letters]
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
