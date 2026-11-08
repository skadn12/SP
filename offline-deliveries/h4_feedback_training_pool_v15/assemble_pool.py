"""Prepare an immutable, audited training pool from explicitly named selections.

No discovery of candidate pools, model calls, question edits or training occurs.
Existing inputs win exact statement-set duplicates; every canonical question has
one row regardless of how many display permutations were evaluated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
VERSION = "h4_feedback_training_pool_v15"
MODEL = "Qwen/Qwen3.6-35B-A3B"
TARGET = "M_clean_policy"
CHECKPOINT = PROJECT / "restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model"
TRAINED_AUDIT_SHA = "add007d4e4f937f2ea1b7e124ccac3c53e5eda5b8bbed1beab3dd4b0855cadd0"
MODEL_FINGERPRINT = "039fa9969ecff33b167ecf62a947fac872ac13905d05947ca2e7825c43d1ca90"
HISTORICAL_VERSIONS = {"h4_r02_yield_pilot_v12_qa", "h4_r02_question_contract_v14"}
ROUTES = {"A": "old_event_continuation", "B": "dataset_new_event"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def content_id(text):
    return "content_" + digest(text.strip().encode())[:16]


def statement_key(question):
    values = question.get("options") if question["question_type"] == "multi_select" else question.get("items")
    return (question["root_id"], question["question_type"],
            tuple(sorted(" ".join(text.casefold().split()) for text in values.values())))


def canonical_question_id(question):
    return "canonical_" + digest(statement_key(question))


def canonical_gold_statements(question):
    values = question.get("options") if question["question_type"] == "multi_select" else question["items"]
    text = [" ".join(values[label].casefold().split()) for label in question["gold"]]
    return tuple(sorted(text)) if question["question_type"] == "multi_select" else tuple(text)


def question_contract(question):
    kind = question.get("question_type")
    require(kind in {"multi_select", "ordering"}, "unsupported_question_type")
    labels = list("ABCDEF") if kind == "multi_select" else ["I" + str(i) for i in range(1, 7)]
    values = question.get("options") if kind == "multi_select" else question.get("items")
    require(isinstance(values, dict) and list(values) == labels
            and all(isinstance(text, str) and text.strip() for text in values.values()), "six_complete_options_or_items_required")
    require(len(set(statement_key(question)[2])) == 6, "duplicate_option_or_item_text")
    require(isinstance(question.get("question"), str) and question["question"].strip(), "question_text_missing")
    gold = question.get("gold")
    require(isinstance(gold, list) and 2 <= len(gold) <= 6
            and all(isinstance(label, str) and label in labels for label in gold)
            and len(set(gold)) == len(gold), "gold_cardinality_or_labels_invalid")
    if kind == "ordering":
        require(len(gold) == 6, "ordering_gold_not_complete_permutation")
    else:
        evidence = question.get("option_evidence")
        require(isinstance(evidence, dict) and set(evidence) == set(labels), "multi_relations_missing")
        require(all(isinstance(evidence[label], dict) and evidence[label].get("relation") in ("supports", "contradicts")
                    and ((evidence[label]["relation"] == "supports") == (label in gold)) for label in labels), "multi_relation_gold_mismatch")
    require(question.get("generator_model") == MODEL and question.get("raw_question_content_preserved") is True,
            "raw_qwen_provenance_required")
    require(question.get("training_target") == "complete_canonical_gold"
            and question.get("loss_weight_from_partial_score") is False, "complete_gold_training_target_required")
    require(not question.get("diagnostic_only"), "diagnostic_candidate_not_formal_selection")


def score(question, variant, raw):
    match = re.fullmatch(r"\s*<box>\s*(.*?)\s*</box>\s*", raw, re.S)
    require(match is not None, "solver_output_parse_failed")
    text = match.group(1).strip()
    labels = [part.strip().upper() for part in text.split(",")] if text else []
    mapping = variant["display_to_content"]
    require(len(labels) == len(set(labels)) and all(label in mapping for label in labels), "solver_output_labels_invalid")
    predicted = [mapping[label] for label in labels]
    values = question.get("options") if question["question_type"] == "multi_select" else question["items"]
    gold = [content_id(values[label]) for label in question["gold"]]
    if question["question_type"] == "multi_select":
        tp, fp = len(set(predicted) & set(gold)), len(set(predicted) - set(gold))
        value = Fraction(tp, len(gold)) - (Fraction(fp, 6 - len(gold)) if len(gold) < 6 else 0)
        exact = set(predicted) == set(gold)
    else:
        require(len(predicted) == 6 and set(predicted) == set(gold), "incomplete_solver_order")
        rank = {cid: index for index, cid in enumerate(predicted)}
        wrong = sum(rank[a] > rank[b] for index, a in enumerate(gold) for b in gold[index + 1:])
        value, exact = Fraction(15 - 2 * wrong, 15), predicted == gold
    return value, exact


def verify_selection(question, manifest, records, result):
    """Independently verify every planned answer and selection of one candidate."""
    qid, kind = question["candidate_id"], question["question_type"]
    variants, slots = (6, 3) if kind == "multi_select" else (12, 1)
    planned = [row for row in manifest["requests"] if row["candidate_id"] == qid]
    require(len(planned) == variants * slots, "wrong_full_call_budget:" + qid)
    expected_slots = {(f"perm_{i:02d}", slot) for i in range(variants) for slot in range(slots)}
    require({(row["variant"]["variant_id"], row["slot"]) for row in planned} == expected_slots, "incomplete_full_variant_slots:" + qid)
    requests = {row["request_id"]: row for row in planned}
    require(len(requests) == len(planned), "duplicate_planned_request")
    observed = [row for row in records if row.get("candidate_id") == qid or row.get("request_id") in requests]
    require(len(observed) == len(planned) and {row["request_id"] for row in observed} == set(requests), "missing_duplicate_or_extra_candidate_response:" + qid)
    values = question.get("options") if kind == "multi_select" else question["items"]
    expected_ids = {content_id(text) for text in values.values()}
    by_variant, exact, displays = defaultdict(list), 0, {}
    for row in observed:
        request = requests[row["request_id"]]
        require(row["evaluation_valid"] is True and row.get("data_errors") == [], "invalid_solver_answer:" + qid)
        for key in ("candidate_id", "root_id", "question_type", "slot", "seed"):
            require(row.get(key) == request.get(key), "response_plan_mapping_mismatch:" + key)
        require(row.get("mode") == request.get("mode") == "full_context" and row["variant_id"] == request["variant"]["variant_id"], "non_full_or_wrong_variant_answer")
        require(row["solver_model"] == TARGET and row["model_path"] == str(CHECKPOINT), "response_solver_checkpoint_mismatch")
        for key in ("temperature", "top_p", "max_tokens", "structured_output_api", "structured_output_version"):
            require(row["metadata"][key] == manifest["decoder_contract"][key], "response_decoder_mismatch:" + key)
        require(row["metadata"]["finish_reason"] in {"stop", "eos"}, "incomplete_solver_answer")
        variant = request["variant"]
        field = "options" if kind == "multi_select" else "items"
        require(set(variant["display_to_content"].values()) == expected_ids
                and len(variant["display_to_content"]) == 6
                and all(variant["display_to_content"][label] == content_id(text) for label, text in variant[field].items()),
                "display_content_mapping_invalid")
        order = tuple(variant["display_to_content"].values())
        require(row["variant_id"] not in displays or displays[row["variant_id"]] == order, "variant_changed_between_rollouts")
        displays[row["variant_id"]] = order
        value, is_exact = score(question, variant, row["raw_output"])
        require(row["score"]["scored"] is True and Fraction(row["score"]["partial_score_fraction"]) == value
                and row["score"]["exact_match"] is is_exact, "stored_score_mismatch")
        by_variant[row["variant_id"]].append(value)
        exact += is_exact
    require(len(set(displays.values())) == variants, "repeated_display_permutations")
    means = {variant: sum(scores, Fraction()) / slots for variant, scores in by_variant.items()}
    mean = sum(means.values(), Fraction()) / variants
    require(0 < mean < 1, "selected_score_outside_open_interval")
    require(result.get("selected_by_policy") is True and result.get("ground_used") is False
            and result.get("data_errors") == [] and Fraction(result["full_partial_mean"]) == mean
            and result["permutation_means"] == {key: str(value) for key, value in means.items()}
            and result["full_exact_rate"] == str(Fraction(exact, variants * slots))
            and result["actual_full_calls"] == result["expected_full_calls"] == variants * slots,
            "selection_result_mismatch")
    require(question["training_selection"] == result, "exported_selection_differs_from_solver_results")
    return {"full_calls": variants * slots, "full_partial_mean": str(mean), "full_exact_rate": str(Fraction(exact, variants * slots))}


class Inspector:
    def __init__(self):
        self.files, self.snapshots, self.models, self.solvers = {}, set(), set(), {}

    def bind(self, path, expected=None, *, snapshot=True):
        path = Path(path).resolve(strict=True)
        if path not in self.files:
            self.files[path] = sha(path)
        actual = self.files[path]
        require(expected is None or expected == actual, "frozen_file_hash_changed:" + str(path))
        if snapshot:
            self.snapshots.add(path)
        return {"path": str(path), "sha256": actual,
                "snapshot_path": "provenance/" + actual + "-" + path.name if snapshot else None}

    def manifest(self, path):
        value = read(path)
        self.bind(path)
        for row in value.get("files", []):
            self.bind(row["path"], row["sha256"], snapshot=False)
        return value

    def model(self, identity):
        require(identity.get("model_identifier") == TARGET and identity.get("model_path") == str(CHECKPOINT)
                and identity.get("model_fingerprint_sha256") == MODEL_FINGERPRINT
                and identity.get("audit_sha256") == TRAINED_AUDIT_SHA
                and identity.get("adapter_already_merged") is True and not identity.get("skipped"), "exact_current_m_clean_checkpoint_required")
        self.bind(identity["audit_path"], TRAINED_AUDIT_SHA)
        self.bind(identity["tokenizer_audit_path"], identity["tokenizer_audit_sha256"])
        trained = read(identity["audit_path"])["models"][TARGET]
        require(trained["inference_model_path"] == str(CHECKPOINT)
                and trained["model_fingerprint_sha256"] == MODEL_FINGERPRINT, "fixed_model_audit_identity_mismatch")
        expected = [{"path": str(CHECKPOINT / row["name"]), "name": row["name"], "bytes": row["size_bytes"], "sha256": row["sha256"]}
                    for row in trained["merged_model_identity"]["files"]]
        expected += [row for row in read(identity["tokenizer_audit_path"])["files"]
                     if row["name"] in {"tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"}]
        require(identity["checkpoint_files"] == expected, "checkpoint_manifest_not_fixed_audit")
        if MODEL_FINGERPRINT not in self.models:
            for row in expected:
                require(Path(row["path"]).stat().st_size == row["bytes"], "checkpoint_file_size_changed")
                self.bind(row["path"], row["sha256"], snapshot=False)
            self.models.add(MODEL_FINGERPRINT)

    def solver(self, directory):
        if directory in self.solvers:
            return self.solvers[directory]
        manifest = read(directory / "manifest.json")
        require(manifest["solver_model"] == TARGET and manifest["model"] == str(CHECKPOINT)
                and manifest["execution"] == "full_only_no_ground_v1" and manifest.get("training_started") is False
                and manifest.get("training_promotion") is False, "solver_scope_or_model_invalid")
        require(manifest["random_display"]["gold_blind"] is True and manifest["random_display"]["cyclic_rebinding"] is False, "old_permutation_policy_not_allowed")
        identity = read(directory / "solver_identity.json")
        require(identity == manifest["solver_identity"], "solver_manifest_identity_changed")
        self.model(identity)
        self.bind(directory / "manifest.json")
        self.bind(directory / "solver_identity.json")
        for path, expected in manifest["code_sha256"].items():
            self.bind(path, expected, snapshot=False)
        for path, expected in manifest["input_sha256"].items():
            self.bind(manifest["pool_path"] if path == "pool" else directory / path, expected)
        results = read(directory / "RESULTS.json")
        complete = read(directory / "COMPLETE.json")
        require(results["solver_model"] == TARGET and results["model_path"] == str(CHECKPOINT)
                and results["ground_calls"] == 0 and results["call_set_errors"] == [], "solver_results_invalid")
        records = []
        for path in sorted((directory / "responses").glob("*.json")):
            records.append(read(path))
            self.bind(path)
        planned_ids = [row["request_id"] for row in manifest["requests"]]
        observed_ids = [row["request_id"] for row in records]
        require(len(set(planned_ids)) == len(planned_ids) == len(records) == manifest["request_count"] == complete["actual_call_count"]
                and Counter(planned_ids) == Counter(observed_ids) and complete["status"] == "COMPLETE", "solver_call_set_incomplete")
        require(all(row.get("mode") == "full_context" and row.get("evaluation_valid") is True for row in records), "ground_or_invalid_solver_calls")
        for request in manifest["requests"]:
            self.bind(directory / "prompts" / (request["request_id"] + ".txt"), request["prompt_sha256"])
        self.bind(directory / "RESULTS.json")
        self.bind(directory / "COMPLETE.json")
        value = manifest, records, {row["candidate_id"]: row for row in results["questions"]}
        self.solvers[directory] = value
        return value

    def feedback_request(self, request_path, generation, root):
        request_path = Path(request_path)
        request, proof = read(request_path), read(request_path.with_name("PROOF.json"))
        require(request["model"] == MODEL and proof["payload_sha256"] == digest(request), "request_model_or_binding_invalid")
        suffix_path = generation / "feedback" / root / "writer__M_clean_policy_feedback.txt"
        suffix, receipt = suffix_path.read_text(), proof["feedback_receipt"]
        text = request["messages"][0]["content"]
        require(text.endswith(suffix) and receipt["feedback_appended"] is True
                and receipt["target_solver_model"] == TARGET and receipt["root_id"] == root
                and receipt["suffix_sha256"] == digest(suffix.encode())
                and receipt["request_user_text_sha256"] == digest(text.encode())
                and receipt["source_lock_sha256"] == sha(generation / "feedback/source_lock.json"), "actual_previous_round_feedback_binding_invalid")
        attempt_path = request_path.with_name("HTTP_ATTEMPT.json")
        attempt = read(attempt_path)
        require(attempt["actual_payload_equals_frozen"] is True and attempt["payload_sha256"] == digest(request), "actual_request_not_frozen_payload")
        return {"request": self.bind(request_path), "proof": self.bind(request_path.with_name("PROOF.json")),
                "actual_http_attempt": self.bind(attempt_path), "feedback_suffix": self.bind(suffix_path),
                "feedback_source_lock": self.bind(generation / "feedback/source_lock.json"),
                "receipt": deepcopy(receipt)}

    def expansion_audit(self, expansion, *, required):
        """Require the new expansion audit; retain available historical names."""
        expansion = Path(expansion).resolve()
        names = ("FINAL_AUDIT.json",) if required else ("FINAL_AUDIT.json", "AUDIT.json")
        available = [expansion / name for name in names if (expansion / name).is_file()]
        if not available:
            require(not required, "new_expansion_final_audit_required")
            return {"required": False, "status": "NOT_PRESENT_IN_HISTORICAL_RUN", "binding": None}
        path = available[0]
        audit = read(path)
        require(audit.get("status") == "PASS"
                and isinstance(audit.get("run"), str) and Path(audit["run"]).resolve() == expansion
                and not audit.get("failures") and audit.get("training_started", False) is False,
                "expansion_independent_audit_failed_or_wrong_run")
        return {"required": required, "status": "PASS", "binding": self.bind(path)}

    def delivery(self, path, *, is_new):
        path = Path(path).resolve(strict=True)
        require(path.name == "SELECTED_QUESTIONS.json", "explicit_selected_delivery_required")
        run = path.parent
        require((run / "MANIFEST.json").is_file(), "top_level_audited_qa_delivery_required")
        selected, manifest, report = read(path), self.manifest(run / "MANIFEST.json"), read(run / "REPORT.json")
        version = manifest.get("version", "")
        require((is_new and version.startswith(VERSION)) or (not is_new and version in HISTORICAL_VERSIONS), "input_version_not_explicitly_allowed_for_role")
        require(not selected.get("diagnostic_only") and not report.get("diagnostic_only")
                and selected.get("training_started") is False and report.get("ground_calls") == report.get("training_calls") == 0,
                "diagnostic_or_training_input_not_allowed")
        require(manifest["solver_model"] == report["solver_model"] == TARGET and report["status"] == "EVALUATION_COMPLETE", "non_current_or_incomplete_selection")
        require(selected.get("training_target") == "complete_canonical_gold"
                and selected.get("loss_weight_from_partial_score") is False, "selected_export_target_invalid")
        audit_paths = [run / name for name in ("AUDIT.json", "QA_INDEPENDENT_AUDIT.json") if (run / name).is_file()]
        require(audit_paths, "independent_source_audit_required")
        audit_path = audit_paths[0]
        audit = read(audit_path)
        require(audit.get("status") == "PASS" and Path(audit["run"]).resolve() == run
                and not audit.get("diagnostic_only"), "independent_source_audit_failed_or_wrong_run")
        qa_chain = {"selected_delivery": self.bind(path), "qa_manifest": self.bind(run / "MANIFEST.json"),
                    "qa_report": self.bind(run / "REPORT.json"), "source_audit": self.bind(audit_path)}
        expansion = Path(manifest["expansion_run"])
        em = self.manifest(expansion / "MANIFEST.json")
        require(em["target_solver_model"] == TARGET and em["training_started"] is False, "expansion_feedback_target_or_scope_mismatch")
        expansion_chain = {"run": str(expansion), "manifest": self.bind(expansion / "MANIFEST.json"),
                           "report": self.bind(expansion / "REPORT.json"),
                           "feedback_manifest": self.bind(expansion / "feedback/COMPILED_FEEDBACK.json"),
                           "feedback_source_lock": self.bind(expansion / "feedback/source_lock.json"),
                           "independent_audit": self.expansion_audit(expansion, required=is_new)}
        questions = selected.get("questions")
        require(isinstance(questions, list) and len({q["candidate_id"] for q in questions}) == len(questions)
                and len(questions) == report["selected_unique_exact"], "selected_export_count_invalid")
        prepared = []
        for question in questions:
            question_contract(question)
            branch, root = question["expansion_branch"], question["root_id"]
            require(branch in ROUTES and question["expansion_route"] == ROUTES[branch], "candidate_branch_invalid")
            generation = run / branch / "generation"
            gm = self.manifest(generation / "MANIFEST.json")
            require(gm["feedback_round"] == "previous_training_round_only" and gm["solver_model"] == TARGET
                    and gm["ground_calls"] == gm["training_calls"] == 0, "question_generation_feedback_or_scope_invalid")
            source = read(question["source_path"])
            self.bind(question["source_path"], question["source_sha256"])
            require(source["root_id"] == root and source["route"] == ROUTES[branch]
                    and source["branch_independent"] is True, "question_context_identity_invalid")
            original = expansion / "roots" / root / branch / "cumulative_dialogue.json"
            self.bind(original, question["source_sha256"])
            prefix_path = expansion / "sources" / (root + ".json")
            prefix = read(prefix_path)
            require(source["turns"][:len(prefix["turns"])] == prefix["turns"]
                    and all(turn["turn_ref"].startswith("R02:" + branch + ":") for turn in source["turns"][len(prefix["turns"]):]), "A_B_contexts_mixed")
            require(len({turn["turn_ref"] for turn in source["turns"]}) == len(source["turns"]), "context_duplicate_turn_ids")
            self.bind(prefix_path)
            generation_pool = read(generation / "SOLVER_POOL.json")
            originals = [q for q in generation_pool["questions"] if q["candidate_id"] == question["candidate_id"]]
            require(len(originals) == 1 and all(question[key] == value for key, value in originals[0].items()), "selected_question_changed_from_solver_pool")
            self.bind(generation / "SOLVER_POOL.json")
            raw_response = read(question["generation_response_path"])
            require(raw_response["metadata"]["requested_model"] == raw_response["metadata"]["returned_model"] == MODEL, "raw_generator_model_invalid")
            raw = next(row for row in json.loads(raw_response["text"])["questions"] if row["candidate_id"] == question["candidate_id"])
            require(raw["question"] == question["question"], "question_text_not_raw_qwen")
            field = "options" if question["question_type"] == "multi_select" else "items"
            raw_values = dict(zip("ABCDEF", [option["proposition"] for option in raw["options"]])) if field == "options" else raw["items"]
            require(raw_values == question[field], "option_or_item_text_not_raw_qwen")
            if field == "options":
                require(question["gold"] == [label for label, option in zip("ABCDEF", raw["options"]) if option["relation"] == "supports"], "raw_qwen_gold_relabelled")
            request_chain = self.feedback_request(question["generation_request_path"], generation, root)
            request_chain["response"] = self.bind(question["generation_response_path"])
            review_chain = None
            if field == "items":
                review = question["qwen_review"]
                require(review["status"] == "PASS" and review["returned_model"] == MODEL, "ordering_qwen_review_required")
                review_chain = self.feedback_request(review["request_path"], generation, root)
                review_chain["response"] = self.bind(review["response_path"], review["response_sha256"])
            solver = run / branch / "solver"
            sm, records, results = self.solver(solver)
            require(question["candidate_id"] in results, "selected_question_missing_solver_result")
            stats = verify_selection(question, sm, records, results[question["candidate_id"]])
            selected_responses = [self.bind(solver / "responses" / (row["request_id"] + ".json"))
                                  for row in sm["requests"] if row["candidate_id"] == question["candidate_id"]]
            provenance = {"input_role": "new" if is_new else "existing", "candidate_id": question["candidate_id"],
                          "source_context_sha256": question["source_sha256"], "branch": branch,
                          "qa": qa_chain, "expansion": expansion_chain, "generation_manifest": self.bind(generation / "MANIFEST.json"),
                          "generation": request_chain, "ordering_review": review_chain,
                          "solver_manifest": self.bind(solver / "manifest.json"), "solver_results": self.bind(solver / "RESULTS.json"),
                          "solver_identity": self.bind(solver / "solver_identity.json"), "full_responses": selected_responses,
                          "selection_verified": stats}
            prepared.append({"question": deepcopy(question), "source": source, "is_new": is_new, "provenance": provenance})
        return prepared, {"path": str(path), "sha256": sha(path), "role": "new" if is_new else "existing", "selected_count": len(questions),
                          "source_audit": qa_chain["source_audit"], "expansion": expansion_chain}


def deduplicate(entries):
    kept, duplicates, by_key = [], [], {}
    for entry in entries:
        key = statement_key(entry["question"])
        if key not in by_key:
            row = deepcopy(entry)
            row["provenance_chain"] = [deepcopy(entry["provenance"])]
            kept.append(row)
            by_key[key] = row
            continue
        retained = by_key[key]
        retained["provenance_chain"].append(deepcopy(entry["provenance"]))
        duplicates.append({"canonical_question_id": canonical_question_id(entry["question"]),
            "retained_candidate_id": retained["question"]["candidate_id"], "excluded_candidate_id": entry["question"]["candidate_id"],
            "excluded_input_role": "new" if entry["is_new"] else "existing",
            "same_context": retained["question"]["source_sha256"] == entry["question"]["source_sha256"],
            "gold_content_disagreement": canonical_gold_statements(retained["question"]) != canonical_gold_statements(entry["question"]),
            "reason": "SAME_ROOT_TYPE_NORMALIZED_FULL_STATEMENT_SET", "retained_context_sha256": retained["question"]["source_sha256"],
            "excluded_context_sha256": entry["question"]["source_sha256"]})
    return kept, duplicates


def write_once(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def assemble(inputs, new_inputs, output):
    output = Path(output).resolve()
    require(output.is_relative_to(HERE) and output != HERE, "output_must_be_a_new_v15_subdirectory")
    require(not output.exists(), "immutable_output_already_exists")
    paths = [Path(path).resolve(strict=True) for path in list(inputs) + list(new_inputs)]
    require(paths and len(set(paths)) == len(paths), "explicit_unique_inputs_required")
    require(all(not output.is_relative_to(path.parent) for path in paths), "output_cannot_modify_input_delivery")
    inspector, entries, input_rows = Inspector(), [], []
    for index, path in enumerate(paths):
        rows, binding = inspector.delivery(path, is_new=index >= len(inputs))
        entries.extend(rows)
        input_rows.append(binding)
    kept, duplicates = deduplicate(entries)
    existing_unique = sum(not entry["is_new"] for entry in kept)
    new_unique = sum(entry["is_new"] for entry in kept)
    inspector.bind(Path(__file__))
    inspector.bind(HERE / "POLICY.md")
    # Everything above is read-only; no partially verified pool is materialized.
    output.mkdir(parents=True, exist_ok=False)
    (output / "provenance").mkdir()
    (output / "contexts").mkdir()
    for path in sorted(inspector.snapshots):
        destination = output / "provenance" / (inspector.files[path] + "-" + path.name)
        if not destination.exists():
            shutil.copyfile(path, destination)
        require(sha(destination) == inspector.files[path], "snapshot_copy_changed")
    context_hashes = set()
    for entry in entries:
        original = Path(entry["question"]["source_path"])
        expected = entry["question"]["source_sha256"]
        destination = output / "contexts" / (expected + ".json")
        if not destination.exists():
            shutil.copyfile(original, destination)
        require(sha(destination) == expected, "context_copy_changed")
        context_hashes.add(expected)
    rows = []
    for entry in kept:
        question = entry["question"]
        field = "options" if question["question_type"] == "multi_select" else "items"
        ids = {label: content_id(text) for label, text in question[field].items()}
        rows.append({"canonical_question_id": canonical_question_id(question), "candidate_id": question["candidate_id"],
            "root_id": question["root_id"], "question_type": question["question_type"], "question": question["question"],
            field: deepcopy(question[field]), "gold": deepcopy(question["gold"]),
            "canonical_content_id_by_label": ids, "complete_gold_content_ids": [ids[label] for label in question["gold"]],
            "context_id": "context_" + question["source_sha256"], "source_sha256": question["source_sha256"],
            "source_path": str(output / "contexts" / (question["source_sha256"] + ".json")),
            "full_context": deepcopy(entry["source"]), "original_selected_question": deepcopy(question),
            "origin": "new" if entry["is_new"] else "existing", "provenance_chain": entry["provenance_chain"],
            "training_target": "complete_canonical_gold", "canonical_exposure_units": 1,
            "loss_weight_from_partial_score": False, "evaluated_permutations_are_separate_training_examples": False})
    with (output / "TRAINING_POOL.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    by_root = {}
    for row in rows:
        counts = by_root.setdefault(row["root_id"], {"existing": 0, "new": 0, "multi_select": 0, "ordering": 0, "total": 0})
        counts[row["origin"]] += 1
        counts[row["question_type"]] += 1
        counts["total"] += 1
    report = {"version": VERSION, "status": "IMMUTABLE_TRAINING_POOL_PREPARED_NOT_TRAINED",
        "input_selected_total": len(entries), "input_existing_selected": sum(not e["is_new"] for e in entries),
        "input_new_selected": sum(e["is_new"] for e in entries), "existing_unique": existing_unique,
        "new_unique_added": new_unique, "total_unique": len(rows), "exact_duplicates_excluded": len(duplicates),
        "duplicates": duplicates, "source_contexts_preserved": len(context_hashes),
        "selected_multi_select": sum(row["question_type"] == "multi_select" for row in rows),
        "selected_ordering": sum(row["question_type"] == "ordering" for row in rows), "per_root": by_root,
        "solver_model": TARGET, "solver_checkpoint": str(CHECKPOINT), "model_fingerprint_sha256": MODEL_FINGERPRINT,
        "training_started": False, "training_calls": 0, "generation_calls": 0, "solver_calls": 0,
        "ground_used_for_selection": False, "full_context_in_every_row": True, "complete_gold_in_every_row": True,
        "canonical_exposure_units_per_row": 1, "loss_weight_from_partial_score": False,
        "semantic_gold_certified": False, "semantic_novelty_certified": False,
        "existing_formal_training_directories_modified": False, "inputs": input_rows}
    write_once(output / "REPORT.json", report)
    lines = ["# Immutable training pool preparation", "",
        f"Existing unique questions: **{existing_unique}**. Newly added unique questions: **{new_unique}**. Total: **{len(rows)}**.",
        f"Multiselect: **{report['selected_multi_select']}**; ordering: **{report['selected_ordering']}**. Exact duplicates excluded: **{len(duplicates)}**.", "",
        "| Root | Existing | New | Multiselect | Ordering | Total |", "|---|---:|---:|---:|---:|---:|"]
    for root, counts in by_root.items():
        lines.append(f"| {root} | {counts['existing']} | {counts['new']} | {counts['multi_select']} | {counts['ordering']} | {counts['total']} |")
    lines += ["", "Every row preserves the complete frozen dialogue, original question text and complete Gold. A canonical question appears once; evaluated permutations do not multiply its exposure.",
        "", "Only explicitly supplied audited M_clean selections were used. The previous-round feedback receipts and expansion/QA/Solver provenance were archived.",
        "", "No generation, Solver inference or training was launched by this assembler. Semantic Gold correctness and semantic novelty are not certified."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    # Recheck all archived inputs; checkpoint/code dependencies are already fixed
    # by their recorded digests and are not copied into the lightweight pool.
    for path in inspector.snapshots:
        require(sha(path) == inspector.files[path], "input_changed_during_assembly:" + str(path))
    manifest = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False, "immutable_write_once": True, "inputs": input_rows,
        "policy": "explicit_audited_current_m_clean_complete_full_selection_then_exact_canonical_dedup",
        "files": [{"path": str(path.relative_to(output)), "sha256": sha(path)} for path in sorted(output.rglob("*")) if path.is_file()],
        "original_bindings": [{"path": str(path), "sha256": value, "snapshot_copied": path in inspector.snapshots}
                              for path, value in sorted(inspector.files.items())], "report": report}
    write_once(output / "MANIFEST.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[], help="Existing audited v12/v14 selected export (repeatable).")
    parser.add_argument("--new-input", type=Path, action="append", default=[], help="New audited v15 QA selected export (repeatable).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.input, args.new_input, args.output)
    print(json.dumps({key: result[key] for key in ("status", "existing_unique", "new_unique_added", "total_unique", "selected_multi_select", "selected_ordering", "exact_duplicates_excluded", "training_started")}, indent=2))


if __name__ == "__main__":
    main()
