"""Independent artifact audit of a completed v14 run; never calls a model.

Run with .venv-local4b/bin/python for the frozen tokenizer's chat template.
This verifies provenance, execution and scoring, not semantic Gold correctness.
Only AUDIT.json is created, exclusively; frozen experiment files are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
MODEL = "Qwen/Qwen3.6-35B-A3B"
TARGET = "M_clean_policy"
MODEL_PATH = PROJECT / "restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model"
AUDIT_SHA = "add007d4e4f937f2ea1b7e124ccac3c53e5eda5b8bbed1beab3dd4b0855cadd0"
POLICY = "v14_full_only_random_mcq6x3_order12x1_v1"
SEED = 2026100800
ROUTES = {"A": "old_event_continuation", "B": "dataset_new_event"}
PUBLIC_ORDERING = ("candidate_id", "root_id", "question_type", "chronology_kind", "question", "items")
SYSTEM_PROMPT = ("You are a strict structured dialogue QA solver. Use only the supplied dialogue when it is present. "
                 "Return the final answer in exactly one <box>...</box> pair. Do not provide explanations, "
                 "reasoning, or any text outside the box.")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def signature(question):
    values = question.get("options") or question.get("items") or question.get("facts")
    values = values.values() if isinstance(values, dict) else values
    return tuple(sorted(" ".join(text.casefold().split()) for text in values))


def content_id(text):
    return "content_" + hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def expected_variants(question):
    """Reproduce the frozen sampler using only public content, ID and seed."""
    field = "options" if question["question_type"] == "multi_select" else "items"
    labels = list(question[field])
    content = [content_id(question[field][label]) for label in labels]
    text_by_id = {content_id(text): text for text in question[field].values()}
    count = 6 if field == "options" else 12
    def rng(index):
        value = hashlib.sha256(f"{SEED}|{question['candidate_id']}|{index}".encode()).digest()
        return random.Random(int.from_bytes(value[:8], "big"))
    rng(-1).shuffle(content)
    used, variants = set(), []
    for index in range(count):
        generator = rng(index)
        while True:
            order = list(content)
            generator.shuffle(order)
            if tuple(order) not in used:
                break
        used.add(tuple(order))
        variants.append({"variant_id": f"perm_{index:02d}", "canonical_order_content_ids": list(content),
            "display_order_content_ids": order, field: {label: text_by_id[cid] for label, cid in zip(labels, order)},
            "display_to_content": dict(zip(labels, order)), "content_to_text": text_by_id})
    return variants


def public_prompt(tokenizer, question, variant, source):
    kind = question["question_type"]
    field, heading = ("options", "Options") if kind == "multi_select" else ("items", "Items")
    chunks = ["Dialogue:\n" + "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source["turns"]),
              f"Question ({kind}):\n{question['question']}",
              heading + ":\n" + "\n".join(f"{label}. {text}" for label, text in variant[field].items())]
    chunks.append("Output only selected letters separated by commas, e.g. <box>A,C</box>." if kind == "multi_select"
                  else "Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.")
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(chunks)}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def independent_score(question, variant, output):
    match = re.fullmatch(r"\s*<box>\s*(.*?)\s*</box>\s*", output, re.S)
    require(match is not None, "unparseable_solver_output")
    text = match.group(1).strip()
    labels = [part.strip().upper() for part in text.split(",")] if text else []
    mapping = variant["display_to_content"]
    require(len(labels) == len(set(labels)) and all(label in mapping for label in labels), "solver_answer_ids_invalid")
    prediction = [mapping[label] for label in labels]
    field = "options" if question["question_type"] == "multi_select" else "items"
    gold = [content_id(question[field][label]) for label in question["gold"]]
    if field == "options":
        pset, gset = set(prediction), set(gold)
        tp, fp, fn = len(pset & gset), len(pset - gset), len(gset - pset)
        score = Fraction(tp, len(gset)) - (Fraction(fp, 6 - len(gset)) if len(gset) < 6 else 0)
        exact, prediction = pset == gset, sorted(pset)
    else:
        require(len(prediction) == 6 and set(prediction) == set(gold), "incomplete_solver_order")
        rank = {cid: index for index, cid in enumerate(prediction)}
        fp = sum(rank[a] > rank[b] for index, a in enumerate(gold) for b in gold[index + 1:])
        tp, fn = 15 - fp, 0
        score, exact = Fraction(tp - fp, 15), prediction == gold
    return {"display_prediction": labels, "content_prediction": prediction, "content_gold": gold,
            "tp": tp, "fp": fp, "fn": fn, "exact_match": exact,
            "partial_score": str(score), "partial_score_fraction": str(score)}


def independent_spans(anchors, source):
    turns = {turn["turn_ref"]: (index, turn) for index, turn in enumerate(source["turns"])}
    require(set(anchors) == {"I" + str(i) for i in range(1, 7)}, "ordering_span_labels_invalid")
    positions = {}
    for label, anchor in anchors.items():
        index, turn = turns[anchor["turn_ref"]]
        quote = anchor["quote"]
        starts = [i for i in range(len(turn["text"])) if turn["text"].startswith(quote, i)] if quote else []
        require(len(starts) == 1, "ordering_span_quote_not_unique:" + label)
        start, end = starts[0], starts[0] + len(quote)
        for key, expected in {"turn_index": index, "char_start": start, "char_end": end, "speaker": turn["speaker"]}.items():
            require(anchor.get(key) == expected, "ordering_span_offset_or_speaker_mismatch:" + label + ":" + key)
        positions[label] = (index, start, end)
    order = sorted(positions, key=lambda label: positions[label])
    for left, right in zip(order, order[1:]):
        a, b = positions[left], positions[right]
        require(a[0] != b[0] or a[2] <= b[1], "ordering_spans_overlap")
    return order


class Audit:
    def __init__(self):
        self.checked = {}
        self.counts = Counter()
        self.question_results = []
        self.tokenizer = None
        self.model_fingerprint = None

    def binding(self, path, expected):
        path = Path(path).resolve()
        if path not in self.checked:
            self.checked[path] = sha(path)
        require(self.checked[path] == expected, "file_hash_changed:" + str(path))

    def bindings(self, rows):
        for row in rows:
            self.binding(row["path"], row["sha256"])

    def model(self, identity):
        require(identity["model_identifier"] == TARGET and identity["model_path"] == str(MODEL_PATH)
                and identity.get("adapter_already_merged") is True and not identity.get("skipped"), "wrong_solver_identity")
        require(identity["audit_sha256"] == AUDIT_SHA, "wrong_trained_audit")
        self.binding(identity["audit_path"], AUDIT_SHA)
        self.binding(identity["tokenizer_audit_path"], identity["tokenizer_audit_sha256"])
        trained = read(identity["audit_path"])["models"][TARGET]
        require(trained["inference_model_path"] == str(MODEL_PATH)
                and trained["model_fingerprint_sha256"] == identity["model_fingerprint_sha256"], "trained_fingerprint_mismatch")
        self.model_fingerprint = identity["model_fingerprint_sha256"]
        expected = [{"path": str(MODEL_PATH / row["name"]), "name": row["name"],
                     "bytes": row["size_bytes"], "sha256": row["sha256"]}
                    for row in trained["merged_model_identity"]["files"]]
        expected += [row for row in read(identity["tokenizer_audit_path"])["files"]
                     if row["name"] in {"tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"}]
        require(identity["checkpoint_files"] == expected, "checkpoint_file_list_differs_from_trained_audit")
        for row in expected:
            require(Path(row["path"]).stat().st_size == row["bytes"], "checkpoint_size_changed")
            self.binding(row["path"], row["sha256"])
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(identity["tokenizer_path"], local_files_only=True)

    def requests(self, generation, sources, mechanical):
        directories = sorted(path for path in (generation / "requests").iterdir() if path.is_dir())
        actual = 0
        for directory in directories:
            request, proof = read(directory / "REQUEST.json"), read(directory / "PROOF.json")
            require(request["model"] == MODEL and request.get("enable_thinking") is False, "wrong_api_model_or_thinking")
            require(proof["payload_sha256"] == digest(request), "api_request_hash_changed")
            text = request["messages"][0]["content"]
            root = proof["root_id"]
            suffix_path = generation / "feedback" / root / "writer__M_clean_policy_feedback.txt"
            suffix = suffix_path.read_text()
            receipt = proof["feedback_receipt"]
            require(text.endswith(suffix) and receipt["feedback_appended"] is True
                    and receipt["suffix_sha256"] == digest(suffix.encode())
                    and receipt["request_user_text_sha256"] == digest(text.encode())
                    and receipt["target_solver_model"] == TARGET
                    and receipt["source_lock_sha256"] == sha(generation / "feedback/source_lock.json"), "api_feedback_mismatch")
            require(proof["source_sha256"] == sha(generation / "sources" / (root + ".json")), "api_source_sha_mismatch")
            public_text = text[:-len(suffix)]
            if proof["role"] == "question_generator":
                data = json.loads(public_text.split("\nINPUT_JSON\n", 1)[1])
                require(data["root_id"] == root and data["source_turns"] == sources[root]["turns"], "generator_source_changed_or_branch_mixed")
                require(data["candidate_ids"] == proof["candidate_ids"], "generator_candidate_id_mismatch")
            else:
                require(proof["role"] == "ordering_reviewer", "unexpected_api_role")
                data = json.loads(public_text.split("\nDATA:\n", 1)[1])
                q = mechanical[proof["candidate_id"]]
                require(data == {"root_id": root, "source_turns": sources[root]["turns"],
                    "candidates": [{key: q[key] for key in PUBLIC_ORDERING}]}, "review_public_input_leaks_gold_or_generated_evidence")
                require(proof["independent_request_without_prior_verdicts"] is True, "review_not_independent")
                self.counts["independent_ordering_review_requests"] += 1
            if (directory / "HTTP_ATTEMPT.json").exists():
                actual += 1
                attempt = read(directory / "HTTP_ATTEMPT.json")
                require(attempt["actual_payload_equals_frozen"] is True and attempt["payload_sha256"] == digest(request)
                        and attempt["requested_model"] == MODEL and attempt["transport_retries"] == 0, "actual_http_payload_mismatch")
            if (directory / "RESPONSE.json").exists():
                response = read(directory / "RESPONSE.json")
                meta = response["metadata"]
                require(meta["requested_model"] == MODEL and meta["returned_model"] == MODEL, "api_returned_model_mismatch")
                if meta.get("raw_response"):
                    raw = meta["raw_response"]
                    require(raw["model"] == MODEL and raw["choices"][0]["message"]["content"] == response["text"], "raw_api_envelope_mismatch")
            else:
                require((directory / "ERROR.json").exists(), "unfinished_api_request")
        self.counts["generation_and_review_http_attempts"] += actual
        return actual

    def candidate(self, question, sources, *, reviewed=False):
        source = sources[question["root_id"]]
        self.binding(question["source_path"], question["source_sha256"])
        require(read(question["source_path"]) == source, "candidate_source_identity_mismatch")
        raw_response = read(question["generation_response_path"])
        raw_candidates = json.loads(raw_response["text"])["questions"]
        raw = next(row for row in raw_candidates if row["candidate_id"] == question["candidate_id"])
        require(question["question"] == raw["question"], "candidate_question_rewritten")
        require(question["generator_model"] == MODEL and question["raw_question_content_preserved"] is True, "candidate_generator_provenance_invalid")
        if question["question_type"] == "multi_select":
            require(len(raw["options"]) == 6 and len(signature(question)) == len(set(signature(question))) == 6,
                    "multi_options_not_six_distinct")
            require(question["options"] == dict(zip("ABCDEF", [option["proposition"] for option in raw["options"]])), "multi_option_text_rewritten")
            gold = [label for label, option in zip("ABCDEF", raw["options"]) if option["relation"] == "supports"]
            require(2 <= len(gold) <= 6 and question["gold"] == gold, "multi_gold_relation_mismatch")
            require(question["raw_model_candidate"] == raw, "raw_candidate_snapshot_mismatch")
            require(question["citation_gate"] == "DIAGNOSTIC_ONLY", "multiselect_citation_gate_enabled")
            for label, option in zip("ABCDEF", raw["options"]):
                evidence = question["option_evidence"][label]
                for key, value in option.items():
                    if key != "proposition":
                        require(evidence.get(key) == value, "raw_multi_evidence_rewritten:" + label + ":" + key)
                require(evidence.get("reason") == option.get("rationale", option.get("reason")), "raw_rationale_rewritten")
            self.counts["multiselect_raw_candidates_checked"] += 1
        else:
            require(question["items"] == raw["items"], "ordering_items_rewritten")
            spans = question["order_anchor_spans"] if reviewed else question["generated_order_anchor_spans"]
            require(question["gold"] == independent_spans(spans, source), "ordering_gold_not_span_order")
            if not reviewed:
                self.counts["generated_ordering_anchor_sets_checked"] += 1
            if reviewed:
                review = question["qwen_review"]
                require(review["status"] == "PASS" and review["returned_model"] == MODEL, "ordering_review_not_qwen_pass")
                self.binding(review["response_path"], review["response_sha256"])
                self.binding(review["request_path"], review["request_file_sha256"])
                verdicts = json.loads(read(review["response_path"])["text"])["reviews"]
                verdict = next(row for row in verdicts if row["candidate_id"] == question["candidate_id"])
                for label, span in spans.items():
                    anchor = verdict["items"][label]["order_anchor"]
                    require(span["turn_ref"] == anchor["turn_ref"] and span["quote"] == anchor["quote"], "final_span_not_raw_reviewer_anchor")
                self.counts["reviewed_ordering_candidates_checked"] += 1
                if len({span["turn_ref"] for span in spans.values()}) < 6:
                    self.counts["same_turn_ordering_candidates_checked"] += 1

    def solver(self, directory, pool, sources):
        questions = {q["candidate_id"]: q for q in pool["questions"]}
        if not questions:
            skipped = read(directory / "SKIPPED.json")
            require(skipped["request_count"] == 0 and not (directory / "responses").exists(), "empty_pool_has_solver_calls")
            return []
        manifest, identity = read(directory / "manifest.json"), read(directory / "solver_identity.json")
        require(manifest["policy_version"] == POLICY and manifest["execution"] == "full_only_no_ground_v1"
                and manifest["solver_model"] == TARGET and manifest["model"] == str(MODEL_PATH), "solver_policy_or_model_wrong")
        require(identity == manifest["solver_identity"] and manifest["training_started"] is False
                and manifest["training_promotion"] is False and manifest["ground_calls"] == 0, "solver_identity_or_execution_scope_wrong")
        self.model(identity)
        for path, expected in manifest["code_sha256"].items():
            self.binding(path, expected)
        for path, expected in manifest["input_sha256"].items():
            self.binding(manifest["pool_path"] if path == "pool" else directory / path, expected)
        require(manifest["pool_sha256"] == sha(manifest["pool_path"]), "solver_pool_hash_changed")
        require(read(manifest["pool_path"]) == pool, "solver_pool_mismatch")
        random_config = manifest["random_display"]
        require(random_config == {"version": "independent_seeded_permutation_v2", "seed": SEED,
                "gold_blind": True, "cyclic_rebinding": False, "shape_filter": False}, "permutation_policy_changed")
        decoder = manifest["decoder_contract"]
        for key, expected in {"temperature": .8, "top_p": .9, "max_tokens": 32,
                              "enable_thinking": False, "structured_output_version": "finite_choice_set_v1"}.items():
            require(decoder[key] == expected, "decoder_contract_changed:" + key)
        public = read(directory / "questions.json")["candidates"]
        require(public == [{k: v for k, v in q.items() if k not in {"gold", "gold_content_ids"}}
                           for q in pool["questions"]], "solver_questions_rewritten")
        gold = read(directory / "gold.json")
        bindings = {row["candidate_id"]: row for row in read(directory / "canonical_bindings.json")["candidates"]}
        requests = manifest["requests"]
        for index, request in enumerate(requests):
            expected_id = f"{request['candidate_id']}__{request['variant']['variant_id']}__full_context__s{request['slot']:02d}"
            require(request["request_id"] == expected_id and request["seed"] == SEED + index, "planned_request_id_or_seed_mismatch")
        records = [read(path) for path in sorted((directory / "responses").glob("*.json"))]
        by_id = {row["request_id"]: row for row in records}
        require(len(by_id) == len(records) == len(requests) == manifest["request_count"], "full_call_count_or_duplicate_invalid")
        require(set(by_id) == {row["request_id"] for row in requests}, "missing_or_extra_full_calls")
        complete = read(directory / "COMPLETE.json")
        require(complete["actual_call_count"] == len(records) and complete["status"] == "COMPLETE", "solver_not_complete")
        results = read(directory / "RESULTS.json")
        require(results["call_set_errors"] == [] and results["ground_calls"] == 0, "reported_call_errors")
        result_by_id = {row["candidate_id"]: row for row in results["questions"]}
        require(set(result_by_id) == set(questions), "result_candidate_set_mismatch")
        selected = []
        for qid, question in questions.items():
            field = "options" if question["question_type"] == "multi_select" else "items"
            label_ids = {label: content_id(text) for label, text in question[field].items()}
            expected_gold = [label_ids[label] for label in question["gold"]]
            require(gold[qid] == expected_gold and bindings[qid]["gold_content_ids"] == expected_gold
                    and bindings[qid]["content_id_by_label"] == label_ids, "canonical_gold_mapping_changed")
            evidence = question["option_evidence" if field == "options" else "item_evidence"]
            require(bindings[qid]["evidence_by_content_id"] == {label_ids[label]: value for label, value in evidence.items()}, "canonical_evidence_mapping_changed")
            variants = expected_variants(question)
            qrequests = [row for row in requests if row["candidate_id"] == qid]
            rollouts = 3 if field == "options" else 1
            require(len(qrequests) == len(variants) * rollouts, "wrong_per_question_full_budget")
            planned_slots = {(v["variant_id"], slot) for v in variants for slot in range(rollouts)}
            require({(r["variant"]["variant_id"], r["slot"]) for r in qrequests} == planned_slots, "variant_slot_set_mismatch")
            scores, exact = defaultdict(list), 0
            display_gold = {}
            for request in qrequests:
                rid, variant = request["request_id"], request["variant"]
                expected_variant = variants[int(variant["variant_id"].split("_")[1])]
                require(variant == expected_variant, "permutation_not_gold_blind_frozen_sampler:" + rid)
                inverse = {cid: label for label, cid in variant["display_to_content"].items()}
                display_gold[variant["variant_id"]] = [inverse[cid] for cid in expected_gold]
                expected_prompt = public_prompt(self.tokenizer, question, variant, sources[question["root_id"]])
                prompt_path = directory / "prompts" / (rid + ".txt")
                require(prompt_path.read_text() == expected_prompt and sha(prompt_path) == request["prompt_sha256"], "solver_public_prompt_mismatch_or_gold_leak:" + rid)
                response = by_id[rid]
                for key in ("candidate_id", "root_id", "question_type", "slot", "seed"):
                    require(response[key] == request[key], "response_call_mapping_changed:" + rid + ":" + key)
                require(response["variant_id"] == variant["variant_id"] and response["mode"] == request["mode"] == "full_context"
                        and response["model_path"] == str(MODEL_PATH) and response["solver_model"] == TARGET, "response_mode_or_model_mismatch")
                require(response["evaluation_valid"] is True and response["data_errors"] == [], "invalid_solver_response")
                metadata = response["metadata"]
                require(metadata["finish_reason"] in {"stop", "eos"} and metadata["sampling_seed"] == request["seed"], "solver_finish_or_seed_mismatch")
                for key in ("temperature", "top_p", "max_tokens", "structured_output_api", "structured_output_version"):
                    require(metadata[key] == decoder[key], "response_decoder_changed:" + key)
                computed = independent_score(question, variant, response["raw_output"])
                for key, value in computed.items():
                    require(response["score"][key] == value, "recomputed_score_mismatch:" + rid + ":" + key)
                require(response["score"]["scored"] is True, "score_not_marked_scored")
                scores[variant["variant_id"]].append(Fraction(computed["partial_score_fraction"]))
                exact += computed["exact_match"]
            means = {key: sum(values, Fraction()) / rollouts for key, values in scores.items()}
            average = sum(means.values(), Fraction()) / len(variants)
            accepted = 0 < average < 1
            result = result_by_id[qid]
            require(result["full_partial_mean"] == str(average) and result["permutation_means"] == {k: str(v) for k, v in means.items()}
                    and result["full_exact_rate"] == str(Fraction(exact, len(qrequests))) and result["selected_by_policy"] is accepted
                    and result["data_errors"] == [] and result["ground_used"] is False, "recomputed_selection_mismatch:" + qid)
            require(result["expected_full_calls"] == result["actual_full_calls"] == len(qrequests), "result_call_count_mismatch")
            if accepted:
                selected.append(qid)
            self.question_results.append({"candidate_id": qid, "root_id": question["root_id"], "question_type": question["question_type"],
                "original_gold": question["gold"], "display_gold_by_permutation": display_gold,
                "full_partial_mean": str(average), "full_exact_rate": str(Fraction(exact, len(qrequests))),
                "selected_by_policy": accepted, "full_calls": len(qrequests)})
        require(results["selected_questions"] == selected, "solver_selected_ids_mismatch")
        self.counts["full_solver_calls_independently_recomputed"] += len(records)
        return selected

    def run(self, run):
        experiment, report = read(run / "MANIFEST.json"), read(run / "REPORT.json")
        require(experiment["version"] == "h4_r02_question_contract_v14" and report["status"] == "EVALUATION_COMPLETE", "wrong_or_incomplete_run")
        self.bindings(experiment["files"])
        expansion = Path(experiment["expansion_run"])
        require("h4_r02_yield_pilot_v12" in str(expansion), "not_frozen_v12_expansion")
        self.bindings(read(expansion / "MANIFEST.json")["files"])
        selected_all = []
        for branch, route in ROUTES.items():
            generation = run / branch / "generation"
            manifest, gen_report = read(generation / "MANIFEST.json"), read(generation / "REPORT.json")
            self.bindings(manifest["files"])
            require(manifest["branch"] == branch and manifest["route"] == route
                    and manifest["solver_model"] == TARGET and manifest["ground_calls"] == manifest["training_calls"] == 0,
                    "generation_scope_wrong")
            sources = {}
            for row in manifest["source_rows"]:
                root = row["root_id"]
                self.binding(row["source_path"], row["source_sha256"])
                original_path = expansion / "roots" / root / branch / "cumulative_dialogue.json"
                require(sha(original_path) == row["source_sha256"], "frozen_branch_source_changed")
                source, base = read(row["source_path"]), read(expansion / "sources" / (root + ".json"))
                require(source["turns"][:len(base["turns"])] == base["turns"] and source["branch_independent"] is True
                        and source["root_id"] == root and source["route"] == route, "branch_base_or_identity_changed")
                suffix = source["turns"][len(base["turns"]):]
                require(len(suffix) == row["added_turns"] and all(t["turn_ref"].startswith("R02:" + branch + ":") for t in suffix), "cross_branch_context_or_turn_count_mismatch")
                sources[root] = source
            mechanical = read(generation / "MECHANICAL_POOL.json")["questions"]
            pool = read(generation / "SOLVER_POOL.json")
            indexed = {q["candidate_id"]: q for q in mechanical}
            require(len(indexed) == len(mechanical) == gen_report["mechanical_pass"], "mechanical_pool_count_mismatch")
            actual = self.requests(generation, sources, indexed)
            require(actual == gen_report["http_attempts"], "generation_http_count_mismatch")
            for q in mechanical:
                self.candidate(q, sources)
            for q in pool["questions"]:
                require(q["candidate_id"] in indexed, "solver_candidate_not_mechanical_pool")
                if q["question_type"] == "ordering":
                    self.candidate(q, sources, reviewed=True)
                else:
                    require(q == indexed[q["candidate_id"]], "multi_candidate_changed_after_admission")
            require(len(pool["questions"]) == gen_report["ready_for_solver"], "ready_pool_count_mismatch")
            for outcome in gen_report["roots"]:
                root = outcome["root_id"]
                prior = read(generation / "prior" / (root + ".json"))
                hard = {signature(row) for row in prior if row.get("hard_dedup") is True and row.get("facts")}
                for rejected in outcome["mechanical_rejected"]:
                    if "duplicate_historical_selected_statement_set" in rejected["errors"]:
                        require(signature(rejected["draft"]) in hard, "dedup_rejected_only_warning_history")
                    if "duplicate_same_batch_statement_set" in rejected["errors"]:
                        matches = [indexed[m["candidate_id"]] for m in rejected["matches"]]
                        require(any(signature(q) == signature(rejected["draft"]) for q in matches), "same_batch_duplicate_not_exact")
            selected_ids = self.solver(run / branch / "solver", pool, sources)
            delivered = read(run / branch / "delivery/SELECTED_QUESTIONS.json")
            selected = delivered["questions"]
            require([q["candidate_id"] for q in selected] == selected_ids and delivered["training_started"] is False, "branch_delivery_selection_mismatch")
            by_id = {q["candidate_id"]: q for q in pool["questions"]}
            for q in selected:
                original = by_id[q["candidate_id"]]
                require(all(q[key] == value for key, value in original.items()), "selected_question_or_complete_gold_changed")
                require(q["training_target"] == "complete_canonical_gold" and q["loss_weight_from_partial_score"] is False, "selected_training_target_not_complete_gold")
            selected_all += selected
        unique, duplicates, seen = [], [], set()
        for q in selected_all:
            key = (q["root_id"], q["question_type"], signature(q))
            if key in seen:
                duplicates.append(q["candidate_id"])
            else:
                seen.add(key)
                unique.append(q)
        exported = read(run / "SELECTED_QUESTIONS.json")
        require(exported["questions"] == unique and exported["training_started"] is False
                and exported["training_target"] == "complete_canonical_gold" and exported["loss_weight_from_partial_score"] is False, "final_export_mismatch")
        require(report["selected_total_before_cross_branch_exact_dedup"] == len(selected_all)
                and report["selected_unique_exact"] == len(unique)
                and [row["candidate_id"] for row in report["cross_branch_exact_duplicates"]] == duplicates, "final_dedup_count_mismatch")
        for kind in ("multi_select", "ordering"):
            require(report["selected_" + kind] == sum(q["question_type"] == kind for q in unique), "final_type_count_mismatch")
        for root, row in report["per_root"].items():
            require(row["selected_unique_exact"] == sum(q["root_id"] == root for q in unique), "final_root_count_mismatch")
            for kind in ("multi_select", "ordering"):
                require(row["selected_" + kind] == sum(q["root_id"] == root and q["question_type"] == kind for q in unique),
                        "final_root_type_count_mismatch")
        require(report["generation_http_attempts"] == self.counts["generation_and_review_http_attempts"]
                and report["full_solver_calls"] == self.counts["full_solver_calls_independently_recomputed"]
                and report["ground_calls"] == report["training_calls"] == 0, "aggregate_call_count_mismatch")
        return {"status": "PASS", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "auditor": "independent_artifact_and_score_recomputation", "run": str(run),
            "audit_script_sha256": sha(Path(__file__)), "frozen_files_checked": len(self.checked),
            "counts": dict(self.counts), "solver_model": TARGET, "solver_model_path": str(MODEL_PATH),
            "model_fingerprint_sha256": self.model_fingerprint,
            "all_multiselect_text_options_relations_and_gold_preserved": True,
            "ordering_review_inputs_exclude_gold_and_generated_evidence": True,
            "ordering_final_spans_independently_verified": True if self.counts["reviewed_ordering_candidates_checked"] else None,
            "live_same_turn_ordering_verified": bool(self.counts["same_turn_ordering_candidates_checked"]),
            "ordering_live_coverage": "REVIEWED_CANDIDATES_PRESENT" if self.counts["reviewed_ordering_candidates_checked"] else "NO_REVIEWED_ORDERING_CANDIDATES_THIS_RUN",
            "branches_independent": True,
            "all_scores_and_means_independently_recomputed": True, "complete_planned_call_set": True,
            "ground_calls": 0, "training_calls": 0, "semantic_gold_certified": False,
            "semantic_novelty_certified": False, "selected_complete_gold_count": len(unique),
            "selected_before_cross_branch_exact_dedup": len(selected_all), "exact_duplicates_removed": duplicates,
            "questions": self.question_results,
            "limitation": "Checks raw model provenance and execution/scoring integrity. Does not certify semantic Gold correctness or semantic novelty."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs/frozen_v12_2roots_01")
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / "AUDIT.json"
    require(not output.exists(), "audit_already_exists_write_once")
    require((run / "REPORT.json").is_file(), "run_delivery_not_complete_no_audit_written")
    auditor = Audit()
    try:
        result = auditor.run(run)
    except Exception as error:
        result = {"status": "FAIL", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
                  "audit_script_sha256": sha(Path(__file__)), "error": type(error).__name__ + ":" + str(error),
                  "frozen_files_checked": len(auditor.checked), "counts": dict(auditor.counts),
                  "semantic_gold_certified": False, "training_started": False}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "questions"}, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
