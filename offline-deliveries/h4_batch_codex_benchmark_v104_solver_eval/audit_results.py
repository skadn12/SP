"""Independently audit the paired v104 full-context Solver evaluation.

Uses only Python stdlib plus the frozen tokenizer; never imports native scoring
or selection code and never calls a model. Artifacts are created exclusively.
Run after both native ``report`` phases with the proven local4b Python.
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
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
VERSION = "v104_paired_solver_independent_audit_v1"
POLICY = "v15_full_only_random_mcq6x3_order12x1_v1"
SELECT_POLICY = "r02_groundless_full_only_random_position_v2"
SEED = 2026100800
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
MODELS = {"base": ("M0", TOKENIZER), "trained": ("M_clean_policy", Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model"))}
BASE_AUDIT_SHA = "59e23710e2373a00aef3b7d50d117d4089eea47cd17fd16a00eebcc83fba0035"
TRAINED_AUDIT_SHA = "add007d4e4f937f2ea1b7e124ccac3c53e5eda5b8bbed1beab3dd4b0855cadd0"
SYSTEM = ("You are a strict structured dialogue QA solver. Use only the supplied dialogue when it is present. "
          "Return the final answer in exactly one <box>...</box> pair. Do not provide explanations, "
          "reasoning, or any text outside the box.")


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_JSON_KEY:" + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_id(text):
    return "content_" + hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def fraction(value):
    return {"fraction": str(value), "value": float(value)} if value is not None else None


def mean(values):
    return sum(values, Fraction()) / len(values) if values else None


def independent_variants(question):
    field = "options" if question["question_type"] == "multi_select" else "items"
    labels = list(question[field])
    content = [content_id(question[field][label]) for label in labels]
    by_content = {content_id(text): text for text in question[field].values()}
    def rng(index):
        seed = hashlib.sha256(f"{SEED}|{question['candidate_id']}|{index}".encode()).digest()
        return random.Random(int.from_bytes(seed[:8], "big"))
    rng(-1).shuffle(content)
    used, result = set(), []
    for index in range(6 if field == "options" else 12):
        generator = rng(index)
        while True:
            order = content[:]
            generator.shuffle(order)
            if tuple(order) not in used:
                break
        used.add(tuple(order))
        result.append({"variant_id": f"perm_{index:02d}",
            "canonical_order_content_ids": content[:], "display_order_content_ids": order,
            field: {label: by_content[cid] for label, cid in zip(labels, order)},
            "display_to_content": dict(zip(labels, order)), "content_to_text": by_content})
    return result


def public_prompt(tokenizer, question, variant, source):
    multi = question["question_type"] == "multi_select"
    field, heading = ("options", "Options") if multi else ("items", "Items")
    chunks = ["Dialogue:\n" + "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source["turns"]),
        f"Question ({question['question_type']}):\n{question['question']}",
        heading + ":\n" + "\n".join(f"{label}. {text}" for label, text in variant[field].items()),
        "Output only selected letters separated by commas, e.g. <box>A,C</box>." if multi else
        "Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>."]
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(chunks)}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def replay_score(question, variant, raw):
    """Parse directly and score with exact arithmetic, without native imports."""
    result = {"parse_status": "success", "parse_reason": None, "scored": False,
        "display_prediction": None, "content_prediction": None,
        "content_gold": [content_id(question["options" if question["question_type"] == "multi_select" else "items"][label])
                         for label in question["gold"]],
        "tp": None, "fp": None, "fn": None, "exact_match": None, "partial_score": None}
    reason = None
    if not isinstance(raw, str):
        reason = "raw_output_missing_or_not_text"
    else:
        text = raw.strip()
        if not (text.startswith("<box>") and text.endswith("</box>")):
            reason = "exactly_one_box_required"
        else:
            inner = text[5:-6].strip()
            labels = [part.strip().upper() for part in inner.split(",")] if inner else []
            if any(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) is None for value in labels):
                reason = "malformed_id_list"
            elif len(labels) != len(set(labels)):
                reason = "duplicate_id"
    if reason:
        result.update(parse_status="failure", parse_reason=reason, scoring_status="PARSE_FAILURE")
        return result
    result["display_prediction"] = labels
    mapping = variant["display_to_content"]
    invalid = [label for label in labels if label not in mapping]
    if invalid:
        result.update(scoring_status="INVALID_ID", invalid_ids=invalid)
        return result
    predicted = [mapping[label] for label in labels]
    if len(predicted) != len(set(predicted)):
        result.update(scoring_status="DUPLICATE_CONTENT_ID", invalid_ids=[])
        return result
    gold = result["content_gold"]
    if question["question_type"] == "multi_select":
        pred_set, gold_set = set(predicted), set(gold)
        tp, fp, fn = len(pred_set & gold_set), len(pred_set - gold_set), len(gold_set - pred_set)
        partial = Fraction(tp, len(gold_set))
        if len(gold_set) < len(mapping):
            partial -= Fraction(fp, len(mapping) - len(gold_set))
        exact, predicted = pred_set == gold_set, sorted(pred_set)
    else:
        if len(predicted) != len(gold) or set(predicted) != set(gold):
            result.update(scoring_status="INCOMPLETE_ORDER", content_prediction=predicted)
            return result
        positions = {cid: index for index, cid in enumerate(predicted)}
        total = len(gold) * (len(gold) - 1) // 2
        fp = sum(positions[a] > positions[b] for index, a in enumerate(gold) for b in gold[index + 1:])
        tp, fn = total - fp, 0
        partial, exact = Fraction(tp - fp, total), predicted == gold
    result.update(scoring_status="SCORED", scored=True, content_prediction=predicted,
        tp=tp, fp=fp, fn=fn, exact_match=exact,
        partial_score=str(partial), partial_score_fraction=str(partial))
    return result


class Audit:
    def __init__(self, directory):
        self.directory = directory
        self.errors, self.hashes = [], {}
        self.calls = []
        self.tokenizer = None

    def check(self, condition, reason, errors=None):
        if not condition:
            (self.errors if errors is None else errors).append(reason)
        return bool(condition)

    def binding(self, path, expected, errors=None):
        path = Path(path).resolve()
        try:
            if str(path) not in self.hashes:
                self.hashes[str(path)] = sha(path)
            return self.check(self.hashes[str(path)] == expected, "HASH_MISMATCH:" + str(path), errors)
        except OSError as exc:
            self.check(False, "BINDING_UNREADABLE:" + str(path) + ":" + str(exc), errors)
            return False

    def source_batch(self, pool):
        """Verify sealed source batch when its manifest is carried by adapter."""
        source_dir = self.directory.parent / "h4_batch_codex_benchmark_v104"
        path = source_dir / "COMPLETE_MANIFEST.json"
        sealed = read(path)
        provenance = pool.get("provenance", {})
        self.check(provenance.get("backend") == "CURRENT_CODEX_SESSION"
            and provenance.get("exact_model_id") is None
            and provenance.get("source_batch_id") == source_dir.name,
            "SOURCE_GENERATOR_PROVENANCE_MISMATCH")
        claimed = provenance.get("source_batch_manifest", {})
        self.check(claimed.get("path") == str(path), "SOURCE_MANIFEST_PROVENANCE_PATH_MISMATCH")
        self.binding(path, claimed.get("sha256"))
        self.check(sealed.get("batch_status") == "CONTENT_REVIEWED_NOT_SOLVER_OR_TRAINING_APPROVED", "SOURCE_BATCH_NOT_CONTENT_SEALED")
        for row in sealed["files"]:
            self.binding(source_dir / row["path"], row["sha256"])
            self.check((source_dir / row["path"]).stat().st_size == row["bytes"],
                "SOURCE_BATCH_SIZE_MISMATCH:" + row["path"])
        self.hashes[str(path.resolve())] = sha(path)
        public = read(source_dir / "SELECTED_PUBLIC_QUESTIONS.json")["questions"] + read(source_dir / "CONTROL_PUBLIC_QUESTIONS.json")["questions"]
        private = read(source_dir / "SELECTED_PRIVATE_GOLD.json")["answers"] + read(source_dir / "CONTROL_PRIVATE_GOLD.json")["answers"]
        public_by = {q["candidate_id"]: q for q in public}
        private_by = {q["candidate_id"]: q for q in private}
        self.check(len(public_by) == len(public) == len(private_by) == len(private) == 22
            and set(public_by) == set(private_by) == {q["candidate_id"] for q in pool["questions"]}, "SOURCE_BATCH_POOL_ID_SET_MISMATCH")
        self.check(Counter(q["question_type"] for q in pool["questions"]) == {"multi_select": 16, "ordering": 6},
            "SOURCE_BATCH_TYPE_COUNTS_MISMATCH")
        self.check(Counter(len(q["gold"]) for q in pool["questions"] if q["question_type"] == "multi_select")
            == {2: 2, 3: 4, 4: 3, 5: 3, 6: 4}, "SOURCE_BATCH_GOLD_COUNTS_MISMATCH")
        for q in pool["questions"]:
            qid = q["candidate_id"]
            if qid not in public_by or qid not in private_by:
                continue
            original = public_by[qid]
            for key in ("candidate_id", "root_id", "question_type", "question", "options", "items", "source_sha256"):
                if key in original:
                    self.check(q.get(key) == original[key], "PUBLIC_CONTENT_CHANGED:" + qid + ":" + key)
            self.check(q.get("gold") == private_by[qid]["gold"], "PRIVATE_GOLD_CHANGED:" + qid)
            self.check(Path(q["source_path"]).resolve() == Path(original["source_path"]).resolve(),
                "PUBLIC_SOURCE_PATH_CHANGED:" + qid)
            self.check(q.get("role") == private_by[qid].get("role"), "PRIVATE_ROLE_CHANGED:" + qid)
        return {"path": str(path), "sha256": sha(path), "sealed_file_count": len(sealed["files"]),
            "sealed_content_verified": not self.errors}

    def adapter(self, pool):
        """Audit receipts and the pre-execution seal without trusting adapter code."""
        binding_path = self.directory / "ADAPTER_BINDINGS.json"
        receipt_path = self.directory / "ADAPTATION_RECEIPT.json"
        preflight_path = self.directory / "PAIRED_PREFLIGHT.json"
        binding, receipt, preflight = read(binding_path), read(receipt_path), read(preflight_path)
        self.check(binding.get("status") == "PASS" and binding.get("candidate_count") == 22
            and binding.get("calls_per_model") == 360
            and binding.get("models") == {label: name for label, (name, _) in MODELS.items()}
            and binding.get("automatic_retry") is False and binding.get("training_started") is False,
            "ADAPTER_BINDING_CONTRACT_MISMATCH")
        source_dir = self.directory.parent / "h4_batch_codex_benchmark_v104"
        seal_path = source_dir / "COMPLETE_MANIFEST.json"
        sealed = read(seal_path)
        expected_sources = {str(seal_path): sha(seal_path)}
        expected_sources.update({str(source_dir / row["path"]): row["sha256"] for row in sealed["files"]})
        source_rows = binding.get("source_batch_files", [])
        self.check(len(source_rows) == len(expected_sources)
            and {row["path"]: row["sha256"] for row in source_rows} == expected_sources,
            "ADAPTER_SOURCE_SEAL_SET_MISMATCH")
        for rows in (source_rows, binding.get("files", []), preflight.get("files", [])):
            self.check(len({row["path"] for row in rows}) == len(rows), "DUPLICATE_RECEIPT_FILE_BINDING")
            for row in rows:
                self.binding(row["path"], row["sha256"])
        bound_paths = {row["path"] for row in binding.get("files", [])}
        required = {str(self.directory / name) for name in
            ("ADAPTATION_RECEIPT.json", "EVALUATION_POOL.json", "adapter_protocol.py", "evaluate.py")}
        required.update(str(self.directory.parent / "h4_feedback_training_pool_v15_qa" / "evaluation" / name)
            for name in ("protocol.py", "run_solver.py", "scoring.py", "selection.py"))
        self.check(required <= bound_paths, "ADAPTER_REQUIRED_FILE_BINDING_MISSING")
        for key in ("native_protocol", "adapter_protocol"):
            self.binding(receipt[key]["path"], receipt[key]["sha256"])
        self.check(receipt.get("solver_protocol") == POLICY and receipt.get("ground_calls") == 0
            and receipt.get("new_semantic_judgments") is False and receipt.get("public_text_edits") is False
            and receipt.get("training_started") is False, "ADAPTATION_RECEIPT_CONTRACT_MISMATCH")
        private = read(source_dir / "SELECTED_PRIVATE_GOLD.json")["answers"] + read(source_dir / "CONTROL_PRIVATE_GOLD.json")["answers"]
        private_by = {q["candidate_id"]: q for q in private}
        receipts = receipt.get("questions", [])
        receipt_by = {row["candidate_id"]: row for row in receipts}
        self.check(len(receipt_by) == len(receipts) == 22 and set(receipt_by) == set(private_by),
            "ADAPTATION_RECEIPT_QUESTION_SET_MISMATCH")
        for question in pool["questions"]:
            qid = question["candidate_id"]
            row = receipt_by.get(qid, {})
            self.check(row.get("original_private") == private_by.get(qid)
                and row.get("public_unchanged_except_absolute_source_path") is True
                and row.get("source") == {"path": question["source_path"], "sha256": question["source_sha256"]},
                "ADAPTATION_RECEIPT_QUESTION_BINDING_MISMATCH:" + qid)
            crosscheck = row.get("final_crosscheck", {})
            expected_path = Path(question["source_path"]).parent / "FINAL_CROSSCHECK.json"
            self.check(crosscheck.get("path") == str(expected_path), "ADAPTATION_CROSSCHECK_PATH_MISMATCH:" + qid)
            self.binding(expected_path, crosscheck.get("sha256"))
            matches = [r for r in read(expected_path)["outcomes"] if r["candidate_id"] == qid]
            self.check(len(matches) == 1 and matches[0].get("status") == "PASS"
                and matches[0].get("original_gold") == question["gold"]
                and matches[0].get("unknown_labels") == []
                and (matches[0].get("independent_supported_labels") == question["gold"]
                     if question["question_type"] == "multi_select" else
                     type(matches[0].get("enumerated_ordering_anchor_combinations")) is int
                     and matches[0]["enumerated_ordering_anchor_combinations"] > 0),
                "ADAPTATION_FINAL_REVIEW_BINDING_MISMATCH:" + qid)
        self.check(preflight.get("status") == "PASS" and preflight.get("candidate_count") == 22
            and preflight.get("calls_per_model") == 360
            and preflight.get("same_prompts_permutations_seeds_decoder_scoring") is True
            and preflight.get("training_started") is False, "PAIRED_PREFLIGHT_CONTRACT_MISMATCH")
        self.check(preflight.get("adapter_binding", {}).get("path") == str(binding_path),
            "PREFLIGHT_ADAPTER_BINDING_PATH_MISMATCH")
        self.binding(binding_path, preflight.get("adapter_binding", {}).get("sha256"))
        expected_preflight = set()
        max_prompt_tokens = 0
        for label in MODELS:
            directory = self.directory / "run" / label
            expected_preflight.update(str(directory / name) for name in
                ("canonical_bindings.json", "gold.json", "manifest.json", "questions.json", "solver_identity.json"))
            manifest = read(directory / "manifest.json")
            for request in manifest["requests"]:
                expected_preflight.add(str(directory / "prompts" / (request["request_id"] + ".txt")))
                max_prompt_tokens = max(max_prompt_tokens, request["prompt_tokens"])
            expected_preflight.update(str(directory / "sources" / (root + ".json"))
                for root in {q["root_id"] for q in pool["questions"]})
        self.check({row["path"] for row in preflight.get("files", [])} == expected_preflight,
            "PREFLIGHT_FROZEN_FILE_SET_MISMATCH")
        self.check(preflight.get("max_prompt_tokens") == max_prompt_tokens, "PREFLIGHT_TOKEN_COUNT_MISMATCH")
        return {"adapter_bindings_sha256": sha(binding_path), "adaptation_receipt_sha256": sha(receipt_path),
            "paired_preflight_sha256": sha(preflight_path), "bound_preflight_files": len(expected_preflight),
            "python_prefix": preflight.get("python_prefix"), "packages": preflight.get("packages")}

    def model(self, identity, expected_name, expected_path):
        self.check(identity.get("model_identifier") == expected_name and identity.get("model_path") == str(expected_path)
            and identity.get("tokenizer_path") == str(TOKENIZER) and not identity.get("skipped"), "MODEL_IDENTITY_INVALID:" + expected_name)
        expected_hash = BASE_AUDIT_SHA if expected_name == "M0" else TRAINED_AUDIT_SHA
        self.check(identity.get("audit_sha256") == expected_hash, "WRONG_FIXED_MODEL_AUDIT:" + expected_name)
        self.binding(identity["audit_path"], expected_hash)
        self.binding(identity["tokenizer_audit_path"], BASE_AUDIT_SHA)
        base = read(identity["tokenizer_audit_path"])
        if expected_name == "M0":
            expected = base["files"]
        else:
            trained = read(identity["audit_path"])["models"][expected_name]
            self.check(trained["inference_model_path"] == str(expected_path)
                and identity.get("model_fingerprint_sha256") == trained["model_fingerprint_sha256"]
                and identity.get("adapter_already_merged") is True, "TRAINED_MODEL_FINGERPRINT_MISMATCH")
            expected = [{"path": str(expected_path / row["name"]), "name": row["name"],
                "bytes": row["size_bytes"], "sha256": row["sha256"]} for row in trained["merged_model_identity"]["files"]]
            expected += [row for row in base["files"] if row["name"] in {"tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"}]
        self.check(identity.get("checkpoint_files") == expected, "CHECKPOINT_FILE_LIST_MISMATCH:" + expected_name)
        for row in expected:
            self.check(Path(row["path"]).stat().st_size == row["bytes"], "CHECKPOINT_SIZE_MISMATCH:" + row["path"])
            self.binding(row["path"], row["sha256"])
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)

    def run(self, label, pool):
        directory = self.directory / "run" / label
        manifest, identity = read(directory / "manifest.json"), read(directory / "solver_identity.json")
        expected_name, expected_path = MODELS[label]
        run_start = len(self.errors)
        self.check(manifest.get("policy_version") == POLICY and manifest.get("execution") == "full_only_no_ground_v1", "POLICY_MISMATCH:" + label)
        self.check(manifest.get("solver_model") == expected_name and manifest.get("model") == str(expected_path)
            and manifest.get("tokenizer") == str(TOKENIZER) and identity == manifest.get("solver_identity"), "MANIFEST_MODEL_MISMATCH:" + label)
        self.check(manifest.get("training_started") is False and manifest.get("training_promotion") is False
            and manifest.get("new_generation_calls") == 0, "UNEXPECTED_TRAINING_OR_GENERATION:" + label)
        self.model(identity, expected_name, expected_path)
        for path, digest in manifest["code_sha256"].items():
            self.binding(path, digest)
        for name, digest in manifest["input_sha256"].items():
            self.binding(Path(manifest["pool_path"]) if name == "pool" else directory / name, digest)
        self.check(Path(manifest["pool_path"]).resolve() == (self.directory / "EVALUATION_POOL.json").resolve()
            and manifest.get("pool_sha256") == sha(self.directory / "EVALUATION_POOL.json")
            and read(manifest["pool_path"]) == pool, "RUN_POOL_MISMATCH:" + label)
        self.check(manifest.get("generator_provenance") == pool.get("provenance"), "PROVENANCE_CHANGED:" + label)
        self.check(manifest.get("random_display") == {"version": "independent_seeded_permutation_v2", "seed": SEED,
            "gold_blind": True, "cyclic_rebinding": False, "shape_filter": False}, "RANDOM_DISPLAY_CONTRACT_MISMATCH:" + label)
        decoder = manifest["decoder_contract"]
        for key, expected in {"temperature": .8, "top_p": .9, "max_tokens": 32, "enable_thinking": False,
            "structured_output_api": "structured_outputs=StructuredOutputsParams(choice)", "structured_output_version": "finite_choice_set_v1"}.items():
            self.check(decoder.get(key) == expected, "DECODER_CONTRACT_MISMATCH:" + label + ":" + key)
        expected_public = [{k: v for k, v in q.items() if k not in {"gold", "gold_content_ids"}} for q in pool["questions"]]
        self.check(read(directory / "questions.json")["candidates"] == expected_public, "RUN_PUBLIC_QUESTIONS_CHANGED:" + label)
        gold, bindings_raw = read(directory / "gold.json"), read(directory / "canonical_bindings.json")["candidates"]
        bindings = {q["candidate_id"]: q for q in bindings_raw}
        q_by = {q["candidate_id"]: q for q in pool["questions"]}
        self.check(len(q_by) == len(pool["questions"]) == 22, "POOL_CANDIDATE_COUNT_OR_DUPLICATE")
        self.check(set(gold) == set(bindings) == set(q_by) and len(bindings) == len(bindings_raw), "GOLD_BINDING_CANDIDATE_SET_MISMATCH:" + label)
        requests = manifest["requests"]
        plans = {r["request_id"]: r for r in requests}
        expected_count = sum(18 if q["question_type"] == "multi_select" else 12 for q in pool["questions"])
        self.check(expected_count == 360 and len(plans) == len(requests) == manifest.get("request_count") == expected_count
            and manifest.get("candidate_count") == len(q_by), "PLANNED_COUNT_MISMATCH:" + label)
        self.check(all(r.get("candidate_id") in q_by for r in requests), "PLANNED_UNKNOWN_CANDIDATE:" + label)
        for index, request in enumerate(requests):
            expected_id = f"{request['candidate_id']}__{request['variant']['variant_id']}__full_context__s{request['slot']:02d}"
            self.check(request["request_id"] == expected_id and type(request.get("seed")) is int
                and request["seed"] == SEED + index and request.get("mode") == "full_context", "REQUEST_ID_MODE_OR_SEED_MISMATCH:" + label + ":" + request["request_id"])
        paths = sorted((directory / "responses").glob("*.json"))
        records = [read(path) for path in paths]
        records_by = {row.get("request_id"): row for row in records}
        self.check(len(records_by) == len(records) == len(requests) and set(records_by) == set(plans), "MISSING_EXTRA_OR_DUPLICATE_RESPONSES:" + label)
        for path, record in zip(paths, records):
            self.check(path.stem == record.get("request_id"), "RESPONSE_FILENAME_MISMATCH:" + str(path))
        complete, started = read(directory / "COMPLETE.json"), read(directory / "STARTED.json")
        self.check(complete.get("status") == "COMPLETE" and complete.get("actual_call_count") == len(records)
            and complete.get("training_started") is False and complete.get("training_promotion") is False
            and not (directory / "FAILED.json").exists(), "RUN_NOT_COMPLETE:" + label)
        self.check(started.get("manifest_sha256") == sha(directory / "manifest.json"), "STARTED_MANIFEST_MISMATCH:" + label)
        native = read(directory / "RESULTS.json")
        native_by = {q["candidate_id"]: q for q in native["questions"]}
        self.check(set(native_by) == set(q_by) and len(native_by) == len(native["questions"])
            and native.get("policy_version") == POLICY and native.get("solver_model") == expected_name
            and native.get("model_path") == str(expected_path) and native.get("ground_calls") == 0, "NATIVE_RESULTS_HEADER_MISMATCH:" + label)
        question_results, run_calls = [], []
        for question in pool["questions"]:
            qid, qtype = question["candidate_id"], question["question_type"]
            field = "options" if qtype == "multi_select" else "items"
            expected_labels = tuple("ABCDEF") if field == "options" else tuple(f"I{i}" for i in range(1, 7))
            qerrors = []
            self.check(tuple(question[field]) == expected_labels and len(set(question[field].values())) == 6, "SIX_DISTINCT_CONTENT_REQUIRED:" + qid, qerrors)
            label_ids = {key: content_id(value) for key, value in question[field].items()}
            expected_gold = [label_ids[key] for key in question["gold"]]
            self.check(gold.get(qid) == expected_gold and bindings.get(qid, {}).get("gold_content_ids") == expected_gold
                and bindings.get(qid, {}).get("content_id_by_label") == label_ids, "CANONICAL_GOLD_BINDING_MISMATCH:" + qid, qerrors)
            evidence = question["option_evidence" if field == "options" else "item_evidence"]
            self.check(bindings.get(qid, {}).get("evidence_by_content_id") == {label_ids[key]: value for key, value in evidence.items()}, "CANONICAL_EVIDENCE_BINDING_MISMATCH:" + qid, qerrors)
            source_path = Path(question["source_path"])
            self.binding(source_path, question["source_sha256"], qerrors)
            source = read(source_path)
            frozen_source = directory / "sources" / (question["root_id"] + ".json")
            self.binding(frozen_source, question["source_sha256"], qerrors)
            self.check(source == read(frozen_source) and source.get("root_id") == question["root_id"], "ROOT_SOURCE_MISMATCH:" + qid, qerrors)
            variants = independent_variants(question)
            qrequests = [r for r in requests if r["candidate_id"] == qid]
            rollouts = 3 if field == "options" else 1
            expected_slots = {(v["variant_id"], slot) for v in variants for slot in range(rollouts)}
            self.check(len(qrequests) == len(expected_slots)
                and {(r["variant"]["variant_id"], r["slot"]) for r in qrequests} == expected_slots,
                "VARIANT_SLOT_SET_MISMATCH:" + qid, qerrors)
            qcalls, by_variant, display_gold = [], defaultdict(list), {}
            for request in qrequests:
                rid, variant = request["request_id"], request["variant"]
                errors = []
                self.check(request.get("root_id") == question["root_id"] and request.get("question_type") == qtype
                    and type(request.get("slot")) is int, "REQUEST_QUESTION_BINDING_MISMATCH", errors)
                expected_variant = next((v for v in variants if v["variant_id"] == variant["variant_id"]), None)
                self.check(variant == expected_variant, "RANDOM_PERMUTATION_MISMATCH", errors)
                inverse = {cid: key for key, cid in variant["display_to_content"].items()}
                display_gold[variant["variant_id"]] = [inverse[cid] for cid in expected_gold]
                text = public_prompt(self.tokenizer, question, variant, source)
                prompt_path = directory / "prompts" / (rid + ".txt")
                self.check(prompt_path.read_text(encoding="utf-8") == text
                    and sha(prompt_path) == request["prompt_sha256"], "PUBLIC_PROMPT_MISMATCH_OR_METADATA_LEAK", errors)
                tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
                self.check(type(request.get("prompt_tokens")) is int and request["prompt_tokens"] == tokens
                    and tokens + 32 <= 16384, "PLANNED_PROMPT_TOKENS_OR_CONTEXT_MISMATCH", errors)
                response = records_by.get(rid)
                if response is None:
                    replay = {"scored": False, "parse_status": "failure", "parse_reason": "missing_response", "scoring_status": "CALL_FAILURE", "exact_match": None}
                    errors.append("MISSING_RESPONSE")
                    metadata = {}
                else:
                    for key, expected in {"candidate_id": qid, "root_id": question["root_id"], "question_type": qtype,
                        "variant_id": variant["variant_id"], "slot": request["slot"], "seed": request["seed"], "mode": "full_context",
                        "solver_model": expected_name, "model_path": str(expected_path)}.items():
                        self.check(response.get(key) == expected and (key not in {"slot", "seed"} or type(response.get(key)) is int), "RESPONSE_BINDING_MISMATCH:" + key, errors)
                    metadata = response.get("metadata") or {}
                    self.check(metadata.get("finish_reason") in {"stop", "eos"}, "INCOMPLETE_GENERATION", errors)
                    self.check(metadata.get("sampling_seed") == request["seed"] and type(metadata.get("sampling_seed")) is int, "RESPONSE_SAMPLING_SEED_MISMATCH", errors)
                    self.check(metadata.get("prompt_token_count") == tokens and type(metadata.get("prompt_token_count")) is int, "RESPONSE_PROMPT_TOKEN_MISMATCH", errors)
                    self.check(type(metadata.get("completion_token_count")) is int and 0 < metadata.get("completion_token_count", 0) <= decoder["max_tokens"], "COMPLETION_TOKEN_COUNT_INVALID", errors)
                    for key in ("temperature", "top_p", "max_tokens", "structured_output_api", "structured_output_version"):
                        self.check(metadata.get(key) == decoder.get(key), "RESPONSE_DECODER_MISMATCH:" + key, errors)
                    if "enable_thinking" in metadata:
                        self.check(metadata["enable_thinking"] is False, "RESPONSE_THINKING_ENABLED", errors)
                    replay = replay_score(question, variant, response.get("raw_output"))
                    self.check(response.get("score") == replay, "NATIVE_STORED_SCORE_MISMATCH", errors)
                    independently_valid = replay["scored"] and metadata.get("finish_reason") in {"stop", "eos"} and not errors
                    self.check(response.get("evaluation_valid") is independently_valid, "NATIVE_VALIDITY_MISMATCH", errors)
                    if independently_valid:
                        self.check(response.get("data_errors") == [], "NATIVE_DATA_ERRORS_NONEMPTY", errors)
                if not replay.get("scored"):
                    errors.append("UNSCORED:" + replay.get("scoring_status", "UNKNOWN"))
                valid = not errors and replay.get("scored") is True
                call = {"model": label, "request_id": rid, "candidate_id": qid, "root_id": question["root_id"], "question_type": qtype,
                    "gold_count": len(question["gold"]), "variant_id": variant["variant_id"], "slot": request["slot"], "seed": request["seed"],
                    "prompt_sha256": request["prompt_sha256"], "prompt_tokens": tokens, "metadata": metadata,
                    "raw_output": response.get("raw_output") if response else None, "independent_score": replay,
                    "evaluation_valid": valid, "data_errors": sorted(set(errors)),
                    "response_path": str(directory / "responses" / (rid + ".json"))}
                if valid:
                    by_variant[variant["variant_id"]].append(Fraction(replay["partial_score_fraction"]))
                qcalls.append(call); run_calls.append(call); self.calls.append(call)
            means = {v["variant_id"]: mean(by_variant[v["variant_id"]]) for v in variants}
            valid_calls = [r for r in qcalls if r["evaluation_valid"]]
            complete_q = len(valid_calls) == len(expected_slots) and not qerrors
            partial_mean = mean(list(means.values())) if complete_q else None
            exact_count = sum(r["independent_score"]["exact_match"] is True for r in valid_calls)
            exact_rate = Fraction(exact_count, len(valid_calls)) if valid_calls else None
            accepted = 0 < partial_mean < 1 if partial_mean is not None else None
            reason = ("EVALUATION_INCOMPLETE_OR_INVALID" if accepted is None else "FULL_PARTIAL_IN_OPEN_INTERVAL" if accepted else
                "FULL_PARTIAL_NONPOSITIVE" if partial_mean <= 0 else "FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM")
            nrow = native_by.get(qid, {})
            expected_native = {"selected_by_policy": accepted, "selection_reason": reason,
                "full_partial_mean": str(partial_mean) if partial_mean is not None else None,
                "permutation_means": {key: str(value) for key, value in means.items() if value is not None},
                "full_exact_rate": str(exact_rate) if exact_rate is not None else None,
                "ground_used": False, "expected_full_calls": len(expected_slots), "actual_full_calls": len(qcalls)}
            for key, value in expected_native.items():
                self.check(nrow.get(key) == value, "NATIVE_SELECTION_MISMATCH:" + qid + ":" + key, qerrors)
            self.check(nrow.get("policy_version") == SELECT_POLICY, "NATIVE_SELECTION_POLICY_MISMATCH:" + qid, qerrors)
            if complete_q:
                self.check(nrow.get("data_errors") == [], "NATIVE_SELECTION_DATA_ERRORS:" + qid, qerrors)
            qerrors += [r["request_id"] + ":" + err for r in qcalls for err in r["data_errors"]]
            self.errors += [label + ":" + qid + ":" + err for err in qerrors]
            question_results.append({"candidate_id": qid, "root_id": question["root_id"], "question_type": qtype,
                "gold_count": len(question["gold"]), "gold": question["gold"], "question": question["question"],
                "content": question[field], "expected_calls": len(expected_slots), "valid_calls": len(valid_calls),
                "exact_correct": exact_count, "exact_rate": fraction(exact_rate), "partial_mean": fraction(partial_mean),
                "permutation_means": {key: fraction(value) for key, value in means.items()},
                "selected_by_policy": accepted if not qerrors else None, "native_selected_by_policy": nrow.get("selected_by_policy"),
                "selection_reason": reason, "displayed_gold": display_gold, "data_errors": sorted(set(qerrors))})
        selected = [r["candidate_id"] for r in question_results if r["selected_by_policy"] is True]
        self.check(native.get("selected_questions") == selected, "NATIVE_SELECTED_ID_SET_MISMATCH:" + label)
        self.check(native.get("call_set_errors") == [], "NATIVE_CALL_SET_ERRORS:" + label)
        health = Counter()
        for call in run_calls:
            score, meta = call["independent_score"], call["metadata"]
            health["invalid_calls"] += not call["evaluation_valid"]
            health["parse_failures"] += score.get("scoring_status") == "PARSE_FAILURE"
            health["invalid_id_failures"] += score.get("scoring_status") == "INVALID_ID"
            health["incomplete_order_failures"] += score.get("scoring_status") == "INCOMPLETE_ORDER"
            health["truncation_failures"] += meta.get("finish_reason") in {"length", "max_tokens"}
            health["non_stop_finish_failures"] += meta.get("finish_reason") not in {"stop", "eos"}
            health["metadata_or_binding_failures"] += any(not e.startswith("UNSCORED:") for e in call["data_errors"])
        return {"solver_model": expected_name, "model_path": str(expected_path), "run_path": str(directory),
            "planned_calls": len(requests), "observed_response_files": len(records), "health": dict(health),
            "status": "VALID" if len(self.errors) == run_start else "INVALID", "questions": question_results,
            "summary": grouped_summary(question_results, run_calls), "all_select_diagnostics": all_select(question_results, run_calls),
            "failure_examples": failure_examples(question_results, run_calls)}, requests


def aggregate(questions, calls):
    valid = [r for r in calls if r["evaluation_valid"]]
    values = [Fraction(r["independent_score"]["partial_score_fraction"]) for r in valid]
    qvalues = [Fraction(q["partial_mean"]["fraction"]) for q in questions if q["partial_mean"]]
    exact = sum(r["independent_score"]["exact_match"] is True for r in valid)
    return {"questions": len(questions), "planned_calls": len(calls), "valid_calls": len(valid),
        "exact_correct": exact, "exact_rate": fraction(Fraction(exact, len(valid)) if valid else None),
        "call_weighted_partial_mean": fraction(mean(values)), "question_equal_partial_mean": fraction(mean(qvalues)),
        "selected_questions": sum(q["selected_by_policy"] is True for q in questions),
        "unresolved_questions": sum(q["selected_by_policy"] is None for q in questions)}


def grouped_summary(questions, calls):
    result = {"overall": aggregate(questions, calls)}
    for name, keyfn in (("by_type", lambda q: q["question_type"]),
        ("by_gold_count", lambda q: f"{q['question_type']}:gold{q['gold_count']}"),
        ("by_root", lambda q: q["root_id"])):
        groups = defaultdict(list)
        for q in questions:
            groups[keyfn(q)].append(q)
        result[name] = {}
        for key, rows in sorted(groups.items()):
            ids = {q["candidate_id"] for q in rows}
            result[name][key] = aggregate(rows, [call for call in calls if call["candidate_id"] in ids])
    return result


def all_select(questions, calls):
    multis = [q for q in questions if q["question_type"] == "multi_select"]
    result = {"baseline": "Always output A,B,C,D,E,F for every multi-select display.",
        "interpretation": "Gold6 exact success is also achieved by all-select and alone does not demonstrate distractor discrimination.",
        "by_gold_count": {}}
    for g in range(2, 7):
        rows = [q for q in multis if q["gold_count"] == g]
        ids = {q["candidate_id"] for q in rows}
        observed = [r for r in calls if r["candidate_id"] in ids and r["evaluation_valid"]]
        chosen = sum(len(r["independent_score"].get("display_prediction") or []) == 6 for r in observed)
        result["by_gold_count"][str(g)] = {"questions": len(rows), "always_all_select_exact_rate": fraction(Fraction(g == 6)),
            "always_all_select_partial_mean": fraction(Fraction(g == 6)), "model_all_select_calls": chosen,
            "valid_calls": len(observed), "model_all_select_rate": fraction(Fraction(chosen, len(observed)) if observed else None)}
    return result


def failure_examples(questions, calls):
    by_id = {q["candidate_id"]: q for q in questions}
    result = []
    for q in sorted(questions, key=lambda row: (row["exact_rate"]["value"] if row["exact_rate"] else -1, row["candidate_id"])):
        wrong = [r for r in calls if r["candidate_id"] == q["candidate_id"]
                 and (not r["evaluation_valid"] or r["independent_score"].get("exact_match") is False)]
        if not wrong:
            continue
        row = wrong[0]
        result.append({"candidate_id": q["candidate_id"], "root_id": q["root_id"], "question_type": q["question_type"],
            "gold_count": q["gold_count"], "question": q["question"], "content": q["content"], "gold": q["gold"],
            "request_id": row["request_id"], "raw_output": row["raw_output"], "score": row["independent_score"],
            "data_errors": row["data_errors"], "response_path": row["response_path"]})
        if len(result) >= 8:
            break
    return result


def compare(models):
    base, trained = models["base"], models["trained"]
    comparisons = {}
    for name in ("by_type", "by_gold_count", "by_root"):
        comparisons[name] = {}
        for key, b in base["summary"][name].items():
            t = trained["summary"][name][key]
            comparisons[name][key] = {"base": b, "trained": t,
                "exact_rate_delta_trained_minus_base": fraction(Fraction(t["exact_rate"]["fraction"]) - Fraction(b["exact_rate"]["fraction"])) if t["exact_rate"] and b["exact_rate"] else None,
                "partial_delta_trained_minus_base": fraction(Fraction(t["question_equal_partial_mean"]["fraction"]) - Fraction(b["question_equal_partial_mean"]["fraction"])) if t["question_equal_partial_mean"] and b["question_equal_partial_mean"] else None}
    b_by = {q["candidate_id"]: q for q in base["questions"]}
    comparisons["by_question"] = []
    for t in trained["questions"]:
        b = b_by[t["candidate_id"]]
        comparisons["by_question"].append({"candidate_id": t["candidate_id"], "root_id": t["root_id"],
            "question_type": t["question_type"], "gold_count": t["gold_count"],
            "base_exact": b["exact_rate"], "trained_exact": t["exact_rate"],
            "base_partial": b["partial_mean"], "trained_partial": t["partial_mean"],
            "exact_delta": fraction(Fraction(t["exact_rate"]["fraction"]) - Fraction(b["exact_rate"]["fraction"])) if t["exact_rate"] and b["exact_rate"] else None,
            "partial_delta": fraction(Fraction(t["partial_mean"]["fraction"]) - Fraction(b["partial_mean"]["fraction"])) if t["partial_mean"] and b["partial_mean"] else None,
            "base_selected": b["selected_by_policy"], "trained_selected": t["selected_by_policy"]})
    return comparisons


def render(result):
    def pct(value):
        return f"{100 * value['value']:.2f}%" if value else "unresolved"
    def val(value):
        return f"{value['value']:.4f}" if value else "unresolved"
    lines = ["# v104 Solver evaluation", "", f"Independent audit: **{result['status']}**.", "",
        "22 questions, 16 multi-select and 6 ordering; 360 full-context calls per model. No ground calls or new training.",
        "Base: M0 (Qwen3-4B). Trained: M_clean_policy (existing merged checkpoint). Displays, seeds, prompts and Gold are paired.", "",
        "| Group | Base exact | Trained exact | Change (pp) | Base partial | Trained partial |",
        "|---|---:|---:|---:|---:|---:|"]
    for group in ("by_type", "by_gold_count", "by_root"):
        for key, row in result["comparison"][group].items():
            b, t, delta = row["base"], row["trained"], row["exact_rate_delta_trained_minus_base"]
            lines.append(f"| {key} | {pct(b['exact_rate'])} | {pct(t['exact_rate'])} | {100*delta['value']:+.2f}" if delta else f"| {key} | {pct(b['exact_rate'])} | {pct(t['exact_rate'])} | unresolved")
            lines[-1] += f" | {val(b['question_equal_partial_mean'])} | {val(t['question_equal_partial_mean'])} |"
    lines += ["", "Exact accuracy uses valid calls; invalid-call counts are shown separately. Partial means weight questions equally; within each question they weight displays equally.",
        "Multi-select partial: TP/g − FP/(6−g), or TP/6 for Gold6. Ordering partial: (correct pairs − wrong pairs)/15.",
        "Native selection is provisional: 0 < mean partial < 1 on complete valid calls. It is not a correctness certificate or a training export.", "",
        "| Model | Calls | Valid | Parse failures | Truncation | Other invalid | Native provisional selected |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for label in ("base", "trained"):
        model = result["models"][label]; h = model["health"]
        lines.append(f"| {label} | {model['planned_calls']} | {model['summary']['overall']['valid_calls']} | {h.get('parse_failures',0)} | {h.get('truncation_failures',0)} | {h.get('invalid_calls',0)-h.get('parse_failures',0)-h.get('truncation_failures',0)} | {model['summary']['overall']['selected_questions']} |")
    lines += ["", "Always selecting all six achieves 100% exact and partial=1 on Gold6 controls, but 0% exact and partial=0 on Gold2–5. Gold6 success alone cannot establish distractor discrimination.", "",
        "| Model | Gold | Model all-select calls / valid |",
        "|---|---:|---:|"]
    for label in ("base", "trained"):
        for gold, row in result["models"][label]["all_select_diagnostics"]["by_gold_count"].items():
            lines.append(f"| {label} | {gold} | {row['model_all_select_calls']}/{row['valid_calls']} |")
    lines += ["", "| Candidate | Root | Type / Gold | Base exact | Trained exact | Base partial | Trained partial | Selected base / trained |",
        "|---|---|---|---:|---:|---:|---:|---|"]
    for q in result["comparison"]["by_question"]:
        lines.append(f"| {q['candidate_id']} | {q['root_id']} | {q['question_type']} / {q['gold_count']} | {pct(q['base_exact'])} | {pct(q['trained_exact'])} | {val(q['base_partial'])} | {val(q['trained_partial'])} | {q['base_selected']} / {q['trained_selected']} |")
    for label in ("base", "trained"):
        lines += ["", f"## {label} failure examples", ""]
        for row in result["models"][label]["failure_examples"]:
            score = row["score"]
            lines.append(f"- `{row['candidate_id']}` ({row['question_type']}, Gold{row['gold_count']}): `{row['raw_output']}`, displayed prediction `{score.get('display_prediction')}`, partial `{score.get('partial_score')}`. Raw record: `{row['response_path']}`.")
    if result["data_errors"]:
        lines += ["", "## Audit errors", ""] + ["- " + e for e in result["data_errors"]]
    lines += ["", "Limits: this audit verifies frozen artifacts, execution and scoring; it does not independently establish semantic Gold correctness. Native response artifacts store model_path/solver_model but omit a separate raw model_identifier; the native live gate checked that identifier before saving. The paired comparison describes this purposively selected 22-question batch from training-exposed roots. It is not an independent holdout or a generalization estimate.", ""]
    return "\n".join(lines)


def write_artifacts(directory, artifacts):
    """Atomic intent check, exclusive file creation; no overwrite or update."""
    paths = [directory / name for name in artifacts]
    if any(path.exists() for path in paths):
        raise FileExistsError("AUDIT_ARTIFACT_ALREADY_EXISTS:" + ",".join(str(p) for p in paths if p.exists()))
    for name, value in artifacts.items():
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        with (directory / name).open("x", encoding="utf-8") as handle:
            handle.write(content)


def self_test():
    q = {"question_type": "multi_select", "options": dict(zip("ABCDEF", ["one", "two", "three", "four", "five", "six"])), "gold": ["B", "D"]}
    variant = {"display_to_content": {key: content_id(value) for key, value in q["options"].items()}}
    assert replay_score(q, variant, "<box>A,B,C,D,E,F</box>")["partial_score"] == "0"
    assert replay_score(q, variant, "<box>B,D</box>")["exact_match"] is True
    assert replay_score(q, variant, "<box>B,B</box>")["scoring_status"] == "PARSE_FAILURE"
    assert replay_score(q, variant, "<box>G</box>")["scoring_status"] == "INVALID_ID"
    assert replay_score(q, variant, "<box>B</box><box>D</box>")["scoring_status"] == "PARSE_FAILURE"
    q["gold"] = list("ABCDEF")
    assert replay_score(q, variant, "<box>A,B,C,D,E</box>")["partial_score"] == "5/6"
    order = {"question_type": "ordering", "items": {f"I{i}": str(i) for i in range(1,7)}, "gold": [f"I{i}" for i in range(1,7)]}
    v = {"display_to_content": {key: content_id(value) for key, value in order["items"].items()}}
    assert replay_score(order, v, "<box>I6,I5,I4,I3,I2,I1</box>")["partial_score"] == "-1"
    assert replay_score(order, v, "<box>I1,I2,I3,I4,I6,I5</box>")["partial_score"] == "13/15"
    assert replay_score(order, v, "<box>I1,I2</box>")["scoring_status"] == "INCOMPLETE_ORDER"
    print(json.dumps({"self_test": "PASS", "cases": 9}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    directory = args.directory.resolve()
    reserved = ["AUDIT_RESULTS.json", "REPORT.md", "SCORED_CALLS.json", "SELECTED_CANDIDATES.json", "EVALUATION_COMPLETE_MANIFEST.json"]
    if any((directory / name).exists() for name in reserved):
        raise FileExistsError("AUDIT_ALREADY_CREATED_USE_NEW_VERSION_DIRECTORY")
    pool = read(directory / "EVALUATION_POOL.json")
    audit = Audit(directory)
    source_batch = audit.source_batch(pool)
    adapter = audit.adapter(pool)
    models, plans = {}, {}
    for label in ("trained", "base"):
        models[label], plans[label] = audit.run(label, pool)
    audit.check(plans["trained"] == plans["base"], "PAIRED_REQUEST_PLANS_DIFFER")
    result = {"audit_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VALID" if not audit.errors else "INVALID", "source_batch": source_batch, "adapter": adapter,
        "evaluation_pool_sha256": sha(directory / "EVALUATION_POOL.json"),
        "audit_script_sha256": sha(Path(__file__)), "full_context_only": True, "ground_calls": 0,
        "training_started": False, "training_promotion": False, "new_generation_calls": 0,
        "sampling": "Purposively selected training-exposed roots; not an independent holdout.",
        "independent_holdout": False,
        "models": models, "comparison": compare(models), "checked_file_hashes": audit.hashes,
        "data_errors": sorted(set(audit.errors))}
    trained_by = {q["candidate_id"]: q for q in models["trained"]["questions"]}
    selected = [q for q in pool["questions"] if trained_by[q["candidate_id"]]["selected_by_policy"] is True]
    controls = [q for q in selected if q["question_type"] == "multi_select" and len(q["gold"]) == 6]
    artifacts = {"AUDIT_RESULTS.json": result,
        "SCORED_CALLS.json": {"audit_version": VERSION, "count": len(audit.calls), "calls": audit.calls},
        "SELECTED_CANDIDATES.json": {"status": "PROVISIONAL_SOLVER_SELECTION_NO_TRAINING" if not audit.errors else "INVALID_AUDIT_NO_PROMOTION",
            "basis_model": "M_clean_policy", "policy": "0 < equal-display full partial mean < 1, complete valid evaluation",
            "candidate_count": len(selected) if not audit.errors else 0, "questions": selected if not audit.errors else [],
            "gold6_controls_selected_by_native_policy": [q["candidate_id"] for q in controls],
            "training_started": False, "training_promotion": False, "sft_export_executed": False},
        "REPORT.md": render(result)}
    write_artifacts(directory, artifacts)
    if not audit.errors:
        files = [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "EVALUATION_COMPLETE_MANIFEST.json"
            and "__pycache__" not in path.parts]
        write_artifacts(directory, {"EVALUATION_COMPLETE_MANIFEST.json": {"status": "PAIRED_SOLVER_EVALUATION_AUDITED_COMPLETE",
            "created_at_utc": result["created_at_utc"], "files": files, "self_hash_excluded": True,
            "training_started": False, "training_promotion": False}})
    print(json.dumps({"status": result["status"], "audited_calls": len(audit.calls),
        "provisional_trained_selected": len(selected) if not audit.errors else 0,
        "errors": len(set(audit.errors))}, ensure_ascii=False))
    if audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

