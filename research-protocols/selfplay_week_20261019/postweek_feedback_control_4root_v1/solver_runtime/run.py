"""FBC1 paired-root B1 runtime: public preparation, sealed raw, then private scoring.

No question is authored here. Direct decoding follows the proven native 32-token
contract, with a local I1..I4 -> A..D label adapter for ordering. Visible reasons
reuse the reviewed v5 reason interface and are observable explanations, not a
claim about faithful internal deliberation. There is no retry or overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
WEEK = EXPERIMENT.parent
HELPER_PATH = WEEK / "postweek_question_quality_pilot_v1/run.py"
ADAPTER_PATH = WEEK / "day2/solver/adapter.py"
ENDPOINT = WEEK / "day5_training_v1/evaluation/models/B1/EXECUTION_MANIFEST.json"
V5_REASON = WEEK / "postweek_live_extension_v5/reason_diagnostic_v5/run.py"
V51_REASON = WEEK / "postweek_live_extension_v5/order_operation_revision_v1/reason_retest_v2/run.py"
MODEL_LEN = 16384
MAX_TOKENS = {"direct": 32, "reason": 3072}
CONDITIONS = ("GENERIC", "TARGETED")
PUBLIC_KEYS = {"candidate_id", "root_id", "condition", "question_type", "task_operation", "question", "source_binding", "display"}
REQUEST_KEYS = PUBLIC_KEYS | {"request_id", "branch", "context_mode", "cohort", "labels", "sampling_seed", "source_turns", "prompt_path", "schema_path", "prompt_tokens", "max_output_tokens"}
OPERATIONS = {
    "MC_SOURCE_STATE": "Evaluate each statement about the records and their status against the supplied dialogue.",
    "MC_ENTITY_BINDING": "Evaluate the relationships and entity bindings asked about in each displayed statement.",
    "MC_ENTITY_STATE": "Evaluate the entity and state relationships asked about in each displayed statement.",
    "MC_STATE_UPDATE": "Evaluate each statement against the relevant updates in the supplied dialogue.",
    "MC_LINKED_RECORDS": "Evaluate the linked details in each statement against the supplied dialogue.",
    "MC_FULL_RECORD_AUDIT": "Evaluate every displayed statement independently against the supplied dialogue.",
    "MC_STATE_UPDATE_CONJUNCTION": "Evaluate every component of each displayed statement against the relevant state updates, speaker attribution and completion status in the supplied dialogue.",
    "MC_CONCLUSION_EVIDENCE": "Select statements that provide source evidence for the exact conclusion requested. A true but irrelevant statement is not sufficient evidence for that conclusion.",
    "MC_JOINT_CONSTRAINT": "Check every linked constraint in each statement, including the entity, speaker, time, scope and requested versus confirmed status. All components must be supported.",
    "MC_RECORD_AUDIT_CONTROL": "Evaluate every displayed statement independently against the supplied dialogue and the precise question.",
    "ORDER_EXPLICIT_TIME": "Order the clock times or dates required by the question, performing stated duration calculations where needed. Distinguish requested cutoffs from actual events. Do not sort by where evidence is mentioned in the dialogue.",
    "ORDER_CLOCK_VALUE": "Sort the reported clock readings or requested clock cutoffs specified by the question. A clock value need not describe a completed event. Do not substitute dialogue mention order, infer travel completion, or combine unrelated records into one journey.",
    "ORDER_EXPLICIT_CLOCK_VALUE": "Sort the reported clock readings or requested clock cutoffs specified by the question. A clock value need not describe a completed event. Do not substitute dialogue mention order, infer travel completion, or combine unrelated records into one journey.",
    "ORDER_ENQUIRY_PROGRESSION": "Reconstruct the enquiry or record disclosure progression requested by the question, grounded in the supplied dialogue. Distinguish statements about constraints, replies and unresolved state from actual journeys or completed visits. This operation is not numeric clock-time sorting."
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


helper = module("fbc1_proven_quality_helpers", HELPER_PATH)
adapter = module("fbc1_proven_native_adapter", ADAPTER_PATH)
read, save, bind, check = helper.read, helper.save, helper.bind, helper.check
diag = helper.diagnostic
diag.OPERATIONS = {**diag.OPERATIONS, **OPERATIONS}


def all_bindings(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            yield {key: value[key] for key in ("path", "sha256", "bytes")}
        else:
            for child in value.values():
                yield from all_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_bindings(child)


def unique_bindings(rows):
    result = {}
    for row in rows:
        if row["path"] in result:
            assert row == result[row["path"]], "CONFLICTING_BINDING"
        result[row["path"]] = row
    return list(result.values())


def safe_id(value):
    assert isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]+", value) and value not in (".", ".."), "UNSAFE_ID"
    return value


def code_paths():
    return [Path(__file__), HELPER_PATH, helper.DIAGNOSTIC_PATH, helper.CAPTURE_PATH,
            diag.REPAIR_PATH, helper.GATE_PATH, ADAPTER_PATH, V5_REASON, V51_REASON]


def environment(endpoint):
    assert endpoint["model_role"] == "B1", "B1_ENDPOINT_REQUIRED"
    assert Path(sys.prefix).resolve() == Path(endpoint["python_prefix"]).resolve(), "USE_PROVEN_LOCAL4B_VENV"
    assert endpoint["decoder_contract"]["max_tokens"] == MAX_TOKENS["direct"]
    assert endpoint["decoder_contract"]["temperature"] == .8 and endpoint["decoder_contract"]["top_p"] == .9
    assert endpoint["decoder_contract"]["enable_thinking"] is False


def model_bindings(endpoint):
    result = endpoint["model_bindings"] + endpoint["tokenizer_bindings"] + endpoint["code_bindings"]
    for row in result:
        check(row)
    return result


def direct_choices(question_type, labels):
    if question_type == "multi_select":
        choices = adapter.choices(question_type, dict.fromkeys(labels))
        assert len(choices) == 63
        return choices
    # Scoped syntax adapter: preserve native finite-permutation constraints and
    # comma/box syntax while using the same displayed labels in both branches.
    native_ids = ["I1", "I2", "I3", "I4"]
    native = adapter.choices("ordering", dict.fromkeys(native_ids))
    conversion = dict(zip(native_ids, labels))
    choices = ["<box>" + ",".join(conversion[key] for key in value[5:-6].split(",")) + "</box>" for value in native]
    assert len(choices) == 24 and len(set(choices)) == 24
    return choices


def direct_prompt(tokenizer, row, source):
    field = "options" if row["question_type"] == "multi_select" else "items"
    prompt = adapter.direct_prompt(tokenizer, row, {field: row["display"]}, source)
    if row["question_type"] == "ordering":
        assert prompt.count("e.g. <box>I2,I4,I1,I3</box>.") == 1, "NATIVE_ORDERING_EXAMPLE_CHANGED"
        prompt = prompt.replace("e.g. <box>I2,I4,I1,I3</box>.", "e.g. <box>B,D,A,C</box>.")
    return prompt


def sampling_seed(question, branch):
    # Deterministic request seeds never depend on Gold. The candidate identifier
    # remains only an artifact identifier and is never in prompts.
    material = "fbc1:20261023:" + question["candidate_id"] + ":" + branch
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big") % 2147483647


def read_export(path):
    frozen = read(path)
    assert frozen["status"] == "FROZEN_BEFORE_SOLVER", "QUESTION_FREEZE_REQUIRED"
    assert frozen["planned_slots"] == 56 and len(frozen["planned_root_ids"]) == 4
    assert len(set(frozen["planned_root_ids"])) == 4
    assert frozen["content_review_status"] == "PASS", "INDEPENDENT_CONTENT_REVIEW_REQUIRED"
    assert frozen["solver_score_filtering"] is False and frozen["full_source_no_truncation"] is True
    check(frozen["public_binding"])
    check(frozen["question_freeze_binding"])
    # Private bytes are not opened or hashed during public preparation/run.
    for row in frozen.get("bindings", []):
        if row != frozen["private_binding"]:
            check(row)
    payload = read(frozen["public_binding"]["path"])
    assert set(payload) == {"questions"}, "PUBLIC_WRAPPER_ALLOWLIST"
    questions = payload["questions"]
    assert 0 < len(questions) <= 56 and len({q["candidate_id"] for q in questions}) == len(questions)
    return frozen, questions


def prepare(args):
    out = args.output.resolve()
    assert not out.exists(), "NO_PREPARATION_OVERWRITE"
    frozen, questions = read_export(args.export_freeze)
    endpoint = read(args.endpoint)
    environment(endpoint)
    bound_model = model_bindings(endpoint)
    from transformers import AutoTokenizer
    import xgrammar as xgr
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    staged = []
    for question in questions:
        assert set(question) == PUBLIC_KEYS, "PUBLIC_FIELD_ALLOWLIST_DRIFT"
        safe_id(question["candidate_id"]); safe_id(question["root_id"])
        assert question["root_id"] in frozen["planned_root_ids"] and question["condition"] in CONDITIONS
        assert question["question_type"] in ("multi_select", "ordering")
        assert question["task_operation"] in diag.OPERATIONS
        diag.operation_text(question)
        expected_labels = list("ABCDEF" if question["question_type"] == "multi_select" else "ABCD")
        assert list(question["display"]) == expected_labels, "DISPLAY_LABEL_ORDER_MUST_BE_FROZEN"
        assert len(set(question["display"].values())) == len(expected_labels), "DUPLICATE_DISPLAY_TEXT"
        assert all(isinstance(v, str) and v.strip() for v in question["display"].values())
        check(question["source_binding"])
        expected_source = args.source_root.resolve() / question["root_id"] / question["condition"] / "FINAL_SOURCE.json"
        assert Path(question["source_binding"]["path"]).resolve() == expected_source, "UNEXPECTED_FINAL_SOURCE_PATH"
        source = read(expected_source)
        assert source["root_id"] == question["root_id"]
        if "condition" in source:
            assert source["condition"] == question["condition"]
        refs = [turn["turn_ref"] for turn in source["turns"]]
        assert refs and len(refs) == len(set(refs)), "INVALID_SOURCE_TURN_REFS"
        for branch in ("direct", "reason"):
            row = dict(question)
            row.update(request_id=question["candidate_id"] + "__B1_FULL_" + branch,
                branch=branch, context_mode="FULL", cohort="FBC1", labels=expected_labels,
                sampling_seed=sampling_seed(question, branch), source_turns=len(refs), max_output_tokens=MAX_TOKENS[branch])
            prompt = direct_prompt(tokenizer, row, source) if branch == "direct" else diag.make_prompt(tokenizer, row, row["display"], source)
            assert all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source["turns"]), "INCOMPLETE_SOURCE_NO_TRUNCATION"
            assert question["candidate_id"] not in prompt, "CANDIDATE_ID_MUST_NOT_REVEAL_GOLD_ARITY"
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            assert len(ids) + row["max_output_tokens"] <= MODEL_LEN, "CONTEXT_OVERFLOW_NO_TRUNCATION:" + row["request_id"]
            if branch == "direct":
                schema = {"choice": direct_choices(row["question_type"], expected_labels)}
            else:
                schema = diag.repair.schema(expected_labels, refs, row["question_type"])
                compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
            staged.append((row, prompt, ids, schema))
    assert len(staged) == 2 * len(questions) <= 112
    bindings = [bind(path) for path in code_paths()] + [bind(args.endpoint), bind(args.export_freeze)]
    bindings += [frozen["public_binding"], frozen["question_freeze_binding"]] + bound_model
    rows, token_ids = [], {}
    for row, prompt, ids, schema in staged:
        rid = row["request_id"]
        prompt_path, schema_path = out / "prompts" / (rid + ".txt"), out / "schemas" / (rid + ".json")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with prompt_path.open("x", encoding="utf-8") as stream:
            stream.write(prompt); stream.flush(); os.fsync(stream.fileno())
        save(schema_path, schema)
        row.update(prompt_path=str(prompt_path), schema_path=str(schema_path), prompt_tokens=len(ids))
        assert set(row) == REQUEST_KEYS
        rows.append(row); token_ids[rid] = ids
        bindings += [bind(prompt_path), bind(schema_path), row["source_binding"]]
    save(out / "PUBLIC_REQUESTS.json", {"requests": rows})
    save(out / "PROMPT_TOKEN_IDS.json", token_ids)
    bindings += [bind(out / "PUBLIC_REQUESTS.json"), bind(out / "PROMPT_TOKEN_IDS.json")]
    prep = {"status": "PREPARED_CPU_ONLY_GPU_NOT_RUN", "requests": len(rows), "questions": len(questions),
        "planned_slots": 56, "planned_root_ids": frozen["planned_root_ids"], "conditions": list(CONDITIONS),
        "fixed_order": [r["request_id"] for r in rows], "bindings": unique_bindings(bindings),
        "private_binding": frozen["private_binding"], "export_freeze_binding": bind(args.export_freeze),
        "endpoint_binding": bind(args.endpoint), "max_model_len": MODEL_LEN, "max_output_tokens": MAX_TOKENS,
        "temperature": .8, "top_p": .9, "native_hidden_thinking": False, "source_truncation": False,
        "automatic_retry": False, "generation_reads_gold": False, "model_role": "B1", "model_directory": endpoint["model_directory"],
        "direct_ordering_label_adapter": "I1..I4 to A..D bijection; output example only changed; same 24 finite permutations",
        "reason_claim": "observable concise explanations; no faithful internal CoT claim", "training_admission": False,
        "max_prompt_tokens": max(row["prompt_tokens"] for row in rows),
        "min_context_headroom": min(MODEL_LEN-row["prompt_tokens"]-row["max_output_tokens"] for row in rows)}
    save(out / "PREPARATION.json", prep)
    print(json.dumps({key: prep[key] for key in ("status", "requests", "questions", "max_prompt_tokens", "min_context_headroom")}))


def verify(args, require_review=False):
    out = args.output.resolve()
    prep = read(out / "PREPARATION.json")
    for row in prep["bindings"]:
        assert row != prep["private_binding"], "PRIVATE_BYTES_IN_PUBLIC_WORKER_BINDINGS"
        check(row)
    rows = read(out / "PUBLIC_REQUESTS.json")["requests"]
    tokens = read(out / "PROMPT_TOKEN_IDS.json")
    ids = [r["request_id"] for r in rows]
    assert len(rows) == prep["requests"] == len(set(ids)) == prep["questions"]*2 <= 112
    assert ids == prep["fixed_order"] and set(tokens) == set(ids)
    assert len({r["candidate_id"] for r in rows}) == prep["questions"]
    for row in rows:
        assert set(row) == REQUEST_KEYS and row["context_mode"] == "FULL"
        assert row["branch"] in MAX_TOKENS and row["condition"] in CONDITIONS
        assert len(tokens[row["request_id"]]) == row["prompt_tokens"]
        assert row["max_output_tokens"] == MAX_TOKENS[row["branch"]]
        assert row["prompt_tokens"]+row["max_output_tokens"] <= MODEL_LEN
    for cid in {r["candidate_id"] for r in rows}:
        pair = [r for r in rows if r["candidate_id"] == cid]
        assert len(pair) == 2 and {r["branch"] for r in pair} == {"direct", "reason"}
        assert all(pair[0][key] == pair[1][key] for key in PUBLIC_KEYS), "BRANCH_QUESTION_OR_DISPLAY_DRIFT"
    if require_review:
        review = read(args.preflight_review)
        assert review["status"] == "PASS_CPU_INTERFACE_ONLY", "INDEPENDENT_PREFLIGHT_REQUIRED"
        assert review.get("reviewer_id") and review.get("reviewer_id") != review.get("author_id"), "INDEPENDENT_PREFLIGHT_REVIEWER_REQUIRED"
        for row in review["bindings"]:
            check(row)
        required = {str(Path(__file__)), str(out / "PREPARATION.json"), str(out / "PUBLIC_REQUESTS.json")}
        assert required <= {r["path"] for r in review["bindings"]}, "PREFLIGHT_REVIEW_CHAIN_INCOMPLETE"
    return prep, rows, tokens


def run(args):
    assert args.execute_gpu, "EXPLICIT_GPU_FLAG_REQUIRED"
    out = args.output.resolve()
    assert not (out / "STARTED.json").exists() and not (out / "RAW_RESPONSE_FREEZE.json").exists(), "NO_RERUN_OR_RETRY"
    prep, rows, tokens = verify(args, True)
    endpoint = read(prep["endpoint_binding"]["path"])
    environment(endpoint); model_bindings(endpoint)
    for key, value in {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ[key] = value
    backend_path = next(Path(b["path"]) for b in endpoint["code_bindings"] if Path(b["path"]).name == "vllm_backends.py")
    sys.path.insert(0, str(backend_path.parents[3]))
    from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
    from vllm import SamplingParams, sampling_params
    save(out / "STARTED.json", {"time_unix": time.time(), "preparation": bind(out / "PREPARATION.json"),
        "preflight_binding": bind(args.preflight_review), "private_binding": prep["private_binding"],
        "requests": len(rows), "request_ids": prep["fixed_order"], "gold_read": False, "automatic_retry": False})
    backend, failure, submitted, errors = None, None, set(), []
    started = time.perf_counter()
    try:
        backend = VLLMSolverBackend(model_path=Path(endpoint["model_directory"]), tokenizer_path=Path(endpoint["tokenizer_directory"]),
            max_model_len=MODEL_LEN, temperature=.8, top_p=.9, max_tokens=MAX_TOKENS["reason"], gpu_memory_utilization=.65)
        assert str(backend.model_identifier) == endpoint["model_directory"]
        for row in rows:
            rid, path = row["request_id"], out / "raw" / (row["request_id"] + ".json")
            try:
                params = {"temperature": .8, "top_p": .9, "max_tokens": row["max_output_tokens"], "seed": row["sampling_seed"]}
                schema = read(row["schema_path"])
                if row["branch"] == "direct":
                    params.update(stop=["</box>"], include_stop_str_in_output=True,
                        structured_outputs=sampling_params.StructuredOutputsParams(choice=schema["choice"]))
                else:
                    params["structured_outputs"] = sampling_params.StructuredOutputsParams(json=schema)
                prompt = Path(row["prompt_path"]).read_text()
                save(out / "submitted" / (rid + ".json"), {"request_id": rid, "time_unix": time.time(), "gold_read": False})
                submitted.add(rid)
                results = backend.engine.generate([prompt], [SamplingParams(**params)], use_tqdm=False)
                raw = helper.preserve_response(row, results, endpoint)
                raw.update(branch=row["branch"], experiment_condition=row["condition"], context_mode="FULL")
                save(path, raw)
                problems = helper.capture.response_errors(raw, prompt, tokens[rid])
                save(out / "raw_validation" / (rid + ".json"), {"request_id": rid, "errors": problems, "raw_binding": bind(path)})
                if problems:
                    errors.append({"request_id": rid, "errors": problems})
                print("RETURNED " + rid, flush=True)
            except Exception:
                error = traceback.format_exc(); errors.append({"request_id": rid, "error": error})
                if not path.exists():
                    save(path, {"request_id": rid, "status": "BACKEND_ERROR" if rid in submitted else "NOT_SUBMITTED", "raw_output": None, "failure": error})
    except BaseException:
        failure = traceback.format_exc()
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                failure = (failure or "") + traceback.format_exc()
        for row in rows:
            path = out / "raw" / (row["request_id"] + ".json")
            if not path.exists():
                save(path, {"request_id": row["request_id"], "status": "BACKEND_ERROR" if row["request_id"] in submitted else "NOT_SUBMITTED", "raw_output": None, "failure": failure})
        paths = sorted((out / "raw").glob("*.json"))
        save(out / "RAW_RESPONSE_FREEZE.json", {"status": "ALL_REQUESTS_TERMINAL_RAW_SEALED", "requests": len(rows), "request_ids": prep["fixed_order"],
            "started_binding": bind(out / "STARTED.json"), "wall_seconds": time.perf_counter()-started, "submitted_requests": len(submitted),
            "returned_requests": sum(read(p)["status"] == "RETURNED" for p in paths), "failure": failure, "request_errors": errors,
            "gold_read": False, "automatic_retry": False, "bindings": [bind(p) for p in paths],
            "validation_bindings": [bind(p) for p in sorted((out / "raw_validation").glob("*.json"))]})
    if failure or errors:
        raise RuntimeError("Frozen failures preserved. No automatic retry; inspect RAW_RESPONSE_FREEZE.json")


def sealed_inputs(args):
    prep, rows, tokens = verify(args)
    out = args.output.resolve()
    frozen = read(out / "RAW_RESPONSE_FREEZE.json")
    expected = {str(out / "raw" / (r["request_id"] + ".json")) for r in rows}
    assert frozen["requests"] == len(rows) == len(frozen["bindings"])
    assert frozen["request_ids"] == prep["fixed_order"]
    assert {r["path"] for r in frozen["bindings"]} == expected
    assert {str(p.resolve()) for p in (out / "raw").glob("*.json")} == expected
    for row in frozen["bindings"] + frozen["validation_bindings"]:
        check(row)
    assert frozen["started_binding"] == bind(out / "STARTED.json")
    start = read(out / "STARTED.json")
    assert start["preparation"] == bind(out / "PREPARATION.json") and start["private_binding"] == prep["private_binding"]
    check(start["preflight_binding"])
    # This is the first read/hash of the private scoring file by this runtime.
    check(prep["private_binding"])
    private = read(prep["private_binding"]["path"])
    for row in private.get("bindings", []):
        check(row)
    items = private["questions"]
    assert len(items) == prep["questions"] and {r["candidate_id"] for r in items} == {r["candidate_id"] for r in rows}
    return prep, rows, tokens, frozen, {r["candidate_id"]: r for r in items}


def analyze(args):
    prep, rows, tokens, frozen, private = sealed_inputs(args)
    out = args.output.resolve()
    records, scores = [], []
    for row in rows:
        rid = row["request_id"]
        path = out / "raw" / (rid + ".json")
        raw = read(path)
        record = {key: row[key] for key in ("request_id", "candidate_id", "root_id", "condition", "branch", "question_type", "task_operation")}
        record.update(request=row, raw_binding=bind(path), semantic_review="PENDING" if row["branch"] == "reason" else "NOT_APPLICABLE", feedback_eligible=False)
        answer, valid = [], False
        try:
            assert raw["request_id"] == rid and raw["status"] == "RETURNED", "BACKEND_OR_SUBMISSION_FAILURE"
            assert not helper.capture.response_errors(raw, Path(row["prompt_path"]).read_text(), tokens[rid]), "RAW_RESPONSE_DRIFT"
            assert raw["finish_reason"] == "stop", "TRUNCATED_OR_NONSTOP_COMPLETION"
            if row["branch"] == "direct":
                assert raw["raw_output"] in read(row["schema_path"])["choice"], "DIRECT_OUTSIDE_FINITE_CHOICES"
                answer = raw["raw_output"][5:-6].split(",")
                parsed = {"final_answer": raw["raw_output"]}
                consistency = {"reason_review_applicable": False}
            else:
                # Final answer syntax is assessed independently of whether
                # explanations/citations pass their stricter interface.
                parsed = json.loads(raw["raw_output"])
                match = re.fullmatch(r"<box>([A-F](?: [A-F]){0,5})</box>", parsed["final_answer"])
                assert match, "REASON_FINAL_BOX_REQUIRED"
                answer = match.group(1).split()
                assert len(answer) == len(set(answer)) and set(answer) <= set(row["labels"]), "INVALID_FINAL_LABEL_SET"
                if row["question_type"] == "ordering":
                    assert set(answer) == set(row["labels"]), "ORDERING_FINAL_INCOMPLETE"
                consistency = {"semantic_correctness": "NOT_INFERRED", "speaker_attribution": "PENDING_INDEPENDENT_REVIEW", "citation_sufficiency": "PENDING_INDEPENDENT_REVIEW"}
                try:
                    checked, checked_answer = diag.repair.validate(raw["raw_output"], row, read(row["schema_path"]))
                    assert checked_answer == answer
                    if row["question_type"] == "ordering":
                        assert all(v["turn_refs"] for v in checked["item_evidence"].values()), "NO_ANCHOR_FOR_ORDERING"
                    consistency.update(diag.mechanical_consistency(row, checked, answer))
                    record["reason_structure_valid"] = True
                except Exception as error:
                    record.update(reason_structure_valid=False, reason_structure_error=str(error))
                items = parsed.get("item_evidence", {})
                usable_items = [item for item in items.values() if isinstance(item, dict)] if isinstance(items, dict) else []
                consistency.update(displayed_items=len(row["labels"]), parsed_evidence_items=len(usable_items),
                    items_with_nonempty_reasons=sum(isinstance(item.get("reason"), str) and bool(item["reason"].strip()) for item in usable_items),
                    items_with_any_reference=sum(bool(item.get("turn_refs")) for item in usable_items),
                    supported_items_without_reference=sum(item.get("judgment") == "supported" and not item.get("turn_refs") for item in usable_items),
                    reference_presence_is_not_semantic_support=True)
            valid = True
            record.update(status="STRUCTURALLY_VALID", parsed=parsed, answer_labels=answer, mechanical_consistency=consistency)
        except Exception as error:
            record.update(status="INVALID", error=str(error))
        presentation = private[row["candidate_id"]]
        assert presentation["question_type"] == row["question_type"]
        assert set(presentation["display_to_canonical"]) == set(row["labels"])
        assert len(set(presentation["display_to_canonical"].values())) == len(row["labels"])
        expected_gold = [k for k in row["labels"] if presentation["display_to_canonical"][k] in presentation["canonical_gold"]] if row["question_type"] == "multi_select" else [next(k for k,v in presentation["display_to_canonical"].items() if v == key) for key in presentation["canonical_gold"]]
        assert (set(presentation["display_gold"]) == set(expected_gold) if row["question_type"] == "multi_select" else presentation["display_gold"] == expected_gold), "PRIVATE_GOLD_REBINDING_DRIFT"
        score = helper.score_record(row, presentation, answer, valid)
        score.update(branch=row["branch"], context_mode="FULL", raw_binding=bind(path), invalid_error=record.get("error"),
            reason_structure_valid=record.get("reason_structure_valid"), reason_structure_error=record.get("reason_structure_error"),
            mc6_control=row["question_type"] == "multi_select" and len(presentation["canonical_gold"]) == 6)
        records.append(record); scores.append(score)
    cells = [{"condition": c, "branch": b, **helper.aggregate([s for s in scores if s["condition"] == c and s["branch"] == b])} for c in CONDITIONS for b in MAX_TOKENS]
    roots = [{"root_id": root, "condition": c, "branch": b, "planned_slots": 7,
        **helper.aggregate([s for s in scores if s["root_id"] == root and s["condition"] == c and s["branch"] == b])}
        for root in prep["planned_root_ids"] for c in CONDITIONS for b in MAX_TOKENS]
    operation_cells = [{"root_id": root, "condition": c, "branch": b, "task_operation": op,
        **helper.aggregate([s for s in scores if s["root_id"] == root and s["condition"] == c and s["branch"] == b and s["task_operation"] == op])}
        for root in prep["planned_root_ids"] for c in CONDITIONS for b in MAX_TOKENS for op in sorted({s["task_operation"] for s in scores})]
    gold_cells = [{"condition": c, "branch": b, "gold_count": count, "all_correct_control": count == 6,
        **helper.aggregate([s for s in scores if s["condition"] == c and s["branch"] == b and s["gold_count"] == count])}
        for c in CONDITIONS for b in MAX_TOKENS for count in range(2, 7)]
    reason_records = [r for r in records if r["branch"] == "reason"]
    mechanical = {"requests": len(reason_records), "final_syntax_valid": sum(r["status"] == "STRUCTURALLY_VALID" for r in reason_records),
        "reason_structure_valid": sum(r.get("reason_structure_valid") is True for r in reason_records),
        "items_with_any_reference": sum(r.get("mechanical_consistency", {}).get("items_with_any_reference", 0) for r in reason_records),
        "supported_items_without_reference": sum(r.get("mechanical_consistency", {}).get("supported_items_without_reference", 0) for r in reason_records),
        "mc_answer_vs_supported_labels_mismatches": sum(r.get("mechanical_consistency", {}).get("answer_matches_supported_labels") is False for r in reason_records),
        "semantic_citation_correctness": "PENDING_INDEPENDENT_REVIEW", "speaker_attribution_correctness": "PENDING_INDEPENDENT_REVIEW",
        "counts_do_not_certify_reason_or_reference_semantics": True}
    summary = {"planned_paired_roots": 4, "independent_unit": "paired root", "planned_question_slots": 56,
        "admitted_questions": prep["questions"], "held_or_failed_slots": 56-prep["questions"], "requests": len(rows),
        "cells": cells, "root_cells": roots, "root_operation_cells": operation_cells, "mc_gold_cells": gold_cells,
        "visible_reason_mechanical_metrics": mechanical, "final_answer_validity_independent_of_reason_structure": True,
        "semantic_review": "PENDING", "speaker_attribution_review": "PENDING",
        "citation_sufficiency_review": "PENDING", "formal_weakness_claim": False, "training_benefit_claim": False,
        "solver_score_filtering": False, "gold_read_after_all_raw_sealed": True, "raw_freeze_binding": bind(out / "RAW_RESPONSE_FREEZE.json")}
    save(out / "SCORES.json", {"records": scores, "gold_read_after_all_raw_sealed": True, "raw_freeze_binding": bind(out / "RAW_RESPONSE_FREEZE.json")})
    save(out / "REASON_REVIEW_PACKET.json", {"requests": [r for r in records if r["branch"] == "reason"], "semantic_review": "PENDING", "feedback_eligible": False})
    save(out / "ALL_STRUCTURAL_RECORDS.json", {"records": records})
    save(out / "ANALYSIS_SUMMARY.json", summary)
    print(json.dumps(summary))


def selfcheck(args):
    # CPU interface check only: no LLM/backend instantiation and no questions
    # authored or written. In-memory labels/schema are protocol fixtures.
    endpoint = read(args.endpoint); environment(endpoint)
    bindings = model_bindings(endpoint)
    from transformers import AutoTokenizer
    import xgrammar as xgr
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    mc = direct_choices("multi_select", list("ABCDEF"))
    order = direct_choices("ordering", list("ABCD"))
    assert "<box>A,C</box>" in mc and "<box>B,D,A,C</box>" in order
    assert set(order) == {"<box>"+",".join(p)+"</box>" for p in itertools.permutations("ABCD")}
    for kind, labels in (("multi_select", list("ABCDEF")), ("ordering", list("ABCD"))):
        schema = diag.repair.schema(labels, ["CHECK:T001", "CHECK:T002"], kind)
        compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
    report = {"status": "PASS_CPU_RUNTIME_CODE_ONLY_INPUTS_PENDING", "gpu_started": False, "questions_authored": 0,
        "model_role": "B1", "model_hashes_checked": len(endpoint["model_bindings"]),
        "native_direct_choices_mc": len(mc), "native_direct_choices_ordering": len(order),
        "reason_schema_compiled": True, "native_hidden_thinking": False,
        "actual_question_preflight_required": True, "independent_preflight_review_required_before_gpu": True,
        "bindings": unique_bindings([bind(p) for p in code_paths()] + [bind(args.endpoint)] + bindings)}
    save(args.selfcheck_output, report)
    print(json.dumps({key:value for key,value in report.items() if key != "bindings"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("selfcheck", "prepare", "verify", "run", "analyze"))
    parser.add_argument("--export-freeze", type=Path, default=EXPERIMENT / "inputs/EXPORT_FREEZE.json")
    parser.add_argument("--source-root", type=Path, default=EXPERIMENT / "sources")
    parser.add_argument("--output", type=Path, default=HERE / "execution")
    parser.add_argument("--endpoint", type=Path, default=ENDPOINT)
    parser.add_argument("--preflight-review", type=Path, default=HERE / "PREFLIGHT_REVIEW.json")
    parser.add_argument("--selfcheck-output", type=Path, default=HERE / "CPU_CODE_CHECK.json")
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    if args.action == "verify":
        prep, rows, tokens = verify(args); print(json.dumps({"status": "PUBLIC_INPUTS_VERIFIED", "requests": len(rows)}))
    else:
        {"selfcheck": selfcheck, "prepare": prepare, "run": run, "analyze": analyze}[args.action](args)
