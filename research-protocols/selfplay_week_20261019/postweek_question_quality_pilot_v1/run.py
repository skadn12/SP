"""Frozen Train-only question-quality pilot; generation never opens scoring keys."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = HERE.parents[2]
OUT = HERE / "execution"
NEW = HERE / "CANDIDATES_PRIVATE.json"
REVIEW = HERE / "CONTENT_REVIEW_AI.json"
OLD = WEEK / "postweek_question_quality_patch_v1/audit/HISTORICAL_CANDIDATES_PRIVATE.json"
GATE_PATH = REPO / "scripts/current_method/question_presentation_gate.py"
DIAGNOSTIC_PATH = WEEK / "postweek_diversity_reason_v1/run.py"
CAPTURE_PATH = WEEK / "postweek_feedback_mainline_v1/question_batch.py"
CONDITIONS = ("FULL", "QUESTION_ONLY")
DISPLAY_SEED = 2026102301
MAX_TOKENS, MODEL_LEN = 3072, 16384
PUBLIC_KEYS = {"schema_version", "candidate_id", "root_id", "question_type", "task_operation",
    "question", "source_binding", "request_id", "cohort", "condition", "display", "labels",
    "sampling_seed", "prompt_path", "schema_path", "prompt_tokens"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostic = load_module("quality_pilot_reason_interface", DIAGNOSTIC_PATH)
gate = load_module("quality_pilot_display_gate", GATE_PATH)
capture = load_module("quality_pilot_response_capture", CAPTURE_PATH)
read, save, bind, check = diagnostic.read, diagnostic.save, diagnostic.bind, diagnostic.check
OPERATION_OVERRIDES = {
    "ORDER_DISCOURSE_RECONSTRUCTION": "Reconstruct the requested relationships among the dialogue contributions using source evidence; do not impose a generic stage order."
}
# Override only this imported instance; historical code and frozen prompts remain intact.
diagnostic.OPERATIONS = {**diagnostic.OPERATIONS, **OPERATION_OVERRIDES}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def sampling_seed(candidate):
    key = candidate.get("parent_candidate_id") or candidate["candidate_id"]
    return int.from_bytes(hashlib.sha256(("quality_pilot_v1:" + key).encode()).digest()[:4], "big") % 2147483647


def reviewed_candidates(candidates, review_doc, historical):
    """Validate the independent review binding, including non-admitted drafts."""
    ids = [c["candidate_id"] for c in candidates]
    reviews = review_doc["reviews"]
    assert len(ids) == len(set(ids)) and not set(ids) & set(historical), "NEW_UNIQUE_IDS_REQUIRED"
    assert len(reviews) == len({r["candidate_id"] for r in reviews}), "DUPLICATE_REVIEW"
    by_id = {r["candidate_id"]: r for r in reviews}
    assert set(by_id) == set(ids), "REVIEW_CANDIDATE_SET_DRIFT"
    admitted, ledger, parent_validity = [], [], {}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        review = by_id[cid]
        assert review["candidate_sha256"] == fingerprint(candidate), "REVIEWED_CANDIDATE_DRIFT:" + cid
        assert candidate.get("author_id") and review["author_id"] == candidate["author_id"], "AUTHOR_ID_DRIFT"
        assert review.get("reviewer_id") and review["reviewer_id"] != review["author_id"], "INDEPENDENT_REVIEW_REQUIRED"
        assert review["decision"] in ("PASS", "HOLD", "FAIL"), "UNKNOWN_REVIEW_DECISION"
        parent_id = candidate.get("parent_candidate_id")
        if parent_id:
            assert parent_id in historical, "UNKNOWN_PARENT"
            valid = review["parent_baseline_gold_valid"]
            assert isinstance(valid, bool), "PARENT_BASELINE_REVIEW_REQUIRED"
            assert parent_id not in parent_validity or parent_validity[parent_id] == valid, "CONFLICTING_PARENT_REVIEW"
            parent_validity[parent_id] = valid
        if review["decision"] == "PASS":
            admitted.append(candidate)
        ledger.append({"candidate_id": cid, "parent_candidate_id": parent_id,
            "decision": review["decision"], "candidate_sha256": fingerprint(candidate),
            "author_id": review["author_id"], "reviewer_id": review["reviewer_id"]})
    assert set(parent_validity) == set(historical), "ALL_NINE_PARENT_BASELINES_REQUIRE_REVIEW"
    return admitted, ledger, parent_validity


def parent_display(parent, paired_public, paired_private):
    """Apply the child's permutation to unchanged parent text and parent Gold."""
    public, private = gate.prepare_display(parent, DISPLAY_SEED)
    field = "options" if parent["question_type"] == "multi_select" else "items"
    mapping = paired_private["display_to_canonical"]
    assert set(mapping.values()) == set(parent[field]), "PAIRED_ITEM_IDS_CHANGED"
    assert all(parent[k] == paired_public[k] for k in ("root_id", "question_type", "source_binding")), "PAIRED_SOURCE_OR_TYPE_CHANGED"
    public[field] = {label: parent[field][key] for label, key in mapping.items()}
    private["display_to_canonical"] = copy.deepcopy(mapping)
    private["canonical_to_display"] = {key: label for label, key in mapping.items()}
    private["display_gold"] = [private["canonical_to_display"][key] for key in parent["gold"]]
    if parent["question_type"] == "multi_select":
        private["display_gold"].sort()
    private["public_sha256"] = fingerprint(public)
    private["nonidentity_permutation"] = list(mapping.values()) != list(parent[field])
    private["permutation_from_candidate_id"] = paired_public["candidate_id"]
    assert private["nonidentity_permutation"], "PAIRED_PERMUTATION_MUST_BE_NONIDENTITY"
    return public, private


def question_only_schema(labels, question_type):
    output_schema = diagnostic.repair.schema(labels, [], question_type)
    # repair.schema shares one item object across labels; independent replacements
    # avoid mutating that module and avoid an invalid empty enum in JSON Schema.
    output_schema = copy.deepcopy(output_schema)
    for item in output_schema["properties"]["item_evidence"]["properties"].values():
        item["properties"]["turn_refs"] = {"type": "array", "maxItems": 0, "items": {"type": "string"}}
    return output_schema


def question_only_prompt(tokenizer, row):
    instruction = diagnostic.operation_text(row)
    task = """No source dialogue or source references are supplied in this condition.
Give your best prediction using the question and displayed items only. For each
displayed label, give a judgment (supported, contradicted, not_supported or
ambiguous) and a concise reason (at most 320 characters). Here supported means
your best assessment from the visible wording, not a verified source claim.
Every turn_refs array must be empty []; do not invent or cite hidden dialogue.
Give answer_reason (at most 800 characters), then final_answer as
<box>A B C</box>: selected labels for multi_select, every label exactly once for
ordering. The example only illustrates format. Return only JSON, with
item_evidence first, answer_reason second and final_answer last. Give brief
visible explanations, not extended private deliberation."""
    user = (f"Question ({row['question_type']}):\n{row['question']}\n\nDisplayed items:\n" +
        "\n".join(f"{label}. {text}" for label, text in row["display"].items()) +
        f"\n\nRequested operation: {row['task_operation']}\n{instruction}\n\n{task}")
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Make a best prediction from the question and displayed items. No dialogue is supplied. Never fabricate source references. Give a brief explanation and final boxed answer."},
        {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def validate_output(raw, row, output_schema):
    if row["condition"] == "FULL":
        parsed, answer = diagnostic.repair.validate(raw, row, output_schema)
        if row["question_type"] == "ordering":
            assert all(item["turn_refs"] for item in parsed["item_evidence"].values()), "NO_ANCHOR_FOR_ORDERING"
        return parsed, answer
    assert row["condition"] == "QUESTION_ONLY"
    from jsonschema import validate as json_validate
    parsed = json.loads(raw)
    json_validate(parsed, output_schema)
    assert list(parsed) == ["item_evidence", "answer_reason", "final_answer"], "REASONS_MUST_PRECEDE_ANSWER"
    assert parsed["answer_reason"].strip(), "EMPTY_ANSWER_REASON"
    for item in parsed["item_evidence"].values():
        assert item["reason"].strip(), "EMPTY_ITEM_REASON"
        assert item["turn_refs"] == [], "QUESTION_ONLY_REFERENCE_FORBIDDEN"
    match = re.fullmatch(r"<box>([A-F](?: [A-F]){0,5})</box>", parsed["final_answer"])
    assert match, "ANSWER_BOX_REQUIRED"
    answer = match.group(1).split()
    assert len(answer) == len(set(answer)) and set(answer) <= set(row["labels"]), "INVALID_LABEL_SET"
    if row["question_type"] == "ordering":
        assert set(answer) == set(row["labels"]), "ORDERING_INCOMPLETE"
    return parsed, answer


def prepare():
    assert not OUT.exists(), "NO_PREPARATION_OVERWRITE"
    candidates = read(NEW)
    assert isinstance(candidates, list) and 1 <= len(candidates) <= 11, "EXPECTED_UP_TO_ELEVEN_NEW_DRAFTS"
    historical_list = read(OLD)["candidates"]
    historical = {c["candidate_id"]: c for c in historical_list}
    assert len(historical_list) == len(historical) == 9, "EXPECTED_NINE_UNCHANGED_PARENTS"
    admitted, ledger, parent_validity = reviewed_candidates(candidates, read(REVIEW), historical)
    coverage = gate.coverage_report(admitted)
    assert coverage["coverage_complete"], "ADMITTED_NEW_MC_MUST_COVER_GOLD_23456:" + json.dumps(coverage)
    prepared_new, paired_children = [], {}
    for candidate in admitted:
        public, private = gate.prepare_display(candidate, DISPLAY_SEED)
        field = "options" if candidate["question_type"] == "multi_select" else "items"
        assert len(candidate[field]) <= 6, "REASON_SCHEMA_MAXIMUM_SIX_ITEMS"
        parent_id = candidate.get("parent_candidate_id")
        strict = bool(candidate.get("paired_item_identity_preserved", False))
        if strict:
            assert parent_id and parent_id not in paired_children, "UNIQUE_STRICT_CHILD_PER_PARENT_REQUIRED"
            parent = historical[parent_id]
            assert parent["question_type"] == candidate["question_type"], "PAIRED_TYPE_CHANGED"
            assert list(parent[field]) == list(candidate[field]), "PAIRED_INPUT_KEY_ORDER_CHANGED"
            assert all(parent[key] == candidate[key] for key in ("root_id", "source_binding")), "PAIRED_SOURCE_CHANGED"
            paired_children[parent_id] = (public, private)
        prepared_new.append((candidate, public, private, "new", parent_id, strict))
    staged_questions = []
    for parent in historical_list:
        cid = parent["candidate_id"]
        if not parent_validity[cid]:
            continue
        if cid in paired_children:
            public, private = parent_display(parent, *paired_children[cid])
        else:
            public, private = gate.prepare_display(parent, DISPLAY_SEED)
        staged_questions.append((parent, public, private, "old", None, False))
    staged_questions.extend(prepared_new)
    from transformers import AutoTokenizer
    import xgrammar as xgr
    endpoint = read(diagnostic.ENDPOINT)
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    staged, sources = [], {}
    for candidate, public, private, cohort, parent_id, strict in staged_questions:
        check(public["source_binding"])
        source = read(public["source_binding"]["path"])
        assert source["root_id"] == public["root_id"], "SOURCE_ROOT_DRIFT"
        refs = [turn["turn_ref"] for turn in source["turns"]]
        assert refs and len(refs) == len(set(refs)), "INVALID_SOURCE_REFERENCES"
        sources[public["source_binding"]["path"]] = public["source_binding"]
        field = "options" if public["question_type"] == "multi_select" else "items"
        for condition in CONDITIONS:
            row = {key: value for key, value in public.items() if key != field}
            row.update(display=public[field], labels=list(public[field]), cohort=cohort,
                condition=condition, sampling_seed=sampling_seed(candidate),
                request_id=f"{public['candidate_id']}__{condition}__B1_quality_v1")
            prompt = diagnostic.make_prompt(tokenizer, row, row["display"], source) if condition == "FULL" else question_only_prompt(tokenizer, row)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            assert len(tokens) + MAX_TOKENS <= MODEL_LEN, "CONTEXT_OVERFLOW_NO_TRUNCATION:" + row["request_id"]
            output_schema = diagnostic.repair.schema(row["labels"], refs, row["question_type"]) if condition == "FULL" else question_only_schema(row["labels"], row["question_type"])
            compiler.compile_json_schema(json.dumps(output_schema), any_whitespace=True)
            scoring = {"request_id": row["request_id"], "candidate_id": candidate["candidate_id"],
                "cohort": cohort, "parent_candidate_id": parent_id,
                "strictly_paired": strict and parent_validity.get(parent_id, False), "presentation": private}
            staged.append((row, prompt, tokens, output_schema, scoring))
    assert len(staged) <= 40 and len({r[0]["request_id"] for r in staged}) == len(staged)
    OUT.mkdir()
    bindings = [bind(path) for path in (Path(__file__), HERE / "PROTOCOL_ZH.md", GATE_PATH,
        DIAGNOSTIC_PATH, CAPTURE_PATH, diagnostic.REPAIR_PATH, diagnostic.ENDPOINT)] + list(sources.values())
    rows, token_ids, private_rows = [], {}, []
    for row, prompt, tokens, output_schema, scoring in staged:
        rid = row["request_id"]
        prompt_path, schema_path = OUT / "prompts" / (rid + ".txt"), OUT / "schemas" / (rid + ".json")
        prompt_path.parent.mkdir(exist_ok=True)
        with prompt_path.open("x", encoding="utf-8") as stream:
            stream.write(prompt)
            stream.flush()
            os.fsync(stream.fileno())
        save(schema_path, output_schema)
        row.update(prompt_path=str(prompt_path), schema_path=str(schema_path), prompt_tokens=len(tokens))
        assert set(row) == PUBLIC_KEYS, "PUBLIC_FIELD_ALLOWLIST_DRIFT"
        rows.append(row)
        token_ids[rid] = tokens
        private_rows.append(scoring)
        bindings.extend((bind(prompt_path), bind(schema_path)))
    save(OUT / "PUBLIC_REQUESTS.json", {"requests": rows})
    save(OUT / "PROMPT_TOKEN_IDS.json", token_ids)
    bindings.extend(bind(OUT / name) for name in ("PUBLIC_REQUESTS.json", "PROMPT_TOKEN_IDS.json"))
    private_path = OUT / "PRIVATE_SCORING.json"
    save(private_path, {"requests": private_rows, "bindings": [bind(NEW), bind(REVIEW), bind(OLD)],
        "public_requests_binding": bind(OUT / "PUBLIC_REQUESTS.json"),
        "prompt_token_ids_binding": bind(OUT / "PROMPT_TOKEN_IDS.json"),
        "new_admission": ledger, "new_mc_coverage": coverage,
        "parent_baselines": [{"candidate_id": cid, "decision": "PASS" if valid else "HOLD",
            "reason": "Independent review of historical Gold", "parent_baseline_gold_valid": valid}
            for cid, valid in parent_validity.items()],
        "planned_questions": {"old": 9, "new": 11}, "submitted_new_drafts": len(candidates),
        "historical_scores_reusable": False})
    private_path.chmod(0o600)
    save(OUT / "PREPARATION.json", {"requests": len(rows), "model_role": "B1", "bindings": bindings,
        "private_scoring_binding": bind(private_path), "max_output_tokens": MAX_TOKENS,
        "max_model_len": MODEL_LEN, "temperature": .8, "top_p": .9,
        "display_seed": DISPLAY_SEED, "conditions": list(CONDITIONS),
        "operation_instruction_overrides": OPERATION_OVERRIDES,
        "fixed_order": [row["request_id"] for row in rows], "automatic_retry": False,
        "source_truncation": False, "native_hidden_thinking": False,
        "independent_requests_no_shared_history_or_memory": True,
        "generation_reads_private_scoring": False, "historical_scores_reusable": False,
        "namespace": "train_only_exploratory_question_quality_diagnostic"})
    print(json.dumps({"prepared": len(rows), "old_questions": sum(r["cohort"] == "old" for r in rows) // 2,
        "new_questions": len(admitted), "max_prompt_tokens": max(r["prompt_tokens"] for r in rows)}))


def public_preparation():
    prep = read(OUT / "PREPARATION.json")
    private_paths = {NEW.resolve(), REVIEW.resolve(), OLD.resolve(), (OUT / "PRIVATE_SCORING.json").resolve()}
    for bound in prep["bindings"]:
        assert Path(bound["path"]).resolve() not in private_paths, "PRIVATE_INPUT_IN_WORKER_BINDINGS"
        check(bound)
    rows = read(OUT / "PUBLIC_REQUESTS.json")["requests"]
    tokens = read(OUT / "PROMPT_TOKEN_IDS.json")
    ids = [r["request_id"] for r in rows]
    assert 0 < len(rows) == prep["requests"] == len(set(ids)) <= 40, "FROZEN_REQUEST_COUNT_DRIFT"
    assert ids == prep["fixed_order"] and set(tokens) == set(ids), "FROZEN_REQUEST_ORDER_OR_TOKEN_SET_DRIFT"
    required = [OUT / "PUBLIC_REQUESTS.json", OUT / "PROMPT_TOKEN_IDS.json", Path(__file__)]
    for row in rows:
        assert set(row) == PUBLIC_KEYS and row["condition"] in CONDITIONS, "NONPUBLIC_REQUEST_FIELDS"
        assert row["cohort"] in ("old", "new") and row["labels"] == list(row["display"])
        assert len(tokens[row["request_id"]]) == row["prompt_tokens"], "PROMPT_TOKEN_COUNT_DRIFT"
        assert row["prompt_tokens"] + MAX_TOKENS <= MODEL_LEN, "CONTEXT_OVERFLOW"
        required.extend(Path(row[key]) for key in ("prompt_path", "schema_path"))
    assert all(bind(path) in prep["bindings"] for path in required), "UNBOUND_PUBLIC_INPUT"
    return prep, rows, tokens


def preserve_response(row, results, endpoint):
    """Serialize all backend returns without checking or discarding their shape."""
    returned = [{"prompt": result.prompt, "prompt_token_ids": list(result.prompt_token_ids or []),
        "outputs": [{"raw_output": comp.text, "finish_reason": comp.finish_reason,
            "stop_reason": comp.stop_reason, "output_token_ids": list(comp.token_ids)}
            for comp in result.outputs]} for result in results]
    raw = {"request_id": row["request_id"], "status": "RETURNED", "returned_result_count": len(returned),
        "returned_results": returned, "raw_output": None, "model_role": "B1",
        "model_directory": endpoint["model_directory"], "sampling_seed": row["sampling_seed"], "gold_read": False}
    if returned:
        raw["prompt_token_ids"] = returned[0]["prompt_token_ids"]
        if returned[0]["outputs"]:
            raw.update(returned[0]["outputs"][0])
    return raw


def run(execute_gpu=False):
    assert execute_gpu, "EXPLICIT_GPU_FLAG_REQUIRED"
    assert not (OUT / "STARTED.json").exists() and not (OUT / "RAW_RESPONSE_FREEZE.json").exists(), "NO_RERUN_OR_OVERWRITE"
    prep, rows, token_ids = public_preparation()
    for key, value in {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ[key] = value
    endpoint = read(diagnostic.ENDPOINT)
    assert Path(sys.prefix).resolve() == Path(endpoint["python_prefix"]).resolve(), "WRONG_PYTHON_ENVIRONMENT"
    for bound in endpoint["model_bindings"] + endpoint["tokenizer_bindings"] + endpoint["code_bindings"]:
        check(bound)
    backend_path = next(Path(b["path"]) for b in endpoint["code_bindings"] if Path(b["path"]).name == "vllm_backends.py")
    sys.path.insert(0, str(backend_path.parents[3]))
    from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
    from vllm import SamplingParams, sampling_params
    ids = [row["request_id"] for row in rows]
    save(OUT / "STARTED.json", {"time_unix": time.time(), "preparation": bind(OUT / "PREPARATION.json"),
        "private_scoring_binding": prep["private_scoring_binding"], "requests": len(rows), "request_ids": ids,
        "automatic_retry": False, "gold_read": False})
    backend, failure, submitted, request_failures = None, None, set(), []
    started = time.perf_counter()
    try:
        backend = VLLMSolverBackend(model_path=Path(endpoint["model_directory"]), tokenizer_path=Path(endpoint["tokenizer_directory"]),
            max_model_len=MODEL_LEN, temperature=.8, top_p=.9, max_tokens=MAX_TOKENS, gpu_memory_utilization=.8)
        assert str(backend.model_identifier) == endpoint["model_directory"], "LOADED_MODEL_DRIFT"
        for row in rows:
            rid, raw_path = row["request_id"], OUT / "raw" / (row["request_id"] + ".json")
            try:
                params = SamplingParams(temperature=.8, top_p=.9, max_tokens=MAX_TOKENS, seed=row["sampling_seed"],
                    structured_outputs=sampling_params.StructuredOutputsParams(json=read(row["schema_path"])))
                prompt = Path(row["prompt_path"]).read_text()
                save(OUT / "submitted" / (rid + ".json"), {"request_id": rid, "seed": row["sampling_seed"], "time_unix": time.time()})
                submitted.add(rid)
                results = backend.engine.generate([prompt], [params], use_tqdm=False)
                raw = preserve_response(row, results, endpoint)
                save(raw_path, raw)
                # Shape/prompt/token checks occur only after all returned text is durable.
                errors = capture.response_errors(raw, prompt, token_ids[rid])
                save(OUT / "raw_validation" / (rid + ".json"), {"request_id": rid, "errors": errors,
                    "raw_binding": bind(raw_path)})
                if errors:
                    request_failures.append({"request_id": rid, "errors": errors})
                print("RETURNED " + rid, flush=True)
            except Exception:
                error = traceback.format_exc()
                request_failures.append({"request_id": rid, "error": error})
                if not raw_path.exists():
                    save(raw_path, {"request_id": rid, "status": "BACKEND_ERROR" if rid in submitted else "NOT_SUBMITTED",
                        "raw_output": None, "failure": error})
    except BaseException:
        failure = traceback.format_exc()
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                failure = (failure or "") + traceback.format_exc()
        for row in rows:
            rid = row["request_id"]
            path = OUT / "raw" / (rid + ".json")
            if not path.exists():
                save(path, {"request_id": rid, "status": "BACKEND_ERROR" if rid in submitted else "NOT_SUBMITTED",
                    "raw_output": None, "failure": failure})
        raw_paths = sorted((OUT / "raw").glob("*.json"))
        save(OUT / "RAW_RESPONSE_FREEZE.json", {"requests": len(rows), "request_ids": ids,
            "started_binding": bind(OUT / "STARTED.json"), "wall_seconds": time.perf_counter() - started,
            "submitted_requests": len(submitted), "returned_requests": sum(read(path)["status"] == "RETURNED" for path in raw_paths),
            "failure": failure, "request_failures": request_failures, "automatic_retry": False,
            "bindings": [bind(path) for path in raw_paths],
            "validation_bindings": [bind(path) for path in sorted((OUT / "raw_validation").glob("*.json"))]})
    if failure or request_failures:
        raise RuntimeError("Frozen failures; no automatic retry. Inspect RAW_RESPONSE_FREEZE.json")


def sealed_inputs():
    """Validate the full public/raw chain before opening any private file."""
    prep, rows, tokens = public_preparation()
    frozen = read(OUT / "RAW_RESPONSE_FREEZE.json")
    ids = [row["request_id"] for row in rows]
    paths = {str((OUT / "raw" / (rid + ".json")).resolve()) for rid in ids}
    assert frozen["requests"] == len(rows) == len(frozen["bindings"]), "RAW_FREEZE_COUNT_DRIFT"
    bound_paths = [bound["path"] for bound in frozen["bindings"]]
    assert len(set(bound_paths)) == len(paths) and set(bound_paths) == paths, "RAW_FREEZE_SET_DRIFT"
    assert {str(path.resolve()) for path in (OUT / "raw").glob("*.json")} == paths, "RAW_DIRECTORY_SET_DRIFT"
    assert frozen["request_ids"] == ids, "RAW_REQUEST_ORDER_DRIFT"
    for bound in frozen["bindings"] + frozen["validation_bindings"]:
        check(bound)
    assert frozen["returned_requests"] == sum(read(path)["status"] == "RETURNED" for path in paths), "RETURNED_COUNT_DRIFT"
    assert frozen["started_binding"] == bind(OUT / "STARTED.json"), "STARTED_BINDING_DRIFT"
    start = read(OUT / "STARTED.json")
    assert start["requests"] == len(rows) and start["request_ids"] == ids, "STARTED_REQUEST_SET_DRIFT"
    assert start["preparation"] == bind(OUT / "PREPARATION.json"), "PREPARATION_CHAIN_DRIFT"
    assert start["private_scoring_binding"] == prep["private_scoring_binding"], "PRIVATE_SCORING_CHAIN_DRIFT"
    check(prep["private_scoring_binding"])
    private = read(prep["private_scoring_binding"]["path"])
    for bound in private["bindings"]:
        check(bound)
    assert private["public_requests_binding"] == bind(OUT / "PUBLIC_REQUESTS.json"), "PRIVATE_PUBLIC_DRIFT"
    assert private["prompt_token_ids_binding"] == bind(OUT / "PROMPT_TOKEN_IDS.json"), "PRIVATE_TOKENS_DRIFT"
    assert [item["request_id"] for item in private["requests"]] == ids, "PRIVATE_REQUEST_SET_DRIFT"
    return prep, rows, tokens, frozen, private


def score_record(row, presentation, answer, valid):
    gold = presentation["canonical_gold"]
    prediction = gate.restore_prediction(presentation, answer) if valid else []
    exact = bool(valid and gate.is_correct_display_answer(presentation, answer))
    score = {key: row[key] for key in ("request_id", "candidate_id", "root_id", "cohort", "condition", "question_type", "task_operation")}
    score.update(valid=valid, exact=exact, prediction=prediction, gold=gold,
        display_prediction=answer if valid else [], display_gold=presentation["display_gold"],
        gold_count=len(gold) if row["question_type"] == "multi_select" else None,
        false_positive=None, false_negative=None, pairwise_correct=None, pairwise_total=None, allselect_exact=None)
    if row["question_type"] == "multi_select":
        # A missing valid answer is represented as an empty prediction for these
        # descriptive counts; exact accuracy still uses every frozen request.
        score.update(false_positive=len(set(prediction) - set(gold)), false_negative=len(set(gold) - set(prediction)),
            allselect_exact=len(gold) == len(presentation["display_to_canonical"]))
    else:
        positions = {key: i for i, key in enumerate(prediction)}
        pairs = [(left, right) for i, left in enumerate(gold) for right in gold[i + 1:]]
        score.update(pairwise_correct=sum(valid and positions[left] < positions[right] for left, right in pairs), pairwise_total=len(pairs))
    return score


def aggregate(records):
    n = len(records)
    mc = [r for r in records if r["question_type"] == "multi_select"]
    ordering = [r for r in records if r["question_type"] == "ordering"]
    pairs = sum(r["pairwise_total"] for r in ordering)
    return {"frozen_requests": n, "valid_requests": sum(r["valid"] for r in records),
        "invalid_requests": sum(not r["valid"] for r in records), "correct": sum(r["exact"] for r in records),
        "exact_accuracy": sum(r["exact"] for r in records) / n if n else None,
        "mc_requests": len(mc), "mc_correct": sum(r["exact"] for r in mc),
        "mc_false_positive": sum(r["false_positive"] for r in mc), "mc_false_negative": sum(r["false_negative"] for r in mc),
        "mc_allselect_correct": sum(r["allselect_exact"] for r in mc),
        "mc_allselect_accuracy": sum(r["allselect_exact"] for r in mc) / len(mc) if mc else None,
        "ordering_requests": len(ordering), "ordering_correct": sum(r["exact"] for r in ordering),
        "ordering_pairwise_correct": sum(r["pairwise_correct"] for r in ordering), "ordering_pairwise_total": pairs,
        "ordering_pairwise_accuracy": sum(r["pairwise_correct"] for r in ordering) / pairs if pairs else None}


def summarize(scores, private):
    cells, gold_buckets = [], []
    for cohort in ("old", "new"):
        for condition in CONDITIONS:
            selected = [s for s in scores if s["cohort"] == cohort and s["condition"] == condition]
            planned = private["planned_questions"][cohort]
            cells.append({"cohort": cohort, "condition": condition, "originally_planned_questions": planned,
                "held_or_unavailable_before_freeze": planned - len(selected), **aggregate(selected)})
    for condition in CONDITIONS:
        for count in (2, 3, 4, 5, 6):
            selected = [s for s in scores if s["cohort"] == "new" and s["condition"] == condition and s["gold_count"] == count]
            gold_buckets.append({"condition": condition, "gold_count": count, **aggregate(selected),
                "six_gold_not_distractor_evidence": count == 6})
    by_id = {s["request_id"]: s for s in scores}
    pairs = []
    for item in private["requests"]:
        if item["cohort"] != "new" or not item["parent_candidate_id"]:
            continue
        new = by_id[item["request_id"]]
        old = next((s for s in scores if s["cohort"] == "old" and s["candidate_id"] == item["parent_candidate_id"] and s["condition"] == new["condition"]), None)
        strict = item["strictly_paired"] and old is not None
        pairs.append({"new_candidate_id": new["candidate_id"], "parent_candidate_id": item["parent_candidate_id"],
            "condition": new["condition"], "strict_paired_comparison": strict,
            "old_exact": old["exact"] if old else None, "new_exact": new["exact"],
            "exact_delta": int(new["exact"]) - int(old["exact"]) if strict else None,
            "note": "Matched source, display IDs, permutation and seed" if strict else "Replacement or historical Gold HOLD; not a strict paired change"})
    return {"requests": len(scores), "correct": sum(s["exact"] for s in scores),
        "valid": sum(s["valid"] for s in scores), "cells": cells, "new_mc_gold_buckets": gold_buckets,
        "paired_changes": pairs, "parent_baselines": private["parent_baselines"],
        "new_admission": private["new_admission"], "new_mc_coverage": private["new_mc_coverage"],
        "all_frozen_requests_in_exact_denominator": True, "invalid_ordering_pairwise_score": 0,
        "invalid_mc_prediction_for_descriptive_fp_fn": "empty_set", "semantic_review": "PENDING",
        "feedback_eligible": False, "training_admission": False, "historical_scores_reused": False,
        "inference_limits": "One stochastic rollout per condition, small heterogeneous Train-only sample. No population difficulty, causal feedback benefit or formal weakness claim."}


def analyze():
    prep, rows, tokens, frozen, private = sealed_inputs()
    scoring = {item["request_id"]: item for item in private["requests"]}
    records, scores = [], []
    for row in rows:
        rid = row["request_id"]
        item = scoring[rid]
        presentation = item["presentation"]
        public = {key: row[key] for key in ("schema_version", "candidate_id", "root_id", "question_type", "task_operation", "question", "source_binding")}
        public["options" if row["question_type"] == "multi_select" else "items"] = row["display"]
        assert presentation["public_sha256"] == fingerprint(public), "PRIVATE_PRESENTATION_DRIFT"
        raw_path = OUT / "raw" / (rid + ".json")
        raw = read(raw_path)
        record = {key: row[key] for key in ("request_id", "candidate_id", "root_id", "question_type", "task_operation", "cohort", "condition")}
        record.update(raw_binding=bind(raw_path), semantic_review="PENDING", feedback_eligible=False,
            request=row, schema_binding=bind(row["schema_path"]))
        answer, valid = [], False
        try:
            assert raw["request_id"] == rid, "RAW_REQUEST_ID_DRIFT"
            assert raw["status"] == "RETURNED", "BACKEND_OR_SUBMISSION_FAILURE:" + raw["status"]
            assert not capture.response_errors(raw, Path(row["prompt_path"]).read_text(), tokens[rid]), "RETURNED_RESPONSE_DRIFT"
            assert raw["finish_reason"] == "stop", "TRUNCATED_OR_NONSTOP_COMPLETION"
            parsed, answer = validate_output(raw["raw_output"], row, read(row["schema_path"]))
            valid = True
            consistency = diagnostic.mechanical_consistency(row, parsed, answer) if row["condition"] == "FULL" else {
                "reference_checks_applicable": False, "reason": "No source or references supplied",
                "answer_matches_supported_labels": set(answer) == {k for k, v in parsed["item_evidence"].items() if v["judgment"] == "supported"} if row["question_type"] == "multi_select" else None}
            record.update(status="STRUCTURALLY_VALID_REASONS_SAVED", parsed=parsed, answer_labels=answer,
                mechanical_consistency=consistency)
        except Exception as error:
            valid = False
            record.update(status="INVALID", error=str(error))
        records.append(record)
        score = score_record(row, presentation, answer, valid)
        score.update(parent_candidate_id=item["parent_candidate_id"], strictly_paired=item["strictly_paired"],
            raw_binding=bind(raw_path), invalid_error=record.get("error"))
        scores.append(score)
    summary = summarize(scores, private)
    save(OUT / "SCORES.json", {"records": scores, "raw_freeze_binding": bind(OUT / "RAW_RESPONSE_FREEZE.json"),
        "private_scoring_binding": prep["private_scoring_binding"], "gold_read_after_all_raw_sealed": True})
    save(OUT / "REASON_REVIEW_PACKET.json", {"requests": records, "semantic_review": "PENDING",
        "feedback_eligible": False, "gold_read_by_generator": False,
        "raw_freeze_binding": bind(OUT / "RAW_RESPONSE_FREEZE.json")})
    save(OUT / "ANALYSIS_SUMMARY.json", summary)
    print(json.dumps({"requests": summary["requests"], "valid": summary["valid"], "correct": summary["correct"], "cells": summary["cells"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "run": lambda: run(args.execute_gpu), "analyze": analyze}[args.action]()
