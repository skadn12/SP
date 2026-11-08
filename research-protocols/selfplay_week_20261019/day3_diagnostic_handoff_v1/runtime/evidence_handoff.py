"""Gold-blind evidence diagnostic handoff for all 23 admitted four-root questions.

This is a separate diagnostic branch, not a replacement for frozen direct scores.
No free-form chain-of-thought is requested: short decisions and auditable citations
are observable output. Every fixed request is retained, including malformed or
truncated outputs. No retry. Gold is never opened by this module.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parents[1]
PILOT = WEEK / "day3_targeted_4root_v1"
DIRECT = PILOT / "runtime/direct_v1"
sys.path.insert(0, str(PILOT / "runtime"))
import common as c

DEFAULT_RUN = HERE / "evidence_v1"
MAX_TOKENS = 2048
LIMIT = 16384
STATUS = ("supported", "contradicted", "not_supported", "ambiguous")
SYSTEM = ("Use only the supplied dialogue. Give short, auditable evidence-based "
          "judgments and a final answer in the requested JSON object. Do not give "
          "extended reasoning or hidden chain-of-thought.")
TASK = ("For every displayed label, give exactly one item_evidence entry. For multi_select, "
        "judge whether the displayed statement answers the precise question, rather than "
        "merely being related to the dialogue. Use judgment supported, contradicted, "
        "not_supported, or ambiguous. For ordering, cite the actual occurrence or explicit "
        "order relation used for that item. Give zero to two short exact quotations, each "
        "with its turn_ref as printed above; use no citations if no source support exists. "
        "Each quotation must be at most 200 characters. Add a brief reason (at most 320 "
        "characters), including only the relevant evidence link or short calculation. "
        "Return only one JSON object with keys item_evidence and final_answer. Each "
        "item_evidence entry has label, judgment, citations (objects with turn_ref and quote), "
        "and reason. final_answer is an array of displayed labels: select all answers for "
        "multi_select, or list every item exactly once in the requested order for ordering.")


def schema(q):
    field = "options" if q["question_type"] == "multi_select" else "items"
    labels = list(q[field])
    n = len(labels)
    return {"type": "object", "additionalProperties": False,
        "required": ["item_evidence", "final_answer"], "properties": {
        "item_evidence": {"type": "array", "minItems": n, "maxItems": n, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["label", "judgment", "citations", "reason"], "properties": {
                "label": {"type": "string", "enum": labels},
                "judgment": {"type": "string", "enum": list(STATUS)},
                "citations": {"type": "array", "maxItems": 2, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["turn_ref", "quote"], "properties": {
                        "turn_ref": {"type": "string", "minLength": 1},
                        "quote": {"type": "string", "minLength": 1, "maxLength": 200}}}},
                "reason": {"type": "string", "minLength": 1, "maxLength": 320}}}},
        "final_answer": {"type": "array", "uniqueItems": True,
            "minItems": n if field == "items" else 1, "maxItems": n,
            "items": {"type": "string", "enum": labels}}}}


def prompt(tokenizer, q, variant, source):
    field = "options" if q["question_type"] == "multi_select" else "items"
    rows = "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source["turns"])
    user = (f"Dialogue:\n{rows}\n\nQuestion ({q['question_type']}):\n{q['question']}\n\n"
        + field.title() + ":\n" + "\n".join(f"{k}. {v}" for k, v in variant[field].items()) + "\n\n" + TASK)
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
        {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def params(row):
    from vllm import SamplingParams
    # Deliberately no finite answer constraint, JSON grammar, stop box, or retries.
    return SamplingParams(temperature=.8, top_p=.9, max_tokens=MAX_TOKENS, seed=row["seed"])


def files_for_code():
    return sorted(set(c.code_paths()) | {Path(__file__).resolve()})


def selected(direct):
    rows = [r for r in direct["requests"] if r["variant_id"] == "perm_00" and r["slot"] == 0]
    c.require(len(rows) == len({r["candidate_id"] for r in rows}) == 23, "EXPECTED_ALL_23_FIXED_BASE_ROWS")
    c.require(collections.Counter(r["question_type"] for r in rows) == {"multi_select": 12, "ordering": 11}, "TYPE_COUNT_DRIFT")
    return rows


def prepare(run):
    c.environment()
    run = Path(run).resolve()
    c.require(c.below(run, HERE) and not run.exists(), "NEW_RUN_MUST_BE_WITHIN_NEW_RUNTIME_NO_OVERWRITE")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    direct = c.read(DIRECT / "manifest.json")
    public = c.read(DIRECT / "PUBLIC_QUESTIONS.json")
    display = c.read(DIRECT / "DISPLAY_MAPPINGS.json")
    admission = c.read(DIRECT / "CONTENT_ADMISSION.json")
    c.require(admission["status"] == "PASS" and admission["admitted_count"] == 23, "CONTENT_ADMISSION_REQUIRED")
    qby = {q["candidate_id"]: q for q in public["questions"]}
    c.require(len(qby) == 23 and set(qby) == set(admission["candidate_ids"]), "ADMITTED_SET_DRIFT")
    rows, prompts, tokens, maps, sources = [], {}, {}, {}, {}
    for base in selected(direct):
        q = qby[base["candidate_id"]]
        c.a.public_question(q)
        variant = display["requests"][base["request_id"]]
        c.require(variant["variant_id"] == "perm_00", "FIXED_MAPPING_REQUIRED")
        source = c.rt.source_check(q)
        sources[q["root_id"]] = Path(q["source_path"])
        text = prompt(tokenizer, q, variant, source)
        c.require(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in text for t in source["turns"]), "FULL_SOURCE_REQUIRED")
        ids = tokenizer.encode(text, add_special_tokens=False)
        c.require(len(ids) + MAX_TOKENS <= LIMIT, "NO_TRUNCATION_CONTEXT_OVERFLOW:" + q["candidate_id"])
        rid = base["request_id"].replace("__direct__", "__evidence__")
        row = {k: base[k] for k in ("candidate_id", "root_id", "question_type", "variant_id", "display", "slot", "seed", "source_turns", "source_sha256")}
        row.update(request_id=rid, branch="evidence", direct_request_id=base["request_id"],
            prompt_sha256=c.rt.textsha(text), prompt_tokens=len(ids), max_output_tokens=MAX_TOKENS, json_schema=schema(q))
        params(row)
        rows.append(row)
        prompts[rid], tokens[rid], maps[rid] = text, ids, variant
    model_bindings = c.rt.models()
    run.mkdir(parents=True)
    c.save(run / "PUBLIC_QUESTIONS.json", public)
    c.save(run / "CONTENT_ADMISSION.json", admission)
    c.save(run / "DISPLAY_MAPPINGS.json", {"requests": maps, "public_to_canonical": display["public_to_canonical"]})
    c.save(run / "PROMPT_TOKEN_IDS.json", tokens)
    c.save(run / "OUTPUT_SCHEMA_POLICY.json", {"policy": "posthoc_only_no_decoding_constraint", "schemas": {r["candidate_id"]: r["json_schema"] for r in rows}, "system": SYSTEM, "task": TASK})
    for root, path in sources.items():
        (run / "sources").mkdir(exist_ok=True)
        shutil.copy2(path, run / "sources" / (root + ".json"))
    for rid, text in prompts.items():
        (run / "prompts").mkdir(exist_ok=True)
        (run / "prompts" / (rid + ".txt")).write_text(text)
    staged = [p for p in sorted(run.rglob("*")) if p.is_file()]
    manifest = {"status": "PREPARED_AWAITING_INDEPENDENT_PREFLIGHT", "prepared_at_utc": c.now(),
        "branch": "evidence", "solver_model": "S0_M_clean", "model": str(c.MODEL), "tokenizer": str(c.TOKENIZER),
        "namespace": "train.feedback.day3_targeted_four_root.evidence_handoff.v1",
        "use_as_final_eval": False, "training_started": False, "question_count": 23, "request_count": 23,
        "question_type_counts": {"multi_select": 12, "ordering": 11}, "root_count": 4,
        "request_selection": "Every content-admitted question; original direct perm_00 and sample s00 mapping and seed, no correctness filtering",
        "requests": rows, "temperature": .8, "top_p": .9, "max_tokens": MAX_TOKENS, "enable_thinking": False,
        "constrained_decoding": False, "json_grammar": False, "finite_answer_choices": False,
        "batch_size": 8, "max_model_len": LIMIT, "gpu_memory_utilization": .8, "prefix_caching": False,
        "automatic_retry": False, "source_truncation": False, "gold_opened_by_prepare": False,
        "gold_access_scope": "This runner never parses or copies Gold, author evidence, direct responses, scores, or review records. Scoring only after RAW_RESPONSE_FREEZE.",
        "raw_invalid_policy": "Keep all raw outputs. Missing, invalid and truncated are retained, with no automatic retry and no denominator removal.",
        "token_input_min": min(map(len, tokens.values())), "token_input_max": max(map(len, tokens.values())),
        "minimum_context_headroom": min(LIMIT - len(v) - MAX_TOKENS for v in tokens.values()),
        "public_input_bindings": [c.bind(p) for p in staged],
        "historical_public_bindings": [c.bind(DIRECT / name) for name in ("manifest.json", "PUBLIC_QUESTIONS.json", "DISPLAY_MAPPINGS.json", "CONTENT_ADMISSION.json")],
        "model_bindings": model_bindings, "code_bindings": [c.bind(p) for p in files_for_code()],
        "versions": c.rt.versions(), "python_prefix": sys.prefix}
    c.save(run / "manifest.json", manifest)
    c.save(run / "PREFLIGHT_FREEZE.json", {"status": manifest["status"], "created_at_utc": c.now(),
        "files": [c.bind(p) for p in sorted(run.rglob("*")) if p.is_file()],
        "questions": 23, "requests": 23, "public_only": True, "gold_opened": False, "inference_started": False})
    print(json.dumps({k: manifest[k] for k in ("status", "question_count", "request_count", "token_input_min", "token_input_max", "minimum_context_headroom")}))


def verify(run, audit=None):
    c.environment()
    run = Path(run).resolve()
    frozen = c.read(run / "PREFLIGHT_FREEZE.json")
    c.require(frozen["status"] == "PREPARED_AWAITING_INDEPENDENT_PREFLIGHT", "INVALID_PREFLIGHT_STATE")
    for row in frozen["files"]:
        c.require(c.below(row["path"], run), "EXECUTION_INPUT_MUST_BE_LOCAL_PUBLIC")
    c.verify_bindings(frozen["files"])
    m = c.read(run / "manifest.json")
    c.require(m["branch"] == "evidence" and m["solver_model"] == "S0_M_clean" and m["question_count"] == m["request_count"] == 23, "FIXED_23_CONTRACT_DRIFT")
    c.require(m["model"] == str(c.MODEL) and m["tokenizer"] == str(c.TOKENIZER), "MODEL_PATH_DRIFT")
    expected = {"temperature": .8, "top_p": .9, "max_tokens": MAX_TOKENS, "enable_thinking": False,
        "constrained_decoding": False, "json_grammar": False, "finite_answer_choices": False, "batch_size": 8,
        "max_model_len": LIMIT, "gpu_memory_utilization": .8, "prefix_caching": False, "automatic_retry": False, "source_truncation": False}
    c.require(all(m[k] == v for k, v in expected.items()), "DECODER_OR_RUNTIME_DRIFT")
    c.verify_bindings(m["historical_public_bindings"] + m["public_input_bindings"] + m["code_bindings"] + m["model_bindings"])
    c.require(m["versions"] == c.rt.versions() and m["python_prefix"] == sys.prefix, "LIBRARY_OR_ENVIRONMENT_DRIFT")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    public = c.read(run / "PUBLIC_QUESTIONS.json")
    qby = {q["candidate_id"]: q for q in public["questions"]}
    maps = c.read(run / "DISPLAY_MAPPINGS.json")
    ids = c.read(run / "PROMPT_TOKEN_IDS.json")
    bases = selected(c.read(DIRECT / "manifest.json"))
    c.require(len(m["requests"]) == len(qby) == len(ids) == len(maps["requests"]) == 23, "COMPLETE_PUBLIC_ARTIFACT_SET_REQUIRED")
    c.require(set(qby) == {r["candidate_id"] for r in bases} == {r["candidate_id"] for r in m["requests"]}, "NO_QUESTION_FILTERING_ALLOWED")
    historical_maps = c.read(DIRECT / "DISPLAY_MAPPINGS.json")
    c.require(maps["public_to_canonical"] == historical_maps["public_to_canonical"], "CANONICAL_MAP_DRIFT")
    for row, base in zip(m["requests"], bases):
        q = qby[base["candidate_id"]]
        c.a.public_question(q)
        variant = historical_maps["requests"][base["request_id"]]
        source = c.rt.source_check(dict(q, source_path=str(run / "sources" / (q["root_id"] + ".json"))))
        text = prompt(tokenizer, q, variant, source)
        tok = tokenizer.encode(text, add_special_tokens=False)
        rid = base["request_id"].replace("__direct__", "__evidence__")
        expected_row = {k: base[k] for k in ("candidate_id", "root_id", "question_type", "variant_id", "display", "slot", "seed", "source_turns", "source_sha256")}
        expected_row.update(request_id=rid, branch="evidence", direct_request_id=base["request_id"],
            prompt_sha256=c.rt.textsha(text), prompt_tokens=len(tok), max_output_tokens=MAX_TOKENS, json_schema=schema(q))
        c.require(row == expected_row, "FROZEN_REQUEST_RECONSTRUCTION_DRIFT")
        c.require(ids[rid] == tok and maps["requests"][rid] == variant and (run / "prompts" / (rid + ".txt")).read_text() == text, "FROZEN_PROMPT_RECONSTRUCTION_DRIFT")
        c.require(len(tok) + MAX_TOKENS <= LIMIT, "CONTEXT_OVERFLOW")
        p = params(row)
        c.require(getattr(p, "structured_outputs", None) is None and getattr(p, "guided_decoding", None) is None, "NO_CONSTRAINED_DECODING")
    if audit is not None:
        a = c.read(audit)
        c.require(a.get("status") == "PASS", "INDEPENDENT_PREFLIGHT_REQUIRED")
        records = a.get("audited_files") or a.get("files") or []
        c.require(any(Path(r["path"]).resolve() == (run / "PREFLIGHT_FREEZE.json").resolve() and r["sha256"] == c.sha(run / "PREFLIGHT_FREEZE.json") for r in records), "AUDIT_MUST_BIND_THIS_PREFLIGHT")
        c.verify_bindings(records)
    return m


def terminal(row, status, **extra):
    return {k: row[k] for k in ("request_id", "candidate_id", "root_id", "question_type", "branch", "variant_id", "slot", "seed", "prompt_sha256", "direct_request_id")} | {
        "status": status, "raw_output": None, "response_committed_at_utc": c.now(), "model_path": str(c.MODEL),
        "gold_read_by_solver": False, "automatic_retry": False, "source_truncation": False, **extra}


def live(run, audit):
    run = Path(run).resolve()
    c.require(not (run / "STARTED.json").exists(), "ALREADY_STARTED_NO_AUTOMATIC_RERUN")
    m = verify(run, audit)
    rows = m["requests"]
    token_ids = c.read(run / "PROMPT_TOKEN_IDS.json")
    c.save(run / "STARTED.json", {"started_at_utc": c.now(), "manifest_binding": c.bind(run / "manifest.json"), "independent_audit_binding": c.bind(audit), "gold_opened": False})
    backend, submitted, done, failure = None, set(), set(), None
    started, load_seconds = time.perf_counter(), None
    try:
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        _, solver = c.a.native()
        solver.ensure_vllm_environment()
        backend = VLLMSolverBackend(model_path=c.MODEL, tokenizer_path=c.TOKENIZER, max_model_len=LIMIT,
            temperature=.8, top_p=.9, max_tokens=MAX_TOKENS, gpu_memory_utilization=.8)
        load_seconds = backend.load_latency_seconds
        c.require(str(backend.model_identifier) == m["model"], "BACKEND_MODEL_MISMATCH")
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            bid = f"batch{offset // 8:04d}"
            prompts = [(run / "prompts" / (r["request_id"] + ".txt")).read_text() for r in batch]
            sampling = [params(r) for r in batch]
            for row, text in zip(batch, prompts):
                c.require(c.rt.textsha(text) == row["prompt_sha256"], "PROMPT_CHANGED")
            submitted_at, batch_start = c.now(), time.perf_counter()
            c.save(run / "batches" / (bid + "_SUBMITTED.json"), {"submitted_at_utc": submitted_at,
                "request_ids": [r["request_id"] for r in batch],
                "sampling_params": [p.to_dict() if hasattr(p, "to_dict") else repr(p) for p in sampling], "automatic_retry": False})
            submitted.update(r["request_id"] for r in batch)
            outputs = backend.engine.generate(prompts, sampling, use_tqdm=False)
            for i, row in enumerate(batch):
                if i >= len(outputs):
                    record = terminal(row, "BACKEND_ERROR", error="MISSING_OUTPUT", usage=None, usage_status="UNKNOWN_NOT_ZERO")
                else:
                    out = outputs[i]
                    completions = list(out.outputs)
                    primary = completions[0] if completions else None
                    errors = []
                    if out.prompt != prompts[i] or list(out.prompt_token_ids) != token_ids[row["request_id"]]:
                        errors.append("PROMPT_OR_TOKEN_DRIFT")
                    if len(completions) != 1:
                        errors.append("COMPLETION_COUNT_MISMATCH")
                    record = terminal(row, "RETURNED", raw_output=primary.text if primary else None,
                        request_submitted_at_utc=submitted_at, prompt_token_ids=list(out.prompt_token_ids),
                        completion_token_ids=list(primary.token_ids) if primary else None, output_validation_errors=errors,
                        all_completions=[{"text": x.text, "token_ids": list(x.token_ids), "finish_reason": x.finish_reason, "stop_reason": x.stop_reason} for x in completions],
                        metadata={"prompt_token_count": len(out.prompt_token_ids),
                            "completion_token_count": len(primary.token_ids) if primary else None,
                            "finish_reason": primary.finish_reason if primary else None, "stop_reason": primary.stop_reason if primary else None,
                            "temperature": .8, "top_p": .9, "max_tokens": MAX_TOKENS, "enable_thinking": False,
                            "structured_output_api": None, "constrained_decoding": False, "backend_request_id": out.request_id,
                            "cached_tokens": getattr(out, "num_cached_tokens", None)})
                c.save(run / "responses" / (row["request_id"] + ".json"), record)
                done.add(row["request_id"])
            if len(outputs) > len(batch):
                c.save(run / "batches" / (bid + "_EXTRA_OUTPUTS.json"), {"extra_outputs": [
                    {"request_id": o.request_id, "prompt": o.prompt, "prompt_token_ids": list(o.prompt_token_ids),
                        "completions": [{"text": x.text, "token_ids": list(x.token_ids), "finish_reason": x.finish_reason, "stop_reason": x.stop_reason} for x in o.outputs]} for o in outputs[len(batch):]]})
            c.require(len(outputs) == len(batch), "OUTPUT_COUNT_MISMATCH")
            c.save(run / "batches" / (bid + "_COMPLETE.json"), {"completed_at_utc": c.now(), "request_ids": [r["request_id"] for r in batch], "wall_seconds": time.perf_counter() - batch_start})
            print(json.dumps({"status": "LIVE", "completed": len(done), "planned": len(rows)}), flush=True)
    except BaseException as error:
        failure = {"error": str(error), "exception_type": type(error).__name__, "traceback": traceback.format_exc()}
    finally:
        if backend:
            try:
                backend.close()
            except BaseException as error:
                failure = failure or {"error": str(error), "exception_type": type(error).__name__, "traceback": traceback.format_exc()}
        if failure:
            c.save(run / "FAILED.json", {"status": "FAILED", "failed_at_utc": c.now(), "submitted_request_count": len(submitted),
                "returned_request_count": len(done), "model_load_wall_seconds": load_seconds, "wall_seconds": time.perf_counter() - started, "automatic_retry": False, **failure})
            for row in rows:
                if row["request_id"] not in done:
                    sent = row["request_id"] in submitted
                    c.save(run / "responses" / (row["request_id"] + ".json"), terminal(row, "BACKEND_ERROR" if sent else "NOT_SUBMITTED",
                        usage=None, usage_status="UNKNOWN_NOT_ZERO" if sent else "NO_REQUEST_SENT", **failure))
        else:
            c.save(run / "COMPLETE.json", {"status": "COMPLETE", "completed_at_utc": c.now(), "actual_submitted_requests": len(submitted),
                "returned_requests": len(done), "model_load_wall_seconds": load_seconds, "wall_seconds": time.perf_counter() - started, "gold_opened": False})
        paths = [run / "responses" / (r["request_id"] + ".json") for r in rows]
        c.require(all(p.is_file() for p in paths) and set((run / "responses").glob("*.json")) == set(paths), "RAW_REQUIRES_ALL_23_TERMINAL_RECORDS")
        ledger = [p for p in sorted(run.rglob("*")) if p.is_file() and (p.parent.name in ("responses", "batches") or p.name in ("STARTED.json", "COMPLETE.json", "FAILED.json"))]
        c.save(run / "RAW_RESPONSE_FREEZE.json", {"status": "COMMITTED", "created_at_utc": c.now(),
            "manifest_binding": c.bind(run / "manifest.json"), "preflight_binding": c.bind(run / "PREFLIGHT_FREEZE.json"),
            "planned_request_count": 23, "files": [c.bind(p) for p in paths], "execution_ledger_bindings": [c.bind(p) for p in ledger],
            "gold_opened_by_live": False, "source_only_prompts": True, "automatic_retry": False, "source_truncation": False})
    if failure:
        raise RuntimeError("LIVE_FAILED_FIXED_DENOMINATOR:" + failure["error"])


def parse(run):
    import jsonschema
    run = Path(run).resolve()
    target = run / "PARSED_RESPONSES.json"
    c.require(not target.exists(), "NO_PARSE_OVERWRITE")
    rawfreeze = c.read(run / "RAW_RESPONSE_FREEZE.json")
    c.require(rawfreeze["status"] == "COMMITTED" and rawfreeze["planned_request_count"] == 23, "RAW_FREEZE_BEFORE_PARSE")
    c.verify_bindings(rawfreeze["files"] + rawfreeze["execution_ledger_bindings"] + [rawfreeze["manifest_binding"], rawfreeze["preflight_binding"]])
    m = c.read(run / "manifest.json")
    c.verify_bindings(c.read(run / "PREFLIGHT_FREEZE.json")["files"])
    c.verify_bindings(m["code_bindings"])
    maps = c.read(run / "DISPLAY_MAPPINGS.json")
    records = []
    for row in m["requests"]:
        rawpath = run / "responses" / (row["request_id"] + ".json")
        raw = c.read(rawpath)
        errors, parsed, fidelity = [], None, []
        truncated = raw.get("metadata", {}).get("finish_reason") == "length"
        if raw.get("status") != "RETURNED" or not isinstance(raw.get("raw_output"), str):
            errors.append("NO_RETURNED_TEXT")
        else:
            try:
                parsed = json.loads(raw["raw_output"].strip())
                jsonschema.validate(parsed, row["json_schema"])
                field = "options" if row["question_type"] == "multi_select" else "items"
                labels = set(row["display"][field])
                c.require(len({x["label"] for x in parsed["item_evidence"]}) == len(labels) and {x["label"] for x in parsed["item_evidence"]} == labels, "DUPLICATE_OR_MISSING_EVIDENCE_LABELS")
            except Exception as error:
                errors.append(type(error).__name__ + ":" + str(error)[:1500])
        if truncated:
            errors.append("OUTPUT_TRUNCATED")
        errors += raw.get("output_validation_errors", [])
        if not errors:
            source = c.read(run / "sources" / (row["root_id"] + ".json"))
            turns = {t["turn_ref"]: t["text"] for t in source["turns"]}
            for item in parsed["item_evidence"]:
                for citation in item["citations"]:
                    exists = citation["turn_ref"] in turns
                    exact = exists and citation["quote"] in turns[citation["turn_ref"]]
                    fidelity.append({"label": item["label"], **citation, "turn_ref_exists": exists, "exact_substring": exact})
        mapping = maps["requests"][row["request_id"]]["display_to_content"]
        records.append({k: row[k] for k in ("request_id", "candidate_id", "root_id", "question_type", "direct_request_id")} | {
            "raw_binding": c.bind(rawpath), "format_valid": not errors, "invalid_reasons": errors, "truncated": truncated,
            "parsed_response": parsed, "display_prediction": parsed["final_answer"] if not errors else None,
            "canonical_prediction": [mapping[x] for x in parsed["final_answer"]] if not errors else None,
            "citation_checks": fidelity, "citation_count": len(fidelity),
            "exact_quote_count": sum(x["exact_substring"] for x in fidelity), "semantic_evidence_review_status": "PENDING",
            "gold_opened": False})
    result = {"status": "PARSED_NO_GOLD_NO_SEMANTIC_REVIEW", "created_at_utc": c.now(),
        "raw_freeze_binding": c.bind(run / "RAW_RESPONSE_FREEZE.json"), "request_count": 23,
        "valid_count": sum(r["format_valid"] for r in records), "invalid_count": sum(not r["format_valid"] for r in records),
        "truncated_count": sum(r["truncated"] for r in records),
        "citation_count": sum(r["citation_count"] for r in records), "exact_quote_count": sum(r["exact_quote_count"] for r in records),
        "invalid_records_retained": True, "automatic_retry": False, "gold_opened": False, "questions": records}
    c.save(target, result)
    print(json.dumps({k: result[k] for k in ("status", "request_count", "valid_count", "invalid_count", "truncated_count", "citation_count", "exact_quote_count")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "live", "parse"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run)
    elif args.command == "verify":
        m = verify(args.run, args.audit)
        print(json.dumps({"status": "PASS", "requests": m["request_count"], "minimum_context_headroom": m["minimum_context_headroom"], "cpu_only": True}))
    elif args.command == "live":
        c.require(args.audit is not None, "INDEPENDENT_AUDIT_REQUIRED")
        live(args.run, args.audit)
    else:
        parse(args.run)


if __name__ == "__main__":
    main()
