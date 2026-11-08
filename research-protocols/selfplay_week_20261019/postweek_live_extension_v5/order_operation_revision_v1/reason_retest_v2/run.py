"""Three fixed V2 operation-only reason retests; old experiments remain immutable."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = Path("/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019")
BASE = WEEK / "postweek_live_extension_v5"
OLD = BASE / "reason_diagnostic_v5"
REVISION = BASE / "order_operation_revision_v1/OPERATION_METADATA_REVISION_V2.json"
REVIEW = BASE / "order_operation_revision_v1/OPERATION_METADATA_REVIEW_V2.json"
REVISION_SHA = "22f3a6093f32773d6daf9ed2d1c063a1029a50301aeef1e1793268793d630a6d"
REVIEW_SHA = "77954b87e68e8e39bbe95e7809da80f376fc8cddcde322f2f0c8094d05039354"
OUT = HERE / "execution"
ENDPOINT = WEEK / "day5_training_v1/evaluation/models/B1/EXECUTION_MANIFEST.json"
HELPER_PATH = WEEK / "postweek_question_quality_pilot_v1/run.py"
spec = importlib.util.spec_from_file_location("v5_reason_shared_interface", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read, save, bind, check = helper.read, helper.save, helper.bind, helper.check
diag = helper.diagnostic
MAX_TOKENS, MODEL_LEN = 3072, 16384
OPERATIONS = {"ORDER_ENQUIRY_PROGRESSION": "Reconstruct the enquiry or record disclosure progression requested by the question, grounded in the supplied dialogue. Distinguish statements about constraints, replies and unresolved state from actual journeys or completed visits. This operation is not numeric clock-time sorting."}
OLD_OPERATIONS = {
    "ORDER_EVENT_TIME": "Apply the time ordering requested by the question using the supplied dialogue.",
    "ORDER_CORRECTION_CHAIN": "Reconstruct the correction sequence requested by the question using the supplied dialogue."
}
diag.OPERATIONS = {**diag.OPERATIONS, **OPERATIONS}
PUBLIC_KEYS = {"candidate_id", "root_id", "question_type", "task_operation", "question", "source_binding", "display"}

def seed(cid):
    return int.from_bytes(hashlib.sha256(("reason_v5:2026102307:"+cid).encode()).digest()[:4], "big") % 2147483647

def all_bindings(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            yield {key: value[key] for key in ("path", "sha256", "bytes")}
        else:
            for child in value.values(): yield from all_bindings(child)
    elif isinstance(value, list):
        for child in value: yield from all_bindings(child)

def metadata_gate():
    assert bind(REVISION)["sha256"] == REVISION_SHA, "V2_PROPOSAL_HASH_CHANGED"
    assert bind(REVIEW)["sha256"] == REVIEW_SHA, "V2_REVIEW_HASH_CHANGED"
    revision, review = read(REVISION), read(REVIEW)
    assert review["status"] == "PASS", "INDEPENDENT_V2_METADATA_PASS_REQUIRED"
    assert any(item == bind(REVISION) for item in all_bindings(review)), "REVIEW_NOT_BOUND_TO_V2"
    assert review["policy_checks"]["old_scores_reusable"] is False and review["policy_checks"]["solver_score_filtering"] is False
    assert all(mapping["verdict"] == "PASS" for mapping in review["mapping_reviews"])
    for item in all_bindings(review): check(item)
    assert len(revision["changes"]) == 3
    assert all(change["new_task_operation"] == "ORDER_ENQUIRY_PROGRESSION" for change in revision["changes"])
    assert revision["operation_definition"] == OPERATIONS["ORDER_ENQUIRY_PROGRESSION"]
    return revision

def export_freeze():
    metadata_gate()
    frozen = read(HERE / "inputs/EXPORT_FREEZE.json")
    for item in frozen["bindings"]:
        check(item)
    return frozen

def prepare():
    assert not OUT.exists(), "NO_PREPARATION_OVERWRITE"
    frozen = export_freeze()
    rows = read(HERE / "inputs/PUBLIC_QUESTIONS.json")["questions"]
    assert len(rows) == 3 and len({q["candidate_id"] for q in rows}) == 3
    revision = metadata_gate()
    assert [q["candidate_id"] for q in rows] == [change["candidate_id"] for change in revision["changes"]]
    old_public = {q["candidate_id"]: q for q in read(OLD / "inputs/PUBLIC_QUESTIONS.json")["questions"]}
    old_requests = {q["candidate_id"]: q for q in read(OLD / "execution/PUBLIC_REQUESTS.json")["requests"]}
    old_preparation = read(OLD / "execution/PREPARATION.json")
    old_bound = {item["path"]: item for item in old_preparation["bindings"]}
    from transformers import AutoTokenizer
    import xgrammar as xgr
    endpoint = read(ENDPOINT)
    assert endpoint["model_role"] == "B1" and Path(sys.prefix).resolve() == Path(endpoint["python_prefix"]).resolve()
    for item in endpoint["model_bindings"] + endpoint["tokenizer_bindings"] + endpoint["code_bindings"]: check(item)
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    bindings = [bind(path) for path in (Path(__file__), HELPER_PATH, helper.DIAGNOSTIC_PATH,
        helper.CAPTURE_PATH, diag.REPAIR_PATH, helper.GATE_PATH, ENDPOINT, HERE / "inputs/EXPORT_FREEZE.json")]
    bindings += frozen["bindings"] + [bind(path) for path in (REVISION, REVIEW, OLD / "run.py",
        OLD / "execution/PREPARATION.json", OLD / "execution/PUBLIC_REQUESTS.json", OLD / "execution/PROMPT_TOKEN_IDS.json")]
    bindings += endpoint["model_bindings"] + endpoint["tokenizer_bindings"] + endpoint["code_bindings"]
    parity = []
    prepared, token_ids = [], {}
    for question in rows:
        assert set(question) == PUBLIC_KEYS and question["task_operation"] in OPERATIONS
        previous = old_public[question["candidate_id"]]
        assert {k:v for k,v in question.items() if k != "task_operation"} == {k:v for k,v in previous.items() if k != "task_operation"}, "PUBLIC_CONTENT_DRIFT"
        check(question["source_binding"])
        source = read(question["source_binding"]["path"])
        assert source["root_id"] == question["root_id"]
        refs = [t["turn_ref"] for t in source["turns"]]
        assert refs and len(refs) == len(set(refs))
        row = dict(question)
        row.update(request_id=question["candidate_id"]+"__v5_reason_operation_retest_v2_s00", labels=list(question["display"]),
            sampling_seed=seed(question["candidate_id"]), cohort="v5_operation_retest_v2", condition="FULL", source_turns=len(refs))
        old_row = old_requests[question["candidate_id"]]
        assert row["sampling_seed"] == old_row["sampling_seed"] and row["display"] == old_row["display"]
        old_prompt_path, old_schema_path = Path(old_row["prompt_path"]), Path(old_row["schema_path"])
        check(old_bound[str(old_prompt_path)]); check(old_bound[str(old_schema_path)])
        old_prompt = old_prompt_path.read_text()
        old_segment = "Requested operation: " + previous["task_operation"] + "\n" + OLD_OPERATIONS[previous["task_operation"]]
        new_segment = "Requested operation: " + row["task_operation"] + "\n" + OPERATIONS[row["task_operation"]]
        assert old_prompt.count(old_segment) == 1, "OLD_OPERATION_SEGMENT_NOT_UNIQUE"
        prompt = diag.make_prompt(tokenizer, row, row["display"], source)
        assert prompt == old_prompt.replace(old_segment, new_segment, 1), "UNAUTHORIZED_PROMPT_BYTE_DIFF"
        assert all(f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" in prompt for turn in source["turns"]), "INCOMPLETE_SOURCE"
        parity.append({"candidate_id":row["candidate_id"], "old_prompt_binding":bind(old_prompt_path),
            "old_task_operation":previous["task_operation"], "new_task_operation":row["task_operation"],
            "only_requested_operation_and_instruction_changed":True, "display_source_stem_seed_unchanged":True})
        assert question["candidate_id"] not in prompt, "CANDIDATE_ID_MUST_NOT_REVEAL_GOLD_ARITY"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(ids) + MAX_TOKENS <= MODEL_LEN, "CONTEXT_OVERFLOW_NO_TRUNCATION"
        schema = diag.repair.schema(row["labels"], refs, row["question_type"])
        assert schema == read(old_schema_path), "SCHEMA_CHANGED"
        compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
        prompt_path = OUT / "prompts" / (row["request_id"]+".txt")
        schema_path = OUT / "schemas" / (row["request_id"]+".json")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with prompt_path.open("x", encoding="utf-8") as stream:
            stream.write(prompt)
            stream.flush()
            os.fsync(stream.fileno())
        save(schema_path, schema)
        assert schema_path.read_bytes() == old_schema_path.read_bytes(), "SCHEMA_BYTES_CHANGED"
        row.update(prompt_path=str(prompt_path), schema_path=str(schema_path), prompt_tokens=len(ids))
        prepared.append(row)
        token_ids[row["request_id"]] = ids
        bindings.extend((bind(prompt_path), bind(schema_path), bind(old_prompt_path), bind(old_schema_path), question["source_binding"]))
    save(OUT / "PUBLIC_REQUESTS.json", {"requests": prepared})
    save(OUT / "PROMPT_TOKEN_IDS.json", token_ids)
    bindings.extend(bind(OUT / name) for name in ("PUBLIC_REQUESTS.json", "PROMPT_TOKEN_IDS.json"))
    save(OUT / "PREPARATION.json", {"requests": 3, "bindings": bindings,
        "private_binding": frozen["private_binding"], "fixed_order": [r["request_id"] for r in prepared],
        "max_output_tokens": MAX_TOKENS, "max_model_len": MODEL_LEN,
        "temperature": .8, "top_p": .9, "automatic_retry": False, "source_truncation": False,
        "native_hidden_thinking": False, "generation_reads_gold": False,
        "semantic_review_required_before_feedback": True, "formal_weakness_admission": False,
        "status":"PREPARED_CPU_ONLY_GPU_NOT_RUN", "model_role":"B1", "model_directory":endpoint["model_directory"],
        "revision_binding":bind(REVISION), "metadata_review_binding":bind(REVIEW),
        "fixed_selection_before_new_outputs":True, "prompt_parity":parity})
    print(json.dumps({"prepared":3,"max_prompt_tokens":max(r["prompt_tokens"] for r in prepared)}), flush=True)

def verify(require_review=False):
    metadata_gate()
    prep = read(OUT / "PREPARATION.json")
    for item in prep["bindings"]:
        check(item)
    rows = read(OUT / "PUBLIC_REQUESTS.json")["requests"]
    tokens = read(OUT / "PROMPT_TOKEN_IDS.json")
    assert len(rows) == 3 and [r["request_id"] for r in rows] == prep["fixed_order"]
    assert [row["candidate_id"] for row in rows] == [change["candidate_id"] for change in metadata_gate()["changes"]]
    assert all(row["task_operation"] == "ORDER_ENQUIRY_PROGRESSION" for row in rows)
    assert set(tokens) == set(prep["fixed_order"])
    for row in rows:
        assert len(tokens[row["request_id"]]) == row["prompt_tokens"]
        assert row["prompt_tokens"]+MAX_TOKENS <= MODEL_LEN
        assert not {"gold", "canonical_gold", "display_gold", "evidence", "private_option_notes"} & set(row)
    if require_review:
        review = read(HERE / "PREFLIGHT_REVIEW.json")
        assert review["status"] == "PASS_CPU_INTERFACE_ONLY"
        for item in review["bindings"]:
            check(item)
        reviewed = {i["path"] for i in review["bindings"]}
        assert {str(Path(__file__)), str(OUT / "PREPARATION.json"), str(OUT / "PUBLIC_REQUESTS.json")} <= reviewed
    return prep, rows, tokens

def run(execute_gpu=False):
    assert execute_gpu, "EXPLICIT_GPU_FLAG_REQUIRED"
    assert not (OUT / "STARTED.json").exists(), "NO_RERUN_OR_AUTOMATIC_RETRY"
    prep, rows, tokens = verify(True)
    for key, value in {"VLLM_WORKER_MULTIPROC_METHOD":"spawn", "VLLM_USE_FLASHINFER_SAMPLER":"0",
        "HF_HUB_OFFLINE":"1", "TOKENIZERS_PARALLELISM":"false"}.items():
        os.environ[key] = value
    endpoint = read(ENDPOINT)
    assert Path(sys.prefix).resolve() == Path(endpoint["python_prefix"]).resolve()
    for item in endpoint["model_bindings"]+endpoint["tokenizer_bindings"]+endpoint["code_bindings"]:
        check(item)
    backend_path = next(Path(i["path"]) for i in endpoint["code_bindings"] if Path(i["path"]).name == "vllm_backends.py")
    sys.path.insert(0, str(backend_path.parents[3]))
    from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
    from vllm import SamplingParams, sampling_params
    save(OUT / "STARTED.json", {"time_unix":time.time(),"preparation":bind(OUT / "PREPARATION.json"),
        "preflight":bind(HERE / "PREFLIGHT_REVIEW.json"),"requests":3,"gold_read":False,"automatic_retry":False})
    backend, failure, submitted = None, None, set()
    errors = []
    started = time.perf_counter()
    try:
        backend = VLLMSolverBackend(model_path=Path(endpoint["model_directory"]),tokenizer_path=Path(endpoint["tokenizer_directory"]),
            max_model_len=MODEL_LEN,temperature=.8,top_p=.9,max_tokens=MAX_TOKENS,gpu_memory_utilization=.65)
        assert str(backend.model_identifier) == endpoint["model_directory"]
        for row in rows:
            rid = row["request_id"]
            path = OUT / "raw" / (rid+".json")
            try:
                prompt = Path(row["prompt_path"]).read_text()
                params = SamplingParams(temperature=.8, top_p=.9,max_tokens=MAX_TOKENS,seed=row["sampling_seed"],
                    structured_outputs=sampling_params.StructuredOutputsParams(json=read(row["schema_path"])))
                save(OUT / "submitted" / (rid+".json"), {"request_id":rid,"time_unix":time.time()})
                submitted.add(rid)
                returned = backend.engine.generate([prompt],[params],use_tqdm=False)
                raw = helper.preserve_response(row, returned, endpoint)
                save(path, raw)
                problems = helper.capture.response_errors(raw,prompt,tokens[rid])
                save(OUT / "raw_validation" / (rid+".json"),{"request_id":rid,"errors":problems,"raw_binding":bind(path)})
                if problems:
                    errors.append({"request_id":rid,"errors":problems})
                print("RETURNED "+rid,flush=True)
            except Exception:
                error = traceback.format_exc()
                errors.append({"request_id":rid,"error":error})
                if not path.exists():
                    save(path,{"request_id":rid,"status":"BACKEND_ERROR" if rid in submitted else "NOT_SUBMITTED","raw_output":None,"failure":error})
    except BaseException:
        failure = traceback.format_exc()
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                failure = (failure or "")+traceback.format_exc()
        for row in rows:
            path = OUT / "raw" / (row["request_id"]+".json")
            if not path.exists():
                save(path,{"request_id":row["request_id"],"status":"BACKEND_ERROR" if row["request_id"] in submitted else "NOT_SUBMITTED","raw_output":None,"failure":failure})
        save(OUT / "RAW_RESPONSE_FREEZE.json",{"requests":3,"request_ids":prep["fixed_order"],"started_binding":bind(OUT / "STARTED.json"),
            "wall_seconds":time.perf_counter()-started,"submitted":len(submitted),"failure":failure,"request_errors":errors,
            "bindings":[bind(p) for p in sorted((OUT / "raw").glob("*.json"))],
            "validation_bindings":[bind(p) for p in sorted((OUT / "raw_validation").glob("*.json"))]})
    if failure or errors:
        raise RuntimeError("Frozen failures preserved; inspect RAW_RESPONSE_FREEZE.json, no automatic retry")

def analyze():
    prep, rows, tokens = verify()
    frozen = read(OUT / "RAW_RESPONSE_FREEZE.json")
    paths = {str(OUT / "raw" / (r["request_id"]+".json")) for r in rows}
    assert frozen["requests"] == 3 and frozen["request_ids"] == prep["fixed_order"]
    assert len(frozen["bindings"]) == 3 and {b["path"] for b in frozen["bindings"]} == paths
    assert {str(p) for p in (OUT / "raw").glob("*.json")} == paths
    for item in frozen["bindings"]+frozen["validation_bindings"]:
        check(item)
    check(frozen["started_binding"])
    start = read(OUT / "STARTED.json")
    assert start["preparation"] == bind(OUT / "PREPARATION.json")
    check(start["preflight"])
    check(prep["private_binding"])
    private = {q["candidate_id"]:q for q in read(prep["private_binding"]["path"])["questions"]}
    records, scores = [], []
    for row in rows:
        raw_path = OUT / "raw" / (row["request_id"]+".json")
        raw = read(raw_path)
        record = {"request":row,"raw_binding":bind(raw_path),"semantic_review":"PENDING","feedback_eligible":False}
        answer, valid = [], False
        try:
            assert raw["request_id"] == row["request_id"] and raw["status"] == "RETURNED"
            assert not helper.capture.response_errors(raw,Path(row["prompt_path"]).read_text(),tokens[row["request_id"]])
            assert raw["finish_reason"] == "stop", "TRUNCATED_OR_NONSTOP_COMPLETION"
            parsed, answer = diag.repair.validate(raw["raw_output"],row,read(row["schema_path"]))
            consistency = diag.mechanical_consistency(row,parsed,answer)
            valid = True
            record.update(status="STRUCTURALLY_VALID_REASONS_SAVED",parsed=parsed,answer_labels=answer,mechanical_consistency=consistency)
        except Exception as error:
            record.update(status="INVALID",error=str(error))
        records.append(record)
        score = helper.score_record(row,private[row["candidate_id"]],answer,valid)
        score.update(raw_binding=bind(raw_path),invalid_error=record.get("error"))
        scores.append(score)
    summary = {**helper.aggregate(scores),"independent_questions":3,"semantic_review":"PENDING","feedback_eligible":False,
        "cot_causal_effect_claim":False,"formal_weakness_claim":False,"raw_freeze_binding":bind(OUT / "RAW_RESPONSE_FREEZE.json"),
        "mc_judgment_answer_inconsistent":sum(r.get("mechanical_consistency",{}).get("answer_matches_supported_labels") is False for r in records)}
    save(OUT / "SCORES.json",{"records":scores,"gold_read_after_all_raw_sealed":True,"raw_freeze_binding":bind(OUT / "RAW_RESPONSE_FREEZE.json")})
    save(OUT / "REASON_REVIEW_PACKET.json",{"requests":records,"semantic_review":"PENDING","feedback_eligible":False})
    save(OUT / "ANALYSIS_SUMMARY.json",summary)
    lines=["# 排序操作 V2 三题可见理由复测", "", f"固定3题；结构有效{summary['valid_requests']}/3，最终答案正确{summary['correct']}/3。简短理由与引用语义尚待独立审核，不能仅凭答案或JSON有效认证反馈。", ""]
    for row, record, score in zip(rows,records,scores):
        lines += ["## "+row["candidate_id"],"",row["question"],"",*(f"- {k}. {v}" for k,v in row["display"].items()),"",
            "显示Gold："+" ".join(score["display_gold"]),"", "最终集合/顺序正确："+str(score["exact"]),"", "```json",
            read(record["raw_binding"]["path"]).get("raw_output") or "null", "```", ""]
    path = OUT / "EXAMPLES_ZH.md"
    with path.open("x",encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps(summary),flush=True)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","verify","run","analyze"))
    parser.add_argument("--execute-gpu",action="store_true")
    args=parser.parse_args()
    {"prepare":prepare,"verify":lambda: (verify(),print("VERIFIED")),"run":lambda:run(args.execute_gpu),"analyze":analyze}[args.action]()
