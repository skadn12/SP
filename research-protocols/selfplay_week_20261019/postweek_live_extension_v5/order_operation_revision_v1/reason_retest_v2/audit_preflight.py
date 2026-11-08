"""Independent CPU interface audit of three fixed V2 operation-only retests.

This process does not parse private scoring, model output or scoring reports.
The reviewer has prior original-Gold exposure; this is not a blind review.
"""
from __future__ import annotations

import ast
import collections
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OLD = BASE / "reason_diagnostic_v5"
OUT = HERE / "execution"
REVISION = HERE.parent / "OPERATION_METADATA_REVISION_V2.json"
METADATA_REVIEW = HERE.parent / "OPERATION_METADATA_REVIEW_V2.json"
REVISION_SHA = "22f3a6093f32773d6daf9ed2d1c063a1029a50301aeef1e1793268793d630a6d"
METADATA_REVIEW_SHA = "77954b87e68e8e39bbe95e7809da80f376fc8cddcde322f2f0c8094d05039354"
FIXED_IDS = ["cc-train-episode-19708__ORDER1", "cc-train-episode-20432__ORDER1", "cc-train-episode-20432__ORDER2"]
NEW_OPERATION = "ORDER_ENQUIRY_PROGRESSION"
NEW_INSTRUCTION = "Reconstruct the enquiry or record disclosure progression requested by the question, grounded in the supplied dialogue. Distinguish statements about constraints, replies and unresolved state from actual journeys or completed visits. This operation is not numeric clock-time sorting."
OLD_INSTRUCTIONS = {
    "ORDER_EVENT_TIME": "Apply the time ordering requested by the question using the supplied dialogue.",
    "ORDER_CORRECTION_CHAIN": "Reconstruct the correction sequence requested by the question using the supplied dialogue.",
}
SOURCE_EXPECTED = {
    "cc-train-episode-19708": (254, "fedfc652a0b2edb88f6089559f4b4d5285d851db3160c78bbb7442da6ac052cc"),
    "cc-train-episode-20432": (263, "b749795d2a10bcf1f5d5a1591caa9e5bc02be823c2e410fe5031627aac4a4699"),
}
READ_LOG = []
HASH_CACHE = {}


def read(path):
    path = Path(path).resolve()
    assert "PRIVATE" not in path.name and path.name not in {"SCORES.json", "ANALYSIS_SUMMARY.json", "REASON_REVIEW_PACKET.json", "EXAMPLES_ZH.md"}, "AUDIT_FORBIDDEN_PARSE: " + str(path)
    assert "raw" not in path.parts, "AUDIT_MUST_PRECEDE_NEW_OUTPUTS"
    READ_LOG.append(str(path))
    return json.loads(path.read_text())


def bind(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        HASH_CACHE[key] = {"path": str(path), "sha256": digest.hexdigest(), "bytes": stat.st_size}
    return HASH_CACHE[key]


def check(item):
    actual = bind(item["path"])
    assert all(actual[k] == item[k] for k in ("path", "sha256", "bytes")), "BINDING_CHANGED: " + item["path"]


def nested_bindings(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            yield {k: value[k] for k in ("path", "sha256", "bytes")}
        else:
            for child in value.values():
                yield from nested_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_bindings(child)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def independently_rebuilt_schema(labels, refs):
    item = {"type": "object", "additionalProperties": False, "required": ["judgment", "turn_refs", "reason"], "properties": {
        "judgment": {"type": "string", "enum": ["supported", "contradicted", "not_supported", "ambiguous"]},
        "turn_refs": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": refs}},
        "reason": {"type": "string", "minLength": 1, "maxLength": 320}}}
    return {"type": "object", "additionalProperties": False, "required": ["item_evidence", "answer_reason", "final_answer"], "properties": {
        "item_evidence": {"type": "object", "additionalProperties": False, "required": labels, "properties": {label: item for label in labels}},
        "answer_reason": {"type": "string", "minLength": 1, "maxLength": 800},
        "final_answer": {"type": "string", "pattern": r"^<box>[A-F]( [A-F]){0,5}</box>$"}}}


def static_execution_review(module):
    text = (HERE / "run.py").read_text()
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    run_text = ast.get_source_segment(text, functions["run"])
    analyze_text = ast.get_source_segment(text, functions["analyze"])
    # Audited exact source ordering: preserve every backend return before validation.
    assert run_text.index("save(path, raw)") < run_text.index("helper.capture.response_errors(raw,")
    assert sum(isinstance(node, ast.Call) and ast.unparse(node.func) == "backend.engine.generate" for node in ast.walk(functions["run"])) == 1
    assert not any(isinstance(node, ast.While) for node in ast.walk(functions["run"]))
    assert "for row in rows:" in run_text and "seed=row[\"sampling_seed\"]" in run_text
    assert "NO_RERUN_OR_AUTOMATIC_RETRY" in run_text and "EXPLICIT_GPU_FLAG_REQUIRED" in run_text
    assert "verify(True)" in run_text and "structured_outputs=" in run_text
    assert "if not path.exists():" in run_text and '"BACKEND_ERROR" if' in run_text and '"NOT_SUBMITTED"' in run_text
    assert run_text.index("finally:") < run_text.index('save(OUT / "RAW_RESPONSE_FREEZE.json"')
    assert not any(isinstance(node, ast.Call) and ast.unparse(node.func) == "read" and "private" in ast.unparse(node) for node in ast.walk(functions["run"]))
    private_parse = analyze_text.index('read(prep["private_binding"]["path"])')
    for guard in ('read(OUT / "RAW_RESPONSE_FREEZE.json")', 'len(frozen["bindings"]) == 3', '{str(p) for p in (OUT / "raw").glob("*.json")} == paths', 'check(frozen["started_binding"])', 'check(start["preflight"])'):
        assert analyze_text.index(guard) < private_parse
    assert '"semantic_review":"PENDING"' in analyze_text and '"feedback_eligible":False' in analyze_text
    assert '"cot_causal_effect_claim":False' in analyze_text and '"formal_weakness_claim":False' in analyze_text
    exporter_text = (HERE / "export_inputs.py").read_text()
    assert "keys.append(private[cid])" in exporter_text and "{'questions': keys}" in exporter_text
    # Original private objects are selected unchanged by the author/exporter.
    # This audit deliberately cannot independently compare parsed private values.
    assert isinstance(module.diag.COMMON, str)
    return {"author_run_line": functions["run"].lineno, "author_analyze_line": functions["analyze"].lineno,
        "one_generate_call_per_question": True, "raw_persisted_before_validation": True,
        "all_three_terminal_raws_frozen_before_private_parse": True,
        "failed_and_not_submitted_retained_in_fixed_denominator": True,
        "no_retry_or_input_truncation_found": True, "exporter_selects_original_private_objects_without_modification": True,
        "current_audit_independently_compared_private_values": False}


def main():
    report_path = HERE / "PREFLIGHT_REVIEW.json"
    md_path = HERE / "PREFLIGHT_REVIEW_ZH.md"
    assert not report_path.exists() and not md_path.exists(), "NO_REVIEW_OVERWRITE"
    assert not (OUT / "STARTED.json").exists() and not (OUT / "RAW_RESPONSE_FREEZE.json").exists(), "NEW_GPU_ALREADY_STARTED"
    spec = importlib.util.spec_from_file_location("independent_v5_operation_retest_v2", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prep, rows, token_ids = module.verify()
    revision, review = read(REVISION), read(METADATA_REVIEW)
    assert bind(REVISION)["sha256"] == REVISION_SHA and bind(METADATA_REVIEW)["sha256"] == METADATA_REVIEW_SHA
    assert review["status"] == "PASS" and len(review["mapping_reviews"]) == 3
    assert [r["candidate_id"] for r in review["mapping_reviews"]] == FIXED_IDS
    assert all(r["verdict"] == "PASS" and r["new_task_operation"] == NEW_OPERATION for r in review["mapping_reviews"])
    assert review["policy_checks"]["old_scores_reusable"] is False and review["policy_checks"]["solver_score_filtering"] is False
    review_bindings = list(nested_bindings(review))
    assert bind(REVISION) in review_bindings
    for item in review_bindings:
        check(item)
    assert [c["candidate_id"] for c in revision["changes"]] == FIXED_IDS
    assert revision["operation_definition"] == NEW_INSTRUCTION
    assert revision["unchanged_ordering_operation"] == {"candidate_id": "cc-train-episode-19708__ORDER2", "task_operation": "ORDER_RECORD_RECONCILIATION"}
    assert module.OPERATIONS == {NEW_OPERATION: NEW_INSTRUCTION} and module.OLD_OPERATIONS == OLD_INSTRUCTIONS
    assert all(c["new_task_operation"] == NEW_OPERATION for c in revision["changes"])
    export = read(HERE / "inputs/EXPORT_FREEZE.json")
    public = read(HERE / "inputs/PUBLIC_QUESTIONS.json")["questions"]
    old_public = {q["candidate_id"]: q for q in read(OLD / "inputs/PUBLIC_QUESTIONS.json")["questions"]}
    old_rows = {q["candidate_id"]: q for q in read(OLD / "execution/PUBLIC_REQUESTS.json")["requests"]}
    old_prep = read(OLD / "execution/PREPARATION.json")
    old_bound = {b["path"]: b for b in old_prep["bindings"]}
    canonical = {q["candidate_id"]: q for q in read(BASE / "solver_v5/inputs/PUBLIC_QUESTIONS.json")["questions"]}
    freeze = read(BASE / "QUESTION_FREEZE.json")
    slots = {q["candidate_id"]: q for q in freeze["questions"]}
    assert freeze["status"] == "FROZEN_BEFORE_SOLVER" and freeze["gates"]["content_review_status"] == "PASS"
    assert [q["candidate_id"] for q in public] == [r["candidate_id"] for r in rows] == FIXED_IDS == export["fixed_candidate_ids"]
    assert len(rows) == prep["requests"] == export["requests"] == 3
    assert export["exporter_gold_aware"] is True and export["inference_gold_aware"] is False
    assert export["stem_display_source_gold_unchanged"] is True and export["operation_only_revision"] is True
    assert prep["private_binding"] == export["private_binding"]
    # Hash bytes only; neither private file is parsed or displayed.
    check(export["private_binding"])
    check(export["original_private_binding"])
    assert prep["revision_binding"] == bind(REVISION) and prep["metadata_review_binding"] == bind(METADATA_REVIEW)
    assert prep["fixed_selection_before_new_outputs"] is True
    assert prep["status"] == "PREPARED_CPU_ONLY_GPU_NOT_RUN"
    assert prep["max_output_tokens"] == module.MAX_TOKENS == 3072
    assert prep["max_model_len"] == module.MODEL_LEN == 16384
    assert prep["temperature"] == .8 and prep["top_p"] == .9
    for key in ("automatic_retry", "source_truncation", "native_hidden_thinking", "generation_reads_gold", "formal_weakness_admission"):
        assert prep[key] is False
    assert prep["semantic_review_required_before_feedback"] is True
    endpoint = read(module.ENDPOINT)
    assert endpoint["model_role"] == prep["model_role"] == "B1"
    assert endpoint["model_directory"] == prep["model_directory"] == "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model"
    assert endpoint["tokenizer_directory"] == "/root/autodl-tmp/Qwen3-4B"
    assert Path(sys.prefix).resolve() == Path(endpoint["python_prefix"]).resolve()
    assert sys.flags.dont_write_bytecode
    dependency_bindings = endpoint["model_bindings"] + endpoint["tokenizer_bindings"] + endpoint["code_bindings"]
    for item in prep["bindings"] + export["bindings"] + dependency_bindings:
        check(item)
    static_review = static_execution_review(module)
    from transformers import AutoTokenizer
    from jsonschema import Draft7Validator
    import xgrammar as xgr
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    question_checks, prompt_schema_paths = [], []
    expected_public_keys = {"candidate_id", "root_id", "question_type", "task_operation", "question", "source_binding", "display"}
    expected_row_keys = expected_public_keys | {"request_id", "labels", "sampling_seed", "cohort", "condition", "source_turns", "prompt_path", "schema_path", "prompt_tokens"}
    assert module.PUBLIC_KEYS == expected_public_keys
    for row, question, change in zip(rows, public, revision["changes"]):
        cid = row["candidate_id"]
        previous, old_row = old_public[cid], old_rows[cid]
        assert set(question) == expected_public_keys and set(row) == expected_row_keys
        assert all(row[k] == value for k, value in question.items())
        assert {k:v for k,v in question.items() if k != "task_operation"} == {k:v for k,v in previous.items() if k != "task_operation"}
        assert question["task_operation"] == NEW_OPERATION and previous["task_operation"] == change["old_task_operation"]
        assert row["question_type"] == "ordering" and row["cohort"] == "v5_operation_retest_v2" and row["condition"] == "FULL"
        assert row["request_id"] == cid + "__v5_reason_operation_retest_v2_s00"
        independent_seed = int.from_bytes(hashlib.sha256(("reason_v5:2026102307:"+cid).encode()).digest()[:4], "big") % 2147483647
        assert row["sampling_seed"] == old_row["sampling_seed"] == independent_seed
        assert row["display"] == old_row["display"] and row["labels"] == list(row["display"]) == list("ABCD")
        assert row["question"] == canonical[cid]["question"] and row["root_id"] == canonical[cid]["root_id"]
        item_by_text = {text: key for key, text in canonical[cid]["items"].items()}
        assert len(item_by_text) == 4 and set(row["display"].values()) == set(item_by_text)
        assert fingerprint([item_by_text[text] for text in row["display"].values()]) == slots[cid]["display_order_digest"]
        check(row["source_binding"])
        expected_turns, expected_sha = SOURCE_EXPECTED[row["root_id"]]
        assert row["source_binding"]["sha256"] == expected_sha == canonical[cid]["source_sha256"]
        frozen_source = {k: slots[cid]["source_binding"][k] for k in ("path", "sha256", "bytes")}
        assert row["source_binding"] == previous["source_binding"] == frozen_source
        source = read(row["source_binding"]["path"])
        refs = [turn["turn_ref"] for turn in source["turns"]]
        assert source["root_id"] == row["root_id"] and len(refs) == len(set(refs)) == expected_turns == row["source_turns"]
        dialogue = "\n".join(f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" for turn in source["turns"])
        user = (f"Dialogue:\n{dialogue}\n\nQuestion ({row['question_type']}):\n{row['question']}\n\nDisplayed items:\n" +
            "\n".join(f"{label}. {text}" for label, text in row["display"].items()) +
            f"\n\nRequested operation: {NEW_OPERATION}\n{NEW_INSTRUCTION}\n\n{module.diag.COMMON}")
        rebuilt_prompt = tokenizer.apply_chat_template([
            {"role":"system", "content":"Use only the supplied dialogue. Follow the question's requested operation. Give concise, verifiable explanations before the final boxed answer."},
            {"role":"user", "content":user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        old_prompt_path, old_schema_path = Path(old_row["prompt_path"]), Path(old_row["schema_path"])
        check(old_bound[str(old_prompt_path)]); check(old_bound[str(old_schema_path)])
        old_bytes = old_prompt_path.read_bytes()
        old_segment = ("Requested operation: " + previous["task_operation"] + "\n" + OLD_INSTRUCTIONS[previous["task_operation"]]).encode()
        new_segment = ("Requested operation: " + NEW_OPERATION + "\n" + NEW_INSTRUCTION).encode()
        assert old_bytes.count(old_segment) == 1
        new_bytes = Path(row["prompt_path"]).read_bytes()
        assert new_bytes == rebuilt_prompt.encode() == old_bytes.replace(old_segment, new_segment, 1)
        assert rebuilt_prompt == module.diag.make_prompt(tokenizer, row, row["display"], source)
        assert cid not in rebuilt_prompt and dialogue in rebuilt_prompt
        assert all(f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" in rebuilt_prompt for turn in source["turns"])
        ids = tokenizer.encode(rebuilt_prompt, add_special_tokens=False)
        assert ids == token_ids[row["request_id"]] and len(ids) == row["prompt_tokens"]
        assert len(ids) + 3072 <= 16384
        schema = read(row["schema_path"])
        assert schema == independently_rebuilt_schema(row["labels"], refs) == module.diag.repair.schema(row["labels"], refs, "ordering")
        assert Path(row["schema_path"]).read_bytes() == old_schema_path.read_bytes()
        assert schema == read(old_schema_path)
        Draft7Validator.check_schema(schema)
        compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
        # Exercise the new operation's generic ordering interface with a synthetic,
        # unscored public label ordering, never with a private expected answer.
        synthetic = {"item_evidence": {label:{"turn_refs":[refs[0]]} for label in row["labels"]}}
        consistency = module.diag.mechanical_consistency(row, synthetic, row["labels"])
        assert consistency["task_operation"] == NEW_OPERATION and consistency["semantic_order_review_required"] is True
        assert consistency["reference_location_order_applicable"] is False
        prompt_schema_paths += [Path(row["prompt_path"]), Path(row["schema_path"]), old_prompt_path, old_schema_path]
        question_checks.append({"candidate_id": cid, "request_id": row["request_id"], "source_turns":len(refs),
            "old_task_operation":previous["task_operation"], "new_task_operation":NEW_OPERATION,
            "sampling_seed":independent_seed, "sampling_seed_unchanged":True, "original_display_digest_verified":True,
            "old_prompt_unique_operation_segment_count":1, "only_operation_and_adjacent_instruction_changed":True,
            "source_stem_display_unchanged":True, "prompt_bytes_independently_rebuilt":True,
            "prompt_tokens":len(ids), "context_headroom":16384-len(ids)-3072,
            "token_ids_independently_rebuilt":True, "schema_bytes_equal_old":True,
            "schema_independently_rebuilt":True, "draft7_checked":True, "xgrammar_compiled":True,
            "generic_ordering_interface_accepts_new_operation":True})
    assert prep["fixed_order"] == [r["request_id"] for r in rows] and set(token_ids) == set(prep["fixed_order"])
    assert len({r["sampling_seed"] for r in rows}) == len({r["request_id"] for r in rows}) == 3
    assert not (OUT / "STARTED.json").exists() and not (OUT / "RAW_RESPONSE_FREEZE.json").exists()
    paths = [Path(__file__), HERE / "run.py", HERE / "export_inputs.py", HERE / "PROTOCOL_ZH.md",
        HERE / "inputs/EXPORT_FREEZE.json", HERE / "inputs/PUBLIC_QUESTIONS.json",
        OUT / "PREPARATION.json", OUT / "PUBLIC_REQUESTS.json", OUT / "PROMPT_TOKEN_IDS.json", REVISION, METADATA_REVIEW,
        OLD / "inputs/EXPORT_FREEZE.json", OLD / "inputs/PUBLIC_QUESTIONS.json", OLD / "execution/PREPARATION.json",
        OLD / "execution/PUBLIC_REQUESTS.json", OLD / "execution/PROMPT_TOKEN_IDS.json", BASE / "QUESTION_FREEZE.json",
        BASE / "solver_v5/inputs/PUBLIC_QUESTIONS.json", module.ENDPOINT]
    all_bindings = [bind(path) for path in paths+prompt_schema_paths] + prep["bindings"] + export["bindings"] + review_bindings + dependency_bindings + [export["private_binding"], export["original_private_binding"]]
    unique = {item["path"]:item for item in all_bindings}
    result = {"schema_version":"live_extension_v5_operation_retest_v2_independent_preflight_v1",
        "status":"PASS_CPU_INTERFACE_ONLY", "reviewed_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "reviewer_id":"/root/solver_prepare/v5_freeze_audit", "independent_of_retest_author":True,
        "scope":"Exactly three fixed V2 operation-only reason retests; CPU interface and immutable input checks only.",
        "requests":3, "fixed_candidate_ids":FIXED_IDS, "question_type_counts":{"ordering":3},
        "root_counts":dict(collections.Counter(r["root_id"] for r in rows)),
        "model_role":"B1", "model_directory":endpoint["model_directory"], "tokenizer_directory":endpoint["tokenizer_directory"],
        "python_prefix":str(Path(sys.prefix).resolve()), "max_output_tokens":3072, "max_model_len":16384,
        "temperature":.8, "top_p":.9, "max_prompt_tokens":max(q["prompt_tokens"] for q in question_checks),
        "minimum_context_headroom":min(q["context_headroom"] for q in question_checks),
        "automatic_retry":False, "source_truncation":False, "native_hidden_thinking":False,
        "new_model_calls":0, "gpu_execution_performed":False, "scores_or_new_raws_parsed":False,
        "gold_exposure":{"prior_original_gold_exposure":True, "review_is_gold_blind":False,
            "current_audit_private_json_parsed":False, "private_files_byte_hashed_only":True,
            "current_audit_independently_compared_private_values":False,
            "assurance":"Author exporter statically selects original private objects unchanged; immutable private file hashes verified without parsing. Prior original-Gold exposure is disclosed."},
        "static_execution_review":static_review,
        "checks_passed":["V2 metadata PASS is bound to exact proposal and all source bindings",
            "fixed three candidate IDs selected before any new outputs; no score filtering",
            "original question, complete source, display, sampling seed and schema preserved",
            "new prompt differs only at unique Requested operation and adjacent instruction segment",
            "three prompts and token IDs independently rebuilt with native thinking disabled",
            "three independent schemas pass Draft7 validation and xgrammar compilation",
            "B1 model, tokenizer, runtime code and old prompt/schema byte bindings verified",
            "3072 output budget fits complete sources under 16384 without truncation",
            "one model request per question, raw saved before validation, all three terminal raws sealed before private scoring",
            "semantic review and feedback admission remain separately gated"],
        "scope_limits":["This is three questions, not a 14-question or 228-request rerun.",
            "No GPU generation, answer reliability certification, formal weakness admission, feedback benefit or CoT causal claim.",
            "Visible reasons are concise checkable explanations; not private internal reasoning.",
            "Private Gold preservation was audited through exporter code and immutable bindings, not independent private-value parsing.",
            "Three remapped questions share one enquiry-progression operation family; no three-family diversity claim."],
        "versions":{name:importlib.metadata.version(name) for name in ("transformers","torch","vllm","tokenizers","xgrammar","jsonschema")},
        "public_json_parse_log":READ_LOG, "questions":question_checks, "bindings":list(unique.values())}
    md = "\n".join(["# V2 三道排序题独立 CPU 接口审核", "", "结论：`PASS_CPU_INTERFACE_ONLY`。仅审核固定三题，不是全 14 题或 228 请求重跑。", "",
        "- 固定题：19708 ORDER1、20432 ORDER1、20432 ORDER2。均采用审核通过的 `ORDER_ENQUIRY_PROGRESSION`。",
        "- B1 模型、Qwen3-4B tokenizer、运行代码与原始依赖均核验 SHA256；使用 B1 venv 的 Python -B。",
        "- 独立重建三题完整 prompt、token IDs、schema；实际完成 Draft7 检查和 xgrammar 编译。",
        "- 254/263 条完整来源、题干、原始打乱 display、采样 seed 和 schema 保持不变；旧 prompt 只有唯一的操作行及紧邻 instruction 段落发生字节变更。",
        f"- 最大 prompt {result['max_prompt_tokens']} tokens；输出上限 3072、上下文 16384，最小余量 {result['minimum_context_headroom']} tokens。没有截断、重试或新模型调用。",
        "- 静态审核确认：每题一次请求，原始输出先保存再验证；三个终态 raw 全部封存后才可解析 private 评分。失败、无效及未提交均保留固定分母 3。",
        "- 新操作经原有 generic ordering mechanical interface 可接受；语义排序仍须独立人工／AI 内容审核，引用位置并不自动充当排序依据。", "",
        "审核者此前接触过原始 Gold，因此这不是完全盲审。本次没有解析新旧 PRIVATE_SCORING、SCORES、ANALYSIS 或新 raw；private 文件只按字节哈希。Gold 保持不变的保证来自导出代码原样选取旧私有对象及文件绑定，未对私有值再次独立解析比对。", "",
        "本审核不认证答案可靠度、不认证弱点或反馈收益，不证明 CoT 因果效果。三题属于一个 enquiry-progression 操作族，不能宣称为三个新的排序族。", "",
        "完整逐题检查、审计脚本、作者脚本、导出冻结、PREPARATION、PUBLIC_REQUESTS、PROMPT_TOKEN_IDS、旧 prompt/schema、V2 proposal/review 和 B1 依赖的绑定见 `PREFLIGHT_REVIEW.json`。", ""])
    with md_path.open("x",encoding="utf-8") as stream:
        stream.write(md); stream.flush(); os.fsync(stream.fileno())
    result["bindings"].append(bind(md_path))
    module.save(report_path,result)
    print(json.dumps({"status":result["status"],"requests":3,"max_prompt_tokens":result["max_prompt_tokens"],
        "minimum_context_headroom":result["minimum_context_headroom"],"new_model_calls":0,"report_binding":bind(report_path),"review_md_binding":bind(md_path)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
