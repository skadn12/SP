"""Independent CPU review of the v5 reason interface; no private Gold parse."""
from __future__ import annotations
import collections
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = HERE / "execution"

def read(path):
    return json.loads(Path(path).read_text())

def bind(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}

def check(item):
    actual = bind(item["path"])
    assert all(actual[key] == item[key] for key in ("path", "sha256", "bytes")), "BINDING_CHANGED:" + item["path"]

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def main():
    result_path = HERE / "PREFLIGHT_REVIEW.json"
    assert not result_path.exists(), "NO_REVIEW_OVERWRITE"
    assert not (OUT / "STARTED.json").exists(), "GPU_ALREADY_STARTED"
    spec = importlib.util.spec_from_file_location("independently_reviewed_v5_reason_run", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prep, rows, token_ids = module.verify()
    freeze = read(BASE / "QUESTION_FREEZE.json")
    assert freeze["status"] == "FROZEN_BEFORE_SOLVER" and freeze["gates"]["content_review_status"] == "PASS"
    canonical = read(BASE / "solver_v5/inputs/PUBLIC_QUESTIONS.json")["questions"]
    canonical = {q["candidate_id"]: q for q in canonical}
    slots = {q["candidate_id"]: q for q in freeze["questions"]}
    exported = read(HERE / "inputs/PUBLIC_QUESTIONS.json")["questions"]
    export = read(HERE / "inputs/EXPORT_FREEZE.json")
    assert len(rows) == len(exported) == 14
    assert {q["candidate_id"] for q in rows} == set(canonical) == set(slots)
    assert prep["private_binding"] == export["private_binding"]
    # Byte checks disclose no Gold JSON to this audit process.
    check(prep["private_binding"])
    assert prep["max_output_tokens"] == module.MAX_TOKENS == 3072
    assert prep["max_model_len"] == module.MODEL_LEN == 16384
    assert prep["temperature"] == .8 and prep["top_p"] == .9
    assert prep["automatic_retry"] is False and prep["source_truncation"] is False
    assert prep["native_hidden_thinking"] is False and prep["generation_reads_gold"] is False
    assert prep["semantic_review_required_before_feedback"] is True and prep["formal_weakness_admission"] is False
    endpoint = read(module.ENDPOINT)
    assert endpoint["model_role"] == "B1"
    assert endpoint["model_directory"] == "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model"
    assert endpoint["tokenizer_directory"] == "/root/autodl-tmp/Qwen3-4B"
    assert Path(sys.prefix).resolve() == Path(endpoint["python_prefix"]).resolve()
    dependency_bindings = endpoint["model_bindings"] + endpoint["tokenizer_bindings"] + endpoint["code_bindings"]
    for item in dependency_bindings:
        check(item)
    from transformers import AutoTokenizer
    from jsonschema import Draft7Validator
    import xgrammar as xgr
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    expected_row_keys = module.PUBLIC_KEYS | {"request_id", "labels", "sampling_seed", "cohort", "condition", "source_turns", "prompt_path", "schema_path", "prompt_tokens"}
    expected_counts = collections.Counter()
    expected_roots = collections.Counter()
    checks = []
    for row, public in zip(rows, exported):
        assert set(row) == expected_row_keys
        assert set(public) == module.PUBLIC_KEYS
        assert all(row[key] == value for key, value in public.items())
        cid = row["candidate_id"]
        q = canonical[cid]
        fs = slots[cid]
        field = "options" if q["question_type"] == "multi_select" else "items"
        assert row["request_id"] == cid + "__v5_reason_s00"
        assert row["cohort"] == "v5" and row["condition"] == "FULL"
        assert row["sampling_seed"] == module.seed(cid)
        assert row["root_id"] == q["root_id"] and row["question_type"] == q["question_type"]
        assert row["question"] == q["question"] and row["task_operation"] == fs["task_operation"]
        assert row["source_binding"]["sha256"] == q["source_sha256"] == fs["source_binding"]["sha256"]
        # Recover the original canonical option IDs from unique public text,
        # then check the frozen original display order without parsing Gold.
        by_text = {text: key for key, text in q[field].items()}
        assert len(by_text) == len(q[field])
        assert list(row["display"]) == list("ABCDEF"[:len(q[field])])
        assert sorted(row["display"].values()) == sorted(q[field].values())
        original_order = [by_text[text] for text in row["display"].values()]
        assert digest(original_order) == fs["display_order_digest"]
        assert row["labels"] == list(row["display"])
        source = read(row["source_binding"]["path"])
        check(row["source_binding"])
        assert source["root_id"] == row["root_id"]
        refs = [t["turn_ref"] for t in source["turns"]]
        assert refs and len(refs) == len(set(refs)) == row["source_turns"]
        prompt = module.diag.make_prompt(tokenizer, row, row["display"], source)
        assert cid not in prompt
        assert all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source["turns"])
        assert Path(row["prompt_path"]).read_text() == prompt
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert ids == token_ids[row["request_id"]] and len(ids) == row["prompt_tokens"]
        assert len(ids) + 3072 <= 16384
        schema = read(row["schema_path"])
        assert schema == module.diag.repair.schema(row["labels"], refs, row["question_type"])
        Draft7Validator.check_schema(schema)
        compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
        assert list(schema["properties"]) == ["item_evidence", "answer_reason", "final_answer"]
        assert schema["properties"]["item_evidence"]["required"] == row["labels"]
        for value in schema["properties"]["item_evidence"]["properties"].values():
            assert value["properties"]["reason"]["maxLength"] == 320
            assert value["properties"]["turn_refs"]["items"]["enum"] == refs
            assert value["properties"]["turn_refs"]["maxItems"] == 3
        assert schema["properties"]["answer_reason"]["maxLength"] == 800
        expected_counts[row["question_type"]] += 1
        expected_roots[row["root_id"]] += 1
        checks.append({"candidate_id": cid, "request_id": row["request_id"], "question_type": row["question_type"],
            "source_turns": len(refs), "prompt_tokens": len(ids), "context_headroom": 16384 - len(ids) - 3072,
            "frozen_display_digest_verified": True, "schema_compiled": True})
    assert dict(expected_counts) == {"multi_select": 10, "ordering": 4}
    assert dict(expected_roots) == {"cc-train-episode-19708": 7, "cc-train-episode-20432": 7}
    assert len({row["request_id"] for row in rows}) == len({row["sampling_seed"] for row in rows}) == 14
    assert not (OUT / "STARTED.json").exists()
    audit_paths = [HERE / "run.py", HERE / "export_inputs.py", HERE / "PROTOCOL_ZH.md", Path(__file__),
        OUT / "PREPARATION.json", OUT / "PUBLIC_REQUESTS.json", OUT / "PROMPT_TOKEN_IDS.json",
        HERE / "inputs/EXPORT_FREEZE.json", HERE / "inputs/PUBLIC_QUESTIONS.json",
        BASE / "QUESTION_FREEZE.json", BASE / "solver_v5/inputs/PUBLIC_QUESTIONS.json"]
    all_bindings = [bind(path) for path in audit_paths] + prep["bindings"] + dependency_bindings
    unique = {item["path"]: item for item in all_bindings}
    result = {"schema_version": "live_extension_v5_reason_independent_preflight_v1", "status": "PASS_CPU_INTERFACE_ONLY",
        "reviewed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "reviewer_id": "/root/solver_prepare/v5_solver_preflight",
        "reviewer_independent_of_reason_author": True, "model_role": "B1", "model_directory": endpoint["model_directory"],
        "tokenizer_directory": endpoint["tokenizer_directory"], "requests": 14, "question_type_counts": dict(expected_counts),
        "root_counts": dict(expected_roots), "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
        "minimum_context_headroom": min(c["context_headroom"] for c in checks), "max_output_tokens": 3072,
        "automatic_retry": False, "source_truncation": False, "gold_parsed_by_reviewer": False,
        "direct_scores_opened_by_reviewer": False, "gpu_execution_performed": False, "new_model_calls": 0,
        "checks_passed": ["all 14 frozen content-admitted candidates retained before solver results",
            "each original shuffled display order independently recovered from public text and frozen digest",
            "B1 endpoint model/tokenizer/code bytes verified", "14 unique request IDs and sampling seeds",
            "native thinking disabled and reasons precede boxed answer", "complete source turns in every rebuilt prompt",
            "all prompt bytes and token IDs independently rebuilt", "14 JSON schemas checked and xgrammar compiled",
            "3072 output budget fits all full inputs without truncation", "one request per question and no automatic retry",
            "all returned backend text preserved before validation", "all 14 terminal raws sealed before private Gold scoring",
            "semantic review remains pending and feedback ineligible"],
        "scope_limits": ["Interface and execution safety review only; no generation or answer accuracy claim",
            "Brief visible explanations are not private internal reasoning", "No CoT causal effect, formal weakness, feedback benefit, or training-admission claim",
            "Author/exporter know Gold; solver prepare/live and this audit do not parse private Gold"],
        "versions": {name: importlib.metadata.version(name) for name in ("transformers", "torch", "vllm", "tokenizers", "xgrammar", "jsonschema")},
        "questions": checks, "bindings": list(unique.values())}
    module.save(result_path, result)
    print(json.dumps({key: result[key] for key in ("status", "requests", "max_prompt_tokens", "minimum_context_headroom", "new_model_calls")}))

if __name__ == "__main__":
    main()
