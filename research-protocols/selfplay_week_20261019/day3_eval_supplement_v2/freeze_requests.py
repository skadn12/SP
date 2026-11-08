"""Public-only CPU request freeze using the existing native prompt contract."""
import ast
import hashlib
import importlib.util
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OUT = Path(__file__).resolve().parent
WEEK = OUT.parent
REPO = WEEK.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(ref):
    assert bind(ref["path"]) == ref, ("HASH_CHANGED", ref["path"])


def write_new(path, data):
    path = Path(path)
    assert not path.exists(), ("REFUSING_OVERWRITE", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


def write_json(path, obj):
    write_new(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    write_new(path, "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))


def main():
    assert not (OUT / "requests/REQUEST_FREEZE.json").exists()
    execution_binding = bind(OUT / "EXECUTION_FREEZE.json")
    freeze = read(OUT / "EXECUTION_FREEZE.json")
    assert freeze.get("status") == "FROZEN_BEFORE_QUESTION_AUTHORING"
    for ref in freeze["upstream"]:
        verify(ref)
    source_refs = [r["source"] for r in freeze["roots"]]
    for ref in source_refs:
        verify(ref)
    admission = read(OUT / "CONTENT_ADMISSION.json")
    assert admission["status"] == "CONTENT_ADMISSION_FINAL"
    for ref in admission["input_bindings"]:
        verify(ref)
    assert admission.get("minimum_coverage_status") == "PENDING_REQUEST_LENGTH_AND_SCOPED_COVERAGE_DECISION"
    frozen_slot_map = {s["candidate_id"]: s for s in freeze["slots"]}
    assert len(frozen_slot_map) == len(freeze["slots"])
    policy_path = WEEK / "day3_common_evaluation_v1/protocol/REQUEST_POLICY_V1.json"
    policy = read(policy_path)
    verify(next(r for r in freeze["upstream"] if r["path"] == str(policy_path)))
    baseline_freeze_path = WEEK / "day3_mainline_readiness_v1/common_eval/COMMON_REGRESSION_REQUEST_FREEZE.json"
    baseline_freeze = read(baseline_freeze_path)
    native_contract_ref = baseline_freeze["native_contract_binding"]
    verify(native_contract_ref)
    adapter_ref = policy["prompt"]["adapter"]
    verify(adapter_ref)
    spec = importlib.util.spec_from_file_location("supplement_public_adapter", adapter_ref["path"])
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    contract_path = REPO / "worktrees/h4-gate-downstream-v1/scripts/current_method/restart_n5_spell/h4_round0_b0_k3_difficulty_v1/contract.py"
    assert bind(contract_path) == native_contract_ref, "NATIVE_CONTRACT_BINDING_MISMATCH"
    source_code = contract_path.read_text()
    nodes = []
    for node in ast.parse(source_code).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STRUCTURED_SYSTEM_PROMPT" for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in ("dialogue_rows", "build_prompt"):
            nodes.append(node)
    namespace = {}
    exec(compile("from __future__ import annotations\n" + "\n\n".join(ast.get_source_segment(source_code, n) for n in nodes), str(contract_path), "exec"), namespace)
    tokenproof_path = WEEK / "day3_mainline_readiness_v1/training/TOKENIZATION_ZERO_TRUNCATION.json"
    tokenproof = read(tokenproof_path)
    tokenproof_ref = bind(tokenproof_path)
    import transformers
    import tokenizers
    declared_tokenizer_dir = Path(policy["length"]["tokenizer_directory"])
    tokenizer_dir = declared_tokenizer_dir
    if not tokenizer_dir.is_dir():
        # Workspace mount path can differ from the one used by earlier bound
        # evidence. Resolve only if every tokenizer artifact matches its bound
        # SHA256 and byte count; do not fetch or substitute a different tokenizer.
        candidates = [Path("/root/autodl-tmp/Qwen3-4B"), WEEK / "models/Qwen3-4B"]
        expected_by_name = {Path(ref["path"]).name: ref for ref in tokenproof["tokenizer_files"]}
        resolved = None
        for candidate in candidates:
            if candidate.is_dir() and all((candidate / name).is_file() and
                    bind(candidate / name)["sha256"] == ref["sha256"] and
                    bind(candidate / name)["bytes"] == ref["bytes"]
                    for name, ref in expected_by_name.items()):
                resolved = candidate
                break
        assert resolved is not None, ("BOUND_TOKENIZER_NOT_FOUND_OR_HASH_MISMATCH", str(declared_tokenizer_dir))
        tokenizer_dir = resolved
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    assert tokenproof["tokenizer_directory"] == policy["length"]["tokenizer_directory"]
    assert policy["length"]["reserved_answer_tokens"] == policy["sampling"]["max_output_tokens"]
    prompts, requests, maps, checks, choices, holds = [], [], [], [], {}, []
    sample = policy["sampling"]
    # Synthetic preflight required by REQUEST_POLICY_V1, independent of any
    # admitted content. Exercise all labels and cardinalities before requests.
    synthetic_roundtrips = 0
    for qtype in ("multi_select", "ordering"):
        for n in (4, 5, 6):
            field = "options" if qtype == "multi_select" else "items"
            synth = {"candidate_id": f"synthetic-{qtype}-{n}", "root_id": "synthetic",
                     "question_type": qtype, "question": "synthetic roundtrip", "source_path": "synthetic",
                     "source_sha256": "0" * 64, field: {label: f"content {i}" for i, label in enumerate(
                         list("ABCDEF")[:n] if field == "options" else [f"I{i+1}" for i in range(n)])}}
            variants = adapter.variants(synth, sample["display_seed"], 12 if qtype == "ordering" else 6)
            keys = list(synth[field])
            all_answers = ([tuple(label for bit, label in enumerate(keys) if mask & (1 << bit))
                            for mask in range(1, 2**n)] if qtype == "multi_select" else list(itertools.permutations(keys)))
            canonical = adapter.content_ids(synth)
            for variant in variants:
                inverse = {content_id: label for label, content_id in variant["display_to_content"].items()}
                for answer in all_answers:
                    content = [canonical[label] for label in answer]
                    display = [inverse[content_id] for content_id in content]
                    assert [variant["display_to_content"][label] for label in display] == content
                    synthetic_roundtrips += 1
    for row in admission["records"]:
        if row["status"] != "CONTENT_ACCEPTED":
            continue
        verify(row["public_question"])
        q = adapter.public_question(read(row["public_question"]["path"]))
        assert q["candidate_id"] == row["candidate_id"]
        slot = frozen_slot_map[q["candidate_id"]]
        assert row["global_slot_index"] == slot["global_slot_index"]
        source_ref = next(r["source"] for r in freeze["roots"] if r["root_id"] == row["root_id"])
        assert bind(q["source_path"]) == source_ref
        assert q["source_sha256"] == source_ref["sha256"]
        source = read(q["source_path"])
        assert source["root_id"] == q["root_id"]
        field = "options" if q["question_type"] == "multi_select" else "items"
        n = len(q[field])
        labels = list(q[field])
        variants_count = sample["multi_select_permutations"] if field == "options" else sample["ordering_permutations"]
        repetitions = sample["multi_select_samples_per_permutation"] if field == "options" else sample["ordering_samples_per_permutation"]
        variants = adapter.variants(q, sample["display_seed"], variants_count)
        assert variants == adapter.variants(q, sample["display_seed"], variants_count)
        assert len({tuple(v["display_order_content_ids"]) for v in variants}) == variants_count
        canonical = adapter.content_ids(q)
        answer_space = [tuple(label for bit, label in enumerate(labels) if mask & (1 << bit)) for mask in range(1, 2**n)] if field == "options" else list(itertools.permutations(labels))
        constraint_key = q["question_type"] + ":" + str(n)
        choices[constraint_key] = ["<box>" + ",".join(answer) + "</box>" for answer in answer_space]
        pending_prompts, pending_requests, pending_maps, pending_checks = [], [], [], []
        for vi, variant in enumerate(variants):
            inverse = {v: k for k, v in variant["display_to_content"].items()}
            assert set(inverse) == set(canonical.values())
            for answer in answer_space:
                content = [canonical[k] for k in answer]
                display = [inverse[k] for k in content]
                assert [variant["display_to_content"][k] for k in display] == content
            prompt = namespace["build_prompt"](tokenizer, q, {field: variant[field]}, namespace["dialogue_rows"](source), context_mode="full_context")
            token_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
            prompt_id = q["candidate_id"] + "::" + variant["variant_id"]
            phash = hashlib.sha256(prompt.encode()).hexdigest()
            pending_prompts.append({"prompt_id": prompt_id, "candidate_id": q["candidate_id"], "variant_id": variant["variant_id"], "prompt": prompt, "prompt_sha256": phash, "prompt_tokens": len(token_ids), "token_ids_sha256": digest(token_ids), "source_sha256": q["source_sha256"]})
            pending_maps.append({"prompt_id": prompt_id, "candidate_id": q["candidate_id"], "canonical_label_to_content": canonical, **variant})
            pending_checks.append({"prompt_id": prompt_id, "prompt_tokens": len(token_ids), "reserved_answer_tokens": sample["max_output_tokens"], "truncated": False, "answer_space_roundtrips": len(answer_space)})
            for si in range(repetitions):
                pending_requests.append({"request_id": prompt_id + "::sample_" + str(si), "candidate_id": q["candidate_id"], "root_id": q["root_id"], "split": "test", "question_type": q["question_type"], "global_slot_index": row["global_slot_index"], "branch": "direct", "variant_id": variant["variant_id"], "sample_index": si, "prompt_id": prompt_id, "prompt_sha256": phash, "display_mapping_sha256": digest(variant), "constraint_key": constraint_key, "sampling": {"temperature": sample["temperature"], "top_p": sample["top_p"], "max_tokens": sample["max_output_tokens"], "seed": sample["seed_base"] + row["global_slot_index"] * 18 + vi * repetitions + si, "stop": sample["stop"], "include_stop_str_in_output": sample["include_stop_str_in_output"]}, "model_binding": None, "runtime_binding": None, "execution_status": "NOT_EXECUTED"})
        if any(c["prompt_tokens"] + c["reserved_answer_tokens"] > policy["length"]["max_model_len"] for c in pending_checks):
            holds.append({"candidate_id": q["candidate_id"], "status": "LENGTH_HOLD", "checks": pending_checks})
            continue
        prompts.extend(pending_prompts)
        requests.extend(pending_requests)
        maps.extend(pending_maps)
        checks.extend(pending_checks)
    assert len({r["request_id"] for r in requests}) == len(requests)
    assert len({r["sampling"]["seed"] for r in requests}) == len(requests)
    dst = OUT / "requests"
    write_jsonl(dst / "PROMPTS.jsonl", prompts)
    write_jsonl(dst / "REQUESTS.jsonl", requests)
    write_json(dst / "DISPLAY_MAPS.json", {"variants": maps})
    write_json(dst / "FINITE_CHOICES.json", {"choices": choices, "gold_cardinality_used": False})
    write_json(dst / "CPU_PREFLIGHT.json", {"status": "PASS" if not holds else "LENGTH_HOLDS_RECORDED", "records": checks, "length_holds": holds, "zero_truncation": True, "answer_space_roundtrips": sum(c["answer_space_roundtrips"] for c in checks), "synthetic_roundtrip_cases": synthetic_roundtrips, "synthetic_roundtrip_all_types_arities": True, "prompt_tokens_min": min((c["prompt_tokens"] for c in checks), default=None), "prompt_tokens_max": max((c["prompt_tokens"] for c in checks), default=None), "tokenizer_files": [bind(tokenizer_dir / Path(ref["path"]).name) for ref in tokenproof["tokenizer_files"]], "declared_tokenizer_directory": str(declared_tokenizer_dir), "resolved_tokenizer_directory": str(tokenizer_dir), "tokenizer_path_relocated_by_bound_hash_only": str(tokenizer_dir) != str(declared_tokenizer_dir), "versions": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__}, "model_weights_loaded": False, "solver_imported": False, "private_gold_read": False})
    record = {"schema_version": "day3_supplement_actual_request_freeze_v2", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "CONTENT_REQUEST_BYTES_FROZEN_FUTURE_MODEL_RUNTIME_PENDING" if requests else "NO_REQUESTS_ACCEPTED", "accepted_content_question_count": sum(r["status"] == "CONTENT_ACCEPTED" for r in admission["records"]), "length_hold_question_count": len(holds), "question_count": len({r["candidate_id"] for r in requests}), "variant_count": len(prompts), "request_count_per_model": len(requests), "content_admission": bind(OUT / "CONTENT_ADMISSION.json"), "execution_freeze": execution_binding, "policy": bind(policy_path), "adapter": adapter_ref, "script": bind(__file__), "native_contract": native_contract_ref, "native_prompt_extraction": ["STRUCTURED_SYSTEM_PROMPT", "dialogue_rows", "build_prompt"], "tokenization_proof": tokenproof_ref, "evidence_bindings": [bind(dst / f) for f in ["PROMPTS.jsonl", "REQUESTS.jsonl", "DISPLAY_MAPS.json", "FINITE_CHOICES.json", "CPU_PREFLIGHT.json"]], "non_scope_sampling_parameters": sample, "index_addendum": "New prospective immutable global slot indices186..197; all preexisting indices and cohorts preserved; no score-based content selection.", "new_solver_calls": 0, "scorer_runs": 0, "new_training_runs": 0, "execution_permitted": False, "training_start_go": False, "model_runtime_status": "Future checkpoint and runtime identities must be bound at Day4 or later before execution.", "test_rule": "FINAL_FROZEN_ENDPOINT_ONLY", "historical_root_exposure": True, "unseen_root_claim": False}
    write_json(dst / "REQUEST_FREEZE.json", record)
    print(json.dumps({k: record[k] for k in ["status", "question_count", "variant_count", "request_count_per_model"]}))


if __name__ == "__main__":
    main()
