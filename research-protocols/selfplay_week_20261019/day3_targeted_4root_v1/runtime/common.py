"""Frozen-input verification for the 24-slot, four-root feasibility probe."""
from __future__ import annotations
from pathlib import Path
import collections
import importlib.util
import sys
import adapter as a

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parents[1]
SOLVER = WEEK / "day2/solver"
spec = importlib.util.spec_from_file_location("crossroot_legacy_runtime", SOLVER / "run.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
require, now, read, sha, bind, save, verify_bindings = rt.require, rt.now, rt.read, rt.sha, rt.bind, rt.save, rt.verify_bindings
MODEL, TOKENIZER, VENV, REPO = rt.MODEL, rt.TOKENIZER, rt.VENV, rt.REPO
PRODUCTION_PLANNED_SLOTS = 24
PRODUCTION_ROOTS = 4
SOLVER_FIXED = {"name": "S0_M_clean", "model": str(MODEL), "tokenizer": str(TOKENIZER),
    "MC_mappings": 6, "MC_samples_each_mapping": 3, "ordering_mappings": 12, "ordering_samples_each_mapping": 1,
    "temperature": .8, "top_p": .9, "max_tokens": 32, "enable_thinking": False, "max_model_len": 16384,
    "source_truncation": False, "automatic_retry": False}


def environment():
    require(Path(sys.prefix).resolve() == VENV.resolve(), "USE_PROVEN_LOCAL4B_VENV")
    require(sys.dont_write_bytecode, "USE_PYTHON_B")


def code_paths():
    return sorted(set(rt.code_paths()) | set(HERE.glob("*.py")))


def safe_id(value):
    require(isinstance(value, str) and value and value not in (".", "..") and all(char.isalnum() or char in "_-.:" for char in value), "UNSAFE_ARTIFACT_ID")
    return value


def below(path, directory):
    return Path(path).resolve().is_relative_to(Path(directory).resolve())


def public_binding(row, run):
    require(below(row["path"], run), "NONPUBLIC_EXECUTION_INPUT:" + row["path"])
    require("private" not in Path(row["path"]).parts, "PRIVATE_EXECUTION_INPUT")
    verify_bindings([row])


def verify(run, audit=None, check_models=True):
    environment()
    run = Path(run).resolve()
    frozen = read(run / "PREFLIGHT_FREEZE.json")
    require(frozen["status"] == "PREPARED_AWAITING_INDEPENDENT_PREFLIGHT", "INVALID_PREFLIGHT_STATE")
    for row in frozen["files"]:
        public_binding(row, run)
    manifest = read(run / "manifest.json")
    policy = read(run / "PROTOCOL.json")
    budget, solver = policy["budget"], policy["solver"]
    require(all(solver[key] == value for key, value in SOLVER_FIXED.items()), "FROZEN_SOLVER_CONTRACT_MISMATCH")
    require(type(solver["display_seed"]) is int and type(solver["sampling_seed"]) is int, "INTEGER_SEEDS_REQUIRED")
    require(manifest["branch"] == "direct" and manifest["solver_model"] == "S0_M_clean", "WRONG_BRANCH_OR_MODEL")
    require(manifest["model"] == str(MODEL) and manifest["tokenizer"] == str(TOKENIZER), "WRONG_MODEL_OR_TOKENIZER")
    require(manifest["automatic_retry"] is False and manifest["source_truncation"] is False, "NO_RETRY_OR_TRUNCATION")
    require(manifest["batch_size"] == 8 and manifest["max_model_len"] == 16384 and manifest["prefix_caching"] is False and manifest["gpu_memory_utilization"] == .8, "BACKEND_SETTINGS_DRIFT")
    require(manifest["production_denominator"] == budget["production_planned_slots"] == 24, "INVALID_PRODUCTION_DENOMINATOR")
    require(manifest["question_count"] == manifest["planned_question_slots"] == budget["initial_question_slots"], "ADMITTED_COUNT_DRIFT")
    rows = manifest["requests"]
    require(len(rows) == manifest["request_count"] == budget["max_direct_requests"], "REQUEST_COUNT_DRIFT")
    require(len({row["request_id"] for row in rows}) == len(rows) and len({row["seed"] for row in rows}) == len(rows), "DUPLICATE_REQUEST_OR_SEED")
    questions = read(run / "PUBLIC_QUESTIONS.json")["questions"]
    require({row["candidate_id"] for row in rows} == {q["candidate_id"] for q in questions}, "REQUEST_QUESTION_SET_DRIFT")
    require(len(questions) == len({q["candidate_id"] for q in questions}) == manifest["question_count"], "PUBLIC_COUNT_DRIFT")
    slots = {slot["candidate_id"]: slot for slot in policy["slots"]}
    require(len(slots) == len(policy["slots"]) == 24 and len({slot["slot_id"] for slot in slots.values()}) == 24 and len({slot["root_id"] for slot in slots.values()}) == 4, "PRODUCTION_SLOT_ROOT_DRIFT")
    require({q["candidate_id"] for q in questions} <= set(slots), "UNPLANNED_PUBLIC_QUESTION")
    require(sum(a.request_count(q) for q in questions) == len(rows), "ARITY_REQUEST_COUNT_DRIFT")
    missing = sorted(set(slots) - {q["candidate_id"] for q in questions})
    require(manifest["missing_candidate_ids"] == missing and manifest["production_unadmitted_slots"] == len(missing) and manifest["production_admitted_numerator"] == len(questions), "PRODUCTION_ADMISSION_DRIFT")
    require(manifest["planned_root_count"] == 4 and manifest["root_count"] == len({q["root_id"] for q in questions}), "ROOT_DENOMINATOR_DRIFT")
    admission = read(run / "CONTENT_ADMISSION.json")
    require(admission["status"] == "PASS" and set(admission["candidate_ids"]) == {q["candidate_id"] for q in questions} and admission["planned_slots"] == 24 and admission["admitted_count"] == len(questions), "PUBLIC_ADMISSION_DRIFT")
    require(policy["namespace"] == manifest["namespace"] == admission["namespace"] and isinstance(admission["namespace"], str) and admission["namespace"].startswith("train.feedback"), "NAMESPACE_DRIFT")
    # Independently reconstruct the entire public plan. No Gold file is opened.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    _, native_solver = a.native()
    require(manifest["decoder_contract"] == native_solver.generation_config("multi_select"), "NATIVE_DECODER_CHANGED")
    maps, token_ids = read(run / "DISPLAY_MAPPINGS.json"), read(run / "PROMPT_TOKEN_IDS.json")
    require(set(maps["requests"]) == set(token_ids) == {row["request_id"] for row in rows}, "PUBLIC_REQUEST_ARTIFACT_SET_DRIFT")
    require(set(maps["public_to_canonical"]) == {q["candidate_id"] for q in questions}, "CANONICAL_QUESTION_SET_DRIFT")
    expected_rows = []
    for q in questions:
        a.public_question(q)
        qid, root = safe_id(q["candidate_id"]), safe_id(q["root_id"])
        source = rt.source_check(dict(q, source_path=str(run / "sources" / (root + ".json"))))
        subset = [row for row in rows if row["candidate_id"] == q["candidate_id"]]
        field = "options" if q["question_type"] == "multi_select" else "items"
        require(slots[qid]["root_id"] == root and slots[qid]["question_type"] == q["question_type"] and slots[qid]["item_count"] == len(q[field]), "SLOT_PUBLIC_IDENTITY_DRIFT")
        require(maps["public_to_canonical"][qid] == a.content_ids(q), "CANONICAL_CONTENT_MAP_DRIFT")
        count = a.mapping_count(q)
        expected_samples = 3 if field == "options" else 1
        require(len(subset) == count * expected_samples, "QUESTION_REQUEST_DENOMINATOR_DRIFT")
        by_variant = collections.defaultdict(list)
        for row in subset:
            require(row["question_type"] == q["question_type"] and row["root_id"] == q["root_id"] and row["branch"] == "direct" and row["max_output_tokens"] == 32, "ROW_IDENTITY_OR_BUDGET_DRIFT")
            require(tuple(row["display"][field]) == a.labels(q["question_type"], len(q[field])), "ROW_LABEL_DRIFT")
            by_variant[row["variant_id"]].append(row)
        require(len(by_variant) == count and all(len(group) == expected_samples for group in by_variant.values()), "MAPPING_REPEAT_DRIFT")
        for variant in a.variants(q, solver["display_seed"], count):
            prompt = a.direct_prompt(tokenizer, q, variant, source)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            require(len(ids) + 32 <= 16384, "FULL_CONTEXT_OVERFLOW_NO_TRUNCATION")
            for sample in range(expected_samples):
                rid = f"{qid}__{variant['variant_id']}__direct__s{sample:02d}"
                require(maps["requests"].get(rid) == variant, "GOLD_BLIND_DISPLAY_RECONSTRUCTION_DRIFT")
                require(token_ids.get(rid) == ids, "PROMPT_TOKEN_IDS_DRIFT")
                require((run / "prompts" / (rid + ".txt")).read_text() == prompt, "PUBLIC_PROMPT_RECONSTRUCTION_DRIFT")
                expected_rows.append({"request_id": rid, "candidate_id": qid, "root_id": root,
                    "question_type": q["question_type"], "branch": "direct", "variant_id": variant["variant_id"],
                    "display": {field: variant[field]}, "slot": sample, "seed": solver["sampling_seed"] + len(expected_rows),
                    "prompt_sha256": rt.textsha(prompt), "prompt_tokens": len(ids), "source_turns": len(source["turns"]),
                    "source_sha256": q["source_sha256"], "max_output_tokens": 32})
    require(rows == expected_rows, "DETERMINISTIC_REQUEST_SCHEDULE_DRIFT")
    require(manifest["random_display"]["seed"] == solver["display_seed"] and manifest["sampling_seed_start"] == solver["sampling_seed"], "SEED_DRIFT")
    for row in manifest["public_input_bindings"]:
        public_binding(row, run)
    verify_bindings(manifest["code_bindings"])
    if check_models:
        verify_bindings(manifest["model_bindings"])
    require(manifest["versions"] == rt.versions(), "LIBRARY_VERSION_CHANGED")
    if audit is not None:
        report = read(audit)
        require(report.get("status") == "PASS", "INDEPENDENT_PREFLIGHT_NOT_PASS")
        bindings = report.get("audited_files") or report.get("files") or []
        preflight = (run / "PREFLIGHT_FREEZE.json").resolve()
        require(any(Path(row["path"]).resolve() == preflight and row["sha256"] == sha(preflight) for row in bindings), "AUDIT_NOT_BOUND_TO_THIS_PREFLIGHT")
        for row in bindings:
            public_binding(row, run)
    return manifest


def freeze_raw(run, manifest):
    run = Path(run).resolve()
    files = [run / "responses" / (row["request_id"] + ".json") for row in manifest["requests"]]
    require(all(path.is_file() for path in files) and set((run / "responses").glob("*.json")) == set(files), "RAW_FREEZE_REQUIRES_EXACT_TERMINAL_SET")
    save(run / "RAW_RESPONSE_FREEZE.json", {"status": "COMMITTED", "created_at_utc": now(), "manifest_binding": bind(run / "manifest.json"),
        "planned_request_count": len(files), "files": [bind(path) for path in files], "gold_opened_by_live": False,
        "source_only_prompts": True, "automatic_retry": False, "source_truncation": False})


def timestamp(value):
    from datetime import datetime
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
