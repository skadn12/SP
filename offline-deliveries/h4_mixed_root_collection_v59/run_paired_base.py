"""Replay three mixed frozen v59 evaluation chunks with M0, without changing inputs.

The native v15 evaluator is reused unchanged. Separate chunks preserve the
original seed reset, request IDs, permutations, prompt bytes, and Gold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
ORIGINAL = DELIVERIES / "h4_sequential_ab_v59_qa/runs"
REPORT = HERE
RUNNER = DELIVERIES / "h4_feedback_training_pool_v15_qa/evaluation/run_solver.py"
PYTHON = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python")
ALLOWED_MANIFEST_DIFFERENCES = {"prepared_at_utc", "model", "solver_model", "solver_identity"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def require(condition, error):
    if not condition:
        raise RuntimeError(error)


def chunks():
    import run_collection as collection
    for chunk in collection.verify_plan()["chunks"]:
        name = f"c{chunk['index']:02d}"
        source = Path(chunk["evaluation_run"]) / "solver"
        yield name, source, HERE / "base_runs" / name / "solver"


def original_bindings():
    import run_collection as collection
    plan = collection.verify_plan()
    complete = collection.completion("evaluation")
    rows = [row for chunk in complete["chunks"] for row in chunk["bindings"]]
    for row in rows:
        require(sha(row["path"]) == row["sha256"], "ORIGINAL_BINDING_CHANGED:" + row["path"])
    return {"binding_path": str(REPORT / "evaluation.COMPLETE.json"),
            "binding_sha256": sha(REPORT / "evaluation.COMPLETE.json"), "verified_files": len(rows)}


def execute(args, log_name):
    log = HERE / "logs" / log_name
    log.parent.mkdir(exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run([str(PYTHON), "-B", str(RUNNER), *args], stdout=stream,
                                stderr=subprocess.STDOUT, cwd=HERE)
    require(result.returncode == 0, "EVALUATOR_FAILED:" + str(log))


def paired_chunk(name, original, base, *, complete=False):
    if (original / "SKIPPED.json").exists():
        require(read(original / "SKIPPED.json")["candidate_count"] == 0, "BAD_EMPTY_TRAINED_POOL")
        require(read(base / "SKIPPED.json")["candidate_count"] == 0, "BAD_EMPTY_BASE_POOL")
        return {"chunk": name, "original_solver_run": str(original), "base_solver_run": str(base),
                "candidate_count": 0, "request_count": 0, "base_actual_calls": 0,
                "status": "EMPTY_POOL_NO_SOLVER", "requests_permutations_seeds_equal": True}
    trained_manifest = read(original / "manifest.json")
    base_manifest = read(base / "manifest.json")
    require(trained_manifest["solver_model"] == "M_clean_policy", "TRAINED_IDENTITY_WRONG")
    require(base_manifest["solver_model"] == "M0", "BASE_IDENTITY_WRONG")
    require(base_manifest["model"] == "/root/autodl-tmp/Qwen3-4B", "BASE_PATH_WRONG")
    keep = lambda value: {k: v for k, v in value.items() if k not in ALLOWED_MANIFEST_DIFFERENCES}
    require(keep(trained_manifest) == keep(base_manifest), "PAIRED_MANIFEST_DIFFERENCE:" + name)
    for relative, expected in trained_manifest["input_sha256"].items():
        if relative == "pool":
            require(sha(trained_manifest["pool_path"]) == expected, "ORIGINAL_POOL_CHANGED")
            continue
        require(sha(original / relative) == sha(base / relative) == expected,
                "PAIRED_INPUT_DIFFERENCE:" + name + ":" + relative)
    for request in trained_manifest["requests"]:
        relative = Path("prompts") / (request["request_id"] + ".txt")
        require(sha(original / relative) == sha(base / relative) == request["prompt_sha256"],
                "PAIRED_PROMPT_DIFFERENCE:" + request["request_id"])
    result = {"chunk": name, "original_solver_run": str(original), "base_solver_run": str(base),
              "original_manifest_sha256": sha(original / "manifest.json"),
              "base_manifest_sha256": sha(base / "manifest.json"),
              "candidate_count": base_manifest["candidate_count"], "request_count": base_manifest["request_count"],
              "requests_permutations_seeds_equal": True, "prompts_byte_identical": True,
              "sources_questions_gold_canonical_bindings_byte_identical": True,
              "decoder_contract_identical": True, "scorer_code_identical": True,
              "allowed_manifest_differences": sorted(ALLOWED_MANIFEST_DIFFERENCES)}
    if complete:
        sys.path.insert(0, str(RUNNER.parent))
        from scoring import score_response
        from selection import audit_complete_call_set
        questions = {q["candidate_id"]: q for q in read(base / "questions.json")["candidates"]}
        gold = read(base / "gold.json")
        requests = {r["request_id"]: r for r in base_manifest["requests"]}
        for directory, manifest in ((original, trained_manifest), (base, base_manifest)):
            require((directory / "COMPLETE.json").is_file(), "INCOMPLETE_SOLVER_RUN:" + str(directory))
            records = [read(p) for p in sorted((directory / "responses").glob("*.json"))]
            require(not audit_complete_call_set(manifest, records), "INVALID_CALL_SET:" + str(directory))
            require(len(records) == manifest["request_count"], "RESPONSE_COUNT_MISMATCH")
            for row in records:
                q = {**questions[row["candidate_id"]], "gold_content_ids": gold[row["candidate_id"]]}
                require(score_response(q, requests[row["request_id"]], row["raw_output"]) == row["score"],
                        "RESPONSE_SCORE_REPLAY_MISMATCH:" + row["request_id"])
                require(row["solver_model"] == manifest["solver_model"] and row["model_path"] == manifest["model"],
                        "RESPONSE_MODEL_MISMATCH")
                require(row["evaluation_valid"] is True and row["data_errors"] == [], "INVALID_RESPONSE")
                request = requests[row["request_id"]]
                require(row["seed"] == row["metadata"]["sampling_seed"] == request["seed"], "RESPONSE_SEED_MISMATCH")
                require(row["metadata"]["prompt_token_count"] == request["prompt_tokens"], "TOKEN_COUNT_MISMATCH")
                for key in ("temperature", "top_p", "max_tokens", "structured_output_api", "structured_output_version"):
                    require(row["metadata"][key] == manifest["decoder_contract"][key], "RESPONSE_DECODER_MISMATCH")
            require(read(directory / "COMPLETE.json")["actual_call_count"] == len(records), "COMPLETE_COUNT_MISMATCH")
        result.update(base_actual_calls=len(records), all_base_calls_valid=True, both_model_scores_replayed=True)
    return result


def prepare():
    require(not (HERE / "BASE_PREPARATION.STARTED.json").exists(), "PREPARATION_ALREADY_STARTED")
    write(HERE / "BASE_PREPARATION.STARTED.json", {"time": now(), "script_sha256": sha(__file__)})
    source_receipt = original_bindings()
    receipts = []
    for name, original, base in chunks():
        if (original / "SKIPPED.json").exists():
            write(base / "SKIPPED.json", {"status": "EMPTY_POOL_NO_SOLVER", "candidate_count": 0})
            receipts.append(paired_chunk(name, original, base))
            continue
        source_manifest = read(original / "manifest.json")
        execute(["preflight", "--pool", source_manifest["pool_path"], "--run", str(base),
                 "--solver-model", "M0", "--require-ordering-review"], name + "_base_preflight.log")
        receipts.append(paired_chunk(name, original, base))
        print(json.dumps({"phase": "preparation", "chunk": name, "paired": True}), flush=True)
    require(sum(r["candidate_count"] for r in receipts) <= 12, "WRONG_QUESTION_COUNT")
    planned = 0
    for _, original, _ in chunks():
        if (original / "SKIPPED.json").exists():
            continue
        questions = read(original / "questions.json")["candidates"]
        planned += sum(18 if q["question_type"] == "multi_select" else 12 for q in questions)
    require(sum(r["request_count"] for r in receipts) == planned <= 180, "WRONG_REQUEST_COUNT")
    write(HERE / "BASE_PAIRED_PREFLIGHT.json", {"time": now(), "status": "PASS", "chunks": receipts,
          "original_bindings": source_receipt, "new_base_calls_planned": sum(r["request_count"] for r in receipts), "new_trained_calls_planned": 0,
          "training_started": False, "script_sha256": sha(__file__)})


def live():
    require((HERE / "BASE_PAIRED_PREFLIGHT.json").is_file(), "PREFLIGHT_REQUIRED")
    require(not (HERE / "BASE_EVALUATION.STARTED.json").exists(), "ALREADY_STARTED_NO_AUTOMATIC_RERUN")
    preflight = read(HERE / "BASE_PAIRED_PREFLIGHT.json")
    require(preflight["script_sha256"] == sha(__file__), "SCRIPT_CHANGED_AFTER_PREFLIGHT")
    original_bindings()
    for name, original, base in chunks():
        paired_chunk(name, original, base)
    write(HERE / "BASE_EVALUATION.STARTED.json", {"time": now(), "paired_preflight_sha256": sha(HERE / "BASE_PAIRED_PREFLIGHT.json"),
          "identity_audits": "native_v15_frozen_base_and_trained_model_audits", "base_calls_planned": preflight["new_base_calls_planned"],
          "new_trained_calls_planned": 0})
    receipts = []
    try:
        for name, original, base in chunks():
            print(json.dumps({"phase": "evaluation_start", "chunk": name, "time": now()}), flush=True)
            if not (base / "SKIPPED.json").exists():
                execute(["live", "--run", str(base)], name + "_base_live.log")
                execute(["report", "--run", str(base)], name + "_base_report.log")
            receipt = paired_chunk(name, original, base, complete=True)
            receipts.append(receipt)
            write(HERE / "base_receipts" / (name + ".json"), receipt)
            print(json.dumps({"phase": "evaluation_complete", "chunk": name, "base_calls": receipt["base_actual_calls"],
                              "time": now()}), flush=True)
        originals = original_bindings()
        write(HERE / "BASE_EVALUATION.COMPLETE.json", {"time": now(), "status": "COMPLETE", "chunks": receipts,
              "base_calls": sum(r["base_actual_calls"] for r in receipts), "new_trained_calls": 0,
              "original_bindings": originals, "training_started": False, "paired_input_equality_verified": True})
    except BaseException as error:
        write(HERE / "BASE_EVALUATION.ERROR.json", {"time": now(), "error": repr(error), "traceback": traceback.format_exc(),
              "completed_chunks": [r["chunk"] for r in receipts], "automatic_rerun_allowed": False})
        raise


def verify():
    rows = [paired_chunk(name, original, base, complete=True) for name, original, base in chunks()]
    print(json.dumps({"status": "PASS", "chunks": rows, "original_bindings": original_bindings()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "live", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "live": live, "verify": verify}[args.phase]()
