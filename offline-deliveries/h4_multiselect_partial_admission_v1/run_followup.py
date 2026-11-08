"""Run exactly the 48 approved local Solver requests with original B0 settings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
PLAN = HERE / "followup_48_requests_revision_2.json"
OUT = HERE / "followup_k3_revision_2"
BASE_RUNNER = HERE.parent / "h4_event_ledger_m0_pilot_v1" / "run.py"
IDENTITY = HERE / "followup_model_identity.json"
EXPECTED_PYTHON = Path("/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_probe():
    spec = importlib.util.spec_from_file_location("approved_followup_original_probe", BASE_RUNNER)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    return probe


def prepare():
    if Path(sys.prefix) != EXPECTED_PYTHON.parent.parent:
        raise RuntimeError("use_original_venv_local4b_python:" + str(EXPECTED_PYTHON))
    plan = read(PLAN)
    baseline_path = Path(plan["baseline_manifest"]["path"])
    if sha(baseline_path) != plan["baseline_manifest"]["sha256"]:
        raise RuntimeError("baseline_manifest_drift")
    baseline = read(baseline_path)
    for name, expected in plan["source_input_sha256"].items():
        if sha(baseline_path.parent / name) != expected:
            raise RuntimeError("frozen_input_drift:" + name)
    identity = read(IDENTITY)
    if identity.get("can_merge_existing_slot0_with_new_slots_under_recorded_model_identity") is not True:
        raise RuntimeError("baseline_model_identity_not_established")
    # Check stats recorded while hashing weights. The identity audit does not
    # pretend that the old run already contained hashes it never recorded.
    for row in identity["files"]:
        p = Path(row["path"])
        s = p.stat()
        if (s.st_size, s.st_mtime_ns, s.st_ctime_ns) != (row["bytes"], row["mtime_ns"], row["ctime_ns"]):
            raise RuntimeError("model_file_changed_since_identity_audit:" + str(p))
    probe = load_probe()
    probe.RUN = baseline_path.parent
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True)
    candidates, dialogue = probe.inputs()
    by_id = {row["candidate_id"]: row for row in candidates}
    baseline_requests = {row["request_id"]: row for row in baseline["requests"]}
    prompts = {}
    seen = set()
    for row in plan["requests"]:
        old = baseline_requests[row["baseline_request_id"]]
        for key in ("candidate_id", "question_type", "variant", "mode", "prompt_sha256", "prompt_tokens"):
            if row[key] != old[key]:
                raise RuntimeError("request_drift:" + key)
        if row["slot"] not in (1, 2) or row["request_id"] in seen:
            raise RuntimeError("invalid_slot_or_duplicate_request")
        seen.add(row["request_id"])
        prompt = probe.contract.build_prompt(tokenizer, by_id[row["candidate_id"]], row["variant"],
                                             dialogue, context_mode=row["mode"])
        if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
            raise RuntimeError("prompt_drift:" + row["request_id"])
        if len(tokenizer.encode(prompt)) != row["prompt_tokens"]:
            raise RuntimeError("prompt_token_count_drift:" + row["request_id"])
        prompts[row["request_id"]] = prompt
    if len(prompts) != 48:
        raise RuntimeError("approved_request_count_mismatch")
    config = probe.solver.generation_config("multi_select")
    for key, value in plan["decoding"].items():
        if config[key] != value:
            raise RuntimeError("decoding_drift:" + key)
    for row in plan["requests"]:
        baseline_response = read(baseline_path.parent / "responses" / (row["baseline_request_id"] + ".json"))
        for key in ("temperature", "top_p", "max_tokens", "structured_output_api", "structured_output_version"):
            if config[key] != baseline_response["metadata"][key]:
                raise RuntimeError("baseline_generation_config_drift:" + key)
    if OUT.exists():
        raise RuntimeError("followup_output_exists")
    OUT.mkdir()
    for name in plan["source_input_sha256"]:
        shutil.copy2(baseline_path.parent / name, OUT / name)
    write_once(OUT / "prompts.json", prompts)
    code_paths = [BASE_RUNNER, Path(probe.contract.__file__), Path(probe.solver.__file__),
                  Path(probe.solver.VLLMSolverBackend.__module__.replace(".", "/") + ".py")]
    code_paths[-1] = probe.WORKTREE / code_paths[-1]
    write_once(OUT / "execution_manifest.json", {
        "status": "PREFLIGHT_PASS", "request_count": 48,
        "authorization": "User confirmed the prepared 48-request Solver follow-up; no training authorized.",
        "plan_sha256": sha(PLAN), "baseline_manifest_sha256": sha(baseline_path),
        "baseline_merge_authorized": True,
        "model_identity_evidence": {"path": str(IDENTITY), "sha256": sha(IDENTITY)},
        "model_identity_scope": identity.get("evidence_strength"),
        "model": plan["model"], "generation_config": config,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers")},
        "python_executable": sys.executable, "python_prefix": sys.prefix,
        "runtime_provenance": {
            "baseline_command": str(EXPECTED_PYTHON) + " " + str(HERE.parent / "h4_259192_manual_qa_v2/run_probe.py") + " live",
            "history_timestamp_utc": "2026-10-04T10:43:09.173Z",
            "history_file": "/root/.codex/sessions/2026/10/04/rollout-2026-10-04T18-29-20-01a0de88-7213-7543-ade5-dd5020c4cbbe_01a10676-0752-7f71-bd96-34abb0ce6d8c.jsonl",
            "baseline_structured_output_api_verified": True,
        },
        "code_sha256": {str(p): sha(p) for p in code_paths},
        "prompt_count": len(prompts), "prompts_sha256": sha(OUT / "prompts.json"),
        "input_sha256": plan["source_input_sha256"], "requests": plan["requests"],
        "training_promotion": False,
    })
    print("PREFLIGHT_PASS: 48 matching prompts; frozen inputs and model identity verified", flush=True)


def live():
    manifest = read(OUT / "execution_manifest.json")
    if sys.prefix != manifest["python_prefix"] or Path(sys.prefix) != EXPECTED_PYTHON.parent.parent:
        raise RuntimeError("runtime_changed_since_preflight")
    for name, version in manifest["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package_changed_since_preflight:" + name)
    if sha(PLAN) != manifest["plan_sha256"] or sha(IDENTITY) != manifest["model_identity_evidence"]["sha256"]:
        raise RuntimeError("execution_evidence_drift")
    for name, expected in manifest["input_sha256"].items():
        if sha(OUT / name) != expected:
            raise RuntimeError("input_changed_since_preflight")
    for path, expected in manifest["code_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("solver_code_changed_since_preflight")
    identity = read(IDENTITY)
    for row in identity["files"]:
        st = Path(row["path"]).stat()
        if (st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (row["bytes"], row["mtime_ns"], row["ctime_ns"]):
            raise RuntimeError("model_changed_since_preflight")
    if sha(OUT / "prompts.json") != manifest["prompts_sha256"]:
        raise RuntimeError("prompts_changed_since_preflight")
    prompts = read(OUT / "prompts.json")
    write_once(OUT / "STARTED.json", {"started_at": time.time(), "request_count": 48})
    probe = load_probe()
    probe.RUN = OUT
    candidates, _ = probe.inputs()
    by_id = {q["candidate_id"]: q for q in candidates}
    from scripts.current_method.restart_n5_spell.backends import SolverRequest
    from scoring import score_response
    backend = probe.solver.create_backend(model_path=Path(manifest["model"]), tokenizer_path=Path(manifest["model"]))
    try:
        for index, row in enumerate(manifest["requests"], 1):
            req = SolverRequest(request_id=row["request_id"], qa_id=row["candidate_id"],
                                evaluation_mode=row["mode"], order_id=row["variant"]["variant_id"],
                                target_slot=row["slot"], retry_index=0, sampling_seed=row["seed"],
                                options=row["variant"]["options"], canonical_answer_text="",
                                prompt_sha256=row["prompt_sha256"], prompt=prompts[row["request_id"]])
            response = probe.solver._structured_generate(backend, req, question_type="multi_select")
            record = {"request_id": row["request_id"], "mode": row["mode"],
                      "variant_id": row["variant"]["variant_id"], "seed": row["seed"],
                      "prompt_sha256": row["prompt_sha256"], "raw_output": response.raw_text,
                      "model_identifier": response.model_identifier, "metadata": response.metadata,
                      "displayed_gold": row["variant"]["displayed_gold"],
                      "score": probe.contract.score_output(response.raw_text, by_id[row["candidate_id"]], row["variant"])}
            record["partial_score_response"] = score_response(by_id[row["candidate_id"]], row, record)
            write_once(OUT / "responses" / (row["request_id"] + ".json"), record)
            if response.metadata.get("finish_reason") not in ("stop", "eos"):
                raise RuntimeError("incomplete_response:" + row["request_id"])
            print(json.dumps({"completed": index, "total": 48, "request_id": row["request_id"],
                              "exact": record["score"]["correct"],
                              "J": record["partial_score_response"]["partial_score"]}), flush=True)
    finally:
        backend.close()
    for row in identity["files"]:
        st = Path(row["path"]).stat()
        if (st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (row["bytes"], row["mtime_ns"], row["ctime_ns"]):
            raise RuntimeError("model_changed_during_run")
    write_once(OUT / "COMPLETE.json", {"completed_at": time.time(), "response_count": 48,
                                       "training_promotion": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "live"))
    args = parser.parse_args()
    {"preflight": prepare, "live": live}[args.phase]()
