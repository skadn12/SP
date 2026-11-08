"""v5 public-only solver primitives; immutable inputs and no Gold parsing."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
WEEK = HERE.parents[2]
SOLVER = WEEK / "day2/solver"
sys.path.insert(0, str(SOLVER))
import adapter as a
_spec = importlib.util.spec_from_file_location("v5_native_runtime", SOLVER / "run.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)
require, now, read, sha, bind, save, verify_bindings = (
    rt.require, rt.now, rt.read, rt.sha, rt.bind, rt.save, rt.verify_bindings)
REPO = rt.REPO
ENDPOINT = WEEK / "day5_training_v1/evaluation/models/B1/EXECUTION_MANIFEST.json"
endpoint = read(ENDPOINT)
require(endpoint["model_role"] == "B1", "B1_ENDPOINT_REQUIRED")
MODEL = Path(endpoint["model_directory"])
TOKENIZER = Path(endpoint["tokenizer_directory"])
VENV = Path(endpoint["python_prefix"])
SOLVER_NAME = "B1"
QUESTION_COUNT = 14
TYPE_COUNTS = {"multi_select": 10, "ordering": 4}
ROOT_COUNTS = {"cc-train-episode-19708": 7, "cc-train-episode-20432": 7}
REQUEST_COUNT = 228
DISPLAY_SEED = 2026102305
SAMPLING_SEED = 2026102306

def environment():
    require(Path(sys.prefix).resolve() == VENV.resolve(), "USE_PROVEN_LOCAL4B_VENV")
    require(sys.dont_write_bytecode, "USE_PYTHON_B")

def models():
    require(endpoint["runtime_versions"] == rt.versions(), "B1_RUNTIME_VERSIONS_CHANGED")
    rows = endpoint["model_bindings"] + endpoint["tokenizer_bindings"] + endpoint["code_bindings"] + [bind(ENDPOINT)]
    verify_bindings(rows)
    return rows

def safe_id(value):
    require(isinstance(value, str) and value and value not in (".", "..") and
            all(char.isalnum() or char in "_-.:" for char in value), "UNSAFE_ARTIFACT_ID")
    return value

def below(path, directory):
    return Path(path).resolve().is_relative_to(Path(directory).resolve())

def code_paths():
    return sorted(set(rt.code_paths()) | set(HERE.glob("*.py")))

def all_bindings(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= set(value):
            yield value
        else:
            for child in value.values():
                yield from all_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_bindings(child)

def public_binding(row, run):
    require(below(row["path"], run), "NONPUBLIC_EXECUTION_INPUT:" + row["path"])
    require("private" not in Path(row["path"]).parts, "PRIVATE_EXECUTION_INPUT")
    verify_bindings([row])

def verify(run, audit=None, check_models=True):
    """Verify public execution records and bytes only; never parse Gold JSON."""
    environment()
    run = Path(run).resolve()
    frozen = read(run / "PREFLIGHT_FREEZE.json")
    require(frozen["status"] == "PREPARED_AWAITING_INDEPENDENT_PREFLIGHT", "INVALID_PREFLIGHT_STATE")
    for row in frozen["files"]:
        public_binding(row, run)
    manifest = read(run / "manifest.json")
    require(manifest["branch"] == "direct" and manifest["solver_model"] == SOLVER_NAME, "WRONG_BRANCH_OR_MODEL")
    require(manifest["model"] == str(MODEL) and manifest["tokenizer"] == str(TOKENIZER), "WRONG_MODEL_OR_TOKENIZER")
    require(manifest["automatic_retry"] is False and manifest["source_truncation"] is False, "RETRY_OR_TRUNCATION_FORBIDDEN")
    require(manifest["question_count"] == QUESTION_COUNT and manifest["question_type_counts"] == TYPE_COUNTS, "FROZEN_QUESTION_COUNT_MISMATCH")
    require(manifest["request_count"] == len(manifest["requests"]) == REQUEST_COUNT, "FROZEN_REQUEST_COUNT_MISMATCH")
    require(manifest["random_display"]["seed"] == DISPLAY_SEED and manifest["sampling_seed_start"] == SAMPLING_SEED, "FROZEN_SEEDS_CHANGED")
    require(len({x["request_id"] for x in manifest["requests"]}) == REQUEST_COUNT, "DUPLICATE_REQUEST_ID")
    require(len({x["seed"] for x in manifest["requests"]}) == REQUEST_COUNT, "DUPLICATE_SEED")
    for row in manifest["public_input_bindings"]:
        public_binding(row, run)
    verify_bindings(manifest["input_bindings"])
    verify_bindings(manifest["code_bindings"])
    if check_models:
        verify_bindings(manifest["model_bindings"])
    require(manifest["versions"] == rt.versions(), "LIBRARY_VERSION_CHANGED")
    if audit is not None:
        report = read(audit)
        require(report.get("status") == "PASS", "INDEPENDENT_PREFLIGHT_NOT_PASS")
        rows = report.get("audited_files") or report.get("files") or []
        preflight = (run / "PREFLIGHT_FREEZE.json").resolve()
        require(any(Path(x["path"]).resolve() == preflight and x["sha256"] == sha(preflight) for x in rows), "AUDIT_NOT_BOUND_TO_PREFLIGHT")
        verify_bindings(rows)
    return manifest

def freeze_raw(run, manifest):
    run = Path(run).resolve()
    files = [run / "responses" / (row["request_id"] + ".json") for row in manifest["requests"]]
    require(all(path.is_file() for path in files), "RAW_FREEZE_REQUIRES_ALL_TERMINAL_RECORDS")
    require(set((run / "responses").glob("*.json")) == set(files), "RAW_RESPONSE_SET_MISMATCH")
    save(run / "RAW_RESPONSE_FREEZE.json", {
        "status": "COMMITTED", "created_at_utc": now(),
        "manifest_binding": bind(run / "manifest.json"),
        "planned_request_count": len(files), "files": [bind(path) for path in files],
        "validation_files": [bind(path) for path in sorted((run / "response_validation").glob("*.json"))],
        "gold_opened_by_live": False, "source_only_prompts": True,
        "automatic_retry": False, "source_truncation": False})

def timestamp(value):
    from datetime import datetime
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
