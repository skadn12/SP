"""Public-only primitives; reuse native prompts, decoder, model and tokenizer."""
from __future__ import annotations
from pathlib import Path
import importlib.util
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parents[1]
SOLVER = WEEK / "day2/solver"
sys.path.insert(0, str(SOLVER))
import adapter as a
spec = importlib.util.spec_from_file_location("day3_solver187_legacy_runtime", SOLVER / "run.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
require, now, read, sha, bind, save, verify_bindings = (
    rt.require, rt.now, rt.read, rt.sha, rt.bind, rt.save, rt.verify_bindings)
MODEL, TOKENIZER, VENV, REPO = rt.MODEL, rt.TOKENIZER, rt.VENV, rt.REPO

DIAGNOSTIC_QUESTIONS = 187
TYPE_COUNTS = {"multi_select": 74, "ordering": 113}
DIRECT_REQUESTS = 2688
PRODUCTION_PLANNED_SLOTS = 240
DISPLAY_SEED = 2026101903
SAMPLING_SEED = 2026101904


def environment():
    require(Path(sys.prefix).resolve() == VENV.resolve(), "USE_PROVEN_LOCAL4B_VENV")
    require(sys.dont_write_bytecode, "USE_PYTHON_B")


def code_paths():
    return sorted(set(rt.code_paths()) | set(HERE.glob("*.py")))


def safe_id(value):
    require(isinstance(value, str) and value and value not in (".", "..") and
            all(char.isalnum() or char in "_-.:" for char in value), "UNSAFE_ARTIFACT_ID")
    return value


def below(path, directory):
    return Path(path).resolve().is_relative_to(Path(directory).resolve())


def public_binding(row, run):
    require(below(row["path"], run), "NONPUBLIC_EXECUTION_INPUT:" + row["path"])
    require("private" not in Path(row["path"]).parts, "PRIVATE_EXECUTION_INPUT")
    verify_bindings([row])


def verify(run, audit=None, check_models=True):
    """Read only public execution inputs, implementation and model files."""
    environment()
    run = Path(run).resolve()
    frozen = read(run / "PREFLIGHT_FREEZE.json")
    require(frozen["status"] == "PREPARED_AWAITING_INDEPENDENT_PREFLIGHT", "INVALID_PREFLIGHT_STATE")
    for row in frozen["files"]:
        public_binding(row, run)
    manifest = read(run / "manifest.json")
    require(manifest["branch"] == "direct" and manifest["solver_model"] == "S0_M_clean", "WRONG_BRANCH_OR_MODEL")
    require(manifest["model"] == str(MODEL) and manifest["tokenizer"] == str(TOKENIZER), "WRONG_MODEL_OR_TOKENIZER_PATH")
    require(manifest["automatic_retry"] is False and manifest["source_truncation"] is False, "RETRY_OR_TRUNCATION_FORBIDDEN")
    require(manifest["request_count"] == len(manifest["requests"]) == DIRECT_REQUESTS, "REQUEST_CAP_EXCEEDED")
    require(manifest["planned_question_slots"] == DIAGNOSTIC_QUESTIONS and manifest["question_count"] == DIAGNOSTIC_QUESTIONS, "INVALID_PLANNED_OR_ADMITTED_COUNT")
    require(manifest["question_type_counts"] == TYPE_COUNTS, "INVALID_TYPE_COUNTS")
    require(manifest["production_denominator"] == PRODUCTION_PLANNED_SLOTS and
            manifest["production_admitted_numerator"] == DIAGNOSTIC_QUESTIONS, "INVALID_PRODUCTION_COUNTS")
    require(manifest["random_display"]["seed"] == DISPLAY_SEED and
            manifest["sampling_seed_start"] == SAMPLING_SEED, "FROZEN_SEEDS_CHANGED")
    require(len({row["request_id"] for row in manifest["requests"]}) == DIRECT_REQUESTS, "DUPLICATE_REQUEST_ID")
    require(len({row["seed"] for row in manifest["requests"]}) == DIRECT_REQUESTS, "DUPLICATE_SAMPLING_SEED")
    for row in manifest["public_input_bindings"]:
        public_binding(row, run)
    verify_bindings(manifest["code_bindings"])
    if check_models:
        verify_bindings(manifest["model_bindings"])
    require(manifest["versions"] == rt.versions(), "LIBRARY_VERSION_CHANGED")
    if audit is not None:
        report = read(audit)
        require(report.get("status") == "PASS", "INDEPENDENT_PREFLIGHT_NOT_PASS")
        rows = report.get("audited_files") or report.get("files") or []
        preflight = (run / "PREFLIGHT_FREEZE.json").resolve()
        require(any(Path(row["path"]).resolve() == preflight and row["sha256"] == sha(preflight) for row in rows), "AUDIT_NOT_BOUND_TO_THIS_PREFLIGHT")
        for row in rows:
            public_binding(row, run)
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
        "gold_opened_by_live": False, "source_only_prompts": True,
        "automatic_retry": False, "source_truncation": False})


def timestamp(value):
    from datetime import datetime
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
