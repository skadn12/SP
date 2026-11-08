#!/usr/bin/env python3
"""Read-only CPU precheck of the exact locked Solver prompt/token construction.

This never calls prepare/run/analyze, never constructs a model, never writes
execution artifacts, and never excludes or edits a semantic candidate.  Run it
after CONTENT_FREEZE.json, before the write-once Solver prepare phase.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
RUNNER_PATH = BASE / "solver_runner.py"
OUTPUT = BASE / "CPU_LENGTH_PRECHECK.json"


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or bind(value.get("path", "")) != {
            key: value.get(key) for key in ("path", "sha256", "bytes")}:
        raise ValueError(f"Immutable binding drift: {value}")


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def checked_input_lock() -> dict[str, Any]:
    lock = read(BASE / "INPUT_LOCK.json")
    rows = lock.get("bindings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("INPUT_LOCK has no bindings")
    for row in rows:
        verify(row)
    if bind(RUNNER_PATH) not in rows:
        raise ValueError("Solver runner is not bound by INPUT_LOCK")
    return lock


def checked_freezes() -> dict[str, Any]:
    checked_input_lock()
    content_path = BASE / "CONTENT_FREEZE.json"
    content = read(content_path)
    if content.get("status") != "FROZEN_BEFORE_SOLVER":
        raise ValueError("Content must be FROZEN_BEFORE_SOLVER")
    content_rows = content.get("bindings")
    if not isinstance(content_rows, list):
        raise ValueError("Content freeze lacks bindings")
    for row in content_rows:
        verify(row)
    for name in ("CANDIDATES_PRIVATE.json", "CONTENT_REVIEW_AI.json", "SOURCE_SELECTION_FINAL.json",
                 "SOURCE_FREEZE.json", "CONFIG.json", "PROTOCOL_ZH.md",
                 "question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md"):
        if bind(BASE / name) not in content_rows:
            raise ValueError(f"Required content input is unbound: {name}")
    analysis_path = BASE / "ANALYSIS_DRIVER_FREEZE.json"
    analysis = read(analysis_path)
    if analysis.get("status") != "FROZEN_BEFORE_SOLVER":
        raise ValueError("Analysis implementation must be FROZEN_BEFORE_SOLVER")
    for field, path in (("script_binding", BASE / "analysis_replication.py"),
                        ("analysis_plan_binding", BASE / "ANALYSIS_PLAN.json")):
        verify(analysis.get(field))
        if analysis[field] != bind(path):
            raise ValueError(f"Analysis {field} path/binding drift")
    if analysis.get("self_check", {}).get("status") != "PASS":
        raise ValueError("Frozen analysis self-check is not PASS")
    initial_ready = analysis.get("initial_ready_receipt")
    if isinstance(initial_ready, dict):
        verify(initial_ready)
    return {"content": content, "analysis": analysis,
            "content_binding": bind(content_path), "analysis_binding": bind(analysis_path)}


def extract_builder() -> tuple[ast.Module, dict[str, Any]]:
    """Compile only the actual runner's in-memory row/prompt/token statements.

    The context-limit assertion is evaluated by the caller for every request so
    an overflow produces a complete receipt rather than stopping the first row.
    All file creation statements are deliberately outside the extracted AST.
    """
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER_PATH))
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    loops = [node for node in ast.walk(prepare) if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
             and node.iter.func.id == "display_variants"]
    if len(loops) != 1:
        raise ValueError("Frozen display loop schema drift")
    loop = loops[0]
    body = []
    found_ids = False
    for node in loop.body:
        body.append(copy.deepcopy(node))
        if isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id == "ids" for item in node.targets):
            found_ids = True
            break
    if not found_ids or not isinstance(body[0], ast.Assign):
        raise ValueError("Frozen prompt/token block is missing")
    # Reuse the exact schema, future output-path, row metadata, and public-key
    # checks too, but omit every mkdir/open/save/write statement.
    for node in loop.body[len(body):]:
        if isinstance(node, ast.Assign):
            names = {item.id for item in ast.walk(node.targets[0]) if isinstance(item, ast.Name)}
            if names in ({"schema"}, {"pp", "sp"}):
                body.append(copy.deepcopy(node))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "row" and call.func.attr == "update"
                    and any(keyword.arg == "prompt_path" for keyword in call.keywords)):
                body.append(copy.deepcopy(node))
            elif isinstance(call.func, ast.Name) and call.func.id == "need" and any(
                    isinstance(item, ast.Constant) and item.value == "PUBLIC_FIELD_ALLOWLIST_DRIFT"
                    for item in ast.walk(call)):
                body.append(copy.deepcopy(node))
                break
    if len(body) != 18:
        raise ValueError(f"Unexpected frozen in-memory prompt block size: {len(body)}")
    forbidden = {"open", "save", "write", "write_text", "write_bytes", "mkdir", "prepare", "run", "analyze", "LLM"}
    for node in body:
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                name = item.func.id if isinstance(item.func, ast.Name) else item.func.attr if isinstance(item.func, ast.Attribute) else ""
                if name in forbidden:
                    raise ValueError(f"Non-CPU/non-readonly call in extracted block: {name}")
    names = ("tok", "candidate", "source", "cid", "field", "base_seed", "index", "public", "presentation", "refs")
    function = ast.FunctionDef(
        name="_frozen_prompt_builder",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in names], vararg=None,
                           kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=body + [ast.Return(value=ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Load())
                                                     for name in ("rid", "row", "prompt", "ids", "messages", "schema")], ctx=ast.Load()))],
        decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    receipt = {"implementation": "AST-extracted exact frozen prepare in-memory row/prompt/token/schema statements",
               "runner_binding": bind(RUNNER_PATH), "statement_count": len(body),
               "source_line_ranges": [[node.lineno, node.end_lineno] for node in body],
               "statements_ast_sha256": hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[]),
                                                                include_attributes=False).encode()).hexdigest(),
               "display_logic": "direct frozen runner.display_variants(candidate)",
               "writes_excluded": True, "context_limit_checked_for_every_request": True}
    compile(module, str(RUNNER_PATH) + "::<readonly_cpu_prompt_block>", "exec")
    return module, receipt


def load_runner(wrapper_path: Path | None):
    path = wrapper_path or RUNNER_PATH
    path = path.resolve()
    path.relative_to(BASE.resolve())
    spec = importlib.util.spec_from_file_location("fgr20_cpu_precheck_runtime", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import the requested runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if wrapper_path is not None:
        runner = module.get_runner()
    else:
        runner = module
    if Path(runner.__file__).resolve() != RUNNER_PATH.resolve():
        raise ValueError("Runtime does not expose the exact frozen Solver module")
    return runner, module if wrapper_path is not None else None


def execution_snapshot() -> list[dict[str, Any]]:
    return [bind(path) for path in sorted((BASE / "execution").rglob("*")) if path.is_file()]


def run_check(wrapper_path: Path | None) -> dict[str, Any]:
    if (BASE / "STOP_REQUEST").exists():
        raise RuntimeError("STOP_REQUEST is present")
    freezes = checked_freezes()
    before_execution = execution_snapshot()
    runner, wrapper = load_runner(wrapper_path)
    endpoint, config_receipt = runner.orig.endpoint_receipt()
    api_receipt = runner.v2.cpu_api_check()
    from transformers import AutoTokenizer
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU precheck must not initialize CUDA")
    tok = AutoTokenizer.from_pretrained(str(runner.orig.TOKENIZER), local_files_only=True)
    reasoner = Qwen3ReasoningParser(tokenizer=tok)
    builder_ast, builder_receipt = extract_builder()
    namespace = dict(vars(runner))
    exec(compile(builder_ast, str(RUNNER_PATH) + "::<readonly_cpu_prompt_block>", "exec"), namespace)
    builder = namespace["_frozen_prompt_builder"]
    design = runner.frozen_design(read(BASE / "CONFIG.json"))
    candidates = read(BASE / "CANDIDATES_PRIVATE.json")
    review_rows = read(BASE / "CONTENT_REVIEW_AI.json")["reviews"]
    reviews = {row["candidate_id"]: row for row in review_rows}
    if not isinstance(candidates, list) or not 0 < len(candidates) <= design["planned_semantic_questions"]:
        raise ValueError("Frozen runner candidate-count precondition fails")
    if (len(candidates) != len({row["candidate_id"] for row in candidates})
            or len(candidates) != len(reviews) or len(reviews) != len(review_rows)
            or set(reviews) != {row["candidate_id"] for row in candidates}):
        raise ValueError("Frozen candidate/review set precondition fails")
    selection = read(BASE / "SOURCE_SELECTION_FINAL.json")
    roots = {row["root_id"]: row for row in selection["roots"]}
    if (not 0 < len(roots) == len(selection["roots"]) <= len(design["root_ids"])
            or not set(roots) <= set(design["root_ids"])
            or not all(row.get("split", selection.get("split")) == "train" for row in roots.values())):
        raise ValueError("Frozen selected-Train-root precondition fails")
    source_bindings = [{key: row[key] for key in ("path", "sha256", "bytes")} for row in selection["roots"]]
    for row in source_bindings:
        verify(row)
    candidate_records, request_records, errors = [], [], []
    seen_slots: set[tuple[str, str, str]] = set()
    for candidate in sorted(candidates, key=lambda row: (row["root_id"], row["slot_id"], row["arm"])):
        cid = candidate["candidate_id"]
        rv = reviews[cid]
        record = {"candidate_id": cid, "candidate_sha256": runner.orig.fingerprint(candidate),
                  "review_record_sha256": digest(rv), "source_binding": candidate.get("source_binding"),
                  "root_id": candidate.get("root_id"), "arm": candidate.get("arm"),
                  "slot_id": candidate.get("slot_id"), "review_decision": rv.get("decision"),
                  "included_by_frozen_runner": False, "request_ids": [], "errors": []}
        candidate_records.append(record)
        try:
            arm, slot, root = candidate["arm"], candidate["slot_id"], candidate["root_id"]
            runner.need(arm in runner.ARMS and slot in runner.SLOTS and root in roots, "UNPLANNED_ARM_ROOT_OR_SLOT:" + cid)
            key = (root, slot, arm)
            runner.need(key not in seen_slots, "DUPLICATE_PLANNED_SLOT:" + cid)
            seen_slots.add(key)
            kind, operation, count = runner.SLOTS[slot]
            runner.need(candidate["question_type"] == kind and candidate["task_operation"] == operation, "SLOT_OPERATION_DRIFT:" + cid)
            field = "options" if kind == "multi_select" else "items"
            runner.need(set(candidate[field]) == set("ABCDEF") and (count is None or len(candidate["gold"]) == count), "SLOT_ITEM_OR_GOLD_COUNT_DRIFT:" + cid)
            selected_binding = {key: roots[root][key] for key in ("path", "sha256", "bytes")}
            runner.need(candidate["source_binding"] == selected_binding, "SOURCE_SELECTION_DRIFT:" + cid)
            runner.need(rv["decision"] in ("PASS", "HOLD", "FAIL"), "UNKNOWN_REVIEW_DECISION:" + cid)
            runner.need(rv["candidate_sha256"] == runner.orig.fingerprint(candidate), "REVIEW_HASH_DRIFT:" + cid)
            runner.need(candidate["author_id"] and rv["author_id"] == candidate["author_id"] != rv["reviewer_id"], "AUTHOR_REVIEW_CONFLICT")
            if rv["decision"] != "PASS":
                record["disposition"] = "NOT_REQUESTED_BY_FROZEN_RUNNER_NONPASS_CONTENT"
                continue
            runner.orig.gate._validate(candidate)
            validated = runner.orig.production_gate.validate_review(candidate, rv, BASE / "SOURCE_SELECTION_FINAL.json")
            runner.need(validated["admitted"], "CONTENT_GATE:" + cid + ":" + json.dumps(validated))
            source = read(candidate["source_binding"]["path"])
            verify(candidate["source_binding"])
            runner.need(source["root_id"] == root, "SOURCE_ROOT_DRIFT")
            refs = [turn["turn_ref"] for turn in source["turns"]]
            runner.need(len(refs) == len(set(refs)), "DUPLICATE_SOURCE_REFS")
            record["included_by_frozen_runner"] = True
            base_seed = int.from_bytes(hashlib.sha256((root + ":" + slot).encode()).digest()[:4], "big") % 2147483647
            for index, public, presentation in runner.display_variants(candidate):
                rid, row, prompt, ids, messages, schema = builder(tok, candidate, source, cid, field, base_seed,
                                                               index, public, presentation, refs)
                total = len(ids) + runner.MAX_TOKENS
                context_ok = total <= runner.MODEL_LEN
                reasoning_ok = reasoner.is_reasoning_end(ids) is True
                failures = []
                if not context_ok:
                    failures.append("CONTEXT_OVERFLOW_NO_TRUNCATION:" + rid + ":" + str(len(ids)))
                if not reasoning_ok:
                    failures.append("THINK_OFF_REASONING_GATE_DRIFT:" + rid)
                request_records.append({"request_id": rid, "candidate_id": cid, "root_id": root,
                                        "arm": arm, "slot_id": slot, "variant_id": f"perm_{index:02}",
                                        "prompt_tokens": len(ids), "reserved_output_tokens": runner.MAX_TOKENS,
                                        "total_reserved_tokens": total, "model_length_limit": runner.MODEL_LEN,
                                        "remaining_prompt_budget_tokens": runner.MODEL_LEN - total,
                                        "context_budget_ok": context_ok, "think_off_gate_ok": reasoning_ok,
                                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                        "token_ids_sha256": digest(ids), "public_request_sha256": digest(row),
                                        "messages_sha256": row["messages_sha256"], "schema_sha256": digest(schema),
                                        "presentation_sha256": digest(presentation), "errors": failures})
                record["request_ids"].append(rid)
                record["errors"].extend(failures)
        except Exception as exc:
            record["errors"].append(f"{type(exc).__name__}: {exc}")
        errors.extend({"candidate_id": cid, "error": error} for error in record["errors"])
    if not request_records:
        errors.append({"candidate_id": None, "error": "REQUEST_COUNT: no content-PASS requests"})
    expected_requests = sum(record["included_by_frozen_runner"] for record in candidate_records) * 2
    if len(request_records) != expected_requests or len({row["request_id"] for row in request_records}) != len(request_records):
        errors.append({"candidate_id": None, "error": "REQUEST_COUNT_OR_DISPLAY_PAIR_DRIFT"})
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU precheck initialized CUDA")
    if execution_snapshot() != before_execution:
        raise RuntimeError("execution/ changed during CPU precheck; no success receipt is written")
    again = checked_freezes()
    if (again["content_binding"] != freezes["content_binding"]
            or again["analysis_binding"] != freezes["analysis_binding"]):
        raise RuntimeError("Frozen inputs changed during CPU precheck")
    code_paths = [Path(__file__).resolve(), RUNNER_PATH,
                  BASE.parent / "thinking_switch_20261024_structured_v2/run.py", *runner.orig.code_paths()]
    if wrapper_path is not None:
        code_paths.append(wrapper_path.resolve())
    bindings = [bind(path) for path in code_paths]
    bindings += api_receipt["implementation_bindings"] + source_bindings
    bindings += [bind(BASE / name) for name in ("INPUT_LOCK.json", "CONFIG.json", "ANALYSIS_DRIVER_FREEZE.json",
                 "ANALYSIS_PLAN.json", "analysis_replication.py", "CONTENT_FREEZE.json", "CANDIDATES_PRIVATE.json",
                 "CONTENT_REVIEW_AI.json", "SOURCE_SELECTION_FINAL.json", "SOURCE_FREEZE.json",
                 "PROTOCOL_ZH.md", "question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md")]
    adaptation_path = BASE / "RUNTIME_INTERPRETER_ADAPTATION.json"
    if adaptation_path.is_file():
        bindings.append(bind(adaptation_path))
    bindings = sorted({row["path"]: row for row in bindings}.values(), key=lambda row: row["path"])
    for row in bindings:
        verify(row)
    return {"schema_version": "fgr20_exact_solver_cpu_length_precheck_v1",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "PASS" if not errors else "FAIL",
            "content_freeze_verified": True, "analysis_driver_freeze_verified": True,
            "content_freeze_binding": freezes["content_binding"], "analysis_driver_freeze_binding": freezes["analysis_binding"],
            "bindings": bindings, "builder": builder_receipt,
            "counts": {"planned_semantic_questions": design["planned_semantic_questions"],
                       "planned_requests": design["planned_solver_requests"], "selected_sources": len(roots),
                       "candidate_records": len(candidate_records), "review_records": len(review_rows),
                       "content_pass_semantic_questions": sum(record["included_by_frozen_runner"] for record in candidate_records),
                       "requests_checked": len(request_records), "overflow_requests": sum(not row["context_budget_ok"] for row in request_records),
                       "failed_think_off_gate_requests": sum(not row["think_off_gate_ok"] for row in request_records)},
            "maximum_prompt_tokens": max((row["prompt_tokens"] for row in request_records), default=None),
            "maximum_total_reserved_tokens": max((row["total_reserved_tokens"] for row in request_records), default=None),
            "candidate_records": candidate_records, "request_records": request_records, "errors": errors,
            "selected_source_bindings": source_bindings, "endpoint": endpoint,
            "compatible_config_receipt": config_receipt, "cpu_api_check": api_receipt,
            "runtime_versions": runner.orig.versions(),
            "runtime_wrapper_binding": bind(wrapper_path) if wrapper_path is not None else None,
            "no_model_or_solver_calls": True, "gpu_model_started": False, "cuda_context_initialized": False,
            "source_truncation": False, "semantic_changes": False, "automatic_exclusions": False,
            "execution_artifacts_untouched": True,
            "prepare_invoked": False, "run_invoked": False, "analyze_invoked": False,
            "failure_policy": "Report every precheck failure; no candidate exclusion, rewrite, regeneration or retry is authorized by this receipt."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-wrapper", type=Path, help="Task-local module exposing get_runner(); its exact bytes are bound")
    parser.add_argument("--self-check", action="store_true", help="Standard-library AST checks only; no tokenizer/model/GPU/input freeze reads or writes")
    args = parser.parse_args()
    if args.self_check:
        _, receipt = extract_builder()
        print(json.dumps({"status": "PASS_STANDARD_LIBRARY_AST_ONLY", "builder": receipt,
                          "model_calls": 0, "gpu_started": False, "files_written": False}, ensure_ascii=False))
        return 0
    if OUTPUT.exists():
        prior = read(OUTPUT)
        checked_freezes()
        for row in prior.get("bindings", []):
            verify(row)
        if prior.get("runtime_wrapper_binding") != (bind(args.runtime_wrapper) if args.runtime_wrapper else None):
            raise RuntimeError("Existing CPU precheck runtime wrapper differs")
        if bind(Path(__file__).resolve()) not in prior.get("bindings", []):
            raise RuntimeError("Existing CPU precheck does not bind this implementation")
        print(json.dumps({"idempotent_reuse": True, "status": prior["status"], "receipt_binding": bind(OUTPUT)}, ensure_ascii=False))
        return 0 if prior["status"] == "PASS" else 2
    value = run_check(args.runtime_wrapper)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": value["status"], "counts": value["counts"],
                      "maximum_prompt_tokens": value["maximum_prompt_tokens"], "receipt_binding": bind(OUTPUT)}, ensure_ascii=False))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
