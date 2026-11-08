"""Bind sealed v104 to the unchanged v15 full-only paired Solver protocol.

The local protocol copy changes only its generator guard and contract path.
No generation, decoding, permutation, scoring, or selection code is replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
BATCH = HERE.parent / "h4_batch_codex_benchmark_v104"
NATIVE = HERE.parent / "h4_feedback_training_pool_v15_qa/evaluation"
MODELS = {"trained": "M_clean_policy", "base": "M0"}
PACKAGES = ("torch", "vllm", "transformers", "tokenizers", "xgrammar")


def now():
    return datetime.now(timezone.utc).isoformat()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify(rows):
    for row in rows:
        need(sha(row["path"]) == row["sha256"], "FROZEN_FILE_CHANGED:" + row["path"])


def sealed_files():
    manifest = read(BATCH / "COMPLETE_MANIFEST.json")
    need(manifest["batch_status"] == "CONTENT_REVIEWED_NOT_SOLVER_OR_TRAINING_APPROVED", "SOURCE_NOT_SEALED")
    rows = [bind(BATCH / "COMPLETE_MANIFEST.json")]
    for row in manifest["files"]:
        path = BATCH / row["path"]
        need(path.is_relative_to(BATCH) and path.stat().st_size == row["bytes"]
             and sha(path) == row["sha256"], "SEALED_CONTENT_CHANGED:" + str(path))
        rows.append({"path": str(path), "sha256": row["sha256"]})
    return rows


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def native():
    sys.path.insert(0, str(NATIVE))
    result = module("v104_unchanged_native_runner", NATIVE / "run_solver.py")
    local = module("v104_adapter_protocol", HERE / "adapter_protocol.py")
    result.validate_pool = local.validate_pool
    return result


def prepare():
    need(not (HERE / "ADAPTER_BINDINGS.json").exists(), "ALREADY_PREPARED_NO_OVERWRITE")
    frozen = sealed_files()
    accounting = read(BATCH / "FINAL_BATCH_ACCOUNTING.json")
    freeze = read(BATCH / "CONTENT_FREEZE.COMPLETE.json")
    need(freeze["status"] == "CONTENT_REVIEWED_NOT_SOLVER_OR_TRAINING_APPROVED"
         and accounting["candidate_count"] == 28 and accounting["content_passed_count"] == 22
         and accounting["new_content_candidates_count"] == 18, "UNEXPECTED_BATCH_STATUS")
    config = read(BATCH / "BENCHMARK_STARTED.json")
    need(config["author_backend"] == "CURRENT_CODEX_SESSION_AND_INHERITED_SUBAGENTS"
         and config["qwen_transport_used"] is False and config["solver_run"] is False
         and config["training_run"] is False, "GENERATOR_PROVENANCE_CHANGED")
    delivery_review = read(BATCH / "evidence/FINAL_DELIVERY_AUDIT.json")
    need(delivery_review["decision"] == "PASS" and not delivery_review["blocking_issues"],
         "FINAL_DELIVERY_REVIEW_NOT_PASS")
    original = (NATIVE / "protocol.py").read_text(encoding="utf-8")
    old_guard = 'if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
    new_guard = ('if (provenance.get("backend") != "CURRENT_CODEX_SESSION"\n'
                 '            or provenance.get("exact_model_id") is not None\n'
                 '            or provenance.get("source_batch_id") != "h4_batch_codex_benchmark_v104"):')
    old_path = 'CONTRACT_DIR = Path(__file__).resolve().parent.parent'
    new_path = "CONTRACT_DIR = Path(" + repr(str(NATIVE.parent)) + ")"
    need(original.count(old_guard) == original.count(old_path) == 1, "PROTOCOL_ADAPTER_PATCH_DRIFT")
    adapted = original.replace(old_guard, new_guard).replace(old_path, new_path)
    with (HERE / "adapter_protocol.py").open("x", encoding="utf-8") as stream:
        stream.write(adapted)
    public = read(BATCH / "SELECTED_PUBLIC_QUESTIONS.json")["questions"] + read(
        BATCH / "CONTROL_PUBLIC_QUESTIONS.json")["questions"]
    private = read(BATCH / "SELECTED_PRIVATE_GOLD.json")["answers"] + read(
        BATCH / "CONTROL_PRIVATE_GOLD.json")["answers"]
    public_ids, private_ids = [q["candidate_id"] for q in public], [q["candidate_id"] for q in private]
    need(len(public_ids) == len(set(public_ids)) == len(private_ids) == len(set(private_ids)) == 22
         and set(public_ids) == set(private_ids), "PUBLIC_PRIVATE_CANDIDATE_SET_MISMATCH")
    gold = {q["candidate_id"]: q for q in private}
    questions, receipt = [], []
    for q in public:
        p = gold[q["candidate_id"]]
        source_path = (PROJECT / q["source_path"]).resolve(strict=True)
        need(source_path.is_relative_to(BATCH) and sha(source_path) == q["source_sha256"], "SOURCE_CHANGED")
        source = read(source_path)
        need(source["root_id"] == q["root_id"], "SOURCE_ROOT_MISMATCH")
        refs = {t["turn_ref"]: t for t in source["turns"]}
        need(len(refs) == len(source["turns"]), "DUPLICATE_SOURCE_TURN")
        root_path = source_path.parent
        root_public = read(root_path / "PUBLIC_QUESTIONS.json")["questions"]
        need(q in root_public, "BATCH_ROOT_PUBLIC_MISMATCH")
        review = read(root_path / "FINAL_CROSSCHECK.json")
        matches = [r for r in review["outcomes"] if r["candidate_id"] == q["candidate_id"]]
        need(len(matches) == 1 and matches[0]["status"] == "PASS"
             and matches[0]["original_gold"] == p["gold"], "FINAL_REVIEW_GOLD_MISMATCH")
        if q["question_type"] == "multi_select":
            need(matches[0]["independent_supported_labels"] == p["gold"]
                 and not matches[0]["unknown_labels"], "FINAL_REVIEW_SUPPORTED_LABEL_MISMATCH")
        labels = tuple("ABCDEF") if q["question_type"] == "multi_select" else tuple(f"I{i}" for i in range(1, 7))
        need(set(p["evidence"]) == set(labels), "INCOMPLETE_PRIVATE_EVIDENCE")
        evidence = {}
        for label in labels:
            citations = p["evidence"][label]
            citations = citations if isinstance(citations, list) else [citations]
            need(bool(citations), "EMPTY_CITATIONS")
            for citation in citations:
                turn = refs.get(citation.get("turn_ref"))
                need(turn is not None and citation["quote"] and citation["quote"] in turn["text"], "NONVERBATIM_CITATION")
                need(citation.get("speaker", turn["speaker"]) == turn["speaker"], "CITATION_SPEAKER_MISMATCH")
            evidence[label] = {**citations[0], "speaker": refs[citations[0]["turn_ref"]]["speaker"]}
            if q["question_type"] == "multi_select":
                evidence[label].update(relation="supports" if label in p["gold"] else "contradicts",
                                       reason="Relation copied from sealed private Gold and PASS review; no new semantic judgment.")
        normalized = {**q, "source_path": str(source_path), "gold": p["gold"], "role": p["role"]}
        key = "option_evidence" if q["question_type"] == "multi_select" else "item_evidence"
        normalized[key] = evidence
        if q["question_type"] == "ordering":
            normalized["order_anchor_spans"] = evidence
        questions.append(normalized)
        receipt.append({"candidate_id": q["candidate_id"], "public_unchanged_except_absolute_source_path": True,
                        "original_private": p, "source": bind(source_path),
                        "final_crosscheck": bind(root_path / "FINAL_CROSSCHECK.json")})
    counts = Counter(q["question_type"] for q in questions)
    need(counts == {"multi_select": 16, "ordering": 6}, "TYPE_COUNTS_CHANGED")
    need(Counter(len(q["gold"]) for q in questions if q["question_type"] == "multi_select") == {2: 2, 3: 4, 4: 3, 5: 3, 6: 4}, "GOLD_COUNTS_CHANGED")
    provenance = {"backend": "CURRENT_CODEX_SESSION", "exact_model_id": None,
                  "source_batch_id": BATCH.name, "source_batch_manifest": bind(BATCH / "COMPLETE_MANIFEST.json"),
                  "semantic_review_performed": True, "reviewer_backend": "CURRENT_CODEX_SESSION",
                  "same_inherited_model_and_shared_session_background": True}
    write(HERE / "EVALUATION_POOL.json", {"status": "SEALED_CONTENT_READY_FOR_SOLVER_EVALUATION",
          "candidate_count": len(questions), "provenance": provenance, "questions": questions})
    write(HERE / "ADAPTATION_RECEIPT.json", {"prepared_at_utc": now(), "questions": receipt,
          "native_protocol": bind(NATIVE / "protocol.py"), "adapter_protocol": bind(HERE / "adapter_protocol.py"),
          "changes": ["Generator provenance guard accepts only sealed v104 CURRENT_CODEX_SESSION; exact model remains unknown.",
                      "CONTRACT_DIR points to unchanged native v15 mechanical gates."],
          "solver_protocol": "v15_full_only_random_mcq6x3_order12x1_v1", "ground_calls": 0,
          "new_semantic_judgments": False, "public_text_edits": False, "training_started": False,
          "sampling": "Different roots from v105, but previously trained roots; fresh question diagnostic, not held-out generalization.",
          "generator_exact_model_id": None, "source_provenance_record": bind(BATCH / "BENCHMARK_STARTED.json")})
    # Validate before the first tokenizer or model call; preserve unchanged native gates.
    n = native()
    _, checked, _ = n.validate_pool(HERE / "EVALUATION_POOL.json")
    need(checked == questions, "VALIDATOR_ALTERED_QUESTIONS")
    paths = [Path(__file__), HERE / "adapter_protocol.py", HERE / "EVALUATION_POOL.json", HERE / "ADAPTATION_RECEIPT.json"]
    paths += [NATIVE / name for name in ("run_solver.py", "protocol.py", "scoring.py", "selection.py")]
    paths += list(NATIVE.parent.glob("*.py"))
    write(HERE / "ADAPTER_BINDINGS.json", {"prepared_at_utc": now(), "status": "PASS",
          "source_batch_files": frozen, "files": [bind(p) for p in sorted(set(paths))],
          "candidate_count": 22, "calls_per_model": 360, "models": MODELS,
          "automatic_retry": False, "training_started": False})
    print(json.dumps({"status": "ADAPTER_READY", "questions": 22, "calls_per_model": 360}), flush=True)


def verify_adapter():
    binding = read(HERE / "ADAPTER_BINDINGS.json")
    verify(binding["source_batch_files"])
    verify(binding["files"])
    return binding


def preflight():
    verify_adapter()
    n = native()
    for side, model in MODELS.items():
        n.preflight(HERE / "EVALUATION_POOL.json", HERE / "run" / side, solver_model=model)
    manifests = {side: read(HERE / "run" / side / "manifest.json") for side in MODELS}
    left, right = manifests.values()
    allowed = {"prepared_at_utc", "model", "solver_model", "solver_identity"}
    need({k: v for k, v in left.items() if k not in allowed} == {k: v for k, v in right.items() if k not in allowed}, "PAIRED_INPUTS_DIFFER")
    need(left["request_count"] == right["request_count"] == 360, "CALL_COUNT_CHANGED")
    for side in MODELS:
        run = HERE / "run" / side
        need(read(run / "questions.json") == read(HERE / "run" / "trained" / "questions.json"), "PAIRED_QUESTIONS_DIFFER")
        need(read(run / "gold.json") == read(HERE / "run" / "trained" / "gold.json"), "PAIRED_GOLD_DIFFER")
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    paths = [p for side in MODELS for p in (HERE / "run" / side).rglob("*") if p.is_file()]
    verify_adapter()
    write(HERE / "PAIRED_PREFLIGHT.json", {"status": "PASS", "prepared_at_utc": now(),
          "files": [bind(p) for p in sorted(paths)], "adapter_binding": bind(HERE / "ADAPTER_BINDINGS.json"),
          "packages": packages, "python_prefix": sys.prefix, "candidate_count": 22,
          "calls_per_model": 360, "same_prompts_permutations_seeds_decoder_scoring": True,
          "max_prompt_tokens": max(r["prompt_tokens"] for r in left["requests"]), "training_started": False})
    print(json.dumps({"status": "PAIRED_PREFLIGHT_PASS", "calls_per_model": 360,
                      "max_prompt_tokens": max(r["prompt_tokens"] for r in left["requests"])}), flush=True)


def live(side):
    verify_adapter()
    pre = read(HERE / "PAIRED_PREFLIGHT.json")
    verify(pre["files"])
    verify([pre["adapter_binding"]])
    need(pre["python_prefix"] == sys.prefix, "PYTHON_PREFIX_CHANGED")
    need(pre["packages"] == {name: importlib.metadata.version(name) for name in PACKAGES}, "PACKAGE_VERSION_CHANGED")
    marker = HERE / (side.upper() + "_EXECUTION")
    need(not marker.with_suffix(".STARTED.json").exists(), "NO_AUTOMATIC_RERUN")
    start = time.monotonic()
    write(marker.with_suffix(".STARTED.json"), {"started_at_utc": now(), "model": MODELS[side],
                                               "paired_preflight": bind(HERE / "PAIRED_PREFLIGHT.json")})
    n = native()
    n.live(HERE / "run" / side)
    n.report(HERE / "run" / side)
    verify_adapter()
    verify(pre["files"])
    write(marker.with_suffix(".COMPLETE.json"), {"completed_at_utc": now(), "wall_seconds": time.monotonic() - start,
          "model": MODELS[side], "calls": read(HERE / "run" / side / "COMPLETE.json")["actual_call_count"],
          "training_started": False, "new_question_generation_calls": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight", "live"))
    parser.add_argument("--side", choices=tuple(MODELS))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "preflight":
        preflight()
    else:
        need(args.side is not None, "LIVE_SIDE_REQUIRED")
        live(args.side)

