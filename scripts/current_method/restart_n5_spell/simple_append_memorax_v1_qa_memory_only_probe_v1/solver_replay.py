"""Local Solver replay for transcript-verified Memory-only QA candidates.

This is intentionally isolated from the main 8-12 question QA runner. It
reuses the existing local no-context grounding and full-context K=5 frontier
logic, while preserving the six-item truth-audited candidate set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..core import atomic_json, sha256_file
from ..ledger import RequestLedger
from ..simple_append_v1.qa import classify_full_context_frontier
from ..simple_append_v1.qa_live import (
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
    LocalSolverRunner,
    SOLVER_GPU_MEMORY_UTILIZATION,
    SOLVER_MAX_MODEL_LEN,
    SOLVER_MAX_TOKENS,
    _evaluate_grounding,
    build_solver_source,
)
from ..vllm_backends import VLLMSolverBackend


DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_memory_only_fast_expansion_v1_live_1"
)
DEFAULT_CANDIDATES = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_qa_memory_only_probe_v1_live_2/"
    "qa/transcript_verified_candidates.json"
)
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_qa_memory_only_probe_v1_solver_replay_1"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        payload = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(str(len(payload)).encode())
        digest.update(hashlib.sha256(payload).hexdigest().encode())
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _qa_rows(candidates: dict[str, Any]) -> list[dict[str, Any]]:
    rows = candidates.get("items")
    if not isinstance(rows, list) or not rows:
        raise ValueError("verified_candidate_items_required")
    result = []
    for row in rows:
        if not isinstance(row, dict) or row.get("transcript_verified") is not True:
            raise ValueError("all_solver_rows_must_be_transcript_verified")
        options = row.get("options")
        if not isinstance(options, dict) or set(options) != set("ABCD"):
            raise ValueError(f"solver_options_invalid:{row.get('qa_id')}")
        result.append({
            "qa_id": str(row["qa_id"]),
            "question": str(row["question"]),
            "question_type": str(row["question_type"]),
            "options": {letter: str(options[letter]) for letter in "ABCD"},
            "canonical_answer_text": str(options[str(row["correct_option"])]),
            "correct_option": str(row["correct_option"]),
            "evidence_turn_refs": list(row.get("evidence_turn_refs") or []),
            "transcript_verified": True,
        })
    return result


def run(*, source_run: Path, candidates_path: Path, run_root: Path,
        tokenizer_path: Path, solver_model: Path) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"run_root_exists:{run_root}")
    source = _load(source_run / "expanded_dialogue.json")
    candidates = _load(candidates_path)
    rows = _qa_rows(candidates)
    run_root.mkdir(parents=True)
    for relative in ("config", "frozen", "raw/prompts", "raw/backend_responses",
                     "ledger", "reports", "solver"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_version": "memory_only_verified_solver_replay_v1",
        "created_at": _now(),
        "source_run": str(source_run),
        "candidates_path": str(candidates_path),
        "source_tree_sha256": _hash_tree(source_run),
        "source_dialogue_sha256": sha256_file(source_run / "expanded_dialogue.json"),
        "candidates_sha256": sha256_file(candidates_path),
        "candidate_count": len(rows),
        "solver_model": str(solver_model),
        "tokenizer": str(tokenizer_path),
        "paid_api_requests": 0,
        "local_policy": {
            "no_context_orders": 4,
            "no_context_samples_per_order": 1,
            "full_context_k": 5,
            "frontier_rule": "spell_frontier_1_to_4_of_5",
        },
    }
    atomic_json(run_root / "config/run_manifest.json", manifest)
    atomic_json(run_root / "frozen/verified_candidates.json", candidates)
    atomic_json(run_root / "frozen/expanded_dialogue.json", source)
    manifest_sha = sha256_file(run_root / "config/run_manifest.json")
    ledger = RequestLedger(run_root, manifest_sha, "local")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    backend = VLLMSolverBackend(
        model_path=solver_model, tokenizer_path=tokenizer_path,
        max_model_len=SOLVER_MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
        max_tokens=SOLVER_MAX_TOKENS,
        gpu_memory_utilization=SOLVER_GPU_MEMORY_UTILIZATION,
    )
    solver_source = build_solver_source(source)
    runner = LocalSolverRunner(
        run_root=run_root, ledger=ledger, tokenizer=tokenizer, backend=backend,
        dialogue=solver_source["combined_dialogue"],
    )
    results: list[dict[str, Any]] = []
    for qa in rows:
        grounding, no_context = _evaluate_grounding(runner, qa)
        full_context = []
        if grounding.get("decision") == "pass_grounding":
            for slot in range(5):
                full_context.append(runner.one(
                    qa=qa, options=qa["options"], mode="full_context",
                    order_id="original", slot=slot,
                ))
        rewards = [row.get("local_reward") for row in full_context]
        frontier = classify_full_context_frontier(rewards)
        results.append({
            "qa_id": qa["qa_id"],
            "question": qa["question"],
            "grounding": grounding,
            "no_context": no_context,
            "full_context": full_context,
            "full_context_rewards": rewards,
            "full_context_frontier": frontier,
            "admitted": bool(grounding.get("decision") == "pass_grounding" and
                              frontier.get("admitted") is True),
        })
    completed = ledger.completed()
    report = {
        "report_version": "memory_only_verified_solver_replay_v1",
        "status": "completed",
        "candidate_count": len(rows),
        "solver_calls": len(completed),
        "paid_api_requests": 0,
        "items": results,
        "summary": {
            "grounding_pass_count": sum(
                row["grounding"].get("decision") == "pass_grounding"
                for row in results),
            "full_context_ran_count": sum(bool(row["full_context"])
                                            for row in results),
            "admitted_count": sum(bool(row["admitted"]) for row in results),
        },
        "source_unchanged": _hash_tree(source_run) == manifest["source_tree_sha256"],
        "retries": 0,
    }
    atomic_json(run_root / "reports/solver_replay.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--solver-model", type=Path, default=DEFAULT_SOLVER_MODEL)
    args = parser.parse_args()
    report = run(source_run=args.source_run, candidates_path=args.candidates,
                 run_root=args.run_root, tokenizer_path=args.tokenizer,
                 solver_model=args.solver_model)
    print(json.dumps({"status": report["status"], **report["summary"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
