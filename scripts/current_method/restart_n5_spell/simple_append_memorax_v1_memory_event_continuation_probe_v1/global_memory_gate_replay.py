"""Zero-API R4 replay for the global Memory candidate check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .contracts import build_semantic_gate_context
from .incremental_round import load_incremental_round_input
from .memory_context import compile_global_memory_state_view
from .prompts import build_semantic_continuation_gate_prompt


RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
R3 = RUNS / "conversation_chronicles_episode_41625_memory_only_incremental_r3_live_2"
R4 = RUNS / "conversation_chronicles_episode_41625_memory_only_incremental_r4_live_2"
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> dict:
    parent, _, validation = load_incremental_round_input(
        R3 / "promotion/promotable_expanded_dialogue.json", round_index=4)
    snapshot = _read(R4 / "memory/recovery_pre_generation_bucket_snapshot.json")
    candidates = _read(R4 / "memory/event_candidates.json")["candidates"]
    plan = _read(R4 / "planner/compiled_plan.json")
    view = compile_global_memory_state_view(snapshot)
    context = build_semantic_gate_context(
        parent=parent, candidates=candidates, plan=plan,
        memory_only_fast=True, global_memory_state_view=view)
    prompt = build_semantic_continuation_gate_prompt(
        gate_context=context)["prompt"]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER,
                                               local_files_only=True)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    parent_texts = [str(row.get("text") or "")
                    for row in parent["dialogue"]["turns"]]
    return {
        "replay_version": "global_memory_gate_r4_replay_v1",
        "api_calls": 0,
        "root_id": parent["root_id"],
        "input_validation": validation,
        "source_snapshot_sha256": view["source_snapshot_sha256"],
        "global_memory_state_view": view,
        "candidate_refs": context["expected_candidate_refs"],
        "candidate_count": context["expected_candidate_count"],
        "parent_transcript_text_in_gate_context": any(
            text and len(text) > 20 and text in str(context)
            for text in parent_texts),
        "gate_prompt_tokens": prompt_tokens,
        "gate_completion_budget": 2048,
        "context_limit": 40960,
        "remaining_context_margin": 40960 - prompt_tokens - 2048,
        "semantic_decisions_generated": 0,
        "interpretation": (
            "The mechanical path is ready. A live Gate call is still required "
            "to measure semantic duplicate decisions."),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) +
                           "\n", encoding="utf-8")
    print(json.dumps({
        "api_calls": 0,
        "bucket_counts": result["global_memory_state_view"]["bucket_counts"],
        "candidate_count": result["candidate_count"],
        "parent_transcript_text_in_gate_context": result[
            "parent_transcript_text_in_gate_context"],
        "gate_prompt_tokens": result["gate_prompt_tokens"],
        "remaining_context_margin": result["remaining_context_margin"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
