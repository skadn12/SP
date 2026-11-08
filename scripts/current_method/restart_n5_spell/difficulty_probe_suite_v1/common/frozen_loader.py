from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io_utils import read_json, sha256_file


DEFAULT_RUN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_41625_child_promotion_qa_live_1")
DEFAULT_MEMORY_RUN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_41625_memory_only_fast_expansion_v1_live_1")


def _turn_ref(row: dict[str, Any]) -> str:
    return str(row.get("turn_ref") or row.get("turn_id") or "")


def load_dialogue(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    turns = payload.get("turns") if isinstance(payload, dict) else None
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"frozen_dialogue_turns_missing:{path}")
    normalized = []
    for index, row in enumerate(turns, 1):
        normalized.append({
            "turn_ref": _turn_ref(row) or f"turn-{index:04d}",
            "speaker": str(row.get("speaker") or ""),
            "text": str(row.get("text") or ""),
            "origin": str(row.get("origin") or "unknown"),
            "scene_ref": row.get("scene_ref"),
            "position": index,
        })
    return {"root_id": str(payload.get("root_id") or ""), "turns": normalized,
            "source_path": str(path), "sha256": sha256_file(path)}


def _eval_for(eval_dir: Path, qa_id: str) -> dict[str, Any]:
    path = eval_dir / f"{qa_id}_evaluation.json"
    if not path.exists():
        return {}
    return read_json(path)


def load_frozen_inputs(run_root: Path = DEFAULT_RUN) -> dict[str, Any]:
    dialogue_path = run_root / "frozen" / "expanded_dialogue.json"
    batch_path = run_root / "qa" / "final_mcq_batch.json"
    eval_dir = run_root / "reports"
    dialogue = load_dialogue(dialogue_path)
    batch = read_json(batch_path)
    items = batch.get("items") if isinstance(batch, dict) else batch
    if not isinstance(items, list):
        raise ValueError("frozen_batch_items_missing")
    rows = []
    turns = {row["turn_ref"]: row for row in dialogue["turns"]}
    for item in items:
        qa_id = str(item.get("qa_id") or item.get("candidate_id") or "")
        refs = [str(x) for x in (item.get("evidence_turn_refs") or item.get("evidence_turn_ids") or [])]
        evidence = [turns[ref] for ref in refs if ref in turns]
        evaluation = _eval_for(eval_dir, qa_id)
        grounding = evaluation.get("grounding") if isinstance(evaluation, dict) else {}
        full = evaluation.get("full_context_rollouts") if isinstance(evaluation, dict) else []
        rows.append({
            "qa_id": qa_id,
            "question": str(item.get("question") or ""),
            "options": dict(item.get("options") or {}),
            "gold_letter": str(item.get("reference_answer_letter") or item.get("correct_option") or ""),
            "gold_text": str(item.get("canonical_answer_text") or item.get("reference_answer_text") or ""),
            "evidence_turn_refs": refs,
            "evidence_texts": [row["text"] for row in evidence],
            "source_scope": item.get("source_scope"),
            "question_type": item.get("question_type"),
            "evaluation": evaluation,
            "grounding_decision": grounding.get("decision") if isinstance(grounding, dict) else None,
            "full_context_correct": int(evaluation.get("full_context_correct") or 0) if isinstance(evaluation, dict) else 0,
            "full_context_total": int(evaluation.get("full_context_total") or 0) if isinstance(evaluation, dict) else 0,
            "historical_full_rollouts": full if isinstance(full, list) else [],
            "eligible_grounded": bool(item.get("evidence_ready")) and grounding.get("decision") == "pass_grounding",
        })
    memory = {}
    overview = DEFAULT_MEMORY_RUN / "memory" / "pre_generation_overview.json"
    if overview.exists():
        memory = read_json(overview)
    return {"dialogue": dialogue, "qa_rows": rows, "memory_overview": memory,
            "inputs": {"dialogue": str(dialogue_path), "batch": str(batch_path),
                        "evaluations": str(eval_dir), "dialogue_sha256": dialogue["sha256"],
                        "batch_sha256": sha256_file(batch_path)}}


def parent_view(dialogue: dict[str, Any]) -> dict[str, Any]:
    turns = [row for row in dialogue["turns"] if row["origin"] == "source_grounded"]
    if not turns:
        turns = dialogue["turns"][:53]
    return {"root_id": dialogue["root_id"], "dialogue": {"turns": [
        {"turn_id": row["turn_ref"], "speaker": row["speaker"], "text": row["text"]}
        for row in turns]}}

