"""Frozen single-call R04 multi-select Writer diagnostic for root 41625."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
PAIRED = HERE.parent / "h4_paired_trainable_qa_yield_pilot_v1"
SOURCE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_41625_memory_only_incremental_r4_live_2/promotion/promotable_expanded_dialogue.json")
SOURCE_SHA256 = "67453cf45b290fe23383325a9756295dbc4ab11a8b6fa1a836d889144e9c964b"
INSTRUCTIONS = HERE.parent / "h4_graph_guided_qa_pilot_v2" / "writer_instructions.txt"
INSTRUCTIONS_SHA256 = "6d78bda6a4609e1af591266da3bf57ffb25f4076159aac3989a066686617fc66"

sys.path.insert(0, str(PAIRED))
from common import call_qwen, sha_bytes, write_once  # noqa: E402


def source_surface() -> dict[str, Any]:
    if sha_bytes(SOURCE.read_bytes()) != SOURCE_SHA256:
        raise RuntimeError("source_hash_mismatch")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if (source.get("root_id"), source.get("turn_count"), source.get("round_count")) != ("cc-train-episode-41625", 234, 4):
        raise RuntimeError("source_counts_mismatch")
    validation = json.loads(SOURCE.with_name("promotable_expanded_dialogue_validation.json").read_text(encoding="utf-8"))
    if validation != {"valid": True, "errors": [], "prior_turn_count": 192, "new_turn_count": 42, "turn_count": 234}:
        raise RuntimeError("promotion_validation_mismatch")
    rows = [{key: turn[key] for key in ("turn_ref", "speaker", "text")} for turn in source["turns"]]
    if len(rows) != 234 or len({row["turn_ref"] for row in rows}) != 234:
        raise RuntimeError("turn_refs_invalid")
    return {"root_id": source["root_id"], "source_sha256": SOURCE_SHA256, "turns": rows}


def prompt(surface: dict[str, Any]) -> str:
    if sha_bytes(INSTRUCTIONS.read_bytes()) != INSTRUCTIONS_SHA256:
        raise RuntimeError("writer_instructions_changed")
    original = INSTRUCTIONS.read_text(encoding="utf-8")
    blocks = original.split("\n\n")
    if len(blocks) != 7 or not blocks[2].startswith("MULTI-SELECT:") or not blocks[3].startswith("ORDERING:") or blocks[-1] != "VISIBLE DIALOGUE:\n":
        raise RuntimeError("writer_instruction_layout_changed")
    introduction = blocks[0].replace(
        "a frozen 188-turn history followed by new G02 scenes",
        "a complete promoted P+R01+R02+R03+R04 dialogue",
    ).replace(
        "Write at most ONE multi_select and ONE ordering question.",
        "Write at most TWO multi_select questions in this one attempt.",
    ).replace("abstain from either type", "abstain from multi_select")
    multi = blocks[2].replace("the frozen R03 history", "the earlier P/R01/R02/R03 history").replace("G02", "R04")
    schema = (
        'Return {"questions":[multi_select candidates],"abstentions":[]} when submitting one or two candidates, '
        'or {"questions":[],"abstentions":[{"question_type":"multi_select","reason":"..."}]} when abstaining. '
        'Each candidate must use the original multi-select schema: candidate_id, question_type, intent, question, '
        'options A-E, gold, and option_evidence A-E with verdict and exactly two clauses. '
        'Each clause needs claim, verdict, refs, quote, and reason. Every cited quote must occur exactly in cited speech. '
        'Do not submit an ordering candidate or ordering abstention. Return JSON only.'
    )
    instructions = "\n\n".join((introduction, blocks[1], multi, schema, blocks[-1]))
    if "G02" in instructions or "frozen R03 history" in instructions:
        raise RuntimeError("unadapted_round_reference")
    return instructions + json.dumps({"turns": surface["turns"]}, ensure_ascii=False, indent=2) + "\n"


def freeze() -> dict[str, Any]:
    surface = source_surface()
    body = prompt(surface)
    manifest = {
        "source_path": str(SOURCE), "source_sha256": SOURCE_SHA256,
        "source_turn_count": len(surface["turns"]),
        "manual_question_sha256": sha_bytes((HERE / "question_manual.json").read_bytes()),
        "manual_audit_sha256": sha_bytes((HERE / "SOURCE_AUDIT.md").read_bytes()),
        "adapter_sha256": sha_bytes(Path(__file__).read_bytes()),
        "base_instructions_sha256": INSTRUCTIONS_SHA256,
        "writer_prompt_sha256": sha_bytes(body.encode("utf-8")),
        "writer_model": "Qwen/Qwen3.6-35B-A3B", "writer_temperature": 0,
        "writer_max_tokens": 6144, "writer_enable_thinking": False,
        "writer_http_retry_limit": 0, "writer_request_cap": 1,
        "writer_candidate_cap": 2, "question_type": "multi_select",
        "training_promotion": False,
    }
    write_once(HERE / "writer_prompt.txt", body, raw=True)
    write_once(HERE / "freeze.json", manifest)
    return manifest


def run() -> None:
    frozen = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
    if frozen != freeze():
        raise RuntimeError("freeze_changed")
    body = (HERE / "writer_prompt.txt").read_text(encoding="utf-8")
    call_qwen(HERE / "writer_run", body, frozen_prompt_sha256=frozen["writer_prompt_sha256"])


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.action == "freeze":
        print(json.dumps(freeze(), ensure_ascii=False, indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
