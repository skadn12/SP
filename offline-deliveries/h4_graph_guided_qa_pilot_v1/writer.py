"""One reserved candidate Writer attempt using only the frozen dialogue surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (  # noqa: E402
    credential_client,
    provider_profile,
)

SOURCE = HERE / "realized_dialogue.json"
SOURCE_SHA256 = "35b2d4521134f75a55427a892322410cf5de9dd047c2e6cea807c6efe8ab3ed3"
INSTRUCTIONS = HERE / "writer_instructions.txt"
OUTPUT_LIMIT = 4096


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_new(path: Path, value: object, *, raw: bool = False) -> None:
    with path.open("x", encoding="utf-8") as handle:
        if raw:
            handle.write(str(value))
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def dialogue_surface() -> dict:
    if sha_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("realized_dialogue_hash_mismatch")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    scenes = source.get("scenes")
    flat = source.get("turns")
    if not isinstance(scenes, list) or not isinstance(flat, list) or len(scenes) != 2:
        raise ValueError("realized_dialogue_shape_invalid")
    surface = []
    projected = []
    for scene in scenes:
        scene_id, relative_time = scene.get("scene_id"), scene.get("relative_time")
        if not isinstance(scene_id, str) or not isinstance(relative_time, str):
            raise ValueError("scene_metadata_invalid")
        turns = scene.get("turns")
        if not isinstance(turns, list):
            raise ValueError("scene_turns_invalid")
        visible = []
        for number, turn in enumerate(turns, 1):
            ref = f"G01:{scene_id}:T{number:03d}"
            speaker, utterance = turn.get("speaker"), turn.get("text")
            if not all(isinstance(value, str) and value.strip() for value in (speaker, utterance)):
                raise ValueError("turn_surface_invalid")
            visible.append({"turn_ref": ref, "speaker": speaker, "text": utterance})
            projected.append((ref, speaker, utterance))
        surface.append({"scene_id": scene_id, "relative_time": relative_time, "turns": visible})
    if projected != [(row.get("turn_ref"), row.get("speaker"), row.get("text")) for row in flat]:
        raise ValueError("flat_turns_differ_from_scenes")
    return {"scenes": surface}


def prompt_for(surface: dict) -> str:
    return INSTRUCTIONS.read_text(encoding="utf-8") + json.dumps(surface, ensure_ascii=False, indent=2) + "\n"


def shape_audit(raw_text: str, valid_refs: set[str]) -> dict:
    result = json.loads(raw_text)
    if not isinstance(result, dict) or not isinstance(result.get("questions"), list):
        raise ValueError("questions_array_missing")
    questions = result["questions"]
    kinds = [question.get("question_type") for question in questions if isinstance(question, dict)]
    if len(questions) > 2 or len(set(kinds)) != len(kinds) or any(kind not in {"multi_select", "ordering"} for kind in kinds):
        raise ValueError("question_type_or_count_invalid")
    candidate_ids = [question.get("candidate_id") for question in questions]
    if any(not isinstance(value, str) or not value.strip() for value in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids_invalid")
    for question in questions:
        if not all(isinstance(question.get(field), str) and question[field].strip() for field in ("intent", "question")):
            raise ValueError("intent_or_question_missing")
        kind = question["question_type"]
        field = "options" if kind == "multi_select" else "items"
        labels = list("ABCDE") if kind == "multi_select" else [f"I{i}" for i in range(1, 5)]
        values = question.get(field)
        if not isinstance(values, dict) or list(values) != labels or any(not isinstance(values[key], str) or not values[key].strip() for key in labels):
            raise ValueError(f"{field}_shape_invalid")
        gold = question.get("gold")
        if not isinstance(gold, list) or any(key not in labels for key in gold) or len(set(gold)) != len(gold):
            raise ValueError("gold_shape_invalid")
        if kind == "multi_select":
            if not 2 <= len(gold) <= 3:
                raise ValueError("multi_select_gold_size_invalid")
            evidence = question.get("option_evidence")
            if not isinstance(evidence, dict) or set(evidence) != set(labels):
                raise ValueError("option_evidence_incomplete")
            for key in labels:
                row = evidence[key]
                if not isinstance(row, dict) or row.get("verdict") != ("TRUE" if key in gold else "FALSE"):
                    raise ValueError("option_verdict_mismatch")
                check_refs(row.get("refs"), valid_refs)
                if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                    raise ValueError("option_reason_missing")
        else:
            if len(gold) != 4:
                raise ValueError("ordering_gold_incomplete")
            evidence = question.get("item_evidence")
            if not isinstance(evidence, dict) or set(evidence) != set(labels):
                raise ValueError("item_evidence_incomplete")
            for row in evidence.values():
                if not isinstance(row, dict):
                    raise ValueError("item_evidence_invalid")
                check_refs(row.get("refs"), valid_refs)
                if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                    raise ValueError("item_reason_missing")
            edges = question.get("adjacent_edges")
            if not isinstance(edges, list) or len(edges) != 3:
                raise ValueError("adjacent_edges_incomplete")
            for index, edge in enumerate(edges):
                if not isinstance(edge, dict) or (edge.get("before"), edge.get("after")) != (gold[index], gold[index + 1]):
                    raise ValueError("adjacent_edge_order_invalid")
                before_refs = check_refs(edge.get("before_refs"), valid_refs)
                after_refs = check_refs(edge.get("after_refs"), valid_refs)
                if set(before_refs) & set(after_refs):
                    raise ValueError("adjacent_edge_same_turn")
                if not isinstance(edge.get("reason"), str) or not edge["reason"].strip():
                    raise ValueError("adjacent_edge_reason_missing")
    abstentions = result.get("abstentions")
    if not isinstance(abstentions, list):
        raise ValueError("abstentions_array_missing")
    for row in abstentions:
        if not isinstance(row, dict) or row.get("question_type") not in {"multi_select", "ordering"} or not isinstance(row.get("reason"), str):
            raise ValueError("abstention_invalid")
    return {"status": "PASS_SHAPE_ONLY_NOT_GOLD", "question_count": len(questions), "question_types": kinds,
            "source_sha256": SOURCE_SHA256, "training_promotion": False}


def check_refs(refs: object, valid_refs: set[str]) -> list[str]:
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in valid_refs for ref in refs):
        raise ValueError("evidence_refs_invalid")
    return refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="show exact prompt without reserving/calling")
    args = parser.parse_args()
    surface = dialogue_surface()
    prompt = prompt_for(surface)
    if args.preview:
        print(prompt)
        return
    profile = provider_profile()
    profile["profile_id"] = "h4_graph_guided_qa_pilot_v1_writer"
    profile["request_parameters"].update(temperature=0, max_tokens=OUTPUT_LIMIT, enable_thinking=False)
    profile["output_limit"] = OUTPUT_LIMIT
    profile["http_retry_limit"] = 0
    write_new(HERE / "writer_attempt.reserved.json", {
        "status": "RESERVED_ONE_ATTEMPT",
        "stage": "candidate_writer_only",
        "source_path": str(SOURCE),
        "source_sha256": SOURCE_SHA256,
        "instructions_sha256": sha_file(INSTRUCTIONS),
        "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
        "profile": profile,
        "maximum_http_attempts": 1,
        "training_promotion": False,
    })
    write_new(HERE / "writer_prompt.txt", prompt, raw=True)

    def on_attempt(payload: dict) -> None:
        write_new(HERE / "writer_http_attempt.json", {
            "attempt_number": 1,
            "payload_sha256": sha_bytes(canonical(payload)),
            "prompt_sha256": sha_bytes(payload["messages"][0]["content"].encode("utf-8")),
            "model": payload.get("model"),
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "enable_thinking": payload.get("enable_thinking"),
        })

    try:
        client = credential_client(profile, on_http_attempt=on_attempt)
        response = client.complete(prompt=prompt, max_tokens=OUTPUT_LIMIT, json_object=True)
        write_new(HERE / "writer_response_raw.txt", response.raw_text, raw=True)
        write_new(HERE / "writer_response_meta.json", {
            "status": "RESPONSE_RECEIVED",
            "raw_sha256": sha_bytes(response.raw_text.encode("utf-8")),
            "metadata": response.metadata,
            "training_promotion": False,
        })
    except Exception as error:
        write_new(HERE / "writer_error.json", {
            "status": "ERROR_NO_RETRY",
            "error_type": type(error).__name__,
            "transport_metadata": getattr(error, "metadata", None),
            "http_attempt_recorded": (HERE / "writer_http_attempt.json").exists(),
            "training_promotion": False,
        })
        raise
    valid_refs = {row["turn_ref"] for scene in surface["scenes"] for row in scene["turns"]}
    try:
        audit = shape_audit(response.raw_text, valid_refs)
        write_new(HERE / "writer_shape_audit.json", audit)
        write_new(HERE / "writer_candidates.json", {
            "status": "UNREVIEWED_CANDIDATES_NOT_GOLD",
            "source_sha256": SOURCE_SHA256,
            "questions": json.loads(response.raw_text)["questions"],
            "abstentions": json.loads(response.raw_text)["abstentions"],
            "training_promotion": False,
        })
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        write_new(HERE / "writer_shape_audit.json", {
            "status": "FAILED_SHAPE_ONLY",
            "error_type": type(error).__name__,
            "reason": str(error),
            "training_promotion": False,
        })
        raise


if __name__ == "__main__":
    main()
