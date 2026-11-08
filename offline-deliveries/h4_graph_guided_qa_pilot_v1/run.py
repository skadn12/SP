"""One reserved SiliconFlow dialogue-realization attempt for a frozen event graph."""

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

GRAPH = HERE / "graph.json"
INSTRUCTIONS = HERE / "prompt_instructions.txt"
OUTPUT_LIMIT = 3000


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def verify_sources(graph: dict) -> None:
    provenance = graph["provenance"]
    sample = Path(provenance["local_sample_path"])
    if sha_file(sample) != provenance["local_sample_sha256"]:
        raise RuntimeError("external_sample_hash_mismatch")
    dialogues = {d["dialogue_id"]: d for d in json.loads(sample.read_text(encoding="utf-8"))}
    for source in provenance["source_dialogues"]:
        dialogue = dialogues[source["dialogue_id"]]
        if sha_bytes(canonical(dialogue)) != source["canonical_dialogue_sha256"]:
            raise RuntimeError("source_dialogue_hash_mismatch")
        for index, expected in zip(source["turn_indices_zero_based"], source["canonical_turn_sha256"], strict=True):
            if sha_bytes(canonical(dialogue["turns"][index])) != expected:
                raise RuntimeError("source_turn_hash_mismatch")
    continuation = graph["continuation"]
    if sha_file(Path(continuation["r03_source_path"])) != continuation["r03_source_sha256"]:
        raise RuntimeError("r03_source_hash_mismatch")


def prompt_for(graph: dict) -> str:
    domain = json.loads(json.dumps(graph["transformed_domain"]))
    for track_key in ("track_A", "track_B"):
        domain[track_key].pop("source_relation")
    specification = {
        "navigation_only": graph["continuation"]["navigation_only"],
        "transformed_domain": domain,
        "events": graph["events"],
        "chronology": graph["chronology"],
        "scene_transition": graph["scene_transition"],
        "evidence_contract": graph["evidence_contract"],
    }
    return INSTRUCTIONS.read_text(encoding="utf-8") + json.dumps(specification, ensure_ascii=False, indent=2) + "\n"


def materialize(raw_text: str, graph: dict) -> dict:
    parsed = json.loads(raw_text)
    scenes = parsed["scenes"]
    if not isinstance(scenes, list) or [s.get("scene_id") for s in scenes] != ["S01", "S02"]:
        raise ValueError("scene_shape_or_order_invalid")
    rows = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        if not isinstance(scene.get("relative_time"), str) or not scene["relative_time"].strip():
            raise ValueError("relative_time_missing")
        turns = scene.get("turns")
        if not isinstance(turns, list):
            raise ValueError("turns_missing")
        for index, turn in enumerate(turns, 1):
            if not isinstance(turn, dict) or turn.get("speaker") not in {"Emily", "Nora"}:
                raise ValueError("speaker_invalid")
            if not isinstance(turn.get("text"), str) or not turn["text"].strip():
                raise ValueError("text_invalid")
            key = turn.get("event_key", "")
            if not isinstance(key, str):
                raise ValueError("event_key_invalid")
            rows.append({
                "turn_ref": f"G01:{scene_id}:T{index:03d}",
                "session_ref": f"G01:{scene_id}",
                "speaker": turn["speaker"],
                "text": turn["text"],
                "event_key": key,
            })
        keys = [row["event_key"] for row in rows if row["session_ref"] == f"G01:{scene_id}" and row["event_key"]]
        if keys != graph["chronology"][scene_id]:
            raise ValueError(f"event_key_order_invalid_{scene_id}")
    return {
        "root_id": graph["continuation"]["root_id"],
        "source_graph_sha256": sha_file(GRAPH),
        "source_r03_sha256": graph["continuation"]["r03_source_sha256"],
        "turn_count": len(rows),
        "scenes": scenes,
        "turns": rows,
        "status": "SHAPE_ONLY_UNREVIEWED",
        "training_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="print the exact frozen prompt; make no call")
    args = parser.parse_args()
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    verify_sources(graph)
    prompt = prompt_for(graph)
    if args.preview:
        print(prompt)
        return
    profile = provider_profile()
    profile["profile_id"] = "h4_graph_guided_qa_pilot_v1_realization"
    profile["request_parameters"].update(temperature=0.25, max_tokens=OUTPUT_LIMIT, enable_thinking=False)
    profile["output_limit"] = OUTPUT_LIMIT
    profile["http_retry_limit"] = 0
    ledger = {
        "status": "RESERVED_ONE_ATTEMPT",
        "stage": "dialogue_realization_only",
        "model": profile["model"],
        "profile": profile,
        "graph_sha256": sha_file(GRAPH),
        "instructions_sha256": sha_file(INSTRUCTIONS),
        "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
        "r03_source_sha256": graph["continuation"]["r03_source_sha256"],
        "external_sample_sha256": graph["provenance"]["local_sample_sha256"],
        "maximum_http_attempts": 1,
        "training_promotion": False,
    }
    write_new(HERE / "attempt.reserved.json", ledger)
    write_new(HERE / "prompt.txt", prompt, raw=True)

    def on_attempt(payload: dict) -> None:
        write_new(HERE / "http_attempt.json", {
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
        write_new(HERE / "response_raw.txt", response.raw_text, raw=True)
        write_new(HERE / "response_meta.json", {
            "status": "RESPONSE_RECEIVED",
            "raw_sha256": sha_bytes(response.raw_text.encode("utf-8")),
            "metadata": response.metadata,
            "training_promotion": False,
        })
    except Exception as error:
        write_new(HERE / "error.json", {
            "status": "ERROR_NO_RETRY",
            "error_type": type(error).__name__,
            "transport_metadata": getattr(error, "metadata", None),
            "http_attempt_recorded": (HERE / "http_attempt.json").exists(),
            "training_promotion": False,
        })
        raise
    try:
        write_new(HERE / "realized_dialogue.json", materialize(response.raw_text, graph))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        write_new(HERE / "shape_audit.json", {
            "status": "FAILED_SHAPE_ONLY",
            "error_type": type(error).__name__,
            "reason": str(error),
            "training_promotion": False,
        })
        raise


if __name__ == "__main__":
    main()
