"""Run one reserved SiliconFlow realization of the frozen three-scene event graph."""

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
    MODEL,
    credential_client,
    provider_profile,
)

GRAPH = HERE / "graph.json"
INSTRUCTIONS = HERE / "realization_instructions.txt"
OUTPUT_LIMIT = 5000
EXPECTED_SCENES = ("S01", "S02", "S03")


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


def verify_graph(graph: dict) -> None:
    chronology = graph["chronology"]
    if not isinstance(chronology, dict) or tuple(chronology) != EXPECTED_SCENES:
        raise ValueError("graph_scene_order_invalid")
    events = graph["events"]
    event_keys = [event["key"] for event in events]
    ordered_keys = [key for scene in EXPECTED_SCENES for key in chronology[scene]]
    if not all(isinstance(keys, list) and keys for keys in chronology.values()):
        raise ValueError("graph_scene_events_missing")
    if len(set(event_keys)) != len(event_keys) or event_keys != ordered_keys:
        raise ValueError("graph_event_order_or_uniqueness_invalid")
    for scene in EXPECTED_SCENES:
        if any(event["scene"] != scene for event in events if event["key"] in chronology[scene]):
            raise ValueError(f"graph_event_scene_invalid_{scene}")


def without_source_links(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: without_source_links(item)
            for key, item in value.items()
            if not key.startswith("source_") and key not in {"provenance", "source_relation", "origin"}
        }
    if isinstance(value, list):
        return [without_source_links(item) for item in value]
    return value


def prompt_for(graph: dict) -> str:
    specification = {
        "navigation_only": graph["continuation"]["navigation_only"],
        "transformed_domain": without_source_links(graph["transformed_domain"]),
        "events": without_source_links(graph["events"]),
        "chronology": graph["chronology"],
        "scene_transition": graph["scene_transition"],
        "evidence_contract": graph["evidence_contract"],
    }
    prompt = INSTRUCTIONS.read_text(encoding="utf-8") + json.dumps(specification, ensure_ascii=False, indent=2) + "\n"
    forbidden = [graph["provenance"]["local_sample_path"], graph["continuation"]["r03_source_path"]]
    forbidden.extend(source["dialogue_id"] for source in graph["provenance"]["source_dialogues"])
    if any(identifier in prompt for identifier in forbidden):
        raise ValueError("source_identifier_leaked_into_prompt")
    return prompt


def materialize(raw_text: str, graph: dict) -> dict:
    parsed = json.loads(raw_text)
    scenes = parsed["scenes"]
    if not isinstance(scenes, list) or [s.get("scene_id") for s in scenes] != list(EXPECTED_SCENES):
        raise ValueError("scene_shape_or_order_invalid")
    actors = {event["key"]: event["actor"] for event in graph["events"]}
    scene_speakers = {
        scene_id: {event["actor"] for event in graph["events"] if event["scene"] == scene_id}
        for scene_id in EXPECTED_SCENES
    }
    rows = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        if not isinstance(scene.get("relative_time"), str) or not scene["relative_time"].strip():
            raise ValueError("relative_time_missing")
        turns = scene.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("turns_missing")
        keys = []
        unkeyed_count = 0
        for index, turn in enumerate(turns, 1):
            if not isinstance(turn, dict) or turn.get("speaker") not in scene_speakers[scene_id]:
                raise ValueError("speaker_invalid")
            if not isinstance(turn.get("text"), str) or not turn["text"].strip():
                raise ValueError("text_invalid")
            key = turn.get("event_key", "")
            if not isinstance(key, str) or (key and actors.get(key) != turn["speaker"]):
                raise ValueError("event_key_or_actor_invalid")
            if key:
                keys.append(key)
            else:
                unkeyed_count += 1
            rows.append({
                "turn_ref": f"G02:{scene_id}:T{index:03d}",
                "session_ref": f"G02:{scene_id}",
                "speaker": turn["speaker"],
                "text": turn["text"],
                "event_key": key,
            })
        if keys != graph["chronology"][scene_id]:
            raise ValueError(f"event_key_order_invalid_{scene_id}")
        if unkeyed_count > 2:
            raise ValueError(f"too_many_unkeyed_turns_{scene_id}")
    return {
        "root_id": graph["continuation"]["root_id"],
        "source_graph_sha256": sha_file(GRAPH),
        "source_r03_sha256": graph["continuation"]["r03_source_sha256"],
        "r03_navigation_only": graph["continuation"]["navigation_only"],
        "turn_count": len(rows),
        "scenes": scenes,
        "turns": rows,
        "status": "SHAPE_ONLY_UNREVIEWED",
        "training_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="print exact frozen prompt; make no call")
    args = parser.parse_args()
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    verify_sources(graph)
    verify_graph(graph)
    prompt = prompt_for(graph)
    if args.preview:
        print(prompt)
        return
    profile = provider_profile()
    if profile["model"] != MODEL or MODEL != "Qwen/Qwen3.6-35B-A3B":
        raise RuntimeError("unexpected_realization_model")
    profile["profile_id"] = "h4_graph_guided_qa_pilot_v2_realization"
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
