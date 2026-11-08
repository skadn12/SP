"""One reserved Writer attempt over the frozen R03 and new G02 dialogue surface."""

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

HISTORY = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json")
HISTORY_SHA256 = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"
SOURCE = HERE / "realized_dialogue.json"
INSTRUCTIONS = HERE / "writer_instructions.txt"
OUTPUT_LIMIT = 6144


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


def projected_turn(row: object) -> dict:
    if not isinstance(row, dict):
        raise ValueError("turn_not_object")
    values = [row.get(key) for key in ("turn_ref", "speaker", "text")]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("turn_surface_invalid")
    return dict(zip(("turn_ref", "speaker", "text"), values))


def dialogue_surface() -> tuple[dict, str]:
    if sha_file(HISTORY) != HISTORY_SHA256:
        raise RuntimeError("historical_source_hash_mismatch")
    historical = json.loads(HISTORY.read_text(encoding="utf-8"))
    history = [projected_turn(row) for row in historical.get("turns", [])]
    if len(history) != 188 or historical.get("turn_count") != 188:
        raise ValueError("historical_turn_count_invalid")
    source_sha = sha_file(SOURCE)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("root_id") != historical.get("root_id"):
        raise ValueError("root_id_mismatch")
    if source.get("source_r03_sha256") != HISTORY_SHA256:
        raise ValueError("new_dialogue_history_hash_mismatch")
    scenes = source.get("scenes")
    flat = source.get("turns")
    if not isinstance(scenes, list) or not scenes or not isinstance(flat, list):
        raise ValueError("new_dialogue_shape_invalid")
    projected = []
    new_scenes = []
    for scene in scenes:
        scene_id, turns = scene.get("scene_id"), scene.get("turns")
        if not isinstance(scene_id, str) or not scene_id.strip() or not isinstance(turns, list) or not turns:
            raise ValueError("new_scene_invalid")
        visible = []
        for number, row in enumerate(turns, 1):
            turn = {"turn_ref": f"G02:{scene_id}:T{number:03d}", "speaker": row.get("speaker"), "text": row.get("text")}
            turn = projected_turn(turn)
            visible.append(turn)
            projected.append(turn)
        new_scenes.append({"scene_id": scene_id, "turns": visible})
    if projected != [projected_turn(row) for row in flat]:
        raise ValueError("flat_turns_differ_from_scenes")
    refs = [row["turn_ref"] for row in history + projected]
    if len(refs) != len(set(refs)):
        raise ValueError("duplicate_turn_refs")
    return {"historical_turns": history, "new_scenes": new_scenes}, source_sha


def check_refs(refs: object, valid_refs: set[str], *, new_only: bool = False) -> list[str]:
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in valid_refs for ref in refs):
        raise ValueError("evidence_refs_invalid")
    if new_only and any(not ref.startswith("G02:") for ref in refs):
        raise ValueError("ordering_requires_new_dialogue_refs")
    return refs


def check_quote(row: dict, ref_map: dict[str, str], refs: list[str]) -> None:
    quote = row.get("quote")
    if not isinstance(quote, str) or not quote.strip() or not any(quote in ref_map[ref] for ref in refs):
        raise ValueError("quote_not_found_in_cited_speech")
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ValueError("evidence_reason_missing")


def scene_of(ref: str) -> str:
    return ref.rsplit(":T", 1)[0]


def shape_audit(raw_text: str, surface: dict, source_sha: str) -> dict:
    result = json.loads(raw_text)
    if not isinstance(result, dict) or not isinstance(result.get("questions"), list):
        raise ValueError("questions_array_missing")
    history = surface["historical_turns"]
    new = [turn for scene in surface["new_scenes"] for turn in scene["turns"]]
    ref_map = {row["turn_ref"]: row["text"] for row in history + new}
    valid_refs = set(ref_map)
    questions = result["questions"]
    if len(questions) > 2 or any(not isinstance(row, dict) for row in questions):
        raise ValueError("questions_shape_invalid")
    kinds = [row.get("question_type") for row in questions]
    if len(set(kinds)) != len(kinds) or any(kind not in {"multi_select", "ordering"} for kind in kinds):
        raise ValueError("question_type_or_count_invalid")
    ids = [row.get("candidate_id") for row in questions]
    if any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate_ids_invalid")
    for question in questions:
        if any(not isinstance(question.get(key), str) or not question[key].strip() for key in ("intent", "question")):
            raise ValueError("intent_or_question_missing")
        kind = question["question_type"]
        field = "options" if kind == "multi_select" else "items"
        labels = list("ABCDE") if kind == "multi_select" else [f"I{i}" for i in range(1, 5)]
        values = question.get(field)
        if not isinstance(values, dict) or list(values) != labels or any(not isinstance(values[key], str) or not values[key].strip() for key in labels):
            raise ValueError(f"{field}_shape_invalid")
        gold = question.get("gold")
        if not isinstance(gold, list) or len(set(gold)) != len(gold) or any(key not in labels for key in gold):
            raise ValueError("gold_shape_invalid")
        if kind == "multi_select":
            if not 2 <= len(gold) <= 3:
                raise ValueError("multi_select_gold_size_invalid")
            evidence = question.get("option_evidence")
            if not isinstance(evidence, dict) or set(evidence) != set(labels):
                raise ValueError("option_evidence_incomplete")
            for label in labels:
                row = evidence[label]
                if not isinstance(row, dict) or row.get("verdict") != ("TRUE" if label in gold else "FALSE"):
                    raise ValueError("option_verdict_mismatch")
                clauses = row.get("clauses")
                if not isinstance(clauses, list) or len(clauses) != 2 or any(not isinstance(clause, dict) for clause in clauses):
                    raise ValueError("two_clauses_required")
                clause_verdicts = []
                cited = []
                for clause in clauses:
                    if not isinstance(clause.get("claim"), str) or not clause["claim"].strip():
                        raise ValueError("clause_claim_missing")
                    verdict = clause.get("verdict")
                    if verdict not in {"TRUE", "FALSE"}:
                        raise ValueError("clause_verdict_invalid")
                    clause_verdicts.append(verdict)
                    refs = check_refs(clause.get("refs"), valid_refs)
                    cited.extend(refs)
                    check_quote(clause, ref_map, refs)
                if len(set(cited)) < 2:
                    raise ValueError("single_turn_option")
                if ("TRUE" if all(v == "TRUE" for v in clause_verdicts) else "FALSE") != row["verdict"]:
                    raise ValueError("clause_option_verdict_mismatch")
                if label in gold:
                    first_scenes = {scene_of(ref) for ref in clauses[0]["refs"]}
                    second_scenes = {scene_of(ref) for ref in clauses[1]["refs"]}
                    crossed = any(not ref.startswith("G02:") for ref in clauses[0]["refs"]) and any(ref.startswith("G02:") for ref in clauses[1]["refs"])
                    crossed |= any(ref.startswith("G02:") for ref in clauses[0]["refs"]) and any(not ref.startswith("G02:") for ref in clauses[1]["refs"])
                    crossed |= any(a != b and a.startswith("G02:") and b.startswith("G02:") for a in first_scenes for b in second_scenes)
                    if not crossed:
                        raise ValueError("true_option_not_cross_scene_or_history")
        else:
            if len(gold) != 4:
                raise ValueError("ordering_gold_incomplete")
            evidence = question.get("item_evidence")
            if not isinstance(evidence, dict) or set(evidence) != set(labels):
                raise ValueError("item_evidence_incomplete")
            tracks = []
            item_refs = {}
            for label in labels:
                row = evidence[label]
                if not isinstance(row, dict) or not isinstance(row.get("track"), str) or not row["track"].strip():
                    raise ValueError("item_track_missing")
                tracks.append(row["track"])
                refs = check_refs(row.get("refs"), valid_refs, new_only=True)
                if len(refs) != 1:
                    raise ValueError("ordering_item_must_cite_one_action_turn")
                item_refs[label] = refs
                check_quote(row, ref_map, refs)
            gold_tracks = [evidence[label]["track"] for label in gold]
            if len(set(tracks)) != 2 or sum(a != b for a, b in zip(gold_tracks, gold_tracks[1:])) < 2:
                raise ValueError("ordering_tracks_not_interleaved")
            gold_scenes = {scene_of(ref) for label in gold for ref in item_refs[label]}
            if len(gold_scenes) < 3:
                raise ValueError("ordering_does_not_span_three_scenes")
            if not any(
                scene_of(item_refs[a][0]) == scene_of(item_refs[b][0]) and evidence[a]["track"] != evidence[b]["track"]
                for a, b in zip(gold, gold[1:])
            ):
                raise ValueError("ordering_lacks_same_scene_cross_track_edge")
            if len({ref for refs in item_refs.values() for ref in refs}) < 4:
                raise ValueError("ordering_actions_not_separate_turns")
            new_positions = {row["turn_ref"]: index for index, row in enumerate(new)}
            if [new_positions[item_refs[label][0]] for label in gold] != sorted(new_positions[item_refs[label][0]] for label in gold):
                raise ValueError("ordering_gold_conflicts_with_turn_sequence")
            edges = question.get("adjacent_edges")
            if not isinstance(edges, list) or len(edges) != 3:
                raise ValueError("adjacent_edges_incomplete")
            for index, edge in enumerate(edges):
                if not isinstance(edge, dict) or (edge.get("before"), edge.get("after")) != (gold[index], gold[index + 1]):
                    raise ValueError("adjacent_edge_order_invalid")
                before = check_refs(edge.get("before_refs"), valid_refs, new_only=True)
                after = check_refs(edge.get("after_refs"), valid_refs, new_only=True)
                check_refs(edge.get("order_refs"), valid_refs, new_only=True)
                if not set(before) & set(item_refs[gold[index]]) or not set(after) & set(item_refs[gold[index + 1]]):
                    raise ValueError("edge_not_grounded_in_item_turns")
                if set(before) & set(after) or not isinstance(edge.get("reason"), str) or not edge["reason"].strip():
                    raise ValueError("adjacent_edge_evidence_invalid")
    abstentions = result.get("abstentions")
    if not isinstance(abstentions, list):
        raise ValueError("abstentions_array_missing")
    for row in abstentions:
        if not isinstance(row, dict) or row.get("question_type") not in {"multi_select", "ordering"} or not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("abstention_invalid")
    if {row["question_type"] for row in abstentions} != {"multi_select", "ordering"} - set(kinds):
        raise ValueError("abstentions_must_cover_omitted_types_only")
    return {"status": "PASS_SHAPE_ONLY_NOT_GOLD", "question_count": len(questions), "question_types": kinds,
            "history_sha256": HISTORY_SHA256, "new_dialogue_sha256": source_sha, "training_promotion": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="print the exact prompt without reserving an API attempt")
    args = parser.parse_args()
    surface, source_sha = dialogue_surface()
    prompt = INSTRUCTIONS.read_text(encoding="utf-8") + json.dumps(surface, ensure_ascii=False, indent=2) + "\n"
    if args.preview:
        print(prompt)
        return
    profile = provider_profile()
    if profile["model"] != MODEL or MODEL != "Qwen/Qwen3.6-35B-A3B":
        raise RuntimeError("unexpected_writer_model")
    profile["profile_id"] = "h4_graph_guided_qa_pilot_v2_writer"
    profile["request_parameters"].update(temperature=0, max_tokens=OUTPUT_LIMIT, enable_thinking=False)
    profile["output_limit"] = OUTPUT_LIMIT
    profile["http_retry_limit"] = 0
    write_new(HERE / "writer_attempt.reserved.json", {
        "status": "RESERVED_ONE_ATTEMPT", "stage": "candidate_writer_only",
        "historical_source_path": str(HISTORY), "historical_source_sha256": HISTORY_SHA256,
        "new_source_path": str(SOURCE), "new_source_sha256": source_sha,
        "instructions_sha256": sha_file(INSTRUCTIONS), "prompt_sha256": sha_bytes(prompt.encode("utf-8")),
        "profile": profile, "maximum_http_attempts": 1, "training_promotion": False,
    })
    write_new(HERE / "writer_prompt.txt", prompt, raw=True)

    def on_attempt(payload: dict) -> None:
        write_new(HERE / "writer_http_attempt.json", {
            "attempt_number": 1, "payload_sha256": sha_bytes(canonical(payload)),
            "prompt_sha256": sha_bytes(payload["messages"][0]["content"].encode("utf-8")),
            "model": payload.get("model"), "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"), "enable_thinking": payload.get("enable_thinking"),
        })

    try:
        response = credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=OUTPUT_LIMIT, json_object=True)
        write_new(HERE / "writer_response_raw.txt", response.raw_text, raw=True)
        write_new(HERE / "writer_response_meta.json", {
            "status": "RESPONSE_RECEIVED", "raw_sha256": sha_bytes(response.raw_text.encode("utf-8")),
            "metadata": response.metadata, "training_promotion": False,
        })
    except Exception as error:
        write_new(HERE / "writer_error.json", {
            "status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
            "transport_metadata": getattr(error, "metadata", None),
            "http_attempt_recorded": (HERE / "writer_http_attempt.json").exists(), "training_promotion": False,
        })
        raise
    try:
        audit = shape_audit(response.raw_text, surface, source_sha)
        write_new(HERE / "writer_shape_audit.json", audit)
        parsed = json.loads(response.raw_text)
        write_new(HERE / "writer_candidates.json", {
            "status": "UNREVIEWED_CANDIDATES_NOT_GOLD", "historical_source_sha256": HISTORY_SHA256,
            "new_source_sha256": source_sha, "questions": parsed["questions"],
            "abstentions": parsed["abstentions"], "training_promotion": False,
        })
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        write_new(HERE / "writer_shape_audit.json", {
            "status": "FAILED_SHAPE_ONLY", "error_type": type(error).__name__,
            "reason": str(error), "training_promotion": False,
        })
        raise


if __name__ == "__main__":
    main()
