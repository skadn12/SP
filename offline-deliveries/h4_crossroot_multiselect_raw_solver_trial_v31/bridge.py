"""Bind frozen v28 Qwen drafts to a raw, mechanically checked Solver pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BATCHES = HERE.parent / "h4_crossroot_multiselect_batch_v28_newroots_quota4to8"
ROOTS = ("cc-train-episode-210046", "cc-train-episode-231899",
         "cc-train-episode-194259", "cc-train-episode-96544")
LABELS = "ABCDEF"
MODEL = "Qwen/Qwen3.6-35B-A3B"
GATE = "raw_source_and_gold_v1"
FROZEN = {
    "cc-train-episode-231899": {
        "source": "269ae15f0f1a8a3e898fd14d2ca669e2e19693b3c0900ab6f195501d26da18d0",
        "request": "ddcac0774e7cee72bc2753938f5d681b759b4e67fcca4504cb9b66cb60d1ab6b",
        "response": "bc604a6be8fcfd607378a33ed2ce2c38b100e0e33606ac778dc732422cc499a3",
        "draft": "4a894bf695c3791c86a9832eb021f3e8b0aa3490f896737393554b374aa45e3e",
    },
    "cc-train-episode-194259": {
        "source": "df30b80f5ed1ba81385f3a2bf52fbce6a30279b7d95cc05cde2ec03772f149f0",
        "request": "71998a07929afd1e3e4bad3d8f444fdf66230a352ac92e702ac0d512a3bc7da4",
        "response": "b99009e828d21b870dc55a48b6b1d9ad38231ca7e523e8e1348717717ad11583",
        "draft": "65311cd93d92d10ed31e149f2ec15c997f2a71bc366b2dcdd80f3d50872cd4f2",
    },
    "cc-train-episode-210046": {
        "source": "785dc3b56a5ae9ba1be0e0ee31538c87ed6bad633e1ff97d44aec824a4a7b3b2",
        "request": "787ce35da0f5750900ef69976aa37a9a2d3e949036c94d13ab49670f41d01d9f",
        "response": "b9a995700f8733acf4fd8e1a41b80be2bc49f7e90ae858126091de37eb42112b",
        "draft": "5dfaafbb4faa2a32addf87a0cf48217232de95ebc4ff3a6a86af453847f324e5",
    },
    "cc-train-episode-96544": {
        "source": "618e772bbc8fa8a53bd29d88d88992edcc3af76adbf9cb8afea5a666aa004a15",
        "request": "160da74df5daecbd866377cfdc49ba7770f4d5f5b6c1eb76617aed316225993f",
        "response": "3ef9c6b7d4136cb087c9c4d2e633af4a5714f3188b8081c44f9ab09de0be4b05",
        "draft": "57b1e4333f3f5b60fbcb8b925030246508abb6ed5b55138dc623da546187d17f",
    },
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def valid_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_candidate(row: object, expected_id: str, turns: dict,
                       root: str, source_path: Path, source_sha: str) -> tuple[list[str], dict | None]:
    errors = []
    if not isinstance(row, dict) or set(row) != {"candidate_id", "topic", "question", "options"}:
        return ["question_schema_invalid"], None
    if row["candidate_id"] != expected_id:
        errors.append("candidate_id_invalid")
    if not valid_text(row["topic"]):
        errors.append("topic_missing")
    if not valid_text(row["question"]):
        errors.append("question_missing")
    options = row["options"]
    if not isinstance(options, list) or len(options) != 6:
        return errors + ["six_options_required"], None
    propositions = []
    for index, option in enumerate(options, 1):
        prefix = f"option_{index}"
        fields = {"proposition", "relation", "turn_ref", "speaker", "quote", "rationale"}
        if not isinstance(option, dict) or set(option) != fields:
            errors.append(f"{prefix}_schema_invalid")
            continue
        for field in ("proposition", "turn_ref", "speaker", "quote", "rationale"):
            if not valid_text(option[field]):
                errors.append(f"{prefix}_{field}_missing")
        if valid_text(option["proposition"]):
            propositions.append(" ".join(option["proposition"].casefold().split()))
        if option["relation"] not in ("supports", "contradicts"):
            errors.append(f"{prefix}_relation_invalid")
        ref = option["turn_ref"]
        turn = turns.get(ref) if isinstance(ref, str) else None
        if turn is None:
            errors.append(f"{prefix}_turn_ref_unavailable")
        else:
            if option["speaker"] != turn["speaker"]:
                errors.append(f"{prefix}_speaker_mismatch")
            if not valid_text(option["quote"]) or option["quote"] not in turn["text"]:
                errors.append(f"{prefix}_quote_not_verbatim")
    if len(propositions) == 6 and len(set(propositions)) != 6:
        errors.append("options_not_unique")
    gold = [label for label, option in zip(LABELS, options)
            if isinstance(option, dict) and option.get("relation") == "supports"]
    if not 2 <= len(gold) <= 6:
        errors.append("gold_cardinality_invalid")
    if errors:
        return errors, None
    evidence = {label: {"turn_ref": option["turn_ref"], "quote": option["quote"],
                        "relation": option["relation"], "reason": option["rationale"]}
                for label, option in zip(LABELS, options)}
    question = {"candidate_id": expected_id, "root_id": root,
                "question_type": "multi_select", "topic": row["topic"],
                "question": row["question"],
                "options": {label: option["proposition"] for label, option in zip(LABELS, options)},
                "gold": gold, "option_evidence": evidence,
                "answer_reason": "Gold labels are the Qwen draft's supports relations; semantic validity has not been reviewed.",
                "source_path": str(source_path), "source_sha256": source_sha}
    return [], question


def load_batch(root: str, base: Path = BATCHES) -> tuple[list[dict], list[dict], dict]:
    directory = base / root / "batch-01"
    paths = {name: directory / name for name in
             ("PREPARED.json", "PREPARED_INPUT.json", "REQUEST.json", "MODEL_RESPONSE.json",
              "MODEL_DRAFT.json", "source.json")}
    prepared, input_data, request, response, draft, source = (read(paths[name]) for name in paths)
    require(prepared.get("root_id") == root and input_data.get("root_id") == root
            and source.get("root_id") == root and draft.get("root_id") == root,
            "root_binding_invalid:" + root)
    require(prepared.get("requested_model") == MODEL and request.get("model") == MODEL
            and prepared.get("requested_question_slots") == 4
            and prepared.get("batch_number") == 1,
            "generator_metadata_invalid:" + root)
    require(prepared.get("request_sha256") == canonical_sha(request)
            and prepared.get("input_sha256") == canonical_sha(input_data)
            and prepared["dependencies"]["source_sha256"] == sha(paths["source.json"])
            and response.get("request_sha256") == prepared["request_sha256"],
            "frozen_request_or_source_binding_invalid:" + root)
    frozen = FROZEN[root]
    require(sha(paths["source.json"]) == frozen["source"]
            and canonical_sha(request) == frozen["request"]
            and canonical_sha(response) == frozen["response"]
            and canonical_sha(draft) == frozen["draft"],
            "v28_frozen_artifact_hash_mismatch:" + root)
    metadata = response.get("metadata")
    require(isinstance(metadata, dict)
            and metadata.get("requested_model", MODEL) == MODEL
            and metadata.get("returned_model") == MODEL
            and metadata.get("finish_reason") != "length",
            "returned_model_or_finish_invalid:" + root)
    require(isinstance(response.get("raw_text"), str)
            and json.loads(response["raw_text"]) == draft,
            "draft_differs_from_model_response:" + root)
    require(set(draft) == {"root_id", "questions", "abstentions"}
            and isinstance(draft["questions"], list)
            and isinstance(draft["abstentions"], list),
            "draft_schema_invalid:" + root)
    ids = prepared.get("candidate_ids")
    require(isinstance(ids, list) and len(ids) == 4 and len(set(ids)) == 4
            and all(isinstance(qid, str) and qid for qid in ids)
            and ids == input_data.get("candidate_ids"),
            "candidate_ids_invalid:" + root)
    abstentions = draft["abstentions"]
    require(all(isinstance(item, dict) and set(item) == {"candidate_id", "reason"}
                and item["candidate_id"] in ids and valid_text(item["reason"])
                for item in abstentions), "abstentions_invalid:" + root)
    abstained = [item["candidate_id"] for item in abstentions]
    rows = draft["questions"]
    question_ids = [row.get("candidate_id") if isinstance(row, dict) else None for row in rows]
    require(len(rows) + len(abstained) == 4
            and len(set(abstained)) == len(abstained)
            and len(set(question_ids)) == len(question_ids)
            and set(question_ids).isdisjoint(abstained)
            and set(question_ids) | set(abstained) == set(ids),
            "candidate_slots_invalid:" + root)
    turns_list = source.get("turns")
    require(isinstance(turns_list, list) and turns_list
            and all(isinstance(turn, dict) and valid_text(turn.get("turn_ref"))
                    and valid_text(turn.get("speaker")) and valid_text(turn.get("text"))
                    for turn in turns_list), "source_turns_invalid:" + root)
    turns = {turn["turn_ref"]: turn for turn in turns_list}
    require(len(turns) == len(turns_list), "source_turn_refs_duplicate:" + root)
    by_id = {row["candidate_id"]: row for row in rows}
    accepted, verdicts = [], []
    source_path = paths["source.json"].resolve(strict=True)
    source_hash = sha(source_path)
    for qid in ids:
        if qid in abstained:
            verdicts.append({"candidate_id": qid, "status": "GENERATOR_ABSTAINED",
                             "errors": [], "reason": next(x["reason"] for x in abstentions if x["candidate_id"] == qid)})
            continue
        errors, question = validate_candidate(by_id[qid], qid, turns, root, source_path, source_hash)
        verdicts.append({"candidate_id": qid,
                         "status": "MECHANICAL_PASS" if not errors else "MECHANICAL_REJECT",
                         "errors": errors})
        if question is not None:
            accepted.append(question)
    bindings = {"root_id": root, "source_sha256": source_hash,
                "prepared_sha256": sha(paths["PREPARED.json"]),
                "prepared_input_sha256": sha(paths["PREPARED_INPUT.json"]),
                "request_sha256": sha(paths["REQUEST.json"]),
                "model_response_sha256": sha(paths["MODEL_RESPONSE.json"]),
                "model_draft_sha256": sha(paths["MODEL_DRAFT.json"])}
    return accepted, verdicts, bindings


def build(base: Path = BATCHES) -> tuple[dict, dict]:
    questions, verdicts, bindings = [], [], []
    for root in ROOTS:
        accepted, rows, binding = load_batch(root, base)
        questions.extend(accepted)
        verdicts.extend({"root_id": root, **row} for row in rows)
        bindings.append(binding)
    require(len(verdicts) == 16 and len({q["candidate_id"] for q in questions}) == len(questions),
            "batch_size_or_candidate_collision")
    pool = {"version": "h4_crossroot_multiselect_raw_solver_trial_v31",
            "status": "RAW_MECHANICALLY_VERIFIED_READY_FOR_SOLVER" if questions else "NO_MECHANICAL_CANDIDATES",
            "candidate_count": len(questions),
            "provenance": {"requested_model": MODEL, "returned_model": MODEL,
                           "semantic_review_performed": False, "codex_question_edits": False,
                           "mechanical_gate_version": GATE,
                           "option_order": "original_qwen_draft_order",
                           "input_bindings": bindings},
            "questions": questions, "solver_calls": 0, "training_started": False}
    report = {"version": pool["version"], "mechanical_gate_version": GATE,
              "draft_count": len(verdicts), "mechanical_pass_count": len(questions),
              "mechanical_reject_count": sum(row["status"] == "MECHANICAL_REJECT" for row in verdicts),
              "generator_abstained_count": sum(row["status"] == "GENERATOR_ABSTAINED" for row in verdicts),
              "semantic_review_performed": False, "solver_calls": 0,
              "verdicts": verdicts}
    return pool, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    pool, report = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("POOL.json", pool), ("REPORT.json", report)):
        with (args.output_dir / name).open("x", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps({"draft_count": report["draft_count"],
                      "mechanical_pass_count": report["mechanical_pass_count"],
                      "mechanical_reject_count": report["mechanical_reject_count"]}))


if __name__ == "__main__":
    main()
