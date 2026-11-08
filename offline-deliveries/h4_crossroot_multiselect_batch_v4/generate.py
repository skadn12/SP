"""Frozen R02-excerpt generation; retain complete roots for subsequent review."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "h4_crossroot_multiselect_batch_v1/generate.py"
BASE_SHA256 = "626752711e39aaef285e4b936048f3eb48ccedab6868c5cb11baeed5dfc26b46"
VIEW_TYPE = "r02_turns_original_order_excerpt_v1"
FIELDS = {
    "candidate_id", "root_id", "question_type", "topic", "question",
    "options", "gold", "option_evidence", "answer_reason",
}


def generation_view(source: dict) -> dict:
    boundaries = source["round_boundaries"]
    latest = source["latest_round_boundary"]
    if (source["round_count"] != 2 or len(boundaries) != 2 or
            latest != boundaries[-1] or latest["round_index"] != 2):
        raise ValueError("r02_boundary_contract_changed")
    turns = source["turns"]
    start, end = latest["first_turn_index_1_based"], latest["last_turn_index_1_based"]
    selected = turns[start - 1:end]
    if (not selected or len(selected) != source["latest_round_child_turn_count"] or
            selected[0]["turn_ref"] != latest["first_turn_ref"] or
            selected[-1]["turn_ref"] != latest["last_turn_ref"] or
            any(not turn["turn_ref"].startswith("R02:") for turn in selected) or
            any(turn["turn_ref"].startswith("R02:") for turn in turns[:start - 1] + turns[end:])):
        raise ValueError("r02_turn_partition_changed")
    return {
        "generation_view_type": VIEW_TYPE,
        "scope_note": "Complete R02 excerpt in original order; parent and R01 omitted; not the complete dialogue.",
        "root_id": source["root_id"],
        "complete_root_turn_count": source["turn_count"],
        "complete_root_round_count": source["round_count"],
        "excerpt_round_index": 2,
        "excerpt_turn_count": len(selected),
        "excerpt_first_turn_index_1_based": start,
        "excerpt_last_turn_index_1_based": end,
        "turns": [
            {"turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "text": turn["text"]}
            for turn in selected
        ],
    }


def runner():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("v1_runner_dependency_changed")
    spec = importlib.util.spec_from_file_location("h4_crossroot_multiselect_batch_v1_runner", BASE)
    if spec is None or spec.loader is None:
        raise ImportError("v1_runner_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.VERSION = HERE.name
    module.__file__ = __file__
    module.OUTPUT_LIMIT = 4096
    original_profile = module.profile
    original_check = module.check_candidate
    original_frozen_inputs = module.frozen_inputs

    def profile() -> dict:
        result = original_profile()
        result["output_limit"] = module.OUTPUT_LIMIT
        result["request_parameters"]["max_tokens"] = module.OUTPUT_LIMIT
        return result

    def prompt_for(root_id: str, source: dict, instructions: str) -> str:
        view = generation_view(source)
        text = instructions.replace("{ROOT_ID}", root_id).replace(
            "{ROOT_SUFFIX}", root_id.rsplit("-", 1)[-1])
        return text + json.dumps(view, ensure_ascii=False, indent=2) + "\n"

    def check_candidate(row: object, root_id: str, turns: list[dict]) -> list[str]:
        errors = original_check(row, root_id, turns)
        if not isinstance(row, dict):
            return errors
        if row.get("candidate_id") != f"{root_id.rsplit('-', 1)[-1]}-qwen36-multi-01":
            errors.append("v4_candidate_id_invalid")
        if set(row) != FIELDS:
            errors.append("candidate_schema_fields_mismatch")
        if isinstance(row.get("question"), str) and "r02" not in row["question"].lower():
            errors.append("question_not_r02_scoped")
        evidence = row.get("option_evidence")
        if isinstance(evidence, dict):
            for label, item in evidence.items():
                if label not in module.LABELS or not isinstance(item, dict):
                    continue
                if set(item) != {"turn_ref", "quote", "relation", "reason"}:
                    errors.append(f"{label}:evidence_schema_fields_mismatch")
                if not isinstance(item.get("turn_ref"), str) or not item["turn_ref"].startswith("R02:"):
                    errors.append(f"{label}:evidence_not_r02")
                quote = item.get("quote")
                if isinstance(quote, str) and len(quote) > 240:
                    errors.append(f"{label}:quote_too_long")
        return errors

    def prepare() -> None:
        if (HERE / "BATCH_PREPARED.json").exists():
            raise FileExistsError("batch_already_prepared_no_overwrite")
        rows = module.ordering.source_manifest()
        instructions = (HERE / "instructions.txt").read_bytes()
        planned = []
        for root_id in module.ROOTS:
            if (HERE / root_id).exists():
                raise FileExistsError(f"root_directory_exists_no_overwrite:{root_id}")
            raw, source = module.ordering.checked_source(root_id, rows[root_id], frozen=False)
            view = generation_view(source)
            prompt = prompt_for(root_id, source, instructions.decode("utf-8"))
            payload = module.ordering.build_payload(profile(), prompt)
            planned.append((root_id, rows[root_id], raw, source, view, prompt, payload))
        frozen = []
        for root_id, row, raw, source, view, prompt, payload in planned:
            directory = HERE / root_id
            directory.mkdir()
            module.write_new(directory / "source.json", raw.decode("utf-8"), raw=True)
            module.write_new(directory / "generation_view.json", view)
            module.write_new(directory / "prompt.txt", prompt, raw=True)
            module.write_new(directory / "request.json", payload)
            frozen.append({
                "root_id": root_id, "source_path": row["source_path"],
                "source_sha256": module.sha(raw), "source_turns": len(source["turns"]),
                "source_round_count": source["round_count"], "partial_promotion": False,
                "generation_view_type": VIEW_TYPE,
                "generation_view_sha256": module.sha((directory / "generation_view.json").read_bytes()),
                "generation_view_turns": view["excerpt_turn_count"],
                "prompt_sha256": module.sha(prompt.encode("utf-8")),
                "payload_sha256": module.sha(module.canonical(payload)),
                "requested_model": module.MODEL, "maximum_candidates": 1,
                "maximum_http_attempts": 1, "status": "PREPARED_NO_API_CALL",
            })
        module.write_new(HERE / "BATCH_PREPARED.json", {
            "version": module.VERSION, "requested_model": module.MODEL,
            "profile": profile(), "generation_view_type": VIEW_TYPE,
            "source_manifest_path": str(module.SOURCE_MANIFEST),
            "source_manifest_sha256": module.sha(module.SOURCE_MANIFEST.read_bytes()),
            "instructions_sha256": module.sha(instructions),
            "script_sha256": module.sha(Path(__file__).read_bytes()),
            "roots": frozen, "generator_calls": 0, "solver_calls": 0,
            "training_started": False,
        })
        print(json.dumps({"status": "PREPARED_NO_API_CALL", "roots": len(frozen),
                          "model": module.MODEL, "maximum_candidates_per_root": 1}))

    def frozen_inputs(root_id: str) -> tuple[dict, dict, str, dict]:
        entry, source, prompt, request = original_frozen_inputs(root_id)
        batch = module.read_json(HERE / "BATCH_PREPARED.json")
        view_path = HERE / root_id / "generation_view.json"
        view_bytes = view_path.read_bytes()
        if (batch.get("generation_view_type") != VIEW_TYPE or
                entry.get("generation_view_type") != VIEW_TYPE or
                entry.get("generation_view_sha256") != module.sha(view_bytes) or
                module.read_json(view_path) != generation_view(source) or
                entry.get("generation_view_turns") != len(generation_view(source)["turns"])):
            raise ValueError("generation_view_freeze_changed")
        return entry, source, prompt, request

    def validate(root_id: str) -> dict:
        entry, source, _, _ = frozen_inputs(root_id)
        directory = HERE / root_id
        response = module.read_json(directory / "RESPONSE.json")
        if (response.get("source_sha256") != entry["source_sha256"] or
                response.get("payload_sha256") != entry["payload_sha256"]):
            raise ValueError("response_binding_changed")
        raw_text, metadata = response["raw_text"], response["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("response_metadata_invalid")
        try:
            parsed = json.loads(raw_text)
        except (TypeError, ValueError):
            parsed = None
        questions = parsed.get("questions") if isinstance(parsed, dict) else None
        abstentions = parsed.get("abstentions") if isinstance(parsed, dict) else None
        global_errors = []
        if not isinstance(parsed, dict) or set(parsed) != {"questions", "abstentions"}:
            global_errors.append("top_level_schema_mismatch")
        if not isinstance(questions, list):
            global_errors.append("questions_array_missing")
            questions = []
        if len(questions) > 1:
            global_errors.append("candidate_count_exceeds_one")
        if not isinstance(abstentions, list) or any(not isinstance(v, str) for v in abstentions):
            global_errors.append("abstentions_array_invalid")
        elif not questions and not abstentions:
            global_errors.append("missing_abstention_reason")
        if metadata.get("returned_model") != module.MODEL:
            global_errors.append("returned_model_mismatch")
        if metadata.get("finish_reason") == "length":
            global_errors.append("response_truncated")
        checks, valid = [], []
        for index, question in enumerate(questions, 1):
            errors = check_candidate(question, root_id, source["turns"])
            candidate_id = question.get("candidate_id") if isinstance(question, dict) else None
            checks.append({"index": index, "candidate_id": candidate_id, "errors": errors})
            if not errors:
                valid.append(question)
        if global_errors:
            valid = []
        provenance = {
            "author": "REMOTE_LARGE_MODEL", "provider": "siliconflow",
            "requested_model": module.MODEL,
            "returned_model": metadata.get("returned_model"),
            "response_id": (metadata.get("raw_response") or {}).get("id"),
            "root_id": root_id, "source_sha256": entry["source_sha256"],
            "generation_view_type": VIEW_TYPE,
            "generation_view_sha256": entry["generation_view_sha256"],
            "prompt_sha256": entry["prompt_sha256"],
            "payload_sha256": entry["payload_sha256"],
            "model_questions_modified_by_codex": False,
            "solver_evaluated": False, "selected_for_training": False,
            "training_started": False,
        }
        status = ("MECHANICAL_REJECTED" if global_errors or (questions and not valid) else
                  "MECHANICAL_CHECKS_ONLY_CONTENT_AUDIT_PENDING" if questions else
                  "NO_CANDIDATE_ABSTAINED")
        result = {
            "status": status, "root_id": root_id,
            "candidate_count": len(questions), "mechanically_valid_count": len(valid),
            "global_errors": global_errors, "checks": checks,
            "semantic_gold_verified": False, "provenance": provenance,
        }
        module.write_new(directory / "PARSED_RESPONSE.json", parsed)
        module.write_new(directory / "VALIDATED_CANDIDATES.json", {
            "status": status, "provenance": provenance, "questions": valid,
        })
        module.write_new(directory / "VALIDATION.json", result)
        return result

    module.profile = profile
    module.prompt_for = prompt_for
    module.check_candidate = check_candidate
    module.prepare = prepare
    module.frozen_inputs = frozen_inputs
    module.validate = validate
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--execute", choices=(
        "cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852"))
    actions.add_argument("--validate", choices=(
        "cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852"))
    actions.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    batch = runner()
    if args.preflight:
        batch.preflight()
    elif args.execute:
        batch.execute(args.execute)
    elif args.validate:
        print(json.dumps(batch.validate(args.validate), ensure_ascii=False))
    elif args.summary:
        batch.summary()
    else:
        batch.prepare()
