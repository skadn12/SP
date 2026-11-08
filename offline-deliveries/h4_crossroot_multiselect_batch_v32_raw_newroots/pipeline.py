"""New-root multi-select trial with four initial slots and an eight-slot cap."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG_PATH = HERE / "batch_config.json"
PROMPT_PATH = HERE / "generation_prompt.txt"
V21_PIPELINE = HERE.parent / "h4_crossroot_multiselect_batch_v21_claim_staged" / "pipeline.py"
V21_PIPELINE_SHA256 = "b354a246dfff105908298707c889e32a77033c97e2ca62d7f997fdf0e75e2c0a"
LABELS = "ABCDEF"
EXPECTED_ROOT_IDS = {
    "cc-train-episode-103866", "cc-train-episode-104814",
    "cc-train-episode-135466", "cc-train-episode-138343",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: object, *, raw: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value if raw else json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_config() -> dict:
    config = read(CONFIG_PATH)
    if not isinstance(config, dict) or config.get("version") != HERE.name:
        raise ValueError("batch_config_invalid")
    if config.get("initial_question_slots_per_root") != 4 or config.get("maximum_question_slots_per_root") != 4:
        raise ValueError("quota_config_changed")
    roots = config.get("root_configs")
    if not isinstance(roots, list):
        raise ValueError("root_configs_invalid")
    ids = [item.get("root_id") for item in roots if isinstance(item, dict)]
    if len(ids) != len(roots) or set(ids) != EXPECTED_ROOT_IDS:
        raise ValueError("root_ids_invalid_or_duplicate")
    split_records(config)
    load_capacity_hints(config)
    return config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def split_records(config: dict) -> dict[str, dict]:
    path = resolve_path(config["split_manifest_path"])
    raw = path.read_bytes()
    if sha(raw) != config["expected_split_manifest_sha256"]:
        raise ValueError("split_manifest_hash_mismatch")
    records = {}
    for line in raw.splitlines():
        row = json.loads(line)
        root_id = row.get("root_id")
        if root_id not in EXPECTED_ROOT_IDS:
            continue
        if root_id in records:
            raise ValueError(f"duplicate_split_record:{root_id}")
        records[root_id] = row
    if set(records) != EXPECTED_ROOT_IDS:
        raise ValueError("split_records_missing")
    for item in config["root_configs"]:
        root_id = item["root_id"]
        row = records[root_id]
        candidate_paths = item["dedup_candidate_paths"]
        if (row.get("split") != "train" or
                row.get("split_version") != config["expected_split_version"] or
                len(candidate_paths) != 1 or
                row.get("candidate_pool_sha256") != item["expected_dedup_sha256"].get(
                    str(resolve_path(candidate_paths[0])))):
            raise ValueError(f"split_or_candidate_pool_binding_invalid:{root_id}")
    return records


def load_capacity_hints(config: dict) -> tuple[dict, bytes]:
    raw = resolve_path(config["capacity_hints_path"]).read_bytes()
    if sha(raw) != config["expected_capacity_hints_sha256"]:
        raise ValueError("capacity_hints_hash_mismatch")
    hints = json.loads(raw)
    if (not isinstance(hints, dict) or set(hints.get("roots", {})) != EXPECTED_ROOT_IDS or
            hints.get("screen_decision") != "APPROVED_FOR_INITIAL_FOUR_SLOT_GENERATION_TRIAL"):
        raise ValueError("capacity_hints_roots_invalid")
    return hints, raw


def root_config(root_id: str, config: dict) -> dict:
    for item in config["root_configs"]:
        if item["root_id"] == root_id:
            return item
    raise ValueError(f"root_not_configured:{root_id}")


def load_source(item: dict) -> tuple[Path, bytes, dict, list[dict]]:
    path = resolve_path(item["source_path"])
    raw = path.read_bytes()
    expected_sha = item["expected_source_sha256"]
    if sha(raw) != expected_sha:
        raise ValueError(f"source_hash_mismatch:{item['root_id']}")
    source = json.loads(raw)
    root_id = item["root_id"]
    if not isinstance(source, dict) or source.get("root_id") != root_id:
        raise ValueError(f"source_root_mismatch:{root_id}")
    turns = source.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"source_turns_missing:{root_id}")
    source_turns = [turn for turn in turns if isinstance(turn, dict)]
    if not source_turns or any(not all(isinstance(turn.get(key), str) and turn[key]
                                       for key in ("turn_ref", "speaker", "text"))
                               for turn in source_turns):
        raise ValueError(f"source_turns_invalid_or_empty:{root_id}")
    refs = [turn["turn_ref"] for turn in source_turns]
    if len(refs) != len(set(refs)):
        raise ValueError(f"source_turn_refs_duplicate:{root_id}")
    if not any(ref.startswith("R01:") for ref in refs):
        raise ValueError(f"r01_appended_turns_missing:{root_id}")
    return path, raw, source, source_turns


def question_rows(value: object) -> list[dict]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict) and isinstance(value.get("questions"), list):
        rows = value["questions"]
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def compact_dedup_row(row: dict, origin: str, review_status: str) -> dict:
    options = row.get("options") or row.get("items") or {}
    if isinstance(options, dict):
        options = list(options.values())
    if not isinstance(options, list):
        options = []
    texts = [option.get("proposition") if isinstance(option, dict) else option
             for option in options]
    evidence = row.get("option_evidence") or row.get("item_evidence") or {}
    refs = set(row.get("evidence_refs") or [])
    if isinstance(evidence, dict):
        refs.update(value.get("turn_ref") for value in evidence.values() if isinstance(value, dict))
    refs.update(option.get("turn_ref") for option in options if isinstance(option, dict))
    evidence_quotes = []
    if isinstance(evidence, dict):
        evidence_quotes.extend({"turn_ref": value.get("turn_ref"), "quote": value.get("quote")}
                               for value in evidence.values() if isinstance(value, dict) and value.get("quote"))
    evidence_quotes.extend({"turn_ref": option.get("turn_ref"), "quote": option.get("quote")}
                           for option in options if isinstance(option, dict) and option.get("quote"))
    return {"candidate_id": row.get("candidate_id"), "origin": origin,
            "review_status": review_status,
            "question_type": row.get("question_type") or ("ordering" if row.get("items") else "multi_select"),
            "topic": row.get("topic"), "question": row.get("question"),
            "fact_texts": [text for text in texts if isinstance(text, str) and text.strip()],
            "evidence_refs": sorted(ref for ref in refs if isinstance(ref, str) and ref),
            "evidence_quotes": evidence_quotes}


def dedup_files(root_id: str, item: dict) -> list[tuple[Path, str, str]]:
    return [(resolve_path(name), "old_candidate_pool_all_types", "candidate_pool_not_selection")
            for name in item["dedup_candidate_paths"]]


def coverage_context(root_id: str, item: dict) -> tuple[dict, dict[str, str]]:
    rows, hashes = [], {}
    if not isinstance(item.get("dedup_candidate_paths"), list):
        raise ValueError(f"dedup_candidate_paths_invalid:{root_id}")
    for path, origin, review_status in dedup_files(root_id, item):
        raw = path.read_bytes()
        actual_sha = sha(raw)
        expected_sha = item["expected_dedup_sha256"][str(path)]
        if actual_sha != expected_sha:
            raise ValueError(f"dedup_candidate_hash_mismatch:{root_id}:{path}")
        hashes[str(path)] = actual_sha
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"candidate_pool_schema_invalid:{root_id}")
        for row in question_rows(data):
            if row.get("root_id") not in (None, root_id):
                raise ValueError(f"dedup_row_root_mismatch:{root_id}:{path}")
            rows.append(compact_dedup_row(row, origin, review_status))
    notes = item.get("covered_fact_notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) or not note.strip() for note in notes):
        raise ValueError(f"covered_fact_notes_invalid:{root_id}")
    return {"prior_candidates_all_types_dedup_only_not_training_selection": rows,
            "covered_fact_notes": notes}, hashes


def candidate_id(root_id: str, ordinal: int) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]+", "", root_id)[-6:]
    return f"{suffix}-qwen36-multi-v32-{ordinal:02d}"


def build_provider_request(prompt: str, config: dict) -> dict:
    if sha(V21_PIPELINE.read_bytes()) != V21_PIPELINE_SHA256:
        raise ValueError("frozen_v21_request_builder_changed")
    spec = importlib.util.spec_from_file_location("v24_frozen_v21_request_builder", V21_PIPELINE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider_module, batch, ordering = module.load_dependencies()
    profile = provider_module.profile(batch)
    profile["profile_id"] = f"{HERE.name}_generation"
    request = ordering.build_payload(profile, prompt)
    if (request.get("model") != config["requested_model"] or
            request.get("max_tokens") != config["max_tokens"] or
            request.get("temperature") != config["temperature"]):
        raise ValueError("provider_request_profile_mismatch")
    return request


def load_provider_dependencies():
    if sha(V21_PIPELINE.read_bytes()) != V21_PIPELINE_SHA256:
        raise ValueError("frozen_v21_request_builder_changed")
    spec = importlib.util.spec_from_file_location("v24_frozen_v21_transport", V21_PIPELINE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider_module, batch, ordering = module.load_dependencies()
    return provider_module, batch, ordering


def batch_path(root_id: str, batch: int) -> Path:
    return HERE / root_id / f"batch-{batch:02d}"


def prior_slot_count(root_id: str, excluding_batch: int | None = None) -> int:
    parent = HERE / root_id
    if not parent.exists():
        return 0
    count = 0
    for path in parent.glob("batch-*/PREPARED.json"):
        if excluding_batch is not None and path.parent == batch_path(root_id, excluding_batch):
            continue
        manifest = read(path)
        count += int(manifest.get("requested_question_slots", 0))
    return count


def supplement_evidence(root_id: str, preceding_batch: int, source_turns: list[dict]) -> tuple[list[dict], dict]:
    review_path = batch_path(root_id, preceding_batch) / "CONTENT_REVIEW.json"
    if not review_path.exists():
        raise ValueError("topup_requires_preceding_batch_content_review")
    review = read(review_path)
    if review.get("supplement_approved") is not True:
        raise ValueError("topup_not_approved_by_content_review")
    if not isinstance(review.get("supplement_reason"), str) or not review["supplement_reason"].strip():
        raise ValueError("topup_reason_missing")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("supplement_reviewer_missing")
    return source_turns, review


def build_material(root_id: str, batch: int) -> tuple[dict, dict, bytes, str, dict, dict[str, str]]:
    if batch != 1:
        raise ValueError("v32_initial_batch_only")
    config = load_config()
    item = root_config(root_id, config)
    source_path, source_raw, source, source_turns = load_source(item)
    coverage, coverage_hashes = coverage_context(root_id, item)
    hints, hints_raw = load_capacity_hints(config)
    root_hints = hints["roots"][root_id]
    scenes = {turn.get("scene_ref") for turn in source_turns}
    if (not isinstance(root_hints, list) or not root_hints or
            any(not isinstance(hint, dict) or hint.get("scene_ref") not in scenes or
                not isinstance(hint.get("topic_hint"), str) or not hint["topic_hint"].strip()
                for hint in root_hints)):
        raise ValueError(f"capacity_hints_scene_invalid:{root_id}")
    used_slots = prior_slot_count(root_id, excluding_batch=batch)
    if batch == 1:
        other_batches = [path for path in (HERE / root_id).glob("batch-*/PREPARED.json")
                         if path.parent != batch_path(root_id, batch)]
        if other_batches:
            raise ValueError("unexpected_prior_batch_for_initial_batch")
        count, refs, supplement_review = 4, source_turns, None
    else:
        if batch not in (2, 3) or used_slots not in (4, 6):
            raise ValueError("topup_requires_prior_four_or_six_slots")
        refs, supplement_review = supplement_evidence(root_id, batch - 1, source_turns)
        count = min(2, 8 - used_slots)
    start = 1 if batch == 1 else used_slots + 1
    ids = [candidate_id(root_id, n) for n in range(start, start + count)]
    prior_model_drafts = []
    for previous in sorted((HERE / root_id).glob("batch-*/MODEL_RESPONSE.json")):
        if int(previous.parent.name.split("-")[-1]) >= batch:
            continue
        response = read(previous)
        try:
            prior_model_drafts.append(json.loads(response["raw_text"]))
        except (KeyError, TypeError, ValueError):
            prior_model_drafts.append({"unparsed_response_sha256": sha(previous.read_bytes())})
    prompt_input = {
        "root_id": root_id,
        "scope": "full_frozen_promoted_dialogue_P_plus_R01",
        "batch_number": batch,
        "requested_candidate_count": count,
        "candidate_ids": ids,
        "source_turns": refs,
        "existing_coverage": coverage,
        "capacity_screen_approval": {
            "screen_version": hints["screen_version"],
            "screen_decision": hints["screen_decision"],
            "screen_scope": hints["screen_scope"],
        },
        "capacity_screen_priority_hints": root_hints,
        "capacity_screen_caveat": hints["use"],
        "prior_model_drafts_to_avoid": prior_model_drafts,
        "supplement_review": supplement_review,
    }
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt.replace("{ROOT_ID}", root_id)
    prompt += json.dumps(prompt_input, ensure_ascii=False, indent=2) + "\n"
    request = build_provider_request(prompt, config)
    deps = {"source_path": str(source_path), "source_sha256": sha(source_raw),
            "coverage_file_sha256": coverage_hashes,
            "coverage_summary_sha256": sha(canonical(coverage)),
            "split_manifest_sha256": config["expected_split_manifest_sha256"],
            "split_record_sha256": sha(canonical(split_records(config)[root_id])),
            "capacity_hints_sha256": sha(hints_raw),
            "config_sha256": sha(CONFIG_PATH.read_bytes()),
            "prompt_template_sha256": sha(PROMPT_PATH.read_bytes()),
            "pipeline_sha256": sha(Path(__file__).read_bytes()),
            "request_builder_sha256": sha(V21_PIPELINE.read_bytes())}
    return prompt_input, request, source_raw, prompt, deps, {"source": source, "root_config": item}


def prepare(root_id: str, batch: int) -> dict:
    directory = batch_path(root_id, batch)
    if directory.exists():
        raise FileExistsError(f"no_overwrite:{directory}")
    prompt_input, request, source_raw, prompt, deps, extra = build_material(root_id, batch)
    config = load_config()
    manifest = {"version": HERE.name, "root_id": root_id, "batch_number": batch,
                "provider": config["provider"], "requested_model": config["requested_model"],
                "requested_question_slots": prompt_input["requested_candidate_count"],
                "candidate_ids": prompt_input["candidate_ids"],
                "source_turn_count": len(prompt_input["source_turns"]),
                "capacity_screen_approval": prompt_input["capacity_screen_approval"],
                "dependencies": deps, "input_sha256": sha(canonical(prompt_input)),
                "prompt_sha256": sha(prompt.encode()), "request_sha256": sha(canonical(request)),
                "prepared_at": datetime.now(timezone.utc).isoformat(),
                "status": "PREPARED_NO_API_CALL", "http_attempts": 0,
                "solver_calls": 0, "training_started": False}
    write_new(directory / "source.json", source_raw.decode("utf-8"), raw=True)
    write_new(directory / "CAPACITY_SCREEN.json", {
        "approval": prompt_input["capacity_screen_approval"],
        "priority_hints": prompt_input["capacity_screen_priority_hints"],
        "caveat": prompt_input["capacity_screen_caveat"],
        "capacity_hints_sha256": deps["capacity_hints_sha256"],
        "split_manifest_sha256": deps["split_manifest_sha256"],
        "split_record_sha256": deps["split_record_sha256"],
        "source_sha256": deps["source_sha256"],
        "candidate_pool_sha256": next(iter(deps["coverage_file_sha256"].values())),
    })
    write_new(directory / "DEDUP_SUMMARY.json", prompt_input["existing_coverage"])
    write_new(directory / "PREPARED_INPUT.json", prompt_input)
    write_new(directory / "PROMPT.txt", prompt, raw=True)
    write_new(directory / "REQUEST.json", request)
    write_new(directory / "PREPARED.json", manifest)
    return {"root_id": root_id, "batch_number": batch, "status": manifest["status"],
            "requested_question_slots": manifest["requested_question_slots"],
            "candidate_ids": manifest["candidate_ids"],
            "source_turn_count": manifest["source_turn_count"],
            "request_sha256": manifest["request_sha256"]}


def preflight(root_id: str, batch: int) -> dict:
    prompt_input, request, source_raw, prompt, deps, _ = build_material(root_id, batch)
    directory = batch_path(root_id, batch)
    manifest = read(directory / "PREPARED.json")
    expected = {"dependencies": deps, "input_sha256": sha(canonical(prompt_input)),
                "prompt_sha256": sha(prompt.encode()), "request_sha256": sha(canonical(request))}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen_hash_changed")
    capacity_snapshot = {
        "approval": prompt_input["capacity_screen_approval"],
        "priority_hints": prompt_input["capacity_screen_priority_hints"],
        "caveat": prompt_input["capacity_screen_caveat"],
        "capacity_hints_sha256": deps["capacity_hints_sha256"],
        "split_manifest_sha256": deps["split_manifest_sha256"],
        "split_record_sha256": deps["split_record_sha256"],
        "source_sha256": deps["source_sha256"],
        "candidate_pool_sha256": next(iter(deps["coverage_file_sha256"].values())),
    }
    if ((directory / "source.json").read_bytes() != source_raw or
            read(directory / "CAPACITY_SCREEN.json") != capacity_snapshot or
            read(directory / "DEDUP_SUMMARY.json") != prompt_input["existing_coverage"] or
            read(directory / "PREPARED_INPUT.json") != prompt_input or
            (directory / "PROMPT.txt").read_text(encoding="utf-8") != prompt or
            read(directory / "REQUEST.json") != request):
        raise ValueError("frozen_artifact_changed")
    return {"root_id": root_id, "batch_number": batch,
            "status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL",
            "requested_model": manifest["requested_model"],
            "request_sha256": manifest["request_sha256"]}


def execute(root_id: str, batch: int) -> dict:
    preflight_result = preflight(root_id, batch)
    directory = batch_path(root_id, batch)
    if (directory / "ATTEMPT_RESERVED.json").exists():
        raise FileExistsError("generation_attempt_already_reserved_no_retry")
    manifest = read(directory / "PREPARED.json")
    prompt = (directory / "PROMPT.txt").read_text(encoding="utf-8")
    request = read(directory / "REQUEST.json")
    provider_module, provider_batch, ordering = load_provider_dependencies()
    profile = provider_module.profile(provider_batch)
    profile["profile_id"] = f"{HERE.name}_generation"
    write_new(directory / "ATTEMPT_RESERVED.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": manifest["provider"], "requested_model": manifest["requested_model"],
        "request_sha256": manifest["request_sha256"], "maximum_http_attempts": 1})

    def on_attempt(actual: dict) -> None:
        if actual != request:
            raise ValueError("actual_payload_changed")
        write_new(directory / "HTTP_ATTEMPT.json", {
            "attempt_number": 1, "request_sha256": sha(canonical(actual)),
            "model": actual.get("model"), "max_tokens": actual.get("max_tokens")})

    try:
        response = ordering.credential_client(profile, on_http_attempt=on_attempt).complete(
            prompt=prompt, max_tokens=read(CONFIG_PATH)["max_tokens"], json_object=True)
    except Exception as error:
        result = {"status": "ERROR_NO_RETRY", "root_id": root_id, "batch_number": batch,
                  "error_type": type(error).__name__, "metadata": getattr(error, "metadata", None),
                  "request_sha256": manifest["request_sha256"],
                  "http_attempt_recorded": (directory / "HTTP_ATTEMPT.json").exists(),
                  "solver_calls": 0, "training_started": False}
        write_new(directory / "ERROR.json", result)
        return result
    write_new(directory / "MODEL_RESPONSE.json", {
        "request_sha256": manifest["request_sha256"],
        "raw_text": response.raw_text, "metadata": response.metadata})
    return {"status": "RESPONSE_PERSISTED", "root_id": root_id, "batch_number": batch,
            "request_sha256": manifest["request_sha256"],
            "preflight_status": preflight_result["status"],
            "solver_calls": 0, "training_started": False}


def check_response(root_id: str, batch: int) -> dict:
    preflight_result = preflight(root_id, batch)
    directory = batch_path(root_id, batch)
    manifest = read(directory / "PREPARED.json")
    response = read(directory / "MODEL_RESPONSE.json")
    if response.get("request_sha256") != manifest["request_sha256"]:
        raise ValueError("model_response_request_binding_invalid")
    fatal = []
    metadata = response.get("metadata", {})
    if not isinstance(metadata, dict):
        fatal.append("response_metadata_invalid")
        metadata = {}
    raw_text = response.get("raw_text")
    try:
        draft = json.loads(raw_text) if isinstance(raw_text, str) else None
    except ValueError:
        draft = None
    if not isinstance(draft, dict) or set(draft) != {"root_id", "questions", "abstentions"}:
        fatal.append("top_level_schema_invalid")
        draft = {"root_id": None, "questions": [], "abstentions": []}
    if draft["root_id"] != root_id:
        fatal.append("root_mismatch")
    if not isinstance(draft["abstentions"], list) or any(
            not isinstance(x, dict) or set(x) != {"candidate_id", "reason"}
            or not isinstance(x.get("candidate_id"), str) or not isinstance(x.get("reason"), str)
            for x in draft["abstentions"]):
        fatal.append("abstentions_invalid")
        draft["abstentions"] = []
    questions = draft["questions"]
    requested = manifest["requested_question_slots"]
    if not isinstance(questions, list) or len(questions) > requested:
        fatal.append("question_count_invalid")
        questions = []
    expected_model = manifest["requested_model"]
    if metadata.get("requested_model", expected_model) != expected_model:
        fatal.append("requested_model_metadata_mismatch")
    if metadata.get("returned_model") != expected_model:
        fatal.append("returned_model_mismatch")
    if metadata.get("finish_reason") == "length":
        fatal.append("response_truncated")
    source = read(directory / "source.json")
    turns = {turn["turn_ref"]: turn for turn in source["turns"]
             if isinstance(turn, dict) and isinstance(turn.get("turn_ref"), str)}
    expected_ids = manifest["candidate_ids"]
    abstention_ids = [item["candidate_id"] for item in draft["abstentions"]]
    if (len(abstention_ids) != len(set(abstention_ids)) or
            any(cid not in expected_ids for cid in abstention_ids)):
        fatal.append("abstention_candidate_ids_invalid_or_duplicate")
    ids, checks = set(), []
    for index, row in enumerate(questions):
        errors = []
        if not isinstance(row, dict) or set(row) != {"candidate_id", "topic", "question", "options"}:
            checks.append({"index": index, "candidate_id": None,
                           "errors": ["question_schema_invalid"], "mechanically_valid": False})
            continue
        cid = row["candidate_id"]
        if not isinstance(cid, str) or cid not in expected_ids or cid in ids:
            errors.append("candidate_id_invalid_or_duplicate")
        else:
            ids.add(cid)
        if not isinstance(row["topic"], str) or not row["topic"].strip():
            errors.append("topic_missing")
        if not isinstance(row["question"], str) or not row["question"].strip():
            errors.append("question_missing")
        options = row["options"]
        if not isinstance(options, list) or len(options) != 6:
            errors.append("six_options_required")
            options = []
        refs, positives = [], 0
        for option_index, option in enumerate(options):
            prefix = f"option_{option_index + 1}"
            fields = {"proposition", "relation", "turn_ref", "speaker", "quote", "rationale"}
            if not isinstance(option, dict) or set(option) != fields:
                errors.append(f"{prefix}_schema_invalid")
                continue
            ref, speaker, quote = option["turn_ref"], option["speaker"], option["quote"]
            if not isinstance(ref, str) or ref not in turns:
                errors.append(f"{prefix}_turn_ref_unavailable")
            else:
                refs.append(ref)
                if turns[ref].get("speaker") != speaker:
                    errors.append(f"{prefix}_speaker_mismatch")
                if not isinstance(quote, str) or not quote.strip() or quote not in turns[ref].get("text", ""):
                    errors.append(f"{prefix}_quote_not_verbatim")
            if not isinstance(quote, str) or len(quote) > 240:
                errors.append(f"{prefix}_quote_too_long_or_invalid")
            for field in ("proposition", "speaker", "rationale"):
                if not isinstance(option[field], str) or not option[field].strip():
                    errors.append(f"{prefix}_{field}_missing")
            if isinstance(option["speaker"], str) and isinstance(option["proposition"], str):
                if option["speaker"].casefold() not in option["proposition"].casefold():
                    errors.append(f"{prefix}_speaker_absent_from_proposition")
            if option["relation"] == "supports":
                positives += 1
            elif option["relation"] != "contradicts":
                errors.append(f"{prefix}_relation_invalid")
        if len(refs) != 6 or len(set(refs)) != 6:
            errors.append("six_distinct_source_turns_required")
        if not any(ref.startswith("R01:") for ref in refs):
            errors.append("candidate_must_anchor_at_least_one_r01_turn")
        if not 2 <= positives <= 6:
            errors.append("gold_cardinality_invalid")
        checks.append({"index": index, "candidate_id": cid, "errors": errors,
                       "inferred_gold_cardinality": positives, "mechanically_valid": not errors})
    if (len(questions) + len(draft["abstentions"]) != requested or
            set(ids) & set(abstention_ids) or set(ids) | set(abstention_ids) != set(expected_ids)):
        fatal.append("requested_slots_not_accounted_for")
    usage = metadata.get("usage", {})
    result = {"version": HERE.name, "root_id": root_id, "batch_number": batch,
              "status": "MECHANICAL_CHECKS_RAW_SOLVER_BRIDGE_NEXT",
              "request_sha256": manifest["request_sha256"], "response_sha256": sha(canonical(response)),
              "draft_sha256": sha(canonical(draft)),
              "fatal_errors": fatal, "question_checks": checks,
              "mechanically_valid_count": 0 if fatal else sum(row["mechanically_valid"] for row in checks),
              "provider": manifest["provider"], "requested_model": expected_model,
              "returned_model": metadata.get("returned_model"),
              "prompt_tokens": usage.get("prompt_tokens"),
              "completion_tokens": usage.get("completion_tokens"),
              "total_tokens": usage.get("total_tokens"),
              "semantic_review_performed": False,
              "solver_calls": 0, "training_started": False,
              "preflight_status": preflight_result["status"]}
    write_new(directory / "MODEL_DRAFT.json", draft)
    write_new(directory / "MECHANICAL_CHECKS.json", result)
    return result


def bind_reviewed(root_id: str, batch: int) -> dict:
    """Bind reviewed proposition/evidence pairs to labels without changing text."""
    preflight(root_id, batch)
    directory = batch_path(root_id, batch)
    checks = read(directory / "MECHANICAL_CHECKS.json")
    draft = read(directory / "MODEL_DRAFT.json")
    review = read(directory / "CONTENT_REVIEW.json")
    response = read(directory / "MODEL_RESPONSE.json")
    if (checks.get("fatal_errors") or checks.get("request_sha256") != read(directory / "PREPARED.json")["request_sha256"] or
            checks.get("response_sha256") != sha(canonical(response)) or
            checks.get("draft_sha256") != sha(canonical(draft)) or
            review.get("draft_sha256") != sha(canonical(draft))):
        raise ValueError("draft_or_content_review_binding_invalid")
    if review.get("response_sha256") != sha(canonical(response)):
        raise ValueError("review_not_bound_to_raw_model_response")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("reviewer_missing")
    decisions = review.get("question_decisions")
    if not isinstance(decisions, list):
        raise ValueError("question_decisions_missing")
    keyed = {}
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != {
                "candidate_id", "approved", "reason", "stem_scope_approved", "option_reviews"}:
            raise ValueError("question_decision_schema_invalid")
        cid = decision["candidate_id"]
        if (cid in keyed or not isinstance(decision["approved"], bool) or
                not isinstance(decision["reason"], str) or not decision["reason"].strip()):
            raise ValueError("question_decision_invalid_or_duplicate")
        if not isinstance(decision["stem_scope_approved"], bool):
            raise ValueError("stem_scope_review_missing")
        reviews = decision["option_reviews"]
        if not isinstance(reviews, list) or len(reviews) != 6:
            raise ValueError("six_per_option_reviews_required")
        option_by_index = {}
        for item in reviews:
            if not isinstance(item, dict) or set(item) != {
                    "option_index", "approved", "relation_confirmed", "semantic_key", "reason"}:
                raise ValueError("option_review_schema_invalid")
            index = item["option_index"]
            if (not isinstance(index, int) or index not in range(6) or index in option_by_index or
                    not isinstance(item["approved"], bool) or
                    not isinstance(item["relation_confirmed"], bool) or
                    not isinstance(item["semantic_key"], str) or not isinstance(item["reason"], str)):
                raise ValueError("option_review_invalid_or_duplicate")
            option_by_index[index] = item
        if set(option_by_index) != set(range(6)) or [item["option_index"] for item in reviews] != list(range(6)):
            raise ValueError("option_reviews_must_cover_all_six_positions")
        keyed[cid] = decision
    draft_rows = draft["questions"]
    if set(keyed) != {row["candidate_id"] for row in draft_rows}:
        raise ValueError("review_must_cover_every_draft_question")
    check_by_id = {row.get("candidate_id"): row for row in checks["question_checks"]}
    approved = []
    for row in draft_rows:
        cid = row["candidate_id"]
        decision = keyed[cid]
        mechanical = check_by_id.get(cid, {}).get("mechanically_valid") is True
        if decision["approved"] and not mechanical:
            raise ValueError(f"content_review_approved_mechanically_invalid_question:{cid}")
        if decision["approved"] and (not decision["stem_scope_approved"] or any(
                not item["approved"] or not item["relation_confirmed"]
                for item in decision["option_reviews"])):
            raise ValueError(f"question_approved_without_six_approved_options_and_stem:{cid}")
        if not decision["approved"]:
            continue
        normalized_keys = [" ".join(item["semantic_key"].casefold().split())
                           for item in decision["option_reviews"]]
        if any(not key for key in normalized_keys) or len(set(normalized_keys)) != 6:
            raise ValueError(f"semantic_duplicate_or_missing_within_question:{cid}")
        approved.append((row, normalized_keys))
    if len(approved) != sum(bool(item["approved"]) for item in decisions):
        raise ValueError("approved_question_count_mismatch")
    # Semantic keys must also be unique across approved questions in this batch.
    all_keys = [key for _, keys in approved for key in keys]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("same_batch_semantic_fact_duplicate")
    normalized = []
    for row, keys in approved:
        options = row["options"]
        cardinality = sum(option["relation"] == "supports" for option in options)
        candidate_seed = int(sha(row["candidate_id"].encode())[:8], 16)
        patterns = list(combinations(range(6), cardinality))
        pattern = patterns[(candidate_seed + len(normalized)) % len(patterns)]
        gold_positions = set(pattern)
        positive = [option for option in options if option["relation"] == "supports"]
        negative = [option for option in options if option["relation"] == "contradicts"]
        ordered = [None] * 6
        positive_iter, negative_iter = iter(positive), iter(negative)
        for index in range(6):
            ordered[index] = next(positive_iter) if index in gold_positions else next(negative_iter)
        labels = list(LABELS)
        candidate = {"candidate_id": row["candidate_id"], "root_id": root_id,
                     "question_type": "multi_select", "topic": row["topic"],
                     "question": row["question"],
                     "options": {label: ordered[index]["proposition"] for index, label in enumerate(labels)},
                     "gold": [labels[index] for index in range(6) if index in gold_positions],
                     "option_evidence": {label: {"turn_ref": ordered[index]["turn_ref"],
                                                  "quote": ordered[index]["quote"],
                                                  "relation": ordered[index]["relation"],
                                                  "reason": ordered[index]["rationale"]}
                                          for index, label in enumerate(labels)},
                     "answer_reason": "Gold labels correspond to the content-reviewed directly supported propositions; other options require explicit same-speaker contradictions."}
        normalized.append(candidate)
    out = {"root_id": root_id, "batch_number": batch, "reviewer": review["reviewer"],
           "source_response_sha256": sha(canonical(response)), "draft_sha256": sha(canonical(draft)),
           "content_review_sha256": sha(canonical(review)), "questions": normalized,
           "status": "CONTENT_APPROVED_NORMALIZED_DRAFT_SOLVER_NOT_RUN",
           "solver_calls": 0, "training_started": False}
    write_new(directory / "NORMALIZED_DRAFT.json", out)
    return {"root_id": root_id, "batch_number": batch,
            "status": out["status"], "approved_question_count": len(normalized),
            "candidate_ids": [row["candidate_id"] for row in normalized]}


def status() -> dict:
    config = load_config()
    roots = []
    for item in config["root_configs"]:
        _, raw, _, turns = load_source(item)
        roots.append({"root_id": item["root_id"], "source_sha256": sha(raw),
                      "source_turn_count": len(turns),
                      "r01_turn_count": sum(turn["turn_ref"].startswith("R01:") for turn in turns),
                      "prepared_slots": prior_slot_count(item["root_id"])})
    return {"version": HERE.name, "status": "ROOT_CENSUS_PENDING" if not roots else "ROOTS_CONFIGURED",
            "initial_slots_per_root": 4, "maximum_slots_per_root": 4, "roots": roots,
            "api_calls": 0, "solver_calls": 0, "training_started": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "prepare", "preflight", "execute",
                                           "check-response"))
    parser.add_argument("root_id", nargs="?")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.action == "status":
        result = status()
    else:
        if not args.root_id:
            parser.error("root_id is required for prepare/preflight")
        if args.action == "prepare":
            result = prepare(args.root_id, args.batch)
        elif args.action == "preflight":
            result = preflight(args.root_id, args.batch)
        elif args.action == "execute":
            result = execute(args.root_id, args.batch)
        elif args.action == "check-response":
            result = check_response(args.root_id, args.batch)
        elif args.action == "bind-reviewed":
            result = bind_reviewed(args.root_id, args.batch)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "ERROR", "error_type": type(error).__name__,
                          "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise
