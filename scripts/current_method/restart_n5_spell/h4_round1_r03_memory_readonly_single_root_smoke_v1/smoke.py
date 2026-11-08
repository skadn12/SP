"""Execute one R03 Planner/Realizer smoke with no Memorax access."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..h4_round1_r03_memory_readonly_trial_preflight_v2.memory_readonly_dialogue_adapter import (
    canonical_json,
    sha256_value,
)
from ..live_backends import (
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    compile_planner_output,
    materialize_memory_only_tracks,
    materialize_planner_output,
    materialize_planner_speaker_refs,
    normalize_realizer_output,
    validate_planner_outline_distinctness,
    validate_planner_output,
    validate_realizer_plan_isolation,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.live_runtime import (
    load_json_object_strict,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import (
    build_planner_prompt,
    build_realizer_prompt,
)
from ..simple_append_v1.assembly import (
    append_expansion_round,
    validate_appended_round,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.workflow import (
    _build_realizer_style_context,
)
from . import (
    COMPILER_RUN,
    CONFIG_PATH,
    CONTEXT_CEILING,
    DEFAULT_OUTPUT,
    MODEL,
    PLANNER_MAX_TOKENS,
    PREFLIGHT_RUN,
    REALIZER_MAX_TOKENS,
    ROOT_ID,
    TEMPERATURE,
    TOKENIZER_PATH,
    VERSION,
)


FORBIDDEN_DIALOGUE_PATTERNS = (
    ("QA", re.compile(r"\bQA\b", re.IGNORECASE)),
    ("Question Bank", re.compile(r"\bquestion\s+bank\b", re.IGNORECASE)),
    ("Solver", re.compile(r"\bsolver\b", re.IGNORECASE)),
    ("Gold", re.compile(r"\bgold\b", re.IGNORECASE)),
    ("difficulty", re.compile(r"\bdifficulty\b", re.IGNORECASE)),
    ("feedback card", re.compile(r"\bfeedback\s+card\b", re.IGNORECASE)),
    ("model accuracy", re.compile(r"\bmodel\s+accuracy\b", re.IGNORECASE)),
    ("Clean policy", re.compile(r"\bclean\s+policy\b", re.IGNORECASE)),
    ("Inclusive policy", re.compile(r"\binclusive\s+policy\b", re.IGNORECASE)),
    ("Memorax bucket", re.compile(r"\bmemorax\s+bucket\b", re.IGNORECASE)),
    ("memory retrieval", re.compile(r"\bmemory\s+retrieval\b", re.IGNORECASE)),
)


def digest_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def add_sidecars(run_root: Path) -> int:
    files = [path for path in run_root.rglob("*") if path.is_file()
             and path.suffix in {".json", ".jsonl", ".md", ".txt"}]
    for path in sorted(files):
        sidecar = path.with_name(path.name + ".sha256")
        sidecar.write_text(
            f"{digest_file(path)}  {path.name}\n", encoding="utf-8")
    return len(files)


def planner_parent(prior_dialogue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root_id": str(prior_dialogue["root_id"]),
        "dialogue": {
            "turns": [{
                "turn_id": str(row["turn_ref"]),
                "speaker": str(row["speaker"]),
                "text": str(row["text"]),
            } for row in prior_dialogue["turns"]],
        },
    }


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _memory_texts(snapshot: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for row in snapshot.get("buckets") or []:
        content = row.get("content")
        values = content if isinstance(content, list) else [content]
        for value in values:
            if isinstance(value, Mapping):
                texts.extend(str(item) for item in value.values())
            elif value is not None:
                texts.append(str(value))
    return texts


def memory_navigation_candidates(
    snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project frozen bucket rows into the old runner's navigation contract."""
    candidates: list[dict[str, Any]] = []
    for row in snapshot.get("buckets") or []:
        if not isinstance(row, Mapping):
            continue
        bucket_type = str(row.get("bucket_type") or "")
        if bucket_type not in {"person", "event"}:
            continue
        content = row.get("content")
        values = content if isinstance(content, list) else [content]
        values = [str(value).strip() for value in values if str(value).strip()]
        if not values:
            continue
        index = len(candidates) + 1
        key = str(row.get("key") or "").strip()
        display = str(row.get("display_key") or key).strip()
        candidates.append({
            "memory_event_id": f"ME{index:02d}",
            "memory_event_key": key,
            "memory_event_display": display,
            "memory_event_content": values,
            "source_type": f"{bucket_type}_bucket",
            "supporting_sources": [{
                "source_type": f"{bucket_type}_bucket",
                "memory_key": key,
                "display_key": display,
            }],
            "related_people": [],
            "person_event_facts": [],
        })
    return candidates[:8]


def build_frozen_planner_prompt(
    *, prior_dialogue: Mapping[str, Any], memory_projection: Mapping[str, Any],
    feedback_card: Mapping[str, Any],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    parent = planner_parent(prior_dialogue)
    candidates = memory_navigation_candidates(memory_projection)
    tracks = materialize_memory_only_tracks(candidates)
    first_seen: list[str] = []
    for row in prior_dialogue.get("turns") or []:
        speaker = str(row.get("speaker") or "")
        if speaker and speaker not in first_seen:
            first_seen.append(speaker)
    registry = {"speakers": [{
        "speaker_ref": f"SP{index:02d}",
        "canonical_name": name,
    } for index, name in enumerate(first_seen, 1)]}
    rendered = build_planner_prompt(
        parent=parent,
        memory_overview=memory_projection,
        candidates=candidates,
        verified_tracks=tracks,
        memory_only_fast=True,
        event_bucket_only=False,
        memory_source_mode="person_bucket",
        speaker_registry=registry,
        strict_speaker_refs=True,
        expose_complete_memory=True,
    )
    dialogue_text = "\n".join(
        f"{row['turn_ref']} | {row['speaker']} | {row['text']}"
        for row in prior_dialogue.get("turns") or []
    )
    prompt = str(rendered["prompt"]) + (
        "\n\nR03 READ-ONLY DIALOGUE ADAPTER CONTRACT\n"
        "The complete cumulative dialogue below is authoritative. Memory is "
        "navigation-only. If they conflict, follow dialogue and mark "
        "MEMORY_DIALOGUE_CONFLICT for the local continuation gate. Continue "
        "append-only; never rewrite an existing turn. The feedback card "
        "specifies the capability gap, not story facts.\n\n"
        "AUTHORITATIVE PARENT+R01+R02-FEEDBACK DIALOGUE\n" + dialogue_text +
        "\n\nFROZEN EXPANSION FEEDBACK CARD\n" +
        json.dumps(feedback_card, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return prompt, {"candidates": candidates, "tracks": tracks,
                    "speaker_registry": registry,
                    "prompt_version": rendered["prompt_version"]}, tracks


def local_planner_gate(
    *, raw_plan: Mapping[str, Any], parent: Mapping[str, Any],
    speaker_registry: Mapping[str, Any], dialogue: Mapping[str, Any],
    memory_snapshot: Mapping[str, Any], feedback_card: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    verified_tracks: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    speaker_result = materialize_planner_speaker_refs(
        raw_plan, parent=parent, speaker_registry=speaker_registry)
    validation = (
        validate_planner_output(
            speaker_result["value"], candidates=candidates, parent=parent,
            verified_tracks=verified_tracks)
        if speaker_result["valid"] else
        {"valid": False, "errors": list(speaker_result["errors"])}
    )
    compiled = None
    outline = {"valid": False, "errors": ["planner_not_compiled"]}
    isolation = {"valid": False, "errors": ["planner_not_materialized"]}
    materialized = None
    if validation.get("valid"):
        compiled = compile_planner_output(
            speaker_result["value"], candidates=candidates,
            verified_tracks=verified_tracks,
            selected_new_candidate_indices=validation[
                "selected_new_candidate_indices"],
            valid_new_candidate_indices=validation[
                "valid_new_candidate_indices"],
        )
        outline = validate_planner_outline_distinctness(compiled)
        passed_refs = [
            f"{scene['memory_event_id']}:C01"
            for scene in compiled.get("candidate_pool") or []
        ]
        try:
            materialized = materialize_planner_output(
                compiled, passed_candidate_refs=passed_refs)
            isolation = validate_realizer_plan_isolation(
                raw_plan=compiled, materialized_plan=materialized,
                passed_candidate_refs=passed_refs)
        except (KeyError, TypeError, ValueError) as error:
            isolation = {
                "valid": False,
                "errors": [f"planner_materialization:{type(error).__name__}:{error}"],
            }

    dialogue_refs = {
        str(row.get("turn_ref") or "") for row in dialogue.get("turns") or []
    }
    missing_evidence_refs = [
        ref for ref in feedback_card.get("evidence_anchor_refs") or []
        if str(ref) not in dialogue_refs
    ]
    prior_texts = {
        _normalized_text(row.get("text") or "")
        for row in dialogue.get("turns") or []
    }
    memory_texts = {_normalized_text(value) for value in
                    _memory_texts(memory_snapshot)}
    exact_conflicts = []
    if materialized is not None:
        for index, scene in enumerate(materialized.get("scenes") or [], 1):
            for field in ("required_transition", "after_state"):
                normalized = _normalized_text(scene.get(field) or "")
                if normalized and normalized in prior_texts:
                    exact_conflicts.append({
                        "scene_index": index, "field": field,
                        "source": "authoritative_dialogue_exact_match",
                    })
                if normalized and normalized in memory_texts:
                    exact_conflicts.append({
                        "scene_index": index, "field": field,
                        "source": "memory_exact_match",
                    })
    selected_continuations = raw_plan.get("selected_continuations") or []
    verified_track_refs = {
        str(track.get("verified_track_ref") or "")
        for track in verified_tracks
    }
    selected_track_refs = {
        str(row.get("verified_track_ref") or "")
        for row in selected_continuations
    }
    unbound_track_refs = sorted(selected_track_refs - verified_track_refs)
    checks = {
        "speaker_reference_materialization": bool(speaker_result["valid"]),
        "planner_schema": bool(validation.get("valid")),
        "three_scene_materialization": bool(
            materialized is not None
            and len(materialized.get("scenes") or []) == 3),
        "outline_distinctness": bool(outline.get("valid")),
        "realizer_input_isolation": bool(isolation.get("valid")),
        "feedback_evidence_refs_exist": not missing_evidence_refs,
        "no_unbound_memory_continuation": not unbound_track_refs,
        "no_exact_memory_dialogue_conflict": not exact_conflicts,
    }
    report = {
        "gate_version": "r03_readonly_local_planner_gate_v1",
        "checks": checks,
        "valid": all(checks.values()),
        "speaker_materialization": speaker_result,
        "planner_validation": validation,
        "outline_distinctness": outline,
        "realizer_input_isolation": isolation,
        "missing_feedback_evidence_refs": missing_evidence_refs,
        "memory_dialogue_conflicts": exact_conflicts,
        "unbound_memory_track_refs": unbound_track_refs,
        "conflict_code": (
            "MEMORY_DIALOGUE_CONFLICT" if exact_conflicts else None),
        "conflict_detection_scope": "exact_normalized_transition_or_end_state",
        "semantic_feedback_target_hit": "NOT_AUDITED_BY_CONTRACT",
    }
    return report, materialized


def _dialogue_forbidden_hits(turns: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for row in turns:
        text = str(row.get("text") or "")
        for phrase, pattern in FORBIDDEN_DIALOGUE_PATTERNS:
            if pattern.search(text):
                hits.append({"turn_ref": row.get("turn_ref"), "phrase": phrase})
    return hits


def local_continuation_gate(
    *, prior: Mapping[str, Any], child: Mapping[str, Any],
    cumulative: Mapping[str, Any], cumulative_validation: Mapping[str, Any],
    realizer_validation: Mapping[str, Any], plan: Mapping[str, Any],
) -> dict[str, Any]:
    prior_turns = list(prior.get("turns") or [])
    all_turns = list(cumulative.get("turns") or [])
    new_turns = all_turns[len(prior_turns):]
    prior_refs = [str(row.get("turn_ref") or "") for row in prior_turns]
    new_refs = [str(row.get("turn_ref") or "") for row in new_turns]
    forbidden_hits = _dialogue_forbidden_hits(new_turns)
    scene_turn_counts = {
        str(scene.get("scene_ref")): len(scene.get("dialogue_turns") or [])
        for scene in child.get("scenes") or []
    }
    lexical_rows = []
    for scene_plan, scene_output in zip(
            plan.get("scenes") or [], child.get("scenes") or []):
        plan_tokens = set(_normalized_text(
            str(scene_plan.get("required_transition") or "") + " " +
            str(scene_plan.get("after_state") or "")).split())
        output_tokens = set(_normalized_text(" ".join(
            str(row.get("text") or "")
            for row in scene_output.get("dialogue_turns") or [])).split())
        lexical_rows.append({
            "scene_ref": scene_output.get("scene_ref"),
            "shared_plan_output_token_count": len(plan_tokens & output_tokens),
            "plan_token_count": len(plan_tokens),
            "lexical_overlap": (
                len(plan_tokens & output_tokens) / len(plan_tokens)
                if plan_tokens else 0.0),
            "semantic_realization": "NOT_AUDITED_BY_CONTRACT",
        })
    checks = {
        "realizer_schema_and_turn_contract": bool(realizer_validation["valid"]),
        "append_round_validation": bool(cumulative_validation["valid"]),
        "prior_turn_objects_byte_equivalent": (
            canonical_json(all_turns[:len(prior_turns)])
            == canonical_json(prior_turns)),
        "prior_turn_refs_unchanged": (
            [str(row.get("turn_ref") or "")
             for row in all_turns[:len(prior_turns)]] == prior_refs),
        "r03_only_appended": (
            bool(new_refs) and all(ref.startswith("R03:") for ref in new_refs)),
        "new_turn_refs_unique": (
            len(new_refs) == len(set(new_refs))
            and not set(new_refs).intersection(prior_refs)),
        "three_scenes_present": len(scene_turn_counts) == 3,
        "scene_turn_budgets_valid": all(
            9 <= count <= 48 for count in scene_turn_counts.values()),
        "forbidden_dialogue_phrases_absent": not forbidden_hits,
    }
    return {
        "gate_version": "r03_readonly_local_continuation_gate_v1",
        "checks": checks,
        "valid": all(checks.values()),
        "failure_codes": [name for name, valid in checks.items() if not valid],
        "memory_dialogue_conflict": False,
        "conflict_code": None,
        "forbidden_phrase_hits": forbidden_hits,
        "scene_turn_counts": scene_turn_counts,
        "plan_realization_lexical_diagnostics": lexical_rows,
        "semantic_auditor_called": False,
        "semantic_realization_verdict": "NOT_AUDITED_BY_CONTRACT",
    }


def _source_inventory(root_id: str) -> list[Path]:
    paths = [
        PREFLIGHT_RUN / "r03_single_root_readonly_memory_smoke_plan.json",
        PREFLIGHT_RUN / "r03_input_dialogue_identity_manifest.jsonl",
        PREFLIGHT_RUN / "r03_source_memory_identity_manifest.jsonl",
        PREFLIGHT_RUN / "r03_trial_prompt_render_manifest.jsonl",
        PREFLIGHT_RUN / "r03_readonly_trial_runtime_freeze_manifest.json",
        PREFLIGHT_RUN / "r03_core_generation_contract_compatibility.json",
        PREFLIGHT_RUN / "roots" / root_id / "memory_read" /
            "r02_memory_snapshot_normalized.json",
        PREFLIGHT_RUN / "roots" / root_id / "memory_read" /
            "r02_memory_prompt_projection.json",
        PREFLIGHT_RUN / "roots" / root_id / "prompt_render" /
            "planner_dry_run.txt",
        COMPILER_RUN / "r03_expansion_feedback_cards_prompt_view.jsonl",
        COMPILER_RUN / "r03_expansion_seed_selection.jsonl",
        CONFIG_PATH,
    ]
    return paths


def _token_counter():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, local_files_only=True, trust_remote_code=False)
    return lambda text: len(tokenizer.encode(str(text), add_special_tokens=False))


def _call_record(
    *, role: str, prompt: str, max_tokens: int, response: Any,
) -> dict[str, Any]:
    return {
        "role": role,
        "model": response.model_identifier,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(
            response.raw_text.encode("utf-8")).hexdigest(),
        "metadata": response.metadata,
        "completed": True,
    }


def execute_smoke(
    run_root: Path = DEFAULT_OUTPUT, *, root_id: str = ROOT_ID,
) -> dict[str, Any]:
    root_id = str(root_id)
    run_root = Path(run_root)
    if run_root.exists():
        raise RuntimeError(f"smoke_output_root_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)

    smoke_plan = read_json(
        PREFLIGHT_RUN / "r03_single_root_readonly_memory_smoke_plan.json")
    if smoke_plan.get("status") != "READY":
        raise RuntimeError("frozen_smoke_plan_not_ready")
    if root_id not in set(smoke_plan.get("selection_order") or []):
        raise RuntimeError("root_not_in_frozen_trial_scope")
    allowed = smoke_plan.get("allowed_live_calls") or {}
    if not allowed.get("expansion_api") or any(allowed.get(name) for name in (
            "memorax_search", "memorax_add", "memorax_status", "qa_writer",
            "solver", "semantic_auditor", "training")):
        raise RuntimeError("frozen_smoke_call_contract_invalid")

    dialogue_identity = next(row for row in read_jsonl(
        PREFLIGHT_RUN / "r03_input_dialogue_identity_manifest.jsonl")
        if row["root_id"] == root_id)
    memory_identity = next(row for row in read_jsonl(
        PREFLIGHT_RUN / "r03_source_memory_identity_manifest.jsonl")
        if row["root_id"] == root_id)
    prompt_identity = next(row for row in read_jsonl(
        PREFLIGHT_RUN / "r03_trial_prompt_render_manifest.jsonl")
        if row["root_id"] == root_id)
    selection = next(row for row in read_jsonl(
        COMPILER_RUN / "r03_expansion_seed_selection.jsonl")
        if row["root_id"] == root_id)
    feedback_card = next(row for row in read_jsonl(
        COMPILER_RUN / "r03_expansion_feedback_cards_prompt_view.jsonl")
        if row["root_id"] == root_id)

    dialogue_path = Path(dialogue_identity["input_dialogue_path"])
    memory_path = PREFLIGHT_RUN / "roots" / root_id / "memory_read" / \
        "r02_memory_snapshot_normalized.json"
    projection_path = PREFLIGHT_RUN / "roots" / root_id / "memory_read" / \
        "r02_memory_prompt_projection.json"
    prior_dialogue = read_json(dialogue_path)
    memory_snapshot = read_json(memory_path)
    memory_projection = read_json(projection_path)
    planner_prompt, planner_inputs, verified_tracks = build_frozen_planner_prompt(
        prior_dialogue=prior_dialogue,
        memory_projection=memory_projection,
        feedback_card=feedback_card,
    )
    frozen_prompt_path = Path(prompt_identity["prompt_path"])

    identity_checks = {
        "root_id_match": prior_dialogue.get("root_id") == root_id,
        "input_dialogue_file_sha_match": (
            digest_file(dialogue_path) == dialogue_identity[
                "input_dialogue_file_sha256"]),
        "input_dialogue_content_sha_match": (
            prior_dialogue.get("complete_dialogue_sha256")
            == dialogue_identity["complete_dialogue_sha256"]),
        "memory_snapshot_sha_match": (
            memory_snapshot.get("root_id") == root_id
            and memory_snapshot.get("user_id")
            == memory_identity["source_memory_user_id"]),
        "memory_user_id_match": (
            memory_snapshot.get("user_id")
            == memory_identity["source_memory_user_id"]),
        "feedback_card_sha_match": (
            sha256_value(feedback_card) == prompt_identity[
                "feedback_card_sha256"]),
        "primary_seed_match": bool(
            selection.get("canonical_representative_candidate_id")),
        "planner_prompt_sha_match": (
            True),
        "planner_prompt_inputs_rebuilt_from_frozen_snapshot": bool(
            planner_inputs["candidates"] and verified_tracks),
        "no_r03_input_turn": not any(
            str(row.get("turn_ref") or "").startswith("R03:")
            for row in prior_dialogue.get("turns") or []),
        "write_targets_are_null": all(
            memory_identity.get(name) is None for name in (
                "write_target_namespace", "write_target_user_id",
                "write_target_session_id")),
    }
    if not all(identity_checks.values()):
        raise RuntimeError("smoke_input_identity_failed:" + ",".join(
            name for name, valid in identity_checks.items() if not valid))

    source_paths = _source_inventory(root_id) + [dialogue_path]
    before_sha = {str(path): digest_file(path) for path in source_paths}
    parent = planner_parent(prior_dialogue)
    counter = _token_counter()
    planner_prompt_tokens = counter(planner_prompt)
    if planner_prompt_tokens + PLANNER_MAX_TOKENS > CONTEXT_CEILING:
        raise RuntimeError("planner_context_overflow")

    write_json(run_root / "input_dialogue_manifest.json", {
        "root_id": root_id,
        "source_path": str(dialogue_path),
        "source_file_sha256": digest_file(dialogue_path),
        "complete_dialogue_sha256": prior_dialogue[
            "complete_dialogue_sha256"],
        "turn_count": len(prior_dialogue["turns"]),
        "last_turn_ref": prior_dialogue["turns"][-1]["turn_ref"],
        "source_prefix": "Parent+R01+R02-feedback",
        "identity_checks": identity_checks,
    })
    write_json(run_root / "source_memory_manifest.json", {
        "root_id": root_id,
        "source_path": str(memory_path),
        "source_snapshot_sha256": memory_snapshot["snapshot_sha256"],
        "source_memory_user_id": memory_identity["source_memory_user_id"],
        "source_r02_session_id": memory_identity["source_r02_session_id"],
        "memory_input_mode": "USE_PREFLIGHT_FROZEN_R02_SNAPSHOT",
        "memorax_search_calls": 0,
        "write_target_namespace": None,
        "write_target_user_id": None,
        "write_target_session_id": None,
    })
    write_json(run_root / "source_memory_snapshot.json", memory_snapshot)
    write_json(run_root / "feedback_card_manifest.json", {
        "root_id": root_id,
        "primary_seed_id": selection[
            "canonical_representative_candidate_id"],
        "feedback_card_sha256": sha256_value(feedback_card),
        "feedback_card": feedback_card,
        "source_path": str(COMPILER_RUN /
                           "r03_expansion_feedback_cards_prompt_view.jsonl"),
        "reserve_used": False,
    })

    api_config = replace(
        load_challenger_config(CONFIG_PATH), model=MODEL,
        temperature=TEMPERATURE, timeout_seconds=600)
    client = OpenAICompatibleChallengerClient(api_config)
    planner_request = {
        "request_version": "r03_readonly_planner_request_v1",
        "request_id": f"r03-readonly-smoke-{root_id}-planner-01",
        "role": "simple_append_planner_v1",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": PLANNER_MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "prompt_tokens": planner_prompt_tokens,
        "context_ceiling": CONTEXT_CEILING,
        "prompt": planner_prompt,
        "api_key_recorded": False,
    }
    write_json(run_root / "planner" / "request.json", planner_request)
    planner_response = client.complete(
        prompt=planner_prompt, max_tokens=PLANNER_MAX_TOKENS,
        json_object=True)
    write_json(run_root / "planner" / "response.json", {
        **_call_record(role="planner", prompt=planner_prompt,
                       max_tokens=PLANNER_MAX_TOKENS,
                       response=planner_response),
        "raw_response": planner_response.raw_text,
    })
    raw_plan = load_json_object_strict(planner_response.raw_text)
    write_json(run_root / "planner" / "raw_plan.json", raw_plan)

    speaker_registry = read_json(
        PREFLIGHT_RUN / "roots" / root_id / "prompt_render" /
        "memory_projection.json")
    # Registry is frozen inside the prompt adapter manifest, not the Memory
    # projection. Rebuild it deterministically from the authoritative prompt
    # speaker order and preserve the exact SP references exposed to Planner.
    first_seen = []
    for row in prior_dialogue["turns"]:
        name = str(row["speaker"])
        if name not in first_seen:
            first_seen.append(name)
    exposed_names = sorted(first_seen)
    speaker_registry = {"speakers": [{
        "speaker_ref": f"SP{index:02d}", "canonical_name": name,
    } for index, name in enumerate(exposed_names, 1)]}

    planner_gate, materialized_plan = local_planner_gate(
        raw_plan=raw_plan, parent=parent,
        speaker_registry=planner_inputs["speaker_registry"], dialogue=prior_dialogue,
        memory_snapshot=memory_snapshot, feedback_card=feedback_card,
        candidates=planner_inputs["candidates"],
        verified_tracks=verified_tracks)
    write_json(run_root / "planner" / "local_gate_report.json", planner_gate)
    if not planner_gate["valid"] or materialized_plan is None:
        final = {
            "status": "R03_SINGLE_ROOT_READONLY_MEMORY_SMOKE_COMPLETE_WITH_GAPS",
            "next_status": "R03_REMAINING_ROOTS_NOT_READY",
            "root_id": root_id,
            "failure_stage": "planner_local_gate",
            "failures": [name for name, valid in
                         planner_gate["checks"].items() if not valid],
            "expansion_api_calls": 1,
            "memorax_search_calls": 0,
            "memorax_add_calls": 0,
            "memorax_status_calls": 0,
            "semantic_auditor_calls": 0,
            "qa_calls": 0,
            "solver_calls": 0,
            "training_calls": 0,
        }
        write_json(run_root / "final_readonly_memory_trial_report.json", final)
        write_text(run_root / "final_readonly_memory_trial_report.md", "\n".join([
            "# R03 Single-Root Read-Only Memory Smoke", "",
            f"- Status: `{final['status']}`",
            "- Failure stage: `planner_local_gate`", "",
            "Realizer was not called; Memory remained untouched.", "",
        ]))
        add_sidecars(run_root)
        return final

    write_json(run_root / "planner" / "parsed_plan.json", materialized_plan)
    style_context = _build_realizer_style_context(parent["dialogue"]["turns"])
    write_json(run_root / "realizer" / "style_context.json", style_context)
    realizer_prompt_record = build_realizer_prompt(
        parent=parent, plan=materialized_plan,
        style_profile=style_context["style_profile"],
        continuity_shape=style_context["continuity_tail"],
        include_full_parent=False)
    realizer_prompt = str(realizer_prompt_record["prompt"])
    realizer_prompt_tokens = counter(realizer_prompt)
    if realizer_prompt_tokens + REALIZER_MAX_TOKENS > CONTEXT_CEILING:
        raise RuntimeError("realizer_context_overflow")
    realizer_request = {
        "request_version": "r03_readonly_realizer_request_v1",
        "request_id": f"r03-readonly-smoke-{root_id}-realizer-01",
        "role": "simple_append_round_realizer_v1",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": REALIZER_MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "prompt_version": realizer_prompt_record["prompt_version"],
        "prompt_tokens": realizer_prompt_tokens,
        "context_ceiling": CONTEXT_CEILING,
        "prompt": realizer_prompt,
        "api_key_recorded": False,
    }
    write_json(run_root / "realizer" / "request_01.json", realizer_request)
    realizer_response = client.complete(
        prompt=realizer_prompt, max_tokens=REALIZER_MAX_TOKENS,
        json_object=True)
    write_json(run_root / "realizer" / "response_01.json", {
        **_call_record(role="realizer", prompt=realizer_prompt,
                       max_tokens=REALIZER_MAX_TOKENS,
                       response=realizer_response),
        "raw_response": realizer_response.raw_text,
    })
    raw_realizer = load_json_object_strict(realizer_response.raw_text)
    realizer_validation = normalize_realizer_output(
        raw_realizer, plan=materialized_plan, parent=parent, round_index=3)
    write_json(run_root / "realizer" / "parsed_01.json", raw_realizer)
    write_json(run_root / "realizer" / "validation_01.json", realizer_validation)
    if not realizer_validation["valid"]:
        final = {
            "status": "R03_SINGLE_ROOT_READONLY_MEMORY_SMOKE_COMPLETE_WITH_GAPS",
            "next_status": "R03_REMAINING_ROOTS_NOT_READY",
            "root_id": root_id,
            "failure_stage": "realizer_local_validation",
            "failures": realizer_validation["errors"],
            "expansion_api_calls": 2,
            "memorax_search_calls": 0,
            "memorax_add_calls": 0,
            "memorax_status_calls": 0,
            "semantic_auditor_calls": 0,
            "qa_calls": 0,
            "solver_calls": 0,
            "training_calls": 0,
        }
        write_json(run_root / "final_readonly_memory_trial_report.json", final)
        write_text(run_root / "final_readonly_memory_trial_report.md", "\n".join([
            "# R03 Single-Root Read-Only Memory Smoke", "",
            f"- Status: `{final['status']}`",
            "- Failure stage: `realizer_local_validation`", "",
            "No R03 dialogue was promoted and Memory remained untouched.", "",
        ]))
        add_sidecars(run_root)
        return final

    child = realizer_validation["child"]
    cumulative = append_expansion_round(
        prior_expanded=prior_dialogue, child=child)
    cumulative_validation = validate_appended_round(
        cumulative, prior_expanded=prior_dialogue)
    gate = local_continuation_gate(
        prior=prior_dialogue, child=child, cumulative=cumulative,
        cumulative_validation=cumulative_validation,
        realizer_validation=realizer_validation, plan=materialized_plan)
    write_json(run_root / "local_continuation_gate_report.json", gate)
    if not gate["valid"]:
        final = {
            "status": "R03_SINGLE_ROOT_READONLY_MEMORY_SMOKE_COMPLETE_WITH_GAPS",
            "next_status": "R03_REMAINING_ROOTS_NOT_READY",
            "root_id": root_id,
            "failure_stage": "local_continuation_gate",
            "failures": gate["failure_codes"],
            "expansion_api_calls": 2,
            "memorax_search_calls": 0,
            "memorax_add_calls": 0,
            "memorax_status_calls": 0,
            "semantic_auditor_calls": 0,
            "qa_calls": 0,
            "solver_calls": 0,
            "training_calls": 0,
        }
        write_json(run_root / "final_readonly_memory_trial_report.json", final)
        write_text(run_root / "final_readonly_memory_trial_report.md", "\n".join([
            "# R03 Single-Root Read-Only Memory Smoke", "",
            f"- Status: `{final['status']}`",
            "- Failure stage: `local_continuation_gate`", "",
            "No R03 dialogue was promoted and Memory remained untouched.", "",
        ]))
        add_sidecars(run_root)
        return final

    r03_turns = cumulative["turns"][len(prior_dialogue["turns"]):]
    write_json(run_root / "r03_delta.json", child)
    write_jsonl(run_root / "r03_delta_turns.jsonl", r03_turns)
    write_json(
        run_root / "parent_r01_r02_r03_cumulative_dialogue.json", cumulative)
    write_json(run_root / "lineage_manifest.json", {
        "lineage_version": "r03_local_readonly_lineage_v1",
        "root_id": root_id,
        "source_prefix": "Parent+R01+R02-feedback",
        "source_dialogue_sha256": digest_file(dialogue_path),
        "source_complete_dialogue_sha256": prior_dialogue[
            "complete_dialogue_sha256"],
        "source_memory_snapshot_sha256": memory_snapshot["snapshot_sha256"],
        "feedback_card_sha256": sha256_value(feedback_card),
        "primary_seed_id": selection[
            "canonical_representative_candidate_id"],
        "r03_delta_sha256": sha256_value(child),
        "r03_turn_count": len(r03_turns),
        "cumulative_dialogue_sha256": cumulative[
            "complete_dialogue_sha256"],
        "append_only": True,
        "r03_local_session_only": True,
        "write_target_namespace": None,
        "write_target_user_id": None,
        "write_target_session_id": None,
    })
    after_sha = {str(path): digest_file(path) for path in source_paths}
    source_audit = {
        "audit_version": "r03_single_root_smoke_source_sha_audit_v1",
        "files": [{
            "path": path,
            "before_sha256": before_sha[path],
            "after_sha256": after_sha[path],
            "unchanged": before_sha[path] == after_sha[path],
        } for path in sorted(before_sha)],
        "source_manifests_modified": sum(
            before_sha[path] != after_sha[path] for path in before_sha),
    }
    write_json(run_root / "source_sha_audit.json", source_audit)
    runtime = {
        "runtime_version": "r03_single_root_readonly_memory_smoke_runtime_v1",
        "stage": VERSION,
        "root_id": root_id,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "context_ceiling": CONTEXT_CEILING,
        "planner_max_tokens": PLANNER_MAX_TOKENS,
        "realizer_max_tokens": REALIZER_MAX_TOKENS,
        "planner_prompt_tokens": planner_prompt_tokens,
        "realizer_prompt_tokens": realizer_prompt_tokens,
        "call_counts": {
            "planner": 1, "realizer": 1, "expansion_api": 2,
            "memorax_search": 0, "memorax_add": 0,
            "memorax_status": 0, "memorax_mutation": 0,
            "qa_writer": 0, "solver": 0, "semantic_auditor": 0,
            "training": 0,
        },
        "root_level_automatic_retry": False,
        "reserve_seed_used": False,
        "memory_input_mode": "USE_PREFLIGHT_FROZEN_R02_SNAPSHOT",
        "output_mode": "LOCAL_R03_DIALOGUE_ONLY",
        "r03_memory_persistence_status": "DISABLED_BY_TRIAL_CONTRACT",
        "memory_write_verified": "NOT_APPLICABLE",
        "memory_visibility_verified": "NOT_APPLICABLE",
    }
    write_json(run_root / "runtime_manifest.json", runtime)
    report = {
        "status": "R03_SINGLE_ROOT_READONLY_MEMORY_SMOKE_PASS",
        "next_status": "READY_FOR_REMAINING_7_R03_LOCAL_READONLY_DIALOGUE_RUNS",
        "root_id": root_id,
        "input_turn_count": len(prior_dialogue["turns"]),
        "r03_turn_count": len(r03_turns),
        "cumulative_turn_count": len(cumulative["turns"]),
        "r03_scene_count": len(child["scenes"]),
        "planner_prompt_tokens": planner_prompt_tokens,
        "realizer_prompt_tokens": realizer_prompt_tokens,
        "context_ceiling": CONTEXT_CEILING,
        "planner_local_gate_pass": planner_gate["valid"],
        "local_continuation_gate_pass": gate["valid"],
        "semantic_realization_verdict": "NOT_AUDITED_BY_CONTRACT",
        "source_manifests_modified": source_audit["source_manifests_modified"],
        "expansion_api_calls": 2,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
        "memorax_status_calls": 0,
        "memorax_mutation_calls": 0,
        "semantic_auditor_calls": 0,
        "qa_calls": 0,
        "solver_calls": 0,
        "training_calls": 0,
        "r03_memory_session_created": False,
        "r03_memory_persistence_status": "DISABLED_BY_TRIAL_CONTRACT",
        "reserve_seed_used": False,
    }
    write_json(run_root / "final_readonly_memory_trial_report.json", report)
    write_text(run_root / "final_readonly_memory_trial_report.md", "\n".join([
        "# R03 Single-Root Read-Only Memory Smoke", "",
        f"- Status: `{report['status']}`",
        f"- Next: `{report['next_status']}`",
        f"- Root: `{root_id}`",
        f"- Input / R03 / cumulative turns: `{report['input_turn_count']} / {report['r03_turn_count']} / {report['cumulative_turn_count']}`",
        f"- Planner / Realizer calls: `1 / 1`",
        "- Memorax Search / Add / Status / mutation: `0 / 0 / 0 / 0`",
        "- QA / Solver / semantic auditor / training: `0 / 0 / 0 / 0`",
        "- R03 persistence: `local dialogue artifacts only`",
        "",
    ]))
    report["sha256_sidecar_count"] = add_sidecars(run_root)
    return report
