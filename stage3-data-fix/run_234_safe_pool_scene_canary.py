#!/usr/bin/env python3
"""Run234: five isolated core-graph + safe-expansion single-scene canaries.

The API sees only the current two-node truth, its directed relation, a protected
surface contract, a safe expansion pool, and the response shape. Parent data is
loaded and frozen locally for provenance but is never copied into an API prompt.
There is no retry, repair, continuation, QA call, graph extraction, training
write, or production Stage3 entry point in this runner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import native_api_graph_dialogue_expansion as native


RUN_ID = "234_safe_pool_scene_canary_n5"
OUTPUT_DIR = ROOT / "base_restart_current_method_validation" / RUN_ID
SOURCE_PARENT_FILE = ROOT / "stage3-data-fix" / "stage3_parent_v4_memory_backfilled.jsonl"
SOURCE_PARENT_FILE_SHA256 = (
    "425efe31adfd9b1a53e30a9c1ab8382db90826ba372181050fac82a4a2394ec9"
)
TOKENIZER_PATH = ROOT.parent / "Qwen3-4B"
MODEL = "Qwen/Qwen3-32B"
TEMPERATURE = 0.25
CANARY_COUNT = 5
MIN_TURNS = 4
MAX_TURNS = 6
MIN_TOKENS = 150
MAX_TOKENS = 220
SCHEMA_VERSION = "stage3_safe_pool_scene_canary_v1"
ALLOWED_CALL_KIND = "api_truth_single_scene_probe"


PARENT_SPECS: list[dict[str, Any]] = [
    {
        "parent_sample_id": "stage3_parent_v4_8be1dd8dbc81fad7e073",
        "lineage_id": "conv-43",
        "subject": "John",
        "speaker_roster": ["John", "Tim"],
        "binder_event_id": "conv-43:OBS:17:001:26ab6a65",
        "binder_turn_id": "D17:3",
    },
    {
        "parent_sample_id": "stage3_parent_v4_1df7edaf7f56821c3572",
        "lineage_id": "conv-42",
        "subject": "Joanna",
        "speaker_roster": ["Joanna", "Nate"],
        "binder_event_id": "conv-42:OBS:22:002:90bd047f",
        "binder_turn_id": "D22:9",
    },
    {
        "parent_sample_id": "stage3_parent_v4_002f2907fb7a3613a280",
        "lineage_id": "conv-26",
        "subject": "Caroline",
        "speaker_roster": ["Caroline", "Melanie"],
        "binder_event_id": "conv-26:OBS:09:004:6f39a12b",
        "binder_turn_id": "D9:12",
    },
    {
        "parent_sample_id": "stage3_parent_v4_625f69365558405c482d",
        "lineage_id": "conv-30",
        "subject": "Gina",
        "speaker_roster": ["Gina", "Jon"],
        "binder_event_id": "conv-30:OBS:17:002:72251f85",
        "binder_turn_id": "D17:3",
    },
    {
        "parent_sample_id": "stage3_parent_v4_0f5c6c3ebd248dfe78a2",
        "lineage_id": "conv-44",
        "subject": "Audrey",
        "speaker_roster": ["Audrey", "Andrew"],
        "binder_event_id": "conv-44:OBS:18:002:3b191ae2",
        "binder_turn_id": "D18:6",
    },
]


TRUTH_SPECS: list[dict[str, Any]] = [
    {
        "truth_plan_id": "TP234-01",
        "slot": "destination",
        "source_value": "Cedar Hall",
        "source_status": "belief_unconfirmed",
        "source_raw_status": "belief",
        "target_value": "London",
        "target_status": "active",
        "source_status_markers": [
            "still only a possibility",
            "not confirmed",
            "tentative",
        ],
        "distractor_role": "same_subject_same_slot_wrong_epistemic_status",
    },
    {
        "truth_plan_id": "TP234-02",
        "slot": "destination",
        "source_value": "Riverside Studio",
        "source_status": "reported_unconfirmed",
        "source_raw_status": "report",
        "target_value": "North Campus",
        "target_status": "active",
        "source_status_markers": [
            "only reported",
            "not verified",
            "an unconfirmed report",
        ],
        "distractor_role": "same_slot_report_not_active",
    },
    {
        "truth_plan_id": "TP234-03",
        "slot": "destination",
        "source_value": "West Annex",
        "source_status": "outdated_no_longer_applicable",
        "source_raw_status": "outdated",
        "target_value": "Harbor Hall",
        "target_status": "active",
        "source_status_markers": [
            "no longer current",
            "outdated",
            "an old entry",
        ],
        "distractor_role": "same_slot_outdated_not_current",
    },
    {
        "truth_plan_id": "TP234-04",
        "slot": "destination",
        "source_value": "East Studio",
        "source_status": "canceled_before_effect",
        "source_raw_status": "canceled",
        "target_value": "Main Hall",
        "target_status": "active",
        "source_status_markers": [
            "was canceled",
            "was called off",
            "did not go ahead",
        ],
        "distractor_role": "same_slot_canceled_before_effect",
    },
    {
        "truth_plan_id": "TP234-05",
        "slot": "destination",
        "source_value": "Library Annex",
        "source_status": "expired_no_longer_valid",
        "source_raw_status": "expired",
        "target_value": "Garden Room",
        "target_status": "active",
        "source_status_markers": [
            "has expired",
            "is no longer valid",
            "has lapsed",
        ],
        "distractor_role": "same_slot_expired_not_current",
    },
]


TARGET_STATUS_MARKERS = [
    "current choice",
    "active destination",
    "what I'm going with",
]

RELATION_MARKERS = [
    "instead",
    "moved on",
    "by contrast",
    "the newer entry",
]

ALL_STATUS_MARKERS: dict[str, list[str]] = {
    "belief_unconfirmed": [
        "still only a possibility",
        "not confirmed",
        "tentative",
    ],
    "reported_unconfirmed": [
        "only reported",
        "not verified",
        "an unconfirmed report",
    ],
    "outdated_no_longer_applicable": [
        "no longer current",
        "outdated",
        "an old entry",
    ],
    "canceled_before_effect": [
        "was canceled",
        "was called off",
        "did not go ahead",
    ],
    "expired_no_longer_valid": [
        "has expired",
        "is no longer valid",
        "has lapsed",
    ],
    "active": TARGET_STATUS_MARKERS,
}

STATUS_VALIDATION_ALIASES: dict[str, list[str]] = {
    "active": ["I'm going with that", "going with that instead"],
}

STATUS_CONTRADICTION_MARKERS: dict[str, list[str]] = {
    "belief_unconfirmed": ["is confirmed", "was confirmed", "is already active"],
    "reported_unconfirmed": ["is verified", "was verified", "is already active"],
    "outdated_no_longer_applicable": ["is still valid", "is still current", "is active now"],
    "canceled_before_effect": ["did go ahead", "was completed", "is active now"],
    "expired_no_longer_valid": ["is still valid", "was renewed", "is active now"],
    "active": [
        "unconfirmed",
        "tentative",
        "only reported",
        "not verified",
        "outdated",
        "was canceled",
        "has expired",
        "not current",
    ],
}


SAFE_UNIT_BASE: list[tuple[str, str]] = [
    (
        "organize_unnamed_personal_notes",
        "The speakers organize an unnamed personal list without naming a file, tool, or source.",
    ),
    (
        "ask_status_difference",
        "One speaker asks about the general difference between the two already supplied states.",
    ),
    (
        "confirm_understanding",
        "One speaker briefly confirms that they understood the existing distinction.",
    ),
    (
        "express_avoid_confusion",
        "A speaker says they want to avoid mixing up the two existing entries.",
    ),
    (
        "agree_separate_records",
        "The speakers agree to keep the two supplied states conceptually separate.",
    ),
    (
        "express_relief_after_clarification",
        "A speaker expresses ordinary relief after the existing distinction is clear.",
    ),
    (
        "general_hesitation",
        "A speaker shows ordinary hesitation without adding a reason, event, or alternative.",
    ),
    (
        "acknowledge_existing_update",
        "A speaker acknowledges the already supplied update without restating both core facts.",
    ),
    (
        "neutral_transition",
        "The conversation moves naturally between the supplied facts with no new world detail.",
    ),
    (
        "natural_close",
        "The speakers close the short exchange naturally without proposing a new action.",
    ),
]

SAFE_UNIT_FORBIDDEN_FIELDS = [
    "new_person",
    "new_location",
    "date_or_deadline",
    "information_source",
    "reason_or_cause",
    "new_destination",
    "new_state_transition",
    "route_or_transport",
    "travel_experience",
    "named_file_software_or_tool",
    "new_plan_or_completed_event",
]

SYSTEM_PROMPT = (
    "Generate one short direct conversation from only the supplied immutable core "
    "truth and local safe expansion units. Do not infer background. Do not add any "
    "world fact beyond the immutable core; the listed present conversational "
    "reactions are ancillary and allowed. Return strict JSON only, with exactly "
    "selected_expansion_ids and turns."
)


class CanaryError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    )
    return " ".join(normalized.casefold().split())


def _assert_output_path(path: Path) -> None:
    root = OUTPUT_DIR.resolve(strict=False)
    resolved = path.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise CanaryError(f"write path escapes isolated run234 directory: {path}")
    current = path
    while current != root.parent and current.exists():
        if current.is_symlink():
            raise CanaryError(f"refusing symlink in run234 output path: {current}")
        current = current.parent


def _atomic_write_json(path: Path, value: Any) -> None:
    _assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"invalid JSON {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CanaryError(f"blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CanaryError(
                    f"truncated JSONL row at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise CanaryError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _paths() -> dict[str, Path]:
    return {
        "manifest": OUTPUT_DIR / "manifest.json",
        "plans": OUTPUT_DIR / "frozen_truth_plans.json",
        "state": OUTPUT_DIR / "state.json",
        "journal": OUTPUT_DIR / "api_scene_calls.jsonl",
        "machine": OUTPUT_DIR / "machine_audit.json",
        "manual": OUTPUT_DIR / "manual_audit.json",
        "report": OUTPUT_DIR / "final_report.json",
    }


def _append_journal(record: dict[str, Any]) -> dict[str, Any]:
    path = _paths()["journal"]
    _assert_output_path(path)
    rows = _read_jsonl(path)
    previous_hash = None
    for row in rows:
        if row.get("previous_record_sha256") != previous_hash:
            raise CanaryError("run234 API journal chain predecessor mismatch")
        unhashed = copy.deepcopy(row)
        declared = unhashed.pop("record_sha256", None)
        if declared != _sha256_json(unhashed):
            raise CanaryError("run234 API journal record hash mismatch")
        previous_hash = declared
    payload = copy.deepcopy(record)
    payload["previous_record_sha256"] = previous_hash
    payload["record_sha256"] = _sha256_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _journal_audit() -> dict[str, Any]:
    rows = _read_jsonl(_paths()["journal"])
    previous_hash = None
    for row in rows:
        if row.get("previous_record_sha256") != previous_hash:
            raise CanaryError("API journal chain is broken")
        unhashed = copy.deepcopy(row)
        declared = unhashed.pop("record_sha256", None)
        if declared != _sha256_json(unhashed):
            raise CanaryError("API journal content hash is broken")
        previous_hash = declared
    intents: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    unmatched: list[str] = []
    index = 0
    expected_ordinal = 1
    while index < len(rows):
        intent = rows[index]
        if intent.get("record_type") != "call_intent":
            raise CanaryError("API journal must alternate call_intent/call_result")
        if intent.get("call_ordinal") != expected_ordinal:
            raise CanaryError("API journal call ordinals are not contiguous")
        intents.append(intent)
        if index + 1 >= len(rows):
            unmatched.append(str(intent.get("call_id") or ""))
            index += 1
            break
        result = rows[index + 1]
        if result.get("record_type") != "call_result":
            raise CanaryError("each API call intent must be followed by its result")
        for key in (
            "call_id",
            "call_ordinal",
            "truth_plan_id",
            "truth_plan_sha256",
            "prompt_sha256",
        ):
            if result.get(key) != intent.get(key):
                raise CanaryError(f"API journal intent/result mismatch for {key}")
        results.append(result)
        expected_ordinal += 1
        index += 2
    intent_ids = [row.get("call_id") for row in intents]
    result_ids = [row.get("call_id") for row in results]
    if len(intent_ids) != len(set(intent_ids)):
        raise CanaryError("API journal contains duplicate call intents")
    if len(result_ids) != len(set(result_ids)):
        raise CanaryError("API journal contains duplicate call results")
    if len(unmatched) > 1 or (unmatched and rows[-1].get("record_type") != "call_intent"):
        raise CanaryError("API journal has an invalid unmatched intent")
    return {
        "rows": rows,
        "intents": intents,
        "results": results,
        "intent_count": len(intents),
        "result_count": len(results),
        "unmatched_intent_ids": unmatched,
        "chain_head": previous_hash,
    }


def _load_parent_rows() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if _sha256_file(SOURCE_PARENT_FILE) != SOURCE_PARENT_FILE_SHA256:
        raise CanaryError("source parent file hash changed")
    wanted = {row["parent_sample_id"] for row in PARENT_SPECS}
    found: dict[str, dict[str, Any]] = {}
    binder_texts: dict[str, str] = {}
    for row in _read_jsonl(SOURCE_PARENT_FILE):
        sample_id = row.get("sample_id")
        if sample_id not in wanted:
            continue
        found[sample_id] = row
    if set(found) != wanted:
        raise CanaryError(f"missing local parents: {sorted(wanted - set(found))}")
    for spec in PARENT_SPECS:
        row = found[spec["parent_sample_id"]]
        if row.get("lineage_group") != spec["lineage_id"]:
            raise CanaryError(f"lineage mismatch for {spec['parent_sample_id']}")
        turns = [
            turn
            for turn in (row.get("dialogue") or {}).get("turns") or []
            if turn.get("turn_id") == spec["binder_turn_id"]
        ]
        if len(turns) != 1 or turns[0].get("speaker") != spec["subject"]:
            raise CanaryError(f"identity binder turn mismatch for {spec['lineage_id']}")
        events = (row.get("semantic_parent_graph") or {}).get("events") or []
        event = next(
            (
                event
                for event in events
                if event.get("event_id") == spec["binder_event_id"]
            ),
            None,
        )
        if not isinstance(event, dict) or event.get("entity") != spec["subject"]:
            raise CanaryError(f"identity binder event mismatch for {spec['lineage_id']}")
        binder_texts[spec["truth_plan_id"] if "truth_plan_id" in spec else spec["lineage_id"]] = str(
            turns[0].get("text") or ""
        )
    return found, binder_texts


def _safe_pool() -> list[dict[str, Any]]:
    return [
        {
            "expansion_id": expansion_id,
            "allowed_meaning": allowed_meaning,
            "forbidden_specific_fields": list(SAFE_UNIT_FORBIDDEN_FIELDS),
            "graph_inclusion": False,
            "inheritance": False,
        }
        for expansion_id, allowed_meaning in SAFE_UNIT_BASE
    ]


def _build_truth_plans() -> tuple[list[dict[str, Any]], dict[str, str]]:
    parents, binder_text_by_lineage = _load_parent_rows()
    plans = []
    parent_text_by_plan: dict[str, str] = {}
    for ordinal, (parent, truth) in enumerate(zip(PARENT_SPECS, TRUTH_SPECS), start=1):
        subject = parent["subject"]
        node_claims = [
            {
                "node_id": "N1",
                "subject": subject,
                "slot": truth["slot"],
                "value": truth["source_value"],
                "status": truth["source_status"],
                "raw_status": truth["source_raw_status"],
                "temporal_index": 1,
                "required_status_markers": list(truth["source_status_markers"]),
            },
            {
                "node_id": "N2",
                "subject": subject,
                "slot": truth["slot"],
                "value": truth["target_value"],
                "status": truth["target_status"],
                "raw_status": "active",
                "temporal_index": 2,
                "required_status_markers": list(TARGET_STATUS_MARKERS),
            },
        ]
        relation = {
            "relation_id": "R1",
            "type": "status_transition",
            "subject": subject,
            "slot": truth["slot"],
            "source_node_id": "N1",
            "target_node_id": "N2",
            "source_value": truth["source_value"],
            "target_value": truth["target_value"],
            "direction": f"{truth['source_value']} -> {truth['target_value']}",
            "temporal_order": ["N1", "N2"],
            "required_relation_markers": list(RELATION_MARKERS),
            "required_relation_surface_patterns": [
                f"moved on to {truth['target_value']}",
                f"instead, {truth['target_value']}",
                f"by contrast, {truth['target_value']}",
                f"the newer entry is {truth['target_value']}",
            ],
            "must_mean": (
                "The earlier non-active entry is followed by a distinct active "
                "entry. The source value keeps its original non-active status."
            ),
        }
        parent_row = parents[parent["parent_sample_id"]]
        parent_provenance = {
            "lineage_id": parent["lineage_id"],
            "parent_sample_id": parent["parent_sample_id"],
            "parent_dialogue_hash": (parent_row.get("dialogue") or {}).get(
                "dialogue_hash"
            ),
            "parent_graph_sha256": _sha256_json(
                parent_row.get("semantic_parent_graph") or {}
            ),
            "identity_binder": {
                "subject": subject,
                "binder_event_id": parent["binder_event_id"],
                "binder_turn_id": parent["binder_turn_id"],
                "local_only_not_in_scene_prompt": True,
            },
        }
        protected_set = {
            "subjects": [subject],
            "slot": truth["slot"],
            "all_core_values": [truth["source_value"], truth["target_value"]],
            "all_core_statuses": [truth["source_status"], truth["target_status"]],
            "relation_type": "status_transition",
            "relation_direction": relation["direction"],
            "temporal_order": ["N1", "N2"],
            "answer_type": "LocationOrScope",
            "predetermined_gold": {
                "node_id": "N2",
                "value": truth["target_value"],
                "status": "active",
            },
            "predetermined_distractor_roles": [
                {
                    "role": truth["distractor_role"],
                    "node_id": "N1",
                    "value": truth["source_value"],
                },
                {"role": "wrong_binder_active_location", "local_only": True},
                {"role": "same_type_other_lineage_location", "local_only": True},
            ],
            "identity_binder": copy.deepcopy(parent_provenance["identity_binder"]),
            "forbid_additional_same_slot_value": True,
            "protected_fact_violation_definition": [
                "change core subject, value, status, relation direction, or temporal order",
                "add another destination value",
                "add a source, reason, time, type, or participant for a core event",
                "add another state transition or inheritable world state",
                "change the identity binder or make a distractor possibly correct",
                "add a shortcut that replaces the frozen reasoning path",
            ],
        }
        qa_target = {
            "operation": "temporal_state_update",
            "target": "status_gated_current_destination_after_nonactive_predecessor",
            "required_reasoning_path": ["N1", "R1", "N2"],
            "answer_type": "LocationOrScope",
            "compiled_gold": truth["target_value"],
            "primary_distractor": truth["source_value"],
            "api_visible": False,
        }
        plan = {
            "schema_version": SCHEMA_VERSION,
            "truth_plan_id": truth["truth_plan_id"],
            "canary_ordinal": ordinal,
            "parent_provenance": parent_provenance,
            "speaker_roster": list(parent["speaker_roster"]),
            "node_claims": node_claims,
            "relation": relation,
            "protected_set": protected_set,
            "predetermined_qa_reasoning_target": qa_target,
            "safe_expansion_pool": _safe_pool(),
        }
        plan["truth_plan_sha256"] = _sha256_json(plan)
        plans.append(plan)
        parent_text_by_plan[truth["truth_plan_id"]] = binder_text_by_lineage[
            parent["lineage_id"]
        ]
    return plans, parent_text_by_plan


def _api_prompt(plan: Mapping[str, Any]) -> dict[str, Any]:
    protected = plan["protected_set"]
    return {
        "speaker_roster": copy.deepcopy(plan["speaker_roster"]),
        "immutable_node_claims": copy.deepcopy(plan["node_claims"]),
        "immutable_relation": copy.deepcopy(plan["relation"]),
        "safe_expansion_pool": copy.deepcopy(plan["safe_expansion_pool"]),
        "protected_field_restrictions": {
            "core_subjects": copy.deepcopy(protected["subjects"]),
            "protected_slot": protected["slot"],
            "allowed_core_values_only": copy.deepcopy(protected["all_core_values"]),
            "allowed_core_statuses_only": copy.deepcopy(protected["all_core_statuses"]),
            "relation_direction": protected["relation_direction"],
            "temporal_order": copy.deepcopy(protected["temporal_order"]),
            "forbid_additional_same_slot_value": True,
            "forbid_core_fact_source_reason_time_type_or_participant": True,
            "forbid_new_state_transition": True,
            "forbid_new_inheritable_world_state": True,
            "forbid_exclusive_answer_summary": True,
        },
        "generation_contract": {
            "turn_count": "4-6 inclusive",
            "tokenizer_token_count": "150-220 inclusive",
            "visible_word_target": (
                "Aim for 125-155 English words total; tokenizer counting also "
                "includes stable turn IDs and speaker labels."
            ),
            "selected_expansion_ids": (
                "Select 2-4 unique IDs only from safe_expansion_pool."
            ),
            "core_realization": [
                "Each core value must appear exactly once with one supplied natural status marker in the same subject-spoken turn.",
                "Put at least one full turn between the source-value turn and target-value turn.",
                "Use one supplied required_relation_surface_pattern verbatim in the target-value turn, without repeating the source value.",
                "Do not put both core values in one turn or one adjacent two-turn exchange.",
                "A core-value question does not establish truth; the subject must state each core claim affirmatively.",
            ],
            "safe_expansion": [
                "Use selected units only for emotion, reaction, clarification, or dialogue management.",
                "Do not concretize a safe unit with a person, place, date, source, reason, event, plan, route, tool, or new state.",
            ],
            "output": (
                "Return one strict JSON object and no markdown or surrounding text."
            ),
        },
        "response_schema": {
            "selected_expansion_ids": ["known expansion_id"],
            "turns": [
                {"speaker": "one exact roster speaker", "text": "unaltered visible text"}
            ],
        },
    }


def _parent_prompt_leak_audit(
    plan: Mapping[str, Any], prompt: Mapping[str, Any], parent_text: str
) -> dict[str, Any]:
    serialized = _canonical_json(prompt)
    local = plan["parent_provenance"]
    forbidden_exact = [
        local["lineage_id"],
        local["parent_sample_id"],
        local["parent_dialogue_hash"],
        local["parent_graph_sha256"],
        local["identity_binder"]["binder_event_id"],
        local["identity_binder"]["binder_turn_id"],
        parent_text,
        str(SOURCE_PARENT_FILE),
    ]
    forbidden_keys_or_phrases = [
        "parent_provenance",
        "identity_binder",
        "parent memory",
        "parent turn",
        "lineage_id",
        "binder_event_id",
        "binder_turn_id",
    ]
    exact_matches = [value for value in forbidden_exact if value and value in serialized]
    phrase_matches = [
        value
        for value in forbidden_keys_or_phrases
        if value.casefold() in serialized.casefold()
    ]
    return {
        "pass": not exact_matches and not phrase_matches,
        "exact_parent_matches": exact_matches,
        "parent_key_or_phrase_matches": phrase_matches,
        "prompt_sha256": _sha256_json(prompt),
    }


def _preflight(output_fresh_required: bool) -> dict[str, Any]:
    plans, parent_texts = _build_truth_plans()
    signatures = [
        _sha256_json(
            {
                "subject": plan["node_claims"][0]["subject"],
                "slot": plan["node_claims"][0]["slot"],
                "nodes": [
                    [node["value"], node["status"]] for node in plan["node_claims"]
                ],
                "relation_type": plan["relation"]["type"],
                "relation_direction": plan["relation"]["direction"],
            }
        )
        for plan in plans
    ]
    prompt_audits = []
    for plan in plans:
        prompt = _api_prompt(plan)
        prompt_audits.append(
            {
                "truth_plan_id": plan["truth_plan_id"],
                **_parent_prompt_leak_audit(
                    plan, prompt, parent_texts[plan["truth_plan_id"]]
                ),
            }
        )
    api_config = (_read_json(ROOT / "config.json").get("api") or {})
    checks = {
        "source_parent_file_hash_exact": _sha256_file(SOURCE_PARENT_FILE)
        == SOURCE_PARENT_FILE_SHA256,
        "five_truth_plans": len(plans) == CANARY_COUNT,
        "five_unique_truth_signatures": len(set(signatures)) == CANARY_COUNT,
        "two_nodes_each": all(len(plan["node_claims"]) == 2 for plan in plans),
        "one_relation_each": all(bool(plan.get("relation")) for plan in plans),
        "safe_pool_8_to_12_each": all(
            8 <= len(plan["safe_expansion_pool"]) <= 12 for plan in plans
        ),
        "safe_units_non_graph_non_inheritable": all(
            unit["graph_inclusion"] is False and unit["inheritance"] is False
            for plan in plans
            for unit in plan["safe_expansion_pool"]
        ),
        "parent_absent_from_all_prompts": all(
            audit["pass"] for audit in prompt_audits
        ),
        "api_config_present": bool(
            api_config.get("api_key")
            and api_config.get("base_url")
        ),
        "curl_available": bool(shutil.which("curl")),
        "tokenizer_path_exists": TOKENIZER_PATH.exists(),
        "output_dir_fresh": (not OUTPUT_DIR.exists())
        if output_fresh_required
        else True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": all(checks.values()),
        "checks": checks,
        "truth_plan_ids": [plan["truth_plan_id"] for plan in plans],
        "truth_signatures": signatures,
        "parent_prompt_leak_audits": prompt_audits,
        "planned_scene_api_calls": 5,
        "planned_internal_retries": 0,
        "planned_external_retries": 0,
        "planned_repairs": 0,
        "planned_continuations": 0,
        "planned_qa_api_calls": 0,
        "planned_graph_api_calls": 0,
        "planned_training_writes": 0,
    }


def _api_args() -> argparse.Namespace:
    api = _read_json(ROOT / "config.json").get("api") or {}
    return argparse.Namespace(
        challenger_api_key=api.get("api_key"),
        challenger_base_url=api.get("base_url"),
        challenger_model=MODEL,
        challenger_temperature=TEMPERATURE,
        challenger_max_tokens=1800,
        max_llm_retries=0,
        llm_timeout=180,
        api_yield_normalizers=False,
    )


def _raw_response_from_meta(meta: Mapping[str, Any]) -> str:
    raw = meta.get("raw_response")
    if isinstance(raw, str):
        return raw
    errors = meta.get("previous_errors")
    if isinstance(errors, list):
        for error in reversed(errors):
            if isinstance(error, Mapping) and isinstance(error.get("raw_response"), str):
                return str(error["raw_response"])
    errors = meta.get("errors")
    if isinstance(errors, list):
        for error in reversed(errors):
            if isinstance(error, Mapping) and isinstance(error.get("raw_response"), str):
                return str(error["raw_response"])
    return ""


def _strict_raw_candidate(raw_response: str) -> tuple[Any, list[str]]:
    issues = []
    if not raw_response:
        return None, ["raw_response_missing"]
    stripped = raw_response.strip()
    try:
        candidate = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, [f"raw_response_not_strict_json:{exc.msg}"]
    if not stripped.startswith("{") or not stripped.endswith("}"):
        issues.append("raw_response_has_surrounding_text")
    return candidate, issues


def _format_audit(
    candidate: Any, raw_issues: list[str], plan: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues = list(raw_issues)
    normalized_turns: list[dict[str, Any]] = []
    selected: list[str] = []
    if not isinstance(candidate, dict) or set(candidate) != {
        "selected_expansion_ids",
        "turns",
    }:
        issues.append("top_level_schema_invalid")
        candidate = candidate if isinstance(candidate, dict) else {}
    raw_selected = candidate.get("selected_expansion_ids")
    allowed_ids = {
        unit["expansion_id"] for unit in plan["safe_expansion_pool"]
    }
    if (
        not isinstance(raw_selected, list)
        or not raw_selected
        or any(not isinstance(item, str) or not item for item in raw_selected)
    ):
        issues.append("selected_expansion_ids_invalid")
    else:
        selected = list(raw_selected)
        if len(selected) != len(set(selected)):
            issues.append("selected_expansion_ids_duplicate")
        if any(item not in allowed_ids for item in selected):
            issues.append("selected_expansion_ids_unknown")
    raw_turns = candidate.get("turns")
    if not isinstance(raw_turns, list):
        issues.append("turns_not_array")
        raw_turns = []
    if not MIN_TURNS <= len(raw_turns) <= MAX_TURNS:
        issues.append("turn_count_out_of_range")
    roster = set(plan["speaker_roster"])
    for index, turn in enumerate(raw_turns, start=1):
        if not isinstance(turn, dict) or set(turn) != {"speaker", "text"}:
            issues.append(f"turn_{index}_schema_invalid")
            continue
        speaker = turn.get("speaker")
        text = turn.get("text")
        if speaker not in roster:
            issues.append(f"turn_{index}_speaker_invalid")
        if not isinstance(text, str) or not text.strip():
            issues.append(f"turn_{index}_text_invalid")
        normalized_turns.append(
            {
                "turn_id": f"C{plan['canary_ordinal']:02d}:T{index:02d}",
                "speaker": speaker,
                "text": text if isinstance(text, str) else "",
            }
        )
    token_count, token_mode = native.dialogue_tokens_for_turns(
        normalized_turns, str(TOKENIZER_PATH)
    )
    if token_mode != "tokenizer":
        issues.append("token_count_not_from_tokenizer")
    if not MIN_TOKENS <= token_count <= MAX_TOKENS:
        issues.append("token_count_out_of_range")
    return (
        {
            "pass": not issues,
            "issues": issues,
            "selected_expansion_ids": selected,
            "turn_count": len(raw_turns),
            "tokenizer_tokens": token_count,
            "token_count_mode": token_mode,
            "token_count_serialization": "[stable_turn_id] speaker: raw_text",
        },
        normalized_turns,
    )


def _marker_matches(text: str, markers: list[str]) -> list[str]:
    folded = _normalized(text)
    return [marker for marker in markers if _normalized(marker) in folded]


def _unnegated_marker_matches(text: str, markers: list[str]) -> list[str]:
    folded = _normalized(text)
    matches = []
    for marker in markers:
        needle = _normalized(marker)
        start = folded.find(needle)
        while start >= 0:
            prefix = folded[max(0, start - 18) : start]
            if not re.search(
                r"(?:\bnot|\bnever|\bisn't|\bwasn't|\bdidn't|\bno longer)\s+"
                r"(?:(?:yet|really|currently|actually)\s+)?"
                r"(?:(?:a|an|the|my|our|his|her|their|your)\s+)?$",
                prefix,
            ):
                matches.append(marker)
                break
            start = folded.find(needle, start + len(needle))
    return matches


def _value_status_context(text: str, value: str) -> str:
    """Keep status checks in the value-bearing clause and an explicit it-clause."""
    clauses = [item.strip() for item in re.split(r"(?<=[.!?;])\s+", text) if item.strip()]
    folded_value = _normalized(value)
    for index, clause in enumerate(clauses):
        if folded_value not in _normalized(clause):
            continue
        selected = [clause]
        if index + 1 < len(clauses) and re.match(
            r"^(?:but\s+|and\s+)?(?:it|that|this)\b", clauses[index + 1], re.I
        ):
            selected.append(clauses[index + 1])
        return " ".join(selected)
    return text


def _core_truth_audit(
    turns: list[dict[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    issues = []
    supports = []
    all_text = "\n".join(str(turn.get("text") or "") for turn in turns)
    value_turn_indices: dict[str, int] = {}
    for node in plan["node_claims"]:
        value = node["value"]
        value_folded = _normalized(value)
        occurrence_count = _normalized(all_text).count(value_folded)
        matching = [
            index
            for index, turn in enumerate(turns)
            if value_folded in _normalized(turn.get("text"))
        ]
        node_issues = []
        if occurrence_count != 1 or len(matching) != 1:
            node_issues.append("core_value_not_expressed_exactly_once")
        turn_index = matching[0] if len(matching) == 1 else None
        status_matches: list[str] = []
        conflicting_statuses: dict[str, list[str]] = {}
        if turn_index is not None:
            turn = turns[turn_index]
            status_context = _value_status_context(turn["text"], value)
            validation_markers = [
                *node["required_status_markers"],
                *STATUS_VALIDATION_ALIASES.get(node["status"], []),
            ]
            status_matches = _marker_matches(
                status_context, validation_markers
            )
            if not status_matches:
                node_issues.append("required_status_not_expressed")
            negated_required = [
                marker
                for marker in status_matches
                if marker not in _unnegated_marker_matches(status_context, [marker])
                and not _normalized(marker).startswith(
                    ("not ", "no longer ", "is no longer ", "did not ")
                )
            ]
            if negated_required:
                node_issues.append("required_status_marker_negated")
            other_status_markers = [
                marker
                for status, markers in ALL_STATUS_MARKERS.items()
                if status != node["status"]
                for marker in markers
            ]
            explicit_conflicts = _unnegated_marker_matches(
                status_context,
                [
                    *STATUS_CONTRADICTION_MARKERS[node["status"]],
                    *other_status_markers,
                ],
            )
            if explicit_conflicts:
                conflicting_statuses["generic_contradiction"] = explicit_conflicts
            if conflicting_statuses:
                node_issues.append("conflicting_status_marker")
            if turn.get("speaker") != node["subject"]:
                node_issues.append("core_subject_speaker_drift")
            if str(turn.get("text") or "").rstrip().endswith("?"):
                node_issues.append("core_claim_only_asked_as_question")
            value_turn_indices[node["node_id"]] = turn_index
        issues.extend(f"{node['node_id']}:{item}" for item in node_issues)
        supports.append(
            {
                "node_id": node["node_id"],
                "value": value,
                "status": node["status"],
                "occurrence_count": occurrence_count,
                "support_turn_indices": (
                    [turn_index + 1] if turn_index is not None else []
                ),
                "matched_status_markers": status_matches,
                "conflicting_status_markers": conflicting_statuses,
                "resolved_subject": (
                    node["subject"] if turn_index is not None and turns[turn_index].get("speaker") == node["subject"] else None
                ),
                "issues": node_issues,
            }
        )
    source_index = value_turn_indices.get("N1")
    target_index = value_turn_indices.get("N2")
    relation_issues = []
    relation_markers: list[dict[str, Any]] = []
    if source_index is None or target_index is None:
        relation_issues.append("relation_endpoints_unbound")
    else:
        if source_index >= target_index:
            relation_issues.append("relation_direction_or_temporal_order_reversed")
        if target_index - source_index < 2:
            relation_issues.append("relation_endpoints_not_separated")
        start, end = sorted((source_index, target_index))
        index = target_index
        matches = _unnegated_marker_matches(
            turns[index]["text"],
            [
                *plan["relation"]["required_relation_surface_patterns"],
                f"{plan['relation']['target_value']} is the newer entry",
            ],
        )
        if matches:
            relation_markers.append(
                {"turn_index": index + 1, "markers": matches}
            )
        if not relation_markers:
            relation_issues.append("directed_relation_marker_missing")
    issues.extend(f"R1:{item}" for item in relation_issues)
    return {
        "pass": not issues,
        "issues": issues,
        "node_supports": supports,
        "relation_support": {
            "source_turn_index": (
                source_index + 1 if source_index is not None else None
            ),
            "target_turn_index": (
                target_index + 1 if target_index is not None else None
            ),
            "matched_direction_markers": relation_markers,
            "direction": plan["relation"]["direction"],
            "issues": relation_issues,
        },
        "subject_preserved": all(
            support["resolved_subject"] == plan["node_claims"][0]["subject"]
            for support in supports
        ),
        "values_preserved": all(support["occurrence_count"] == 1 for support in supports),
        "statuses_preserved": all(
            support["matched_status_markers"]
            and not support["conflicting_status_markers"]
            and "required_status_marker_negated" not in support["issues"]
            for support in supports
        ),
        "relation_direction_preserved": not relation_issues,
    }


SOURCE_PATTERN = re.compile(
    r"\b(?:recommended|suggested|someone told|told me|was told|heard from|according to|read about|found online|from someone)\b",
    re.I,
)
CAUSE_PATTERN = re.compile(r"\b(?:because|due to|caused by|as a result of)\b", re.I)
TIME_PATTERN = re.compile(
    r"\b(?:today|tomorrow|yesterday|next week|last week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2})\b",
    re.I,
)
TYPE_PATTERN = re.compile(
    r"\b(?:is|was|it's|that is)\s+(?:a|an|the)\s+(?:hotel|library|school|office|studio|campus|hall|room|venue|building|town|city|trail|station)\b",
    re.I,
)
PARTICIPANT_PATTERN = re.compile(
    r"\b(?:with my|with our|together with|joined by|recommended by|suggested by)\b",
    re.I,
)
NEW_EVENT_PATTERN = re.compile(
    r"\b(?:I|we)\s+(?:booked|bought|scheduled|visited|traveled|will go|plan to go)\b"
    r"|\b(?:I|we)\s+(?:finished|completed|started)\s+"
    r"(?:(?:a|an|the|my|our)\s+)?(?:new\s+)?"
    r"(?:trip|journey|visit|booking|plan|project|route|move|event)\b",
    re.I,
)
EXTRA_STATE_PATTERN = re.compile(
    r"\b(?:canceled again|expired again|became active too|became current too|also active|also current|another transition)\b",
    re.I,
)
ANAPHORIC_STATE_CHANGE_PATTERN = re.compile(
    r"\b(?:it|that one|this place|that place|the destination|the choice|the entry)\b.{0,35}\b(?:could|might|will|can|would)?\s*(?:become|turn|remain|be)\s+(?:active|current|confirmed|canceled|outdated|expired)\b",
    re.I,
)
ANAPHORIC_TARGET_PATTERN = re.compile(
    r"\b(?:it|that one|this place|that place|the destination|the choice|the entry)\b",
    re.I,
)
EXTRA_SLOT_INTRO_PATTERN = re.compile(
    r"\b(?:another|new|other|third)\s+(?:destination|place|location|option)\b",
    re.I,
)
NAMED_SLOT_CANDIDATE_PATTERNS = [
    re.compile(
        r"\b(?:destination|place|location|option)\s+(?:is|was|called|named)?\s*([A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,2})"
    ),
    re.compile(
        r"\b(?:considering|suggesting|maybe|going to|heading to)\s+([A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,2})"
    ),
]
ANAPHORIC_SOURCE_ASSERTION = re.compile(
    r"(?:\b(?:it|that one|this place|that place|the destination|the choice|the entry)\b"
    r".{0,24}\b(?:was|is|came)?\s*(?:recommended|suggested|reported|mentioned)\b"
    r".{0,16}\b(?:by|from)\b)"
    r"|(?:\b(?:someone|a coworker|a friend|they)\b.{0,18}"
    r"\b(?:recommended|suggested|reported|mentioned)\b.{0,14}"
    r"\b(?:it|that one|this place|that place)\b)",
    re.I,
)
ANAPHORIC_CAUSE_ASSERTION = re.compile(
    r"\b(?:it|that one|this place|that place|the destination|the choice|the entry)\b"
    r".{0,30}\b(?:because|due to|caused by|as a result of)\b",
    re.I,
)
ANAPHORIC_TIME_ASSERTION = re.compile(
    r"\b(?:it|that one|this place|that place|the destination|the choice|the entry)\b"
    r".{0,30}\b(?:today|tomorrow|next week|last week|on monday|on tuesday|on wednesday|on thursday|on friday|on saturday|on sunday)\b",
    re.I,
)
ANAPHORIC_TYPE_ASSERTION = re.compile(
    r"\b(?:it|that one|this place|that place|the destination|the choice|the entry)\b"
    r"\s+(?:is|was|would be)\s+(?:a|an|the)\s+"
    r"(?:hotel|library|school|office|studio|campus|hall|room|venue|building|town|city|trail|station)\b",
    re.I,
)
ANAPHORIC_PARTICIPANT_ASSERTION = re.compile(
    r"\b(?:it|that one|this place|that place|the destination|the choice|the entry)\b"
    r".{0,30}\b(?:with my|with our|together with|joined by)\b",
    re.I,
)


def _fact_linked_violations(text: str, value: str) -> tuple[list[str], list[str]]:
    escaped = re.escape(value)
    strong: list[str] = []
    ambiguous: list[str] = []
    source_link = re.compile(
        rf"(?:\b(?:recommended|suggested)\s+(?:that\s+)?\b{escaped}\b)"
        rf"|(?:\b(?:told\s+me\s+about|heard\s+about|read\s+about)\s+\b{escaped}\b)"
        rf"|(?:\b{escaped}\b.{{0,24}}\b(?:was|is)\s+(?:recommended|suggested)\b)"
        rf"|(?:\b(?:someone|a\s+coworker|a\s+friend|they)\s+"
        rf"(?:reported|mentioned)\s+\b{escaped}\b)"
        rf"|(?:\b{escaped}\b.{{0,24}}\b(?:was|is)\s+"
        rf"(?:reported|mentioned)\s+(?:by|from)\b)",
        re.I,
    )
    cause_link = re.compile(
        rf"\b{escaped}\b.{{0,35}}\b(?:because|due to|caused by|as a result of)\b",
        re.I,
    )
    time_link = re.compile(
        rf"\b{escaped}\b.{{0,35}}\b(?:today|tomorrow|next week|last week|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
        re.I,
    )
    type_link = re.compile(
        rf"\b{escaped}\b\s+(?:is|was|would be)\s+(?:a|an|the)\s+"
        r"(?:hotel|library|school|office|studio|campus|hall|room|venue|building|town|city|trail|station)\b",
        re.I,
    )
    participant_link = re.compile(
        rf"\b{escaped}\b.{{0,35}}\b(?:with my|with our|together with|joined by)\b",
        re.I,
    )
    for code, pattern in (
        ("target_fact_source_added", source_link),
        ("target_fact_reason_added", cause_link),
        ("target_fact_time_added", time_link),
        ("target_fact_type_added", type_link),
        ("target_fact_participant_added", participant_link),
    ):
        if pattern.search(text):
            strong.append(code)
    for code, pattern in (
        ("ambiguous_source_language", SOURCE_PATTERN),
        ("ambiguous_reason_language", CAUSE_PATTERN),
        ("ambiguous_time_language", TIME_PATTERN),
        ("ambiguous_type_language", TYPE_PATTERN),
        ("ambiguous_participant_language", PARTICIPANT_PATTERN),
    ):
        if pattern.search(text):
            ambiguous.append(code)
    return strong, ambiguous


def _unknown_proper_name_candidates(
    turns: list[dict[str, Any]], plan: Mapping[str, Any]
) -> list[str]:
    allowed = {
        token
        for value in [*plan["speaker_roster"], *plan["protected_set"]["all_core_values"]]
        for token in re.findall(r"[A-Z][A-Za-z-]+", value)
    }
    common = {
        "I",
        "A",
        "An",
        "The",
        "We",
        "You",
        "He",
        "She",
        "My",
        "Our",
        "Your",
        "So",
        "But",
        "And",
        "That",
        "This",
        "It",
        "Yes",
        "No",
        "Right",
        "Exactly",
        "Good",
        "Okay",
        "Well",
        "Now",
        "Then",
        "Still",
        "By",
        "Instead",
        "Saying",
    }
    found = []
    for turn in turns:
        for token in re.findall(r"\b[A-Z][A-Za-z-]+\b", turn["text"]):
            if token not in allowed and token not in common and token not in found:
                found.append(token)
    return found


def _protected_audit(
    turns: list[dict[str, Any]], plan: Mapping[str, Any], core: Mapping[str, Any]
) -> dict[str, Any]:
    hard = []
    candidates = []
    support_values_by_index = {
        index: support["value"]
        for support in core.get("node_supports") or []
        for index in support.get("support_turn_indices") or []
    }
    support_indices = set(support_values_by_index)
    for index in support_indices:
        text = turns[index - 1]["text"]
        strong, ambiguous = _fact_linked_violations(
            text, support_values_by_index[index]
        )
        hard.extend({"code": code, "turn_index": index} for code in strong)
        candidates.extend(
            {"code": code, "turn_index": index} for code in ambiguous
        )
    for index, turn in enumerate(turns, start=1):
        if NEW_EVENT_PATTERN.search(turn["text"]):
            hard.append({"code": "new_inheritable_world_event", "turn_index": index})
        extra_state_match = EXTRA_STATE_PATTERN.search(turn["text"])
        if extra_state_match:
            if turn["text"].rstrip().endswith("?") or re.search(
                r"\b(?:not|never|no|neither|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|won't|can't|cannot)\b",
                turn["text"],
                re.I,
            ):
                candidates.append(
                    {"code": "negated_or_questioned_extra_state", "turn_index": index}
                )
            else:
                hard.append({"code": "new_state_transition", "turn_index": index})
        anaphoric_state_match = (
            ANAPHORIC_STATE_CHANGE_PATTERN.search(turn["text"])
            if index not in support_indices
            else None
        )
        if anaphoric_state_match:
            if turn["text"].rstrip().endswith("?") or re.search(
                r"\b(?:not|never|no|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|won't|can't|cannot|couldn't|wouldn't)\b",
                turn["text"],
                re.I,
            ):
                candidates.append(
                    {"code": "negated_or_questioned_anaphoric_state", "turn_index": index}
                )
            else:
                hard.append(
                    {"code": "anaphoric_new_state_transition", "turn_index": index}
                )
        distractor_match = re.search(
            r"\b(?:both|either)\b.*\b(?:active|current)\b", turn["text"], re.I
        )
        if distractor_match:
            if turn["text"].rstrip().endswith("?") or re.search(
                r"\b(?:not|never|no|neither|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|won't|can't)\b",
                turn["text"],
                re.I,
            ):
                candidates.append(
                    {"code": "negated_or_questioned_distractor_language", "turn_index": index}
                )
            else:
                hard.append(
                    {"code": "distractor_made_possibly_correct", "turn_index": index}
                )
        extra_slot_match = EXTRA_SLOT_INTRO_PATTERN.search(turn["text"])
        if extra_slot_match:
            prefix = turn["text"][max(0, extra_slot_match.start() - 35) : extra_slot_match.start()]
            if turn["text"].rstrip().endswith("?") or re.search(
                r"\b(?:not|never|no|without|don't|doesn't|didn't|won't|can't)\b",
                prefix,
                re.I,
            ):
                candidates.append(
                    {"code": "negated_or_questioned_extra_slot_language", "turn_index": index}
                )
            else:
                hard.append(
                    {"code": "additional_same_slot_value_introduced", "turn_index": index}
                )
        for pattern in NAMED_SLOT_CANDIDATE_PATTERNS:
            for match in pattern.finditer(turn["text"]):
                candidate = _normalized(match.group(1))
                allowed = {
                    _normalized(value)
                    for value in plan["protected_set"]["all_core_values"]
                }
                if candidate and candidate not in allowed:
                    hard.append(
                        {
                            "code": "additional_named_same_slot_value",
                            "turn_index": index,
                            "candidate": match.group(1),
                        }
                    )
        if index not in support_indices and ANAPHORIC_TARGET_PATTERN.search(turn["text"]):
            for code, pattern in (
                ("anaphoric_target_source_added", ANAPHORIC_SOURCE_ASSERTION),
                ("anaphoric_target_reason_added", ANAPHORIC_CAUSE_ASSERTION),
                ("anaphoric_target_time_added", ANAPHORIC_TIME_ASSERTION),
                ("anaphoric_target_type_added", ANAPHORIC_TYPE_ASSERTION),
                (
                    "anaphoric_target_participant_added",
                    ANAPHORIC_PARTICIPANT_ASSERTION,
                ),
            ):
                if pattern.search(turn["text"]) and not turn["text"].rstrip().endswith("?"):
                    hard.append({"code": code, "turn_index": index})
            for code, pattern in (
                ("ambiguous_anaphoric_source_language", SOURCE_PATTERN),
                ("ambiguous_anaphoric_reason_language", CAUSE_PATTERN),
                ("ambiguous_anaphoric_time_language", TIME_PATTERN),
                ("ambiguous_anaphoric_type_language", TYPE_PATTERN),
                ("ambiguous_anaphoric_participant_language", PARTICIPANT_PATTERN),
            ):
                if pattern.search(turn["text"]):
                    candidates.append({"code": code, "turn_index": index})
    unknown_names = _unknown_proper_name_candidates(turns, plan)
    if unknown_names:
        candidates.append(
            {"code": "unknown_proper_name_candidates", "values": unknown_names}
        )
    node_issue_set = {
        issue
        for support in core.get("node_supports") or []
        for issue in support.get("issues") or []
    }
    if "core_subject_speaker_drift" in node_issue_set:
        hard.append({"code": "identity_binder_or_subject_changed"})
    if node_issue_set.intersection(
        {"required_status_marker_negated", "conflicting_status_marker"}
    ):
        hard.append({"code": "core_status_changed"})
    relation_issues = set((core.get("relation_support") or {}).get("issues") or [])
    if "relation_direction_or_temporal_order_reversed" in relation_issues:
        hard.append({"code": "relation_direction_or_temporal_order_changed"})
    return {
        "pass": not hard,
        "protected_fact_violation": bool(hard),
        "hard_violations": hard,
        "manual_review_candidates": candidates,
        "open_world_absence_requires_manual_review": True,
        "automatic_coverage": [
            "core subject/value/status/direction contradictions",
            "explicit added same-slot value constructions",
            "core-window source/reason/time/type/participant assertions",
            "anaphoric adjacent source/reason/time/type/participant assertions",
            "explicit new events, state changes, and extra-correct-option cues",
        ],
    }


def _qa_shortcut_audit(
    turns: list[dict[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    gold = _normalized(plan["predetermined_qa_reasoning_target"]["compiled_gold"])
    source = _normalized(plan["predetermined_qa_reasoning_target"]["primary_distractor"])
    escaped_gold = r"\s+".join(re.escape(token) for token in gold.split())
    exclusive_gold_patterns = [
        re.compile(
            rf"\bthe\s+(?:correct\s+|final\s+)?answer\s+(?:is|would\s+be)\s+{escaped_gold}\b",
            re.I,
        ),
        re.compile(
            rf"\b{escaped_gold}\b.{{0,35}}\b(?:is|remains)\s+"
            r"(?:the\s+|my\s+)?(?:only|sole|unique|correct|final)\s+"
            r"(?:active\s+|current\s+)?(?:destination|choice|answer|entry)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:only|sole|unique|correct|final)\s+"
            r"(?:active\s+|current\s+)?(?:destination|choice|answer|entry)\s+"
            rf"(?:is|would\s+be|:)\s*{escaped_gold}\b",
            re.I,
        ),
        re.compile(
            rf"\bonly\s+{escaped_gold}\b.{{0,24}}\b(?:active|current)\b",
            re.I,
        ),
        re.compile(
            rf"\b{escaped_gold}\b.{{0,60}}\b(?:the\s+)?only\s+"
            r"(?:active|current)\s+(?:destination|choice|entry)\b",
            re.I,
        ),
    ]
    windows = []
    hard = []
    for index, turn in enumerate(turns):
        windows.append((f"turn:{index + 1}", turn["text"]))
        if index + 1 < len(turns):
            windows.append(
                (
                    f"adjacent:{index + 1}-{index + 2}",
                    turn["text"] + "\n" + turns[index + 1]["text"],
                )
            )
    for window_id, text in windows:
        folded = _normalized(text)
        if gold in folded and any(pattern.search(text) for pattern in exclusive_gold_patterns):
            hard.append(
                {"window": window_id, "reason": "exclusive_gold_summary"}
            )
        if gold in folded and source in folded:
            hard.append(
                {"window": window_id, "reason": "full_reasoning_path_collapsed"}
            )
    return {
        "pass": not hard,
        "qa_shortcut": bool(hard),
        "hard_shortcut_windows": hard,
        "implicit_shortcut_requires_manual_review": True,
    }


def _naturalness_flags(
    turns: list[dict[str, Any]], plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    flags = []
    question_count = sum(turn["text"].rstrip().endswith("?") for turn in turns)
    if len(turns) >= 4 and question_count >= 2:
        flags.append({"code": "interview_style_candidate", "question_turns": question_count})
    meta_pattern = re.compile(
        r"\b(?:graph|node|slot|relation|field|database|to summarize|status is)\b",
        re.I,
    )
    meta_turns = [
        index + 1 for index, turn in enumerate(turns) if meta_pattern.search(turn["text"])
    ]
    if meta_turns:
        flags.append({"code": "database_or_label_recital_candidate", "turns": meta_turns})
    for status in plan["protected_set"]["all_core_statuses"]:
        if "_" in status and any(status in turn["text"] for turn in turns):
            flags.append({"code": "literal_internal_status_label", "status": status})
    acknowledgements = Counter(
        _normalized(turn["text"]).strip(".!?")
        for turn in turns
        if len(_normalized(turn["text"]).split()) <= 5
    )
    repeated = [text for text, count in acknowledgements.items() if count > 1]
    if repeated:
        flags.append({"code": "repeated_low_information_acknowledgement", "texts": repeated})
    value_repetitions = {
        value: _normalized("\n".join(turn["text"] for turn in turns)).count(_normalized(value))
        for value in plan["protected_set"]["all_core_values"]
    }
    if any(count > 1 for count in value_repetitions.values()):
        flags.append({"code": "repeated_core_value_candidate", "counts": value_repetitions})
    return flags


def _machine_audit(
    raw_response: str, plan: Mapping[str, Any]
) -> dict[str, Any]:
    candidate, raw_issues = _strict_raw_candidate(raw_response)
    format_report, turns = _format_audit(candidate, raw_issues, plan)
    core_report = _core_truth_audit(turns, plan)
    protected_report = _protected_audit(turns, plan, core_report)
    shortcut_report = _qa_shortcut_audit(turns, plan)
    return {
        "schema_version": SCHEMA_VERSION,
        "format_gate": format_report,
        "core_truth_gate": core_report,
        "protected_set_gate": protected_report,
        "qa_shortcut_gate": shortcut_report,
        "naturalness_flags": _naturalness_flags(turns, plan),
        "parsed_strict_response": candidate,
        "deterministic_semantic_gate_pass": (
            format_report["pass"]
            and core_report["pass"]
            and protected_report["pass"]
            and shortcut_report["pass"]
        ),
        "manual_semantic_and_naturalness_review_required": True,
    }


def _command_preflight(_: argparse.Namespace) -> int:
    report = _preflight(output_fresh_required=True)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["pass"] else 2


def _command_init(_: argparse.Namespace) -> int:
    if OUTPUT_DIR.exists():
        raise CanaryError(f"refusing to reuse run234 directory: {OUTPUT_DIR}")
    preflight = _preflight(output_fresh_required=True)
    if not preflight["pass"]:
        raise CanaryError(f"run234 preflight failed: {preflight['checks']}")
    plans, parent_texts = _build_truth_plans()
    prompts = []
    for plan in plans:
        prompt = _api_prompt(plan)
        leak = _parent_prompt_leak_audit(
            plan, prompt, parent_texts[plan["truth_plan_id"]]
        )
        if not leak["pass"]:
            raise CanaryError(f"parent leaked into {plan['truth_plan_id']} prompt")
        prompts.append(
            {
                "truth_plan_id": plan["truth_plan_id"],
                "prompt": prompt,
                "prompt_sha256": _sha256_json(prompt),
                "parent_prompt_leak_audit": leak,
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "created_unix": int(time.time()),
        "output_dir": str(OUTPUT_DIR),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "canary_count": CANARY_COUNT,
        "calls_per_canary": 1,
        "max_internal_retries": 0,
        "max_external_retries": 0,
        "repair_calls": 0,
        "continuation_calls": 0,
        "qa_api_calls": 0,
        "graph_extraction_api_calls": 0,
        "training_writes": 0,
        "source_parent_file": str(SOURCE_PARENT_FILE),
        "source_parent_file_sha256": SOURCE_PARENT_FILE_SHA256,
        "preflight": preflight,
    }
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "truth_plans": plans,
        "api_prompts": prompts,
    }
    frozen["frozen_bundle_sha256"] = _sha256_json(frozen)
    state = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "ready_for_distinct_canary",
        "frozen_bundle_sha256": frozen["frozen_bundle_sha256"],
        "completed_canary_count": 0,
        "scene_api_call_intents": 0,
        "scene_api_call_results": 0,
        "qa_api_calls": 0,
        "graph_api_calls": 0,
        "repair_calls": 0,
        "continuation_calls": 0,
        "training_files_written": 0,
    }
    _atomic_write_json(_paths()["manifest"], manifest)
    _atomic_write_json(_paths()["plans"], frozen)
    _atomic_write_json(_paths()["state"], state)
    print(
        json.dumps(
            {
                "initialized": True,
                "run_id": RUN_ID,
                "truth_plan_count": len(plans),
                "api_calls": 0,
                "next_truth_plan_id": plans[0]["truth_plan_id"],
            },
            sort_keys=True,
        )
    )
    return 0


def _load_initialized() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not OUTPUT_DIR.is_dir():
        raise CanaryError("run234 is not initialized")
    frozen = _read_json(_paths()["plans"])
    state = _read_json(_paths()["state"])
    unhashed = copy.deepcopy(frozen)
    declared = unhashed.pop("frozen_bundle_sha256", None)
    if declared != _sha256_json(unhashed):
        raise CanaryError("frozen truth-plan bundle hash mismatch")
    if state.get("frozen_bundle_sha256") != declared:
        raise CanaryError("state points to a different frozen bundle")
    journal = _journal_audit()
    for intent in journal["intents"]:
        ordinal = int(intent.get("call_ordinal") or 0)
        if not 1 <= ordinal <= CANARY_COUNT:
            raise CanaryError("API journal ordinal is outside the frozen plan")
        plan = frozen["truth_plans"][ordinal - 1]
        prompt_row = frozen["api_prompts"][ordinal - 1]
        expected_call_id = f"run234_scene_call_{ordinal:02d}_{plan['truth_plan_id']}"
        expected = {
            "call_id": expected_call_id,
            "truth_plan_id": plan["truth_plan_id"],
            "truth_plan_sha256": plan["truth_plan_sha256"],
            "prompt_sha256": prompt_row["prompt_sha256"],
            "model": MODEL,
            "temperature": TEMPERATURE,
            "call_kind": ALLOWED_CALL_KIND,
            "max_internal_retries": 0,
            "max_external_retries": 0,
            "repair": False,
            "continuation": False,
        }
        mismatches = {
            key: {"expected": value, "actual": intent.get(key)}
            for key, value in expected.items()
            if intent.get(key) != value
        }
        if mismatches:
            raise CanaryError(f"API intent diverges from frozen plan: {mismatches}")
        if intent.get("prompt") != prompt_row["prompt"]:
            raise CanaryError("journaled prompt diverges from frozen prompt")
        if intent.get("system_prompt") != SYSTEM_PROMPT:
            raise CanaryError("journaled system prompt changed")
    state_intents = int(state.get("scene_api_call_intents") or 0)
    state_results = int(state.get("scene_api_call_results") or 0)
    state_completed = int(state.get("completed_canary_count") or 0)
    if state_intents > journal["intent_count"]:
        raise CanaryError("state claims an API intent absent from the journal")
    if state_results > journal["result_count"] or state_completed > journal["result_count"]:
        raise CanaryError("state claims an API result absent from the journal")
    if journal["intent_count"] - state_intents > 1:
        raise CanaryError("state is more than one API intent behind the journal")
    if journal["result_count"] - min(state_results, state_completed) > 1:
        raise CanaryError("state is more than one API result behind the journal")
    return frozen, state, journal


def _command_dispatch_next(args: argparse.Namespace) -> int:
    if not args.execute_live:
        raise CanaryError("dispatch-next-canary requires --execute-live")
    frozen, state, journal = _load_initialized()
    if journal["unmatched_intent_ids"]:
        raise CanaryError(
            "terminal indeterminate call exists; reissue is forbidden: "
            f"{journal['unmatched_intent_ids']}"
        )
    if journal["result_count"] >= CANARY_COUNT:
        raise CanaryError("all five distinct canaries have already been dispatched")
    ordinal = journal["result_count"] + 1
    plan = frozen["truth_plans"][ordinal - 1]
    prompt_row = frozen["api_prompts"][ordinal - 1]
    prompt = prompt_row["prompt"]
    api_args = _api_args()
    if not (
        api_args.challenger_api_key
        and api_args.challenger_base_url
        and api_args.challenger_model == MODEL
    ):
        raise CanaryError("live API configuration changed after run234 initialization")
    # Force all local request construction errors to occur before the call intent.
    native.build_json_request(
        api_args,
        prompt,
        SYSTEM_PROMPT,
        None,
        1800,
        TEMPERATURE,
    )
    call_id = f"run234_scene_call_{ordinal:02d}_{plan['truth_plan_id']}"
    intent_record = _append_journal(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "call_intent",
            "call_id": call_id,
            "call_ordinal": ordinal,
            "truth_plan_id": plan["truth_plan_id"],
            "truth_plan_sha256": plan["truth_plan_sha256"],
            "prompt_sha256": prompt_row["prompt_sha256"],
            "system_prompt": SYSTEM_PROMPT,
            "prompt": prompt,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "call_kind": ALLOWED_CALL_KIND,
            "max_internal_retries": 0,
            "max_external_retries": 0,
            "repair": False,
            "continuation": False,
            "unix_time": int(time.time()),
        }
    )
    state.update(
        {
            "scene_api_call_intents": ordinal,
            "scene_api_call_results": journal["result_count"],
            "completed_canary_count": journal["result_count"],
            "status": "scene_api_call_inflight",
            "inflight_call_id": call_id,
            "journal_chain_head": intent_record["record_sha256"],
        }
    )
    _atomic_write_json(_paths()["state"], state)
    started = time.monotonic()
    try:
        candidate, meta = native.call_raw_json_api(
            api_args,
            prompt,
            system_prompt=SYSTEM_PROMPT,
            required_keys=("selected_expansion_ids", "turns"),
            call_kind=ALLOWED_CALL_KIND,
            max_tokens_override=1800,
            max_retries_override=0,
            temperature_override=TEMPERATURE,
            include_raw_response=True,
        )
        physical_dispatch_proven = int(meta.get("attempts") or 0) == 1
    except subprocess.TimeoutExpired as exc:
        candidate = None
        meta = {
            "used": False,
            "call_kind": ALLOWED_CALL_KIND,
            "attempts": 1,
            "max_retries_override": 0,
            "request_dispatch_invoked": True,
            "reason": "uncaught_transport_or_client_exception",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        physical_dispatch_proven = True
    except Exception as exc:
        candidate = None
        meta = {
            "used": False,
            "call_kind": ALLOWED_CALL_KIND,
            "attempts": 0,
            "max_retries_override": 0,
            "request_dispatch_invoked": False,
            "reason": "uncaught_exception_without_transport_proof",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        physical_dispatch_proven = False
    meta = copy.deepcopy(meta)
    meta["request_dispatch_invoked"] = physical_dispatch_proven
    raw_response = _raw_response_from_meta(meta)
    machine = _machine_audit(raw_response, plan)
    result_record = _append_journal(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "call_result",
            "call_id": call_id,
            "call_ordinal": ordinal,
            "truth_plan_id": plan["truth_plan_id"],
            "truth_plan_sha256": plan["truth_plan_sha256"],
            "prompt_sha256": prompt_row["prompt_sha256"],
            "helper_candidate": candidate,
            "raw_response": raw_response,
            "raw_response_sha256": _sha256_text(raw_response),
            "api_meta": meta,
            "request_dispatch_invoked": physical_dispatch_proven,
            "max_internal_retries": 0,
            "max_external_retries": 0,
            "repair": False,
            "continuation": False,
            "machine_audit": machine,
            "latency_ms": int(round((time.monotonic() - started) * 1000)),
            "manual_text_edits": 0,
            "manual_review_required": True,
        }
    )
    journal = _journal_audit()
    state.update(
        {
            "completed_canary_count": journal["result_count"],
            "scene_api_call_intents": journal["intent_count"],
            "scene_api_call_results": journal["result_count"],
            "journal_chain_head": journal["chain_head"],
            "inflight_call_id": None,
            "status": (
                "awaiting_manual_audit"
                if journal["result_count"] == CANARY_COUNT
                else "ready_for_distinct_canary"
            ),
        }
    )
    _atomic_write_json(_paths()["state"], state)
    print(
        json.dumps(
            {
                "truth_plan_id": plan["truth_plan_id"],
                "call_ordinal": ordinal,
                "physical_call_dispatched": physical_dispatch_proven,
                "api_used": bool(meta.get("used")),
                "api_attempts": meta.get("attempts"),
                "format_pass": machine["format_gate"]["pass"],
                "core_truth_pass": machine["core_truth_gate"]["pass"],
                "protected_violation_auto": machine["protected_set_gate"][
                    "protected_fact_violation"
                ],
                "qa_shortcut_auto": machine["qa_shortcut_gate"]["qa_shortcut"],
                "turn_count": machine["format_gate"]["turn_count"],
                "tokenizer_tokens": machine["format_gate"]["tokenizer_tokens"],
                "completed_canary_count": journal["result_count"],
                "next_status": state["status"],
                "result_record_sha256": result_record["record_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _machine_summary(
    frozen: Mapping[str, Any], journal: Mapping[str, Any]
) -> dict[str, Any]:
    result_by_plan = {
        row["truth_plan_id"]: row for row in journal["results"]
    }
    rows = []
    for plan, prompt_row in zip(frozen["truth_plans"], frozen["api_prompts"]):
        result = result_by_plan.get(plan["truth_plan_id"])
        if result is None:
            rows.append(
                {
                    "truth_plan_id": plan["truth_plan_id"],
                    "complete": False,
                }
            )
            continue
        recomputed_machine = _machine_audit(result["raw_response"], plan)
        call_time_machine = copy.deepcopy(result["machine_audit"])
        rows.append(
            {
                "truth_plan_id": plan["truth_plan_id"],
                "complete": True,
                "truth_plan": copy.deepcopy(plan),
                "complete_prompt": copy.deepcopy(prompt_row["prompt"]),
                "system_prompt": SYSTEM_PROMPT,
                "raw_response": result["raw_response"],
                "raw_response_sha256": result["raw_response_sha256"],
                "api_meta": copy.deepcopy(result["api_meta"]),
                "machine_audit": recomputed_machine,
                "machine_audit_at_call_time": call_time_machine,
                "machine_audit_recomputed_from_immutable_raw": True,
                "machine_audit_changed_since_call": (
                    _sha256_json(recomputed_machine) != _sha256_json(call_time_machine)
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "canaries": rows,
        "journal_intent_count": journal["intent_count"],
        "journal_result_count": journal["result_count"],
        "unmatched_intent_ids": journal["unmatched_intent_ids"],
    }


def _command_status(_: argparse.Namespace) -> int:
    frozen, state, journal = _load_initialized()
    result = {
        "run_id": RUN_ID,
        "status": state["status"],
        "completed_canary_count": journal["result_count"],
        "scene_call_intent_count": journal["intent_count"],
        "unmatched_intent_ids": journal["unmatched_intent_ids"],
        "completed_truth_plan_ids": [
            row["truth_plan_id"] for row in journal["results"]
        ],
        "next_truth_plan_id": (
            frozen["truth_plans"][journal["result_count"]]["truth_plan_id"]
            if journal["result_count"] < CANARY_COUNT
            else None
        ),
        "retry_commands_available": False,
        "qa_api_calls": state["qa_api_calls"],
        "graph_api_calls": state["graph_api_calls"],
        "repair_calls": state["repair_calls"],
        "continuation_calls": state["continuation_calls"],
        "training_files_written": state["training_files_written"],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _command_finalize(args: argparse.Namespace) -> int:
    frozen, state, journal = _load_initialized()
    existing_outputs = [
        str(path)
        for path in (_paths()["manual"], _paths()["machine"], _paths()["report"])
        if path.exists()
    ]
    if existing_outputs:
        raise CanaryError(
            f"run234 final audit is immutable and already exists: {existing_outputs}"
        )
    if journal["unmatched_intent_ids"]:
        raise CanaryError("cannot finalize an indeterminate physical call")
    if journal["intent_count"] != CANARY_COUNT or journal["result_count"] != CANARY_COUNT:
        raise CanaryError("finalize requires exactly five call intents and five results")
    manual = _read_json(args.manual_audit)
    rows = manual.get("canaries") if isinstance(manual, dict) else None
    if not isinstance(rows, list) or len(rows) != CANARY_COUNT:
        raise CanaryError("manual audit must contain exactly five canaries")
    manual_by_plan = {
        row.get("truth_plan_id"): row for row in rows if isinstance(row, dict)
    }
    expected_ids = {plan["truth_plan_id"] for plan in frozen["truth_plans"]}
    if set(manual_by_plan) != expected_ids:
        raise CanaryError("manual audit truth-plan coverage mismatch")
    result_by_plan = {
        row["truth_plan_id"]: row for row in journal["results"]
    }
    for plan_id, row in manual_by_plan.items():
        result = result_by_plan[plan_id]
        required_keys = {
            "truth_plan_id",
            "raw_response_sha256",
            "semantic_clean",
            "natural_dialogue",
            "protected_fact_violation",
            "qa_shortcut",
            "manual_text_edits",
            "failure_reasons",
            "reviewer",
        }
        if set(row) != required_keys:
            raise CanaryError(f"manual audit schema mismatch for {plan_id}")
        if row["raw_response_sha256"] != result["raw_response_sha256"]:
            raise CanaryError(f"manual audit raw-response hash mismatch for {plan_id}")
        if row["manual_text_edits"] != 0:
            raise CanaryError(f"manual text edits are forbidden for {plan_id}")
        for field in (
            "semantic_clean",
            "natural_dialogue",
            "protected_fact_violation",
            "qa_shortcut",
        ):
            if not isinstance(row[field], bool):
                raise CanaryError(f"manual {field} must be boolean for {plan_id}")
        if not isinstance(row["failure_reasons"], list):
            raise CanaryError(f"manual failure_reasons must be a list for {plan_id}")
    machine_summary = _machine_summary(frozen, journal)
    final_canaries = []
    semantic_clean_count = 0
    natural_count = 0
    protected_violation_count = 0
    shortcut_count = 0
    for row in machine_summary["canaries"]:
        plan_id = row["truth_plan_id"]
        review = manual_by_plan[plan_id]
        machine = row["machine_audit"]
        protected_violation = bool(
            machine["protected_set_gate"]["protected_fact_violation"]
            or review["protected_fact_violation"]
        )
        shortcut = bool(
            machine["qa_shortcut_gate"]["qa_shortcut"]
            or review["qa_shortcut"]
        )
        semantic_clean = bool(
            machine["format_gate"]["pass"]
            and machine["core_truth_gate"]["pass"]
            and review["semantic_clean"]
            and not protected_violation
            and not shortcut
        )
        natural = bool(review["natural_dialogue"])
        semantic_clean_count += int(semantic_clean)
        natural_count += int(natural)
        protected_violation_count += int(protected_violation)
        shortcut_count += int(shortcut)
        plan = row["truth_plan"]
        parsed = machine.get("parsed_strict_response") or {}
        failure_reasons = []
        failure_reasons.extend(
            f"format:{issue}" for issue in machine["format_gate"]["issues"]
        )
        failure_reasons.extend(
            f"core_truth:{issue}" for issue in machine["core_truth_gate"]["issues"]
        )
        failure_reasons.extend(
            f"protected:{item['code']}"
            for item in machine["protected_set_gate"]["hard_violations"]
        )
        failure_reasons.extend(
            f"qa_shortcut:{item['reason']}"
            for item in machine["qa_shortcut_gate"]["hard_shortcut_windows"]
        )
        failure_reasons.extend(str(item) for item in review["failure_reasons"])
        failure_reasons = list(dict.fromkeys(failure_reasons))
        final_canaries.append(
            {
                "truth_plan_id": plan_id,
                "parent_lineage_provenance": plan["parent_provenance"],
                "frozen_node_claims": plan["node_claims"],
                "frozen_relation": plan["relation"],
                "protected_set": plan["protected_set"],
                "safe_expansion_pool": plan["safe_expansion_pool"],
                "selected_expansion_ids": parsed.get("selected_expansion_ids"),
                "complete_system_prompt": row["system_prompt"],
                "complete_prompt": row["complete_prompt"],
                "complete_raw_response": row["raw_response"],
                "raw_response_sha256": row["raw_response_sha256"],
                "turn_count": machine["format_gate"]["turn_count"],
                "tokenizer_tokens": machine["format_gate"]["tokenizer_tokens"],
                "format_gate": machine["format_gate"],
                "core_truth_gate": machine["core_truth_gate"],
                "protected_set_gate": machine["protected_set_gate"],
                "qa_shortcut_gate": machine["qa_shortcut_gate"],
                "naturalness_flags": machine["naturalness_flags"],
                "machine_audit_at_call_time": row["machine_audit_at_call_time"],
                "machine_audit_recomputed_from_immutable_raw": row[
                    "machine_audit_recomputed_from_immutable_raw"
                ],
                "machine_audit_changed_since_call": row[
                    "machine_audit_changed_since_call"
                ],
                "manual_audit": review,
                "final_semantic_clean": semantic_clean,
                "final_natural_dialogue": natural,
                "final_protected_fact_violation": protected_violation,
                "final_qa_shortcut": shortcut,
                "failure_reasons": failure_reasons,
            }
        )
    call_attempts = [
        int((row.get("api_meta") or {}).get("attempts") or 0)
        for row in journal["results"]
    ]
    physical_call_count = sum(
        int(row.get("request_dispatch_invoked") is True)
        for row in journal["results"]
    )
    physical_calls_exact = (
        journal["intent_count"] == 5
        and journal["result_count"] == 5
        and len({row["truth_plan_id"] for row in journal["results"]}) == 5
        and call_attempts == [1, 1, 1, 1, 1]
        and all(
            (row.get("api_meta") or {}).get("max_retries_override") == 0
            for row in journal["results"]
        )
        and all(
            row.get("request_dispatch_invoked") is True
            and row.get("max_internal_retries") == 0
            and row.get("max_external_retries") == 0
            and row.get("repair") is False
            and row.get("continuation") is False
            for row in journal["results"]
        )
        and all(
            row.get("call_kind") == ALLOWED_CALL_KIND
            for row in journal["intents"]
        )
    )
    source_hash_unchanged = (
        _sha256_file(SOURCE_PARENT_FILE) == SOURCE_PARENT_FILE_SHA256
    )
    current_plans, _ = _build_truth_plans()
    graph_hashes_before = {
        plan["parent_provenance"]["lineage_id"]: plan["parent_provenance"][
            "parent_graph_sha256"
        ]
        for plan in frozen["truth_plans"]
    }
    graph_hashes_after = {
        plan["parent_provenance"]["lineage_id"]: plan["parent_provenance"][
            "parent_graph_sha256"
        ]
        for plan in current_plans
    }
    source_graph_hashes_unchanged = graph_hashes_before == graph_hashes_after
    manual_text_edits = sum(int(row["manual_text_edits"]) for row in rows)
    overall_pass = (
        semantic_clean_count >= 4
        and natural_count >= 4
        and protected_violation_count == 0
        and shortcut_count == 0
        and manual_text_edits == 0
        and physical_calls_exact
        and state["qa_api_calls"] == 0
        and state["graph_api_calls"] == 0
        and state["repair_calls"] == 0
        and state["continuation_calls"] == 0
        and state["training_files_written"] == 0
        and source_hash_unchanged
        and source_graph_hashes_unchanged
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "final_status": "PASS" if overall_pass else "FAIL",
        "canaries": final_canaries,
        "summary": {
            "physical_scene_api_calls": physical_call_count,
            "physical_scene_api_calls_exactly_proved": physical_calls_exact,
            "api_attempts_by_canary": call_attempts,
            "one_call_semantic_clean_count": semantic_clean_count,
            "natural_dialogue_pass_count": natural_count,
            "protected_fact_violation_count": protected_violation_count,
            "qa_shortcut_count": shortcut_count,
            "manual_text_edits": manual_text_edits,
            "overall_threshold_pass": overall_pass,
            "internal_retry_count": 0,
            "external_retry_count": 0,
            "repair_count": state["repair_calls"],
            "continuation_count": state["continuation_calls"],
            "qa_api_call_count": state["qa_api_calls"],
            "graph_extraction_api_call_count": state["graph_api_calls"],
            "training_files_written": state["training_files_written"],
            "source_parent_file_hash_unchanged": source_hash_unchanged,
            "source_parent_file_sha256_before": SOURCE_PARENT_FILE_SHA256,
            "source_parent_file_sha256_after": _sha256_file(SOURCE_PARENT_FILE),
            "source_graph_hashes_unchanged": source_graph_hashes_unchanged,
            "source_graph_hashes_before": graph_hashes_before,
            "source_graph_hashes_after": graph_hashes_after,
            "automatic_next_stage_started": False,
            "machine_audit_reclassification_count": sum(
                int(row["machine_audit_changed_since_call"])
                for row in machine_summary["canaries"]
            ),
        },
    }
    report["report_sha256"] = _sha256_json(report)
    _atomic_write_json(_paths()["manual"], manual)
    _atomic_write_json(_paths()["machine"], machine_summary)
    _atomic_write_json(_paths()["report"], report)
    state["status"] = report["final_status"]
    state["final_status"] = report["final_status"]
    state["report_sha256"] = report["report_sha256"]
    _atomic_write_json(_paths()["state"], state)
    print(json.dumps(report["summary"] | {"final_status": report["final_status"]}, sort_keys=True))
    return 0 if overall_pass else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight").set_defaults(handler=_command_preflight)
    subparsers.add_parser("init").set_defaults(handler=_command_init)
    dispatch = subparsers.add_parser("dispatch-next-canary")
    dispatch.add_argument("--execute-live", action="store_true")
    dispatch.set_defaults(handler=_command_dispatch_next)
    subparsers.add_parser("status").set_defaults(handler=_command_status)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manual-audit", type=Path, required=True)
    finalize.set_defaults(handler=_command_finalize)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return int(args.handler(args))
    except CanaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
