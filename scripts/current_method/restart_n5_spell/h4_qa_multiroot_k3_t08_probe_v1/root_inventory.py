"""Read-only inventory and deterministic name-cleaning views for expansion roots."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from ..core import canonical_json, sha256_bytes, sha256_file


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")

# These are existing completed expansions.  The probe never edits them.  The
# cleaned view is written below the new probe run so no historical artifact is
# overwritten.
DEFAULT_ROOT_SOURCES = (
    RUN_BASE / "conversation_chronicles_episode_178392_round2_career_trial_followup_scene_release_v1_live_2" / "expanded_dialogue_round2.json",
    RUN_BASE / "conversation_chronicles_episode_91716_named_speakers_event_bucket_only_expansion_v1_live_1" / "expanded_dialogue.json",
    RUN_BASE / "conversation_chronicles_episode_13836_memory_event_candidate_isolation_probe_v1_recovery_2" / "expanded_dialogue.json",
    RUN_BASE / "conversation_chronicles_episode_146700_simple_append_memorax_fullr1_durable_devdiv_local_anchor_span_probe_v1_live_1" / "expanded_dialogue.json",
    RUN_BASE / "conversation_chronicles_episode_34406_memory_event_continuation_contract_v2_live_1" / "expanded_dialogue.json",
)

# The older generic expansions use speaker aliases such as "Neighbors A".
# This deterministic per-root map keeps the dialogue facts unchanged while
# satisfying the named-speaker input requirement for this new probe.
SPEAKER_RENAMES: dict[str, dict[str, str]] = {
    "cc-train-episode-178392": {"Maya": "Maya", "Jordan": "Jordan"},
    "cc-train-episode-91716": {"Nora": "Nora", "Caleb": "Caleb"},
    "cc-train-episode-13836": {"Neighbors A": "Amelia", "Neighbors B": "Victor"},
    "cc-train-episode-146700": {"Co-workers A": "Lucas", "Co-workers B": "Harper"},
    "cc-train-episode-34406": {"Classmates A": "Sofia", "Classmates B": "Leo"},
    "cc-train-episode-41625": {"Classmates A": "Aria", "Classmates B": "Owen"},
    "cc-train-episode-61555": {"Co-workers A": "Mia", "Co-workers B": "Ethan"},
}


def load_expanded(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("turns"), list):
        raise ValueError(f"expanded_dialogue_invalid:{path}")
    return value


def clean_named_view(expanded: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    root_id = str(expanded.get("root_id") or "")
    mapping = dict(SPEAKER_RENAMES.get(root_id) or {})
    cleaned = copy.deepcopy(dict(expanded))
    turns = []
    for raw in expanded.get("turns") or []:
        row = dict(raw)
        speaker = str(row.get("speaker") or "")
        replacement = mapping.get(speaker, speaker)
        row["speaker"] = replacement
        text = str(row.get("text") or "")
        for old, new in mapping.items():
            text = text.replace(old, new)
        row["text"] = text
        turns.append(row)
    cleaned["turns"] = turns
    cleaned["speaker_name_cleaning"] = {
        "cleaning_version": "h4_named_speaker_view_v1",
        "mapping": mapping,
        "source_dialogue_sha256": sha256_bytes(canonical_json(dict(expanded)).encode()),
    }
    cleaned["turn_count"] = len(turns)
    speakers = [str(row.get("speaker") or "") for row in turns]
    if any(re.search(r"(?:^| )(?:A|B)$", speaker) for speaker in speakers):
        raise ValueError(f"speaker_alias_remaining:{root_id}")
    mapped_names = [str(name).strip() for name in mapping.values()]
    if any(not name for name in mapped_names) or len(mapped_names) != len(set(mapped_names)):
        raise ValueError(f"speaker_name_mapping_not_unique:{root_id}")
    return cleaned, mapping


def inventory_sources(paths: tuple[Path, ...] | None = None) -> list[dict[str, Any]]:
    paths = DEFAULT_ROOT_SOURCES if paths is None else paths
    rows: list[dict[str, Any]] = []
    seen_roots: set[str] = set()
    for path in paths:
        row: dict[str, Any] = {
            "source_path": str(path),
            "source_exists": path.is_file(),
            "source_sha256": sha256_file(path) if path.is_file() else None,
            "eligible": False,
            "eligibility_errors": [],
        }
        if not path.is_file():
            row["eligibility_errors"].append("source_missing")
            rows.append(row)
            continue
        try:
            expanded = load_expanded(path)
            root_id = str(expanded.get("root_id") or "")
            row.update({
                "root_id": root_id,
                "turn_count": len(expanded.get("turns") or []),
                "qa_fact_source": expanded.get("qa_fact_source"),
                "memory_is_qa_fact_source": expanded.get("memory_is_qa_fact_source"),
                "unique_root_id": bool(root_id and root_id not in seen_roots),
            })
            if not root_id:
                row["eligibility_errors"].append("root_id_missing")
            if root_id in seen_roots:
                row["eligibility_errors"].append("duplicate_root_id")
            seen_roots.add(root_id)
            if not expanded.get("turns"):
                row["eligibility_errors"].append("turns_missing")
            cleaned, mapping = clean_named_view(expanded)
            row["speaker_name_mapping"] = mapping
            row["cleaned_dialogue_sha256"] = sha256_bytes(canonical_json(cleaned).encode())
            row["cleaned_speakers"] = sorted(set(str(x.get("speaker")) for x in cleaned["turns"]))
        except Exception as error:
            row["eligibility_errors"].append(f"{type(error).__name__}:{error}")
        row["eligible"] = not row["eligibility_errors"]
        rows.append(row)
    return rows
