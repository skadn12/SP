"""Build a provenance-preserving, minimal model-visible Realizer input.

This module deliberately performs no model or network operation.  Historical
summaries and rejected/previous raw dialogue are returned in a separate
provenance object and are never copied into the model-visible payload.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from . import PROMPT_VERSION, VERSION

VISIBLE_FIELDS = {
    "goal", "required_transition", "actor_bindings", "required_participants",
    "interaction_beats", "dependencies", "minimal_current_state",
    "already_established", "do_not_replay", "open_transition",
    "structured_actual_progress", "selected_candidate_ref", "verified_track_ref",
}
REDACTED_FIELDS = {
    "memory_event_content", "supporting_sources", "verified_current_state",
    "before_state", "retained_actual_prefix",
}


def _sha(value) -> str:
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def _status(value, status, provenance):
    return {"fact": value, "status": status, "provenance": provenance}


def structured_actual_progress(*, completed_actions=(), confirmed_decisions=(),
                               confirmed_arrangements=(), unresolved_targets=(),
                               prerequisite_status=(), source_turn_refs=(),
                               source_sha256=None, extraction_method="manual"):
    """Return conservative progress data; no semantic extraction is attempted."""
    if extraction_method not in {"manual", "rule_based"}:
        raise ValueError("progress_extraction_method_must_be_manual_or_rule_based")
    def rows(values, status="observed"):
        return [{"fact": str(v["fact"] if isinstance(v, dict) else v,
                              ), "status": v.get("status", status) if isinstance(v, dict) else status,
                 "provenance": v.get("provenance", "accepted_generated_dialogue") if isinstance(v, dict) else "accepted_generated_dialogue"}
                for v in values]
    return {
        "completed_actions": rows(completed_actions),
        "confirmed_decisions": rows(confirmed_decisions),
        "confirmed_arrangements": rows(confirmed_arrangements),
        "unresolved_targets": rows(unresolved_targets, "unknown"),
        "prerequisite_status": rows(prerequisite_status),
        "source_turn_refs": list(source_turn_refs),
        "source_sha256": source_sha256,
        "extraction_method": extraction_method,
    }


def minimal_current_state(scene, *, source_ref="selected_plan"):
    """Make navigation state without copying any free-form historical summary."""
    transition = scene.get("required_transition") or scene.get("new_development") or ""
    event_display = scene.get("memory_event_display")
    return {
        "scene_anchor": {
            "candidate_ref": scene.get("selected_candidate_ref"),
            "event_key": scene.get("memory_event_key"),
            "event_display": event_display,
            "verified_track_ref": scene.get("verified_track_ref"),
            "status": "navigation_only",
            "provenance": source_ref + ".memory_event_identity",
        },
        "established_facts": [],
        "planned_but_not_observed": [_status(transition, "planned", source_ref + ".required_transition")] if transition else [],
        "unknowns": [_status("Whether the planned transition has already happened", "unknown", source_ref + ".no_actual_evidence")],
    }


def project_scene(scene, *, actual_progress=None, source_ref="selected_plan"):
    """Project one planner scene while retaining typed dependency identity."""
    deps = []
    for dep in scene.get("dependencies") or []:
        deps.append({k: dep.get(k) for k in ("phase_id", "predecessor_candidate_ref", "relation_kind", "condition")})
    visible = {
        "selected_candidate_ref": scene.get("selected_candidate_ref"),
        "verified_track_ref": scene.get("verified_track_ref"),
        "goal": scene.get("goal"),
        "required_transition": scene.get("required_transition"),
        "actor_bindings": deepcopy(scene.get("actor_bindings") or []),
        "required_participants": list(scene.get("required_participants") or scene.get("participants") or []),
        "interaction_beats": list(scene.get("interaction_beats") or []),
        "dependencies": deps,
        "minimal_current_state": minimal_current_state(scene, source_ref=source_ref),
        "already_established": [],
        "do_not_replay": [
            "Any historical dialogue or summary not represented in this input.",
            "Previously proposed intentions unless actual progress is evidenced here.",
        ],
        "open_transition": _status(scene.get("required_transition"), "planned", source_ref + ".required_transition"),
        "structured_actual_progress": deepcopy(actual_progress or structured_actual_progress()),
    }
    return visible


def build_provenance(scene, *, prior_dialogue=(), rejected_output=None, source_ref="selected_plan"):
    """Keep excluded source material outside the payload sent to the model."""
    return {
        "adapter_version": VERSION,
        "source_ref": source_ref,
        "excluded_fields": sorted(REDACTED_FIELDS),
        "original_scene": deepcopy(scene),
        "prior_dialogue": deepcopy(list(prior_dialogue)),
        "rejected_output": deepcopy(rejected_output),
        "excluded_content_sha256": {field: _sha(scene.get(field)) for field in REDACTED_FIELDS if field in scene},
        "prior_dialogue_sha256": _sha(list(prior_dialogue)),
    }


def render_prompt(scenes, *, actual_progress_by_ref=None):
    actual_progress_by_ref = actual_progress_by_ref or {}
    projected = []
    for scene in scenes:
        ref = scene.get("selected_candidate_ref")
        projected.append(project_scene(scene, actual_progress=actual_progress_by_ref.get(ref), source_ref="selected_plan." + str(ref)))
    payload = {
        "planned_phases_minimal_state": projected,
        "adapter_version": VERSION,
        "input_policy": {
            "historical_dialogue_included": False,
            "historical_freeform_summaries_included": False,
            "failed_outputs_included": False,
            "actual_progress_must_be_evidence_backed": True,
        },
    }
    instructions = (
        "You are the Natural Realizer. Output only new dialogue for the supplied scenes. "
        "Use minimal_current_state as navigation, not as proof. Distinguish observed, planned, "
        "and unknown. Do not replay any historical dialogue or prior intention. Establish only "
        "progress supported by this scene and structured_actual_progress. Return ONLY this JSON shape, "
        'with no extra top-level fields: {"scenes":[{"dialogue_turns":[{"speaker":"...","text":"..."}]}]}. '
        "The scenes array length must equal the number of input scenes. Each scene must have exactly 14 "
        "dialogue_turns by default; extend only to 15-18 when the selected development genuinely requires it, "
        "and never exceed 18 or fall below 9. Never return a singular `scene` object, state summary, plan, or prose.\n\n"
        "INPUT DATA\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return {"prompt": instructions, "payload": payload, "prompt_version": PROMPT_VERSION}


def _tokens(text):
    return re.findall(r"[\w']+", str(text).lower())


def leakage_scan(model_input, excluded_texts, *, min_long_substring=40, ngram_size=5):
    """Content-level redaction scan; short names/common terms are ignored."""
    serialized = json.dumps(model_input, ensure_ascii=False, sort_keys=True)
    normalized = re.sub(r"\s+", " ", serialized).strip().lower()
    findings = []
    for label, text in excluded_texts.items():
        if not text:
            continue
        source = re.sub(r"\s+", " ", str(text)).strip().lower()
        if len(source) >= min_long_substring and source in normalized:
            findings.append({"source": label, "kind": "long_substring", "length": len(source)})
        src_tokens = _tokens(source)
        dst_tokens = _tokens(normalized)
        src_grams = {tuple(src_tokens[i:i + ngram_size]) for i in range(max(0, len(src_tokens) - ngram_size + 1))}
        dst_grams = {tuple(dst_tokens[i:i + ngram_size]) for i in range(max(0, len(dst_tokens) - ngram_size + 1))}
        overlap = len(src_grams & dst_grams) / max(1, len(src_grams))
        # Very short plan anchors can legitimately share a few event words.
        # Require a substantial source before treating aggregate overlap as a
        # leakage finding; long contiguous source text remains a hard finding.
        if len(src_tokens) >= 20 and overlap >= 0.25:
            findings.append({"source": label, "kind": "ngram_overlap", "ratio": round(overlap, 4)})
    return {"valid": not findings, "findings": findings, "excluded_source_count": len(excluded_texts)}


def freeze_manifest(root):
    root = Path(root)
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        entries.append({"path": str(path.relative_to(root)), "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"root": str(root), "file_count": len(entries), "files": entries,
            "manifest_sha256": _sha(entries)}
