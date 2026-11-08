"""Frozen local dataset inputs and evidence-bound model extraction requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "h4.r02.dataset_cards.v1"
MODEL = "Qwen/Qwen3.6-35B-A3B"
SOURCE_DIR = Path(__file__).resolve().parent.parent / "h4_external_event_seed_selection_v1"
SODA_INDICES = (6027743, 6027790, 6027845)
MULTIWOZ_SPANS = {
    "SNG1013.json": (0, 9),
    "SSNG0142.json": (0, 9),
    "PMUL0843.json": (8, 13),
}
STEP_STATES = {
    "requested", "proposed", "planned", "reported_action", "reported_success",
    "reported_failure", "reported_condition", "reported_preference", "pending",
}
CARD_KEYS = {
    "card_id", "source_id", "guidance_level", "actors", "steps", "dependencies",
    "confusion_axes", "open_ends", "downgrade_reason",
}
ACTOR_KEYS = {"actor_id", "source_speaker", "role"}
STEP_KEYS = {"step_id", "actor_id", "action", "state", "evidence"}
EVIDENCE_KEYS = {"unit_id", "quote"}
DEPENDENCY_KEYS = {"before", "after", "relation", "evidence"}
DEPENDENCY_RELATIONS = {"order", "retry", "condition_change", "correction", "prerequisite"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def _read_source(path: Path, provenance_path: Path, dataset: str) -> tuple[Any, dict[str, Any]]:
    raw = path.read_bytes()
    provenance_raw = provenance_path.read_bytes()
    provenance = json.loads(provenance_raw)
    receipt = {
        "dataset": dataset,
        "split": "train",
        "local_path": str(path.resolve()),
        "local_sha256": _sha(raw),
        "local_bytes": len(raw),
        "origin_url": provenance["url"],
        "revision": None,
        "revision_pinned": False,
        "revision_note": "Existing source URL has no immutable revision; exact local bytes are frozen.",
        "source_receipt_path": str(provenance_path.resolve()),
        "source_receipt_sha256": _sha(provenance_raw),
    }
    if dataset == "allenai/soda":
        if receipt["local_sha256"] != provenance["sha256"]:
            raise ValueError("SODA local bytes differ from existing download receipt")
        receipt["download_sha256"] = provenance["sha256"]
    else:
        # MultiWOZ's receipt hashes the upstream full file, while the local file is a subset.
        receipt["upstream_download_sha256"] = provenance["download_sha256"]
        receipt["upstream_download_bytes"] = provenance["download_bytes"]
        receipt["local_file_is_download_subset"] = True
    return json.loads(raw), receipt


def freeze_source_subset(source_dir: str | Path = SOURCE_DIR) -> dict[str, Any]:
    """Return complete frozen prompt excerpts and local/origin provenance, without writes."""
    base = Path(source_dir)
    soda, soda_receipt = _read_source(
        base / "soda_train_sample.json", base / "soda_train_sample_source.json", "allenai/soda"
    )
    multiwoz, multiwoz_receipt = _read_source(
        base / "multiwoz_train_sample.json", base / "multiwoz_train_source.json", "MultiWOZ_2.2"
    )
    items: list[dict[str, Any]] = []
    soda_rows = {entry["row"]["original_index"]: entry for entry in soda["rows"]}
    for original_index in SODA_INDICES:
        entry = soda_rows[original_index]
        row = entry["row"]
        if row["split"] != "train" or entry.get("truncated_cells"):
            raise ValueError("SODA sample must be complete train data")
        if len(row["dialogue"]) != len(row["speakers"]):
            raise ValueError("SODA speaker/dialogue lengths differ")
        units = [{"unit_id": "narrative", "kind": "narrative", "speaker": None, "text": row["narrative"]}]
        units.extend(
            {"unit_id": f"dialogue:{i}", "kind": "dialogue", "speaker": speaker, "text": text}
            for i, (speaker, text) in enumerate(zip(row["speakers"], row["dialogue"]))
        )
        item = {
            "source_id": f"soda:train:{original_index}",
            "dataset": "allenai/soda", "split": "train", "item_id": str(original_index),
            "local_row_index": entry["row_idx"],
            "source_file_sha256": soda_receipt["local_sha256"],
            "source_item_sha256": _sha(_canonical(row)),
            "source_span": {"narrative": "full", "dialogue_start": 0, "dialogue_end": len(row["dialogue"]) - 1},
            "source_speakers": sorted(set(row["speakers"])),
            "units": units,
        }
        item["excerpt_sha256"] = _sha(_canonical(units))
        items.append(item)
    multiwoz_items = {item["dialogue_id"]: item for item in multiwoz}
    for dialogue_id, (start, end) in MULTIWOZ_SPANS.items():
        original = multiwoz_items[dialogue_id]
        turns = [turn for turn in original["turns"] if start <= int(turn["turn_id"]) <= end]
        if [int(turn["turn_id"]) for turn in turns] != list(range(start, end + 1)):
            raise ValueError("MultiWOZ source span is not complete")
        units = [
            {"unit_id": f"turn:{turn['turn_id']}", "kind": "dialogue", "speaker": turn["speaker"], "text": turn["utterance"]}
            for turn in turns
        ]
        item = {
            "source_id": f"multiwoz22:train:{dialogue_id}",
            "dataset": "MultiWOZ_2.2", "split": "train", "item_id": dialogue_id,
            "source_file_sha256": multiwoz_receipt["local_sha256"],
            "source_item_sha256": _sha(_canonical(original)),
            "source_span": {"turn_start": start, "turn_end": end},
            "source_speakers": sorted({unit["speaker"] for unit in units}),
            "units": units,
        }
        item["excerpt_sha256"] = _sha(_canonical(units))
        items.append(item)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_method": "fixed_local_pilot_subset_daily_social_and_constraint_changes",
        "selection_is_complexity_or_difficulty_certificate": False,
        "files": [soda_receipt, multiwoz_receipt],
        "items": items,
    }
    manifest["manifest_sha256"] = _sha(_canonical(manifest))
    return manifest


def select_source_subset(source_manifest: dict[str, Any], source_ids: list[str]) -> dict[str, Any]:
    """Select a frozen batch with a new manifest hash and linkage to its parent manifest."""
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("source_ids must be nonempty and unique")
    lookup = {item["source_id"]: item for item in source_manifest["items"]}
    if any(source_id not in lookup for source_id in source_ids):
        raise ValueError("unknown extraction batch source_id")
    batch = {
        **source_manifest,
        "selection_parent_manifest_sha256": source_manifest["manifest_sha256"],
        "items": [lookup[source_id] for source_id in source_ids],
    }
    del batch["manifest_sha256"]
    batch["manifest_sha256"] = _sha(_canonical(batch))
    return batch


def build_extraction_request(
    source_manifest: dict[str, Any] | None = None,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create the exact bounded request payload; caller owns credentials and HTTP ledger."""
    manifest = source_manifest if source_manifest is not None else freeze_source_subset()
    if source_ids is not None:
        manifest = select_source_subset(manifest, source_ids)
    instructions = """Extract at most one reusable event card per supplied source item, at most six cards total.
Use English. You are extracting source structure, not writing a new episode or questions.
Return ONLY JSON with exactly schema_version and cards. schema_version must be h4.r02.dataset_cards.v1.
Every card must have exactly: card_id, source_id, guidance_level, actors, steps, dependencies,
confusion_axes, open_ends, downgrade_reason. Unique card_id; source_id must match input.
guidance_level is scaffold or seed. A scaffold retains evidenced distinct stages and a concrete
dependency/condition update, correction, or retry. Simple/single-event material should be a seed;
give a short downgrade_reason. Do not fabricate complexity or a completed result to obtain scaffold.
actors is a list of {actor_id: actor_1/actor_2/etc, source_speaker: exact input speaker, role: abstract role}.
Use only actor_N placeholders in action descriptions and abstract roles, not original person/place names.
steps is a list (1-6) of {step_id: s1/s2/etc, actor_id, action: short abstract action,
state, evidence: [{unit_id, quote}]}. State must be requested, proposed, planned, reported_action,
reported_success, reported_failure, reported_condition, reported_preference, or pending.
Each quote must be copied exactly from one referenced input unit; no invented or paraphrased quotes.
Preserve attribution and modality: a proposed purchase or booking is not completed, a preference is
not an action, reported success is not observed execution. Every step needs its own evidence.
SODA narrative and dialogue are separate evidence kinds. Narrative-only steps remain narrative-only;
do not claim they were said in dialogue. Do not merge a narrative outcome with a pending dialogue
proposal; select one consistent evidence track or use seed and explain the limitation in open_ends.
dependencies is a list of {before: step_id, after: step_id, relation, evidence: [{unit_id,quote}]}.
relation is order, retry, condition_change, correction, or prerequisite. Keep only evidenced links;
adjacent mentions alone do not establish causality. No dependency may refer to missing steps.
confusion_axes and open_ends are lists of short strings. Describe distinguishable source relations
and genuinely unknown results, not planned new source facts. Complexity never predicts Solver score.
All source-specific names, locations, quantities and quotations are provenance only and must not
be copied as the new root's experiences. Future new details belong to realization, not extraction.
Omit an unusable item rather than repairing it. No questions, Gold, difficulty claims or extra fields.
Keep each card concise: short actions, at most 3 dependencies, at most 3 confusion_axes and open_ends.
"""
    visible_items = [
        {key: item[key] for key in ("source_id", "dataset", "split", "source_span", "source_speakers", "units")}
        for item in manifest["items"]
    ]
    return {
        "model": MODEL,
        "temperature": 0.25,
        "max_tokens": 4096,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "messages": [{
            "role": "user",
            "content": instructions + "\nSOURCE_ITEMS_JSON:\n" + json.dumps(
                {"source_manifest_sha256": manifest["manifest_sha256"], "source_items": visible_items},
                ensure_ascii=True,
            ),
        }],
    }


def _object_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label}: invalid object fields")


def _text(value: Any, label: str, *, maximum: int = 500) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label}: expected nonempty bounded text")


def _evidence(value: Any, units: dict[str, dict[str, Any]], label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError(f"{label}: evidence must have 1-3 references")
    normalized = []
    for ref in value:
        _object_keys(ref, EVIDENCE_KEYS, label)
        unit_id = ref["unit_id"]
        if not isinstance(unit_id, str) or unit_id not in units:
            raise ValueError(f"{label}: unknown source unit")
        quote = ref["quote"]
        _text(quote, label + ".quote", maximum=2000)
        if quote not in units[unit_id]["text"]:
            raise ValueError(f"{label}: quote is not exact source text")
        start = units[unit_id]["text"].index(quote)
        normalized.append({
            **ref,
            "kind": units[unit_id]["kind"],
            "source_speaker": units[unit_id]["speaker"],
            "char_start": start,
            "char_end": start + len(quote),
            "quote_sha256": _sha(quote.encode()),
        })
    return normalized


def _normalize_card(card: Any, item: dict[str, Any]) -> dict[str, Any]:
    _object_keys(card, CARD_KEYS, "card")
    _text(card["card_id"], "card_id", maximum=80)
    if card["guidance_level"] not in {"seed", "scaffold"}:
        raise ValueError("guidance_level: expected seed or scaffold")
    actors = card["actors"]
    if not isinstance(actors, list) or not 1 <= len(actors) <= 6:
        raise ValueError("actors: expected 1-6 actors")
    actor_ids: set[str] = set()
    for index, actor in enumerate(actors, start=1):
        _object_keys(actor, ACTOR_KEYS, "actor")
        if actor["actor_id"] != f"actor_{index}":
            raise ValueError("actor_id: require consecutive abstract placeholders")
        if actor["source_speaker"] not in item["source_speakers"]:
            raise ValueError("actor: source_speaker not found in source item")
        _text(actor["role"], "actor.role", maximum=160)
        actor_ids.add(actor["actor_id"])
    steps = card["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        raise ValueError("steps: expected 1-6 steps")
    units = {unit["unit_id"]: unit for unit in item["units"]}
    step_ids: set[str] = set()
    normalized_steps = []
    for index, step in enumerate(steps, start=1):
        _object_keys(step, STEP_KEYS, "step")
        if step["step_id"] != f"s{index}":
            raise ValueError("step_id: require consecutive step IDs")
        if step["actor_id"] not in actor_ids:
            raise ValueError("step: unknown actor")
        _text(step["action"], "step.action", maximum=400)
        if step["state"] not in STEP_STATES:
            raise ValueError("step.state: unsupported source status")
        refs = _evidence(step["evidence"], units, "step.evidence")
        normalized_steps.append({**step, "evidence": refs})
        step_ids.add(step["step_id"])
    dependencies = card["dependencies"]
    if not isinstance(dependencies, list) or len(dependencies) > 3:
        raise ValueError("dependencies: expected at most 3 relations")
    normalized_dependencies = []
    for dep in dependencies:
        _object_keys(dep, DEPENDENCY_KEYS, "dependency")
        if dep["before"] not in step_ids or dep["after"] not in step_ids or dep["before"] == dep["after"]:
            raise ValueError("dependency: invalid step references")
        if int(dep["before"][1:]) >= int(dep["after"][1:]):
            raise ValueError("dependency: before step must precede after step")
        if dep["relation"] not in DEPENDENCY_RELATIONS:
            raise ValueError("dependency: unsupported relation")
        normalized_dependencies.append({**dep, "evidence": _evidence(dep["evidence"], units, "dependency.evidence")})
    for field in ("confusion_axes", "open_ends"):
        if not isinstance(card[field], list) or len(card[field]) > 3:
            raise ValueError(f"{field}: expected at most 3 strings")
        for value in card[field]:
            _text(value, field, maximum=300)
    if not isinstance(card["downgrade_reason"], str) or len(card["downgrade_reason"]) > 400:
        raise ValueError("downgrade_reason: invalid text")
    if card["guidance_level"] == "seed" and not card["downgrade_reason"].strip():
        raise ValueError("seed: downgrade reason required")
    if card["guidance_level"] == "scaffold" and (len(steps) < 2 or not dependencies):
        raise ValueError("scaffold: needs at least two evidenced steps and a dependency")
    normalized = {
        **card,
        "steps": normalized_steps,
        "dependencies": normalized_dependencies,
        "lineage": {
            "dataset": item["dataset"], "split": item["split"], "item_id": item["item_id"],
            "source_span": item["source_span"],
            "source_file_sha256": item["source_file_sha256"],
            "source_item_sha256": item["source_item_sha256"],
            "excerpt_sha256": item["excerpt_sha256"],
            "semantic_fidelity_certified": False,
        },
    }
    normalized["card_sha256"] = _sha(_canonical(normalized))
    return normalized


def validate_extracted_cards(
    response: str | dict[str, Any],
    source_manifest: dict[str, Any],
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Strictly parse/filter extraction output; never repair text, references or partial JSON."""
    if source_ids is not None:
        source_manifest = select_source_subset(source_manifest, source_ids)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "accepted_cards": [], "rejected_records": [],
        "semantic_fidelity_certified": False,
    }
    try:
        parsed = json.loads(response) if isinstance(response, str) else response
        _object_keys(parsed, {"schema_version", "cards"}, "response")
        if parsed["schema_version"] != SCHEMA_VERSION:
            raise ValueError("response: schema version mismatch")
        if not isinstance(parsed["cards"], list) or len(parsed["cards"]) > 6:
            raise ValueError("response: cards must be a list of at most 6")
    except (ValueError, TypeError) as exc:
        result["rejected_records"].append({"scope": "response", "reason": str(exc)})
        return result
    items = {item["source_id"]: item for item in source_manifest["items"]}
    source_counts: dict[str, int] = {}
    card_counts: dict[str, int] = {}
    for card in parsed["cards"]:
        if isinstance(card, dict):
            for field, counts in (("source_id", source_counts), ("card_id", card_counts)):
                value = card.get(field)
                if isinstance(value, str):
                    counts[value] = counts.get(value, 0) + 1
    for index, card in enumerate(parsed["cards"]):
        try:
            _object_keys(card, CARD_KEYS, "card")
            source_id = card["source_id"]
            if not isinstance(source_id, str) or source_id not in items:
                raise ValueError("card: unknown source_id")
            if source_counts[source_id] != 1:
                raise ValueError("card: duplicate source_id")
            if not isinstance(card["card_id"], str) or card_counts.get(card["card_id"]) != 1:
                raise ValueError("card: missing or duplicate card_id")
            result["accepted_cards"].append(_normalize_card(card, items[source_id]))
        except (ValueError, TypeError) as exc:
            result["rejected_records"].append({
                "scope": "card", "index": index,
                "source_id": card.get("source_id") if isinstance(card, dict) else None,
                "reason": str(exc),
            })
    result["missing_source_ids"] = sorted(set(items) - {card["source_id"] for card in result["accepted_cards"]})
    return result


def build_realization_guidance(card: dict[str, Any]) -> dict[str, Any]:
    """Strip raw provenance/quote fields; abstractions remain model-authored, not certified."""
    return {
        "card_id": card["card_id"],
        "card_sha256": card["card_sha256"],
        "guidance_level": card["guidance_level"],
        "actors": [{"actor_id": actor["actor_id"], "role": actor["role"]} for actor in card["actors"]],
        "steps": [{key: step[key] for key in ("step_id", "actor_id", "action", "state")} for step in card["steps"]],
        "dependencies": [{key: dep[key] for key in ("before", "after", "relation")} for dep in card["dependencies"]],
        "confusion_axes": card["confusion_axes"],
        "open_ends": card["open_ends"],
        "downgrade_reason": card["downgrade_reason"],
        "source_is_root_evidence": False,
        "new_details_must_be_realized_in_root_dialogue": True,
    }
