"""Small deterministic contracts for Memory-event continuation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import combinations
import re
from typing import Any, Mapping, Sequence

PARENT_RELATIONS = {
    "REALIZED", "ONGOING", "PLANNED_OR_INTENDED", "RELATED_ONLY", "NONE",
}


def build_speaker_registry(parent: Mapping[str, Any]) -> dict[str, Any]:
    """Create a deterministic registry for speakers already in the Parent.

    The registry is the only name source accepted by the strict live Planner
    contract.  Names are preserved exactly as they occur in the Parent; Memory
    aliases and mentioned-only people never enter this registry.
    """
    names: list[str] = []
    for turn in (parent.get("dialogue") or {}).get("turns") or []:
        name = str(turn.get("speaker") or "").strip()
        if name and name not in names:
            names.append(name)
    return {
        "version": "canonical_speaker_registry_v1",
        "speakers": [
            {"speaker_ref": f"SP{index:02d}", "canonical_name": name}
            for index, name in enumerate(names, 1)
        ],
    }


def _speaker_maps(parent: Mapping[str, Any],
                  speaker_registry: Mapping[str, Any] | None = None
                  ) -> tuple[dict[str, str], dict[str, str]]:
    registry = speaker_registry or build_speaker_registry(parent)
    ref_to_name = {
        str(row.get("speaker_ref")): str(row.get("canonical_name"))
        for row in registry.get("speakers") or []
        if isinstance(row, Mapping) and row.get("speaker_ref") and
        row.get("canonical_name")
    }
    return ref_to_name, {name: ref for ref, name in ref_to_name.items()}


def materialize_planner_speaker_refs(
    value: Any, *, parent: Mapping[str, Any],
    speaker_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize the strict Planner speaker contract.

    Live Planner output uses only ``speaker_refs`` and ``actor_ref``.  This
    helper converts those stable references into the legacy participant and
    actor-name fields consumed by the frozen downstream contracts. Memorax's
    ``user``/``assistant`` values are transport roles, never person aliases.
    A declared ``new_people`` name is removed from ``speaker_refs`` because the
    local compiler adds it to participants. Other unknown refs remain rejected.
    """
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": ["planner_speaker_top_level"],
                "value": value}
    # Expanded Planner calls may return several complete single-plan objects.
    # Materialize every variant through this same speaker contract before the
    # workflow selects one; this prevents a non-selected variant from hiding
    # invalid speaker refs while preserving the legacy single-plan shape.
    if "plan_variants" in value:
        variants = value.get("plan_variants")
        if (set(value) != {"plan_variants"} or not isinstance(variants, list)
                or not variants):
            return {"valid": False,
                    "errors": ["planner_plan_variants_shape"],
                    "value": deepcopy(dict(value))}
        materialized_variants: list[Any] = []
        errors: list[str] = []
        for index, variant in enumerate(variants, 1):
            materialized = materialize_planner_speaker_refs(
                variant, parent=parent, speaker_registry=speaker_registry)
            materialized_variants.append(materialized.get("value"))
            errors.extend(
                f"plan_variants_{index}_{error}"
                for error in materialized.get("errors") or [])
        return {
            "valid": not errors,
            "errors": errors,
            "value": {"plan_variants": materialized_variants},
            "speaker_registry": deepcopy(dict(speaker_registry or
                                               build_speaker_registry(parent))),
        }
    ref_to_name, _ = _speaker_maps(parent, speaker_registry)
    result = deepcopy(dict(value))
    errors: list[str] = []
    # All Planner modes must enter through this same materializer.  The
    # single-event ordered route uses `steps`; keeping it here prevents that
    # route from silently growing a second speaker contract.
    for group_name in ("selected_continuations", "new_development_candidates",
                       "steps"):
        items = result.get(group_name)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items, 1):
            prefix = f"{group_name}_{index}"
            if not isinstance(item, dict):
                errors.append(f"{prefix}_object")
                continue
            if "participants" in item or "required_participants" in item:
                errors.append(f"{prefix}_legacy_speaker_fields")
                continue
            raw_refs = item.get("speaker_refs")
            if (not isinstance(raw_refs, list) or not raw_refs):
                errors.append(f"{prefix}_speaker_refs")
                continue
            new_people = item.get("new_people") or []
            if not isinstance(new_people, list):
                errors.append(f"{prefix}_new_people_shape")
                new_people = []
            declared_new_names = {
                str(person.get("name") or "").strip()
                for person in new_people
                if isinstance(person, Mapping) and
                str(person.get("name") or "").strip()
            }
            refs: list[str] = []
            for raw_ref in raw_refs:
                ref_text = str(raw_ref).strip()
                if ref_text in declared_new_names:
                    continue
                if _UNBOUND_SPEAKER_ALIAS.fullmatch(ref_text):
                    errors.append(f"{prefix}_transport_role_ref_forbidden")
                    continue
                if ref_text and ref_text not in refs:
                    refs.append(ref_text)
            if not refs:
                errors.append(f"{prefix}_speaker_refs")
                continue
            for ref in refs:
                if ref not in ref_to_name:
                    errors.append(f"{prefix}_unknown_speaker_ref:{ref}")
            names: list[str] = []
            for ref in refs:
                name = ref_to_name.get(str(ref))
                if name and name not in names:
                    names.append(name)
            if group_name == "selected_continuations" and new_people:
                errors.append(f"{prefix}_new_people_forbidden")
            for person in new_people:
                if isinstance(person, Mapping) and str(person.get("name") or "").strip():
                    name = str(person["name"]).strip()
                    if name not in names:
                        names.append(name)
            participant_field = (
                "required_participants" if group_name == "steps"
                else "participants")
            item[participant_field] = names
            raw_bindings = item.pop("actor_bindings", None)
            if raw_bindings is None:
                errors.append(f"{prefix}_actor_refs")
                continue
            if not isinstance(raw_bindings, list) or not raw_bindings:
                errors.append(f"{prefix}_actor_refs")
                continue
            bindings: list[dict[str, str]] = []
            for binding in raw_bindings:
                if (not isinstance(binding, Mapping) or
                        set(binding) != {"actor_ref", "binding"}):
                    errors.append(f"{prefix}_actor_ref_shape")
                    continue
                actor_ref = str(binding.get("actor_ref") or "").strip()
                if _UNBOUND_SPEAKER_ALIAS.fullmatch(actor_ref):
                    errors.append(f"{prefix}_transport_role_actor_forbidden")
                    continue
                actor_name = ref_to_name.get(actor_ref)
                if not actor_name or actor_name not in names:
                    errors.append(f"{prefix}_actor_ref_unknown_or_not_participant")
                    continue
                text = str(binding.get("binding") or "").strip()
                if not text:
                    errors.append(f"{prefix}_actor_binding_empty")
                bindings.append({"actor": actor_name, "binding": text})

            # Memorax transport roles carry no character identity. They must
            # never leak from a bucket summary into Planner action semantics.
            action_fields = (
                "goal", "transition_from_previous", "required_transition",
                "after_state", "before_state", "interaction_beats",
            )
            unbound_aliases: set[str] = set()
            for field in action_fields:
                raw_value = item.get(field)
                values = raw_value if isinstance(raw_value, list) else [raw_value]
                unbound_aliases.update(
                    alias for value in values
                    for alias in _unbound_speaker_aliases(value))
            for binding in bindings:
                aliases = _unbound_speaker_aliases(binding.get("binding"))
                unbound_aliases.update(aliases)
            if unbound_aliases:
                errors.append(f"{prefix}_transport_role_alias_forbidden")
            item["actor_bindings"] = bindings
            item.pop("speaker_refs", None)
    return {"valid": not errors, "errors": errors, "value": result,
            "speaker_registry": deepcopy(dict(speaker_registry or
                                               build_speaker_registry(parent)))}


def select_planner_variant(
    value: Mapping[str, Any], *, max_plan_variants: int = 1,
) -> dict[str, Any]:
    """Select the first explicit plan variant without changing legacy output."""
    limit = int(max_plan_variants)
    if limit < 1:
        return {"valid": False, "errors": ["max_plan_variants_must_be_positive"]}
    if "plan_variants" not in value:
        return {"valid": True, "value": deepcopy(dict(value)),
                "variant_count": 1, "selected_index": 0,
                "selection_mode": "single_plan_legacy"}
    variants = value.get("plan_variants")
    if not isinstance(variants, list) or not variants:
        return {"valid": False, "errors": ["planner_plan_variants_shape"]}
    if len(variants) > limit:
        return {"valid": False,
                "errors": [f"planner_plan_variants_max_{limit}"]}
    if not all(isinstance(variant, Mapping) for variant in variants):
        return {"valid": False,
                "errors": ["planner_plan_variant_object"]}
    return {"valid": True, "value": deepcopy(dict(variants[0])),
            "variant_count": len(variants), "selected_index": 0,
            "selection_mode": "first_variant_deterministic"}


def build_event_candidates(
    overview: Mapping[str, Any], *, max_candidates: int = 8,
    person_events_only: bool = False,
) -> list[dict[str, Any]]:
    """Build bounded event candidates while binding person-bucket event facts.

    An event bucket key remains the preferred candidate identity.  Person
    buckets add the actor-specific facts that event summaries often collapse
    to ``user``.  If no event bucket can match a concrete person event, that
    fact is retained as a deterministic person-event candidate instead of
    being silently discarded.  This is lexical binding only; it is not a
    semantic thread classifier.
    """
    candidates: list[dict[str, Any]] = []
    event_candidates_by_key: dict[str, dict[str, Any]] = {}

    def event_text(value: Any) -> str:
        text = " ".join(str(value or "").split()).strip()
        if text.startswith("[事件") and "]" in text:
            text = text.split("]", 1)[1].strip()
        return text

    def append_unique(values: list[str], value: str) -> None:
        if value and value.casefold() not in {row.casefold() for row in values}:
            values.append(value)

    def fact_tokens(value: Any) -> set[str]:
        """Return small, deterministic content tokens for source binding."""
        text = str(value or "").casefold()
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "for", "in",
            "on", "at", "by", "with", "from", "that", "this", "their",
            "they", "them", "he", "she", "it", "user", "someone",
            "person", "classmates", "said", "says", "mentioned", "felt",
            "expressed", "noted", "discussed", "event", "recent", "more",
            "due", "not", "have", "has", "had", "been", "being", "about",
            "now", "made", "make", "also", "very", "just", "still", "then",
            "there", "here", "because", "despite", "after", "before", "into",
            "over", "under", "through", "while", "would", "could", "should",
            "will", "was", "were", "are", "is", "but", "view",
        }
        result: set[str] = set()
        for token in tokens:
            if token in stop or (len(token) < 3 and token not in {"tv"}):
                continue
            # Collapse common English inflections so ``studied`` and
            # ``studying`` can bind to ``study`` without an LLM.
            stem = token
            for suffix in ("ingly", "edly", "ing", "ed", "ies", "s"):
                if len(stem) > len(suffix) + 2 and stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    if suffix == "ies":
                        stem += "y"
                    break
            if stem not in stop:
                result.add(stem)
        return result

    def fact_identity(value: Any) -> str:
        """Return a stable identity for one concrete fact, independent of time tags."""
        normalized = event_text(value).casefold()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        return digest

    def facts_are_same_event(left: Any, right: Any) -> bool:
        """Conservatively bind two person facts when their content overlaps.

        ``past``/``current`` are temporal hints only. They never participate in
        this decision. A high overlap ratio prevents generic references such
        as an apology and a later report that merely mentions that apology from
        becoming one event.
        """
        left_tokens = fact_tokens(left)
        right_tokens = fact_tokens(right)
        overlap = left_tokens & right_tokens
        if len(overlap) < 3 or not left_tokens or not right_tokens:
            return False
        return len(overlap) / min(len(left_tokens), len(right_tokens)) >= 0.5

    def create_candidate(*, key: str, display: str, contents: Sequence[str],
                         source_type: str,
                         supporting_source: Mapping[str, Any]) -> dict[str, Any] | None:
        normalized_contents: list[str] = []
        for content in contents:
            append_unique(normalized_contents, event_text(content))
        if not key or not normalized_contents or len(candidates) >= int(max_candidates):
            return None
        candidate = {
            "memory_event_id": f"ME{len(candidates) + 1:02d}",
            "memory_event_key": key,
            "memory_event_display": display or key,
            "memory_event_content": normalized_contents,
            "source_type": source_type,
            "supporting_sources": [dict(supporting_source)],
            "related_people": [],
            "person_event_facts": [],
        }
        candidates.append(candidate)
        return candidate

    def identity_matches(content: str, value: str) -> bool:
        alias = " ".join(re.findall(r"\w+", str(value or "").casefold()))
        normalized = " ".join(re.findall(r"\w+", content.casefold()))
        if len(alias) < 2:
            return False
        if any(ord(character) > 127 for character in alias):
            return alias in normalized
        return bool(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized))

    def matching_event_candidates(content: str) -> list[dict[str, Any]]:
        content_key = event_text(content).casefold()
        content_tokens = fact_tokens(content_key)
        matches: list[dict[str, Any]] = []
        for candidate in event_candidates_by_key.values():
            candidate_text = " | ".join([
                str(candidate.get("memory_event_key") or ""),
                str(candidate.get("memory_event_display") or ""),
                *[str(value) for value in
                  candidate.get("memory_event_content") or []],
            ])
            candidate_tokens = fact_tokens(candidate_text)
            overlap = len(content_tokens & candidate_tokens)
            exact_identity = (
                content_key in {
                    str(value).casefold()
                    for value in candidate["memory_event_content"]} or
                identity_matches(content, candidate["memory_event_key"]) or
                identity_matches(content, candidate["memory_event_display"]))
            # Two meaningful shared tokens are the minimum lexical binding
            # signal.  This avoids merging on generic words such as ``event``
            # or on the transport-role word ``user``.
            if exact_identity or overlap >= 2:
                matches.append(candidate)
        return matches

    def add_person_event_fact(candidate: dict[str, Any], *,
                               person_key: str, person_display: str,
                               event_tag: str, fact: str) -> None:
        row = {
            "person_key": person_key,
            "person_display": person_display,
            "event_tag": event_tag,
            "fact_text": fact,
        }
        identity = (person_key.casefold(), event_tag.casefold(),
                    fact.casefold())
        existing = {
            (str(item.get("person_key") or "").casefold(),
             str(item.get("event_tag") or "").casefold(),
             str(item.get("fact_text") or "").casefold())
            for item in candidate.setdefault("person_event_facts", [])
            if isinstance(item, Mapping)
        }
        if identity not in existing:
            candidate["person_event_facts"].append(row)

    person_fact_fields = {
        "身份": "identity_facts",
        "identity": "identity_facts",
        "关系": "relationship_facts",
        "relationship": "relationship_facts",
        "偏好": "preference_facts",
        "preference": "preference_facts",
        "性格": "trait_facts",
        "trait": "trait_facts",
        "personality": "trait_facts",
    }

    def split_person_fact(value: Any) -> tuple[str, str]:
        text = " ".join(str(value or "").split()).strip()
        match = re.match(r"^\[([^]]+)]\s*(.*)$", text)
        if not match:
            return "", text
        return match.group(1).strip(), match.group(2).strip()

    def person_profile(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
        person_key = str(
            row.get("canonical_key") or row.get("display_key") or "").strip()
        person_display = str(row.get("display_key") or person_key).strip()
        profile: dict[str, Any] = {
            "name": person_display,
            "identity_facts": [],
            "relationship_facts": [],
            "preference_facts": [],
            "trait_facts": [],
        }
        recent_events: list[tuple[str, str]] = []
        for raw_content in row.get("content") or []:
            label, fact = split_person_fact(raw_content)
            normalized_label = label.casefold()
            if normalized_label.startswith("事件") or normalized_label.startswith("event"):
                if fact:
                    event = (label.split("@", 1)[1].strip()
                             if "@" in label else label)
                    identity = (event.casefold(), fact.casefold())
                    if identity not in {
                            (old_event.casefold(), old_fact.casefold())
                            for old_event, old_fact in recent_events}:
                        recent_events.append((event, fact))
                continue
            field = person_fact_fields.get(normalized_label)
            if field and fact:
                append_unique(profile[field], fact)
        return profile, recent_events

    def add_person_context(candidate: dict[str, Any], *, person_key: str,
                           profile: Mapping[str, Any]) -> None:
        related_people = candidate.setdefault("related_people", [])
        name = str(profile.get("name") or "")
        if name.casefold() not in {
                str(person.get("name") or "").casefold()
                for person in related_people if isinstance(person, Mapping)}:
            related_people.append(deepcopy(dict(profile)))
        source = {
            "source_type": "person_bucket",
            "memory_key": person_key,
            "display_key": name,
            "fact_categories": [
                field for field in (
                    "identity_facts", "relationship_facts",
                    "preference_facts", "trait_facts")
                if profile.get(field)
            ],
        }
        if source not in candidate["supporting_sources"]:
            candidate["supporting_sources"].append(source)

    events = overview.get("events") or []
    if isinstance(events, list):
        for row in events:
            if not isinstance(row, Mapping):
                continue
            key = str(row.get("canonical_key") or row.get("display_key") or "").strip()
            display = str(row.get("display_key") or key).strip()
            content_values = row.get("content") or []
            if not key or not isinstance(content_values, list):
                continue
            identity = key.casefold()
            candidate = event_candidates_by_key.get(identity)
            normalized_contents = [event_text(content) for content in content_values]
            normalized_contents = [content for content in normalized_contents if content]
            source = {"source_type": "event_bucket", "memory_key": key,
                      "display_key": display, "contents": normalized_contents}
            if candidate is None:
                candidate = create_candidate(
                    key=key, display=display, contents=normalized_contents,
                    source_type="event_bucket", supporting_source=source)
                if candidate is not None:
                    event_candidates_by_key[identity] = candidate
            else:
                for content in normalized_contents:
                    append_unique(candidate["memory_event_content"], content)
                candidate["supporting_sources"].append(source)

    people = overview.get("people") or []
    if isinstance(people, list):
        for row in people:
            if not isinstance(row, Mapping):
                continue
            person_key = str(row.get("canonical_key") or row.get("display_key") or "").strip()
            profile, recent_events = person_profile(row)
            person_display = str(profile.get("name") or person_key)
            if not isinstance(row.get("content") or [], list):
                continue

            for candidate in event_candidates_by_key.values():
                event_identity_text = " | ".join([
                    str(candidate.get("memory_event_key") or ""),
                    str(candidate.get("memory_event_display") or ""),
                    *[str(value) for value in
                      candidate.get("memory_event_content") or []],
                ])
                if (identity_matches(event_identity_text, person_key) or
                        identity_matches(event_identity_text, person_display)):
                    add_person_context(
                        candidate, person_key=person_key, profile=profile)

            for event_tag, normalized in recent_events:
                if (person_events_only and event_tag.casefold() in {
                        "recent", "近期", "最新", "general"}):
                    # A broad roll-up cannot identify one concrete event. Do
                    # not split or merge it by guesswork in the current path.
                    continue
                source = {"source_type": "person_recent_event",
                          "memory_key": person_key,
                          "display_key": person_display,
                          "event_tag": event_tag,
                          "content": normalized}
                matches = matching_event_candidates(
                    f"{event_tag} {normalized}")
                if matches:
                    for candidate in matches:
                        if source not in candidate["supporting_sources"]:
                            candidate["supporting_sources"].append(source)
                        add_person_context(
                            candidate, person_key=person_key, profile=profile)
                        add_person_event_fact(
                            candidate, person_key=person_key,
                            person_display=person_display,
                            event_tag=event_tag, fact=normalized)
                    continue

                # Retain a concrete person-bucket event when no event bucket
                # owns it.  Reuse an existing person candidate when its event
                # facts have a strong lexical overlap; otherwise derive a
                # stable identity from the person key and fact content.
                person_candidates = [
                    row for row in candidates
                    if str(row.get("source_type") or "") ==
                    "person_bucket_event"
                ]
                person_matches: list[dict[str, Any]] = []
                for existing_candidate in person_candidates:
                    existing_facts = [
                        str(row.get("fact_text") or "")
                        for row in existing_candidate.get(
                            "person_event_facts") or []
                        if isinstance(row, Mapping)
                    ]
                    # A fact repeated for the same person is the same source
                    # even when Memorax assigned a different temporal label.
                    # Cross-person binding still requires strong content
                    # overlap; matching ``past``/``current`` alone is invalid.
                    if any(
                            event_text(normalized).casefold() ==
                            event_text(existing).casefold()
                            for existing in existing_facts):
                        person_matches.append(existing_candidate)
                        continue
                    if any(facts_are_same_event(normalized, existing)
                           for existing in existing_facts):
                        person_matches.append(existing_candidate)
                if person_matches:
                    for candidate in person_matches:
                        if source not in candidate["supporting_sources"]:
                            candidate["supporting_sources"].append(source)
                        add_person_context(
                            candidate, person_key=person_key,
                            profile=profile)
                        add_person_event_fact(
                            candidate, person_key=person_key,
                            person_display=person_display,
                            event_tag=event_tag, fact=normalized)
                        append_unique(candidate["memory_event_content"],
                                      normalized)
                    continue
                person_key_identity = person_key.casefold()
                fact_digest = fact_identity(normalized)
                candidate = create_candidate(
                    key=f"person_event:{person_key_identity}:fact:{fact_digest}",
                    display=normalized,
                    contents=[normalized], source_type="person_bucket_event",
                    supporting_source=source)
                if candidate is not None:
                    add_person_context(
                        candidate, person_key=person_key, profile=profile)
                    add_person_event_fact(
                        candidate, person_key=person_key,
                        person_display=person_display,
                        event_tag=event_tag, fact=normalized)

    return candidates[:int(max_candidates)]


def build_person_event_candidates(
    overview: Mapping[str, Any], *, max_candidates: int = 8,
) -> list[dict[str, Any]]:
    """Build auxiliary person-bucket event candidates for diagnostics.

    The restored Expansion route is event-bucket-first.  This helper remains
    available for legacy/diagnostic callers and must not be described as the
    primary Planner source.
    """
    return build_event_candidates(
        {"events": [], "people": list(overview.get("people") or [])},
        max_candidates=max_candidates, person_events_only=True)


def _parent_turn_map(parent: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("turn_id")): row for row in parent["dialogue"]["turns"]}


_VERIFIER_PLACEHOLDER_STATES = {
    "latest state",
    "current state",
    "latest state already true in parent",
    "the latest state already true in parent",
    "latest state directly supported by parent",
    "the latest state directly supported by parent",
}


def validate_event_verifier_output(
    value: Any, *, candidates: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the thin transcript-grounding response before planning."""
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {"verified_tracks"}:
        return {"valid": False, "errors": ["event_verifier_top_level_shape"],
                "verified_track_count": 0}
    tracks = value.get("verified_tracks")
    if not isinstance(tracks, list):
        return {"valid": False, "errors": ["verified_tracks_shape"],
                "verified_track_count": 0}
    if len(tracks) > len(candidates):
        errors.append("verified_track_count_exceeds_candidates")
    candidate_refs = {str(row.get("memory_event_id") or "")
                      for row in candidates}
    parent_turn_count = len(parent["dialogue"]["turns"])
    used_event_refs: set[str] = set()
    for index, track in enumerate(tracks, 1):
        prefix = f"verified_track_{index}"
        if not isinstance(track, Mapping):
            errors.append(f"{prefix}_not_object")
            continue
        expected_track_keys = {
                "memory_event_ref", "verified_current_state",
                "evidence_positions"}
        legacy_state_keys = {"event_ref", "intention", "action", "result"}
        if set(track) == legacy_state_keys:
            errors.append(f"{prefix}_legacy_state_split_fields")
        elif set(track) != expected_track_keys:
            errors.append(f"{prefix}_shape")
        event_ref = str(track.get("memory_event_ref") or "")
        if event_ref not in candidate_refs:
            errors.append(f"{prefix}_unknown_memory_event")
        if event_ref in used_event_refs:
            errors.append(f"{prefix}_duplicate_memory_event")
        used_event_refs.add(event_ref)
        current_state = " ".join(str(
            track.get("verified_current_state") or "").split()).strip()
        if not current_state:
            errors.append(f"{prefix}_empty_current_state")
        elif current_state.casefold().rstrip(".") in _VERIFIER_PLACEHOLDER_STATES:
            errors.append(f"{prefix}_placeholder_current_state")
        positions = track.get("evidence_positions")
        if (not isinstance(positions, list) or
                not 1 <= len(positions) <= 4 or
                any(type(position) is not int for position in positions) or
                len(set(positions)) != len(positions) or
                any(position < 1 or position > parent_turn_count
                    for position in positions)):
            errors.append(f"{prefix}_evidence_positions")
    return {"valid": not errors, "errors": errors,
            "verified_track_count": len(tracks),
            "verified_memory_event_refs": sorted(used_event_refs)}


def materialize_verified_tracks(
    value: Mapping[str, Any], *, candidates: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind verifier positions to authoritative Parent turns and local IDs."""
    validation = validate_event_verifier_output(
        value, candidates=candidates, parent=parent)
    if not validation["valid"]:
        raise ValueError("event_verifier_invalid:" + ",".join(
            validation["errors"]))
    candidate_by_ref = {
        str(row.get("memory_event_id") or ""): row for row in candidates}
    parent_turns = list(parent["dialogue"]["turns"])
    tracks: list[dict[str, Any]] = []
    for index, raw in enumerate(value.get("verified_tracks") or [], 1):
        event_ref = str(raw["memory_event_ref"])
        source = candidate_by_ref[event_ref]
        positions = [int(position) for position in raw["evidence_positions"]]
        evidence = [{
            "position": position,
            "turn_ref": str(parent_turns[position - 1]["turn_id"]),
            "speaker": str(parent_turns[position - 1]["speaker"]),
            "text": str(parent_turns[position - 1]["text"]),
        } for position in positions]
        tracks.append({
            "verified_track_ref": f"VT{index:02d}",
            "memory_event_ref": event_ref,
            "memory_event_key": source.get("memory_event_key"),
            "memory_event_display": source.get("memory_event_display"),
            "memory_event_content": deepcopy(
                source.get("memory_event_content") or []),
            "verified_current_state": str(
                raw["verified_current_state"]).strip(),
            "evidence_positions": positions,
            "evidence_turn_refs": [row["turn_ref"] for row in evidence],
            "evidence_source": evidence,
        })
    return tracks


def materialize_memory_only_tracks(
    candidates: Sequence[Mapping[str, Any]],
    *, event_bucket_only: bool = False,
) -> list[dict[str, Any]]:
    """Create bounded planning tracks directly from Memorax bucket contents.

    This is intentionally not a transcript verifier.  The bucket key is the
    event identity and the bucket contents are non-authoritative planning
    hints.  Empty candidates are omitted; no source turn/evidence refs are
    fabricated in this mode.
    """
    tracks: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, 1):
        contents = [" ".join(str(value or "").split()).strip()
                    for value in candidate.get("memory_event_content") or []]
        contents = [value for value in contents if value]
        if not contents:
            continue
        all_supporting_sources = deepcopy(
            candidate.get("supporting_sources") or [])
        if event_bucket_only:
            # Person buckets remain available in the raw snapshot for audit,
            # but cannot become Planner event context in this pilot.  Keep
            # only provenance owned by the selectable event bucket.
            supporting_sources = [
                source for source in all_supporting_sources
                if isinstance(source, Mapping) and
                str(source.get("source_type") or "") == "event_bucket"
            ]
            if not supporting_sources:
                continue
            related_people = []
            person_event_facts = []
        else:
            supporting_sources = all_supporting_sources
            related_people = deepcopy(candidate.get("related_people") or [])
            person_event_facts = deepcopy(
                candidate.get("person_event_facts") or [])
        planning_packet = {
            "memory_event_ref": str(candidate.get("memory_event_id") or ""),
            "memory_event_key": candidate.get("memory_event_key"),
            "memory_event_display": candidate.get("memory_event_display"),
            "event_contents": deepcopy(contents),
            "person_event_facts": person_event_facts,
            "related_people": related_people,
            "supporting_sources": supporting_sources,
            "state_authority": "memorax_unverified",
            "event_bucket_only": bool(event_bucket_only),
        }
        tracks.append({
            "verified_track_ref": f"VT{index:02d}",
            "memory_event_ref": str(candidate.get("memory_event_id") or ""),
            "memory_event_key": candidate.get("memory_event_key"),
            "memory_event_display": candidate.get("memory_event_display"),
            "memory_event_content": deepcopy(contents),
            "person_event_facts": deepcopy(
                candidate.get("person_event_facts") or []),
            "memory_planning_packet": planning_packet,
            "related_people": related_people,
            "supporting_sources": supporting_sources,
            "memory_state_hint": "Unverified Memorax summaries: " +
                                 " | ".join(contents),
            # Compatibility-only: frozen downstream contracts still consume
            # this key. It is explicitly not a verified latest state.
            "verified_current_state": "Unverified Memorax summaries: " +
                                      " | ".join(contents),
            "evidence_positions": [],
            "evidence_turn_refs": [],
            "evidence_source": [],
            "state_source": "memorax_memory_only_unverified",
        })
    return tracks


_WEAK_TRANSITION = re.compile(
    r"\b(?:mentions?|acknowledges?|discusses?|talks?|chats?|considers?|"
    r"thinks?\s+about)\b",
    re.IGNORECASE,
)
_EXPLICIT_TRANSITION = re.compile(
    r"\b(?:agrees?|obtains?|confirms?|completes?|decides?|accepts?|"
    r"starts?|begins?|commits?|finishes?|meets?|exchanges?|establishes?|"
    r"participates?|takes?\s+responsibility|makes?\s+a\s+concrete\s+decision|"
    r"calls?|contacts?|sends?|books?|checks?|schedules?|creates?|"
    r"receives?|gets?|offers?|shares?|suggests?|proposes?|encourages?|commends?|"
    r"recommends?|reaffirms?|renews?|recommits?|"
    r"sets?\s+up|arranges?)\b",
    re.IGNORECASE,
)
# This is intentionally a narrow mechanical guard. It catches an outcome that
# is stated only as a future intention, while allowing a current agreement or
# schedule whose subject happens to include a future date.
_FUTURE_ONLY_OUTCOME = re.compile(
    r"\b(?:will|would|eventually|in\s+the\s+future|someday|one\s+day|"
    r"plans?\s+to|intends?\s+to|hopes?\s+to|may\s+later)\b",
    re.IGNORECASE,
)
_CURRENT_OUTCOME_SIGNAL = re.compile(
    r"\b(?:agrees?|decides?|accepts?|confirm(?:s|ed)?|establishes?|"
    r"sent|"
    r"completes?|starts?|begins?|finishes?|is\s+introduced|"
    r"is\s+scheduled|(?:has|have)\s+scheduled|arranges?|feels?|"
    r"has\s+a\s+(?:concrete\s+)?plan|"
    r"have\s+a\s+(?:concrete\s+)?plan)\b",
    re.IGNORECASE,
)
_COMPOUND_PERSON_NAME = re.compile(
    r"(?:\s+(?:and|&)\s+|[,;/、，]|(?:和|与))", re.IGNORECASE)

# Memorax summaries may use ``user`` as a role label, but it is not a
# canonical Parent speaker. In strict Memory-only planning it is unsafe to
# let the Realizer guess which canonical person that label means. This is a
# narrow deterministic contract check, not a semantic classifier.
_UNBOUND_SPEAKER_ALIAS = re.compile(
    r"(?<![\w])(?:the\s+user|user|用户)(?![\w])", re.IGNORECASE)
_TRANSPORT_SPEAKER_LABEL = re.compile(
    r"(?:the\s+user|user|用户|the\s+assistant|assistant|助手)",
    re.IGNORECASE,
)


def _unbound_speaker_aliases(value: Any) -> list[str]:
    text = " ".join(str(value or "").split())
    return sorted({match.group(0).casefold() for match in
                   _UNBOUND_SPEAKER_ALIAS.finditer(text)})


def is_transport_speaker_label(value: Any) -> bool:
    """Return true only for Memorax/chat transport roles used as a name."""
    return bool(_TRANSPORT_SPEAKER_LABEL.fullmatch(str(value or "").strip()))


def _is_obviously_weak_transition(value: Any) -> bool:
    """Reject only the explicit weak-action forms frozen by this contract."""
    text = " ".join(str(value or "").split())
    return bool(_WEAK_TRANSITION.search(text) and
                not _EXPLICIT_TRANSITION.search(text))


def _is_obviously_future_only_outcome(value: Any) -> bool:
    """Return true only for an explicitly future-only end-state sentence.

    This does not attempt semantic planning.  A present-tense agreement or
    schedule such as ``A agrees to meet next week`` remains valid; a sentence
    that only says ``A will eventually meet`` is rejected before it reaches
    the Gate because it cannot be a scene-completable durable outcome.
    """
    text = " ".join(str(value or "").split())
    future_match = _FUTURE_ONLY_OUTCOME.search(text)
    if not future_match:
        return False
    current_match = _CURRENT_OUTCOME_SIGNAL.search(text)
    # A current decision/agreement can legitimately mention a later date
    # ("A agrees to walk on Saturday").  A future verb that comes first
    # ("A will eventually complete...") remains future-only.
    return current_match is None or current_match.start() > future_match.start()


def _validate_transition_fields(value: Mapping[str, Any], *, prefix: str,
                                allowed_participants: set[str],
                                errors: list[str],
                                required_keys: set[str]) -> None:
    if set(value) != required_keys:
        errors.append(f"{prefix}_shape")
    for field in ("before_state", "required_transition", "after_state"):
        if not str(value.get(field) or "").strip():
            errors.append(f"{prefix}_empty_{field}")
    if _is_obviously_future_only_outcome(value.get("after_state")):
        errors.append(f"{prefix}_future_only_after_state")
    bindings = value.get("actor_bindings")
    if not isinstance(bindings, list) or not bindings:
        errors.append(f"{prefix}_actor_bindings")
    elif any(not isinstance(binding, Mapping) or
             set(binding) != {"actor", "binding"} or
             not str(binding.get("actor") or "").strip() or
             not str(binding.get("binding") or "").strip()
             for binding in bindings):
        errors.append(f"{prefix}_actor_bindings")
    elif any(str(binding.get("actor")) not in allowed_participants
             for binding in bindings):
        errors.append(f"{prefix}_actor_not_declared")
    beats = value.get("interaction_beats")
    if (not isinstance(beats, list) or not 3 <= len(beats) <= 7 or
            any(not str(beat or "").strip() for beat in beats)):
        errors.append(f"{prefix}_interaction_beats")
    if _is_obviously_weak_transition(value.get("required_transition")):
        errors.append(f"{prefix}_weak_transition_candidate")


def _validate_scene_metadata(value: Mapping[str, Any], *, prefix: str,
                             allowed_participants: set[str],
                             parent_speakers: set[str],
                             errors: list[str]) -> None:
    if not str(value.get("goal") or "").strip():
        errors.append(f"{prefix}_empty_goal")
    if not str(value.get("transition_from_previous") or "").strip():
        errors.append(f"{prefix}_empty_transition_from_previous")
    participants = value.get("participants")
    if not isinstance(participants, list) or not participants:
        errors.append(f"{prefix}_participants")
    elif any(str(item) not in allowed_participants for item in participants):
        errors.append(f"{prefix}_participant_not_declared")
    elif not any(str(item) in parent_speakers for item in participants):
        errors.append(f"{prefix}_parent_participant_required")


def _validate_new_people(value: Any, *, prefix: str,
                         errors: list[str]) -> set[str]:
    if not isinstance(value, list) or len(value) > 2:
        errors.append(f"{prefix}_new_people")
        return set()
    names: list[str] = []
    for index, person in enumerate(value, 1):
        item_prefix = f"{prefix}_new_person_{index}"
        if (not isinstance(person, Mapping) or
                set(person) != {"name", "relationship_basis"}):
            errors.append(f"{item_prefix}_shape")
            continue
        name = str(person.get("name") or "").strip()
        basis = str(person.get("relationship_basis") or "").strip()
        if not name:
            errors.append(f"{item_prefix}_name")
        elif _COMPOUND_PERSON_NAME.search(name):
            errors.append(f"{item_prefix}_compound_name")
        if not basis:
            errors.append(f"{item_prefix}_relationship_basis")
        if name:
            names.append(name)
    if len(names) != len(set(names)):
        errors.append(f"{prefix}_duplicate_new_person_name")
    return set(names)


def _validate_new_development_candidate(
    item: Any, *, index: int, parent_speakers: set[str],
) -> list[str]:
    """Return local errors for one new-development candidate.

    Candidate semantic weakness is intentionally scoped to the candidate so
    a ranked fallback can replace it. Schema and structural violations remain
    hard Planner-contract failures; fallback must not weaken that contract.
    """
    prefix = f"new_development_candidate_{index}"
    errors: list[str] = []
    if not isinstance(item, Mapping):
        return [f"{prefix}_not_object"]
    new_keys = {
        "source_basis", "new_people", "goal", "participants",
        "transition_from_previous", "before_state", "required_transition",
        "after_state", "actor_bindings", "interaction_beats",
    }
    if set(item) != new_keys:
        errors.append(f"{prefix}_shape")
    basis = item.get("source_basis")
    if (not isinstance(basis, list) or not basis or len(basis) > 4 or
            any(not str(row or "").strip() for row in basis)):
        errors.append(f"{prefix}_source_basis")
    new_people = _validate_new_people(
        item.get("new_people"), prefix=prefix, errors=errors)
    _validate_scene_metadata(
        item, prefix=prefix, allowed_participants=parent_speakers | new_people,
        parent_speakers=parent_speakers, errors=errors)
    _validate_transition_fields(
        item, prefix=prefix,
        allowed_participants={str(value) for value in item.get("participants") or []},
        errors=errors, required_keys=new_keys)
    return errors


# These words are deliberately small and mechanical.  The check below is a
# conservative outline collision detector, not a semantic thread classifier.
_OUTLINE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "could", "for", "from", "had", "has", "have", "in", "into",
    "is", "it", "its", "of", "on", "or", "so", "that", "the", "their",
    "them", "there", "they", "this", "to", "was", "were", "with", "would",
    "will", "already", "now", "still", "then", "here", "just", "really",
    "also", "more", "some", "very", "own", "each", "other", "new",
    "concrete", "transition", "state", "end", "selected", "rejected",
    "current", "supported", "shared", "specific", "item", "now", "true",
}
_OUTLINE_STAGE_WORDS = {
    # Include the small normalized forms produced by _outline_token_set.
    "completed": {"complete", "complet", "finish", "achiev", "success",
                  "successful", "successfully", "done", "resolv"},
    "planned": {"plan", "plann", "intend", "agre", "decid", "propos",
                "schedul"},
    "active": {"start", "beg", "ongo", "continu", "tri", "attempt"},
}


def _outline_token_set(value: Any, *, participants: Sequence[str] = ()) -> set[str]:
    text = str(value or "").casefold()
    # Participant names identify the actors, not the development itself.
    for participant in participants:
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+",
                                str(participant).casefold()):
            text = re.sub(rf"\b{re.escape(token)}\b", " ", text)
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
    normalized: set[str] = set()
    for token in tokens:
        if token in _OUTLINE_STOPWORDS or len(token) < 3:
            continue
        # Small suffix normalization is enough for complete/completed,
        # walk/walked, and similar wording drift seen in Planner outlines.
        if token.endswith("ied") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        normalized.add(token)
    return normalized


def _outline_stage(tokens: set[str]) -> set[str]:
    return {
        stage for stage, words in _OUTLINE_STAGE_WORDS.items()
        if tokens.intersection(words)
    }


def validate_planner_outline_distinctness(
    plan: Mapping[str, Any], *, min_shared_tokens: int = 3,
    min_containment: float = 0.45,
) -> dict[str, Any]:
    """Reject only obvious cross-scene development collisions.

    This intentionally operates after deterministic Planner compilation and
    before any Gate/Realizer request.  It does not decide whether two scenes
    are thematically related; it catches an exact repeated transition, or a
    close completed transition, when the same actors are also shared.
    """
    scenes = [row for row in (plan.get("scenes") or [])
              if isinstance(row, Mapping)]
    all_participants = {
        str(value).strip()
        for scene in scenes
        for value in (scene.get("participants") or [])
        if str(value).strip()
    }
    all_participants.update({
        str(person.get("name") or "").strip()
        for scene in scenes
        for person in (scene.get("new_people") or [])
        if isinstance(person, Mapping) and str(person.get("name") or "").strip()
    })
    prepared: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        candidate = ((scene.get("next_step_candidates") or [None])[0]
                     if isinstance(scene.get("next_step_candidates"), list)
                     else None)
        candidate = candidate if isinstance(candidate, Mapping) else scene
        participants = [str(value) for value in scene.get("participants") or []]
        transition = str(candidate.get("required_transition") or
                          scene.get("required_transition") or
                          candidate.get("new_development") or "").strip()
        outcome = str(candidate.get("after_state") or
                      scene.get("after_state") or
                      candidate.get("durable_outcome") or "").strip()
        transition_tokens = _outline_token_set(
            transition, participants=sorted(all_participants))
        outcome_tokens = _outline_token_set(
            outcome, participants=sorted(all_participants))
        combined = transition_tokens | outcome_tokens
        prepared.append({
            "scene_index": index,
            "scene_ref": str(scene.get("scene_ref") or f"S{index:02d}"),
            "memory_event_id": str(scene.get("memory_event_id") or ""),
            "participants": participants,
            "transition": transition,
            "outcome": outcome,
            "transition_tokens": transition_tokens,
            "outcome_tokens": outcome_tokens,
            "tokens": combined,
            "stages": _outline_stage(transition_tokens | outcome_tokens),
        })

    collisions: list[dict[str, Any]] = []
    non_blocking_similarities: list[dict[str, Any]] = []
    for left_index, left in enumerate(prepared):
        for right in prepared[left_index + 1:]:
            shared_participants = sorted(
                set(left["participants"]) & set(right["participants"]))
            shared = left["transition_tokens"] & right["transition_tokens"]
            smaller = min(len(left["transition_tokens"]),
                          len(right["transition_tokens"]))
            containment = (len(shared) / smaller) if smaller else 0.0
            exact = (
                re.sub(r"\s+", " ", left["transition"].casefold()).strip() ==
                re.sub(r"\s+", " ", right["transition"].casefold()).strip() and
                bool(left["transition"]) and bool(right["transition"]))
            same_completion = "completed" in left["stages"] and \
                "completed" in right["stages"]
            fuzzy_similarity = bool(
                same_completion and len(shared) >= int(min_shared_tokens) and
                containment >= float(min_containment))
            if shared_participants and fuzzy_similarity and not exact:
                non_blocking_similarities.append({
                    "scene_refs": [left["scene_ref"], right["scene_ref"]],
                    "memory_event_ids": [left["memory_event_id"],
                                          right["memory_event_id"]],
                    "shared_participants": shared_participants,
                    "shared_tokens": sorted(shared),
                    "containment": round(containment, 3),
                    "reason": "lexically_similar_non_blocking",
                })
            if shared_participants and exact:
                collisions.append({
                    "scene_refs": [left["scene_ref"], right["scene_ref"]],
                    "memory_event_ids": [left["memory_event_id"],
                                          right["memory_event_id"]],
                    "shared_participants": shared_participants,
                    "shared_tokens": sorted(shared),
                    "containment": round(containment, 3),
                    "reason": "exact_transition_duplicate",
                })
    return {
        "valid": not collisions,
        "errors": [
            "planner_outline_duplicate:" + ":".join(row["scene_refs"])
            for row in collisions
        ],
        "scene_count": len(prepared),
        "collisions": collisions,
        "non_blocking_similarities": non_blocking_similarities,
        "policy": {
            "min_shared_tokens": int(min_shared_tokens),
            "min_containment": float(min_containment),
            "requires_shared_participant": True,
            "exact_transition_is_duplicate": True,
            "fuzzy_similarity_is_non_blocking": True,
            "fuzzy_overlap_requires_same_completion_stage": True,
        },
    }


def build_planner_outline_pairs(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build a compact pair menu for the complete candidate pool.

    The semantic auditor must see ranked fallback candidates as well as the
    first three scene slots.  Candidate refs and Memory provenance are derived
    locally; the Planner never supplies either one.  Parent transcript and
    turn evidence remain deliberately absent.
    """
    scenes = [row for row in (plan.get("candidate_pool") or
                             plan.get("scenes") or [])
              if isinstance(row, Mapping)]
    views: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        candidate = ((scene.get("next_step_candidates") or [None])[0]
                     if isinstance(scene.get("next_step_candidates"), list)
                     else None)
        candidate = candidate if isinstance(candidate, Mapping) else scene
        event_id = str(scene.get("memory_event_id") or "")
        candidate_ref = str(scene.get("candidate_ref") or "")
        if not candidate_ref:
            candidate_ref = str(scene.get("selected_candidate_ref") or "")
        if not candidate_ref and event_id:
            candidate_ref = _candidate_ref(event_id, 1)
        memory_source = {
            "memory_event_ref": scene.get("memory_event_ref"),
            "memory_event_key": scene.get("memory_event_key"),
            "memory_event_display": scene.get("memory_event_display"),
            "memory_event_content": deepcopy(
                scene.get("memory_event_content") or []),
            "supporting_sources": deepcopy(
                scene.get("supporting_sources") or []),
        }
        views.append({
            "candidate_ref": candidate_ref,
            "scene_ref": str(scene.get("scene_ref") or
                              candidate_ref or f"S{index:02d}"),
            "memory_event_ref": event_id,
            "development_mode": str(scene.get("development_mode") or ""),
            "participants": [str(value) for value in
                             scene.get("participants") or []],
            "current_state": str(scene.get("verified_current_state") or
                                  candidate.get("before_state") or "").strip(),
            "required_transition": str(
                candidate.get("required_transition") or
                scene.get("required_transition") or
                candidate.get("new_development") or "").strip(),
            "after_state": str(candidate.get("after_state") or
                                scene.get("after_state") or
                                candidate.get("durable_outcome") or "").strip(),
            "source_basis": deepcopy(scene.get("source_basis") or
                                      candidate.get("source_basis") or []),
            "memory_source": memory_source,
        })
    pairs: list[dict[str, Any]] = []
    pair_index = 1
    for left_index, left in enumerate(views):
        for right in views[left_index + 1:]:
            pairs.append({
                "pair_ref": f"P{pair_index:02d}",
                "left": deepcopy(left),
                "right": deepcopy(right),
            })
            pair_index += 1
    return pairs


def validate_planner_outline_semantic_output(
    value: Any, *, plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the one-call semantic duplicate review.

    A structurally valid ``duplicate`` or ``uncertain`` result is a filtering
    signal, not a contract error.  The local program retains the earlier
    ranked candidate and filters later conflicts.  This keeps model-format
    failures separate from the model's actual outline judgment.
    """
    expected_pairs = build_planner_outline_pairs(plan)
    expected_refs = [str(row["pair_ref"]) for row in expected_pairs]
    errors: list[str] = []
    reviews = value.get("pair_reviews") if isinstance(value, Mapping) else None
    if not isinstance(value, Mapping) or set(value) != {"pair_reviews"}:
        errors.append("planner_outline_semantic_top_level_shape")
        reviews = []
    if not isinstance(reviews, list):
        errors.append("planner_outline_semantic_pair_reviews_shape")
        reviews = []
    by_ref: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(reviews, 1):
        if not isinstance(row, Mapping) or set(row) != {
                "pair_ref", "decision", "reason"}:
            errors.append(f"planner_outline_semantic_review_{index}_shape")
            continue
        pair_ref = str(row.get("pair_ref") or "")
        if pair_ref in by_ref:
            errors.append(f"planner_outline_semantic_duplicate_ref:{pair_ref}")
            continue
        if pair_ref not in expected_refs:
            errors.append(f"planner_outline_semantic_unknown_ref:{pair_ref}")
        decision = str(row.get("decision") or "").strip()
        if decision not in {"distinct", "duplicate", "uncertain"}:
            errors.append(f"planner_outline_semantic_decision:{pair_ref}")
        if not isinstance(row.get("reason"), str):
            errors.append(f"planner_outline_semantic_reason:{pair_ref}")
        by_ref[pair_ref] = row
    missing = [ref for ref in expected_refs if ref not in by_ref]
    # A missing review is not evidence of distinctness.  Keep it as an
    # explicitly conservative uncertainty edge so the workflow can select a
    # different legal scene set without accepting an unreviewed pair.
    errors.extend(f"planner_outline_semantic_missing_ref:{ref}" for ref in missing)
    for ref in missing:
        by_ref[ref] = {
            "pair_ref": ref,
            "decision": "uncertain",
            "reason": "missing model review; conservatively treated as uncertain",
        }
    duplicate_refs = [ref for ref, row in by_ref.items()
                      if row.get("decision") == "duplicate"]
    uncertain_refs = [ref for ref, row in by_ref.items()
                      if row.get("decision") == "uncertain"]
    # Only malformed/unknown/duplicate-ref/invalid-decision errors make the
    # response unusable.  Missing pair reviews are handled conservatively by
    # the conflict graph below, not by treating them as distinct.
    hard_error_prefixes = (
        "planner_outline_semantic_top_level_shape",
        "planner_outline_semantic_pair_reviews_shape",
        "planner_outline_semantic_review_",
        "planner_outline_semantic_duplicate_ref:",
        "planner_outline_semantic_unknown_ref:",
        "planner_outline_semantic_decision:",
        "planner_outline_semantic_reason:",
    )
    structural_valid = not any(
        any(error.startswith(prefix) for prefix in hard_error_prefixes)
        for error in errors)
    # Build a conflict graph.  Both explicit duplicate/uncertain decisions and
    # missing reviews are forbidden edges for candidate selection.
    pair_by_ref = {str(row.get("pair_ref")): row
                   for row in expected_pairs}
    decision_by_ref = {
        str(ref): str(row.get("decision"))
        for ref, row in by_ref.items()
    }
    ordered_candidate_refs: list[str] = []
    for row in expected_pairs:
        for side in ("left", "right"):
            candidate_ref = str(row.get(side, {}).get("candidate_ref") or "")
            if candidate_ref and candidate_ref not in ordered_candidate_refs:
                ordered_candidate_refs.append(candidate_ref)
    conflict_edges = {
        tuple(sorted((
            str(pair.get("left", {}).get("candidate_ref") or ""),
            str(pair.get("right", {}).get("candidate_ref") or ""),
        )))
        for pair_ref, pair in pair_by_ref.items()
        if decision_by_ref.get(pair_ref) in {"duplicate", "uncertain"}
        and pair.get("left", {}).get("candidate_ref")
        and pair.get("right", {}).get("candidate_ref")
    }

    def legal(combo: Sequence[str]) -> bool:
        return all(tuple(sorted((left, right))) not in conflict_edges
                   for left, right in combinations(combo, 2))

    # Select the lexicographically highest-ranked maximum independent set for
    # the audit artifact.  The workflow later asks for exactly three members.
    kept_candidate_refs: list[str] = []
    for size in range(len(ordered_candidate_refs), -1, -1):
        for combo in combinations(ordered_candidate_refs, size):
            if legal(combo):
                kept_candidate_refs = list(combo)
                break
        if kept_candidate_refs or size == 0:
            break
    blocked_candidate_refs = [ref for ref in ordered_candidate_refs
                              if ref not in set(kept_candidate_refs)]
    if not structural_valid:
        overall_decision = "invalid"
    elif duplicate_refs:
        overall_decision = "duplicate"
    elif uncertain_refs or missing:
        overall_decision = "uncertain"
    else:
        overall_decision = "pass"
    return {
        "valid": structural_valid,
        "errors": errors + [
            f"planner_outline_semantic_duplicate:{ref}"
            for ref in duplicate_refs] + [
            f"planner_outline_semantic_uncertain:{ref}"
            for ref in uncertain_refs],
        "overall_decision": overall_decision,
        "expected_pair_refs": expected_refs,
        "returned_pair_refs": list(by_ref),
        "duplicate_pair_refs": duplicate_refs,
        "uncertain_pair_refs": uncertain_refs,
        "missing_pair_refs": missing,
        "conflict_edges": [list(edge) for edge in sorted(conflict_edges)],
        "blocked_candidate_refs": blocked_candidate_refs,
        "kept_candidate_refs": kept_candidate_refs,
        "candidate_filtering": bool(blocked_candidate_refs),
        "pair_reviews": [dict(by_ref[ref]) for ref in expected_refs
                         if ref in by_ref],
    }


def select_ranked_legal_candidate_set(
    candidate_refs: Sequence[str], *,
    outline_audit: Mapping[str, Any],
    required_count: int = 3,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Choose the highest-ranked legal scene set after semantic review.

    Candidate order is already deterministic: continuation rows first, then
    Planner's new-candidate order.  Duplicate and uncertain edges are both
    treated as conflicts.  Missing reviews therefore cannot silently pass.
    """
    ranked = list(dict.fromkeys(str(ref) for ref in candidate_refs if ref))
    edges = {
        tuple(sorted((str(edge[0]), str(edge[1]))))
        for edge in outline_audit.get("conflict_edges") or []
        if isinstance(edge, (list, tuple)) and len(edge) == 2
    }
    legal_sets: list[tuple[str, ...]] = []
    target_count = min(int(required_count), len(ranked))
    if target_count > 0:
        for combo in combinations(ranked, target_count):
            if all(tuple(sorted(pair)) not in edges
                       for pair in combinations(combo, 2)):
                legal_sets.append(combo)
    if legal_sets:
        selected = list(legal_sets[0])
    elif allow_partial:
        # Expanded profiles treat the scene limit as an upper bound.  Select
        # the highest-ranked legal subset when duplicate/uncertain edges make
        # the requested count impossible; never synthesize a replacement.
        selected = []
        for size in range(target_count - 1, 0, -1):
            partial_sets = [combo for combo in combinations(ranked, size)
                            if all(tuple(sorted(pair)) not in edges
                                   for pair in combinations(combo, 2))]
            if partial_sets:
                selected = list(partial_sets[0])
                break
    else:
        selected = []
    return {
        "required_count": int(required_count),
        "allow_partial": bool(allow_partial),
        "ranked_candidate_refs": ranked,
        "selected_candidate_refs": selected,
        "blocked_candidate_refs": [ref for ref in ranked
                                    if ref not in set(selected)],
        "legal_set_count": len(legal_sets),
        "valid": (len(selected) == int(required_count) if not allow_partial
                  else bool(selected) or int(required_count) == 0),
    }


def validate_planner_output(
    value: Any, *, candidates: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    verified_tracks: Sequence[Mapping[str, Any]] | None = None,
    max_new_candidates: int = 4,
    max_continuations: int = 3,
    scene_quota: int = 3,
    new_development_slots: int | None = None,
    require_full_scene_quota: bool = True,
) -> dict[str, Any]:
    """Validate split continuation/new-development Planner output."""
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": ["planner_not_object"],
                "scene_count": 0}
    if set(value) != {"round_goal", "selected_continuations",
                      "new_development_candidates"}:
        errors.append("planner_top_level_shape")
    if not str(value.get("round_goal") or "").strip():
        errors.append("planner_empty_round_goal")
    continuations = value.get("selected_continuations")
    new_candidates = value.get("new_development_candidates")
    if not isinstance(continuations, list):
        errors.append("selected_continuations_shape")
        continuations = []
    if not isinstance(new_candidates, list):
        errors.append("new_development_candidates_shape")
        new_candidates = []
    if len(continuations) > int(max_continuations):
        errors.append("selected_continuations_max_" + str(int(max_continuations)))
    if len(new_candidates) > int(max_new_candidates):
        errors.append(
            "new_development_candidates_max_" + str(int(max_new_candidates)))
    remaining_quota = max(0, int(scene_quota) - len(continuations))
    independent_new_slots = (remaining_quota if new_development_slots is None
                              else min(remaining_quota, int(new_development_slots)))
    # Keep one ranked fallback whenever the fixed three-scene quota requires
    # new developments. The pool may contain one additional ranked fallback;
    # it is still distinct from the three materialized scene slots.
    required_new_candidates = (
        min(3, remaining_quota + (1 if remaining_quota else 0))
        if new_development_slots is None else
        (independent_new_slots if require_full_scene_quota else 0))
    if len(continuations) <= int(max_continuations) and len(new_candidates) < required_new_candidates:
        errors.append("insufficient_new_development_candidates")
    parent_speakers = {str(row.get("speaker"))
                       for row in parent["dialogue"]["turns"]}
    candidate_validation: list[dict[str, Any]] = []
    valid_new_candidate_indices: list[int] = []
    for index, item in enumerate(new_candidates, 1):
        candidate_errors = _validate_new_development_candidate(
            item, index=index, parent_speakers=parent_speakers)
        candidate_validation.append({
            "index": index,
            "valid": not candidate_errors,
            "errors": candidate_errors,
        })
        if not candidate_errors:
            valid_new_candidate_indices.append(index)
        # Candidate-local semantic prechecks are quarantinable. Shape,
        # reference, participant and other structural contract violations make
        # the Planner response invalid even when later candidates could fill
        # quota. A future-only fallback must not invalidate two later,
        # scene-completable candidates that already fill the remaining slots.
        hard_candidate_errors = [
            error for error in candidate_errors
            if not (error.endswith("_weak_transition_candidate") or
                    error.endswith("_future_only_after_state"))
        ]
        errors.extend(hard_candidate_errors)
    selected_new_candidate_indices = valid_new_candidate_indices[:independent_new_slots]
    materialized_new_candidates = [
        new_candidates[index - 1] for index in selected_new_candidate_indices
    ]

    candidate_by_id = {str(row.get("memory_event_id")): row for row in candidates}
    if verified_tracks is None:
        verified_tracks = [{
            "verified_track_ref": f"VT{index:02d}",
            "memory_event_ref": row.get("memory_event_id"),
            "verified_current_state": "fixture current state",
            "evidence_turn_refs": [parent["dialogue"]["turns"][0]["turn_id"]],
        } for index, row in enumerate(candidates, 1)]
    track_by_ref = {str(row.get("verified_track_ref") or ""): row
                    for row in verified_tracks}
    event_keys = {str(row.get("memory_event_key") or "").casefold()
                  for row in candidates}
    used_refs: set[str] = set()
    used_keys: set[str] = set()
    continuation_keys = {
        "verified_track_ref", "goal", "participants", "transition_from_previous",
        "required_transition", "after_state", "actor_bindings",
        "interaction_beats", "new_people",
    }
    new_keys = {
        "source_basis", "new_people", "goal", "participants",
        "transition_from_previous", "before_state", "required_transition",
        "after_state", "actor_bindings", "interaction_beats",
    }
    for index, item in enumerate(continuations, 1):
        prefix = f"continuation_{index}"
        if not isinstance(item, Mapping):
            errors.append(f"{prefix}_not_object")
            continue
        track_ref = str(item.get("verified_track_ref") or "")
        track = track_by_ref.get(track_ref)
        if track is None:
            errors.append(f"{prefix}_unknown_verified_track")
        ref = str((track or {}).get("memory_event_ref") or "")
        if ref not in candidate_by_id:
            errors.append(f"{prefix}_unknown_memory_event")
        if track_ref in used_refs:
            errors.append(f"{prefix}_duplicate_verified_track_ref")
        used_refs.add(track_ref)
        key = str((candidate_by_id.get(ref) or {}).get("memory_event_key") or "").casefold()
        if key and key in used_keys:
            errors.append(f"{prefix}_duplicate_memory_key")
        if key:
            used_keys.add(key)
        if set(item) != continuation_keys:
            errors.append(f"{prefix}_shape")
        if item.get("new_people") != []:
            errors.append(f"{prefix}_new_people_forbidden")
        _validate_scene_metadata(item, prefix=prefix,
                                 allowed_participants=parent_speakers,
                                 parent_speakers=parent_speakers, errors=errors)
        transition_item = {**dict(item), "before_state": str(
            (track or {}).get("verified_current_state") or "")}
        transition_keys = set(continuation_keys) | {"before_state"}
        transition_keys.remove("verified_track_ref")
        transition_item.pop("verified_track_ref", None)
        _validate_transition_fields(transition_item, prefix=prefix,
                                    allowed_participants={
                                        str(value)
                                        for value in item.get("participants") or []},
                                    errors=errors,
                                    required_keys=transition_keys)
    if require_full_scene_quota and len(selected_new_candidate_indices) < independent_new_slots:
        errors.append("insufficient_valid_new_development_candidates")
        # Preserve the concrete contract diagnostics when the fallback pool
        # cannot fill the fixed quota.  Valid fallback candidates remain
        # usable; these diagnostics only become blocking at quota exhaustion.
        for row in candidate_validation:
            if not row["valid"]:
                errors.extend(row["errors"])
    return {"valid": not errors, "errors": errors,
            "scene_count": len(continuations) + len(materialized_new_candidates),
            "selected_verified_track_refs": sorted(used_refs),
            "selected_continuation_count": len(continuations),
            "raw_new_candidate_count": len(new_candidates),
            "scene_quota": int(scene_quota),
            "max_continuations": int(max_continuations),
            "new_development_slots": (None if new_development_slots is None else int(new_development_slots)),
            "require_full_scene_quota": bool(require_full_scene_quota),
            "remaining_new_quota": remaining_quota,
            "required_new_candidate_count": required_new_candidates,
            "materialized_new_development_count": len(
                materialized_new_candidates),
            "unused_new_candidate_count": max(
                0, len(new_candidates) - len(materialized_new_candidates)),
            "candidate_validation": candidate_validation,
            "valid_new_candidate_indices": valid_new_candidate_indices,
            "selected_new_candidate_indices": selected_new_candidate_indices,
            "skipped_new_candidate_indices": [
                row["index"] for row in candidate_validation if not row["valid"]
            ],
            "final_materialized_scene_count": (
                len(continuations) + len(materialized_new_candidates))}


def validate_planner_repair_output(
    value: Any, *, parent: Mapping[str, Any], required_count: int,
) -> dict[str, Any]:
    """Validate the one-shot replacement pool before merging it."""
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {
            "new_development_candidates"}:
        return {"valid": False, "errors": ["planner_repair_top_level_shape"],
                "candidate_count": 0}
    items = value.get("new_development_candidates")
    required_count = int(required_count)
    if not isinstance(items, list):
        return {"valid": False, "errors": ["planner_repair_candidates_shape"],
                "candidate_count": 0}
    if len(items) != required_count:
        errors.append("planner_repair_candidate_count")
    parent_speakers = {str(row.get("speaker"))
                       for row in parent["dialogue"]["turns"]}
    candidate_validation: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        candidate_errors = _validate_new_development_candidate(
            item, index=index, parent_speakers=parent_speakers)
        candidate_validation.append({
            "index": index, "valid": not candidate_errors,
            "errors": candidate_errors,
        })
        errors.extend(candidate_errors)
    return {"valid": not errors, "errors": errors,
            "candidate_count": len(items),
            "required_count": required_count,
            "candidate_validation": candidate_validation}


def compile_planner_output(
    value: Mapping[str, Any], *, candidates: Sequence[Mapping[str, Any]],
    verified_tracks: Sequence[Mapping[str, Any]] | None = None,
    selected_new_candidate_indices: Sequence[int] | None = None,
    valid_new_candidate_indices: Sequence[int] | None = None,
    scene_quota: int = 3,
    new_development_slots: int | None = None,
) -> dict[str, Any]:
    """Compile split Planner output into the existing internal scene contract."""
    candidate_by_id = {str(row.get("memory_event_id")): row for row in candidates}
    if verified_tracks is None:
        verified_tracks = [{
            "verified_track_ref": f"VT{index:02d}",
            "memory_event_ref": row.get("memory_event_id"),
            "verified_current_state": "fixture current state",
            "evidence_turn_refs": [],
        } for index, row in enumerate(candidates, 1)]
    track_by_ref = {str(row.get("verified_track_ref") or ""): row
                    for row in verified_tracks}
    scenes: list[dict[str, Any]] = []
    candidate_pool: list[dict[str, Any]] = []
    next_person_index = 1
    for item in list(value.get("selected_continuations") or []):
        track_ref = str(item["verified_track_ref"])
        track = track_by_ref[track_ref]
        ref = str(track["memory_event_ref"])
        source = candidate_by_id[ref]
        candidate = {**dict(item),
                     "before_state": str(track["verified_current_state"])}
        candidate.pop("verified_track_ref", None)
        candidate.update({"development_mode": "continuation",
                          "memory_event_ref": ref,
                          "memory_event_id": ref,
                          "memory_event_key": source.get("memory_event_key"),
                          "memory_event_display": source.get("memory_event_display"),
                          "memory_event_content": deepcopy(
                              source.get("memory_event_content") or [])})
        scene = {"memory_event_id": ref, "memory_event_ref": ref,
                       "verified_track_ref": track_ref,
                       "memory_event_key": source.get("memory_event_key"),
                       "memory_event_display": source.get("memory_event_display"),
                       "memory_event_content": deepcopy(
                           source.get("memory_event_content") or []),
                       "supporting_sources": deepcopy(
                           source.get("supporting_sources") or []),
                       "development_mode": "continuation",
                       "new_people": [],
                       "verified_current_state": track["verified_current_state"],
                       "evidence_turn_refs": deepcopy(
                           track.get("evidence_turn_refs") or []),
                       "evidence_source": deepcopy(
                           track.get("evidence_source") or []),
                       "goal": candidate["goal"],
                       "participants": candidate["participants"],
                       "transition_from_previous": candidate[
                           "transition_from_previous"],
                       "next_step_candidates": [candidate]}
        scenes.append(scene)
        candidate_pool.append(deepcopy(scene))
    continuation_count = len(value.get("selected_continuations") or [])
    remaining_quota = max(0, int(scene_quota) - continuation_count)
    independent_new_slots = (remaining_quota if new_development_slots is None
                              else min(remaining_quota, int(new_development_slots)))
    raw_new_candidates = list(value.get("new_development_candidates") or [])
    if selected_new_candidate_indices is None:
        candidate_indices = list(range(1, len(raw_new_candidates) + 1))
    else:
        candidate_indices = [int(index) for index in selected_new_candidate_indices]
    candidate_indices = candidate_indices[:independent_new_slots]
    selected_new_candidates = [raw_new_candidates[index - 1]
                               for index in candidate_indices]
    all_new_scenes: dict[int, dict[str, Any]] = {}
    for materialized_index, item in enumerate(raw_new_candidates, 1):
        ref = f"ND{materialized_index:02d}"
        candidate = dict(item)
        materialized_people: list[dict[str, str]] = []
        for person in item.get("new_people") or []:
            materialized_people.append({
                "person_ref": f"NP{next_person_index:02d}",
                "name": str(person["name"]),
                "relationship_basis": str(person["relationship_basis"]),
            })
            next_person_index += 1
        candidate["new_people"] = deepcopy(materialized_people)
        candidate.update({"development_mode": "new_development",
                          "memory_event_ref": None,
                          "memory_event_id": ref,
                          "memory_event_key": None,
                          "memory_event_display": ref})
        scene = {"memory_event_id": ref, "memory_event_ref": None,
                       "development_mode": "new_development",
                       "source_basis": deepcopy(candidate["source_basis"]),
                       "new_people": deepcopy(materialized_people),
                       "verified_current_state": candidate["before_state"],
                       "evidence_turn_refs": [],
                       "goal": candidate["goal"],
                       "participants": candidate["participants"],
                       "transition_from_previous": candidate[
                           "transition_from_previous"],
                       "next_step_candidates": [candidate]}
        all_new_scenes[materialized_index] = scene
    # Keep the historical three-slot view for compatibility, while retaining
    # every ranked new candidate in a separate pool for Gate fallback.
    scenes.extend(deepcopy(all_new_scenes[index]) for index in candidate_indices)
    gate_candidate_indices = (
        [int(index) for index in valid_new_candidate_indices]
        if valid_new_candidate_indices is not None else
        list(range(1, len(raw_new_candidates) + 1)))
    candidate_pool.extend(
        deepcopy(all_new_scenes[index]) for index in gate_candidate_indices)
    return {"round_goal": value.get("round_goal"), "scenes": scenes,
            "candidate_pool": candidate_pool,
            "scene_quota_audit": {
                "scene_quota": int(scene_quota),
                "selected_continuation_count": continuation_count,
                "new_development_slots": (None if new_development_slots is None
                                           else int(new_development_slots)),
                "raw_new_candidate_count": len(raw_new_candidates),
                "remaining_new_quota": remaining_quota,
                "required_new_candidate_count": min(
                    len(raw_new_candidates), independent_new_slots +
                    (1 if independent_new_slots and new_development_slots is None else 0)),
                "materialized_new_development_count": len(
                    selected_new_candidates),
                "unused_new_candidate_count": max(
                    0, len(raw_new_candidates) - len(selected_new_candidates)),
                "selected_new_candidate_indices": candidate_indices,
                "skipped_new_candidate_indices": [
                    index for index in range(1, len(raw_new_candidates) + 1)
                    if index not in candidate_indices
                ],
                "final_materialized_scene_count": len(scenes),
                "unused_new_development_candidates": deepcopy(
                    [item for index, item in enumerate(raw_new_candidates, 1)
                     if index not in candidate_indices]),
            }}


def _candidate_ref(event_id: str, index: int) -> str:
    return f"{event_id}:C{index:02d}"


def validate_event_binding_alignment(
        plan: Mapping[str, Any], *, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report deterministic ref/identity mismatches without semantic LLM gating.

    A legal VT/ME lookup proves identity and uniqueness only.  This report
    flags a clear lexical collision with another event, while leaving ordinary
    semantic fit as ``UNKNOWN`` for the existing Gate/adherence path.
    """
    by_id = {str(row.get("memory_event_id") or ""): row for row in candidates}
    identities: dict[str, set[str]] = {}
    for event_id, row in by_id.items():
        raw = " ".join([str(row.get("memory_event_key") or ""),
                         str(row.get("memory_event_display") or ""),
                         *[str(x) for x in row.get("memory_event_content") or []]])
        identities[event_id] = {token.casefold() for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{3,}", raw)}
    rows: list[dict[str, Any]] = []
    for index, scene in enumerate(plan.get("scenes") or [], 1):
        if not isinstance(scene, Mapping) or scene.get("development_mode") != "continuation":
            continue
        event_id = str(scene.get("memory_event_id") or "")
        text = " ".join(str(scene.get(key) or "") for key in (
            "goal", "required_transition", "after_state"))
        tokens = {token.casefold() for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{3,}", text)}
        foreign = []
        for other_id, other_tokens in identities.items():
            if other_id == event_id:
                continue
            overlap = sorted(tokens & other_tokens)
            if len(overlap) >= 2:
                foreign.append({"memory_event_id": other_id, "overlap_tokens": overlap})
        status = "CLEAR_FOREIGN_EVENT_OVERLAP" if foreign else "UNKNOWN"
        rows.append({"scene_index": index, "memory_event_id": event_id,
                     "status": status, "foreign_event_overlaps": foreign,
                     "semantic_fit": "UNKNOWN"})
    return {"valid": True, "authority": "mechanical_identity_only",
            "semantic_gate_unchanged": True, "scenes": rows,
            "mismatch_count": sum(row["status"] == "CLEAR_FOREIGN_EVENT_OVERLAP" for row in rows)}


def materialize_planner_output(
    value: Mapping[str, Any], *, passed_candidate_refs: Sequence[str],
    scene_count: int = 3,
) -> dict[str, Any]:
    """Select the first Gate-approved candidate in Planner recommendation order."""
    materialized = deepcopy(dict(value))
    # Quota audit may retain unused candidate semantics for reporting. It must
    # never be part of the clean Realizer input.
    materialized.pop("scene_quota_audit", None)
    candidate_pool = materialized.pop("candidate_pool", None)
    passed = {str(ref) for ref in passed_candidate_refs}
    scenes = candidate_pool if isinstance(candidate_pool, list) else materialized.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("planner_materialization_scenes_required")
    selected_scenes: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            raise ValueError(f"planner_materialization_scene_{index}_object")
        event_id = str(scene.get("memory_event_id") or "")
        candidates = scene.get("next_step_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"planner_materialization_scene_{index}_candidates")
        selected_ref = next((
            _candidate_ref(event_id, candidate_index)
            for candidate_index in range(1, len(candidates) + 1)
            if _candidate_ref(event_id, candidate_index) in passed), None)
        if selected_ref is None:
            continue
        selected_index = int(selected_ref.rsplit("C", 1)[1]) - 1
        candidate = candidates[selected_index]
        if not isinstance(candidate, Mapping):
            raise ValueError(f"planner_materialization_scene_{index}_candidate")
        scene.pop("next_step_candidates", None)
        scene.pop("interaction_beats", None)
        scene["selected_candidate_ref"] = selected_ref
        scene["before_state"] = str(candidate["before_state"]).strip()
        scene["required_transition"] = str(
            candidate["required_transition"]).strip()
        scene["after_state"] = str(candidate["after_state"]).strip()
        scene["actor_bindings"] = deepcopy(candidate["actor_bindings"])
        scene["interaction_beats"] = deepcopy(candidate["interaction_beats"])
        scene["development_mode"] = str(candidate["development_mode"])
        # Planner participants are required coverage, not an exhaustive
        # speaker allowlist. Keep the legacy field for frozen consumers while
        # exposing the authoritative meaning explicitly.
        scene["required_participants"] = deepcopy(
            scene.get("required_participants") or scene.get("participants") or [])
        if scene["development_mode"] == "new_development":
            scene["source_basis"] = deepcopy(candidate["source_basis"])
            scene["new_people"] = deepcopy(candidate.get("new_people") or [])
        # Compatibility fields are deterministic aliases, never Planner output.
        scene["new_development"] = scene["required_transition"]
        scene["durable_outcome"] = scene["after_state"]
        selected_scenes.append(scene)
        if len(selected_scenes) >= int(scene_count):
            break
    if len(selected_scenes) != int(scene_count):
        raise ValueError("planner_materialization_insufficient_pass_candidates")
    materialized["scenes"] = selected_scenes
    return materialized


def validate_realizer_plan_isolation(
    *, raw_plan: Mapping[str, Any], materialized_plan: Mapping[str, Any],
    passed_candidate_refs: Sequence[str],
    scene_count: int = 3,
) -> dict[str, Any]:
    """Prove that only the selected PASS candidate reaches the Realizer plan."""
    errors: list[str] = []
    raw_scenes = raw_plan.get("candidate_pool") or raw_plan.get("scenes")
    final_scenes = materialized_plan.get("scenes")
    if not isinstance(raw_scenes, list) or not isinstance(final_scenes, list):
        return {"valid": False, "errors": ["realizer_isolation_scene_shape"],
                "rejected_content_leak_count": 0}
    if len(final_scenes) != int(scene_count):
        errors.append("realizer_isolation_scene_count")
    if "scene_quota_audit" in materialized_plan:
        errors.append("realizer_isolation_quota_audit_leak")
    passed = {str(value) for value in passed_candidate_refs}
    selected_refs: list[str] = []
    leak_count = 0
    forbidden_container_keys = {
        "next_step_candidates", "candidates", "all_candidates",
        "rejected_candidates", "selected_candidate_index",
    }
    selected_raw_scenes: list[Mapping[str, Any]] = []
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, Mapping):
            continue
        event_id = str(raw_scene.get("memory_event_id") or "")
        candidates = raw_scene.get("next_step_candidates")
        if not isinstance(candidates, list):
            continue
        if any(_candidate_ref(event_id, i) in passed
               for i in range(1, len(candidates) + 1)):
            selected_raw_scenes.append(raw_scene)
    # Materialization consumes the first three passing scenes in the same
    # recommendation order. Later passing fallbacks must remain outside the
    # Realizer plan and therefore outside the isolation comparison.
    selected_raw_scenes = selected_raw_scenes[:int(scene_count)]
    if len(selected_raw_scenes) != len(final_scenes):
        errors.append("realizer_isolation_selected_scene_count")
    for scene_index, (raw_scene, final_scene) in enumerate(
            zip(selected_raw_scenes, final_scenes), 1):
        prefix = f"scene_{scene_index}"
        if not isinstance(raw_scene, Mapping) or not isinstance(final_scene, Mapping):
            errors.append(f"{prefix}_realizer_isolation_object")
            continue
        event_id = str(raw_scene.get("memory_event_id") or "")
        candidates = raw_scene.get("next_step_candidates")
        if not isinstance(candidates, list):
            errors.append(f"{prefix}_realizer_isolation_candidates")
            continue
        selected_index = next((
            index for index in range(1, len(candidates) + 1)
            if _candidate_ref(event_id, index) in passed), None)
        if selected_index is None:
            errors.append(f"{prefix}_realizer_isolation_no_pass")
            continue
        selected_ref = _candidate_ref(event_id, selected_index)
        selected_refs.append(selected_ref)
        selected = candidates[selected_index - 1]
        if not isinstance(selected, Mapping):
            errors.append(f"{prefix}_realizer_isolation_selected_object")
            continue
        if str(final_scene.get("selected_candidate_ref") or "") != selected_ref:
            errors.append(f"{prefix}_realizer_isolation_selected_ref")
        for field in ("before_state", "required_transition", "after_state",
                      "actor_bindings", "interaction_beats",
                      "development_mode"):
            if final_scene.get(field) != selected.get(field):
                errors.append(f"{prefix}_realizer_isolation_{field}")
        if selected.get("development_mode") == "new_development":
            for field in ("source_basis", "new_people"):
                if final_scene.get(field) != selected.get(field):
                    errors.append(f"{prefix}_realizer_isolation_{field}")
        if final_scene.get("new_development") != selected.get(
                "required_transition"):
            errors.append(f"{prefix}_realizer_isolation_new_development")
        if final_scene.get("durable_outcome") != selected.get("after_state"):
            errors.append(f"{prefix}_realizer_isolation_durable_outcome")
        if "completion_conditions" in final_scene:
            errors.append(f"{prefix}_realizer_isolation_completion_conditions")
        if forbidden_container_keys & set(final_scene):
            errors.append(f"{prefix}_realizer_isolation_candidate_container")

        final_strings: set[str] = set()
        def collect_strings(value: Any) -> None:
            if isinstance(value, str):
                final_strings.add(value)
            elif isinstance(value, Mapping):
                for nested in value.values(): collect_strings(nested)
            elif isinstance(value, list):
                for nested in value: collect_strings(nested)
        collect_strings(final_scene)
        selected_strings: set[str] = set()
        def collect_selected_strings(value: Any) -> None:
            if isinstance(value, str):
                selected_strings.add(value)
            elif isinstance(value, Mapping):
                for nested in value.values(): collect_selected_strings(nested)
            elif isinstance(value, list):
                for nested in value: collect_selected_strings(nested)
        collect_selected_strings(selected)
        for candidate_index, candidate in enumerate(candidates, 1):
            if candidate_index == selected_index or not isinstance(candidate, Mapping):
                continue
            for field in ("before_state", "required_transition", "after_state"):
                value = candidate.get(field)
                if (isinstance(value, str) and value not in selected_strings and
                        value in final_strings):
                    leak_count += 1
                    errors.append(f"{prefix}_unselected_{field}_leak")
            for beat in candidate.get("interaction_beats") or []:
                if (isinstance(beat, str) and beat not in selected_strings and
                        beat in final_strings):
                    leak_count += 1
                    errors.append(f"{prefix}_unselected_interaction_beat_leak")
            for binding in candidate.get("actor_bindings") or []:
                value = binding.get("binding") if isinstance(binding, Mapping) else None
                if (isinstance(value, str) and value not in selected_strings and
                        value in final_strings):
                    leak_count += 1
                    errors.append(f"{prefix}_unselected_actor_binding_leak")
    return {"valid": not errors, "errors": errors,
            "selected_candidate_refs": selected_refs,
            "rejected_content_leak_count": leak_count}


def build_semantic_gate_context(*, parent: Mapping[str, Any],
                                candidates: Sequence[Mapping[str, Any]],
                                plan: Mapping[str, Any],
                                memory_only_fast: bool = False,
                                global_memory_state_view: Mapping[str, Any] | None = None,
                                minimum_pass_candidates: int = 3,
                                ) -> dict[str, Any]:
    """Build candidate coverage context.

    The historical path exposes authoritative Parent evidence.  The fast
    Memory-only path deliberately omits transcript text: Planner owns novelty
    relative to the Memory state, while this Gate only filters candidates.
    """
    candidate_by_id = {
        str(row.get("memory_event_id")): dict(row) for row in candidates}
    turn_by_ref = _parent_turn_map(parent)
    parent_turns = list(parent["dialogue"]["turns"])
    scenes: list[dict[str, Any]] = []
    expected_candidate_refs: list[str] = []
    source_scenes = plan.get("candidate_pool") or plan.get("scenes") or []
    for scene in source_scenes:
        event_id = str(scene.get("memory_event_id") or "")
        mode = str(scene.get("development_mode") or "continuation")
        refs = [str(value) for value in scene.get("evidence_turn_refs") or []]
        session_ids = list(dict.fromkeys(
            str(turn_by_ref[ref].get("session_id"))
            for ref in refs if ref in turn_by_ref))
        # Fast Memory-only Gate input never carries Parent transcript text.
        # Keep the legacy evidence window only for historical/non-fast runs.
        if memory_only_fast:
            evidence_window = []
        else:
            scoped_parent_turns = (parent_turns if mode == "new_development" else [
                row for row in parent_turns
                if str(row.get("session_id")) in session_ids])
            evidence_window = [{
                "turn_ref": str(row.get("turn_id")),
                "speaker": str(row.get("speaker")),
                "text": str(row.get("text")),
            } for row in scoped_parent_turns]
        candidate_rows = [{
            "candidate_ref": _candidate_ref(event_id, candidate_index),
            "before_state": candidate.get("before_state"),
            "planner_current_state_stage": candidate.get(
                "planner_current_state_stage"),
            "planner_current_state_summary": candidate.get(
                "planner_current_state_summary"),
            "planner_state_basis_refs": candidate.get(
                "planner_state_basis_refs"),
            "planner_completion_condition": candidate.get(
                "planner_completion_condition"),
            "required_transition": candidate.get("required_transition"),
            "after_state": candidate.get("after_state"),
            "actor_bindings": candidate.get("actor_bindings"),
            "interaction_beats": candidate.get("interaction_beats"),
            "development_mode": candidate.get("development_mode"),
            "source_basis": candidate.get("source_basis"),
            "new_people": candidate.get("new_people"),
        } for candidate_index, candidate in enumerate(
            scene.get("next_step_candidates") or [], 1)
            if isinstance(candidate, Mapping)]
        expected_candidate_refs.extend(
            str(candidate["candidate_ref"]) for candidate in candidate_rows)
        scenes.append({
            "memory_event_id": event_id,
            "development_mode": mode,
            "memory_event": (candidate_by_id.get(event_id) or
                             candidate_by_id.get(str(
                                 scene.get("memory_event_ref") or ""), {})),
            "source_basis": scene.get("source_basis"),
            "new_people": scene.get("new_people"),
            "planner_verified_current_state": scene.get(
                "verified_current_state"),
            "planner_evidence_turn_refs": refs,
            "authoritative_parent_evidence": evidence_window,
            "candidates": candidate_rows,
        })
    return {"gate_context_version": (
                "semantic_gate_memory_candidate_coverage_v1"
                if memory_only_fast else
                "semantic_gate_candidate_coverage_v1"),
            "memory_only_fast": bool(memory_only_fast),
            "global_memory_state_view": dict(global_memory_state_view or {}),
            "expected_candidate_count": len(expected_candidate_refs),
            "expected_candidate_refs": expected_candidate_refs,
            "scenes": scenes,
            "minimum_pass_candidates": int(minimum_pass_candidates)}


def _global_memory_fact_refs(view: Mapping[str, Any] | None) -> set[str]:
    refs: set[str] = set()
    for rows in ((view or {}).get("buckets") or {}).values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for fact in row.get("memory_facts") or []:
                if isinstance(fact, Mapping) and fact.get("memory_ref"):
                    refs.add(str(fact["memory_ref"]))
    return refs


def semantic_gate_review_fields(*, strict_memory_contract: bool) -> dict[str, Any]:
    """Shared field definition; mode-specific validation behavior is unchanged."""
    evidence_key = "supporting_memory_refs" if strict_memory_contract else "evidence_turn_refs"
    required = {"candidate_ref", "decision", "duplicate_with_parent", evidence_key,
                "parent_relation"}
    return {"evidence_key": evidence_key, "required": required,
            "allowed": required | {"reason"}}


def validate_semantic_gate_output(
        value: Any, *, plan: Mapping[str, Any], parent: Mapping[str, Any],
        memory_only_fast: bool = False,
        global_memory_state_view: Mapping[str, Any] | None = None,
        minimum_pass_candidates: int = 3,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {"candidate_reviews"}:
        return {"valid": False, "errors": ["gate_top_level_shape"],
                "overall_decision": "invalid"}
    reviews = value.get("candidate_reviews")
    expected_by_ref: dict[str, str] = {}
    refs_by_event: dict[str, list[str]] = {}
    source_scenes = plan.get("candidate_pool") or plan.get("scenes") or []
    for scene in source_scenes:
        event_id = str(scene.get("memory_event_id") or "")
        refs_by_event[event_id] = []
        for index, _ in enumerate(scene.get("next_step_candidates") or [], 1):
            ref = _candidate_ref(event_id, index)
            expected_by_ref[ref] = event_id
            refs_by_event[event_id].append(ref)
    parent_refs = set(_parent_turn_map(parent))
    strict_memory_contract = bool(memory_only_fast and
                                  global_memory_state_view is not None)
    memory_refs = _global_memory_fact_refs(global_memory_state_view)
    if strict_memory_contract and not memory_refs:
        errors.append("gate_global_memory_refs_required")
    if not isinstance(reviews, list) or len(reviews) != len(expected_by_ref):
        errors.append("gate_review_count")
        reviews = reviews if isinstance(reviews, list) else []
    seen: set[str] = set()
    passed: list[str] = []
    rejected: list[str] = []
    normalized_reviews: list[dict[str, Any]] = []
    for index, review in enumerate(reviews, 1):
        prefix = f"gate_review_{index}"
        if not isinstance(review, Mapping):
            errors.append(f"{prefix}_not_object")
            continue
        fields = semantic_gate_review_fields(strict_memory_contract=strict_memory_contract)
        evidence_key = fields["evidence_key"]
        required_keys = fields["required"]
        allowed_keys = fields["allowed"]
        if not required_keys.issubset(review) or not set(review).issubset(
                allowed_keys):
            errors.append(f"{prefix}_shape")
        candidate_ref = str(review.get("candidate_ref") or "")
        if candidate_ref not in expected_by_ref:
            errors.append(f"{prefix}_unknown_candidate")
        if candidate_ref in seen:
            errors.append(f"{prefix}_duplicate_candidate")
        seen.add(candidate_ref)
        decision = str(review.get("decision") or "")
        duplicate = review.get("duplicate_with_parent")
        refs = review.get(evidence_key)
        parent_relation = str(review.get("parent_relation") or "")
        reason = str(review.get("reason") or "").strip()
        if decision not in {"pass", "reject"}:
            errors.append(f"{prefix}_decision")
        if parent_relation not in PARENT_RELATIONS:
            errors.append(f"{prefix}_parent_relation")
        if not isinstance(duplicate, bool):
            errors.append(f"{prefix}_duplicate_boolean")
        if parent_relation in PARENT_RELATIONS:
            expected_decision = "reject" if parent_relation == "REALIZED" else "pass"
            if decision != expected_decision:
                errors.append(f"{prefix}_parent_relation_decision_conflict")
            if duplicate is not (parent_relation == "REALIZED"):
                errors.append(f"{prefix}_parent_relation_duplicate_conflict")
        if not isinstance(refs, list):
            errors.append(f"{prefix}_evidence_shape")
            refs = []
        if len(refs) != len(set(str(ref) for ref in refs)):
            errors.append(f"{prefix}_evidence_duplicate")
        if strict_memory_contract:
            if any(str(ref) not in memory_refs for ref in refs):
                errors.append(f"{prefix}_evidence_not_memory")
        elif any(str(ref) not in parent_refs for ref in refs):
            errors.append(f"{prefix}_evidence_not_parent")
        if parent_relation == "REALIZED" and not refs:
            errors.append(f"{prefix}_reject_evidence_required")
        if strict_memory_contract and parent_relation == "REALIZED" and not reason:
            errors.append(f"{prefix}_reject_reason_required")
        if not reason:
            reason = (
                "The candidate duplicates a transition already realized in "
                "the authoritative Parent."
                if parent_relation == "REALIZED"
                else "No realized Parent transition was identified."
            )
        normalized_reviews.append({
            "candidate_ref": candidate_ref,
            "parent_relation": parent_relation,
            "decision": decision,
            "duplicate_with_parent": duplicate,
            "evidence_turn_refs": [str(ref) for ref in refs],
            "supporting_memory_refs": (
                [str(ref) for ref in refs] if strict_memory_contract else []),
            "reason": reason,
        })
        if decision == "reject":
            rejected.append(candidate_ref)
        elif decision == "pass":
            passed.append(candidate_ref)
    if sorted(seen) != sorted(expected_by_ref):
        errors.append("gate_candidate_coverage")
    events_without_pass = [
        event_id for event_id, refs in refs_by_event.items()
        if not any(ref in passed for ref in refs)]
    minimum = int(minimum_pass_candidates)
    overall = "invalid" if errors else (
        "reject" if len(passed) < minimum else "pass")
    return {"valid": not errors, "errors": errors,
            "overall_decision": overall,
            "passed_candidate_refs": passed,
            "rejected_candidate_refs": rejected,
            "events_without_pass": events_without_pass,
            "passed_candidate_count": len(passed),
            "minimum_pass_candidates": minimum,
            "realizer_allowed": not errors and len(passed) >= minimum,
            "normalized_candidate_reviews": normalized_reviews}


def validate_realized_speaker_coverage(
    turns: Sequence[Any], *, scene: Mapping[str, Any], prefix: str,
) -> dict[str, Any]:
    """Validate identity syntax and required-person coverage only.

    Extra speakers are legal. Whether an extra person steals a bound action is
    a semantic Adherence decision, not a mechanical speaker-list decision.
    """
    errors: list[str] = []
    raw_required = scene.get("required_participants")
    if raw_required is None:
        raw_required = scene.get("participants")
    required = [str(value).strip() for value in (raw_required or [])
                if str(value).strip()]
    if not required:
        errors.append(f"{prefix}:required_participants_empty")
    actual: list[str] = []
    for index, row in enumerate(turns, 1):
        speaker = (str(row.get("speaker") or "").strip()
                   if isinstance(row, Mapping) else "")
        if not speaker or is_transport_speaker_label(speaker):
            errors.append(f"{prefix}:turn_{index}_speaker")
            continue
        if speaker not in actual:
            actual.append(speaker)
    for participant in required:
        if participant not in actual:
            errors.append(
                f"{prefix}:required_participant_missing:{participant}")
    return {
        "valid": not errors,
        "errors": errors,
        "required_participants": required,
        "actual_speakers": actual,
        "additional_speakers": [speaker for speaker in actual
                                if speaker not in set(required)],
    }


def normalize_realizer_output(value: Any, *, plan: Mapping[str, Any],
                              parent: Mapping[str, Any],
                              round_index: int = 1,
                              scene_index_offset: int = 0) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {"scenes"}:
        return {"valid": False, "errors": ["realizer_top_level_shape"]}
    raw_scenes = value.get("scenes")
    planned = list(plan.get("scenes") or [])
    if not isinstance(raw_scenes, list) or len(raw_scenes) != len(planned):
        return {"valid": False, "errors": ["realizer_scene_count"]}
    parent_texts = {str(row.get("text")) for row in parent["dialogue"]["turns"]}
    child_scenes: list[dict[str, Any]] = []
    speaker_coverage: list[dict[str, Any]] = []
    for idx, (raw, plan_scene) in enumerate(zip(raw_scenes, planned), 1):
        ref = str(plan_scene.get("scene_ref") or
                  f"R{int(round_index):02d}:S{idx + int(scene_index_offset):02d}")
        if not isinstance(raw, Mapping): errors.append(f"{ref}:scene_not_object"); continue
        turns = raw.get("dialogue_turns")
        if not isinstance(turns, list) or not 9 <= len(turns) <= 48:
            errors.append(f"{ref}:turn_budget")
            turns = turns if isinstance(turns, list) else []
        coverage = validate_realized_speaker_coverage(
            turns, scene=plan_scene, prefix=ref)
        errors.extend(coverage["errors"])
        speaker_coverage.append({"scene_ref": ref, **coverage})
        normalized=[]
        seen_text=set()
        for j, row in enumerate(turns,1):
            if not isinstance(row, Mapping): errors.append(f"{ref}:turn_{j}_object"); continue
            speaker=str(row.get("speaker") or "").strip(); text=str(row.get("text") or "").strip()
            if not text: errors.append(f"{ref}:turn_{j}_empty")
            if text in parent_texts: errors.append(f"{ref}:turn_{j}_parent_exact_copy")
            if text in seen_text: errors.append(f"{ref}:turn_{j}_duplicate_text")
            seen_text.add(text)
            normalized.append({"turn_ref":f"{ref}:T{j:03d}","speaker":speaker,"text":text})
        child_scenes.append({"scene_ref":ref,"session_ref":f"{parent['root_id']}:r{int(round_index):02d}:s{idx:02d}",
                             "memory_event_id":plan_scene.get("memory_event_id"),
                             "dialogue_turns":normalized})
    return {"valid": not errors, "errors": errors,
            "speaker_coverage": speaker_coverage,
            "child":{"schema_version":"memory_event_continuation_child_v1",
                      "root_id":parent["root_id"],"round_index":int(round_index),"scenes":child_scenes}}
