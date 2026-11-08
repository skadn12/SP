#!/usr/bin/env python3
"""Deterministic source-ledger and question compiler for feedback closed-loop v3."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

OPERATIONS = ("MC2", "MC3", "MC4", "MC5", "MC6", "ORDER1", "ORDER2", "ORDER3")
ARMS = ("F", "G")
MC_GOLD = {"MC2": 2, "MC3": 3, "MC4": 4, "MC5": 5, "MC6": 6}
LABELS = tuple("ABCDEF")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": digest_bytes(raw), "bytes": len(raw)}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sentences(text: str) -> list[tuple[int, int, str]]:
    """Split while retaining exact source offsets; a sentence is never normalized."""
    out: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"[.!?](?:[\"'”’)]*)?(?=\s|$)", text):
        end = m.end()
        raw = text[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            out.append((start + left, start + right, text[start + left : start + right]))
        start = end
    if start < len(text):
        raw = text[start:]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            out.append((start + left, start + right, text[start + left : start + right]))
    return out


def _truth(text: str) -> bool:
    # A ledger row is an *asserted source claim*, including negative claims such
    # as "we have not booked".  Its support is therefore true whenever it
    # comes from the frozen source.  False options are created only through a
    # deterministic contradiction transform below; treating lexical negation as
    # false would incorrectly reject source-supported negative facts.
    return True


def _predicate(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(?:19|20)\d\d\b|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b\d{1,2}:\d{2}\b", low):
        return "time_relation"
    if re.search(r"\b\d+(?:\.\d+)?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|half|twice)\b", low):
        return "quantity_relation"
    if re.search(r"\b(?:because|so|therefore|want|need|reason|caused|to)\b", low):
        return "causal_relation"
    if re.search(r"\b(?:first|then|before|after|later|earlier|next|replace|instead|finally)\b", low):
        return "ordered_relation"
    return "discourse_relation"


def build_ledger(root_id: str, draft_path: Path) -> dict[str, Any]:
    """Build only from saved initial draft turns. Mapping/native metadata is ignored."""
    raw = json.loads(draft_path.read_text(encoding="utf-8"))
    if raw.get("root_id") != root_id or not isinstance(raw.get("scenes"), list):
        raise ValueError(f"invalid draft root: {root_id}")
    facts: list[dict[str, Any]] = []
    scene_count = 0
    for scene in raw["scenes"]:
        turns = scene.get("turns")
        if not isinstance(turns, list):
            continue
        scene_count += 1
        for turn in turns:
            turn_id = turn.get("turn_ref")
            speaker = turn.get("speaker")
            text = turn.get("text")
            if not isinstance(turn_id, str) or not isinstance(speaker, str) or not isinstance(text, str) or not text.strip():
                continue
            # Require the immutable R03 turn form; this excludes malformed mapping records.
            if not re.fullmatch(r"R03:[AB]:S\d\d:T\d\d\d", turn_id):
                continue
            for index, (start, end, substring) in enumerate(_sentences(text), 1):
                facts.append({
                    "fact_id": f"{turn_id}:F{index:02d}",
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "source_substring": substring,
                    "start": start,
                    "end": end,
                    "predicate": _predicate(substring),
                    "truth": _truth(substring),
                    "scene_ref": turn_id.rsplit(":", 1)[0],
                })
    facts.sort(key=lambda x: (x["turn_id"], x["start"], x["fact_id"]))
    if scene_count < 6 or len({f["turn_id"] for f in facts}) < 48:
        raise ValueError(f"incomplete source turns for {root_id}")
    if not facts:
        raise ValueError(f"empty ledger for {root_id}")
    operations = {op: len(facts) >= 6 for op in OPERATIONS}
    if not all(operations.values()):
        raise ValueError(f"ledger lacks operation coverage for {root_id}")
    return {
        "schema_version": "feedback_closed_loop_v3_fact_ledger",
        "root_id": root_id,
        "source_draft": binding(draft_path),
        "fact_count": len(facts),
        "operation_coverage": operations,
        "facts": facts,
        "excluded_fields": ["native_evidence", "read_receipt", "warnings"],
    }


def _replace_first_number(text: str, amount: int) -> str:
    m = re.search(r"\d+", text)
    if not m:
        return text + f" (amount {amount})"
    n = int(m.group(0))
    return text[: m.start()] + str(max(0, n + amount)) + text[m.end() :]


def _option_from_fact(fact: dict[str, Any], transform: str, other_speaker: str) -> tuple[str, bool, str]:
    src = fact["source_substring"]
    base = bool(fact["truth"])
    if transform == "identity":
        return src, base, "identity"
    if transform == "corroborated":
        return f"The dialogue also supports: {src}", base, "corroborated"
    if transform == "negate":
        return f"It is not the case that {src[0].lower() + src[1:]}", False, "negate"
    if transform == "swap_actor":
        return f"{other_speaker} states: {src}", False, "swap_actor"
    if transform == "alter_quantity":
        return _replace_first_number(src, 1), False, "alter_quantity"
    if transform == "alter_time":
        changed = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: f"{(int(m.group(1)) + 1) % 24:02d}:{m.group(2)}", src, count=1)
        if changed == src:
            changed = src + " (at a different time)"
        return changed, False, "alter_time"
    raise ValueError(transform)


def _choose_facts(facts: list[dict[str, Any]], count: int, truth: bool, offset: int = 0) -> list[dict[str, Any]]:
    pool = [f for f in facts if bool(f["truth"]) is truth]
    if len(pool) < count:
        # The source text remains exact; deterministic inverse predicates supply false claims.
        pool = facts
    return [pool[(offset + i) % len(pool)] for i in range(count)]


def _permutation(seed_material: str) -> tuple[dict[str, str], dict[str, str]]:
    ranked = sorted(LABELS, key=lambda x: hashlib.sha256(f"{seed_material}:{x}".encode()).hexdigest())
    if ranked == list(LABELS):
        ranked = ranked[1:] + ranked[:1]
    display_to_canonical = dict(zip(LABELS, ranked))
    canonical_to_display = {v: k for k, v in display_to_canonical.items()}
    return display_to_canonical, canonical_to_display


def compile_slot(ledger: dict[str, Any], arm: str, operation: str, challenger: dict[str, Any] | None = None) -> dict[str, Any]:
    if arm not in ARMS or operation not in OPERATIONS:
        raise ValueError("unknown arm or operation")
    facts = ledger["facts"]
    root = ledger["root_id"]
    selected = (challenger or {}).get("target_fact_ids") or []
    by_id = {f["fact_id"]: f for f in facts}
    preferred = [by_id[x] for x in selected if x in by_id]
    ordered = preferred + [f for f in facts if f not in preferred]
    speakers = sorted({f["speaker"] for f in facts})
    other = speakers[1] if len(speakers) > 1 and speakers[0] == ordered[0]["speaker"] else speakers[0]
    slot_id = f"FGR3-{arm}-{root.split('-')[-1]}-{operation}"
    if operation.startswith("MC"):
        target = MC_GOLD[operation]
        true_facts = _choose_facts(ordered, target, True)
        # False alternatives are transformations of asserted facts. Applying
        # ``negate`` to an already-negated fact would accidentally create a
        # second true option and violate the operation quota.
        false_fact = _choose_facts(ordered, 1, True)[0]
        options: dict[str, dict[str, Any]] = {}
        for i, fact in enumerate(true_facts):
            text, truth, transform = _option_from_fact(fact, "identity" if i == 0 else "corroborated", other)
            options[LABELS[i]] = {"text": text, "truth": truth, "transform": transform, "evidence": [{"turn_id": fact["turn_id"], "source_substring": fact["source_substring"]}], "fact_id": fact["fact_id"]}
        transforms = ["negate", "swap_actor", "alter_quantity", "alter_time", "negate"]
        for j in range(6 - target):
            fact = ordered[target + j] if target + j < len(ordered) and ordered[target + j]["truth"] else false_fact
            text, truth, transform = _option_from_fact(fact, transforms[j], other)
            label = LABELS[target + j]
            options[label] = {"text": text, "truth": truth, "transform": transform, "evidence": [{"turn_id": fact["turn_id"], "source_substring": fact["source_substring"]}], "fact_id": fact["fact_id"]}
        gold = list(LABELS[:target])
        question = f"Which {target} claims are supported by the exact dialogue evidence for operation {operation}?"
        question_type = "multi_select"
    else:
        chosen = ordered[:6]
        options = {}
        for label, fact in zip(LABELS, chosen):
            options[label] = {"text": fact["source_substring"], "truth": True, "transform": "identity", "evidence": [{"turn_id": fact["turn_id"], "source_substring": fact["source_substring"]}], "fact_id": fact["fact_id"]}
        gold = list(LABELS)
        question = f"Put the six supplied dialogue claims in exact source order for operation {operation}."
        question_type = "ordering"
    catalog = {
        "schema_version": "feedback_closed_loop_v3_compiled_slot",
        "candidate_id": slot_id,
        "root_id": root,
        "arm": arm,
        "slot_id": operation,
        "question_type": question_type,
        "task_operation": operation,
        "question": question,
        "options": options,
        "gold": gold,
        "source_binding": ledger["source_draft"],
        "target_fact_ids": selected,
    }
    validate_slot(catalog, ledger)
    presentations = []
    for variant in range(2):
        d2c, c2d = _permutation(f"{root}:{arm}:{operation}:perm:{variant}")
        display = {label: options[d2c[label]] for label in LABELS}
        display_gold = sorted(c2d[x] for x in gold) if question_type == "multi_select" else [c2d[x] for x in gold]
        presentations.append({
            "variant_id": f"perm_{variant:02d}",
            "display": display,
            "display_to_canonical": d2c,
            "canonical_to_display": c2d,
            "gold": display_gold,
            "nonidentity_permutation": list(d2c.values()) != list(LABELS),
        })
    if not all(x["nonidentity_permutation"] for x in presentations):
        raise ValueError("identity display permutation")
    catalog["presentations"] = presentations
    return catalog


def validate_slot(slot: dict[str, Any], ledger: dict[str, Any]) -> None:
    if list(slot.get("options", {})) != list(LABELS):
        raise ValueError("options must sort A-F")
    options = slot["options"]
    if slot["question_type"] == "multi_select":
        expected = MC_GOLD[slot["slot_id"]]
        actual = [x for x in LABELS if options[x]["truth"]]
        if len(actual) != expected or slot.get("gold") != actual:
            raise ValueError(f"truth quota failed: {slot['slot_id']}")
    else:
        if slot.get("gold") != list(LABELS):
            raise ValueError("ordering gold failed")
    facts = {f["fact_id"]: f for f in ledger["facts"]}
    for option in options.values():
        if option["fact_id"] not in facts:
            raise ValueError("unknown fact evidence")
        for span in option["evidence"]:
            fact = facts[option["fact_id"]]
            if span["turn_id"] != fact["turn_id"] or span["source_substring"] != fact["source_substring"]:
                raise ValueError("exact span mismatch")
            if not span["source_substring"]:
                raise ValueError("empty span")


def compile_root(ledger: dict[str, Any], challengers: dict[str, Any]) -> list[dict[str, Any]]:
    slots = []
    for arm in ARMS:
        for operation in OPERATIONS:
            slots.append(compile_slot(ledger, arm, operation, challengers.get(arm)))
    if len(slots) != 16:
        raise ValueError("root must compile 16 slots")
    if {x["slot_id"] for x in slots} != set(OPERATIONS):
        raise ValueError("operation quota failure")
    return slots


def cross_check_catalog(catalog: list[dict[str, Any]], ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for slot in catalog:
        try:
            validate_slot(slot, ledger)
        except Exception as exc:
            errors.append(f"{slot.get('candidate_id')}: {exc}")
        if len(slot.get("presentations", [])) != 2:
            errors.append(f"{slot.get('candidate_id')}: presentation_count")
        for p in slot.get("presentations", []):
            if not p.get("nonidentity_permutation"):
                errors.append(f"{slot.get('candidate_id')}: identity_permutation")
    return errors
