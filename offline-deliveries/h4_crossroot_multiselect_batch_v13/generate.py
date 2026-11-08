"""Source-diverse new candidates from unused R02 scene-three turns."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
V12_PATH = PARENT / "h4_crossroot_multiselect_batch_v12/generate.py"
V12_SHA256 = "ef450533d4ef8356e54233accc53a58c148cae8f0391122feaa3f3cc8afb6025"
V8_ROOT = PARENT / "h4_crossroot_multiselect_batch_v8"
V12_ROOT = PARENT / "h4_crossroot_multiselect_batch_v12"
SLOT = 4


def load_base():
    if hashlib.sha256(V12_PATH.read_bytes()).hexdigest() != V12_SHA256:
        raise ValueError("v12_runner_dependency_changed")
    spec = importlib.util.spec_from_file_location("v13_frozen_v12", V12_PATH)
    assert spec and spec.loader
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.HERE = HERE
    base.__file__ = __file__
    base.SLOTS = (SLOT,)
    return base


base = load_base()
original_check = base.check_candidate


def prior_questions(root_id: str, slot: int) -> list[dict]:
    if slot != SLOT:
        raise ValueError("v13_slot_must_be_04")
    rows = base.selected_questions(root_id)
    original = V8_ROOT / root_id / "PARSED_RESPONSE.json"
    if not original.exists():
        raise ValueError("v8_source_candidate_missing")
    for path in [original] + [V12_ROOT / root_id / f"slot-{number:02d}/PARSED_RESPONSE.json"
                              for number in (2, 3)]:
        parsed = json.loads(path.read_text())
        questions = parsed.get("questions") if isinstance(parsed, dict) else None
        if not isinstance(questions, list):
            raise ValueError(f"prior_questions_invalid:{path}")
        rows.extend(row for row in questions if isinstance(row, dict))
    return rows


def anchor_refs(questions: list[dict]) -> set[str]:
    return {item["turn_ref"] for row in questions
            for item in (row.get("option_evidence") or {}).values()
            if isinstance(item, dict) and isinstance(item.get("turn_ref"), str)}


def allowed_turns(source: dict, prior: list[dict]) -> list[dict]:
    used = anchor_refs(prior)
    allowed = [turn for turn in source["turns"]
               if turn["turn_ref"].startswith("R02:S03:") and turn["turn_ref"] not in used]
    if len(allowed) < 6:
        raise ValueError("insufficient_unused_scene_three_turns")
    return allowed


def prompt_for(root_id: str, slot: int, source: dict, instructions: str,
               prior: list[dict], v8) -> str:
    view = v8.generation_view(source)
    view["turns"] = allowed_turns(source, prior)
    view["excerpt_turn_count"] = len(view["turns"])
    view["generation_view_type"] = "r02_scene_three_unused_anchors_v1"
    view["scope_note"] = "Unused R02 scene-three turns only; complete root retained for content review."
    context = {"previous_questions_avoid_repeating": base.dedup_view(prior),
               "source": view}
    text = instructions.replace("{ROOT_ID}", root_id).replace(
        "{CANDIDATE_ID}", base.slot_id(root_id, slot))
    return text + json.dumps(context, ensure_ascii=False, indent=2) + "\n"


def check_candidate(row: object, root_id: str, slot: int, turns: list[dict],
                    prior: list[dict], v8) -> list[str]:
    errors = original_check(row, root_id, slot, turns, prior, v8)
    if not isinstance(row, dict):
        return errors
    evidence = row.get("option_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(base.LABELS):
        return errors
    permitted = {turn["turn_ref"] for turn in allowed_turns({"turns": turns}, prior)}
    refs = [item.get("turn_ref") if isinstance(item, dict) else None
            for item in evidence.values()]
    if any(ref not in permitted for ref in refs):
        errors.append("anchor_not_from_unused_r02_scene_three")
    if len(set(refs)) != 6:
        errors.append("six_distinct_evidence_turns_required")
    return errors


base.prior_questions = prior_questions
base.prompt_for = prompt_for
base.check_candidate = check_candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "execute", "validate"))
    parser.add_argument("root_id", choices=(
        "cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852"))
    args = parser.parse_args()
    if args.action == "prepare":
        result = base.prepare(args.root_id, SLOT)
    elif args.action == "preflight":
        _, _, _, manifest, source, prior, prompt, _ = base.inputs(args.root_id, SLOT)
        result = {"status": "FROZEN_INPUTS_VERIFIED_NO_API_CALL", "root_id": args.root_id,
                  "slot": SLOT, "prior_count": len(prior),
                  "allowed_turns": len(allowed_turns(source, prior)),
                  "prompt_bytes": len(prompt.encode()),
                  "payload_sha256": manifest["payload_sha256"]}
    elif args.action == "execute":
        result = base.execute(args.root_id, SLOT)
    else:
        result = base.validate(args.root_id, SLOT)
    print(json.dumps(result, ensure_ascii=False))
