"""Independent model revision of v13's scene-three candidates."""

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
V13_ROOT = PARENT / "h4_crossroot_multiselect_batch_v13"
SLOT = 4
ROOTS = ("cc-train-episode-136202", "cc-train-episode-197635", "cc-train-episode-254852")
SEEDS = {
    "cc-train-episode-136202": {
        "A": "Emily said she feels more in control of her reactions now.",
        "C": "Emily said she used to get frustrated when people interrupted her.",
        "E": "Emily said James is not jumping in as much as before.",
    },
    "cc-train-episode-197635": {
        "A": "Taylor said they planned to bring a notebook to jot down notes.",
        "C": "Taylor said they planned to try candle lighting at home.",
        "D": "Casey said they planned to start practicing some mindfulness techniques they had been taught.",
        "E": "Taylor said the workshop was more hands-on than expected.",
    },
    "cc-train-episode-254852": {
        "A": "Jamie said he felt overwhelmed and needed a break and fresh air.",
        "B": "Robin said she would bring her phone to take pictures of flowers and birds along the way.",
        "E": "Jamie said he had not taken a proper walk in a while.",
        "F": "Robin said she would take her time and not rush.",
    },
}
FORBIDDEN_THEMES = {
    "cc-train-episode-136202": ["Sarah friendship status", "handling everything alone", "therapy session appointment", "sending a message"],
    "cc-train-episode-197635": ["reunion music and playlist", "friends' RSVPs", "photo booth", "old clothing", "Jewish candle ritual already performed"],
    "cc-train-episode-254852": ["flight and accommodation booking", "coffee proposal", "cushion", "thermos and pastries", "surfboard and swimsuit"],
}
CONTRAST_ANCHORS = {
    "cc-train-episode-136202": ["R02:S03:T008", "R02:S03:T010"],
    "cc-train-episode-197635": ["R02:S03:T009", "R02:S03:T020"],
    "cc-train-episode-254852": ["R02:S03:T006", "R02:S03:T007"],
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_base():
    if sha(V12_PATH.read_bytes()) != V12_SHA256:
        raise ValueError("v12_runner_dependency_changed")
    spec = importlib.util.spec_from_file_location("v14_frozen_v12", V12_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.__file__ = __file__
    module.SLOTS = (SLOT,)
    return module


base = load_base()
original_check = base.check_candidate
original_validate = base.validate


def parent_input(root_id: str) -> tuple[dict, bytes]:
    path = V13_ROOT / root_id / "slot-04/PARSED_RESPONSE.json"
    raw = path.read_bytes()
    parsed = json.loads(raw)
    questions = parsed.get("questions") if isinstance(parsed, dict) else None
    if not isinstance(questions, list) or len(questions) != 1:
        raise ValueError("v13_parent_question_count_changed")
    row = questions[0]
    if row.get("root_id") != root_id or row.get("candidate_id") != base.slot_id(root_id, SLOT):
        raise ValueError("v13_parent_identity_changed")
    return row, raw


def older_questions(root_id: str) -> list[dict]:
    rows = base.selected_questions(root_id)
    paths = [V8_ROOT / root_id / "PARSED_RESPONSE.json"] + [
        V12_ROOT / root_id / f"slot-{number:02d}/PARSED_RESPONSE.json" for number in (2, 3)]
    for path in paths:
        parsed = json.loads(path.read_text())
        questions = parsed.get("questions") if isinstance(parsed, dict) else None
        if not isinstance(questions, list):
            raise ValueError(f"prior_questions_invalid:{path}")
        rows.extend(row for row in questions if isinstance(row, dict))
    return rows


def prior_questions(root_id: str, slot: int) -> list[dict]:
    if slot != SLOT or root_id not in ROOTS:
        raise ValueError("unknown_revision_target")
    parent, _ = parent_input(root_id)
    return older_questions(root_id) + [parent]


def refs(questions: list[dict]) -> set[str]:
    return {item["turn_ref"] for row in questions
            for item in (row.get("option_evidence") or {}).values()
            if isinstance(item, dict) and isinstance(item.get("turn_ref"), str)}


def seed_view(root_id: str, parent: dict) -> list[dict]:
    source = json.loads((V13_ROOT / root_id / "slot-04/source.json").read_text())
    source_by_ref = {turn["turn_ref"]: turn for turn in source["turns"]}
    result = []
    for label, revised_claim in SEEDS[root_id].items():
        item = parent["option_evidence"][label]
        if label not in parent["gold"] or item["relation"] != "supports":
            raise ValueError("v13_seed_not_true")
        full_quote = source_by_ref[item["turn_ref"]]["text"]
        if len(full_quote) > 240:
            raise ValueError("seed_full_quote_too_long")
        result.append({"parent_label": label, "revised_supported_claim": revised_claim,
                       "turn_ref": item["turn_ref"], "quote": full_quote})
    return result


def visible_turns(source: dict, root_id: str, parent: dict, old: list[dict]) -> list[dict]:
    unused_old = refs(old)
    seed_refs = {item["turn_ref"] for item in seed_view(root_id, parent)}
    view = [turn for turn in source["turns"] if turn["turn_ref"].startswith("R02:S03:")
            and (turn["turn_ref"] not in unused_old or turn["turn_ref"] in seed_refs)]
    if len(view) < 6 or len(seed_refs) != len(SEEDS[root_id]):
        raise ValueError("insufficient_scene_three_revision_anchors")
    return view


def prompt_for(root_id: str, slot: int, source: dict, instructions: str,
               prior: list[dict], v8) -> str:
    parent, old = prior[-1], prior[:-1]
    view = v8.generation_view(source)
    view["turns"] = visible_turns(source, root_id, parent, old)
    view["excerpt_turn_count"] = len(view["turns"])
    view["generation_view_type"] = "r02_scene_three_targeted_revision_v14"
    context = {
        "revision_parent_candidate_id": parent["candidate_id"],
        "revision_parent_parsed_sha256": sha(parent_input(root_id)[1]),
        "safe_true_seed_claims_and_quotes": seed_view(root_id, parent),
        "disallowed_old_source_refs": sorted(refs(old)),
        "disallowed_old_themes": FORBIDDEN_THEMES[root_id],
        "suggested_new_false_contrast_turn_refs": CONTRAST_ANCHORS[root_id],
        "prior_gold_patterns": ["".join(sorted(row["gold"]))
            for row in prior if isinstance(row.get("gold"), list)],
        "source": view,
    }
    text = instructions.replace("{ROOT_ID}", root_id).replace(
        "{CANDIDATE_ID}", base.slot_id(root_id, slot))
    return text + json.dumps(context, ensure_ascii=False, indent=2) + "\n"


def check_candidate(row: object, root_id: str, slot: int, turns: list[dict],
                    prior: list[dict], v8) -> list[str]:
    old, parent = prior[:-1], prior[-1]
    errors = original_check(row, root_id, slot, turns, old, v8)
    if not isinstance(row, dict):
        return errors
    gold = row.get("gold")
    if v8.valid_gold(gold) and len(gold) == len(parent["gold"]) and set(gold) == set(parent["gold"]):
        errors.append("gold_position_pattern_repeated_from_parent")
    evidence = row.get("option_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(base.LABELS):
        return errors
    permitted = {turn["turn_ref"] for turn in visible_turns(
        {"turns": turns}, root_id, parent, old)}
    cited = [item.get("turn_ref") if isinstance(item, dict) else None
             for item in evidence.values()]
    if any(ref not in permitted for ref in cited):
        errors.append("anchor_not_permitted_for_revision")
    if len(set(cited)) != 6:
        errors.append("six_distinct_evidence_turns_required")
    for seed in seed_view(root_id, parent):
        matching = [label for label, item in evidence.items() if isinstance(item, dict) and
                    item.get("turn_ref") == seed["turn_ref"]]
        if len(matching) != 1 or matching[0] not in (gold or []) or \
           evidence[matching[0]].get("relation") != "supports":
            errors.append("supported_seed_anchor_missing:" + seed["turn_ref"])
    return errors


def check_binding(root_id: str) -> dict:
    path = HERE / root_id / "slot-04/REVISION_BINDING.json"
    binding = json.loads(path.read_text())
    _, raw = parent_input(root_id)
    if binding.get("parent_parsed_sha256") != sha(raw) or \
       binding.get("parent_candidate_id") != base.slot_id(root_id, SLOT) or \
       binding.get("revision_kind") != "MODEL_TARGETED_REVISION_OF_V13":
        raise ValueError("revision_parent_binding_changed")
    return binding


def prepare(root_id: str) -> dict:
    parent, raw = parent_input(root_id)
    result = base.prepare(root_id, SLOT)
    _, batch = base.dependency()
    batch.ordering.write_new(HERE / root_id / "slot-04/REVISION_BINDING.json", {
        "revision_kind": "MODEL_TARGETED_REVISION_OF_V13",
        "parent_candidate_id": parent["candidate_id"],
        "parent_path": str(V13_ROOT / root_id / "slot-04/PARSED_RESPONSE.json"),
        "parent_parsed_sha256": sha(raw),
        "requested_model": batch.MODEL,
        "model_authored_revision": True,
        "codex_content_review_completed": False,
        "solver_evaluated": False, "training_started": False,
    })
    return result


def validate(root_id: str) -> dict:
    check_binding(root_id)
    result = original_validate(root_id, SLOT)
    _, batch = base.dependency()
    batch.ordering.write_new(HERE / root_id / "slot-04/REVISION_RESULT.json", {
        "revision_kind": "MODEL_TARGETED_REVISION_OF_V13",
        "parent_candidate_id": base.slot_id(root_id, SLOT),
        "model": batch.MODEL,
        "mechanical_status": result["status"],
        "content_review_pending": True,
        "solver_evaluated": False, "training_started": False,
    })
    return result


base.prior_questions = prior_questions
base.prompt_for = prompt_for
base.check_candidate = check_candidate
base.validate = validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "execute", "validate"))
    parser.add_argument("root_id", choices=ROOTS)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.root_id)
    elif args.action == "preflight":
        binding = check_binding(args.root_id)
        _, _, _, manifest, source, prior, prompt, _ = base.inputs(args.root_id, SLOT)
        result = {"status": "FROZEN_REVISION_INPUTS_VERIFIED_NO_API_CALL",
                  "root_id": args.root_id, "parent": binding["parent_candidate_id"],
                  "seed_count": len(seed_view(args.root_id, prior[-1])),
                  "visible_turn_count": len(visible_turns(source, args.root_id, prior[-1], prior[:-1])),
                  "prompt_bytes": len(prompt.encode()),
                  "payload_sha256": manifest["payload_sha256"]}
    elif args.action == "execute":
        check_binding(args.root_id)
        result = base.execute(args.root_id, SLOT)
    else:
        result = validate(args.root_id)
    print(json.dumps(result, ensure_ascii=False))
