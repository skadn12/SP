"""Pure input and display checks for the cross-root multiselect trial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


LABELS = tuple("ABCDEF")
VARIANTS = tuple(f"perm_{shift:02d}" for shift in range(6))
GROUND_SLOTS = tuple(range(5))
FULL_SLOTS = tuple(range(3))
POOL_STATUS = "CONTENT_AUDITED_READY_FOR_SOLVER"
GENERATOR_MODEL = "Qwen/Qwen3.6-35B-A3B"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def display(question, shift):
    if shift not in range(6):
        raise ValueError("DISPLAY_SHIFT_INVALID")
    options = question["options"]
    mapping = dict(zip(LABELS, LABELS[shift:] + LABELS[:shift]))
    inverse = {canonical: shown for shown, canonical in mapping.items()}
    return {"variant_id": VARIANTS[shift],
            "options": {shown: options[canonical] for shown, canonical in mapping.items()},
            "display_to_canonical": mapping,
            "displayed_gold": [inverse[label] for label in question["gold"]]}


def validate_pool(path):
    pool = read(path)
    questions = pool.get("questions")
    provenance = pool.get("provenance", {})
    if (pool.get("status") != POOL_STATUS or not isinstance(questions, list) or not questions
            or pool.get("candidate_count", len(questions)) != len(questions)
            or provenance.get("requested_model") != GENERATOR_MODEL
            or provenance.get("returned_model") != GENERATOR_MODEL):
        raise ValueError("CONTENT_AUDITED_MODEL_POOL_REQUIRED")
    ids, sources = set(), {}
    for question in questions:
        qid = question.get("candidate_id")
        root = question.get("root_id")
        if (not isinstance(qid, str) or not qid or not all(c.isalnum() or c in "-_" for c in qid)
                or qid in ids or not isinstance(root, str) or not root.startswith("cc-train-episode-")
                or question.get("question_type") != "multi_select"
                or not isinstance(question.get("question"), str) or not question["question"].strip()):
            raise ValueError("QUESTION_ID_ROOT_OR_TYPE_INVALID:" + str(qid))
        ids.add(qid)
        options, gold = question.get("options"), question.get("gold")
        if (not isinstance(options, dict) or tuple(options) != LABELS
                or any(not isinstance(value, str) or not value.strip() for value in options.values())
                or len({value.strip().casefold() for value in options.values()}) != 6
                or not isinstance(gold, list) or len(gold) != 3
                or len(set(gold)) != 3 or not set(gold) <= set(LABELS)):
            raise ValueError("SIX_OPTION_THREE_GOLD_CONTRACT_INVALID:" + qid)
        evidence = question.get("option_evidence")
        if not isinstance(evidence, dict) or set(evidence) != set(LABELS):
            raise ValueError("COMPLETE_OPTION_EVIDENCE_REQUIRED:" + qid)
        source_path = Path(question.get("source_path", ""))
        source_hash = question.get("source_sha256")
        if (not source_path.is_absolute() or not source_path.is_file()
                or not isinstance(source_hash, str) or sha(source_path) != source_hash):
            raise ValueError("SOURCE_PATH_OR_SHA_INVALID:" + qid)
        source = read(source_path)
        if source.get("root_id") != root or not isinstance(source.get("turns"), list) or not source["turns"]:
            raise ValueError("SOURCE_ROOT_OR_TURNS_INVALID:" + qid)
        turns = {row["turn_ref"]: row for row in source["turns"]}
        if len(turns) != len(source["turns"]):
            raise ValueError("SOURCE_TURN_REFS_NOT_UNIQUE:" + qid)
        for label, citation in evidence.items():
            if (not isinstance(citation, dict) or citation.get("turn_ref") not in turns
                    or not isinstance(citation.get("quote"), str) or not citation["quote"]
                    or citation["quote"] not in turns[citation["turn_ref"]].get("text", "")
                    or citation.get("relation") not in {"supports", "contradicts"}
                    or (citation["relation"] == "supports") != (label in gold)
                    or not isinstance(citation.get("reason"), str) or not citation["reason"].strip()):
                raise ValueError("OPTION_EVIDENCE_SOURCE_MISMATCH:" + qid + ":" + label)
        previous = sources.get(root)
        if previous is not None and previous["sha256"] != source_hash:
            raise ValueError("ROOT_SOURCE_VERSION_CONFLICT:" + root)
        sources[root] = {"path": str(source_path), "sha256": source_hash,
                         "turn_count": len(source["turns"]), "source": source}
    return pool, questions, sources


def request_plan(questions, sources, make_prompt):
    requests, prompts = [], {}
    for question in questions:
        root = question["root_id"]
        for mode, cells in (("no_context", ((0, slot) for slot in GROUND_SLOTS)),
                            ("full_context", ((shift, slot) for shift in range(6)
                                              for slot in FULL_SLOTS))):
            for shift, slot in cells:
                variant = display(question, shift)
                prompt = make_prompt(question, variant, sources[root]["source"], mode)
                request_id = f"{question['candidate_id']}__{variant['variant_id']}__{mode}__s{slot:02d}"
                if request_id in prompts:
                    raise ValueError("DUPLICATE_REQUEST_ID:" + request_id)
                prompts[request_id] = prompt
                requests.append({"request_id": request_id, "candidate_id": question["candidate_id"],
                                 "root_id": root, "source_sha256": sources[root]["sha256"],
                                 "question_type": "multi_select", "variant": variant,
                                 "mode": mode, "slot": slot, "seed": 2026100800 + len(requests),
                                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    if len(requests) != len(questions) * 23:
        raise ValueError("REQUEST_PLAN_COUNT_MISMATCH")
    return requests, prompts
