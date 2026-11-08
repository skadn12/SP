"""Input validation, gold-blind random display rebinding, and prompt contract."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

LABELS = {"multi_select": tuple("ABCDEF"), "ordering": tuple(f"I{i}" for i in range(1, 7))}


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _content_ids(question: dict[str, Any], field: str) -> tuple[list[str], dict[str, str]]:
    values = question.get(field)
    if not isinstance(values, dict):
        raise ValueError("OPTIONS_OR_ITEMS_REQUIRED")
    labels = LABELS[question["question_type"]]
    if (tuple(values) != labels
            or any(not isinstance(value, str) or not value.strip() for value in values.values())
            or len(set(str(v).strip().casefold() for v in values.values())) != len(labels)):
        raise ValueError("SIX_UNIQUE_OPTIONS_OR_ITEMS_REQUIRED")
    # IDs are derived from option content, so a display reorder never changes
    # the identity used by Gold, evidence, or response records.
    content = ["content_" + hashlib.sha256(str(values[label]).strip().encode("utf-8")).hexdigest()[:16]
               for label in labels]
    if len(set(content)) != len(content):
        raise ValueError("CONTENT_ID_COLLISION")
    return content, {label: content[index] for index, label in enumerate(labels)}


def validate_pool(path: Path, *, require_ordering_review: bool = False) -> tuple[dict, list[dict], dict]:
    pool = read(path)
    questions = pool.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("NONEMPTY_QUESTION_POOL_REQUIRED")
    # Accept the reviewed ordering pool's historical envelope without changing
    # it on disk: its question and provenance are nested per candidate.
    normalized: list[dict[str, Any]] = []
    for original in questions:
        if isinstance(original.get("question"), dict):
            inner = dict(original["question"])
            per_candidate = original.get("provenance") or {}
            inner.setdefault("candidate_id", original.get("candidate_id"))
            inner.setdefault("root_id", original.get("root_id"))
            if per_candidate.get("source_path"):
                inner.setdefault("source_path", per_candidate["source_path"])
            if per_candidate.get("source_sha256"):
                inner.setdefault("source_sha256", per_candidate["source_sha256"])
            if per_candidate.get("requested_model"):
                inner.setdefault("generator_model", per_candidate["requested_model"])
            if per_candidate.get("qwen_review"):
                inner.setdefault("qwen_review", per_candidate["qwen_review"])
            if original.get("qwen_review"):
                inner.setdefault("qwen_review", original["qwen_review"])
            normalized.append(inner)
        else:
            normalized.append(original)
    questions = normalized
    provenance = pool.get("provenance") or {}
    if not provenance:
        for original in pool["questions"]:
            candidate_provenance = original.get("provenance") or {}
            if candidate_provenance:
                provenance = candidate_provenance
                break
    pool["provenance"] = provenance
    if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":
        raise ValueError("UNSUPPORTED_GENERATOR_MODEL")
    seen: set[str] = set(); sources: dict[str, dict] = {}
    for question in questions:
        qid, root, qtype = question.get("candidate_id"), question.get("root_id"), question.get("question_type")
        if not isinstance(qid, str) or qid in seen or not isinstance(root, str) or qtype not in LABELS:
            raise ValueError("QUESTION_ID_ROOT_OR_TYPE_INVALID")
        seen.add(qid)
        field = "options" if qtype == "multi_select" else "items"
        labels = LABELS[qtype]
        if tuple((question.get(field) or {})) != labels:
            raise ValueError("QUESTION_LABEL_CONTRACT_INVALID:" + qid)
        if not isinstance(question.get("gold"), list) or not 2 <= len(question["gold"]) <= 6:
            raise ValueError("GOLD_CARDINALITY_INVALID:" + qid)
        if len(set(question["gold"])) != len(question["gold"]) or not set(question["gold"]) <= set(labels):
            raise ValueError("GOLD_LABELS_INVALID:" + qid)
        if qtype == "ordering" and len(question["gold"]) != 6:
            raise ValueError("ORDERING_REQUIRES_SIX_GOLD_ITEMS")
        evidence = question.get("option_evidence" if qtype == "multi_select" else "item_evidence")
        if not isinstance(evidence, dict) or set(evidence) != set(labels):
            raise ValueError("COMPLETE_EVIDENCE_REQUIRED:" + qid)
        source_path = Path(question.get("source_path", ""))
        expected = question.get("source_sha256")
        if not source_path.is_absolute() or not source_path.is_file() or sha(source_path) != expected:
            raise ValueError("SOURCE_PATH_OR_SHA_INVALID:" + qid)
        source = read(source_path)
        if source.get("root_id") != root or not isinstance(source.get("turns"), list) or not source["turns"]:
            raise ValueError("SOURCE_ROOT_OR_TURNS_INVALID:" + qid)
        turns = {row.get("turn_ref"): row for row in source["turns"]}
        if len(turns) != len(source["turns"]):
            raise ValueError("SOURCE_TURN_REFS_NOT_UNIQUE:" + qid)
        for label, citation in evidence.items():
            if (not isinstance(citation, dict) or citation.get("turn_ref") not in turns
                    or not isinstance(citation.get("quote"), str)
                    or not citation["quote"] or citation["quote"] not in turns[citation["turn_ref"]].get("text", "")
                    or (qtype == "multi_select" and citation.get("relation") not in {"supports", "contradicts"})
                    or (qtype == "multi_select" and
                        ((citation.get("relation") == "supports") != (label in question["gold"])))
                    or not isinstance(citation.get("reason"), str) or not citation["reason"].strip()):
                raise ValueError("EVIDENCE_SOURCE_MISMATCH:" + qid + ":" + label)
        if qtype == "ordering" and require_ordering_review:
            review = question.get("qwen_review") or {}
            if review.get("status") != "PASS" or review.get("returned_model") != provenance.get("requested_model"):
                raise ValueError("ORDERING_QWEN_REVIEW_REQUIRED:" + qid)
        sources.setdefault(root, {"path": str(source_path), "sha256": expected, "source": source})
        if sources[root]["sha256"] != expected:
            raise ValueError("ROOT_SOURCE_VERSION_CONFLICT:" + root)
    return pool, questions, sources


def _seed(seed: int, qid: str, variant: int) -> int:
    raw = f"{seed}|{qid}|{variant}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def make_variants(question: dict[str, Any], *, seed: int, count: int) -> list[dict[str, Any]]:
    qtype = question["question_type"]
    field = "options" if qtype == "multi_select" else "items"
    labels = list(LABELS[qtype])
    content, label_to_content = _content_ids(question, field)
    content_to_text = {label_to_content[label]: question[field][label] for label in labels}
    base_rng = random.Random(_seed(seed, question["candidate_id"], -1))
    base_rng.shuffle(content)
    variants: list[dict[str, Any]] = []
    used: set[tuple[str, ...]] = set()
    for index in range(count):
        rng = random.Random(_seed(seed, question["candidate_id"], index))
        order = content[:]
        rng.shuffle(order)
        key = tuple(order)
        if key in used:
            # A duplicate check is independent of Gold; this is not a truth-based resample.
            for offset in range(1, len(content) + 1):
                order = content[offset:] + content[:offset]
                if tuple(order) not in used:
                    break
        used.add(tuple(order))
        display_labels = list(labels)
        displayed = {display_labels[pos]: content_to_text[content_id] for pos, content_id in enumerate(order)}
        mapping = {display_labels[pos]: content_id for pos, content_id in enumerate(order)}
        variants.append({"variant_id": f"perm_{index:02d}",
                         "canonical_order_content_ids": list(content),
                         "display_order_content_ids": order,
                         field: displayed, "display_to_content": mapping,
                         "content_to_text": content_to_text})
    return variants


def prompt(question: dict[str, Any], variant: dict[str, Any], source: dict, *, model_prompt: str | None = None) -> str:
    """Build public Solver prompt. It deliberately excludes Gold, mappings, and evidence."""
    field = "options" if question["question_type"] == "multi_select" else "items"
    heading = "Options" if field == "options" else "Items"
    lines = [f"Dialogue:\n" + "\n".join(f"[{row.get('turn_ref','')}] {row.get('speaker','')}: {row.get('text','')}" for row in source["turns"]),
             f"Question:\n{question['question']}", heading + ":\n" + "\n".join(f"{key}. {text}" for key, text in variant[field].items())]
    lines.append("Output only one <box>... </box> answer. For multi-select use comma-separated letters; for ordering use comma-separated item IDs.")
    return model_prompt or "\n\n".join(lines)
