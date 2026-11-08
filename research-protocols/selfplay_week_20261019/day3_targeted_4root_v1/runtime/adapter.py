"""Gold-blind local extension: six-option MC and 3–6-item total permutations.

The native prompt, finite box protocol, stochastic parameters and scoring stay
unchanged. The local public validator and shuffle allow ordering arity three.
"""
from __future__ import annotations
import hashlib
import importlib.util
from itertools import permutations
import math
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
LEGACY = Path(__file__).resolve().parents[2] / "day2/solver/adapter.py"
spec = importlib.util.spec_from_file_location("crossroot_legacy_adapter", LEGACY)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
PUBLIC_KEYS = old.PUBLIC_KEYS
REPO, WORKTREE, NATIVE = old.REPO, old.WORKTREE, old.NATIVE
native, native_scoring, content_ids = old.native, old.native_scoring, old.content_ids
direct_prompt = old.direct_prompt


def labels(qtype, n):
    if qtype == "multi_select" and n == 6:
        return tuple("ABCDEF")
    if qtype == "ordering" and n in (3, 4, 5, 6):
        return tuple(f"I{x}" for x in range(1, n + 1))
    raise ValueError("REQUIRE_MC_SIX_OR_ORDERING_THREE_TO_SIX")


def public_question(q):
    if not isinstance(q, dict) or set(q) - PUBLIC_KEYS:
        raise ValueError("PUBLIC_WHITELIST_VIOLATION")
    for key in ("candidate_id", "root_id", "question_type", "question", "source_path", "source_sha256"):
        if not isinstance(q.get(key), str) or not q[key].strip():
            raise ValueError("INVALID_PUBLIC_FIELD:" + key)
    field = "options" if q["question_type"] == "multi_select" else "items"
    if not isinstance(q.get(field), dict):
        raise ValueError("PUBLIC_ITEMS_REQUIRED")
    keys = labels(q["question_type"], len(q[field]))
    if tuple(q[field]) != keys:
        raise ValueError("CANONICAL_INPUT_LABELS_INVALID")
    values = list(q[field].values())
    if any(not isinstance(v, str) or not v.strip() for v in values) or len(set(" ".join(v.casefold().split()) for v in values)) != len(keys):
        raise ValueError("ITEMS_NOT_NONEMPTY_UNIQUE")
    if ("items" if field == "options" else "options") in q:
        raise ValueError("CONFLICTING_TASK_FIELDS")
    return q


def _seed(seed, qid, variant):
    return int.from_bytes(hashlib.sha256(f"{seed}|{qid}|{variant}".encode()).digest()[:8], "big")


def variants(q, seed, count):
    # Identical legacy v15 seed/rebind algorithm, only labels() is generalized.
    public_question(q)
    field = "options" if q["question_type"] == "multi_select" else "items"
    keys = list(labels(q["question_type"], len(q[field])))
    mapping = content_ids(q)
    content = list(mapping.values())
    texts = {mapping[k]: q[field][k] for k in keys}
    if not 1 <= count <= math.factorial(len(content)):
        raise ValueError("INVALID_VARIANT_COUNT")
    random.Random(_seed(seed, q["candidate_id"], -1)).shuffle(content)
    used, result = set(), []
    for i in range(count):
        rng = random.Random(_seed(seed, q["candidate_id"], i))
        while True:
            order = content[:]
            rng.shuffle(order)
            if tuple(order) not in used:
                break
        used.add(tuple(order))
        result.append({"variant_id": f"perm_{i:02d}", field: {keys[p]: texts[c] for p, c in enumerate(order)},
            "display_to_content": dict(zip(keys, order)), "canonical_order_content_ids": content[:],
            "display_order_content_ids": order, "content_to_text": texts})
    return result


def mapping_count(q):
    field = "options" if q["question_type"] == "multi_select" else "items"
    return 6 if field == "options" else min(12, math.factorial(len(q[field])))


def request_count(q):
    return mapping_count(q) * (3 if q["question_type"] == "multi_select" else 1)


def choices(qtype, display):
    """Keep native vocabulary byte-for-byte, extending ordering only to n=3."""
    if qtype == "ordering" and len(display) == 3:
        keys = tuple(display)
        if set(keys) != set(labels(qtype, 3)):
            raise ValueError("INVALID_ORDERING_THREE_KEYS")
        return ["<box>" + ",".join(order) + "</box>" for order in permutations(keys)]
    return old.choices(qtype, display)


def sampling_params(row, expected_api):
    # Local choice lookup is required: old.sampling_params binds old.choices,
    # whose original native implementation deliberately rejects n=3.
    from vllm import SamplingParams, sampling_params as sp
    _, solver = native()
    api, _ = solver.detect_structured_output_api()
    if api != expected_api:
        raise RuntimeError("STRUCTURED_API_CHANGED")
    if row["branch"] != "direct":
        raise ValueError("DIRECT_ONLY_RUNTIME")
    field = "options" if row["question_type"] == "multi_select" else "items"
    constraint = {"choice": choices(row["question_type"], row["display"][field])}
    kwargs = {"temperature": .8, "top_p": .9, "max_tokens": 32, "seed": row["seed"],
        "stop": ["</box>"], "include_stop_str_in_output": True}
    if api.startswith("structured_outputs="):
        kwargs["structured_outputs"] = sp.StructuredOutputsParams(**constraint)
    else:
        kwargs["guided_decoding"] = sp.GuidedDecodingParams(**constraint)
    return SamplingParams(**kwargs)
