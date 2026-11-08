"""One-pass Composition/Fresh QA Writer comparison over frozen R03 inputs.

This module intentionally owns only the two newly-authorized QA Writer paths.
It imports Targeted Evolution as an immutable reference and never calls a
solver, a semantic auditor, Memorax, or any dialogue-expansion component.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from jsonschema import validate

from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from . import (
    ALLOWED_LIFECYCLE, API_CONFIG, CAPACITY_RUN, CLOSURE_RUN, COMPILER_RUN,
    CONTEXT_LIMIT, DEFAULT_OUTPUT, LIFECYCLE_RUN, MAX_OUTPUT_TOKENS,
    QUARANTINE_ROOT, REROUTE_RUN, SUCCESS_ROOTS, TARGETED_CLOSURE_RUN,
    TARGETED_RUN, TEMPERATURE, VERSION,
)


COMPOSITION_TEMPLATE = """# MULTI_QA_COMPOSITION_TEMPLATE_V1

Create at most one new QA from the supplied independent canonical clusters.
This is a composition task, not a rewrite task. Use only the authoritative
cumulative dialogue as factual authority. A QA outcome is allowed only when
the Gold requires at least two distinct evidence relations and cannot be
answered by replaying any one supplied source question. Otherwise return
NO_SAFE_EVOLUTION. R03 is optional: cite it only if it contributes to the
reasoning, and state whether the answer would remain available without it.

For QA, create exactly one complete single_choice, multi_select, or ordering
item with a unique Gold and valid distractors. All evidence refs must be turn
refs from the supplied dialogue. Do not invent facts. Do not provide analysis,
markdown, more than one item, bank actions, or recommendations.
"""

FRESH_TEMPLATE = """# FRESH_EXPLORATION_TEMPLATE_V1

Create at most one independently useful new QA from the authoritative
cumulative dialogue and its R03 evidence bundles. This is not source-QA
evolution: do not reproduce, paraphrase, or target the supplied historical
surface/answer signatures. Use only the dialogue for facts. Prefer a question
whose answer needs R03 when the dialogue supports one, but R03 is optional;
state whether it semantically contributes and whether the answer is available
without R03. If a new, unambiguous QA cannot be formed safely, return
NO_SAFE_EVOLUTION.

For QA, create exactly one complete single_choice, multi_select, or ordering
item with a unique Gold and valid distractors. All evidence refs must be turn
refs from the supplied dialogue. Do not invent facts. Do not provide analysis,
markdown, more than one item, bank actions, or recommendations.
"""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = sha_bytes(data)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def write_json(output: Path, name: str, value: Any) -> str:
    return write_bytes(
        output / name,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def write_jsonl(output: Path, name: str, values: list[Mapping[str, Any]]) -> str:
    return write_bytes(output / name, "".join(canonical(value) + "\n" for value in values).encode("utf-8"))


def write_text(output: Path, name: str, value: str) -> str:
    return write_bytes(output / name, (value if value.endswith("\n") else value + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_source(path: Path, source_hashes: dict[str, str]) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing_input:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing_input_sidecar:{sidecar}")
    observed = sha_file(path)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError(f"input_sha_mismatch:{path}")
    source_hashes[str(path)] = observed
    return observed


def norm(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_jaccard(left: Any, right: Any) -> float:
    a = set(re.findall(r"\w+", norm(left), flags=re.UNICODE))
    b = set(re.findall(r"\w+", norm(right), flags=re.UNICODE))
    return len(a & b) / len(a | b) if a | b else 1.0


def question_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else row
    return {
        "candidate_id": str(row.get("candidate_id") or candidate.get("candidate_id") or ""),
        "root_id": str(row.get("root_id") or candidate.get("root_id") or ""),
        "split": str(row.get("split") or candidate.get("split") or ""),
        "question": str(candidate.get("question") or ""),
        "question_type": str(candidate.get("question_type") or ""),
        "options_or_items": candidate.get("options") or candidate.get("items") or {},
        "gold": candidate.get("gold") or [],
        "evidence_refs": list(candidate.get("evidence_refs") or []),
        "reasoning_groups": list(candidate.get("reasoning_groups_used") or candidate.get("required_reasoning") or []),
        "normalized_stem_signature": row.get("normalized_stem_signature"),
        "surface_signature": row.get("surface_signature"),
    }


def turn_segment(turn_ref: str) -> str:
    if turn_ref.startswith("R03:"):
        return "R03"
    if turn_ref.startswith("R02:"):
        return "R02"
    if turn_ref.startswith("R01:"):
        return "R01"
    if ":P" in turn_ref or ":PARENT" in turn_ref:
        return "PARENT"
    return "UNKNOWN"


def relation_label(refs: list[str]) -> str:
    labels = sorted({turn_segment(str(ref)) for ref in refs})
    return "_AND_".join(labels) if labels else "NO_EVIDENCE"


def render_dialogue(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(
        f"[{turn.get('turn_ref')}] {turn.get('speaker')}: {turn.get('text')}"
        for turn in dialogue.get("turns") or []
    )


def output_schema(pass_name: str) -> dict[str, Any]:
    common_properties: dict[str, Any] = {
        "outcome": {"enum": ["QA", "NO_SAFE_EVOLUTION"]},
        "question_type": {"type": ["string", "null"], "enum": ["single_choice", "multi_select", "ordering", None]},
        "new_question": {"type": ["string", "null"]},
        "options_or_items": {"type": ["object", "null"], "additionalProperties": {"type": "string"}},
        "canonical_gold": {"type": ["array", "null"], "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "new_r03_evidence_refs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "selected_evidence_relations": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "expected_context_dependency": {"type": ["string", "null"]},
        "r03_semantic_contribution_claimed": {"type": ["boolean", "null"]},
        "answerable_without_r03_claim": {"type": ["boolean", "null"]},
        "no_safe_evolution_reason": {"type": "string"},
    }
    if pass_name == "composition":
        common_properties.update({
            "source_cluster_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "composition_relation": {"type": ["string", "null"]},
        })
        qa_required = ["source_cluster_ids", "composition_relation"]
        qa_properties = {
            "source_cluster_ids": {"minItems": 2},
            "composition_relation": {"type": "string", "minLength": 1},
            "selected_evidence_relations": {"minItems": 2},
        }
    else:
        common_properties.update({"independent_new_qa_reason": {"type": ["string", "null"]}, "novelty_basis": {"type": ["string", "null"]}})
        qa_required = ["independent_new_qa_reason", "novelty_basis"]
        qa_properties = {
            "independent_new_qa_reason": {"type": "string", "minLength": 1},
            "novelty_basis": {"type": "string", "minLength": 1},
            "selected_evidence_relations": {"minItems": 1},
        }
    # The reason is conditional: it is required only for NO_SAFE_EVOLUTION.
    # A valid QA outcome may omit it (or return an empty/null value).
    required = [field for field in common_properties if field != "no_safe_evolution_reason"]
    qa_fields = {
        "question_type": {"type": "string", "enum": ["single_choice", "multi_select", "ordering"]},
        "new_question": {"type": "string", "minLength": 1},
        "options_or_items": {"type": "object", "minProperties": 4},
        "canonical_gold": {"type": "array", "minItems": 1},
        "evidence_refs": {"minItems": 1},
        "expected_context_dependency": {"type": "string", "minLength": 1},
        "r03_semantic_contribution_claimed": {"type": "boolean"},
        "answerable_without_r03_claim": {"type": "boolean"},
        "no_safe_evolution_reason": {"type": ["string", "null"], "enum": ["", None]},
        **qa_properties,
    }
    no_safe_fields: dict[str, Any] = {
        "question_type": {"type": "null"}, "new_question": {"type": "null"},
        "options_or_items": {"type": "null"}, "canonical_gold": {"type": "null"},
        "evidence_refs": {"const": []}, "new_r03_evidence_refs": {"const": []},
        "selected_evidence_relations": {"const": []},
        "expected_context_dependency": {"type": "null"},
        "r03_semantic_contribution_claimed": {"type": "null"},
        "answerable_without_r03_claim": {"type": "null"},
        "no_safe_evolution_reason": {"type": "string", "minLength": 1},
    }
    if pass_name == "composition":
        no_safe_fields.update({"source_cluster_ids": {"type": "array", "minItems": 2}, "composition_relation": {"type": "null"}})
    else:
        no_safe_fields.update({"independent_new_qa_reason": {"type": "null"}, "novelty_basis": {"type": "null"}})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"h4.r03.multi_pass.{pass_name}.output.v1",
        "type": "object", "additionalProperties": False, "required": required,
        "properties": common_properties,
        "allOf": [
            {"if": {"properties": {"outcome": {"const": "QA"}}}, "then": {"properties": qa_fields}},
            {"if": {"properties": {"outcome": {"const": "NO_SAFE_EVOLUTION"}}}, "then": {"required": ["no_safe_evolution_reason"], "properties": no_safe_fields}},
        ],
    }


def load_inputs() -> dict[str, Any]:
    hashes: dict[str, str] = {}
    paths = {
        "r03_root_manifest": CLOSURE_RUN / "r03_local_batch_root_manifest.jsonl",
        "r03_closure": CLOSURE_RUN / "final_r03_local_batch_closure_report.json",
        "capacity_scene": CAPACITY_RUN / "r03_scene_evidence_manifest.jsonl",
        "capacity_joint": CAPACITY_RUN / "r03_joint_evidence_bundle_manifest.jsonl",
        "capacity_report": CAPACITY_RUN / "final_r03_novel_question_capacity_preflight.json",
        "composition_pool": COMPILER_RUN / "questioner_composition_parent_pool.jsonl",
        "pair_feasibility": COMPILER_RUN / "questioner_composition_pair_feasibility.jsonl",
        "fresh_signatures": COMPILER_RUN / "fresh_exploration_negative_signature_view.jsonl",
        "compiler_report": COMPILER_RUN / "final_round1_question_bank_feedback_compiler_report.json",
        "cluster_representatives": LIFECYCLE_RUN / "cluster_canonical_representative_manifest.jsonl",
        "composition_eligibility": LIFECYCLE_RUN / "composition_parent_eligibility.jsonl",
        "candidate_lifecycle": LIFECYCLE_RUN / "candidate_lifecycle_view.jsonl",
        "provisional_bank": REROUTE_RUN / "provisional_question_bank_entries.jsonl",
        "targeted_closure": TARGETED_CLOSURE_RUN / "offline_feedback_qa_closure_review.json",
        "targeted_reviews": TARGETED_CLOSURE_RUN / "raw_successor_content_review.jsonl",
        "targeted_call_closure": TARGETED_RUN / "qa_writer_call_closure.json",
        "targeted_report": TARGETED_RUN / "final_r03_feedback_guided_targeted_evolution_pilot_report.json",
    }
    for path in paths.values():
        verify_source(path, hashes)
    roots = {str(row["root_id"]): row for row in read_jsonl(paths["r03_root_manifest"])}
    if set(SUCCESS_ROOTS) - set(roots):
        raise RuntimeError("successful_root_missing_from_closure")
    for root in SUCCESS_ROOTS:
        row = roots[root]
        if row.get("disposition") != "R03_LOCAL_SUCCESS":
            raise RuntimeError(f"root_not_success:{root}:{row.get('disposition')}")
        dialogue_path = Path(str((row.get("dialogue_artifacts") or {}).get("cumulative_dialogue_path") or ""))
        verify_source(dialogue_path, hashes)
    quarantine = roots.get(QUARANTINE_ROOT)
    if not quarantine or quarantine.get("disposition") != "R03_LOCAL_QUARANTINE":
        raise RuntimeError("quarantine_root_disposition_drift")
    bank = {item["candidate_id"]: item for item in (question_payload(row) for row in read_jsonl(paths["provisional_bank"])) if item["candidate_id"]}
    return {
        "paths": paths, "source_hashes": hashes, "roots": roots,
        "scenes": read_jsonl(paths["capacity_scene"]), "joint": read_jsonl(paths["capacity_joint"]),
        "composition_pool": read_jsonl(paths["composition_pool"]),
        "pair_feasibility": read_jsonl(paths["pair_feasibility"]),
        "fresh_signatures": read_jsonl(paths["fresh_signatures"]),
        "canonical_clusters": read_jsonl(paths["cluster_representatives"]),
        "composition_eligibility": read_jsonl(paths["composition_eligibility"]),
        "candidate_lifecycle": read_jsonl(paths["candidate_lifecycle"]),
        "bank": bank,
        "targeted_closure": read_json(paths["targeted_closure"]),
        "targeted_reviews": read_jsonl(paths["targeted_reviews"]),
        "targeted_call_closure": read_json(paths["targeted_call_closure"]),
        "targeted_report": read_json(paths["targeted_report"]),
    }


def lifecycle_rank(state: str) -> int:
    return {"ACTIVE_FRONTIER_BOTH": 3, "ACTIVE_FRONTIER_ONE": 2, "RESOLVED_WITH_CONTEXT": 1}.get(state, 0)


def build_composition_samples(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    bank: Mapping[str, Mapping[str, Any]] = inputs["bank"]
    pool_by_root: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in inputs["composition_pool"]:
        root = str(row.get("root_id") or "")
        candidate_id = str(row.get("canonical_representative_candidate_id") or "")
        if root not in SUCCESS_ROOTS or candidate_id not in bank:
            continue
        if not bool(row.get("canonical_representative")):
            continue
        if str(row.get("lifecycle_state")) not in ALLOWED_LIFECYCLE:
            continue
        if not bool(row.get("r03_feedback_compatible")) or not row.get("evidence_refs"):
            continue
        if bank[candidate_id]["split"] != "train":
            continue
        candidate = dict(row)
        candidate["candidate"] = bank[candidate_id]
        candidate["cluster_id"] = str(row.get("solver_prompt_fingerprint") or "")
        candidate["evidence_relation"] = relation_label([str(x) for x in row.get("evidence_refs") or []])
        if candidate["cluster_id"]:
            pool_by_root[root][candidate["cluster_id"]] = candidate
    pairs_by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in inputs["pair_feasibility"]:
        if str(row.get("root_id") or "") in SUCCESS_ROOTS:
            pairs_by_root[str(row["root_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for root in SUCCESS_ROOTS:
        available = pool_by_root.get(root, {})
        eligible = {item["candidate"]["candidate_id"]: item for item in available.values()}
        ranked = sorted(
            available.values(),
            key=lambda item: (-lifecycle_rank(str(item.get("lifecycle_state"))), -float(item.get("raw_weight") or 0.0), str(item["cluster_id"])),
        )
        rank = {item["candidate"]["candidate_id"]: index for index, item in enumerate(ranked)}
        candidates: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], Mapping[str, Any]]] = []
        for pair in pairs_by_root.get(root, []):
            left_id, right_id = str(pair.get("left_candidate_id") or ""), str(pair.get("right_candidate_id") or "")
            left, right = eligible.get(left_id), eligible.get(right_id)
            if not left or not right or left["cluster_id"] == right["cluster_id"]:
                continue
            evidence_jaccard = pair.get("evidence_jaccard")
            if evidence_jaccard is None or float(evidence_jaccard) != 0.0:
                continue
            score = (
                lifecycle_rank(str(left.get("lifecycle_state"))) + lifecycle_rank(str(right.get("lifecycle_state"))),
                float(left.get("raw_weight") or 0.0) + float(right.get("raw_weight") or 0.0),
                -(rank[left_id] + rank[right_id]),
                "|".join(sorted((left["cluster_id"], right["cluster_id"]))),
            )
            candidates.append((score, left, right, pair))
        if not candidates:
            selected.append({
                "root_id": root, "pass": "MULTI_QA_COMPOSITION", "sampling_status": "EXCLUDED_NO_LEGAL_COMPLEMENTARY_CANONICAL_PAIR",
                "reason": "No two canonical, train-scope, composition-eligible candidates had disjoint evidence.",
                "candidate_pool_cluster_count": len(available), "api_call_planned": False,
                "targeted_seed_dependency": False,
            })
            continue
        _, left, right, pair = max(candidates, key=lambda item: item[0])
        sources = []
        for item in (left, right):
            question = item["candidate"]
            sources.append({
                "cluster_id": item["cluster_id"], "canonical_representative_candidate_id": question["candidate_id"],
                "lifecycle_state": item["lifecycle_state"], "question_type": question["question_type"],
                "source_question": question["question"], "evidence_refs": item["evidence_refs"],
                "evidence_relation": item["evidence_relation"], "reasoning_groups": question["reasoning_groups"],
                "normalized_stem_signature": item.get("normalized_stem_signature"),
            })
        selected.append({
            "root_id": root, "pass": "MULTI_QA_COMPOSITION", "sampling_status": "ELIGIBLE_SELECTED",
            "api_call_planned": True, "targeted_seed_dependency": False,
            "source_cluster_ids": [source["cluster_id"] for source in sources], "source_clusters": sources,
            "selected_source_evidence_relations": [source["evidence_relation"] for source in sources],
            "selection_rule": "canonical_train_scope + lifecycle priority + zero evidence overlap; the output contract then requires two distinct composed evidence relations",
            "pair_feasibility_basis": pair.get("feasibility_basis"),
            "pair_evidence_jaccard": pair.get("evidence_jaccard"),
            "root_96462_independent_legality": ("INDEPENDENTLY_LEGAL_NO_TARGETED_SEED_DEPENDENCY" if root == "cc-train-episode-96462" else "NOT_APPLICABLE"),
        })
    return selected


def build_fresh_samples(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenes_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene in inputs["scenes"]:
        if str(scene.get("root_id") or "") in SUCCESS_ROOTS:
            scenes_by_root[str(scene["root_id"])].append(scene)
    joint_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for joint in inputs["joint"]:
        if str(joint.get("root_id") or "") in SUCCESS_ROOTS:
            joint_by_root[str(joint["root_id"])].append(joint)
    signatures_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signature in inputs["fresh_signatures"]:
        root = str(signature.get("root_id") or "")
        if root in SUCCESS_ROOTS:
            signatures_by_root[root].append({
                "normalized_stem_signature": signature.get("normalized_stem_signature"),
                "prompt_cluster_count": signature.get("prompt_cluster_count"),
                "candidate_count": signature.get("candidate_count"),
            })
    samples: list[dict[str, Any]] = []
    for root in SUCCESS_ROOTS:
        scenes = sorted(scenes_by_root[root], key=lambda item: str(item.get("scene_ref") or ""))
        joint = sorted(joint_by_root[root], key=lambda item: (str(item.get("r03_scene_ref") or ""), str(item.get("prior_segment_id") or "")))
        if not scenes:
            samples.append({"root_id": root, "pass": "FRESH_EXPLORATION", "sampling_status": "EXCLUDED_NO_R03_BUNDLE", "api_call_planned": False})
            continue
        samples.append({
            "root_id": root, "pass": "FRESH_EXPLORATION", "sampling_status": "ELIGIBLE_SELECTED", "api_call_planned": True,
            "selection_rule": "one attempt per authoritative R03-success root with structured R03 bundles and signature-only novelty exclusions",
            "r03_scene_bundles": [{
                "scene_ref": item["scene_ref"], "descriptor": item["descriptor"], "memory_event_id": item.get("memory_event_id"),
                "turn_refs": item["turn_refs"], "state_change_candidate": item.get("state_change_candidate"),
            } for item in scenes],
            "joint_bundle_candidates": [{
                "r03_scene_ref": item["r03_scene_ref"], "prior_segment_id": item["prior_segment_id"],
                "memory_event_id": item.get("memory_event_id"), "bundle_signature": item["bundle_signature"],
                "evidence_ref_count": item["evidence_ref_count"],
            } for item in joint],
            "historical_duplicate_signatures": sorted(signatures_by_root[root], key=lambda item: str(item.get("normalized_stem_signature") or ""))[:24],
            "root_96462_fresh_eligibility": ("INCLUDED_R03_VALID_AND_INDEPENDENTLY_QUALIFIED" if root == "cc-train-episode-96462" else "NOT_APPLICABLE"),
        })
    return samples


def composition_prompt(sample: Mapping[str, Any], dialogue: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    return "\n".join([
        "You are the QA Writer for one bounded MULTI_QA_COMPOSITION attempt.",
        "Return exactly one JSON object matching OUTPUT_SCHEMA. This is the only attempt; do not retry or force a QA.",
        "SOURCE_CLUSTER_IDS in a QA outcome must exactly match the sampled cluster ids.",
        "MULTI_RELATION_REQUIRED: select at least two distinct evidence relations and cite real dialogue refs for the composed answer.",
        "No Solver, training, Memorax, auditor, lifecycle update, dialogue append, or recommendation is authorized.",
        "\nFROZEN_TEMPLATE:\n" + COMPOSITION_TEMPLATE,
        "\nOUTPUT_SCHEMA:\n" + canonical(schema),
        "\nCOMPOSITION_SAMPLE:\n" + canonical(sample),
        "\nAUTHORITATIVE_CUMULATIVE_DIALOGUE:\n" + render_dialogue(dialogue),
    ])


def fresh_prompt(sample: Mapping[str, Any], dialogue: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    fresh_context = {key: value for key, value in sample.items() if key not in {"pass", "sampling_status", "api_call_planned"}}
    return "\n".join([
        "You are the QA Writer for one bounded FRESH_EXPLORATION attempt.",
        "Return exactly one JSON object matching OUTPUT_SCHEMA. This is the only attempt; do not retry or force a QA.",
        "INDEPENDENT_NEW_QA_REQUIRED: the QA must not be a surface or answer-target replay of supplied signature-only historical coverage.",
        "No historical question text, options, Gold labels, or question-bank contents are provided. Do not infer them.",
        "No Solver, training, Memorax, auditor, lifecycle update, dialogue append, or recommendation is authorized.",
        "\nFROZEN_TEMPLATE:\n" + FRESH_TEMPLATE,
        "\nOUTPUT_SCHEMA:\n" + canonical(schema),
        "\nFRESH_SAMPLE:\n" + canonical(fresh_context),
        "\nAUTHORITATIVE_CUMULATIVE_DIALOGUE:\n" + render_dialogue(dialogue),
    ])


def single_http_json(config: Any, prompt: str) -> tuple[str, dict[str, Any]]:
    """Exactly one transport attempt. Any failure is terminal for this seed."""
    payload = {
        "model": config.model, "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE, "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"}, "enable_thinking": False,
    }
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(180, int(config.timeout_seconds))) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body["choices"][0]
    return str(choice["message"]["content"]), {
        "model": config.model, "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"), "transport_attempt_count": 1, "transport_retry_count": 0,
    }


class BackendResponse:
    def __init__(self, result: tuple[str, dict[str, Any]]):
        self.raw_text, self.metadata = result
        self.model_identifier = str(self.metadata.get("model") or "")


def parse_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return None, f"json_parse_error:{error}"
    return (value, None) if isinstance(value, dict) else (None, "top_level_not_object")


def inferred_type(options: Any, gold: Any) -> str:
    if not isinstance(options, dict) or not isinstance(gold, list):
        return "UNKNOWN"
    keys = set(options)
    if keys == {"A", "B", "C", "D"} and len(gold) == 1:
        return "single_choice"
    if keys == {"A", "B", "C", "D", "E"} and gold and set(gold) <= keys and len(gold) == len(set(gold)):
        return "multi_select"
    if keys == {"I1", "I2", "I3", "I4"} and set(gold) == keys and len(gold) == 4:
        return "ordering"
    return "UNKNOWN"


def existing_questions_for_root(bank: Mapping[str, Mapping[str, Any]], root: str) -> list[Mapping[str, Any]]:
    return [question for question in bank.values() if question["root_id"] == root and question["split"] == "train"]


def duplicate_relation(output: Mapping[str, Any], existing: list[Mapping[str, Any]]) -> dict[str, Any]:
    question = norm(output.get("new_question"))
    options = output.get("options_or_items") or {}
    gold = list(output.get("canonical_gold") or [])
    evidence = {str(item) for item in output.get("evidence_refs") or []}
    lexical = False
    for old in existing:
        old_question = norm(old["question"])
        exact_surface = question == old_question and canonical(options) == canonical(old["options_or_items"]) and gold == list(old["gold"])
        if exact_surface:
            kind = "exact_surface_same_provenance" if evidence == set(old["evidence_refs"]) else "exact_surface_changed_provenance"
            return {"duplicate_class": kind, "exact_duplicate": True, "lexical_paraphrase_candidate": False, "matched_candidate_id": old["candidate_id"]}
        sequence = difflib.SequenceMatcher(None, question, old_question).ratio()
        option_overlap = token_jaccard(" ".join(map(str, options.values())), " ".join(map(str, old["options_or_items"].values())))
        lexical = lexical or bool(sequence >= 0.90 or (sequence >= 0.75 and option_overlap >= 0.50))
    return {"duplicate_class": "lexical_paraphrase_candidate" if lexical else "distinct_surface", "exact_duplicate": False, "lexical_paraphrase_candidate": lexical, "matched_candidate_id": None}


def structural_closure(pass_name: str, sample: Mapping[str, Any], result: Mapping[str, Any], dialogue: Mapping[str, Any], bank: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    closure: dict[str, Any] = {
        "schema_valid": False, "content_valid": False, "content_valid_definition": "offline structural completeness, real evidence refs, nonduplicate surface, and pass-specific contract; Gold/distractor semantics remain unadjudicated without a semantic auditor",
        "gold_unique_supported": "UNKNOWN_NO_SEMANTIC_AUDITOR", "distractors_valid": "UNKNOWN_NO_SEMANTIC_AUDITOR",
        "semantic_duplicate_status": "UNKNOWN_NO_SEMANTIC_AUDITOR", "r03_semantic_contribution": "UNKNOWN_NO_SEMANTIC_AUDITOR",
    }
    schema = output_schema(pass_name)
    try:
        validate(result, schema)
        closure["schema_valid"] = True
    except Exception as error:
        closure.update({"structural_status": "SCHEMA_INVALID", "schema_error": f"{type(error).__name__}:{error}".split("\n", 1)[0]})
        return closure
    if result.get("outcome") == "NO_SAFE_EVOLUTION":
        closure.update({"structural_status": "NO_SAFE_EVOLUTION", "no_safe_evolution": True, "content_valid": False})
        return closure
    closure["no_safe_evolution"] = False
    refs = [str(value) for value in result.get("evidence_refs") or []]
    r03_refs = [str(value) for value in result.get("new_r03_evidence_refs") or []]
    known_refs = {str(turn.get("turn_ref") or "") for turn in dialogue.get("turns") or []}
    errors: list[str] = []
    inferred = inferred_type(result.get("options_or_items"), result.get("canonical_gold"))
    if inferred == "UNKNOWN" or inferred != result.get("question_type"):
        errors.append("question_type_or_gold_structure_invalid")
    if any(ref not in known_refs for ref in refs):
        errors.append("evidence_ref_unknown")
    if any(ref not in refs or not ref.startswith("R03:") for ref in r03_refs):
        errors.append("new_r03_evidence_ref_invalid")
    if pass_name == "composition":
        source_ids = list(sample.get("source_cluster_ids") or [])
        if list(result.get("source_cluster_ids") or []) != source_ids:
            errors.append("source_cluster_ids_not_exact_sample")
        if len(result.get("selected_evidence_relations") or []) < 2:
            errors.append("MULTI_RELATION_REQUIRED")
    else:
        if not str(result.get("independent_new_qa_reason") or "").strip():
            errors.append("INDEPENDENT_NEW_QA_REQUIRED")
    duplicate = duplicate_relation(result, existing_questions_for_root(bank, str(sample["root_id"])))
    if duplicate["exact_duplicate"]:
        errors.append("exact_duplicate")
    if duplicate["lexical_paraphrase_candidate"]:
        errors.append("lexical_paraphrase_candidate")
    evidence_segments = sorted({turn_segment(ref) for ref in refs})
    r03_touch = any(ref.startswith("R03:") for ref in refs)
    r03_claimed = result.get("r03_semantic_contribution_claimed")
    closure.update({
        "structural_status": "PASS" if not errors else "FAIL", "structural_errors": errors,
        "content_valid": not errors, "question_type_inferred": inferred,
        "evidence_refs_real": not any(ref not in known_refs for ref in refs),
        "evidence_ref_count": len(refs), "evidence_segments": evidence_segments,
        "evidence_relation_count": len(result.get("selected_evidence_relations") or []),
        "r03_evidence_touch": r03_touch, "r03_evidence_ref_count": sum(ref.startswith("R03:") for ref in refs),
        "r03_semantic_contribution_claimed": r03_claimed,
        "answerable_without_r03_claim": result.get("answerable_without_r03_claim"),
        "r03_semantic_contribution": ("CLAIMED_UNVERIFIED" if r03_touch and r03_claimed else "NOT_CLAIMED_OR_NOT_TOUCHED"),
        "cross_stage_context": len([segment for segment in evidence_segments if segment != "UNKNOWN"]) >= 2,
        "multi_event_context": len({ref.split(":T", 1)[0] for ref in refs}) >= 2,
        "context_dependency_claimed": bool(str(result.get("expected_context_dependency") or "").strip()),
        **duplicate,
    })
    return closure


def make_attempts(
    output: Path, pass_name: str, samples: list[dict[str, Any]], dialogues: Mapping[str, Mapping[str, Any]],
    config: Any, ledger: RequestLedger, total_budget: int, bank: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    schema = output_schema(pass_name)
    attempts: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.get("api_call_planned"):
            attempts.append({"root_id": sample["root_id"], "pass": pass_name.upper(), "status": "NOT_CALLED_INELIGIBLE", "api_attempts": 0, "retry_count": 0, "sampling_status": sample["sampling_status"]})
            continue
        root = str(sample["root_id"])
        prompt = composition_prompt(sample, dialogues[root], schema) if pass_name == "composition" else fresh_prompt(sample, dialogues[root], schema)
        request_id = f"{root}:{pass_name}:attempt-001"
        prompt_sha = sha_bytes(prompt.encode("utf-8"))
        write_text(output, f"raw/prompts/{pass_name}/{root}.txt", prompt)
        record: dict[str, Any] = {
            "root_id": root, "pass": pass_name.upper(), "request_id": request_id, "attempt": 1,
            "prompt_sha256": prompt_sha, "template_sha256": sha_bytes((COMPOSITION_TEMPLATE if pass_name == "composition" else FRESH_TEMPLATE).encode("utf-8")),
            "sampling_status": sample["sampling_status"], "status": "FAILED", "api_attempts": 0, "retry_count": 0,
        }
        try:
            response = ledger.invoke(
                request_id=request_id, intent=f"r03_{pass_name}_qa_writer", attempt=1, prompt_sha256=prompt_sha,
                call=lambda prompt=prompt: BackendResponse(single_http_json(config, prompt)), budget_class="qa_writer",
                budget_limit=total_budget, total_budget_limit=total_budget,
                request_metadata={"root_id": root, "pass": pass_name, "one_attempt": True, "retry_policy": "none", "template_sha256": record["template_sha256"]},
                backend_identity={"transport": "single_http_openai_compatible", "model": config.model, "temperature": TEMPERATURE, "enable_thinking": False},
            )
            raw = response.raw_text
            write_text(output, f"raw/responses/{pass_name}/{root}.txt", raw)
            record.update({"status": "RESPONSE_RECEIVED", "api_attempts": 1, "response_metadata": response.metadata})
            parsed, error = parse_object(raw)
            if error:
                record.update({"status": "PARSE_FAILED", "parse_error": error})
            else:
                record["parsed_output"] = parsed
                closure = structural_closure(pass_name, sample, parsed, dialogues[root], bank)
                record["closure"] = closure
                record["status"] = "NO_SAFE_EVOLUTION" if closure.get("no_safe_evolution") else ("SCHEMA_VALID" if closure.get("schema_valid") else "SCHEMA_INVALID")
        except Exception as error:
            record.update({"status": "API_FAILED", "api_attempts": 1, "error": f"{type(error).__name__}:{error}"})
        attempts.append(record)
    return attempts


def pass_summary(pass_name: str, attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
    called = [item for item in attempts if item.get("api_attempts") == 1]
    parsed = [item for item in called if isinstance(item.get("parsed_output"), dict)]
    closures = [item.get("closure") or {} for item in parsed]
    qa = [item for item in closures if not item.get("no_safe_evolution") and item.get("schema_valid")]
    return {
        "pass": pass_name, "sampling_rows": len(attempts), "attempt_count": len(called),
        "no_safe_evolution_count": sum(bool(item.get("no_safe_evolution")) for item in closures),
        "raw_generated_count": len(parsed), "schema_valid_count": sum(bool(item.get("schema_valid")) for item in closures),
        "content_valid_count": sum(bool(item.get("content_valid")) for item in qa),
        "genuine_failure_count": sum(item.get("status") in {"API_FAILED", "PARSE_FAILED", "SCHEMA_INVALID"} or (item.get("closure") or {}).get("structural_status") == "FAIL" for item in attempts),
        "exact_duplicate_count": sum(bool(item.get("exact_duplicate")) for item in qa),
        "lexical_paraphrase_candidate_count": sum(bool(item.get("lexical_paraphrase_candidate")) for item in qa),
        "r03_evidence_touch_count": sum(bool(item.get("r03_evidence_touch")) for item in qa),
        "r03_semantic_contribution_claimed_count": sum(item.get("r03_semantic_contribution") == "CLAIMED_UNVERIFIED" for item in qa),
        "valid_gold_structural_count": sum(item.get("gold_unique_supported") == "UNKNOWN_NO_SEMANTIC_AUDITOR" and item.get("content_valid") for item in qa),
        "valid_distractor_structural_count": sum(item.get("distractors_valid") == "UNKNOWN_NO_SEMANTIC_AUDITOR" and item.get("content_valid") for item in qa),
        "semantic_correctness_status": "UNKNOWN_NO_SEMANTIC_AUDITOR",
        "multi_relation_required_pass_count": sum(item.get("structural_status") == "PASS" and pass_name == "MULTI_QA_COMPOSITION" for item in qa),
        "independent_new_qa_required_pass_count": sum(item.get("structural_status") == "PASS" and pass_name == "FRESH_EXPLORATION" for item in qa),
        "cross_stage_context_count": sum(bool(item.get("cross_stage_context")) for item in qa),
        "multi_event_context_count": sum(bool(item.get("multi_event_context")) for item in qa),
        "context_dependency_claimed_count": sum(bool(item.get("context_dependency_claimed")) for item in qa),
    }


def targeted_reference(inputs: Mapping[str, Any]) -> dict[str, Any]:
    closure = inputs["targeted_closure"]
    call_closure = inputs["targeted_call_closure"]
    if closure.get("attempt_count") != 6 or closure.get("legitimate_no_safe_evolution_count") != 4 or closure.get("genuine_qa_failure_count") != 2:
        raise RuntimeError("targeted_baseline_drift")
    if call_closure.get("qa_writer_calls") != 6 or call_closure.get("retry_count") != 0:
        raise RuntimeError("targeted_call_baseline_drift")
    return {
        "reference_only": True, "source_run": str(TARGETED_RUN), "source_closure_run": str(TARGETED_CLOSURE_RUN),
        "attempt_count": 6, "no_safe_evolution_count": 4, "raw_generated_count": 2,
        "schema_valid_count": 0, "content_valid_count": 0, "genuine_failure_count": 2,
        "retry_count": 0, "solver_calls": 0, "training_calls": 0, "memorax_calls": 0, "semantic_auditor_calls": 0,
        "interpretation": "Imported as an existing result. No Targeted request was scheduled or made by this comparison pilot.",
        "closure_status": closure.get("status"), "targeted_report_status": inputs["targeted_report"].get("status"),
    }


def output_sidecar_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name.endswith(".sha256") or path.name == "output_sidecar_sha_manifest.json":
            continue
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.is_file():
            continue
        observed = sha_file(path)
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        files.append({"path": str(path.relative_to(output)), "sha256": observed, "sidecar_matches": observed == expected})
    return {"status": "PASS" if all(item["sidecar_matches"] for item in files) else "FAIL", "file_count": len(files), "files": files, "excludes_self": True}


def run(output: Path, *, preflight_only: bool = False) -> int:
    if output.exists():
        raise RuntimeError(f"immutable_output_exists:{output}")
    output.mkdir(parents=True, exist_ok=False)
    inputs = load_inputs()
    composition_samples = build_composition_samples(inputs)
    fresh_samples = build_fresh_samples(inputs)
    if any(sample.get("api_call_planned") and sample["root_id"] == QUARANTINE_ROOT for sample in composition_samples + fresh_samples):
        raise RuntimeError("quarantine_root_call_planned")
    composition_planned = sum(bool(sample.get("api_call_planned")) for sample in composition_samples)
    fresh_planned = sum(bool(sample.get("api_call_planned")) for sample in fresh_samples)
    if fresh_planned != len(SUCCESS_ROOTS):
        raise RuntimeError(f"fresh_coverage_not_all_success_roots:{fresh_planned}")
    if not any(sample.get("root_id") == "cc-train-episode-96462" and sample.get("api_call_planned") for sample in fresh_samples):
        raise RuntimeError("96462_fresh_not_included")
    if composition_planned > len(SUCCESS_ROOTS) or fresh_planned > len(SUCCESS_ROOTS):
        raise RuntimeError("per_root_attempt_budget_exceeded")
    composition_schema, fresh_schema = output_schema("composition"), output_schema("fresh")
    comp_template_sha = sha_bytes(COMPOSITION_TEMPLATE.encode("utf-8"))
    fresh_template_sha = sha_bytes(FRESH_TEMPLATE.encode("utf-8"))
    config = load_challenger_config(API_CONFIG)
    contract = {
        "version": VERSION, "status": "PREFLIGHT_PASS", "backend_mode": "live" if not preflight_only else "preflight_only_no_api",
        "allowed_new_qa_writer_passes": ["MULTI_QA_COMPOSITION", "FRESH_EXPLORATION"], "targeted_mode": "REFERENCE_ONLY_NO_RERUN",
        "success_roots": list(SUCCESS_ROOTS), "quarantine_root": QUARANTINE_ROOT,
        "composition_calls_planned": composition_planned, "fresh_calls_planned": fresh_planned,
        "max_one_attempt_per_success_root_per_pass": True, "retry_policy": "none", "reserve_seed_switching": False,
        "model": config.model, "temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS, "context_limit": CONTEXT_LIMIT,
        "composition_template_sha256": comp_template_sha, "fresh_template_sha256": fresh_template_sha,
        "solver_calls": 0, "training_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0, "memorax_status_calls": 0,
        "memorax_mutation_calls": 0, "semantic_auditor_calls": 0, "dialogue_mutation_calls": 0,
    }
    write_json(output, "pilot_contract.json", contract)
    write_text(output, "multi_qa_composition_template_v1.md", COMPOSITION_TEMPLATE)
    write_text(output, "fresh_exploration_template_v1.md", FRESH_TEMPLATE)
    write_json(output, "multi_qa_composition_output_schema_v1.json", composition_schema)
    write_json(output, "fresh_exploration_output_schema_v1.json", fresh_schema)
    write_jsonl(output, "composition_sampling_manifest.jsonl", composition_samples)
    write_jsonl(output, "fresh_sampling_manifest.jsonl", fresh_samples)
    write_json(output, "targeted_existing_result_reference.json", targeted_reference(inputs))
    dialogue_manifest: list[dict[str, Any]] = []
    dialogues: dict[str, Any] = {}
    for root in SUCCESS_ROOTS:
        dialogue_path = Path(str((inputs["roots"][root].get("dialogue_artifacts") or {}).get("cumulative_dialogue_path") or ""))
        dialogue = read_json(dialogue_path)
        dialogues[root] = dialogue
        dialogue_manifest.append({"root_id": root, "path": str(dialogue_path), "sha256": sha_file(dialogue_path), "turn_count": len(dialogue.get("turns") or []), "r03_turn_count": sum(str(turn.get("turn_ref") or "").startswith("R03:") for turn in dialogue.get("turns") or [])})
    write_jsonl(output, "authoritative_dialogue_manifest.jsonl", dialogue_manifest)
    write_json(output, "source_sha_manifest.json", {"status": "PASS", "files": [{"path": path, "sha256_before": digest} for path, digest in sorted(inputs["source_hashes"].items())]})
    if preflight_only:
        write_jsonl(output, "composition_attempt_results.jsonl", [])
        write_jsonl(output, "fresh_attempt_results.jsonl", [])
        write_json(output, "composition_call_closure.json", {"status": "PREFLIGHT_ONLY", "qa_writer_calls": 0, "retry_count": 0})
        write_json(output, "fresh_call_closure.json", {"status": "PREFLIGHT_ONLY", "qa_writer_calls": 0, "retry_count": 0})
        write_json(output, "composition_offline_closure.json", {"status": "PREFLIGHT_ONLY"})
        write_json(output, "fresh_offline_closure.json", {"status": "PREFLIGHT_ONLY"})
        report = {"version": VERSION, "status": "R03_MULTI_PASS_QA_GENERATION_COMPARISON_PREFLIGHT_PASS", "next_status": "READY_FOR_MULTI_PASS_QA_WRITER_CALLS", "contract": contract}
        write_json(output, "multi_pass_comparison_report.json", report)
        write_text(output, "multi_pass_comparison_report.md", "# R03 Multi-Pass QA Comparison Pilot v1\n\nStatus: `R03_MULTI_PASS_QA_GENERATION_COMPARISON_PREFLIGHT_PASS`\n")
        write_json(output, "output_sidecar_sha_manifest.json", output_sidecar_manifest(output))
        return 0
    total_budget = composition_planned + fresh_planned
    ledger = RequestLedger(output, sha_bytes(canonical(contract).encode("utf-8")), "live")
    composition_attempts = make_attempts(output, "composition", composition_samples, dialogues, config, ledger, total_budget, inputs["bank"])
    fresh_attempts = make_attempts(output, "fresh", fresh_samples, dialogues, config, ledger, total_budget, inputs["bank"])
    write_jsonl(output, "composition_attempt_results.jsonl", composition_attempts)
    write_jsonl(output, "fresh_attempt_results.jsonl", fresh_attempts)
    comp_summary, fresh_summary = pass_summary("MULTI_QA_COMPOSITION", composition_attempts), pass_summary("FRESH_EXPLORATION", fresh_attempts)
    for pass_name, samples, attempts, summary in (("composition", composition_samples, composition_attempts, comp_summary), ("fresh", fresh_samples, fresh_attempts, fresh_summary)):
        planned = sum(bool(sample.get("api_call_planned")) for sample in samples)
        calls = sum(int(item.get("api_attempts") or 0) for item in attempts)
        closure = {
            "status": "PASS" if calls == planned and all(int(item.get("retry_count") or 0) == 0 for item in attempts) else "COMPLETE_WITH_GAPS",
            "pass": pass_name.upper(), "qa_writer_calls": calls, "planned_calls": planned,
            "attempts_per_root": {str(item["root_id"]): int(item.get("api_attempts") or 0) for item in attempts}, "retry_count": 0,
            "solver_calls": 0, "training_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0, "memorax_status_calls": 0,
            "memorax_mutation_calls": 0, "semantic_auditor_calls": 0, "dialogue_mutation_calls": 0,
            "ledger_summary": ledger.summary(),
        }
        write_json(output, f"{pass_name}_call_closure.json", closure)
        write_json(output, f"{pass_name}_offline_closure.json", {"status": "COMPLETE", **summary, "semantic_limits": "Gold uniqueness, distractor validity, semantic duplication, and causal R03 contribution are UNKNOWN without a semantic auditor; content_valid is explicitly structural."})
    targeted = targeted_reference(inputs)
    comparison_rows = [
        {"pass": "TARGETED_EVOLUTION_REFERENCE", "attempts": targeted["attempt_count"], "no_safe": targeted["no_safe_evolution_count"], "raw_generated": targeted["raw_generated_count"], "schema_valid": targeted["schema_valid_count"], "content_valid": targeted["content_valid_count"], "genuine_failures": targeted["genuine_failure_count"], "exact_duplicates": "UNKNOWN_NOT_RECALCULATED", "lexical_candidates": "UNKNOWN_NOT_RECALCULATED", "r03_touch": "REFERENCE_CLOSURE_ONLY", "r03_semantic_contribution": "REFERENCE_CLOSURE_ONLY", "valid_gold": 0, "valid_distractors": 0},
        {"pass": "MULTI_QA_COMPOSITION", "attempts": comp_summary["attempt_count"], "no_safe": comp_summary["no_safe_evolution_count"], "raw_generated": comp_summary["raw_generated_count"], "schema_valid": comp_summary["schema_valid_count"], "content_valid": comp_summary["content_valid_count"], "genuine_failures": comp_summary["genuine_failure_count"], "exact_duplicates": comp_summary["exact_duplicate_count"], "lexical_candidates": comp_summary["lexical_paraphrase_candidate_count"], "r03_touch": comp_summary["r03_evidence_touch_count"], "r03_semantic_contribution": comp_summary["r03_semantic_contribution_claimed_count"], "valid_gold": comp_summary["valid_gold_structural_count"], "valid_distractors": comp_summary["valid_distractor_structural_count"]},
        {"pass": "FRESH_EXPLORATION", "attempts": fresh_summary["attempt_count"], "no_safe": fresh_summary["no_safe_evolution_count"], "raw_generated": fresh_summary["raw_generated_count"], "schema_valid": fresh_summary["schema_valid_count"], "content_valid": fresh_summary["content_valid_count"], "genuine_failures": fresh_summary["genuine_failure_count"], "exact_duplicates": fresh_summary["exact_duplicate_count"], "lexical_candidates": fresh_summary["lexical_paraphrase_candidate_count"], "r03_touch": fresh_summary["r03_evidence_touch_count"], "r03_semantic_contribution": fresh_summary["r03_semantic_contribution_claimed_count"], "valid_gold": fresh_summary["valid_gold_structural_count"], "valid_distractors": fresh_summary["valid_distractor_structural_count"]},
    ]
    gaps = comp_summary["genuine_failure_count"] + fresh_summary["genuine_failure_count"]
    report = {
        "version": VERSION, "status": "R03_MULTI_PASS_QA_GENERATION_COMPARISON_PILOT_COMPLETE" if not gaps else "R03_MULTI_PASS_QA_GENERATION_COMPARISON_PILOT_COMPLETE_WITH_GAPS",
        "next_status": "READY_FOR_MULTI_PASS_OFFLINE_REVIEW", "comparison_table": comparison_rows,
        "targeted_reference_only": True, "new_qa_writer_calls": total_budget, "retry_count": 0,
        "Q1_comparison": "Descriptive counts are reported in comparison_table. They do not identify whether any difference is caused by template or data because Targeted has six different eligible seeds and the two new passes use different pass-specific inputs; no causal conclusion is made.",
        "Q2_r03_contribution": "R03 touch and model-claimed contribution are counted, but semantic contribution is UNKNOWN without a semantic auditor or an independent offline adjudication. A non-R03-answerability claim is not treated as proof.",
        "Q3_minimum_next_evidence": "Before any larger generation, independently adjudicate each structurally valid new QA for Gold uniqueness, distractor validity, source-family duplication, and necessity of R03; then use matched seed sets if a template-versus-data comparison is required. This pilot does not advance automatically.",
        "limits": ["No Solver", "No training", "No Memorax read/write/status/mutation", "No semantic auditor", "No dialogue mutation", "No Targeted rerun", "No retry or reserve seed switch", "No Dev/Test"],
    }
    write_json(output, "multi_pass_comparison_report.json", report)
    lines = ["# R03 Multi-Pass QA Generation Comparison Pilot v1", "", f"Status: `{report['status']}`", f"Next: `{report['next_status']}`", "", "| Pass | Attempts | NO_SAFE | Schema valid | Structural content valid | Genuine failures | R03 touch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {row['pass']} | {row['attempts']} | {row['no_safe']} | {row['schema_valid']} | {row['content_valid']} | {row['genuine_failures']} | {row['r03_touch']} |" for row in comparison_rows)
    lines.extend(["", "Q1: Descriptive only; unmatched pass inputs prevent a template-versus-data causal conclusion.", "", "Q2: R03 semantic contribution remains UNKNOWN without independent adjudication.", "", "Q3: Independently review structurally valid QAs before any larger generation. No next stage is started.", ""])
    write_text(output, "multi_pass_comparison_report.md", "\n".join(lines))
    source_after = [{"path": path, "sha256_before": digest, "sha256_after": sha_file(Path(path)), "unchanged": digest == sha_file(Path(path))} for path, digest in sorted(inputs["source_hashes"].items())]
    write_json(output, "source_sha_manifest.json", {"status": "PASS" if all(item["unchanged"] for item in source_after) else "FAIL", "files": source_after})
    write_json(output, "output_sidecar_sha_manifest.json", output_sidecar_manifest(output))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    return run(args.output.resolve(), preflight_only=args.preflight_only)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
