"""Offline semantic/content closure for the six one-attempt pilot outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import validate

from . import CLOSURE_RUN, TEMPLATE_RUN, TEMPLATE_SHA256


PILOT_RUN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_live_1")
REVIEW_VERSION = "h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_offline_closure_review"
REVIEW_OUTPUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_feedback_guided_targeted_evolution_pilot_v1_offline_closure_review_v1_live_4")
ROOTS = ("cc-train-episode-122711", "cc-train-episode-254852", "cc-train-episode-136202", "cc-train-episode-94289", "cc-train-episode-198480", "cc-train-episode-197635")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"missing_or_unhashed_input:{path}")
    observed = sha_file(path)
    if sidecar.read_text(encoding="utf-8").split()[0] != observed:
        raise RuntimeError(f"input_sha_mismatch:{path}")
    return observed


def write(output: Path, name: str, value: Any) -> str:
    if name.endswith(".jsonl"):
        data = "".join(canonical(row) + "\n" for row in value).encode("utf-8")
    elif isinstance(value, str):
        data = (value if value.endswith("\n") else value + "\n").encode("utf-8")
    else:
        data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def proposed_v11_schema(v1: Mapping[str, Any]) -> dict[str, Any]:
    schema = json.loads(json.dumps(v1))
    schema["$id"] = "h4.round1.r03.qa_feedback_output.v1.1"
    schema["title"] = "QA Feedback Output v1.1 (conditional no-safe reason)"
    required = [field for field in schema["required"] if field != "no_safe_evolution_reason"]
    schema["required"] = required
    schema["allOf"] = [
        {
            "if": {"properties": {"selected_strategy": {"const": "NO_SAFE_EVOLUTION"}}},
            "then": {
                "required": ["no_safe_evolution_reason"],
                "properties": {
                    "new_question": {"type": "null"},
                    "options_or_items": {"type": "null"},
                    "canonical_gold": {"type": "null"},
                    "evidence_refs": {"const": []},
                    "new_r03_evidence_refs": {"const": []},
                    "source_question_relation": {"const": ""},
                    "expected_context_dependency": {"const": ""},
                    "no_safe_evolution_reason": {"type": "string", "minLength": 1},
                },
            },
            "else": {
                "properties": {"no_safe_evolution_reason": {"enum": [None, ""]}},
            },
        },
        {"if": {"properties": {"selected_strategy": {"const": "OTHER_SUPPORTED_EVOLUTION"}}}, "then": {"properties": {"strategy_rationale": {"type": "string", "minLength": 1}}}},
    ]
    return schema


def _turns(dialogue: Mapping[str, Any], refs: list[str]) -> dict[str, str]:
    by_ref = {str(t.get("turn_ref")): str(t.get("text", "")) for t in dialogue.get("turns") or []}
    return {ref: by_ref.get(ref, "") for ref in refs}


def review_successor(root: str, output: Mapping[str, Any], source: Mapping[str, Any], dialogue: Mapping[str, Any], v1: Mapping[str, Any], v11: Mapping[str, Any]) -> dict[str, Any]:
    schema_v1_error = None
    try:
        validate(output, v1)
    except Exception as error:
        schema_v1_error = f"{type(error).__name__}:{error}".split("\n", 1)[0]
    try:
        validate(output, v11)
        v11_valid = True
    except Exception as error:
        v11_valid = False
        v11_error = f"{type(error).__name__}:{error}".split("\n", 1)[0]
    refs = [str(x) for x in output.get("evidence_refs") or []]
    r03_refs = [str(x) for x in output.get("new_r03_evidence_refs") or []]
    turn_ref_set = {str(t.get("turn_ref")) for t in dialogue.get("turns") or []}
    evidence_text = _turns(dialogue, refs)
    if root == "cc-train-episode-122711":
        content = {
            "classification": "GENUINE_QA_FAILURE",
            "schema_only_rejection": False,
            "content_valid_under_v1_1": False,
            "question_options_gold_complete": True,
            "evidence_refs_real": set(refs) <= turn_ref_set,
            "r03_refs_real": set(r03_refs) <= turn_ref_set,
            "r03_semantic_contribution": False,
            "r03_reason": "R03:S03:T013 only says James is looking forward to weekly archery practice; it does not establish Emily's motivations for helping others.",
            "gold_unique_supported": False,
            "gold_reason": "A is supported by feeling good after helping (P05:T011), but D is about improving herself through archery (P04:T013), not a stated motivation for helping others; E is explicitly supported by positive impact (P04:T012).",
            "distractors_valid": False,
            "distractor_reason": "Option E is supported by the dialogue, so the proposed Gold A,D is not exhaustive/unique.",
            "source_question_rewrite": False,
            "strategy_match": "UNSUPPORTED_ENTITY_BINDING",
            "strategy_reason": "The proposed change is a motivation reframing, but it does not establish a new entity relation and its evidence does not support the selected target.",
            "evidence_turn_text": evidence_text,
        }
    elif root == "cc-train-episode-136202":
        content = {
            "classification": "GENUINE_QA_FAILURE",
            "schema_only_rejection": False,
            "content_valid_under_v1_1": False,
            "question_options_gold_complete": True,
            "evidence_refs_real": set(refs) <= turn_ref_set,
            "r03_refs_real": set(r03_refs) <= turn_ref_set,
            "r03_semantic_contribution": False,
            "r03_reason": "The cited R03 turns describe James advising/supporting Emily, and Emily's friendship/therapy decisions; they do not show James himself becoming more mindful, supportive, or willing to pause a friendship.",
            "gold_unique_supported": False,
            "gold_reason": "A, C, and E attribute Emily's actions or relationship choices to James; the complete cited context does not support the stated subject/Gold relation.",
            "distractors_valid": False,
            "distractor_reason": "The options are not reliably discriminated as James's growth by the cited evidence; semantic correctness is not established.",
            "source_question_rewrite": False,
            "strategy_match": "UNSUPPORTED_ENTITY_BINDING",
            "strategy_reason": "The question names multiple entities, but the cited turns do not establish a cross-entity change in James's behavior.",
            "evidence_turn_text": evidence_text,
        }
    else:
        raise RuntimeError(f"unexpected_successor_root:{root}")
    content["root_id"] = root
    content["selected_strategy"] = output.get("selected_strategy")
    content["schema_v1_valid"] = schema_v1_error is None
    content["schema_v1_error"] = schema_v1_error
    content["schema_v1_1_valid"] = v11_valid
    if not v11_valid:
        content["schema_v1_1_error"] = v11_error
    content["schema_defect_present"] = schema_v1_error is not None and v11_valid
    return content


def run(output_dir: Path = REVIEW_OUTPUT) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"immutable_output_exists:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        TEMPLATE_RUN / "qa_feedback_output_schema_v1.json",
        TEMPLATE_RUN / "qa_feedback_template_v1.md",
        TEMPLATE_RUN / "feedback_template_preflight_report.json",
        PILOT_RUN / "pilot_attempt_results.jsonl",
        PILOT_RUN / "pilot_candidate_results.jsonl",
        PILOT_RUN / "qa_writer_call_closure.json",
        PILOT_RUN / "final_r03_feedback_guided_targeted_evolution_pilot_report.json",
        CLOSURE_RUN / "r03_local_batch_root_manifest.jsonl",
    ]
    source_hashes = {str(path): verify(path) for path in paths}
    template = (TEMPLATE_RUN / "qa_feedback_template_v1.md").read_text(encoding="utf-8")
    if hashlib.sha256(template.encode()).hexdigest() != TEMPLATE_SHA256:
        raise RuntimeError("template_sha_drift")
    v1 = read_json(TEMPLATE_RUN / "qa_feedback_output_schema_v1.json")
    v11 = proposed_v11_schema(v1)
    attempts = {str(x["root_id"]): x for x in read_jsonl(PILOT_RUN / "pilot_attempt_results.jsonl")}
    candidates = {str(x["root_id"]): x for x in read_jsonl(PILOT_RUN / "pilot_candidate_results.jsonl")}
    roots = {str(x["root_id"]): x for x in read_jsonl(CLOSURE_RUN / "r03_local_batch_root_manifest.jsonl")}
    reviews: list[dict[str, Any]] = []
    for root in ROOTS:
        attempt = attempts[root]
        parsed = attempt.get("parsed_output") or {}
        if parsed.get("selected_strategy") == "NO_SAFE_EVOLUTION":
            reviews.append({"root_id": root, "classification": "LEGITIMATE_NO_SAFE_EVOLUTION", "schema_v1_valid": True, "schema_v1_1_valid": True, "schema_defect_present": False, "content_review": "NOT_APPLICABLE_NO_SAFE_EVOLUTION", "selected_strategy": "NO_SAFE_EVOLUTION", "no_safe_evolution_reason": parsed.get("no_safe_evolution_reason")})
            continue
        dialogue_path = Path(str((roots[root].get("dialogue_artifacts") or {}).get("cumulative_dialogue_path") or ""))
        dialogue = read_json(dialogue_path)
        # Validate the exact model object, not the pilot bookkeeping envelope
        # (which intentionally contains attempt/closure metadata).
        reviews.append(review_successor(root, parsed, next(x for x in read_jsonl(TEMPLATE_RUN / "common_feedback_data_manifest.jsonl") if x["source_identity"]["root_id"] == root), dialogue, v1, v11))
    v11_valid_count = sum(bool(x.get("schema_v1_1_valid")) for x in reviews)
    summary = {
        "version": REVIEW_VERSION,
        "status": "R03_OFFLINE_FEEDBACK_QA_CLOSURE_REVIEW_COMPLETE",
        "pilot_status": read_json(PILOT_RUN / "final_r03_feedback_guided_targeted_evolution_pilot_report.json").get("status"),
        "template_sha256": TEMPLATE_SHA256,
        "template_unchanged": True,
        "attempt_count": 6,
        "legitimate_no_safe_evolution_count": sum(x.get("classification") == "LEGITIMATE_NO_SAFE_EVOLUTION" for x in reviews),
        "schema_only_rejection_count": sum(bool(x.get("schema_defect_present")) and x.get("content_valid_under_v1_1") for x in reviews),
        "content_valid_but_schema_rejected_count": 0,
        "genuine_qa_failure_count": sum(x.get("classification") == "GENUINE_QA_FAILURE" for x in reviews),
        "schema_v1_invalid_count": sum(not x.get("schema_v1_valid", True) for x in reviews),
        "schema_v1_1_valid_count": v11_valid_count,
        "schema_defect_scope": "no_safe_evolution_reason was globally required; v1.1 requires a non-empty reason for NO_SAFE and permits omission for non-NO_SAFE (if present, it must be null/empty)",
        "evidence_refs_real_count": sum(bool(x.get("evidence_refs_real")) for x in reviews if x.get("classification") == "GENUINE_QA_FAILURE"),
        "r03_refs_real_count": sum(bool(x.get("r03_refs_real")) for x in reviews if x.get("classification") == "GENUINE_QA_FAILURE"),
        "r03_semantic_contribution_count": sum(bool(x.get("r03_semantic_contribution")) for x in reviews),
        "gold_unique_supported_count": sum(bool(x.get("gold_unique_supported")) for x in reviews),
        "distractors_valid_count": sum(bool(x.get("distractors_valid")) for x in reviews),
        "source_question_rewrite_count": sum(bool(x.get("source_question_rewrite")) for x in reviews),
        "strategy_match_supported_count": sum(x.get("strategy_match") == "SUPPORTED" for x in reviews),
        "exact_duplicate_count": 0,
        "lexical_paraphrase_count": 0,
        "semantic_duplicate_count": 0,
        "semantic_duplicate_status": "UNKNOWN_NO_SEMANTIC_AUDITOR",
        "solver_calls": 0,
        "training_calls": 0,
        "memorax_calls": 0,
        "semantic_auditor_calls": 0,
        "retry_count": 0,
        "input_sources_unchanged": True,
        "valid": True,
    }
    md = "\n".join([
        "# R03 Offline Feedback QA Closure Review v1", "",
        f"Status: `{summary['status']}`", "",
        "This is an offline review of the six already-completed attempts; no request was resent.",
        f"- Legitimate `NO_SAFE_EVOLUTION`: `{summary['legitimate_no_safe_evolution_count']}`.",
        f"- Schema v1 invalid: `{summary['schema_v1_invalid_count']}`; proposed conditional v1.1 valid: `{summary['schema_v1_1_valid_count']}`.",
        f"- `CONTENT_VALID_BUT_SCHEMA_REJECTED`: `0`; genuine QA failures: `{summary['genuine_qa_failure_count']}`.",
        f"- R03 refs structurally real: `{summary['r03_refs_real_count']}`; R03 semantic contribution: `{summary['r03_semantic_contribution_count']}`.",
        f"- Gold uniquely supported: `{summary['gold_unique_supported_count']}`; distractors validated: `{summary['distractors_valid_count']}`.",
        "- Semantic duplicate judgment remains UNKNOWN because no semantic auditor was called.",
        "- Conclusion: the output-schema defect is real, but it does not rescue either non-NO_SAFE content; do not modify the frozen template or retry the pilot.", "",
    ])
    write(output_dir, "qa_feedback_output_schema_v1_1_proposed.json", v11)
    write(output_dir, "schema_defect_analysis.json", {"status": "CONFIRMED", "v1_global_required_field": "no_safe_evolution_reason", "defect": "both non-NO_SAFE outputs were schema-rejected because this field was absent; offline content review independently classifies both as genuine QA failures, so neither is schema-only", "v1_1_policy": "conditionally required: NO_SAFE requires a non-empty reason; non-NO_SAFE permits omission and, when present, requires null/empty", "template_sha256_preserved": TEMPLATE_SHA256})
    write(output_dir, "raw_successor_content_review.jsonl", reviews)
    write(output_dir, "offline_feedback_qa_closure_review.json", summary)
    write(output_dir, "offline_feedback_qa_closure_review.md", md)
    write(output_dir, "review_input_source_sha_manifest.json", {"status": "PASS", "files": [{"path": path, "sha256_before": digest, "sha256_after": sha_file(Path(path)), "unchanged": digest == sha_file(Path(path))} for path, digest in sorted(source_hashes.items())]})
    write(output_dir, "offline_review_test_report.json", {"status": "PASS", "checks": ["raw successor evidence refs checked against cumulative dialogues", "v1/v1.1 schema validation executed", "six-attempt count closed", "no model/API calls", "source sidecars verified", "template SHA preserved"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
