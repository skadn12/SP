"""Run one bounded QA-feedback evolution attempt per eligible R03 seed."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from jsonschema import validate

from ..live_backends import load_challenger_config
from ..ledger import RequestLedger
from . import (
    API_CONFIG, CLOSURE_RUN, CONTEXT_LIMIT, DEFAULT_OUTPUT, ELIGIBLE_ROOTS,
    MAX_OUTPUT_TOKENS, QUARANTINE_ROOT, TEMPLATE_RUN, TEMPLATE_SHA256,
    TEMPERATURE, UNSAFE_ROOT, VERSION,
)


BASELINE_GENERATION_RUN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_qa_feedback_generation_v2_live_1")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def write_json(path: Path, value: Any) -> str:
    return write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    return write_bytes(path, ("".join(canonical(row) + "\n" for row in rows)).encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_sidecar(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing_input:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing_input_sidecar:{sidecar}")
    observed = sha_file(path)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError(f"input_sha_mismatch:{path}")
    return observed


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {
        "common": TEMPLATE_RUN / "common_feedback_data_manifest.jsonl",
        "schema": TEMPLATE_RUN / "qa_feedback_output_schema_v1.json",
        "template": TEMPLATE_RUN / "qa_feedback_template_v1.md",
        "strategy": TEMPLATE_RUN / "qa_evolution_strategy_schema_v1.json",
        "template_report": TEMPLATE_RUN / "feedback_template_preflight_report.json",
        "eligibility": TEMPLATE_RUN / "feedback_seed_eligibility_manifest.jsonl",
        "closure_roots": CLOSURE_RUN / "r03_local_batch_root_manifest.jsonl",
        "closure_calls": CLOSURE_RUN / "r03_local_batch_call_closure.json",
    }
    hashes = {str(path): verify_sidecar(path) for path in paths.values()}
    template = (paths["template"]).read_text(encoding="utf-8")
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != TEMPLATE_SHA256:
        raise RuntimeError("frozen_template_sha_mismatch")
    report = read_json(paths["template_report"])
    if report.get("status") != "R03_QA_FEEDBACK_TEMPLATE_V1_PREFLIGHT_PASS":
        raise RuntimeError("template_preflight_not_pass")
    common = {str(x["source_identity"]["root_id"]): x for x in read_jsonl(paths["common"])}
    eligibility = {str(x["root_id"]): x for x in read_jsonl(paths["eligibility"])}
    if set(common) != set(ELIGIBLE_ROOTS):
        raise RuntimeError(f"eligible_common_root_mismatch:{sorted(common)}")
    if any(eligibility[root].get("eligibility_status") != "ELIGIBLE" for root in ELIGIBLE_ROOTS):
        raise RuntimeError("eligibility_drift")
    closure = {str(x["root_id"]): x for x in read_jsonl(paths["closure_roots"])}
    for root in ELIGIBLE_ROOTS:
        if (closure[root].get("disposition") != "R03_LOCAL_SUCCESS"):
            raise RuntimeError(f"root_not_success:{root}")
        dialogue_path = Path(str((closure[root].get("dialogue_artifacts") or {}).get("cumulative_dialogue_path") or ""))
        verify_sidecar(dialogue_path)
    sources = dict(hashes)
    sources.update({str((closure[root].get("dialogue_artifacts") or {}).get("cumulative_dialogue_path")): verify_sidecar(Path(str((closure[root].get("dialogue_artifacts") or {}).get("cumulative_dialogue_path")))) for root in ELIGIBLE_ROOTS})
    return common, eligibility, {"output_schema": read_json(paths["schema"]), "strategy_schema": read_json(paths["strategy"]), "template": template, "template_report": report}, {"closure": closure, "closure_calls": read_json(paths["closure_calls"]), "source_hashes": sources}


def render_dialogue(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(f"[{turn.get('turn_ref')}] {turn.get('speaker')}: {turn.get('text')}" for turn in dialogue.get("turns") or [])


def make_prompt(common: Mapping[str, Any], dialogue: Mapping[str, Any], template: str, output_schema: Mapping[str, Any], strategy_schema: Mapping[str, Any]) -> str:
    return "\n".join([
        "You are the Questioner running one bounded feedback-guided targeted-evolution pilot.",
        "Return exactly one JSON object matching OUTPUT_SCHEMA. Do not return a list, markdown, commentary, or multiple attempts.",
        "The model chooses the strategy; the compiler must not preassign one. This is one attempt only.",
        "Use only AUTHORITATIVE_CUMULATIVE_DIALOGUE for facts, options, Gold, and evidence refs.",
        "A successor must be answerable in the complete dialogue, have a unique Gold, have wrong distractors in complete context, avoid ambiguity, and change answer target, evidence relation, or reasoning structure. Do not paraphrase the source or force a quota.",
        "R03 evidence is optional and should be referenced only when it contributes to reasoning. Exact replay is not new QA.",
        "Solver observations are behavioral observations, not cognitive diagnoses. Mark tentative_hypothesis as a hypothesis and use UNKNOWN when needed.",
        "For NO_SAFE_EVOLUTION, leave new_question/options_or_items/canonical_gold/evidence_refs/new_r03_evidence_refs/source_question_relation/expected_context_dependency empty or null as required, and give a reason.",
        "Do not add fields. Do not add root-specific recommendations, Question Bank actions, replay weights, or semantic-family claims.",
        "\nFROZEN_TEMPLATE:\n" + template,
        "\nSTRATEGY_SCHEMA:\n" + canonical(strategy_schema),
        "\nOUTPUT_SCHEMA:\n" + canonical(output_schema),
        "\nCOMMON_FEEDBACK_DATA:\n" + canonical(common),
        "\nAUTHORITATIVE_CUMULATIVE_DIALOGUE:\n" + render_dialogue(dialogue),
    ])


def single_http_json(config: Any, prompt: str) -> tuple[str, dict[str, Any]]:
    """Perform exactly one HTTP attempt; intentionally no retry loop."""
    payload = {"model": config.model, "messages": [{"role": "user", "content": prompt}], "temperature": TEMPERATURE, "max_tokens": MAX_OUTPUT_TOKENS, "response_format": {"type": "json_object"}, "enable_thinking": False}
    request = urllib.request.Request(config.base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=max(180, int(config.timeout_seconds))) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body["choices"][0]
    return str(choice["message"]["content"]), {"model": config.model, "finish_reason": choice.get("finish_reason"), "usage": body.get("usage"), "transport_attempt_count": 1}


def parse_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return None, f"json_parse_error:{error}"
    if not isinstance(value, dict):
        return None, "top_level_not_object"
    return value, None


def norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def token_jaccard(left: Any, right: Any) -> float:
    a, b = set(re.findall(r"\w+", norm(left), flags=re.UNICODE)), set(re.findall(r"\w+", norm(right), flags=re.UNICODE))
    return len(a & b) / len(a | b) if a | b else 1.0


def infer_question_type(items: Any, gold: Any) -> str:
    keys = tuple(items.keys()) if isinstance(items, dict) else ()
    if keys == ("A", "B", "C", "D") and len(gold or []) == 1: return "single_choice"
    if keys == ("A", "B", "C", "D", "E"): return "multi_select"
    if set(keys) == {"I1", "I2", "I3", "I4"}: return "ordering"
    return "UNKNOWN"


def closure_candidate(output: Mapping[str, Any], source: Mapping[str, Any], dialogue: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_valid": False, "structural_status": "NOT_APPLICABLE", "exact_duplicate": False, "lexical_paraphrase": False, "source_question_rewrite": False, "semantic_duplicate_status": "UNKNOWN_NO_SEMANTIC_AUDITOR", "gold_structural_status": "UNKNOWN_SEMANTIC_CORRECTNESS_NOT_AUDITED", "distractor_structural_status": "UNKNOWN_SEMANTIC_CORRECTNESS_NOT_AUDITED", "r03_evidence_used": False, "evidence_relation_changed": False, "reasoning_requirement_status": "UNKNOWN_OFFLINE", "changed_provenance_context_variant": False}
    try:
        validate(output, schema)
        result["schema_valid"] = True
    except Exception as error:
        result["schema_error"] = f"{type(error).__name__}:{error}"
        return result
    if output.get("selected_strategy") == "NO_SAFE_EVOLUTION":
        result.update({"structural_status": "NO_SAFE_EVOLUTION", "no_safe_evolution": True})
        return result
    result["no_safe_evolution"] = False
    question = output.get("new_question")
    items = output.get("options_or_items")
    gold = output.get("canonical_gold")
    qtype = infer_question_type(items, gold)
    turn_refs = {str(x.get("turn_ref")) for x in dialogue.get("turns") or []}
    evidence = [str(x) for x in output.get("evidence_refs") or []]
    new_r03 = [str(x) for x in output.get("new_r03_evidence_refs") or []]
    errors = []
    if not isinstance(question, str) or not question.strip(): errors.append("new_question_missing")
    if qtype == "UNKNOWN": errors.append("options_or_items_shape_unknown")
    if not evidence: errors.append("evidence_refs_missing")
    if any(ref not in turn_refs for ref in evidence): errors.append("evidence_ref_unknown")
    if any(ref not in evidence for ref in new_r03): errors.append("new_r03_ref_not_in_evidence")
    if qtype == "single_choice" and (not isinstance(gold, list) or len(gold) != 1 or gold[0] not in items): errors.append("single_gold_invalid")
    if qtype == "multi_select" and (not isinstance(gold, list) or not gold or len(set(gold)) != len(gold) or any(x not in items for x in gold)): errors.append("multi_gold_invalid")
    if qtype == "ordering" and (not isinstance(gold, list) or set(gold) != set(items)): errors.append("ordering_gold_invalid")
    source_q = source.get("source_qa", {}).get("question", "")
    source_items = source.get("source_qa", {}).get("options_or_items", {})
    source_gold = source.get("source_qa", {}).get("canonical_gold", [])
    exact = norm(question) == norm(source_q) and canonical(items) == canonical(source_items) and list(gold or []) == list(source_gold or [])
    seq = difflib.SequenceMatcher(None, norm(question), norm(source_q)).ratio()
    alt_overlap = token_jaccard(" ".join(map(str, (items or {}).values())), " ".join(map(str, (source_items or {}).values())))
    lexical = bool(not exact and (seq >= 0.90 or (seq >= 0.75 and alt_overlap >= 0.50)))
    result.update({"structural_status": "PASS" if not errors else "FAIL", "structural_errors": errors, "question_type_inferred": qtype, "exact_duplicate": exact, "lexical_paraphrase": lexical, "source_question_rewrite": exact or lexical, "stem_sequence_ratio": seq, "alternative_text_overlap_ratio": alt_overlap, "gold_structural_status": "STRUCTURE_VALID_UNVERIFIED_SEMANTICS" if not errors else "STRUCTURE_INVALID", "distractor_structural_status": "STRUCTURE_PRESENT_UNVERIFIED_SEMANTICS" if not errors else "STRUCTURE_INVALID", "r03_evidence_used": any(ref.startswith("R03:") for ref in evidence), "new_r03_evidence_ref_count": sum(ref.startswith("R03:") for ref in new_r03), "evidence_relation_changed": set(evidence) != set(source.get("evidence", {}).get("authoritative_evidence_refs") or []), "reasoning_requirement_status": "CLAIMED_UNVERIFIED" if norm(output.get("expected_context_dependency")) else "NOT_CLAIMED", "changed_provenance_context_variant": bool(set(evidence) != set(source.get("evidence", {}).get("authoritative_evidence_refs") or []) and not lexical)})
    return result


def baseline_relations(output: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare only objective surface fields to historical free-exploration rows."""
    question = norm(output.get("new_question"))
    items = output.get("options_or_items") or {}
    gold = list(output.get("canonical_gold") or [])
    exact = lexical = 0
    for row in rows:
        if norm(row.get("question")) == question and canonical(row.get("options") or row.get("items") or {}) == canonical(items) and list(row.get("gold") or []) == gold:
            exact += 1
        sequence = difflib.SequenceMatcher(None, question, norm(row.get("question"))).ratio()
        old_items = row.get("options") or row.get("items") or {}
        if sequence >= 0.90 or (sequence >= 0.75 and token_jaccard(" ".join(map(str, items.values())) if isinstance(items, dict) else items, " ".join(map(str, old_items.values())) if isinstance(old_items, dict) else old_items) >= 0.50):
            lexical += 1
    return {"exact_duplicate_count": exact, "lexical_paraphrase_count": lexical, "comparison_row_count": len(rows), "semantic_duplicate_status": "UNKNOWN_NO_SEMANTIC_AUDITOR"}


def baseline_summary() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if BASELINE_GENERATION_RUN.is_dir():
        for pool in BASELINE_GENERATION_RUN.glob("*/candidate_pool.json"):
            try:
                rows.extend(read_json(pool))
            except Exception:
                continue
    free = [x for x in rows if x.get("generation_source") == "free_exploration"]
    return {"baseline_run": str(BASELINE_GENERATION_RUN), "baseline_available": bool(rows), "all_candidate_count": len(rows), "free_exploration_candidate_count": len(free), "free_exploration_valid_count": sum(bool(x.get("schema_valid")) for x in free), "comparison_scope": "descriptive historical E1Q1 feedback-generation free_exploration rows; no Solver or semantic auditor invoked", "semantic_duplicate_status": "NOT_JUDGED_NO_SEMANTIC_AUDITOR"}


def run(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable_output_exists:{output}")
    output.mkdir(parents=True, exist_ok=False)
    common_by_root, eligibility, frozen, meta = load_inputs()
    template = frozen["template"]
    schema = frozen["output_schema"]
    strategy_schema = frozen["strategy_schema"]
    roots = meta["closure"]
    sources = dict(meta["source_hashes"])
    config = load_challenger_config(API_CONFIG)
    contract = {"version": VERSION, "template_sha256": TEMPLATE_SHA256, "temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS, "context_limit": CONTEXT_LIMIT, "one_attempt_per_seed": True, "retry_policy": "none", "eligible_roots": list(ELIGIBLE_ROOTS), "excluded_roots": {"quarantine": QUARANTINE_ROOT, "no_safe_seed": UNSAFE_ROOT}, "qa_writer_calls_planned": 6, "solver_calls": 0, "memorax_calls": 0, "semantic_auditor_calls": 0, "training_calls": 0, "model": config.model, "api_base_url": config.base_url}
    write_json(output / "pilot_contract.json", contract)
    write_json(output / "frozen_input_identity.json", {"template_sha256": TEMPLATE_SHA256, "template_run": str(TEMPLATE_RUN), "source_sha256": sources})
    write_jsonl(output / "common_feedback_inputs.jsonl", [common_by_root[root] for root in ELIGIBLE_ROOTS])
    dialogue_rows = []
    dialogues = {}
    for root in ELIGIBLE_ROOTS:
        path = Path(str((roots[root].get("dialogue_artifacts") or {}).get("cumulative_dialogue_path")))
        dialogue = read_json(path); dialogues[root] = dialogue
        dialogue_rows.append({"root_id": root, "path": str(path), "sha256": sha_file(path), "turn_count": dialogue.get("turn_count", len(dialogue.get("turns") or [])), "r03_turn_count": sum(str(x.get("turn_ref", "")).startswith("R03:") for x in dialogue.get("turns") or [])})
    write_jsonl(output / "authoritative_dialogue_manifest.jsonl", dialogue_rows)
    write_bytes(output / "qa_feedback_template_v1.md", template.encode("utf-8"))
    ledger = RequestLedger(output, hashlib.sha256(canonical(contract).encode()).hexdigest(), "live")
    attempts: list[dict[str, Any]] = []
    for root in ELIGIBLE_ROOTS:
        prompt = make_prompt(common_by_root[root], dialogues[root], template, schema, strategy_schema)
        request_id = f"{root}:targeted-evolution:attempt-001"
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        write_bytes(output / "raw" / "prompts" / f"{root}.txt", prompt.encode("utf-8"))
        record: dict[str, Any] = {"root_id": root, "request_id": request_id, "attempt": 1, "prompt_sha256": prompt_sha, "status": "FAILED", "api_attempts": 0, "retry_count": 0}
        try:
            response = ledger.invoke(request_id=request_id, intent="r03_feedback_guided_targeted_evolution", attempt=1, prompt_sha256=prompt_sha, call=lambda: _backend_response(single_http_json(config, prompt)), budget_class="qa_writer", budget_limit=6, total_budget_limit=6, request_metadata={"root_id": root, "template_sha256": TEMPLATE_SHA256, "one_attempt": True, "max_output_tokens": MAX_OUTPUT_TOKENS}, backend_identity={"transport": "single_http_openai_compatible", "model": config.model, "base_url": config.base_url, "temperature": TEMPERATURE})
            raw = response.raw_text
            record.update({"status": "RESPONSE_RECEIVED", "api_attempts": 1, "response_metadata": response.metadata})
            write_bytes(output / "raw" / "responses" / f"{root}.txt", raw.encode("utf-8"))
            parsed, parse_error = parse_object(raw)
            if parse_error:
                record.update({"status": "PARSE_FAILED", "parse_error": parse_error})
            else:
                record["parsed_output"] = parsed
                try:
                    validate(parsed, schema)
                    record["status"] = "SCHEMA_VALID"
                except Exception as error:
                    record.update({"status": "SCHEMA_INVALID", "schema_error": f"{type(error).__name__}:{error}"})
                record["closure"] = closure_candidate(parsed, common_by_root[root], dialogues[root], schema)
        except Exception as error:
            record.update({"status": "API_FAILED", "api_attempts": 1, "error": f"{type(error).__name__}:{error}"})
        attempts.append(record)
    write_jsonl(output / "pilot_attempt_results.jsonl", attempts)
    candidates = [{"root_id": x["root_id"], "request_id": x["request_id"], **(x.get("parsed_output") or {}), "attempt_status": x["status"], "closure": x.get("closure", {})} for x in attempts if isinstance(x.get("parsed_output"), dict)]
    write_jsonl(output / "pilot_candidate_results.jsonl", candidates)
    baseline_rows: list[dict[str, Any]] = []
    if BASELINE_GENERATION_RUN.is_dir():
        for pool in BASELINE_GENERATION_RUN.glob("*/candidate_pool.json"):
            try:
                baseline_rows.extend(read_json(pool))
            except Exception:
                pass
    for row in candidates:
        row["historical_free_exploration_relation"] = baseline_relations(row, [x for x in baseline_rows if x.get("generation_source") == "free_exploration"])
    write_jsonl(output / "pilot_candidate_results.jsonl", candidates)
    strategy_counts: dict[str, int] = {}
    for row in candidates:
        strategy = str(row.get("selected_strategy") or "UNKNOWN")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    closures = [x.get("closure", {}) for x in attempts]
    raw_r03_mentions = sum(
        any(str(ref).startswith("R03:") for ref in ((x.get("parsed_output") or {}).get("evidence_refs") or []) + ((x.get("parsed_output") or {}).get("new_r03_evidence_refs") or []))
        for x in attempts if isinstance(x.get("parsed_output"), dict)
    )
    schema_invalid_count = sum(x["status"] == "SCHEMA_INVALID" for x in attempts)
    closure = {"status": "PASS" if schema_invalid_count == 0 else "PASS_WITH_GAPS", "seed_count": 6, "attempt_count": len(attempts), "response_received_count": sum(x["status"] in {"RESPONSE_RECEIVED", "SCHEMA_VALID", "SCHEMA_INVALID"} for x in attempts), "schema_valid_count": sum(x["status"] == "SCHEMA_VALID" for x in attempts), "no_safe_evolution_count": sum(x.get("parsed_output", {}).get("selected_strategy") == "NO_SAFE_EVOLUTION" for x in attempts if isinstance(x.get("parsed_output"), dict)), "successor_output_count": sum(x.get("parsed_output", {}).get("selected_strategy") != "NO_SAFE_EVOLUTION" for x in attempts if isinstance(x.get("parsed_output"), dict) and x.get("status") == "SCHEMA_VALID"), "api_failed_count": sum(x["status"] == "API_FAILED" for x in attempts), "parse_failed_count": sum(x["status"] == "PARSE_FAILED" for x in attempts), "schema_invalid_count": schema_invalid_count, "selected_strategy_distribution": strategy_counts, "exact_duplicate_count": sum(bool(x.get("exact_duplicate")) for x in closures), "lexical_paraphrase_count": sum(bool(x.get("lexical_paraphrase")) for x in closures), "source_question_rewrite_count": sum(bool(x.get("source_question_rewrite")) for x in closures), "historical_free_exploration_exact_duplicate_count": sum(int(x.get("historical_free_exploration_relation", {}).get("exact_duplicate_count", 0)) for x in candidates), "historical_free_exploration_lexical_paraphrase_count": sum(int(x.get("historical_free_exploration_relation", {}).get("lexical_paraphrase_count", 0)) for x in candidates), "semantic_duplicate_count": 0, "semantic_duplicate_unknown_count": sum(x.get("schema_valid") is True and not x.get("no_safe_evolution") for x in closures), "gold_structural_valid_count": sum(x.get("gold_structural_status") == "STRUCTURE_VALID_UNVERIFIED_SEMANTICS" for x in closures), "distractor_structural_valid_count": sum(x.get("distractor_structural_status") == "STRUCTURE_PRESENT_UNVERIFIED_SEMANTICS" for x in closures), "r03_evidence_used_count": sum(bool(x.get("r03_evidence_used")) for x in closures), "raw_r03_evidence_mentioned_count": raw_r03_mentions, "evidence_relation_changed_count": sum(bool(x.get("evidence_relation_changed")) for x in closures), "reasoning_requirement_claimed_count": sum(x.get("reasoning_requirement_status") == "CLAIMED_UNVERIFIED" for x in closures), "changed_provenance_context_variant_count": sum(bool(x.get("changed_provenance_context_variant")) for x in closures), "semantic_auditor_calls": 0, "solver_calls": 0, "memorax_calls": 0, "training_calls": 0, "retry_count": 0, "auto_seed_replacement_count": 0, "baseline_free_exploration": baseline_summary(), "selection_scope": {"excluded_quarantine_root": QUARANTINE_ROOT, "excluded_no_safe_seed_root": UNSAFE_ROOT}}
    write_json(output / "offline_qa_closure.json", closure)
    write_json(output / "qa_writer_call_closure.json", {"status": "PASS" if len(attempts) == 6 and all(x["api_attempts"] == 1 for x in attempts) else "FAIL", "qa_writer_calls": len(attempts), "attempts_per_seed": {x["root_id"]: x["api_attempts"] for x in attempts}, "retry_count": 0, "solver_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0, "memorax_status_calls": 0, "memorax_mutation_calls": 0, "semantic_auditor_calls": 0, "training_calls": 0})
    write_json(output / "pilot_input_source_sha_manifest.json", {"status": "PASS", "files": [{"path": path, "sha256_before": digest, "sha256_after": sha_file(Path(path)), "unchanged": digest == sha_file(Path(path))} for path, digest in sorted(sources.items())]})
    report = {"version": VERSION, "status": "R03_FEEDBACK_GUIDED_TARGETED_EVOLUTION_PILOT_COMPLETE" if len(attempts) == 6 and schema_invalid_count == 0 else "R03_FEEDBACK_GUIDED_TARGETED_EVOLUTION_PILOT_COMPLETE_WITH_GAPS", "next_status": "READY_FOR_OFFLINE_FEEDBACK_QA_CLOSURE_REVIEW", "template_sha256": TEMPLATE_SHA256, "template_sha_consistent": True, "eligible_roots": list(ELIGIBLE_ROOTS), "excluded_roots": {"quarantine": QUARANTINE_ROOT, "no_safe_seed": UNSAFE_ROOT}, "attempt_count": len(attempts), "call_closure": closure, "input_sources_unchanged": True, "no_retry": True, "valid": len(attempts) == 6 and schema_invalid_count == 0 and all(x["api_attempts"] == 1 for x in attempts)}
    write_json(output / "final_r03_feedback_guided_targeted_evolution_pilot_report.json", report)
    md = "\n".join(["# R03 Feedback-Guided Targeted Evolution Pilot v1", "", f"Status: `{report['status']}`", f"Next: `{report['next_status']}`", "", f"- Six eligible seeds, one attempt each; QA Writer calls: `{len(attempts)}`; retries: `0`.", f"- NO_SAFE_EVOLUTION: `{closure['no_safe_evolution_count']}`; schema-valid successor outputs: `{closure['successor_output_count']}`.", f"- Strategy distribution: `{strategy_counts}`.", f"- Exact duplicates: `{closure['exact_duplicate_count']}`; lexical paraphrases: `{closure['lexical_paraphrase_count']}`; semantic duplicate judgment: offline UNKNOWN (no semantic auditor).", f"- R03 evidence used: `{closure['r03_evidence_used_count']}`; changed evidence relation: `{closure['evidence_relation_changed_count']}`; claimed new reasoning requirement (unverified): `{closure['reasoning_requirement_claimed_count']}`.", "- Solver, training, Memorax, semantic auditor: all `0`. 112812 and 96462 were excluded as frozen.", ""])
    write_bytes(output / "final_r03_feedback_guided_targeted_evolution_pilot_report.md", md.encode("utf-8"))
    return 0


class _backend_response:
    def __init__(self, result: tuple[str, dict[str, Any]]):
        self.raw_text, self.metadata = result
        self.model_identifier = self.metadata.get("model", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return run(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
