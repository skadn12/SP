"""Render and validate the current revision-1 multi-pass request plan.

This module deliberately stops before the live QA Writer call.  It uses the
already frozen seven-root sampling selection from the historical multi-pass
pilot, but rebuilds prompts and schemas through the active v1.1 revision
implementation.  No Targeted request is scheduled or rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import validate

from ..h4_round1_r03_multi_pass_qa_generation_comparison_pilot_v1.run import (
    COMPOSITION_TEMPLATE,
    FRESH_TEMPLATE,
)
from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.contract import build_multi_pass_schema
from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.prompts import multi_pass_prompt
from ..h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight.run import verify
from . import (
    CONTEXT_LIMIT,
    MAX_OUTPUT_TOKENS,
    MAX_REQUESTS,
    OLD_PILOT,
    OUTPUT,
    QUARANTINE_ROOT,
    REVISION_PREFLIGHT,
    SUCCESS_ROOTS,
    TEMPLATE_RUN,
    TEMPERATURE,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    sha = digest(path)
    path.with_name(path.name + ".sha256").write_text(f"{sha}  {path.name}\n", encoding="utf-8")
    return sha


def write_json(path: Path, value: Any) -> str:
    return write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return write_bytes(path, data.encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_text(path: Path) -> str:
    return verify(path)["sha256"]


def load_frozen() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    required = [
        OLD_PILOT / "composition_sampling_manifest.jsonl",
        OLD_PILOT / "fresh_sampling_manifest.jsonl",
        OLD_PILOT / "authoritative_dialogue_manifest.jsonl",
        OLD_PILOT / "multi_qa_composition_template_v1.md",
        OLD_PILOT / "fresh_exploration_template_v1.md",
        OLD_PILOT / "multi_qa_composition_output_schema_v1.json",
        OLD_PILOT / "fresh_exploration_output_schema_v1.json",
        REVISION_PREFLIGHT / "qa_output_schema_v1_1_revision_1_composition.json",
        REVISION_PREFLIGHT / "qa_output_schema_v1_1_revision_1_fresh.json",
        TEMPLATE_RUN / "qa_feedback_output_schema_v1.json",
    ]
    hashes = {str(path): verify_text(path) for path in required}
    composition = read_jsonl(OLD_PILOT / "composition_sampling_manifest.jsonl")
    fresh = read_jsonl(OLD_PILOT / "fresh_sampling_manifest.jsonl")
    dialogue_manifest = read_jsonl(OLD_PILOT / "authoritative_dialogue_manifest.jsonl")
    dialogues: dict[str, Any] = {}
    for row in dialogue_manifest:
        root = str(row["root_id"])
        path = Path(str(row["path"]))
        hashes[f"dialogue:{root}"] = verify_text(path)
        observed = read_json(path)
        if int(row.get("turn_count") or 0) != len(observed.get("turns") or []):
            raise RuntimeError(f"dialogue_turn_count_drift:{root}")
        dialogues[root] = observed
    return composition, fresh, {"dialogues": dialogues, "manifest": dialogue_manifest}, hashes


def planned(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("api_call_planned"))]


def render_pass(pass_name: str, rows: list[dict[str, Any]], dialogues: Mapping[str, Any], schema: Mapping[str, Any], template: str, output: Path) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for row in rows:
        root = str(row["root_id"])
        prompt = multi_pass_prompt(pass_name, row, dialogues[root], template=template, schema=schema)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        request_id = f"{root}:{pass_name}:revision-1:attempt-001"
        relative = f"rendered_prompts/{pass_name}/{root}.txt"
        write_bytes(output / relative, prompt.encode("utf-8"))
        checks = {
            "revision_schema_id_present": schema["$id"] in prompt,
            "question_type_explicit": "explicit question_type" in prompt,
            "single_choice_four": "exactly A-D" in prompt,
            "multi_select_four_to_six": "4-6 contiguous keys" in prompt,
            "ordering_four_to_six": "I1-I4, I1-I5, or I1-I6" in prompt,
            "evidence_refs_once": "evidence_refs once" in prompt,
            "no_new_r03_evidence_refs": "Do not emit new_r03_evidence_refs" in prompt,
            "minimal_real_evidence_policy": "Evidence completeness is not required" in prompt,
            "one_attempt": "one bounded R03" in prompt,
        }
        rendered.append({
            "request_id": request_id,
            "root_id": root,
            "pass": pass_name.upper(),
            "attempt": 1,
            "api_call_authorized": False,
            "api_call_planned": True,
            "prompt_path": str(output / relative),
            "prompt_sha256": prompt_sha,
            "prompt_character_count": len(prompt),
            "prompt_word_count_approx": len(prompt.split()),
            "context_limit_tokens": CONTEXT_LIMIT,
            "checks": checks,
            "source_sample": row,
        })
    return rendered


def run(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable_output_exists:{output}")
    output.mkdir(parents=True, exist_ok=True)
    composition, fresh, dialogue_data, source_hashes = load_frozen()
    if len(composition) != len(SUCCESS_ROOTS) or len(fresh) != len(SUCCESS_ROOTS):
        raise RuntimeError("frozen_sampling_row_count_drift")
    comp_planned, fresh_planned = planned(composition), planned(fresh)
    expected_roots = set(SUCCESS_ROOTS)
    checks: dict[str, Any] = {
        "composition_7_rows": len(comp_planned) == 7,
        "fresh_7_rows": len(fresh_planned) == 7,
        "composition_root_set": {str(x["root_id"]) for x in comp_planned} == expected_roots,
        "fresh_root_set": {str(x["root_id"]) for x in fresh_planned} == expected_roots,
        "quarantine_root_excluded": all(str(x["root_id"]) != QUARANTINE_ROOT for x in comp_planned + fresh_planned),
        "96462_composition_included": any(str(x["root_id"]) == "cc-train-episode-96462" for x in comp_planned),
        "96462_fresh_included": any(str(x["root_id"]) == "cc-train-episode-96462" for x in fresh_planned),
        "one_attempt_per_root_per_pass": all(int(x.get("attempt") or 1) == 1 for x in comp_planned + fresh_planned),
        "targeted_not_scheduled": True,
    }
    source_schema = read_json(TEMPLATE_RUN / "qa_feedback_output_schema_v1.json")
    schemas = {"composition": build_multi_pass_schema(source_schema, "composition"), "fresh": build_multi_pass_schema(source_schema, "fresh")}
    for name, schema in schemas.items():
        expected_path = REVISION_PREFLIGHT / f"qa_output_schema_v1_1_revision_1_{name}.json"
        checks[f"{name}_schema_matches_revision_preflight"] = schema == read_json(expected_path)
    rendered_comp = render_pass("composition", comp_planned, dialogue_data["dialogues"], schemas["composition"], (OLD_PILOT / "multi_qa_composition_template_v1.md").read_text(encoding="utf-8"), output)
    rendered_fresh = render_pass("fresh", fresh_planned, dialogue_data["dialogues"], schemas["fresh"], (OLD_PILOT / "fresh_exploration_template_v1.md").read_text(encoding="utf-8"), output)
    rendered = rendered_comp + rendered_fresh
    checks["14_total_requests"] = len(rendered) == MAX_REQUESTS
    checks["all_prompt_contract_checks"] = all(all(item["checks"].values()) for item in rendered)
    checks["no_prompt_over_context_limit_approx"] = all(item["prompt_word_count_approx"] < CONTEXT_LIMIT for item in rendered)
    # Validate the revised schemas' conditional behavior with the existing
    # deterministic fixtures; this remains entirely offline.
    fixture_no_safe = {
        "outcome": "NO_SAFE_EVOLUTION", "question_type": None, "new_question": None,
        "options_or_items": None, "canonical_gold": None, "evidence_refs": [],
        "selected_evidence_relations": [], "expected_context_dependency": None,
        "r03_semantic_contribution_claimed": None, "answerable_without_r03_claim": None,
        "no_safe_evolution_reason": "No safe successor is supported.",
    }
    fixture_qa = {
        "outcome": "QA", "question_type": "multi_select", "new_question": "Which?",
        "options_or_items": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "canonical_gold": ["A", "B"], "evidence_refs": ["R03:S01:T001"],
        "selected_evidence_relations": ["relation-a"], "expected_context_dependency": "dialogue",
        "r03_semantic_contribution_claimed": True, "answerable_without_r03_claim": False,
        "no_safe_evolution_reason": None, "independent_new_qa_reason": "new relation", "novelty_basis": "R03 state change",
    }
    for name in ("composition", "fresh"):
        no_safe = dict(fixture_no_safe)
        if name == "composition":
            no_safe.update({"source_cluster_ids": ["cluster-a", "cluster-b"], "composition_relation": None})
        else:
            no_safe.update({"independent_new_qa_reason": None, "novelty_basis": None})
        try:
            validate(no_safe, schemas[name])
            checks[f"{name}_no_safe_fixture_valid"] = True
        except Exception:
            checks[f"{name}_no_safe_fixture_valid"] = False
        qa = dict(fixture_qa)
        if name == "composition":
            qa.pop("independent_new_qa_reason", None)
            qa.pop("novelty_basis", None)
            qa.update({"source_cluster_ids": ["cluster-a", "cluster-b"], "composition_relation": "joint", "selected_evidence_relations": ["relation-a", "relation-b"]})
        else:
            qa.pop("source_cluster_ids", None)
            qa.pop("composition_relation", None)
        try:
            validate(qa, schemas[name])
            checks[f"{name}_qa_fixture_valid"] = True
        except Exception:
            checks[f"{name}_qa_fixture_valid"] = False
    write_json(output / "pilot_contract.json", {
        "version": "h4_round1_r03_multi_pass_qa_generation_comparison_revision_1_preflight",
        "status": "PREFLIGHT_ONLY_NO_API",
        "current_output_contract": "qa_output_contract_v1_1_revision_1",
        "allowed_passes": ["MULTI_QA_COMPOSITION", "FRESH_EXPLORATION"],
        "targeted_mode": "NOT_SCHEDULED_NO_RERUN",
        "success_roots": list(SUCCESS_ROOTS), "quarantine_root": QUARANTINE_ROOT,
        "composition_requests_planned": len(rendered_comp), "fresh_requests_planned": len(rendered_fresh),
        "total_requests_planned": len(rendered), "max_requests": MAX_REQUESTS,
        "one_attempt_per_root_per_pass": True, "retry_policy": "none", "reserve_seed_switching": False,
        "temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS, "context_limit": CONTEXT_LIMIT,
        "qa_writer_calls": 0, "solver_calls": 0, "training_calls": 0, "memorax_calls": 0,
        "semantic_auditor_calls": 0, "dialogue_mutation_calls": 0,
        "authorization_required_before_live": True,
    })
    write_json(output / "qa_output_schema_v1_1_revision_1_composition.json", schemas["composition"])
    write_json(output / "qa_output_schema_v1_1_revision_1_fresh.json", schemas["fresh"])
    write_jsonl(output / "composition_request_manifest.jsonl", rendered_comp)
    write_jsonl(output / "fresh_request_manifest.jsonl", rendered_fresh)
    write_json(output / "source_sha_manifest.json", {"status": "PASS", "files": [{"path": path, "sha256": sha} for path, sha in sorted(source_hashes.items())]})
    report = {
        "status": "R03_MULTI_PASS_REVISION_1_PREFLIGHT_PASS" if all(checks.values()) else "R03_MULTI_PASS_REVISION_1_PREFLIGHT_FAIL",
        "next_status": "AWAITING_EXPLICIT_AUTHORIZATION_FOR_MAX_14_QA_WRITER_CALLS" if all(checks.values()) else "BLOCKED_BY_PREFLIGHT_CHECKS",
        "checks": checks,
        "composition_requests_planned": len(rendered_comp), "fresh_requests_planned": len(rendered_fresh),
        "total_requests_planned": len(rendered), "qa_writer_calls": 0,
        "targeted_rerun": 0, "solver_calls": 0, "training_calls": 0, "memorax_calls": 0, "semantic_auditor_calls": 0,
        "old_pilot_immutable_reference": str(OLD_PILOT),
        "interpretation": "This preflight proves only that the active revision-1 prompts and schemas can be rendered for the frozen seven-root selection. It is not a live generation result.",
    }
    write_json(output / "revision_1_multi_pass_preflight_report.json", report)
    markdown = "\n".join([
        "# R03 Multi-Pass Revision-1 Preflight",
        "",
        f"Status: `{report['status']}`",
        f"Planned requests: Composition `{len(rendered_comp)}`, Fresh `{len(rendered_fresh)}`, total `{len(rendered)}/{MAX_REQUESTS}`.",
        "",
        "This run only renders the current revision-1 prompts and validates frozen inputs. No QA Writer request was sent.",
        "Targeted is not scheduled or rerun. Solver, training, Memorax, semantic auditor, and dialogue mutation calls are all zero.",
        "",
        f"Next: `{report['next_status']}`",
        "",
    ])
    write_bytes(output / "revision_1_multi_pass_preflight_report.md", (markdown + "\n").encode("utf-8"))
    manifest = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and not path.name.endswith(".sha256") and path.name != "output_sha_manifest.json":
            manifest.append({"path": str(path.relative_to(output)), "sha256": digest(path), "sidecar_present": path.with_name(path.name + ".sha256").is_file()})
    # All files created above carry a sidecar; the manifest itself is excluded
    # from its own entries to avoid self-hash recursion.
    write_json(output / "output_sha_manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
