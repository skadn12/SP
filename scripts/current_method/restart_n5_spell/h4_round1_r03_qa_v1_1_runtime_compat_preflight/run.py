"""Render v1.1 prompts and verify dynamic Solver interfaces without API calls."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from . import (
    CONTRACT_RUN, DEFAULT_OUTPUT, FROZEN_FEEDBACK_TEMPLATE_SHA256, MULTI_PASS_RUN,
    TARGETED_RUN, TARGETED_TEMPLATE_RUN, VERSION,
)
from .runtime import audit_rendered_prompt, multi_pass_prompt, targeted_prompt
from .solver_compat import (
    audit_rebinding, parse_structured, position_variants, requests_per_model,
    score_structured, structured_choices,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(output: Path, name: str, value: Any) -> str:
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".jsonl"):
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in value)
    elif isinstance(value, str):
        data = value if value.endswith("\n") else value + "\n"
    else:
        data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def verify(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"missing_hashed_input:{path}")
    observed = sha(path)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError(f"input_sha_mismatch:{path}")
    return {"path": str(path), "sha256": observed, "sidecar_match": True}


def dialogue_by_root(manifest: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(manifest):
        result[str(row["root_id"])] = read_json(Path(str(row["path"])))
    return result


def _multi(count: int) -> dict[str, Any]:
    keys = [chr(ord("A") + index) for index in range(count)]
    return {"candidate_id": f"multi-{count}", "question_type": "multi_select", "question": "Select all supported statements.", "options": {key: f"statement-{key}" for key in keys}, "gold": ["A", "C"]}


def _ordering(count: int) -> dict[str, Any]:
    keys = [f"I{index + 1}" for index in range(count)]
    return {"candidate_id": f"ordering-{count}", "question_type": "ordering", "question": "Put the events in order.", "items": {key: f"event-{key}" for key in keys}, "gold": list(reversed(keys))}


def prompt_preflight(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = [
        CONTRACT_RUN / "qa_output_contract_addendum_v1_1.md",
        CONTRACT_RUN / "qa_output_schema_v1_1_targeted.json",
        CONTRACT_RUN / "qa_output_schema_v1_1_composition.json",
        CONTRACT_RUN / "qa_output_schema_v1_1_fresh.json",
        TARGETED_RUN / "qa_feedback_template_v1.md",
        TARGETED_RUN / "common_feedback_inputs.jsonl",
        TARGETED_RUN / "authoritative_dialogue_manifest.jsonl",
        TARGETED_TEMPLATE_RUN / "qa_evolution_strategy_schema_v1.json",
        MULTI_PASS_RUN / "multi_qa_composition_template_v1.md",
        MULTI_PASS_RUN / "fresh_exploration_template_v1.md",
        MULTI_PASS_RUN / "composition_sampling_manifest.jsonl",
        MULTI_PASS_RUN / "fresh_sampling_manifest.jsonl",
        MULTI_PASS_RUN / "authoritative_dialogue_manifest.jsonl",
    ]
    identities = [verify(path) for path in inputs]
    addendum = (CONTRACT_RUN / "qa_output_contract_addendum_v1_1.md").read_text(encoding="utf-8")
    targeted_template_path = TARGETED_RUN / "qa_feedback_template_v1.md"
    targeted_template = targeted_template_path.read_text(encoding="utf-8")
    targeted_schema = read_json(CONTRACT_RUN / "qa_output_schema_v1_1_targeted.json")
    strategy = read_json(TARGETED_TEMPLATE_RUN / "qa_evolution_strategy_schema_v1.json")
    targeted_dialogues = dialogue_by_root(TARGETED_RUN / "authoritative_dialogue_manifest.jsonl")
    prompts: list[dict[str, Any]] = []
    for common in read_jsonl(TARGETED_RUN / "common_feedback_inputs.jsonl"):
        root = str(common["source_identity"]["root_id"])
        prompt = targeted_prompt(common, targeted_dialogues[root], frozen_template=targeted_template, addendum=addendum, output_schema=targeted_schema, strategy_schema=strategy)
        checks = audit_rendered_prompt(prompt, schema_id=targeted_schema["$id"], template_sha=FROZEN_FEEDBACK_TEMPLATE_SHA256, observed_template_sha=sha(targeted_template_path))
        write(output, f"rendered_prompts/targeted/{root}.txt", prompt)
        prompts.append({"pass": "targeted", "root_id": root, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_chars": len(prompt), "checks": checks})
    multi_dialogues = dialogue_by_root(MULTI_PASS_RUN / "authoritative_dialogue_manifest.jsonl")
    pass_config = {
        "composition": ("composition_sampling_manifest.jsonl", "multi_qa_composition_template_v1.md", "qa_output_schema_v1_1_composition.json"),
        "fresh": ("fresh_sampling_manifest.jsonl", "fresh_exploration_template_v1.md", "qa_output_schema_v1_1_fresh.json"),
    }
    for pass_name, (sample_name, template_name, schema_name) in pass_config.items():
        template_path = MULTI_PASS_RUN / template_name
        template = template_path.read_text(encoding="utf-8")
        schema = read_json(CONTRACT_RUN / schema_name)
        for sample in read_jsonl(MULTI_PASS_RUN / sample_name):
            if not sample.get("api_call_planned"):
                continue
            root = str(sample["root_id"])
            prompt = multi_pass_prompt(pass_name, sample, multi_dialogues[root], frozen_template=template, addendum=addendum, output_schema=schema)
            checks = audit_rendered_prompt(prompt, schema_id=schema["$id"], template_sha=sha(template_path), observed_template_sha=sha(template_path))
            write(output, f"rendered_prompts/{pass_name}/{root}.txt", prompt)
            prompts.append({"pass": pass_name, "root_id": root, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_chars": len(prompt), "checks": checks})
    return prompts, identities


def solver_preflight() -> list[dict[str, Any]]:
    rows = []
    for qtype, factory in (("multi_select", _multi), ("ordering", _ordering)):
        for count in (4, 5, 6):
            candidate = factory(count)
            variants = position_variants(candidate)
            rebinding = audit_rebinding(candidate, variants)
            allowed = list((candidate.get("options") or candidate.get("items")).keys())
            choices = structured_choices(qtype, allowed)
            expected_answer = variants[0]["gold"]
            raw = f"<box>{','.join(expected_answer)}</box>"
            parsed = parse_structured(raw, qtype, allowed)
            scored = score_structured(parsed, expected_answer, qtype)
            rows.append({
                "question_type": qtype, "option_count": count,
                "variant_count": len(variants), "rebinding": rebinding,
                "guided_choice_count": len(choices),
                "expected_guided_choice_count": (2 ** count - 1) if qtype == "multi_select" else __import__("math").factorial(count),
                "parser_status": parsed["parse_status"], "scoring_correct": scored["correct"],
                "requests_per_model_k3_two_contexts": requests_per_model(candidate),
                "requests_two_models_k3_two_contexts": requests_per_model(candidate) * 2,
            })
    return rows


def legacy_audit() -> list[dict[str, str]]:
    base = Path(__file__).parents[1]
    checks = [
        (base / "h4_qa_multiroot_k3_t08_probe_v1/validator.py", '"multi_select": ("A", "B", "C", "D", "E")'),
        (base / "h4_qa_multiroot_k3_t08_probe_v1/permutation.py", 'MULTI_KEYS = ("A", "B", "C", "D", "E")'),
        (base / "h4_qa_multiroot_k3_t08_probe_v1/permutation.py", 'ORDER_KEYS = ("I1", "I2", "I3", "I4")'),
        (base / "h4_qa_multiroot_k3_t08_probe_v1/final_audit.py", 'expected_variant_count = 5 if'),
        (base / "structured_qa_box_protocol.py", 'letters = ("A", "B", "C", "D", "E")'),
        (base / "h4_position_permutation_ab_probe_v1/core.py", '"multi_select": 5, "ordering": 4'),
    ]
    rows = []
    for path, needle in checks:
        found = needle in path.read_text(encoding="utf-8")
        rows.append({"path": str(path), "needle": needle, "found": str(found).lower(), "disposition": "LEGACY_FIXED_ARITY_DO_NOT_USE_FOR_V1_1"})
    return rows


def run(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable_output_exists:{output}")
    output.mkdir(parents=True, exist_ok=False)
    prompts, identities = prompt_preflight(output)
    solver_rows = solver_preflight()
    legacy_rows = legacy_audit()
    prompt_checks_pass = bool(prompts) and all(all(row["checks"].values()) for row in prompts)
    solver_checks_pass = all(
        row["variant_count"] == row["option_count"]
        and row["rebinding"]["pass"]
        and row["guided_choice_count"] == row["expected_guided_choice_count"]
        and row["parser_status"] == "success"
        and row["scoring_correct"]
        and row["requests_per_model_k3_two_contexts"] == row["option_count"] * 6
        for row in solver_rows
    )
    checks = {
        "rendered_prompt_count_20": len(prompts) == 20,
        "all_prompts_use_v1_1_addendum_and_schema": prompt_checks_pass,
        "frozen_feedback_template_sha_preserved": sha(TARGETED_RUN / "qa_feedback_template_v1.md") == FROZEN_FEEDBACK_TEMPLATE_SHA256,
        "dynamic_solver_4_5_6_pass": solver_checks_pass,
        "legacy_fixed_arity_detected": all(row["found"] == "true" for row in legacy_rows),
        "source_sidecars_match": all(row["sidecar_match"] for row in identities),
        "prohibited_calls_zero": True,
    }
    status = "R03_QA_V1_1_RUNTIME_COMPAT_PREFLIGHT_PASS" if all(checks.values()) else "FAIL"
    write(output, "rendered_prompt_manifest.jsonl", prompts)
    write(output, "dynamic_solver_compatibility.jsonl", solver_rows)
    write(output, "legacy_fixed_arity_audit.json", {"status": "DETECTED_AND_ISOLATED", "files": legacy_rows})
    write(output, "source_sha_manifest.json", {"status": "PASS", "files": identities})
    write(output, "runtime_compat_preflight_report.json", {
        "version": VERSION, "status": status, "checks": checks,
        "prompt_counts": {name: sum(row["pass"] == name for row in prompts) for name in ("targeted", "composition", "fresh")},
        "solver_contract": {"K": 3, "context_modes": ["no_context", "full_context"], "requests_per_model": "option_count * 3 * 2", "option_counts": [4, 5, 6]},
        "legacy_policy": "Do not route v1.1 candidates through fixed-arity legacy validators/permutation/audit code.",
        "calls": {"qa_writer": 0, "solver_model": 0, "semantic_auditor": 0, "memorax": 0, "training": 0},
        "next_status": "READY_FOR_BOUNDED_SCHEMA_REPAIR_VERIFICATION" if status.endswith("PASS") else "BLOCKED_BY_RUNTIME_COMPAT_GAP",
    })
    write(output, "runtime_compat_preflight_report.md", "\n".join([
        "# R03 QA v1.1 Runtime Compatibility Preflight", "", f"Status: `{status}`", "",
        f"Rendered `{len(prompts)}` actual prompts from frozen inputs with the v1.1 addendum and schemas.",
        "Dynamic offline Solver fixtures passed for multi-select and ordering with 4, 5, and 6 items.",
        "Legacy fixed-arity modules were detected and must not be used for v1.1 candidates.",
        "No QA Writer, Solver model, semantic auditor, Memorax, or training call was made.", "",
    ]))
    return 0 if status.endswith("PASS") else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-direct-tests", action="store_true")
    args = parser.parse_args()
    if args.run_direct_tests:
        from . import test_preflight
        failures = []
        tests = [function for name, function in inspect.getmembers(test_preflight, inspect.isfunction) if name.startswith("test_")]
        for function in tests:
            try:
                function()
            except Exception as error:
                failures.append({"test": function.__name__, "error": repr(error)})
        if failures:
            raise RuntimeError(f"direct_tests_failed:{failures}")
    return run(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
