"""Run at most two isolated live requests against the revised targeted schema.

This is a verification harness, not a new QA pilot. It consumes only frozen
Common Feedback Data and successful local R03 dialogue artifacts, and never
writes Question Bank, Memorax, Solver, or training state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from jsonschema import validate

from ..backends import BackendResponse
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.contract import (
    build_targeted_schema,
    local_closure,
)
from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.prompts import targeted_prompt
from ..h4_round1_r03_qa_output_contract_v1_1_revision_1_preflight.run import verify
from . import (
    API_CONFIG,
    CLOSURE_RUN,
    FROZEN_TEMPLATE_SHA256,
    MAX_OUTPUT_TOKENS,
    MAX_REQUESTS,
    OUTPUT,
    REVISION_PREFLIGHT_RUN,
    ROOTS,
    TEMPLATE_RUN,
    TEMPERATURE,
)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    sha = digest_bytes(data)
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


def single_http_json(config: Any, prompt: str) -> BackendResponse:
    """Exactly one HTTP attempt; no retry or automatic resend is permitted."""
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=max(180, int(config.timeout_seconds))) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body["choices"][0]
    return BackendResponse(
        str(choice["message"]["content"]),
        config.model,
        {
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
            "transport_attempt_count": 1,
            "transport_retry_count": 0,
            "rate_limit_retry_count": 0,
        },
    )


def frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {
        "common": TEMPLATE_RUN / "common_feedback_data_manifest.jsonl",
        "schema": TEMPLATE_RUN / "qa_feedback_output_schema_v1.json",
        "template": TEMPLATE_RUN / "qa_feedback_template_v1.md",
        "strategy": TEMPLATE_RUN / "qa_evolution_strategy_schema_v1.json",
        "root_manifest": CLOSURE_RUN / "r03_local_batch_root_manifest.jsonl",
        "revision_schema": REVISION_PREFLIGHT_RUN / "qa_output_schema_v1_1_revision_1_targeted.json",
    }
    source_hashes = {name: verify(path) for name, path in paths.items()}
    template = paths["template"].read_text(encoding="utf-8")
    if digest_bytes(template.encode("utf-8")) != FROZEN_TEMPLATE_SHA256:
        raise RuntimeError("frozen_feedback_template_sha_mismatch")
    common_rows = {str(row["source_identity"]["root_id"]): row for row in read_jsonl(paths["common"])}
    roots = {str(row["root_id"]): row for row in read_jsonl(paths["root_manifest"])}
    dialogues: dict[str, dict[str, Any]] = {}
    for root in ROOTS:
        if root not in common_rows or root not in roots:
            raise RuntimeError(f"missing_frozen_seed:{root}")
        row = roots[root]
        if row.get("disposition") != "R03_LOCAL_SUCCESS":
            raise RuntimeError(f"seed_not_local_success:{root}")
        dialogue_path = Path(str(row["dialogue_artifacts"]["cumulative_dialogue_path"]))
        source_hashes[f"dialogue:{root}"] = verify(dialogue_path)
        dialogues[root] = read_json(dialogue_path)
    return common_rows, dialogues, {
        "template": template,
        "strategy": read_json(paths["strategy"]),
        "source_schema": read_json(paths["schema"]),
        "targeted_schema": read_json(paths["revision_schema"]),
    }, source_hashes


def known_refs(dialogue: Mapping[str, Any]) -> set[str]:
    return {str(turn.get("turn_ref")) for turn in dialogue.get("turns") or [] if turn.get("turn_ref")}


def result_for_output(output: dict[str, Any] | None, parse_error: str | None, schema: Mapping[str, Any], dialogue: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"parse_error": parse_error, "schema_valid": False, "closure": None}
    if output is None:
        return result
    try:
        validate(output, schema)
        result["schema_valid"] = True
    except Exception as error:
        result["schema_error"] = f"{type(error).__name__}:{str(error).splitlines()[0]}"
        return result
    result["selected_strategy"] = output.get("selected_strategy")
    result["question_type"] = output.get("question_type")
    result["closure"] = local_closure(output, mode="targeted", known_refs=known_refs(dialogue))
    result["derived_new_r03_evidence_refs"] = result["closure"].get("derived_new_r03_evidence_refs", [])
    return result


def run(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output_not_empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    common, dialogues, inputs, source_hashes = frozen_inputs()
    schema = build_targeted_schema(inputs["source_schema"])
    if schema != inputs["targeted_schema"]:
        raise RuntimeError("revision_schema_drift")
    config = load_challenger_config(API_CONFIG)
    ledger_manifest = digest_bytes(json.dumps({"roots": ROOTS, "schema": schema, "template_sha256": FROZEN_TEMPLATE_SHA256}, sort_keys=True).encode())
    ledger = RequestLedger(output_dir, ledger_manifest, "live")
    write_json(output_dir / "qa_output_schema_v1_1_revision_1_targeted.json", schema)
    write_json(output_dir / "input_contract.json", {
        "status": "FROZEN_INPUTS_VERIFIED",
        "roots": list(ROOTS),
        "source_hashes": source_hashes,
        "frozen_feedback_template_sha256": FROZEN_TEMPLATE_SHA256,
        "revision_preflight_run": str(REVISION_PREFLIGHT_RUN),
        "old_pilot_untouched": True,
    })
    rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for index, root in enumerate(ROOTS, start=1):
        common_row = common[root]
        prompt = targeted_prompt(common_row, dialogues[root], template=inputs["template"], schema=schema, strategy_schema=inputs["strategy"])
        request_id = f"bounded-{index:02d}-{root}"
        write_bytes(output_dir / "raw" / "prompts" / f"{request_id}.txt", prompt.encode("utf-8"))
        prompt_sha = digest_bytes(prompt.encode("utf-8"))
        call_row: dict[str, Any] = {"request_id": request_id, "root_id": root, "attempt": 1, "prompt_sha256": prompt_sha, "transport_attempt_limit": 1}
        raw: str | None = None
        try:
            response = ledger.invoke(
                request_id=request_id,
                intent="r03_qa_output_contract_v1_1_revision_1_bounded_verification",
                attempt=1,
                prompt_sha256=prompt_sha,
                call=lambda: single_http_json(config, prompt),
                budget_class="bounded_verification",
                budget_limit=MAX_REQUESTS,
                total_budget_limit=MAX_REQUESTS,
                request_metadata={"root_id": root, "verification_only": True, "solver": False, "memorax": False},
                backend_identity={"model": config.model, "base_url": config.base_url, "temperature": TEMPERATURE, "max_tokens": MAX_OUTPUT_TOKENS},
            )
            raw = response.raw_text
            write_bytes(output_dir / "raw" / "model_outputs" / f"{request_id}.txt", raw.encode("utf-8"))
            parsed, parse_error = parse_object(raw)
            evaluation = result_for_output(parsed, parse_error, schema, dialogues[root])
            call_row.update({"status": "COMPLETED", "model": response.model_identifier, "response_metadata": response.metadata, "response_sha256": digest_bytes(raw.encode("utf-8")), "evaluation": evaluation})
            decision = "SCHEMA_PASS_NO_SAFE" if evaluation.get("schema_valid") and evaluation.get("closure", {}).get("status") == "NO_SAFE_EVOLUTION" else ("SCHEMA_PASS_QA" if evaluation.get("schema_valid") and evaluation.get("closure", {}).get("status") == "PASS" else "SCHEMA_PASS_BUT_CLOSURE_FAIL" if evaluation.get("schema_valid") else "SCHEMA_FAIL")
            ledger.record_decision(request_id, decision)
        except Exception as error:
            call_row.update({"status": "TRANSPORT_OR_EXECUTION_FAILURE", "error": f"{type(error).__name__}:{error}"})
        calls.append(call_row)
    write_jsonl(output_dir / "bounded_request_closure.jsonl", calls)
    completed = [row for row in calls if row["status"] == "COMPLETED"]
    schema_pass = [row for row in completed if row.get("evaluation", {}).get("schema_valid")]
    no_safe = [row for row in schema_pass if row.get("evaluation", {}).get("closure", {}).get("status") == "NO_SAFE_EVOLUTION"]
    qa_pass = [row for row in schema_pass if row.get("evaluation", {}).get("closure", {}).get("status") == "PASS"]
    report = {
        "status": "R03_QA_OUTPUT_CONTRACT_V1_1_REVISION_BOUNDED_VERIFICATION_PASS" if len(completed) == MAX_REQUESTS and len(schema_pass) == MAX_REQUESTS else "R03_QA_OUTPUT_CONTRACT_V1_1_REVISION_BOUNDED_VERIFICATION_COMPLETE_WITH_GAPS",
        "next_status": "READY_FOR_BOUNDED_GENERATION_REVIEW" if len(completed) == MAX_REQUESTS else "REVIEW_TRANSPORT_FAILURES_BEFORE_PROGRESS",
        "request_count": len(calls), "completed_request_count": len(completed), "schema_valid_count": len(schema_pass), "qa_structural_pass_count": len(qa_pass), "no_safe_count": len(no_safe),
        "technical_failures": [row for row in calls if row["status"] != "COMPLETED"],
        "forbidden_calls": {"memorax": 0, "solver": 0, "qa_generation_outside_bounded_verification": 0, "semantic_auditor": 0, "training": 0},
        "old_pilot_untouched": True,
        "max_request_budget": MAX_REQUESTS,
        "calls": calls,
    }
    write_json(output_dir / "bounded_verification_report.json", report)
    markdown = "\n".join([
        "# Bounded Generation Verification",
        "",
        f"Status: `{report['status']}`",
        f"Requests: `{len(calls)}/{MAX_REQUESTS}`; schema-valid: `{len(schema_pass)}`; QA closure PASS: `{len(qa_pass)}`; NO_SAFE: `{len(no_safe)}`.",
        "",
        "This run is isolated from the historical pilot and performs no Solver, Memorax, semantic-auditor, training, or Question Bank operation.",
    ]) + "\n"
    write_bytes(output_dir / "bounded_verification_report.md", markdown.encode("utf-8"))
    all_files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and not path.name.endswith(".sha256") and path.name != "output_sha_manifest.json":
            all_files.append({"path": str(path.relative_to(output_dir)), "sha256": digest(path), "sidecar_present": path.with_name(path.name + ".sha256").is_file()})
    write_json(output_dir / "output_sha_manifest.json", all_files)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
