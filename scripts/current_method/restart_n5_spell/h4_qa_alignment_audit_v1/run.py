"""Read-only runner for the H4 QA reasoning-contract alignment audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text, canonical_json, sha256_file
from .audit import audit_bundle


DEFAULT_INPUT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_mining_probe_v1_phase0_4"
)
DEFAULT_OUTPUT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_178392_h4_qa_alignment_audit_v1_phase0_4"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run(*, input_run: Path, output_run: Path) -> dict[str, Any]:
    if output_run.exists():
        raise RuntimeError(f"output_run_already_exists:{output_run}")
    required = {
        "manifest": input_run / "manifest.json",
        "candidate_pool": input_run / "candidate_pool.json",
        "contracts": input_run / "reasoning_contracts.json",
        "intents": input_run / "intents.json",
        "static": input_run / "static_filter.json",
        "difficulty": input_run / "difficulty.json",
        "report": input_run / "report.json",
    }
    for path in required.values():
        _require(path)
    manifest = _load(required["manifest"])
    candidates = _load(required["candidate_pool"])
    contracts = _load(required["contracts"])
    intents = _load(required["intents"])
    static = _load(required["static"])
    difficulty = _load(required["difficulty"])
    source_path = Path(str(manifest.get("source_dialogue") or ""))
    _require(source_path)
    source = _load(source_path)
    source_hash = sha256_file(source_path)
    if source_hash != manifest.get("source_dialogue_sha256"):
        raise RuntimeError("source_dialogue_hash_mismatch")
    if not isinstance(candidates, list) or not isinstance(contracts, list):
        raise ValueError("phase0_candidate_or_contract_schema_invalid")
    difficulty_rows = list((difficulty or {}).get("candidate_summaries") or [])
    turn_text = {
        str(row.get("turn_ref")): str(row.get("text") or "")
        for row in source.get("turns") or [] if row.get("turn_ref")
    }
    audit = audit_bundle(
        candidates=candidates, contracts=contracts, turn_text=turn_text,
        static_rows=static if isinstance(static, dict) else {},
        difficulty_rows=difficulty_rows,
    )
    input_hashes = {key: sha256_file(path) for key, path in required.items()}
    audit["input_run"] = str(input_run)
    audit["input_hashes"] = input_hashes
    audit["source_dialogue"] = str(source_path)
    audit["source_dialogue_sha256"] = source_hash
    audit["intent_count"] = len((intents or {}).get("validation", {}).get("intents") or [])
    audit["contract_count"] = len(contracts)
    audit["production_mainline_modified"] = False
    audit["phase0_input_modified"] = False
    output_run.mkdir(parents=True, exist_ok=False)
    atomic_json(output_run / "alignment_audit.json", audit)
    atomic_json(output_run / "input_manifest.json", {
        "audit_version": "h4_qa_alignment_audit_v1",
        "input_run": str(input_run), "input_hashes": input_hashes,
        "source_dialogue": str(source_path), "source_dialogue_sha256": source_hash,
        "offline_only": True, "challenger_api_calls": 0, "solver_calls": 0,
    })
    report = {
        "report_version": "h4_qa_alignment_audit_report_v1",
        "status": "completed",
        "output_run": str(output_run),
        "input_run": str(input_run),
        "candidate_count": audit["candidate_count"],
        "contract_alignment_pass_count": audit["contract_alignment_pass_count"],
        "structural_relation_cue_pass_count": audit["structural_relation_cue_pass_count"],
        "alignment_bucket_counts": audit["alignment_bucket_counts"],
        "difficulty_by_alignment": audit["difficulty_by_alignment"],
        "issue_counts": audit["issue_counts"],
        "challenger_api_calls": 0,
        "solver_calls": 0,
        "offline_only": True,
        "phase0_input_modified": False,
        "production_mainline_modified": False,
    }
    atomic_json(output_run / "report.json", report)
    lines = [
        "# H4 QA Alignment Audit v1",
        "",
        "Status: **completed**",
        f"Input: `{input_run}`",
        f"Candidates audited: {audit['candidate_count']}",
        f"Alignment buckets: `{json.dumps(audit['alignment_bucket_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"Contract alignment pass: {audit['contract_alignment_pass_count']}/{audit['candidate_count']}",
        f"Structural relation-cue pass: {audit['structural_relation_cue_pass_count']}/{audit['candidate_count']}",
        f"Issue counts: `{json.dumps(audit['issue_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "This is a deterministic offline audit. It made zero Challenger API calls and zero Solver calls.",
        "Lexical evidence support and variable overlap are diagnostics, not semantic entailment proofs.",
        "The Phase-0 input run and production mainline were not modified.",
    ]
    atomic_text(output_run / "summary.md", "\n".join(lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT_RUN)
    parser.add_argument("--output-run", type=Path, default=DEFAULT_OUTPUT_RUN)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(**vars(parse_args())), ensure_ascii=False, indent=2))
