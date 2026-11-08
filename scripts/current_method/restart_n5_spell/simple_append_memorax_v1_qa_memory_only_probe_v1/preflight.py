"""Zero-API preflight for the isolated Memory-only Writer branch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import sha256_file
from .memory_only_contract import MAX_QUESTIONS, memory_source_refs, materialize_solver_rows, validate_writer_output
from .memory_only_prompts import build_memory_only_writer_prompt

DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_memory_only_fast_expansion_v1_live_1")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_qa_memory_only_probe_v1_preflight")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _tree_sha(root: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\0")
    return digest.hexdigest()


def _fixture(count: int, refs: list[str]) -> dict[str, Any]:
    rows = []
    for index in range(count):
        gold = "A fact"
        rows.append({"question_type": "state_change", "question": f"Q{index}",
                     "options": {"A": gold, "B": f"B{index}", "C": f"C{index}", "D": f"D{index}"},
                     "gold_text": gold, "memory_source_refs": [refs[0]]})
    return {"questions": rows}


def run(*, source_run: Path = DEFAULT_SOURCE_RUN,
        run_root: Path = DEFAULT_RUN_ROOT) -> dict[str, Any]:
    memory_path = source_run / "memory" / "post_generation_overview.json"
    dialogue_path = source_run / "expanded_dialogue.json"
    checks: dict[str, bool] = {
        "source_run_exists": source_run.is_dir(),
        "memory_snapshot_exists": memory_path.is_file(),
        "expanded_dialogue_exists_for_later_validation": dialogue_path.is_file(),
        "run_root_absent": not run_root.exists(),
        "api_calls_zero": True, "writer_calls_zero": True,
        "solver_calls_zero": True, "expansion_calls_zero": True,
        "memorax_calls_zero": True,
    }
    details: dict[str, Any] = {}
    if all(checks.values()):
        overview = _load(memory_path)
        refs = memory_source_refs(overview)
        prompt = build_memory_only_writer_prompt(overview)
        dialogue = _load(dialogue_path)
        dialogue_texts = {str(row.get("text") or "") for row in dialogue.get("turns") or []}
        prompt_has_turn_serialization = any(text and text in prompt["prompt"] for text in dialogue_texts)
        invalid_ref = {"questions": [{
            "question_type": "state_change", "question": "q", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "gold_text": "a", "memory_source_refs": ["MZ999"]}]}
        multi_gold = {"questions": [{
            "question_type": "state_change", "question": "q", "options": {"A": "same", "B": "same", "C": "c", "D": "d"},
            "gold_text": "same", "memory_source_refs": [next(iter(refs))]}]}
        valid = validate_writer_output(_fixture(3, list(refs)), source_refs=refs)
        empty = validate_writer_output({"questions": []}, source_refs=refs)
        twelve = validate_writer_output(_fixture(12, list(refs)), source_refs=refs)
        thirteen = validate_writer_output(_fixture(13, list(refs)), source_refs=refs)
        invalid = validate_writer_output(invalid_ref, source_refs=refs)
        ambiguous = validate_writer_output(multi_gold, source_refs=refs)
        solver_rows = materialize_solver_rows(valid, root_id=str(overview.get("root_id")))
        checks.update({
            "memory_overview_not_qa_fact_source": overview.get("memory_is_qa_fact_source") is False,
            "memory_has_event_or_person_records": bool(overview.get("events") or overview.get("people")),
            "writer_prompt_memory_only": not prompt_has_turn_serialization and "No transcript" in prompt["prompt"],
            # The prompt may mention turn IDs in its prohibition text; it must
            # not contain an actual serialized canonical ref or dialogue row.
            "writer_prompt_no_canonical_turn_ids": "D001" not in prompt["prompt"] and "source_turn_refs" not in prompt["prompt"],
            "zero_questions_valid": empty["valid"], "one_to_three_valid": valid["valid"],
            "twelve_capacity_valid": twelve["valid"], "thirteen_rejected": not thirteen["valid"],
            "unknown_memory_ref_rejected": (not invalid["questions"] and
                any("memory_source_refs" in q["errors"] for q in invalid["quarantine"])),
            "gold_ambiguity_rejected": (not ambiguous["questions"] and
                any("gold_text_not_unique" in q["errors"] for q in ambiguous["quarantine"])),
            "solver_shape_compatible": all(set(row) >= {"qa_id", "question", "options", "correct_option", "canonical_answer_text"} for row in solver_rows),
        })
        details = {"root_id": overview.get("root_id"), "memory_path": str(memory_path),
                   "dialogue_path": str(dialogue_path), "source_tree_sha256": _tree_sha(source_run),
                   "memory_file_sha256": sha256_file(memory_path),
                   "memory_overview_declared_sha256": overview.get("memory_overview_sha256"),
                   "memory_ref_count": len(refs), "memory_refs": refs,
                   "prompt_sha256": prompt["prompt_sha256"], "prompt_character_count": prompt["character_count"],
                   "prompt_token_estimate": max(1, (len(prompt["prompt"]) + 3) // 4),
                   "dialogue_sha256": __import__("hashlib").sha256(dialogue_path.read_bytes()).hexdigest(),
                   "dialogue_turn_count": len(dialogue.get("turns") or [])}
    failed = sorted(key for key, value in checks.items() if not value)
    report = {"preflight_version": "qa_memory_only_probe_v1", "decision": "pass" if not failed else "fail",
              "checks": checks, "failed_checks": failed, "details": details,
              "api_calls": 0, "writer_calls": 0, "solver_calls": 0,
              "method_contract_changed": False,
              "mainline_modified": False,
              "branch_scope": "QA-only; Writer input is frozen Memory only; transcript reserved for later audit"}
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_root / "preflight_report.md").write_text("\n".join([
        "# QA Memory-only Probe v1 Preflight", "", f"Decision: `{report['decision']}`",
        f"Frozen source: `{source_run}`", f"Memory records: `{details.get('memory_ref_count', 0)}`",
        f"Writer prompt estimate: `{details.get('prompt_token_estimate', 0)}` tokens",
        "Writer/QA/Solver/Expansion/Memorax calls: `0 / 0 / 0 / 0 / 0`", "",
        "This is an isolated branch. Existing modules and historical runs were not modified.",
        "", "Failed checks: " + (", ".join(failed) if failed else "none"), ""
    ]), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["decision"] == "pass" else 1)
