from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.simple_append_v1.qa_live import (
    CONTEXT_LIMIT,
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
    MAX_QUESTION_COUNT,
    QUESTION_COUNT,
    WRITER_MAX_TOKENS,
    build_solver_source,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.prompts import build_mcq_batch_prompt
from scripts.current_method.restart_n5_spell.simple_append_v1.qa import build_evidence_alias_registry

PROBE_VERSION = "qa_difficulty_integration_materialize_v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_expanded_dialogue(*, source_record: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    source_path = Path(str(source_record["dialogue_path"]))
    frozen = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(frozen, Mapping) or not isinstance(frozen.get("turns"), list):
        raise ValueError("frozen_source_dialogue_invalid")
    parent_count = int(frozen.get("original_parent_turn_count") or 0)
    if parent_count < 1:
        parent_count = sum(row.get("origin") == "source_grounded" for row in frozen["turns"] if isinstance(row, Mapping))
    parent_rows = list(frozen["turns"][:parent_count])
    if len(parent_rows) != parent_count or any(row.get("origin") != "source_grounded" for row in parent_rows):
        raise ValueError("source_parent_prefix_invalid")
    if str(child.get("root_id") or "") != str(source_record.get("root_id") or ""):
        raise ValueError("scene_release_root_mismatch")
    scenes = child.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 3:
        raise ValueError("scene_release_requires_three_scenes")
    turns = copy.deepcopy(parent_rows)
    scene_refs: list[str] = []
    for scene in scenes:
        scene_ref = str(scene.get("scene_ref") or "")
        session_ref = str(scene.get("session_ref") or "")
        if not scene_ref or not session_ref:
            raise ValueError("scene_release_scene_refs_missing")
        scene_refs.append(scene_ref)
        for row in scene.get("dialogue_turns") or []:
            turns.append({
                "turn_ref": str(row.get("turn_ref") or ""),
                "session_ref": session_ref,
                "scene_ref": scene_ref,
                "speaker": str(row.get("speaker") or ""),
                "text": str(row.get("text") or ""),
                "origin": "generated_extension",
            })
    refs = [str(row.get("turn_ref") or "") for row in turns]
    if any(not ref for ref in refs) or len(refs) != len(set(refs)):
        raise ValueError("scene_release_turn_refs_invalid")
    result = {
        "expanded_dialogue_version": "simple_append_expanded_dialogue_v1",
        "root_id": str(source_record["root_id"]),
        "append_mode": "parent_tail_only",
        "parent_dialogue_sha256": str(frozen.get("parent_dialogue_sha256") or ""),
        "parent_turn_count": parent_count,
        "child_turn_count": len(turns) - parent_count,
        "turn_count": len(turns),
        "child_scene_refs": scene_refs,
        "child_boundary": {
            "first_child_turn_index_1_based": parent_count + 1,
            "last_parent_turn_ref": str(parent_rows[-1]["turn_ref"]),
            "first_child_turn_ref": str(turns[parent_count]["turn_ref"]),
        },
        "turns": turns,
        "qa_fact_source": "complete_parent_plus_child",
        "memory_is_qa_fact_source": False,
        "source_scene_release_probe": str(source_record.get("case_id") or ""),
    }
    result["complete_dialogue_sha256"] = _sha(result)
    return result


def materialize(*, scene_run: Path, output_source: Path) -> dict[str, Any]:
    if output_source.exists() and any(output_source.iterdir()):
        raise FileExistsError(f"refusing_to_overwrite_nonempty_source:{output_source}")
    source_record = json.loads((scene_run / "source_record.json").read_text(encoding="utf-8"))
    child = json.loads((scene_run / "child_dialogue.json").read_text(encoding="utf-8"))
    expanded = build_expanded_dialogue(source_record=source_record, child=child)
    output_source.mkdir(parents=True, exist_ok=True)
    _write_json(output_source / "expanded_dialogue.json", expanded)
    _write_json(output_source / "reports" / "ROUND_SUMMARY.json", {
        "status": "completed",
        "generation_ready": True,
        "qa_source_ready": True,
        "root_id": expanded["root_id"],
        "turn_count": expanded["turn_count"],
        "child_scene_count": len(expanded["child_scene_refs"]),
        "source_scene_release_run": str(scene_run.resolve()),
        "source_child_sha256": hashlib.sha256((scene_run / "child_dialogue.json").read_bytes()).hexdigest(),
    })
    source = build_solver_source(expanded)
    aliases = build_evidence_alias_registry(expanded)
    prompt = build_mcq_batch_prompt(expanded, target_questions=QUESTION_COUNT, maximum_questions=MAX_QUESTION_COUNT, focus_round_index=1)
    checks = {
        "expanded_contract": expanded["qa_fact_source"] == "complete_parent_plus_child" and expanded["memory_is_qa_fact_source"] is False,
        "parent_present": aliases["parent_count"] == expanded["parent_turn_count"],
        "child_present": aliases["child_count"] == expanded["child_turn_count"],
        "turn_count_matches": len(source["combined_dialogue"]) == expanded["turn_count"],
        "unique_turn_refs": len({row["turn_id"] for row in source["combined_dialogue"]}) == expanded["turn_count"],
        "writer_prompt_fits": prompt["token_estimate"] + WRITER_MAX_TOKENS <= CONTEXT_LIMIT,
        "solver_assets_present": DEFAULT_TOKENIZER.is_dir() and DEFAULT_SOLVER_MODEL.is_dir(),
        "no_future_scene_plan_in_source": True,
    }
    failed = sorted(key for key, value in checks.items() if not value)
    metrics = {
        "probe_version": PROBE_VERSION,
        "scene_release_run": str(scene_run.resolve()),
        "output_source": str(output_source.resolve()),
        "root_id": expanded["root_id"],
        "expanded_dialogue_sha256": hashlib.sha256((output_source / "expanded_dialogue.json").read_bytes()).hexdigest(),
        "turn_count": expanded["turn_count"],
        "parent_turn_count": expanded["parent_turn_count"],
        "child_turn_count": expanded["child_turn_count"],
        "writer_prompt_token_estimate": prompt["token_estimate"],
        "checks": checks,
        "failed_checks": failed,
        "decision": "pass" if not failed else "fail",
        "api_calls": 0,
    }
    _write_json(output_source / "QA_INTEGRATION_PREFLIGHT.json", metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="materialize scene-release output for isolated QA integration")
    parser.add_argument("--scene-run", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(scene_run=args.scene_run.resolve(), output_source=args.output_source.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
