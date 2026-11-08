"""Reproducible CPU-only P0 smoke; creates new artifacts, never calls a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from scripts.current_method import base_qa_pipeline_v1 as qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    artifacts = Path(__file__).resolve().parent
    prior = artifacts.parent / "natural_question_forms_fg_15root_extension_20261026_v1"
    root_ids = ["cc-train-episode-109164", "cc-train-episode-24719", "cc-train-episode-761"]
    original_selection = prior / "SOURCE_SELECTION_FINAL.json"
    selection = qa.read(original_selection)
    selection["roots"] = [r for r in selection["roots"] if r["root_id"] in root_ids]
    if {r["root_id"] for r in selection["roots"]} != set(root_ids):
        raise ValueError("expected three reviewed sources")
    selection_path = out / "SOURCE_SELECTION_THREE_ROOTS.json"
    qa.write_new(selection_path, selection)

    bindings = {b["path"]: b for b in qa.read(artifacts / "CALLCHAIN_AND_DEFECTS.json")["cited_file_bindings"]}
    for b in qa.read(artifacts / "EXISTING_CASE_REVIEW.json")["input_bindings"]:
        bindings[b["path"]] = b
    bindings[str(original_selection)] = qa.legacy.file_binding(original_selection)
    for entry in selection["roots"]:
        for filename in (entry["path"], entry["warnings_path"]):
            bindings[filename] = qa.legacy.file_binding(filename)
    if any(qa.legacy.file_binding(b["path"]) != b for b in bindings.values()):
        raise ValueError("historical binding already differs from audit")

    suite = subprocess.run([sys.executable, "-B", "-m", "unittest",
        "scripts.current_method.test_base_qa_pipeline_v1",
        "scripts.current_method.test_production_question_gate",
        "scripts.current_method.test_question_presentation_gate"],
        cwd=PROJECT, text=True, capture_output=True)
    (out / "UNITTEST_OUTPUT.txt").write_text(suite.stdout + suite.stderr)
    if suite.returncode:
        raise RuntimeError("offline regression failed; inspect UNITTEST_OUTPUT.txt")

    source_smokes = []
    for root_id in root_ids:
        packets = []
        for arm in ("F", "G"):
            run = out / "prepared" / root_id / arm
            qa.main(["prepare", "--selection", str(selection_path), "--root", root_id,
                     "--arm", arm, "--run-dir", str(run)])
            context, source = qa._context(run)
            packets.append(qa.read(run / "DISCOVERY_PACKET.json"))
        if packets[0] != packets[1]:
            raise AssertionError("F/G base discovery inputs diverged")
        source_smokes.append({"root_id": root_id, "source_binding": context["source_binding"],
            "source_turn_count": len(source["turns"]), "source_character_count": sum(len(t["text"]) for t in source["turns"]),
            "fg_discovery_packets_identical": True, "feedback_injected": False,
            "status": "READY_FOR_DISCOVERY_ONLY_NOT_NEW_QA"})

    old_path = prior / "CANDIDATES_PRIVATE.json"
    old = qa.read(old_path)
    valid = [c for c in old if qa.text(c.get("question"))]
    if len(old) != 150 or len(valid) != 141:
        raise AssertionError("historical candidate count differs from audit")
    adapted = [{k: c[k] for k in ("candidate_id", "question_type", "question", "options", "gold")}
               | {"relation_type": "historical_operation_label:" + c["task_operation"]} for c in valid]
    adapter_path = out / "HISTORICAL_141_AUDIT_INPUT.json"
    qa.write_new(adapter_path, {"adapter_note": "Format adapter only; no candidate edits, semantic review, or new generation.",
                               "original_binding": qa.legacy.file_binding(old_path), "candidates": adapted})
    qa.main(["audit-batch", "--input", str(adapter_path), "--output", str(out / "HISTORICAL_141_PATTERN_AUDIT.json")])
    audit = qa.read(out / "HISTORICAL_141_PATTERN_AUDIT.json")["audit"]
    if len(audit["canonical_first_k_candidates"]) != 138:
        raise AssertionError("canonical prefix pattern disagrees with original audit")
    source_cache, rows = {}, []
    for c in valid:
        source_path = c["source_binding"]["path"]
        if source_path not in source_cache:
            source_cache[source_path] = qa.source_from(c["source_binding"], c["root_id"])
        turns = source_cache[source_path]["turns"]
        full_turn_choices = [k for k, value in c["options"].items()
            if any(t["text"] in value for t in turns if len(t["text"]) >= 15)]
        rows.append({"candidate_id": c["candidate_id"], "arm": c["arm"], "slot_id": c["slot_id"],
            "operation_label": c["task_operation"], "gold_count": len(c["gold"]),
            "canonical_prefix_gold": set(list(c["options"])[:len(c["gold"])]) == set(c["gold"]),
            "options_containing_verbatim_full_source_turn": full_turn_choices,
            "all_options_contain_verbatim_full_source_turn": len(full_turn_choices) == len(c["options"]),
            "evidence_count_per_option": {k: len(c["item_analysis"][k].get("evidence", [])) for k in c["options"]},
            "semantic_necessity": "NOT_DETERMINED_BY_CITATION_COUNTS"})
    qa.write_new(out / "HISTORICAL_141_STATIC_CHECKS.json", {"original_binding": qa.legacy.file_binding(old_path),
        "scope": "All 141 records mechanically inspected; not 141 new independent semantic reviews.",
        "verbatim_metric": "Case-sensitive full-turn substring of at least 15 characters; no punctuation normalization.",
        "all_six_verbatim_full_turn_questions": sum(r["all_options_contain_verbatim_full_source_turn"] for r in rows),
        "rows": rows})
    unchanged = [{**b, "unchanged": qa.legacy.file_binding(b["path"]) == b} for b in bindings.values()]
    if not all(b["unchanged"] for b in unchanged):
        raise AssertionError("historical source or code changed")
    qa.write_new(out / "HISTORICAL_INTEGRITY.json", {"bindings": unchanged, "all_unchanged": True})
    result = {"version": qa.VERSION, "model_calls": 0, "solver_calls": 0, "training_runs": 0,
        "actual_api_cost": 0, "unit_tests_passed": True, "unittest_output": str(out / "UNITTEST_OUTPUT.txt"),
        "sources": source_smokes, "historical_candidates": len(old), "historical_valid_questions": len(valid),
        "canonical_first_k": len(audit["canonical_first_k_candidates"]), "historical_binding_count": len(unchanged),
        "historical_bindings_unchanged": True, "new_questions_generated": 0,
        "limits": ["Source preparation and software regressions only; no fresh relation or QA authored.",
                   "No empirical naturalness, semantic necessity, shortcut, or Solver improvement measured.",
                   "No Qwen availability probe, model identity attestation, or API price verification."],
        "implementation_bindings": qa.implementation_bindings()}
    qa.write_new(out / "SMOKE_RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
