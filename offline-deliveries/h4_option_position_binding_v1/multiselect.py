"""Rebind mechanically checked Qwen options before multiselect Solver evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / "h4_crossroot_multiselect_solver_selection_v2/protocol.py"
LABELS = "ABCDEF"
VERSION = "multiselect_source_bound_positions_v1"


def load_protocol():
    spec = importlib.util.spec_from_file_location("position_binding_solver_protocol", PROTOCOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def stable_order(question, counter):
    identity = {"version": VERSION, "candidate_id": question["candidate_id"],
                "root_id": question["root_id"], "source_sha256": question["source_sha256"],
                "original_options": question["options"], "counter": counter}
    seed = hashlib.sha256(canonical_bytes(identity)).digest()
    return tuple(sorted(LABELS, key=lambda label: hashlib.sha256(seed + label.encode()).digest()))


def bind_question(question):
    old_gold = set(question["gold"])
    if not 2 <= len(old_gold) <= 6:
        raise ValueError("GOLD_CARDINALITY_INVALID:" + question["candidate_id"])
    prefix = set(LABELS[:len(old_gold)])
    for counter in range(720):
        order = stable_order(question, counter)
        new_gold = [label for label, original in zip(LABELS, order) if original in old_gold]
        if len(old_gold) == 6 or set(new_gold) != prefix:
            break
    else:
        raise ValueError("NO_NONPREFIX_BINDING:" + question["candidate_id"])
    rebound = {**question,
               "options": {label: question["options"][original]
                           for label, original in zip(LABELS, order)},
               "option_evidence": {label: question["option_evidence"][original]
                                   for label, original in zip(LABELS, order)},
               "gold": new_gold}
    if ({(question["options"][label], canonical_bytes(question["option_evidence"][label]))
         for label in LABELS}
        != {(rebound["options"][label], canonical_bytes(rebound["option_evidence"][label]))
            for label in LABELS}):
        raise ValueError("OPTION_EVIDENCE_IDENTITY_CHANGED:" + question["candidate_id"])
    if any((rebound["option_evidence"][label]["relation"] == "supports")
           != (label in new_gold) for label in LABELS):
        raise ValueError("GOLD_MAPPING_INVALID:" + question["candidate_id"])
    audit = {"candidate_id": question["candidate_id"],
             "source_sha256": question["source_sha256"],
             "display_to_original": dict(zip(LABELS, order)),
             "original_gold": question["gold"], "rebound_gold": new_gold,
             "all_six_true": len(old_gold) == 6,
             "nonprefix_gold": len(old_gold) < 6 and set(new_gold) != prefix,
             "deterministic_counter": counter}
    return rebound, audit


def build(input_pool):
    protocol = load_protocol()
    source, questions, _ = protocol.validate_pool(input_pool)
    provenance = source["provenance"]
    if (source["status"] != "RAW_MECHANICALLY_VERIFIED_READY_FOR_SOLVER"
            or provenance.get("option_order") != "original_qwen_draft_order"):
        raise ValueError("ORIGINAL_QWEN_RAW_POOL_REQUIRED")
    rebound, audits = [], []
    for question in questions:
        row, audit = bind_question(question)
        rebound.append(row)
        audits.append(audit)
    output = {**source, "version": VERSION,
              "provenance": {**provenance,
                             "option_order": "deterministic_source_bound_rebinding_v1",
                             "position_binding_version": VERSION,
                             "input_pool_path": str(input_pool.resolve(strict=True)),
                             "input_pool_sha256": protocol.sha(input_pool)},
              "questions": rebound}
    report = {"version": VERSION, "input_pool_path": str(input_pool.resolve(strict=True)),
              "input_pool_sha256": protocol.sha(input_pool),
              "candidate_count": len(audits),
              "non_six_count": sum(not row["all_six_true"] for row in audits),
              "non_six_nonprefix_count": sum(row["nonprefix_gold"] for row in audits),
              "six_true_count": sum(row["all_six_true"] for row in audits),
              "bindings": audits, "new_solver_calls": 0, "training_started": False}
    return output, report


def verify(input_pool, output_pool, report_path):
    expected_pool, expected_report = build(input_pool)
    actual_pool = json.loads(output_pool.read_text(encoding="utf-8"))
    actual_report = json.loads(report_path.read_text(encoding="utf-8"))
    if actual_pool != expected_pool or actual_report != expected_report:
        raise ValueError("POSITION_BINDING_REPLAY_MISMATCH")
    load_protocol().validate_pool(output_pool)
    return expected_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "verify"))
    parser.add_argument("--input-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pool_path = args.output_dir / "POOL.json"
    report_path = args.output_dir / "POSITION_BINDING.json"
    if args.phase == "build":
        pool, report = build(args.input_pool)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for path, value in ((pool_path, pool), (report_path, report)):
            with path.open("x", encoding="utf-8") as output:
                json.dump(value, output, ensure_ascii=False, indent=2)
                output.write("\n")
    else:
        report = verify(args.input_pool, pool_path, report_path)
    print(json.dumps({"status": "POSITION_BINDING_VALID", "candidates": report["candidate_count"],
                      "non_six_nonprefix": report["non_six_nonprefix_count"],
                      "six_true": report["six_true_count"]}))


if __name__ == "__main__":
    main()
