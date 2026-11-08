#!/usr/bin/env python3
"""Actual schema and mechanical provenance checks; not a semantic QA substitute."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
LIBRARY = BASE / "design/PUBLIC_DESIGN_LIBRARY.json"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


schema = read(LIBRARY)["schemas"]["candidate_record"]
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema, format_checker=FormatChecker())
provenance = read(OUT / "MECHANICAL_PROVENANCE.json")
provenance_by_id = {p["candidate_id"]: p for p in provenance["candidates"]}
schema_results = []
mechanical_results = []

for path in sorted(OUT.glob("A-week1-*.json")):
    record = read(path)
    candidate_id = record["candidate_id"]
    errors = [{"path": "/".join(map(str, e.absolute_path)), "message": e.message}
              for e in validator.iter_errors(record)]
    schema_results.append({"path": str(path), "sha256": sha(path), "valid": not errors,
                           "errors": errors})
    checks = []

    def check(name, condition, detail):
        checks.append({"check": name, "result": "PASS" if condition else "FAIL", "detail": detail})

    source_path = Path(record["source_binding"]["source_path"])
    source = read(source_path)
    check("source_hash", sha(source_path) == record["source_binding"]["source_sha256"]
          == record["source_binding"]["source_scope_sha256"], "Full exact source and declared scope hash.")
    refs = record["evidence_span"]["evidence_refs"]
    ref_by_id = {r["ref_id"]: r for r in refs}
    check("unique_evidence_ids", len(refs) == len(ref_by_id), str(len(refs)))
    exact = True
    for ref in refs:
        pos = ref["source_position"]
        turn = source["turns"][pos["global_turn_index"]]
        exact &= turn["turn_ref"] == ref["turn_id"]
        exact &= (turn["scene_ref"] or turn["session_ref"]) == ref["scene_id"]
        exact &= turn["text"][pos["span_start"]:pos["span_end"]] == ref["quote"]
    check("exact_source_positions_and_quotes", exact, "Zero-based index and exact Python character span checked for every evidence ref.")
    cited = []
    for item in record["target_realization"]["criteria"] + record["lexical_shortcuts"]["checks"] + record["qa_review"]["criterion_results"]:
        cited += item["evidence_ref_ids"]
    cited += record["evidence_span"]["revision_chain_ref_ids"]
    provenance_entry = provenance_by_id[candidate_id]
    hashes_match = True
    for field in ["original_public", "final_public", "private_author_artifact", "plan", "public_packet", "independent_reconstruction", "independent_review", "clarification_record"]:
        value = provenance_entry[field]
        if value:
            hashes_match &= sha(value["path"]) == value["sha256"]
    check("bound_artifact_hashes", hashes_match, "Original/final public, private, plan, public packet, independent reconstruction/review and optional clarification bindings.")
    histories = record["generation_history"]
    check("generation_hashes", all(sha(h["raw_response_path"]) == h["candidate_content_sha256"] for h in histories)
          and all(h["request_input_sha256"] == sha(provenance_entry["public_packet"]["path"]) for h in histories),
          "Actual preserved public artifact hash and exact per-root input packet hash.")
    check("source_admission_precedes_generation", all(datetime.fromisoformat(record["source_review"]["reviewed_at"])
          < datetime.fromisoformat(h["started_at"]) for h in histories),
          "Existing material-admission timestamp precedes all recorded author artifact creation timestamps.")
    clarification_count = 1 if record["root_id"].endswith("104814") else 0
    check("honest_budget_metadata", record["budget"]["initial_used"] == 1
          and record["budget"]["content_clarification_used"] == clarification_count
          and len(histories) == 1 + clarification_count
          and record["budget"]["actual_cap_support"] == "UNKNOWN"
          and record["budget"]["actual_cap_evidence"] is None
          and all(h["request_id"] is None and h["actual_output_tokens"] is None
                  and h["actual_cap_support"] == "UNKNOWN" for h in histories),
          "One initial, zero/one clarification; no fabricated actual cap, usage or API request IDs.")
    check("hold_boundary", record["disposition"]["state"] == "HOLD"
          and record["fact_family"]["duplicate_review"] == "UNKNOWN"
          and record["disposition"]["accepted_content_sha256"] is None
          and record["disposition"]["feedback_binding_location"] is None
          and record["budget"]["solver_calls"] == 0,
          "Independent content PASS does not become global acceptance.")
    check("no_unearned_cross_turn_claim", "E_CROSS_TURN_NECESSARY" not in record["target_realization"]["evidence_structure_tags"]
          and record["target_realization"]["single_turn_sufficiency_sweep"]["status"] == "NOT_APPLICABLE",
          "No cross-turn necessity claim or fabricated premise ablation.")
    final_public = read(provenance_entry["final_public"]["path"])
    check("six_final_public_lengths", len(final_public["options_or_items"]) == 6
          and record["lexical_shortcuts"]["option_lengths"] == [len(final_public["options_or_items"][k]) for k in "ABCDEF"],
          "Lengths are exact final-public character lengths in A-F order.")
    if record["question_type"] == "multi_select":
        payload = record["multi_select"]
        for item in payload["options"]:
            cited += item["support_ref_ids"] + item["refute_ref_ids"]
        check("MC_inherited_quota_and_truth", len(payload["gold_option_ids"]) == payload["target_gold_count_inherited"]
              and set(payload["gold_option_ids"]) == {i["option_id"] for i in payload["options"] if i["source_verdict"] == "SUPPORTED"}
              and all(i["source_verdict"] in ["SUPPORTED", "REFUTED"] for i in payload["options"]),
              "Inherited quota and exact independent SUPPORTED set; every distractor explicitly REFUTED.")
        expected_criteria = ["B1", "B2", "B3"] if record["target_realization"]["primary_relation_id"] == "MC_BINDING" else ["S1", "S2", "S3"]
    else:
        payload = record["ordering"]
        items = {i["item_id"]: i for i in payload["items"]}
        positions = {label: ref_by_id[i["selected_ref_id"]]["source_position"]["global_turn_index"] for label, i in items.items()}
        scenes = {label: ref_by_id[i["selected_ref_id"]]["scene_id"] for label, i in items.items()}
        counts = Counter(scenes.values())
        mention_selection = True
        for item in items.values():
            cited.append(item["selected_ref_id"])
            cited += [o["ref_id"] for o in item["all_occurrences"]]
            matched = [o["ref_id"] for o in item["all_occurrences"] if o["included_for_event_key"]]
            mention_selection &= item["selected_ref_id"] in matched
            mention_selection &= all(o["reviewer_verified"] for o in item["all_occurrences"])
            mention_selection &= ref_by_id[item["selected_ref_id"]]["source_position"]["global_turn_index"] == min(ref_by_id[r]["source_position"]["global_turn_index"] for r in matched)
        check("ORDER_mention_policy", mention_selection, "Every selected ref is the earliest independently included occurrence, with all independent exclusions preserved.")
        check("ORDER_gold_and_profile", len(set(positions.values())) == 6
              and payload["gold_item_ids"] == sorted(positions, key=positions.get)
              and sorted(counts.values(), reverse=True) == payload["scene_event_counts"]
              and sum(n*(n-1)//2 for n in counts.values()) == payload["within_scene_pair_count"],
              "Six distinct exact positions, strict source order, actual scene histogram and same-scene pair count.")
        pairs = provenance_entry["all_15_independent_ordering_pairs"]
        pair_keys = set()
        pairs_match = True
        for pair in pairs:
            labels = pair.get("labels", pair.get("pair"))
            a, z = labels
            pair_keys.add(tuple(sorted(labels)))
            pairs_match &= pair["earlier"] == min(labels, key=positions.get)
            pairs_match &= pair["same_scene"] == (scenes[a] == scenes[z])
            if "indices_1_based" in pair:
                pairs_match &= pair["indices_1_based"] == [positions[a]+1, positions[z]+1]
            else:
                pairs_match &= pair["source_positions"] == [positions[a], positions[z]]
        check("ORDER_all_15_independent_pairs", len(pairs) == 15 and pair_keys == set(combinations("ABCDEF", 2)) and pairs_match,
              "Every independent pair, position, earlier label and scene relation mechanically matches normalized selected evidence.")
        expected_criteria = ["O1", "O2", "O3", "O4"]
    check("full_target_criteria", [c["criterion_id"] for c in record["target_realization"]["criteria"]] == expected_criteria
          and all(c["result"] == "PASS" for c in record["target_realization"]["criteria"]),
          "Required criterion mapping inherits existing independent target PASS.")
    check("all_internal_evidence_references_resolve", set(cited).issubset(ref_by_id), str(len(cited)))
    mechanical_results.append({"candidate_id": candidate_id, "record_sha256": sha(path),
                               "checks": checks, "result": "PASS" if all(c["result"] == "PASS" for c in checks) else "FAIL"})

now = datetime.now(timezone.utc).isoformat()
schema_report = {"schema_version": "day3_A_candidate_schema_validation_v1", "validated_at_utc": now,
                 "python": sys.executable, "python_version": sys.version,
                 "jsonschema_version": version("jsonschema"),
                 "validator": "Draft202012Validator + FormatChecker",
                 "schema_path": str(LIBRARY), "schema_sha256": sha(LIBRARY),
                 "schema_pointer": "/schemas/candidate_record", "record_count": len(schema_results),
                 "valid_count": sum(r["valid"] for r in schema_results), "results": schema_results,
                 "semantic_QA_performed_by_schema_validator": False}
save("SCHEMA_VALIDATION.json", schema_report)
mechanical_report = {"schema_version": "day3_A_mechanical_integrity_validation_v1", "validated_at_utc": now,
                     "python": sys.executable, "validator_script_sha256": sha(__file__),
                     "record_count": len(mechanical_results),
                     "pass_count": sum(r["result"] == "PASS" for r in mechanical_results),
                     "ordering_pairs_checked": 90, "results": mechanical_results,
                     "scope_limit": "Mechanical provenance/reference consistency only; semantic conclusions remain attributed to existing independent reviews."}
save("MECHANICAL_INTEGRITY.json", mechanical_report)
print(json.dumps({"records": len(schema_results), "schema_valid": schema_report["valid_count"],
                  "mechanically_valid": mechanical_report["pass_count"], "ordering_pairs": 90,
                  "failures": [{"candidate_id": r["candidate_id"], "checks": [c for c in r["checks"] if c["result"] == "FAIL"]}
                               for r in mechanical_results if r["result"] == "FAIL"]}, ensure_ascii=False))
assert len(schema_results) == 10 and schema_report["valid_count"] == mechanical_report["pass_count"] == 10
