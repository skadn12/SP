"""CPU-only adversarial checks for strict parsing, positions and paired metrics."""
from __future__ import annotations

import copy
import json
import pathlib
import sys

sys.dont_write_bytecode = True
import score_probe as s

HERE = pathlib.Path(__file__).resolve().parent


def main():
    checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append({"name": name, "status": "PASS"})

    labels = [f"I{index}" for index in range(1, 7)]
    canonical = {label: "content_" + label for label in labels}
    display_to_content = {label: canonical[labels[5 - index]] for index, label in enumerate(labels)}
    # Non-lexicographic refs deliberately make text sorting incorrect.
    source = {"root_id": "synthetic-root", "turns": [{"turn_ref": ref, "speaker": "A", "text": "first" + label + "; second" + label}
              for label, ref in zip(labels, ["T9", "T2", "T18", "T1", "T7", "T0"])]}
    schema = {"type": "object", "additionalProperties": False, "required": ["evidence", "answer"], "properties": {
        "evidence": {"type": "array", "minItems": 6, "maxItems": 6, "items": {"type": "object", "additionalProperties": False,
                     "required": ["label", "status", "turn_ref", "quote"], "properties": {
                         "label": {"type": "string", "enum": labels}, "status": {"type": "string", "enum": ["supported", "contradicted", "not_supported", "ambiguous"]},
                         "turn_ref": {"type": "string"}, "quote": {"type": "string"}}}},
        "answer": {"type": "array", "minItems": 6, "maxItems": 6, "uniqueItems": True, "items": {"type": "string", "enum": labels}}}}
    row = {"request_id": "r-control", "candidate_id": "q", "root_id": "synthetic-root", "probe_arm": "control", "cohort": "replication",
           "pair_id": "p", "variant_id": "perm_00", "seed": 12, "question_type": "ordering", "branch": "evidence",
           "display": {"items": dict.fromkeys(labels, "synthetic")}, "json_schema": schema, "max_output_tokens": 1024}
    ordered_display = labels[::-1]
    obj = {"evidence": [{"label": ordered_display[index], "status": "supported", "turn_ref": turn["turn_ref"], "quote": "first" + labels[index]}
                        for index, turn in enumerate(source["turns"])], "answer": ordered_display}
    def raw_for(value, finish="stop", status="RETURNED"):
        return {"status": status, "raw_output": json.dumps(value), "metadata": {"finish_reason": finish}, "response_committed_at_utc": "2026-10-19T01:00:00+00:00"}
    def score(value=None, raw=None, source_value=None, row_value=None):
        return s.score_one(row_value or row, raw or raw_for(value or obj), {"display_to_content": display_to_content}, canonical, labels, source_value or source)

    baseline = score()
    check("valid_remapped_order_exact", baseline["valid"] and baseline["exact"] and baseline["correct_pairs"] == 15)
    check("position_uses_turn_array_not_ref_lexical_order", baseline["citation_observation"]["citation_sorted_display"] == ordered_display)
    check("positions_not_semantic_admission", baseline["citation_observation"]["semantic_support"] == "NOT_REVIEWED")
    reverse = copy.deepcopy(obj)
    reverse["answer"] = labels
    wrong = score(reverse)
    check("reverse_answer_zero_pairs", wrong["valid"] and not wrong["exact"] and wrong["correct_pairs"] == 0)
    check("external_sort_rescues_observable_wrong_answer", wrong["external_citation_sort"]["rescued_wrong_to_right"] and wrong["external_citation_sort"]["exact"] is True)
    swapped = copy.deepcopy(obj)
    swapped["answer"][0], swapped["answer"][1] = swapped["answer"][1], swapped["answer"][0]
    check("one_inversion_fourteen_pairs", score(swapped)["correct_pairs"] == 14)
    for name, mutate in [
        ("duplicate_answer", lambda value: value["answer"].__setitem__(0, value["answer"][1])),
        ("duplicate_evidence_label", lambda value: value["evidence"][0].__setitem__("label", value["evidence"][1]["label"])),
        ("incomplete_evidence", lambda value: value["evidence"].pop()),
        ("extra_json_key", lambda value: value.__setitem__("reasoning", "unrequested")),
        ("unknown_answer_label", lambda value: value["answer"].__setitem__(0, "I7")),
    ]:
        bad = copy.deepcopy(obj)
        mutate(bad)
        result = score(bad)
        check(name + "_invalid_fixed_denominator", not result["valid"] and not result["exact"] and result["correct_pairs"] == 0 and result["total_pairs"] == 15)
    truncated = score(raw=raw_for(obj, finish="length"))
    check("length_finish_invalid_even_valid_json", not truncated["valid"] and truncated["correct_pairs"] == 0)
    check("invalid_external_sort_unknown", truncated["external_citation_sort"]["exact"] is None)
    for status in ("BACKEND_ERROR", "NOT_SUBMITTED", "MISSING"):
        result = score(raw=raw_for(obj, status=status))
        check(status + "_fixed_denominator", not result["valid"] and not result["exact"] and result["correct_pairs"] == 0)
    duplicate_key_raw = raw_for(obj)
    duplicate_key_raw["raw_output"] = duplicate_key_raw["raw_output"][:-1] + ', "answer": ' + json.dumps(ordered_display) + "}"
    check("duplicate_object_key_rejected", not score(raw=duplicate_key_raw)["valid"])
    nonexact = copy.deepcopy(obj)
    nonexact["evidence"][0]["quote"] = "not in source"
    result = score(nonexact)
    check("nonexact_quote_does_not_invalidate_answer_schema", result["valid"] and result["exact"])
    check("nonexact_quote_position_unknown", not result["citation_observation"]["computable"] and result["external_citation_sort"]["exact"] is None)
    empty = copy.deepcopy(obj)
    empty["evidence"][0]["quote"] = ""
    check("empty_quote_position_unknown", not score(empty)["citation_observation"]["computable"])
    missing_ref = copy.deepcopy(obj)
    missing_ref["evidence"][0]["turn_ref"] = "absent"
    check("missing_ref_position_unknown", not score(missing_ref)["citation_observation"]["computable"])
    repeated_source = copy.deepcopy(source)
    repeated_source["turns"][0]["text"] += " firstI1"
    check("multiple_match_position_unknown", not score(source_value=repeated_source)["citation_observation"]["computable"])
    overlap_source = copy.deepcopy(source)
    overlap_source["turns"][0]["text"] = "aaaa"
    overlap_obj = copy.deepcopy(obj)
    overlap_obj["evidence"][0]["quote"] = "aa"
    overlap = score(overlap_obj, source_value=overlap_source)
    check("overlapping_quote_occurrences_counted", overlap["citation_observation"]["entries"][0]["quote_start_offsets"] == [0, 1, 2])
    tied = copy.deepcopy(obj)
    tied["evidence"][1]["turn_ref"] = tied["evidence"][0]["turn_ref"]
    tied["evidence"][1]["quote"] = tied["evidence"][0]["quote"]
    check("tied_positions_unknown", not score(tied)["citation_observation"]["computable"])
    within_source = copy.deepcopy(source)
    within_source["turns"][0]["text"] += " lateritem"
    within_obj = copy.deepcopy(obj)
    within_obj["evidence"][1]["turn_ref"] = within_obj["evidence"][0]["turn_ref"]
    within_obj["evidence"][1]["quote"] = "lateritem"
    check("same_turn_char_offset_orders_distinct_spans", score(within_obj, source_value=within_source)["citation_observation"]["citation_sorted_display"] == ordered_display)
    bad_cites = copy.deepcopy(obj)
    refs = [entry["turn_ref"] for entry in bad_cites["evidence"]]
    quotes = [entry["quote"] for entry in bad_cites["evidence"]]
    for index, entry in enumerate(bad_cites["evidence"]):
        entry["turn_ref"], entry["quote"] = refs[5 - index], quotes[5 - index]
    cite_loss = score(bad_cites)
    check("external_sort_can_harm_original_exact", cite_loss["exact"] and cite_loss["external_citation_sort"]["changed_exact_to_wrong"])
    agg = s.aggregate([baseline, wrong, truncated, result])
    check("aggregate_keeps_four_call_denominator", agg["calls"] == 4 and agg["total_pairs"] == 60 and agg["valid_calls"] == 3)
    check("citation_consistency_keeps_unknowns", agg["citation_consistency_fixed_denominator"] == 4 and agg["citation_consistency_unknown_calls"] == 2)
    check("external_fixed_denominator_and_unknown", agg["external_citation_sort"]["fixed_denominator"] == 4 and agg["external_citation_sort"]["unknown_calls"] == 2)

    records = []
    for cohort, root_count in (("discovery", 3), ("replication", 6)):
        for index in range(root_count):
            root = cohort + str(index)
            for arm in s.ARMS:
                record = copy.deepcopy(wrong if arm == "control" else baseline)
                record.update(root_id=root, candidate_id="q-" + root, cohort=cohort, probe_arm=arm, pair_id=root, request_id=root + "-" + arm)
                records.append(record)
    summary = s.summarize(records)
    check("five_gate_conditions_present", len(summary["primary_endpoint"]["conditions"]) == 5)
    check("positive_three_root_gate_met", summary["primary_endpoint"]["met"] and len(summary["primary_endpoint"]["positive_replication_roots"]) == 6)
    check("root_equal_primary_difference", summary["primary_replication_root_equal_exact_difference"] == 1)
    check("matched_wrong_to_correct_transitions", summary["matched_pairs"]["transitions"]["wrong_to_correct"] == 9)
    no_consistency = copy.deepcopy(records)
    for record in no_consistency:
        if record["probe_arm"] == "treatment":
            record["citation_observation"]["answer_consistent_with_own_citation_positions"] = False
    summary = s.summarize(no_consistency)
    check("consistency_no_improvement_blocks_gate", not summary["primary_endpoint"]["met"] and not summary["primary_endpoint"]["conditions"]["replication_citation_answer_consistent_count_net_positive"])
    invalid_treatment = copy.deepcopy(records)
    invalid_treatment[1]["valid"] = False
    check("single_invalid_blocks_gate", not s.summarize(invalid_treatment)["primary_endpoint"]["met"])
    missing_pair = copy.deepcopy(records[:-1])
    caught = False
    try:
        s.summarize(missing_pair)
    except RuntimeError as error:
        caught = str(error) == "MISSING_PAIRED_ARM"
    check("missing_pair_detected", caught)
    duplicate_pair = copy.deepcopy(records + [records[0]])
    caught = False
    try:
        s.summarize(duplicate_pair)
    except RuntimeError as error:
        caught = str(error) == "DUPLICATE_PAIRED_ARM"
    check("duplicate_pair_detected", caught)
    design_rows = []
    for record in records:
        for index in range(6):
            trial = copy.deepcopy(row)
            trial.update({key: record[key] for key in ("root_id", "candidate_id", "cohort", "probe_arm")})
            trial.update(request_id=record["request_id"] + str(index), pair_id=record["pair_id"] + str(index), variant_id="perm_" + str(index))
            design_rows.append(trial)
    manifest = {"branch": "evidence", "requests": design_rows, "request_count": 108}
    s.validate_design(manifest)
    check("complete_108_design_passes", True)
    changed_design = copy.deepcopy(manifest)
    changed_design["requests"][0]["seed"] += 1
    caught = False
    try:
        s.validate_design(changed_design)
    except RuntimeError as error:
        caught = str(error) == "PAIRED_DESIGN_DIFFERS:seed"
    check("changed_paired_seed_rejected", caught)

    output = {"status": "PASS", "created_at_utc": s.now(), "scope": "CPU synthetic scoring tests only; no actual model ability or semantic validation claims.",
              "checks_passed": len(checks), "checks": checks, "files": [s.bind(__file__), s.bind(s.__file__)]}
    s.save(HERE / "SCORING_CPU_PRECHECK.json", output)
    print(json.dumps({"status": "PASS", "checks_passed": len(checks), "files": output["files"]}))


if __name__ == "__main__":
    main()
