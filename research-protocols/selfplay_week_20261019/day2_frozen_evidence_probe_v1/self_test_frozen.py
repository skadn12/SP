"""CPU boundary checks using synthetic citations, answers and costs only."""
from __future__ import annotations
import copy
import itertools
import json
import pathlib
import tempfile

import score_frozen as s


def main():
    tests = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        tests.append(name)
    def rejected(name, function):
        try:
            function()
        except RuntimeError:
            tests.append(name)
            return
        raise AssertionError(name)

    source = {"root_id": "synthetic-a", "turns": [
        {"turn_ref": "Z99", "text": "qq1 first qq2 next qq3 third"},
        {"turn_ref": "A00", "text": "qq4 fourth qq5 fifth qq6 last"}]}
    frozen = {"evidence": [{"label": f"I{x}", "turn_ref": "Z99" if x <= 3 else "A00", "quote": f"qq{x}", "status": "supported"} for x in range(1, 7)]}
    observation = s.locate_citations(frozen, source)
    check("source_array_order_beats_lexical_turn_ref", observation["citation_sorted_display"] == list(s.LABELS))
    check("same_turn_offsets_and_offset_zero", [x["quote_start_offset"] for x in observation["positions"]] == [0, 10, 19, 0, 11, 21])
    bad = copy.deepcopy(frozen)
    bad["evidence"][0]["quote"] = ""
    rejected("empty_quote_rejected", lambda: s.locate_citations(bad, source))
    bad = copy.deepcopy(frozen)
    bad["evidence"][0]["quote"] = "q"
    rejected("nonunique_overlapping_quote_rejected", lambda: s.locate_citations(bad, source))
    bad = copy.deepcopy(frozen)
    bad["evidence"][1]["quote"] = "qq1 first"
    rejected("same_start_offset_tie_rejected", lambda: s.locate_citations(bad, source))
    bad = copy.deepcopy(frozen)
    bad["citation_sorted_display"] = list(reversed(s.LABELS))
    rejected("stored_reference_must_recompute", lambda: s.locate_citations(bad, source))
    raw = {"status": "RETURNED", "raw_output": "<box>I1,I2,I3,I4,I5,I6</box>", "metadata": {"finish_reason": "stop"}}
    check("all_720_finite_permutations_valid", all(s.parse_strict(dict(raw, raw_output="<box>" + ",".join(p) + "</box>"))[1] for p in itertools.permutations(s.LABELS)))
    bad_outputs = [" <box>I1,I2,I3,I4,I5,I6</box>", "<box>I1,I2,I3,I4,I5,I6</box>\n", "<box>I1,I1,I2,I3,I4,I5</box>", "<box>I1,I2,I3,I4,I5</box>", "<box>I1,I2,I3,I4,I5,I7</box>", "<box>I1, I2,I3,I4,I5,I6</box>"]
    check("strict_box_rejects_variants", not any(s.parse_strict(dict(raw, raw_output=x))[1] for x in bad_outputs))
    check("length_finish_invalid", not s.parse_strict(dict(raw, metadata={"finish_reason": "length"}))[1])
    check("backend_error_invalid", not s.parse_strict(dict(raw, status="BACKEND_ERROR"))[1])
    row = {"request_id": "a__r00__freeze_quotes", "candidate_id": "synthetic-question", "root_id": "synthetic-a", "probe_arm": s.ARMS[0], "pair_id": "a__r00", "frozen_record_id": "a", "variant_id": "perm_00", "repeat_id": "r00", "seed": 111}
    canonical = {x: "content" + x for x in s.LABELS}
    variant = {"display_to_content": canonical}
    scored = s.score_one(row, raw, variant, canonical, list(reversed(s.LABELS)), observation)
    check("citation_and_gold_are_distinct_targets", scored["citation_exact"] and not scored["gold_exact"] and scored["citation_correct_pairs"] == 15 and scored["gold_correct_pairs"] == 0)
    invalid = s.score_one(row, dict(raw, metadata={"finish_reason": "length"}), variant, canonical, list(s.LABELS), observation)
    check("invalid_fixed_denominator_both_references_zero", not invalid["valid"] and not invalid["citation_exact"] and not invalid["gold_exact"] and invalid["citation_correct_pairs"] == invalid["gold_correct_pairs"] == 0 and invalid["total_pairs"] == 15)
    check("pair_count_adjacent_swap", s.correct_pairs(["I2", "I1", "I3", "I4", "I5", "I6"], list(s.LABELS)) == 14)

    # Root A contributes two frozen records, root B one; root equal differs from pooled.
    records = []
    for root, frozen_id, control_ok, treatment_ok in [("a", "a1", False, True), ("a", "a2", False, True), ("b", "b1", True, False)]:
        for arm, correct in zip(s.ARMS, [control_ok, treatment_ok]):
            item = copy.deepcopy(scored)
            item.update({"root_id": root, "frozen_record_id": frozen_id, "pair_id": frozen_id + "__r00", "request_id": frozen_id + "__" + arm,
                         "candidate_id": root, "probe_arm": arm, "citation_exact": correct, "gold_exact": correct,
                         "citation_correct_pairs": 15 * correct, "gold_correct_pairs": 15 * correct})
            records.append(item)
    summary = s.summarize(records)
    check("primary_root_equal_not_pooled", summary["primary_root_equal_citation_exact_difference"] == 0 and summary["by_arm"][s.ARMS[1]]["citation_exact_rate"] > summary["by_arm"][s.ARMS[0]]["citation_exact_rate"])
    check("matched_pair_transitions", summary["matched_pairs"]["citation_transitions"] == {"correct_to_correct": 0, "correct_to_wrong": 1, "wrong_to_correct": 2, "wrong_to_wrong": 0})
    check("no_old_working_gate_or_cause_admission", summary["primary_endpoint"]["working_support_gate"] is None and not summary["primary_endpoint"]["cause_auto_published"])
    bad_records = copy.deepcopy(records)
    bad_records[1]["seed"] = 112
    rejected("unpaired_seed_rejected", lambda: s.paired_contrasts(bad_records))

    with tempfile.TemporaryDirectory() as temp:
        run = pathlib.Path(temp)
        manifest = {"requests": [{"request_id": "c", "probe_arm": s.ARMS[0]}, {"request_id": "t", "probe_arm": s.ARMS[1]}, {"request_id": "u", "probe_arm": s.ARMS[0]}]}
        raws = {"c": dict(raw, metadata={"prompt_token_count": 12, "completion_token_count": 4}), "t": dict(raw, metadata={}), "u": {"status": "NOT_SUBMITTED"}}
        cost = s.runtime_cost(run, manifest, raws)
        check("usage_unknown_not_zero_or_double_counted", cost["requests_with_unknown_usage"] == 1 and cost["by_arm"][s.ARMS[1]]["requests_with_unknown_usage"] == 1 and cost["known_total_tokens"] == 16)
        check("not_submitted_retains_planned_cost_denominator", cost["planned_calls"] == 3 and cost["not_submitted_calls"] == 1 and cost["submitted_attempts"] == 2)
        (run / "file").write_text("a")
        binding = s.bind(run / "file")
        (run / "file").write_text("b")
        rejected("changed_binding_rejected", lambda: s.verify_bindings([binding]))

    output = {"status": "PASS", "synthetic_only": True, "private_gold_read": False, "checks": len(tests), "tests": tests,
              "scoring_code_binding": s.bind(s.__file__), "self_test_code_binding": s.bind(__file__), "created_at_utc": s.now()}
    s.save(pathlib.Path(__file__).parent / "SCORING_CPU_PRECHECK.json", output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
