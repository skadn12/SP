"""Offline regression tests for the versioned P0 QA boundary.

Synthetic proof objects below are software fixtures, not generated dataset QA or
claims of semantic quality. This suite makes no model/API/training calls.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import base_qa_pipeline_v1 as qa
except ImportError:
    import base_qa_pipeline_v1 as qa


def fixture_source():
    return {"root_id": "synthetic-root", "turns": [
        {"turn_ref": "T1", "speaker": "fixture_actor", "text": "Fixture premise alpha."},
        {"turn_ref": "T2", "speaker": "fixture_actor", "text": "Fixture premise beta."},
        {"turn_ref": "T3", "speaker": "fixture_actor", "text": "Fixture extra premise."},
    ]}


def evidence(source, *refs):
    return [{"turn_ref": t["turn_ref"], "speaker": t["speaker"], "quote": t["text"]}
            for t in source["turns"] if t["turn_ref"] in refs]


def fixture(kind="single_choice", numeric=False):
    source = fixture_source()
    values = {"c1": "fixture outcome alpha", "c2": "fixture outcome beta", "c3": "fixture outcome gamma"}
    gold = ["c1"] if kind == "single_choice" else ["c1", "c2"] if kind == "multi_select" else list(values)
    if numeric:
        values = {"c1": {"value": "2/5", "unit": "ratio"},
                  "c2": {"value": "1/2", "unit": "ratio"},
                  "c3": {"value": "3/5", "unit": "ratio"}}
    relation = {"relation_id": "fixture-relation", "author_id": "fixture-author",
        "target": "synthetic target", "relation_type": "fixture_relation", "why_ask": "synthetic test only",
        "expected_answer": [values[k] for k in gold], "requires_deterministic_check": False,
        "facts": [
            {"fact_id": "f1", "claim": "fixture alpha", "evidence": evidence(source, "T1"), "evidence_turn_ids": ["T1"]},
            {"fact_id": "f2", "claim": "fixture beta", "evidence": evidence(source, "T2"), "evidence_turn_ids": ["T2"]}],
        "inference_steps": [{"step_id": "s1", "inputs": ["f1", "f2"], "conclusion": "synthetic joint outcome"}],
        "necessity_groups": [
            {"group_id": "g1", "fact_ids": ["f1"], "turn_refs": ["T1"], "why_necessary": "fixture alpha needed", "removal_effect": "fixture target unresolved"},
            {"group_id": "g2", "fact_ids": ["f2"], "turn_refs": ["T2"], "why_necessary": "fixture beta needed", "removal_effect": "fixture target unresolved"}]}
    candidate = {"candidate_id": "fixture-question", "root_id": source["root_id"], "author_id": relation["author_id"],
        "question_type": kind, "question": "Synthetic software fixture target?", "gold": gold,
        "options": {k: "Fixture display " + k for k in values}, "item_analysis": {}}
    for key, val in values.items():
        candidate["item_analysis"][key] = {"answer_value": val, "evidence": evidence(source, "T1", "T2"),
            "rationale": "Synthetic fixture rationale", "status": "supported" if key in gold else "contradicted",
            "error_reason": "Synthetic fixture distractor differs"}
    if kind == "ordering":
        candidate["ordering_edges"] = [{"before": a, "after": b, "evidence": evidence(source, "T1", "T2")}
                                        for a, b in zip(gold, gold[1:])]
    if numeric:
        relation["requires_deterministic_check"] = True
        candidate["deterministic_checks"] = [{"operation": "ratio", "inputs": {
            "numerator": {"fact_id": "f1", "value": "8", "unit": "person_minute"}, "denominator": {"fact_id": "f2", "value": "20", "unit": "person_minute"}},
            "operands": ["numerator", "denominator"], "expected_value": "2/5", "unit": "ratio", "answer_index": 0}]
    candidate["relation_sha256"] = qa.legacy.fingerprint(relation)
    return source, relation, candidate


def review_fixture(source, relation, candidate, packet=None):
    packet = packet or qa.blind_packet(candidate, source)
    blind = {"packet_sha256": qa.legacy.fingerprint(packet), "reviewer_id": "fixture-reviewer",
        "model_id": "offline synthetic review fixture; no model call", "decision": "SOLVED", "ambiguities": [],
        "independent_gold": candidate["gold"][:], "evidence": evidence(source, "T1", "T2"),
        "reasoning_steps": ["Synthetic independent fixture reasoning"]}
    comparison = {"blind_sha256": qa.legacy.fingerprint(blind), "candidate_sha256": qa.legacy.fingerprint(candidate),
        "reviewer_id": blind["reviewer_id"], "decision": "PASS",
        "checks": {k: {"decision": "PASS", "rationale": "Synthetic fixture judgment"} for k in qa.CHECKS},
        "necessity_review": [{"group_id": g["group_id"], "removed_turn_refs": g["turn_refs"],
            "target_underdetermined": True, "single_group_insufficient": True, "no_remaining_bypass": True,
            "rationale": "Synthetic fixture deletion judgment"} for g in relation["necessity_groups"]]}
    return packet, blind, comparison


class MechanicalBoundaryTests(unittest.TestCase):
    def test_all_three_types_accept_consistent_fixtures_only_mechanically(self):
        for kind in ("single_choice", "multi_select", "ordering"):
            with self.subTest(kind=kind):
                s, r, c = fixture(kind)
                result = qa.validate_candidate(c, r, s)
                self.assertTrue(result["mechanical_pass"], result)
                self.assertFalse(result["semantic_quality_proved_by_code"])

    def test_disconnected_fact_rejected(self):
        s, r, _ = fixture()
        r["inference_steps"][0]["inputs"] = ["f1"]
        self.assertTrue(any("all facts" in e for e in qa.validate_relation(r, s)))

    def test_unknown_or_cyclic_forward_step_input_rejected(self):
        s, r, _ = fixture()
        for bad_input in ("missing", "s1"):
            with self.subTest(bad_input=bad_input):
                rr = copy.deepcopy(r)
                rr["inference_steps"][0]["inputs"] = ["f1", bad_input]
                self.assertTrue(qa.validate_relation(rr, s))

    def test_exact_evidence_and_removal_coverage_required(self):
        s, r, _ = fixture()
        r["facts"][0]["evidence"][0]["quote"] = "Fabricated quotation"
        self.assertTrue(qa.validate_relation(r, s))
        s, r, _ = fixture()
        r["necessity_groups"][0]["turn_refs"] = ["T3"]
        self.assertTrue(any("omits cited" in e for e in qa.validate_relation(r, s)))

    def test_gold_semantics_cannot_change_after_relation_freeze(self):
        for kind in ("single_choice", "multi_select", "ordering"):
            s, r, c = fixture(kind)
            c["item_analysis"][c["gold"][0]]["answer_value"] = "different outcome"
            self.assertFalse(qa.validate_candidate(c, r, s)["mechanical_pass"])

    def test_dataset_option_length_audit_excludes_ordering(self):
        _, _, choice = fixture()
        _, _, ordering = fixture("ordering")
        ordering["options"] = {k: "very " * 100 for k in ordering["options"]}
        single_only = qa.dataset_audit([choice])
        mixed = qa.dataset_audit([choice, ordering])
        self.assertEqual(single_only["mean_option_words"], mixed["mean_option_words"])
        self.assertEqual(mixed["type_counts"]["ordering"], 1)
        self.assertEqual(qa.dataset_audit([ordering])["mean_option_words"], {"supported": None, "unsupported": None})

    def test_single_unique_and_multiselect_truth_status(self):
        s, r, c = fixture()
        c["gold"] = ["c1", "c2"]
        self.assertFalse(qa.validate_candidate(c, r, s)["mechanical_pass"])
        s, r, c = fixture("multi_select")
        c["item_analysis"]["c3"]["status"] = "supported"
        self.assertFalse(qa.validate_candidate(c, r, s)["mechanical_pass"])

    def test_order_requires_all_items_unique_full_chain(self):
        s, r, c = fixture("ordering")
        c["gold"].pop()
        self.assertFalse(qa.validate_candidate(c, r, s)["mechanical_pass"])
        s, r, c = fixture("ordering")
        c["ordering_edges"].pop()
        self.assertFalse(qa.validate_candidate(c, r, s)["mechanical_pass"])
        s, r, c = fixture("ordering")
        c["ordering_edges"].append({"before": "c3", "after": "c1", "evidence": evidence(s, "T1")})
        self.assertFalse(qa.validate_candidate(c, r, s)["mechanical_pass"])

    def test_order_rejects_invalid_extra_edges(self):
        for before, after in (("unknown", "c1"), ("c1", "c1")):
            with self.subTest(before=before, after=after):
                s, r, c = fixture("ordering")
                c["ordering_edges"].append({"before": before, "after": after, "evidence": evidence(s, "T1")})
                self.assertFalse(qa.validate_candidate(c, r, s)["mechanical_pass"])

    def test_exact_numeric_certificate_and_answer_value_binding(self):
        s, r, c = fixture(numeric=True)
        result = qa.validate_candidate(c, r, s)
        self.assertTrue(result["mechanical_pass"], result)
        self.assertEqual(result["deterministic_checks"][0]["result"], "2/5")
        self.assertFalse(result["deterministic_checks"][0]["source_value_grounding_verified"])
        c["deterministic_checks"][0]["expected_value"] = "1/2"
        self.assertFalse(qa.validate_candidate(c, r, s)["mechanical_pass"])

    def test_numeric_required_missing_check_rejected(self):
        s, r, c = fixture(numeric=True)
        c.pop("deterministic_checks")
        self.assertFalse(qa.validate_candidate(c, r, s)["mechanical_pass"])

    def test_numeric_target_cannot_disable_deterministic_check_with_flag(self):
        for flag in (False, None, "true"):
            with self.subTest(flag=flag):
                source, relation, candidate = fixture(numeric=True)
                if flag is None:
                    relation.pop("requires_deterministic_check")
                else:
                    relation["requires_deterministic_check"] = flag
                candidate.pop("deterministic_checks")
                candidate["relation_sha256"] = qa.legacy.fingerprint(relation)
                self.assertFalse(qa.validate_candidate(candidate, relation, source)["mechanical_pass"])

    def test_each_numeric_answer_needs_covered_exact_check(self):
        source, relation, candidate = fixture("multi_select", numeric=True)
        self.assertEqual(len(relation["expected_answer"]), 2)
        self.assertEqual(len(candidate["deterministic_checks"]), 1)
        self.assertFalse(qa.validate_candidate(candidate, relation, source)["mechanical_pass"])

    def test_numeric_unit_mismatch_rejected(self):
        source, relation, candidate = fixture(numeric=True)
        candidate["deterministic_checks"][0]["inputs"]["denominator"]["unit"] = "count"
        self.assertFalse(qa.validate_candidate(candidate, relation, source)["mechanical_pass"])

    def test_iso_time_deterministic_duration(self):
        source, relation, candidate = fixture(numeric=True)
        relation["expected_answer"] = [{"value": "79", "unit": "minute"}]
        candidate["item_analysis"]["c1"]["answer_value"] = relation["expected_answer"][0]
        candidate["relation_sha256"] = qa.legacy.fingerprint(relation)
        candidate["deterministic_checks"] = [{"operation": "elapsed_minutes", "inputs": {
            "start": {"fact_id": "f1", "value": "2026-10-27T08:36:00", "unit": "iso_datetime"},
            "end": {"fact_id": "f2", "value": "2026-10-27T09:55:00", "unit": "iso_datetime"}},
            "operands": ["start", "end"], "expected_value": "79", "unit": "minute", "answer_index": 0}]
        result = qa.validate_candidate(candidate, relation, source)
        self.assertTrue(result["mechanical_pass"], result)
        self.assertEqual(result["deterministic_checks"][0]["result"], "79")

    def test_position_shuffle_and_gold_roundtrip_for_all_types(self):
        for kind in ("single_choice", "multi_select", "ordering"):
            for seed in range(7):
                with self.subTest(kind=kind, seed=seed):
                    _, _, c = fixture(kind)
                    public, private = qa.prepare_display(c, seed)
                    self.assertEqual(set(public), {"question_type", "question", "options"})
                    self.assertEqual(set(private["display_to_canonical"].values()), set(c["options"]))
                    self.assertTrue(qa.same_answer(kind, [private["display_to_canonical"][x] for x in private["display_gold"]], c["gold"]))
                    self.assertTrue(qa.diagnose_response(private, private["display_gold"])["answer_correct"])

    def test_shuffle_independent_of_gold(self):
        _, _, c = fixture("multi_select")
        c2 = copy.deepcopy(c)
        c2["gold"] = ["c3"]
        first, _ = qa.prepare_display(c, 12)
        second, _ = qa.prepare_display(c2, 12)
        self.assertEqual(first, second)

    def test_missing_order_and_numeric_option_error_diagnosed_separately(self):
        _, _, c = fixture("ordering")
        _, p = qa.prepare_display(c, 1)
        d = qa.diagnose_response(p, p["display_gold"][:-1])
        self.assertFalse(d["response_valid"])
        self.assertEqual(d["diagnosis"], "ORDERING_MISSING_ITEMS")
        self.assertEqual(len(d["missing_labels"]), 1)
        _, _, c = fixture()
        _, p = qa.prepare_display(c, 1)
        wrong = next(k for k, v in p["display_to_canonical"].items() if v != c["gold"][0])
        d = qa.diagnose_response(p, [wrong], reported_value="2/5", expected_value="2/5")
        self.assertEqual(d["diagnosis"], "VALUE_CORRECT_OPTION_WRONG")
        self.assertTrue(d["legacy_score_unchanged"])


class IndependentReviewTests(unittest.TestCase):
    def test_blind_packet_whitelist_hides_gold_proof_and_author(self):
        s, r, c = fixture()
        c.update({"observed_solver_error": "SECRET", "weakness_feedback": "SECRET"})
        packet = qa.blind_packet(c, s)
        self.assertEqual(set(packet), {"stage", "language", "instruction", "candidate_id", "source", "question_type", "question", "options"})
        for key in ("gold", "item_analysis", "relation", "relation_sha256", "author_id", "weakness_feedback"):
            self.assertNotIn(key, packet)
        self.assertNotIn("SECRET", json.dumps(packet))

    def test_blind_source_view_strips_authoring_provenance(self):
        source, _, candidate = fixture()
        source["continuation_provenance"] = {"expected_answer": "SECRET"}
        source["turns"][0]["author_gold"] = "SECRET"
        packet = qa.blind_packet(candidate, source)
        self.assertNotIn("continuation_provenance", packet["source"])
        self.assertNotIn("author_gold", packet["source"]["turns"][0])
        self.assertNotIn("SECRET", json.dumps(packet))

    def test_malformed_review_shapes_hold_instead_of_crashing(self):
        source, relation, candidate = fixture()
        packet, blind, comparison = review_fixture(source, relation, candidate)
        variants = [([], comparison), (blind, []), (blind, {**comparison, "checks": []}),
            (blind, {**comparison, "checks": {**comparison["checks"], "gold_truth": []}}),
            (blind, {**comparison, "necessity_review": [None, {}]})]
        for bad_blind, bad_comparison in variants:
            with self.subTest(blind_type=type(bad_blind).__name__, comparison=bad_comparison):
                result = qa.validate_review(candidate, relation, source, packet, bad_blind, bad_comparison)
                self.assertEqual(result["decision"], "HOLD")
                self.assertFalse(result["admitted"])

    def test_missing_nonempty_or_nonlist_ambiguities_hold(self):
        for value in (None, ["fixture ambiguity"], "", {}, False):
            source, relation, candidate = fixture()
            packet, blind, comparison = review_fixture(source, relation, candidate)
            if value is None:
                blind.pop("ambiguities")
            else:
                blind["ambiguities"] = value
            comparison["blind_sha256"] = qa.legacy.fingerprint(blind)
            result = qa.validate_review(candidate, relation, source, packet, blind, comparison)
            self.assertEqual(result["decision"], "HOLD")
            self.assertTrue(any("ambiguities" in e for e in result["errors"]))

    def test_blind_options_shuffle_stable_ids_independent_of_gold(self):
        source, _, candidate = fixture()
        packet = qa.blind_packet(candidate, source)
        other = copy.deepcopy(candidate)
        other["gold"] = ["c3"]
        second = qa.blind_packet(other, source)
        self.assertEqual(packet, second)
        self.assertEqual(packet["options"], candidate["options"])
        self.assertNotEqual(list(packet["options"]), list(candidate["options"]))
        self.assertEqual(packet, qa.blind_packet(candidate, source))
        self.assertNotIn("gold", packet)
        self.assertNotIn("display_to_canonical", packet)

    def test_bound_independent_review_can_only_claim_static_pass(self):
        s, r, c = fixture()
        p, b, comp = review_fixture(s, r, c)
        d = qa.validate_review(c, r, s, p, b, comp)
        self.assertEqual(d["decision"], "PASS_STATIC_REVIEW")
        self.assertTrue(d["admitted"])
        self.assertFalse(d["empirical_quality_established"])

    def test_gold_conflict_hold_no_automatic_winner(self):
        s, r, c = fixture()
        p, b, comp = review_fixture(s, r, c)
        b["independent_gold"] = ["c2"]
        comp["blind_sha256"] = qa.legacy.fingerprint(b)
        d = qa.validate_review(c, r, s, p, b, comp)
        self.assertEqual(d["decision"], "HOLD")
        self.assertTrue(any("Gold disagreement" in e for e in d["errors"]))

    def test_no_independent_identity_or_evidence_hold(self):
        for mutation in ("same_author", "missing_model", "no_evidence", "empty_reasoning"):
            s, r, c = fixture()
            p, b, comp = review_fixture(s, r, c)
            if mutation == "same_author": b["reviewer_id"] = c["author_id"]
            if mutation == "missing_model": b.pop("model_id")
            if mutation == "no_evidence": b["evidence"] = []
            if mutation == "empty_reasoning": b["reasoning_steps"] = []
            comp["blind_sha256"] = qa.legacy.fingerprint(b)
            self.assertEqual(qa.validate_review(c, r, s, p, b, comp)["decision"], "HOLD")

    def test_missing_or_failed_necessity_and_semantic_check_hold(self):
        for mutation in ("missing_group", "duplicate_group", "determined", "bypass", "missing_check", "review_risk"):
            s, r, c = fixture()
            p, b, comp = review_fixture(s, r, c)
            if mutation == "missing_group": comp["necessity_review"].pop()
            if mutation == "duplicate_group": comp["necessity_review"][1] = copy.deepcopy(comp["necessity_review"][0])
            if mutation == "determined": comp["necessity_review"][0]["target_underdetermined"] = False
            if mutation == "bypass": comp["necessity_review"][0]["no_remaining_bypass"] = False
            if mutation == "missing_check": comp["checks"].pop("natural_question")
            if mutation == "review_risk": comp["checks"]["no_context_shortcut"]["decision"] = "RISK"
            self.assertEqual(qa.validate_review(c, r, s, p, b, comp)["decision"], "HOLD")

    def test_blind_hash_and_candidate_freeze_binding_drift_hold(self):
        for field in ("packet_sha256", "candidate_sha256", "blind_sha256"):
            s, r, c = fixture()
            p, b, comp = review_fixture(s, r, c)
            (b if field == "packet_sha256" else comp)[field] = "bad-binding"
            self.assertEqual(qa.validate_review(c, r, s, p, b, comp)["decision"], "HOLD")


class OfflineWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source, self.relation, self.candidate = fixture()
        self.sourcepath = self.put("source.json", self.source)
        self.selection = self.put("selection.json", {"roots": [{"root_id": self.source["root_id"], **qa.legacy.file_binding(self.sourcepath)}]})

    def put(self, name, value):
        path = self.base / name
        path.write_text(json.dumps(value))
        return path

    def call(self, *args):
        with redirect_stdout(StringIO()):
            return qa.main(list(map(str, args)))

    def prepare(self, arm="G", suffix="run"):
        run = self.base / suffix
        self.call("prepare", "--selection", self.selection, "--root", self.source["root_id"], "--arm", arm, "--run-dir", run)
        return run

    def freeze_candidate(self, run):
        self.call("freeze-relation", "--run-dir", run, "--input", self.put("relation.json", self.relation))
        self.call("freeze-candidate", "--run-dir", run, "--input", self.put("candidate.json", self.candidate))

    def admit(self, run):
        self.freeze_candidate(run)
        packet = qa.read(run / "BLIND_PACKET.json")
        _, blind, comparison = review_fixture(self.source, self.relation, self.candidate, packet)
        self.call("freeze-blind", "--run-dir", run, "--input", self.put("blind.json", blind))
        self.assertEqual(self.call("finalize", "--run-dir", run, "--input", self.put("comparison.json", comparison)), 0)

    def test_audit_batch_accepts_list_and_wrapped_list_without_overwrite(self):
        for index, value in enumerate(([self.candidate], {"candidates": [self.candidate]})):
            output = self.base / ("audit-" + str(index) + ".json")
            incoming = self.put("audit-input-" + str(index) + ".json", value)
            self.assertEqual(self.call("audit-batch", "--input", incoming, "--output", output), 0)
            result = qa.read(output)
            self.assertEqual(result["model_calls"], 0)
            self.assertEqual(result["input_binding"], qa.legacy.file_binding(incoming))
            self.assertEqual(result["audit"]["type_counts"], {"single_choice": 1})
            with self.assertRaises(FileExistsError):
                self.call("audit-batch", "--input", incoming, "--output", output)
        with self.assertRaises(ValueError):
            self.call("audit-batch", "--input", self.put("bad-audit.json", {}), "--output", self.base / "bad-audit-out.json")

    def test_frozen_qa_receipt_and_display_only_after_pass(self):
        run = self.prepare()
        self.admit(run)
        frozen = qa.read(run / "FROZEN_QA.json")
        self.assertEqual(qa.read(run / "FROZEN_QA_FREEZE.json")["sha256"], qa.legacy.fingerprint(frozen))
        output = self.base / "display"
        self.assertEqual(self.call("prepare-display", "--run-dir", run, "--seed", "fixture-seed", "--output-dir", output), 0)
        public, private = qa.read(output / "PUBLIC_QUESTION.json"), qa.read(output / "PRIVATE_MAPPING.json")
        self.assertEqual(set(public), {"question_type", "question", "options"})
        self.assertNotIn("gold", public)
        self.assertNotIn("relation", public)
        self.assertEqual(private["public_sha256"], qa.legacy.fingerprint(public))
        self.assertTrue(qa.diagnose_response(private, private["display_gold"])["answer_correct"])
        with self.assertRaises(FileExistsError):
            self.call("prepare-display", "--run-dir", run, "--seed", "fixture-seed", "--output-dir", output)

    def test_display_rejects_missing_pass_or_tampered_frozen_question(self):
        missing_run = self.prepare(suffix="missing-pass")
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.call("prepare-display", "--run-dir", missing_run, "--seed", "x", "--output-dir", self.base / "missing-output")
        self.assertFalse((self.base / "missing-output").exists())
        run = self.prepare(suffix="tampered")
        self.admit(run)
        frozen = qa.read(run / "FROZEN_QA.json")
        frozen["candidate"]["question"] = "Altered frozen fixture"
        (run / "FROZEN_QA.json").write_text(json.dumps(frozen))
        with self.assertRaises(ValueError):
            self.call("prepare-display", "--run-dir", run, "--seed", "x", "--output-dir", self.base / "tampered-output")
        self.assertFalse((self.base / "tampered-output").exists())

    def test_comparison_source_once_and_reference_contexts_reconstruct(self):
        run = self.prepare()
        self.freeze_candidate(run)
        packet = qa.read(run / "BLIND_PACKET.json")
        _, blind, _ = review_fixture(self.source, self.relation, self.candidate, packet)
        self.call("freeze-blind", "--run-dir", run, "--input", self.put("blind.json", blind))
        comparison = qa.read(run / "COMPARISON_PACKET.json")
        self.assertEqual(comparison["necessity_check_mode"], "STATIC_COUNTERFACTUAL_REVIEW_NOT_BLIND_ABLATION")
        turns = comparison["source"]["turns"]
        refs = [turn["turn_ref"] for turn in turns]
        by_ref = {turn["turn_ref"]: turn for turn in turns}
        for context in comparison["removal_contexts"]:
            self.assertEqual(set(context), {"group_id", "removed_turn_refs", "remaining_turn_refs", "single_group_turn_refs"})
            removed = context["removed_turn_refs"]
            self.assertEqual(context["remaining_turn_refs"], [ref for ref in refs if ref not in removed])
            self.assertEqual(context["single_group_turn_refs"], [ref for ref in refs if ref in removed])
            self.assertEqual([by_ref[ref] for ref in context["remaining_turn_refs"]], [turn for turn in turns if turn["turn_ref"] not in removed])
            self.assertEqual([by_ref[ref] for ref in context["single_group_turn_refs"]], [turn for turn in turns if turn["turn_ref"] in removed])
        # The candidate/relation contain quotations, but the full turn list is stored once.
        def count_turn_lists(value):
            if isinstance(value, dict):
                return int("turns" in value) + sum(count_turn_lists(v) for v in value.values())
            if isinstance(value, list):
                return sum(count_turn_lists(v) for v in value)
            return 0
        self.assertEqual(count_turn_lists(comparison), 1)

    def test_fg_same_base_packet_and_feedback_disabled(self):
        f, g = self.prepare("F", "F"), self.prepare("G", "G")
        self.assertEqual(qa.read(f / "DISCOVERY_PACKET.json"), qa.read(g / "DISCOVERY_PACKET.json"))
        self.assertFalse(qa.read(f / "INPUT.json")["policy"]["feedback_enabled"])
        self.assertFalse(qa.read(g / "INPUT.json")["policy"]["model_calls_enabled"])
        self.assertNotEqual(qa.read(f / "INPUT.json")["arm"], qa.read(g / "INPUT.json")["arm"])
        with self.assertRaises(ValueError):
            self.call("prepare", "--selection", self.selection, "--root", self.source["root_id"], "--run-dir", self.base / "bad", "--feedback", self.base / "feedback.json")
        self.assertFalse((self.base / "bad").exists())

    def test_no_safe_qa_freezes_stop_without_author_packet(self):
        run = self.prepare()
        val = self.put("no_safe.json", {"status": "NO_SAFE_QA", "reason": "Synthetic fixture has no safe relation"})
        self.call("freeze-relation", "--run-dir", run, "--input", val)
        self.assertTrue((run / "RELATION_FREEZE.json").exists())
        self.assertFalse((run / "AUTHOR_PACKET.json").exists())
        with self.assertRaises(ValueError):
            self.call("freeze-candidate", "--run-dir", run, "--input", self.put("candidate.json", self.candidate))

    def test_cannot_skip_relation_and_blind_stages(self):
        run = self.prepare()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.call("freeze-candidate", "--run-dir", run, "--input", self.put("candidate.json", self.candidate))
        self.assertFalse((run / "CANDIDATE.json").exists())
        self.freeze_candidate(run)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.call("finalize", "--run-dir", run, "--input", self.put("comparison.json", {}))
        self.assertFalse((run / "COMPARISON_REVIEW.json").exists())

    def test_blind_is_frozen_before_gold_comparison_and_cannot_overwrite(self):
        run = self.prepare()
        self.freeze_candidate(run)
        packet = qa.read(run / "BLIND_PACKET.json")
        self.assertNotIn("gold", packet)
        self.assertFalse((run / "COMPARISON_PACKET.json").exists())
        _, blind, comparison = review_fixture(self.source, self.relation, self.candidate, packet)
        blindpath = self.put("blind.json", blind)
        self.call("freeze-blind", "--run-dir", run, "--input", blindpath)
        self.assertTrue((run / "COMPARISON_PACKET.json").exists())
        with self.assertRaises(FileExistsError):
            self.call("freeze-blind", "--run-dir", run, "--input", blindpath)
        code = self.call("finalize", "--run-dir", run, "--input", self.put("comparison.json", comparison))
        self.assertEqual(code, 0)
        self.assertTrue((run / "FROZEN_QA.json").exists())

    def test_unresolved_blind_is_frozen_but_cannot_admit(self):
        run = self.prepare()
        self.freeze_candidate(run)
        packet = qa.read(run / "BLIND_PACKET.json")
        _, blind, comparison = review_fixture(self.source, self.relation, self.candidate, packet)
        blind["decision"] = "UNRESOLVED"
        comparison["blind_sha256"] = qa.legacy.fingerprint(blind)
        self.call("freeze-blind", "--run-dir", run, "--input", self.put("blind.json", blind))
        code = self.call("finalize", "--run-dir", run, "--input", self.put("comparison.json", comparison))
        self.assertEqual(code, 2)
        self.assertEqual(qa.read(run / "DECISION.json")["decision"], "HOLD")
        self.assertFalse((run / "FROZEN_QA.json").exists())

    def test_candidate_tampering_after_freeze_is_rejected(self):
        run = self.prepare()
        self.freeze_candidate(run)
        changed = qa.read(run / "CANDIDATE.json")
        changed["question"] = "Changed after freeze"
        (run / "CANDIDATE.json").write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            self.call("freeze-blind", "--run-dir", run, "--input", self.put("blind.json", {}))
        self.assertFalse((run / "BLIND_SOLUTION.json").exists())

    def test_prepare_input_cannot_switch_source_after_freeze(self):
        run = self.prepare()
        self.freeze_candidate(run)
        new_source = copy.deepcopy(self.source)
        new_source["turns"][2]["text"] = "Changed reviewed source outside selected corpus"
        other = self.put("other-source.json", new_source)
        context = qa.read(run / "INPUT.json")
        context["source_binding"] = qa.legacy.file_binding(other)
        (run / "INPUT.json").write_text(json.dumps(context))
        with self.assertRaises(ValueError):
            self.call("freeze-blind", "--run-dir", run, "--input", self.put("blind.json", {}))
        self.assertFalse((run / "BLIND_SOLUTION.json").exists())


if __name__ == "__main__":
    unittest.main()
