"""Offline synthetic answer/mapping/strict-review regression tests.

The inline scope is a low-level file-contract fixture, never a formal material
grant.  No Reviewer, Solver, network API, or real-source generation is invoked.
"""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    from . import base_qa_answers_v2_2_1 as a
    from . import base_qa_option_freeze_v2_2_1 as qa
except ImportError:
    import base_qa_answers_v2_2_1 as a
    import base_qa_option_freeze_v2_2_1 as qa


def candidate(qtype="single_choice"):
    base = {"candidate_id": "synthetic-q", "author_id": "synthetic-author",
            "question_type": qtype, "question": "Which arrangement is supported?",
            "private_certificate": {"fixture_only": True}}
    if qtype == "single_choice":
        base.update(options={"o3": "Review, installation, then opening", "o1": "Installation, review, then opening",
                             "o2": "Installation, opening, then review"},
                    item_analysis={"o3": {"semantic_value": ["review", "installation", "opening"]},
                                   "o1": {"semantic_value": ["installation", "review", "opening"]},
                                   "o2": {"semantic_value": ["installation", "opening", "review"]}},
                    semantic_answer=a.semantic_answer("sequence", ["installation", "review", "opening"]),
                    gold_answer={"kind": "single_option", "option_id": "o1"})
    else:
        base.update(options={"o3": "Opening", "o1": "Installation", "o2": "Review"},
                    item_analysis={"o3": {"semantic_value": "opening"}, "o1": {"semantic_value": "installation"},
                                   "o2": {"semantic_value": "review"}})
        if qtype == "multi_select":
            base.update(semantic_answer=a.semantic_answer("set", ["installation", "review"]),
                        gold_answer={"kind": "option_set", "option_ids": ["o2", "o1"]})
        elif qtype == "ordering":
            base.update(semantic_answer=a.semantic_answer("sequence", ["installation", "review", "opening"]),
                        gold_answer={"kind": "option_sequence", "option_ids": ["o1", "o2", "o3"]})
        elif qtype == "partial_order":
            base.update(semantic_answer=a.semantic_answer("relation_set", [["installation", "review"],
                                                                          ["review", "opening"]]),
                        gold_answer={"kind": "partial_order", "pairs": [["o1", "o2"], ["o2", "o3"]]})
    return base


class TypedAnswersV221Tests(unittest.TestCase):
    def test_single_choice_can_select_whole_sequence_without_nesting(self):
        value = candidate()
        checked = a.validate_candidate_answer(value)
        self.assertEqual(checked["derived_semantic"], value["semantic_answer"])
        self.assertEqual(checked["canonical_gold"], {"kind": "single_option", "option_id": "o1"})

    def test_single_choice_can_select_object_or_local_scalar(self):
        for kind, values in (("object", [{"person": "Ann", "room": "Hall"}, {"person": "Ann", "room": "Studio"}]),
                             ("scalar", [True, 1]), ("scalar", [61, 71])):
            with self.subTest(kind=kind, values=values):
                value = candidate()
                value["options"] = {"o1": "First arrangement", "o2": "Second arrangement"}
                value["item_analysis"] = {key: {"semantic_value": item} for key, item in zip(value["options"], values)}
                value["semantic_answer"] = a.semantic_answer(kind, values[0])
                self.assertEqual(a.validate_candidate_answer(value)["derived_semantic"], value["semantic_answer"])

    def test_bool_int_float_remain_typed(self):
        self.assertFalse(a.typed_equal(True, 1))
        self.assertFalse(a.typed_equal(1, 1.0))
        self.assertFalse(a.typed_equal({"value": False}, {"value": 0}))
        self.assertEqual(len(a.semantic_answer("set", [True, 1, 1.0])["value"]), 3)

    def test_set_order_canonical_but_sequence_meaning_preserved(self):
        self.assertEqual(a.semantic_answer("set", ["b", "a"]), a.semantic_answer("set", ["a", "b"]))
        self.assertNotEqual(a.semantic_answer("sequence", ["b", "a"]), a.semantic_answer("sequence", ["a", "b"]))

    def test_partial_edges_compare_by_closure(self):
        reduced = [["a", "b"], ["b", "c"]]
        full = [["a", "c"], ["b", "c"], ["a", "b"]]
        self.assertEqual(a.semantic_answer("relation_set", reduced), a.semantic_answer("relation_set", full))
        self.assertEqual(a.canonical_answer({"kind": "partial_order", "pairs": reduced}, "partial_order", ["a", "b", "c"]),
                         a.canonical_answer({"kind": "partial_order", "pairs": full}, "partial_order", ["a", "b", "c"]))

    def test_partial_order_rejects_unknown_self_cycle_and_bad_pairs(self):
        for pairs in ([["a", "d"]], [["a", "a"]], [["a", "b"], ["b", "a"]],
                      [["a", "b"], ["b", "c"], ["c", "a"]], [["a"]], [["a", "b"], ["a", "b"]]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                a.canonical_answer({"kind": "partial_order", "pairs": pairs}, "partial_order", ["a", "b", "c"])

    def test_partial_order_supports_incomparable_items(self):
        value = candidate("partial_order")
        value["gold_answer"]["pairs"] = [["o1", "o3"]]
        value["semantic_answer"] = a.semantic_answer("relation_set", [["installation", "opening"]])
        checked = a.validate_candidate_answer(value)
        self.assertEqual(checked["canonical_gold"]["pairs"], [["o1", "o3"]])

    def test_empty_partial_relation_is_explicit_not_missing(self):
        self.assertEqual(a.canonical_answer({"kind": "partial_order", "pairs": []}, "partial_order", ["a", "b"]),
                         {"kind": "partial_order", "pairs": []})
        with self.assertRaises(ValueError):
            a.canonical_answer({"kind": "partial_order"}, "partial_order", ["a", "b"])

    def test_complete_order_rejects_partial_coverage(self):
        with self.assertRaisesRegex(ValueError, "COMPLETE_ORDER_COVERAGE"):
            a.canonical_answer({"kind": "option_sequence", "option_ids": ["a", "b"]}, "ordering", ["a", "b", "c"])

    def test_exact_schema_legacy_and_extra_gold_rejected(self):
        value = candidate()
        value["gold"] = ["o1"]
        with self.assertRaisesRegex(ValueError, "LEGACY_GOLD"):
            a.validate_candidate_answer(value)
        for answer in ({"kind": "single_option", "option_id": "o1", "extra": 1},
                       {"kind": "option_set", "option_ids": ["o1"]}, ["o1"]):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                a.canonical_answer(answer, "single_choice", ["o1", "o2"])

    def test_wrong_semantic_gold_and_candidate_target_conflict_rejected(self):
        value = candidate()
        value["gold_answer"]["option_id"] = "o2"
        with self.assertRaises(ValueError):
            a.validate_candidate_answer(value)
        value = candidate()
        with self.assertRaisesRegex(ValueError, "TARGET_SEMANTIC_MISMATCH"):
            a.validate_candidate_answer(value, a.semantic_answer("sequence", ["opening", "installation", "review"]))

    def test_semantically_duplicate_single_options_are_ambiguous(self):
        value = candidate()
        value["semantic_answer"] = a.semantic_answer("set", ["a", "b"])
        value["item_analysis"]["o1"]["semantic_value"] = ["a", "b"]
        value["item_analysis"]["o2"]["semantic_value"] = ["b", "a"]
        with self.assertRaisesRegex(ValueError, "NOT_UNIQUE"):
            a.validate_candidate_answer(value)

    def test_multi_rejects_wrong_kind_and_duplicate_selected_ids(self):
        value = candidate("multi_select")
        value["semantic_answer"] = a.semantic_answer("sequence", ["installation", "review"])
        with self.assertRaisesRegex(ValueError, "SEMANTIC_KIND_MISMATCH"):
            a.validate_candidate_answer(value)
        value = candidate("multi_select")
        value["gold_answer"]["option_ids"] = ["o1", "o1"]
        with self.assertRaises(ValueError):
            a.validate_candidate_answer(value)

    def test_non_json_nonfinite_and_wrong_semantic_types_rejected(self):
        invalid = [("scalar", []), ("object", []), ("set", ["a", "a"]), ("sequence", "a"),
                   ("scalar", float("nan")), ("object", {1: "a"}), ("set", {"a"}), ("scalar", (1, 2))]
        for kind, value in invalid:
            with self.subTest(kind=kind, value=value), self.assertRaises(ValueError):
                a.semantic_answer(kind, value)

    def test_normalization_idempotent_nonmutating_and_unordered_gold_canonical(self):
        for qtype in a.QUESTION_TYPES:
            with self.subTest(qtype=qtype):
                value = candidate(qtype)
                before = copy.deepcopy(value)
                normalized = a.normalize_candidate(value)
                self.assertEqual(value, before)
                self.assertEqual(normalized, a.normalize_candidate(normalized))
                self.assertEqual(list(normalized["options"]), ["o1", "o2", "o3"])
                self.assertEqual(list(normalized["item_analysis"]), ["o1", "o2", "o3"])


class FrozenMappingV221Tests(unittest.TestCase):
    def setUp(self):
        self.source = {"root_id": "synthetic-fixture", "source_id": "synthetic-source", "turns": [
            {"turn_ref": "S1:T1", "speaker": "A", "text": "Installation must precede review."},
            {"turn_ref": "S2:T1", "speaker": "B", "text": "Review must precede opening."}]}
        self.scope = {"source_id": "synthetic-source", "source_sha256": qa.fingerprint(self.source),
                      "source_admission_complete": True, "fixture_only": True,
                      "turns": [{"turn_id": "S1:T1", "formal_qa_allowed": True},
                                {"turn_id": "S2:T1", "formal_qa_allowed": True}]}

    def freeze(self, path, value=None, scope=None, **kwargs):
        return qa.freeze_bundle(path, value or candidate(), self.source, scope or self.scope,
                                reviewer_id="synthetic-reviewer", context_id="fresh-synthetic-context",
                                issuer_id="synthetic-orchestrator", receipt_id="synthetic-receipt", **kwargs)

    def response(self, path, decision="SOLVED"):
        bundle = qa.load_bundle(path)
        value = bundle["candidate"]
        answer = a.map_answer(value["gold_answer"], value["question_type"],
                              bundle["render_contract"]["canonical_to_display"])
        return {"schema_version": qa.BLIND_SCHEMA, "candidate_id": value["candidate_id"],
                "packet_sha256": bundle["packet"]["packet_sha256"],
                "source_scope_sha256": bundle["packet"]["source_scope_sha256"],
                "reviewer_id": "synthetic-reviewer", "model_id": "offline-fixture-no-model",
                "reviewer_mode": "same_model_blind", "decision": decision,
                "independent_answer": answer if decision == "SOLVED" else None,
                "evidence": [{"turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "quote": turn["text"]}
                             for turn in self.source["turns"]],
                "reasoning_steps": ["Synthetic schema fixture; no Reviewer was run."],
                "ambiguities": [] if decision == "SOLVED" else ["Fixture unresolved event identity."],
                "source_limitations": ["Synthetic file-contract fixture, not formal material authorization."],
                "isolation_receipt": qa.reviewer_receipt_projection(path)}

    def raw(self, response):
        return json.dumps(response, ensure_ascii=False).encode("utf-8")

    def test_four_task_forms_roundtrip_reviewer_and_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            for qtype in sorted(a.QUESTION_TYPES):
                with self.subTest(qtype=qtype):
                    path = Path(tmp, qtype)
                    self.freeze(path, candidate(qtype))
                    response = self.response(path)
                    self.assertEqual(qa.freeze_blind_response(path, self.raw(response))["status"], "VALID")
                    disposition = qa.blind_disposition(path)
                    self.assertEqual(disposition["decision"], "REVIEWER_MATCH", disposition)
                    score = qa.score_frozen_answer(path, response["independent_answer"])
                    self.assertTrue(score["correct"], score)
                    self.assertEqual(score["canonical_answer"], disposition["author_gold"])
                    self.assertEqual(score["mapping_sha256"], disposition["mapping_sha256"])

    def test_all_option_insertions_and_item_orderings_preserve_hash_and_labels(self):
        expected = None
        with tempfile.TemporaryDirectory() as tmp:
            for index, ids in enumerate(itertools.permutations(["o1", "o2", "o3"])):
                value = candidate("multi_select")
                value["options"] = {key: value["options"][key] for key in ids}
                value["item_analysis"] = {key: value["item_analysis"][key] for key in ids[::-1]}
                if index % 2:
                    value["gold_answer"]["option_ids"].reverse()
                    value["semantic_answer"]["value"].reverse()
                path = Path(tmp, str(index))
                envelope = self.freeze(path, value)
                current = (envelope["candidate_sha256"], qa.frozen_packet(path), qa.frozen_mapping(path))
                expected = current if expected is None else expected
                self.assertEqual(current, expected)

    def test_partial_reduced_full_edges_have_same_freeze_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = candidate("partial_order")
            first = self.freeze(Path(tmp, "one"), value)
            value["gold_answer"]["pairs"].append(["o1", "o3"])
            second = self.freeze(Path(tmp, "two"), value)
            self.assertEqual(first["candidate_sha256"], second["candidate_sha256"])
            self.assertEqual(qa.frozen_packet(Path(tmp, "one")), qa.frozen_packet(Path(tmp, "two")))

    def test_reviewer_reduced_partial_edges_match_expanded_author(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp, candidate("partial_order"))
            response = self.response(tmp)
            reverse = qa.load_bundle(tmp)["render_contract"]["canonical_to_display"]
            response["independent_answer"] = {"kind": "partial_order", "pairs": [
                [reverse["o1"], reverse["o2"]], [reverse["o2"], reverse["o3"]]]}
            self.assertEqual(qa.freeze_blind_response(tmp, self.raw(response))["status"], "VALID")
            self.assertEqual(qa.blind_disposition(tmp)["decision"], "REVIEWER_MATCH")
            self.assertTrue(qa.score_frozen_answer(tmp, response["independent_answer"])["correct"])

    def test_set_answer_order_ignored_but_sequence_order_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            for qtype in ("multi_select", "ordering"):
                path = Path(tmp, qtype)
                self.freeze(path, candidate(qtype))
                answer = self.response(path)["independent_answer"]
                answer["option_ids"].reverse()
                score = qa.score_frozen_answer(path, answer)
                self.assertEqual(score["status"], "VALID")
                self.assertEqual(score["correct"], qtype == "multi_select")

    def test_gold_change_is_private_and_does_not_bias_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp, "one"), Path(tmp, "two")
            self.freeze(one)
            value = candidate()
            value["gold_answer"]["option_id"] = "o2"
            value["semantic_answer"] = a.semantic_answer("sequence", value["item_analysis"]["o2"]["semantic_value"])
            self.freeze(two, value)
            self.assertEqual(qa.frozen_packet(one), qa.frozen_packet(two))
            self.assertEqual(qa.frozen_mapping(one), qa.frozen_mapping(two))
            self.assertNotEqual(qa.load_bundle(one)["envelope"]["candidate_sha256"],
                                qa.load_bundle(two)["envelope"]["candidate_sha256"])

    def test_whole_sequence_single_and_ordering_have_same_semantics_different_task_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp, "one"), Path(tmp, "two")
            self.freeze(one, candidate("single_choice"))
            self.freeze(two, candidate("ordering"))
            first, second = qa.load_bundle(one)["candidate"], qa.load_bundle(two)["candidate"]
            self.assertEqual(first["semantic_answer"], second["semantic_answer"])
            self.assertNotEqual(first["gold_answer"], second["gold_answer"])

    def test_map_getters_are_copies_and_loading_never_rerenders(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            before = qa.frozen_mapping(tmp)
            local = qa.frozen_mapping(tmp)
            local["A"] = "invented"
            with mock.patch.object(qa, "_render_once", side_effect=AssertionError("unexpected rerender")):
                self.assertEqual(qa.frozen_mapping(tmp), before)
                self.assertEqual(qa.load_bundle(tmp)["render_contract"]["display_to_canonical"], before)

    def test_public_packet_excludes_private_answers_ids_and_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            payload = qa.frozen_packet(tmp)["packet"]
            self.assertEqual(set(payload), {"stage", "instruction", "candidate_id", "reviewer_mode", "source",
                                           "source_scope", "question_type", "question", "options"})
            self.assertFalse(set(candidate()["options"]) & set(payload["options"]))
            self.assertNotIn("semantic_answer", payload)
            self.assertNotIn("gold_answer", payload)
            self.assertNotIn("item_analysis", payload)
            self.assertNotIn("private_certificate", payload)

    def test_new_process_reads_same_authoritative_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            code = ("import json,sys; import base_qa_option_freeze_v2_2_1 as q; "
                    "print(json.dumps({'packet':q.frozen_packet(sys.argv[1]),'mapping':q.frozen_mapping(sys.argv[1])}))")
            result = subprocess.run([sys.executable, "-c", code, tmp], cwd=Path(qa.__file__).parent,
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), {"packet": qa.frozen_packet(tmp), "mapping": qa.frozen_mapping(tmp)})

    def test_implementation_binding_includes_new_and_old_dependencies_not_tests(self):
        names = {Path(row["path"]).name for row in qa.implementation_bindings()["files"]}
        for name in ("base_qa_answers_v2_2_1.py", "base_qa_pipeline_v2_2_1.py", "base_qa_schema_v2_2_1.json",
                     "base_qa_pipeline_v2_2.py", "base_qa_pipeline_v2_1.py", "base_qa_pipeline_v1.py"):
            self.assertIn(name, names)
        self.assertFalse(any(name.startswith("test_") for name in names))

    def test_implementation_extra_binding_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            extra = Path(tmp, "bound.txt")
            extra.write_text("fixture-v1")
            path = Path(tmp, "bundle")
            self.freeze(path, extra_paths=[extra])
            extra.write_text("fixture-v2")
            with self.assertRaisesRegex(qa.FreezeError, "IMPLEMENTATION_DRIFT"):
                qa.load_bundle(path)

    def test_each_frozen_file_and_envelope_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in [*qa._FILES, qa.ENVELOPE]:
                with self.subTest(name=name):
                    path = Path(tmp, name)
                    self.freeze(path)
                    target = path / name
                    target.write_bytes(target.read_bytes() + b" ")
                    if name == qa.ENVELOPE:
                        # JSON whitespace is semantically harmless for the outer
                        # envelope; corrupt a bound value rather than require a
                        # nonexistent external signature on that envelope file.
                        value = json.loads(target.read_bytes())
                        value["packet_sha256"] = "0" * 64
                        target.write_text(json.dumps(value))
                    with self.assertRaises(qa.FreezeError):
                        qa.load_bundle(path)
                    self.assertEqual(qa.score_frozen_answer(path, {"kind": "single_option", "option_id": "A"})["status"], "HOLD")

    def test_source_input_and_scope_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            source = copy.deepcopy(self.source)
            source["turns"][0]["text"] += " altered"
            with self.assertRaisesRegex(qa.FreezeError, "SOURCE_INPUT_DRIFT"):
                qa.load_bundle(tmp, source=source)
            scope = dict(self.scope, fixture_only=False)
            with self.assertRaisesRegex(qa.FreezeError, "SOURCE_SCOPE_INPUT_DRIFT"):
                qa.load_bundle(tmp, source_scope=scope)

    def test_raw_malformed_missing_type_extra_utf8_duplicates_all_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in ("missing", "type", "extra", "utf8", "duplicate", "unparseable", "legacy_answer", "overflow"):
                with self.subTest(case=case):
                    path = Path(tmp, case)
                    self.freeze(path)
                    response = self.response(path)
                    if case == "missing":
                        del response["independent_answer"]
                    elif case == "type":
                        response["independent_answer"] = ["A"]
                    elif case == "extra":
                        response["unexpected"] = True
                    elif case == "legacy_answer":
                        response["independent_gold"] = ["A"]
                        del response["independent_answer"]
                    raw = self.raw(response)
                    if case == "utf8":
                        raw = b"\xff"
                    elif case == "duplicate":
                        raw = raw[:-1] + b',"candidate_id":"duplicate"}'
                    elif case == "unparseable":
                        raw = b"{"
                    elif case == "overflow":
                        raw = b'{"value":1e9999}'
                    self.assertEqual(qa.freeze_blind_response(path, raw)["status"], "INVALID")
                    self.assertEqual((path / "BLIND_RAW.bin").read_bytes(), raw)
                    disposition = qa.blind_disposition(path)
                    self.assertEqual(disposition["decision"], "HOLD")
                    self.assertFalse(disposition["gold_compared"])

    def test_reviewer_gold_conflict_frozen_before_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            response = self.response(tmp)
            mapping = qa.load_bundle(tmp)["render_contract"]["canonical_to_display"]
            response["independent_answer"] = {"kind": "single_option", "option_id": mapping["o2"]}
            raw = self.raw(response)
            parsed = qa.freeze_blind_response(tmp, raw)
            self.assertEqual(parsed["status"], "VALID")
            self.assertEqual(Path(tmp, "BLIND_RAW.bin").read_bytes(), raw)
            self.assertEqual(qa.blind_disposition(tmp)["errors"], ["REVIEWER_GOLD_DISAGREEMENT"])
            self.assertTrue(qa.blind_disposition(tmp)["gold_compared"])
            self.assertFalse(qa.score_frozen_answer(tmp, response["independent_answer"])["correct"])

    def test_reviewer_hold_is_valid_but_not_gold_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            response = self.response(tmp, "HOLD")
            self.assertEqual(qa.freeze_blind_response(tmp, self.raw(response))["status"], "VALID")
            self.assertEqual(qa.blind_disposition(tmp), {"decision": "HOLD", "errors": ["REVIEWER_HOLD"], "gold_compared": False})

    def test_bad_identity_hash_receipt_ambiguity_evidence_and_admission_rejected(self):
        cases = ("reviewer", "packet", "scope", "receipt", "ambiguity", "evidence", "admission", "mode")
        with tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                with self.subTest(case=case):
                    path = Path(tmp, case)
                    scope = copy.deepcopy(self.scope)
                    if case == "admission":
                        scope["source_admission_complete"] = False
                    self.freeze(path, scope=scope)
                    response = self.response(path)
                    if case == "reviewer":
                        response["reviewer_id"] = "synthetic-author"
                    elif case == "packet":
                        response["packet_sha256"] = "0" * 64
                    elif case == "scope":
                        response["source_scope_sha256"] = "0" * 64
                    elif case == "receipt":
                        response["isolation_receipt"]["author_material_visible"] = True
                    elif case == "ambiguity":
                        response["ambiguities"] = ["Unresolved event identity."]
                    elif case == "evidence":
                        response["evidence"][0]["quote"] = "Invented quotation."
                    elif case == "mode":
                        response["reviewer_mode"] = "independent_model"
                    self.assertEqual(qa.freeze_blind_response(path, self.raw(response))["status"], "INVALID")

    def test_no_refreeze_no_response_retry_no_partial_directory_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            with self.assertRaises(qa.FreezeError):
                self.freeze(tmp)
            qa.freeze_blind_response(tmp, b"{")
            with self.assertRaises(FileExistsError):
                qa.freeze_blind_response(tmp, self.raw(self.response(tmp)))
            self.assertEqual(Path(tmp, "BLIND_RAW.bin").read_bytes(), b"{")
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "PARTIAL").write_text("previous failed attempt")
            with self.assertRaises(qa.FreezeError):
                self.freeze(tmp)

    def test_invalid_typed_scoring_is_hold_not_solver_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            for answer in (["A"], {"kind": "single_option", "option_id": "unknown"},
                           {"kind": "option_set", "option_ids": ["A"]}):
                with self.subTest(answer=answer):
                    score = qa.score_frozen_answer(tmp, answer)
                    self.assertEqual(score["status"], "HOLD")
                    self.assertIsNone(score["correct"])

    def test_blind_freeze_after_context_drift_keeps_received_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            raw = self.raw(self.response(tmp))
            path = Path(tmp, qa.CONTRACT)
            path.write_bytes(path.read_bytes() + b" ")
            parsed = qa.freeze_blind_response(tmp, raw)
            self.assertEqual(parsed["status"], "INVALID")
            self.assertEqual(Path(tmp, "BLIND_RAW.bin").read_bytes(), raw)

    def test_blind_raw_parse_and_solution_tamper_never_compares_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("BLIND_RAW.bin", "BLIND_RAW_FREEZE.json", "BLIND_PARSE.json", "BLIND_SOLUTION.json", "BLIND_SOLUTION_FREEZE.json"):
                with self.subTest(name=name):
                    path = Path(tmp, name)
                    self.freeze(path)
                    qa.freeze_blind_response(path, self.raw(self.response(path)))
                    target = path / name
                    target.write_bytes(target.read_bytes() + b" ")
                    if name.endswith("FREEZE.json"):
                        value = json.loads(target.read_bytes())
                        value["sha256"] = "0" * 64
                        target.write_text(json.dumps(value))
                    disposition = qa.blind_disposition(path)
                    self.assertEqual(disposition["decision"], "HOLD")
                    self.assertFalse(disposition["gold_compared"])


if __name__ == "__main__":
    unittest.main()
