"""Offline option-freeze contracts; fixtures are not formal QA source approval."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    from . import base_qa_option_freeze_v2_2 as qa
except ImportError:
    import base_qa_option_freeze_v2_2 as qa


class OptionFreezeV22Tests(unittest.TestCase):
    def setUp(self):
        self.source = {"root_id": "synthetic-fixture", "source_id": "fixture-source", "turns": [
            {"turn_ref": "S1:T1", "speaker": "A", "text": "The event was moved to Tuesday."},
            {"turn_ref": "S2:T1", "speaker": "B", "text": "The new time is 14:00."},
        ]}
        self.scope = {"source_id": "fixture-source", "source_sha256": qa.fingerprint(self.source),
                      "source_admission_complete": True, "fixture_only": True,
                      "turns": [{"turn_id": "S1:T1", "formal_qa_allowed": True},
                                {"turn_id": "S2:T1", "formal_qa_allowed": True}]}
        self.candidate = {
            "candidate_id": "fixture-q", "author_id": "fixture-author",
            "question_type": "single_choice", "question": "Which arrangement now applies?",
            "options": {"opt_d": "Tuesday at 16:00", "opt_b": "Tuesday at 14:00",
                        "opt_a": "Monday at 14:00", "opt_c": "Monday at 16:00"},
            "gold": ["opt_b"], "private_certificate": {"fixture": True},
        }

    def freeze(self, path, candidate=None, source=None, scope=None, **kwargs):
        return qa.freeze_bundle(path, candidate or self.candidate, source or self.source,
                                scope or self.scope, reviewer_id="fixture-reviewer",
                                context_id="fresh-fixture", issuer_id="fixture-orchestrator",
                                receipt_id="fixture-receipt", **kwargs)

    def response(self, path, *, answer=None, decision="SOLVED"):
        bundle = qa.load_bundle(path)
        mapping = bundle["render_contract"]["canonical_to_display"]
        return {
            "schema_version": qa.legacy.BLIND_SCHEMA, "candidate_id": "fixture-q",
            "packet_sha256": bundle["packet"]["packet_sha256"],
            "source_scope_sha256": bundle["packet"]["source_scope_sha256"],
            "reviewer_id": "fixture-reviewer", "model_id": "offline-schema-fixture",
            "reviewer_mode": "same_model_blind", "decision": decision,
            "independent_gold": answer if answer is not None else [mapping[key] for key in bundle["candidate"]["gold"]],
            "evidence": [{"turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "quote": turn["text"]}
                         for turn in self.source["turns"]],
            "reasoning_steps": ["Fixture reasoning; no real model was called."],
            "ambiguities": [] if decision == "SOLVED" else ["Fixture HOLD."],
            "source_limitations": ["Synthetic fixture; not formal source admission."],
            "isolation_receipt": qa.reviewer_receipt_projection(path),
        }

    def raw(self, response):
        return json.dumps(response, ensure_ascii=False).encode("utf-8")

    def test_normalization_is_idempotent_input_unchanged_and_gold_order_preserved(self):
        before = copy.deepcopy(self.candidate)
        normalized = qa.normalize_candidate(self.candidate)
        self.assertEqual(self.candidate, before)
        self.assertEqual(list(self.candidate["options"]), ["opt_d", "opt_b", "opt_a", "opt_c"])
        self.assertEqual(list(normalized["options"]), sorted(self.candidate["options"]))
        self.assertEqual(qa.normalize_candidate(normalized), normalized)
        for kind, gold in (("multi_select", ["opt_d", "opt_a"]),
                           ("ordering", ["opt_d", "opt_a", "opt_c", "opt_b"])):
            candidate = dict(self.candidate, question_type=kind, gold=gold)
            self.assertEqual(qa.normalize_candidate(candidate)["gold"], gold)

    def test_all_24_internal_insertions_and_storage_formats_have_same_presentation(self):
        expected = None
        with tempfile.TemporaryDirectory() as tmp:
            for index, ids in enumerate(itertools.permutations(self.candidate["options"])):
                candidate = dict(self.candidate, options={key: self.candidate["options"][key] for key in ids})
                # JSON sorting, formatting and reload are part of the boundary.
                candidate = json.loads(json.dumps(candidate, sort_keys=index % 2 == 0,
                                                   indent=2 if index % 3 == 0 else None))
                path = Path(tmp, str(index))
                self.freeze(path, candidate)
                packet, mapping = qa.frozen_packet(path), qa.frozen_mapping(path)
                current = (packet, mapping)
                if expected is None:
                    expected = current
                self.assertEqual(current, expected, index)
                self.assertEqual(qa.load_bundle(path)["envelope"]["packet_sha256"], packet["packet_sha256"])

    def test_26_options_bounded_reverse_rotate_and_seeded_permutations(self):
        options = {"option_%02d" % i: "Different choice %02d" % i for i in range(26)}
        ids = list(options)
        permutations = [ids, ids[::-1], ids[1:] + ids[:1], ids[-1:] + ids[:-1]]
        for seed in range(4):
            shuffled = ids[:]
            random.Random(seed).shuffle(shuffled)
            permutations.append(shuffled)
        expected = None
        with tempfile.TemporaryDirectory() as tmp:
            for index, keys in enumerate(permutations):
                candidate = dict(self.candidate, options={key: options[key] for key in keys}, gold=[ids[0]])
                path = Path(tmp, str(index))
                self.freeze(path, candidate)
                current = (qa.frozen_packet(path), qa.frozen_mapping(path))
                if expected is None:
                    expected = current
                self.assertEqual(current, expected)
                self.assertEqual(set(current[1]), set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    def test_two_option_and_unicode_boundary(self):
        candidate = dict(self.candidate, question="现在的安排是什么？",
                         options={"choice_2": "周二下午", "choice_1": "周一上午"}, gold=["choice_2"])
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp, candidate)
            packet = qa.frozen_packet(tmp)
            self.assertEqual(packet["packet"]["question"], candidate["question"])
            self.assertEqual(set(packet["packet"]["options"].values()), set(candidate["options"].values()))

    def test_single_multi_and_ordering_review_and_frozen_scoring_roundtrip(self):
        variants = [("single_choice", ["opt_b"]), ("multi_select", ["opt_d", "opt_a"]),
                    ("ordering", ["opt_d", "opt_a", "opt_c", "opt_b"])]
        with tempfile.TemporaryDirectory() as tmp:
            for index, (kind, gold) in enumerate(variants):
                path = Path(tmp, str(index))
                self.freeze(path, dict(self.candidate, question_type=kind, gold=gold))
                response = self.response(path)
                self.assertEqual(qa.freeze_blind_response(path, self.raw(response))["status"], "VALID")
                disposition = qa.blind_disposition(path)
                self.assertEqual(disposition["decision"], "REVIEWER_MATCH")
                self.assertEqual(disposition["author_gold"], gold)
                score = qa.score_frozen_answer(path, response["independent_gold"])
                self.assertTrue(score["correct"])
                self.assertEqual(score["canonical_answer"], gold)

    def test_new_process_reload_uses_identical_frozen_packet_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            expected = {"packet": qa.frozen_packet(tmp), "mapping": qa.frozen_mapping(tmp)}
            code = ("import json,sys; import base_qa_option_freeze_v2_2 as q; "
                    "print(json.dumps({'packet':q.frozen_packet(sys.argv[1]),"
                    "'mapping':q.frozen_mapping(sys.argv[1])},ensure_ascii=False,sort_keys=True))")
            result = subprocess.run([sys.executable, "-c", code, tmp],
                                    cwd=str(Path(qa.__file__).parent), check=True,
                                    capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), expected)

    def test_public_packet_excludes_gold_canonical_ids_certificate_and_private_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            payload = qa.frozen_packet(tmp)["packet"]
            self.assertEqual(set(payload), {"stage", "instruction", "candidate_id", "reviewer_mode",
                                           "source", "source_scope", "question_type", "question", "options"})
            self.assertFalse(set(self.candidate["options"]) & set(payload["options"]))
            self.assertNotIn("gold", payload)
            self.assertNotIn("private_certificate", payload)
            projection = qa.reviewer_receipt_projection(tmp)
            self.assertEqual(set(projection), {"schema_version", "receipt_id", "context_id",
                                               "context_mode", "author_material_visible"})

    def test_gold_changes_do_not_change_public_rendering_but_are_privately_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp, "one"), Path(tmp, "two")
            self.freeze(one)
            self.freeze(two, dict(self.candidate, gold=["opt_a"]))
            self.assertEqual(qa.frozen_packet(one), qa.frozen_packet(two))
            self.assertEqual(qa.frozen_mapping(one), qa.frozen_mapping(two))
            self.assertNotEqual(qa.load_bundle(one)["envelope"]["candidate_sha256"],
                                qa.load_bundle(two)["envelope"]["candidate_sha256"])

    def test_immutable_getters_do_not_modify_stored_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            before = qa.frozen_mapping(tmp)
            local = qa.frozen_mapping(tmp)
            local["A"] = "invented"
            packet = qa.frozen_packet(tmp)
            packet["packet"]["options"]["A"] = "altered"
            self.assertEqual(qa.frozen_mapping(tmp), before)
            self.assertNotEqual(qa.frozen_packet(tmp), packet)

    def test_changed_legacy_rendering_cannot_reassign_frozen_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            packet_bytes = Path(tmp, qa.PACKET).read_bytes()
            bundle = qa.load_bundle(tmp)
            public, private = qa.legacy.v1.prepare_display(bundle["candidate"], seed=qa.SEED)
            private["display_to_canonical"]["A"], private["display_to_canonical"]["B"] = (
                private["display_to_canonical"]["B"], private["display_to_canonical"]["A"])
            with mock.patch.object(qa.legacy.v1, "prepare_display", return_value=(public, private)):
                with self.assertRaises(qa.FreezeError):
                    qa.frozen_packet(tmp)
            self.assertEqual(Path(tmp, qa.PACKET).read_bytes(), packet_bytes)

    def test_no_refreeze_no_response_retry_and_partial_directories_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            with self.assertRaises(qa.FreezeError):
                self.freeze(tmp)
            raw = b"{"
            qa.freeze_blind_response(tmp, raw)
            with self.assertRaises(FileExistsError):
                qa.freeze_blind_response(tmp, self.raw(self.response(tmp)))
            self.assertEqual(Path(tmp, "BLIND_RAW.bin").read_bytes(), raw)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "PARTIAL").write_text("failed attempt")
            with self.assertRaises(qa.FreezeError):
                self.freeze(tmp)

    def test_missing_type_extra_duplicate_invalid_utf8_and_unparseable_raw_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, case in enumerate(("missing", "type", "extra", "duplicate", "utf8", "unparseable")):
                path = Path(tmp, case)
                self.freeze(path)
                response = self.response(path)
                if case == "missing":
                    response.pop("packet_sha256")
                elif case == "type":
                    response["independent_gold"] = "A"
                elif case == "extra":
                    response["unexpected"] = True
                raw = self.raw(response)
                if case == "duplicate":
                    raw = raw[:-1] + b',"candidate_id":"duplicate"}'
                elif case == "utf8":
                    raw = b"\xff"
                elif case == "unparseable":
                    raw = b"{"
                parsed = qa.freeze_blind_response(path, raw)
                self.assertEqual(parsed["status"], "INVALID", index)
                self.assertEqual(Path(path, "BLIND_RAW.bin").read_bytes(), raw)
                result = qa.blind_disposition(path)
                self.assertEqual(result["decision"], "HOLD")
                self.assertFalse(result["gold_compared"])

    def test_reviewer_conflict_holds_and_never_repairs_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            other = qa.load_bundle(tmp)["render_contract"]["canonical_to_display"]["opt_a"]
            response = self.response(tmp, answer=[other])
            raw = self.raw(response)
            self.assertEqual(qa.freeze_blind_response(tmp, raw)["status"], "VALID")
            disposition = qa.blind_disposition(tmp)
            self.assertEqual(disposition["decision"], "HOLD")
            self.assertTrue(disposition["gold_compared"])
            self.assertEqual(disposition["reviewer_gold"], ["opt_a"])
            self.assertEqual(Path(tmp, "BLIND_RAW.bin").read_bytes(), raw)

    def test_reviewer_hold_never_compares_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            response = self.response(tmp, answer=[], decision="HOLD")
            self.assertEqual(qa.freeze_blind_response(tmp, self.raw(response))["status"], "VALID")
            self.assertFalse(qa.blind_disposition(tmp)["gold_compared"])

    def test_every_frozen_core_file_tamper_holds_before_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, name in enumerate(qa._FILES):
                path = Path(tmp, str(index))
                self.freeze(path)
                qa.freeze_blind_response(path, self.raw(self.response(path)))
                original = Path(path, name).read_bytes()
                Path(path, name).write_bytes(original + b" ")
                result = qa.blind_disposition(path)
                self.assertEqual(result["decision"], "HOLD", name)
                self.assertFalse(result["gold_compared"], name)

    def test_envelope_duplicate_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            file = Path(tmp, qa.ENVELOPE)
            raw = file.read_bytes().rstrip()
            file.write_bytes(raw[:-1] + b',"schema_version":"duplicate"}')
            with self.assertRaises(qa.FreezeError):
                qa.frozen_packet(tmp)

    def test_mapping_packet_scope_seed_and_renderer_contract_drift_hold(self):
        # Updating an outer file checksum must not bypass semantic binding checks.
        with tempfile.TemporaryDirectory() as tmp:
            for name, key, value in ((qa.CONTRACT, "seed", "different"),
                                     (qa.CONTRACT, "renderer_version", "different"),
                                     (qa.CONTRACT, "display_to_canonical", {"A": "opt_a"}),
                                     (qa.PACKET, "packet_sha256", "0" * 64),
                                     ("SOURCE_SCOPE.json", "source_sha256", "0" * 64),
                                     (qa.RECEIPT, "unexpected", True)):
                path = Path(tmp, str(len(list(Path(tmp).iterdir()))))
                self.freeze(path)
                target = Path(path, name)
                data = json.loads(target.read_text())
                data[key] = value
                target.write_bytes(qa._bytes(data) + b"\n")
                envelope_path = Path(path, qa.ENVELOPE)
                envelope = json.loads(envelope_path.read_text())
                raw = target.read_bytes()
                envelope["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                envelope_path.write_bytes(qa._bytes(envelope) + b"\n")
                with self.assertRaises(qa.FreezeError, msg=name + ":" + key):
                    qa.load_bundle(path)

    def test_external_source_binding_drift_blocks_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp, "actual-source.json")
            source_path.write_bytes(qa._bytes(self.source))
            scope = dict(self.scope, source_binding=qa._file_binding(source_path),
                         source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
            path = Path(tmp, "run")
            self.freeze(path, scope=scope)
            source_path.write_bytes(source_path.read_bytes() + b" ")
            with self.assertRaises(qa.FreezeError):
                qa.frozen_packet(path)

    def test_extra_implementation_binding_drift_blocks_all_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            extra = Path(tmp, "extra_code.txt")
            extra.write_text("first version")
            path = Path(tmp, "run")
            self.freeze(path, extra_paths=[extra])
            extra.write_text("modified version")
            with self.assertRaises(qa.FreezeError):
                qa.load_bundle(path)
            self.assertEqual(qa.score_frozen_answer(path, ["A"])["status"], "HOLD")

    def test_received_raw_is_preserved_even_when_context_has_drifted(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            raw = self.raw(self.response(tmp))
            Path(tmp, qa.CONTRACT).write_bytes(b"{}")
            parsed = qa.freeze_blind_response(tmp, raw)
            self.assertEqual(parsed["status"], "INVALID")
            self.assertEqual(Path(tmp, "BLIND_RAW.bin").read_bytes(), raw)
            self.assertFalse(qa.blind_disposition(tmp)["gold_compared"])

    def test_source_scope_argument_drift_and_invalid_scoring_labels_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.freeze(tmp)
            with self.assertRaises(qa.FreezeError):
                qa.frozen_packet(tmp, source=dict(self.source, source_id="wrong"))
            with self.assertRaises(qa.FreezeError):
                qa.frozen_packet(tmp, source_scope=dict(self.scope, source_sha256="0" * 64))
            for answer in (["unknown"], ["A", "A"], "A", [], ["A", "B"]):
                self.assertEqual(qa.score_frozen_answer(tmp, answer)["status"], "HOLD")

    def test_invalid_ids_cardinality_nonfinite_and_ordering_coverage_are_rejected(self):
        for candidate in (
            dict(self.candidate, options={"invalid id": "x", "valid": "y"}),
            dict(self.candidate, options={"only_one": "x"}, gold=["only_one"]),
            dict(self.candidate, gold=["missing"]),
            dict(self.candidate, gold=["opt_a", "opt_a"]),
            dict(self.candidate, gold=["opt_a", "opt_b"]),
            dict(self.candidate, question_type="ordering", gold=["opt_a", "opt_b"]),
            dict(self.candidate, private_numeric=float("nan")),
        ):
            with self.assertRaises(ValueError):
                qa.normalize_candidate(candidate)

    def test_historical_104814_failure_is_retained_and_not_migrated(self):
        root = Path(__file__).resolve().parents[2]
        history = root / "research-protocols/selfplay_week_20261019/base_qa_v2_1_dual_root_real_20261101/cc-train-episode-104814"
        paths = sorted(path for path in history.rglob("*") if path.is_file())
        self.assertTrue(paths)
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        original = json.loads((history / "author/CANDIDATE_01.json").read_text())
        frozen = json.loads((history / "review_01/AUTHOR_CANDIDATE.json").read_text())
        self.assertEqual(qa.legacy.fingerprint(original), qa.legacy.fingerprint(frozen))
        old_public, _ = qa.legacy.v1.prepare_display(original, seed=qa.SEED)
        reread_public, _ = qa.legacy.v1.prepare_display(frozen, seed=qa.SEED)
        self.assertNotEqual(old_public["options"], reread_public["options"])
        # The regression is reproduced read-only. No new freeze of this question.
        self.assertEqual(before, {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


if __name__ == "__main__":
    unittest.main()
