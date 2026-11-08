"""Offline tamper/failure tests for v91 export safeguards; no delivery seal."""
from pathlib import Path
import importlib.util
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v91_exporter_under_test", HERE / "export_expansion_v91.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e.json_bytes(value))


class ExportSafeguards(unittest.TestCase):
    def test_protected_inventory_rejects_add_delete_and_modify(self):
        with TemporaryDirectory() as temporary:
            home = Path(temporary)
            prior = home / "prior"
            prior.mkdir()
            target = prior / "original.json"
            target.write_bytes(b"original")
            before = home / "BEFORE.json"
            write(before, {"directories": [str(prior)], "files": [e.bind(target)], "protected_count": 1})
            self.assertEqual(e.verify_preservation(before)["status"], "PASS")
            extra = prior / "extra.json"
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "protected_inventory_changed"):
                e.verify_preservation(before)
            extra.unlink()
            target.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "protected_sha_changed"):
                e.verify_preservation(before)
            target.unlink()
            with self.assertRaisesRegex(RuntimeError, "protected_inventory_changed"):
                e.verify_preservation(before)

    def test_sealed_inventory_rejects_all_file_mutations(self):
        with TemporaryDirectory() as temporary:
            home = Path(temporary)
            target = home / "source.json"
            target.write_bytes(b"original")
            files = e.inventory(home, (e.SEAL_NAME,))
            write(home / e.SEAL_NAME, {"version": "v91_exact_delivery_seal_v1", "status": "COMPLETE",
                "files": files, "file_count": len(files)})
            self.assertEqual(e.verify_seal_inventory(home)["status"], "COMPLETE")
            target.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "sealed_inventory_or_sha_changed"):
                e.verify_seal_inventory(home)
            target.write_bytes(b"original")
            extra = home / "evidence" / "extra.json"
            extra.parent.mkdir()
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "sealed_inventory_or_sha_changed"):
                e.verify_seal_inventory(home)
            extra.unlink()
            target.unlink()
            with self.assertRaisesRegex(RuntimeError, "sealed_inventory_or_sha_changed"):
                e.verify_seal_inventory(home)

    def test_immutable_exports_and_nested_seal_protection(self):
        with TemporaryDirectory() as temporary:
            home = Path(temporary)
            target = home / "evidence" / "export.json"
            e.immutable_write(target, b"original")
            e.immutable_write(target, b"original")
            with self.assertRaisesRegex(RuntimeError, "immutable_export_changed"):
                e.immutable_write(target, b"different")
            (home / e.SEAL_NAME).write_bytes(b"sealed")
            with self.assertRaisesRegex(RuntimeError, "write_after_seal"):
                e.immutable_write(home / "evidence" / "new.json", b"new")
            self.assertFalse((home / "evidence" / "new.json").exists())

    def quality_fixture(self, home):
        inputs, outputs, results, rows = {}, {}, {}, []
        for number in range(10):
            root = "root" + str(number)
            input_path, output_path = home / (root + "_input.json"), home / (root + "_output.json")
            input_path.write_bytes(b"original " + root.encode())
            output_path.write_bytes(input_path.read_bytes() + (b" appended" if number == 0 else b""))
            inputs[root], outputs[root] = e.bind(input_path), e.bind(output_path)
            results[root] = {"accepted_scenes": int(number == 0), "added_turns": int(number == 0)}
            rows.append({"root_id": root, "input_source": inputs[root],
                         "output_source": outputs[root] if number == 0 else None,
                         "new_turn_count": int(number == 0), "read_complete_source": True,
                         "verdict": "HOLD" if number == 0 else "NO_EXPANSION", "findings": []})
        for label, chunk in (("A", rows[:5]), ("B", rows[5:])):
            write(home / "evidence" / ("EXPANSION_QUALITY_REVIEW_" + label + ".json"), {"roots": chunk})
        return inputs, outputs, results, rows

    def test_quality_requires_exact_shards_and_preserves_hold(self):
        with TemporaryDirectory() as temporary:
            home = Path(temporary)
            inputs, outputs, results, rows = self.quality_fixture(home)
            records, bindings = e.quality_records(home, inputs, outputs, results)
            self.assertEqual(len(records), 10)
            self.assertEqual(len(bindings), 2)
            self.assertEqual(e.final_quality(records["root0"], True), "HOLD")
            self.assertEqual(e.final_quality(records["root1"], False), "NO_EXPANSION")
            write(home / "evidence/EXPANSION_QUALITY_REVIEW_B.json", {"roots": rows[4:9]})
            with self.assertRaisesRegex(RuntimeError, "quality_shard_not_assigned_five_roots"):
                e.quality_records(home, inputs, outputs, results)

    def test_quality_rejects_wrong_output_unread_source_and_unexpanded_usable(self):
        with TemporaryDirectory() as temporary:
            home = Path(temporary)
            inputs, outputs, results, rows = self.quality_fixture(home)
            rows[0]["output_source"] = outputs["root1"]
            write(home / "evidence/EXPANSION_QUALITY_REVIEW_A.json", {"roots": rows[:5]})
            with self.assertRaisesRegex(RuntimeError, "quality_wrong_output"):
                e.quality_records(home, inputs, outputs, results)
            rows[0]["output_source"] = outputs["root0"]
            rows[0]["read_complete_source"] = False
            write(home / "evidence/EXPANSION_QUALITY_REVIEW_A.json", {"roots": rows[:5]})
            with self.assertRaisesRegex(RuntimeError, "quality_complete_source_not_read"):
                e.quality_records(home, inputs, outputs, results)
            rows[0]["read_complete_source"] = True
            rows[1]["verdict"] = "USABLE"
            write(home / "evidence/EXPANSION_QUALITY_REVIEW_A.json", {"roots": rows[:5]})
            with self.assertRaisesRegex(RuntimeError, "unexpanded_root_marked_usable"):
                e.quality_records(home, inputs, outputs, results)

    def test_cardinality_repair_is_separate_from_later_failure(self):
        with TemporaryDirectory() as temporary:
            home = Path(temporary)
            stages = []
            for number, (count, status) in enumerate(((6, "REJECT"), (5, "PASS")), 1):
                request = home / ("planner-" + str(number))
                write(request / "PROOF.json", {"role": "planner", "root_id": "root", "scene_id": "S01"})
                write(request / "RESPONSE.json", {"text": json.dumps({"scenes": [{"developments": ["x"] * count}]})})
                verdict = {"status": status}
                if status == "REJECT":
                    verdict["errors"] = [{"code": "scene_developments_count_invalid"}]
                stages.append({"request_directory": str(request), "role": "planner", "status": status, "verdict": verdict})
            request = home / "realizer-1"
            write(request / "PROOF.json", {"role": "realizer", "root_id": "root", "scene_id": "S01"})
            stages.append({"request_directory": str(request), "role": "realizer", "status": "REJECT",
                           "verdict": {"status": "REJECT", "reason": "card_realization_failed"}})
            analysis = e.stage_analysis(stages, SimpleNamespace(parse_object=json.loads))
            self.assertEqual(analysis["planner_cardinality_status"], "REPAIRED")
            self.assertEqual(analysis["planner_count_error_attempts"], 1)
            self.assertEqual(analysis["logical_repeat_requests_by_role"]["planner"], 1)
            self.assertEqual(len(analysis["rejected_stages"]), 2)

    def test_audit_replay_captures_expected_write_without_touching_disk(self):
        with TemporaryDirectory() as temporary:
            home = Path(temporary)
            run = home / "run"
            run.mkdir()
            audit_path = run / "FINAL_AUDIT.json"
            saved = {"status": "PASS", "failures": [], "audit_time_utc": "old", "http_attempts": 1}
            write(audit_path, saved)
            original_bytes = audit_path.read_bytes()
            original_loader = e.load_module
            base = SimpleNamespace(write=write)
            recalculated = {**saved, "audit_time_utc": "new"}
            runtime = SimpleNamespace(base=base)
            module = SimpleNamespace(r=runtime)
            module.audit_run = lambda path: (base.write(path / "FINAL_AUDIT.json", recalculated), recalculated)[1]
            e.load_module = lambda *args: module
            try:
                self.assertIs(e.replay_engine_audit(home, run, saved), runtime)
                self.assertEqual(audit_path.read_bytes(), original_bytes)
                self.assertIs(base.write, write)
                module.audit_run = lambda path: (base.write(path / "UNEXPECTED.json", recalculated), recalculated)[1]
                with self.assertRaisesRegex(RuntimeError, "unexpected_audit_write"):
                    e.replay_engine_audit(home, run, saved)
                self.assertFalse((run / "UNEXPECTED.json").exists())
                self.assertIs(base.write, write)
            finally:
                e.load_module = original_loader

    def test_real_4b_tokenizer_identity_and_body_definition(self):
        manifest = e.read(e.HOME / "expansion_engine/runs" / e.RUN_NAME / "MANIFEST.json")
        tokenizer, receipt = e.body_tokenizer(manifest)
        self.assertEqual(receipt["model"], "Qwen/Qwen3-4B")
        self.assertEqual(e.read(receipt["files"][0]["path"])["hidden_size"], 2560)
        self.assertFalse(receipt["prepare_capacity_checks_certify_qwen3_4b"])
        self.assertEqual(receipt["prepare_capacity_tokenizer_model"], "Qwen/Qwen3-14B")
        self.assertGreater(len(tokenizer.encode("新的步骤需要保留其行动主体和完成状态。", add_special_tokens=False).ids), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
