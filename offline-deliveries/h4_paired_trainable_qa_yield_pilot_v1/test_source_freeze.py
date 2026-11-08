"""Source preflight and predeclared backup behavior."""

import unittest
from unittest.mock import patch

import common
import prepare


class SourceFreezeTests(unittest.TestCase):
    def test_all_predeclared_promoted_sources_pass_integrity(self):
        for root_id, spec in common.SOURCES.items():
            with self.subTest(root_id=root_id):
                view = common.load_surface(root_id)
                self.assertEqual(len(view["turns"]), spec["turn_count"])
                self.assertEqual(view["source_sha256"], spec["sha256"])

    def test_second_primary_failure_replaces_only_its_slot(self):
        primary_ids = list(common.ROOTS)

        def load(root_id):
            if root_id == primary_ids[1]:
                raise RuntimeError("source_hash_mismatch")
            return {"root_id": root_id}

        with patch.object(prepare, "load_surface", side_effect=load) as mocked:
            selected = prepare.select_roots()
        self.assertEqual([row["root_id"] for row in selected],
                         [primary_ids[0], common.BACKUP_ROOT])
        self.assertEqual([row["slot_index"] for row in selected], [0, 1])
        self.assertEqual(selected[1]["substitution"]["replaced_root_id"], primary_ids[1])
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(prepare.arm_order(0), list(reversed(prepare.arm_order(1))))

    def test_backup_failure_aborts_before_freeze(self):
        def load(root_id):
            if root_id == next(iter(common.ROOTS)):
                raise RuntimeError("source_hash_mismatch")
            if root_id == common.BACKUP_ROOT:
                raise ValueError("promoted_source_integrity_failed")
            return {"root_id": root_id}

        with patch.object(prepare, "load_surface", side_effect=load):
            with self.assertRaisesRegex(RuntimeError, "backup_source_integrity_failed"):
                prepare.select_roots()

    def test_two_invalid_primaries_cannot_reuse_backup(self):
        def load(root_id):
            if root_id in common.ROOTS:
                raise RuntimeError("source_hash_mismatch")
            return {"root_id": root_id}

        with patch.object(prepare, "load_surface", side_effect=load) as mocked:
            with self.assertRaisesRegex(RuntimeError, "both_primary_roots_invalid_one_backup_only"):
                prepare.select_roots()
        self.assertEqual(mocked.call_count, 3)

    def test_prior_hash_change_rejects_source(self):
        root_id = next(iter(common.ROOTS))
        prior_path = common.ROOTS[root_id]["prior_path"]
        original_sha_file = common.sha_file

        def sha_file(path):
            return "0" * 64 if path == prior_path else original_sha_file(path)

        with patch.object(common, "sha_file", side_effect=sha_file):
            with self.assertRaisesRegex(RuntimeError, "prior_hash_mismatch"):
                common.load_surface(root_id)


if __name__ == "__main__":
    unittest.main()
