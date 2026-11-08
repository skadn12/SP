"""Offline checks for source-item identity and the v2 request boundary."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import runner
import unique_events


class UniqueEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = runner.base.dataset_cards.freeze_source_subset()
        cls.runs = unique_events.prior_runs(runner.V1_RUNS, runner.V2_RUNS)
        cls.prior = unique_events.collect_prior_usage(cls.runs)
        cls.snapshot = unique_events.allocate(cls.manifest, runner.base.ROOTS, cls.prior)

    def test_only_accepted_prior_scenes_consume_source_events(self):
        self.assertEqual(len(self.prior["accepted_uses"]), 3)
        self.assertEqual({row["source_id"] for row in self.prior["accepted_uses"]}, {
            "multiwoz22:train:SNG1013.json",
            "multiwoz22:train:SSNG0142.json",
            "multiwoz22:train:PMUL0843.json",
        })
        self.assertEqual(set(self.snapshot["excluded_prior_source_ids"]),
                         {row["source_id"] for row in self.prior["accepted_uses"]})
        self.assertEqual(self.snapshot["assigned_slots"], 3)
        self.assertEqual(self.snapshot["target_slots"], 9)
        self.assertEqual([len(rows) for rows in self.snapshot["assignments"].values()], [1, 1, 1])
        self.assertTrue(all(row["dataset"] == "allenai/soda"
                            for rows in self.snapshot["assignments"].values() for row in rows))

    def test_same_item_with_different_card_id_and_span_is_duplicate(self):
        card = deepcopy(runner.base.read(runner.V1_RUNS / "pilot_02" / "dataset_cards_result.json")["accepted_cards"][0])
        alias = deepcopy(card)
        alias["card_id"] = "another-card-id"
        alias["lineage"]["source_span"] = {"turn_start": 2, "turn_end": 4}
        self.assertEqual(unique_events.event_key(card), unique_events.event_key(alias))
        synthetic = unique_events.allocate(self.manifest, ["one"], {"accepted_uses": []}, 3)
        synthetic["assignments"]["one"] = [{
            "dataset": card["lineage"]["dataset"], "split": card["lineage"]["split"],
            "item_id": card["lineage"]["item_id"], "source_id": card["source_id"],
            "source_item_sha256": card["lineage"]["source_item_sha256"],
        }]
        with self.assertRaisesRegex(ValueError, "duplicate_event_card_for_root"):
            unique_events.verify_cards_for_root([card, alias], synthetic, "one")

    def test_prior_used_card_is_not_visible_to_any_new_root(self):
        cards = runner.base.read(runner.V1_RUNS / "pilot_02" / "dataset_cards_result.json")["accepted_cards"]
        for root in runner.base.ROOTS:
            self.assertEqual(runner.cards_for_root(cards, self.snapshot, root), [])
            with self.assertRaisesRegex(ValueError, "card_source_not_allocated_to_root"):
                unique_events.verify_cards_for_root(cards[:1], self.snapshot, root)

    def test_changed_historical_evidence_or_allocation_is_rejected(self):
        unique_events.verify_allocation(self.snapshot, self.manifest, runner.base.ROOTS)
        changed = deepcopy(self.snapshot)
        changed["assignments"][runner.base.ROOTS[0]].append(changed["assignments"][runner.base.ROOTS[1]][0])
        with self.assertRaisesRegex(ValueError, "allocation_changed"):
            unique_events.verify_allocation(changed, self.manifest, runner.base.ROOTS)
        changed = deepcopy(self.snapshot)
        changed["prior"]["prior_evidence"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            unique_events.verify_allocation(changed, self.manifest, runner.base.ROOTS)

    def test_prepare_freezes_prior_usage_and_balanced_assignments_without_http(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            prepared = runner.prepare(path)
            self.assertEqual(prepared["http_calls"], 0)
            self.assertEqual(prepared["assigned_B_slots"], 3)
            self.assertEqual([len(rows) for rows in prepared["per_root"].values()], [1, 1, 1])
            self.assertFalse((path / "LIVE_STARTED.json").exists())
            runner._verify_v2(path)

    def test_interrupted_live_run_blocks_reallocation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            interrupted = base / "runs" / "interrupted"
            interrupted.mkdir(parents=True)
            (interrupted / "LIVE_STARTED.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "incomplete_prior_live_run_usage_unknown"):
                unique_events.prior_runs(base / "runs", base / "other")


if __name__ == "__main__":
    unittest.main()
