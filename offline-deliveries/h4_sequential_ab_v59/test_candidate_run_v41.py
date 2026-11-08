"""Temporary whole-run collection; no network or production source reservations."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoTokenizer  # Load SSL before prohibiting socket use.
import candidate_runner as c


class OfflineCaller:
    def __init__(self, run):
        self.run = run
        self.request_counter = 0
        self.http_counter = 0

    def call(self, label, payload, proof):
        self.request_counter += 1
        self.http_counter += 1  # Synthetic receipt count; this fake never opens a socket.
        directory = self.run / "requests" / f"{self.request_counter:03d}-{label}"
        c.r.base.write(directory / "REQUEST.json", payload)
        c.r.base.write(directory / "PROOF.json", {**proof, "payload_sha256": c.r.base.digest(payload)})
        c.r.base.write(directory / "RESERVATION.json", {"request_number": self.request_counter})
        c.r.base.write(directory / "HTTP_ATTEMPT.json", {"mock_only": True, "http_attempt_number": self.http_counter})
        context = json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split("\nCANDIDATE_INPUT_JSON\n", 1)[1])[0]
        assert len(context["cumulative_history"]) >= 132
        if context['dataset_card'] is not None:
            assert context['dataset_card']['steps'][0]['evidence'][0]['quote'] == 'MOCK ONLY source quotation'
            prompt = payload['messages'][0]['content']
            assert prompt.rfind('MOCK ONLY source quotation') > prompt.index('CURRENT TASK AFTER ALL HISTORICAL DATA')
            if proof['role'] in ('planner', 'realizer'):
                assert prompt.index('CURRENT TASK AFTER ALL HISTORICAL DATA') > prompt.index('[/PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]')
        ident = {key: context[key] for key in c.IDENTITY_KEYS}
        role = proof["role"]
        if role == "planner":
            result = {"broken_optional_plan": True}
        elif role == "realizer":
            result = {**ident, "status": "scene", "turns": [
                {"speaker": speaker, "text": f"MOCK ONLY {context['route']} {context['scene_id']} speaker {i}: new synthetic fixture detail for collection integration."}
                for i, speaker in enumerate(context["allowed_speakers"][:2])],
                "dataset_card_id": context["dataset_card"]["card_id"] if context["dataset_card"] else None}
        else:
            result = {**ident, "verdict": "uncertain", "summary": "MOCK ONLY missing auxiliary checks."}
        raw = json.dumps(result)
        c.r.base.write(directory / "RESPONSE.json", {"text": raw, "metadata": {"mock_only": True}})
        c.r.base.write(directory / "HTTP_RESPONSE_01.json", {"text": raw, "metadata": {"mock_only": True}})
        return raw, directory


def fixture_cards(caller, run, allocation):
    # Transport/card extraction is separately validated; these explicit test
    # cards exercise reserved-source identity and downstream cumulative wiring.
    cards = []
    for i, item in enumerate(allocation["assignments"][c.ROOT], 1):
        card = {**item, "card_id": f"MOCK_CARD_{i}", "card_sha256": "0" * 64,
                "lineage": {k: item[k] for k in ("dataset", "split", "item_id", "source_item_sha256")},
                "guidance_level": "seed", "actors": [{"actor_id": "a1", "role": "mock speaker"}],
                "steps": [{"step_id": "s1", "actor_id": "a1", "action": "MOCK ONLY", "state": "proposed",
                           "evidence": [{"unit_id": "MOCK:0", "quote": "MOCK ONLY source quotation"}]}],
                "dependencies": [], "confusion_axes": [], "open_ends": [], "downgrade_reason": "MOCK ONLY"}
        c.r.source_inventory.verify_cards_for_root([card], allocation, c.ROOT)
        c.r.base.write(run / "extractions" / f"{i:02d}.json", {"source_id": item["source_id"],
            "card": card, "attempts": [], "source_remains_consumed": True})
        cards.append(card)
    return cards


class WholeRunTests(unittest.TestCase):
    def test_prepare_collect_b2_and_replay_without_training_admission(self):
        self.run_flow((c.r.ROUTES[1],), scenes=2, expected_records=2, expected_B=2)

    def test_same_policy_supports_A_only_and_A_then_B(self):
        self.run_flow((c.r.ROUTES[0],), scenes=1, expected_records=1, expected_B=0)
        self.run_flow(c.r.ROUTES, scenes=1, expected_records=2, expected_B=1)

    def run_flow(self, routes, *, scenes, expected_records, expected_B):
        original_here = c.HERE
        original_collect = c.r.source_inventory.collect_prior
        with tempfile.TemporaryDirectory(prefix="V41_MOCK_ONLY_") as tmp, ExitStack() as stack:
            delivery = Path(tmp) / "h4_sequential_ab_v41"
            delivery.mkdir()
            for source in original_here.iterdir():
                if source.is_file() and source.suffix in (".py", ".md", ".json"):
                    shutil.copyfile(source, delivery / source.name)
            unique = delivery.parent / "h4_r02_source_unique_v2"
            unique.mkdir()
            shutil.copyfile(original_here.parent / "h4_r02_source_unique_v2/unique_events.py", unique / "unique_events.py")
            stack.enter_context(patch.object(c, "HERE", delivery))
            stack.enter_context(patch.object(c.r, "LOCK_PATH", Path(tmp) / "lock"))
            stack.enter_context(patch.object(c.r.source_inventory, "collect_prior",
                side_effect=lambda: original_collect(current_runs=delivery / "runs")))
            stack.enter_context(patch("candidate_transport.Caller", OfflineCaller))
            stack.enter_context(patch.object(c, "extract_cards", side_effect=fixture_cards))
            stack.enter_context(patch("socket.socket", side_effect=AssertionError("network_forbidden")))
            run = delivery / "runs/MOCK_COLLECTION"
            before = c.r.source_inventory.collect_prior()
            prepared = c.prepare(run, routes=routes, scenes=scenes)
            self.assertEqual(prepared["assigned_B_slots"], expected_B)
            self.assertFalse((run / "SOURCE_RESERVATIONS.json").exists())
            self.assertEqual(c.r.source_inventory.collect_prior(), before)
            self.assertEqual(c.verify(run)[0]["formal_history_sha256"], c.r.AUDITED_V34_A3_HISTORY_SHA256)
            report = c.live(run)
            self.assertEqual(report["metrics"]["retained"]["scenes"], expected_records)
            self.assertEqual(report["metrics"]["admitted"]["body_tokens"], 0)
            self.assertFalse(report["training_pool_eligible"])
            self.assertFalse((run / "MERGE_MANIFEST.json").exists())
            self.assertFalse((run / "sources_merged").exists())
            self.assertEqual(len(c.r.base.read(run / "SOURCE_RESERVATIONS.json")["reservations"]), expected_B)
            self.assertEqual(len(c.r.source_inventory.collect_prior()["accepted_uses"]),
                             len(before["accepted_uses"]) + expected_B)
            self.assertEqual(c.audit(run)["status"], "ARTIFACT_REPLAY_VERIFIED")
            final_history = c.r.base.read(run / "CANDIDATE_HISTORY.json")["turns"]
            baseline = c.r.base.read(run / "FORMAL_BASELINE.json")["turns"]
            self.assertEqual(final_history[:132], baseline)
            self.assertEqual(len(final_history), 132 + expected_records * 2)
            self.assertTrue(all(t["origin"] == "provisional_candidate" for t in final_history[132:]))
            # Byte bindings independently reject later dialogue tampering.
            history_path = run / "CANDIDATE_HISTORY.json"
            history_path.write_text(history_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "frozen_file_changed"):
                c.audit(run)


if __name__ == "__main__":
    unittest.main()
