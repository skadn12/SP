"""No-network regressions using frozen real planner and copied-scene responses."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import candidate_runner as c
from candidate_request_audit import _context, audit_requests
import grounded_task
import retry_a_diagnostic as d

read = c.r.base.read
ROOT, ROUTE, SEED = d.ROOT, d.ROUTE, d.SEED


def fixture(scene, role="planner", attempt=1):
    req = next((SEED / "requests").glob(
        f"*-A-{scene}-{role}" + (f"-{attempt}" if role == "realizer" else "")))
    raw = read(req / "RESPONSE.json")["text"]
    history = read(SEED / "candidates" / ROOT / "A" / scene / "INPUT.json")["history"]
    return raw, history


class AnchorTests(unittest.TestCase):
    def test_all_three_real_planners_recover_exact_quoted_anchors(self):
        for scene in ("S01", "S02", "S03"):
            with self.subTest(scene=scene):
                raw, history = fixture(scene)
                original = json.loads(raw)
                plan, provenance = c.validate_plan_with_provenance(
                    raw, c.identity(ROOT, ROUTE, scene), history, None)
                self.assertEqual(provenance["history_sha256"], c.r.base.digest(history))
                self.assertFalse(provenance["semantic_certified"])
                self.assertEqual({k: v for k, v in plan.items() if k != "source_anchors"},
                                 {k: v for k, v in original.items() if k != "source_anchors"})
                for old, anchor, receipt in zip(original["source_anchors"], plan["source_anchors"], provenance["anchors"]):
                    self.assertEqual(old["quote"], anchor["quote"])
                    matching = [t for t in history if anchor["quote"] in t["text"]
                                and t["speaker"] == receipt["matched_history_speaker"]]
                    self.assertEqual(len(matching), 1)
                    self.assertEqual(matching[0]["turn_ref"], anchor["turn_ref"])
                    self.assertEqual(receipt["original"], old)
                    self.assertEqual(receipt["normalized"], anchor)
                    if old["turn_ref"] != anchor["turn_ref"]:
                        self.assertIn("derive_speaker_from_unique_exact_quote", receipt["operations"])
                    else:
                        self.assertEqual(anchor, old)
                self.assertEqual(json.loads(raw), original)

    def check(self, mutate, history_mutate=None, error="plan_anchor"):
        raw, history = fixture("S01")
        plan = json.loads(raw)
        mutate(plan["source_anchors"][0])
        if history_mutate:
            history_mutate(history)
        with self.assertRaisesRegex(ValueError, error):
            c.validate_plan_with_provenance(json.dumps(plan), c.identity(ROOT, ROUTE, "S01"), history, None)

    def test_wrong_root_cannot_be_recovered_by_quote(self):
        self.check(lambda a: a.update(turn_ref="cc-train-episode-135466:R01:S01:T001"))

    def test_wrong_position_cannot_be_recovered_by_quote(self):
        self.check(lambda a: a.update(turn_ref=ROOT + ":R01:S01:T002"))

    def test_speaker_conflict_cannot_be_recovered(self):
        self.check(lambda a: a.update(speaker="Emily"), error="speaker_mismatch")

    def test_ambiguous_legacy_quote_is_not_speaker_inference(self):
        def extra(history):
            turn = deepcopy(next(t for t in history if t["turn_ref"] == "R01:S01:T001"))
            turn["turn_ref"] = "CANDIDATE:A:S00:T001"
            history.append(turn)
        self.check(lambda a: None, extra, "ambiguous_quote_speaker")

    def test_alias_with_speaker_still_requires_unique_quote_speaker(self):
        def extra(history):
            turn = deepcopy(next(t for t in history if t["turn_ref"] == "R01:S01:T001"))
            turn["turn_ref"] = "CANDIDATE:A:S00:T001"
            history.append(turn)
        self.check(lambda a: a.update(speaker="David"), extra, "ambiguous_quote_speaker")

    def test_other_speaker_quote_is_not_ambiguous_when_speaker_explicit(self):
        raw, history = fixture("S01")
        plan = json.loads(raw)
        plan["source_anchors"][0]["speaker"] = "David"
        turn = deepcopy(next(t for t in history if t["turn_ref"] == "R01:S01:T001"))
        turn.update(turn_ref="CANDIDATE:A:S00:T001", speaker="Emily")
        history.append(turn)
        result, _ = c.validate_plan_with_provenance(json.dumps(plan),
            c.identity(ROOT, ROUTE, "S01"), history, None)
        self.assertEqual(result["source_anchors"][0]["speaker"], "David")

    def test_quote_not_exact_is_rejected(self):
        self.check(lambda a: a.update(quote=a["quote"].lower()))

    def test_exact_legacy_anchor_does_not_gain_a_new_ambiguity_gate(self):
        raw, history = fixture("S01")
        plan = json.loads(raw)
        anchor = plan["source_anchors"][0]
        anchor["turn_ref"] = "R01:S01:T001"
        turn = deepcopy(next(t for t in history if t["turn_ref"] == anchor["turn_ref"]))
        turn["turn_ref"] = "CANDIDATE:A:S00:T001"
        history.append(turn)
        result, provenance = c.validate_plan_with_provenance(json.dumps(plan),
            c.identity(ROOT, ROUTE, "S01"), history, None)
        self.assertEqual(result["source_anchors"][0], anchor)
        self.assertEqual(provenance["anchors"][0]["operations"], [])

    def test_B_aliases_are_not_normalized(self):
        raw, history = fixture("S01")
        plan = json.loads(raw)
        plan.update(route=c.r.ROUTES[1], dataset_card_id="MOCK", actor_mapping=[])
        with self.assertRaisesRegex(ValueError, "plan_anchor_invalid"):
            c.validate_plan_with_provenance(json.dumps(plan),
                c.identity(ROOT, c.r.ROUTES[1], "S01"), history, {"card_id": "MOCK", "actors": []})


class RepairTests(unittest.TestCase):
    def test_four_real_copies_still_reject_and_repair_locates_all_thirteen_turns(self):
        for scene in ("S02", "S03"):
            for attempt in (1, 2):
                raw, history = fixture(scene, "realizer", attempt)
                baseline = history[:110]
                obj, failures, _, dup = c.inspect_realization(raw, c.identity(ROOT, ROUTE, scene),
                                                             history, baseline, None)
                self.assertEqual(failures, [])
                self.assertEqual(dup["status"], "REJECT")
                repair = c.mechanical_repair.build(raw, failures, dup)
                span = repair["copied_spans"][0]
                self.assertEqual((span["new_start_position"], span["new_end_position"]), (1, 13))
                self.assertEqual((span["source_start_position"], span["source_end_position"]), (111, 123))
                self.assertEqual(span["source_start_ref"], "CANDIDATE:A:S01:T001")
                data, proof = c.request(SEED, ROOT, ROUTE, scene, "realizer", history, baseline,
                                        repair=repair)
                prompt = data["messages"][0]["content"]
                context = _context(data)
                tail = json.JSONDecoder().raw_decode(prompt.split(grounded_task.TARGET_MARKER)[1])[0]
                self.assertEqual(context["cumulative_history"], history)
                self.assertNotIn("previous_response", context["mechanical_repair"])
                self.assertEqual(tail["mechanical_repair"]["previous_response"], raw)
                self.assertEqual(prompt.count('"previous_response":'), 1)
                # One occurrence in history, one escaped in the single rejected
                # response; no third reinforcement from a duplicated repair.
                self.assertEqual(prompt.count(obj["turns"][0]["text"]), 2)
                self.assertEqual(proof["final_payload_sha256"], c.r.base.digest(data))
                self.assertLess(prompt.index("[/PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]"),
                                prompt.index(grounded_task.TAIL_MARKER))
                self.assertIn("genuinely new interaction", repair["instruction"])


class DiagnosticFakeCaller:
    def __init__(self, run):
        self.run, self.request_counter, self.http_counter = run, 0, 0

    def call(self, label, payload, proof):
        self.request_counter += 1
        self.http_counter += 1
        folder = self.run / "requests" / f"{self.request_counter:03d}-{label}"
        ctx, scene, role = _context(payload), proof["scene_id"], proof["role"]
        assert len(ctx["cumulative_history"]) == (123 if scene == "S02" else 125)
        assert ctx["dataset_card"] is None
        assert all("CANDIDATE:B:" not in t["turn_ref"] for t in ctx["cumulative_history"])
        identity = c.identity(ROOT, ROUTE, scene)
        if role == "planner":
            raw = fixture(scene)[0]
        elif role == "realizer" and label.endswith("-1"):
            raw = fixture(scene, "realizer", 1)[0]
        elif role == "realizer":
            assert ctx["advisory_plan"] is not None
            assert ctx["mechanical_repair"]["previous_response_occurrences"] == 1
            raw = json.dumps({**identity, "status": "scene", "dataset_card_id": None,
                "turns": [{"speaker": speaker,
                    "text": f"MOCK ONLY regression {scene}: unique new action {i}, deliberately distinct from the historical quoted scene."}
                    for i, speaker in enumerate(ctx["allowed_speakers"])],
                "feedback_topic_ids": [], "no_scene_reason": ""})
        else:
            raw = json.dumps({**identity, "verdict": "uncertain", "summary": "MOCK ONLY"})
        c.r.base.write(folder / "REQUEST.json", payload)
        c.r.base.write(folder / "PROOF.json", {**proof, "payload_sha256": c.r.base.digest(payload)})
        c.r.base.write(folder / "RESERVATION.json", {"request_number": self.request_counter})
        c.r.base.write(folder / "HTTP_ATTEMPT_01.json", {"http_attempt_number": self.http_counter, "mock_only": True})
        c.r.base.write(folder / "RESPONSE.json", {"text": raw, "mock_only": True})
        c.r.base.write(folder / "HTTP_RESPONSE_01.json", {"text": raw, "mock_only": True})
        return raw, folder


class DiagnosticFlowTests(unittest.TestCase):
    def test_real_seed_fake_two_slot_repair_and_audit_and_tamper(self):
        with tempfile.TemporaryDirectory(prefix="V52_OFFLINE_ONLY_") as tmp, ExitStack() as stack:
            dest = Path(tmp)
            for source in d.HERE.iterdir():
                if source.is_file() and source.suffix in (".py", ".json", ".md"):
                    shutil.copyfile(source, dest / source.name)
            stack.enter_context(patch.object(d, "HERE", dest))
            stack.enter_context(patch.object(c.r, "LOCK_PATH", dest / "MOCK.lock"))
            stack.enter_context(patch("candidate_transport.Caller", DiagnosticFakeCaller))
            stack.enter_context(patch.object(c, "token_counter", return_value=lambda text: len(text.split())))
            stack.enter_context(patch.object(c.r, "transport_inputs", return_value=({"offline_only": True}, [], [])))
            stack.enter_context(patch.object(c.r.base, "transport", return_value=(lambda: {"offline_only": True},)))
            stack.enter_context(patch("socket.socket", side_effect=AssertionError("network_forbidden")))
            run = dest / "runs/mock_A2_A3"
            self.assertEqual(d.prepare(run)["initial_history_turns"], 123)
            report = d.live(run)
            self.assertEqual((report["logical_calls"], report["http_attempts"]), (8, 8))
            self.assertEqual(report["metrics"]["new_A_responses_only"]["retained"]["scenes"], 2)
            self.assertEqual(report["metrics"]["new_A_responses_only"]["rejected"]["scenes"], 2)
            self.assertEqual(d.audit(run)["status"], "ARTIFACT_REPLAY_VERIFIED")
            self.assertEqual(len(read(run / "CANDIDATE_HISTORY.json")["turns"]), 127)
            self.assertFalse((run / "SOURCE_RESERVATIONS.json").exists())
            # B cannot silently enter a retry via its seed.
            initial = run / "INITIAL_HISTORY.json"
            original = initial.read_bytes()
            value = read(initial)
            value["turns"][-1]["turn_ref"] = "CANDIDATE:B:S01:T001"
            initial.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                d.verify(run)
            initial.write_bytes(original)
            # Independently replay provenance, even bypassing collection hashes.
            advisory_path = run / "candidates" / ROOT / "A/S02/PLAN_ADVISORY.json"
            advisory = read(advisory_path)
            advisory["anchor_normalization"]["anchors"][0]["normalized"]["speaker"] = "WRONG"
            advisory_path.write_text(json.dumps(advisory))
            with self.assertRaisesRegex(ValueError, "anchor_normalization_changed"):
                audit_requests(run, read(run / "CANDIDATE_RECORDS.json"),
                               read(run / "FORMAL_BASELINE.json")["turns"])


if __name__ == "__main__":
    unittest.main()
