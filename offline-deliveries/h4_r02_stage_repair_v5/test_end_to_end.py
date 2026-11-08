"""Offline branch/feedback/repair regression with a contract-faithful provider.

These are scripted control-flow fixtures, not Qwen outputs or semantic quality
evidence. Branch tests copy frozen feedback into TemporaryDirectory. The live
control-flow tests also run prepare()/live(), with temporary inventory/lock and
an injected provider. No real transport or persistent run is used.
"""

from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runner


PREFLIGHT = runner.HERE / "runs/r02_stage_repair_v5_preflight_02"


def input_context(prompt):
    marker = next(marker for marker in (
        "\nINPUT_JSON\n", "\nREVIEW INPUT DATA:\n", "\nSOURCE_ITEMS_JSON:\n"
    ) if marker in prompt)
    # Feedback and reviewer repair instructions follow the first JSON object.
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]


def fixture_cards(count):
    result = []
    for number in range(1, count + 1):
        card = {
            "card_id": f"MOCK_ONLY_CARD_{number}",
            "guidance_level": "scaffold", "downgrade_reason": "",
            "actors": [{"actor_id": "person_A", "role": "requester"},
                       {"actor_id": "person_B", "role": "coordinator"}],
            "steps": [
                {"step_id": "s1", "actor_id": "person_A", "action": "request a booking",
                 "state": "proposed"},
                {"step_id": "s2", "actor_id": "person_B", "action": "propose an available time",
                 "state": "reported_action"},
            ],
            "dependencies": [{"before": "s1", "after": "s2", "relation": "order"}],
            "confusion_axes": ["requested versus offered time"], "open_ends": [],
        }
        card["card_sha256"] = runner.base.digest(card)
        result.append(card)
    return result


class ScriptedProvider:
    """Reconstruct the real provider envelope, and return its response shape."""

    def __init__(self, mode=None):
        self.mode = mode
        self.observed = []
        self.used_injections = set()
        self.role_counts = Counter()
        self.before_request = lambda: None
        self.omit_source_id = None

    def inject(self, mode, role, ctx):
        # Inject at most once, in A/S01 of the first root exercised by a test.
        if (self.mode == mode and mode not in self.used_injections
                and ctx["route"] == runner.ROUTES[0]
                and ctx.get("target_scene_id", ctx.get("scene_id")) == "S01"):
            self.used_injections.add(mode)
            return True
        return False

    def factory(self, _profile, *, on_http_attempt):
        owner = self

        class Client:
            def complete(self, *, prompt, max_tokens, json_object):
                actual = {
                    "model": runner.MODEL, "temperature": .25,
                    "max_tokens": max_tokens, "enable_thinking": False,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                }
                assert max_tokens == 4096 and json_object is True
                owner.before_request()
                on_http_attempt(actual)
                ctx = input_context(prompt)
                role = "dataset_extractor" if "source_items" in ctx else ctx.get("role", "scene_reviewer")
                owner.role_counts[role] += 1
                owner.observed.append({"role": role, "context": ctx, "payload": actual})
                if role == "planner" and owner.inject("http_503", role, ctx):
                    return SimpleNamespace(raw_text="", metadata={"http_status": 503})
                result = getattr(owner, role)(ctx)
                return SimpleNamespace(raw_text=json.dumps(result), metadata={
                    "returned_model": runner.MODEL, "finish_reason": "stop", "http_status": 200,
                })

        return Client()

    def dataset_extractor(self, ctx):
        item = ctx["source_items"][0]
        if item["source_id"] == self.omit_source_id:
            return {"schema_version": runner.card_selection.SCHEMA_VERSION, "cards": []}
        unit = next(unit for unit in item["units"] if unit["kind"] == "dialogue")
        card = {
            "card_id": "MOCK_SEED_" + runner.base.digest(item["source_id"])[:12],
            "source_id": item["source_id"], "guidance_level": "seed",
            "actors": [{"actor_id": "actor_1", "source_speaker": unit["speaker"],
                        "role": "participant"}],
            "steps": [{"step_id": "s1", "actor_id": "actor_1",
                       "action": "actor_1 introduces a conversational premise", "state": "proposed",
                       "evidence": [{"unit_id": unit["unit_id"], "quote": unit["text"][:500]}]}],
            "dependencies": [], "confusion_axes": [], "open_ends": [],
            "downgrade_reason": "MOCK ONLY: one quoted premise exercises the seed schema.",
        }
        return {"schema_version": runner.card_selection.SCHEMA_VERSION, "cards": [card]}

    def planner(self, ctx):
        card = ctx["dataset_cards"][0] if ctx["dataset_cards"] else None
        anchor = ctx["cumulative_history"][-1]
        tag = f"{ctx['root_id']} {ctx['route']} {ctx['target_scene_id']}"
        scene = {
            "scene_id": ctx["target_scene_id"], "actors": ctx["allowed_speakers"],
            "source_anchors": [{"turn_ref": anchor["turn_ref"], "quote": anchor["text"][:100]}],
            "goal": f"MOCK ONLY: arrange the next appointment for {tag}.",
            "developments": [f"The requester asks for an appointment ({tag}).",
                             f"The coordinator offers an available time after that request ({tag})."],
            "dataset_card_id": card["card_id"] if card else None,
            "card_development_links": [{
                "card_step_ids": [step["step_id"] for step in card["steps"]],
                "dependency_indices": list(range(1, len(card["dependencies"]) + 1)),
                "development_indices": [1, 2],
            }] if card else [],
            "feedback_topic_ids": [],
        }
        if self.inject("planner_schema", "planner", ctx):
            scene["scene_id"] = "INVALID"
        return {"root_id": ctx["root_id"], "route": ctx["route"], "scenes": [scene],
                "shortfall_reason": ""}

    def realizer(self, ctx):
        scene = ctx["scene_plan"]
        tag = f"{ctx['root_id']} {ctx['route']} {scene['scene_id']}"
        lines = [
            f"I would like to request the next appointment. Fixture {tag}.",
            f"After that request, I can offer a time next Tuesday. Fixture {tag}.",
            "Would the morning appointment leave enough time to travel?",
            "Yes, the new appointment starts at ten thirty in the morning.",
            "I will check the bus timetable before agreeing to that time.",
            "The bus timetable should let you compare the available options.",
            "The earlier bus would get me there twenty minutes before the appointment.",
            "You could use that spare time to find the entrance to the building.",
            "I am keeping this proposed appointment separate from the earlier one.",
            "I have written down that this time is offered but not yet confirmed.",
            "I will decide after checking whether I have another commitment that morning.",
            "Let me know your decision after checking that commitment.",
        ]
        turns = [{"speaker": scene["actors"][index % len(scene["actors"])],
                  "text": text if index < 2 else f"{text} Fixture {tag}, turn {index + 1}."}
                 for index, text in enumerate(lines)]
        if self.inject("realizer_schema", "realizer", ctx):
            turns[2]["speaker"] = "MOCK_INVALID_SPEAKER"
        return {"root_id": ctx["root_id"], "route": ctx["route"],
                "scene_id": scene["scene_id"], "status": "scene", "turns": turns,
                "feedback_topic_ids": [], "no_scene_reason": ""}

    def scene_reviewer(self, ctx):
        if self.inject("reviewer_schema", "scene_reviewer", ctx):
            return {"malformed_mock_review": True}
        checks = [{"development_index": index, "realized": True,
                   "evidence": [{"turn_index": index,
                                 "quote": ctx["new_scene"]["turns"][index - 1]["text"]}],
                   "reason": "Scripted review cites the corresponding fixture turn."}
                  for index in range(1, len(ctx["scene_plan"]["developments"]) + 1)]
        semantic_reject = self.inject("reviewer_semantic", "scene_reviewer", ctx)
        if semantic_reject:
            checks[0].update(realized=False, evidence=[], reason="Scripted unmet development.")
        return {"root_id": ctx["root_id"], "route": ctx["route"], "scene_id": ctx["scene_id"],
                "checks": checks, "new_event_or_progress": not semantic_reject,
                "consistent_with_history": True, "duplicates_existing_event": False,
                "reason": "MOCK ONLY: this verdict exercises acceptance/repair control flow."}


class EndToEndMockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r02-v5-offline-test-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        for name in ("sources", "feedback"):
            shutil.copytree(PREFLIGHT / name, self.run / name)
        shutil.copy2(PREFLIGHT / "MANIFEST.json", self.run / "MANIFEST.json")
        # A future accidental transport change must fail before any socket opens.
        self.network_guard = patch("socket.socket", side_effect=AssertionError("network_forbidden_in_mock"))
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)

    def exercise(self, *, scenes=1, mode=None, all_roots=False):
        fake = ScriptedProvider(mode)
        caller = runner.Caller.__new__(runner.Caller)
        caller.run, caller.limit = self.run, 200
        caller.profile, caller.factory = object(), fake.factory
        caller.capacity = lambda *_args: {"status": "PASS", "mock_capacity_only": True}
        caller.annotate, caller.tokenizer = lambda value: value, object()
        caller.http_counter = caller.request_counter = 0
        roots = runner.base.ROOTS if all_roots else runner.base.ROOTS[:1]
        results = []
        for root in roots:
            results.extend(runner.run_root(caller, self.run, root, fixture_cards(scenes),
                                           target_scenes=scenes))
        for row in results:
            self.assertEqual(row["accepted_scenes"], scenes, row)
            self.assertEqual(row["failures"], [], row)
            self.assertEqual(row["added_turns"], 12 * scenes)
        self.assertEqual(list(self.run.rglob("ERROR.json")), [])

        for root in roots:
            frozen = runner.base.read(self.run / "sources" / f"{root}.json")["turns"]
            for letter in ("A", "B"):
                folder = self.run / "roots" / root / letter
                self.assertEqual(runner.base.read(folder / "INPUT.json")["turns"], frozen)
                turns = runner.base.read(folder / "cumulative_dialogue.json")["turns"]
                self.assertEqual(turns[:len(frozen)], frozen)
                self.assertTrue(all(turn["turn_ref"].startswith(f"R02:{letter}:")
                                    for turn in turns[len(frozen):]))
            branches = runner.base.read(self.run / "roots" / root / "BRANCHES.json")
            self.assertTrue(branches["independent_branches"])
            self.assertIsNone(branches["merged_dialogue"])

        requests = sorted((self.run / "requests").iterdir())
        self.assertEqual(len(requests), caller.request_counter)
        self.assertEqual(len(list(self.run.rglob("HTTP_ATTEMPT*.json"))), caller.http_counter)
        for directory in requests:
            payload = runner.base.read(directory / "REQUEST.json")
            proof = runner.base.read(directory / "PROOF.json")
            self.assertTrue((directory / "RESPONSE.json").exists())
            self.assertEqual(proof["payload_sha256"], runner.base.digest(payload))
            for attempt_path in directory.glob("HTTP_ATTEMPT*.json"):
                attempt = runner.base.read(attempt_path)
                self.assertTrue(attempt["actual_payload_equals_frozen"])
                self.assertEqual(attempt["payload_sha256"], runner.base.digest(payload))
            if proof["role"] in ("planner", "realizer"):
                receipt = proof["feedback_receipt"]
                self.assertTrue(receipt["feedback_appended"])
                self.assertFalse(receipt["current_generation_feedback_used"])
                self.assertEqual(receipt["request_payload_sha256"], runner.base.digest(payload))
                suffix = (self.run / "feedback" / proof["root_id"] /
                          f"{proof['route']}__{proof['role']}.txt").read_text()
                self.assertTrue(payload["messages"][0]["content"].endswith(suffix))
                context = input_context(payload["messages"][0]["content"])
                runner.branch_history_ok(self.run, proof["root_id"], proof["route"],
                                         context["cumulative_history"])
        # B's reviewer sees A solely as a duplicate-comparison corpus.
        reviewers_B = [row for row in fake.observed if row["role"] == "scene_reviewer"
                       and row["context"]["route"] == runner.ROUTES[1]]
        self.assertTrue(reviewers_B)
        self.assertTrue(all(len(row["context"]["other_branch_added_turns_for_review_only"])
                            == 12 * scenes for row in reviewers_B))
        return fake, caller, requests

    def test_three_roots_one_scene_each_route_with_real_frozen_feedback(self):
        fake, caller, _ = self.exercise(all_roots=True)
        self.assertEqual(caller.request_counter, 18)
        self.assertEqual(caller.http_counter, 18)
        self.assertEqual(fake.role_counts, {"planner": 6, "realizer": 6, "scene_reviewer": 6})

    def test_three_roots_three_scene_slots_preserve_cumulative_branch_history(self):
        fake, caller, _ = self.exercise(scenes=3, all_roots=True)
        self.assertEqual(caller.request_counter, 54)
        self.assertEqual(caller.http_counter, 54)
        for observation in fake.observed:
            if observation["role"] not in ("planner", "realizer"):
                continue
            ctx = observation["context"]
            frozen = runner.base.read(self.run / "sources" / f"{ctx['root_id']}.json")["turns"]
            expected_additions = (int(ctx["target_scene_id"][1:]) - 1) * 12
            self.assertEqual(len(ctx["cumulative_history"]), len(frozen) + expected_additions)

    def test_transient_503_reuses_identical_payload_then_finishes_both_routes(self):
        fake, caller, requests = self.exercise(mode="http_503")
        self.assertEqual(caller.request_counter, 6)
        self.assertEqual(caller.http_counter, 7)
        self.assertEqual(fake.observed[0]["payload"], fake.observed[1]["payload"])
        response = runner.base.read(requests[0] / "RESPONSE.json")
        self.assertEqual([row["http_status"] for row in response["attempts"]], [503, 200])
        self.assertTrue((requests[0] / "HTTP_ATTEMPT_02.json").exists())

    def test_invalid_plan_retries_planner_without_rewriting_history(self):
        fake, caller, _ = self.exercise(mode="planner_schema")
        self.assertEqual(caller.request_counter, 7)
        self.assertEqual(fake.role_counts, {"planner": 3, "realizer": 2, "scene_reviewer": 2})
        second = [row for row in fake.observed if row["role"] == "planner"][1]["context"]
        self.assertIn("target_scene_id_mismatch", second["targeted_repair"]["failure"])

    def test_invalid_speaker_repairs_realizer_before_review(self):
        fake, caller, _ = self.exercise(mode="realizer_schema")
        self.assertEqual(caller.request_counter, 7)
        self.assertEqual(fake.role_counts, {"planner": 2, "realizer": 3, "scene_reviewer": 2})
        second = [row for row in fake.observed if row["role"] == "realizer"][1]["context"]
        self.assertEqual(second["targeted_repair"]["invalid_speakers"],
                         [{"turn_position": 3, "speaker": "MOCK_INVALID_SPEAKER"}])

    def test_malformed_review_retries_reviewer_without_rewriting_scene(self):
        fake, caller, _ = self.exercise(mode="reviewer_schema")
        self.assertEqual(caller.request_counter, 7)
        self.assertEqual(fake.role_counts, {"planner": 2, "realizer": 2, "scene_reviewer": 3})
        reviews = [row for row in fake.observed if row["role"] == "scene_reviewer"]
        self.assertEqual(reviews[0]["context"], reviews[1]["context"])
        self.assertIn("REVIEW_FORMAT_REPAIR", reviews[1]["payload"]["messages"][0]["content"])

    def test_semantic_rejection_returns_only_structured_findings_to_realizer(self):
        fake, caller, _ = self.exercise(mode="reviewer_semantic")
        self.assertEqual(caller.request_counter, 8)
        self.assertEqual(fake.role_counts, {"planner": 2, "realizer": 3, "scene_reviewer": 3})
        second = [row for row in fake.observed if row["role"] == "realizer"][1]["context"]
        self.assertEqual(second["targeted_repair"]["unmet_development_indices"], [1])
        self.assertNotIn("other_branch_added_turns_for_review_only", second)
        self.assertNotIn("Scripted unmet development", json.dumps(second["targeted_repair"]))


class LiveControlFlowMockTests(unittest.TestCase):
    """Exercise real prepare, extraction validation, reservation, and final report."""

    def exercise_live(self, *, scenes, omit_first_source=False):
        with tempfile.TemporaryDirectory(prefix="r02-v5-live-control-mock-") as temp:
            temp = Path(temp)
            delivery = temp / "offline-deliveries/h4_r02_stage_repair_v5"
            delivery.mkdir(parents=True)
            # prepare binds code files; supply a frozen temporary copy.
            for path in runner.HERE.glob("*.py"):
                shutil.copy2(path, delivery / path.name)
            unique_dir = delivery.parent / "h4_r02_source_unique_v2"
            unique_dir.mkdir()
            shutil.copy2(runner.HERE.parent / "h4_r02_source_unique_v2/unique_events.py",
                         unique_dir / "unique_events.py")
            run = delivery / "runs/MOCK_ONLY"
            fake = ScriptedProvider()
            # Reservation must precede even the first extraction request.
            def assert_reservation_before_request():
                self.assertTrue((run / "SOURCE_RESERVATIONS.json").exists())
                self.assertTrue((run / "LIVE_STARTED.json").exists())
            fake.before_request = assert_reservation_before_request

            def fake_init(caller, run_path):
                caller.run = run_path
                caller.limit = runner.base.read(run_path / "MANIFEST.json")["http_max"]
                caller.profile, caller.factory = object(), fake.factory
                caller.capacity = lambda *_args: {"status": "PASS", "mock_capacity_only": True}
                caller.annotate, caller.tokenizer = lambda value: value, object()
                caller.http_counter = caller.request_counter = 0

            prior = {"prior_runs": [], "prior_evidence": [], "accepted_uses": [],
                     "reserved_uses": [], "consumption_policy": "MOCK_ONLY_EMPTY_INVENTORY"}
            with patch.object(runner, "HERE", delivery), \
                    patch.object(runner, "LOCK_PATH", temp / "MOCK_ONLY.lock"), \
                    patch.object(runner.source_inventory, "collect_prior", return_value=prior), \
                    patch.object(runner.Caller, "__init__", fake_init), \
                    patch("socket.socket", side_effect=AssertionError("network_forbidden_in_mock")):
                prepared = runner.prepare(run, scenes=scenes)
                self.assertEqual(prepared["http_calls"], 0)
                self.assertEqual(prepared["assigned_B_slots"], scenes * 3)
                manifest, allocation = runner.verify(run)
                if omit_first_source:
                    fake.omit_source_id = allocation["assignments"][runner.base.ROOTS[0]][0]["source_id"]
                report = runner.live(run)
                runner.verify(run)
                self.assertEqual(runner.base.read(run / "REPORT.json"), report)
                self.assertLessEqual(report["http_attempts"], manifest["http_max"])
                with self.assertRaisesRegex(ValueError, "already_started_no_automatic_replay"):
                    runner.live(run)

            omitted = int(omit_first_source)
            self.assertEqual(report["status"], "COMPLETE_WITH_SHORTFALL" if omitted else "COMPLETE")
            self.assertEqual(report["accepted_scenes"], 6 * scenes - omitted)
            self.assertEqual(len(report["accepted_uses"]), 3 * scenes - omitted)
            expected_http = 3 * scenes + 3 * (6 * scenes - omitted)
            self.assertEqual(report["http_attempts"], expected_http)
            self.assertEqual(report["http_request_reservations"], expected_http)
            self.assertEqual(report["http_attempt_log_count"], expected_http)
            self.assertEqual(fake.role_counts["dataset_extractor"], 3 * scenes)
            self.assertEqual(report["solver_calls"], 0)
            self.assertEqual(report["question_calls"], 0)
            self.assertFalse(report["training_started"])
            self.assertFalse(report["semantic_certified"])
            self.assertTrue(report["branches_remain_separate"])
            reservations = runner.base.read(run / "SOURCE_RESERVATIONS.json")["reservations"]
            self.assertEqual(len(reservations), 3 * scenes)
            self.assertEqual(list(run.rglob("ERROR.json")), [])
            self.assertTrue(all(Path(row["run"]).is_relative_to(temp) for row in reservations))
            for request in (run / "requests").iterdir():
                payload = runner.base.read(request / "REQUEST.json")
                proof = runner.base.read(request / "PROOF.json")
                self.assertEqual(proof["payload_sha256"], runner.base.digest(payload))
                attempt = runner.base.read(request / "HTTP_ATTEMPT.json")
                self.assertEqual(attempt["payload_sha256"], runner.base.digest(payload))
                self.assertTrue((request / "RESPONSE.json").exists())
                if proof["role"] in ("planner", "realizer"):
                    receipt = proof["feedback_receipt"]
                    self.assertTrue(receipt["feedback_appended"])
                    self.assertEqual(receipt["request_payload_sha256"], runner.base.digest(payload))
                    suffix = (run / "feedback" / proof["root_id"] /
                              f"{proof['route']}__{proof['role']}.txt").read_text()
                    self.assertEqual(payload["messages"][0]["content"].count(suffix), 1)
            if omit_first_source:
                row = next(row for row in report["routes"] if row["root_id"] == runner.base.ROOTS[0]
                           and row["route"] == runner.ROUTES[1])
                self.assertEqual([scene["scene_id"] for scene in row["scene_metadata"]], ["S02", "S03"])
                self.assertEqual(row["failures"][0]["scene_id"], "S01")
                self.assertEqual(row["failures"][0]["reason"], "reserved_source_card_extraction_failed")
                # The omitted source remains reserved despite no accepted scene.
                self.assertIn(fake.omit_source_id, {row["source_id"] for row in reservations})

    def test_prepare_live_reserve_extract_and_report_three_roots_one_scene_per_route(self):
        self.exercise_live(scenes=1)

    def test_missing_first_card_preserves_later_slots_and_reservation(self):
        self.exercise_live(scenes=3, omit_first_source=True)


if __name__ == "__main__":
    unittest.main()
