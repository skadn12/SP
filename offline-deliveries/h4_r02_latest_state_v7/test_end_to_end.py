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


PREFLIGHT = runner.HERE.parent / "h4_r02_stage_repair_v5/runs/r02_stage_repair_v5_preflight_02"
HISTORICAL_V5 = runner.HERE.parent / "h4_r02_stage_repair_v5/runs/r02_stage_repair_v5_01"


def input_context(prompt):
    marker = next(marker for marker in (
        "\nINPUT_JSON\n", "\nREVIEW INPUT DATA:\n", "\nSOURCE_ITEMS_JSON:\n", "\nSTATE_INPUT_JSON\n"
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
                role = ("state_extractor" if "\nSTATE_INPUT_JSON\n" in prompt else
                        "dataset_extractor" if "source_items" in ctx else ctx.get("role", "scene_reviewer"))
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
            "state_transition": {
                "basis_fact_ids": [ctx["latest_state"]["facts"][0]["fact_id"]],
                "new_event": "MOCK ONLY: a separate future appointment is proposed.",
                "temporal_placement": "MOCK ONLY: later than the cited historical report.",
            },
        }
        if self.inject("planner_schema", "planner", ctx):
            scene["scene_id"] = "INVALID"
        return {"root_id": ctx["root_id"], "route": ctx["route"], "scenes": [scene],
                "shortfall_reason": ""}

    def state_extractor(self, ctx):
        if (self.inject("state_schema", "state_extractor", ctx) or
                self.mode == "state_schema_exhausted" and ctx["route"] == runner.ROUTES[1]):
            return {"malformed_mock_state": True}
        anchor = ctx["cumulative_history"][-1]
        # The fixture copies evidence without claiming this is a model-derived,
        # correct summary. It exercises the actual extraction/validation stage.
        return {"root_id": ctx["root_id"], "route": ctx["route"], "scene_id": ctx["scene_id"],
                "facts": [{"fact_id": "f1", "entity": anchor["speaker"],
                           "event": "MOCK ONLY: latest cited historical report",
                           "status": "condition", "statement": anchor["text"],
                           "evidence": [{"turn_ref": anchor["turn_ref"], "quote": anchor["text"]}]}],
                "conflicts": [], "unknowns": []}

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
        result = {"root_id": ctx["root_id"], "route": ctx["route"], "scene_id": ctx["scene_id"],
                  "checks": checks, "new_event_or_progress": not semantic_reject,
                  "consistent_with_history": True, "duplicates_existing_event": False,
                  "state_checks": [{
                      "fact_id": fact["fact_id"],
                      "verdict": "contradicted" if self.mode == "reviewer_state_reject" else "consistent",
                      "history_evidence": fact["evidence"],
                      "new_evidence": [{"turn_index": 1, "quote": ctx["new_scene"]["turns"][0]["text"]}],
                      "reason": "MOCK ONLY: scripted state contradiction." if self.mode == "reviewer_state_reject"
                                else "MOCK ONLY: scripted consistency, not semantic quality evidence.",
                  } for fact in ctx["latest_state"]["facts"]],
                  "reason": "MOCK ONLY: this verdict exercises acceptance/repair control flow."}
        card = ctx.get("dataset_card_guidance")
        if card is not None:
            references = [("step", step["step_id"]) for step in card["steps"]]
            references += [("dependency", index) for index in range(1, len(card["dependencies"]) + 1)]
            result["card_checks"] = [
                {"kind": kind, "id": identifier, "realized": True,
                 "evidence": [{"turn_index": 1, "quote": ctx["new_scene"]["turns"][0]["text"]}],
                 "reason": "MOCK ONLY: scripted card verdict, not a semantic assessment."}
                for kind, identifier in references
            ]
            if self.mode == "reviewer_card_reject":
                result["card_checks"][0].update(
                    realized=False, evidence=[], reason="Scripted unmet card step despite a realized plan.")
        return result


class EndToEndMockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r02-v7-offline-test-")
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
                for scene_number in range(1, scenes + 1):
                    snapshot = runner.base.read(folder / f"S{scene_number:02d}.LATEST_STATE.json")
                    self.assertEqual(snapshot["root_id"], root)
                    self.assertEqual(snapshot["scene_id"], f"S{scene_number:02d}")
                    self.assertEqual(snapshot["facts"][0]["fact_id"], "f1")
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
            if proof["role"] == "state_extractor":
                context = input_context(payload["messages"][0]["content"])
                self.assertNotIn("other_branch_added_turns_for_review_only", context)
                self.assertNotIn("feedback", context)
                for generator_role in ("planner", "realizer"):
                    suffix = (self.run / "feedback" / proof["root_id"] /
                              f"{proof['route']}__{generator_role}.txt").read_text()
                    self.assertNotIn(suffix, payload["messages"][0]["content"])
                runner.branch_history_ok(self.run, proof["root_id"], proof["route"],
                                         context["cumulative_history"])
        snapshots = {}
        for observation in fake.observed:
            role, ctx = observation["role"], observation["context"]
            scene_id = ctx.get("target_scene_id", ctx.get("scene_id"))
            key = ctx["root_id"], ctx["route"], scene_id
            if role == "state_extractor":
                snapshots[key] = ctx["cumulative_history"][-1]
                continue
            self.assertIn(key, snapshots, "Every scene must extract state before Planner.")
            anchor = snapshots[key]
            self.assertEqual(ctx["latest_state"]["facts"][0]["statement"], anchor["text"])
            self.assertEqual(ctx["latest_state"]["facts"][0]["evidence"],
                             [{"turn_ref": anchor["turn_ref"], "quote": anchor["text"]}])
        # B's reviewer sees A solely as a duplicate-comparison corpus.
        reviewers_B = [row for row in fake.observed if row["role"] == "scene_reviewer"
                       and row["context"]["route"] == runner.ROUTES[1]]
        self.assertTrue(reviewers_B)
        self.assertTrue(all(len(row["context"]["other_branch_added_turns_for_review_only"])
                            == 12 * scenes for row in reviewers_B))
        for directory in requests:
            proof = runner.base.read(directory / "PROOF.json")
            if proof["role"] == "scene_reviewer" and proof["route"] == runner.ROUTES[1]:
                review = runner.base.read(directory / "REVIEW.json")
                self.assertEqual(review["status"], "PASS")
                self.assertEqual({(check["kind"], check["id"]) for check in review["parsed"]["card_checks"]},
                                 {("step", "s1"), ("step", "s2"), ("dependency", 1)})
        return fake, caller, requests

    def test_three_roots_one_scene_each_route_with_real_frozen_feedback(self):
        fake, caller, _ = self.exercise(all_roots=True)
        self.assertEqual(caller.request_counter, 24)
        self.assertEqual(caller.http_counter, 24)
        self.assertEqual(fake.role_counts, {"state_extractor": 6, "planner": 6, "realizer": 6, "scene_reviewer": 6})

    def test_three_roots_three_scene_slots_preserve_cumulative_branch_history(self):
        fake, caller, _ = self.exercise(scenes=3, all_roots=True)
        self.assertEqual(caller.request_counter, 72)
        self.assertEqual(caller.http_counter, 72)
        for observation in fake.observed:
            if observation["role"] not in ("state_extractor", "planner", "realizer"):
                continue
            ctx = observation["context"]
            frozen = runner.base.read(self.run / "sources" / f"{ctx['root_id']}.json")["turns"]
            scene_id = ctx.get("target_scene_id", ctx.get("scene_id"))
            expected_additions = (int(scene_id[1:]) - 1) * 12
            self.assertEqual(len(ctx["cumulative_history"]), len(frozen) + expected_additions)

    def test_invalid_state_retries_extractor_before_first_planner(self):
        fake, caller, _ = self.exercise(mode="state_schema")
        self.assertEqual(caller.request_counter, 9)
        self.assertEqual(fake.role_counts,
                         {"state_extractor": 3, "planner": 2, "realizer": 2, "scene_reviewer": 2})
        self.assertEqual([row["role"] for row in fake.observed[:3]],
                         ["state_extractor", "state_extractor", "planner"])
        first, repaired = fake.observed[0]["context"], fake.observed[1]["context"]
        self.assertEqual(first["cumulative_history"], repaired["cumulative_history"])
        self.assertTrue(repaired["targeted_repair"]["errors"])

    def test_transient_503_reuses_identical_payload_then_finishes_both_routes(self):
        fake, caller, requests = self.exercise(mode="http_503")
        self.assertEqual(caller.request_counter, 8)
        self.assertEqual(caller.http_counter, 9)
        self.assertEqual(fake.observed[1]["payload"], fake.observed[2]["payload"])
        response = runner.base.read(requests[1] / "RESPONSE.json")
        self.assertEqual([row["http_status"] for row in response["attempts"]], [503, 200])
        self.assertTrue((requests[1] / "HTTP_ATTEMPT_02.json").exists())

    def test_invalid_plan_retries_planner_without_rewriting_history(self):
        fake, caller, _ = self.exercise(mode="planner_schema")
        self.assertEqual(caller.request_counter, 9)
        self.assertEqual(fake.role_counts, {"state_extractor": 2, "planner": 3, "realizer": 2, "scene_reviewer": 2})
        second = [row for row in fake.observed if row["role"] == "planner"][1]["context"]
        self.assertIn("target_scene_id_mismatch", second["targeted_repair"]["failure"])

    def test_invalid_speaker_repairs_realizer_before_review(self):
        fake, caller, _ = self.exercise(mode="realizer_schema")
        self.assertEqual(caller.request_counter, 9)
        self.assertEqual(fake.role_counts, {"state_extractor": 2, "planner": 2, "realizer": 3, "scene_reviewer": 2})
        second = [row for row in fake.observed if row["role"] == "realizer"][1]["context"]
        self.assertEqual(second["targeted_repair"]["invalid_speakers"],
                         [{"turn_position": 3, "speaker": "MOCK_INVALID_SPEAKER"}])

    def test_malformed_review_retries_reviewer_without_rewriting_scene(self):
        fake, caller, _ = self.exercise(mode="reviewer_schema")
        self.assertEqual(caller.request_counter, 9)
        self.assertEqual(fake.role_counts, {"state_extractor": 2, "planner": 2, "realizer": 2, "scene_reviewer": 3})
        reviews = [row for row in fake.observed if row["role"] == "scene_reviewer"]
        self.assertEqual(reviews[0]["context"], reviews[1]["context"])
        self.assertIn("REVIEW_FORMAT_REPAIR", reviews[1]["payload"]["messages"][0]["content"])

    def test_semantic_rejection_returns_only_structured_findings_to_realizer(self):
        fake, caller, _ = self.exercise(mode="reviewer_semantic")
        self.assertEqual(caller.request_counter, 10)
        self.assertEqual(fake.role_counts, {"state_extractor": 2, "planner": 2, "realizer": 3, "scene_reviewer": 3})
        second = [row for row in fake.observed if row["role"] == "realizer"][1]["context"]
        self.assertEqual(second["targeted_repair"]["unmet_development_indices"], [1])
        self.assertNotIn("other_branch_added_turns_for_review_only", second)
        self.assertNotIn("Scripted unmet development", json.dumps(second["targeted_repair"]))


class LiveControlFlowMockTests(unittest.TestCase):
    """Exercise real prepare, extraction validation, reservation, and final report."""

    def exercise_live(self, *, scenes, omit_first_source=False, roots=None, routes=None,
                      historical_inventory=False, mode=None):
        selected_roots = list(runner.base.ROOTS if roots is None else roots)
        selected_routes = tuple(runner.ROUTES if routes is None else routes)
        b_slots = scenes * len(selected_roots)
        scene_slots = b_slots * len(selected_routes)
        with tempfile.TemporaryDirectory(prefix="r02-v7-live-control-mock-") as temp:
            temp = Path(temp)
            delivery = temp / "offline-deliveries/h4_r02_latest_state_v7"
            delivery.mkdir(parents=True)
            # prepare binds code files; supply a frozen temporary copy.
            for path in runner.HERE.glob("*.py"):
                shutil.copy2(path, delivery / path.name)
            unique_dir = delivery.parent / "h4_r02_source_unique_v2"
            unique_dir.mkdir()
            shutil.copy2(runner.HERE.parent / "h4_r02_source_unique_v2/unique_events.py",
                         unique_dir / "unique_events.py")
            comparison_turns = {}
            if selected_routes == (runner.ROUTES[1],):
                for root in selected_roots:
                    # Copy the historical reviewer-only corpus into the isolated
                    # layout expected by real prepare(); never modify its source.
                    original = HISTORICAL_V5 / "roots" / root / "A/cumulative_dialogue.json"
                    copied = (delivery.parent / "h4_r02_stage_repair_v5/runs/r02_stage_repair_v5_01"
                              / "roots" / root / "A/cumulative_dialogue.json")
                    copied.parent.mkdir(parents=True)
                    shutil.copy2(original, copied)
            run = delivery / "runs/MOCK_ONLY"
            fake = ScriptedProvider(mode)
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

            collect_prior = runner.source_inventory.collect_prior
            generations = {
                key: getattr(runner.source_inventory, key.upper()) if historical_inventory else temp / key
                for key in ("v1_runs", "v2_runs", "v4_runs", "v5_runs", "v6_runs")
            }
            # Explicitly isolate v5/v6 as well as all other generations for ordinary
            # fixtures. The B-only regression reads real history but writes only
            # current reservations under the temporary delivery.
            def collect_test_inventory():
                return collect_prior(delivery / "runs", **generations)

            prior = collect_test_inventory()
            self.assertEqual(len(prior["accepted_uses"]), 25 if historical_inventory else 0)
            used_keys = {runner.source_inventory.event_key(row) for row in prior["accepted_uses"]}
            with patch.object(runner, "HERE", delivery), \
                    patch.object(runner, "LOCK_PATH", temp / "MOCK_ONLY.lock"), \
                    patch.object(runner.source_inventory, "collect_prior", side_effect=collect_test_inventory), \
                    patch.object(runner.Caller, "__init__", fake_init), \
                    patch("socket.socket", side_effect=AssertionError("network_forbidden_in_mock")):
                prepared = runner.prepare(run, scenes=scenes, roots=roots, routes=routes)
                self.assertEqual(prepared["http_calls"], 0)
                self.assertEqual(prepared["assigned_B_slots"], b_slots)
                self.assertEqual(prepared["independent_branch_previews"], 2 * len(selected_roots) * len(selected_routes))
                manifest, allocation = runner.verify(run)
                self.assertEqual(manifest["roots"], selected_roots)
                self.assertEqual(manifest["selected_routes"], list(selected_routes))
                self.assertEqual(set(allocation["assignments"]), set(selected_roots))
                allocated_keys = {runner.source_inventory.event_key(item)
                                  for rows in allocation["assignments"].values() for item in rows}
                self.assertFalse(allocated_keys & used_keys)
                self.assertEqual(len(runner.base.read(run / "dataset_sources.json")["items"]), b_slots)
                if historical_inventory:
                    self.assertEqual(len(allocation["prior"]["accepted_uses"]), 25)
                    for row in prior["accepted_uses"]:
                        candidate = {key: value for key, value in row.items() if key not in {"run", "root_id"}}
                        with self.subTest(consumed_source=row["source_id"]), self.assertRaisesRegex(
                                ValueError, "source_event_already_consumed"):
                            runner.source_inventory.validate_available({selected_roots[0]: [candidate]}, prior)
                if selected_routes == (runner.ROUTES[1],):
                    for root in selected_roots:
                        compared = runner.base.read(run / "comparison_corpora" / f"{root}.A.json")
                        comparison_turns[root] = compared["turns"]
                        self.assertTrue(compared["turns"])
                        self.assertEqual(compared["usage"], "duplicate_checker_and_reviewer_only")
                if omit_first_source:
                    fake.omit_source_id = allocation["assignments"][selected_roots[0]][0]["source_id"]
                report = runner.live(run)
                runner.verify(run)
                self.assertEqual(runner.base.read(run / "REPORT.json"), report)
                self.assertLessEqual(report["http_attempts"], manifest["http_max"])
                with self.assertRaisesRegex(ValueError, "already_started_no_automatic_replay"):
                    runner.live(run)

            omitted = int(omit_first_source)
            review_rejects = b_slots - omitted if mode in ("reviewer_card_reject", "reviewer_state_reject") else 0
            state_rejects = b_slots - omitted if mode == "state_schema_exhausted" else 0
            rejected = review_rejects + state_rejects
            self.assertEqual(report["status"], "COMPLETE_WITH_SHORTFALL" if omitted + rejected else "COMPLETE")
            self.assertEqual(report["accepted_scenes"], scene_slots - omitted - rejected)
            self.assertEqual(len(report["accepted_uses"]), b_slots - omitted - rejected)
            expected_http = b_slots + 4 * (scene_slots - omitted - state_rejects) + 2 * review_rejects + 2 * state_rejects
            self.assertEqual(report["http_attempts"], expected_http)
            self.assertEqual(report["http_request_reservations"], expected_http)
            self.assertEqual(report["http_attempt_log_count"], expected_http)
            self.assertEqual(fake.role_counts["dataset_extractor"], b_slots)
            self.assertEqual(fake.role_counts["state_extractor"], scene_slots - omitted + state_rejects)
            self.assertEqual(report["solver_calls"], 0)
            self.assertEqual(report["question_calls"], 0)
            self.assertFalse(report["training_started"])
            self.assertFalse(report["semantic_certified"])
            self.assertTrue(report["branches_remain_separate"])
            reservations = runner.base.read(run / "SOURCE_RESERVATIONS.json")["reservations"]
            self.assertEqual(len(reservations), b_slots)
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
                    self.assertIn("latest_state", input_context(payload["messages"][0]["content"]))
                if proof["role"] == "state_extractor":
                    state_context = input_context(payload["messages"][0]["content"])
                    self.assertNotIn("other_branch_added_turns_for_review_only", state_context)
                    self.assertNotIn("feedback", state_context)
                    for generator_role in ("planner", "realizer"):
                        suffix = (run / "feedback" / proof["root_id"] /
                                  f"{proof['route']}__{generator_role}.txt").read_text()
                        self.assertNotIn(suffix, payload["messages"][0]["content"])
                if proof["role"] != "dataset_extractor":
                    self.assertIn(proof["root_id"], selected_roots)
                    self.assertIn(proof["route"], selected_routes)
                if proof["role"] == "scene_reviewer" and proof["route"] == runner.ROUTES[1]:
                    review = runner.base.read(request / "REVIEW.json")
                    self.assertTrue(review["schema_valid"], review)
                    self.assertEqual({(row["kind"], row["id"]) for row in review["parsed"]["card_checks"]},
                                     {("step", "s1")})
                    if mode == "reviewer_card_reject":
                        self.assertTrue(all(row["realized"] for row in review["parsed"]["checks"]))
                        self.assertTrue(review["parsed"]["new_event_or_progress"])
                        self.assertTrue(review["parsed"]["consistent_with_history"])
                        self.assertFalse(review["parsed"]["duplicates_existing_event"])
                        self.assertEqual(review["status"], "REJECT")
                        self.assertTrue(any("card_step[s1]:not_realized" in code for code in review["failures"]))
                    if mode == "reviewer_state_reject":
                        self.assertTrue(all(row["realized"] for row in review["parsed"]["checks"]))
                        self.assertTrue(all(row["realized"] for row in review["parsed"]["card_checks"]))
                        self.assertTrue(review["parsed"]["new_event_or_progress"])
                        self.assertTrue(review["parsed"]["consistent_with_history"])
                        self.assertFalse(review["parsed"]["duplicates_existing_event"])
                        self.assertEqual(review["status"], "REJECT")
                        self.assertTrue(any(code.startswith("state_fact[f1]:contradicted")
                                            for code in review["failures"]))
            self.assertEqual({row["root_id"] for row in report["routes"]}, set(selected_roots))
            self.assertEqual({row["route"] for row in report["routes"]}, set(selected_routes))
            if selected_routes == (runner.ROUTES[1],):
                for root in selected_roots:
                    self.assertFalse((run / "roots" / root / "A").exists())
                    branches = runner.base.read(run / "roots" / root / "BRANCHES.json")
                    self.assertEqual(set(branches["dialogues"]), {"B"})
                for observation in fake.observed:
                    ctx = observation["context"]
                    if observation["role"] == "dataset_extractor":
                        continue
                    self.assertEqual(ctx["route"], runner.ROUTES[1])
                    if observation["role"] == "scene_reviewer":
                        self.assertEqual(ctx["other_branch_added_turns_for_review_only"],
                                         comparison_turns[ctx["root_id"]])
                    else:
                        self.assertNotIn("other_branch_added_turns_for_review_only", ctx)
                        self.assertFalse(any(turn["turn_ref"].startswith("R02:A:")
                                             for turn in ctx["cumulative_history"]))
                        if observation["role"] != "state_extractor":
                            self.assertIn("card_mapping_context", ctx)
                        if ctx.get("targeted_repair") and mode == "reviewer_card_reject":
                            self.assertEqual(ctx["targeted_repair"]["unmet_development_indices"], [])
                            self.assertEqual(ctx["targeted_repair"]["unmet_card_elements"],
                                             [{"kind": "step", "id": "s1"}])
                            self.assertNotIn("Scripted unmet card", json.dumps(ctx["targeted_repair"]))
                if mode in ("reviewer_card_reject", "reviewer_state_reject"):
                    for root in selected_roots:
                        result = runner.base.read(run / "roots" / root / "B/RESULT.json")
                        self.assertEqual(result["accepted_scenes"], 0)
                        self.assertEqual(result["failures"][0]["reason"], "no_accepted_scene")
                        attempts = runner.base.read(run / "roots" / root / "B/S01.ATTEMPTS.json")
                        self.assertEqual(len(attempts), runner.SCENE_ATTEMPTS)
                        self.assertTrue(all(row["status"] == "REJECT" for row in attempts))
                        self.assertFalse((run / "roots" / root / "B/S01.json").exists())
                        self.assertTrue((run / "roots" / root / "B/S01.PLAN.json").exists())
                    self.assertEqual(fake.role_counts["realizer"], runner.SCENE_ATTEMPTS)
                    self.assertEqual(fake.role_counts["scene_reviewer"], runner.SCENE_ATTEMPTS)
                if mode == "state_schema_exhausted":
                    self.assertEqual(fake.role_counts["state_extractor"], 2)
                    self.assertEqual(fake.role_counts["planner"], 0)
                    self.assertEqual(fake.role_counts["realizer"], 0)
                    self.assertEqual(fake.role_counts["scene_reviewer"], 0)
                    for root in selected_roots:
                        result = runner.base.read(run / "roots" / root / "B/RESULT.json")
                        self.assertEqual(result["accepted_scenes"], 0)
                        self.assertEqual(result["failures"][0]["reason"], "state_rejected_after_retry")
                        self.assertFalse((run / "roots" / root / "B/S01.PLAN.json").exists())
                        self.assertFalse((run / "roots" / root / "B/S01.json").exists())
            consumed_after = collect_test_inventory()
            self.assertEqual(len(consumed_after["accepted_uses"]), len(used_keys) + b_slots)
            self.assertTrue(used_keys <= {runner.source_inventory.event_key(row)
                                         for row in consumed_after["accepted_uses"]})
            if omit_first_source:
                row = next(row for row in report["routes"] if row["root_id"] == selected_roots[0]
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

    def test_one_root_b_only_prepares_one_unused_source_and_never_generates_a(self):
        self.exercise_live(scenes=1, roots=runner.base.ROOTS[:1], routes=(runner.ROUTES[1],),
                           historical_inventory=True)

    def test_negative_card_review_rejects_a_realized_plan_and_keeps_source_consumed(self):
        self.exercise_live(scenes=1, roots=runner.base.ROOTS[:1], routes=(runner.ROUTES[1],),
                           historical_inventory=True, mode="reviewer_card_reject")

    def test_negative_state_check_cannot_be_overridden_by_all_other_positive_verdicts(self):
        self.exercise_live(scenes=1, roots=runner.base.ROOTS[:1], routes=(runner.ROUTES[1],),
                           historical_inventory=True, mode="reviewer_state_reject")

    def test_state_extraction_exhaustion_stops_before_planner_and_keeps_source_consumed(self):
        self.exercise_live(scenes=1, roots=runner.base.ROOTS[:1], routes=(runner.ROUTES[1],),
                           historical_inventory=True, mode="state_schema_exhausted")


if __name__ == "__main__":
    unittest.main()
