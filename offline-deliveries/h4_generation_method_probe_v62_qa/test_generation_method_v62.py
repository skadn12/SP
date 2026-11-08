"""Offline method comparison: real prompt replay and bounded mocked HTTP execution."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import generate as gen
import qa_transport
import evaluate_and_pool as pipeline
import test_generate_adapter as fixtures


class GenerationMethodTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CandidateGenerationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def prepare(self, method="staged", name="generation"):
        original = gen.prepare
        def bound(*args, **kwargs):
            return original(*args, **kwargs, generation_method=method)
        with patch.object(gen, "prepare", side_effect=bound):
            return self.fixture.prepare(run_name=name, comparison_id="paired-fixture")

    def plan(self, context, count=1):
        turn = self.fixture.turns[1]
        return {"root_id": self.root, "plans": [{"candidate_id": qid,
            "topic": "Synthetic source worksheet", "evidence": [{
                "turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "quote": turn["text"],
                "source_fact": "Synthetic supported fact.", "contradicted_alternative": None,
                "contradiction_reason": None} for _ in range(count)]}
            for qid in context["candidate_ids"]]}

    def exercise(self, *, bad_plan=False, failed_http=False, repair=False, missing_usage=False):
        run = self.prepare()
        fixture, root, owner = self.fixture, self.root, self
        seen = []
        class MockCaller:
            def __init__(self, run):
                self.run, self.counter = run, 0
            def call(self, label, payload, proof):
                self.counter += 1
                seen.append((deepcopy(payload), deepcopy(proof)))
                directory = self.run / "requests" / f"{self.counter:03d}-{label}"
                gen.write(directory / "REQUEST.json", payload)
                gen.write(directory / "PROOF.json", {**proof, "payload_sha256": gen.base.digest(payload)})
                gen.write(directory / "HTTP_ATTEMPT.json", {"synthetic_test_only": True,
                    "actual_payload_equals_frozen": True, "payload_sha256": gen.base.digest(payload)})
                if proof["role"] == "evidence_planner" and failed_http:
                    gen.write(directory / "ERROR.json", {"error_type": "SyntheticTransportFailure"})
                    return None, directory, {}
                context = fixtures.input_json(payload, "\nINPUT_JSON\n")
                if proof["role"] == "evidence_planner":
                    raw = "{bad json" if bad_plan else json.dumps(owner.plan(context), ensure_ascii=False, indent=2)
                else:
                    questions = []
                    for qid in context["candidate_ids"]:
                        options = [{"turn_ref": t["turn_ref"], "speaker": t["speaker"], "quote": t["text"],
                            "proposition": f"Synthetic {qid} fact {i}.", "relation": "supports",
                            "rationale": "Synthetic exact quote."} for i, t in enumerate(fixture.turns[1:])]
                        if repair and label.endswith("generate-1") and qid.endswith("-02"):
                            options[0]["proposition"] = ""
                        questions.append({"candidate_id": qid, "topic": "Synthetic coherent fixture",
                            "question": "Fixture? Select all that apply.", "options": options})
                    raw = json.dumps({"root_id": root, "questions": questions, "abstentions": []})
                metadata = {"requested_model": gen.MODEL, "returned_model": gen.MODEL,
                    "finish_reason": "stop", "http_status": 200, "synthetic_test_only": True}
                if not missing_usage:
                    metadata["usage"] = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
                gen.write(directory / "RESPONSE.json", {"text": raw, "metadata": metadata})
                return raw, directory, metadata
        with patch.object(qa_transport, "Caller", MockCaller), patch.object(qa_transport,
                "shared_lock", return_value=nullcontext()), redirect_stdout(io.StringIO()):
            report = gen.generate(run)
        return run, report, seen

    def test_direct_payload_byte_identical_to_frozen_v61(self):
        run = self.prepare("direct")
        old_path = Path(gen.__file__).parent.parent / "h4_multiselect_probe_v61_qa/generate.py"
        paths = sys.path[:]
        try:
            spec = importlib.util.spec_from_file_location("v61_frozen_direct_equivalence", old_path)
            old = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(old)
        finally:
            sys.path[:] = paths
        context = gen.initial_context(run, self.root, "multi_select")
        self.assertEqual(gen.make_payload(run, self.root, "multi_select", context),
                         old.make_payload(run, self.root, "multi_select", context))
        self.assertEqual(gen.generation_budget(1, 2, "direct"), old.generation_budget(1, 2))
        self.assertEqual(gen.read(run / "MANIFEST.json")["generation_method"], "direct")

    def test_staged_keeps_every_source_prior_feedback_and_raw_plan_and_final_six_truths(self):
        run, report, seen = self.exercise()
        self.assertEqual([proof["role"] for _, proof in seen], ["evidence_planner", "question_generator"])
        original = gen.initial_context(run, self.root, "multi_select")
        planner = fixtures.input_json(seen[0][0], "\nINPUT_JSON\n")
        compose = fixtures.input_json(seen[1][0], "\nINPUT_JSON\n")
        self.assertEqual(planner, original)
        for key, value in original.items():
            self.assertEqual(compose[key], value)
        plan = gen.verified_evidence_plan(run, self.root)
        self.assertEqual(compose["frozen_evidence_plan_raw_json"], plan["raw_text"])
        suffix = gen.legacy.feedback.build_feedback_suffix(run / "feedback", self.root, "writer", gen.TARGET)
        self.assertTrue(seen[0][0]["messages"][0]["content"].endswith(suffix))
        self.assertIn(suffix, seen[1][0]["messages"][0]["content"])
        self.assertTrue(all(payload["max_tokens"] == 4096 for payload, _ in seen))
        self.assertEqual(report["actual_generation_requests"], 1)
        self.assertEqual(report["evidence_plan_requests"], 1)
        self.assertEqual(report["candidate_slots"], 2)
        self.assertEqual(report["ready_for_solver"], 2)
        questions = gen.read(run / "SOLVER_POOL.json")["questions"]
        self.assertTrue(all(q["gold"] == list("ABCDEF") for q in questions))
        self.assertEqual(len(pipeline.verify_generation(run)[1]), 2)
        self.assertEqual(report["request_costs"]["evidence_planner"]["total_tokens"], 110)

    def test_planner_counts_are_guidelines_not_gates(self):
        run = self.prepare()
        context = gen.initial_context(run, self.root, "multi_select")
        for count in (0, 1, 6, 8, 12):
            value = self.plan(context, count)
            self.assertEqual(gen.parse_evidence_plan(json.dumps(value), self.root,
                context["candidate_ids"], {"turns": self.fixture.turns}), value)

    def test_planner_bad_json_stops_without_compose_or_retry(self):
        run, report, seen = self.exercise(bad_plan=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(report["actual_generation_requests"], 0)
        self.assertEqual(report["ready_for_solver"], 0)
        self.assertEqual(report["http_attempts"], 1)
        self.assertEqual(report["evidence_plan_status"], "FAILED")
        self.assertFalse(gen.read(run / "EVIDENCE_PLAN_FAILURE.json")["direct_fallback"])
        self.assertEqual(gen.read(run / "MECHANICAL_POOL.json")["questions"], [])
        self.assertEqual(len(pipeline.verify_generation(run)[1]), 0)

    def test_planner_transport_failure_stops_and_cost_remains_unknown(self):
        run, report, seen = self.exercise(failed_http=True)
        self.assertEqual(len(seen), 1)
        cost = report["request_costs"]["evidence_planner"]
        self.assertEqual(cost["unknown_request_count"], 1)
        self.assertIsNone(cost["total_tokens"])
        self.assertEqual(cost["reported_total_tokens"], 0)
        self.assertEqual(report["actual_generation_requests"], 0)

    def test_success_missing_usage_is_unknown_not_zero(self):
        _, report, _ = self.exercise(missing_usage=True)
        self.assertIsNone(report["request_costs"]["evidence_planner"]["input_tokens"])
        self.assertIsNone(report["request_costs"]["question_generator"]["total_tokens"])

    def test_repair_uses_same_raw_plan_and_bounded_three_calls(self):
        run, report, seen = self.exercise(repair=True)
        self.assertEqual(len(seen), 3)
        self.assertEqual(report["http_attempts"], gen.generation_budget(1, 2, "staged")["http_attempts_max"])
        self.assertEqual(report["actual_generation_requests"], 2)
        contexts = [fixtures.input_json(payload, "\nINPUT_JSON\n") for payload, _ in seen[1:]]
        self.assertEqual(contexts[0]["frozen_evidence_plan_raw_json"], contexts[1]["frozen_evidence_plan_raw_json"])
        self.assertEqual(contexts[0]["source_turns"], contexts[1]["source_turns"])
        self.assertEqual(len(contexts[1]["candidate_ids"]), 1)
        self.assertEqual(report["ready_for_solver"], 2)
        self.assertEqual(len(pipeline.verify_generation(run)[1]), 2)
        with patch.object(qa_transport, "shared_lock", return_value=nullcontext()):
            with self.assertRaisesRegex(ValueError, "generation_already_started"):
                gen.generate(run)

    def test_plan_and_compose_tampering_is_detected(self):
        run, _, seen = self.exercise()
        request_path = next(path for path in (run / "requests").glob("*/REQUEST.json")
                            if gen.read(path.with_name("PROOF.json"))["role"] == "question_generator")
        payload, proof = gen.read(request_path), gen.read(request_path.with_name("PROOF.json"))
        gen.verify_prompt_request(run, payload, proof)
        changed = deepcopy(payload)
        changed["messages"][0]["content"] = changed["messages"][0]["content"].replace(
            "fallible worksheet", "certified worksheet")
        with self.assertRaises(ValueError):
            gen.verify_prompt_request(run, changed, proof)
        path = run / "EVIDENCE_PLAN.json"
        plan = gen.read(path)
        plan["raw_text"] += " "
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "evidence_plan_raw_binding_changed"):
            gen.verify_prompt_request(run, payload, proof)


if __name__ == "__main__":
    unittest.main()
