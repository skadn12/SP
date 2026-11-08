"""Contract-alignment regression using immutable inputs and blocked network I/O."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

from . import POLICY_ID
from .fixtures import fixture_patch
from .input_views import (
    _resolve_at, project_supplement_inputs, reconstruct_supplement_inputs,
)
from .planning import merge_patch, prepare_original, sha
from .prompts import render_supplement, TARGET_RULES
from .run import main, implementation_manifest, offline_network_guard, production_registry
from .supplement_contract_alignment import (
    ALIGNED_VERSION, LEGACY_VERSION, CONTRACT_ALIGNMENT_RULES,
    allowed_source_view, select_supplement_version,
)
from .test_readiness import Response


PACKAGE = Path(__file__).resolve().parent
RUNS = PACKAGE.parents[4] / "restart-n5-runs"
ACTUAL = RUNS / "h4_round1_r03_incremental_retention_authorized_single_root_live_2"
RUN_MODULE = __package__ + ".run"
EXAMPLE_PHASE_IDS = ("phase:followup_1", "phase:followup-2")
EXAMPLE_DEPENDENCY = {
    "phase_id": "phase:followup_1",
    "condition": "The prerequisite established by the earlier phase actually holds.",
}
EXPECTED_ERRORS = [
    "patch_phase_1_identity", "patch_phase_1_original_goal_changed",
    "patch_phase_2_unrelated_candidate", "patch_phase_2_identity",
    "patch_phase_2_original_goal_changed", "patch_phase_3_unrelated_candidate",
    "patch_phase_3_identity", "patch_phase_3_original_goal_changed",
]


def read_json(path):
    return json.loads(Path(path).read_text())


def actual_inputs():
    bundle = read_json(ACTUAL / "inputs/bundle.json")
    prepared = read_json(ACTUAL / "execution/original_plan_new_policy_diagnostics.json")
    inputs = {
        "identity": bundle["identity"], "original_plan_version": "original_v1",
        "original_plan_sha256": prepared["original_sha256"], "original_plan": prepared["original"],
        "stable_candidates": prepared["candidate_registry"],
        "affected_candidate_ids": prepared["affected_candidate_ids"],
        "goal_diagnostics": prepared["goal_diagnostics"],
        "authoritative_history": bundle["parent"]["dialogue"]["turns"],
        "retained_actual_r03_dialogue": [], "budget": read_json(ACTUAL / "inputs/budget.json"),
        "shared_feedback_view": bundle["shared_feedback_view"],
    }
    request = read_json(ACTUAL / "execution/requests/001_supplement.json")
    projected = json.loads(request["messages"][0]["content"].rsplit("\n\nINPUT DATA\n", 1)[1])
    mapping = read_json(ACTUAL / "execution/supplement_input_projection_mapping.json")
    assert reconstruct_supplement_inputs(projected, mapping) == inputs
    return inputs


def readonly_ledger_snapshot():
    root = production_registry()
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()} if root.exists() else {}


def exercise_alignment_cli(output=None, *, prompt_version=ALIGNED_VERSION, case="normal"):
    """Capture the real CLI-to-HTTP path using a temporary ledger and mock opener.

    No real credential is read. All sockets are blocked. Only the transient mock
    authorization includes this execution identity, and it is never exported.
    """
    before = readonly_ledger_snapshot()
    exported = {}
    with tempfile.TemporaryDirectory(prefix="h4-alignment-MOCK-") as temporary:
        root = Path(temporary)
        bundle = read_json(ACTUAL / "inputs/bundle.json")
        bundle["identity"]["branch_id"] = "MOCK-CONTRACT-ALIGNMENT-NOT-AUTHORIZED-PRODUCTION"
        bundle["supplement_prompt_version"] = prompt_version
        budget = read_json(ACTUAL / "inputs/budget.json")
        model = read_json(ACTUAL / "inputs/model_config.json")
        authorization = {
            "explicitly_authorized": True, "policy_id": POLICY_ID,
            "identity": bundle["identity"], "bundle_sha256": sha(bundle),
            "budget_sha256": sha(budget), "model_config_sha256": sha(model),
            "implementation_manifest": implementation_manifest(),
        }
        for name, value in (("bundle", bundle), ("budget", budget), ("model", model),
                            ("authorization", authorization)):
            with (root / (name + ".json")).open("x") as handle:
                json.dump(value, handle)
        content = json.dumps({
            "status": "NO_SUPPLEMENT", "original_plan_sha256": bundle["original_plan_sha256"],
            "phases": [], "reason": "OFFLINE MOCK: contract routing only, no model generation.",
        })
        response = Response(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        }).encode())
        request_payloads = []
        request_urls = []
        opener = Mock()
        opener.open.return_value = response
        if case == "http_400":
            opener.open.side_effect = urllib.error.HTTPError(
                model["endpoint"], 400, "MOCK rejection", response.headers,
                io.BytesIO(b'{"error":"MOCK full provider rejection body"}'))
        elif case != "normal":
            raise ValueError("unknown_mock_case")

        def record_request(request, **kwargs):
            request_urls.append(request.full_url)
            request_payloads.append(json.loads(request.data))
            return opener.open(request, **kwargs)

        args = ["--execute", "--bundle", str(root / "bundle.json"),
                "--budget", str(root / "budget.json"), "--model-config", str(root / "model.json"),
                "--authorization", str(root / "authorization.json"), "--run-dir", str(root / "MOCK_run")]
        with offline_network_guard(), \
             patch.dict("os.environ", {model["api_key_env"]: "mock-secret-never-export"}), \
             patch("urllib.request.build_opener", return_value=Mock(open=record_request)), \
             patch(RUN_MODULE + ".production_registry", return_value=root / "MOCK_state") as registry, \
             patch("sys.stdout", new=io.StringIO()):
            if prompt_version not in (ALIGNED_VERSION, LEGACY_VERSION):
                with patch(RUN_MODULE + ".SingleShotTransport") as constructor:
                    try:
                        main(args)
                    except ValueError as error:
                        exported["result_MOCK_ONLY.json"] = {
                            "unknown_version_rejected": True, "exception_type": type(error).__name__,
                            "error_code": str(error), "mock_http_attempt_count": 0,
                            "real_external_calls": 0, "production_ledger_touched": False,
                        }
                    else:
                        raise AssertionError("unknown version was not rejected")
                    constructor.assert_not_called()
                    registry.assert_not_called()
                    assert not (root / "MOCK_state").exists()
                    assert not (root / "MOCK_run").exists()
            else:
                main(args)
        if prompt_version in (ALIGNED_VERSION, LEGACY_VERSION):
            assert opener.open.call_count == 1
            assert request_urls == [model["endpoint"]]
            run = root / "MOCK_run"
            request = read_json(run / "requests/001_supplement.json")
            report = read_json(run / "final_report.json")
            mapping = read_json(run / "supplement_input_projection_mapping.json")
            projected = json.loads(request["messages"][0]["content"].rsplit("\n\nINPUT DATA\n", 1)[1])
            restored = reconstruct_supplement_inputs(projected, mapping)
            assert request_payloads[0]["messages"] == request["messages"]
            assert request["prompt_version"] == prompt_version
            assert restored["shared_feedback_view"] == bundle["shared_feedback_view"]
            assert restored["authoritative_history"] == bundle["parent"]["dialogue"]["turns"]
            assert restored["original_plan"] == bundle["original_plan"]
            assert len(report["new_calls"]) == 1
            assert report["new_calls"][0]["role"] == "supplement"
            assert report["retained_scene_count"] == report["retained_turn_count"] == 0
            assert not list(run.glob("*cumulative*"))
            assert report["attempted_request_counts"] == {
                role: int(role == "supplement") for role in budget["role_requests"]}
            for directory in ("responses", "failures"):
                for path in (run / directory).glob("*.json"):
                    exported[directory + "/" + path.name] = read_json(path)
            identity_path = run / "supplement_contract_identity.json"
            if prompt_version == ALIGNED_VERSION:
                assert identity_path.exists()
                exported["supplement_contract_identity.json"] = read_json(identity_path)
            else:
                assert not identity_path.exists()
            exported.update({
                "actual_request.json": request,
                "actual_http_payload.json": request_payloads[0],
                "final_messages.json": request_payloads[0]["messages"],
                "model_parameters.json": model,
                "input_projection_mapping.json": mapping,
                "actual_request_url.json": {"url": request_urls[0], "matches_frozen_model_config": True},
                "result_MOCK_ONLY.json": {
                    "case": case, "status": report["status"], "prompt_version": prompt_version,
                    "mock_http_attempt_count": 1, "real_external_calls": 0,
                    "production_ledger_touched": False, "no_real_credentials_read": True,
                    "no_retry": True, "mock_retained_turn_count": 0,
                    "complete_feedback_matches_source": True,
                    "complete_history_turn_count": len(restored["authoritative_history"]),
                    "candidate_count": len(restored["stable_candidates"]),
                    "allowed_source_ids": restored["affected_candidate_ids"],
                    "model_quality": "UNKNOWN", "mock_authorization_exported": False,
                    "workflow_live_counter_counts_mock_transport_only": True,
                },
            })
        assert readonly_ledger_snapshot() == before
        assert "mock-secret-never-export" not in json.dumps(exported)
        if output is not None:
            from .readiness import save
            for name, value in exported.items():
                save(output, name, value)
    return exported


class SupplementAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = offline_network_guard()
        cls.guard.__enter__()
        cls.bundle = read_json(ACTUAL / "inputs/bundle.json")
        cls.prepared = read_json(ACTUAL / "execution/original_plan_new_policy_diagnostics.json")
        cls.inputs = actual_inputs()
        cls.ledger_before = readonly_ledger_snapshot()

    @classmethod
    def tearDownClass(cls):
        if readonly_ledger_snapshot() != cls.ledger_before:
            raise AssertionError("production ledger changed during offline tests")
        cls.guard.__exit__(None, None, None)

    def examples_patch(self):
        value = fixture_patch(self.bundle)
        for row, phase_id in zip(value["phases"], EXAMPLE_PHASE_IDS):
            row["phase_id"] = phase_id
        value["phases"][1]["dependencies"] = [deepcopy(EXAMPLE_DEPENDENCY)]
        return value

    def test_prompt_id_and_dependency_examples_pass_existing_validator(self):
        value = self.examples_patch()
        fragment = json.dumps({"dependencies": [EXAMPLE_DEPENDENCY]}, separators=(",", ":"))
        self.assertIn(fragment, CONTRACT_ALIGNMENT_RULES)
        for phase_id in EXAMPLE_PHASE_IDS:
            self.assertIn(phase_id, CONTRACT_ALIGNMENT_RULES)
        result = merge_patch(self.prepared, value)
        self.assertTrue(result["valid"], result)
        self.assertEqual(len(result["merged"]["added_phases"]), 2)
        self.assertEqual(result["diff"]["original_candidate_changes"], [])
        self.assertEqual(result["diff"]["original_goal_changes"], [])
        self.assertEqual(len({r["source_candidate_id"] for r in value["phases"]}), 1)

    def test_original_illegal_second_colon_examples_still_rejected(self):
        for phase_id in ("phase:ND01:C01", "phase:ND02:C01", "phase:ND03:C01"):
            with self.subTest(phase_id=phase_id):
                value = self.examples_patch()
                value["phases"] = value["phases"][:1]
                value["phases"][0]["phase_id"] = phase_id
                self.assertEqual(merge_patch(self.prepared, value)["errors"], ["patch_phase_1_identity"])

    def test_phase_identity_no_normalization_or_duplicate_waiver(self):
        for phase_id in (" phase:valid", "phase:valid ", "phase:a.b", "phase:"):
            with self.subTest(phase_id=phase_id):
                value = self.examples_patch()
                value["phases"] = value["phases"][:1]
                value["phases"][0]["phase_id"] = phase_id
                self.assertIn("patch_phase_1_identity", merge_patch(self.prepared, value)["errors"])
        value = self.examples_patch()
        value["phases"][1]["phase_id"] = EXAMPLE_PHASE_IDS[0]
        self.assertIn("patch_phase_2_identity", merge_patch(self.prepared, value)["errors"])

    def test_forward_self_unknown_and_malformed_dependencies_rejected(self):
        for dependency in (
            {"phase_id": EXAMPLE_PHASE_IDS[1], "condition": "later"},
            {"phase_id": EXAMPLE_PHASE_IDS[0], "condition": "self"},
            {"phase_id": "phase:unknown", "condition": "unknown"},
            {"phase_id": "initial:ME07:C01", "condition": " "},
            {"phase_id": "initial:ME07:C01", "condition": "known", "extra": True},
        ):
            with self.subTest(dependency=dependency):
                value = self.examples_patch()
                value["phases"][0]["dependencies"] = [dependency]
                self.assertIn("patch_phase_1_dependency", merge_patch(self.prepared, value)["errors"])

    def test_background_initial_dependency_is_not_a_source_violation(self):
        value = self.examples_patch()
        value["phases"][0]["dependencies"] = [{
            "phase_id": "initial:ME08:C01", "condition": "A background prerequisite is established.",
        }]
        self.assertTrue(merge_patch(self.prepared, value)["valid"])

    def test_other_candidate_goal_wrong_field_and_rewrite_rejected(self):
        scene = self.prepared["candidate_registry"]["ME07:C01"]["scene"]
        other_goal = self.prepared["candidate_registry"]["ND01:C01"]["scene"]["goal"]
        for goal in (other_goal, scene["required_transition"], scene["after_state"],
                     scene["goal"].removesuffix("."), " " + scene["goal"], scene["goal"] + " "):
            with self.subTest(goal=goal):
                value = self.examples_patch()
                value["phases"][0]["original_goal"] = goal
                self.assertEqual(merge_patch(self.prepared, value)["errors"],
                                 ["patch_phase_1_original_goal_changed"])

    def test_json_escaping_preserves_parsed_goal_not_semantic_validation(self):
        prepared = deepcopy(self.prepared)
        prepared["candidate_registry"]["ME07:C01"]["scene"]["goal"] = 'A quoted "goal" with a newline\nremains exact.'
        value = self.examples_patch()
        for row in value["phases"]:
            row["original_goal"] = prepared["candidate_registry"]["ME07:C01"]["scene"]["goal"]
        decoded = json.loads(json.dumps(value))
        self.assertTrue(merge_patch(prepared, decoded)["valid"])

    def test_known_but_unallowed_sources_still_rejected(self):
        for ref in ("ME08:C01", "ME03:C01"):
            with self.subTest(ref=ref):
                value = self.examples_patch()
                value["phases"] = value["phases"][:1]
                value["phases"][0]["source_candidate_id"] = ref
                value["phases"][0]["original_goal"] = self.prepared["candidate_registry"][ref]["scene"]["goal"]
                self.assertIn("patch_phase_1_unrelated_candidate", merge_patch(self.prepared, value)["errors"])

    def test_original_real_patch_still_same_eight_errors_without_input_mutation(self):
        validation = read_json(ACTUAL / "execution/increment_patch_and_diff.json")
        raw = json.loads(read_json(ACTUAL / "execution/responses/001_supplement.json"))
        self.assertEqual(raw, validation["raw_patch"])
        identity = read_json(ACTUAL / "implementation_sha256.json")
        path = PACKAGE / "planning.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), identity[str(path)])
        before = sha({"prepared": self.prepared, "patch": raw})
        reproduced = merge_patch(self.prepared, raw)
        self.assertEqual(reproduced, validation)
        self.assertEqual(reproduced["errors"], EXPECTED_ERRORS)
        self.assertEqual(sha({"prepared": self.prepared, "patch": raw}), before)

    def test_prepared_registry_reproduces_frozen_candidates_and_allowed_scope(self):
        actual = prepare_original(self.bundle["original_plan"], parent=self.bundle["parent"],
                                  candidates=self.bundle["candidates"], verified_tracks=self.bundle["verified_tracks"])
        self.assertEqual(actual, self.prepared)
        self.assertEqual(actual["affected_candidate_ids"], ["ME07:C01"])
        self.assertEqual(len(actual["candidate_registry"]), 11)

    def test_allowed_source_map_resolves_raw_and_lossless_projected_goals(self):
        projected, _ = project_supplement_inputs(self.inputs)
        for data in (self.inputs, projected):
            with self.subTest(projected=data is projected):
                rows = allowed_source_view(data)
                self.assertEqual([row["source_candidate_id"] for row in rows], ["ME07:C01"])
                expected = self.prepared["candidate_registry"]["ME07:C01"]["scene"]["goal"]
                self.assertEqual(rows[0]["original_goal"], expected)
                resolved, _ = _resolve_at(data, rows[0]["goal_source"])
                self.assertEqual(resolved, expected)

    def test_distinct_fixture_allowlist_is_data_driven_not_hardcoded(self):
        data = deepcopy(self.inputs)
        data["affected_candidate_ids"] = ["ND02:C01"]
        built = render_supplement(data, prompt_version=ALIGNED_VERSION)
        rows = built["supplement_contract_identity"]["allowed_source_view"]
        self.assertEqual([row["source_candidate_id"] for row in rows], ["ND02:C01"])
        self.assertEqual(rows[0]["original_goal"], data["stable_candidates"]["ND02:C01"]["scene"]["goal"])

    def test_invalid_allowed_source_data_fails_closed(self):
        for bad in (["UNKNOWN:C01"], "ME07:C01"):
            with self.subTest(value=bad):
                data = deepcopy(self.inputs)
                data["affected_candidate_ids"] = bad
                with self.assertRaises((ValueError, KeyError)):
                    allowed_source_view(data)

    def test_legacy_renderer_reproduces_actual_frozen_messages_exactly(self):
        expected = read_json(ACTUAL / "execution/requests/001_supplement.json")
        for built in (render_supplement(self.inputs),
                      render_supplement(self.inputs, prompt_version=LEGACY_VERSION)):
            self.assertEqual(built["messages"], expected["messages"])
            self.assertEqual(built["prompt_version"], expected["prompt_version"])
            self.assertNotIn("supplement_contract_identity", built)

    def test_aligned_renderer_has_single_map_and_same_input_projection(self):
        original = deepcopy(self.inputs)
        old = render_supplement(self.inputs)
        new = render_supplement(self.inputs, prompt_version=ALIGNED_VERSION)
        old_data = old["prompt"].rsplit("\n\nINPUT DATA\n", 1)[1]
        new_data = new["prompt"].rsplit("\n\nINPUT DATA\n", 1)[1]
        self.assertEqual(new_data, old_data)
        self.assertEqual(new["input_projection_mapping"], old["input_projection_mapping"])
        self.assertEqual(self.inputs, original)
        sidecar = new["supplement_contract_identity"]
        self.assertEqual(sidecar["prompt_version"], ALIGNED_VERSION)
        self.assertEqual(sidecar["allowed_source_view_sha256"], sha(sidecar["allowed_source_view"]))
        self.assertEqual(sidecar["allowed_source_view"], allowed_source_view(json.loads(new_data)))
        header = new["prompt"].rsplit("\n\nINPUT DATA\n", 1)[0]
        self.assertIn(CONTRACT_ALIGNMENT_RULES, header)
        self.assertIn(TARGET_RULES, header)
        self.assertEqual(header.count('"source_candidate_id": "ME07:C01"'), 1)
        self.assertEqual(header.count(sidecar["allowed_source_view"][0]["original_goal"]), 1)
        for example in EXAMPLE_PHASE_IDS:
            self.assertIn(example, header)
        self.assertIn("phase:[A-Za-z0-9_-]+", header)
        self.assertIn("original_goal", header)
        self.assertIn("scene.goal", header)
        self.assertIn("INCREMENTAL PATCH CONTRACT EXPLANATION v1", header)

    def test_version_selection_is_explicit_and_unknown_fails(self):
        self.assertEqual(select_supplement_version({}), LEGACY_VERSION)
        self.assertEqual(select_supplement_version({"supplement_prompt_version": LEGACY_VERSION}), LEGACY_VERSION)
        self.assertEqual(select_supplement_version({"supplement_prompt_version": ALIGNED_VERSION}), ALIGNED_VERSION)
        for value in ("future_unknown_version", "", {}, 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    select_supplement_version({"supplement_prompt_version": value})

    def test_actual_cli_to_http_messages_identity_and_parameter_preservation(self):
        result = exercise_alignment_cli()
        request = result["actual_request.json"]
        payload = result["actual_http_payload.json"]
        self.assertEqual(request["prompt_version"], ALIGNED_VERSION)
        self.assertEqual(result["supplement_contract_identity.json"]["prompt_version"], ALIGNED_VERSION)
        self.assertEqual(payload["messages"], request["messages"])
        self.assertEqual(set(payload), {"model", "messages", "temperature", "max_tokens", "stream", "chat_template_kwargs"})
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["model"], "Qwen/Qwen3-14B")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertTrue(request["capacity"]["fits"])
        self.assertEqual(result["result_MOCK_ONLY.json"]["complete_history_turn_count"], 143)
        self.assertEqual(result["result_MOCK_ONLY.json"]["candidate_count"], 11)
        self.assertEqual(result["result_MOCK_ONLY.json"]["allowed_source_ids"], ["ME07:C01"])
        self.assertEqual(result["result_MOCK_ONLY.json"]["status"], "NO_SUPPLEMENT_STOP")

    def test_legacy_version_still_routes_through_actual_cli(self):
        result = exercise_alignment_cli(prompt_version=LEGACY_VERSION)
        self.assertEqual(result["actual_request.json"]["prompt_version"], LEGACY_VERSION)
        self.assertNotIn("supplement_contract_identity.json", result)

    def test_unknown_cli_version_blocks_before_transport_registry_and_request(self):
        result = exercise_alignment_cli(prompt_version="unknown_NOT_ALLOWED")
        self.assertTrue(result["result_MOCK_ONLY.json"]["unknown_version_rejected"])
        self.assertEqual(result["result_MOCK_ONLY.json"]["mock_http_attempt_count"], 0)

    def test_http_rejection_keeps_complete_evidence_and_never_retries(self):
        result = exercise_alignment_cli(case="http_400")
        self.assertEqual(result["result_MOCK_ONLY.json"]["status"], "SUPPLEMENT_FAILED_STOP_NO_RETRY")
        evidence = result["failures/001_http_evidence.json"]
        self.assertEqual(evidence["http_status"], 400)
        self.assertTrue(evidence["response_body_complete"])
        self.assertIn("MOCK full provider rejection body", json.dumps(evidence))
        self.assertEqual(result["result_MOCK_ONLY.json"]["mock_http_attempt_count"], 1)


if __name__ == "__main__":
    unittest.main()
