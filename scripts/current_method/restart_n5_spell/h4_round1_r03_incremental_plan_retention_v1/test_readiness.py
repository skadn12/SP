"""Real CLI, workflow and HTTP construction with only local mocked I/O."""
from copy import deepcopy
import io
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

from . import POLICY_ID
from .bundle import build_failed_bundle, validate_bundle_feedback
from .capacity import local_counter, proposed_budget, measure
from .fixtures import failed_bundle, FixtureBackend
from .planning import sha
from .run import main, implementation_manifest, offline_network_guard
from .workflow import execute_policy


class Response(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(body)
        self.status = status
        self.headers = {"content-type": "application/json", "x-siliconcloud-trace-id": "MOCK-trace"}


def model_fixture():
    from .capacity import TOKENIZER_PATH, TOKENIZER_SHA
    return {"endpoint": "https://offline.invalid/v1/chat/completions", "model": "Qwen/Qwen3-14B",
            "temperature": 0.25, "timeout_seconds": 600, "api_key_env": "H4_READINESS_MOCK_KEY",
            "tokenizer_path": TOKENIZER_PATH, "tokenizer_json_sha256": TOKENIZER_SHA}


def exercise_http_cli(output, case="normal", *, model_config=None, use_credential_file=False):
    """Temporary authorization exercises the real branch, never production state.

    Mock-only inputs/authorizations are removed with the TemporaryDirectory.
    Optional evidence export has no usable authorization or production ledger.
    """
    exported = {}
    with tempfile.TemporaryDirectory(prefix="h4-readiness-http-MOCK-") as temp:
        root = Path(temp)
        bundle = build_failed_bundle(branch_id="MOCK-HTTP-CONSTRUCTION-NOT-PRODUCTION")
        budget = proposed_budget()
        model = deepcopy(model_config) if model_config is not None else model_fixture()
        auth = {"explicitly_authorized": True, "policy_id": POLICY_ID, "identity": bundle["identity"],
                "bundle_sha256": sha(bundle), "budget_sha256": sha(budget),
                "model_config_sha256": sha(model), "implementation_manifest": implementation_manifest()}
        credential_args = []
        if use_credential_file:
            credential = root / "MOCK_project_credentials.json"
            credential.write_text(json.dumps({"api": {"model": model["model"],
                "base_url": model["endpoint"].removesuffix("/chat/completions"), "api_key": "mock-secret-never-export"}}))
            auth["credential_source"] = {"path": str(credential), "sha256": hashlib.sha256(credential.read_bytes()).hexdigest()}
            credential_args = ["--credential-config", str(credential)]
        for name, obj in (("bundle", bundle), ("budget", budget), ("model", model), ("authorization", auth)):
            (root / (name + ".json")).write_text(json.dumps(obj))
        content = json.dumps({"status": "NO_SUPPLEMENT", "original_plan_sha256": bundle["original_plan_sha256"],
                              "phases": [], "reason": "MOCK ONLY: no generation performed"})
        if case == "business_parse":
            content = "not a JSON business object"
        body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()
        response = Response(body)
        error = None
        if case in {"400", "429", "500"}:
            error = urllib.error.HTTPError(model["endpoint"], int(case), "MOCK rejection", response.headers,
                io.BytesIO(b'{"error":"MOCK complete provider rejection"}'))
        elif case == "non_json":
            response = Response(b"MOCK non-JSON complete response")
        elif case == "empty":
            response = Response(b"")
        elif case == "timeout":
            error = TimeoutError("MOCK response unavailable")
        opener = Mock()
        opener.open.return_value = response
        opener.open.side_effect = error
        captured = []
        captured_urls = []
        original_open = opener.open
        def record_request(req, **kwargs):
            captured_urls.append(req.full_url)
            captured.append(json.loads(req.data))
            return original_open(req, **kwargs)
        with offline_network_guard(), patch.dict("os.environ", {model["api_key_env"]: "" if use_credential_file else "mock-secret-never-export"}), \
             patch("urllib.request.build_opener", return_value=Mock(open=record_request)), \
             patch("scripts.current_method.restart_n5_spell.h4_round1_r03_incremental_plan_retention_v1.run.production_registry", return_value=root / "MOCK_state"), \
             patch("sys.stdout", new=io.StringIO()):
            main(["--execute", "--bundle", str(root / "bundle.json"), "--budget", str(root / "budget.json"),
                  "--model-config", str(root / "model.json"), "--authorization", str(root / "authorization.json"),
                  "--run-dir", str(root / "MOCK_run")] + credential_args)
            if use_credential_file:
                assert os.environ[model["api_key_env"]] == ""
        assert opener.open.call_count == 1
        assert captured_urls == [model["endpoint"]]
        assert model["endpoint"].endswith("/chat/completions")
        request = json.loads((root / "MOCK_run/requests/001_supplement.json").read_text())
        sent_data = json.loads(captured[0]["messages"][0]["content"].rsplit("\n\nINPUT DATA\n", 1)[1])
        assert sent_data["shared_feedback_view"] == bundle["shared_feedback_view"]
        assert captured[0]["messages"] == request["messages"]
        assert captured[0]["max_tokens"] == 8192
        assert captured[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert not list((root / "MOCK_run").glob("*cumulative*"))
        report = json.loads((root / "MOCK_run/final_report.json").read_text())
        assert len(report["new_calls"]) == 1
        for directory in ("responses", "failures"):
            for path in (root / "MOCK_run" / directory).glob("*.json"):
                exported[directory + "/" + path.name] = json.loads(path.read_text())
        exported["actual_request.json"] = request
        exported["actual_http_payload.json"] = captured[0]
        exported["actual_request_url.json"] = {"url": captured_urls[0], "matches_frozen_model_config": True,
            "credential_handoff_used": use_credential_file, "credential_value_logged": False}
        exported["result_MOCK_ONLY.json"] = {"case": case, "status": report["status"],
            "mock_http_attempt_count": 1, "real_external_calls": 0, "production_ledger_touched": False,
            "complete_feedback_matches_source": True, "no_retry": True}
        if output is not None:
            from .readiness import save
            for name, obj in exported.items():
                save(output, name, obj)
    return exported


class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = offline_network_guard()
        cls.guard.__enter__()
        cls.counter = local_counter()
        cls.bundle = failed_bundle()

    @classmethod
    def tearDownClass(cls):
        cls.guard.__exit__(None, None, None)

    def test_actual_cli_to_http_feedback_and_failures(self):
        for case in ("normal", "400", "429", "500", "non_json", "business_parse", "empty", "timeout"):
            with self.subTest(case=case):
                result = exercise_http_cli(None, case)
                self.assertTrue(result["result_MOCK_ONLY.json"]["complete_feedback_matches_source"])

    def test_capacity_fail_before_any_registry_or_attempt(self):
        budget = proposed_budget()
        budget["role_capacity"]["supplement"]["max_input_tokens"] = 1
        with tempfile.TemporaryDirectory(prefix="h4-preclaim-") as temp:
            registry = Path(temp) / "not_created"
            backend = FixtureBackend(self.bundle)
            with self.assertRaisesRegex(ValueError, "known_capacity_blocked_before_attempt"):
                execute_policy(bundle=self.bundle, budget=budget, backend=backend,
                               state_registry=registry, artifact_dir=Path(temp) / "run")
            self.assertFalse(registry.exists())
            self.assertEqual(backend.requests, [])

    def test_first_audit_role_budget_fail_before_attempt(self):
        budget = proposed_budget()
        budget["role_requests"]["adherence"] = 0
        with tempfile.TemporaryDirectory(prefix="h4-preclaim-") as temp:
            backend = FixtureBackend(self.bundle)
            with self.assertRaisesRegex(ValueError, "known_first_batch_budget"):
                execute_policy(bundle=self.bundle, budget=budget, backend=backend,
                    state_registry=Path(temp) / "absent", artifact_dir=Path(temp) / "run")
            self.assertEqual(backend.requests, [])
            self.assertFalse((Path(temp) / "absent").exists())

    def test_actual_tokens_not_http_json_bytes(self):
        messages = [{"role": "user", "content": "Text with \\" * 200}]
        result = measure(messages, counter=self.counter, budget=proposed_budget(), role="supplement")
        self.assertEqual(result["known_final_messages_tokens"], self.counter.count_tokens(messages))
        self.assertNotEqual(result["known_final_messages_tokens"], result["http_json_bytes_DIAGNOSTIC_ONLY"])
        self.assertEqual(result["output_reserve_tokens"], 8192)
        self.assertEqual(result["thinking_reserve_tokens"], 0)

    def test_17_requests_four_batches_growth_raw_audit_bound(self):
        with tempfile.TemporaryDirectory(prefix="h4-growth-") as temp:
            backend = FixtureBackend(self.bundle)
            report = execute_policy(bundle=self.bundle, budget=proposed_budget(), backend=backend,
                state_registry=Path(temp) / "state", artifact_dir=Path(temp) / "run")
            self.assertEqual(report["mock_call_count"], 17)
            self.assertEqual(report["retained_scene_count"], 8)
            self.assertEqual(report["external_call_count"], 0)
            reserves = [c for c in report["capacity_checks"] if c["label"].startswith("batch_")]
            self.assertEqual(len(reserves), 4)
            self.assertEqual([r["retained_prefix_turn_count"] for r in reserves], [0, 28, 56, 84])
            audits = [r for r in backend.requests if r["role"] == "adherence"]
            for reserve, audit in zip(reserves, audits):
                self.assertLessEqual(self.counter.count_tokens(audit["messages"]),
                    reserve["known_final_messages_tokens"] + reserve["future_content_bound_tokens"])
                self.assertEqual(len(audit["messages"]), 2)
            self.assertFalse(list((Path(temp) / "run").glob("*cumulative*")))

    def test_no_authorization_blocks_before_transport(self):
        with patch("scripts.current_method.restart_n5_spell.h4_round1_r03_incremental_plan_retention_v1.run.SingleShotTransport") as transport, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main(["--execute", "--run-dir", "/tmp/h4-NOT-CREATED-readiness"])
            transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
