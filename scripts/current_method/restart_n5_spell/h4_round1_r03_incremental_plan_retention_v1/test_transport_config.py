"""Offline production-config URL and scoped-credential regressions."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .transport_config import (chat_completion_endpoint, validate_complete_endpoint,
                               model_config_from_historical, credential_environment)
from .run import SingleShotTransport, validate_authorization, offline_network_guard
from .test_readiness import exercise_http_cli

PROJECT = Path("/root/autodl-tmp/gxy-selfplay")
FROZEN = PROJECT / "restart-n5-runs/h4_round1_r03_incremental_readiness_v2_offline_preflight_live_1"
FAILED = PROJECT / "restart-n5-runs/h4_round1_r03_expanded_single_plan_fullflow_debug_live_3"


def corrected_production_model():
    old = json.loads((FROZEN / "model_config_PENDING_AUTHORIZATION.json").read_text())
    historical = json.loads((FAILED / "execution_plan.json").read_text())
    return model_config_from_historical(historical, tokenizer_path=old["tokenizer_path"],
                                       tokenizer_sha256=old["tokenizer_json_sha256"])


class TransportConfigTests(unittest.TestCase):
    def setUp(self):
        self.guard = offline_network_guard()
        self.guard.__enter__()
        self.addCleanup(self.guard.__exit__, None, None, None)

    def test_base_converted_once_only_at_configuration_build(self):
        full = "https://api.siliconflow.cn/v1/chat/completions"
        for source in ("https://api.siliconflow.cn/v1", "https://api.siliconflow.cn/v1/", full):
            self.assertEqual(chat_completion_endpoint(source), full)
        self.assertEqual(validate_complete_endpoint(full), full)

    def test_runtime_rejects_old_base_before_tokenizer_or_request(self):
        old = json.loads((FROZEN / "model_config_PENDING_AUTHORIZATION.json").read_text())
        with patch("scripts.current_method.restart_n5_spell.h4_round1_r03_incremental_plan_retention_v1.run.local_counter") as counter:
            with self.assertRaisesRegex(ValueError, "complete_chat_completions_endpoint"):
                SingleShotTransport(old, {})
            counter.assert_not_called()
        with self.assertRaisesRegex(ValueError, "complete_chat_completions_endpoint"):
            validate_authorization(authorization={}, bundle={}, budget={}, model_config=old)

    def test_invalid_or_credential_containing_url_rejected(self):
        for url in ("http://example.org/v1", "https://user:secret@example.org/v1", "https://example.org/v1?key=x",
                    "https://example.org/v1#fragment", " https://example.org/v1", "https://example.org/v1\n", None):
            with self.subTest(url=url), self.assertRaises(ValueError):
                chat_completion_endpoint(url)

    def test_production_config_only_endpoint_changes(self):
        old = json.loads((FROZEN / "model_config_PENDING_AUTHORIZATION.json").read_text())
        new = corrected_production_model()
        self.assertEqual([k for k in old if old[k] != new[k]], ["endpoint"])
        self.assertEqual(new["endpoint"], "https://api.siliconflow.cn/v1/chat/completions")

    def test_formal_config_through_cli_to_final_http_url(self):
        result = exercise_http_cli(None, model_config=corrected_production_model())
        self.assertEqual(result["actual_request_url.json"]["url"], "https://api.siliconflow.cn/v1/chat/completions")
        self.assertTrue(result["result_MOCK_ONLY.json"]["complete_feedback_matches_source"])

    def test_explicit_credentials_through_cli_and_error_cleanup(self):
        for case in ("normal", "400", "timeout"):
            with self.subTest(case=case):
                result = exercise_http_cli(None, case, model_config=corrected_production_model(), use_credential_file=True)
                self.assertTrue(result["actual_request_url.json"]["credential_handoff_used"])
                self.assertNotIn("mock-secret-never-export", json.dumps(result))

    def test_scoped_source_validation_and_env_restore(self):
        config = {**corrected_production_model(), "api_key_env": "H4_TEST_SCOPED_CREDENTIAL"}
        with tempfile.TemporaryDirectory(prefix="h4-credential-MOCK-") as temp, patch.dict(os.environ, {config["api_key_env"]: ""}):
            path = Path(temp) / "credential.json"
            record = {"api": {"base_url": "https://api.siliconflow.cn/v1", "model": config["model"], "api_key": "MOCK-scoped-value"}}
            path.write_text(json.dumps(record))
            source = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(RuntimeError, "MOCK stage error"):
                with credential_environment(config, source) as metadata:
                    self.assertEqual(os.environ[config["api_key_env"]], "MOCK-scoped-value")
                    self.assertNotIn("MOCK-scoped-value", json.dumps(metadata))
                    raise RuntimeError("MOCK stage error")
            self.assertEqual(os.environ[config["api_key_env"]], "")
            with self.assertRaisesRegex(ValueError, "sha_mismatch"):
                with credential_environment(config, {**source, "sha256": "0" * 64}):
                    self.fail("invalid source accepted")
            for field, value, expected in (("model", "different-model", "model_mismatch"),
                                          ("base_url", "https://other.invalid/v1", "endpoint_mismatch"),
                                          ("api_key", "", "key_missing")):
                altered = json.loads(json.dumps(record))
                altered["api"][field] = value
                path.write_text(json.dumps(altered))
                identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                with self.assertRaisesRegex(ValueError, expected):
                    with credential_environment(config, identity):
                        self.fail("invalid source accepted")
            self.assertEqual(os.environ[config["api_key_env"]], "")

    def test_existing_environment_not_overwritten_and_missing_no_discovery(self):
        config = {**corrected_production_model(), "api_key_env": "H4_TEST_SCOPED_CREDENTIAL"}
        with patch.dict(os.environ, {config["api_key_env"]: "MOCK-existing"}):
            with credential_environment(config) as meta:
                self.assertFalse(meta["temporary_injection"])
            self.assertEqual(os.environ[config["api_key_env"]], "MOCK-existing")
        with patch.dict(os.environ, {config["api_key_env"]: ""}):
            with self.assertRaisesRegex(ValueError, "explicit_source_required"):
                with credential_environment(config):
                    self.fail("should not discover credentials")


if __name__ == "__main__":
    unittest.main()
