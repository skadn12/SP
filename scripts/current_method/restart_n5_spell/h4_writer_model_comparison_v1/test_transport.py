from __future__ import annotations

import copy
import io
import json
import os
import ssl
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

from .transport import (ComparisonTransportError, ProfileClient, _NoRedirect,
                        build_payload, resolve_credential)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "model": "test-version-1", "base_url": "https://provider.invalid/v3",
            "endpoint": "https://provider.invalid/v3/chat/completions",
            "api_key_env": "H4_TRANSPORT_TEST_KEY", "timeout_seconds": 12,
            "output_limit": 8192,
            "request_parameters": {"temperature": 0.1, "max_tokens": 8192,
                                   "enable_thinking": False,
                                   "response_format": {"type": "json_object"}},
        }
        self.secret = "test-credential-never-log"

    def client(self, profile=None, **kwargs):
        with patch("urllib.request.build_opener") as factory:
            opener = factory.return_value
            client = ProfileClient(profile or self.profile, credential=self.secret, **kwargs)
        return client, opener

    def respond(self, opener, body):
        response = opener.open.return_value.__enter__.return_value
        response.read.return_value = json.dumps(body).encode("utf-8")
        response.getcode.return_value = 200

    def test_frozen_messages_and_only_explicit_parameters(self):
        messages = [{"role": "system", "content": "exact\ntext"},
                    {"role": "user", "content": "完整 133 条对话\n"}]
        original = copy.deepcopy(messages)
        profile = dict(self.profile, request_parameters={"seed": None})
        payload = build_payload(profile, messages)
        self.assertEqual(payload, {"model": "test-version-1", "messages": original})
        payload["messages"][0]["content"] = "changed"
        self.assertEqual(messages, original)

    def test_endpoint_payload_returned_model_usage_and_callback(self):
        callback = Mock()
        client, opener = self.client(on_http_attempt=callback)
        self.respond(opener, {"id": "completion-123", "model": "actual-fixed-version",
            "choices": [{"finish_reason": "length", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 71, "completion_tokens": 29, "total_tokens": 100,
                      "completion_tokens_details": {"reasoning_tokens": 11}}})
        result = client.complete(prompt="完整冻结正文", max_tokens=8192, json_object=True)
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, self.profile["endpoint"])
        self.assertEqual(payload, build_payload(self.profile, "完整冻结正文"))
        self.assertEqual(opener.open.call_count, 1)
        callback.assert_called_once_with(payload)
        self.assertEqual(result.model_identifier, "actual-fixed-version")
        self.assertEqual(result.metadata["finish_reason"], "length")
        self.assertEqual([result.metadata[k] for k in ("input_tokens", "output_tokens", "reasoning_tokens")], [71, 29, 11])
        self.assertEqual(result.metadata["raw_response"]["id"], "completion-123")
        self.assertNotIn(self.secret, repr(client) + repr(client.config) + repr(result))

    def test_missing_model_and_usage_remain_unknown(self):
        client, opener = self.client()
        self.respond(opener, {"choices": [{"message": {"content": "{}"}}]})
        result = client.complete(prompt="x", max_tokens=8192, json_object=True)
        self.assertEqual(result.model_identifier, "UNKNOWN")
        for key in ("actual_model", "usage", "input_tokens", "output_tokens", "reasoning_tokens", "finish_reason"):
            self.assertIsNone(result.metadata[key])

    def test_429_is_one_attempt_and_error_body_is_redacted(self):
        client, opener = self.client()
        opener.open.side_effect = urllib.error.HTTPError(self.profile["endpoint"], 429, "Too many requests", {},
                                                         io.BytesIO(self.secret.encode()))
        with self.assertRaises(ComparisonTransportError) as caught:
            client.complete(prompt="x", max_tokens=8192, json_object=True)
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(caught.exception.metadata["http_status"], 429)
        self.assertEqual(caught.exception.metadata["transport_retry_count"], 0)
        self.assertNotIn(self.secret, repr(caught.exception.metadata) + str(caught.exception))

    def test_parameter_and_context_rejections_never_change_or_resend_request(self):
        for code, message in ((400, b"unsupported response_format"), (413, b"context too long")):
            with self.subTest(http_status=code):
                callback = Mock()
                client, opener = self.client(on_http_attempt=callback)
                opener.open.side_effect = urllib.error.HTTPError(
                    self.profile["endpoint"], code, "rejected", {}, io.BytesIO(message))
                with self.assertRaises(ComparisonTransportError) as caught:
                    client.complete(prompt="original frozen transcript", max_tokens=8192, json_object=True)
                self.assertEqual(opener.open.call_count, 1)
                sent = json.loads(opener.open.call_args.args[0].data)
                self.assertEqual(sent, build_payload(self.profile, "original frozen transcript"))
                callback.assert_called_once_with(sent)
                self.assertEqual(caught.exception.metadata["http_status"], code)
                self.assertEqual(caught.exception.metadata["transport_retry_count"], 0)

    def test_timeout_and_tls_failure_are_never_retried(self):
        for error in (TimeoutError(), urllib.error.URLError(ssl.SSLError("EOF occurred"))):
            with self.subTest(error=type(error).__name__):
                client, opener = self.client()
                opener.open.side_effect = error
                with self.assertRaises(ComparisonTransportError) as caught:
                    client.complete(prompt="x", max_tokens=8192, json_object=True)
                self.assertEqual(opener.open.call_count, 1)
                self.assertEqual(caught.exception.metadata["transport_attempt_count"], 1)
                self.assertEqual(caught.exception.metadata["completion_status"], "UNKNOWN")

    def test_redirect_handler_refuses_follow_up(self):
        handler = _NoRedirect()
        request = urllib.request.Request(self.profile["endpoint"])
        self.assertIsNone(handler.redirect_request(request, None, 302, "Found", {}, "https://elsewhere.invalid"))
        client, opener = self.client()
        opener.open.side_effect = urllib.error.HTTPError(self.profile["endpoint"], 302, "Found", {}, io.BytesIO(b""))
        with self.assertRaises(ComparisonTransportError) as caught:
            client.complete(prompt="x", max_tokens=8192, json_object=True)
        self.assertEqual(caught.exception.status, "redirect_blocked")
        self.assertEqual(opener.open.call_count, 1)

    def test_incompatible_budget_fails_before_attempt(self):
        callback = Mock()
        client, opener = self.client(on_http_attempt=callback)
        with self.assertRaises(ValueError):
            client.complete(prompt="x", max_tokens=4096, json_object=True)
        opener.open.assert_not_called()
        callback.assert_not_called()

    def test_json_format_is_not_injected_for_other_provider(self):
        other = dict(self.profile, model="other-model", endpoint="https://other.invalid/custom",
                     request_parameters={"max_completion_tokens": 8192})
        client, opener = self.client(other)
        self.respond(opener, {"model": "other-model", "choices": [{"message": {"content": "{}"}}]})
        client.complete(prompt="same text", max_tokens=8192, json_object=True)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://other.invalid/custom")
        self.assertEqual(json.loads(request.data), {"model": "other-model", "max_completion_tokens": 8192,
                                                   "messages": [{"role": "user", "content": "same text"}]})

    def test_missing_or_invalid_envelope_preserves_evidence(self):
        client, opener = self.client()
        self.respond(opener, {"model": "actual", "error": "bad response", "usage": {"prompt_tokens": 9}})
        with self.assertRaises(ComparisonTransportError) as caught:
            client.complete(prompt="x", max_tokens=8192, json_object=True)
        self.assertEqual(caught.exception.status, "invalid_response_schema")
        self.assertEqual(caught.exception.metadata["input_tokens"], 9)
        self.assertEqual(caught.exception.metadata["raw_response"]["error"], "bad response")
        self.assertEqual(opener.open.call_count, 1)

    def test_response_echo_of_secret_is_not_logged(self):
        client, opener = self.client()
        self.respond(opener, {"choices": [{"message": {"content": self.secret}}], "authorization": "Bearer " + self.secret})
        result = client.complete(prompt="x", max_tokens=8192, json_object=True)
        self.assertNotIn(self.secret, repr(result))

    def test_credential_environment_only_explicit_name(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": self.secret}, clear=True):
            with self.assertRaises(ValueError):
                resolve_credential(self.profile)
        with patch.dict(os.environ, {"H4_TRANSPORT_TEST_KEY": self.secret}, clear=True):
            self.assertEqual(resolve_credential(self.profile), self.secret)

    def test_credential_file_requires_owned_mode_0600(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credential"
            path.write_text(self.secret + "\n")
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                resolve_credential(self.profile, path)
            path.chmod(0o600)
            self.assertEqual(resolve_credential(self.profile, path), self.secret)
            link = Path(directory) / "symlink"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                resolve_credential(self.profile, link)

    def test_inline_secret_and_message_override_rejected(self):
        for values in ({"api_key": self.secret}, {"request_parameters": {"messages": []}}):
            with self.subTest(keys=list(values)):
                with self.assertRaises(ValueError):
                    build_payload(dict(self.profile, **values), "x")


if __name__ == "__main__":
    unittest.main()
