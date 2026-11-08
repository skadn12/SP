"""HTTP fixtures only; sockets blocked and no production state is constructed."""
import base64
from copy import deepcopy
import hashlib
import io
from http.client import IncompleteRead
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

from .http_evidence import request_once


class _Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {"content-type": "application/json", "x-request-id": "mock-request-1"}


class HttpEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch("socket.socket.connect", side_effect=AssertionError("NETWORK_BLOCKED"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.env = patch.dict("os.environ", {"H4_MOCK_HTTP_CREDENTIAL": "mock-sensitive-key-123"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.events = []
        self.transport = SimpleNamespace(
            config={"endpoint": "https://offline.invalid/chat/completions", "model": "Qwen/Qwen3-14B",
                "temperature": 0.25, "timeout_seconds": 600, "api_key_env": "H4_MOCK_HTTP_CREDENTIAL"},
            tokenizer=SimpleNamespace(encode=lambda value, **kwargs: list(value)),
            response_evidence_sink=lambda event: self.events.append(deepcopy(event)))
        self.request = {"role": "supplement", "call_id": 1, "prompt_version": "offline.mock.v2",
            "messages": [{"role": "user", "content": "OFFLINE MOCK"}], "max_tokens": 8192,
            "identity": {"root_id": "mock-root", "round_index": 3, "branch_id": "offline-http-test"}}

    def run_response(self, response=None, error=None):
        opener = Mock()
        opener.open.side_effect = error
        opener.open.return_value = response
        with patch("urllib.request.build_opener", return_value=opener):
            try:
                result = request_once(self.transport, self.request)
            finally:
                self.assertEqual(opener.open.call_count, 1)
        return result, opener

    def test_http_400_429_500_capture_before_failure(self):
        for status in (400, 429, 500):
            with self.subTest(status=status):
                body = json.dumps({"error": {"message": "mock complete error", "status": status}}).encode()
                exc = urllib.error.HTTPError("https://offline.invalid/", status, "mock HTTP error",
                    {"x-request-id": "trace-" + str(status)}, io.BytesIO(body))
                with self.assertRaises(urllib.error.HTTPError):
                    self.run_response(error=exc)
                e = self.transport.last_response_evidence
                self.assertEqual(e["http_status"], status)
                self.assertEqual(e["response_body"].encode(), body)
                self.assertEqual(e["failure_classification"], "HTTP_NON_SUCCESS")
                self.assertEqual(self.transport.last_response_envelope, json.loads(body))
                self.assertEqual(self.events[-2]["evidence_event"], "response_received")

    def test_200_non_json_body_preserved(self):
        with self.assertRaises(json.JSONDecodeError):
            self.run_response(_Response(b"complete non-json body\n"))
        self.assertEqual(self.transport.last_response_evidence["response_body"], "complete non-json body\n")
        self.assertEqual(self.events[0]["evidence_event"], "response_received")

    def test_business_failure_preserves_envelope(self):
        body = b'{"error":{"message":"business schema rejected"}}'
        with self.assertRaises(ValueError):
            self.run_response(_Response(body))
        self.assertEqual(self.transport.last_response_evidence["response_body"].encode(), body)
        self.assertEqual(self.transport.last_response_envelope, json.loads(body))

    def test_empty_response_is_not_no_response(self):
        with self.assertRaises(json.JSONDecodeError):
            self.run_response(_Response(b""))
        evidence = self.transport.last_response_evidence
        self.assertFalse(evidence["response_unavailable"])
        self.assertFalse(evidence["response_body_unavailable"])
        self.assertEqual(evidence["response_body"], "")
        self.assertEqual(evidence["http_status"], 200)

    def test_timeout_and_no_response_do_not_invent_http(self):
        for exc in (TimeoutError("mock timeout"), urllib.error.URLError("mock no response")):
            with self.subTest(error=type(exc).__name__):
                with self.assertRaises(type(exc)):
                    self.run_response(error=exc)
                evidence = self.transport.last_response_evidence
                self.assertTrue(evidence["response_unavailable"])
                self.assertIsNone(evidence["http_status"])
                self.assertNotIn("response_body", evidence)
                self.assertNotIn("response_body_base64", evidence)

    def test_normal_parameters_and_body_unchanged(self):
        body = b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}],"usage":{"total_tokens":5}}'
        result, opener = self.run_response(_Response(body))
        self.assertEqual(result, "{}")
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload, {"model": "Qwen/Qwen3-14B", "messages": self.request["messages"],
            "temperature": 0.25, "max_tokens": 8192, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(self.transport.last_response_evidence["response_body"].encode(), body)
        self.assertEqual(self.transport.last_response_envelope["usage"]["total_tokens"], 5)
        self.assertFalse(self.transport.last_response_evidence["redaction_applied"])
        self.assertNotIn("response_format_sha256", self.transport.last_response_evidence["request_identity"])

    def test_adherence_response_format_sent_exactly_and_bound_to_evidence(self):
        self.request["role"] = "adherence"
        self.request["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "adherence_batch", "strict": True, "schema": {"type": "object",
                "properties": {"condition_results": {"type": "array", "maxItems": 0}},
                "required": ["condition_results"], "additionalProperties": False}}}
        original = deepcopy(self.request)
        body = b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}'
        _, opener = self.run_response(_Response(body))
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload["response_format"], original["response_format"])
        self.assertEqual(self.request, original)
        expected = hashlib.sha256(json.dumps(payload["response_format"], ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for event in self.events:
            self.assertEqual(event["request_identity"]["response_format_sha256"], expected)
            self.assertEqual(event["request_attempt_count"], 1)
            self.assertEqual(event["retry_count"], 0)
            self.assertEqual(event["fallback_count"], 0)

    def test_response_format_invalid_config_never_opens_network(self):
        valid = {"type": "json_schema", "json_schema": {"name": "audit", "schema": {"type": "object"}}}
        invalid = [None, [], {}, {"type": "json_object"}, {"type": "json_schema", "json_schema": None},
            {"type": "json_schema", "json_schema": {"name": "audit"}},
            {"type": "json_schema", "json_schema": {"schema": {}}},
            {"type": "json_schema", "json_schema": {"name": "", "schema": {}}},
            {"type": "json_schema", "json_schema": {"name": "audit", "schema": []}},
            {"type": "json_schema", "json_schema": {"name": "audit", "schema": {}, "strict": 1}},
            {"type": "json_schema", "json_schema": {"name": "audit", "schema": {"minimum": float("nan")}}},
            {"type": "json_schema", "json_schema": {"name": "audit", "schema": {"enum": ("a", "b")}}}]
        for role, value in [("adherence", item) for item in invalid] + [("supplement", valid)]:
            with self.subTest(role=role, config=value):
                self.request.update(role=role, response_format=value)
                with patch("urllib.request.build_opener") as build:
                    with self.assertRaisesRegex(ValueError, "response_format"):
                        request_once(self.transport, self.request)
                    build.assert_not_called()
                self.assertIsNone(self.transport.last_response_evidence)
                self.assertIsNone(self.transport.last_response_envelope)
        self.assertEqual(self.events, [])

    def test_provider_schema_rejection_does_not_retry_without_schema(self):
        self.request["role"] = "adherence"
        self.request["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "audit", "strict": True, "schema": {"type": "object"}}}
        error = urllib.error.HTTPError("https://offline.invalid/", 400, "unsupported schema",
            {}, io.BytesIO(b'{"error":"response_format unsupported"}'))
        with self.assertRaises(urllib.error.HTTPError):
            self.run_response(error=error)
        evidence = self.transport.last_response_evidence
        self.assertEqual(evidence["failure_classification"], "HTTP_NON_SUCCESS")
        self.assertIn("response_format_sha256", evidence["request_identity"])
        self.assertEqual(evidence["retry_count"], 0)
        self.assertEqual(evidence["fallback_count"], 0)

    def test_sensitive_body_headers_and_exception_are_redacted(self):
        body = b'{"error":"Bearer mock-sensitive-key-123", "api_key":"different-secret", "detail":"useful detail"}'
        exc = urllib.error.HTTPError("https://offline.invalid/?api_key=mock-sensitive-key-123", 400,
            "Bearer mock-sensitive-key-123", {"Authorization": "Bearer header-secret", "Set-Cookie": "cookie-secret",
            "x-request-id": "trace-safe"}, io.BytesIO(body))
        with self.assertRaises(urllib.error.HTTPError):
            self.run_response(error=exc)
        all_saved = json.dumps([self.events, self.transport.last_response_envelope])
        for secret in ("mock-sensitive-key-123", "different-secret", "header-secret", "cookie-secret"):
            self.assertNotIn(secret, all_saved)
        self.assertIn("useful detail", all_saved)
        self.assertTrue(self.transport.last_response_evidence["redaction_applied"])
        self.assertEqual(self.transport.last_response_evidence["response_headers"], {"x-request-id": "trace-safe"})

    def test_siliconcloud_trace_id_header_retained(self):
        body = b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}'
        self.run_response(_Response(body, headers={"X-SiliconCloud-Trace-Id": "mock-siliconcloud-trace"}))
        self.assertEqual(self.transport.last_response_evidence["response_headers"],
            {"x-siliconcloud-trace-id": "mock-siliconcloud-trace"})

    def test_success_redaction_blocks_returning_changed_model_content(self):
        for content, secret in (("mock-sensitive-key-123", "mock-sensitive-key-123"),
                (json.dumps({"api_key": "nested-secret"}), "nested-secret")):
            with self.subTest(content_type="nested" if content.startswith("{") else "direct"):
                self.events.clear()
                body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()
                with self.assertRaisesRegex(ValueError, "credential_redaction_changed_success_response"):
                    self.run_response(_Response(body))
                evidence = self.transport.last_response_evidence
                self.assertEqual(evidence["failure_classification"], "CREDENTIAL_REDACTION_CHANGED_SUCCESS_RESPONSE")
                self.assertTrue(evidence["body_redacted"])
                self.assertNotIn("completed", [event["evidence_event"] for event in self.events])
                self.assertNotIn(secret, json.dumps([self.events, self.transport.last_response_envelope]))

    def test_sensitive_header_only_does_not_alter_or_reject_clean_content(self):
        body = b'{"choices":[{"finish_reason":"stop","message":{"content":"unchanged output"}}]}'
        result, _ = self.run_response(_Response(body, headers={"Authorization": "Bearer omitted-secret"}))
        self.assertEqual(result, "unchanged output")
        self.assertFalse(self.transport.last_response_evidence["body_redacted"])
        self.assertTrue(self.transport.last_response_evidence["redaction_applied"])
        self.assertNotIn("omitted-secret", json.dumps(self.events))

    def test_non_utf8_saved_as_lossless_base64(self):
        body = b"\xff\xfe complete non-UTF8 mock body"
        with self.assertRaises((ValueError, UnicodeError)):
            self.run_response(_Response(body))
        e = self.transport.last_response_evidence
        self.assertEqual(base64.b64decode(e["response_body_base64"]), body)
        self.assertEqual(e["response_body_encoding"], "base64_non_utf8")

    def test_nested_error_content_secret_is_redacted(self):
        nested = json.dumps({"api_key": "nested-sensitive-key", "detail": "keep detail"})
        body = json.dumps({"error": {"message": nested}}).encode()
        with self.assertRaises(ValueError):
            self.run_response(_Response(body))
        saved = json.dumps([self.events, self.transport.last_response_envelope])
        self.assertNotIn("nested-sensitive-key", saved)
        self.assertIn("keep detail", saved)
        self.assertTrue(self.transport.last_response_evidence["body_redacted"])

    def test_timeout_exception_secret_is_redacted(self):
        with self.assertRaises(TimeoutError):
            self.run_response(error=TimeoutError("mock failure api_key=mock-sensitive-key-123"))
        self.assertNotIn("mock-sensitive-key-123", json.dumps(self.events))
        self.assertTrue(self.transport.last_response_evidence["redaction_applied"])

    def test_output_overflow_preserves_raw_business_content(self):
        self.request["max_tokens"] = 1
        body = b'{"choices":[{"finish_reason":"stop","message":{"content":"too long"}}]}'
        with self.assertRaisesRegex(ValueError, "response_output_budget_exceeded"):
            self.run_response(_Response(body))
        self.assertEqual(self.transport.last_response_evidence["response_body"].encode(), body)
        self.assertEqual(self.transport.last_response_envelope["choices"][0]["message"]["content"], "too long")

    def test_evidence_sink_runs_before_business_parsing(self):
        seen = []
        self.transport.response_evidence_sink = lambda event: seen.append(
            (event["evidence_event"], self.transport.last_response_envelope))
        with self.assertRaises(ValueError):
            self.run_response(_Response(b'{"error":"business failure"}'))
        self.assertEqual(seen[0], ("response_received", None))
        self.assertEqual(seen[1][0], "failure")

    def test_sink_failure_does_not_resend(self):
        self.transport.response_evidence_sink = Mock(side_effect=OSError("mock local evidence disk error"))
        with self.assertRaises(OSError):
            self.run_response(_Response(b'{"error":"useful raw response"}'))
        self.assertIn("useful raw response", self.transport.last_response_evidence["response_body"])
        self.assertIsNone(self.transport.last_response_envelope)

    def test_partial_response_keeps_available_bytes_not_complete_claim(self):
        class Partial(_Response):
            def read(self):
                raise IncompleteRead(b"available partial body", 100)
        with self.assertRaises(IncompleteRead):
            self.run_response(Partial(b""))
        e = self.transport.last_response_evidence
        self.assertFalse(e["response_unavailable"])
        self.assertFalse(e["response_body_complete"])
        self.assertEqual(e["response_body"], "available partial body")
        self.assertEqual(e["failure_classification"], "HTTP_RESPONSE_READ_FAILURE")

    def test_redirection_handler_will_not_follow(self):
        opener = Mock()
        opener.open.return_value = _Response(b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}')
        with patch("urllib.request.build_opener", return_value=opener) as build:
            request_once(self.transport, self.request)
        handler = build.call_args.args[0]()
        self.assertIsNone(handler.redirect_request(None, None, 302, "mock", {}, "https://different.invalid/"))
        self.assertEqual(opener.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
