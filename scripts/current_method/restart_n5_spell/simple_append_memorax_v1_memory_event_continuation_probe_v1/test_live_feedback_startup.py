"""Exercise live startup with offline gateways before any model request."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import live
from .profiles import R1_ALIGNED_FEEDBACK_PROFILE


class _ReachedWorkflow(BaseException):
    pass


class FeedbackLiveStartupTest(unittest.TestCase):
    def test_r2_startup_keeps_history_and_blocks_memory_writes(self):
        prior = {"root_id": "fixture", "turn_count": 2, "turns": [
            {"turn_ref": "fixture:P01:T001", "session_ref": "parent",
             "speaker": "A", "text": "We booked the workshop."},
            {"turn_ref": "R01:S01:T001", "session_ref": "r01",
             "speaker": "B", "text": "The booking is now confirmed."},
        ]}
        prior["complete_dialogue_sha256"] = hashlib.sha256(json.dumps(
            prior, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        backend = Mock()
        backend.calls = []
        backend.search_buckets.return_value = {"buckets": []}

        def reached(**kwargs):
            self.assertTrue(kwargs["no_memorax_writes"])
            self.assertTrue(kwargs["reuse_existing_namespace"])
            self.assertEqual(kwargs["feedback_dialogue"]["turns"], [
                {key: row[key] for key in ("turn_ref", "speaker", "text")}
                for row in prior["turns"]])
            gateway = kwargs["memorax"]
            self.assertEqual(gateway.search_buckets({}, request_id="read"),
                             {"buckets": []})
            with self.assertRaisesRegex(RuntimeError, "memorax_add_blocked"):
                gateway.add({}, request_id="forbidden")
            with self.assertRaisesRegex(RuntimeError, "memorax_add_status_blocked"):
                gateway.add_status("task", request_id="forbidden")
            backend.add.assert_not_called()
            backend.add_status.assert_not_called()
            raise _ReachedWorkflow()

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            directory = Path(tmp)
            input_path = directory / "input.json"
            input_path.write_text(json.dumps(prior))
            feedback_path = directory / "feedback.json"
            feedback_path.write_text(json.dumps({"root_id": "fixture"}))
            for name in ("RequestLedger", "FreezeGuard", "_config",
                         "load_memorax_config", "SimpleAppendLiveCaller",
                         "OpenAICompatibleChallengerClient"):
                stack.enter_context(patch.object(live, name))
            stack.enter_context(patch.object(live, "MemoraxClient", return_value=backend))
            stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained",
                                      return_value=Mock()))
            workflow = stack.enter_context(patch.object(
                live, "execute_round", autospec=True, side_effect=reached))
            with self.assertRaises(_ReachedWorkflow):
                live.execute_live(
                    report={"decision": "pass", "root_id": "fixture",
                            "source_manifest": {"manifest_sha256": "a" * 64}},
                    input_path=input_path, run_root=directory / "run",
                    config_path=directory / "config.json",
                    tokenizer_path=directory / "tokenizer",
                    credentials_path=directory / "credentials",
                    allow_insecure_http=False, reuse_existing_namespace=True,
                    round_index=2, no_memorax_writes=True,
                    execution_profile=R1_ALIGNED_FEEDBACK_PROFILE,
                    feedback_input=feedback_path)
            workflow.assert_called_once()


if __name__ == "__main__":
    unittest.main()
