from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.live_backends import ChallengerAPIConfig
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    ROUND_VALIDATOR_MAX_OUTPUT_TOKENS as ORCHESTRATOR_MAX_OUTPUT_TOKENS,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_transport_v11 import (
    ROUND_VALIDATOR_MAX_OUTPUT_TOKENS as TRANSPORT_MAX_OUTPUT_TOKENS,
    build_frozen_round_validator_transport_request_v11,
    round_validator_transport_configuration_v11,
    validate_frozen_round_validator_request_v11,
)


RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_31"
)
RUN_SHA = "a883d35655c68bb4b510088e0876f87b9e8303b5ea98985937f10d5a0afa10e7"


class Canary31TransportFreezeAlignmentTests(unittest.TestCase):
    def test_history_is_unchanged_and_failure_preceded_network_send(self):
        self.assertEqual(hash_tree(RUN)["tree_sha256"], RUN_SHA)
        attempt = json.loads(
            (RUN / "attempts/round_validator_attempt_01.json").read_text())
        self.assertFalse(attempt["request_completed"])
        self.assertIn("12288!=8192", attempt["runtime_error"])
        self.assertEqual(attempt["completion_tokens_local"], 0)

    def test_orchestrator_transport_and_wire_body_share_one_limit(self):
        self.assertEqual(ORCHESTRATOR_MAX_OUTPUT_TOKENS, 12_288)
        self.assertEqual(TRANSPORT_MAX_OUTPUT_TOKENS, 12_288)
        self.assertEqual(round_validator_transport_configuration_v11()[
            "max_output_tokens"], 12_288)
        request = build_frozen_round_validator_transport_request_v11(
            config=ChallengerAPIConfig(
                base_url="https://api.invalid/v1", api_key="not-used",
                model="Qwen/Qwen3-14B", timeout_seconds=180, temperature=0.25),
            prompt="frozen validator prompt",
            authoritative_input_sha256="a" * 64,
            orchestration_seed=61555,
            prompt_schema_contract_sha256="b" * 64,
        )
        self.assertEqual(request.body["max_tokens"], 12_288)
        self.assertTrue(validate_frozen_round_validator_request_v11(request)["valid"])


if __name__ == "__main__":
    unittest.main()
