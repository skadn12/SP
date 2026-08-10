from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.core import atomic_json, sha256_file
from scripts.current_method.restart_n5_spell.realizer_s1_canary import (
    DEFAULT_SOURCE_RUN, load_frozen_s1_source,
)
from scripts.current_method.restart_n5_spell.realizer_s2_s10_canary import (
    DEFAULT_S1_RUN, load_accepted_s1, load_accepted_scene_prefix,
    seed_frozen_s1, seed_frozen_scene_prefix,
)
from scripts.current_method.restart_n5_spell.stateful_graph_orchestrator import (
    StatefulGraphRoundOrchestrator, build_stateful_realizer_retry_suffix,
    stateful_dialogue_turns_sha256,
    stateful_realizer_dialogue_preserved,
    stateful_realizer_handoff_diagnostics,
    stateful_realizer_metadata_only_retry_available,
    stateful_realizer_retry_error_feedback,
    stateful_realizer_stage_diagnostics,
)
from scripts.current_method.restart_n5_spell.stateful_graph_expansion import (
    validate_and_apply_stateful_scene,
)


FAILED_S2_S10_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_realizer_s2_s10_"
    "sequential_canary_v1")
FAILED_S2_S10_V2_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_realizer_s2_s10_"
    "sequential_canary_v2")
FAILED_S2_S10_V3_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_realizer_s2_s10_"
    "sequential_canary_v3")
ACCEPTED_S2_PREFIX_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_realizer_s2_only_"
    "metadata_canary_v1")


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


class _CaptureInvalidBackend:
    model_identifier = "mock"

    def __init__(self) -> None:
        self.requests = []

    def generate_session(self, request):
        self.requests.append(request)
        return BackendResponse("{}", "mock", {"finish_reason": "stop"})


@unittest.skipUnless(DEFAULT_SOURCE_RUN.is_dir() and DEFAULT_S1_RUN.is_dir(),
                     "authoritative Planner/Graph or S1 source absent")
class RealizerS2S10CanaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_frozen_s1_source(DEFAULT_SOURCE_RUN)
        cls.s1 = load_accepted_s1(DEFAULT_S1_RUN)
        cls.contracts = cls.source["contracts_blob"]["contracts"]

    def _seed(self, root: Path):
        manifest = {"run_id": "test:s2-s10", "mode": "test"}
        atomic_json(root / "config/run_manifest.json", manifest)
        seeded = seed_frozen_s1(
            realizer_root=root, realizer_manifest=manifest,
            contracts=self.contracts, accepted_s1=self.s1)
        return manifest, seeded

    def test_s1_seed_rehydrates_full_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, seeded = self._seed(Path(directory))
            self.assertEqual(len(seeded["ledger"]["stage_states"]), 16)
            self.assertEqual(seeded["ledger"]["stage_states"][
                "E1:stage_01"]["status"], "completed")
            self.assertEqual(seeded["ledger"]["stage_states"][
                "E1:stage_02"]["status"], "assigned")

    def test_s1_seed_preserves_model_dialogue_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, seeded = self._seed(Path(directory))
            actual = [{"speaker": row["speaker"], "text": row["text"]}
                      for row in seeded["session"]["dialogue_turns"]]
            self.assertEqual(
                actual, self.s1["model_payload"]["dialogue_turns"])

    def test_seed_updates_only_run_local_ledger_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, seeded = self._seed(root)
            frozen = json.loads((root /
                "frozen/s1_seed_attestation.json").read_text())
            self.assertTrue(frozen["source_dialogue_preserved"])
            self.assertEqual(frozen["full_ledger_stage_count"], 16)
            self.assertFalse(frozen["source_realizer_raw_read"])
            self.assertFalse(frozen["source_request_ledger_read"])
            self.assertEqual(seeded["session"]["source_session_sha256"],
                             self.s1["sha256"]["session"])

    def test_seed_does_not_modify_source_s1(self) -> None:
        source_path = DEFAULT_S1_RUN / "realizer/accepted/session_01.json"
        before = sha256_file(source_path)
        with tempfile.TemporaryDirectory() as directory:
            self._seed(Path(directory))
        self.assertEqual(sha256_file(source_path), before)

    def test_executor_starts_at_s2_and_has_no_planner_intent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._seed(root)
            backend = _CaptureInvalidBackend()
            result = StatefulGraphRoundOrchestrator(
                run_root=root, manifest=manifest, contracts=self.contracts,
                tokenizer=_WhitespaceTokenizer(), realizer=backend,
                backend_mode="mock", max_attempts_per_scene=3,
                max_realizer_generations=27).run(request_limit=1)
            self.assertEqual(len(result["sessions"]), 1)
            self.assertEqual(backend.requests[0].request_id,
                             "stateful-realizer-s02-a01")
            states = [json.loads(path.read_text()) for path in
                      (root / "ledger/requests").glob("*.json")]
            self.assertEqual({row["intent"] for row in states},
                             {"stateful_realize_scene"})

    @unittest.skipUnless(ACCEPTED_S2_PREFIX_RUN.is_dir(),
                         "accepted S2 prefix run absent")
    def test_frozen_s2_prefix_revalidates_and_executor_starts_at_s3(self):
        prefix = load_accepted_scene_prefix(ACCEPTED_S2_PREFIX_RUN)
        self.assertEqual(2, prefix["scene_count"])
        self.assertEqual("S2", prefix["last_scene_id"])
        self.assertFalse(prefix["source_realizer_raw_read"])
        self.assertFalse(prefix["source_request_ledger_read"])
        self.assertTrue(prefix["source_state_ledger_read"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"run_id": "test:s3-only", "mode": "test"}
            atomic_json(root / "config/run_manifest.json", manifest)
            seeded = seed_frozen_scene_prefix(
                realizer_root=root, realizer_manifest=manifest,
                contracts=self.contracts, prefix=prefix)
            self.assertEqual(prefix["ledger"]["ledger_sha256"],
                             seeded["ledger"]["ledger_sha256"])
            backend = _CaptureInvalidBackend()
            result = StatefulGraphRoundOrchestrator(
                run_root=root, manifest=manifest, contracts=self.contracts,
                tokenizer=_WhitespaceTokenizer(), realizer=backend,
                backend_mode="mock", max_attempts_per_scene=3,
                max_realizer_generations=3).run(
                    request_limit=1, scene_limit=3)
            self.assertEqual(2, len(result["sessions"]))
            self.assertEqual("stateful-realizer-s03-a01",
                             backend.requests[0].request_id)

    @unittest.skipUnless(FAILED_S2_S10_RUN.is_dir(),
                         "historical S7 failure run absent")
    def test_historical_s7_curly_apostrophe_output_now_replays(self) -> None:
        ledger = json.loads((FAILED_S2_S10_RUN /
            "realizer/frozen/state_ledger_after_S6.json").read_text())
        wrapper = json.loads((FAILED_S2_S10_RUN /
            "realizer/raw/stateful-realizer-s07-a02.json").read_text())
        parsed = json.loads(wrapper["raw_response"])
        result = validate_and_apply_stateful_scene(
            parsed, self.contracts[6], ledger)
        self.assertEqual("accept_stateful_scene", result["decision"])
        self.assertEqual([], result["errors"])

    @unittest.skipUnless(FAILED_S2_S10_V2_RUN.is_dir(),
                         "historical S3 multi-stage failure run absent")
    def test_historical_s3_retry_targets_only_invalid_stage(self) -> None:
        contract = self.contracts[2]
        wrapper = json.loads((FAILED_S2_S10_V2_RUN /
            "realizer/raw/stateful-realizer-s03-a02.json").read_text())
        parsed = json.loads(wrapper["raw_response"])
        report = json.loads((FAILED_S2_S10_V2_RUN /
            "realizer/reports/stateful-realizer-s03-a02.json").read_text())

        diagnostics = stateful_realizer_stage_diagnostics(
            contract=contract, parsed=parsed)
        by_stage = {row["stage_id"]: row for row in diagnostics}
        self.assertFalse(by_stage["E1:stage_03"]["valid"])
        self.assertEqual(
            by_stage["E1:stage_03"]["reported_evidence_turn_index"], 6)
        self.assertEqual(
            by_stage["E1:stage_03"]["actual_indexed_speaker"],
            "Co-workers B")
        self.assertEqual(
            by_stage["E1:stage_03"]["recommended_evidence_turn_indices"],
            [5])
        self.assertTrue(by_stage["E2:stage_01"]["valid"])

        feedback = stateful_realizer_retry_error_feedback(
            contract=contract, errors=report["reasons"], parsed=parsed,
            actual_tokens=385, actual_turns=11)
        stage_feedback = [row for row in feedback
                          if row["code"] == "stage_contract_violation"]
        self.assertEqual([row["stage_id"] for row in stage_feedback],
                         ["E1:stage_03"])

        suffix = build_stateful_realizer_retry_suffix(
            contract=contract, errors=report["reasons"], parsed=parsed,
            actual_tokens=385, actual_turns=11)
        self.assertIn("Preserve dialogue_turns verbatim", suffix)
        self.assertIn('"recommended_evidence_turn_indices": [\n      5', suffix)
        self.assertIn("VALID STAGES — PRESERVE THESE UPDATES EXACTLY", suffix)
        self.assertIn('"stage_id": "E2:stage_01"', suffix)
        self.assertNotIn("final turn must contain", suffix)
        self.assertIn("do not need to be in the final dialogue turn", suffix)
        self.assertTrue(stateful_realizer_metadata_only_retry_available(
            contract=contract, errors=report["reasons"], parsed=parsed))
        self.assertTrue(stateful_realizer_dialogue_preserved(
            parsed["dialogue_turns"], parsed))
        changed = json.loads(json.dumps(parsed))
        changed["dialogue_turns"][0]["text"] += " changed"
        self.assertFalse(stateful_realizer_dialogue_preserved(
            parsed["dialogue_turns"], changed))

        frozen_suffix = build_stateful_realizer_retry_suffix(
            contract=contract,
            errors=["metadata_only_retry_dialogue_changed"], parsed=changed,
            actual_tokens=385, actual_turns=11,
            frozen_dialogue_turns=parsed["dialogue_turns"])
        self.assertIn("FROZEN DIALOGUE_TURNS — COPY VERBATIM", frozen_suffix)
        self.assertIn("restore the frozen dialogue exactly", frozen_suffix)

    @unittest.skipUnless(FAILED_S2_S10_V2_RUN.is_dir(),
                         "historical S3 multi-stage failure run absent")
    def test_historical_s3_metadata_binding_fix_replays_offline(self) -> None:
        contract = self.contracts[2]
        ledger = json.loads((FAILED_S2_S10_V2_RUN /
            "realizer/frozen/state_ledger_after_S2.json").read_text())
        wrapper = json.loads((FAILED_S2_S10_V2_RUN /
            "realizer/raw/stateful-realizer-s03-a02.json").read_text())
        parsed = json.loads(wrapper["raw_response"])
        original = json.loads(json.dumps(parsed))
        for update in parsed["candidate_stage_updates"]:
            if update["stage_id"] == "E1:stage_03":
                update["evidence_turn_index"] = 5
                update["result_speaker"] = "Co-workers A"
        result = validate_and_apply_stateful_scene(parsed, contract, ledger)
        self.assertEqual("accept_stateful_scene", result["decision"])
        self.assertEqual([], result["errors"])
        # The historical source remains a read-only fixture.
        fresh = json.loads(wrapper["raw_response"])
        self.assertEqual(original, fresh)

    @unittest.skipUnless(FAILED_S2_S10_V3_RUN.is_dir(),
                         "historical S2 metadata failure run absent")
    def test_s2_attempt3_metadata_only_replay_passes_strict_validator(self):
        contract = self.contracts[1]
        ledger = json.loads((FAILED_S2_S10_V3_RUN /
            "realizer/frozen/state_ledger_after_S1.json").read_text())
        wrapper = json.loads((FAILED_S2_S10_V3_RUN /
            "realizer/raw/stateful-realizer-s02-a03.json").read_text())
        parsed = json.loads(wrapper["raw_response"])
        report = json.loads((FAILED_S2_S10_V3_RUN /
            "realizer/reports/stateful-realizer-s02-a03.json").read_text())
        before_turns = json.loads(json.dumps(parsed["dialogue_turns"]))
        before_sha = stateful_dialogue_turns_sha256(before_turns)

        handoffs = stateful_realizer_handoff_diagnostics(
            parsed=parsed, allowed_handoff_ids=[])
        self.assertEqual(handoffs, [{
            "reported_handoff_id": "H0001",
            "allowed_handoff_ids": [],
            "valid": False,
            "error": "unknown_handoff_id",
            "required_action": (
                "omit this handoff update; no unknown ID may be registered"),
        }])
        self.assertTrue(stateful_realizer_metadata_only_retry_available(
            contract=contract, errors=report["reasons"], parsed=parsed,
            allowed_handoff_ids=[]))

        fixed = json.loads(json.dumps(parsed))
        fixed["candidate_stage_updates"][0]["evidence_turn_index"] = 12
        fixed["candidate_handoff_updates"] = []
        self.assertEqual(before_turns, fixed["dialogue_turns"])
        self.assertEqual(before_sha, stateful_dialogue_turns_sha256(
            fixed["dialogue_turns"]))
        result = validate_and_apply_stateful_scene(fixed, contract, ledger)
        self.assertEqual("accept_stateful_scene", result["decision"])
        self.assertEqual([], result["errors"])
        self.assertEqual({}, ledger["handoffs"])

        suffix = build_stateful_realizer_retry_suffix(
            contract=contract, errors=report["reasons"], parsed=parsed,
            actual_tokens=433, actual_turns=13,
            allowed_handoff_ids=[])
        self.assertIn('"recommended_evidence_turn_indices": [\n      12',
                      suffix)
        self.assertIn('"reported_handoff_id": "H0001"', suffix)
        self.assertIn('"allowed_handoff_ids": []', suffix)
        self.assertIn("candidate_handoff_updates as an empty array", suffix)
        self.assertIn("Preserve dialogue_turns verbatim", suffix)
        self.assertIn(
            "required marker is already present at turn 12 and is spoken by "
            "Co-workers A", suffix)
        self.assertIn(
            "H0001 is not a valid handoff ID. The current ledger contains no "
            "available handoffs", suffix)

    @unittest.skipUnless(FAILED_S2_S10_V3_RUN.is_dir(),
                         "historical S2 metadata failure run absent")
    def test_s2_metadata_only_requires_real_authorized_terminal_marker(self):
        contract = self.contracts[1]
        wrapper = json.loads((FAILED_S2_S10_V3_RUN /
            "realizer/raw/stateful-realizer-s02-a03.json").read_text())
        parsed = json.loads(wrapper["raw_response"])

        no_marker = json.loads(json.dumps(parsed))
        no_marker["dialogue_turns"][12]["text"] = "The lunch is over."
        self.assertFalse(stateful_realizer_metadata_only_retry_available(
            contract=contract,
            errors=["terminal_evidence_marker_missing", "unknown_handoff_id"],
            parsed=no_marker, allowed_handoff_ids=[]))

        wrong_speaker = json.loads(json.dumps(parsed))
        wrong_speaker["dialogue_turns"][12]["speaker"] = "Co-workers B"
        self.assertFalse(stateful_realizer_metadata_only_retry_available(
            contract=contract,
            errors=["stage_result_speaker_unauthorized",
                    "terminal_evidence_marker_missing", "unknown_handoff_id"],
            parsed=wrong_speaker, allowed_handoff_ids=[]))

        future = json.loads(json.dumps(parsed))
        future["dialogue_turns"][12]["text"] = (
            "Tomorrow I will say: I held out Co-workers B's chair at lunch "
            "and they acknowledged it.")
        self.assertFalse(stateful_realizer_metadata_only_retry_available(
            contract=contract,
            errors=["terminal_evidence_is_future_or_conditional",
                    "unknown_handoff_id"], parsed=future,
            allowed_handoff_ids=[]))


if __name__ == "__main__":
    unittest.main()
