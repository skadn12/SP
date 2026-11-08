from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..backends import BackendResponse
from ..core import atomic_json, sha256_bytes
from . import (
    FORBIDDEN_SOLVER_MODEL,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    ROOT_SPLIT_COUNTS,
)
from .contract import (
    audit_rebinding,
    build_prompt,
    position_variants,
    score_output,
)
from .difficulty import (
    classify_difficulty_counts,
    position_diagnostics,
    summarize_candidate_result,
)
from .gate import (
    CONTEXT_OVERFLOW,
    PROGRAM_PASS,
    PROGRAM_REJECT,
    _contract_source_identity,
    assert_solver_paths,
    build_solver_plan,
    detect_structured_output_api,
    directory_identity,
    gate_candidate,
    output_vocabulary_budget,
    validate_candidate_structure,
)
from .inventory import freeze_root_split
from .io_utils import RunLock, stable_seed, verify_sha_sidecar, write_once_json
from .request_ledger import Round0RequestLedger, request_cache_key
from .run import main
from . import run as run_module
from .solver import candidate_result_path, evaluate_candidate, load_or_recover_candidate_result


class FakeTokenizer:
    chat_template = "fake-chat-template-v1"

    def __init__(self, multiplier: int = 1):
        self.multiplier = multiplier

    def encode(self, text, add_special_tokens=False):
        return list(str(text)) * self.multiplier

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{row['role']}: {row['content']}" for row in messages)


def single_candidate() -> dict:
    return {
        "candidate_id": "r1:C1",
        "root_id": "r1",
        "question_type": "single_choice",
        "question": "Which?",
        "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "items": {},
        "correct_option": "B",
        "gold": ["B"],
        "schema_valid": False,
        "schema_errors": ["semantic_field_that_must_be_ignored"],
        "evidence_refs": [],
    }


def multi_candidate() -> dict:
    return {
        "candidate_id": "r1:C2",
        "root_id": "r1",
        "question_type": "multi_select",
        "question": "Which apply?",
        "options": {
            "A": "alpha", "B": "beta", "C": "gamma", "D": "delta", "E": "epsilon",
        },
        "items": {},
        "correct_option": None,
        "gold": ["A", "D"],
    }


def ordering_candidate() -> dict:
    return {
        "candidate_id": "r1:C3",
        "root_id": "r1",
        "question_type": "ordering",
        "question": "Order?",
        "options": {},
        "items": {"I1": "one", "I2": "two", "I3": "three", "I4": "four"},
        "correct_option": None,
        "gold": ["I2", "I1", "I4", "I3"],
    }


def expanded() -> dict:
    return {
        "root_id": "r1",
        "turns": [
            {"turn_ref": "D001", "session_ref": "P01", "speaker": "Ada", "text": "Alpha."},
            {"turn_ref": "D002", "session_ref": "R01", "speaker": "Ben", "text": "Beta."},
        ],
    }


class ContractTests(unittest.TestCase):
    def test_single_rebinding_and_canonical_scoring(self):
        candidate = single_candidate()
        variants = position_variants(candidate)
        self.assertEqual(len(variants), 4)
        self.assertTrue(audit_rebinding(candidate, variants)["pass"])
        shift = variants[1]
        display_gold = shift["displayed_gold"][0]
        self.assertNotEqual(display_gold, "B")
        scored = score_output(f"<box>{display_gold}</box>", candidate, shift)
        self.assertTrue(scored["correct"])
        self.assertEqual(scored["canonical_prediction"], ["B"])

    def test_multi_rebinding_is_set_scored_after_canonical_mapping(self):
        candidate = multi_candidate()
        variant = position_variants(candidate)[2]
        raw = "<box>" + ",".join(variant["displayed_gold"][::-1]) + "</box>"
        scored = score_output(raw, candidate, variant)
        self.assertTrue(scored["correct"])
        self.assertEqual(scored["canonical_prediction"], ["A", "D"])

    def test_ordering_rotates_display_order_without_relabeling_items(self):
        candidate = ordering_candidate()
        variants = position_variants(candidate)
        self.assertEqual(list(variants[1]["items"]), ["I2", "I3", "I4", "I1"])
        self.assertEqual(variants[1]["display_to_canonical"], {
            "I1": "I1", "I2": "I2", "I3": "I3", "I4": "I4",
        })
        scored = score_output("<box>I2,I1,I4,I3</box>", candidate, variants[3])
        self.assertTrue(scored["correct"])

    def test_prompt_contract_has_context_only_in_full_mode(self):
        tokenizer = FakeTokenizer()
        candidate = single_candidate()
        variant = position_variants(candidate)[0]
        dialogue = [{"turn_id": "D001", "speaker": "Ada", "text": "secret"}]
        no_prompt = build_prompt(
            tokenizer, candidate, variant, dialogue, context_mode="no_context",
        )
        full_prompt = build_prompt(
            tokenizer, candidate, variant, dialogue, context_mode="full_context",
        )
        self.assertNotIn("secret", no_prompt)
        self.assertIn("secret", full_prompt)
        self.assertIn("A. alpha", full_prompt)


class GateTests(unittest.TestCase):
    def test_contract_identity_covers_runtime_modules_but_not_tests(self):
        names = {Path(row["path"]).name for row in _contract_source_identity()}
        self.assertTrue({
            "__init__.py", "contract.py", "difficulty.py", "gate.py", "inventory.py",
            "io_utils.py", "request_ledger.py", "run.py", "solver.py", "backends.py",
            "core.py", "prompts.py", "structured_qa_box_protocol.py", "vllm_backends.py",
        } <= names)
        self.assertNotIn("test_round0.py", names)

    def test_directory_identity_hashes_every_file(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "ordinary.txt").write_text("frozen", encoding="utf-8")
            identity = directory_identity(root)
            self.assertEqual(len(identity["files"]), 1)
            self.assertIn("sha256", identity["files"][0])

    def test_legal_output_vocabulary_fits_generation_budget(self):
        budget = output_vocabulary_budget(FakeTokenizer())
        self.assertEqual(set(budget), {"single_choice", "multi_select", "ordering"})
        self.assertTrue(all(
            row["max_legal_output_tokens"] < row["max_tokens"]
            for row in budget.values()
        ))

    def test_structured_api_matches_installed_vllm_capability(self):
        api_name, version = detect_structured_output_api()
        self.assertTrue(version)
        self.assertTrue(
            api_name.startswith("guided_decoding=")
            or api_name.startswith("structured_outputs=")
        )

    def test_gate_ignores_semantic_schema_and_evidence_fields(self):
        row = gate_candidate(
            single_candidate(), expanded(), FakeTokenizer(), root_id="r1",
        )
        self.assertEqual(row["gate_status"], PROGRAM_PASS)
        self.assertFalse(row["semantic_judgment_applied"])
        self.assertFalse(row["evidence_gate_applied"])

    def test_gate_rejects_duplicate_options_and_invalid_gold(self):
        candidate = multi_candidate()
        candidate["options"]["E"] = " alpha "
        candidate["gold"] = []
        errors = validate_candidate_structure(
            candidate, root_id="r1", dialogue_root_id="r1",
        )
        self.assertIn("options_duplicate_text", errors)
        self.assertIn("multi_gold_nonempty_unique_labels", errors)
        row = gate_candidate(candidate, expanded(), FakeTokenizer(), root_id="r1")
        self.assertEqual(row["gate_status"], PROGRAM_REJECT)

    def test_gate_marks_context_overflow(self):
        row = gate_candidate(
            single_candidate(), expanded(), FakeTokenizer(multiplier=MAX_MODEL_LEN), root_id="r1",
        )
        self.assertEqual(row["gate_status"], CONTEXT_OVERFLOW)

    def test_forbidden_model_is_rejected_before_existence_check(self):
        with self.assertRaisesRegex(RuntimeError, "forbidden_solver_model"):
            assert_solver_paths(FORBIDDEN_SOLVER_MODEL, Path("/root/autodl-tmp/Qwen3-4B"))


class SplitAndPlanTests(unittest.TestCase):
    def _inventory(self):
        return {
            "roots": [
                {
                    "root_id": f"r{index:03d}",
                    "candidate_count": 3,
                    "candidate_pool_sha256": f"pool-{index}",
                    "dialogue_sha256": f"dialogue-{index}",
                }
                for index in range(101)
            ]
        }

    def test_split_is_stable_and_test_is_unexposed(self):
        exposure = {"exposed_root_ids": [f"r{index:03d}" for index in range(25)]}
        rows1, summary1 = freeze_root_split(self._inventory(), exposure)
        rows2, summary2 = freeze_root_split(self._inventory(), exposure)
        self.assertEqual(rows1, rows2)
        self.assertEqual(summary1, summary2)
        self.assertEqual(summary1["root_counts"], ROOT_SPLIT_COUNTS)
        self.assertFalse(any(row["analysis_exposed"] for row in rows1 if row["split"] == "test"))

    def test_split_blocks_when_unexposed_pool_is_too_small(self):
        exposed = [f"r{index:03d}" for index in range(90)]
        with self.assertRaisesRegex(RuntimeError, "insufficient_unexposed"):
            freeze_root_split(self._inventory(), {"exposed_root_ids": exposed})

    def test_plan_uses_exact_type_weight_and_six_candidate_smoke(self):
        split_rows = []
        gate_rows = []
        for root_index in range(3):
            root_id = f"r{root_index}"
            split_rows.append({"root_id": root_id, "split": "train"})
            for qtype in ("single_choice", "multi_select", "ordering"):
                gate_rows.append({
                    "root_id": root_id,
                    "candidate_id": f"{root_id}:{qtype}",
                    "question_type": qtype,
                    "gate_status": PROGRAM_PASS,
                })
        plan = build_solver_plan(gate_rows, split_rows)
        self.assertEqual(plan["planned_request_count"], 3 * (24 + 30 + 24))
        self.assertEqual(plan["smoke_candidate_count"], 6)
        self.assertEqual(plan["smoke_request_count"], 156)


class CliSafetyTests(unittest.TestCase):
    def test_smoke_without_live_stops_before_reading_or_loading_any_run(self):
        with self.assertRaisesRegex(RuntimeError, "--live_required_for_smoke"):
            main(["--phase", "smoke", "--run-root", "/path/that/does/not/exist"])

    def test_solver_runtime_contract_is_checked_before_gpu_or_backend(self):
        with tempfile.TemporaryDirectory() as value:
            run_root = Path(value) / "run"
            source_run = Path(value) / "source"
            run_root.mkdir()
            source_run.mkdir()
            write_once_json(run_root / "run_identity.json", {
                "probe_version": run_module.PROBE_VERSION,
                "run_root": str(run_root.resolve()),
                "source_run": str(source_run.resolve()),
            })
            write_once_json(run_root / "preflight_report.json", {"status": "PASS"})
            with patch.object(run_module, "_tokenizer", return_value=FakeTokenizer()), \
                    patch.object(
                        run_module, "validate_solver_runtime",
                        side_effect=RuntimeError("contract sentinel"),
                    ), \
                    patch.object(
                        run_module, "assert_gpu_idle",
                        side_effect=AssertionError("GPU check ran too early"),
                    ), \
                    patch.object(
                        run_module, "create_backend",
                        side_effect=AssertionError("backend loaded too early"),
                    ):
                with self.assertRaisesRegex(RuntimeError, "contract sentinel"):
                    main([
                        "--phase", "smoke", "--live",
                        "--run-root", str(run_root),
                        "--source-run", str(source_run),
                    ])


class LedgerTests(unittest.TestCase):
    @staticmethod
    def _contract(prompt="prompt"):
        return {
            "solver_contract_sha256": "solver-sha",
            "root_id": "r1",
            "prompt_sha256": sha256_bytes(prompt.encode()),
            "context_mode": "full_context",
            "variant_id": "shift_0",
            "parser_version": "boxed_letter_v1",
            "backend_identity": {"model": "base"},
            "generation_config": {"temperature": 0.7},
        }

    def test_sha_sidecar_is_verified(self):
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "manifest.json"
            write_once_json(path, {"a": 1})
            self.assertEqual(len(verify_sha_sidecar(path)), 64)
            path.write_text('{"a": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "sha256_sidecar_mismatch"):
                verify_sha_sidecar(path)

    def test_completed_response_reuses_only_exact_contract(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            ledger = Round0RequestLedger(root, solver_contract_sha256="solver-sha")
            prompt = "prompt"
            contract = self._contract(prompt)
            calls = []

            def call():
                calls.append(True)
                return BackendResponse("<box>A</box>", "base", {})

            first, reused1 = ledger.invoke(contract=contract, prompt=prompt, call=call)
            second, reused2 = ledger.invoke(contract=contract, prompt=prompt, call=call)
            self.assertEqual(first.raw_text, second.raw_text)
            self.assertFalse(reused1)
            self.assertTrue(reused2)
            self.assertEqual(len(calls), 1)
            drift = {**contract, "generation_config": {"temperature": 0.8}}
            self.assertNotEqual(request_cache_key(contract), request_cache_key(drift))

    def test_interrupted_attempt_is_recovered_and_retried_with_history(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            ledger = Round0RequestLedger(root, solver_contract_sha256="solver-sha")
            prompt = "prompt"
            contract = self._contract(prompt)
            with self.assertRaises(KeyboardInterrupt):
                ledger.invoke(
                    contract=contract,
                    prompt=prompt,
                    call=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
                )
            self.assertEqual(ledger.states()[0]["status"], "interrupted")
            recovery = ledger.recover_orphaned_requests()
            self.assertEqual(recovery["retryable_request_count"], 1)
            response, reused = ledger.invoke(
                contract=contract,
                prompt=prompt,
                call=lambda: BackendResponse("<box>A</box>", "base", {}),
            )
            self.assertEqual(response.raw_text, "<box>A</box>")
            self.assertFalse(reused)
            state = ledger.states()[0]
            self.assertEqual(state["status"], "completed")
            self.assertEqual(
                [row["status"] for row in state["attempts"]],
                ["interrupted", "completed"],
            )

    def test_reserved_attempt_without_envelope_becomes_retryable(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            ledger = Round0RequestLedger(root, solver_contract_sha256="solver-sha")
            prompt = "prompt"
            contract = self._contract(prompt)
            with self.assertRaises(KeyboardInterrupt):
                ledger.invoke(
                    contract=contract,
                    prompt=prompt,
                    call=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
                )
            cache_key = request_cache_key(contract)
            state_path = ledger._state_path("r1", cache_key)
            state = ledger.states()[0]
            state["status"] = "reserved"
            state["attempts"][-1]["status"] = "reserved"
            atomic_json(state_path, state)
            recovery = ledger.recover_orphaned_requests()
            self.assertEqual(recovery["retryable_request_count"], 1)
            ledger.invoke(
                contract=contract,
                prompt=prompt,
                call=lambda: BackendResponse("<box>A</box>", "base", {}),
            )
            self.assertEqual(
                [row["status"] for row in ledger.states()[0]["attempts"]],
                ["abandoned", "completed"],
            )

    def test_response_envelope_recovers_without_second_generation(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            ledger = Round0RequestLedger(root, solver_contract_sha256="solver-sha")
            prompt = "prompt"
            contract = self._contract(prompt)
            with self.assertRaises(KeyboardInterrupt):
                ledger.invoke(
                    contract=contract,
                    prompt=prompt,
                    call=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
                )
            cache_key = request_cache_key(contract)
            raw = "<box>A</box>"
            write_once_json(ledger._response_path(cache_key), {
                "response_envelope_version": "h4_round0_response_envelope_v1",
                "request_cache_key": cache_key,
                "attempt_id": 1,
                "raw_text": raw,
                "raw_response_sha256": sha256_bytes(raw.encode()),
                "model_identifier": "base",
                "response_metadata": {"finish_reason": "stop"},
            }, with_sha=False)
            recovery = ledger.recover_orphaned_requests()
            self.assertEqual(recovery["response_envelope_recovered_count"], 1)
            response, reused = ledger.invoke(
                contract=contract,
                prompt=prompt,
                call=lambda: self.fail("recovered response must not regenerate"),
            )
            self.assertTrue(reused)
            self.assertEqual(response.raw_text, raw)
            self.assertEqual(ledger.states()[0]["attempts"][-1]["status"], "completed")

    def test_run_lock_rejects_concurrent_holder_and_can_be_reacquired(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            with RunLock(root):
                with self.assertRaisesRegex(RuntimeError, "solver_run_lock_busy"):
                    with RunLock(root):
                        self.fail("second lock must not be acquired")
            with RunLock(root):
                pass


class FakeBackend:
    model_identifier = "/root/autodl-tmp/Qwen3-4B"
    tokenizer_path = Path("/root/autodl-tmp/Qwen3-4B")
    max_model_len = MAX_MODEL_LEN

    def __init__(self):
        self.seeds = []

    def generate(self, request):
        self.seeds.append((request.evaluation_mode, request.order_id, request.target_slot, request.sampling_seed))
        return BackendResponse(request.canonical_answer_text.join(("<box>", "</box>")), self.model_identifier, {})

    def generate_structured(self, request, *, question_type):
        self.seeds.append((request.evaluation_mode, request.order_id, request.target_slot, request.sampling_seed))
        return BackendResponse(f"<box>{request.canonical_answer_text}</box>", self.model_identifier, {})


class OneFailureBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.failed = False

    def generate(self, request):
        if not self.failed:
            self.failed = True
            raise TimeoutError("isolated timeout")
        return super().generate(request)


class FirstThreeFailuresBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.calls <= 3:
            raise TimeoutError(f"timeout {self.calls}")
        return super().generate(request)


class OneTruncatedBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.truncated = False

    def generate(self, request):
        if not self.truncated:
            self.truncated = True
            return BackendResponse(
                f"<box>{request.canonical_answer_text}</box>",
                self.model_identifier,
                {"finish_reason": "length"},
            )
        return super().generate(request)


class SolverTests(unittest.TestCase):
    @staticmethod
    def _gate(tokenizer):
        gate = gate_candidate(single_candidate(), expanded(), tokenizer, root_id="r1")
        gate.update({"candidate_pool_sha256": "pool", "dialogue_sha256": "dialogue"})
        return gate

    def test_evaluate_candidate_uses_paired_seeds_and_canonical_scoring(self):
        tokenizer = FakeTokenizer()
        candidate = single_candidate()
        gate = gate_candidate(candidate, expanded(), tokenizer, root_id="r1")
        gate.update({
            "candidate_pool_sha256": "pool",
            "dialogue_sha256": "dialogue",
        })
        with tempfile.TemporaryDirectory() as value:
            backend = FakeBackend()
            ledger = Round0RequestLedger(Path(value), solver_contract_sha256="solver-sha")
            result = evaluate_candidate(
                candidate=candidate,
                expanded=expanded(),
                gate_row=gate,
                tokenizer=tokenizer,
                backend=backend,
                ledger=ledger,
                solver_contract_sha256="solver-sha",
            )
            self.assertEqual(result["request_count"], 24)
            self.assertEqual(result["technical_failure_count"], 0)
            self.assertTrue(all(row["correct"] for row in result["requests"]))
            self.assertTrue(all("response_reused" not in row for row in result["requests"]))
            replay = evaluate_candidate(
                candidate=candidate,
                expanded=expanded(),
                gate_row=gate,
                tokenizer=tokenizer,
                backend=backend,
                ledger=ledger,
                solver_contract_sha256="solver-sha",
            )
            self.assertEqual(result, replay)
            self.assertEqual(len(backend.seeds), 24)
            for slot in range(3):
                seeds = {seed for _, _, observed_slot, seed in backend.seeds if observed_slot == slot}
                self.assertEqual(seeds, {stable_seed("r1", "r1:C1", slot)})

    def test_isolated_solver_failure_becomes_technical_candidate(self):
        tokenizer = FakeTokenizer()
        candidate = single_candidate()
        gate = gate_candidate(candidate, expanded(), tokenizer, root_id="r1")
        gate.update({"candidate_pool_sha256": "pool", "dialogue_sha256": "dialogue"})
        with tempfile.TemporaryDirectory() as value:
            backend = OneFailureBackend()
            ledger = Round0RequestLedger(Path(value), solver_contract_sha256="solver-sha")
            result = evaluate_candidate(
                candidate=candidate,
                expanded=expanded(),
                gate_row=gate,
                tokenizer=tokenizer,
                backend=backend,
                ledger=ledger,
                solver_contract_sha256="solver-sha",
            )
            self.assertEqual(result["request_count"], 24)
            self.assertEqual(result["technical_failure_count"], 1)
            self.assertEqual(ledger.summary()["failed_request_count"], 1)
            self.assertEqual(
                summarize_candidate_result(result)["difficulty_bucket"],
                "technical_excluded",
            )

    def test_cached_failures_do_not_retrigger_consecutive_failure_stop(self):
        tokenizer = FakeTokenizer()
        candidate = single_candidate()
        gate = self._gate(tokenizer)
        with tempfile.TemporaryDirectory() as value:
            ledger = Round0RequestLedger(Path(value), solver_contract_sha256="solver-sha")
            backend = FirstThreeFailuresBackend()
            with self.assertRaisesRegex(RuntimeError, "consecutive_solver_failure_stop"):
                evaluate_candidate(
                    candidate=candidate,
                    expanded=expanded(),
                    gate_row=gate,
                    tokenizer=tokenizer,
                    backend=backend,
                    ledger=ledger,
                    solver_contract_sha256="solver-sha",
                )
            result = evaluate_candidate(
                candidate=candidate,
                expanded=expanded(),
                gate_row=gate,
                tokenizer=tokenizer,
                backend=backend,
                ledger=ledger,
                solver_contract_sha256="solver-sha",
            )
            self.assertEqual(result["request_count"], 24)
            self.assertEqual(result["technical_failure_count"], 3)
            replay = evaluate_candidate(
                candidate=candidate,
                expanded=expanded(),
                gate_row=gate,
                tokenizer=tokenizer,
                backend=backend,
                ledger=ledger,
                solver_contract_sha256="solver-sha",
            )
            self.assertEqual(result, replay)

    def test_finish_reason_length_is_never_scored(self):
        tokenizer = FakeTokenizer()
        candidate = single_candidate()
        with tempfile.TemporaryDirectory() as value:
            result = evaluate_candidate(
                candidate=candidate,
                expanded=expanded(),
                gate_row=self._gate(tokenizer),
                tokenizer=tokenizer,
                backend=OneTruncatedBackend(),
                ledger=Round0RequestLedger(Path(value), solver_contract_sha256="solver-sha"),
                solver_contract_sha256="solver-sha",
            )
            truncated = result["requests"][0]
            self.assertEqual(truncated["parse_status"], "output_truncated")
            self.assertEqual(truncated["failure_type"], "solver_output_truncated")
            self.assertFalse(truncated["scored"])
            self.assertIsNone(truncated["correct"])

    def test_candidate_result_without_sidecar_is_rebuilt_without_generation(self):
        tokenizer = FakeTokenizer()
        candidate = single_candidate()
        gate = self._gate(tokenizer)
        with tempfile.TemporaryDirectory() as value:
            run_root = Path(value)
            backend = FakeBackend()
            ledger = Round0RequestLedger(run_root, solver_contract_sha256="solver-sha")
            result = evaluate_candidate(
                candidate=candidate,
                expanded=expanded(),
                gate_row=gate,
                tokenizer=tokenizer,
                backend=backend,
                ledger=ledger,
                solver_contract_sha256="solver-sha",
            )
            calls_before_recovery = len(backend.seeds)
            result_path = candidate_result_path(run_root, candidate["candidate_id"])
            atomic_json(result_path, result)
            recovered = load_or_recover_candidate_result(
                run_root=run_root,
                candidate=candidate,
                expanded=expanded(),
                gate_row=gate,
                tokenizer=tokenizer,
                backend=backend,
                ledger=ledger,
                solver_contract_sha256="solver-sha",
            )
            self.assertEqual(recovered, result)
            self.assertEqual(len(backend.seeds), calls_before_recovery)
            self.assertTrue(result_path.with_name(result_path.name + ".sha256").is_file())


class DifficultyTests(unittest.TestCase):
    def test_all_mutually_exclusive_buckets(self):
        cases = [
            ({"technical": True, "full_correct": 5, "full_total": 12, "no_correct": 1, "no_total": 12}, "technical_excluded"),
            ({"full_correct": 0, "full_total": 12, "no_correct": 0, "no_total": 12}, "full_zero_review"),
            ({"full_correct": 2, "full_total": 12, "no_correct": 3, "no_total": 12}, "full_context_worse"),
            ({"full_correct": 2, "full_total": 12, "no_correct": 2, "no_total": 12}, "hard_candidate"),
            ({"full_correct": 12, "full_total": 12, "no_correct": 11, "no_total": 12}, "full_context_resolved"),
            ({"full_correct": 12, "full_total": 12, "no_correct": 12, "no_total": 12}, "easy"),
        ]
        for values, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_difficulty_counts(**values), expected)

    @staticmethod
    def _requests(position_predictions):
        rows = []
        for shift, values in enumerate(position_predictions):
            for slot, value in enumerate(values):
                rows.append({
                    "variant_id": f"shift_{shift}",
                    "context_mode": "full_context",
                    "slot_id": slot,
                    "scored": True,
                    "correct": value == "A",
                    "canonical_prediction": [value],
                })
        return rows

    def test_position_mechanisms(self):
        cases = [
            ([['A'] * 3] * 4, "none"),
            ([['A', 'B', 'A']] * 4, "sampling_only"),
            ([['A'] * 3, ['B'] * 3, ['A'] * 3, ['A'] * 3], "position_only"),
            ([['A', 'B', 'A'], ['B'] * 3, ['A'] * 3, ['A'] * 3], "mixed"),
        ]
        for predictions, expected in cases:
            with self.subTest(expected=expected):
                diagnostic = position_diagnostics(
                    self._requests(predictions),
                    question_type="single_choice",
                    context_mode="full_context",
                )
                self.assertEqual(diagnostic["position_mechanism"], expected)

    def test_candidate_summary_uses_full_context_mechanism(self):
        full = self._requests([['A', 'B', 'A'], ['B'] * 3, ['A'] * 3, ['A'] * 3])
        no = [{**row, "context_mode": "no_context"} for row in self._requests([['B'] * 3] * 4)]
        result = {"question_type": "single_choice", "requests": no + full}
        summary = summarize_candidate_result(result)
        self.assertEqual(summary["difficulty_bucket"], "hard_candidate")
        self.assertEqual(summary["position_mechanism"], "mixed")
        self.assertTrue(summary["context_helpful"])


if __name__ == "__main__":
    unittest.main()
