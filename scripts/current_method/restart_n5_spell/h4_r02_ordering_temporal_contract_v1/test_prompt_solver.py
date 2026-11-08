"""Offline checks of the Writer prompt and actual Solver read entry."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from ..core import atomic_json, sha256_file
from ..h4_qa_mining_hybrid_generation_v1 import contract_guided_generator as guided
from ..h4_qa_mining_hybrid_generation_v1 import free_exploration_generator as free
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_qa_multiroot_k3_t08_probe_v1 import solver_adapter as solver
from ..h4_qa_multiroot_k3_t08_probe_v1.permutation import audit_gold_rebinding, position_variants
from ..h4_qa_multiroot_k3_t08_probe_v1.validator import validate_candidate
from .contract import REGISTRY_FILENAME, VERSION, ordering_schema
from .guidance import synthetic_example


class TextTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


def projected_fixture(directory: Path, node_count: int = 4):
    example = synthetic_example(node_count)
    policy = {"version": VERSION, "node_count": node_count}
    raw_path = directory / "raw" / "responses" / "fixture-primary.json.txt"
    atomic_json(raw_path, example["raw"])
    context = {
        "root_dir": directory, "run_id": "fixture-run", "arm_id": "TEMPORAL_CONTRACT_ORDERING",
        "root_id": "fixture-root", "intent_id": "intent-1", "source": "contract_guided",
        "stage": "primary", "request_id": "fixture-primary", "raw_sha256": sha256_file(raw_path),
    }
    registry = {key: context[key] for key in ("run_id", "arm_id", "root_id")}
    registry.update({"ordering_policy": policy, "requests": {
        context["request_id"]: {key: context[key] for key in ("intent_id", "source", "stage", "raw_sha256")},
    }})
    atomic_json(directory / REGISTRY_FILENAME, registry)
    normalized = normalize_batch(example["raw"], intent_id="intent-1", source="contract_guided",
                                 contract={}, expanded=example["expanded"], ordering_policy=policy,
                                 ordering_context=context)
    candidate = normalized["candidates"][0]
    candidate["candidate_id"] = "fixture-root:" + candidate["candidate_id"]
    candidate["root_id"] = context["root_id"]
    return example, candidate, policy, context


class PromptSolverTests(unittest.TestCase):
    def test_complete_examples_use_real_schema_normalizer_and_final_validator(self):
        for n in (4, 5, 6):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                example, candidate, _, _ = projected_fixture(Path(tmp), n)
                Draft202012Validator(ordering_schema(n)).validate(example["raw"]["candidates"][0])
                self.assertTrue(candidate["schema_valid"], candidate["schema_errors"])
                result = validate_candidate(candidate, valid_turn_refs={row["turn_ref"] for row in example["expanded"]["turns"]})
                self.assertEqual("pass", result["decision"], result)
                self.assertEqual(n, len(candidate["gold"]))

    def test_prompt_modes_primary_repair_and_legacy_identity(self):
        expanded = {"turns": [{"turn_ref": "CURRENT-T1", "speaker": "Someone", "text": "Current dialogue only."}]}
        intent = {"intent_id": "original-intent", "intent_type": "temporal", "discriminative_variable": "original variable"}
        raw = {"candidates": [], "raw_marker": "UNPROJECTED_PRIMARY_RAW"}
        legacy = {"version": "legacy", "node_count": 4}
        self.assertEqual(guided.build_prompt(expanded, intent), guided.build_prompt(expanded, intent, ordering_policy=legacy))
        self.assertEqual(free.build_prompt(expanded, intent), free.build_prompt(expanded, intent, ordering_policy=legacy))
        self.assertEqual(guided.build_repair_prompt(expanded, intent, raw, "contract_guided"),
                         guided.build_repair_prompt(expanded, intent, raw, "contract_guided", ordering_policy=legacy))
        self.assertEqual(free.build_repair_prompt(expanded, intent, raw),
                         free.build_repair_prompt(expanded, intent, raw, ordering_policy=legacy))
        for n in (4, 5, 6):
            policy = {"version": VERSION, "node_count": n}
            prompts = [guided.build_prompt(expanded, intent, ordering_policy=policy),
                       guided.build_repair_prompt(expanded, intent, raw, "contract_guided", ordering_policy=policy),
                       free.build_prompt(expanded, intent, ordering_policy=policy),
                       free.build_repair_prompt(expanded, intent, raw, ordering_policy=policy)]
            for result in prompts:
                prompt = result["prompt"]
                self.assertIn(f"exactly {n} event_nodes", prompt)
                self.assertNotIn("ordering items I1-I4 with a permutation gold", prompt)
                self.assertNotIn("ordering I1-I4 + permutation gold", prompt)
                self.assertIn("EXAMPLE_ONLY_T1", prompt)
                self.assertIn("not current-root facts or usable refs", prompt)
                self.assertIn("omit gold or set it to null", prompt)
                self.assertIn("CURRENT-T1 | Someone | Current dialogue only.", prompt)
            for result in (prompts[1], prompts[3]):
                self.assertTrue(result["prompt"].endswith(json.dumps(raw, ensure_ascii=False)))
            self.assertEqual({"single_choice": 2, "multi_select": 1, "ordering": 1}, prompts[0]["expected_counts"])
            self.assertEqual({"single_choice": 4, "multi_select": 2, "ordering": 2}, prompts[2]["expected_counts"])

    def test_rebinding_preserves_positions_and_item_identity_four_to_six(self):
        for n in (4, 5, 6):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                _, candidate, _, _ = projected_fixture(Path(tmp), n)
                variants = position_variants(candidate)
                self.assertTrue(audit_gold_rebinding(candidate, variants)["pass"])
                self.assertNotEqual(list(variants[1]["items"]), list(candidate["items"]))
                wrong_gold = copy.deepcopy(variants)
                wrong_gold[0]["gold"].reverse()
                self.assertFalse(audit_gold_rebinding(candidate, wrong_gold)["pass"])
                wrong_text = copy.deepcopy(variants)
                wrong_text[0]["items"]["I1"], wrong_text[0]["items"]["I2"] = wrong_text[0]["items"]["I2"], wrong_text[0]["items"]["I1"]
                self.assertFalse(audit_gold_rebinding(candidate, wrong_text)["pass"])

    def test_solver_render_has_only_original_visible_fields(self):
        forbidden = ("event_nodes", "temporal_relations", "temporal_basis", "derived_gold", "derivation_status", "semantic_support_status", "SECRET_SIDECAR", "SECRET_EDGE", "SECRET_GOLD")
        for n in (4, 5, 6):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                example, candidate, _, _ = projected_fixture(Path(tmp), n)
                candidate.update({"event_nodes": "SECRET_SIDECAR", "temporal_relations": "SECRET_EDGE", "gold": ["SECRET_GOLD"]})
                variant = {"items": candidate["items"], "gold": ["SECRET_GOLD"]}
                rendered = solver._structured_prompt(TextTokenizer(), candidate, variant, solver._dialogue_rows(example["expanded"]))
                self.assertIn(candidate["question"], rendered)
                self.assertIn(f"I{n}. ", rendered)
                for marker in forbidden:
                    self.assertNotIn(marker, rendered)
                item_lines = rendered.split("Items:\n", 1)[1].split("\n\n", 1)[0].splitlines()
                self.assertEqual(list(candidate["items"]), [line.split(".", 1)[0] for line in item_lines])

    def test_actual_solver_entry_resolves_registry_and_rejects_tampering_before_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            example, candidate, policy, _ = projected_fixture(root)
            arguments = {"expanded": example["expanded"], "root_id": "fixture-root", "run_root": root,
                         "ledger": None, "tokenizer": TextTokenizer(), "backend": None}
            stub = {"scored": True, "correct": True, "failure_type": None, "parse_status": "success"}
            with patch.object(solver, "_invoke_rollout", return_value=stub) as invoke:
                result = solver.evaluate_candidates(candidates=[candidate], **arguments)
                self.assertEqual(24, invoke.call_count)
                self.assertTrue(result[0]["gold_rebinding_audit"]["pass"])
            corruptions = []
            wrong = copy.deepcopy(candidate)
            wrong["gold"].reverse()
            corruptions.append(wrong)
            wrong = copy.deepcopy(candidate)
            wrong["items"]["I1"] = "Another event"
            corruptions.append(wrong)
            wrong = copy.deepcopy(candidate)
            wrong.pop("ordering_derivation")
            wrong.pop("ordering_contract_version")
            corruptions.append(wrong)
            wrong = copy.deepcopy(candidate)
            wrong["candidate_id"] = "fixture-root:other-intent-C-Q01"
            corruptions.append(wrong)
            for wrong in corruptions:
                with patch.object(solver, "_invoke_rollout") as invoke, self.assertRaises(ValueError):
                    solver.evaluate_candidates(candidates=[wrong], **arguments)
                invoke.assert_not_called()
            altered_expanded = copy.deepcopy(example["expanded"])
            altered_expanded["turns"][0]["text"] = "Changed transcript"
            with patch.object(solver, "_invoke_rollout") as invoke, self.assertRaises(ValueError):
                solver.evaluate_candidates(candidates=[candidate], **{**arguments, "expanded": altered_expanded})
            invoke.assert_not_called()
            with patch.object(solver, "_invoke_rollout") as invoke, self.assertRaises(ValueError):
                solver.evaluate_candidates(candidates=[candidate], ordering_policy={"version": "legacy", "node_count": 4}, **arguments)
            invoke.assert_not_called()
            sidecar = root / candidate["ordering_derivation"]["path"]
            sidecar.unlink()
            with patch.object(solver, "_invoke_rollout") as invoke, self.assertRaises(ValueError):
                solver.evaluate_candidates(candidates=[candidate], ordering_policy=policy, **arguments)
            invoke.assert_not_called()

    def test_legacy_solver_context_without_sidecar_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = {"candidate_id": "legacy", "question_type": "ordering", "question": "Order events",
                   "items": {f"I{x}": f"Event {x}" for x in range(1, 5)}, "gold": ["I2", "I1", "I4", "I3"]}
            with patch.object(solver, "_invoke_rollout", return_value={"scored": True, "correct": True}) as invoke:
                result = solver.evaluate_candidates(candidates=[row], expanded={"turns": []}, root_id="legacy-root",
                    run_root=Path(tmp), ledger=None, tokenizer=TextTokenizer(), backend=None,
                    ordering_policy={"version": "legacy", "node_count": 4})
                self.assertEqual(24, invoke.call_count)
                self.assertTrue(result[0]["gold_rebinding_audit"]["pass"])


if __name__ == "__main__":
    unittest.main()
