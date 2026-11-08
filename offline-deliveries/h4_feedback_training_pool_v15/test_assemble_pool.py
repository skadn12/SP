"""Targeted checks for provenance-bound pool assembly; no model calls."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v15_pool_tests", HERE / "assemble_pool.py")
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def question(qid="root-q", source="a" * 64):
    return {"candidate_id": qid, "root_id": "root", "question_type": "multi_select",
        "question": "Which statements are supported?", "options": {c: "Claim " + c for c in "ABCDEF"},
        "gold": list("ABC"), "option_evidence": {c: {"relation": "supports" if c in "ABC" else "contradicts"} for c in "ABCDEF"},
        "generator_model": pool.MODEL, "raw_question_content_preserved": True,
        "training_target": "complete_canonical_gold", "loss_weight_from_partial_score": False,
        "source_sha256": source}


def selection_fixture():
    q = question()
    manifest = {"requests": [], "decoder_contract": {"temperature": .8, "top_p": .9, "max_tokens": 32,
                "structured_output_api": "finite", "structured_output_version": "finite_choice_set_v1"}}
    records = []
    for variant in range(6):
        content = list(q["options"].values())
        content = content[variant:] + content[:variant]
        options = dict(zip("ABCDEF", content))
        mapping = {label: pool.content_id(text) for label, text in options.items()}
        display = {"variant_id": f"perm_{variant:02d}", "options": options, "display_to_content": mapping}
        answer = next(label for label, text in options.items() if text == "Claim A")
        for slot in range(3):
            rid = f"request-{variant}-{slot}"
            request = {"request_id": rid, "candidate_id": q["candidate_id"], "root_id": q["root_id"],
                "question_type": "multi_select", "variant": display, "slot": slot, "seed": len(records), "mode": "full_context"}
            manifest["requests"].append(request)
            records.append({**{key: request[key] for key in ("request_id", "candidate_id", "root_id", "question_type", "slot", "seed", "mode")},
                "variant_id": display["variant_id"], "raw_output": "<box>" + answer + "</box>",
                "metadata": {**manifest["decoder_contract"], "finish_reason": "stop"},
                "solver_model": pool.TARGET, "model_path": str(pool.CHECKPOINT), "evaluation_valid": True, "data_errors": [],
                "score": {"scored": True, "partial_score_fraction": "1/3", "exact_match": False}})
    result = {"selected_by_policy": True, "ground_used": False, "data_errors": [], "full_partial_mean": "1/3",
        "permutation_means": {f"perm_{i:02d}": "1/3" for i in range(6)}, "full_exact_rate": "0", "actual_full_calls": 18, "expected_full_calls": 18}
    q["training_selection"] = deepcopy(result)
    return q, manifest, records, result


class AssemblyTests(unittest.TestCase):
    def test_complete_full_selection_checks_every_answer(self):
        q, manifest, records, result = selection_fixture()
        self.assertEqual(pool.verify_selection(q, manifest, records, result)["full_calls"], 18)
        records.pop()
        with self.assertRaisesRegex(ValueError, "missing_duplicate_or_extra"):
            pool.verify_selection(q, manifest, records, result)

    def test_partial_negative_scores_are_not_zero_clipped(self):
        q, manifest, records, result = selection_fixture()
        records[0]["raw_output"] = "<box>D</box>"
        records[0]["score"]["partial_score_fraction"] = "-1/3"
        result["permutation_means"]["perm_00"] = "1/9"
        result["full_partial_mean"] = "8/27"
        q["training_selection"] = deepcopy(result)
        self.assertEqual(pool.verify_selection(q, manifest, records, result)["full_partial_mean"], "8/27")

    def test_wrong_canonical_mapping_or_checkpoint_rejects(self):
        q, manifest, records, result = selection_fixture()
        records[0]["model_path"] = "/model/base"
        with self.assertRaisesRegex(ValueError, "checkpoint_mismatch"):
            pool.verify_selection(q, manifest, records, result)
        records[0]["model_path"] = str(pool.CHECKPOINT)
        manifest["requests"][0]["variant"]["display_to_content"]["A"] = "wrong"
        with self.assertRaisesRegex(ValueError, "display_content_mapping_invalid"):
            pool.verify_selection(q, manifest, records, result)

    def test_empty_gold_and_diagnostic_pools_reject(self):
        q = question()
        q["gold"] = []
        with self.assertRaisesRegex(ValueError, "gold_cardinality"):
            pool.question_contract(q)
        q = question()
        q["diagnostic_only"] = True
        with self.assertRaisesRegex(ValueError, "diagnostic_candidate"):
            pool.question_contract(q)

    def test_all_six_gold_supported_and_loss_not_derived_from_score(self):
        q = question()
        q["gold"] = list("ABCDEF")
        for value in q["option_evidence"].values():
            value["relation"] = "supports"
        pool.question_contract(q)
        variant = {"display_to_content": {label: pool.content_id(text) for label, text in q["options"].items()}}
        self.assertEqual(pool.score(q, variant, "<box>A</box>")[0], Fraction(1, 6))
        q["loss_weight_from_partial_score"] = True
        with self.assertRaisesRegex(ValueError, "complete_gold_training_target"):
            pool.question_contract(q)

    def test_reversed_ordering_keeps_negative_score(self):
        q = question()
        q.update(question_type="ordering", items={"I" + str(i): "Item " + str(i) for i in range(1, 7)},
                 gold=["I" + str(i) for i in range(1, 7)])
        variant = {"display_to_content": {label: pool.content_id(text) for label, text in q["items"].items()}}
        self.assertEqual(pool.score(q, variant, "<box>I6,I5,I4,I3,I2,I1</box>")[0], Fraction(-1))
        q["gold"].pop()
        with self.assertRaisesRegex(ValueError, "ordering_gold_not_complete"):
            pool.question_contract(q)

    def test_canonical_identity_ignores_display_and_context_but_not_root(self):
        q = question()
        other = question("new-id", "b" * 64)
        other["options"] = dict(zip("ABCDEF", reversed(list(other["options"].values()))))
        other["options"]["F"] = "  CLAIM   A "
        self.assertEqual(pool.canonical_question_id(q), pool.canonical_question_id(other))
        other["root_id"] = "different-root"
        self.assertNotEqual(pool.canonical_question_id(q), pool.canonical_question_id(other))

    def test_existing_first_dedup_retains_context_and_gold_conflict_diagnostic(self):
        a, b = question("old", "a" * 64), question("new", "b" * 64)
        b["gold"] = list("ABD")
        entries = [{"question": q, "source": {"turns": [origin]}, "is_new": is_new,
                    "provenance": {"candidate_id": q["candidate_id"]}}
                   for q, origin, is_new in [(a, "old context", False), (b, "new context", True)]]
        original = deepcopy(entries)
        kept, duplicates = pool.deduplicate(entries)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["question"], a)
        self.assertEqual(len(kept[0]["provenance_chain"]), 2)
        self.assertTrue(duplicates[0]["gold_content_disagreement"])
        self.assertFalse(duplicates[0]["same_context"])
        self.assertEqual(entries, original)

    def test_formal_output_outside_version_directory_rejects(self):
        with self.assertRaisesRegex(ValueError, "output_must_be_a_new_v15"):
            pool.assemble([], [], Path("/tmp/formal-training-output"))

    def test_archived_baseline_six_integrates_without_inference(self):
        old = HERE.parent / "h4_r02_yield_pilot_v12/runs/contract_smoke_2roots_01_qa/SELECTED_QUESTIONS.json"
        current = HERE.parent / "h4_r02_question_contract_v14/runs/frozen_v12_2roots_01/SELECTED_QUESTIONS.json"
        with tempfile.TemporaryDirectory(prefix=".assembly-test-", dir=HERE) as folder:
            destination = Path(folder) / "pool"
            report = pool.assemble([old, current], [], destination)
            self.assertEqual((report["existing_unique"], report["new_unique_added"], report["total_unique"]), (6, 0, 6))
            self.assertEqual((report["selected_multi_select"], report["selected_ordering"]), (4, 2))
            rows = [json.loads(line) for line in (destination / "TRAINING_POOL.jsonl").read_text().splitlines()]
            self.assertEqual(len({row["canonical_question_id"] for row in rows}), 6)
            self.assertTrue(all(row["canonical_exposure_units"] == 1 for row in rows))
            for row in rows:
                self.assertEqual(row["full_context"], pool.read(row["source_path"]))
                self.assertEqual(row["gold"], row["original_selected_question"]["gold"])
                self.assertEqual(pool.sha(row["source_path"]), row["source_sha256"])
            with self.assertRaisesRegex(ValueError, "immutable_output_already_exists"):
                pool.assemble([old, current], [], destination)


if __name__ == "__main__":
    unittest.main()
