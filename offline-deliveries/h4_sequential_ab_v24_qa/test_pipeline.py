"""Offline selection/provenance boundaries; never load a Solver or call APIs."""
from copy import deepcopy
from fractions import Fraction
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_and_pool as pipeline


def question(kind="multi_select"):
    q = {"candidate_id": "fixture-" + kind, "root_id": "fixture", "question_type": kind,
         "question": "Which statements are supported?" if kind == "multi_select" else "Order these mentions.",
         "generator_model": pipeline.generation.MODEL, "raw_question_content_preserved": True,
         "source_path": "/fixture/full.json", "source_sha256": "a" * 64}
    if kind == "multi_select":
        q.update(options={label: "Claim " + label for label in "ABCDEF"}, gold=list("ABC"),
                 option_evidence={label: {"relation": "supports" if label in "ABC" else "contradicts"}
                                  for label in "ABCDEF"})
    else:
        q.update(items={f"I{i}": f"Event {i}" for i in range(1, 7)}, gold=[f"I{i}" for i in range(1, 7)],
                 original_gold=[f"I{i}" for i in range(6, 0, -1)])
    return q


def solver_fixture(questions):
    manifest = {"policy_version": pipeline.v15.POLICY, "solver_model": pipeline.assembler.TARGET,
        "model": str(pipeline.assembler.CHECKPOINT), "execution": "full_only_no_ground_v1",
        "training_started": False, "training_promotion": False, "requests": [],
        "decoder_contract": {"temperature": .8, "top_p": .9, "max_tokens": 32,
                             "structured_output_api": "finite", "structured_output_version": "finite_choice_set_v1"}}
    records, results = [], []
    for q in questions:
        variants, _ = pipeline.evaluator._variants_for(q)
        kind = q["question_type"]
        slots = 3 if kind == "multi_select" else 1
        score_map = {}
        for variant in variants:
            values = q.get("options", q.get("items"))
            gold_text = [values[label] for label in q["gold"]]
            displayed = variant["options" if kind == "multi_select" else "items"]
            if kind == "multi_select":
                prediction = [gold_text[0]]
            else:
                prediction = [gold_text[1], gold_text[0], *gold_text[2:]]
            answer = "<box>" + ",".join(next(label for label, text in displayed.items() if text == wanted)
                                            for wanted in prediction) + "</box>"
            value, exact = pipeline.assembler.score(q, variant, answer)
            score_map[variant["variant_id"]] = [value] * slots
            for slot in range(slots):
                request = {"request_id": f"{q['candidate_id']}-{variant['variant_id']}-{slot}",
                    "candidate_id": q["candidate_id"], "root_id": q["root_id"], "question_type": kind,
                    "variant": variant, "mode": "full_context", "slot": slot,
                    "seed": pipeline.evaluator.SEED + len(records)}
                manifest["requests"].append(request)
                records.append({**{k: request[k] for k in ("request_id", "candidate_id", "root_id", "question_type", "mode", "slot", "seed")},
                    "variant_id": variant["variant_id"], "raw_output": answer,
                    "metadata": {**manifest["decoder_contract"], "finish_reason": "stop"},
                    "evaluation_valid": True, "data_errors": [], "solver_model": pipeline.assembler.TARGET,
                    "model_path": str(pipeline.assembler.CHECKPOINT),
                    "score": {"scored": True, "partial_score_fraction": str(value), "exact_match": exact}})
        result = pipeline.evaluator.select(score_map, list(score_map), exact_rate=Fraction(0), data_errors=[])
        result.update(candidate_id=q["candidate_id"], root_id=q["root_id"], question_type=kind,
                      expected_full_calls=len(variants) * slots, actual_full_calls=len(variants) * slots)
        results.append(result)
    return manifest, records, {"questions": results, "selected_questions": [q["candidate_id"] for q in questions]}


class PipelineTests(unittest.TestCase):
    def test_full_mock_generation_evaluation_and_pool_integration(self):
        """Real stage code with transport and model execution boundaries mocked."""
        adapter = pipeline.generation
        root_id = "cc-train-episode-104814"
        refs = ["P:T001", "P:T002", "R02:A:S01:T001", "R02:A:S02:T001", "R02:B:S01:T001", "R02:B:S02:T001"]
        source = {"root_id": root_id, "turns": [{"turn_ref": ref, "speaker": "Alex",
                   "text": "SYNTHETIC fixture event " + word + "."}
                  for ref, word in zip(refs, ("one", "two", "three", "four", "five", "six"))]}

        class MockCaller:
            def __init__(self, run):
                self.run, self.counter = run, 0

            def call(self, label, payload, proof):
                self.counter += 1
                directory = self.run / "requests" / f"{self.counter:03d}-{label}"
                adapter.write(directory / "REQUEST.json", payload)
                adapter.write(directory / "PROOF.json", {**proof, "payload_sha256": adapter.base.digest(payload)})
                adapter.write(directory / "HTTP_ATTEMPT.json", {"synthetic_test_only": True,
                    "actual_payload_equals_frozen": True, "payload_sha256": adapter.base.digest(payload)})
                if proof["role"] == "ordering_reviewer":
                    context = pipeline._input_json(payload, "\nDATA:\n")
                    q = context["candidates"][0]
                    items = {}
                    for key, text in q["items"].items():
                        turn = next(t for t in source["turns"] if t["text"] == text)
                        items[key] = {"supported": True, "order_anchor": {"turn_ref": turn["turn_ref"],
                            "quote": text, "reason": "SYNTHETIC exact fixture."}, "support_refs": [],
                            "alternative_matching_anchors": [], "actor_ok": True, "modality_ok": True,
                            "state_ok": True, "neutral_stem_ok": True, "reason": "SYNTHETIC support."}
                    value = {"reviews": [{"candidate_id": q["candidate_id"], "items": items,
                        "coherent": True, "unique_order": True, "alternative_orders": [],
                        "neutral_stem_ok": True, "reason": "SYNTHETIC review."}]}
                else:
                    context = pipeline._input_json(payload, "\nINPUT_JSON\n")
                    qid = context["candidate_ids"][0]
                    if proof["question_type"] == "multi_select":
                        q = {"candidate_id": qid, "topic": "SYNTHETIC test", "question": "Fixture? Select all that apply.",
                             "options": [{"proposition": f"Fixture multiselect claim {i}.",
                                          "relation": "supports" if i < 3 else "contradicts",
                                          "turn_ref": t["turn_ref"], "speaker": t["speaker"],
                                          "quote": t["text"], "rationale": "SYNTHETIC relation."}
                                         for i, t in enumerate(source["turns"])]}
                    else:
                        q = {"candidate_id": qid, "root_id": root_id, "question_type": "ordering",
                             "chronology_kind": "conversation_order",
                             "question": "Order statements as they APPEAR IN THE SUPPLIED DIALOGUE.",
                             "items": {f"I{i}": t["text"] for i, t in enumerate(source["turns"], 1)},
                             "gold": [f"I{i}" for i in range(6, 0, -1)],
                             "item_evidence": {f"I{i}": {"turn_ref": t["turn_ref"], "speaker": t["speaker"],
                                 "quote": t["text"], "status": "statement", "reason": "SYNTHETIC quote."}
                                 for i, t in enumerate(source["turns"], 1)}}
                    value = {"root_id": root_id, "questions": [q], "abstentions": []}
                raw = json.dumps(value)
                metadata = {"requested_model": adapter.MODEL, "returned_model": adapter.MODEL,
                            "finish_reason": "stop", "http_status": 200, "synthetic_test_only": True}
                adapter.write(directory / "RESPONSE.json", {"text": raw, "metadata": metadata})
                return raw, directory, metadata

        real_preflight = pipeline.evaluator.preflight

        def mock_model_preflight(pool_path, run, **options):
            self.assertTrue(options["require_model"])
            self.assertEqual(options["solver_model"], pipeline.assembler.TARGET)
            # Execute the existing real planning path, but explicitly replace
            # only model-loading verification in this temporary synthetic run.
            probe = pipeline.evaluator.load_probe("v24_offline_pipeline_probe")
            decoder = {"temperature": .8, "top_p": .9, "max_tokens": 32, "enable_thinking": False,
                       "structured_output_api": "SYNTHETIC", "structured_output_version": "finite_choice_set_v1"}
            with patch.object(pipeline.evaluator, "load_probe", return_value=probe), \
                 patch.object(probe.solver, "generation_config", return_value=decoder):
                real_preflight(pool_path, run, require_ordering_review=True,
                               require_model=False, solver_model=pipeline.assembler.TARGET)
            identity = pipeline.read(run / "solver_identity.json")
            identity.update(synthetic_test_only=True)
            (run / "solver_identity.json").write_text(json.dumps(identity))
            manifest = pipeline.read(run / "manifest.json")
            manifest["solver_identity"] = identity
            (run / "manifest.json").write_text(json.dumps(manifest))

        def mock_model_check(inspector, identity):
            self.assertTrue(identity.get("synthetic_test_only"))
            self.assertEqual(identity["model_identifier"], pipeline.assembler.TARGET)

        def mock_solver_live(run):
            manifest = pipeline.read(run / "manifest.json")
            gold = pipeline.read(run / "gold.json")
            qs = {q["candidate_id"]: q for q in pipeline.read(run / "questions.json")["candidates"]}
            for request in manifest["requests"]:
                qid = request["candidate_id"]
                answer = gold[qid][:1] if request["question_type"] == "multi_select" else [gold[qid][1], gold[qid][0], *gold[qid][2:]]
                mapping = {content: label for label, content in request["variant"]["display_to_content"].items()}
                raw = "<box>" + ",".join(mapping[cid] for cid in answer) + "</box>"
                scored = pipeline.evaluator.score_response({**qs[qid], "gold_content_ids": gold[qid]}, request, raw)
                adapter.write(run / "responses" / (request["request_id"] + ".json"), {
                    **{key: request[key] for key in ("request_id", "candidate_id", "root_id", "question_type", "mode", "slot", "seed")},
                    "variant_id": request["variant"]["variant_id"], "raw_output": raw,
                    "solver_model": pipeline.assembler.TARGET, "model_path": str(pipeline.assembler.CHECKPOINT),
                    "metadata": {**manifest["decoder_contract"], "finish_reason": "stop", "synthetic_test_only": True},
                    "score": scored, "evaluation_valid": True, "data_errors": []})
            adapter.write(run / "COMPLETE.json", {"status": "COMPLETE", "actual_call_count": len(manifest["requests"]),
                "training_started": False, "training_promotion": False, "synthetic_test_only": True})

        with tempfile.TemporaryDirectory(prefix="v24-qa-SYNTHETIC-") as directory, \
             patch("socket.socket", side_effect=AssertionError("No network in integration")), \
             contextlib.redirect_stdout(io.StringIO()):
            temp = Path(directory)
            expansion, generated, evaluated = temp / "expansion", temp / "generation", temp / "evaluation"
            source_path = expansion / "sources_merged" / (root_id + ".json")
            adapter.write(source_path, source)
            merge_path = expansion / "MERGE_MANIFEST.json"
            adapter.write(merge_path, {"version": adapter.EXPANSION_VERSION, "status": "PASS",
                "sequence": "frozen_base -> accepted_A -> accepted_B", "files": adapter.base.bindings([source_path]),
                "source_rows": [{"root_id": root_id, "path": str(source_path),
                    "sha256": adapter.base.digest(source_path.read_bytes()), "accepted_scenes": 4,
                    "accepted_A_scenes": 2, "accepted_B_scenes": 2, "status": "PASS"}]})
            adapter.write(expansion / "FINAL_AUDIT.json", {"status": "PASS", "failures": [], "run": str(expansion),
                "run_status": "COMPLETE_WITH_SHORTFALL", "roots": [root_id], "model": adapter.MODEL, "synthetic_test_only": True})
            adapter.write(expansion / "MANIFEST.json", {"version": adapter.EXPANSION_VERSION,
                "roots": [root_id], "target_solver_model": adapter.TARGET, "files": []})
            (temp / "prior.jsonl").write_text("")
            with patch.object(adapter, "_dependency_paths", return_value=({}, {Path(adapter.__file__)})), \
                 patch.object(adapter, "DEFAULT_PRIOR", temp / "prior.jsonl"), \
                 patch.object(adapter.v15, "prior_candidates", return_value=([], [])):
                adapter.prepare(generated, merge_path, slots=1)
            with patch.object(adapter.legacy, "Caller", MockCaller), patch.object(adapter, "INTERVAL", 0):
                self.assertEqual(adapter.generate(generated)["ready_for_solver"], 2)
            self.assertEqual(len(pipeline.verify_generation(generated)[1]), 2)
            with patch.object(pipeline.evaluator, "preflight", side_effect=mock_model_preflight):
                self.assertEqual(pipeline.preflight(generated, evaluated)["full_calls_planned"], 30)
            with patch.object(pipeline.assembler.Inspector, "model", mock_model_check), \
                 patch.object(pipeline.evaluator, "live", side_effect=mock_solver_live):
                self.assertEqual(pipeline.live(evaluated)["actual_call_count"], 30)
                result = pipeline.finalize(evaluated)
            self.assertEqual(result["new_unique_added"], 2)
            self.assertEqual(result["ground_calls"], 0)
            self.assertEqual(result["training_calls"], 0)
            rows = [json.loads(line) for line in (evaluated / "training_pool/TRAINING_POOL.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["full_context"] == source for row in rows))
            ordering = next(row for row in rows if row["question_type"] == "ordering")
            self.assertEqual(ordering["gold"], [f"I{i}" for i in range(1, 7)])
            self.assertEqual(ordering["original_selected_question"]["original_gold"], [f"I{i}" for i in range(6, 0, -1)])
            before = (evaluated / "training_pool/TRAINING_POOL.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "already_exists"):
                pipeline.finalize(evaluated)
            self.assertEqual((evaluated / "training_pool/TRAINING_POOL.jsonl").read_bytes(), before)

    def test_actual_v15_policy_replay_preserves_both_gold_contracts(self):
        qs = [question(), question("ordering")]
        manifest, records, results = solver_fixture(qs)
        before = deepcopy(qs)
        selected = pipeline.verify_scores(qs, manifest, records, results)
        self.assertEqual(len(records), 30)
        self.assertEqual([q["gold"] for q in selected], [q["gold"] for q in qs])
        self.assertEqual(selected[1]["original_gold"], qs[1]["original_gold"])
        self.assertEqual(selected[0]["training_selection"]["full_partial_mean"], "1/3")
        self.assertEqual(selected[1]["training_selection"]["full_partial_mean"], "13/15")
        self.assertEqual(qs, before)

    def test_missing_answers_score_changes_and_base_model_never_enter_pool(self):
        qs = [question()]
        for mutation in ("missing", "score", "base_model", "ground", "permutation"):
            manifest, records, results = solver_fixture(qs)
            if mutation == "missing":
                records.pop()
            elif mutation == "score":
                records[0]["score"]["partial_score_fraction"] = "1"
            elif mutation == "base_model":
                manifest["model"] = str(pipeline.evaluator.TOKENIZER)
            elif mutation == "ground":
                records[0]["mode"] = manifest["requests"][0]["mode"] = "ground"
            else:
                manifest["requests"][0]["variant"] = deepcopy(manifest["requests"][0]["variant"])
                manifest["requests"][0]["variant"]["options"]["A"] = "Changed statement"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                pipeline.verify_scores(qs, manifest, records, results)

    def test_complete_score_boundaries_are_not_relaxed_for_yield(self):
        for mean in (Fraction(0), Fraction(1), Fraction(-1, 3)):
            result = pipeline.evaluator.select({f"perm_{i:02d}": [mean] * 3 for i in range(6)},
                                               [f"perm_{i:02d}" for i in range(6)])
            self.assertFalse(result["selected_by_policy"])
            self.assertFalse(result["ground_used"])

    def test_pool_rows_preserve_full_A_B_history_and_original_gold_dedup_exact_only(self):
        qs = [question(), question("ordering")]
        manifest, records, results = solver_fixture(qs)
        selected = pipeline.verify_scores(qs, manifest, records, results)
        source = {"root_id": "fixture", "turns": [
            {"turn_ref": ref, "speaker": "A", "text": ref} for ref in
            ("P:T001", "R02:A:S01:T001", "R02:B:S01:T001")]}
        entries = [{"question": q, "source": source, "is_new": True, "provenance": {"candidate": q["candidate_id"]}}
                   for q in selected]
        duplicate = deepcopy(entries[0])
        duplicate["question"]["candidate_id"] = "same-set-new-id"
        entries.append(duplicate)
        paraphrase = deepcopy(entries[0])
        paraphrase["question"]["candidate_id"] = "paraphrase-allowed"
        paraphrase["question"]["options"]["A"] = "A supported paraphrase."
        entries.append(paraphrase)
        before = deepcopy(entries)
        rows, duplicates = pipeline.pool_rows(entries, Path("/new/output"))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(rows[1]["original_selected_question"]["original_gold"], qs[1]["original_gold"])
        self.assertTrue(all(row["full_context"] == source and row["canonical_exposure_units"] == 1 for row in rows))
        self.assertTrue(all(row["gold"] == row["original_selected_question"]["gold"] for row in rows))
        self.assertEqual(entries, before)

    def test_failed_generation_audit_blocks_before_solver_preflight_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(pipeline, "verify_generation", side_effect=ValueError("source_audit_failed")), \
                 patch.object(pipeline.evaluator, "preflight", side_effect=AssertionError("Must not plan Solver")):
                with self.assertRaisesRegex(ValueError, "source_audit_failed"):
                    pipeline.preflight(root / "generation", root / "output")
            self.assertFalse((root / "output").exists())

    def test_skipped_or_base_model_identity_cannot_start_live(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            pipeline.write(run / "solver/manifest.json", {
                "policy_version": pipeline.v15.POLICY, "solver_model": pipeline.assembler.TARGET,
                "model": str(pipeline.assembler.CHECKPOINT), "execution": "full_only_no_ground_v1",
                "training_started": False, "training_promotion": False, "requests": [],
                "solver_identity": {"skipped": True}})
            with patch.object(pipeline, "verify", return_value={"candidate_count": 1}), \
                 patch.object(pipeline.evaluator, "live", side_effect=AssertionError("Must not load Solver")):
                with self.assertRaisesRegex(ValueError, "exact_current_m_clean_checkpoint_required"):
                    pipeline.live(run)


if __name__ == "__main__":
    unittest.main()
