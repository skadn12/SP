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
import qa_transport


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
        self._run_full_mock_pipeline()

    def test_actual_rejected_generation_finishes_empty_without_solver(self):
        self._run_full_mock_pipeline(reject_questions=True)

    def _run_full_mock_pipeline(self, reject_questions=False):
        """Real stage code with transport and model execution boundaries mocked."""
        adapter = pipeline.generation
        root_id = "cc-train-episode-104814"
        refs = ["P:T001", "P:T002", "R02:A:S01:T001", "R02:A:S02:T001", "CANDIDATE:B:S01:T001", "CANDIDATE:B:S02:T001"]
        source = {"root_id": root_id, "new_admitted_scenes": 0, "retained_candidate_scenes": 2,
                  "training_pool_eligible": False, "formal_baseline_turns": 4,
                  "history_policy": "formal_prefix_then_provisional_retained_dialogue", "turns": [{"turn_ref": ref, "speaker": "Alex",
                   "text": "SYNTHETIC fixture event " + word + "."}
                  for ref, word in zip(refs, ("one", "two", "three", "four", "five", "six"))]}

        for turn in source["turns"][4:]:
            turn.update(origin="provisional_candidate", training_pool_eligible=False)

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
                        "coherent": not reject_questions, "unique_order": True, "alternative_orders": [],
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
                    if reject_questions and proof["question_type"] == "multi_select":
                        # Existing text contract rejects a missing proposition;
                        # raw Qwen evidence and relations remain unmodified.
                        q["options"][0]["proposition"] = ""
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
            probe = pipeline.evaluator.load_probe("v45_offline_pipeline_probe")
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

        with tempfile.TemporaryDirectory(prefix="v45-qa-SYNTHETIC-") as directory, \
             patch("socket.socket", side_effect=AssertionError("No network in integration")), \
             contextlib.redirect_stdout(io.StringIO()):
            temp = Path(directory)
            expansion, generated, evaluated = temp / "expansion", temp / "generation", temp / "evaluation"
            source_path = expansion / "CANDIDATE_HISTORY.json"
            adapter.write(source_path, source)
            adapter.write(expansion / "FORMAL_BASELINE.json", {"turns": source["turns"][:4]})
            adapter.write(expansion / "CANDIDATE_RECORDS.json", [
                {"candidate_id": "candidate-1", "disposition": "retained_pending_review", "realization": {"turns": source["turns"][4:5]}},
                {"candidate_id": "candidate-2", "disposition": "retained_pending_review", "realization": {"turns": source["turns"][5:]}}])
            adapter.write(expansion / "COLLECTION_BINDINGS.json", {"files": adapter.base.bindings([
                source_path, expansion / "FORMAL_BASELINE.json", expansion / "CANDIDATE_RECORDS.json"])})
            adapter.write(expansion / "MANIFEST.json", {"version": adapter.EXPANSION_VERSION,
                "scope": "fresh_collection", "root_id": root_id, "model": adapter.MODEL,
                "training_pool_eligible": False})
            audit = {"status": "ARTIFACT_REPLAY_VERIFIED", "semantic_certified": False,
                     "training_pool_eligible": False, "synthetic_test_only": True}
            (temp / "prior.jsonl").write_text("")
            stack = contextlib.ExitStack()
            self.addCleanup(stack.close)
            stack.enter_context(patch.object(adapter.collection_source, "replay_audit", return_value=audit))
            with patch.object(adapter, "_dependency_paths", return_value=({}, {Path(adapter.__file__)})), \
                 patch.object(adapter, "DEFAULT_PRIOR", temp / "prior.jsonl"), \
                 patch.object(adapter.v15, "prior_candidates", return_value=([], [])):
                adapter.prepare(generated, source_path, slots=1)
            source = adapter.read(generated / "sources" / (root_id + ".json"))
            with patch.object(qa_transport, "Caller", MockCaller), \
                 patch.object(qa_transport, "shared_lock", contextlib.nullcontext):
                report = adapter.generate(generated)
                self.assertEqual(report["ready_for_solver"], 0 if reject_questions else 2)
            self.assertEqual(len(pipeline.verify_generation(generated)[1]), 0 if reject_questions else 2)
            raw_responses = {path: path.read_bytes() for path in (generated / "requests").glob("*/RESPONSE.json")}
            if reject_questions:
                by_type = {row["question_type"]: row for row in report["roots"]}
                self.assertTrue(by_type["multi_select"]["mechanical_rejected"])
                self.assertEqual(len(by_type["ordering"]["review_rejected"]), 1)
                mechanical = pipeline.read(generated / "MECHANICAL_POOL.json")["questions"]
                self.assertEqual(len(mechanical), 1)
                self.assertEqual(mechanical[0]["question_type"], "ordering")
                self.assertEqual(mechanical[0]["original_gold"], [f"I{i}" for i in range(6, 0, -1)])
                with patch.object(pipeline.evaluator, "preflight", side_effect=AssertionError("No rejected Solver planning")), \
                     patch.object(pipeline.evaluator, "live", side_effect=AssertionError("No rejected Solver inference")):
                    self.assertEqual(pipeline.preflight(generated, evaluated)["full_calls_planned"], 0)
                    self.assertEqual(pipeline.live(evaluated)["status"], "EMPTY_POOL_NO_SOLVER")
                    result = pipeline.finalize(evaluated)
                self.assertEqual(result["new_unique_added"], 0)
                self.assertEqual(result["full_solver_calls"], 0)
                self.assertFalse(result["training_allowed"])
                self.assertEqual((evaluated / "training_pool/TRAINING_POOL.jsonl").read_text(), "")
                self.assertTrue(all(path.read_bytes() == raw for path, raw in raw_responses.items()))
                return
            ordering_question = next(q for q in pipeline.read(generated / "SOLVER_POOL.json")["questions"]
                                     if q["question_type"] == "ordering")
            review_request = Path(ordering_question["qwen_review"]["request_path"])
            original_request = review_request.read_bytes()
            tampered_request = pipeline.read(review_request)
            tampered_request["messages"][0]["content"] = tampered_request["messages"][0]["content"].replace(
                source["turns"][-1]["text"], "SYNTHETIC removed final event")
            review_request.write_text(json.dumps(tampered_request))
            with self.assertRaisesRegex(ValueError, "actual_ordering_review_context_or_contract_changed"):
                pipeline.verify_generation(generated)
            review_request.write_bytes(original_request)
            with patch.object(pipeline.evaluator, "preflight", side_effect=mock_model_preflight):
                self.assertEqual(pipeline.preflight(generated, evaluated)["full_calls_planned"], 30)
            with patch.object(pipeline.assembler.Inspector, "model", mock_model_check), \
                 patch.object(pipeline.evaluator, "live", side_effect=mock_solver_live):
                self.assertEqual(pipeline.live(evaluated)["actual_call_count"], 30)
                result = pipeline.finalize(evaluated)
            self.assertEqual(result["new_unique_added"], 2)
            self.assertEqual(result["training_admitted_questions"], 0)
            self.assertEqual(result["upstream_admitted_scenes"], 0)
            self.assertFalse(result["training_allowed"])
            self.assertEqual(result["qa_selected_unique_candidates"], 2)
            self.assertEqual(result["ground_calls"], 0)
            self.assertEqual(result["training_calls"], 0)
            rows = [json.loads(line) for line in (evaluated / "training_pool/TRAINING_POOL.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["full_context"] == source for row in rows))
            self.assertTrue(all(row["training_allowed"] is False and row["admission_required"]
                                and row["upstream_admitted_scenes"] == 0 for row in rows))
            ordering = next(row for row in rows if row["question_type"] == "ordering")
            self.assertEqual(ordering["gold"], [f"I{i}" for i in range(1, 7)])
            self.assertEqual(ordering["original_selected_question"]["original_gold"], [f"I{i}" for i in range(6, 0, -1)])
            multiselect = next(row for row in rows if row["question_type"] == "multi_select")
            self.assertEqual(multiselect["gold"], list("ABC"))
            self.assertEqual([multiselect["original_selected_question"]["option_evidence"][label]["relation"]
                              for label in "ABCDEF"], ["supports"] * 3 + ["contradicts"] * 3)
            self.assertTrue(all(path.read_bytes() == raw for path, raw in raw_responses.items()))
            before = (evaluated / "training_pool/TRAINING_POOL.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "already_exists"):
                pipeline.finalize(evaluated)
            self.assertEqual((evaluated / "training_pool/TRAINING_POOL.jsonl").read_bytes(), before)
            with (evaluated / "training_pool/TRAINING_POOL.jsonl").open("a") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "candidate_pool_artifact_changed"):
                pipeline.verify(evaluated)

    def test_empty_generation_finishes_with_zero_provisional_yield_without_solver(self):
        with tempfile.TemporaryDirectory(prefix="v45-qa-empty-SYNTHETIC-") as directory:
            temp = Path(directory)
            generation, run = temp / "generation", temp / "evaluation"
            pipeline.write(generation / "MANIFEST.json", {"synthetic_test_only": True})
            inspector = pipeline.assembler.Inspector()
            inspector.bind(generation / "MANIFEST.json")
            verified = ({"roots": ["fixture"]}, [], {}, inspector)
            with patch.object(pipeline, "verify_generation", return_value=verified), \
                 patch.object(pipeline.evaluator, "preflight", side_effect=AssertionError("No empty Solver preflight")), \
                 patch.object(pipeline.evaluator, "live", side_effect=AssertionError("No empty Solver load")):
                self.assertEqual(pipeline.preflight(generation, run)["full_calls_planned"], 0)
                self.assertEqual(pipeline.live(run)["status"], "EMPTY_POOL_NO_SOLVER")
                report = pipeline.finalize(run)
                self.assertEqual(report["qa_selected_unique_candidates"], 0)
                self.assertFalse(report["training_allowed"])
                self.assertFalse((run / "delivery").exists())
                self.assertEqual((run / "training_pool/TRAINING_POOL.jsonl").read_text(), "")
                manifest = pipeline.read(run / "MANIFEST.json")
                manifest["training_allowed"] = True
                (run / "MANIFEST.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "provisional_candidate_scope_changed"):
                    pipeline.verify(run)

    def test_accepted_upstream_status_cannot_be_substituted_for_pending(self):
        for row in ({"qa_status": "READY_CHECKED_MERGED_ROOT", "new_admitted_scenes": 0},
                    {"qa_status": "READY_PROVISIONAL_CANDIDATE_HISTORY", "new_admitted_scenes": 2}):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "source_not_explicitly_provisional"):
                pipeline._verify_source_scope({"source_rows": [row]})

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
