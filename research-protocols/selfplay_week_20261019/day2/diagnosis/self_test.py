"""Synthetic executable edge cases; no experimental weaknesses are created."""
import copy
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import controller as C
from tokenizers import Tokenizer


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return C.reference(path)


def modify(path, callback):
    value = json.loads(Path(path).read_text())
    callback(value)
    write(path, value)


def refresh(directory):
    """Rebind a synthetic DAG after a mutation, so rejection tests semantics."""
    for _ in range(40):
        changed = False
        for path in sorted(Path(directory).rglob("*.json")):
            value = json.loads(path.read_text())
            for ref in C.all_refs(value):
                target = Path(ref["path"])
                if target.is_file() and target.is_relative_to(directory):
                    digest = C.sha(target)
                    if digest != ref["sha256"]:
                        ref["sha256"] = digest
                        changed = True
            write(path, value)
        if not changed:
            return
    raise RuntimeError("synthetic fixture reference graph did not converge")


def fixture(directory, status="reproduced", diagnostic=False):
    directory = Path(directory)
    frozen = json.loads((C.DAY1 / "MODEL_BINDINGS.json").read_text())
    token_ref = next(f for f in frozen["models"]["S0_M_clean_merged"]["files"] if Path(f["path"]).name == "tokenizer.json")
    token_ref = {k: token_ref[k] for k in ["path", "sha256"]}
    tokenizer = Tokenizer.from_file(token_ref["path"])
    roots = json.loads((C.DAY1 / "TRAIN_ROOTS.json").read_text())["root_ids"][:5]
    initial = C.Controller(C.DAY1 / "WEAKNESS_MEMORY_B_v000.json")
    doc, _ = initial.legacy.fixture()
    models = [{"model_id": "S0", "canonical_model_id": "S0_M_clean_merged", "parent_model_id": None, "binding": C.reference(C.DAY1 / "MODEL_BINDINGS.json"), "training_record": None}]
    for mid, parent, start, end in [("B1", "S0", 5, 6), ("B2", "B1", 11, 12)]:
        weight = directory / (mid + "-synthetic.safetensors")
        weight.write_bytes(b"Synthetic fixture bytes; not a loadable model and not actual weights.")
        binding = write(directory / (mid + "-model.json"), {"record_type": "model_file_manifest", "model_id": mid, "canonical_model_id": mid + "-synthetic", "files": [C.reference(weight)], "not_real_weights": True})
        train = write(directory / (mid + "-train.json"), {"record_type": "training_lineage", "parent_model_id": parent, "child_model_id": mid, "parent_binding": models[-1]["binding"], "child_binding": binding, "training_question_ids": [], "training_family_keys": [], "started_at": f"2026-10-19T00:{start:02}:00Z", "completed_at": f"2026-10-19T00:{end:02}:00Z"})
        models.append({"model_id": mid, "canonical_model_id": mid + "-synthetic", "parent_model_id": parent, "binding": binding, "training_record": train})
    questions = {}
    for n, root in enumerate(roots):
        source = write(directory / f"q{n}-source.json", {"turns": [{"speaker": "Synthetic", "text": f"Synthetic source for q{n}; this is not experiment data."}]})
        public = write(directory / f"q{n}-public.json", {"question": f"Synthetic question q{n}", "options": {"A": f"Option zero q{n}", "B": f"Option one q{n}", "C": f"Option two q{n}", "D": f"Option three q{n}"}})
        mapping = write(directory / f"q{n}-mapping.json", {"display_to_content": dict(zip("ABCD", ["i0", "i1", "i2", "i3"])), "content_to_text": {f"i{i}": text for i, text in enumerate(json.loads(Path(public["path"]).read_text())["options"].values())}})
        provenance = {"split": "train", "namespace": "train.feedback", "root_id": root, "family_id": f"family{n}", "source_id": f"synthetic-source{n}", "source_sha256": source["sha256"]}
        gold = write(directory / f"q{n}-gold.json", {"record_type": "private_gold", "question_id": f"q{n}", "task_type": "multi_select", "gold_content_ids": ["i1"]})
        review = write(directory / f"q{n}-review.json", {"decision": "PASS", "synthetic": True, "not_experimental_evidence": True})
        gate = write(directory / f"q{n}-gate.json", {"record_type": "content_admission", "question_id": f"q{n}", "provenance": provenance, "decision": "PASS", "score_used": False, "completed_at": "2026-10-19T00:00:01Z", "public_question": public, "source_input": source, "gold_reference": gold, "review_record": review})
        questions[n] = {"source": source, "public": public, "mapping": mapping, "provenance": provenance, "gold": gold, "gate": gate}

    def run(n, name, model="S0", correct=False, minute=0, mode="direct"):
        q = questions[n]
        start, committed, read_at, finished = [f"2026-10-19T00:{minute:02}:{seconds:02}Z" for seconds in [10, 20, 25, 30]]
        public = json.loads(Path(q["public"]["path"]).read_text())
        text = "<box>B</box>" if correct else "<box>A</box>"
        if mode == "diagnostic": text = json.dumps({"evidence": [{"label": label, "status": "supported", "turn_ref": "", "quote": ""} for label in "ABCD"], "answer": ["B" if correct else "A"]})
        request = write(directory / (name + "-request.json"), {"record_type": "solver_request", "request_id": name, "model_id": model, "model_binding": next(m["binding"] for m in models if m["model_id"] == model), "question_id": f"q{n}", "provenance": q["provenance"], "task_type": "multi_select", "mode": mode, "max_output_tokens": 1024 if mode == "diagnostic" else 32, "public_question": q["public"], "source_input": q["source"], "display_mapping": q["mapping"], "prompt": "\n".join([json.loads(Path(q["source"]["path"]).read_text())["turns"][0]["text"], public["question"], *public["options"].values()]), "gold_visible": False, "author_evidence_visible": False, "started_at": start})
        if name == "hint":
            modify(request["path"], lambda x: x.update(prompt=x["prompt"] + "\nHint: inspect the cited synthetic source carefully."))
            request = C.reference(request["path"])
        response = write(directory / (name + "-response.json"), {"record_type": "solver_response", "request_id": name, "model_id": model, "question_id": f"q{n}", "text": text, "output_token_ids": tokenizer.encode(text, add_special_tokens=False).ids, "committed_at": committed, "format_valid": True, "full_evidence_visible": True, "prediction_content_ids": ["i1"] if correct else ["i0"]})
        raw = write(directory / (name + "-run.json"), {"record_type": "solver_run", "request": request, "response": response, "tokenizer": token_ref})
        observation = write(directory / (name + "-eval.json"), {"record_type": "gold_evaluation", "solver_run": raw, "model_id": model, "question_id": f"q{n}", "provenance": q["provenance"], "gold_first_read_at": read_at, "gold_read_scope": "this_scoring_run", "gold_read_scope_id": "synthetic-scoring-run", "completed_at": finished, "gold_reference": q["gold"], "content_gate": q["gate"], "exact_correct": correct})
        return raw, observation

    direct_observations, failure_bindings = [], []
    for n, case in enumerate(doc["failure_cases"]):
        q = questions[n]
        raw, observation = run(n, f"fail{n}")
        group = write(directory / f"fail{n}-group.json", {"record_type": "solver_run_group", "runs": [raw]})
        evaluated = write(directory / f"fail{n}-eval-group.json", {"record_type": "evaluation_group", "observations": [observation]})
        case.update(provenance=q["provenance"], question_id=f"q{n}", direct_answer_record=group, postgold_evaluation=evaluated)
        case["content_gate"]["record"] = q["gate"]
        case["repeat_aggregation"] = {"protocol": C.reference(C.DAY1 / "PROTOCOL_V1.json"), "valid_calls": 1, "content_failures": 1, "failed_display_mappings": 1, "total_display_mappings": 1}
        case["evidence_refs"] = [{"source_id": q["provenance"]["source_id"], "source_sha256": q["source"]["sha256"], "scene_id": "scene0", "turn_start": 0, "turn_end": 0, "speaker": "Synthetic", "quote": json.loads(Path(q["source"]["path"]).read_text())["turns"][0]["text"]}]
        failure_bindings.append({"failure_id": case["failure_id"], "direct_group": group})
        direct_observations.append(observation)
    probe = doc["probes"][0]
    probe.update(provenance=questions[0]["provenance"])
    controlled = write(directory / "probe-protocol.json", {"record_type": "controlled_probe_protocol", "probe_id": "probe0", "cause_ids": ["cause0"], "failure_ids": ["failure0"], "changed_factor": "evidence_hint", "expected_exact_direction": "increase", "frozen_at": "2026-10-19T00:00:00Z"})
    probe["frozen_protocol"] = controlled
    _, hint = run(0, "hint", correct=True, minute=1)
    baseline = write(directory / "probe-before.json", {"record_type": "evaluation_group", "observations": [direct_observations[0]]})
    intervention = write(directory / "probe-after.json", {"record_type": "evaluation_group", "observations": [hint]})
    review = write(directory / "probe-review.json", {"record_type": "probe_review", "decision": "PASS", "controlled": True, "probe_id": "probe0", "cause_ids": ["cause0"], "outcome": "supports_hypothesis", "completed_at": "2026-10-19T00:02:10Z", "synthetic": True})
    probe["record"] = write(directory / "probe-record.json", {"record_type": "controlled_probe", "probe_id": "probe0", "cause_ids": ["cause0"], "failure_ids": ["failure0"], "model_id": "S0", "changed_factor": "evidence_hint", "expected_exact_direction": "increase", "protocol": controlled, "frozen_at": "2026-10-19T00:00:00Z", "completed_at": "2026-10-19T00:02:00Z", "baseline_evaluations": baseline, "intervention_evaluations": intervention, "unchanged_controls": ["model", "source", "question_semantics", "scoring"], "outcome": "supports_hypothesis", "review": review})
    cause = doc["causes"][0]
    cause.update(supporting_root_ids=roots[:3], supporting_family_keys=[C.family(questions[n]["provenance"]) for n in range(3)], evidence_refs=doc["failure_cases"][0]["evidence_refs"])
    cause["state_history"] = [{"at": "2026-10-19T00:03:00Z", "model_id": "S0", "from": "hypothesis", "to": "reproduced", "reason": "Synthetic fixture, not a finding", "record": probe["record"]}]
    result_bindings = []

    def comparison(name, before_model, after_model, before_ok, after_ok, minute):
        _, bef = run(3, name + "-before", model=before_model, correct=before_ok, minute=minute)
        _, aft = run(3, name + "-after", model=after_model, correct=after_ok, minute=minute + 1)
        bg = write(directory / (name + "-before-group.json"), {"record_type": "evaluation_group", "observations": [bef]})
        ag = write(directory / (name + "-after-group.json"), {"record_type": "evaluation_group", "observations": [aft]})
        protocol = write(directory / (name + "-protocol.json"), {"record_type": "new_question_comparison_protocol", "frozen_at": "2026-10-19T00:00:00Z", "metric": "exact_rate"})
        review = write(directory / (name + "-review.json"), {"record_type": "comparison_review", "decision": "PASS", "result_id": name, "new_untrained_family": True, "completed_at": f"2026-10-19T00:{minute + 2:02}:10Z", "synthetic": True})
        record = write(directory / (name + "-comparison.json"), {"record_type": "new_question_comparison", "result_id": name, "cause_id": "cause0", "before_protocol": protocol, "after_protocol": protocol, "before_evaluations": bg, "after_evaluations": ag, "completed_at": f"2026-10-19T00:{minute + 2:02}:00Z", "review": review})
        result = {"result_id": name, "provenance": questions[3]["provenance"], "new_family_for_memory": True, "before_model_id": before_model, "after_model_id": after_model, "same_protocol": True, "before_exact_rate": int(before_ok), "after_exact_rate": int(after_ok), "record": record, "interpretation": "Synthetic boundary fixture only"}
        cause["new_question_results"].append(result)
        result_bindings.append({"result_id": name, "record": record})
        return record

    if status in {"improving", "recurred"}:
        record = comparison("gain", "S0", "B1", False, True, 7)
        cause["state_history"].append({"at": "2026-10-19T00:10:00Z", "model_id": "B1", "from": "reproduced", "to": "improving", "reason": "Synthetic positive change", "record": record})
    if status == "recurred":
        record = comparison("decline", "B1", "B2", True, False, 13)
        cause["state_history"].append({"at": "2026-10-19T00:16:00Z", "model_id": "B2", "from": "improving", "to": "recurred", "reason": "Synthetic later decline", "record": record})
    cause["confidence_status"] = status
    if diagnostic:
        raw, observation = run(0, "diagnostic", minute=4, mode="diagnostic")
        run_doc = json.loads(Path(raw["path"]).read_text())
        response = json.loads(Path(run_doc["response"]["path"]).read_text())
        trace = {"document_type": "solver_diagnostic_trace", "schema_version": "1.0", "trace_id": "synthetic-trace", "request_id": "diagnostic", "created_at": "2026-10-19T00:04:21Z", "model_id": "S0", "provenance": questions[0]["provenance"], "question_id": "q0", "task_type": "multi_select", "public_question": questions[0]["public"], "source_input": questions[0]["source"], "display_mapping": questions[0]["mapping"], "protocol": C.reference(C.DAY1 / "PROTOCOL_V1.json"), "gold_visible": False, "author_evidence_visible": False, "output_budget_tokens": 1024, "actual_output_tokens": len(response["output_token_ids"]), "per_item": [{"content_id": f"i{i}", "judgment": "supported", "evidence_refs": [], "brief_basis": "Synthetic only"} for i in range(4)], "order_pairs": [], "final_prediction_content_ids": ["i0"], "format_valid": True, "raw_response": run_doc["response"]}
        doc["failure_cases"][0]["diagnostic_trace"] = write(directory / "diagnostic-trace.json", trace)
        failure_bindings[0].update(diagnostic_run=raw, diagnostic_evaluation=observation)
    doc.update(created_at="2026-10-19T00:20:00Z", taxonomy=C.reference(C.DAY1 / "diagnosis/WEAKNESS_TAXONOMY.json"), allowed_train_roots_manifest=C.reference(C.DAY1 / "TRAIN_ROOTS.json"))
    target = write(directory / "snapshot.json", doc)
    sidecar = write(directory / "controller.json", {"record_type": "diagnosis_controller_manifest", "schema_version": 1, "target": target, "frozen_protocol": C.reference(C.DAY1 / "PROTOCOL_V1.json"), "models": models, "failure_bindings": failure_bindings, "probe_bindings": [{"probe_id": "probe0", "record": probe["record"]}], "result_bindings": result_bindings})
    return Path(target["path"]), Path(sidecar["path"])


def as_ordering(directory):
    modify(directory / "q0-public.json", lambda x: x.update(items={"I" + str(n + 1): text for n, text in enumerate(x.pop("options").values())}))
    modify(directory / "q0-mapping.json", lambda x: x.update(display_to_content={"I" + str(n + 1): "i" + str(n) for n in range(4)}))
    modify(directory / "q0-gold.json", lambda x: x.update(task_type="ordering", gold_content_ids=["i0", "i1", "i2", "i3"]))
    tokenizer_ref = json.loads((directory / "fail0-run.json").read_text())["tokenizer"]
    tokenizer = Tokenizer.from_file(tokenizer_ref["path"])
    for name, answer, predicted in [("fail0", "<box>I2,I1,I3,I4</box>", ["i1", "i0", "i2", "i3"]), ("hint", "<box>I1,I2,I3,I4</box>", ["i0", "i1", "i2", "i3"])]:
        modify(directory / (name + "-request.json"), lambda x: x.update(task_type="ordering"))
        modify(directory / (name + "-response.json"), lambda x: x.update(text=answer, output_token_ids=tokenizer.encode(answer, add_special_tokens=False).ids, prediction_content_ids=predicted))
    modify(directory / "snapshot.json", lambda x: x["failure_cases"][0].update(observed_errors=["obs.order.within_scene_pair"]))


def main():
    checks = []
    def check(name, status="reproduced", diagnostic=False, mutate=None, valid=False, error_contains=None):
        with tempfile.TemporaryDirectory(prefix="day2-controller-synthetic-") as tmp:
            target, sidecar = fixture(tmp, status, diagnostic)
            if mutate:
                mutate(Path(tmp))
                refresh(Path(tmp))
            result = C.validate_file(target, sidecar)
            passed = (result["status"] == "PASS") == valid
            if error_contains and result["status"] == "FAIL": passed &= error_contains in result["error"]
            checks.append({"name": name, "passed": bool(passed), "expected_valid": valid, "result": result})
    for state in ["reproduced", "improving", "recurred"]:
        check("bound_" + state + "_valid", state, valid=True)
    check("mixed_ordering_bound_valid", mutate=as_ordering, valid=True)
    check("bound_diagnostic_valid", diagnostic=True, valid=True)
    # The hash test intentionally skips DAG refresh, unlike semantic tests below.
    with tempfile.TemporaryDirectory(prefix="day2-controller-synthetic-") as tmp:
        target, sidecar = fixture(tmp)
        Path(tmp, "fail0-response.json").write_text("tampered")
        result = C.validate_file(target, sidecar)
        checks.append({"name": "raw_response_hash_tamper_rejected", "passed": result["status"] == "FAIL" and "hash mismatch" in result["error"], "result": result})
    check("gold_read_before_response_rejected", mutate=lambda p: modify(p / "fail0-eval.json", lambda x: x.update(gold_first_read_at="2026-10-19T00:00:19Z")), error_contains="strictly before Gold")
    check("global_first_gold_read_claim_rejected", mutate=lambda p: modify(p / "fail0-eval.json", lambda x: x.update(gold_read_scope="entire_conversation_first_read")), error_contains="this scoring run")
    check("late_content_gate_rejected", mutate=lambda p: modify(p / "q0-gate.json", lambda x: x.update(completed_at="2026-10-19T00:00:11Z")), error_contains="admission must precede")
    check("forged_prediction_rejected", mutate=lambda p: modify(p / "fail0-response.json", lambda x: x.update(prediction_content_ids=["i3"])), error_contains="original raw answer")
    check("forged_rate_rejected", status="improving", mutate=lambda p: modify(p / "snapshot.json", lambda x: x["causes"][0]["new_question_results"][0].update(after_exact_rate=0.5)), error_contains="rate differs")
    check("invented_failure_denominator_rejected", mutate=lambda p: modify(p / "snapshot.json", lambda x: x["failure_cases"][0]["repeat_aggregation"].update(valid_calls=18)), error_contains="numerator/denominator")
    check("unrelated_probe_rejected", mutate=lambda p: modify(p / "probe-record.json", lambda x: x.update(cause_ids=["unrelated-cause"])), error_contains="missing cause")
    check("cross_model_probe_rejected", mutate=lambda p: modify(p / "probe-record.json", lambda x: x.update(model_id="B1")), error_contains="changes current model")
    check("probe_effect_direction_changed_rejected", mutate=lambda p: modify(p / "probe-record.json", lambda x: x.update(expected_exact_direction="decrease")), error_contains="effect was changed")
    check("probe_without_controls_rejected", mutate=lambda p: modify(p / "probe-record.json", lambda x: x.update(unchanged_controls=["model"])), error_contains="common controls")
    check("probe_without_actual_intervention_rejected", mutate=lambda p: modify(p / "hint-request.json", lambda x: x.update(prompt=json.loads((p / "fail0-request.json").read_text())["prompt"])), error_contains="actual changed input")
    check("probe_frozen_after_collection_rejected", mutate=lambda p: [modify(p / n, lambda x: x.update(frozen_at="2026-10-19T00:01:11Z")) for n in ["probe-protocol.json", "probe-record.json"]], error_contains="frozen after collection")
    check("decline_cannot_be_improving", status="recurred", mutate=lambda p: modify(p / "snapshot.json", lambda x: x["causes"][0].update(confidence_status="improving", state_history=[x["causes"][0]["state_history"][0], dict(x["causes"][0]["state_history"][-1], to="improving", **{"from": "reproduced"})])), error_contains="positive effect")
    check("none_to_recurred_rejected", status="recurred", mutate=lambda p: modify(p / "snapshot.json", lambda x: x["causes"][0].update(state_history=[dict(x["causes"][0]["state_history"][-1], **{"from": "none"})])), error_contains="invalid cause-state transition")
    check("recurred_without_recorded_improvement_rejected", status="recurred", mutate=lambda p: modify(p / "snapshot.json", lambda x: x["causes"][0].update(state_history=[x["causes"][0]["state_history"][-1]])), error_contains="prior improvement")
    check("recurrence_before_improvement_rejected", status="recurred", mutate=lambda p: modify(p / "snapshot.json", lambda x: x["causes"][0]["state_history"][-1].update(at="2026-10-19T00:09:30Z")), error_contains="history must be chronological")
    check("trained_on_family_not_new_rejected", status="improving", mutate=lambda p: modify(p / "B1-train.json", lambda x: x.update(training_family_keys=[C.family(json.loads((p / "q3-gate.json").read_text())["provenance"])])), error_contains="trained-on question/family")
    check("training_before_checkpoint_completion_rejected", status="improving", mutate=lambda p: modify(p / "B1-train.json", lambda x: x.update(completed_at="2026-10-19T00:08:11Z")), error_contains="before its training completed")
    check("self_raised_diagnostic_budget_rejected", diagnostic=True, mutate=lambda p: [modify(p / n, lambda x: x.update(**({"max_output_tokens": 2048} if n.endswith("request.json") else {"output_budget_tokens": 2048}))) for n in ["diagnostic-request.json", "diagnostic-trace.json"]], error_contains="1024 cap")
    check("inflated_original_diagnostic_tokens_rejected", diagnostic=True, mutate=lambda p: modify(p / "diagnostic-response.json", lambda x: x.update(output_token_ids=[1] * 1025)), error_contains="exceeds request budget")
    check("trace_actual_tokens_disagree_rejected", diagnostic=True, mutate=lambda p: modify(p / "diagnostic-trace.json", lambda x: x.update(actual_output_tokens=1)), error_contains="token counts differ")
    check("trace_changes_raw_answer_rejected", diagnostic=True, mutate=lambda p: modify(p / "diagnostic-trace.json", lambda x: x.update(final_prediction_content_ids=["i1"])), error_contains="final prediction differs")
    check("trace_changes_raw_item_judgment_rejected", diagnostic=True, mutate=lambda p: modify(p / "diagnostic-trace.json", lambda x: x["per_item"][0].update(judgment="refuted")), error_contains="judgment differs")
    check("model_lineage_cycle_rejected", status="improving", mutate=lambda p: modify(p / "controller.json", lambda x: x["models"][0].update(parent_model_id="B2")), error_contains=None)
    check("unrelated_improvement_model_rejected", status="improving", mutate=lambda p: modify(p / "controller.json", lambda x: x["models"][1].update(parent_model_id=None, training_record=None)), error_contains="unrelated root model")
    check("public_gold_field_rejected", mutate=lambda p: modify(p / "q0-public.json", lambda x: x.update(gold=["i1"])), error_contains="private field")
    check("omitted_source_turn_rejected", mutate=lambda p: modify(p / "fail0-request.json", lambda x: x.update(prompt=x["prompt"].split("\n", 1)[1])), error_contains="complete source")
    check("invented_source_quote_rejected", mutate=lambda p: modify(p / "snapshot.json", lambda x: x["failure_cases"][0]["evidence_refs"][0].update(quote="Not in this source")), error_contains="quote/speaker")
    check("unknown_gold_content_id_rejected", mutate=lambda p: modify(p / "q0-gold.json", lambda x: x.update(gold_content_ids=["unknown-id"])), error_contains="nonempty content-ID answer")
    check("reproduction_state_before_probe_review_rejected", mutate=lambda p: modify(p / "snapshot.json", lambda x: x["causes"][0]["state_history"][0].update(at="2026-10-19T00:02:09Z")), error_contains="precedes controlled-probe review")
    empty = C.validate_file(C.DAY1 / "WEAKNESS_MEMORY_B_v000.json")
    checks.append({"name": "frozen_empty_v000_compatible", "passed": empty["status"] == "PASS", "result": empty})
    report = {"status": "PASS" if all(t["passed"] for t in checks) else "FAIL", "record_type": "synthetic_controller_boundary_validation", "created_at_utc": datetime.now(timezone.utc).isoformat(), "test_count": len(checks), "passed_count": sum(t["passed"] for t in checks), "synthetic_fixtures_only": True, "real_failures_created": 0, "real_causes_created": 0, "model_calls": 0, "training_runs": 0, "checks": checks, "files": {p.name: {"sha256": C.sha(p), "bytes": p.stat().st_size} for p in [C.HERE / "controller.py", Path(__file__)]}}
    write(C.HERE / "VALIDATION.json", report)
    print(json.dumps({k: report[k] for k in ["status", "test_count", "passed_count", "model_calls", "training_runs"]}))
    for check in checks:
        if not check["passed"]: print(json.dumps(check, ensure_ascii=False))
    raise SystemExit(report["status"] != "PASS")


if __name__ == "__main__":
    main()
