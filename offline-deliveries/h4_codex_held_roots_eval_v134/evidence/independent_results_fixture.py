"""One synthetic integration check for independent audit versus actual report."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile

sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parents[1]
REPO = BATCH.parent.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle)


def main():
    independent = module("independent_v134_fixture_audit", BATCH / "evidence/independent_results_audit.py")
    scorer = module("native_v134_fixture_scorer",
                    REPO / "offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation/scoring.py")
    report = module("production_v134_report_fixture", BATCH / "evaluation/report.py")
    synthetic = Path(tempfile.mkdtemp(prefix="v134-independent-audit-fixture-"))
    ev = synthetic / "evaluation"
    for name in ("SHARED_REQUESTS.json", "SCOPE.json", "EVALUATION_POOL.json"):
        write(ev / name, read(BATCH / "evaluation" / name))
    plan = read(ev / "SHARED_REQUESTS.json")
    canonical = read(ev / "EVALUATION_POOL.json")["questions"]
    by_id = {q["candidate_id"]: q for q in canonical}
    for side in ("base", "clean", "new"):
        actual = BATCH / "evaluation/run" / side
        run = ev / "run" / side
        manifest, gold = read(actual / "manifest.json"), read(actual / "gold.json")
        write(run / "manifest.json", manifest)
        write(run / "gold.json", gold)
        write(run / "COMPLETE.json", {"actual_call_count": len(plan["requests"])})
        write(run / "RESULTS.json", {"questions": [{"candidate_id": q["candidate_id"]} for q in canonical]})
        (run / "prompts").mkdir()
        for r in plan["requests"]:
            q = by_id[r["candidate_id"]]
            mapping = r["variant"]["display_to_content"]
            inverse = {value: key for key, value in mapping.items()}
            correct = [inverse[value] for value in gold[q["candidate_id"]]]
            if side == "base" and q["question_type"] == "multi_select":
                prediction = list(mapping)
            elif side == "new" and q["question_type"] == "ordering":
                prediction = list(reversed(correct))
            else:
                prediction = correct
            raw = "<box>" + ",".join(prediction) + "</box>"
            score = scorer.score_response(dict(q, gold_content_ids=gold[q["candidate_id"]]), r, raw)
            response = {key: r[key] for key in ("request_id", "candidate_id", "root_id",
                                                "question_type", "slot", "seed", "mode")}
            response.update(variant_id=r["variant"]["variant_id"],
                            solver_model=manifest["solver_model"], model_path=manifest["model"],
                            raw_output=raw, score=score, evaluation_valid=True, data_errors=[],
                            metadata={**plan["decoder_contract"], "finish_reason": "stop"})
            write(run / "responses" / (r["request_id"] + ".json"), response)
            (run / "prompts" / (r["request_id"] + ".txt")).symlink_to(
                actual / "prompts" / (r["request_id"] + ".txt"))
    # In-process fixture guard only. No production file, run or guard is edited.
    report.HERE = ev
    report.ev.frozen = lambda **kwargs: None
    result = report.main(ev / "reports")
    assert result == 0
    result = independent.main(synthetic, ev / "reports", synthetic / "SYNTHETIC_AUDIT.json")
    assert result == 0
    audit = read(synthetic / "SYNTHETIC_AUDIT.json")
    assert audit["status"] == "PASS" and audit["failure_count"] == 0
    assert audit["models"]["base"]["multi_select"]["known_fp_total"] > 0
    assert audit["models"]["new"]["ordering"]["correct_pairs"]["mean"] == 0
    assert audit["models"]["clean"]["exact_rate_fraction"] == "1"
    output = BATCH / "evidence/INDEPENDENT_RESULTS_AUDIT_FIXTURE.json"
    write(output, {"status": "PASS", "kind": "SYNTHETIC_NOT_LIVE_RESULTS",
                   "temporary_fixture": str(synthetic), "checks": audit["checks"],
                   "failure_count": audit["failure_count"], "formal_run_modified": False,
                   "inference_or_api_calls": 0, "production_files_modified": False,
                   "fixture_conditions": ["base selects all on every multi-select",
                                          "clean returns exact Gold for every request",
                                          "new reverses every ordering prediction"],
                   "script_bindings": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in [Path(__file__).resolve(),
                                  BATCH / "evidence/independent_results_audit.py",
                                  BATCH / "evaluation/report.py"]]})
    print(json.dumps({"status": "PASS", "synthetic_checks": audit["checks"],
                      "fixture_evidence": str(output)}))


if __name__ == "__main__":
    main()
