"""Synthetic guard checks only; never load this batch's QA or invoke a Solver."""
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
BATCH = PROJECT / "offline-deliveries/h4_codex_sequential_ab_pilot_v112"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BATCH / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def make_solver_fixture(folder):
    for side in ("base", "trained"):
        run = folder / "evaluation/run" / side
        requests = []
        results = []
        gold = {}
        for cid, qtype in [("synthetic_mc", "multi_select"), ("synthetic_order", "ordering")]:
            labels = list("ABCDEF") if qtype == "multi_select" else [f"I{i}" for i in range(1, 7)]
            mapping = {label: f"content{i}" for i, label in enumerate(labels)}
            gold[cid] = ["content0", "content1"] if qtype == "multi_select" else list(mapping.values())
            for variant in range(6 if qtype == "multi_select" else 12):
                for slot in range(3 if qtype == "multi_select" else 1):
                    rid = f"{cid}_{variant}_{slot}"
                    prompt = run / "prompts" / (rid + ".txt")
                    prompt.parent.mkdir(parents=True, exist_ok=True)
                    prompt.write_text("Synthetic prompt without current QA.")
                    request = {"request_id": rid, "candidate_id": cid, "root_id": "synthetic_root", "question_type": qtype,
                               "seed": len(requests), "slot": slot, "prompt_sha256": sha(prompt),
                               "variant": {"variant_id": f"perm_{variant:02d}", "display_to_content": mapping}}
                    requests.append(request)
                    raw = "<box>A,B</box>" if qtype == "multi_select" else "<box>I1,I2,I3,I4,I5,I6</box>"
                    write(run / "responses" / (rid + ".json"), {
                        **{k: request[k] for k in ("request_id", "candidate_id", "root_id", "question_type", "seed", "slot")},
                        "variant_id": f"perm_{variant:02d}", "evaluation_valid": True, "raw_output": raw,
                        "score": {"partial_score_fraction": "1", "exact_match": True}})
            results.append({"candidate_id": cid, "root_id": "synthetic_root", "question_type": qtype,
                            "full_partial_mean": "1", "full_exact_rate": "1", "selected_by_policy": False})
        write(run / "manifest.json", {"requests": requests, "request_count": len(requests), "candidate_count": 2})
        write(run / "gold.json", gold)
        write(run / "RESULTS.json", {"questions": results, "selected_questions": [], "call_set_errors": []})


class ToolingGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def auditor(self):
        result = module("synthetic_audit_v112", "audit_solver.py")
        result.HERE = self.folder
        return result

    def test_complete_synthetic_call_set_recomputes_and_binds_inputs(self):
        make_solver_fixture(self.folder)
        with contextlib.redirect_stdout(io.StringIO()):
            self.auditor().main()
        audit = read(self.folder / "INDEPENDENT_SOLVER_AUDIT.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["results"]["base"], {"calls": 30, "candidates": 2, "errors": 0})
        self.assertTrue(audit["input_bindings"])

    def test_result_omission_and_duplicate_are_rejected(self):
        for mode in ("omission", "duplicate"):
            case = self.folder / mode
            make_solver_fixture(case)
            path = case / "evaluation/run/base/RESULTS.json"
            value = read(path)
            value["questions"] = value["questions"][:1] if mode == "omission" else [value["questions"][0]] * 2
            write(path, value)
            audit = self.auditor()
            audit.HERE = case
            with self.assertRaises(AssertionError):
                audit.main()
            self.assertFalse((case / "INDEPENDENT_SOLVER_AUDIT.json").exists())

    def test_final_audit_gate_rejects_stale_audited_inputs(self):
        make_solver_fixture(self.folder)
        with contextlib.redirect_stdout(io.StringIO()):
            self.auditor().main()
        delivery = module("synthetic_delivery_v112", "deliver.py")
        delivery.HERE = self.folder
        self.assertEqual(delivery.require_independent_audit()["status"], "PASS")
        path = self.folder / "evaluation/run/base/RESULTS.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(AssertionError, "AUDITED_INPUT_CHANGED"):
            delivery.require_independent_audit()

    def test_final_audit_gate_requires_PASS(self):
        delivery = module("synthetic_delivery_failure_v112", "deliver.py")
        delivery.HERE = self.folder
        write(self.folder / "INDEPENDENT_SOLVER_AUDIT.json", {"status": "FAIL", "errors": ["fixture"]})
        with self.assertRaises(AssertionError):
            delivery.require_independent_audit()

    def test_frozen_tooling_gate_detects_tool_change(self):
        delivery = module("synthetic_delivery_freeze_v112", "deliver.py")
        delivery.HERE = self.folder
        files = []
        for name in ("evaluate.py", "evaluate_v2.py", "audit_solver.py", "deliver.py"):
            path = self.folder / name
            path.write_text("synthetic frozen code")
            files.append({"path": str(path.resolve()), "sha256": sha(path)})
        write(self.folder / "evidence/TOOLING_CORRECTION_AND_FREEZE.json", {
            "status": "FROZEN_BEFORE_SOLVER_LIVE", "created_at_utc": datetime.now(timezone.utc).isoformat(), "files": files})
        delivery.verify_tooling()
        (self.folder / "audit_solver.py").write_text("changed fixture")
        with self.assertRaisesRegex(AssertionError, "TOOLING_HASH_CHANGED"):
            delivery.verify_tooling()

    def test_v2_checks_preflight_again_after_stubbed_run(self):
        evaluation = module("synthetic_eval_v2", "evaluate_v2.py")
        evaluation.HERE = self.folder
        planned = self.folder / "planned.txt"
        planned.write_text("frozen")
        binding = self.folder / "evaluation/ADAPTER_BINDINGS.json"
        write(binding, {"files": []})
        write(self.folder / "evaluation/PAIRED_PREFLIGHT.json", {
            "files": [{"path": str(planned), "sha256": sha(planned)}],
            "adapter_binding": {"path": str(binding), "sha256": sha(binding)}})
        evaluation.native = lambda: SimpleNamespace(live=lambda run: planned.write_text("changed during synthetic run"), report=lambda run: None)
        with self.assertRaisesRegex(ValueError, "BOUND_FILE_CHANGED"):
            evaluation.live("base")
        self.assertFalse((self.folder / "evaluation/BASE_COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
