"""Offline regressions for model identity and complete response admission."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_solver


class RunnerTest(unittest.TestCase):
    def test_checkpoint_names_bind_to_historical_audits(self):
        base = run_solver.model_identity("M0")
        clean = run_solver.model_identity("M_clean_policy")
        self.assertEqual(base["model_path"], "/root/autodl-tmp/Qwen3-4B")
        self.assertTrue(clean["model_path"].endswith("/clean_merged_model"))
        self.assertEqual(clean["tokenizer_path"], base["tokenizer_path"])
        self.assertTrue(clean["adapter_already_merged"])
        self.assertEqual(clean["audit_sha256"], run_solver.TRAINED_AUDIT_SHA256)
        with self.assertRaisesRegex(RuntimeError, "UNSUPPORTED_SOLVER_MODEL"):
            run_solver.model_identity("latest")

    def test_preflight_original_evidence_binding_and_complete_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text(json.dumps({"root_id": "test-root", "turns": [
                {"turn_ref": f"T{i}", "session_ref": "S1", "speaker": "user", "text": f"source {i}"}
                for i in range(6)]}))
            q = {"candidate_id": "test-multi", "root_id": "test-root", "question_type": "multi_select",
                 "question": "Select the supported claims.", "options": {c: f"claim {c}" for c in "ABCDEF"},
                 "gold": ["A", "C"], "source_path": str(source), "source_sha256": run_solver.sha(source),
                 "option_evidence": {c: {"turn_ref": f"T{i}", "quote": f"source {i}",
                    "relation": "supports" if c in "AC" else "contradicts", "reason": f"reason {c}"}
                    for i, c in enumerate("ABCDEF")}}
            pool = root / "pool.json"
            pool.write_text(json.dumps({"questions": [q], "provenance": {"requested_model": "Qwen/Qwen3.6-35B-A3B"}}))
            run = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                run_solver.preflight(pool, run, require_model=False, solver_model="M_clean_policy")
            manifest = run_solver.read(run / "manifest.json")
            self.assertEqual(len(manifest["requests"]), 18)
            self.assertEqual({x["mode"] for x in manifest["requests"]}, {"full_context"})
            bindings = run_solver.read(run / "canonical_bindings.json")["candidates"][0]
            for label, text in q["options"].items():
                content_id = "content_" + hashlib.sha256(text.encode()).hexdigest()[:16]
                self.assertEqual(bindings["content_id_by_label"][label], content_id)
                self.assertEqual(bindings["evidence_by_content_id"][content_id], q["option_evidence"][label])
            responses = run / "responses"
            responses.mkdir()
            # One response per permutation used to permit admission with 12 missing calls.
            for request in manifest["requests"]:
                if request["slot"] != 0:
                    continue
                record = {k: request[k] for k in ("request_id", "candidate_id", "root_id", "question_type", "slot", "seed")}
                record.update(variant_id=request["variant"]["variant_id"], evaluation_valid=True,
                              score={"partial_score_fraction": "1/2", "exact_match": False})
                run_solver.write_once(responses / (request["request_id"] + ".json"), record)
            with contextlib.redirect_stdout(io.StringIO()):
                run_solver.report(run)
            result = run_solver.read(run / "RESULTS.json")["questions"][0]
            self.assertIsNone(result["selected_by_policy"])
            self.assertEqual(result["selection_reason"], "EVALUATION_INCOMPLETE_OR_INVALID")
            self.assertEqual(sum(x.startswith("MISSING_RESPONSE:") for x in result["data_errors"]), 12)
            # A dry preflight must never be accepted as a live checkpoint identity.
            with self.assertRaises(RuntimeError):
                run_solver._validate_model(run_solver.read(run / "solver_identity.json"))


if __name__ == "__main__":
    unittest.main()
