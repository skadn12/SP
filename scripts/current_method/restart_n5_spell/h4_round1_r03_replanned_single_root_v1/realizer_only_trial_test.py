import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import realizer_only_trial as trial
from ..h4_round1_r03_incremental_plan_retention_v1 import planner_relations as relations


class _Counter:
    pass


class _Backend:
    def __init__(self, *, invalid_first=False):
        self.invalid_first = invalid_first
        self.calls = []
        self.response_evidence_sink = None
        self.last_response_envelope = None

    def request(self, request):
        self.calls.append(request)
        if self.response_evidence_sink:
            self.response_evidence_sink({"evidence_event": "response_received", "http_status": 200})
        if self.invalid_first and len(self.calls) == 1:
            raw = '{"scenes": []}'
        else:
            raw = json.dumps({"scenes": [
                {"dialogue_turns": [
                    {"speaker": "A" if i % 2 == 0 else "B",
                     "text": f"batch{len(self.calls)} scene{j} turn{i} unique"}
                    for i in range(9)]}
                for j in range(2)]}, ensure_ascii=False)
        self.last_response_envelope = {"choices": [{"finish_reason": "stop"}]}
        if self.response_evidence_sink:
            self.response_evidence_sink({"evidence_event": "completed", "http_status": 200})
        return raw


class _RepeatThirdBackend(_Backend):
    def request(self, request):
        if len(self.calls) == 1:
            # Capture the complete second response; call 3 replays it verbatim.
            raw = super().request(request)
            self.second_raw = raw
            return raw
        if len(self.calls) == 2:
            self.calls.append(request)
            return self.second_raw
        return super().request(request)


class _DuplicateBackend(_Backend):
    def request(self, request):
        self.calls.append(request)
        return '{"scenes": [], "scenes": []}'


def _scene(index):
    return {
        "memory_event_id": f"ME{index:02d}",
        "selected_candidate_ref": f"ME{index:02d}:C01",
        "phase_id": f"initial:ME{index:02d}:C01",
        "required_participants": ["A", "B"],
        "speaker_refs": ["A", "B"],
        "goal": "make progress",
        "before_state": "starting",
        "required_transition": f"perform action {index}",
        "after_state": "new state",
        "actor_bindings": [{"actor_ref": "A", "binding": "acts"}],
        "interaction_beats": ["A proposes", "B checks", "A records"],
        "dependencies": [{"phase_id": "initial:ME01:C01", "predecessor_candidate_ref": "ME01:C01",
                           "relation_kind": "natural_continuity", "condition": None}]
        if index > 1 else [],
    }


def _prepared(source):
    parent = {"root_id": "root-test", "dialogue": {"turns": [
        {"turn_id": "P:T01", "speaker": "A", "text": "old parent fact"},
        {"turn_id": "P:T02", "speaker": "B", "text": "old parent answer"},
    ]}}
    scenes = [_scene(i) for i in range(1, 7)]
    return {
        "root_id": "root-test", "source_run": str(source), "bundle_file": str(source / "bundle.json"),
        "bundle_sha256": "x", "source_hashes": {}, "parent": parent, "scenes": scenes,
        "relation_projection": {"selected_candidate_refs": [s["selected_candidate_ref"] for s in scenes], "relations": []},
        "relation_sidecar": {"relations": [], "output_phase_map": {f"/x/{i}": f"initial:ME{i:02d}:C01" for i in range(1, 7)}, "source_candidate_phase_map": {}, "nodes": []},
        "model": {"model": "mock", "temperature": 0, "tokenizer_path": "mock", "tokenizer_json_sha256": "mock"},
        "budget": {"role_requests": {role: (3 if role == "realizer" else 0) for role in trial.validate_budget.__globals__["ROLES"]},
                   "max_output_tokens": {role: 1000 for role in trial.validate_budget.__globals__["ROLES"]}},
    }


class RealizerOnlyTrialTests(unittest.TestCase):
    def test_three_batches_prefix_ids_relations_and_http_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source"; source.mkdir()
            prepared = _prepared(source)
            prepared["source_hashes"] = trial.snapshot(source)
            out = Path(td) / "out"
            backend = _Backend()
            fake_measure = lambda messages, counter, budget, role: {"fits": True, "input_tokens": 1}
            with patch.object(trial, "local_counter", return_value=_Counter()), patch.object(trial, "measure", fake_measure):
                result = trial.execute(prepared, out, backend)
            self.assertEqual(result["status"], "REALIZER_GENERATION_COMPLETE_STRUCTURE_VALID")
            self.assertEqual(result["realizer_real_calls"], 3)
            self.assertEqual([json.loads((out / "batches" / f"{i:03d}/request_manifest.json").read_text())["prefix_turn_count"] for i in range(1, 4)], [0, 18, 36])
            child = json.loads((out / "generated_child.json").read_text())["scenes"]
            self.assertEqual([s["scene_ref"] for s in child], [f"R03:S{i:02d}" for i in range(1, 7)])
            self.assertEqual([s["session_ref"] for s in child], [s["scene_ref"] for s in child])
            self.assertEqual(len({t["turn_ref"] for s in child for t in s["dialogue_turns"]}), 54)
            for i in range(1, 4):
                self.assertTrue((out / "batches" / f"{i:03d}/http_completed.json").exists())
            second_text = (out / "batches/002/prompt.txt").read_text()
            second = json.loads(second_text.split("INPUT DATA\n", 1)[1])
            self.assertEqual(len(second["retained_actual_prefix"]), 18)
            self.assertTrue(any(d.get("relation_kind") == "natural_continuity" for d in second["planned_phases_not_facts"][0]["dependencies"]))
            third_text = (out / "batches/003/prompt.txt").read_text()
            third = json.loads(third_text.split("INPUT DATA\n", 1)[1])
            self.assertEqual(len(third["retained_actual_prefix"]), 36)
            self.assertEqual(len(json.loads((out / "batches/002/request_manifest.json").read_text())["candidate_refs"]), 2)

    def test_first_invalid_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source"; source.mkdir()
            prepared = _prepared(source); prepared["source_hashes"] = trial.snapshot(source)
            backend = _Backend(invalid_first=True)
            with patch.object(trial, "local_counter", return_value=_Counter()), patch.object(trial, "measure", lambda *args, **kwargs: {"fits": True}):
                result = trial.execute(prepared, Path(td) / "out", backend)
            self.assertEqual(result["status"], "REALIZER_OUTPUT_INVALID_STOP")
            self.assertEqual(result["realizer_real_calls"], 1)
            self.assertEqual(len(backend.calls), 1)

    def test_third_batch_copy_is_rejected_and_raw_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source"; source.mkdir()
            prepared = _prepared(source); prepared["source_hashes"] = trial.snapshot(source)
            backend = _RepeatThirdBackend()
            out = Path(td) / "out"
            with patch.object(trial, "local_counter", return_value=_Counter()), patch.object(trial, "measure", lambda *args, **kwargs: {"fits": True}):
                result = trial.execute(prepared, out, backend)
            self.assertEqual(result["status"], "REALIZER_OUTPUT_INVALID_STOP")
            self.assertEqual(result["realizer_real_calls"], 3)
            self.assertEqual(result["scene_count"], 4)
            self.assertEqual(result["turn_count"], 36)
            self.assertEqual([json.loads((out / "batches" / f"{i:03d}/request_manifest.json").read_text())["prefix_turn_count"] for i in range(1, 4)], [0, 18, 36])
            errors = json.loads((out / "batches/003/validation.json").read_text())["errors"]
            self.assertTrue(any("parent_exact_copy" in error for error in errors))
            self.assertTrue((out / "batches/003/raw_response.json").exists())
            self.assertEqual(len(json.loads((out / "generated_child.json").read_text())["scenes"]), 4)

    def test_duplicate_json_parse_error_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source"; source.mkdir()
            prepared = _prepared(source); prepared["source_hashes"] = trial.snapshot(source)
            backend = _DuplicateBackend()
            with patch.object(trial, "local_counter", return_value=_Counter()), patch.object(trial, "measure", lambda *args, **kwargs: {"fits": True}):
                result = trial.execute(prepared, Path(td) / "out", backend)
            self.assertEqual(result["status"], "REALIZER_TECHNICAL_OR_PARSE_ERROR_STOP")
            self.assertEqual(result["realizer_real_calls"], 1)
            self.assertEqual(len(backend.calls), 1)

    def test_project_scenes_rejects_selected_rejected_and_hard_prerequisite(self):
        registry = {s["selected_candidate_ref"]: {"scene": s} for s in (_scene(i) for i in range(1, 7))}
        compiled = {"prepared": {"candidate_registry": registry,
                                  "compiled": {"scenes": [{"memory_event_id": f"ME{i:02d}"} for i in range(1, 7)]}}}
        sidecar = {"relations": []}
        decisions = [{"candidate_ref": f"ME{i:02d}:C01", "candidate_id": f"ME{i:02d}:C01", "decision": "pass"} for i in range(1, 7)]
        bad = list(decisions); bad[0] = {"candidate_ref": "ME01:C01", "candidate_id": "ME01:C01", "decision": "reject"}
        with self.assertRaises(ValueError):
            trial.project_scenes(compiled, bad, sidecar)
        sidecar["relations"] = [{"owner_phase_id": "initial:ME02:C01", "predecessor_phase_id": "initial:ME01:C01",
                                  "relation_kind": relations.HARD, "condition": "fact"}]
        with self.assertRaises(ValueError):
            trial.project_scenes(compiled, decisions, sidecar)


if __name__ == "__main__":
    unittest.main()
