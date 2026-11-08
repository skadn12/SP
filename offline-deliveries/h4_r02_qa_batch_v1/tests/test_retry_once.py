"""Offline request-cache and transport-retry boundary tests; no client creation."""
import importlib.util
from pathlib import Path
import tempfile
import json
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retry_once_test_module", HERE / "retry_once.py")
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


def fixture(directory, *, failed=False):
    payload = {"model": retry.gen.MODEL, "messages": [{"role": "user", "content": "same frozen payload"}]}
    proof = {"root_id": "root", "role": "question_generator", "question_type": "ordering",
             "candidate_ids": ["q1", "q2"], "source_sha256": "source", "feedback_receipt": {"target_solver_model": "M_clean_policy"}}
    retry.base.write(directory / "REQUEST.json", payload)
    retry.base.write(directory / "PROOF.json", {**proof, "payload_sha256": retry.base.digest(payload), "capacity": {"status": "PASS"}})
    retry.base.write(directory / "PROFILE.json", {"model": retry.gen.MODEL})
    retry.base.write(directory / "HTTP_ATTEMPT.json", {"actual_payload_equals_frozen": True, "payload_sha256": retry.base.digest(payload)})
    if failed:
        retry.base.write(directory / "ERROR.json", {"error_type": "TransportError", "metadata": {"http_status": 503}})
    else:
        # Successful HTTP remains reusable even if the raw draft fails content/schema validation.
        retry.base.write(directory / "RESPONSE.json", {"text": "invalid question JSON", "metadata": {"returned_model": retry.gen.MODEL, "finish_reason": "stop"}})
        retry.base.write(directory / "VALIDATION.json", {"accepted_ids": [], "rejected": ["q1", "q2"]})
    return payload, proof


def caller(run, row):
    result = retry.ReplayCaller.__new__(retry.ReplayCaller)
    result.run = run; result.counter = 0; result.new_http = 0
    result.policy = {"max_new_http_attempts": 12}
    result.cache = {row["payload_sha256"]: row}; result.old_stages = {row["stage"]: row}
    result.retried = set(); result.recovered = {}
    return result


def row_for(directory, payload, failed=False):
    return {"directory": str(directory), "payload_sha256": retry.base.digest(payload), "stage": "root-order-generate-1",
            "disposition": "RETRY_TRANSPORT_ONCE" if failed else "REUSE_SUCCESSFUL_HTTP"}


def test_mechanically_rejected_success_is_reused_without_http():
    with tempfile.TemporaryDirectory() as temp:
        parent = Path(temp) / "parent"; run = Path(temp) / "new"
        payload, proof = fixture(parent)
        obj = caller(run, row_for(parent, payload))
        with patch.object(retry.ORIGINAL_CALLER, "call", side_effect=AssertionError("unexpected API route")):
            raw, directory, metadata = obj.call("root-order-generate-1", payload, proof)
        assert raw == "invalid question JSON"
        assert (directory / "REUSE.json").exists() and not (directory / "HTTP_ATTEMPT.json").exists()
        assert obj.new_http == 0


def test_retry_once_unlocks_only_required_downstream_slots():
    def fake_call(self, label, payload, proof):
        self.counter += 1
        directory = self.run / "requests" / f"{self.counter:03d}-{label}"
        retry.base.write(directory / "HTTP_ATTEMPT.json", {"mock": True})
        return "recovered", directory, {"returned_model": retry.gen.MODEL}
    with tempfile.TemporaryDirectory() as temp:
        parent = Path(temp) / "parent"; run = Path(temp) / "new"
        payload, proof = fixture(parent, failed=True)
        obj = caller(run, row_for(parent, payload, failed=True))
        with patch.object(retry.ORIGINAL_CALLER, "call", fake_call):
            obj.call("root-order-generate-1", payload, proof)
            assert obj.new_http == 1 and obj.recovered == {("root", "ordering"): {"q1", "q2"}}
            try:
                obj.call("root-order-generate-1", payload, proof)
            except ValueError as error:
                assert str(error) == "failed_payload_already_retried"
            else:
                raise AssertionError("same failure retried twice")
            review = {"root_id": "root", "role": "ordering_reviewer", "candidate_id": "q1"}
            obj.call("root-order-review-01", {"new": "review"}, review)
            assert obj.new_http == 2
            for forbidden in ({**review, "candidate_id": "q3"}, {**review, "root_id": "other_root"}):
                try:
                    obj.call("root-order-review-03", {"new": "forbidden"}, forbidden)
                except ValueError as error:
                    assert str(error) == "new_request_outside_recovered_slots"
                else:
                    raise AssertionError("unrelated candidate generated")


def test_existing_stage_cannot_be_regenerated_with_changed_payload():
    with tempfile.TemporaryDirectory() as temp:
        parent = Path(temp) / "parent"; payload, proof = fixture(parent)
        obj = caller(Path(temp) / "new", row_for(parent, payload))
        obj.recovered = {("root", "ordering"): {"q1", "q2"}}
        try:
            obj.call("root-order-generate-1", {"different": "payload"}, proof)
        except ValueError as error:
                assert str(error) in ("existing_stage_payload_changed_no_regeneration", "frozen_prior_message_schema")
        else:
            raise AssertionError("existing stage regenerated")


def test_parent_success_content_change_is_detected():
    with tempfile.TemporaryDirectory() as temp:
        parent, run = Path(temp) / "parent", Path(temp) / "new"
        q = {"candidate_id": "q", "question_type": "multi_select", "question": "same", "gold": ["A", "D"]}
        for directory in (parent, run):
            for name in ("MECHANICAL_POOL.json", "SOLVER_POOL.json"):
                retry.base.write(directory / name, {"questions": [q]})
        assert retry.verify_preserved(parent, run)["status"] == "PASS"
        (run / "SOLVER_POOL.json").write_text('{"questions":[{"candidate_id":"q","gold":["B","D"]}]}')
        try:
            retry.verify_preserved(parent, run)
        except ValueError as error:
            assert str(error) == "parent_success_content_changed:q"
        else:
            raise AssertionError("parent Gold change passed")


def test_frozen_ordering_branch_replays_actual_old_context_and_no_http():
    with tempfile.TemporaryDirectory() as temp:
        parent = Path(temp) / "parent"
        context = {"root_id": "root", "candidate_ids": ["q1", "q2"], "source_turns": [{"text": "frozen source"}],
                   "previous_candidates_dedup_only": [{"candidate_id": "old", "facts": ["fact"]}]}
        def payload_for(data):
            return {"model": retry.gen.MODEL, "messages": [{"role": "user", "content":
                    "instructions\nINPUT_JSON\n" + json.dumps(data) + "\nUNCHANGED_FEEDBACK"}]}
        payload = payload_for(context)
        proof = {"root_id": "root", "role": "question_generator", "question_type": "ordering",
                 "candidate_ids": ["q1", "q2"], "source_sha256": "same", "feedback_receipt": {"frozen": True}}
        retry.base.write(parent / "REQUEST.json", payload)
        retry.base.write(parent / "PROOF.json", {**proof, "payload_sha256": retry.base.digest(payload), "capacity": {"status": "PASS"}})
        retry.base.write(parent / "PROFILE.json", {})
        retry.base.write(parent / "RESPONSE.json", {"text": "old ordering draft", "metadata": {"returned_model": retry.gen.MODEL}})
        obj = caller(Path(temp) / "new", row_for(parent, payload))
        obj.recovered = {("root", "multi_select"): {"recovered_multi"}}
        context["previous_candidates_dedup_only"].append({"candidate_id": "recovered_multi", "facts": ["new fact"]})
        proposed = payload_for(context)
        with patch.object(retry.ORIGINAL_CALLER, "call", side_effect=AssertionError("unexpected API route")):
            raw, directory, _ = obj.call("root-order-generate-1", proposed, {**proof, "feedback_receipt": {"new_hash": True}})
        assert raw == "old ordering draft"
        assert retry.base.read(directory / "REQUEST.json") == payload
        note = retry.base.read(directory / "FROZEN_PRIOR_CONTEXT_REPLAY.json")
        assert note["proposed_payload_sha256"] != note["actual_frozen_payload_sha256"]
        assert note["recovered_multiselect_ids_not_visible_to_old_ordering"] == ["recovered_multi"]
        assert not (directory / "HTTP_ATTEMPT.json").exists()


def test_transmitted_unknown_transport_error_is_explicitly_retryable():
    with tempfile.TemporaryDirectory() as temp:
        parent = Path(temp)
        directory = parent / "requests/001-root-order-generate-1"
        fixture(directory, failed=True)
        (directory / "ERROR.json").write_text(json.dumps({"error_type": "ComparisonTransportError",
            "metadata": {"http_attempt_count": 1, "completion_status": "UNKNOWN", "timeout": False}}))
        retry.base.write(parent / "MANIFEST.json", {"files": []})
        retry.base.write(parent / "REPORT.json", {"http_attempts": 1})
        rows, count = retry.parent_inventory(parent)
        assert count == 1 and rows[0]["disposition"] == "RETRY_TRANSPORT_ONCE"
        assert rows[0]["prior_completion_unknown"] is True and rows[0]["http_status"] is None
