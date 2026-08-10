import json
import tempfile
import unittest
from pathlib import Path

import httpx

from scripts.current_method.restart_n5_spell.simple_append_v1.contracts import (
    PLANNER_SCHEMA_VERSION,
    REALIZER_SCHEMA_VERSION,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1 import (
    MemoraxClient,
    MemoraxAsyncWriteError,
    MemoraxConfig,
    MemoryWorkflowError,
    execute_single_round,
    submit_add_without_wait,
    submit_add_and_wait,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.async_write import (
    DEFAULT_POLL_DELAYS_SECONDS,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.async_recovery import (
    preflight as recovery_preflight,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.single_root_live import (
    _artifact_sink,
    run_preflight,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1.write_state import (
    apply_add_response,
    apply_task_response,
    apply_visibility_result,
    new_write_state,
    retry_allowed,
)


PARENT_PATH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/"
    "unified_root.json")
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def planner_fixture(parent):
    speakers = list(dict.fromkeys(
        str(row["speaker"]) for row in parent["dialogue"]["turns"]))[:2]
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "Continue the existing story through three developments.",
        "scenes": [{
            "goal": f"Develop a connected consequence in scene {index}.",
            "participants": speakers,
            "new_development": f"A distinct practical development {index} occurs.",
            "interaction_beats": [
                "A concrete fact emerges.",
                "The speakers respond differently.",
                "They choose a next step.",
            ],
            "transition_from_previous": (
                "Continue the parent tail." if index == 1 else
                f"Continue the actual result of scene {index - 1}."),
            "desired_outcome": f"The scene reaches distinct outcome {index}.",
        } for index in range(1, 4)],
    }


def realizer_fixture(parent):
    speakers = list(dict.fromkeys(
        str(row["speaker"]) for row in parent["dialogue"]["turns"]))[:2]
    return {
        "realizer_schema_version": REALIZER_SCHEMA_VERSION,
        "scene_dialogues": {
            f"R01:S{scene:02d}": [{
                "speaker": speakers[(turn + scene) % 2],
                "text": (
                    f"Fresh scene {scene} turn {turn} develops a concrete "
                    f"consequence that did not appear earlier."),
            } for turn in range(1, 10)]
            for scene in range(1, 4)
        },
    }


def bucket_response(include_child=False):
    events = {
        "earlier planning": {
            "display_key": "Earlier planning",
            "content": ["The coworkers previously discussed practical planning."],
        },
    }
    if include_child:
        events["new consequence"] = {
            "display_key": "New consequence",
            "content": ["The latest accepted dialogue adds a new consequence."],
        }
    return {"success": True, "data": {"buckets": {
        "person": {
            "co-workers a": {
                "display_key": "Co-workers A",
                "content": ["Co-workers A participates in the story."],
            },
            "co-workers b": {
                "display_key": "Co-workers B",
                "content": ["Co-workers B participates in the story."],
            },
        },
        "location": {}, "event": events, "item": {}, "dreaming": {},
    }}}


def mock_client():
    state = {"searches": 0, "requests": []}

    def handler(request):
        payload = json.loads(request.content.decode("utf-8"))
        state["requests"].append({"path": request.url.path, "payload": payload})
        if request.url.path == "/v1/memories/add":
            return httpx.Response(200, json={
                "success": True,
                "data": {
                    "status": "completed",
                    "session_id": payload["session_id"],
                },
            })
        if request.url.path == "/v1/memory-buckets/search":
            state["searches"] += 1
            return httpx.Response(
                200, json=bucket_response(include_child=state["searches"] > 1))
        return httpx.Response(404, json={"success": False})

    client = MemoraxClient(
        MemoraxConfig(base_url="https://memorax.example", api_key="fixture"),
        transport=httpx.MockTransport(handler),
    )
    return client, state


class MemoraxSingleRootWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))

    def test_complete_mock_round_adds_parent_then_only_accepted_child(self):
        client, state = mock_client()
        planner_prompts = []
        realizer_prompts = []
        result = execute_single_round(
            parent=self.parent,
            memorax=client,
            planner_call=lambda prompt: (
                planner_prompts.append(prompt) or planner_fixture(self.parent)),
            realizer_call=lambda prompt: (
                realizer_prompts.append(prompt) or realizer_fixture(self.parent)),
        )
        self.assertTrue(result["generation_ready"])
        self.assertTrue(result["qa_prompt_ready"])
        self.assertFalse(result["qa_pipeline_ready"])
        self.assertFalse(result["memory_is_qa_fact_source"])
        self.assertEqual(27, result["expanded_dialogue_validation"]["child_turn_count"])
        self.assertEqual(4, len(client.calls))
        self.assertEqual([
            "/v1/memories/add", "/v1/memory-buckets/search",
            "/v1/memories/add", "/v1/memory-buckets/search",
        ], [row["path"] for row in state["requests"]])
        add_requests = [row["payload"] for row in state["requests"]
                        if row["path"] == "/v1/memories/add"]
        self.assertEqual(60, len(add_requests[0]["messages"]))
        self.assertEqual(27, len(add_requests[1]["messages"]))
        self.assertEqual(0, add_requests[0]["metadata"]["round_index"])
        self.assertEqual(1, add_requests[1]["metadata"]["round_index"])
        self.assertTrue(add_requests[0]["async_mode"])
        self.assertTrue(add_requests[1]["async_mode"])
        self.assertIn("NON-AUTHORITATIVE MEMORAX MEMORY",
                      planner_prompts[0]["prompt"])
        self.assertIn("NON-AUTHORITATIVE MEMORAX MEMORY",
                      realizer_prompts[0]["prompt"])
        self.assertIn("QUESTION TOPIC SELECTION", result["mcq_prompt"]["prompt"])
        self.assertIn("new consequence", json.dumps(
            result["post_generation_memory_overview"]))

    def test_parent_memory_timeout_is_pending_and_does_not_block_generation(self):
        state = {"add_count": 0, "search_count": 0}

        def handler(request):
            path = request.url.path
            if path == "/v1/memories/add":
                state["add_count"] += 1
                if state["add_count"] == 1:
                    return httpx.Response(202, json={
                        "success": True,
                        "data": {"status": "accepted", "task_id": "parent-pending"},
                    })
                return httpx.Response(200, json={
                    "success": True,
                    "data": {"status": "completed"},
                })
            if path.startswith("/v1/memories/add/status/"):
                return httpx.Response(200, json={
                    "status": "processing", "memory": {}, "error": None,
                })
            if path == "/v1/memory-buckets/search":
                state["search_count"] += 1
                return httpx.Response(200, json=bucket_response(
                    include_child=state["search_count"] > 1))
            return httpx.Response(404, json={"success": False})

        client = MemoraxClient(
            MemoraxConfig(base_url="https://memorax.example", api_key="fixture"),
            transport=httpx.MockTransport(handler),
        )
        result = execute_single_round(
            parent=self.parent, memorax=client,
            planner_call=lambda prompt: planner_fixture(self.parent),
            realizer_call=lambda prompt: realizer_fixture(self.parent),
            poll_delays_seconds=(0, 0), sleep_fn=lambda value: None,
        )
        self.assertTrue(result["generation_ready"])
        self.assertTrue(result["qa_prompt_ready"])
        self.assertFalse(result["memory_write_verified"])
        self.assertEqual(["parent_memory_add"], [
            row["stage"] for row in result["pending_memory_writes"]])
        self.assertEqual(2, state["add_count"])

    def test_child_memory_timeout_is_pending_but_qa_handoff_remains_ready(self):
        state = {"add_count": 0, "search_count": 0}

        def handler(request):
            path = request.url.path
            if path == "/v1/memories/add":
                state["add_count"] += 1
                return httpx.Response(200, json={
                    "success": True,
                    "data": {"status": "completed"},
                }) if state["add_count"] == 1 else httpx.Response(202, json={
                    "success": True,
                    "data": {"status": "accepted", "task_id": "child-pending"},
                })
            if path.startswith("/v1/memories/add/status/"):
                return httpx.Response(200, json={
                    "status": "processing", "memory": {}, "error": None,
                })
            if path == "/v1/memory-buckets/search":
                state["search_count"] += 1
                return httpx.Response(200, json=bucket_response(
                    include_child=state["search_count"] > 1))
            return httpx.Response(404, json={"success": False})

        client = MemoraxClient(
            MemoraxConfig(base_url="https://memorax.example", api_key="fixture"),
            transport=httpx.MockTransport(handler),
        )
        result = execute_single_round(
            parent=self.parent, memorax=client,
            planner_call=lambda prompt: planner_fixture(self.parent),
            realizer_call=lambda prompt: realizer_fixture(self.parent),
            poll_delays_seconds=(0, 0), sleep_fn=lambda value: None,
        )
        self.assertTrue(result["generation_ready"])
        self.assertTrue(result["qa_prompt_ready"])
        self.assertFalse(result["memory_write_verified"])
        self.assertEqual(["child_memory_add"], [
            row["stage"] for row in result["pending_memory_writes"]])
        self.assertIsNone(result["verified_child_add"])

    def test_invalid_planner_never_adds_child(self):
        client, state = mock_client()
        invalid = {"planner_schema_version": PLANNER_SCHEMA_VERSION,
                   "round_goal": "bad", "scenes": []}
        with self.assertRaisesRegex(MemoryWorkflowError, "planner_contract"):
            execute_single_round(
                parent=self.parent, memorax=client,
                planner_call=lambda prompt: invalid,
                realizer_call=lambda prompt: self.fail("Realizer must not run"),
            )
        self.assertEqual(2, len(client.calls))
        self.assertEqual(1, sum(row["path"] == "/v1/memories/add"
                                for row in state["requests"]))

    def test_nonblocking_add_persists_task_id_and_continues_round(self):
        state = {"add_count": 0, "status_count": 0, "search_count": 0}

        def handler(request):
            path = request.url.path
            if path == "/v1/memories/add":
                state["add_count"] += 1
                return httpx.Response(202, json={
                    "success": True,
                    "data": {
                        "status": "accepted",
                        "task_id": f"pending-task-{state['add_count']:02d}",
                    },
                })
            if path.startswith("/v1/memories/add/status/"):
                state["status_count"] += 1
                return httpx.Response(200, json={"status": "processing"})
            if path == "/v1/memory-buckets/search":
                state["search_count"] += 1
                return httpx.Response(200, json=bucket_response())
            return httpx.Response(404, json={"success": False})

        client = MemoraxClient(
            MemoraxConfig(base_url="https://memorax.example", api_key="fixture"),
            transport=httpx.MockTransport(handler),
        )
        artifacts = {}
        result = execute_single_round(
            parent=self.parent,
            memorax=client,
            planner_call=lambda prompt: planner_fixture(self.parent),
            realizer_call=lambda prompt: realizer_fixture(self.parent),
            wait_for_memory_completion=False,
            artifact_sink=lambda name, value: artifacts.__setitem__(name, value),
        )
        self.assertTrue(result["generation_ready"])
        self.assertTrue(result["qa_prompt_ready"])
        self.assertEqual(2, state["add_count"])
        self.assertEqual(0, state["status_count"])
        self.assertEqual(2, state["search_count"])
        self.assertEqual(2, len(result["pending_memory_writes"]))
        self.assertEqual(
            "pending-task-01",
            artifacts["memory/parent_add_write_state"]["task_id"],
        )
        self.assertEqual(
            "pending-task-02",
            artifacts["memory/child_add_write_state"]["task_id"],
        )
        self.assertFalse(result["memory_write_verified"])

    def test_invalid_realizer_never_adds_child(self):
        client, state = mock_client()
        with self.assertRaisesRegex(MemoryWorkflowError, "realizer_contract"):
            execute_single_round(
                parent=self.parent, memorax=client,
                planner_call=lambda prompt: planner_fixture(self.parent),
                realizer_call=lambda prompt: {
                    "realizer_schema_version": REALIZER_SCHEMA_VERSION,
                    "scene_dialogues": {},
                },
            )
        self.assertEqual(2, len(client.calls))
        self.assertEqual(1, sum(row["path"] == "/v1/memories/add"
                                for row in state["requests"]))

    def test_existing_parent_memory_skips_parent_add(self):
        client, state = mock_client()
        result = execute_single_round(
            parent=self.parent, memorax=client,
            planner_call=lambda prompt: planner_fixture(self.parent),
            realizer_call=lambda prompt: realizer_fixture(self.parent),
            initialize_parent_memory=False,
        )
        self.assertEqual(3, result["memory_call_count"])
        add_requests = [row for row in state["requests"]
                        if row["path"] == "/v1/memories/add"]
        self.assertEqual(1, len(add_requests))
        self.assertEqual(1, add_requests[0]["payload"]["metadata"]["round_index"])

    def test_artifacts_are_emitted_before_and_after_memory(self):
        client, _ = mock_client()
        artifacts = {}
        execute_single_round(
            parent=self.parent, memorax=client,
            planner_call=lambda prompt: planner_fixture(self.parent),
            realizer_call=lambda prompt: realizer_fixture(self.parent),
            artifact_sink=lambda name, value: artifacts.setdefault(name, value),
        )
        for required in (
                "memory/parent_add_intent",
                "memory/pre_generation_overview",
                "planner/validation",
                "realizer/validation",
                "expanded_dialogue_validation",
                "memory/child_add_intent",
                "memory/post_generation_overview",
                "qa/mcq_prompt"):
            self.assertIn(required, artifacts)

    def test_artifact_sink_splits_mcq_prompt_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "qa").mkdir()
            _artifact_sink(root)("qa/mcq_prompt", {
                "prompt": "fixture prompt",
                "prompt_version": "fixture_v1",
                "prompt_sha256": "fixture_sha",
            })
            prompt_path = root / "qa/mcq_prompt.txt"
            manifest_path = root / "qa/mcq_prompt_manifest.json"
            self.assertEqual("fixture prompt", prompt_path.read_text())
            manifest = json.loads(manifest_path.read_text())
            self.assertNotIn("prompt", manifest)
            self.assertEqual("fixture_sha", manifest["prompt_sha256"])
            self.assertFalse((root / "qa/mcq_prompt.json").exists())

    def test_live_preflight_blocks_plaintext_http_without_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            credentials = root / "credentials.env"
            credentials.write_text(
                "MEMORAX_BASE_URL=http://memorax.example\n"
                "MEMORAX_API_KEY=fixture\n", encoding="utf-8")
            credentials.chmod(0o600)
            report = run_preflight(
                input_path=PARENT_PATH,
                run_root=root / "absent-run",
                api_config_path=API_CONFIG,
                tokenizer_path=TOKENIZER,
                memorax_credentials_path=credentials,
                allow_insecure_memorax_http=False,
            )
            self.assertEqual("fail", report["decision"])
            self.assertIn("memorax_transport_authorized", report["failed_checks"])
            self.assertEqual(0, report["api_calls"])
            self.assertFalse((root / "absent-run").exists())
            self.assertNotIn("fixture", json.dumps(report))

    def test_live_preflight_accepts_explicit_http_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            credentials = root / "credentials.env"
            credentials.write_text(
                "MEMORAX_BASE_URL=http://memorax.example\n"
                "MEMORAX_API_KEY=fixture\n", encoding="utf-8")
            credentials.chmod(0o600)
            report = run_preflight(
                input_path=PARENT_PATH,
                run_root=root / "absent-run",
                api_config_path=API_CONFIG,
                tokenizer_path=TOKENIZER,
                memorax_credentials_path=credentials,
                allow_insecure_memorax_http=True,
            )
            self.assertEqual("pass", report["decision"], report)
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(0, report["api_calls"])
            self.assertFalse((root / "absent-run").exists())

    def test_async_add_polls_processing_until_success(self):
        class Gateway:
            def __init__(self):
                self.statuses = iter(("processing", "processing", "success"))
                self.status_calls = []

            def add(self, payload, *, request_id):
                return {"success": True, "data": {
                    "status": "accepted", "task_id": "task_fixture_001"}}

            def add_status(self, task_id, *, request_id):
                self.status_calls.append((task_id, request_id))
                return {"status": next(self.statuses), "memory": {}, "error": None}

        gateway = Gateway()
        result = submit_add_and_wait(
            gateway=gateway,
            intent={"request_id": "add-fixture", "payload": {"x": 1}},
            poll_delays_seconds=(0, 0, 0), sleep_fn=lambda value: None)
        self.assertTrue(result["verified_complete"])
        self.assertEqual("success", result["final_status"])
        self.assertEqual(3, result["poll_count"])
        self.assertEqual(3, len(gateway.status_calls))

    def test_workflow_searches_only_after_each_async_add_succeeds(self):
        state = {"task_count": 0, "polls": {}, "completed": set(),
                 "search_completion_counts": []}

        def handler(request):
            path = request.url.path
            if path == "/v1/memories/add":
                state["task_count"] += 1
                task = f"task_async_{state['task_count']:02d}"
                state["polls"][task] = 0
                return httpx.Response(202, json={
                    "success": True,
                    "data": {"status": "accepted", "task_id": task},
                })
            if path.startswith("/v1/memories/add/status/"):
                task = path.rsplit("/", 1)[-1]
                state["polls"][task] += 1
                status = "success" if state["polls"][task] == 2 else "processing"
                if status == "success":
                    state["completed"].add(task)
                return httpx.Response(200, json={
                    "status": status, "memory": {}, "error": None})
            if path == "/v1/memory-buckets/search":
                state["search_completion_counts"].append(len(state["completed"]))
                return httpx.Response(200, json=bucket_response(
                    include_child=len(state["completed"]) > 1))
            return httpx.Response(404, json={"success": False})

        client = MemoraxClient(
            MemoraxConfig(base_url="https://memorax.example", api_key="fixture"),
            transport=httpx.MockTransport(handler))
        result = execute_single_round(
            parent=self.parent, memorax=client,
            planner_call=lambda prompt: planner_fixture(self.parent),
            realizer_call=lambda prompt: realizer_fixture(self.parent),
            poll_delays_seconds=(0, 0), sleep_fn=lambda value: None,
        )
        self.assertTrue(result["generation_ready"])
        self.assertEqual([1, 2], state["search_completion_counts"])
        self.assertEqual(8, result["memory_call_count"])
        self.assertEqual(2, result["verified_parent_add"]["poll_count"])
        self.assertEqual(2, result["verified_child_add"]["poll_count"])

    def test_nonblocking_submit_emits_accepted_state_before_any_poll(self):
        class Gateway:
            add_calls = 0

            def add(self, payload, *, request_id):
                self.add_calls += 1
                return {"success": True, "data": {
                    "status": "accepted", "task_id": "task_fixture_saved"}}

            def add_status(self, task_id, *, request_id):
                raise AssertionError("nonblocking submit must not poll")

        states = []
        gateway = Gateway()
        result = submit_add_without_wait(
            gateway=gateway,
            intent={
                "request_id": "add-fixture",
                "payload_sha256": "a" * 64,
                "payload": {"x": 1},
            },
            stage="child_memory_add",
            on_state_change=lambda value: states.append(dict(value)),
        )
        self.assertEqual(1, gateway.add_calls)
        self.assertTrue(result["pending"])
        self.assertEqual("not_submitted", states[0]["status"])
        self.assertEqual("accepted_pending", states[-1]["status"])
        self.assertEqual("task_fixture_saved", states[-1]["task_id"])
        self.assertFalse(states[-1]["automatic_retry_allowed"])

    def test_write_state_requires_confirmed_absence_before_retry(self):
        intent = {
            "request_id": "write-intent-fixture",
            "payload_sha256": "b" * 64,
            "payload": {"x": 1},
        }
        state = new_write_state(intent=intent, stage="child_memory_add")
        state = apply_add_response(state, {"data": {
            "status": "accepted", "task_id": "task_failed_fixture"}})
        state = apply_task_response(state, {"status": "failed"})
        self.assertEqual("ambiguous", state["status"])
        self.assertFalse(retry_allowed(state))
        still_ambiguous = apply_visibility_result(
            state, visible=False, absence_confirmed=False)
        self.assertEqual("ambiguous", still_ambiguous["status"])
        retryable = apply_visibility_result(
            state, visible=False, absence_confirmed=True)
        self.assertEqual("failed_retryable", retryable["status"])
        self.assertTrue(retry_allowed(retryable))

    def test_completed_task_missing_from_bucket_is_visibility_pending(self):
        intent = {
            "request_id": "write-intent-completed",
            "payload_sha256": "c" * 64,
            "payload": {"x": 1},
        }
        state = new_write_state(intent=intent, stage="parent_memory_add")
        state = apply_add_response(state, {"data": {
            "status": "accepted", "task_id": "task_completed_fixture"}})
        state = apply_task_response(state, {"status": "completed"})
        pending = apply_visibility_result(
            state, visible=False, absence_confirmed=True)
        self.assertEqual("completed_visibility_pending", pending["status"])
        self.assertTrue(pending["memory_write_verified"])
        self.assertFalse(pending["bucket_visibility_verified"])
        self.assertFalse(retry_allowed(pending))

    def test_async_add_failed_status_is_terminal_without_resubmit(self):
        class Gateway:
            add_calls = 0

            def add(self, payload, *, request_id):
                self.add_calls += 1
                return {"success": True, "data": {
                    "status": "accepted", "task_id": "task_fixture_002"}}

            def add_status(self, task_id, *, request_id):
                return {"status": "failed", "memory": {}, "error": "fixture"}

        gateway = Gateway()
        with self.assertRaisesRegex(MemoraxAsyncWriteError, "async_add_failed"):
            submit_add_and_wait(
                gateway=gateway,
                intent={"request_id": "add-fixture", "payload": {"x": 1}},
                poll_delays_seconds=(0,), sleep_fn=lambda value: None)
        self.assertEqual(1, gateway.add_calls)

    def test_async_add_poll_exhaustion_is_terminal_without_resubmit(self):
        class Gateway:
            add_calls = 0

            def add(self, payload, *, request_id):
                self.add_calls += 1
                return {"success": True, "data": {
                    "status": "accepted", "task_id": "task_fixture_003"}}

            def add_status(self, task_id, *, request_id):
                return {"status": "processing", "memory": {}, "error": None}

        gateway = Gateway()
        with self.assertRaisesRegex(MemoraxAsyncWriteError, "poll_exhausted:2"):
            submit_add_and_wait(
                gateway=gateway,
                intent={"request_id": "add-fixture", "payload": {"x": 1}},
                poll_delays_seconds=(0, 0), sleep_fn=lambda value: None)
        self.assertEqual(1, gateway.add_calls)

    def test_reuse_preflight_removes_parent_add_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            credentials = root / "credentials.env"
            credentials.write_text(
                "MEMORAX_BASE_URL=http://memorax.example\n"
                "MEMORAX_API_KEY=fixture\n", encoding="utf-8")
            credentials.chmod(0o600)
            report = run_preflight(
                input_path=PARENT_PATH,
                run_root=root / "absent-run",
                api_config_path=API_CONFIG,
                tokenizer_path=TOKENIZER,
                memorax_credentials_path=credentials,
                allow_insecure_memorax_http=True,
                reuse_existing_parent_memory=True,
            )
            self.assertEqual("pass", report["decision"], report)
            self.assertEqual("reuse_existing", report["parent_memory_mode"])
            self.assertEqual(1, report["request_budget"]["memorax_add"])
            self.assertEqual(18, report["request_budget"]["memorax_add_status_max"])

    def test_default_async_poll_window_is_about_five_minutes(self):
        self.assertEqual(18, len(DEFAULT_POLL_DELAYS_SECONDS))
        self.assertGreaterEqual(sum(DEFAULT_POLL_DELAYS_SECONDS), 285)
        self.assertLessEqual(sum(DEFAULT_POLL_DELAYS_SECONDS), 300)

    def test_recovery_preflight_is_read_only_and_requires_valid_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "reports").mkdir(parents=True)
            (source / "reports/SIMPLE_APPEND_MEMORAX_LIVE_REPORT.json").write_text(
                json.dumps({
                    "failure_stage": "child_memory_add",
                    "memorax_call_records": [{
                        "path": (
                            "/v1/memories/add/status/"
                            "b89c0dac-862d-45c9-ad4c-58833fe0a638"),
                    }],
                }), encoding="utf-8")
            (source / "expanded_dialogue.json").write_text(json.dumps({
                "root_id": "fixture-root",
                "qa_fact_source": "complete_parent_plus_child",
            }), encoding="utf-8")
            (source / "child_dialogue.json").write_text(json.dumps({
                "root_id": "fixture-root", "scenes": []}), encoding="utf-8")
            (source / "expanded_dialogue_validation.json").write_text(json.dumps({
                "valid": True, "child_turn_count": 1}), encoding="utf-8")
            output = root / "recovery"
            report = recovery_preflight(source_run=source, output_root=output)
            self.assertEqual("pass", report["decision"], report)
            self.assertEqual(0, report["api_calls"])
            self.assertFalse(output.exists())
            self.assertEqual(
                "b89c0dac-862d-45c9-ad4c-58833fe0a638",
                report["details"]["task_id"])
            self.assertEqual(0, report["details"]["request_budget"]["memorax_add"])

    def test_recovery_preflight_reads_immediately_persisted_task_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "reports").mkdir(parents=True)
            (source / "memory").mkdir()
            (source / "reports/SIMPLE_APPEND_MEMORAX_LIVE_REPORT.json").write_text(
                json.dumps({
                    "status": "completed",
                    "pending_memory_writes": [{
                        "stage": "child_memory_add",
                        "task_id": "task-immediate-fixture",
                    }],
                    "memorax_call_records": [],
                }), encoding="utf-8")
            (source / "memory/child_add_write_state.json").write_text(
                json.dumps({
                    "write_state_version": "simple_append_memorax_write_state_v1",
                    "write_intent_id": "intent-fixture",
                    "attempt_id": "intent-fixture:attempt:01",
                    "attempt_index": 1,
                    "stage": "child_memory_add",
                    "request_id": "intent-fixture",
                    "payload_sha256": "d" * 64,
                    "task_id": "task-immediate-fixture",
                    "status": "accepted_pending",
                    "task_status": "accepted",
                    "add_submitted": True,
                    "memory_write_verified": False,
                    "bucket_visibility_verified": False,
                    "automatic_retry_allowed": False,
                    "last_error": None,
                }), encoding="utf-8")
            (source / "expanded_dialogue.json").write_text(json.dumps({
                "root_id": "fixture-root",
                "qa_fact_source": "complete_parent_plus_child",
            }), encoding="utf-8")
            (source / "child_dialogue.json").write_text(json.dumps({
                "root_id": "fixture-root", "scenes": []}), encoding="utf-8")
            (source / "expanded_dialogue_validation.json").write_text(json.dumps({
                "valid": True, "child_turn_count": 1}), encoding="utf-8")
            output = root / "recovery"
            report = recovery_preflight(source_run=source, output_root=output)
        self.assertEqual("pass", report["decision"], report)
        self.assertEqual("task-immediate-fixture", report["details"]["task_id"])
        self.assertEqual(
            "accepted_pending", report["details"]["write_state"]["status"])
        self.assertEqual(0, report["api_calls"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
