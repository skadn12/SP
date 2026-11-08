"""Offline-only preflight; file/main guard is required by spawn-based tests."""
import json
import unittest
from pathlib import Path


def main():
    from scripts.current_method import base_qa_capability_probe_v1 as p
    from scripts.current_method.base_qa_executor_v2 import LocalTokenizerCounter
    modules = ["scripts.current_method.test_base_qa_capability_probe_v1",
        "scripts.current_method.test_base_qa_executor_v2", "scripts.current_method.test_qa_api_budget_v2",
        "scripts.current_method.test_qa_api_transport_v2", "scripts.current_method.test_base_qa_executor_v2_adversarial",
        "scripts.current_method.test_base_qa_pipeline_v1", "scripts.current_method.test_question_presentation_gate",
        "scripts.current_method.test_production_question_gate"]
    with (p.ARTIFACTS / "OFFLINE_TEST_OUTPUT_FINAL.txt").open("x") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    assert result.wasSuccessful()
    events_before_tokenizer = (p.ARTIFACTS / "OFFLINE_NETWORK_EVENTS.jsonl").read_text().splitlines()
    directory = Path("/root/autodl-tmp/Qwen3-4B")
    try:
        counter = LocalTokenizerCounter(directory)
        proxy = {"available": True, "local_prompt_proxy_tokens": counter(p.body()),
            "local_visible_OK_tokens": len(counter.tokenizer.encode("OK", add_special_tokens=False)),
            "method": "AutoTokenizer local_files_only=True trust_remote_code=False apply_chat_template tokenize=True add_generation_prompt=True",
            "tokenizer_directory": str(directory), "model_weights_loaded": False, "remote_compatibility_verified": False,
            "limitation": "Local Qwen3-4B tokenizer proxy; remote qwen3.6-35b-a3b may use a different template/tokenizer or hidden tokens.",
            "file_bindings": [p.file_binding(directory / name) for name in
                ("tokenizer.json", "tokenizer_config.json", "config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")
                if (directory / name).exists()]}
    except Exception as error:
        proxy = {"available": False, "error_type": type(error).__name__, "remote_compatibility_verified": False}
    p.durable_write(p.ARTIFACTS / "INPUT_TOKEN_PROXY.json", proxy)
    events_after_tokenizer = (p.ARTIFACTS / "OFFLINE_NETWORK_EVENTS.jsonl").read_text().splitlines()
    p.durable_write(p.ARTIFACTS / "OFFLINE_TEST_RESULT.json", {"tests_run": result.testsRun,
        "passed": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
        "blocked_socket_events_before_tokenizer": len(events_before_tokenizer),
        "blocked_socket_events_after_tokenizer": len(events_after_tokenizer),
        "real_api_requests": 0, "modules": modules,
        "earlier_stdin_harness": "Interrupted before completion because spawn cannot import <stdin>; final check uses this guarded file entry"})
    print(json.dumps({"tests_run": result.testsRun, "passed": result.wasSuccessful(),
                      "blocked_socket_events": len(events_after_tokenizer), "real_api_requests": 0,
                      "tokenizer_proxy": proxy}, ensure_ascii=False))


if __name__ == "__main__":
    main()
