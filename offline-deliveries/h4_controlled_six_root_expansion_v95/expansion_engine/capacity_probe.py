"""Offline exact-preview capacity and conservative post-expansion solver estimate.

Does not authorize truncation. All real expansion requests and eventual actual
solver prompts must pass their existing exact capacity gates independently.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner as r


def check(run):
    run = Path(run).resolve()
    manifest, _ = r.verify(run)
    profile, _factory, annotate, capacity, dev = r.base.transport()
    tokenizer = dev.tokenizer_from(str(r.base.TOKENIZER))
    previews = []
    for path in sorted((run / "previews").rglob("*.json")):
        row = r.base.read(path)
        proof = annotate(capacity(profile(), tokenizer, row["payload"]["messages"], 4096))
        previews.append({"path": str(path), "capacity": proof})
    r.base.require(all(row["capacity"]["status"] == "PASS" for row in previews),
                   "offline_expansion_preview_capacity_blocked")
    from transformers import AutoTokenizer
    local_tokenizer_path = Path("/root/autodl-tmp/Qwen3-4B")
    local = AutoTokenizer.from_pretrained(str(local_tokenizer_path), local_files_only=True)
    native_path = r.base.PROJECT / "offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation"
    sys.path.insert(0, str(native_path))
    spec = importlib.util.spec_from_file_location("v90_offline_solver_capacity_native", native_path / "run_solver.py")
    n = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n)
    probe = n.load_probe("v90_offline_solver_capacity_contract")
    solver_rows = []
    for root in manifest["roots"]:
        source = r.base.read(run / "sources" / (root + ".json"))
        rows = probe.contract.dialogue_rows(source)
        for kind in ("multi_select", "ordering"):
            field = "options" if kind == "multi_select" else "items"
            labels = tuple("ABCDEF") if kind == "multi_select" else tuple("I" + str(i) for i in range(1, 7))
            question = {"question_type": kind, "question": "OFFLINE CAPACITY PLACEHOLDER"}
            variant = {field: {label: "OFFLINE CAPACITY PLACEHOLDER" for label in labels}}
            prompt = probe.contract.build_prompt(local, question, variant, rows, context_mode="full_context")
            current = len(local.encode(prompt, add_special_tokens=False))
            # Estimates, not silent new QA restrictions: actual QA is unknown.
            suffix_reserve = 4096
            qa_reserve = 2048
            estimated = current + suffix_reserve + qa_reserve
            solver_rows.append({"root_id": root, "question_type": kind,
                "source_turns": len(source["turns"]), "complete_baseline_prompt_tokens": current,
                "new_scene_estimate_token_reserve": suffix_reserve,
                "question_and_six_options_estimate_token_reserve": qa_reserve,
                "estimated_post_expansion_prompt_tokens": estimated,
                "model_context_tokens": 16384, "max_output_tokens": 32,
                "estimated_margin_tokens": 16384 - estimated - 32,
                "estimate_status": "PASS" if estimated + 32 <= 16384 else "HOLD_ESTIMATE",
                "actual_solver_prompt_requires_separate_exact_preflight": True})
    result = {"status": "PASS" if all(x["estimate_status"] == "PASS" for x in solver_rows) else "HOLD_ESTIMATE",
        "network_requests": 0, "historical_prefix_truncation": False,
        "expansion_exact_preview_capacity": previews,
        "local4b_tokenizer_path": str(local_tokenizer_path), "solver_estimates": solver_rows,
        "estimate_is_not_actual_post_generation_capacity_proof": True,
        "actual_dynamic_expansion_requests_checked_by_unchanged_Caller": True}
    r.base.write(run / "OFFLINE_CAPACITY_CHECKS.json", result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(check(parser.parse_args().run), ensure_ascii=False, indent=2))
