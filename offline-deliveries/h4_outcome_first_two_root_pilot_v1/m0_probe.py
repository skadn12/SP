"""Eight local M0 responses for one manually source-reviewed candidate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
sys.path.insert(0, str(WORKTREE))
from transformers import AutoTokenizer  # noqa: E402
from scripts.current_method.restart_n5_spell.backends import SolverRequest  # noqa: E402
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import (  # noqa: E402
    BASE_SOLVER_MODEL, MAX_MODEL_LEN, MAX_TOKENS,
)
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.contract import (  # noqa: E402
    build_prompt, dialogue_rows, score_output,
)
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.run import (  # noqa: E402
    assert_gpu_idle,
)
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.solver import (  # noqa: E402
    _structured_generate, ensure_vllm_environment,
)
from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend  # noqa: E402

from run import HERE, SOURCES, write_new

ORDERS = ("ACBDE", "EABCD", "EBADC", "CEDAB")
MODES = ("no_context", "full_context")


def make_prompt(tokenizer, candidate, variant, dialogue, mode):
    standard = build_prompt(tokenizer, candidate, variant, dialogue, context_mode=mode)
    return standard.replace(
        "Output only selected letters separated by commas, e.g. <box>A,C</box>.",
        "Output only selected letters separated by commas inside one <box>...</box> pair.",
    )


def main(version: str = "v1") -> None:
    if version not in {"v1", "v2"}:
        raise RuntimeError("version must be v1 or v2")
    candidate_path = HERE / ("candidate_112812_v2.json" if version == "v2" else "candidate_112812.json")
    output_prefix = "m0_v2" if version == "v2" else "m0"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    source_path, expected_sha = SOURCES["112812"]
    actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha or candidate["frozen_source_sha256"] != actual_sha:
        raise RuntimeError("source mismatch")
    assert_gpu_idle()
    ensure_vllm_environment()
    tokenizer = AutoTokenizer.from_pretrained(BASE_SOLVER_MODEL, local_files_only=True)
    dialogue = dialogue_rows(json.loads(source_path.read_text(encoding="utf-8")))
    planned = []
    for i, order in enumerate(ORDERS):
        mapping = dict(zip("ABCDE", order))
        variant = {
            "variant_id": f"perm_{i:02d}",
            "options": {d: candidate["options"][c] for d, c in mapping.items()},
            "display_to_canonical": mapping,
            "displayed_gold": [d for d, c in mapping.items() if c in candidate["gold"]],
        }
        for mode in MODES:
            prompt = make_prompt(tokenizer, candidate, variant, dialogue, mode)
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if tokens + MAX_TOKENS > MAX_MODEL_LEN:
                raise RuntimeError("context overflow")
            planned.append((variant, mode, prompt, tokens))
    manifest = {
        "candidate_path": str(candidate_path),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "source_path": str(source_path), "source_sha256": actual_sha,
        "model": str(BASE_SOLVER_MODEL), "request_count": len(planned),
        "orders": ORDERS, "modes": MODES,
        "sampling": {"temperature": 0.8, "top_p": 0.9, "max_tokens": MAX_TOKENS},
        "prompt_note": "B0 structured prompt with answer-example removed to avoid A,C priming",
        "limitation": "Four displays are nonrotation-equivalent but not position-balanced; one sample per mode/display",
        "training_promotion": False,
        "requests": [{"variant": v["variant_id"], "mode": m,
                      "prompt_sha256": hashlib.sha256(p.encode()).hexdigest(),
                      "prompt_tokens": n} for v, m, p, n in planned],
    }
    write_new(HERE / f"{output_prefix}_manifest.json", manifest)
    write_new(HERE / f"{output_prefix}_started.json", {"request_limit": len(planned)})
    backend = VLLMSolverBackend(
        model_path=BASE_SOLVER_MODEL, tokenizer_path=BASE_SOLVER_MODEL,
        max_model_len=MAX_MODEL_LEN, temperature=0.8, top_p=0.9,
        max_tokens=MAX_TOKENS, gpu_memory_utilization=0.80,
    )
    results = []
    try:
        for i, (variant, mode, prompt, tokens) in enumerate(planned):
            request = SolverRequest(
                request_id=f"{candidate['candidate_id']}__{variant['variant_id']}__{mode}",
                qa_id=candidate["candidate_id"], evaluation_mode=mode,
                order_id=variant["variant_id"], target_slot=0, retry_index=0,
                sampling_seed=20261003 + i, options=variant["options"],
                canonical_answer_text=",".join(variant["displayed_gold"]),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), prompt=prompt,
            )
            response = _structured_generate(backend, request, question_type="multi_select")
            row = {"request_id": request.request_id, "mode": mode,
                   "variant": variant, "prompt": prompt, "prompt_tokens": tokens,
                   "raw_output": response.raw_text, "backend_metadata": response.metadata,
                   "score": score_output(response.raw_text, candidate, variant),
                   "training_promotion": False}
            write_new(HERE / f"{output_prefix}_response_{i:02d}.json", row)
            results.append({"mode": mode, "variant": variant["variant_id"],
                            "score": row["score"]})
            print(json.dumps({"done": i + 1, "mode": mode, "correct": row["score"]["correct"]}), flush=True)
        write_new(HERE / f"{output_prefix}_summary.json", {
            "status": "COMPLETE", "request_count": len(results),
            "rates": {mode: {"correct": sum(bool(r["score"]["correct"]) for r in results if r["mode"] == mode),
                             "total": sum(r["mode"] == mode for r in results)} for mode in MODES},
            "results": results, "training_promotion": False,
        })
    finally:
        backend.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
