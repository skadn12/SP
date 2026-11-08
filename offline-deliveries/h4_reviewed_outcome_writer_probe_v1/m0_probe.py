"""Position-balanced local M0 source-dependence probe for 96462-Q3."""

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
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.run import assert_gpu_idle  # noqa: E402
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.solver import (  # noqa: E402
    _structured_generate, ensure_vllm_environment,
)
from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend  # noqa: E402

from run import HERE, SOURCE_DIR, SOURCES, write_new

ORDERS = (
    "ACBD", "BDAC", "CADB", "DBCA",
    "ADCB", "BCDA", "CBAD", "DABC",
)
MODES = ("no_context", "full_context")


def main() -> None:
    candidate_path = HERE / "candidate_96462_q3.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    source = SOURCE_DIR / "cc-train-episode-96462.json"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != SOURCES["96462"] or candidate["frozen_source_sha256"] != source_sha:
        raise RuntimeError("frozen source mismatch")
    for label in "ABCD":
        positions = [order.index(label) for order in ORDERS]
        if sorted(positions) != [0, 0, 1, 1, 2, 2, 3, 3]:
            raise RuntimeError(f"unbalanced label positions: {label}")
    if len(set(ORDERS)) != len(ORDERS):
        raise RuntimeError("duplicate display")
    assert_gpu_idle()
    ensure_vllm_environment()
    tokenizer = AutoTokenizer.from_pretrained(BASE_SOLVER_MODEL, local_files_only=True)
    dialogue = dialogue_rows(json.loads(source.read_text(encoding="utf-8")))
    planned = []
    for index, order in enumerate(ORDERS):
        mapping = dict(zip(("I1", "I2", "I3", "I4"), (f"I{ord(x) - ord('A') + 1}" for x in order)))
        variant = {
            "variant_id": f"perm_{index:02d}",
            "items": {display: candidate["items"][canonical] for display, canonical in mapping.items()},
            "display_to_canonical": mapping,
            "displayed_gold": [display for canonical in candidate["gold"] for display, original in mapping.items() if original == canonical],
        }
        for mode in MODES:
            prompt = build_prompt(tokenizer, candidate, variant, dialogue, context_mode=mode).replace(
                "Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.",
                "Output only ordered item IDs separated by commas inside one <box>...</box> pair.",
            )
            token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            if token_count + MAX_TOKENS > MAX_MODEL_LEN:
                raise RuntimeError("context overflow")
            planned.append((variant, mode, prompt, token_count))
    write_new(HERE / "m0_manifest.json", {
        "candidate_path": str(candidate_path),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "source_path": str(source), "source_sha256": source_sha,
        "model": str(BASE_SOLVER_MODEL), "orders": ORDERS, "modes": MODES,
        "request_count": len(planned),
        "position_balance": "each canonical event appears twice at every display position; IDs are relabeled to displayed text",
        "sampling": {"temperature": 0.8, "top_p": 0.9, "max_tokens": MAX_TOKENS},
        "prompt_note": "B0 structured prompt with example answer removed",
        "requests": [{"variant": v["variant_id"], "mode": m,
                      "prompt_sha256": hashlib.sha256(p.encode()).hexdigest(),
                      "prompt_tokens": n} for v, m, p, n in planned],
        "training_promotion": False,
    })
    backend = VLLMSolverBackend(
        model_path=BASE_SOLVER_MODEL, tokenizer_path=BASE_SOLVER_MODEL,
        max_model_len=MAX_MODEL_LEN, temperature=0.8, top_p=0.9,
        max_tokens=MAX_TOKENS, gpu_memory_utilization=0.80,
    )
    results = []
    try:
        for index, (variant, mode, prompt, token_count) in enumerate(planned):
            request = SolverRequest(
                request_id=f"{candidate['candidate_id']}__{variant['variant_id']}__{mode}",
                qa_id=candidate["candidate_id"], evaluation_mode=mode,
                order_id=variant["variant_id"], target_slot=0, retry_index=0,
                sampling_seed=20261003 + index, options=variant["items"],
                canonical_answer_text=",".join(variant["displayed_gold"]),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), prompt=prompt,
            )
            response = _structured_generate(backend, request, question_type="ordering")
            row = {"request_id": request.request_id, "mode": mode,
                   "variant": variant, "prompt": prompt, "prompt_tokens": token_count,
                   "raw_output": response.raw_text, "backend_metadata": response.metadata,
                   "score": score_output(response.raw_text, candidate, variant),
                   "training_promotion": False}
            write_new(HERE / f"m0_response_{index:02d}.json", row)
            results.append({"mode": mode, "variant": variant["variant_id"], "score": row["score"]})
            print(json.dumps({"done": index + 1, "mode": mode, "correct": row["score"]["correct"]}), flush=True)
        write_new(HERE / "m0_summary.json", {
            "status": "COMPLETE", "request_count": len(results),
            "rates": {mode: {"correct": sum(bool(r["score"]["correct"]) for r in results if r["mode"] == mode),
                             "total": sum(r["mode"] == mode for r in results)} for mode in MODES},
            "results": results, "training_promotion": False,
        })
    finally:
        backend.close()


if __name__ == "__main__":
    main()
