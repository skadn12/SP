"""Bounded B0 M0 pilot for one source-reviewed typed appointment question."""

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
from scripts.current_method.restart_n5_spell.core import atomic_json, sha256_file  # noqa: E402
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


HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "candidate_v3.json"
SOURCE = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json")
EXPECTED_SOURCE_SHA = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"
ORDER_STRINGS = (
    "ACBDE", "EABCD", "EBADC", "CEDAB", "BEDCA",
    "ACEBD", "BDCEA", "DBEAC", "DACBE", "CDAEB",
)
MODES = ("no_context", "full_context")


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def orders() -> list[tuple[str, ...]]:
    rows = [tuple(value) for value in ORDER_STRINGS]
    keys = "ABCDE"
    assert len(rows) == len(set(rows)) == 10
    assert all(set(row) == set(keys) for row in rows)
    assert all(sum(row[position] == key for row in rows) == 2
               for key in keys for position in range(5))
    rotation_orbits = {min("".join(row[shift:] + row[:shift]) for shift in range(5))
                       for row in rows}
    assert len(rotation_orbits) == 10
    return rows


def prepare() -> dict:
    if (HERE / "manifest.json").exists():
        raise RuntimeError("pilot_already_prepared")
    candidate = json.loads(INPUT.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if sha256_file(SOURCE) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("frozen_source_drift")
    if source["root_id"] != candidate["root_id"] or len(source["turns"]) != 188:
        raise RuntimeError("source_identity_mismatch")
    if (candidate["question_type"] != "multi_select" or
            tuple(candidate["options"]) != tuple("ABCDE") or
            candidate["gold"] != ["A", "C"] or
            candidate["frozen_source_sha256"] != EXPECTED_SOURCE_SHA or
            candidate["training_promotion"] is not False):
        raise RuntimeError("candidate_contract_mismatch")
    tokenizer = AutoTokenizer.from_pretrained(BASE_SOLVER_MODEL, local_files_only=True)
    presentation = {key: candidate[key] for key in
                    ("candidate_id", "root_id", "question_type", "question", "options", "gold")}
    dialogue = dialogue_rows(source)
    requests = []
    for number, order in enumerate(orders()):
        mapping = {display: canonical for display, canonical in zip("ABCDE", order)}
        variant = {
            "variant_id": f"perm_{number:02d}",
            "options": {display: candidate["options"][canonical]
                        for display, canonical in mapping.items()},
            "display_to_canonical": mapping,
            "displayed_gold": [display for display, canonical in mapping.items()
                               if canonical in candidate["gold"]],
        }
        for mode in MODES:
            prompt = build_prompt(tokenizer, presentation, variant, dialogue, context_mode=mode)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if prompt_tokens + MAX_TOKENS > MAX_MODEL_LEN:
                raise RuntimeError("context_overflow")
            if mode == "no_context" and source["turns"][0]["text"] in prompt:
                raise RuntimeError("no_context_leakage")
            requests.append({
                "request_id": f"{candidate['candidate_id']}__perm_{number:02d}__{mode}",
                "variant": variant, "mode": mode, "prompt_sha256": sha_text(prompt),
                "prompt_tokens": prompt_tokens, "seed": 20261003 + number * 2 + MODES.index(mode),
            })
    manifest = {
        "status": "PREPARED", "protocol": "B0 local structured Solver prompt/finite-choice/scoring",
        "model": str(BASE_SOLVER_MODEL), "candidate_path": str(INPUT),
        "candidate_sha256": sha256_file(INPUT), "source_path": str(SOURCE),
        "source_sha256": EXPECTED_SOURCE_SHA, "runner_sha256": sha256_file(Path(__file__)),
        "permutation_policy": "10 distinct non-rotation-equivalent displays; each canonical option occupies each position twice",
        "rollouts_per_display_mode": 1, "temperature": 0.8, "top_p": 0.9,
        "max_tokens": MAX_TOKENS, "enable_thinking": False,
        "request_limit": 20, "training_promotion": False, "requests": requests,
    }
    atomic_json(HERE / "manifest.json", manifest)
    return {"status": "PREPARED", "requests": len(requests),
            "max_prompt_tokens": max(row["prompt_tokens"] for row in requests)}


def live() -> dict:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if (HERE / "summary.json").exists() or (HERE / "STARTED.json").exists():
        raise RuntimeError("pilot_already_started")
    for path, expected in ((INPUT, manifest["candidate_sha256"]),
                           (SOURCE, manifest["source_sha256"]),
                           (Path(__file__), manifest["runner_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"input_drift:{path}")
    if len(manifest["requests"]) != 20:
        raise RuntimeError("request_limit_drift")
    assert_gpu_idle()
    ensure_vllm_environment()
    tokenizer = AutoTokenizer.from_pretrained(BASE_SOLVER_MODEL, local_files_only=True)
    candidate = json.loads(INPUT.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    presentation = {key: candidate[key] for key in
                    ("candidate_id", "root_id", "question_type", "question", "options", "gold")}
    dialogue = dialogue_rows(source)
    atomic_json(HERE / "STARTED.json", {"request_limit": 20, "model": str(BASE_SOLVER_MODEL)})
    backend = None
    rows = []
    try:
        backend = VLLMSolverBackend(
            model_path=BASE_SOLVER_MODEL, tokenizer_path=BASE_SOLVER_MODEL,
            max_model_len=MAX_MODEL_LEN, temperature=0.8, top_p=0.9,
            max_tokens=MAX_TOKENS, gpu_memory_utilization=0.80)
        for planned in manifest["requests"]:
            variant, mode = planned["variant"], planned["mode"]
            prompt = build_prompt(tokenizer, presentation, variant, dialogue, context_mode=mode)
            if sha_text(prompt) != planned["prompt_sha256"]:
                raise RuntimeError("prompt_drift")
            request = SolverRequest(
                request_id=planned["request_id"], qa_id=candidate["candidate_id"],
                evaluation_mode=mode, order_id=variant["variant_id"],
                target_slot=0, retry_index=0, sampling_seed=planned["seed"],
                options=variant["options"], canonical_answer_text=",".join(variant["displayed_gold"]),
                prompt_sha256=planned["prompt_sha256"], prompt=prompt,
            )
            response = _structured_generate(backend, request, question_type="multi_select")
            score = score_output(response.raw_text, presentation, variant)
            row = {**planned, "raw_output": response.raw_text,
                   "model_identifier": response.model_identifier,
                   "backend_metadata": response.metadata, "score": score}
            atomic_json(HERE / "responses" / (planned["request_id"] + ".json"), row)
            rows.append(row)
            print(json.dumps({"completed": len(rows), "request_id": planned["request_id"],
                              "correct": score["correct"],
                              "canonical_prediction": score["canonical_prediction"]}), flush=True)
            if response.metadata.get("finish_reason") not in ("stop", "eos") or not score["scored"]:
                raise RuntimeError("unscorable_response_no_retry")
        summary = {"status": "COMPLETE", "candidate_id": candidate["candidate_id"],
                   "question_count": 1, "display_count": 10, "request_count": len(rows),
                   "gold": candidate["gold"], "training_promotion": False,
                   "rates": {mode: {"correct": sum(row["score"]["correct"] for row in rows
                                                    if row["mode"] == mode),
                                    "total": sum(row["mode"] == mode for row in rows)}
                             for mode in MODES},
                   "per_display": {f"perm_{i:02d}": {
                       mode: next(row["score"]["correct"] for row in rows
                                  if row["variant"]["variant_id"] == f"perm_{i:02d}" and row["mode"] == mode)
                       for mode in MODES} for i in range(10)},
                   "source_sha256": manifest["source_sha256"],
                   "candidate_sha256": manifest["candidate_sha256"]}
        atomic_json(HERE / "summary.json", summary)
        return summary
    except BaseException as error:
        atomic_json(HERE / "STOPPED.json", {"completed_requests": len(rows),
                                           "error_type": type(error).__name__,
                                           "error": str(error), "automatic_retry": False})
        raise
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase not in {"prepare", "live"}:
        raise SystemExit("usage: run.py prepare|live")
    print(json.dumps(prepare() if phase == "prepare" else live()), flush=True)
