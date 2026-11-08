"""Auditable six-display M0 source-dependence probe for an accepted QA item.

Default is a local dry run. A live run requires the reviewed question's SHA-256
on the command line; neither the answer nor review metadata enters the prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
SOURCE = Path(
    "/root/autodl-tmp/gxy-selfplay/offline-deliveries/"
    "h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json"
)
SOURCE_SHA256 = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"
MODES = ("no_context", "full_context")
SEED = 20261004


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> tuple[Any, str]:
    data = path.read_bytes()
    return json.loads(data), sha256(data)


def write_once(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"existing_artifact_differs:{path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def normalized_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    qtype = record.get("question_type")
    if qtype not in {"multi_select", "ordering"}:
        raise ValueError("question_type_must_be_multi_select_or_ordering")
    field = "items" if qtype == "ordering" else "options"
    raw = record.get(field)
    if not isinstance(raw, dict) or not 4 <= len(raw) <= 6:
        raise ValueError(f"{field}_must_have_4_to_6_entries")
    ids = ([f"I{i + 1}" for i in range(len(raw))] if qtype == "ordering"
           else [chr(ord("A") + i) for i in range(len(raw))])
    if list(raw) != ids or any(not isinstance(raw[key], str) or not raw[key].strip() for key in ids):
        raise ValueError(f"{field}_ids_or_text_invalid")
    if len(set(raw.values())) != len(raw):
        raise ValueError(f"{field}_texts_not_unique")
    question = record.get("question", record.get("stem"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question_or_stem_missing")
    gold = record.get("gold")
    if not isinstance(gold, list) or any(key not in ids for key in gold) or len(set(gold)) != len(gold):
        raise ValueError("gold_invalid")
    if qtype == "ordering" and len(gold) != len(ids):
        raise ValueError("ordering_gold_must_be_full_permutation")
    if qtype == "multi_select" and not 1 <= len(gold) < len(ids):
        raise ValueError("multi_select_gold_must_be_proper_nonempty_subset")
    candidate_id = record.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id_missing")
    return {
        "candidate_id": candidate_id, "question_type": qtype,
        "question": question, field: raw, "gold": gold,
    }


def _turns(container: Any) -> list[dict[str, str]]:
    if isinstance(container, list):
        rows = container
    elif isinstance(container, dict) and isinstance(container.get("turns"), list):
        rows = container["turns"]
    elif isinstance(container, dict) and isinstance(container.get("scenes"), list):
        rows = []
        for scene in container["scenes"]:
            if not isinstance(scene, dict) or not isinstance(scene.get("turns"), list):
                raise ValueError("scene_turns_missing")
            rows.extend(scene["turns"])
    else:
        raise ValueError("dialogue_requires_turns_or_scenes")
    turns = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("turn_must_be_object")
        ref = row.get("turn_ref", row.get("turn_id"))
        speaker, text = row.get("speaker"), row.get("text")
        if not all(isinstance(value, str) and value.strip() for value in (ref, speaker, text)):
            raise ValueError("turn_ref_speaker_text_required")
        turns.append({
            "turn_id": ref, "session_id": str(row.get("session_ref") or row.get("session_id") or ""),
            "speaker": speaker, "text": text,
        })
    return turns


def combined_dialogue(historical: Any, realized: Any) -> list[dict[str, str]]:
    old, new = _turns(historical), _turns(realized)
    if not new:
        raise ValueError("realized_dialogue_has_no_turns")
    old_by_ref = {row["turn_id"]: row for row in old}
    if len(old_by_ref) != len(old):
        raise ValueError("historical_turn_ref_duplicate")
    seen = set(old_by_ref)
    appended = []
    for row in new:
        ref = row["turn_id"]
        if ref in seen:
            if ref not in old_by_ref or old_by_ref[ref] != row:
                raise ValueError(f"duplicate_or_changed_turn_ref:{ref}")
            continue
        seen.add(ref)
        appended.append(row)
    if not appended:
        raise ValueError("realized_dialogue_has_no_new_turns")
    return old + appended


def display_orders(ids: list[str], seed: int = SEED) -> list[tuple[str, ...]]:
    """Find distinct randomized displays with a balanced non-rotational core."""
    rng = random.Random(seed)
    n = len(ids)
    pool = list(itertools.permutations(ids))
    rng.shuffle(pool)
    first = pool[0]
    rotations = {first[i:] + first[:i] for i in range(n)}
    candidates = [row for row in pool if row not in rotations]
    core = [first]
    occupied = [set([first[pos]]) for pos in range(n)]

    def extend() -> bool:
        if len(core) == n:
            return True
        eligible = [row for row in candidates if all(row[pos] not in occupied[pos] for pos in range(n))]
        rng.shuffle(eligible)
        for row in eligible:
            core.append(row)
            for pos, key in enumerate(row):
                occupied[pos].add(key)
            if extend():
                return True
            core.pop()
            for pos, key in enumerate(row):
                occupied[pos].remove(key)
        return False

    if not extend():
        raise RuntimeError("could_not_construct_balanced_nonrotational_displays")
    while len(core) < 6:
        choices = [row for row in candidates if row not in core]
        counts = {(key, pos): sum(row[pos] == key for row in core)
                  for key in ids for pos in range(n)}
        best = min(sum((counts[(key, pos)] + (row[pos] == key)) ** 2
                       for key in ids for pos in range(n)) for row in choices)
        core.append(rng.choice([
            row for row in choices
            if sum((counts[(key, pos)] + (row[pos] == key)) ** 2
                   for key in ids for pos in range(n)) == best
        ]))
    rng.shuffle(core)
    return core


def variants(candidate: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    qtype = candidate["question_type"]
    field = "items" if qtype == "ordering" else "options"
    ids = list(candidate[field])
    orders = display_orders(ids)
    rows = []
    counts = {key: {display: 0 for display in ids} for key in ids}
    for index, order in enumerate(orders):
        mapping = dict(zip(ids, order))
        reverse = {canonical: display for display, canonical in mapping.items()}
        rows.append({
            "variant_id": f"perm_{index:02d}",
            field: {display: candidate[field][canonical] for display, canonical in mapping.items()},
            "display_to_canonical": mapping,
            "displayed_gold": [reverse[canonical] for canonical in candidate["gold"]],
        })
        for display, canonical in mapping.items():
            counts[canonical][display] += 1
    if len({tuple(row["display_to_canonical"].values()) for row in rows}) != 6:
        raise RuntimeError("display_orders_not_distinct")
    if any(max(positions.values()) - min(positions.values()) > 1 for positions in counts.values()):
        raise RuntimeError("display_positions_unbalanced")
    return rows, counts


def plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if str(WORKTREE) not in sys.path:
        sys.path.insert(0, str(WORKTREE))
    from transformers import AutoTokenizer
    from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import (
        BASE_SOLVER_MODEL, MAX_MODEL_LEN, MAX_TOKENS,
    )
    from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.contract import build_prompt

    raw_question, question_sha = read_json(args.question)
    candidate = normalized_candidate(raw_question)
    historical, source_sha = read_json(args.historical_source)
    if args.historical_source.resolve() == SOURCE.resolve() and source_sha != SOURCE_SHA256:
        raise RuntimeError("frozen_R03_source_mismatch")
    realized, realized_sha = read_json(args.dialogue)
    dialogue = combined_dialogue(historical, realized)
    rows, counts = variants(candidate)
    tokenizer = AutoTokenizer.from_pretrained(BASE_SOLVER_MODEL, local_files_only=True)
    planned = []
    # build_prompt reads only the public question fields; scoring keeps gold private.
    public_question = {key: candidate[key] for key in ("question_type", "question")}
    for variant in rows:
        field = "items" if candidate["question_type"] == "ordering" else "options"
        public_variant = {field: variant[field]}
        for mode in MODES:
            prompt = build_prompt(tokenizer, public_question, public_variant, dialogue, context_mode=mode)
            prompt = prompt.replace(
                "Output only selected letters separated by commas, e.g. <box>A,C</box>.",
                "Output only selected letters separated by commas inside one <box>...</box> pair.",
            ).replace(
                "Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.",
                "Output only ordered item IDs separated by commas inside one <box>...</box> pair.",
            )
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if tokens + MAX_TOKENS > MAX_MODEL_LEN:
                raise RuntimeError(f"context_overflow:{mode}:{tokens}")
            planned.append({
                "variant": variant, "mode": mode, "prompt": prompt,
                "prompt_sha256": sha256(prompt.encode("utf-8")), "prompt_tokens": tokens,
            })
    manifest = {
        "status": "DRY_RUN", "question_path": str(args.question.resolve()),
        "question_sha256": question_sha,
        "historical_source_path": str(args.historical_source.resolve()),
        "historical_source_sha256": source_sha,
        "realized_dialogue_path": str(args.dialogue.resolve()),
        "realized_dialogue_sha256": realized_sha,
        "combined_turn_count": len(dialogue),
        "model": str(BASE_SOLVER_MODEL), "question_type": candidate["question_type"],
        "modes": MODES, "display_count": len(rows), "request_count": len(planned),
        "seed": SEED, "sampling": {"temperature": 0.8, "top_p": 0.9, "max_tokens": MAX_TOKENS},
        "position_counts": counts,
        "position_balance_note": "Six distinct randomized permutations; each canonical item appears at each display position once or twice when item count is below six.",
        "requests": [{
            "request_id": f"{candidate['candidate_id']}__{row['variant']['variant_id']}__{row['mode']}",
            "variant_id": row["variant"]["variant_id"], "mode": row["mode"],
            "display_to_canonical": row["variant"]["display_to_canonical"],
            "prompt_sha256": row["prompt_sha256"], "prompt_tokens": row["prompt_tokens"],
        } for row in planned],
        "training_promotion": False,
    }
    return manifest, planned, candidate


def run_live(args: argparse.Namespace, manifest: dict[str, Any], planned: list[dict[str, Any]], candidate: dict[str, Any]) -> None:
    if args.accepted_sha256 != manifest["question_sha256"]:
        raise RuntimeError("live_requires_exact_reviewed_question_sha256")
    if str(WORKTREE) not in sys.path:
        sys.path.insert(0, str(WORKTREE))
    from scripts.current_method.restart_n5_spell.backends import SolverRequest
    from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import (
        BASE_SOLVER_MODEL, MAX_MODEL_LEN, MAX_TOKENS,
    )
    from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.contract import score_output
    from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.run import assert_gpu_idle
    from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.solver import (
        _structured_generate, ensure_vllm_environment,
    )
    from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend

    response_paths = [args.output_dir / f"solver_probe_response_{i:02d}.json" for i in range(len(planned))]
    if any(path.exists() for path in response_paths) or (args.output_dir / "solver_probe_summary.json").exists():
        raise RuntimeError("live_artifacts_exist; refusing_to_overwrite_or_mix_runs")
    assert_gpu_idle()
    ensure_vllm_environment()
    backend = VLLMSolverBackend(
        model_path=BASE_SOLVER_MODEL, tokenizer_path=BASE_SOLVER_MODEL,
        max_model_len=MAX_MODEL_LEN, temperature=0.8, top_p=0.9,
        max_tokens=MAX_TOKENS, gpu_memory_utilization=0.80,
    )
    results = []
    field = "items" if candidate["question_type"] == "ordering" else "options"
    try:
        for index, row in enumerate(planned):
            variant = row["variant"]
            request = SolverRequest(
                request_id=manifest["requests"][index]["request_id"],
                qa_id=candidate["candidate_id"], evaluation_mode=row["mode"],
                order_id=variant["variant_id"], target_slot=0, retry_index=0,
                sampling_seed=SEED + index, options=variant[field],
                canonical_answer_text="", prompt_sha256=row["prompt_sha256"],
                prompt=row["prompt"],
            )
            response = _structured_generate(backend, request, question_type=candidate["question_type"])
            score = score_output(response.raw_text, candidate, variant)
            artifact = {
                "request_id": request.request_id, "mode": row["mode"],
                "variant_id": variant["variant_id"], "display_to_canonical": variant["display_to_canonical"],
                "question_sha256": manifest["question_sha256"],
                "historical_source_sha256": manifest["historical_source_sha256"],
                "realized_dialogue_sha256": manifest["realized_dialogue_sha256"],
                "prompt": row["prompt"], "prompt_sha256": row["prompt_sha256"],
                "prompt_tokens": row["prompt_tokens"], "raw_output": response.raw_text,
                "model": response.model_identifier, "usage_and_backend_metadata": response.metadata,
                "score": score, "training_promotion": False,
            }
            write_once(response_paths[index], artifact)
            results.append({
                "request_id": request.request_id, "mode": row["mode"],
                "variant_id": variant["variant_id"], "score": score,
            })
            print(json.dumps({"done": index + 1, "mode": row["mode"], "correct": score["correct"]}), flush=True)
    finally:
        backend.close()
    write_once(args.output_dir / "solver_probe_summary.json", {
        "status": "COMPLETE", "request_count": len(results),
        "question_sha256": manifest["question_sha256"],
        "historical_source_sha256": manifest["historical_source_sha256"],
        "realized_dialogue_sha256": manifest["realized_dialogue_sha256"],
        "rates": {mode: {
            "correct": sum(row["mode"] == mode and row["score"]["correct"] is True for row in results),
            "parse_failures": sum(row["mode"] == mode and not row["score"]["scored"] for row in results),
            "total": sum(row["mode"] == mode for row in results),
        } for mode in MODES},
        "results": results, "training_promotion": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", type=Path, default=HERE / "accepted_question.json")
    parser.add_argument("--dialogue", type=Path, default=HERE / "realized_dialogue.json")
    parser.add_argument("--historical-source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--accepted-sha256", default="")
    args = parser.parse_args()
    manifest, planned, candidate = plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_once(args.output_dir / "solver_probe_manifest.json", manifest)
    if args.live:
        run_live(args, manifest, planned, candidate)
    else:
        print(json.dumps({"status": "DRY_RUN", "request_count": len(planned),
                          "question_sha256": manifest["question_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
