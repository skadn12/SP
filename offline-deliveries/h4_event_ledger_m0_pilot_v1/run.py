"""One-shot local M0 probe of the four reviewed event-ledger questions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1")
MODEL = Path("/root/autodl-tmp/Qwen3-4B")
REVIEW = DELIVERIES / "h4_event_ledger_qa_review_repeat01"
DEFAULT_SOURCE = DELIVERIES / "h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json"
OUT = HERE
RUN = OUT / "run"
SEED = 20261003
ORDERING_ORDERS = ("ABCD", "DCBA", "ABDC", "CDBA", "BCAD", "DACB", "BDAC", "CADB")

sys.path.insert(0, str(WORKTREE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import contract
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import solver


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refuse_overwrite:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_orders(candidate, protocol):
    field = "items" if candidate["question_type"] == "ordering" else "options"
    keys = list(candidate[field])
    if protocol == "balanced":
        if field == "items":
            if len(keys) != 4:
                raise ValueError("balanced_ordering_requires_four_items")
            return [(f"perm_{index:02d}", [keys[ord(char) - ord("A")] for char in order])
                    for index, order in enumerate(ORDERING_ORDERS)]
        if len(keys) != 5:
            raise ValueError("balanced_multi_select_requires_five_options")
        reverse = list(reversed(keys))
        orders = [keys[shift:] + keys[:shift] for shift in range(5)]
        orders += [reverse[shift:] + reverse[:shift] for shift in range(5)]
        return [(f"perm_{index:02d}", order) for index, order in enumerate(orders)]
    if protocol != "pilot":
        raise ValueError("unknown_protocol")
    permutations = {4: (2, 3, 1, 0), 5: (2, 4, 0, 3, 1),
                    6: (2, 5, 0, 4, 1, 3)}
    if len(keys) not in permutations:
        raise ValueError("pilot_requires_four_to_six_items")
    return [("standard", keys),
            ("scrambled", [keys[index] for index in permutations[len(keys)]])]


def variant(candidate, kind, ordered):
    field = "items" if candidate["question_type"] == "ordering" else "options"
    keys = list(candidate[field])
    assert set(ordered) == set(keys)
    mapping = dict(zip(keys, ordered))
    inverse = {canonical: display for display, canonical in mapping.items()}
    result = {
        "variant_id": kind,
        field: {display: candidate[field][canonical] for display, canonical in mapping.items()},
        "display_to_canonical": mapping,
        "displayed_gold": [inverse[key] for key in candidate["gold"]],
    }
    probe = "<box>" + ",".join(result["displayed_gold"]) + "</box>"
    assert contract.score_output(probe, candidate, result)["correct"] is True
    return result


def inputs():
    questions = read(RUN / "questions.json")["candidates"]
    gold = read(RUN / "gold.json")
    source = read(RUN / "source.json")
    assert questions and source["turns"]
    assert {q["candidate_id"] for q in questions} == set(gold)
    assert len({q["candidate_id"] for q in questions}) == len(questions)
    assert all(q.get("root_id", source.get("root_id")) == source.get("root_id") for q in questions)
    candidates = [{**q, "gold": gold[q["candidate_id"]]} for q in questions]
    dialogue = contract.dialogue_rows(source)
    return candidates, dialogue


def preflight(questions_path: Path, gold_path: Path | None, source_path: Path,
              protocol: str, rollouts: int):
    if RUN.exists():
        raise RuntimeError("run_exists_do_not_overwrite")
    from transformers import AutoTokenizer

    questions_input = read(questions_path)
    questions = questions_input["candidates"] if isinstance(questions_input, dict) else questions_input
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions_must_be_nonempty_list_or_candidates_object")
    if gold_path is None and questions_path.resolve() == (REVIEW / "solver_questions.json").resolve():
        gold_path = REVIEW / "solver_gold.json"
    if gold_path is None:
        if any("gold" not in question for question in questions):
            raise ValueError("gold_required_for_candidates_without_embedded_gold")
        gold = {question["candidate_id"]: question["gold"] for question in questions}
    else:
        gold = read(gold_path)
        if any("gold" in question and question["gold"] != gold.get(question["candidate_id"])
               for question in questions):
            raise ValueError("embedded_and_separate_gold_disagree")
    public_questions = [{key: value for key, value in question.items() if key != "gold"}
                        for question in questions]
    if len({q["candidate_id"] for q in public_questions}) != len(public_questions):
        raise ValueError("duplicate_candidate_id")
    if {q["candidate_id"] for q in public_questions} != set(gold):
        raise ValueError("question_gold_id_mismatch")
    RUN.mkdir(parents=True)
    write(RUN / "questions.json", {"candidates": public_questions})
    write(RUN / "gold.json", gold)
    shutil.copy2(source_path, RUN / "source.json")
    candidates, dialogue = inputs()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    requests = []
    if not 1 <= rollouts <= 5:
        raise ValueError("rollouts_must_be_1_to_5")
    for candidate in candidates:
        for kind, order in display_orders(candidate, protocol):
            display = variant(candidate, kind, order)
            for mode in ("no_context", "full_context"):
                prompt = contract.build_prompt(tokenizer, candidate, display, dialogue, context_mode=mode)
                token_count = len(tokenizer.encode(prompt))
                assert token_count + 32 <= 16384
                for slot in range(rollouts):
                    request_id = f"{candidate['candidate_id']}__{kind}__{mode}__s{slot:02d}"
                    requests.append({"request_id": request_id, "candidate_id": candidate["candidate_id"],
                                     "question_type": candidate["question_type"], "variant": display,
                                     "mode": mode, "slot": slot,
                                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                     "prompt_tokens": token_count, "seed": SEED + len(requests)})
    expected = sum(len(display_orders(candidate, protocol)) for candidate in candidates) * 2 * rollouts
    assert len(requests) == expected
    write(RUN / "manifest.json", {
        "model": str(MODEL), "protocol": "B0 local structured Solver",
        "display_protocol": protocol, "rollouts_per_position": rollouts,
        "temperature": 0.8, "top_p": 0.9, "max_tokens": 32, "enable_thinking": False,
        "candidate_count": len(candidates), "request_count": len(requests),
        "training_promotion": False,
        "input_sha256": {name: sha(RUN / name) for name in
                         ("questions.json", "gold.json", "source.json")},
        "source_paths": {"questions": str(questions_path.resolve()),
                         "gold": str(gold_path.resolve()) if gold_path else "embedded",
                         "source": str(source_path.resolve())},
        "requests": requests,
    })
    return {"status": "PASS", "requests": len(requests),
            "maximum_prompt_tokens": max(row["prompt_tokens"] for row in requests)}


def live():
    from transformers import AutoTokenizer

    manifest = read(RUN / "manifest.json")
    if (RUN / "STARTED.json").exists():
        raise RuntimeError("live_already_started_no_automatic_rerun")
    for name, expected in manifest["input_sha256"].items():
        if sha(RUN / name) != expected:
            raise RuntimeError(f"frozen_input_changed:{name}")
    candidates, dialogue = inputs()
    by_id = {q["candidate_id"]: q for q in candidates}
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    prompts = []
    for row in manifest["requests"]:
        prompt = contract.build_prompt(tokenizer, by_id[row["candidate_id"]], row["variant"],
                                       dialogue, context_mode=row["mode"])
        if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
            raise RuntimeError("prompt_drift:" + row["request_id"])
        prompts.append(prompt)
    write(RUN / "STARTED.json", {"request_count": len(prompts), "started_at": time.time()})
    backend = solver.create_backend(model_path=MODEL, tokenizer_path=MODEL)
    try:
        from scripts.current_method.restart_n5_spell.backends import SolverRequest
        for row, prompt in zip(manifest["requests"], prompts):
            request = SolverRequest(
                request_id=row["request_id"], qa_id=row["candidate_id"],
                evaluation_mode=row["mode"], order_id=row["variant"]["variant_id"],
                target_slot=row["slot"], retry_index=0, sampling_seed=row["seed"],
                options=row["variant"].get("options") or row["variant"].get("items"),
                canonical_answer_text="", prompt_sha256=row["prompt_sha256"], prompt=prompt,
            )
            response = solver._structured_generate(backend, request, question_type=row["question_type"])
            if response.metadata.get("finish_reason") not in ("stop", "eos"):
                raise RuntimeError("incomplete_generation:" + row["request_id"])
            score = contract.score_output(response.raw_text, by_id[row["candidate_id"]], row["variant"])
            write(RUN / "responses" / (row["request_id"] + ".json"), {
                "request_id": row["request_id"], "mode": row["mode"],
                "variant_id": row["variant"]["variant_id"], "seed": row["seed"],
                "prompt_sha256": row["prompt_sha256"], "raw_output": response.raw_text,
                "model_identifier": response.model_identifier, "metadata": response.metadata,
                "displayed_gold": row["variant"]["displayed_gold"], "score": score,
            })
            print(json.dumps({"request_id": row["request_id"], "correct": score["correct"],
                              "raw_output": response.raw_text}), flush=True)
    finally:
        backend.close()
    return report()


def report():
    manifest = read(RUN / "manifest.json")
    rows = []
    for request in manifest["requests"]:
        path = RUN / "responses" / (request["request_id"] + ".json")
        if not path.exists():
            raise RuntimeError("missing_response:" + request["request_id"])
        rows.append(read(path))
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["request_id"].split("__")[0]][row["variant_id"], row["mode"], row["seed"]] = row
    lines = ["# M0 question probe", "", "Local Qwen3-4B base; B0 structured prompt and exact scoring. "
             "Repeated positions and samples are correlated observations of the same question, "
             "not independent questions or admission decisions.", "",
             "| Question | No context | Full context |", "|---|---:|---:|"]
    summary = {"status": "COMPLETE", "model": str(MODEL), "request_count": len(rows),
               "training_promotion": False, "questions": {},
               "input_tokens": sum(r["metadata"]["prompt_token_count"] for r in rows),
               "output_tokens": sum(r["metadata"]["completion_token_count"] for r in rows)}
    for cid, cells in sorted(grouped.items()):
        scores = {}
        for mode in ("no_context", "full_context"):
            selected = [row for (variant_id, m, seed), row in cells.items() if m == mode]
            scores[mode] = {"correct": sum(row["score"]["correct"] is True for row in selected),
                            "total": len(selected),
                            "parsed": sum(row["score"]["scored"] is True for row in selected)}
        summary["questions"][cid] = {"scores": scores, "positions": {
            variant_id: {mode: {
                "correct": sum(row["score"]["correct"] is True for (v, m, _), row in cells.items()
                               if v == variant_id and m == mode),
                "total": sum(v == variant_id and m == mode for v, m, _ in cells)}
                for mode in ("no_context", "full_context")}
            for variant_id in sorted({key[0] for key in cells})},
            "responses": {f"{variant_id}_{mode}_{seed}": {
                "raw_output": row["raw_output"],
                "canonical_prediction": row["score"]["canonical_prediction"],
                "correct": row["score"]["correct"]}
                for (variant_id, mode, seed), row in cells.items()}}
        lines.append(f"| {cid} | {scores['no_context']['correct']}/{scores['no_context']['total']} "
                     f"| {scores['full_context']['correct']}/{scores['full_context']['total']} |")
    lines += ["", "Per-position scores, parse counts, raw outputs and predictions are in "
              "summary.json; individual metadata is in responses/.", ""]
    write(OUT / "summary.json", summary)
    path = OUT / "report.md"
    if path.exists():
        raise RuntimeError("refuse_overwrite:" + str(path))
    path.write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "live", "report"))
    parser.add_argument("--questions", type=Path, default=REVIEW / "solver_questions.json")
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--protocol", choices=("pilot", "balanced"), default="pilot")
    parser.add_argument("--rollouts", type=int, default=1)
    args = parser.parse_args()
    OUT = args.out.resolve()
    RUN = OUT / "run"
    action = {"preflight": lambda: preflight(args.questions, args.gold, args.source,
                                             args.protocol, args.rollouts),
              "live": live, "report": report}[args.phase]
    print(json.dumps(action(),
                     ensure_ascii=False), flush=True)
