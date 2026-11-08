"""Replay Q1 full-context variants with greedy structured decoding."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "h4_event_ledger_m0_pilot_v1" / "run.py"
ORDER_BASE = ("412356", "345162", "256431", "624513", "531624", "163245")

spec = importlib.util.spec_from_file_location("base_probe_temp0", BASE)
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.RUN = HERE / "probe_v3" / "run"
probe.solver.STRUCTURED_TEMPERATURE = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--varied", action="store_true")
    args = parser.parse_args()
    out = HERE / ("temp0_q1_varied_results.json" if args.varied else "temp0_q1_results.json")
    if out.exists():
        raise RuntimeError("refuse_overwrite_temp0_results")
    from transformers import AutoTokenizer
    from scripts.current_method.restart_n5_spell.backends import SolverRequest

    manifest = probe.read(probe.RUN / "manifest.json")
    for name, expected in manifest["input_sha256"].items():
        if probe.sha(probe.RUN / name) != expected:
            raise RuntimeError(f"frozen_input_changed:{name}")
    candidates, dialogue = probe.inputs()
    candidate = next(q for q in candidates if q["candidate_id"] == "259192-manual-v2-status-01")
    if args.varied:
        keys = list(candidate["options"])
        base = [[keys[int(char) - 1] for char in order] for order in ORDER_BASE]
        orders = base + [list(reversed(order)) for order in base]
        requests = [{"request_id": f"{candidate['candidate_id']}__varied_{index:02d}",
                     "variant": probe.variant(candidate, f"varied_{index:02d}", order),
                     "seed": 20262000 + index}
                    for index, order in enumerate(orders)]
    else:
        requests = [row for row in manifest["requests"]
                    if row["candidate_id"] == candidate["candidate_id"] and row["mode"] == "full_context"]
    assert len(requests) == 12
    tokenizer = AutoTokenizer.from_pretrained(str(probe.MODEL), local_files_only=True)
    rows = []
    backend = probe.solver.create_backend(model_path=probe.MODEL, tokenizer_path=probe.MODEL)
    try:
        for row in requests:
            prompt = probe.contract.build_prompt(tokenizer, candidate, row["variant"],
                                                 dialogue, context_mode="full_context")
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            if row.get("prompt_sha256") and prompt_sha != row["prompt_sha256"]:
                raise RuntimeError("prompt_drift:" + row["request_id"])
            request = SolverRequest(
                request_id=row["request_id"] + "__temp0", qa_id=candidate["candidate_id"],
                evaluation_mode="full_context", order_id=row["variant"]["variant_id"],
                target_slot=0, retry_index=0, sampling_seed=row["seed"],
                options=row["variant"]["options"], canonical_answer_text="",
                prompt_sha256=prompt_sha, prompt=prompt,
            )
            response = probe.solver._structured_generate(backend, request,
                                                         question_type="multi_select")
            if response.metadata.get("finish_reason") not in ("stop", "eos"):
                raise RuntimeError("incomplete_generation:" + row["request_id"])
            score = probe.contract.score_output(response.raw_text, candidate, row["variant"])
            rows.append({"variant_id": row["variant"]["variant_id"],
                         "canonical_prediction": score["canonical_prediction"],
                         "correct": score["correct"], "raw_output": response.raw_text,
                         "prompt_sha256": prompt_sha,
                         "display_to_canonical": row["variant"]["display_to_canonical"]})
            print(json.dumps(rows[-1]), flush=True)
    finally:
        backend.close()
    out.write_text(json.dumps({"candidate_id": candidate["candidate_id"],
                               "model": str(probe.MODEL), "temperature": 0.0,
                               "top_p": probe.solver.TOP_P, "gold": candidate["gold"],
                               "display_protocol": "varied" if args.varied else "original_balanced",
                               "rows": rows, "training_promotion": False}, indent=2) + "\n",
                   encoding="utf-8")


if __name__ == "__main__":
    main()
