#!/usr/bin/env python3
"""Fixed B1 inference worker. It never reads Gold or invokes Codex."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    requests_path = Path(args.requests).resolve()
    out_path = Path(args.out).resolve()
    rows = [json.loads(line) for line in requests_path.read_text().splitlines() if line.strip()]
    records = []
    started = time.time()
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
        import vllm  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        from vllm import LLM, SamplingParams  # type: ignore

        model_dir = "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        prompts = []
        for row in rows:
            messages = [{"role": "user", "content": row["prompt"]}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "<think>\n\n</think>\n\n")
        llm = LLM(model=model_dir, tokenizer=model_dir, dtype="bfloat16", max_model_len=16384,
                  gpu_memory_utilization=0.85, max_num_seqs=8, seed=0,
                  enable_prefix_caching=False, enforce_eager=True, disable_log_stats=True)
        for start in range(0, len(rows), 8):
            batch = rows[start : start + 8]
            params = [SamplingParams(temperature=0.8, top_p=0.9, top_k=-1,
                                     repetition_penalty=1.0, max_tokens=6144,
                                     seed=int(row.get("sampling_seed", 0))) for row in batch]
            outputs = llm.generate(prompts[start : start + 8], params, use_tqdm=False)
            for row, output in zip(batch, outputs):
                text = output.outputs[0].text if output.outputs else ""
                records.append({"request_id": row["request_id"], "returned": bool(text), "text": text,
                                "mode": "FULL THINK_OFF", "gold_read": False})
        meta = {"backend": "vllm", "torch": torch.__version__, "vllm": vllm.__version__,
                "transformers": transformers.__version__, "model": model_dir,
                "mode": "FULL THINK_OFF", "training": False}
    except Exception as exc:
        meta = {"backend": "unavailable", "error": repr(exc), "model": "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model",
                "mode": "FULL THINK_OFF", "training": False}
    payload = {"schema_version": "feedback_closed_loop_v3_raw_scores", "started_at": started,
               "finished_at": time.time(), "prepared_count": len(rows), "returned_count": len(records),
               "records": records, "backend": meta, "gold_read": False, "training": False}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, out_path)
    return 0 if len(records) == len(rows) and all(r["returned"] for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
