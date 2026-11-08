#!/usr/bin/env python3
"""Small, fixed external-evaluation smoke test for MultiRC and BAMBOO.

This runner is intentionally independent of the training/evaluation code.  It
does not truncate prompts: records over the prompt budget are written as
skipped rows and never sent to vLLM.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_multirc(zip_path: Path, n: int, one_per_paragraph: bool = True) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("splitv2/dev_83-fixedIds.json"))["data"]
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for pidx, item in enumerate(data):
        pid = str(item.get("id", pidx))
        para = item["paragraph"]
        context = html.unescape(re.sub(r"<[^>]+>", " ", str(para.get("text", ""))))
        context = re.sub(r"\s+", " ", context).strip()
        for q in para.get("questions", []):
            answers = q.get("answers", [])
            gold = [i for i, a in enumerate(answers) if a.get("isAnswer") is True]
            if not gold or (one_per_paragraph and pid in used):
                continue
            if one_per_paragraph:
                used.add(pid)
            rows.append({
                "dataset": "MultiRC",
                "task": "multirc",
                "id": f"{pid}::{q.get('idx')}",
                "context": context,
                "question": str(q.get("question", "")),
                "options": [str(a.get("text", "")) for a in answers],
                "gold": gold,
            })
            if n > 0 and len(rows) >= n:
                return rows
    return rows


def load_bamboo(path: Path, task: str, n: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            rows.append({"dataset": "BAMBOO", "task": task, "id": f"{task}:{i}", "raw": r, "gold": r["answer"]})
            if n > 0 and len(rows) >= n:
                break
    return rows


def multirc_messages(r: dict[str, Any]) -> list[dict[str, str]]:
    opts = "\n".join(f"{LETTERS[i]}. {x}" for i, x in enumerate(r["options"]))
    user = (
        "Context:\n" + r["context"] + "\n\nQuestion:\n" + r["question"] +
        "\n\nOptions:\n" + opts +
        "\n\nSelect every option that is correct. Output only the option letters separated by commas."
    )
    return [{"role": "system", "content": "You are a careful reading comprehension assistant."}, {"role": "user", "content": user}]


def bamboo_messages(r: dict[str, Any], prompt_spec: dict[str, Any]) -> list[dict[str, str]]:
    p = r["raw"]
    if r["task"].startswith("showssort"):
        content = "\n".join(f"[{i}] ```{x}```" for i, x in enumerate(p["content"]))
        spec = prompt_spec["showssort"]
        user = spec["final_answer"].format(content=content)
        system = spec["system"]
    else:
        events = "\n".join(f"[{i}] ```{x}```" for i, x in enumerate(p["summaries"]))
        spec = prompt_spec["reportsumsort"]
        user = spec["final_answer"].format(content=p["content"], events=events)
        system = spec["system"]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{m['role'].title()}: {m['content']}" for m in messages) + "\n\nAssistant:"


def parse_multirc(text: str, count: int) -> list[int]:
    # Prefer bracket/comma lists, then accept standalone option letters.
    found = re.findall(r"\b([A-Z])\b", text.upper())
    out = []
    for x in found:
        i = LETTERS.index(x) if x in LETTERS else -1
        if 0 <= i < count and i not in out:
            out.append(i)
    return sorted(out)


def parse_order(text: str) -> list[int] | None:
    nums = [int(x) for x in re.findall(r"\[(\d)\]", text)]
    if len(nums) != 5:
        nums = [int(x) for x in re.findall(r"\b([0-4])\b", text)]
    if len(nums) != 5 or sorted(nums) != list(range(5)):
        return None
    return nums


def kendall(pred: list[int], gold: list[int]) -> float:
    pos = {x: i for i, x in enumerate(gold)}
    a = [pos[x] for x in pred]
    discord = sum(a[i] > a[j] for i in range(5) for j in range(i + 1, 5))
    return 1.0 - discord / 10.0


def score(row: dict[str, Any], text: str) -> dict[str, Any]:
    if row["task"] == "multirc":
        pred = parse_multirc(text, len(row["options"]))
        gold = sorted(row["gold"])
        inter = len(set(pred) & set(gold))
        precision = inter / len(pred) if pred else 0.0
        recall = inter / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"prediction": pred, "gold": gold, "exact": pred == gold, "f1": f1, "parse_ok": bool(pred)}
    pred = parse_order(text)
    gold = list(row["gold"])
    return {"prediction": pred, "gold": gold, "exact": pred == gold, "kendall": kendall(pred, gold) if pred else 0.0, "parse_ok": pred is not None}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by task while retaining parser and overflow health counts."""
    out: dict[str, Any] = {}
    tasks = sorted({str(r.get("task")) for r in results})
    for task in tasks:
        rows = [r for r in results if r.get("task") == task]
        scored = [r for r in rows if not r.get("overflow")]
        exact = sum(bool(r.get("exact")) for r in scored)
        item: dict[str, Any] = {
            "total": len(rows),
            "generated": sum("raw_output" in r for r in rows),
            "overflow": sum(bool(r.get("overflow")) for r in rows),
            "parse_failures": sum(not bool(r.get("parse_ok")) for r in scored),
            "exact": exact,
            "exact_rate": exact / len(scored) if scored else None,
        }
        if task == "multirc":
            f1s = [float(r.get("f1", 0.0)) for r in scored]
            tp = fp = fn = 0
            for r in scored:
                p, g = set(r.get("prediction") or []), set(r.get("gold") or [])
                tp += len(p & g); fp += len(p - g); fn += len(g - p)
            item["f1m"] = sum(f1s) / len(f1s) if f1s else None
            item["f1a"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
            item.update({"tp": tp, "fp": fp, "fn": fn})
        else:
            vals = [float(r.get("kendall", 0.0)) for r in scored if r.get("parse_ok")]
            item["pairwise_agreement"] = sum(vals) / len(vals) if vals else None
            item["kendall_tau"] = 2 * item["pairwise_agreement"] - 1 if vals else None
        out[task] = item
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--multirc-zip", required=True)
    ap.add_argument("--bamboo-dir", required=True)
    ap.add_argument("--n-multirc", type=int, default=5)
    ap.add_argument("--n-bamboo", type=int, default=3)
    ap.add_argument("--all-multirc", action="store_true", help="Use every scored question in the public MultiRC dev split.")
    ap.add_argument("--all-bamboo", action="store_true", help="Use every example in all four BAMBOO sorting files.")
    ap.add_argument("--batch-size", type=int, default=1, help="Number of prompts per vLLM request; use 1 for smoke, 2-4 for full runs.")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if args.all_multirc:
        rows = load_multirc(Path(args.multirc_zip), 0, one_per_paragraph=False)
    else:
        rows = load_multirc(Path(args.multirc_zip), args.n_multirc)
    bdir = Path(args.bamboo_dir)
    prompt_spec = json.loads((bdir / "prompt.json").read_text(encoding="utf-8"))
    if args.all_bamboo:
        rows += load_bamboo(bdir / "showssort_4k.jsonl", "showssort_4k", 0)
        rows += load_bamboo(bdir / "reportsumsort_4k.jsonl", "reportsumsort_4k", 0)
        rows += load_bamboo(bdir / "showssort_16k.jsonl", "showssort_16k", 0)
        rows += load_bamboo(bdir / "reportsumsort_16k.jsonl", "reportsumsort_16k", 0)
    else:
        rows += load_bamboo(bdir / "showssort_4k.jsonl", "showssort", args.n_bamboo)
        rows += load_bamboo(bdir / "reportsumsort_4k.jsonl", "reportsumsort", args.n_bamboo)
        # Include one 16k-context case of each sorting task to test length handling.
        rows += load_bamboo(bdir / "showssort_16k.jsonl", "showssort_16k", 1)
        rows += load_bamboo(bdir / "reportsumsort_16k.jsonl", "reportsumsort_16k", 1)
    llm = LLM(model=args.model, tokenizer=args.tokenizer, trust_remote_code=True, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.80, max_num_seqs=2, enforce_eager=True)
    tok = llm.get_tokenizer()
    budget = args.max_model_len - args.max_new_tokens
    prepared = []
    for row in rows:
        messages = multirc_messages(row) if row["task"] == "multirc" else bamboo_messages(row, prompt_spec)
        prompt = render(tok, messages)
        ids = tok(prompt, add_special_tokens=False).input_ids
        row["prompt_tokens"] = len(ids); row["prompt_budget"] = budget
        row["messages"] = messages
        row["overflow"] = len(ids) > budget
        if not row["overflow"]:
            prepared.append((row, ids))
    params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, seed=20261108)
    results = []
    t0 = time.time()
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids in chunk], params, use_tqdm=False)
        if len(outputs) != len(chunk):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for batch of {len(chunk)}")
        for (row, _), o in zip(chunk, outputs):
            text = o.outputs[0].text.strip() if o.outputs else ""
            s = score(row, text)
            results.append({"id": row["id"], "dataset": row["dataset"], "task": row["task"], "prompt_tokens": row["prompt_tokens"], "raw_output": text, **s})
            print(row["id"], row["prompt_tokens"], repr(text[:160]), s, flush=True)
    for row in rows:
        if row["overflow"]:
            results.append({"id": row["id"], "dataset": row["dataset"], "task": row["task"], "prompt_tokens": row["prompt_tokens"], "overflow": True, "parse_ok": False})
    payload = {"model": args.model, "tokenizer": args.tokenizer, "max_model_len": args.max_model_len, "max_new_tokens": args.max_new_tokens,
               "batch_size": batch_size, "all_multirc": bool(args.all_multirc), "all_bamboo": bool(args.all_bamboo),
               "source": {"multirc_zip_sha256": sha256(Path(args.multirc_zip)), "bamboo_manifest": str(bdir / "MANIFEST.json")},
               "n_total": len(rows), "n_generated": len(prepared), "elapsed_sec": round(time.time()-t0, 3),
               "summary": summarize_results(results), "results": results}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
