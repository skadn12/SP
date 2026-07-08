#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path

import requests


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def load_config(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_api_config(config):
    api_cfg = config.get("api", {}) if isinstance(config.get("api"), dict) else {}

    base_url = (
        api_cfg.get("base_url")
        or api_cfg.get("api_base")
        or api_cfg.get("openai_base_url")
        or config.get("base_url")
        or config.get("api_base")
        or config.get("openai_base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("SILICONFLOW_BASE_URL")
        or "https://api.siliconflow.cn/v1"
    )

    api_key = (
        api_cfg.get("api_key")
        or api_cfg.get("openai_api_key")
        or api_cfg.get("siliconflow_api_key")
        or config.get("api_key")
        or config.get("openai_api_key")
        or config.get("siliconflow_api_key")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
    )

    model = (
        api_cfg.get("model")
        or config.get("model")
        or os.environ.get("OPENAI_MODEL")
        or "Qwen/Qwen3-14B"
    )

    return base_url.rstrip("/"), api_key, model


def normalize_turns(sample, mode="full", tail_turns=40):
    turns = sample.get("dialogue", {}).get("turns", []) or []

    if mode == "tail40":
        turns = turns[-tail_turns:]

    return turns


def normalize_options(sample):
    opts = sample.get("options")
    if opts is None:
        opts = sample.get("qa", {}).get("options")
    return opts or []


def get_question(sample):
    return sample.get("question") or sample.get("qa", {}).get("question") or ""


def get_gold(sample):
    gold = (
        sample.get("gold_answer")
        or sample.get("gold_options")
        or sample.get("gold_answers")
        or sample.get("qa", {}).get("gold")
        or sample.get("qa", {}).get("gold_answer")
    )
    if gold is None:
        return []
    if isinstance(gold, list):
        return [str(x).strip() for x in gold]
    return [str(gold).strip()]


def format_options(options):
    lines = []
    for o in options:
        if isinstance(o, dict):
            oid = o.get("id")
            text = o.get("text")
            lines.append(f"{oid}. {text}")
        else:
            lines.append(str(o))
    return "\n".join(lines)


def answer_instruction(question_type):
    if question_type == "evidence_selection":
        return "Answer only with turn IDs, e.g. D19:1 or D19:1,D19:3. Do not explain."
    if question_type == "ordering":
        return "Answer only with option letters separated by >, e.g. A>B>D>C. Do not explain."
    if question_type == "multi_select":
        return "Answer only with option letters separated by commas, e.g. A,C. Do not explain."
    if question_type == "single_choice":
        return "Answer only with one option letter, e.g. B. Do not explain."
    if question_type == "state_reconstruction":
        return "Answer only with the exact phrase. Do not explain."
    return "Answer only in the required format. Do not explain."


def build_prompt(sample, mode="full", tail_turns=40):
    qtype = sample.get("question_type") or sample.get("qa", {}).get("question_type") or ""
    turns = normalize_turns(sample, mode=mode, tail_turns=tail_turns)
    options = normalize_options(sample)
    question = get_question(sample)

    dialogue_lines = []
    for t in turns:
        tid = t.get("turn_id", "")
        speaker = t.get("speaker", "")
        text = t.get("text", "")
        dialogue_lines.append(f"[{tid}] {speaker}: {text}")

    prompt = f"""You are answering a structured question about a dialogue.

Dialogue:
{chr(10).join(dialogue_lines)}

Question:
{question}

Options:
{format_options(options)}

{answer_instruction(qtype)}
"""
    return prompt


def call_chat_api(base_url, api_key, model, prompt, max_tokens, temperature=0.0, retries=3, timeout=120):
    url = base_url + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}: {r.text[:500]}"
                time.sleep(2 * (attempt + 1))
                continue
            data = r.json()
            return data["choices"][0]["message"]["content"].strip(), data
        except Exception as e:
            last_err = repr(e)
            time.sleep(2 * (attempt + 1))

    return "", {"error": last_err}


def parse_prediction(pred, qtype):
    s = (pred or "").strip()

    if qtype == "evidence_selection":
        return re.findall(r"D\d+:\d+", s)

    if qtype in ("multi_select", "single_choice"):
        letters = re.findall(r"\b[A-F]\b", s.upper())
        seen = []
        for x in letters:
            if x not in seen:
                seen.append(x)
        return seen

    if qtype == "ordering":
        letters = re.findall(r"\b[A-F]\b", s.upper())
        seen = []
        for x in letters:
            if x not in seen:
                seen.append(x)
        return seen

    if qtype == "state_reconstruction":
        return s.strip()

    return s.strip()


def exact_match(parsed, gold, qtype):
    gold = [str(x).strip() for x in gold]

    if qtype == "state_reconstruction":
        if not gold:
            return False
        return str(parsed).strip().lower() == gold[0].strip().lower()

    if isinstance(parsed, list):
        if qtype in ("multi_select", "evidence_selection"):
            return set(parsed) == set(gold)
        if qtype in ("ordering", "single_choice"):
            return parsed == gold

    return False


def partial_credit(parsed, gold, qtype):
    gold = [str(x).strip() for x in gold]
    if not gold:
        return 0.0

    if qtype == "state_reconstruction":
        return 1.0 if exact_match(parsed, gold, qtype) else 0.0

    if not isinstance(parsed, list):
        return 0.0

    if qtype in ("multi_select", "evidence_selection"):
        ps, gs = set(parsed), set(gold)
        if not ps:
            return 0.0
        precision = len(ps & gs) / max(1, len(ps))
        recall = len(ps & gs) / max(1, len(gs))
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    if qtype == "ordering":
        if not parsed:
            return 0.0
        correct_pos = sum(1 for i, x in enumerate(parsed[:len(gold)]) if i < len(gold) and x == gold[i])
        return correct_pos / len(gold)

    if qtype == "single_choice":
        return 1.0 if parsed == gold else 0.0

    return 0.0


def max_tokens_for_qtype(qtype):
    if qtype == "single_choice":
        return 8
    if qtype == "multi_select":
        return 16
    if qtype in ("ordering", "evidence_selection"):
        return 24
    if qtype == "state_reconstruction":
        return 64
    return 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", choices=["full", "tail40"], default="full")
    ap.add_argument("--tail-turns", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    config = load_config(args.config)
    base_url, api_key, config_model = get_api_config(config)
    if args.model == 'Qwen/Qwen3-14B' and config_model:
        args.model = config_model

    samples = load_jsonl(args.samples)[:args.limit]
    rows = []

    exact_count = 0
    partial_sum = 0.0
    qtype_counts = {}
    bucket_counts = {}

    for i, s in enumerate(samples):
        qtype = s.get("question_type") or s.get("qa", {}).get("question_type") or ""
        qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1
        gold = get_gold(s)
        prompt = build_prompt(s, mode=args.mode, tail_turns=args.tail_turns)

        preds = []
        exacts = 0
        partials = []

        for kk in range(args.k):
            pred, raw = call_chat_api(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=max_tokens_for_qtype(qtype),
                temperature=args.temperature if args.k == 1 else max(args.temperature, 0.2),
            )
            parsed = parse_prediction(pred, qtype)
            ex = exact_match(parsed, gold, qtype)
            pc = partial_credit(parsed, gold, qtype)
            exacts += int(ex)
            partials.append(pc)

            preds.append({
                "prediction": pred,
                "parsed": parsed,
                "exact_correct": ex,
                "partial_credit": pc,
                "raw_error": raw.get("error") if isinstance(raw, dict) else None,
            })

        soft = exacts / max(1, args.k)
        partial = sum(partials) / max(1, len(partials))
        bucket = f"{exacts}/{args.k}"
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        if args.k == 1 and exacts == 1:
            exact_count += 1
        partial_sum += partial

        rows.append({
            "sample_id": s.get("sample_id"),
            "question_type": qtype,
            "attack_skill": s.get("attack_skill"),
            "gold": gold,
            "api_eval": {
                "model": args.model,
                "mode": args.mode,
                "k": args.k,
                "soft_accuracy": soft,
                "partial_credit": partial,
                "bucket": bucket,
                "predictions": preds,
                "prompt_char_len": len(prompt),
                "prompt_token_estimate": int(len(prompt) / 4),
            }
        })

        print(f"[{i+1}/{len(samples)}] {s.get('sample_id')} qtype={qtype} bucket={bucket} partial={partial:.3f}")

    dump_jsonl(Path(args.out), rows)

    summary = {
        "model": args.model,
        "mode": args.mode,
        "sample_count": len(samples),
        "k": args.k,
        "exact_accuracy_k1": exact_count / max(1, len(samples)) if args.k == 1 else None,
        "soft_accuracy": sum(r["api_eval"]["soft_accuracy"] for r in rows) / max(1, len(rows)),
        "partial_credit_mean": partial_sum / max(1, len(samples)),
        "bucket_counts": bucket_counts,
        "question_type_counts": qtype_counts,
        "prompt_token_estimate_mean": sum(r["api_eval"]["prompt_token_estimate"] for r in rows) / max(1, len(rows)),
        "prompt_token_estimate_max": max([r["api_eval"]["prompt_token_estimate"] for r in rows] or [0]),
    }

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
