"""Diagnostic-only driver for the Self-Play r3_long Challenger/Solver pipeline.

This script is READ-ONLY with respect to the protected generation / alignment /
validator / frontier / reward code. It reuses the *already patched* solver-eval
building blocks from scripts/old_method/run_old_improved_gate200.py (imported as
`base`) so that prompt construction, decoding, stop sequences, max_new_tokens and
scoring are identical to the shipped pipeline. The model is loaded once and reused
across every eval pass.

Tasks:
  1. r2 eval sanity check on existing r2 parent samples (schema-converted).
  2. Same-sample context ablation on r3_long gate20 samples
     (full_dialogue / tail_40_turns / evidence_window_only).
  3. Parent/appended template-contamination counters (pure text, no model).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Blackwell SM 12.x + vLLM 0.21: the default FlashInfer sampler JIT-fails and silently degrades
# decoding to gibberish, which would corrupt the controller's "clean" per-cell eval too. Force it
# off in-process here (belt-and-suspenders alongside the env the controller passes to subprocess).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from typing import Any

ROOT = Path("/root/autodl-tmp/gxy-selfplay")
BASE_SCRIPT = ROOT / "scripts/old_method/run_old_improved_gate200.py"
DEFAULT_BASE_MODEL = "/root/autodl-tmp/Qwen3-4B"
DEFAULT_SOLVER_CKPT = ROOT / "outputs/qwen3_4b_lora_r2_frontier_plus_clean_replay_s0_s2_from_r1_e1_ctx9216_full"
R2_SAMPLES = ROOT / "eval_sets/graph_expansion_challenger_v1/r2_auto_full_sample0_5_complete/samples.jsonl"
GATE20 = ROOT / "current_method_runs/current_method_r3_long_append_from_r2_gate20_det_align_fastpath_provfix"
DIAG_DIR = ROOT / "current_method_runs/diagnostics_r3_long_ablation"

MAX_MODEL_LEN = 9216
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
TOP_P = 0.9
GPU_MEM = 0.90
SEED = 20260615

TEMPLATE_PHRASES = [
    "kept it tied to that the part of the conversation",
    "I logged state update",
    "The detail for the detail",
    "When we checked the what was said earlier",
    "When they looked back at what was said earlier",
    "state update",
    "I will rely only on what was clearly said in this conversation",
    "I logged",
    "the person's detail for",
]


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("diag_base_gate200", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["diag_base_gate200"] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def eval_args(
    k_eval: int,
    debug_dir: Path | None = None,
    *,
    tokenizer_path: str = DEFAULT_BASE_MODEL,
    max_model_len: int = MAX_MODEL_LEN,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    gpu_memory_utilization: float = GPU_MEM,
) -> argparse.Namespace:
    return argparse.Namespace(
        k_eval=k_eval,
        seed=SEED,
        eval_backend="vllm",
        base_model=tokenizer_path,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        gpu_memory_utilization=gpu_memory_utilization,
        prompt_debug_dir=str(debug_dir) if debug_dir else None,
    )


class SharedSolver:
    """Loads the merged solver once; reuses base's prompt/decode/score helpers."""

    def __init__(
        self,
        *,
        solver_model: str | Path = DEFAULT_SOLVER_CKPT,
        solver_tokenizer: str | Path | None = None,
        base_model: str | Path = DEFAULT_BASE_MODEL,
        max_model_len: int = MAX_MODEL_LEN,
        gpu_memory_utilization: float = GPU_MEM,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer_path = str(solver_tokenizer or base_model)
        self.model_path, self.adapter_path = base.resolve_solver_checkpoint(str(solver_model))
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.llm = LLM(
            model=self.model_path,
            tokenizer=self.tokenizer_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

    def _generate(self, samples: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[list[str]], list[dict[str, Any]]]:
        """Mirror of base.evaluate_with_vllm grouping, but on the shared LLM."""
        from vllm import SamplingParams

        prompts = [base.render_prompt(self.tokenizer, sample) for sample in samples]
        diagnostics = [
            base.solver_prompt_diagnostics(
                sample=sample,
                prompt=prompt,
                tokenizer=self.tokenizer,
                args=args,
                max_new_tokens=base.effective_solver_max_new_tokens(sample, args),
            )
            for sample, prompt in zip(samples, prompts)
        ]
        raw_predictions: list[list[str]] = [base.solver_overflow_prediction(s, args) for s in samples]
        grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for index, (prompt, diag) in enumerate(zip(prompts, diagnostics)):
            if diag.get("eval_context_overflow"):
                continue
            grouped[int(diag["max_new_tokens"])].append((index, prompt))
        for max_tokens, items in sorted(grouped.items()):
            params = SamplingParams(
                n=args.k_eval,
                temperature=args.temperature if args.k_eval > 1 else 0.0,
                top_p=args.top_p if args.k_eval > 1 else 1.0,
                max_tokens=max_tokens,
                stop=list(base.SOLVER_STOP_SEQUENCES),
            )
            outputs = self.llm.generate([p for _, p in items], params)
            for (sample_index, _), request_output in zip(items, outputs):
                raw_predictions[sample_index] = [base.stop_at_solver_sequence(o.text) for o in request_output.outputs]
        return raw_predictions, diagnostics

    def evaluate(self, samples: list[dict[str, Any]], k_eval: int, debug_dir: Path | None = None) -> list[dict[str, Any]]:
        """Mirror of base.run_solver_eval scoring on shared-LLM predictions."""
        args = eval_args(
            k_eval,
            debug_dir,
            tokenizer_path=self.tokenizer_path,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        if debug_dir:
            base.write_solver_prompt_debug_files(samples, self.tokenizer, args)
        raw_predictions, prompt_diagnostics = self._generate(samples, args)
        eval_rows: list[dict[str, Any]] = []
        for sample, predictions, diagnostics in zip(samples, raw_predictions, prompt_diagnostics):
            scored = []
            exact_count = 0
            partials: list[float] = []
            for idx, prediction in enumerate(predictions[:k_eval]):
                exact, partial, parsed = base.score_prediction(sample, prediction)
                exact_count += 1 if exact else 0
                partials.append(partial)
                scored.append(
                    {
                        "sample_index": idx,
                        "prediction": prediction,
                        "parsed": parsed,
                        "exact_correct": exact,
                        "partial_credit": round(partial, 6),
                    }
                )
            while len(scored) < k_eval:
                scored.append({"sample_index": len(scored), "prediction": "", "parsed": "", "exact_correct": False, "partial_credit": 0.0})
                partials.append(0.0)
            soft_accuracy = exact_count / k_eval if k_eval else 0.0
            partial_credit = sum(partials) / k_eval if k_eval else 0.0
            sample["solver_eval"] = {
                "k": k_eval,
                "predictions": scored,
                "exact_correct_count": exact_count,
                "soft_accuracy": round(soft_accuracy, 6),
                "partial_credit": round(partial_credit, 6),
                "bucket": f"{exact_count}/{k_eval}",
                "diagnostics": diagnostics,
            }
            eval_rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "question_type": sample.get("qa", {}).get("question_type"),
                    "attack_skill": sample.get("attack_plan", {}).get("attack_skill"),
                    "solver_eval": sample["solver_eval"],
                }
            )
        return eval_rows


def summarize(samples: list[dict[str, Any]], k_eval: int, n_preds: int = 5) -> dict[str, Any]:
    n = len(samples)
    buckets = {f"{i}/{k_eval}": 0 for i in range(k_eval + 1)}
    diags = []
    soft_sum = 0.0
    exact_k1 = 0
    cat = {"bad_question": 0, "true_too_hard": 0, "near_miss": 0}
    first_preds: list[dict[str, Any]] = []
    for sample in samples:
        se = sample.get("solver_eval", {})
        buckets[str(se.get("bucket") or f"0/{k_eval}")] += 1
        soft = float(se.get("soft_accuracy") or 0.0)
        partial = float(se.get("partial_credit") or 0.0)
        soft_sum += soft
        preds = se.get("predictions", [])
        if preds and preds[0].get("exact_correct"):
            exact_k1 += 1
        # frontier-style coarse categorisation (diagnostic, mirrors gate200 bands)
        if soft == 0.0 and partial < 0.4:
            cat["bad_question"] += 1
        elif soft == 0.0 and partial >= 0.4:
            cat["true_too_hard"] += 1
        elif 0.2 <= soft <= 0.6:
            cat["near_miss"] += 1
        if isinstance(se.get("diagnostics"), dict):
            diags.append(se["diagnostics"])
        if len(first_preds) < n_preds:
            first_preds.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "question_type": sample.get("qa", {}).get("question_type"),
                    "bucket": se.get("bucket"),
                    "prediction_0": (preds[0].get("prediction") if preds else "")[:240],
                    "parsed_0": preds[0].get("parsed") if preds else None,
                    "gold": sample.get("qa", {}).get("gold"),
                }
            )
    tok = [int(d.get("prompt_token_estimate") or 0) for d in diags]
    return {
        "n": n,
        "k_eval": k_eval,
        "solver_exact_accuracy_k1": round(exact_k1 / n, 6) if n else 0.0,
        "solver_soft_accuracy_k5": round(soft_sum / n, 6) if n else 0.0,
        "soft_buckets": buckets,
        "bad_question_count": cat["bad_question"],
        "true_too_hard_count": cat["true_too_hard"],
        "near_miss_count": cat["near_miss"],
        "prompt_token_mean": round(sum(tok) / len(tok), 3) if tok else 0.0,
        "prompt_token_max": max(tok) if tok else 0,
        "eval_context_overflow_count": sum(1 for d in diags if d.get("eval_context_overflow") is True),
        "truncated_count": sum(1 for d in diags if d.get("truncated") is True),
        "answer_format_instruction_missing_count": sum(
            1 for d in diags if d.get("answer_format_instruction_present") is not True
        ),
        "first_5_predictions": first_preds,
    }


# --------------------------------------------------------------------------- #
# Task 1: r2 parent schema -> nested qa/dialogue (eval-only conversion)        #
# --------------------------------------------------------------------------- #
def r2_to_eval_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    qtype = str(row.get("question_type") or "")
    turns = base.parse_dialogue_turns(row)
    if len(turns) < 4:
        return None
    options_map = row.get("options")
    options: list[dict[str, Any]] = []
    if isinstance(options_map, dict):
        for label in sorted(options_map.keys()):
            options.append({"id": label, "text": str(options_map[label])})
    gold: list[str]
    if qtype == "state_reconstruction":
        gold = [str(row.get("gold_answer") or "")]
    elif qtype == "ordering":
        raw = str(row.get("gold_answer") or "")
        gold = [t for t in re.split(r"\s*>\s*", raw) if t.strip()]
    elif qtype == "multi_select":
        gold = [str(x) for x in (row.get("gold_options") or [])]
    else:
        gold = [str(x) for x in (row.get("gold_options") or [])] or [str(row.get("gold_answer") or "")]
    evidence = [str(x) for x in (row.get("evidence_nodes") or []) if str(x).strip()]
    qa = {
        "question_type": qtype,
        "question": row.get("question"),
        "options": options,
        "gold": gold,
        "gold_source": row.get("gold_source") or "r2_parent",
        "evidence_turn_ids": evidence,
    }
    return {
        "sample_id": row.get("sample_id"),
        "attack_plan": {"attack_skill": row.get("attack_type") or row.get("trap_type") or "r2_parent"},
        "qa": qa,
        "dialogue": {"turns": [{"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"]} for t in turns]},
    }


# --------------------------------------------------------------------------- #
# Task 2: context ablation modes (diagnostic-only prompt reshaping)           #
# --------------------------------------------------------------------------- #
def _option_turn_ids(qa: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for opt in qa.get("options") or []:
        if not isinstance(opt, dict):
            continue
        for tid in opt.get("turn_ids") or []:
            ids.append(str(tid))
        oid = str(opt.get("id") or "")
        if re.match(r"^D\d+:\d+$", oid):  # evidence_selection option ids are turn ids
            ids.append(oid)
    return ids


def make_full_dialogue(sample: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(sample)


def make_tail40(sample: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(sample)
    turns = out.get("dialogue", {}).get("turns") or []
    out["dialogue"]["turns"] = turns[-40:]
    return out


def make_evidence_window(sample: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(sample)
    turns = out.get("dialogue", {}).get("turns") or []
    qa = out.get("qa", {})
    index_by_id = {str(t.get("turn_id")): i for i, t in enumerate(turns)}
    keep_ids = set(str(x) for x in (qa.get("evidence_turn_ids") or [])) | set(_option_turn_ids(qa))
    keep_idx: set[int] = set()
    for tid in keep_ids:
        if tid in index_by_id:
            i = index_by_id[tid]
            for off in (-1, 0, 1):
                if 0 <= i + off < len(turns):
                    keep_idx.add(i + off)
    # Fallback: if nothing matched, keep tail 12 so the prompt is still well-formed.
    if not keep_idx:
        keep_idx = set(range(max(0, len(turns) - 12), len(turns)))
    out["dialogue"]["turns"] = [turns[i] for i in sorted(keep_idx)]
    return out


ABLATION_BUILDERS = {
    "full_dialogue": make_full_dialogue,
    "tail_40_turns": make_tail40,
    "evidence_window_only": make_evidence_window,
}


# --------------------------------------------------------------------------- #
# Task 3: template contamination counters (pure text)                         #
# --------------------------------------------------------------------------- #
def _phrase_counts(text: str) -> tuple[int, list[str]]:
    low = text.lower()
    total = 0
    hits: list[str] = []
    for phrase in TEMPLATE_PHRASES:
        c = low.count(phrase.lower())
        if c:
            total += c
            hits.append(phrase)
    return total, hits


def template_contamination(samples: list[dict[str, Any]]) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        final_turns = sample.get("dialogue", {}).get("turns") or []
        final_text = "\n".join(str(t.get("text") or "") for t in final_turns)
        parent_text = str(sample.get("expanded_dialogue_text") or sample.get("new_dialogue_text") or "")
        final_count, _ = _phrase_counts(final_text)
        parent_count, _ = _phrase_counts(parent_text)
        turns_with = sum(1 for t in final_turns if _phrase_counts(str(t.get("text") or ""))[0] > 0)
        per_sample.append(
            {
                "sample_id": sample.get("sample_id"),
                "question_type": sample.get("qa", {}).get("question_type"),
                "final_template_phrase_count": final_count,
                "parent_template_phrase_count": parent_count,
                "template_phrase_count": final_count,
                "turns_containing_template_phrases": turns_with,
                "final_turn_count": len(final_turns),
                "turns_with_template_ratio": round(turns_with / len(final_turns), 4) if final_turns else 0.0,
            }
        )
    n = len(per_sample) or 1
    def agg(key: str) -> dict[str, float]:
        vals = [p[key] for p in per_sample]
        return {"mean": round(sum(vals) / n, 4), "max": max(vals) if vals else 0, "min": min(vals) if vals else 0}
    return {
        "n_samples": len(per_sample),
        "final_template_phrase_count": agg("final_template_phrase_count"),
        "parent_template_phrase_count": agg("parent_template_phrase_count"),
        "turns_containing_template_phrases": agg("turns_containing_template_phrases"),
        "turns_with_template_ratio": agg("turns_with_template_ratio"),
        "per_sample": per_sample,
    }


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic context ablation + r2 sanity check (read-only).")
    parser.add_argument("--task", choices=("template", "r2", "ablation", "all", "verify"), default="all")
    parser.add_argument("--r2-limit", type=int, default=10)
    parser.add_argument("--k-eval", type=int, default=5)
    parser.add_argument("--out-dir", default=str(DIAG_DIR))
    parser.add_argument("--samples", default=None, help="Path to a samples.jsonl to eval verbatim (task=verify)")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--solver-model", default=str(DEFAULT_SOLVER_CKPT))
    parser.add_argument("--solver-tokenizer", default=None)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEM)
    return parser.parse_args()


def main() -> int:
    cli = parse_cli()
    out_dir = Path(cli.out_dir)
    gen_dir = out_dir / "generation"
    rep_dir = out_dir / "reports"
    gen_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # --- verify mode: eval an arbitrary samples.jsonl verbatim (full_dialogue) ---
    if cli.task == "verify":
        target = Path(cli.samples)
        samples = read_jsonl(target)
        # Deep-copy so we never read any stale solver_eval already embedded in the file;
        # summarize the freshly-evaluated objects, not the originals.
        eval_samples = [copy.deepcopy(s) for s in samples]
        solver = SharedSolver(
            solver_model=cli.solver_model,
            solver_tokenizer=cli.solver_tokenizer,
            base_model=cli.base_model,
            max_model_len=cli.max_model_len,
            gpu_memory_utilization=cli.gpu_memory_utilization,
        )
        rows = solver.evaluate(eval_samples, k_eval=cli.k_eval, debug_dir=rep_dir / "verify_prompt_debug")
        write_jsonl(gen_dir / "verify_k5_eval.jsonl", rows)
        vs = summarize(eval_samples, cli.k_eval)
        write_json(rep_dir / "verify_summary.json", {"samples_path": str(target), **vs})
        print("[verify]", {k: vs[k] for k in ("solver_exact_accuracy_k1", "solver_soft_accuracy_k5", "soft_buckets", "bad_question_count")})
        for fp in vs["first_5_predictions"]:
            print("   ", fp["question_type"], fp["bucket"], repr(str(fp["prediction_0"])[:80]))
        return 0

    r3_samples = read_jsonl(GATE20 / "generation/samples.jsonl")

    # Task 3 (no model) first so it is available even if eval is skipped.
    contamination = template_contamination(r3_samples)
    write_json(rep_dir / "template_contamination.json", contamination)
    print("[task3] template contamination written:", {
        "final_mean": contamination["final_template_phrase_count"]["mean"],
        "final_max": contamination["final_template_phrase_count"]["max"],
        "parent_mean": contamination["parent_template_phrase_count"]["mean"],
        "turns_ratio_mean": contamination["turns_with_template_ratio"]["mean"],
    })

    if cli.task == "template":
        return 0

    solver = SharedSolver(
        solver_model=cli.solver_model,
        solver_tokenizer=cli.solver_tokenizer,
        base_model=cli.base_model,
        max_model_len=cli.max_model_len,
        gpu_memory_utilization=cli.gpu_memory_utilization,
    )
    summary: dict[str, Any] = {
        "solver_model_path": solver.model_path,
        "solver_adapter_path": solver.adapter_path,
        "solver_tokenizer": solver.tokenizer_path,
        "max_model_len": solver.max_model_len,
        "max_new_tokens": MAX_NEW_TOKENS,
        "k_eval": cli.k_eval,
    }

    # ---- Task 1: r2 sanity check ----
    if cli.task in ("r2", "all"):
        raw_r2 = read_jsonl(R2_SAMPLES, limit=2000)
        # Stratify across question types so the sanity check is not all multi_select
        # (r3_long is dominated by ordering / evidence_selection, so we must cover those).
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in raw_r2:
            by_type[str(r.get("question_type"))].append(r)
        per_type = max(1, cli.r2_limit // max(1, len(by_type)))
        picked: list[dict[str, Any]] = []
        for t, group in by_type.items():
            picked.extend(group[:per_type])
        # top up to r2_limit from whatever remains
        if len(picked) < cli.r2_limit:
            picked_ids = {id(x) for x in picked}
            for r in raw_r2:
                if id(r) not in picked_ids:
                    picked.append(r)
                if len(picked) >= cli.r2_limit:
                    break
        r2_eval_samples = [s for s in (r2_to_eval_sample(r) for r in picked) if s][: cli.r2_limit]
        rows = solver.evaluate(r2_eval_samples, k_eval=cli.k_eval, debug_dir=rep_dir / "r2_prompt_debug")
        write_jsonl(gen_dir / "r2_sanity_k5_eval.jsonl", rows)
        r2_summary = summarize(r2_eval_samples, cli.k_eval)
        summary["task1_r2_sanity"] = r2_summary
        write_json(rep_dir / "r2_sanity_summary.json", r2_summary)
        print("[task1] r2 sanity:", {k: r2_summary[k] for k in ("solver_exact_accuracy_k1", "solver_soft_accuracy_k5", "soft_buckets", "bad_question_count")})

    # ---- Task 2: context ablation on r3_long ----
    if cli.task in ("ablation", "all"):
        ablation: dict[str, Any] = {}
        file_map = {
            "full_dialogue": "solver_k5_eval_full_dialogue.jsonl",
            "tail_40_turns": "solver_k5_eval_tail40.jsonl",
            "evidence_window_only": "solver_k5_eval_evidence_window.jsonl",
        }
        for mode, builder in ABLATION_BUILDERS.items():
            mode_samples = [builder(s) for s in r3_samples]
            rows = solver.evaluate(mode_samples, k_eval=cli.k_eval, debug_dir=rep_dir / f"ablation_prompt_debug_{mode}")
            write_jsonl(gen_dir / file_map[mode], rows)
            ablation[mode] = summarize(mode_samples, cli.k_eval)
            print(f"[task2] {mode}:", {k: ablation[mode][k] for k in ("solver_soft_accuracy_k5", "soft_buckets", "bad_question_count", "prompt_token_mean")})
        summary["task2_context_ablation"] = ablation
        write_json(rep_dir / "context_ablation_summary.json", {"modes": ablation, "meta": {k: summary[k] for k in ("solver_model_path", "max_model_len", "max_new_tokens", "k_eval")}})

    summary["task3_template_contamination"] = {
        k: contamination[k] for k in ("final_template_phrase_count", "parent_template_phrase_count", "turns_with_template_ratio")
    }
    write_json(rep_dir / "diagnostics_master_summary.json", summary)
    print("\n[done] master summary:", rep_dir / "diagnostics_master_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
