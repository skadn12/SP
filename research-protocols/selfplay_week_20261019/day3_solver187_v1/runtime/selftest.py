"""Synthetic CPU-only regression: no diagnostic question, private file, model or backend."""
from __future__ import annotations
import copy
import json
import argparse
from pathlib import Path
import common as c
import score
import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=c.HERE / "CPU_SELFTEST.json")
    args = parser.parse_args()
    c.environment()
    checks = []
    def check(condition, name):
        c.require(condition, name)
        checks.append(name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    _, solver = c.a.native()
    api, _ = solver.detect_structured_output_api()
    source = {"root_id": "synthetic-root", "turns": [
        {"turn_ref": "S1:T1", "speaker": "Alice", "text": "First synthetic marker."},
        {"turn_ref": "S2:T1", "speaker": "Bob", "text": "Last synthetic marker."}]}
    count = 0
    for typ in ("multi_select", "ordering"):
        field = "options" if typ == "multi_select" else "items"
        labels = c.a.labels(typ, 6)
        question = {"candidate_id": "synthetic-" + typ, "root_id": "synthetic-root",
                    "question_type": typ, "question": "Synthetic check only.",
                    field: {label: "Unique synthetic item " + str(index) for index, label in enumerate(labels)},
                    "source_path": "/synthetic-source", "source_sha256": "0" * 64}
        c.a.public_question(question)
        variants = c.a.variants(question, 2026101917, 6 if typ == "multi_select" else 12)
        poison = copy.deepcopy(question)
        poison["gold"] = list(reversed(labels))
        poison["secret"] = "SHOULD_NOT_APPEAR"
        check(variants == c.a.variants(poison, 2026101917, len(variants)), typ + ":gold_blind_mapping")
        canonical = c.a.content_ids(question)
        gold = list(labels[:3]) if typ == "multi_select" else list(labels)
        canonical_gold = [canonical[label] for label in gold]
        for index, variant in enumerate(variants):
            check(len(set(variant["display_to_content"].values())) == 6, typ + ":bijection:" + str(index))
            prompt = c.a.direct_prompt(tokenizer, question, variant, source)
            check(prompt == c.a.direct_prompt(tokenizer, poison, variant, source) and "SHOULD_NOT_APPEAR" not in prompt, typ + ":public_prompt:" + str(index))
            check(all(turn["text"] in prompt for turn in source["turns"]), typ + ":full_context:" + str(index))
            choices = c.a.choices(typ, variant[field])
            check(len(choices) == (63 if typ == "multi_select" else 720), typ + ":finite_choice_count:" + str(index))
            row = {"request_id": "synthetic-request", "candidate_id": question["candidate_id"],
                "root_id": question["root_id"], "question_type": typ, "branch": "direct",
                "variant_id": variant["variant_id"], "slot": 0, "seed": 2026101917,
                "prompt_sha256": "0" * 64, "display": {field: variant[field]}}
            params = c.a.sampling_params(row, api)
            check(params.max_tokens == 32 and params.temperature == .8 and params.top_p == .9, typ + ":native_params:" + str(index))
            reverse = {content: label for label, content in variant["display_to_content"].items()}
            answer = [reverse[content] for content in canonical_gold]
            if typ == "multi_select":
                answer.sort()
            raw = execute.terminal(row, "RETURNED", raw_output="<box>" + ",".join(answer) + "</box>",
                metadata={"finish_reason": "stop"})
            correct = score.score_one(row, raw, variant, canonical, gold)
            check(correct["valid"] and correct["exact"], typ + ":canonical_roundtrip:" + str(index))
            if typ == "multi_select":
                check(correct["metrics"]["fp"] == correct["metrics"]["fn"] == 0 and len(correct["items"]) == 6, typ + ":item_metrics:" + str(index))
            else:
                check(correct["metrics"]["correct_pairs"] == 15 and len(correct["pairs"]) == 15, typ + ":all_pairs:" + str(index))
            bad = copy.deepcopy(raw)
            bad["raw_output"] = " " + raw["raw_output"]
            invalid = score.score_one(row, bad, variant, canonical, gold)
            check(not invalid["valid"] and not invalid["exact"], typ + ":strict_native_box:" + str(index))
            for status in ("BACKEND_ERROR", "NOT_SUBMITTED", "MISSING"):
                failed = score.score_one(row, execute.terminal(row, status), variant, canonical, gold)
                check(not failed["valid"] and not failed["exact"], typ + ":fixed_failure:" + status + ":" + str(index))
                if typ == "multi_select":
                    check(all(item["outcome"] == "INVALID" and item["fp"] is None and item["fn"] is None for item in failed["items"]), typ + ":unknown_not_fake_fpfn:" + status + ":" + str(index))
                else:
                    check(failed["metrics"]["total_pairs"] == 15 and failed["metrics"]["correct_pairs"] == 0, typ + ":invalid_zero_pairs:" + status + ":" + str(index))
            count += 1
        summary = score.summarize_question(question, [correct, invalid, failed])
        check(summary["denominator_calls"] == 3 and summary["valid_calls"] == 1 and summary["exact_rate"] == 1 / 3, typ + ":summary_fixed_denominator")
    output = {"status": "PASS", "cpu_only": True, "synthetic_only": True,
        "diagnostic_questions_opened": False, "private_gold_opened": False, "model_backend_created": False,
        "new_solver_calls": 0, "checks": len(checks), "variant_roundtrips": count,
        "versions": c.rt.versions(), "code_bindings": [c.bind(path) for path in sorted(c.HERE.glob("*.py"))]}
    c.save(args.output, output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
