"""Build immutable, pre-expanded Day 4 schedules without loading model weights."""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
from pathlib import Path

from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent
WEEK = OUT.parent


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def save(name, value):
    path = OUT / name
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_lines(name, records):
    path = OUT / name
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records))


def verify_ref(ref):
    actual = bind(ref["path"])
    assert actual["sha256"] == ref["sha256"], ref["path"]
    assert "bytes" not in ref or actual["bytes"] == ref["bytes"], ref["path"]


def main():
    assert not (OUT / "SELECTION.json").exists(), "Already built; use a new version"
    policy_path = OUT / "EXECUTION_POLICY.json"
    policy = read(policy_path)
    ledger_path = WEEK / "day3_mainline_readiness_v1/training/PAIR88_ELIGIBILITY_AND_FAMILY_LEDGER.json"
    reserve_path = WEEK / "day4_common_set_admission_v1/COMMON_ELIGIBLE_BASE_SET.json"
    sets_decision_path = WEEK / "day4_common_set_admission_v1/DECISION.json"
    old_path = WEEK / "day3_mainline_readiness_v1/training/COMMON_OLD_BASE_SET.json"
    ledger, reserve, sets_decision, old = map(read, [ledger_path, reserve_path, sets_decision_path, old_path])
    assert sets_decision["D4_SETS"].startswith("PASS")
    assert reserve["foundation_eligible_A"] == reserve["foundation_eligible_B"]
    assert reserve["legacy_replay_selected_count"] == 0
    verify_ref(reserve["admission_ledger_binding"])
    pairs = ledger["pairs"]
    assert len(pairs) == 88 and all(p["candidate_pair_CPU_usable"] for p in pairs)
    details = {r["candidate_id"]: r for r in ledger["candidate_records"]}
    private = {}
    for directory in ["day3_production_v1", "day3_production_batch2_v1"]:
        private.update({r["candidate_id"]: r for r in lines(WEEK / directory / "exports/content_accepted.private.jsonl")})
    foundation_path = Path(old["foundation"]["candidate_data_binding"]["path"])
    lineage_path = Path(old["foundation"]["lineage_binding"]["path"])
    verify_ref(old["foundation"]["candidate_data_binding"])
    verify_ref(old["foundation"]["lineage_binding"])
    foundations = {r["candidate_id"]: r for r in lines(foundation_path)}
    lineage = {r["candidate_id"]: r for r in lines(lineage_path)}
    tok = AutoTokenizer.from_pretrained(policy["serialization"]["base_tokenizer"], local_files_only=True,
                                       trust_remote_code=False)
    seed = policy["serialization"]["display_seed"]
    max_length = policy["serialization"]["full_input_max_tokens"]
    epoch_count = policy["serialization"]["epochs"]
    suffix = "\n\nOutput only one <box>... </box> answer. For multi-select use comma-separated letters; for ordering use comma-separated item IDs."
    system = "You are a strict structured dialogue QA solver. Use only the supplied dialogue when it is present. Return the final answer in exactly one <box>...</box> pair. Do not provide explanations, reasoning, or any text outside the box."
    input_refs = {str(p.resolve()): bind(p) for p in [policy_path, ledger_path, reserve_path,
                  sets_decision_path, old_path, foundation_path, lineage_path]}
    mapping_rows, all_token_rows = [], []
    source_cache = {}

    def measure(messages):
        prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                       enable_thinking=False)
        pi = tok(prompt, add_special_tokens=False).input_ids
        fi = tok(full, add_special_tokens=False).input_ids
        if tok.eos_token_id is not None and (not fi or fi[-1] != tok.eos_token_id):
            fi.append(tok.eos_token_id)
        assert fi[:len(pi)] == pi and len(fi) > len(pi), "Prompt/answer mask mismatch"
        return {"full_input_tokens": len(fi), "prompt_tokens": len(pi),
                "supervised_tokens": len(fi) - len(pi), "token_ids_sha256": digest(fi),
                "over_limit": len(fi) > max_length, "zero_truncation": True}

    def expand(cid, key, root_id, qtype, category, options, gold, prefix, arm):
        labels = list(options)
        assert len(labels) in (4, 5, 6) and len(set(labels)) == len(labels)
        assert set(gold) <= set(labels)
        if qtype == "ordering":
            assert len(gold) == len(labels) and len(set(gold)) == len(gold)
        result, previous = [], None
        for epoch in range(1, epoch_count + 1):
            order = sorted(labels, key=lambda label: hashlib.sha256(
                f"{seed}|{epoch}|{key}|{label}".encode()).hexdigest())
            collision_rotated = order == previous
            if collision_rotated:
                order = order[1:] + order[:1]
            previous = order
            dto = dict(zip(labels, order))
            otd = {original: display for display, original in dto.items()}
            mapped_gold = [otd[label] for label in gold]
            if qtype == "multi_select":
                mapped_gold.sort(key=labels.index)
                answer_space = (list(c) for size in range(len(labels) + 1)
                                for c in itertools.combinations(labels, size))
            else:
                answer_space = itertools.permutations(labels)
            cases = 0
            for answer in answer_space:
                assert [dto[otd[label]] for label in answer] == list(answer)
                cases += 1
            recovered = [dto[label] for label in mapped_gold]
            assert recovered == gold if qtype == "ordering" else set(recovered) == set(gold)
            user = prefix + "\n".join(f"{label}. {options[dto[label]]}" for label in labels) + suffix
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user},
                        {"role": "assistant", "content": "<box>" + ",".join(mapped_gold) + "</box>"}]
            tokens = measure(messages)
            record = {"candidate_id": cid, "schedule_key": key, "root_id": root_id, "question_type": qtype,
                      "category": category, "arm": arm, "epoch": epoch, "split": "train",
                      "messages": messages, "dependence_group_id": root_id + ":" + qtype,
                      **{"expected_" + k: v for k, v in tokens.items() if k != "over_limit"},
                      "gold_cardinality": len(gold)}
            result.append(record)
            mapping_rows.append({"candidate_id": cid, "arm": arm, "epoch": epoch, "category": category,
                                 "schedule_key": key, "native_labels": labels, "display_to_original": dto,
                                 "original_to_display": otd, "original_gold": gold, "mapped_gold": mapped_gold,
                                 "collision_rotated": collision_rotated, "all_answer_space_cases": cases,
                                 "all_roundtrips_pass": True})
            all_token_rows.append({"candidate_id": cid, "arm": arm, "epoch": epoch,
                                   "category": category, "messages_sha256": digest(messages), **tokens})
        return result

    new_rows = {"A": [], "B": []}
    for pair in pairs:
        for arm in ("A", "B"):
            cid = pair[arm]
            detail = details[cid]
            row = private[cid]
            for key in ["public", "record", "source"]:
                verify_ref(row[key])
                input_refs[row[key]["path"]] = bind(row[key]["path"])
            pb = read(row["public"]["path"])
            src_path = row["source"]["path"]
            if src_path not in source_cache:
                source_cache[src_path] = read(src_path)
            source = source_cache[src_path]
            dialogue = "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source["turns"])
            prefix = "Dialogue:\n" + dialogue + "\n\nQuestion:\n" + pb["question"] + "\n\n" + (
                "Options:\n" if row["question_type"] == "multi_select" else "Items:\n")
            assert detail["content_and_lineage_status"] == "PASS_CANDIDATE_PREFLIGHT"
            if arm == "B" and row["question_type"] == "ordering":
                assert detail["feedback_lineage"]["qualified_for_narrow_feedback_target"]
            new_rows[arm] += expand(cid, pair["slot_id"], row["root_id"], row["question_type"],
                                    "new_structure", pb["options_or_items"], row["gold"], prefix, arm)

    rank_seed = policy["common"]["selection_seed"]
    rank = sorted(reserve["foundation_eligible_A"], key=lambda cid: (
        hashlib.sha256(f"{rank_seed}|foundation|{cid}".encode()).hexdigest(), cid))
    base_expanded = {}
    for cid in rank:
        row, lin = foundations[cid], lineage[cid]
        assert row["split"] == "train" and row["root_id"] == lin["root_id"]
        assert row["question_type"] == lin["question_type"]
        verify_ref({"path": lin["source_binding"]["path"], "sha256": lin["source_binding"]["sha256"]})
        input_refs[lin["source_binding"]["path"]] = bind(lin["source_binding"]["path"])
        kind = "options" if row["question_type"] == "multi_select" else "items"
        options = lin["display_variant"][kind]
        original_block = "\n".join(f"{label}. {text}" for label, text in options.items())
        original_user = row["messages"][1]["content"]
        expected_tail = original_block + suffix
        assert original_user.endswith(expected_tail), cid
        prefix = original_user[:-len(expected_tail)]
        assert prefix.endswith("Options:\n" if kind == "options" else "Items:\n")
        gold = lin["displayed_gold"]
        assert row["messages"][-1]["content"] == "<box>" + ",".join(gold) + "</box>", cid
        assert row["messages"][0]["content"] == system
        base_expanded[cid] = expand(cid, "foundation|" + cid, row["root_id"], row["question_type"],
                                    "foundation", options, gold, prefix, "COMMON")

    def totals(rows):
        return {key: sum(row["expected_" + key] for row in rows)
                for key in ["full_input_tokens", "prompt_tokens", "supervised_tokens"]}

    new_totals = {arm: totals(rows) for arm, rows in new_rows.items()}
    alternatives = []
    for count in range(4, len(rank) + 1, 4):
        br = [row for cid in rank[:count] for row in base_expanded[cid]]
        bt = totals(br)
        shares = {arm: bt["supervised_tokens"] / (new_totals[arm]["supervised_tokens"] + bt["supervised_tokens"])
                  for arm in ("A", "B")}
        deviation = [abs(share - 0.25) for share in shares.values()]
        alternatives.append({"foundation_count": count, "foundation_tokens": bt, "foundation_shares": shares,
                             "max_target_deviation": max(deviation), "mean_target_deviation": sum(deviation) / 2})
    chosen = min(alternatives, key=lambda x: (x["max_target_deviation"], x["mean_target_deviation"],
                                             x["foundation_count"]))
    selected_base = rank[:chosen["foundation_count"]]
    selected_ids = {"A": {p["A"] for p in pairs}, "B": {p["B"] for p in pairs}, "COMMON": set(selected_base)}
    selected_maps = [r for r in mapping_rows if r["candidate_id"] in selected_ids[r["arm"]]]
    selected_tokens = [r for r in all_token_rows if r["candidate_id"] in selected_ids[r["arm"]]]
    assert all(not row["over_limit"] for row in selected_tokens), "Length gate failed"
    summary, schedules = {}, {}
    order_seed = policy["serialization"]["training_seed"]
    for arm in ("A", "B"):
        schedule = new_rows[arm] + [{**r, "arm": arm} for cid in selected_base for r in base_expanded[cid]]
        schedule.sort(key=lambda r: (r["epoch"], hashlib.sha256(
            f"{order_seed}|{r['epoch']}|{r['schedule_key']}".encode()).hexdigest(), r["schedule_key"]))
        for index, row in enumerate(schedule, 1):
            row["ordinal"] = index
        assert len(schedule) % 4 == 0
        schedules[arm] = schedule
        total = totals(schedule)
        categories = {category: totals([r for r in schedule if r["category"] == category])
                      for category in ("new_structure", "foundation")}
        designs = collections.Counter(details[p[arm]]["design_id"] for p in pairs)
        summary[arm] = {"unique_questions": len({r["candidate_id"] for r in schedule}),
                        "new_structure": len(pairs), "foundation": len(selected_base), "legacy": 0,
                        "question_types_unique": dict(collections.Counter(
                            r["question_type"] for r in schedule if r["epoch"] == 1)),
                        "roots": len({r["root_id"] for r in schedule}),
                        "new_design_counts": dict(designs), "microsteps": len(schedule),
                        "optimizer_steps": len(schedule) // 4, "epochs_preexpanded": epoch_count,
                        "runtime_epochs": 1, "totals": total, "by_category": categories,
                        "supervised_token_shares": {category: v["supervised_tokens"] / total["supervised_tokens"]
                                                    for category, v in categories.items()}}
        assert sum(k.startswith("MC_") for k in designs) >= 2
        assert sum(k.startswith("ORDER_") for k in designs) >= 2
    assert [(r["epoch"], r["schedule_key"]) for r in schedules["A"]] == [
        (r["epoch"], r["schedule_key"]) for r in schedules["B"]]
    input_diff = abs(summary["A"]["totals"]["full_input_tokens"] - summary["B"]["totals"]["full_input_tokens"]) / max(
        summary[a]["totals"]["full_input_tokens"] for a in ("A", "B"))
    assert input_diff <= policy["budget"]["full_input_token_relative_difference_max"]
    assert summary["A"]["optimizer_steps"] == summary["B"]["optimizer_steps"]
    for arm, rows in schedules.items():
        write_lines(f"TRAIN_POOL_{arm}.jsonl", rows)
    save("SELECTION.json", {"status": "ACTUAL_POOLS_SELECTED_SCORE_INDEPENDENT",
         "policy": bind(policy_path), "paired_slots": pairs, "foundation_hash_rank": rank,
         "prefix_alternatives": alternatives, "selected_prefix": chosen, "selected_foundation": selected_base,
         "selected_new_A": [p["A"] for p in pairs], "selected_new_B": [p["B"] for p in pairs],
         "legacy_selected": [], "q": 0.0, "unmatched_not_selected": 11,
         "partial_overlap_policy": "retained with inherited root/type and upstream fact-family dependencies",
         "input_bindings": list(input_refs.values())})
    save("DISPLAY_MAP_CHECKS.json", {"status": "PASS", "records": selected_maps,
         "answer_space_roundtrips": sum(r["all_answer_space_cases"] for r in selected_maps),
         "original_arity_and_labels_preserved": True, "gold_independent": True,
         "two_distinct_displays_per_question": True, "not_new_independent_questions": True})
    save("TOKENIZATION_ZERO_TRUNCATION.json", {"status": "PASS", "records": selected_tokens,
         "tokenizer_directory": str(Path(policy["serialization"]["base_tokenizer"])),
         "max_length": max_length, "full_input_tokens_min": min(r["full_input_tokens"] for r in selected_tokens),
         "full_input_tokens_max": max(r["full_input_tokens"] for r in selected_tokens), "length_holds": [],
         "prompt_is_exact_full_token_prefix": True, "zero_truncation": True,
         "eos": "append eos iff final full token is not eos; inherited native training tokenization"})
    save("SCHEDULE_SUMMARY.json", {"status": "INPUTS_FROZEN_PENDING_INDEPENDENT_AND_RUNTIME_CHECKS",
         "arms": summary, "relative_full_input_token_difference": input_diff, "q": 0.0,
         "target_supervised_token_shares": policy["common"]["target_supervised_token_shares"],
         "new_solver_calls": 0, "new_training_runs": 0, "model_weights_loaded": False,
         "training_start_go": False, "formal_execution_requires": "Day5 isolated longest-input smoke and final artifact checks",
         "script": bind(__file__), "policy": bind(policy_path),
         "train_pools": {a: bind(OUT / f"TRAIN_POOL_{a}.jsonl") for a in ("A", "B")}})
    print(json.dumps({"arms": summary, "input_token_difference": input_diff}, ensure_ascii=False))


if __name__ == "__main__":
    main()
