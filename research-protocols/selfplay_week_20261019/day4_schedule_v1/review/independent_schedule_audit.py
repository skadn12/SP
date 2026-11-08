"""Independent CPU audit; does not import the schedule builder or load weights."""
from __future__ import annotations

import collections
import datetime
import hashlib
import importlib.metadata
import itertools
import json
import math
import pathlib
import sys

from transformers import AutoTokenizer

REVIEW = pathlib.Path(__file__).resolve().parent
OUT = REVIEW.parent
WEEK = OUT.parent
CHECKS = collections.Counter()
REFS = {}


def require(condition, label):
    if not condition:
        raise AssertionError(label)
    CHECKS[label] += 1


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def binding(path):
    path = pathlib.Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    ref = {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}
    REFS[str(path)] = ref
    return ref


def read(path):
    binding(path)
    return json.loads(pathlib.Path(path).read_text())


def jsonl(path):
    binding(path)
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def verify(ref):
    actual = binding(ref["path"])
    require(actual["sha256"] == ref["sha256"], "input_hash_matches")
    size = ref.get("bytes", ref.get("size_bytes"))
    if size is not None:
        require(actual["bytes"] == size, "input_size_matches")


def audit():
    policy = read(OUT / "EXECUTION_POLICY.json")
    selection = read(OUT / "SELECTION.json")
    summary = read(OUT / "SCHEDULE_SUMMARY.json")
    mapping_report = read(OUT / "DISPLAY_MAP_CHECKS.json")
    token_report = read(OUT / "TOKENIZATION_ZERO_TRUNCATION.json")
    ledger = read(WEEK / "day3_mainline_readiness_v1/training/PAIR88_ELIGIBILITY_AND_FAMILY_LEDGER.json")
    reserve = read(WEEK / "day4_common_set_admission_v1/COMMON_ELIGIBLE_BASE_SET.json")
    sets = read(WEEK / "day4_common_set_admission_v1/DECISION.json")
    old = read(WEEK / "day3_mainline_readiness_v1/training/COMMON_OLD_BASE_SET.json")
    require(sets["D4_SETS"].startswith("PASS"), "common_reserve_admitted")
    require(reserve["foundation_eligible_A"] == reserve["foundation_eligible_B"], "common_reserve_identical")
    require(selection["q"] == summary["q"] == policy["common"]["q"] == 0.0, "q_zero")
    require(selection["legacy_selected"] == [] and reserve["legacy_replay_selected_count"] == 0,
            "no_unadmitted_legacy_replay")
    for ref in selection["input_bindings"]:
        verify(ref)
    for ref in [selection["policy"], summary["script"], summary["policy"], *summary["train_pools"].values()]:
        verify(ref)
    require(summary["new_solver_calls"] == summary["new_training_runs"] == 0, "no_new_solver_or_training_claim")
    require(summary["training_start_go"] is False and summary["model_weights_loaded"] is False,
            "no_gpu_or_training_go_claim")

    tok = AutoTokenizer.from_pretrained(policy["serialization"]["base_tokenizer"],
                                       local_files_only=True, trust_remote_code=False)
    limit = policy["serialization"]["full_input_max_tokens"]
    require(limit == 12288, "inherited_length_limit")
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]:
        path = pathlib.Path(policy["serialization"]["base_tokenizer"]) / name
        if path.exists():
            binding(path)

    def tokens(messages):
        prompt_text = tok.apply_chat_template(messages[:-1], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        all_text = tok.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=False, enable_thinking=False)
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        full_ids = tok.encode(all_text, add_special_tokens=False)
        if tok.eos_token_id is not None and (not full_ids or full_ids[-1] != tok.eos_token_id):
            full_ids += [tok.eos_token_id]
        require(full_ids[:len(prompt_ids)] == prompt_ids, "prompt_exact_token_prefix")
        require(len(full_ids) > len(prompt_ids), "nonempty_answer_supervision")
        require(len(full_ids) <= limit, "full_input_length_pass_no_truncation")
        return {"full_input_tokens": len(full_ids), "prompt_tokens": len(prompt_ids),
                "supervised_tokens": len(full_ids) - len(prompt_ids),
                "token_ids_sha256": canonical_hash(full_ids), "zero_truncation": True}

    pairs = ledger["pairs"]
    details = {r["candidate_id"]: r for r in ledger["candidate_records"]}
    require(len(pairs) == 88 and selection["paired_slots"] == pairs, "all_88_matched_pairs_unchanged")
    require(all(p["candidate_pair_CPU_usable"] for p in pairs), "matched_pair_preflight_pass")
    originals = {}
    for folder in ["day3_production_v1", "day3_production_batch2_v1"]:
        for r in jsonl(WEEK / folder / "exports/content_accepted.private.jsonl"):
            require(r["candidate_id"] not in originals, "candidate_identity_unique")
            originals[r["candidate_id"]] = r
    foundations = {r["candidate_id"]: r for r in jsonl(old["foundation"]["candidate_data_binding"]["path"])}
    lineages = {r["candidate_id"]: r for r in jsonl(old["foundation"]["lineage_binding"]["path"])}
    rank = sorted(reserve["foundation_eligible_A"],
                  key=lambda cid: (sha(f"{policy['common']['selection_seed']}|foundation|{cid}"), cid))
    require(selection["foundation_hash_rank"] == rank, "foundation_hash_rank_reproduced")

    system = "You are a strict structured dialogue QA solver. Use only the supplied dialogue when it is present. Return the final answer in exactly one <box>...</box> pair. Do not provide explanations, reasoning, or any text outside the box."
    suffix = "\n\nOutput only one <box>... </box> answer. For multi-select use comma-separated letters; for ordering use comma-separated item IDs."
    expected = {}
    expected_maps = {}
    source_cache = {}
    ordering_anchor_count = 0

    def source(ref):
        verify(ref)
        if ref["path"] not in source_cache:
            source_cache[ref["path"]] = read(ref["path"])
        return source_cache[ref["path"]]

    def dialogue_text(src):
        return "\n".join("[" + turn["turn_ref"] + "] " + turn["speaker"] + ": " + turn["text"]
                         for turn in src["turns"])

    def create(cid, key, root, kind, category, options, gold, prefix, arm):
        labels = list(options)
        require(len(labels) in [4, 5, 6] and len(labels) == len(set(labels)), "native_labels_and_arity")
        require(len(set(gold)) == len(gold) and set(gold) <= set(labels), "original_gold_valid")
        if kind == "ordering":
            require(set(gold) == set(labels), "ordering_all_items_gold")
        prior = None
        rows = []
        for epoch in [1, 2]:
            order = sorted(labels, key=lambda label: sha(f"{policy['serialization']['display_seed']}|{epoch}|{key}|{label}"))
            rotated = prior == order
            if rotated:
                order = order[1:] + order[:1]
            require(prior != order, "epoch_displays_distinct")
            prior = order
            dto = dict(zip(labels, order))
            inverse = {original: display for display, original in dto.items()}
            mapped = [inverse[g] for g in gold]
            if kind == "multi_select":
                mapped = sorted(mapped, key=labels.index)
                recovered = {dto[g] for g in mapped}
                require(recovered == set(gold), "multiselect_gold_content_preserved")
                count = 2 ** len(labels)
                space = itertools.chain.from_iterable(itertools.combinations(labels, n) for n in range(len(labels) + 1))
            else:
                require([dto[g] for g in mapped] == gold, "ordering_gold_semantics_preserved")
                count = math.factorial(len(labels))
                space = itertools.permutations(labels)
            actual_cases = 0
            for answer in space:
                require(tuple(dto[inverse[g]] for g in answer) == tuple(answer), "complete_answer_space_roundtrip")
                actual_cases += 1
            require(actual_cases == count, "answer_space_count")
            user = prefix + "\n".join(label + ". " + options[dto[label]] for label in labels) + suffix
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user},
                        {"role": "assistant", "content": "<box>" + ",".join(mapped) + "</box>"}]
            mt = tokens(messages)
            row = {"candidate_id": cid, "schedule_key": key, "root_id": root, "question_type": kind,
                   "category": category, "arm": arm, "epoch": epoch, "split": "train", "messages": messages,
                   "dependence_group_id": root + ":" + kind, "gold_cardinality": len(gold),
                   **{"expected_" + k: v for k, v in mt.items()}}
            key_tuple = (cid, arm, epoch)
            require(key_tuple not in expected, "display_identity_unique")
            expected[key_tuple] = row
            expected_maps[key_tuple] = {"candidate_id": cid, "arm": arm, "epoch": epoch, "category": category,
                                       "schedule_key": key, "native_labels": labels, "display_to_original": dto,
                                       "original_to_display": inverse, "original_gold": gold, "mapped_gold": mapped,
                                       "collision_rotated": rotated, "all_answer_space_cases": count,
                                       "all_roundtrips_pass": True}
            rows.append(row)
        return rows

    new_rows = {"A": [], "B": []}
    for pair in pairs:
        for arm in ["A", "B"]:
            cid = pair[arm]
            original = originals[cid]
            for tag in ["public", "record", "source"]:
                verify(original[tag])
            public = read(original["public"]["path"])
            record = read(original["record"]["path"])
            src = source(original["source"])
            require(record["qa_review"]["status"] == "PASS", "inherited_content_qa_pass")
            require(details[cid]["content_and_lineage_status"] == "PASS_CANDIDATE_PREFLIGHT", "new_candidate_eligibility")
            ev = {e["ref_id"]: e for e in record["evidence_span"]["evidence_refs"]}
            turns = {t["turn_ref"]: (n, t) for n, t in enumerate(src["turns"])}
            for evidence in ev.values():
                ref = evidence.get("turn_id", evidence.get("turn_ref"))
                require(ref in turns and evidence["quote"] in turns[ref][1]["text"], "source_evidence_quote_exact")
            if original["question_type"] == "multi_select":
                gold = [o["option_id"] for o in record["multi_select"]["options"] if o["source_verdict"] == "SUPPORTED"]
                require(set(gold) == set(original["gold"]), "gold_matches_frozen_option_verdicts")
            else:
                gold = record["ordering"]["gold_item_ids"]
                selected = [(turns[ev[item["selected_ref_id"]]["turn_id"]][0], item["item_id"])
                            for item in record["ordering"]["items"]]
                require(len({pos for pos, _ in selected}) == len(selected), "ordering_selected_source_positions_unique")
                require([item for _, item in sorted(selected)] == gold, "ordering_gold_matches_selected_source_order")
                ordering_anchor_count += len(selected)
                require(gold == original["gold"], "ordering_gold_matches_frozen_record")
                if arm == "B":
                    require(details[cid]["feedback_lineage"]["qualified_for_narrow_feedback_target"], "B_ordering_feedback_qualified")
            prefix = "Dialogue:\n" + dialogue_text(src) + "\n\nQuestion:\n" + public["question"] + "\n\n" + (
                "Options:\n" if original["question_type"] == "multi_select" else "Items:\n")
            new_rows[arm].extend(create(cid, pair["slot_id"], original["root_id"], original["question_type"],
                                        "new_structure", public["options_or_items"], original["gold"], prefix, arm))
    base_rows = {}
    for cid in rank:
        row, lin = foundations[cid], lineages[cid]
        src = source(lin["source_binding"])
        require(row["split"] == "train", "foundation_train_split")
        require(row["root_id"] == lin["root_id"] == src["root_id"], "foundation_root_lineage")
        require(row["question_type"] == lin["question_type"], "foundation_type_lineage")
        original_user = row["messages"][1]["content"]
        lead = "Dialogue:\n" + dialogue_text(src) + "\n\nQuestion:\n"
        require(original_user.startswith(lead), "foundation_full_source_exact")
        kind = row["question_type"]
        options = lin["display_variant"]["options" if kind == "multi_select" else "items"]
        block = "\n".join(label + ". " + text for label, text in options.items()) + suffix
        require(original_user.endswith(block), "foundation_options_exact_to_lineage")
        prefix = original_user[:-len(block)]
        require(prefix.endswith("Options:\n" if kind == "multi_select" else "Items:\n"), "foundation_prefix_preserved")
        gold = lin["displayed_gold"]
        require(row["messages"][-1]["content"] == "<box>" + ",".join(gold) + "</box>", "foundation_gold_original_messages")
        canonical = [lin["display_to_canonical"][g] for g in gold]
        require(set(canonical) == set(lin["canonical_gold"]) if kind == "multi_select"
                else canonical == lin["canonical_gold"], "foundation_gold_canonical_roundtrip")
        require(row["messages"][0]["content"] == system, "foundation_system_preserved")
        base_rows[cid] = create(cid, "foundation|" + cid, row["root_id"], kind, "foundation", options, gold, prefix, "COMMON")

    def total(rows):
        return {field: sum(r["expected_" + field] for r in rows)
                for field in ["full_input_tokens", "prompt_tokens", "supervised_tokens"]}

    new_totals = {arm: total(rows) for arm, rows in new_rows.items()}
    alternatives = []
    for count in range(1, len(rank) + 1):
        if (88 + count) % policy["budget"]["gradient_accumulation"]:
            continue
        bt = total([r for cid in rank[:count] for r in base_rows[cid]])
        shares = {a: bt["supervised_tokens"] / (new_totals[a]["supervised_tokens"] + bt["supervised_tokens"])
                  for a in ["A", "B"]}
        deviations = [abs(shares[a] - 0.25) for a in ["A", "B"]]
        alternatives.append({"foundation_count": count, "foundation_tokens": bt, "foundation_shares": shares,
                             "max_target_deviation": max(deviations), "mean_target_deviation": sum(deviations) / 2})
    require(alternatives == selection["prefix_alternatives"], "all_prefix_alternatives_independently_recomputed")
    choice = min(alternatives, key=lambda alt: (alt["max_target_deviation"], alt["mean_target_deviation"], alt["foundation_count"]))
    require(choice == selection["selected_prefix"], "minimax_prefix_selection_reproduced")
    selected_base = rank[:choice["foundation_count"]]
    require(selected_base == selection["selected_foundation"], "selected_foundation_exact_prefix")
    allowed_ids = {"A": set(selection["selected_new_A"]), "B": set(selection["selected_new_B"]), "COMMON": set(selected_base)}
    for arm in ["A", "B"]:
        require(allowed_ids[arm] == {p[arm] for p in pairs}, "all_new_ids_selected_without_scores")

    selected_expected_maps = {k: v for k, v in expected_maps.items() if k[0] in allowed_ids[k[1]]}
    reported_maps = {(r["candidate_id"], r["arm"], r["epoch"]): r for r in mapping_report["records"]}
    require(reported_maps == selected_expected_maps, "selected_map_report_exact_reconstruction")
    require(mapping_report["answer_space_roundtrips"] == sum(r["all_answer_space_cases"] for r in reported_maps.values()),
            "selected_answer_space_total")
    reported_tokens = {(r["candidate_id"], r["arm"], r["epoch"]): r for r in token_report["records"]}
    require(set(reported_tokens) == set(selected_expected_maps), "selected_token_rows_complete")
    for key, rt in reported_tokens.items():
        er = expected[key]
        for field in ["full_input_tokens", "prompt_tokens", "supervised_tokens", "token_ids_sha256", "zero_truncation"]:
            require(rt[field] == er["expected_" + field], "token_report_exact_reconstruction")
        require(rt["messages_sha256"] == canonical_hash(er["messages"]) and rt["over_limit"] is False,
                "token_message_binding_and_length")

    schedules = {}
    arm_results = {}
    for arm in ["A", "B"]:
        expected_schedule = new_rows[arm] + [{**r, "arm": arm} for cid in selected_base for r in base_rows[cid]]
        expected_schedule.sort(key=lambda r: (r["epoch"], sha(f"{policy['serialization']['training_seed']}|{r['epoch']}|{r['schedule_key']}"), r["schedule_key"]))
        for n, row in enumerate(expected_schedule, 1):
            row["ordinal"] = n
        schedules[arm] = jsonl(OUT / f"TRAIN_POOL_{arm}.jsonl")
        require(schedules[arm] == expected_schedule, "entire_schedule_exact_source_gold_messages_metadata_tokens_order")
        counts = collections.Counter((r["candidate_id"], r["epoch"]) for r in schedules[arm])
        require(set(counts.values()) == {1}, "once_per_question_per_epoch")
        require({r["epoch"] for r in schedules[arm]} == {1, 2}, "exactly_two_preexpanded_epochs")
        totals = total(schedules[arm])
        epoch_counts = collections.Counter(r["epoch"] for r in schedules[arm])
        require(all(n % 4 == 0 for n in epoch_counts.values()), "complete_accumulation_each_epoch")
        num = len(schedules[arm])
        require(summary["arms"][arm]["microsteps"] == num and summary["arms"][arm]["optimizer_steps"] == num // 4,
                "summary_equal_exact_steps")
        require(summary["arms"][arm]["totals"] == totals, "summary_tokens_exact")
        designs = collections.Counter(details[p[arm]]["design_id"] for p in pairs)
        require(sum(k.startswith("MC_") for k in designs) >= 2 and sum(k.startswith("ORDER_") for k in designs) >= 2,
                "new_diversity_gate_preserved")
        require(summary["arms"][arm]["new_design_counts"] == dict(designs), "summary_design_counts_exact")
        for category in ["new_structure", "foundation"]:
            require(summary["arms"][arm]["by_category"][category] == total([r for r in schedules[arm] if r["category"] == category]),
                    "summary_category_tokens_exact")
        arm_results[arm] = {"unique_questions": len(counts) // 2, "new_structure": 88,
                            "foundation": len(selected_base), "microsteps": num, "optimizer_steps": num // 4,
                            "full_input_tokens": totals["full_input_tokens"], "supervised_tokens": totals["supervised_tokens"],
                            "foundation_supervised_share": choice["foundation_shares"][arm]}
    require([(r["epoch"], r["schedule_key"]) for r in schedules["A"]] == [(r["epoch"], r["schedule_key"]) for r in schedules["B"]],
            "identical_pairwise_schedule_order")
    common_a = [{k: v for k, v in r.items() if k != "arm"} for r in schedules["A"] if r["category"] == "foundation"]
    common_b = [{k: v for k, v in r.items() if k != "arm"} for r in schedules["B"] if r["category"] == "foundation"]
    require(common_a == common_b, "foundation_id_messages_tokens_identical_across_arms")
    diff = abs(arm_results["A"]["full_input_tokens"] - arm_results["B"]["full_input_tokens"]) / max(r["full_input_tokens"] for r in arm_results.values())
    require(diff <= 0.05 and diff == summary["relative_full_input_token_difference"], "complete_input_difference_le5percent")
    require(policy["serialization"]["runtime_epochs_over_preexpanded_data"] == 1 and
            policy["serialization"]["runtime_shuffle"] is False and policy["serialization"]["runtime_permutation"] is False,
            "no_runtime_epoch_or_permutation_double_count")
    return {"arms": arm_results, "relative_input_token_difference": diff,
            "foundation_candidates_measured": len(rank), "prefix_alternatives_verified": len(alternatives),
            "selected_mapping_records": len(reported_maps), "actual_training_records_both_arms": sum(len(r) for r in schedules.values()),
            "new_ordering_selected_anchors_checked": ordering_anchor_count,
            "selected_input_length_min": min(r["expected_full_input_tokens"] for rows in schedules.values() for r in rows),
            "selected_input_length_max": max(r["expected_full_input_tokens"] for rows in schedules.values() for r in rows)}


def main():
    report = {"schema_version": "independent_day4_schedule_audit_v1",
              "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "auditor": "/root/day4_contract_review", "status": "FAIL",
              "environment": {"python_executable": sys.executable, "python": sys.version,
                              **{name: importlib.metadata.version(name) for name in ["transformers", "tokenizers"]}},
              "implementation_independence": "Does not import builder; independently reconstructs source, options, Gold mapping, display permutations, token streams, prefix selection and ordering.",
              "historical_environment_note": "This build/audit uses transformers 4.57.3, whereas the inherited TP88 CPU preflight recorded 5.9.0. New files are verified with the actual current tokenizer implementation; GPU runtime remains untested.",
              "scope_limits": ["Mechanical replay audit retains existing semantic QA; it is not a fresh semantic truth review of every accepted question.",
                               "No Solver scores are read. No model weights, inference, training, GPU smoke, or causal-effect test is executed.",
                               "Source ordering checks use frozen selected evidence anchors and do not independently redo all semantic first-mention searches.",
                               "D4-SCHEDULE success is readiness for the separately required Day5 runtime smoke, not training GO."],
              "new_solver_calls": 0, "new_training_runs": 0, "model_weights_loaded": False,
              "gpu_runtime_verified": False, "training_start_go": False}
    try:
        report["results"] = audit()
        report["status"] = "PASS_INDEPENDENT_CPU_SCHEDULE_REPLAY"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["checks"] = dict(CHECKS)
    report["total_check_evaluations"] = sum(CHECKS.values())
    report["script_binding"] = binding(__file__)
    report["input_bindings"] = list(REFS.values())
    destination = REVIEW / "INDEPENDENT_SCHEDULE_AUDIT.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "error": report.get("error"), "results": report.get("results"),
                      "total_check_evaluations": report["total_check_evaluations"]}, ensure_ascii=False))
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
