"""Post-freeze scoring of fixed-citation ordering; no inference or cause admission.

The primary reference is the positions of the same frozen model citations, not
the private answer. All terminal raw responses are bound before Gold is read.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import pathlib
import re
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
ARMS = ("freeze_quotes", "explicit_positions")
LABELS = tuple(f"I{x}" for x in range(1, 7))
REPEATS = tuple(f"r{x:02d}" for x in range(4))
PUBLIC_KEYS = {"candidate_id", "root_id", "question_type", "question", "items", "source_path", "source_sha256"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path):
    path = pathlib.Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def verify_bindings(bindings):
    for item in bindings:
        path = pathlib.Path(item["path"])
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "BOUND_FILE_CHANGED:" + str(path))


def save(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def locate_citations(frozen, source):
    """Locate literal spans independently in source array order, never ref order."""
    require(len(source["turns"]) == len({turn["turn_ref"] for turn in source["turns"]}), "DUPLICATE_SOURCE_TURN_REF")
    turns = {turn["turn_ref"]: (i, turn["text"]) for i, turn in enumerate(source["turns"])}
    evidence = frozen["evidence"]
    require(tuple(entry["label"] for entry in evidence) == LABELS, "FROZEN_EVIDENCE_NOT_LABEL_ORDER")
    positions = []
    for entry in evidence:
        located = turns.get(entry["turn_ref"])
        quote = entry["quote"]
        offsets = []
        if located and isinstance(quote, str) and quote:
            start = 0
            while True:
                offset = located[1].find(quote, start)
                if offset < 0:
                    break
                offsets.append(offset)
                start = offset + 1
        require(located is not None and len(offsets) == 1, "FROZEN_CITATION_NOT_LITERAL_UNIQUE:" + entry["label"])
        positions.append({"label": entry["label"], "source_turn_index": located[0], "quote_start_offset": offsets[0]})
    pairs = [(position["source_turn_index"], position["quote_start_offset"]) for position in positions]
    require(len(set(pairs)) == 6, "FROZEN_CITATION_POSITION_TIE")
    ordered = [position["label"] for position in sorted(positions, key=lambda x: (x["source_turn_index"], x["quote_start_offset"]))]
    if "positions" in frozen:
        require(frozen["positions"] == positions, "FROZEN_POSITION_METADATA_MISMATCH")
    if "citation_sorted_display" in frozen:
        require(frozen["citation_sorted_display"] == ordered, "FROZEN_CITATION_ORDER_MISMATCH")
    return {"positions": positions, "citation_sorted_display": ordered, "semantic_support": "NOT_REVIEWED"}


def parse_strict(raw):
    if raw.get("status") != "RETURNED":
        return None, False, "NOT_RETURNED:" + str(raw.get("status"))
    output = raw.get("raw_output")
    if not isinstance(output, str) or not re.fullmatch(r"<box>I[1-6](?:,I[1-6]){5}</box>", output):
        return None, False, "NOT_EXACT_FINITE_BOX_FORMAT"
    labels = output[5:-6].split(",")
    if len(set(labels)) != 6 or set(labels) != set(LABELS):
        return labels, False, "NOT_COMPLETE_UNIQUE_LABEL_PERMUTATION"
    if raw.get("metadata", {}).get("finish_reason") not in ("stop", "eos"):
        return labels, False, "INVALID_FINISH_REASON:" + str(raw.get("metadata", {}).get("finish_reason"))
    return labels, True, None


def correct_pairs(prediction, target):
    require(len(target) == 6 and len(set(target)) == 6, "REFERENCE_NOT_SIX_UNIQUE_ITEMS")
    rank = {label: index for index, label in enumerate(prediction)}
    return sum(rank[left] < rank[right] for index, left in enumerate(target) for right in target[index + 1:])


def score_one(row, raw, variant, canonical, gold_labels, observation):
    mapping = variant["display_to_content"]
    require(set(mapping) == set(LABELS) and len(set(mapping.values())) == 6, "DISPLAY_MAPPING_NOT_BIJECTION")
    gold = [canonical[label] for label in gold_labels]
    require(set(gold) == set(mapping.values()) and len(gold) == 6, "GOLD_CONTENTS_MISMATCH")
    content_to_display = {content: label for label, content in mapping.items()}
    gold_display = [content_to_display[content] for content in gold]
    citation_display = observation["citation_sorted_display"]
    prediction, valid, error = parse_strict(raw)
    citation_pairs = correct_pairs(prediction, citation_display) if valid else 0
    gold_pairs = correct_pairs(prediction, gold_display) if valid else 0
    return {"request_id": row["request_id"], "candidate_id": row["candidate_id"], "root_id": row["root_id"],
            "probe_arm": row["probe_arm"], "pair_id": row["pair_id"], "frozen_record_id": row["frozen_record_id"],
            "variant_id": row["variant_id"], "repeat_id": row["repeat_id"], "seed": row["seed"],
            "status": raw.get("status"), "valid": valid, "parse_error": error,
            "prediction_display": prediction, "prediction_content": [mapping[x] for x in prediction] if prediction else None,
            "citation_reference_display": citation_display, "gold_reference_display": gold_display,
            "citation_reference_content": [mapping[x] for x in citation_display], "gold_reference_content": gold,
            "citation_order_matches_gold": citation_display == gold_display,
            "citation_exact": bool(valid and prediction == citation_display),
            "citation_correct_pairs": citation_pairs, "citation_pair_accuracy": citation_pairs / 15,
            "gold_exact": bool(valid and prediction == gold_display),
            "gold_correct_pairs": gold_pairs, "gold_pair_accuracy": gold_pairs / 15,
            "total_pairs": 15, "citation_positions": observation["positions"],
            "semantic_support": "NOT_REVIEWED_BY_SCORER",
            "finish_reason": raw.get("metadata", {}).get("finish_reason"),
            "response_committed_at_utc": raw.get("response_committed_at_utc")}


def aggregate(records):
    n = len(records)
    result = {"calls": n, "questions": len({x["candidate_id"] for x in records}),
              "roots": len({x["root_id"] for x in records}),
              "frozen_records": len({x["frozen_record_id"] for x in records}),
              "valid_calls": sum(x["valid"] for x in records), "total_pairs": 15 * n}
    result["invalid_calls"] = n - result["valid_calls"]
    for target in ("citation", "gold"):
        result[target + "_exact_calls"] = sum(x[target + "_exact"] for x in records)
        result[target + "_exact_rate"] = result[target + "_exact_calls"] / n if n else None
        result[target + "_correct_pairs"] = sum(x[target + "_correct_pairs"] for x in records)
        result[target + "_pair_accuracy"] = result[target + "_correct_pairs"] / (15 * n) if n else None
    return result


def paired_contrasts(records):
    grouped = collections.defaultdict(dict)
    for record in records:
        require(record["probe_arm"] not in grouped[record["pair_id"]], "DUPLICATE_PAIRED_ARM")
        grouped[record["pair_id"]][record["probe_arm"]] = record
    output = []
    transitions = {target: collections.Counter() for target in ("citation", "gold")}
    for pair_id, arms in sorted(grouped.items()):
        require(set(arms) == set(ARMS), "MISSING_PAIRED_ARM")
        control, treatment = (arms[arm] for arm in ARMS)
        for key in ("candidate_id", "root_id", "frozen_record_id", "variant_id", "repeat_id", "seed", "citation_reference_display", "gold_reference_display"):
            require(control[key] == treatment[key], "PAIRED_FIELD_DIFFERS:" + key)
        row = {key: control[key] for key in ("pair_id", "candidate_id", "root_id", "frozen_record_id", "variant_id", "repeat_id", "seed")}
        row.update({"freeze_quotes_request_id": control["request_id"], "explicit_positions_request_id": treatment["request_id"],
                    "freeze_quotes_valid": control["valid"], "explicit_positions_valid": treatment["valid"]})
        for target in ("citation", "gold"):
            name = ("correct" if control[target + "_exact"] else "wrong") + "_to_" + ("correct" if treatment[target + "_exact"] else "wrong")
            transitions[target][name] += 1
            row[target + "_transition"] = name
            row[target + "_exact_net_change"] = int(treatment[target + "_exact"]) - int(control[target + "_exact"])
            row[target + "_correct_pair_net_change"] = treatment[target + "_correct_pairs"] - control[target + "_correct_pairs"]
        output.append(row)
    names = ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")
    return {"pairs": len(output), "citation_transitions": {key: transitions["citation"][key] for key in names},
            "gold_transitions": {key: transitions["gold"][key] for key in names}, "records": output}


def summarize(records):
    by_arm = {arm: aggregate([x for x in records if x["probe_arm"] == arm]) for arm in ARMS}
    by_root, by_frozen = [], []
    for key, result in (("root_id", by_root), ("frozen_record_id", by_frozen)):
        for value in sorted({x[key] for x in records}):
            subset = [x for x in records if x[key] == value]
            arms = {arm: aggregate([x for x in subset if x["probe_arm"] == arm]) for arm in ARMS}
            item = {key: value, "root_id": subset[0]["root_id"], **arms}
            for target in ("citation", "gold"):
                item[target + "_exact_rate_difference"] = arms[ARMS[1]][target + "_exact_rate"] - arms[ARMS[0]][target + "_exact_rate"]
                item[target + "_exact_net_change"] = arms[ARMS[1]][target + "_exact_calls"] - arms[ARMS[0]][target + "_exact_calls"]
                item[target + "_correct_pair_net_change"] = arms[ARMS[1]][target + "_correct_pairs"] - arms[ARMS[0]][target + "_correct_pairs"]
            if key == "frozen_record_id":
                item.update({"candidate_id": subset[0]["candidate_id"], "variant_id": subset[0]["variant_id"],
                             "citation_order_matches_gold": subset[0]["citation_order_matches_gold"]})
            result.append(item)
    root_equal = {}
    for target in ("citation", "gold"):
        root_equal[target] = {arm: sum(root[arm][target + "_exact_rate"] for root in by_root) / len(by_root) if by_root else None for arm in ARMS}
        root_equal[target]["difference"] = sum(root[target + "_exact_rate_difference"] for root in by_root) / len(by_root) if by_root else None
    return {"by_arm": by_arm, "by_root": by_root, "by_frozen_record": by_frozen,
            "matched_pairs": paired_contrasts(records), "root_equal_exact": root_equal,
            "primary_root_equal_citation_exact_difference": root_equal["citation"]["difference"],
            "secondary_root_equal_gold_exact_difference": root_equal["gold"]["difference"],
            "primary_endpoint": {"reference": "FROZEN_CITATION_POSITION_ORDER", "unit": "root_equal", "direction": "explicit_positions_minus_freeze_quotes",
                                 "effect_only": True, "statistical_significance_claim": False, "working_support_gate": None,
                                 "cause_auto_published": False}}


def validate_design(manifest, inventory):
    rows = manifest["requests"]
    require(manifest["branch"] == "direct", "DIRECT_BRANCH_REQUIRED")
    require(manifest["request_count"] == 88 and len(rows) == 88, "EXPECTED_88_REQUESTS")
    require(len({x["request_id"] for x in rows}) == 88, "DUPLICATE_REQUEST_ID")
    require(len({x["candidate_id"] for x in rows}) == 5 and len({x["root_id"] for x in rows}) == 5, "EXPECTED_FIVE_QUESTIONS_FIVE_ROOTS")
    frozen = inventory["records"]
    require(len(frozen) == 11 and len(inventory["excluded_records"]) == 43, "EXPECTED_11_SELECTED_43_UNKNOWN")
    require(len({x["frozen_record_id"] for x in frozen}) == 11, "DUPLICATE_FROZEN_RECORD")
    require({x["frozen_record_id"] for x in rows} == {x["frozen_record_id"] for x in frozen}, "FROZEN_RECORD_SET_MISMATCH")
    frozen_by_id = {x["frozen_record_id"]: x for x in frozen}
    require(all(x["question_type"] == "ordering" and x["branch"] == "direct" and x["probe_arm"] in ARMS and x["repeat_id"] in REPEATS for x in rows), "INVALID_DESIGN_ROW")
    for frozen_id, record in frozen_by_id.items():
        subset = [x for x in rows if x["frozen_record_id"] == frozen_id]
        require(len(subset) == 8 and {(x["probe_arm"], x["repeat_id"]) for x in subset} == set(itertools.product(ARMS, REPEATS)), "FROZEN_RECORD_DENOMINATOR_CHANGED")
        for row in subset:
            require(row["pair_id"] == frozen_id + "__" + row["repeat_id"], "PAIR_ID_CHANGED")
            require(row["display"] == record["display"], "FROZEN_DISPLAY_CHANGED")
            require(tuple(row["display"]["items"]) == LABELS and row["max_output_tokens"] == 32, "LABELS_OR_OUTPUT_BUDGET_CHANGED")
            for key in ("candidate_id", "root_id", "variant_id"):
                require(row[key] == record[key], "FROZEN_ID_CHANGED:" + key)
    pairs = collections.defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    require(len(pairs) == 44, "EXPECTED_44_MATCHED_PAIRS")
    for pair in pairs.values():
        require(len(pair) == 2 and {x["probe_arm"] for x in pair} == set(ARMS), "PAIR_ARM_MISMATCH")
        for key in ("candidate_id", "root_id", "variant_id", "repeat_id", "seed", "display", "max_output_tokens", "frozen_record_id"):
            require(pair[0][key] == pair[1][key], "PAIRED_DESIGN_DIFFERS:" + key)


def runtime_cost(run, manifest, raws):
    returned = [raw for raw in raws.values() if raw["status"] == "RETURNED"]
    usage_keys = ("prompt_token_count", "completion_token_count")
    unknown = [raw for raw in raws.values() if raw["status"] == "BACKEND_ERROR" or (raw["status"] == "RETURNED" and any(type(raw.get("metadata", {}).get(key)) is not int for key in usage_keys))]
    def usage(subset, key):
        return sum(raw.get("metadata", {}).get(key) for raw in subset if type(raw.get("metadata", {}).get(key)) is int)
    by_arm = {}
    for arm in ARMS:
        ids = [row["request_id"] for row in manifest["requests"] if row["probe_arm"] == arm]
        subset = [raws[rid] for rid in ids if raws[rid]["status"] == "RETURNED"]
        by_arm[arm] = {"planned_calls": len(ids), "returned_calls": len(subset),
                       "known_prompt_tokens": usage(subset, usage_keys[0]), "known_completion_tokens": usage(subset, usage_keys[1]),
                       "requests_with_unknown_usage": sum(any(type(raw.get("metadata", {}).get(key)) is not int for key in usage_keys) for raw in subset)}
    execution_path = run / ("COMPLETE.json" if (run / "COMPLETE.json").exists() else "FAILED.json")
    execution = read(execution_path) if execution_path.exists() else {}
    prompt, completion = (usage(returned, key) for key in usage_keys)
    return {"planned_calls": len(raws), "returned_calls": len(returned),
            "submitted_attempts": sum(raw["status"] not in ("NOT_SUBMITTED", "MISSING") for raw in raws.values()),
            "not_submitted_calls": sum(raw["status"] == "NOT_SUBMITTED" for raw in raws.values()),
            "missing_terminal_calls": sum(raw["status"] == "MISSING" for raw in raws.values()),
            "requests_with_unknown_usage": len(unknown), "known_prompt_tokens": prompt, "known_completion_tokens": completion,
            "known_total_tokens": prompt + completion, "by_arm": by_arm,
            "live_wall_seconds": execution.get("wall_seconds"), "model_load_wall_seconds": execution.get("model_load_wall_seconds"),
            "execution_binding": bind(execution_path) if execution_path.exists() else None,
            "wall_scope": "Shared local GPU live runtime; includes model load, excludes preparation, audits and reporting. No independent per-arm latency inferred.",
            "currency_cost": None, "currency_cost_reason": "Local GPU inference; monetary billing unavailable. Unknown usage is not counted as zero."}


def score(run):
    run = pathlib.Path(run).resolve()
    require(not (run / "SCORING_STARTED.json").exists() and not (run / "SCORES.json").exists(), "SCORING_ALREADY_STARTED_NO_OVERWRITE")
    manifest = read(run / "manifest.json")
    inventory_path = run / "FROZEN_EVIDENCE.json"
    inventory = read(inventory_path)
    validate_design(manifest, inventory)
    preflight = read(run / "PREFLIGHT_FREEZE.json")
    verify_bindings(preflight["files"])
    for key in ("protocol_binding", "frozen_evidence_binding", "baseline_inventory_binding"):
        verify_bindings([manifest[key]])
    require(pathlib.Path(manifest["frozen_evidence_binding"]["path"]).resolve() == inventory_path, "WRONG_FROZEN_EVIDENCE_BINDING")
    freeze_path = run / "RAW_RESPONSE_FREEZE.json"
    freeze = read(freeze_path)
    require(freeze["status"] == "COMMITTED", "COMMITTED_RAW_FREEZE_REQUIRED")
    verify_bindings(freeze["files"] + [freeze["manifest_binding"]])
    require(pathlib.Path(freeze["manifest_binding"]["path"]).resolve() == run / "manifest.json", "WRONG_FROZEN_MANIFEST")
    expected = {run / "responses" / (row["request_id"] + ".json") for row in manifest["requests"]}
    require(len(freeze["files"]) == 88 and {pathlib.Path(item["path"]).resolve() for item in freeze["files"]} == expected, "RAW_DENOMINATOR_MISMATCH")
    require(set((run / "responses").glob("*.json")) == expected, "UNEXPECTED_RESPONSE_FILES")
    raws = {}
    for row in manifest["requests"]:
        raw = read(run / "responses" / (row["request_id"] + ".json"))
        require(all(raw[key] == row[key] for key in ("request_id", "candidate_id", "root_id")), "RESPONSE_ID_MISMATCH")
        require(raw["status"] in ("RETURNED", "BACKEND_ERROR", "NOT_SUBMITTED", "MISSING"), "NONTERMINAL_RESPONSE_STATUS")
        require(timestamp(raw["response_committed_at_utc"]) <= timestamp(freeze["created_at_utc"]), "RESPONSE_AFTER_FREEZE")
        if raw["status"] == "RETURNED":
            require(all(raw[key] == row[key] for key in ("prompt_sha256", "seed", "variant_id")), "RESPONSE_PROMPT_SEED_VARIANT_MISMATCH")
        raws[row["request_id"]] = raw
    public = {q["candidate_id"]: q for q in read(run / "PUBLIC_QUESTIONS.json")["questions"]}
    maps_path = run / "private/DISPLAY_MAPPINGS.json"
    mappings = read(maps_path)
    observations, frozen_by_id = {}, {record["frozen_record_id"]: record for record in inventory["records"]}
    for frozen_id, record in frozen_by_id.items():
        source_path = run / "sources" / (record["root_id"] + ".json")
        require(sha(source_path) == public[record["candidate_id"]]["source_sha256"], "SOURCE_BYTES_CHANGED")
        verify_bindings([record["raw_binding"], record["source_binding"]])
        source = read(source_path)
        require(source["root_id"] == record["root_id"], "SOURCE_ROOT_MISMATCH")
        observations[frozen_id] = locate_citations(record, source)
    private_binding = manifest["private_answer_binding"]
    verify_bindings([private_binding])
    first_gold = now()
    require(timestamp(first_gold) >= timestamp(freeze["created_at_utc"]), "CLOCK_BEFORE_FREEZE")
    save(run / "SCORING_STARTED.json", {"gold_first_read_at_utc": first_gold,
         "raw_response_freeze_binding": bind(freeze_path), "private_answer_binding": private_binding,
         "all_88_terminal_raws_committed_before_gold": True,
         "gold_read_scope": "First private QA JSON read by this scoring process; prior author, reviewer and parent exposure is not denied."})
    private = {q["candidate_id"]: q for q in read(private_binding["path"])["questions"]}
    gold_read_done = now()
    for candidate_id, question in public.items():
        require({key: private[candidate_id][key] for key in PUBLIC_KEYS if key in private[candidate_id]} == question, "PRIVATE_PUBLIC_QUESTION_MISMATCH")
    records = []
    for row in manifest["requests"]:
        record = score_one(row, raws[row["request_id"]], mappings["requests"][row["request_id"]],
                           mappings["public_to_canonical"][row["candidate_id"]], private[row["candidate_id"]]["gold"], observations[row["frozen_record_id"]])
        record.update({"response_binding": bind(run / "responses" / (row["request_id"] + ".json")),
                       "source_binding": bind(run / "sources" / (row["root_id"] + ".json")),
                       "frozen_evidence_binding": bind(inventory_path), "display_mappings_binding": bind(maps_path),
                       "gold_first_read_at_utc": first_gold})
        records.append(record)
    result = {"status": "COMPLETE" if (run / "COMPLETE.json").exists() else "FAILED_WITH_FIXED_DENOMINATOR",
              "scored_at_utc": now(), "namespace": "train.feedback", "branch": "direct",
              "gold_first_read_at_utc": first_gold, "gold_read_completed_at_utc": gold_read_done,
              "manifest_binding": bind(run / "manifest.json"), "raw_response_freeze_binding": bind(freeze_path),
              "preflight_freeze_binding": bind(run / "PREFLIGHT_FREEZE.json"), "frozen_evidence_binding": bind(inventory_path),
              "public_questions_binding": bind(run / "PUBLIC_QUESTIONS.json"), "private_answer_binding": private_binding,
              "display_mappings_binding": bind(maps_path), "scoring_code_binding": bind(__file__),
              "protocol_binding": manifest["protocol_binding"], "baseline_inventory_binding": manifest["baseline_inventory_binding"],
              "denominator_calls": 88, "denominator_questions": 5, "denominator_roots": 5,
              "denominator_frozen_records": 11, "denominator_matched_pairs": 44,
              "valid_calls": sum(record["valid"] for record in records), "cost": runtime_cost(run, manifest, raws),
              **summarize(records), "requests": records,
              "subset_accounting": {"old_control_records": 54, "included_literal_unique_untied_records": 11,
                                    "excluded_unknown_records": 43, "post_output_subset": True,
                                    "sample_changed_by_gold_or_new_output": False,
                                    "citation_orders_matching_gold": sum(item["citation_order_matches_gold"] for item in summarize(records)["by_frozen_record"])},
              "semantic_support_reviewed_by_scorer": False, "independent_semantic_review": "Separate artifact; not consulted for sample selection or numerical scoring.",
              "cause_auto_published": False, "weakness_memory_upgraded": False,
              "use_as_final_eval": False, "training_started": False, "new_training_questions_generated": 0,
              "limitations": ["Post-output literal-location subset: 11 of 54 old control records from 5 observed Train roots; 43 unknown records retained in inventory, not represented by this estimand.",
                              "Four repeated seeds and multiple permutations per question are paired observations, not independent questions or roots.",
                              "The primary target is the model's fixed literal citation order; semantic support and agreement with the task Gold are separate.",
                              "Explicit coordinates directly supply the positions used in the mechanical primary reference; improvement is task scaffolding, not proof of an isolated latent cause.",
                              "The small selected sample supports exploratory effect reporting only; no significance, cause admission, memory promotion, feedback-training or self-play gain is claimed."]}
    save(run / "SCORES.json", result)
    print(json.dumps({"status": result["status"], "valid": result["valid_calls"], "by_arm": result["by_arm"], "root_equal_exact": result["root_equal_exact"]}, ensure_ascii=False))
    return result


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def report(run, output):
    run, output = pathlib.Path(run).resolve(), pathlib.Path(output).resolve()
    scores_path = run / "SCORES.json"
    scores = read(scores_path)
    verify_bindings([scores[key] for key in ("manifest_binding", "raw_response_freeze_binding", "scoring_code_binding", "frozen_evidence_binding", "protocol_binding")])
    lines = ["# 固定引文与显式位置最小对照", "",
             "本轮固定上轮 control 产生的同一组 6 条引文，在相同完整对话、题目和展示标签下，对比仅给固定引文（freeze_quotes）与额外给核验后位置坐标（explicit_positions）。两臂不重新生成引文，只输出 `<box>I#,...</box>`；同一 720 种合法排列约束、32 token 上限、temperature=0.8、top_p=0.9、4 个配对 seed。", "",
             "样本是事后按字面定位可计算性选出的子集：上轮 54 条 control 记录中，11 条的 6 项引文均逐字唯一且位置无并列，来自 5 个已观察过的 Train root；其余 43 条 unknown 保留在清单中。不是随机抽样，也不代表全部排序题。11 条记录 × 4 次 seed × 2 臂 = 88 次计划作答，重复 seed 和同题多个置换不构成独立题目。", "",
             "主要终点是 **root 等权的自身固定引文位置排序正确率之差**；Gold 准确率仅为次要终点。引文位置按源文件 turns 数组索引，再按引文起始字符偏移排序；并非按 turn_ref 字符串排序。", "",
             "| 臂 | 严格有效 / 计划 | 遵循固定引文顺序 / 计划 | 正确率 | 引文正确成对 / 固定成对 | 成对正确率 | Gold 全对 / 计划 | Gold 正确率 | Gold 成对正确率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = scores["by_arm"][arm]
        lines.append(f"| {arm} | {row['valid_calls']}/{row['calls']} | {row['citation_exact_calls']}/{row['calls']} | {percent(row['citation_exact_rate'])} | {row['citation_correct_pairs']}/{row['total_pairs']} | {percent(row['citation_pair_accuracy'])} | {row['gold_exact_calls']}/{row['calls']} | {percent(row['gold_exact_rate'])} | {percent(row['gold_pair_accuracy'])} |")
    root = scores["root_equal_exact"]
    lines += ["", f"主要终点：freeze_quotes {percent(root['citation'][ARMS[0]])}，explicit_positions {percent(root['citation'][ARMS[1]])}，差值 **{100 * root['citation']['difference']:+.2f} 个百分点**。", "",
              f"次要终点（Gold，root 等权）：{percent(root['gold'][ARMS[0]])} → {percent(root['gold'][ARMS[1]])}，差值 **{100 * root['gold']['difference']:+.2f} 个百分点**。以上为效应描述，不设通过门槛、不作统计显著性声明。", "",
              "| Root | 固定记录数 | 引文顺序：仅引文 | 引文顺序：加位置 | 引文正确率差 | Gold：仅引文 | Gold：加位置 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in scores["by_root"]:
        a, b = (row[arm] for arm in ARMS)
        lines.append(f"| {row['root_id']} | {a['frozen_records']} | {a['citation_exact_calls']}/{a['calls']} | {b['citation_exact_calls']}/{b['calls']} | {100 * row['citation_exact_rate_difference']:+.2f} pp | {a['gold_exact_calls']}/{a['calls']} | {b['gold_exact_calls']}/{b['calls']} |")
    lines += ["", "同一固定引文、展示排列和 seed 的 44 对转移：", "", "| 参照 | 对→对 | 对→错 | 错→对 | 错→错 |", "|---|---:|---:|---:|---:|"]
    for target in ("citation", "gold"):
        t = scores["matched_pairs"][target + "_transitions"]
        lines.append(f"| {target} | {t['correct_to_correct']} | {t['correct_to_wrong']} | {t['wrong_to_correct']} | {t['wrong_to_wrong']} |")
    matching = scores["subset_accounting"]["citation_orders_matching_gold"]
    semantic_path = HERE / "SEMANTIC_REVIEW_ZH.md"
    semantic_link = f"[独立语义审核]({os.path.relpath(semantic_path, output.parent)})" if semantic_path.exists() else "独立语义审核（另行记录）"
    lines += ["", f"11 组固定引文的位置排序中，{matching} 组与 Gold 顺序一致。逐字唯一定位只证明引文在何处，**不证明引文支持选项语义**。评分器未进行语义审核，逐请求记 NOT_REVIEWED_BY_SCORER；{semantic_link}不改变本轮样本、分母或数值评分。", "",
              "| 固定记录（旧 control request_id） | 引文：仅引文 | 引文：加位置 | Gold：仅引文 | Gold：加位置 | 引文顺序等于 Gold |", "|---|---:|---:|---:|---:|---|"]
    for row in scores["by_frozen_record"]:
        a, b = (row[arm] for arm in ARMS)
        lines.append(f"| {row['frozen_record_id']} | {a['citation_exact_calls']}/{a['calls']} | {b['citation_exact_calls']}/{b['calls']} | {a['gold_exact_calls']}/{a['calls']} | {b['gold_exact_calls']}/{b['calls']} | {'是' if row['citation_order_matches_gold'] else '否'} |")
    invalid = [row for row in scores["requests"] if not row["valid"]]
    lines += ["", f"无效作答 {len(invalid)} 次。无效、缺失或未提交请求保留每臂 44 次的固定分母，对引文参照和 Gold 的整题及 15 个成对均计 0。", ""]
    if invalid:
        lines += ["| 请求 | 状态 | 原因 |", "|---|---|---|"]
        for row in invalid:
            reason = str(row["parse_error"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['request_id']} | {row['status']} | {reason} |")
    cost = scores["cost"]
    lines += ["", f"已知输入 {cost['known_prompt_tokens']:,} tokens、输出 {cost['known_completion_tokens']:,} tokens，共 {cost['known_total_tokens']:,} tokens；使用量未知 {cost['requests_with_unknown_usage']} 个请求，未知不记为 0。", "",
              f"模型运行 {cost['live_wall_seconds']} 秒，其中加载 {cost['model_load_wall_seconds']} 秒。两臂共享加载，不推算独立单臂耗时；不含准备、审核及报告时间。金额未知。", "",
              "显式位置直接提供主要机械参照所用坐标，提升只说明该辅助形式在此子集上有用；无法单独证明模型内部错误发生在哪一步，也不能证明针对性出题、memory 或 self-play 训练有效。本轮未生成新题、未训练、未自动发布 cause 或升级弱点 memory。", "",
              f"完整逐请求结果：[SCORES.json]({os.path.relpath(scores_path, output.parent)})；原始返回封存：[RAW_RESPONSE_FREEZE.json]({os.path.relpath(run / 'RAW_RESPONSE_FREEZE.json', output.parent)})；固定引文与 43 条排除清单：[FROZEN_EVIDENCE.json]({os.path.relpath(run / 'FROZEN_EVIDENCE.json', output.parent)})。", "",
              f"输入分数 SHA256：`{sha(scores_path)}`。", ""]
    with output.open("x") as stream:
        stream.write("\n".join(lines))
    return bind(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for command in ("score", "report"):
        item = sub.add_parser(command)
        item.add_argument("--run", type=pathlib.Path, default=HERE / "run")
        if command == "report":
            item.add_argument("--output", type=pathlib.Path, default=HERE / "REPORT_ZH.md")
    args = parser.parse_args()
    if args.cmd == "score":
        score(args.run)
    else:
        print(json.dumps(report(args.run, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
