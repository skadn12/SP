"""Strict, fixed-denominator scoring for the paired ordering construction probe.

The scoring command is a separate post-freeze process. It first checks every
terminal response and commits SCORING_STARTED, then opens private Gold JSON.
No inference and no semantic admission are performed by this module.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "run"
ARMS = ("control", "treatment")
COHORTS = ("discovery", "replication")
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


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("DUPLICATE_JSON_OBJECT_KEY:" + key)
        obj[key] = value
    return obj


def parse_strict(raw, row):
    """Use the frozen strict schema, including uniqueItems, after generation."""
    import jsonschema
    if raw.get("status") != "RETURNED":
        return None, False, "NOT_RETURNED:" + str(raw.get("status"))
    try:
        obj = json.loads(raw.get("raw_output"), object_pairs_hook=_unique_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError("NONFINITE_JSON:" + value)))
        jsonschema.validate(obj, row["json_schema"])
        labels = list(row["display"]["items"])
        require(len(labels) == 6 and len(set(labels)) == 6, "EXPECTED_SIX_DISPLAY_LABELS")
        require(len(obj["evidence"]) == 6 and {entry["label"] for entry in obj["evidence"]} == set(labels), "EVIDENCE_LABELS_NOT_COMPLETE_UNIQUE")
        require(len(obj["answer"]) == 6 and set(obj["answer"]) == set(labels), "ANSWER_NOT_COMPLETE_UNIQUE")
    except Exception as error:
        return None, False, "SCHEMA_OR_PARSE:" + type(error).__name__ + ":" + str(error)
    if raw.get("metadata", {}).get("finish_reason") not in ("stop", "eos"):
        return obj, False, "INVALID_FINISH_REASON:" + str(raw.get("metadata", {}).get("finish_reason"))
    return obj, True, None


def citation_observation(structured, source, display_to_content, gold):
    """Verbatim positions establish location only, never semantic support."""
    if structured is None:
        return {"computable": False, "entries": [], "reason": "NO_STRICT_SCHEMA_OBJECT",
                "citation_sorted_display": None, "citation_sorted_content": None,
                "answer_consistent_with_own_citation_positions": None,
                "citation_position_order_matches_gold": None, "semantic_support": "NOT_REVIEWED"}
    turns = {turn["turn_ref"]: (index, turn) for index, turn in enumerate(source["turns"])}
    entries = []
    for item in structured["evidence"]:
        located = turns.get(item["turn_ref"])
        quote = item["quote"]
        offsets = []
        if located and quote:
            start = 0
            while True:
                offset = located[1]["text"].find(quote, start)
                if offset < 0:
                    break
                offsets.append(offset)
                start = offset + 1
        entries.append({"display_label": item["label"], "content_id": display_to_content[item["label"]],
                        "declared_status": item["status"], "turn_ref": item["turn_ref"], "quote": quote,
                        "quote_exact_in_cited_turn": bool(offsets), "exact_occurrence_count": len(offsets),
                        "source_turn_index": located[0] if located else None, "quote_start_offsets": offsets,
                        "position": [located[0], offsets[0]] if located and len(offsets) == 1 else None,
                        "semantic_support": "NOT_REVIEWED"})
    positions = [tuple(entry["position"]) for entry in entries if entry["position"] is not None]
    computable = len(entries) == 6 and len(positions) == 6 and len(set(positions)) == 6
    ordered = sorted(entries, key=lambda entry: entry["position"]) if computable else []
    display = [entry["display_label"] for entry in ordered] if computable else None
    content = [entry["content_id"] for entry in ordered] if computable else None
    return {"computable": computable, "entries": entries,
            "reason": None if computable else "NONEXACT_NONUNIQUE_OR_TIED_POSITIONS",
            "citation_sorted_display": display, "citation_sorted_content": content,
            "answer_consistent_with_own_citation_positions": structured["answer"] == display if computable else None,
            "citation_position_order_matches_gold": content == gold if computable else None,
            "semantic_support": "NOT_REVIEWED"}


def score_one(row, raw, variant, canonical, gold_labels, source):
    gold = [canonical[label] for label in gold_labels]
    require(len(gold) == 6 and len(set(gold)) == 6, "GOLD_NOT_SIX_UNIQUE_CONTENTS")
    mapping = variant["display_to_content"]
    require(set(mapping) == set(row["display"]["items"]) and set(mapping.values()) == set(gold), "DISPLAY_MAPPING_NOT_BIJECTION")
    structured, valid, error = parse_strict(raw, row)
    prediction = [mapping[label] for label in structured["answer"]] if structured else None
    pair_correct = 0
    if valid:
        rank = {content: index for index, content in enumerate(prediction)}
        pair_correct = sum(rank[left] < rank[right] for index, left in enumerate(gold) for right in gold[index + 1:])
    observation = citation_observation(structured, source, mapping, gold)
    derived_exact = observation["citation_position_order_matches_gold"] if valid and observation["computable"] else None
    return {"request_id": row["request_id"], "candidate_id": row["candidate_id"], "root_id": row["root_id"],
            "probe_arm": row["probe_arm"], "cohort": row["cohort"], "pair_id": row["pair_id"],
            "variant_id": row["variant_id"], "seed": row["seed"], "status": raw.get("status"),
            "valid": valid, "exact": bool(valid and prediction == gold), "parse_error": error,
            "prediction_display": structured["answer"] if structured else None, "prediction_content": prediction,
            "gold_content": gold, "correct_pairs": pair_correct, "total_pairs": 15,
            "pair_accuracy": pair_correct / 15, "citation_observation": observation,
            "external_citation_sort": {"status": "KNOWN" if derived_exact is not None else "UNKNOWN",
                                       "exact": derived_exact,
                                       "rescued_wrong_to_right": bool(derived_exact is True and not (valid and prediction == gold)),
                                       "changed_exact_to_wrong": bool(derived_exact is False and valid and prediction == gold),
                                       "interpretation": "Deterministic external sorting of the model's cited spans; not the model answer or model ability. Semantic support not reviewed."},
            "format_finish_reason": raw.get("metadata", {}).get("finish_reason"),
            "response_committed_at_utc": raw.get("response_committed_at_utc")}


def aggregate(records):
    n = len(records)
    valid = sum(record["valid"] for record in records)
    exact = sum(record["exact"] for record in records)
    pair_correct = sum(record["correct_pairs"] for record in records)
    pair_total = sum(record["total_pairs"] for record in records)
    computable = [record for record in records if record["valid"] and record["citation_observation"]["computable"]]
    return {"calls": n, "questions": len({record["candidate_id"] for record in records}),
            "roots": len({record["root_id"] for record in records}), "valid_calls": valid,
            "invalid_calls": n - valid, "exact_calls": exact, "exact_rate": exact / n if n else None,
            "correct_pairs": pair_correct, "total_pairs": pair_total,
            "pair_accuracy": pair_correct / pair_total if pair_total else None,
            "citation_positions_computable_valid_calls": len(computable),
            "answer_consistent_with_own_citation_positions": sum(record["citation_observation"]["answer_consistent_with_own_citation_positions"] for record in computable),
            "citation_consistency_fixed_denominator": n,
            "citation_consistency_unknown_calls": n - len(computable),
            "citation_position_order_matches_gold": sum(record["citation_observation"]["citation_position_order_matches_gold"] for record in computable),
            "external_citation_sort": {"fixed_denominator": n,
                                       "known_calls": sum(record["external_citation_sort"]["exact"] is not None for record in records),
                                       "unknown_calls": sum(record["external_citation_sort"]["exact"] is None for record in records),
                                       "exact_calls": sum(record["external_citation_sort"]["exact"] is True for record in records),
                                       "rescued_wrong_to_right": sum(record["external_citation_sort"]["rescued_wrong_to_right"] for record in records),
                                       "changed_exact_to_wrong": sum(record["external_citation_sort"]["changed_exact_to_wrong"] for record in records)},
            "exact_quotes": sum(entry["quote_exact_in_cited_turn"] for record in records for entry in record["citation_observation"]["entries"]),
            "expected_citations": 6 * n}


def contrasts(records):
    grouped = collections.defaultdict(dict)
    for record in records:
        require(record["probe_arm"] not in grouped[record["pair_id"]], "DUPLICATE_PAIRED_ARM")
        grouped[record["pair_id"]][record["probe_arm"]] = record
    result = []
    transitions = collections.Counter()
    for pair_id, arms in sorted(grouped.items()):
        require(set(arms) == set(ARMS), "MISSING_PAIRED_ARM")
        control, treatment = arms["control"], arms["treatment"]
        for key in ("candidate_id", "root_id", "cohort", "variant_id", "seed"):
            require(control[key] == treatment[key], "PAIRED_FIELD_DIFFERS:" + key)
        transition = ("correct" if control["exact"] else "wrong") + "_to_" + ("correct" if treatment["exact"] else "wrong")
        transitions[transition] += 1
        result.append({"pair_id": pair_id, "candidate_id": control["candidate_id"], "root_id": control["root_id"],
                       "cohort": control["cohort"], "variant_id": control["variant_id"], "seed": control["seed"],
                       "control_request_id": control["request_id"], "treatment_request_id": treatment["request_id"],
                       "control_valid": control["valid"], "treatment_valid": treatment["valid"],
                       "control_exact": control["exact"], "treatment_exact": treatment["exact"],
                       "exact_net_change": int(treatment["exact"]) - int(control["exact"]),
                       "correct_pair_net_change": treatment["correct_pairs"] - control["correct_pairs"], "transition": transition})
    return {"pairs": len(result), "transitions": {key: transitions[key] for key in ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")}, "records": result}


def summarize(records):
    matched = contrasts(records)
    arms = {arm: aggregate([record for record in records if record["probe_arm"] == arm]) for arm in ARMS}
    cohorts = {cohort: {arm: aggregate([record for record in records if record["cohort"] == cohort and record["probe_arm"] == arm]) for arm in ARMS} for cohort in COHORTS}
    roots = []
    for root in sorted({record["root_id"] for record in records}):
        subset = [record for record in records if record["root_id"] == root]
        require(len({record["cohort"] for record in subset}) == 1, "ROOT_IN_MULTIPLE_COHORTS")
        summaries = {arm: aggregate([record for record in subset if record["probe_arm"] == arm]) for arm in ARMS}
        roots.append({"root_id": root, "cohort": subset[0]["cohort"], **summaries,
                      "exact_net_change": summaries["treatment"]["exact_calls"] - summaries["control"]["exact_calls"],
                      "correct_pair_net_change": summaries["treatment"]["correct_pairs"] - summaries["control"]["correct_pairs"]})
    rep = cohorts["replication"]
    positive = [root["root_id"] for root in roots if root["cohort"] == "replication" and root["exact_net_change"] > 0]
    conditions = {"at_least_three_positive_replication_roots": len(positive) >= 3,
                  "positive_replication_aggregate_exact_net_change": rep["treatment"]["exact_calls"] > rep["control"]["exact_calls"],
                  "replication_pair_correct_count_not_decreased": rep["treatment"]["correct_pairs"] >= rep["control"]["correct_pairs"],
                  "replication_citation_answer_consistent_count_net_positive": rep["treatment"]["answer_consistent_with_own_citation_positions"] > rep["control"]["answer_consistent_with_own_citation_positions"],
                  "both_arms_all_strict_valid": all(value["calls"] > 0 and value["valid_calls"] == value["calls"] for value in arms.values())}
    endpoint = {"name": "PREDECLARED_REPLICATION_WORKING_SUPPORT_GATE", "conditions": conditions,
                "positive_replication_roots": positive, "met": all(conditions.values()),
                "statistical_significance_claim": False, "cause_auto_published": False,
                "interpretation": "A working support threshold only; not a significance test or proof of an internal mechanism."}
    replication_roots = [root for root in roots if root["cohort"] == "replication"]
    root_equal_difference = sum(root["treatment"]["exact_rate"] - root["control"]["exact_rate"] for root in replication_roots) / len(replication_roots) if replication_roots else None
    return {"by_arm": arms, "by_cohort": cohorts, "by_root": roots, "matched_pairs": matched,
            "primary_replication_root_equal_exact_difference": root_equal_difference, "primary_endpoint": endpoint}


def validate_design(manifest):
    rows = manifest["requests"]
    require(manifest["branch"] == "evidence", "EVIDENCE_BRANCH_REQUIRED")
    require(len(rows) == 108 and manifest["request_count"] == 108, "EXPECTED_108_FIXED_REQUESTS")
    require(len({row["request_id"] for row in rows}) == 108, "DUPLICATE_REQUEST_ID")
    require(len({row["candidate_id"] for row in rows}) == 9 and len({row["root_id"] for row in rows}) == 9, "EXPECTED_NINE_QUESTIONS_NINE_ROOTS")
    require(all(row["question_type"] == "ordering" and row["branch"] == "evidence" and row["probe_arm"] in ARMS and row["cohort"] in COHORTS for row in rows), "INVALID_DESIGN_ROW")
    for cohort, expected in (("discovery", 3), ("replication", 6)):
        subset = [row for row in rows if row["cohort"] == cohort]
        require(len({row["root_id"] for row in subset}) == expected and len(subset) == expected * 12, "COHORT_DENOMINATOR_CHANGED")
    for candidate_id in {row["candidate_id"] for row in rows}:
        subset = [row for row in rows if row["candidate_id"] == candidate_id]
        require(len({row["root_id"] for row in subset}) == 1 and len(subset) == 12, "QUESTION_DENOMINATOR_CHANGED")
        for arm in ARMS:
            variants = [row["variant_id"] for row in subset if row["probe_arm"] == arm]
            require(len(variants) == 6 and len(set(variants)) == 6, "EXPECTED_SIX_VARIANTS_PER_ARM")
    pairs = collections.defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    require(len(pairs) == 54, "EXPECTED_54_PAIRS")
    for pair in pairs.values():
        require(len(pair) == 2 and {row["probe_arm"] for row in pair} == set(ARMS), "PAIR_ARM_MISMATCH")
        for key in ("candidate_id", "root_id", "cohort", "variant_id", "seed", "display", "json_schema", "max_output_tokens"):
            require(pair[0][key] == pair[1][key], "PAIRED_DESIGN_DIFFERS:" + key)
        require(pair[0]["max_output_tokens"] == 1024, "OUTPUT_BUDGET_CHANGED")


def score(run):
    run = pathlib.Path(run).resolve()
    require(not (run / "SCORING_STARTED.json").exists() and not (run / "SCORES.json").exists(), "SCORING_ALREADY_STARTED_NO_OVERWRITE")
    manifest = read(run / "manifest.json")
    validate_design(manifest)
    preflight = read(run / "PREFLIGHT_FREEZE.json")
    verify_bindings(preflight["files"])
    freeze_path = run / "RAW_RESPONSE_FREEZE.json"
    freeze = read(freeze_path)
    require(freeze["status"] == "COMMITTED", "COMMITTED_RAW_FREEZE_REQUIRED")
    verify_bindings(freeze["files"] + [freeze["manifest_binding"]])
    require(pathlib.Path(freeze["manifest_binding"]["path"]).resolve() == run / "manifest.json", "WRONG_FROZEN_MANIFEST")
    expected = {run / "responses" / (row["request_id"] + ".json") for row in manifest["requests"]}
    require(len(freeze["files"]) == 108 and {pathlib.Path(item["path"]).resolve() for item in freeze["files"]} == expected, "RAW_DENOMINATOR_MISMATCH")
    require(set((run / "responses").glob("*.json")) == expected, "UNEXPECTED_RESPONSE_FILES")
    raw_records = {}
    for row in manifest["requests"]:
        raw = read(run / "responses" / (row["request_id"] + ".json"))
        require(raw["request_id"] == row["request_id"] and raw["candidate_id"] == row["candidate_id"] and raw["root_id"] == row["root_id"], "RESPONSE_ID_MISMATCH")
        require(raw["status"] in ("RETURNED", "BACKEND_ERROR", "NOT_SUBMITTED", "MISSING"), "NONTERMINAL_RESPONSE_STATUS")
        require(timestamp(raw["response_committed_at_utc"]) <= timestamp(freeze["created_at_utc"]), "RESPONSE_AFTER_FREEZE")
        if raw["status"] == "RETURNED":
            require(raw["prompt_sha256"] == row["prompt_sha256"] and raw["seed"] == row["seed"], "RESPONSE_PROMPT_OR_SEED_MISMATCH")
            require(raw["variant_id"] == row["variant_id"], "RESPONSE_VARIANT_MISMATCH")
        raw_records[row["request_id"]] = raw
    private_binding = manifest["private_answer_binding"]
    verify_bindings([private_binding])  # Hashing bytes does not parse private Gold.
    before = now()
    require(timestamp(before) >= timestamp(freeze["created_at_utc"]), "CLOCK_BEFORE_FREEZE")
    save(run / "SCORING_STARTED.json", {"gold_first_read_at_utc": before,
         "raw_response_freeze_binding": bind(freeze_path), "private_answer_binding": private_binding,
         "all_raw_responses_committed_before_gold": True,
         "gold_read_scope": "First private QA JSON read by this scoring process; historical author, reviewer and parent exposure is not denied."})
    private = {q["candidate_id"]: q for q in read(private_binding["path"])["questions"]}
    after = now()
    public = {q["candidate_id"]: q for q in read(run / "PUBLIC_QUESTIONS.json")["questions"]}
    maps_path = run / "private/DISPLAY_MAPPINGS.json"
    maps = read(maps_path)
    sources = {}
    for row in manifest["requests"]:
        question = public[row["candidate_id"]]
        gold_question = private[row["candidate_id"]]
        require({key: gold_question[key] for key in PUBLIC_KEYS if key in gold_question} == question, "PRIVATE_PUBLIC_QUESTION_MISMATCH")
        source_path = run / "sources" / (row["root_id"] + ".json")
        require(sha(source_path) == question["source_sha256"], "SOURCE_BYTES_CHANGED")
        if row["root_id"] not in sources:
            source = read(source_path)
            require(source["root_id"] == row["root_id"] and source["turns"], "SOURCE_ROOT_MISMATCH")
            require(len({turn["turn_ref"] for turn in source["turns"]}) == len(source["turns"]), "SOURCE_DUPLICATE_TURN_REF")
            sources[row["root_id"]] = source
    scored = []
    for row in manifest["requests"]:
        raw = raw_records[row["request_id"]]
        record = score_one(row, raw, maps["requests"][row["request_id"]], maps["public_to_canonical"][row["candidate_id"]],
                           private[row["candidate_id"]]["gold"], sources[row["root_id"]])
        record.update({"response_binding": bind(run / "responses" / (row["request_id"] + ".json")),
                       "source_binding": bind(run / "sources" / (row["root_id"] + ".json")),
                       "display_mappings_binding": bind(maps_path), "gold_first_read_at_utc": before})
        scored.append(record)
    costs = runtime_cost(run, manifest, raw_records)
    status = "COMPLETE" if (run / "COMPLETE.json").exists() else "FAILED_WITH_FIXED_DENOMINATOR"
    result = {"status": status, "scored_at_utc": now(), "namespace": "train.feedback", "branch": "evidence",
              "gold_first_read_at_utc": before, "gold_read_completed_at_utc": after,
              "manifest_binding": bind(run / "manifest.json"), "raw_response_freeze_binding": bind(freeze_path),
              "preflight_freeze_binding": bind(run / "PREFLIGHT_FREEZE.json"),
              "public_questions_binding": bind(run / "PUBLIC_QUESTIONS.json"), "private_answer_binding": private_binding,
              "display_mappings_binding": bind(maps_path), "scoring_code_binding": bind(__file__),
              "denominator_calls": 108, "denominator_questions": 9, "denominator_roots": 9,
              "valid_calls": sum(record["valid"] for record in scored), "exact_calls": sum(record["exact"] for record in scored),
              "cost": costs, **summarize(scored), "requests": scored,
              "semantic_support_reviewed": False, "use_as_final_eval": False, "training_started": False,
              "new_training_questions_generated": 0, "cause_auto_published": False, "weakness_memory_upgraded": False,
              "memory_three_arm_planned_scenes": 54, "memory_three_arm_executed_scenes": 0,
              "limitations": ["All questions belong to previously observed Train roots; replication is not a pristine test set.",
                              "Repeated permutations and samples do not create independent questions.",
                              "Verbatim citation location does not establish semantic support or faithful internal reasoning.",
                              "The endpoint is a predeclared working threshold, not a statistical significance test.",
                              "No automatic cause publication, new question generation, training, or demonstration of self-play gains."]}
    protocol_binding = manifest.get("protocol_binding")
    if protocol_binding:
        verify_bindings([protocol_binding])
        result["protocol_binding"] = protocol_binding
    save(run / "SCORES.json", result)
    print(json.dumps({"status": status, "valid": result["valid_calls"], "exact": result["exact_calls"],
                      "by_arm": result["by_arm"], "primary_endpoint": result["primary_endpoint"]}, ensure_ascii=False))
    return result


def runtime_cost(run, manifest, raw_records):
    returned = [raw for raw in raw_records.values() if raw["status"] == "RETURNED"]
    unknown = [raw for raw in raw_records.values() if raw["status"] == "BACKEND_ERROR" or
               (raw["status"] == "RETURNED" and any(not isinstance(raw.get("metadata", {}).get(key), int) for key in ("prompt_token_count", "completion_token_count")))]
    complete_path = run / "COMPLETE.json"
    failed_path = run / "FAILED.json"
    execution_path = complete_path if complete_path.exists() else failed_path
    execution = read(execution_path) if execution_path.exists() else {}
    by_arm = {}
    for arm in ARMS:
        ids = [row["request_id"] for row in manifest["requests"] if row["probe_arm"] == arm]
        subset = [raw_records[rid] for rid in ids if raw_records[rid]["status"] == "RETURNED"]
        by_arm[arm] = {"known_prompt_tokens": sum(raw.get("metadata", {}).get("prompt_token_count", 0) or 0 for raw in subset),
                       "known_completion_tokens": sum(raw.get("metadata", {}).get("completion_token_count", 0) or 0 for raw in subset),
                       "planned_calls": len(ids), "returned_calls": len(subset)}
    prompt = sum(raw.get("metadata", {}).get("prompt_token_count", 0) or 0 for raw in returned)
    completion = sum(raw.get("metadata", {}).get("completion_token_count", 0) or 0 for raw in returned)
    return {"planned_calls": len(raw_records), "returned_calls": len(returned),
            "submitted_attempts": sum(raw["status"] not in ("NOT_SUBMITTED", "MISSING") for raw in raw_records.values()),
            "not_submitted_calls": sum(raw["status"] == "NOT_SUBMITTED" for raw in raw_records.values()),
            "missing_terminal_calls": sum(raw["status"] == "MISSING" for raw in raw_records.values()),
            "requests_with_unknown_usage": len(unknown), "known_prompt_tokens": prompt,
            "known_completion_tokens": completion, "known_total_tokens": prompt + completion,
            "by_arm": by_arm, "live_wall_seconds": execution.get("wall_seconds"),
            "model_load_wall_seconds": execution.get("model_load_wall_seconds"),
            "execution_binding": bind(execution_path) if execution_path.exists() else None,
            "wall_scope": "Reported live runtime includes model loading; protocol preparation, audits and report writing are excluded. Matched arms share the same model load; per-arm latency is not inferred.",
            "currency_cost": None, "currency_cost_reason": "Local GPU inference; monetary billing is unavailable. Unknown usage is not counted as zero."}


def _percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def report(run, output):
    run, output = pathlib.Path(run).resolve(), pathlib.Path(output).resolve()
    scores_path = run / "SCORES.json"
    scores = read(scores_path)
    verify_bindings([scores["manifest_binding"], scores["raw_response_freeze_binding"], scores["scoring_code_binding"]])
    def relative(path):
        return os.path.relpath(path, output.parent)
    lines = ["# 排序答案构造最小对照", "",
             "本轮检验：仅增加固定的引用位置核对指令，是否能提高排序答案准确率。两臂保留相同全文、展示置换、采样 seed、结构化输出规则与 1024 token 预算；没有提供作者证据、Gold 或正确排序关系。", "",
             "共 9 道题、9 个 Train root、每题 6 个置换，两臂合计 108 次计划作答。3 个既有异常案例为 discovery，另 6 个 root 按事前哈希规则选入 replication。全部来自已观察过的训练侧数据，replication 不等于全新测试集。", "",
             "| 范围 | 臂 | 严格有效 / 计划 | 整题全对 / 计划 | 整题正确率 | 正确成对 / 固定成对 | 成对正确率 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for cohort in ("all",) + COHORTS:
        arms = scores["by_arm"] if cohort == "all" else scores["by_cohort"][cohort]
        for arm in ARMS:
            row = arms[arm]
            lines.append(f"| {cohort} | {arm} | {row['valid_calls']}/{row['calls']} | {row['exact_calls']}/{row['calls']} | {_percent(row['exact_rate'])} | {row['correct_pairs']}/{row['total_pairs']} | {_percent(row['pair_accuracy'])} |")
    endpoint = scores["primary_endpoint"]
    lines += ["", f"主要终点：replication root 等权整题正确率之差（treatment − control）为 **{100 * scores['primary_replication_root_equal_exact_difference']:+.2f} 个百分点**。", "",
              f"事前规定的 replication 工作支持门槛：**{'达到' if endpoint['met'] else '未达到'}**。这是操作门槛，不是统计显著性检验，也不证明内部推理机制。", "",
              "| 事前条件 | 是否满足 |", "|---|---|"]
    translations = {"at_least_three_positive_replication_roots": "至少 3 个不同 replication root 的整题全对数净增大于 0",
                    "positive_replication_aggregate_exact_net_change": "replication 聚合整题全对数净增大于 0",
                    "replication_pair_correct_count_not_decreased": "replication 正确成对数不下降",
                    "replication_citation_answer_consistent_count_net_positive": "replication 固定分母下自身引文位置与最终答案一致次数净增大于 0",
                    "both_arms_all_strict_valid": "两臂所有计划作答均严格有效"}
    for key, met in endpoint["conditions"].items():
        lines.append(f"| {translations[key]} | {'是' if met else '否'} |")
    lines += ["", "| Root | Cohort | control 全对 | treatment 全对 | 全对净增 | control 正确成对 | treatment 正确成对 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in scores["by_root"]:
        control, treatment = row["control"], row["treatment"]
        lines.append(f"| {row['root_id']} | {row['cohort']} | {control['exact_calls']}/{control['calls']} | {treatment['exact_calls']}/{treatment['calls']} | {row['exact_net_change']:+d} | {control['correct_pairs']}/{control['total_pairs']} | {treatment['correct_pairs']}/{treatment['total_pairs']} |")
    lines += ["", "逐对转移（相同题目、置换及 seed）：", ""]
    transitions = scores["matched_pairs"]["transitions"]
    lines.append("；".join(f"{key}: {value}" for key, value in transitions.items()) + "。")
    lines += ["", "引文位置仅在每项引文逐字匹配指定 turn、出现位置唯一且 6 项无位置并列时可计算。逐字匹配不等于语义支持；自身引文也不被假设为模型内部思考的真实记录。", "",
              "| 臂 | 严格有效且引文位置可计算 | 最终答案遵循自身引文位置 | 自身引文排序符合 Gold | 逐字匹配引文 / 预期引文 |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = scores["by_arm"][arm]
        lines.append(f"| {arm} | {row['citation_positions_computable_valid_calls']} | {row['answer_consistent_with_own_citation_positions']} | {row['citation_position_order_matches_gold']} | {row['exact_quotes']}/{row['expected_citations']} |")
    lines += ["", "上表机械一致性使用每臂全部 54 次计划作答作固定分母，replication 每臂 36 次；不可定位的引文单独记 unknown，不剔除分母。", "",
              "另外对模型自己的引文位置进行确定性排序，得到以下外部后处理结果。这些结果不替代模型实际作答、不计为模型能力，也没有进行新的引文语义复核。", "",
              "| 范围 | 臂 | 外部排序全对 / 固定分母 | 原答案错→外部对 | 原答案对→外部错 | unknown |", "|---|---|---:|---:|---:|---:|"]
    for cohort in ("all",) + COHORTS:
        arms = scores["by_arm"] if cohort == "all" else scores["by_cohort"][cohort]
        for arm in ARMS:
            derived = arms[arm]["external_citation_sort"]
            lines.append(f"| {cohort} | {arm} | {derived['exact_calls']}/{derived['fixed_denominator']} | {derived['rescued_wrong_to_right']} | {derived['changed_exact_to_wrong']} | {derived['unknown_calls']} |")
    invalid = [row for row in scores["requests"] if not row["valid"]]
    lines += ["", f"格式或完成状态失败：{len(invalid)} 次。无效、缺失或未提交请求保留在固定分母中，整题与 15 个成对均计 0；无效作答没有通过丢弃而提高准确率。", ""]
    if invalid:
        lines += ["| 请求 | 臂 | 原始状态 | 原因 |", "|---|---|---|---|"]
        for row in invalid:
            reason = str(row["parse_error"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['request_id']} | {row['probe_arm']} | {row['status']} | {reason} |")
    cost = scores["cost"]
    lines += ["", f"已知输入 {cost['known_prompt_tokens']:,} tokens，输出 {cost['known_completion_tokens']:,} tokens，共 {cost['known_total_tokens']:,} tokens；使用量未知请求 {cost['requests_with_unknown_usage']} 次，未知不记为 0。", "",
              f"模型运行记录耗时 {cost['live_wall_seconds']} 秒，其中加载 {cost['model_load_wall_seconds']} 秒；不包括准备、审核和报告时间。两臂共享加载，未推算独立单臂耗时。金额计费未知。", "",
              "本轮不会自动升级弱点 memory 或发布已确认原因。事实 memory 三臂的 54 场实验尚未执行；本轮没有生成新训练题、没有训练，也没有证明 self-play 或反馈训练收益。", "",
              f"分数及全部逐请求绑定：[SCORES.json]({relative(scores_path)})；原始记录封存：[RAW_RESPONSE_FREEZE.json]({relative(run / 'RAW_RESPONSE_FREEZE.json')})。", "",
              f"报告输入分数 SHA256：`{sha(scores_path)}`。", ""]
    with output.open("x") as stream:
        stream.write("\n".join(lines))
    return bind(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    for command in ("score", "report"):
        child = commands.add_parser(command)
        child.add_argument("--run", type=pathlib.Path, default=DEFAULT_RUN)
        if command == "report":
            child.add_argument("--output", type=pathlib.Path, default=HERE / "REPORT_ZH.md")
    args = parser.parse_args()
    if args.cmd == "score":
        score(args.run)
    else:
        print(json.dumps(report(args.run, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
