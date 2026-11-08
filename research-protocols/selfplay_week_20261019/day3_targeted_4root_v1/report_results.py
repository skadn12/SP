"""Publish a post-freeze diagnostic report; stdlib only, no inference or private JSON.

Writing outputs requires --post-freeze-authorized. This explicit execution flag
is used by the parent only after raw freeze and scoring. Gold is then obtained
from SCORES.json, never from a private sidecar. Merely importing this module
performs no input reads or writes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
from statistics import mean, median

HERE = Path(__file__).resolve().parent
NAMESPACE = "train.feedback.targeted4root_v1"
TYPES = ("multi_select", "ordering")
LIMITS = [
    "Four historically exposed Train roots; this is an operational feasibility and S0 diagnosis pilot.",
    "Two targeted extensions and two reused sources are not randomized matched A/B groups; their scores do not estimate a feedback effect.",
    "Arity, relation, Gold count and source/question reuse differ; no controlled difficulty-increase claim.",
    "No training, training-benefit estimate, new training admission or DEV/TEST evaluation is performed by this report.",
    "Content acceptance and fixed production yield are distinct from Solver accuracy and from training admission.",
    "Finite legal-answer constrained decoding is used; validity is not free-form format compliance.",
    "Direct outputs support option/pair observations, not a causal diagnosis of hidden reasoning or measured memory reliance.",
    "Invalid calls remain in exact, Jaccard and pair denominators; invalid MC selections and FP/FN are unknown, not zero observations.",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verify(record, expected_path=None):
    current = binding(record["path"])
    require(all(current[key] == record[key] for key in ("sha256", "bytes")), "BINDING_MISMATCH:" + current["path"])
    if expected_path is not None:
        require(Path(record["path"]).resolve() == Path(expected_path).resolve(), "BINDING_PATH_MISMATCH")


def unique(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), "DUPLICATE:" + key)
    return result


def average(values):
    values = list(values)
    return mean(values) if values else None


def divide(a, b):
    return a / b if b else None


def same(actual, expected, label):
    if isinstance(expected, float):
        require(isinstance(actual, (float, int)) and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), label)
    else:
        require(actual == expected, label)


def grouped(questions):
    calls = sum(q["denominator_calls"] for q in questions)
    exact = sum(q["exact_calls"] for q in questions)
    roots = {q["root_id"] for q in questions}
    result = {"questions": len(questions), "calls": calls, "exact_calls": exact,
        "question_equal_exact": average(q["exact_rate"] for q in questions), "call_equal_exact": divide(exact, calls),
        "admitted_roots": len(roots),
        "root_equal_exact": average(average(q["exact_rate"] for q in questions if q["root_id"] == root) for root in sorted(roots)),
        "question_equal_partial_metric": average(q["partial_metric"] for q in questions),
        "nonexact_with_partial_calls": sum(q["nonexact_with_partial_calls"] for q in questions),
        "nonexact_with_partial_rate": divide(sum(q["nonexact_with_partial_calls"] for q in questions), calls)}
    if len({q["question_type"] for q in questions}) > 1:
        result.pop("question_equal_partial_metric")
    if questions and all(q["question_type"] == "ordering" for q in questions):
        correct, total = sum(q["correct_pairs"] for q in questions), sum(q["total_pairs"] for q in questions)
        result.update(correct_pairs=correct, total_pairs=total, question_equal_pair_accuracy=average(q["pair_accuracy"] for q in questions), pair_weighted_accuracy=divide(correct, total))
    if questions and all(q["question_type"] == "multi_select" for q in questions):
        result.update(question_equal_jaccard=average(q["mean_jaccard"] for q in questions),
            FP_option_events=sum(q["fp_calls"] for q in questions), FN_option_events=sum(q["fn_calls"] for q in questions))
    return result


def reconstruct(manifest, scores, public, mappings, raw_by_id):
    native = unique(scores["requests"], "request_id")
    require(set(native) == {r["request_id"] for r in manifest["requests"]}, "SCORED_REQUEST_SET")
    calls = []
    gold_by_q = {}
    for request in manifest["requests"]:
        rid, qid, typ = request["request_id"], request["candidate_id"], request["question_type"]
        raw, scored, question = raw_by_id[rid], native[rid], public[qid]
        field = "options" if typ == "multi_select" else "items"
        canonical = {label: "content_" + hashlib.sha256(text.strip().encode()).hexdigest()[:16] for label, text in question[field].items()}
        require(mappings["public_to_canonical"][qid] == canonical, "CANONICAL_CONTENT_HASH")
        reverse = {content: label for label, content in canonical.items()}
        mapping = mappings["requests"][rid]["display_to_content"]
        require(set(mapping.values()) == set(reverse) and len(mapping) == len(reverse), "DISPLAY_CONTENT_BIJECTION")
        display_to_canonical = {label: reverse[content] for label, content in mapping.items()}
        gold = scored["gold_canonical"]  # Accessed only after verified raw freeze; no private JSON.
        require(len(gold) == len(set(gold)) >= 2 and set(gold) <= set(canonical), "INVALID_SCORED_GOLD")
        require(typ != "ordering" or set(gold) == set(canonical), "INCOMPLETE_ORDERING_GOLD")
        require(qid not in gold_by_q or gold_by_q[qid] == gold, "GOLD_CHANGED_BETWEEN_CALLS")
        gold_by_q[qid] = gold
        match = re.fullmatch(r"<box>([^<>]+)</box>", raw.get("raw_output") or "")
        display = match.group(1).split(",") if match else None
        valid = bool(match and raw["status"] == "RETURNED" and not raw.get("output_validation_errors")
            and raw.get("metadata", {}).get("finish_reason") in ("stop", "eos")
            and len(display) == len(set(display)) and set(display) <= set(mapping))
        valid = valid and (display == sorted(display) if typ == "multi_select" else set(display or []) == set(mapping))
        prediction = [display_to_canonical[label] for label in display] if valid else None
        predset, goldset = set(prediction or []), set(gold)
        exact = bool(valid and (predset == goldset if typ == "multi_select" else prediction == gold))
        same(valid, scored["valid"], rid + ":valid")
        same(exact, scored["exact"], rid + ":exact")
        require((predset == set(scored["canonical_prediction"] or []) if typ == "multi_select" else prediction == scored["canonical_prediction"]), rid + ":prediction")
        row = {"request_id": rid, "candidate_id": qid, "root_id": request["root_id"], "question_type": typ,
            "display_prediction": display, "canonical_prediction": prediction, "canonical_gold": gold,
            "display_to_canonical": display_to_canonical, "raw_output": raw.get("raw_output"), "raw_status": raw["status"],
            "valid": valid, "exact": exact, "item_count": len(canonical), "prompt_tokens": request["prompt_tokens"],
            "source_turns": request["source_turns"], "pair_correct": None, "pair_total": None,
            "false_positive_options": None, "false_negative_options": None, "true_positive_options": None,
            "FP_labels": None, "FN_labels": None, "jaccard": None, "nonexact_with_partial": False,
            "raw_response_path": str((Path(manifest["_run"]) / "responses" / (rid + ".json")).resolve())}
        if typ == "multi_select":
            fp, fn, tp = predset - goldset, goldset - predset, predset & goldset
            for key, val in (("fp", len(fp)), ("fn", len(fn)), ("tp", len(tp))):
                same(scored["metrics"][key], val if valid else None, rid + ":" + key)
            row.update(false_positive_options=len(fp) if valid else None, false_negative_options=len(fn) if valid else None,
                true_positive_options=len(tp) if valid else None, FP_labels=sorted(fp) if valid else None, FN_labels=sorted(fn) if valid else None,
                jaccard=len(tp) / len(predset | goldset) if valid else 0,
                nonexact_with_partial=bool(valid and not exact and tp))
        else:
            rank = {label: i for i, label in enumerate(prediction or [])}
            pairs = list(itertools.combinations(gold, 2))
            correct = sum(bool(valid and rank[a] < rank[b]) for a, b in pairs)
            same(scored["metrics"]["correct_pairs"], correct, rid + ":correct_pairs")
            same(scored["metrics"]["total_pairs"], len(pairs), rid + ":total_pairs")
            row.update(pair_correct=correct, pair_total=len(pairs), nonexact_with_partial=bool(valid and not exact and correct > 0))
        calls.append(row)
    return calls, gold_by_q


def source_stats(generation, source_freeze, admitted_roots):
    require(source_freeze["status"] == "CONTENT_ACCEPTED_BEFORE_QUESTIONS", "SOURCE_CONTENT_FREEZE_REQUIRED")
    result = {}
    for root in generation["roots"]:
        rid = root["root_id"]
        record = source_freeze["sources"].get(rid, {})
        item = {"mode": root["mode"], "source_accepted": bool(record.get("source")),
            "prefix_turns": root["prefix_turn_count"], "final_turns": None, "new_turns": None,
            "accepted_extension_scenes": None, "scene_counts_by_route": None, "source": record.get("source")}
        if record.get("source"):
            verify(record["source"])
            verify(root["prefix"])
            source, prefix = read(record["source"]["path"]), read(root["prefix"]["path"])
            require(source["root_id"] == rid and source["turns"][:len(prefix["turns"])] == prefix["turns"], "SOURCE_PREFIX_CHANGED")
            require(len(prefix["turns"]) == root["prefix_turn_count"], "PREFIX_TURN_COUNT")
            if root["mode"] == "reuse_existing":
                require(source["turns"] == prefix["turns"], "REUSED_SOURCE_TURNS_CHANGED")
            scenes = source.get("targeted_extension_scenes", [])
            item.update(final_turns=len(source["turns"]), new_turns=len(source["turns"]) - len(prefix["turns"]),
                accepted_extension_scenes=len(scenes), scene_counts_by_route=dict(Counter(s["scene_ref"].split(":")[1] for s in scenes)))
        require(item["source_accepted"] or rid not in admitted_roots, "ADMITTED_SOURCE_NOT_ACCEPTED")
        result[rid] = item
    return result


def contradictions(path, public_path, public, calls):
    if path is None:
        return {"annotation_supplied": False, "checks": [], "note": "No explicit reviewed incompatibility annotations supplied; no automatic contradiction claim is made."}
    payload = read(path)
    require(payload["status"] == "PASS", "CONTRADICTION_ANNOTATION_NOT_REVIEWED")
    verify(payload["public_questions"], public_path)
    results = []
    for spec in payload["checks"]:
        q = public[spec["candidate_id"]]
        labels = spec["canonical_labels"]
        require(q["question_type"] == "multi_select" and spec["kind"] == "mutually_exclusive_selection", "CONTRADICTION_KIND")
        require(spec["semantic_review_status"] == "PASS" and spec["reviewed_by"] and spec["reason"], "CONTRADICTION_SEMANTIC_REVIEW_REQUIRED")
        require(len(labels) == len(set(labels)) == 2 and set(labels) <= set(q["options"]), "CONTRADICTION_REQUIRES_ONE_EXPLICIT_PAIR")
        source = read(q["source_path"])
        turns = {t["turn_ref"]: t["text"] for t in source["turns"]}
        require(spec["evidence"] and all(e["turn_ref"] in turns and e["quote"] and e["quote"] in turns[e["turn_ref"]] for e in spec["evidence"]), "CONTRADICTION_EXACT_EVIDENCE")
        subset = [c for c in calls if c["candidate_id"] == spec["candidate_id"]]
        require(not set(labels) <= set(subset[0]["canonical_gold"]), "GOLD_CONFLICTS_WITH_INCOMPATIBILITY_ANNOTATION")
        hits = [c["request_id"] for c in subset if c["valid"] and set(labels) <= set(c["canonical_prediction"])]
        results.append({**spec, "option_texts": {k: q["options"][k] for k in labels}, "denominator_calls": len(subset),
            "valid_calls": sum(c["valid"] for c in subset), "unknown_invalid_calls": sum(not c["valid"] for c in subset),
            "co_selected_calls": len(hits), "request_ids": hits})
    return {"annotation_supplied": True, "annotation_binding": binding(path), "checks": results,
        "note": "Counts apply only to these explicitly reviewed canonical option sets; no automatic cross-option or hidden-reasoning inference."}


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(summary):
    d, mc = summary["diagnosis"], summary["MC_observable_errors"]
    lines = ["# 定向扩充四 root 操作试点", "",
        f"固定 4 个历史已暴露 Train roots、24 个题位：准入 {summary['admitted_questions']}/24，未准入 {24-summary['admitted_questions']}/24。状态分布：{summary['production_status_counts']}。两 root 定向扩充，两 root 复用既有对话；不是随机对照实验。", "",
        f"S0=M-clean，直接答案，共 {d['calls']} 个固定调用，有效 {d['valid_calls']}、无效/终态失败 {d['invalid_calls']}、完整正确 {d['exact_calls']}。全部准入题均评估；未按分数重新准入，无重试、无截断。未准入题保留在生产分母，但没有伪造 Solver 作答。", "",
        "|题型|题数|固定调用|完整正确（题等权）|完整正确（调用等权）|部分指标（题等权）|错答但含部分正确内容（调用数）|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for typ, label in (("multi_select", "多选（部分指标=Jaccard）"), ("ordering", "排序（部分指标=序对正确率）")):
        g = summary["by_type"][typ]
        lines.append(f"|{label}|{g['questions']}|{g['calls']}|{pct(g['question_equal_exact'])}|{pct(g['call_equal_exact'])}|{pct(g['question_equal_partial_metric'])}|{g['nonexact_with_partial_calls']}|")
    lines += ["", "多选 Jaccard=所选正确选项数/预测与 Gold 的并集大小；排序部分指标=正确 Gold 序对数/全部 Gold 序对数。无效调用两指标计零。最后一列仅指非全对但命中至少一个 Gold 选项/序对，不等同于通过题目；这些调用可能仍包含错误选项或错误关系。总体混合两种部分指标没有统一含义，因此不报告混合部分正确率。", "",
        "|root|来源方式|对话 turns（前缀+新增）|准入/计划|多选/排序|未准入|完整正确（题等权）|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for root, r in summary["roots"].items():
        s = summary["sources"][root]
        lines.append(f"|{root}|{s['mode']}|{s['final_turns']}（{s['prefix_turns']}+{s['new_turns']}）|{r['admitted']}/{r['planned_slots']}|{r['MC']}/{r['ordering']}|{r['unadmitted']}|{pct(r['scores']['question_equal_exact'])}|")
    lines += ["", "扩充/复用模式各自的题目分数只作描述，素材、题目和历史暴露不匹配，不能用于证明扩充反馈有效。", "",
        "|排序项数|准入题/计划题位|固定调用|完整正确（题等权）|完整正确（调用等权）|序对正确（题等权）|序对正确（序对加权）|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for n in (3, 4, 5, 6):
        g = summary["ordering_by_arity"][str(n)]
        lines.append(f"|{n}|{g['questions']}/{g['planned_slots_with_actual_arity']}|{g['calls']}|{pct(g['question_equal_exact'])}|{pct(g['call_equal_exact'])}|{pct(g.get('question_equal_pair_accuracy'))}|{pct(g.get('pair_weighted_accuracy'))}|")
    lines += ["", "排序未准入题若没有实际项数，不分配到任何项数层；3 项仅 6 种全序，6 项有 720 种，不可把不同项数的总分当成同等难度。", "",
        "|关系/模板|计划题位|准入|未准入|调用|完整正确（题等权）|",
        "|---|---:|---:|---:|---:|---:|"]
    for name, g in summary["by_template"].items():
        lines.append(f"|{cell(name)}|{g['planned_slots']}|{g['questions']}|{g['unadmitted']}|{g['calls']}|{pct(g['question_equal_exact'])}|")
    lines += ["", f"多选实际 Gold 数量分布（准入）：{summary['gold_count_distribution_MC']}；计划目标：{summary['planned_MC_gold_count_distribution']}。排序 primary_relation 分层详见 SUMMARY.json 的 ordering_by_relation；原始作者标注保留，未臆测补齐。", "",
        "|排序 primary_relation（作者标注）|准入/计划|准入项数分布|完整正确（题等权）|序对正确（题等权）|",
        "|---|---:|---|---:|---:|"]
    for relation, g in summary["ordering_by_relation"].items():
        lines.append(f"|{cell(relation)}|{g['questions']}/{g['planned_slots']}|{g['admitted_arity_counts']}|{pct(g['question_equal_exact'])}|{pct(g.get('question_equal_pair_accuracy'))}|")
    lines += ["",
        f"多选有效调用观察到 {mc['FP_option_events']} 次错误选项入选、{mc['FN_option_events']} 次正确选项漏选；分别涉及 {mc['responses_with_FP']} 和 {mc['responses_with_FN']} 个作答。另有 {mc['unknown_invalid_calls']} 个无效作答的选择情况未知。这些是选项事件，不是作答次数，也不证明模型的具体内部错误原因。", "",
        "|题号|典型错误选项|选项原文|选入/有效调用|",
        "|---|---|---|---:|"]
    options = [o for o in summary["MC_option_observations"] if not o["gold_selected"] and o["selected_calls"]]
    for o in sorted(options, key=lambda x: (-x["selected_calls"], x["candidate_id"], x["canonical_label"]))[:6]:
        lines.append(f"|{o['candidate_id']}|{o['canonical_label']}|{cell(o['text'])}|{o['selected_calls']}/{o['valid_calls']}|")
    contradiction = summary["explicit_option_contradictions"]
    lines += ["", "矛盾核查：" + ("仅统计经明确审核的指定选项组合。" if contradiction["annotation_supplied"] else "没有提供带精确证据的显式互斥选项审核标注，本报告不自动声称存在矛盾。")]
    for c in contradiction["checks"]:
        lines.append(f"- {c['candidate_id']} 的 {','.join(c['canonical_labels'])} 同选 {c['co_selected_calls']}/{c['denominator_calls']} 次；限定依据：{c['reason']}")
    cost = d["cost"]
    lines += ["", f"完整提示长度 min/median/max={d['prompt_tokens']['min']}/{d['prompt_tokens']['median']}/{d['prompt_tokens']['max']} tokens（含对话、题面、选项与提示模板，按固定请求统计）。执行耗时 {cost['live_wall_seconds']} 秒，含加载 {cost['model_load_wall_seconds']} 秒；这不是扩充、作者出题、审核与准备的总耗时。", "",
        f"本地 Solver 已知输入 {cost['known_prompt_tokens']}、输出 {cost['known_completion_tokens']}、总计 {cost['known_total_tokens']} tokens，usage 未知请求 {cost['requests_with_unknown_usage']}。未知量和本地货币费用均不按零费用解释。", "",
        "本轮仅验证固定流程能否扩充、出题、审核并完成 S0 诊断。没有反馈因果 A/B、难度提升、训练收益、未见泛化或 memory 作用的证据；下一步若主张这些效果，须另冻结相应对照。", "",
        "文件：`SUMMARY.json` 保留全部分母、来源模式、关系分层、错误与限制；`exports/SOLVER_RESPONSES.jsonl` 保留每次展示答案与还原答案；`exports/PER_QUESTION.json` 和 `per_question.csv` 保留逐题 exact/partial；`review/REPORT_RECOMPUTATION.json` 记录重算与输入绑定。", ""]
    return "\n".join(lines)


def build(base, run, contradiction_path=None):
    # Gate raw freeze and its exact terminal set BEFORE reading SCORES with Gold.
    freeze = read(run / "RAW_RESPONSE_FREEZE.json")
    require(freeze["status"] == "COMMITTED", "COMMITTED_RAW_FREEZE_REQUIRED")
    verify(freeze["manifest_binding"], run / "manifest.json")
    manifest = read(run / "manifest.json")
    require(manifest["namespace"] == NAMESPACE and manifest["solver_model"] == "S0_M_clean", "WRONG_NAMESPACE_OR_SOLVER")
    require(manifest["automatic_retry"] is False and manifest["source_truncation"] is False, "RETRY_OR_TRUNCATION")
    require(manifest["random_display"]["seed"] == 2026102004 and manifest["sampling_seed_start"] == 2026102040, "WRONG_SEEDS")
    expected = {run / "responses" / (r["request_id"] + ".json") for r in manifest["requests"]}
    require(len(expected) == len(manifest["requests"]) == len(freeze["files"]), "DUPLICATE_RAW_OR_REQUESTS")
    require({Path(b["path"]).resolve() for b in freeze["files"]} == expected == set((run / "responses").glob("*.json")), "EXACT_FROZEN_RAW_SET_REQUIRED")
    for record in freeze["files"]:
        verify(record)
    for record in manifest["public_input_bindings"]:
        verify(record)
    raws = {}
    for request in manifest["requests"]:
        raw = read(run / "responses" / (request["request_id"] + ".json"))
        require(raw["status"] in ("RETURNED", "BACKEND_ERROR", "NOT_SUBMITTED", "MISSING"), "NONTERMINAL_RAW")
        require(all(raw[k] == request[k] for k in ("request_id", "candidate_id", "root_id", "question_type", "slot", "seed", "prompt_sha256")), "RAW_IDENTITY_CHANGED")
        require(datetime.fromisoformat(raw["response_committed_at_utc"]) <= datetime.fromisoformat(freeze["created_at_utc"]), "RAW_AFTER_FREEZE")
        raws[request["request_id"]] = raw
    scores = read(run / "SCORES.json")
    verify(scores["manifest_binding"], run / "manifest.json")
    verify(scores["raw_response_freeze_binding"], run / "RAW_RESPONSE_FREEZE.json")
    require(datetime.fromisoformat(scores["gold_first_read_at_utc"]) >= datetime.fromisoformat(freeze["created_at_utc"]), "SCORING_GOLD_BEFORE_FREEZE")
    generation, source_freeze = read(base / "GENERATION_FREEZE.json"), read(base / "SOURCE_FREEZE.json")
    admission, ledger = read(base / "CONTENT_ADMISSION.json"), read(base / "exports/CANDIDATE_LEDGER.json")
    protocol = read(base / "EXECUTION_FREEZE.json")
    require(admission["namespace"] == generation["namespace"] == protocol["namespace"] == NAMESPACE, "CONTENT_NAMESPACE")
    require(admission["status"] in ("PASS", "FROZEN", "CONTENT_FROZEN"), "CONTENT_NOT_ADMITTED")
    verify(admission["all_slot_ledger"], base / "exports/CANDIDATE_LEDGER.json")
    verify(admission["public_questions"], base / "exports/PUBLIC_QUESTIONS.json")
    verify(protocol["generation_freeze"], base / "GENERATION_FREEZE.json")
    verify(protocol["source_freeze"], base / "SOURCE_FREEZE.json")
    verify(protocol["content_admission"], base / "CONTENT_ADMISSION.json")
    public_path = run / "PUBLIC_QUESTIONS.json"
    public = unique(read(public_path)["questions"], "candidate_id")
    require(public == unique(read(base / "exports/PUBLIC_QUESTIONS.json")["questions"], "candidate_id"), "PUBLIC_QUESTION_EXPORT_DRIFT")
    meta, slots = unique(ledger["records"], "candidate_id"), unique(protocol["slots"], "candidate_id")
    require(len(meta) == len(slots) == ledger["planned_slots"] == admission["planned_slots"] == 24 and set(meta) == set(slots), "24_SLOT_DENOMINATOR")
    require(set(meta) == {f"T4R-R{r}-{task}{i}" for r in range(1, 5) for task in ("MC", "ORD") for i in range(1, 4)}, "EXPECTED_TARGETED_CANDIDATE_IDS")
    require(all(m["status"] in ("CONTENT_PASS", "CONTENT_HOLD", "SOURCE_SHORTFALL") for m in meta.values()), "UNRECOGNIZED_CONTENT_STATUS")
    require(set(meta) == {s["candidate_id"] for s in generation["slots"]}, "GENERATION_SLOT_SET")
    require(len({q["root_id"] for q in meta.values()}) == len(generation["roots"]) == 4, "FOUR_ROOT_DENOMINATOR")
    require({q["candidate_id"] for q in meta.values() if q["status"] == "CONTENT_PASS"} == set(public) == {q["candidate_id"] for q in admission["questions"]}, "ADMITTED_SET_DRIFT")
    require(ledger["solver_scores_used_for_admission"] is False and admission["solver_scores_available"] is False, "SCORE_BASED_ADMISSION")
    require(ledger["counts"] == dict(Counter(q["status"] for q in meta.values())), "LEDGER_COUNT_DRIFT")
    missing = sorted(set(meta) - set(public))
    require(manifest["missing_candidate_ids"] == scores["missing_candidate_ids"] == missing, "MISSING_SLOT_DENOMINATOR")
    require(manifest["production_denominator"] == scores["production_denominator"] == protocol["budget"]["production_planned_slots"] == 24, "PRODUCTION_DENOMINATOR_DRIFT")
    require(admission["admitted_count"] == protocol["budget"]["initial_question_slots"] == scores["denominator_questions"] == manifest["question_count"] == len(public), "ADMITTED_QUESTION_DENOMINATOR")
    require(scores["production_unadmitted_slots"] == len(missing) and scores["planned_root_count"] == manifest["planned_root_count"] == 4, "PLANNED_FAILURE_OR_ROOT_DENOMINATOR")
    require(scores["denominator_roots"] == len({q["root_id"] for q in public.values()}), "ADMITTED_ROOT_DENOMINATOR")
    require(len(scores["planned_slot_outcomes"]) == 24 and {s["candidate_id"] for s in scores["planned_slot_outcomes"]} == set(meta), "SCORES_PLANNED_SLOT_OUTCOMES")
    require(scores["training_started"] is False and manifest["training_started"] is False and scores["use_as_final_eval"] is False, "PILOT_SCOPE_DRIFT")
    manifest["_run"] = str(run)
    calls, gold = reconstruct(manifest, scores, public, read(run / "DISPLAY_MAPPINGS.json"), raws)
    sources = source_stats(generation, source_freeze, {q["root_id"] for q in public.values()})
    require(Counter(s["mode"] for s in sources.values()) == {"targeted_extension": 2, "reuse_existing": 2}, "TWO_EXTEND_TWO_REUSE")
    questions, options = [], []
    native_questions = unique(scores["questions"], "candidate_id")
    require(set(native_questions) == set(public), "SCORED_QUESTION_SET")
    for qid, q in public.items():
        subset, native, m = [r for r in calls if r["candidate_id"] == qid], native_questions[qid], meta[qid]
        field = "options" if q["question_type"] == "multi_select" else "items"
        n, nc = len(q[field]), len(subset)
        require(nc == (18 if field == "options" else min(12, math.factorial(n))), "FIXED_REQUEST_COUNT")
        require(q["source_sha256"] == sources[q["root_id"]]["source"]["sha256"], "QUESTION_FINAL_SOURCE")
        row = {k: v for k, v in native.items() if k not in ("items", "pairs", "design")}
        for k, v in {"item_count": n, "denominator_calls": nc, "valid_calls": sum(r["valid"] for r in subset), "invalid_calls": sum(not r["valid"] for r in subset), "exact_calls": sum(r["exact"] for r in subset), "exact_rate": mean(r["exact"] for r in subset)}.items():
            same(row[k], v, qid + ":" + k)
        same(row["stratum"], "incorrect" if row["exact_calls"] == 0 else "stably_correct" if row["exact_calls"] == nc else "occasionally_correct", qid + ":stratum")
        row.update({k: m.get(k) for k in ("template_id", "primary_relation", "timeline_kind", "fact_family_id", "self_contained_operands_or_constraints", "strong_intrinsic_procedure_cues", "easy_control", "reuse_lineage")})
        row.update(question=q["question"], canonical_gold=gold[qid], source_mode=sources[q["root_id"]]["mode"],
            gold_count=len(gold[qid]) if field == "options" else None,
            nonexact_with_partial_calls=sum(r["nonexact_with_partial"] for r in subset))
        row["nonexact_with_partial_rate"] = row["nonexact_with_partial_calls"] / nc
        if field == "options":
            row.update(mean_jaccard=mean(r["jaccard"] for r in subset), fp_calls=sum(r["false_positive_options"] or 0 for r in subset), fn_calls=sum(r["false_negative_options"] or 0 for r in subset))
            row.update(partial_metric=row["mean_jaccard"], partial_metric_name="Jaccard; invalid=0")
            same(row["fp_calls"], native["fp_calls"], qid + ":FP")
            same(row["fn_calls"], native["fn_calls"], qid + ":FN")
            same(m["actual_gold_count"], len(gold[qid]), qid + ":gold_count")
            for label, text in q["options"].items():
                valid = [r for r in subset if r["valid"]]
                options.append({"candidate_id": qid, "canonical_label": label, "text": text, "gold_selected": label in gold[qid],
                    "denominator_calls": nc, "valid_calls": len(valid), "unknown_invalid_calls": nc - len(valid),
                    "selected_calls": sum(label in r["canonical_prediction"] for r in valid),
                    "FP_events": sum(label in r["FP_labels"] for r in valid), "FN_events": sum(label in r["FN_labels"] for r in valid)})
        else:
            row.update(correct_pairs=sum(r["pair_correct"] for r in subset), total_pairs=sum(r["pair_total"] for r in subset))
            same(row["correct_pairs"], native["correct_pairs"], qid + ":correct_pairs")
            same(row["total_pairs"], native["total_pairs"], qid + ":total_pairs")
            row["pair_accuracy"] = row["correct_pairs"] / row["total_pairs"]
            same(row["pair_accuracy"], native["pair_accuracy"], qid + ":pair_rate")
            row.update(partial_metric=row["pair_accuracy"], partial_metric_name="Gold pair accuracy; invalid=0")
        questions.append(row)
        for call in subset:
            call.update(template_id=row["template_id"], primary_relation=row["primary_relation"], source_mode=row["source_mode"])
    overall = grouped(questions)
    for key in ("denominator_calls", "valid_calls", "exact_calls"):
        expected = len(calls) if key == "denominator_calls" else sum(r["valid" if key == "valid_calls" else "exact"] for r in calls)
        same(scores[key], expected, "OVERALL:" + key)
    same(scores["question_equal_exact"], overall["question_equal_exact"], "QUESTION_EQUAL_EXACT")
    same(scores["root_equal_exact"], overall["root_equal_exact"], "ROOT_EQUAL_EXACT")
    by_type = {typ: grouped([q for q in questions if q["question_type"] == typ]) for typ in TYPES}
    for typ in TYPES:
        for key in ("questions", "calls", "question_equal_exact", "root_equal_exact"):
            same(by_type[typ][key], scores["by_type"][typ][key], typ + ":" + key)
    def groups(key, typ=None):
        result = {}
        for value in sorted({str(m.get(key) or "UNSPECIFIED") for m in meta.values() if typ is None or m["question_type"] == typ}):
            planned = [m for m in meta.values() if str(m.get(key) or "UNSPECIFIED") == value and (typ is None or m["question_type"] == typ)]
            qs = [q for q in questions if q["candidate_id"] in {m["candidate_id"] for m in planned}]
            result[value] = {**grouped(qs), "planned_slots": len(planned), "unadmitted": len(planned)-len(qs), "production_status_counts": dict(Counter(m["status"] for m in planned)), "admitted_arity_counts": dict(Counter(str(q["item_count"]) for q in qs))}
        return result
    arity = {}
    for n in (3, 4, 5, 6):
        qs = [q for q in questions if q["question_type"] == "ordering" and q["item_count"] == n]
        arity[str(n)] = {**grouped(qs), "planned_slots_with_actual_arity": sum(m["question_type"] == "ordering" and m.get("item_count") == n for m in meta.values())}
        if qs:
            for key, val in scores["by_arity"]["ordering:" + str(n)].items():
                if key in arity[str(n)]:
                    same(arity[str(n)][key], val, f"ARITY{n}:{key}")
    roots = {}
    for root in sources:
        planned = [m for m in meta.values() if m["root_id"] == root]
        qs = [q for q in questions if q["root_id"] == root]
        require(len(planned) == 6, "SIX_PLANNED_SLOTS_PER_ROOT")
        roots[root] = {"planned_slots": 6, "admitted": len(qs), "unadmitted": 6-len(qs), "MC": sum(q["question_type"] == "multi_select" for q in qs),
            "ordering": sum(q["question_type"] == "ordering" for q in qs), "production_status_counts": dict(Counter(m["status"] for m in planned)), "scores": grouped(qs)}
    cost = scores["cost"]
    returned = [r for r in raws.values() if r["status"] == "RETURNED"]
    for key, rawkey in (("known_prompt_tokens", "prompt_token_count"), ("known_completion_tokens", "completion_token_count")):
        same(cost[key], sum(r.get("metadata", {}).get(rawkey) for r in returned if type(r.get("metadata", {}).get(rawkey)) is int), "USAGE:" + key)
    same(cost["known_total_tokens"], cost["known_prompt_tokens"] + cost["known_completion_tokens"], "TOTAL_USAGE")
    if cost.get("execution_binding"):
        verify(cost["execution_binding"])
    prompt_lengths = [r["prompt_tokens"] for r in calls]
    mc_calls = [r for r in calls if r["question_type"] == "multi_select"]
    source_mode_groups = {mode: grouped([q for q in questions if q["source_mode"] == mode]) for mode in ("targeted_extension", "reuse_existing")}
    # A single partial metric for mixed types would mix Jaccard with pair credit.
    overall.pop("question_equal_partial_metric", None)
    for g in source_mode_groups.values():
        g.pop("question_equal_partial_metric", None)
    summary = {"schema_version": "targeted4root_pilot_summary_v1", "status": "COMPLETE_OPERATIONAL_DIAGNOSTIC" if scores["status"] == "COMPLETE" else "FAILED_WITH_FIXED_DENOMINATORS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "namespace": NAMESPACE, "production_planned_slots": 24,
        "production_status_counts": ledger["counts"], "admitted_questions": len(questions), "unadmitted_slots": [m for m in meta.values() if m["status"] != "CONTENT_PASS"],
        "production_yield": {"admitted_slots": len(questions), "planned_slots": 24, "rate": len(questions)/24,
            "roots_with_any_admitted_question": sum(r["admitted"] > 0 for r in roots.values()), "roots_all_six_admitted": sum(r["admitted"] == 6 for r in roots.values()), "planned_roots": 4},
        "sources": sources, "roots": roots, "overall": overall, "by_type": by_type, "by_source_mode_descriptive_only": source_mode_groups,
        "by_template": groups("template_id"), "ordering_by_relation": groups("primary_relation", "ordering"), "ordering_by_arity": arity,
        "ordering_unadmitted_slots_without_actual_arity": [m["candidate_id"] for m in meta.values() if m["question_type"] == "ordering" and m["status"] != "CONTENT_PASS" and m.get("item_count") is None],
        "gold_count_distribution_MC": dict(Counter(str(q["gold_count"]) for q in questions if q["question_type"] == "multi_select")),
        "planned_MC_gold_count_distribution": dict(Counter(str(m["target_gold_count"]) for m in meta.values() if m["question_type"] == "multi_select")),
        "fact_families_author_annotated": len({q["fact_family_id"] for q in questions if q.get("fact_family_id")}),
        "MC_observable_errors": {"denominator_calls": len(mc_calls), "valid_calls": sum(r["valid"] for r in mc_calls), "unknown_invalid_calls": sum(not r["valid"] for r in mc_calls),
            "FP_option_events": sum(r["false_positive_options"] or 0 for r in mc_calls), "FN_option_events": sum(r["false_negative_options"] or 0 for r in mc_calls),
            "responses_with_FP": sum(bool(r["false_positive_options"]) for r in mc_calls), "responses_with_FN": sum(bool(r["false_negative_options"]) for r in mc_calls)},
        "MC_option_observations": options, "explicit_option_contradictions": contradictions(contradiction_path, base / "exports/PUBLIC_QUESTIONS.json", public, calls),
        "diagnosis": {"calls": len(calls), "valid_calls": sum(r["valid"] for r in calls), "invalid_calls": sum(not r["valid"] for r in calls), "exact_calls": sum(r["exact"] for r in calls),
            "prompt_tokens": {"min": min(prompt_lengths, default=None), "median": median(prompt_lengths) if prompt_lengths else None, "max": max(prompt_lengths, default=None), "includes":"full dialogue + question/items + prompt/chat template; repeated request weighted"},
            "cost": cost, "source_authoring_wall_seconds": None, "source_authoring_token_usage": source_freeze.get("usage", None), "authoring_usage_note": "Only explicit source-freeze usage is exposed, if present. Missing stage time/usage is unknown; Solver wall is not total pipeline wall.",
            "raw_freeze_verified_before_gold_in_scores_read": True, "private_gold_json_opened_by_report": False},
        "questions": questions, "solver": protocol["solver"], "limits": LIMITS, "training_started": False, "new_training_admitted": 0,
        "feedback_causal_effect_established": False, "controlled_difficulty_increase_established": False, "training_benefit_established": False,
        "dev_test_calls": 0, "source_masked_control_performed": False}
    paths = [base / name for name in ("GENERATION_FREEZE.json", "SOURCE_FREEZE.json", "CONTENT_ADMISSION.json", "EXECUTION_FREEZE.json", "exports/CANDIDATE_LEDGER.json", "exports/PUBLIC_QUESTIONS.json")]
    paths += [run / name for name in ("manifest.json", "RAW_RESPONSE_FREEZE.json", "SCORES.json", "PUBLIC_QUESTIONS.json", "DISPLAY_MAPPINGS.json")]
    if contradiction_path:
        paths.append(contradiction_path)
    inputs = [binding(p) for p in paths]
    summary["input_bindings"] = inputs
    fields = ["candidate_id", "root_id", "source_mode", "question_type", "template_id", "primary_relation", "item_count", "gold_count", "denominator_calls", "valid_calls", "invalid_calls", "exact_calls", "exact_rate", "partial_metric_name", "partial_metric", "nonexact_with_partial_calls", "mean_jaccard", "pair_accuracy", "correct_pairs", "total_pairs", "fp_calls", "fn_calls", "stratum", "fact_family_id"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(questions)
    encode = lambda value: json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    recomputation = {"status": "PASS", "created_at_utc": summary["created_at_utc"], "request_count": len(calls), "question_count": len(questions),
        "scope": "Literal-box parsing, canonical content hashing/remapping, exact/FP/FN/Jaccard/ordering pair recomputation, fixed denominators and source/public bindings; does not newly validate every Gold semantically.",
        "private_json_opened": False, "inference_calls": 0, "inputs": inputs, "report_code": binding(Path(__file__))}
    outputs = {base / "exports/SOLVER_RESPONSES.jsonl": "".join(json.dumps(c, ensure_ascii=False)+"\n" for c in calls),
        base / "exports/PER_QUESTION.json": encode({"questions": questions}), base / "per_question.csv": buffer.getvalue(),
        base / "SUMMARY.json": encode(summary), base / "REPORT_ZH.md": markdown(summary), base / "review/REPORT_RECOMPUTATION.json": encode(recomputation)}
    return outputs, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-freeze-authorized", action="store_true", help="Parent authorizes reporting only after raw freeze and scoring.")
    parser.add_argument("--contradictions", type=Path, help="Optional explicitly reviewed incompatibility annotations; no automatic inference.")
    args = parser.parse_args()
    require(args.post_freeze_authorized, "EXPLICIT_POST_FREEZE_REPORT_EXECUTION_FLAG_REQUIRED")
    outputs, summary = build(HERE, HERE / "runtime/direct_v1", args.contradictions.resolve() if args.contradictions else None)
    require(all(not p.exists() for p in outputs), "REPORT_OUTPUT_EXISTS_NO_OVERWRITE")
    # All consistency checks and rendering finish before any report output write.
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(text)
    print(json.dumps({"status": summary["status"], "admitted_questions": summary["admitted_questions"], "production_status_counts": summary["production_status_counts"], "output_files": [str(p) for p in outputs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
