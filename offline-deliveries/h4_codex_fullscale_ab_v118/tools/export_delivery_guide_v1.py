"""Add a mechanically reconciled Chinese entry point after all final exports.

Exclusive additions only. No model calls, semantic judgments, source edits,
question edits, Gold edits, selection edits, training, or pool promotion.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
B = HERE.parent
NEW_ACTIONS = {"SCREEN", "INHERIT_SCREEN"}
SCENES = {f"{route}_S{number:02d}" for route in "AB" for number in range(1, 4)}


class Inputs:
    def __init__(self):
        self.bindings = {}

    def bind(self, path):
        path = Path(path).resolve()
        value = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if str(path) in self.bindings:
            assert self.bindings[str(path)] == value, "INPUT_CHANGED_DURING_EXPORT:" + str(path)
        self.bindings[str(path)] = value
        return value

    def read(self, path):
        self.bind(path)
        return json.loads(Path(path).read_text())

    def verify(self, value):
        assert self.bind(value["path"]) == value, "BOUND_INPUT_CHANGED:" + value["path"]


def distribution(questions):
    assert all(q["question_type"] in {"multi_select", "ordering"} for q in questions)
    gold = Counter(str(len(q["gold"])) for q in questions if q["question_type"] == "multi_select")
    assert set(gold) <= {"2", "3", "4", "5", "6"}
    return {"questions": len(questions),
            "type_counts": dict(Counter(q["question_type"] for q in questions)),
            "gold_counts": dict(gold)}


def summarize_entries(entries):
    valid = [r for r in entries if r["valid"]]
    result = {"calls": len(entries), "valid_calls": len(valid),
              "invalid_calls": len(entries) - len(valid),
              "correct": sum(r["exact"] for r in entries),
              "exact_rate": sum(r["exact"] for r in entries) / len(entries) if entries else None,
              "partial_mean": str(sum((r["partial"] for r in valid), Fraction()) / len(valid)) if valid else None}
    mcq = [r for r in valid if "predicted_size" in r]
    if mcq:
        result.update(false_positive_total=sum(r["fp"] for r in mcq),
                      false_negative_total=sum(r["fn"] for r in mcq),
                      calls_with_false_positive=sum(r["fp"] > 0 for r in mcq),
                      calls_with_false_negative=sum(r["fn"] > 0 for r in mcq),
                      selected_all_six_calls=sum(r["predicted_size"] == 6 for r in mcq),
                      over_count_calls=sum(r["predicted_size"] > r["gold_size"] for r in mcq),
                      under_count_calls=sum(r["predicted_size"] < r["gold_size"] for r in mcq),
                      mean_predicted_size=statistics.mean(r["predicted_size"] for r in mcq))
    return result


def recompute_wave(batch, wave, exported_metric, inputs):
    scope = batch / "evaluations" / wave
    pool = inputs.read(scope / "EVALUATION_POOL.json")["questions"]
    qmap = {q["candidate_id"]: q for q in pool}
    assert len(qmap) == len(pool)
    adapter = inputs.read(scope / "ADAPTER_BINDINGS.json")
    assert adapter["candidate_count"] == len(pool)
    assert adapter["calls_per_model"] == sum(18 if q["question_type"] == "multi_select" else 12 for q in pool)
    audit = inputs.read(scope / "INDEPENDENT_SOLVER_AUDIT.json")
    assert audit["status"] == "PASS" and not audit["errors"]
    inputs.verify(exported_metric["pairing"]["audit_binding"])
    assert exported_metric["pairing"]["calls_per_model"] == adapter["calls_per_model"]
    waves = {}
    paired_requests = None
    paired_gold = None
    for side in ("base", "trained"):
        run = scope / "run" / side
        manifest = inputs.read(run / "manifest.json")
        requests = {r["request_id"]: r for r in manifest["requests"]}
        assert len(requests) == manifest["request_count"] == adapter["calls_per_model"]
        gold = inputs.read(run / "gold.json")
        assert set(gold) == set(qmap)
        if paired_requests is None:
            paired_requests, paired_gold = requests, gold
        else:
            assert gold == paired_gold
            assert set(requests) == set(paired_requests)
            for request_id, request in requests.items():
                first = paired_requests[request_id]
                for key in ("candidate_id", "root_id", "question_type", "seed", "slot", "variant", "prompt_sha256"):
                    assert request[key] == first[key], "UNPAIRED_SOLVER_REQUEST:" + request_id
        complete = inputs.read(run / "COMPLETE.json")
        wrapper = inputs.read(scope / (side.upper() + "_COMPLETE.json"))
        assert complete["status"] == "COMPLETE"
        assert complete["actual_call_count"] == wrapper["actual_calls"] == len(requests)
        assert audit["results"][side]["calls"] == len(requests)
        buckets = defaultdict(list)
        per_candidate = defaultdict(list)
        seen = set()
        for path in sorted((run / "responses").glob("*.json")):
            row = inputs.read(path)
            request_id = row["request_id"]
            assert request_id not in seen and request_id in requests
            seen.add(request_id)
            request = requests[request_id]
            for key in ("candidate_id", "root_id", "question_type", "seed", "slot"):
                assert row[key] == request[key]
            assert row["variant_id"] == request["variant"]["variant_id"]
            assert row["evaluation_valid"] is True, "INVALID_RAW_SOLVER_CALL:" + request_id
            match = re.fullmatch(r"\s*<box>\s*([^<>]*)\s*</box>\s*", row["raw_output"])
            assert match
            labels = [v.strip().upper() for v in match.group(1).split(",") if v.strip()]
            assert len(labels) == len(set(labels))
            mapping = request["variant"]["display_to_content"]
            assert all(v in mapping for v in labels)
            predicted = [mapping[v] for v in labels]
            cid = row["candidate_id"]
            q = qmap[cid]
            expected = gold[cid]  # Canonical content IDs, never public display labels.
            assert len(expected) == len(q["gold"])
            entry = {"valid": True}
            keys = [q["question_type"]]
            if q["question_type"] == "multi_select":
                fp = len(set(predicted) - set(expected))
                fn = len(set(expected) - set(predicted))
                partial = Fraction(len(set(predicted) & set(expected)), len(expected))
                if len(expected) != len(mapping):
                    partial -= Fraction(fp, len(mapping) - len(expected))
                exact = set(predicted) == set(expected)
                entry.update(fp=fp, fn=fn, predicted_size=len(predicted), gold_size=len(expected))
                keys.append("gold" + str(len(expected)))
                if 2 <= len(expected) <= 5:
                    keys.append("multi_select_gold2_5")
                if q.get("role") == "control6":
                    keys.append("gold6_control")
            else:
                assert len(predicted) == 6 and set(predicted) == set(expected)
                ranks = {v: i for i, v in enumerate(predicted)}
                wrong = sum(ranks[a] > ranks[b] for a, b in itertools.combinations(expected, 2))
                partial = Fraction(15 - 2 * wrong, 15)
                exact = predicted == expected
            assert row["score"]["exact_match"] == exact
            assert Fraction(row["score"]["partial_score_fraction"]) == partial
            entry.update(exact=exact, partial=partial)
            for key in keys:
                buckets[key].append(entry)
            per_candidate[cid].append((row["variant_id"], entry))
        assert seen == set(requests)
        results = inputs.read(run / "RESULTS.json")
        assert not results["call_set_errors"]
        assert {r["candidate_id"] for r in results["questions"]} == set(qmap)
        for result in results["questions"]:
            rows = per_candidate[result["candidate_id"]]
            by_variant = defaultdict(list)
            for variant_id, entry in rows:
                by_variant[variant_id].append(entry["partial"])
            is_ms = qmap[result["candidate_id"]]["question_type"] == "multi_select"
            assert len(by_variant) == (6 if is_ms else 12)
            assert all(len(values) == (3 if is_ms else 1) for values in by_variant.values())
            partial_mean = sum((sum(v, Fraction()) / len(v) for v in by_variant.values()), Fraction()) / len(by_variant)
            assert partial_mean == Fraction(result["full_partial_mean"])
            assert Fraction(sum(r["exact"] for _, r in rows), len(rows)) == Fraction(result["full_exact_rate"])
            assert result["selected_by_policy"] == (0 < partial_mean < 1)
        summary = {key: summarize_entries(rows) for key, rows in buckets.items()}
        for key, values in summary.items():
            assert values == exported_metric["by_side"][side][key], "FINAL_SOLVER_METRIC_MISMATCH:" + wave + ":" + side + ":" + key
        waves[side] = {"calls": len(seen), "by_type_and_gold": summary}
    return {"calls_per_model": adapter["calls_per_model"], "candidate_count": len(pool), "by_side": waves}


def collect(output, batch=B):
    inputs = Inputs()
    final = inputs.read(output / "FINAL_STATUS.json")
    manifest = inputs.read(batch / "MANIFEST.json")
    attempts = inputs.read(output / "ALL_ATTEMPTS_INDEX.json")
    content = inputs.read(output / "CONTENT_QUESTIONS.json")["questions"]
    selected = inputs.read(output / "SELECTED_CANDIDATES.json")["questions"]
    for name in ("REPORT_ZH.md", "EXECUTION_REPORT_ZH.md", "EXCLUDED_CANDIDATES.json",
                 "PROCEDURE_DEVIATIONS.json", "PROCEDURE_NOTES_ZH.md", "QUESTION_STAGE_LEDGER.json"):
        inputs.bind(output / name)
    assert final["status"] == "FULL_FIXED_POPULATION_PROCESSED"
    roots = {r["root_id"]: r for r in final["roots"]}
    indexed = {r["root_id"]: r for r in attempts["roots"]}
    fixed = {r["root_id"]: r for r in manifest["roots"]}
    denominator = manifest["population_denominator"]
    assert len(roots) == len(final["roots"]) == len(indexed) == len(attempts["roots"]) == len(fixed) == denominator
    assert set(roots) == set(indexed) == set(fixed)
    assert final["population_denominator"] == final["terminal_roots"] == attempts["population_denominator"] == attempts["terminal_roots"] == denominator
    assert all(r["terminal"] and r["status"] == indexed[rid]["status"] and r["action"] == fixed[rid]["action"] for rid, r in roots.items())
    statuses = dict(Counter(r["status"] for r in roots.values()))
    assert statuses == final["statuses"] == attempts["statuses"]
    for key in ("training_started", "training_pool_merged", "external_semantic_certification", "unseen_holdout", "tokens_are_billed_usage"):
        assert final[key] is False
    qmap = {q["candidate_id"]: q for q in content}
    smap = {q["candidate_id"]: q for q in selected}
    assert len(qmap) == len(content) == final["content_questions"]
    assert len(smap) == len(selected) == final["selected_candidates"]
    assert set(smap) <= set(qmap) and all(q == qmap[cid] for cid, q in smap.items())
    assert all(q.get("role") != "control6" and (q["question_type"] != "multi_select" or len(q["gold"]) != 6) for q in selected)
    cohorts = {}
    for name, actions in (("new", NEW_ACTIONS), ("reused", {"REUSE"}), ("all", NEW_ACTIONS | {"REUSE"})):
        rows = [r for r in roots.values() if r["action"] in actions]
        ids = {r["root_id"] for r in rows}
        cq = [q for q in content if q["root_id"] in ids]
        sq = [q for q in selected if q["root_id"] in ids]
        cohorts[name] = {"roots": len(rows), "content": distribution(cq), "selected": distribution(sq)}
    for prefix, questions in (("content", content), ("selected", selected)):
        dist = distribution(questions)
        assert dist["type_counts"] == final[prefix + "_type_counts"]
        assert dist["gold_counts"] == final[prefix + "_gold_counts"]
    assert cohorts["all"]["content"]["questions"] == len(content)
    assert cohorts["new"]["content"]["questions"] == final["new_content_questions"]
    assert cohorts["new"]["selected"]["questions"] == final["new_selected_candidates"]
    assert cohorts["new"]["roots"] == final["new_fixed_attempt_roots"]
    assert cohorts["reused"]["roots"] == final["reused_complete_roots"]
    counts, chosen = Counter(q["root_id"] for q in content), Counter(q["root_id"] for q in selected)
    for rid, root in roots.items():
        assert root["content_questions"] == counts[rid] and root["selected_candidates"] == chosen[rid]
        if root["action"] in NEW_ACTIONS and root["status"] != "NEW_CONTENT_COMPLETED":
            assert counts[rid] == chosen[rid] == 0
    success = [r for r in roots.values() if r["status"] == "NEW_CONTENT_COMPLETED"]
    assert all(r["action"] in NEW_ACTIONS and r["complete_dialogue"] and counts[r["root_id"]] > 0 for r in success)
    main = [r for r in roots.values() if r["complete_dialogue"]]
    new_main = [r for r in main if r["action"] in NEW_ACTIONS]
    assert len(main) == final["complete_dialogues"] and len(new_main) == final["new_complete_dialogues"]
    for root in main:
        inputs.verify(root["source_binding"])
        inputs.bind(root["readable_dialogue_path"])
        inputs.bind(root["readable_questions_path"])
    source_only = [r for r in indexed.values() if r["source_only_complete_qa_hold"]]
    assert {r["root_id"] for r in source_only} == set(attempts["source_only_complete_qa_hold_roots"])
    assert not {r["root_id"] for r in main} & {r["root_id"] for r in source_only}
    for row in source_only:
        assert row["six_scene_reviews_ready"] and row["qa_procedure_hold"]
        assert counts[row["root_id"]] == chosen[row["root_id"]] == 0
        inputs.verify(row["source_only_readable_dialogue_binding"])
    lengths = {}
    for name, rows in (("main_all", main), ("main_new", new_main)):
        lengths[name] = {"dialogues": len(rows),
                         "mean_full_tokens": statistics.mean(r["dialogue_tokens_speakers_and_body"] for r in rows) if rows else None,
                         "mean_six_scene_tokens": statistics.mean(r["new_dialogue_tokens_speakers_and_body"] for r in rows) if rows else None}
    assert math.isclose(lengths["main_all"]["mean_full_tokens"], final["mean_dialogue_tokens_speakers_and_body"], abs_tol=1e-9)
    assert math.isclose(lengths["main_all"]["mean_six_scene_tokens"], final["mean_new_dialogue_tokens_speakers_and_body"], abs_tol=1e-9)
    naturalness = {}
    success_ids = {r["root_id"] for r in success}
    for name, rows in (("new_content_success_scenes", success),
                       ("all_new_attempt_reviewed_scenes", [r for r in roots.values() if r["action"] in NEW_ACTIONS])):
        statuses_seen, decisions = Counter(), Counter()
        review_roots = set()
        for root in rows:
            rid = root["root_id"]
            unit = Path(indexed[rid]["unit_path"])
            folder = unit / "roots" / unit.name
            paths = sorted(folder.glob("scenes/*/SOURCE_REVIEW.json"))
            stage_bindings = {x["path"]: x for x in indexed[rid]["stage_artifacts"]}
            if rid in success_ids:
                assert {p.parent.name for p in paths} == SCENES and len(paths) == 6
            local = Counter()
            for path in paths:
                inputs.verify(stage_bindings[str(path.resolve())])
                review = inputs.read(path)
                assert review["root_id"] == rid
                status = review["checks"]["naturalness"]["status"]
                assert status in {"PASS", "WARNING", "PASS_WITH_WARNING", "FAIL"}
                statuses_seen[status] += 1
                local["naturalness:" + status] += 1
                decisions[review["decision"]] += 1
                review_roots.add(rid)
                if rid in success_ids:
                    assert review["decision"] == "READY" and status != "FAIL"
            if "scene_check_counts" in root:
                assert dict(local) == {k: v for k, v in root["scene_check_counts"].items() if k.startswith("naturalness:")}
        naturalness[name] = {"reviewed_roots": len(review_roots), "reviews": sum(statuses_seen.values()),
                             "status_counts": {k: statuses_seen[k] for k in ("PASS", "WARNING", "PASS_WITH_WARNING", "FAIL")},
                             "decision_counts": dict(decisions)}
    assert naturalness["new_content_success_scenes"]["reviews"] == len(success) * 6
    solver = {wave: recompute_wave(batch, wave, final["solver"][wave], inputs) for wave in ("validation", "production")}
    overall = {}
    for side in ("base", "trained"):
        overall[side] = {}
        for key in ("multi_select_gold2_5", "ordering", "gold6_control"):
            values = [v["by_side"][side]["by_type_and_gold"].get(key, {"correct": 0, "calls": 0}) for v in solver.values()]
            correct, calls = sum(v["correct"] for v in values), sum(v["calls"] for v in values)
            overall[side][key] = {"correct": correct, "calls": calls, "exact_rate": correct / calls if calls else None}
    inputs.bind(Path(__file__))
    facts = {"status": "MECHANICALLY_RECONCILED_DELIVERY_GUIDE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "population": denominator, "terminal_roots": final["terminal_roots"], "statuses": statuses,
             "new_content_success_roots": len(success), "reused_complete_roots": cohorts["reused"]["roots"],
             "main_complete_dialogues": len(main), "main_new_complete_dialogues": len(new_main),
             "main_new_complete_statuses": dict(Counter(r["status"] for r in new_main)),
             "source_only_qa_procedure_hold_roots": [r["root_id"] for r in source_only],
             "readable_complete_dialogues_including_source_only": len(main) + len(source_only),
             "cohorts": cohorts, "tokens": {"tokenizer": final["tokenizer"], "tokens_are_billed_usage": False, **lengths},
             "naturalness_existing_labels": naturalness, "solver_new_content_only": {"waves": solver, "overall": overall},
             "training_started": False, "training_pool_merged": False, "external_semantic_certification": False,
             "new_semantic_judgment_by_export": False, "unseen_holdout": False,
             "scope": "Mechanical aggregation of retained exports, canonical raw answers, and existing review labels; not new semantic certification.",
             "input_bindings": sorted(inputs.bindings.values(), key=lambda x: x["path"])}
    return facts


def rate(value):
    return f"{value['correct']}/{value['calls']}（{100 * value['exact_rate']:.2f}%）" if value["calls"] else "0/0"


def render(f):
    lines = ["# 本批交付入口", "",
             f"固定 {f['population']} 个 root 已全部终态；本批内容题成功 {f['new_content_success_roots']} 根，复用历史完成 {f['reused_complete_roots']} 根。全部终态包含失败/HOLD，不代表全部成功。", "",
             f"主交付完整对话 {f['main_complete_dialogues']} 份，其中本批新增 {f['main_new_complete_dialogues']} 份（状态分布：{f['main_new_complete_statuses']}），历史复用 {f['reused_complete_roots']} 份。",
             f"另有 {len(f['source_only_qa_procedure_hold_roots'])} 份六场源 READY、QA 程序 HOLD 的 source-only 对话：{', '.join(f['source_only_qa_procedure_hold_roots']) or '无'}。含此附录共有 {f['readable_complete_dialogues_including_source_only']} 份可读完整对话。source-only 未产出内容通过题或入选候选；QA 程序 HOLD 不等于源语义失败。", "",
             "|范围|题集|总题数|多选|排序|Gold2|Gold3|Gold4|Gold5|Gold6|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cohort, label in (("new", "本批新增"), ("reused", "历史复用"), ("all", "合计")):
        for kind, title in (("content", "内容通过"), ("selected", "Solver 入选候选")):
            d = f["cohorts"][cohort][kind]
            values = [d["questions"], d["type_counts"].get("multi_select", 0), d["type_counts"].get("ordering", 0)] + [d["gold_counts"].get(str(k), 0) for k in range(2, 7)]
            lines.append("|" + "|".join([label, title, *map(str, values)]) + "|")
    lines += ["", "Gold6 为对照，全部不入选。新增候选使用既有训练后模型的完整有效调用、等展示 partial 均分 0<均分<1；复用候选保留历史选择与 v114 隔离/覆盖，没有重新跑 Solver。", "",
              "下表仅汇总本批新增内容题的配对 Solver：", "",
              "|范围|M0 完全正确/调用|既有训练后模型 完全正确/调用|", "|---|---:|---:|"]
    for key, label in (("multi_select_gold2_5", "多选 Gold2–5"), ("ordering", "排序"), ("gold6_control", "Gold6 对照")):
        lines.append(f"|{label}|{rate(f['solver_new_content_only']['overall']['base'][key])}|{rate(f['solver_new_content_only']['overall']['trained'][key])}|")
    lines += ["", "两波各模型调用数：" + "；".join(f"{wave} {v['calls_per_model']} 次/模型" for wave, v in f["solver_new_content_only"]["waves"].items()) + "。", "同题展示和同 family 多选变体相关，调用数不是独立题数；这些结果不构成本轮新训练效果，全部 root 均有训练侧/会话曝光。", ""]
    for key, label in (("main_all", "全部主交付"), ("main_new", "其中本批新增主交付")):
        t = f["tokens"][key]
        lines.append(f"{label} {t['dialogues']} 份：完整对话平均 {t['mean_full_tokens']:.2f} tokens，新增六场平均 {t['mean_six_scene_tokens']:.2f} tokens。")
    lines += [f"长度沿用 {f['tokens']['tokenizer']} tokenizer 的 speaker＋正文统计；上述均值不含 source-only 附录，产物 tokens 不是 API usage 或计费量。", "",
              "|既有自然度审核统计范围|审核场数|PASS|WARNING|PASS_WITH_WARNING|FAIL|", "|---|---:|---:|---:|---:|---:|"]
    for key, label in (("new_content_success_scenes", "本批内容题成功根的六场"), ("all_new_attempt_reviewed_scenes", "全部本批尝试中已有审核的场景")):
        n = f["naturalness_existing_labels"][key]
        lines.append("|" + "|".join([label, str(n["reviews"]), *[str(n["status_counts"][s]) for s in ("PASS", "WARNING", "PASS_WITH_WARNING", "FAIL")]]) + "|")
    lines += ["", "这是原 SOURCE_REVIEW 中每场累计源审核的标签计数，并非独立对话样本，没有新增语义审核。WARNING 与 PASS_WITH_WARNING 原样分列；内容通过仍可能有清单化、服务转述公式化等自然度限制。第二行包含后来停止或失败根，不能当作入选源的失败率。", "",
              "可读完整对话与对应题答在 [roots/](roots/) 下各 root 的 FULL_DIALOGUE.md / QUESTIONS_AND_ANSWERS.md；仅源附录在 [source_only/](source_only/)。", "",
              "- [全部内容通过题、答案与完整引用](CONTENT_QUESTIONS.json)",
              "- [Solver 入选暂存候选](SELECTED_CANDIDATES.json)",
              "- [全部尝试、首稿、审核与失败路径](ALL_ATTEMPTS_INDEX.json)",
              "- [逐 root 终态与主要统计](FINAL_STATUS.json)",
              "- [组合前未填槽位和最终题审台账](QUESTION_STAGE_LEDGER.json)",
              "- [程序偏差说明](PROCEDURE_NOTES_ZH.md)及[原始记录](PROCEDURE_DEVIATIONS.json)",
              "- [全量报告](REPORT_ZH.md)及[失败处置报告](EXECUTION_REPORT_ZH.md)",
              "- [本入口的机器可读核对数据和输入哈希](DELIVERY_GUIDE_FACTS.json)", "",
              "未新训练，未合并正式训练池。作者与审者是继承同一模型的不同会话代理，共享文件系统；不是外部/异模型语义认证或严格盲审。", ""]
    return "\n".join(lines)


def main():
    output = B / "delivery/all"
    assert output.is_dir(), "WAIT_FOR_FINAL_DELIVERY_EXPORT"
    assert not (B / "SEAL_RECEIPT.json").exists() and not (B / "COMPLETE_MANIFEST.json").exists(), "BATCH_ALREADY_SEALED"
    targets = [output / "READ_ME_ZH.md", output / "DELIVERY_GUIDE_FACTS.json"]
    assert all(not p.exists() for p in targets), "GUIDE_OUTPUT_ALREADY_EXISTS"
    facts = collect(output)
    text = render(facts)
    facts["generated_report_binding"] = {"path": str(targets[0].resolve()), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    for item in facts["input_bindings"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"], "INPUT_CHANGED_BEFORE_WRITE"
    with targets[0].open("x") as stream:
        stream.write(text)
    with targets[1].open("x") as stream:
        json.dump(facts, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": facts["status"], "population": facts["population"],
                      "new_content_success_roots": facts["new_content_success_roots"],
                      "readable_complete_dialogues_including_source_only": facts["readable_complete_dialogues_including_source_only"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
