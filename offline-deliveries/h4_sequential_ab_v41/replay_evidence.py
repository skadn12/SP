"""Replay real historical B responses under candidate retention, with no API.

This is a diagnostic comparison of two reviews of one existing dialogue. It
never reserves a source, appends a production history or admits training data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import candidate_contract
import candidate_metrics
import duplicate_check


HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
ROOT = "cc-train-episode-104814"
ROUTE = "dataset_new_event"
V37 = DELIVERIES / "h4_sequential_ab_v37/experiments/fixed_soda_a3_01"
V40 = DELIVERIES / "h4_sequential_ab_v40/probes/two_stage_01"
TOKENIZER = HERE.parents[1] / "restart-n5-runs/h4_writer_model_comparison_v1_live_1/policy_snapshot/tokenizer/tokenizer.json"
BASELINE_SHA256 = "fe7a1ab35ed3efb4fad16c0d81501e6b80d8cd3615ef344de465bc4dfa63e997"


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_inputs():
    realization_dir = V37 / f"requests/006-{ROOT}-B-S01-realizer-1"
    original_review_dir = V37 / f"requests/007-{ROOT}-B-S01-review-1-1"
    later_review_dir = V40 / "requests/002-full-review-original_decision_scene"
    inherited_path = V37 / "INHERITED_A.json"
    inherited = read(inherited_path)
    history_path = Path(inherited["local_path"])
    source_history_path = Path(inherited["source_path"])
    history = read(history_path)["turns"]
    if candidate_metrics.digest(history) != BASELINE_SHA256 or history != read(source_history_path)["turns"]:
        raise ValueError("replay_exact_A3_parent_changed")
    original_manifest = read(V37 / "EXPERIMENT_OUTPUT_MANIFEST.json")
    # Existing output bindings protect the replay from trusting edited legacy
    # artifacts simply because the replay would bind their new bytes.
    candidate_metrics.verify_files(original_manifest["files"])
    realization_response = read(realization_dir / "RESPONSE.json")
    realization = json.loads(realization_response["text"])
    saved_scene_path = V37 / f"roots/{ROOT}/B/S01.json"
    if realization != read(saved_scene_path):
        raise ValueError("replay_realization_differs_from_original_response")
    plan_path = V37 / f"roots/{ROOT}/B/S01.PLAN.json"
    scene = read(plan_path)["scenes"][0]
    # Replay mechanical checks from actual dialogue, not the old accepted flag.
    mechanical_failures = []
    if not all(realization.get(key) == value for key, value in
               (("root_id", ROOT), ("route", ROUTE), ("scene_id", "S01"), ("status", "scene"))):
        mechanical_failures.append("realizer_identity_or_status_mismatch")
    turns = realization.get("turns")
    if not isinstance(turns, list) or not 12 <= len(turns) <= 16:
        mechanical_failures.append("realizer_turn_count_outside_12_16")
    else:
        for turn in turns:
            if not isinstance(turn, dict) or set(turn) != {"speaker", "text"}:
                mechanical_failures.append("turn_fields_invalid")
                continue
            if turn["speaker"] not in scene["actors"]:
                mechanical_failures.append("turn_speaker_invalid")
            if not isinstance(turn["text"], str) or not turn["text"].strip() or len(turn["text"]) > 2000:
                mechanical_failures.append("turn_text_invalid")
    if mechanical_failures:
        raise ValueError("historical_replay_dialogue_damaged:" + ",".join(mechanical_failures))
    duplicate = duplicate_check.inspect_duplicates(turns, {"branch_history": history,
                                                           "other_branch_accepted_additions": []})
    if duplicate != read(realization_dir / "DUPLICATE_CHECK.json"):
        raise ValueError("replay_duplicate_result_changed")
    card_path, source_path = V37 / "FROZEN_CARD.json", V37 / "FROZEN_SOURCE.json"
    card, source = read(card_path), read(source_path)
    receipt_path = V37 / "SOURCE_REUSE_RECEIPT.json"
    receipt = read(receipt_path)
    if (card["source_id"] != source["source_id"] or
            card["lineage"]["source_item_sha256"] != source["source_item_sha256"] or
            receipt["training_pool_eligible"] is not False or receipt["new_source_consumption"] != 0):
        raise ValueError("diagnostic_source_identity_or_scope_changed")
    reviews = [
        {"name": "original_v37_review", "directory": original_review_dir,
         "validation": read(original_review_dir / "REVIEW.json"),
         "validation_path": original_review_dir / "REVIEW.json"},
        {"name": "v40_missing_history_goal_checks", "directory": later_review_dir,
         "validation": read(later_review_dir / "EVALUATION.json")["review"],
         "validation_path": later_review_dir / "EVALUATION.json"},
    ]
    paths = [inherited_path, history_path, source_history_path, saved_scene_path, plan_path,
             card_path, source_path, receipt_path, V37 / "EXPERIMENT_OUTPUT_MANIFEST.json",
             realization_dir / "RESPONSE.json", realization_dir / "HTTP_RESPONSE_01.json",
             realization_dir / "DUPLICATE_CHECK.json", TOKENIZER,
             HERE / "replay_evidence.py", HERE / "candidate_contract.py",
             HERE / "candidate_metrics.py", HERE / "duplicate_check.py"]
    for review in reviews:
        response = read(review["directory"] / "RESPONSE.json")
        http = read(review["directory"] / "HTTP_RESPONSE_01.json")
        if response["text"] != http["text"]:
            raise ValueError("replay_review_differs_from_http_body")
        review["raw"] = response["text"]
        review["parsed"] = json.loads(response["text"])
        paths.extend([review["directory"] / name for name in ("RESPONSE.json", "HTTP_RESPONSE_01.json", "REQUEST.json")])
        paths.append(review["validation_path"])
    return history, realization, duplicate, source, receipt, reviews, paths


def build_replay():
    from tokenizers import Tokenizer
    history, realization, duplicate, source, receipt, reviews, paths = load_inputs()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    scenarios = []
    artifacts = {}
    for review in reviews:
        decision = candidate_contract.decide_candidate(
            duplicate_result=duplicate, review_raw=review["raw"],
            review_validation=review["validation"],
            expected_identity={"root_id": ROOT, "route": ROUTE, "scene_id": "S01"})
        record = {"candidate_id": review["name"], "route": ROUTE,
                  "disposition": decision["disposition"], "realization": realization,
                  "provenance": "diagnostic_replay", "raw_review": review["raw"],
                  "historical_validator_status": review["validation"].get("status"),
                  "decision": decision, "training_pool_eligible": False}
        report = candidate_metrics.summarize_records([record], count_text_tokens=count,
                                                    baseline_turns=history, scope="offline_diagnostic")
        check = candidate_metrics.verify_summary(report, [record], count_text_tokens=count,
                                                 baseline_turns=history, scope="offline_diagnostic")
        for name, obj in (("DECISION", decision), ("RECORD", record), ("METRICS", report), ("ACCOUNTING_CHECK", check)):
            artifacts[f"scenarios/{review['name']}/{name}.json"] = obj
        scenarios.append({"name": review["name"], "disposition": decision["disposition"],
                          "historical_validator_status": review["validation"].get("status"),
                          "historical_schema_valid": review["validation"].get("schema_valid"),
                          "raw_model_aggregate_verdict": decision["raw_review_verdict"],
                          "audit_issues": decision["audit_issues"], "content_risks": decision["content_risks"],
                          "diagnostic_body_tokens": report["diagnostic_replay"]["generated"]["body_tokens"],
                          "fresh_generated_body_tokens": report["generated"]["body_tokens"],
                          "fresh_retained_body_tokens": report["retained"]["body_tokens"],
                          "training_admitted_body_tokens": report["admitted"]["body_tokens"]})
    measure, _ = candidate_metrics.measure_turns(realization["turns"], count)
    baseline, _ = candidate_metrics.measure_turns(history, count)
    artifacts["SOURCE_BINDINGS.json"] = {"files": candidate_metrics.bind_files(paths)}
    artifacts["DUPLICATE_CHECK.json"] = duplicate
    artifacts["REPORT.json"] = {
        "version": "v41_offline_candidate_retention_replay", "scope": "offline_diagnostic",
        "source_identity": {key: source[key] for key in ("dataset", "split", "item_id", "source_id", "source_item_sha256")},
        "source_reuse_receipt": str(V37 / "SOURCE_REUSE_RECEIPT.json"),
        "source_policy": "Already-consumed diagnostic source; never a new production or training example.",
        "scenarios": scenarios, "unique_diagnostic_dialogues": 1, "unique_diagnostic_body": measure,
        "comparison_policy": "Two reviewer scenarios reuse one dialogue; scenario tokens must not be summed as unique yield.",
        "formal_accepted_baseline": baseline, "formal_accepted_cumulative": baseline,
        "fresh_A_body_tokens": 0, "fresh_B_body_tokens": 0,
        "http_calls": 0, "new_source_reservations": 0, "training_admitted_scenes": 0,
        "training_admitted_body_tokens": 0, "training_pool_eligible": False,
        "semantic_certified": False,
        "known_reviewer_metadata_risks": [
            "Both historical reports contain consumption/rollout summary claims whose accuracy this replay does not certify.",
            "Historical PASS/REJECT are validator results; original model JSON has no aggregate status/verdict field.",
        ],
    }
    return artifacts


def replay(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("replay_output_already_exists")
    artifacts = build_replay()
    for relative, obj in artifacts.items():
        write(output / relative, obj)
    report = artifacts["REPORT.json"]
    rows = [f"| {item['name']} | {item['historical_validator_status']} | {item['disposition']} | {item['diagnostic_body_tokens']} | 0 |"
            for item in report["scenarios"]]
    prose = "\n".join([
        "# v41 候选保留离线回放", "",
        "复用 v37 的真实 B 对白和 v37/v40 两份真实审核响应，没有调用模型或占用新来源。",
        "同一对白为 16 轮、504 个正文 token；两次审核情景不能相加为两条新样本。", "",
        "| 审核情景 | 历史校验结果 | v41 候选处置 | 复用正文 token | 新增 token |",
        "| --- | --- | --- | ---: | ---: |", *rows, "",
        "v40 的历史校验 REJECT 原因是缺少 history_goal_checks。新规则把缺字段保留为审核问题，不因此丢弃对白。",
        "原始模型 JSON 没有总 status/verdict 字段，历史 PASS/REJECT 来自校验器，所有原始字段和历史结果都保留。", "",
        "历史摘要中关于吃完三明治、完成 rollout 的说法仍需复核；本报告不把候选保留解释为语义认证。",
        "正式基线仍为 132 轮、3672 个正文 token；新增 A/B 均为 0，训练准入为 0。",
        "SOURCE_BINDINGS.json 绑定原始响应、对白、来源记录、精确 A3 历史、tokenizer 和回放代码。", "",
    ])
    (output / "REPLAY_REVIEW.md").write_text(prose)
    write(output / "OUTPUT_BINDINGS.json", {"files": candidate_metrics.bind_files(output.rglob("*json")) +
                                            candidate_metrics.bind_files([output / "REPLAY_REVIEW.md"])})
    return verify(output)


def verify(output):
    output = Path(output).resolve()
    candidate_metrics.verify_files(read(output / "SOURCE_BINDINGS.json")["files"])
    candidate_metrics.verify_files(read(output / "OUTPUT_BINDINGS.json")["files"])
    expected = build_replay()
    for relative, obj in expected.items():
        if read(output / relative) != obj:
            raise ValueError("offline_replay_artifact_changed:" + relative)
    if (output / "SOURCE_RESERVATIONS.json").exists() or (output / "requests").exists():
        raise ValueError("offline_replay_has_live_artifacts")
    return {"status": "ARTIFACTS_CONSISTENT", "scenarios": len(expected["REPORT.json"]["scenarios"]),
            "unique_diagnostic_body_tokens": expected["REPORT.json"]["unique_diagnostic_body"]["body_tokens"],
            "fresh_A_body_tokens": 0, "fresh_B_body_tokens": 0, "http_calls": 0,
            "new_source_reservations": 0, "training_admitted_body_tokens": 0,
            "training_pool_eligible": False, "semantic_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("replay", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.output) if args.action == "replay" else verify(args.output), indent=2))
