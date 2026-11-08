"""Independent batch2 artifact audit; candidate content is never rewritten.

Prepared stage does not inspect solver results. Completed stage verifies raw hashes
and independently recomputes score metrics and reports from the sealed records.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def check(bound):
    assert bind(bound["path"]) == bound, "BINDING_DRIFT:" + bound["path"]


def main(completed):
    from independent_collection_checks import audit_collection, audit_completed_summary
    collection_audit = audit_collection(HERE)
    spec = importlib.util.spec_from_file_location("independent_b_actual_execution", HERE / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    out = HERE / "execution"
    prep = read(out / "PREPARATION.json")
    for b in prep["bindings"]:
        check(b)
    code_review = read(HERE / "audit/EXECUTION_CODE_REVIEW.json")
    assert code_review["status"].startswith("PASS")
    for b in code_review["bindings"]:
        check(b)
    rows = read(out / "PUBLIC_REQUESTS.json")["requests"]
    tokens = read(out / "PROMPT_TOKEN_IDS.json")
    check(prep["private_scoring_binding"])
    private = read(out / "PRIVATE_SCORING.json")
    assert (out / "PRIVATE_SCORING.json").stat().st_mode & 0o777 == 0o600
    for b in private["bindings"]:
        check(b)
    assert private["public_requests_binding"] == bind(out / "PUBLIC_REQUESTS.json")
    assert private["prompt_token_ids_binding"] == bind(out / "PROMPT_TOKEN_IDS.json")
    candidates = {c["candidate_id"]: c for c in read(HERE / "CANDIDATES_PRIVATE.json")}
    reviews = {r["candidate_id"]: r for r in read(HERE / "CONTENT_REVIEW_AI.json")["reviews"]}
    admitted = {cid: c for cid, c in candidates.items() if reviews[cid]["decision"] == "PASS"}
    ids = [r["request_id"] for r in rows]
    assert ids == prep["fixed_order"] == [r["request_id"] for r in private["requests"]]
    assert len(ids) == len(set(ids)) == prep["requests"] == 4 * len(admitted)
    assert set(ids) == set(tokens)
    assert private["planned_questions"] == {"new": 70}
    assert prep["automatic_retry"] is False and prep["source_truncation"] is False
    assert prep["native_hidden_thinking"] is False and prep["generation_reads_private_scoring"] is False
    scoring = {r["request_id"]: r["presentation"] for r in private["requests"]}
    from transformers import AutoTokenizer
    endpoint = read(run.diagnostic.ENDPOINT)
    tokenizer = AutoTokenizer.from_pretrained(endpoint["tokenizer_directory"], local_files_only=True)
    selection = read(HERE / "SELECTION_FREEZE.json")
    sources = {}
    for root in selection["roots"]:
        b = {k: root[k] for k in ("path", "sha256", "bytes")}
        check(b)
        source = read(root["path"])
        assert source["root_id"] == root["root_id"]
        sources[root["root_id"]] = source
    variant_maps = {}
    source_turn_checks = full = qonly = 0
    public_keys = {"schema_version", "candidate_id", "root_id", "question_type", "task_operation",
        "question", "source_binding", "request_id", "cohort", "condition", "display", "labels",
        "sampling_seed", "prompt_path", "schema_path", "prompt_tokens", "variant_id"}
    for row in rows:
        assert set(row) == public_keys
        cid, rid = row["candidate_id"], row["request_id"]
        candidate, mapping = admitted[cid], scoring[rid]
        field = "options" if row["question_type"] == "multi_select" else "items"
        assert row["root_id"] == candidate["root_id"] and row["source_binding"] == candidate["source_binding"]
        assert row["question"] == candidate["question"] and row["task_operation"] == candidate["task_operation"]
        assert row["labels"] == list(row["display"]) == [chr(65 + i) for i in range(len(row["display"]))]
        by_text = {text: key for key, text in candidate[field].items()}
        assert len(by_text) == len(candidate[field])
        direct = {label: by_text[text] for label, text in row["display"].items()}
        assert set(direct.values()) == set(candidate[field])
        assert direct == mapping["display_to_canonical"]
        inverse = {canonical: label for label, canonical in direct.items()}
        assert inverse == mapping["canonical_to_display"]
        assert mapping["canonical_gold"] == candidate["gold"]
        display_gold = [inverse[k] for k in candidate["gold"]]
        if row["question_type"] == "multi_select":
            display_gold.sort()
        assert display_gold == mapping["display_gold"]
        perm = tuple(direct.values())
        assert perm != tuple(candidate[field])
        key = (cid, row["variant_id"])
        if key in variant_maps:
            assert variant_maps[key] == perm
        variant_maps[key] = perm
        prompt = Path(row["prompt_path"]).read_text()
        source = sources[row["root_id"]]
        expected = run.diagnostic.make_prompt(tokenizer, row, row["display"], source) if row["condition"] == "FULL" else run.base.question_only_prompt(tokenizer, row)
        assert prompt == expected, "PROMPT_NOT_PUBLIC_ONLY_REBUILD:" + rid
        actual_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        assert actual_tokens == tokens[rid] and len(actual_tokens) == row["prompt_tokens"]
        assert len(actual_tokens) + prep["max_output_tokens"] <= prep["max_model_len"]
        refs = [t["turn_ref"] for t in source["turns"]]
        schema = read(row["schema_path"])
        if row["condition"] == "FULL":
            serialized = "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source["turns"])
            assert "Dialogue:\n" + serialized + "\n\nQuestion (" in prompt
            assert schema == run.diagnostic.repair.schema(row["labels"], refs, row["question_type"])
            full += 1
            source_turn_checks += len(refs)
        else:
            assert row["condition"] == "QUESTION_ONLY" and "Dialogue:\n" not in prompt
            assert schema == run.base.question_only_schema(row["labels"], row["question_type"])
            qonly += 1
    gold_independent = 0
    for cid, candidate in admitted.items():
        assert len({variant_maps[(cid, v)] for v in ("perm_00", "perm_01")}) == 2
        alternate = dict(candidate)
        field = "options" if candidate["question_type"] == "multi_select" else "items"
        alternate["gold"] = list(candidate[field]) if field == "options" else candidate["gold"][1:] + candidate["gold"][:1]
        if alternate["gold"] == candidate["gold"]:
            alternate["gold"] = list(candidate[field])[:2]
        for variant, public, mapping in run.displays(alternate):
            assert tuple(mapping["display_to_canonical"].values()) == variant_maps[(cid, variant)]
        gold_independent += 1
    inventory = read(HERE / "SOURCE_INVENTORY.json")
    source_inventory = {r["root_id"]: r for r in inventory["sources"]}
    token_counts = []
    total_turns = 0
    for root_id, source in sources.items():
        record = source_inventory[root_id]
        check(record["source_binding"])
        serialized = "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source["turns"])
        count = len(tokenizer.encode(serialized, add_special_tokens=False))
        assert record["turns"] == len(source["turns"]) and record["serialized_dialogue_tokens"] == count
        total_turns += len(source["turns"])
        token_counts.append(count)
    assert total_turns == inventory["total_turns"]
    assert sum(token_counts) / len(token_counts) == inventory["mean_source_tokens"]
    assert min(token_counts) == inventory["min_source_tokens"] and max(token_counts) == inventory["max_source_tokens"]
    paths = [Path(__file__), HERE / "audit/independent_collection_checks.py", HERE / "run.py", HERE / "audit/EXECUTION_CODE_REVIEW.json",
        HERE / "CANDIDATES_PRIVATE.json", HERE / "CONTENT_REVIEW_AI.json", HERE / "SELECTION_FREEZE.json",
        HERE / "SLOT_LEDGER.json", HERE / "COLLECTION_AUDIT.json", HERE / "SOURCE_INVENTORY.json",
        out / "PREPARATION.json", out / "PUBLIC_REQUESTS.json", out / "PROMPT_TOKEN_IDS.json", out / "PRIVATE_SCORING.json"]
    result = {"schema_version": "production_independent_execution_artifact_review_v1",
        "status": "PASS_INDEPENDENT_COMPLETED_EXECUTION" if completed else "PASS_INDEPENDENT_PREPARED_EXECUTION",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "reviewer_id": "/root/production_gate_finish",
        "script_authored_by_reviewer": True, "root_execution_scripts_modified_by_reviewer": False,
        "question_semantics_certified_here": False, "raw_solver_text_opened": False,
        "collection_checks": collection_audit,
        "candidate_repairs_based_on_solver_outputs": False, "admitted_questions": len(admitted), "requests": len(rows),
        "full_prompt_rebuilds": full, "question_only_prompt_rebuilds": qonly,
        "full_turns_checked_across_requests": source_turn_checks,
        "every_prompt_token_id_recomputed": True, "every_schema_rebuilt": True,
        "all_mappings_recovered_independently_from_item_text": True,
        "gold_counterfactual_display_checks": gold_independent,
        "two_distinct_nonidentity_displays_per_question": True,
        "source_inventory_recomputed": {"roots": len(sources), "total_turns": total_turns,
            "mean_source_tokens": sum(token_counts) / len(token_counts), "min_source_tokens": min(token_counts), "max_source_tokens": max(token_counts)},
        "private_scoring_mode_0600": True, "planned_denominator": 70,
        "limitations": ["Public-only reconstruction uses the separately statically audited prompt builders.",
            "This audit recomputes scores from saved validity and displayed predictions; raw JSON parsing/validity is covered by completed integrity and analysis gates.",
            "This audit does not certify visible reason semantics or source question semantics."]}
    if completed:
        freeze = read(out / "RAW_RESPONSE_FREEZE.json")
        start = read(out / "STARTED.json")
        assert freeze["request_ids"] == ids and freeze["requests"] == len(ids)
        raw_paths = {str((out / "raw" / (rid + ".json")).resolve()) for rid in ids}
        assert {b["path"] for b in freeze["bindings"]} == raw_paths
        assert len(freeze["bindings"]) == len(raw_paths)
        assert {str(p.resolve()) for p in (out / "raw").glob("*.json")} == raw_paths
        for b in freeze["bindings"] + freeze["validation_bindings"]:
            check(b)
        assert freeze["started_binding"] == bind(out / "STARTED.json")
        assert start["preparation"] == bind(out / "PREPARATION.json")
        assert start["private_scoring_binding"] == prep["private_scoring_binding"]
        assert start["request_ids"] == ids and start["requests"] == len(ids)
        assert freeze["automatic_retry"] is False and start["gold_read"] is False
        score_doc = read(out / "SCORES.json")
        scores = score_doc["records"]
        assert score_doc["gold_read_after_all_raw_sealed"] is True
        assert score_doc["raw_freeze_binding"] == bind(out / "RAW_RESPONSE_FREEZE.json")
        assert score_doc["private_scoring_binding"] == prep["private_scoring_binding"]
        assert [s["request_id"] for s in scores] == ids
        for row, score in zip(rows, scores):
            m = scoring[row["request_id"]]
            gold = m["canonical_gold"]
            answer = score["display_prediction"]
            prediction = [m["display_to_canonical"][label] for label in answer] if score["valid"] else []
            if row["question_type"] == "multi_select":
                prediction.sort()
                exact = bool(score["valid"] and set(prediction) == set(gold))
                assert score["false_positive"] == len(set(prediction) - set(gold))
                assert score["false_negative"] == len(set(gold) - set(prediction))
                assert score["allselect_exact"] == (len(gold) == len(m["display_to_canonical"]))
            else:
                exact = bool(score["valid"] and prediction == gold)
                pos = {k: i for i, k in enumerate(prediction)}
                pairs = [(left, right) for i, left in enumerate(gold) for right in gold[i + 1:]]
                assert score["pairwise_total"] == len(pairs)
                assert score["pairwise_correct"] == sum(bool(score["valid"] and pos[left] < pos[right]) for left, right in pairs)
            assert score["prediction"] == prediction and score["exact"] == exact
            assert score["gold"] == gold and score["display_gold"] == m["display_gold"]
            assert score["variant_id"] == row["variant_id"]
            assert score["raw_binding"] == bind(out / "raw" / (row["request_id"] + ".json"))
        summary = read(out / "ANALYSIS_SUMMARY.json")
        assert summary == run.summarize(scores, private)
        audit_completed_summary(HERE, rows, scores, private, summary)
        assert summary["requests"] == len(rows) and summary["planned_questions"] == 70
        assert summary["initial_content_pass_rate"] == summary["initial_content_pass"] / 70
        collection = read(HERE / "COLLECTION_AUDIT.json")
        assert summary["initial_content_pass"] == collection["initial_joint_pass"]
        pool = read(HERE / "POOL_MANIFEST.json")
        root_report = read(HERE / "ROOT_VALIDATION.json")
        pool_rows = [json.loads(line) for line in (HERE / "TRAINING_POOL_CANDIDATES.jsonl").read_text().splitlines()]
        assert {c["candidate_id"] for c in pool_rows} == set(admitted)
        assert pool["exported_unique_questions"] == len(admitted) and pool["planned_slots"] == 70
        assert pool["solver_scores_used_for_admission"] is False
        assert root_report["root_inventory"] == pool["root_inventory"] and root_report["solver_summary"] == summary
        assert pool["question_types"] == dict(Counter(c["question_type"] for c in admitted.values()))
        assert sum(r["admitted"] for r in pool["root_inventory"]) == len(admitted)
        report = (HERE / "REPORT_ZH.md").read_text()
        assert f"70个冻结槽位；提交{len(candidates)}道初稿" in report
        assert f"最终内容审核通过{len(admitted)}题" in report
        assert f"{collection['initial_joint_pass']}/70={collection['initial_joint_pass_rate_planned']:.1%}" in report
        result.update(raw_freeze_hashes_verified_without_reading_text=len(freeze["bindings"]),
            score_records_independently_recomputed=len(scores), all_frozen_requests_retained=True,
            reported_counts={"submitted": len(candidates), "admitted": len(admitted), "requests": len(rows),
                "initial_joint_pass": collection["initial_joint_pass"], "initial_joint_pass_rate_planned": collection["initial_joint_pass_rate_planned"],
                "material_shortfalls": len(collection["material_shortfalls"]), "question_types": pool["question_types"],
                "valid_requests": summary["valid"], "correct_requests": summary["correct"]},
            report_and_pool_counts_verified=True)
        paths.extend(out / name for name in ("STARTED.json", "RAW_RESPONSE_FREEZE.json", "SCORES.json", "ANALYSIS_SUMMARY.json", "REASON_REVIEW_PACKET.json"))
        paths.extend(HERE / name for name in ("POOL_MANIFEST.json", "ROOT_VALIDATION.json", "TRAINING_POOL_CANDIDATES.jsonl", "REPORT_ZH.md"))
        paths.append(HERE / "audit/COMPLETED_INTEGRITY.json")
    result["bindings"] = [bind(p) for p in paths]
    target = HERE / "audit" / ("INDEPENDENT_EXECUTION_AUDIT.json" if completed else "INDEPENDENT_PREPARED_AUDIT.json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "requests": len(rows), "questions": len(admitted), "path": str(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepared", "completed"))
    main(parser.parse_args().stage == "completed")
