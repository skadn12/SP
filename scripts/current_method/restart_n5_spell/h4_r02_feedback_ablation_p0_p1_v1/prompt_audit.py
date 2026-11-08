"""Offline stage-specific parity audit of the unchanged Miner and Writer builders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file
from ..h4_batch30_memory_event_k3_probe_v1 import qa_generation_only as original
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation
from ..h4_qa_multiroot_k3_t08_probe_v1.root_inventory import clean_named_view, load_expanded
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_qa_mining_probe_v1.reasoning_contract import build_reasoning_contract, contract_has_three_disjoint_groups
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt as guided, build_repair_prompt as guided_repair
from ..h4_qa_mining_hybrid_generation_v1.free_exploration_generator import build_prompt as free, build_repair_prompt as free_repair
from ..simple_append_v1.live_runtime import load_json_object_strict
from .qa_run import ARMS, ROOTS, MODES, read, verify_schedule


def digest(value):
    return sha256_bytes(canonical_json(value).encode())


def appended(base, suffix):
    return base + "\n" + suffix if suffix else base


def raw_object(root: Path, request_id: str):
    path = root / "raw/responses" / f"{request_id}.json.txt"
    if path.is_file():
        try:
            return load_json_object_strict(path.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def fixed_contract(expanded):
    refs = [row["turn_ref"] for row in expanded["turns"][:3]]
    if len(refs) != 3:
        raise ValueError("fixed_contract_needs_three_real_refs")
    return {"intent_id": "OFFLINE_FIXED_CONTRACT_NOT_A_LIVE_INTENT", "intent_type": "causal",
            "discriminative_variable": "offline parity fixture",
            "required_evidence_groups": dict(zip(("before_state", "trigger", "after_state"), [[r] for r in refs]))}


def audit(run: Path):
    verify_schedule(run)
    checks, errors, pending, rows = {}, [], [], []

    def check(label, condition):
        checks[label] = bool(condition)
        if not condition:
            errors.append(label)

    check("same_original_generate_root_function", original.generate_root_qa is generation.generate_root_qa)
    check("same_original_miner_builder", generation.build_intent_prompt is build_intent_prompt)
    check("same_original_guided_builder", generation.build_guided_prompt is guided)
    check("same_original_free_builder", generation.build_free_prompt is free)
    check("same_original_guided_repair_builder", generation.build_guided_repair is guided_repair)
    check("same_original_free_repair_builder", generation.build_free_repair is free_repair)
    check("same_original_minimum_acceptance", generation.MINIMUM_COUNTS == {"single_choice": 2, "multi_select": 1, "ordering": 1})
    manifests = {arm: read(run / arm / "qa/manifest.json") for arm in ARMS}
    contexts = {arm: original._verify_feedback_context(manifests[arm], None) for arm in ARMS}
    completed = 0
    fixed_rows, pairs = [], []
    for root_id in ROOTS:
        clean_by_arm, miner_bases, fixture_bases = {}, {}, {}
        for arm in ARMS:
            key = f"{arm}:{root_id}"
            manifest = manifests[arm]
            entry = next(row for row in manifest["roots"] if row["root_id"] == root_id)
            root = original._root_dir(run / arm / "qa", entry["shard"], root_id)
            source = Path(entry["source_path"])
            check(key + ":source_sha256", sha256_file(source) == entry["source_sha256"])
            expected_clean, mapping = clean_named_view(load_expanded(source))
            cleaned = read(root / "expanded_dialogue_clean.json")
            check(key + ":cleaned_equals_original_cleaner", cleaned == expected_clean)
            check(key + ":mapping_equals_original_cleaner", read(root / "source_metadata.json")["speaker_name_mapping"] == mapping)
            clean_by_arm[arm] = cleaned
            context = original._load_feedback_context(contexts[arm]["roots"][root_id], root_id)
            check(key + ":feedback_empty_iff_off", bool(context["intent_suffix"]) == (arm == "feedback_on"))
            if arm == "feedback_off":
                check(key + ":off_exact_empty_context", context == {"intent_suffix": "", "slot_suffixes": ["", "", ""]})
            miner_base = build_intent_prompt(cleaned)
            miner_bases[arm] = miner_base
            fixture = fixed_contract(cleaned)
            fixture_bases[arm] = {}
            for mode, builder in (("contract_guided", guided), ("free_exploration", free)):
                base = builder(cleaned, fixture)
                expected_counts = {"single_choice": 2, "multi_select": 1, "ordering": 1}
                if mode == "free_exploration":
                    expected_counts = {k: 2 * v for k, v in expected_counts.items()}
                check(key + ":fixed_contract_quotas:" + mode, base["expected_counts"] == expected_counts)
                fixture_bases[arm][mode] = base
                for index, suffix in enumerate(context["slot_suffixes"]):
                    built = generation._append_feedback_context(base, suffix)
                    check(key + f":fixed_contract_append:{mode}:{index}", built["prompt"] == appended(base["prompt"], suffix))
                    fixed_rows.append({"arm": arm, "root_id": root_id, "mode": mode, "slot": index,
                                       "fixture_sha256": digest(fixture), "base_sha256": base["prompt_sha256"],
                                       "suffix_sha256": sha256_bytes(suffix.encode()), "final_sha256": built["prompt_sha256"],
                                       "live_intent": False})
            report_path = root / "qa_generation_report.json"
            has_report = report_path.is_file()
            completed += int(has_report)
            intents_path = root / "intents.json"
            if not intents_path.is_file():
                pending.append(key + ":live_intent_not_generated")
                continue
            saved_intents = read(intents_path)
            contracts = read(root / "reasoning_contracts.json")
            reconstructed = []
            for intent in saved_intents["validation"].get("intents", []):
                if intent.get("valid"):
                    contract = build_reasoning_contract(intent)
                    contract["contract_valid"] = contract_has_three_disjoint_groups(contract)
                    if contract["contract_valid"]:
                        reconstructed.append(contract)
            check(key + ":live_contracts_derive_from_own_intents", contracts == reconstructed)
            check(key + ":max_three_live_contracts", len(contracts) <= 3)
            records_path = root / "candidate_generation.json"
            records = read(records_path) if records_path.is_file() else []
            expected = {f"{root_id}-intent-miner": {"base": miner_base, "suffix": context["intent_suffix"],
                        "mode": "intent_miner", "intent_id": None, "attempt": "primary", "saved": saved_intents["prompt"]}}
            records_by_id = {row["request_id"]: row for row in records}
            check(key + ":unique_generation_records", len(records_by_id) == len(records))
            for index, contract in enumerate(contracts):
                intent_id = contract["intent_id"]
                for mode, builder in (("contract_guided", guided), ("free_exploration", free)):
                    request_id = f"{root_id}-{mode}-generator-{intent_id}"
                    record = records_by_id.get(request_id)
                    expected[request_id] = {"base": builder(cleaned, contract), "suffix": context["slot_suffixes"][index],
                                            "mode": mode, "intent_id": intent_id, "attempt": "primary",
                                            "saved": record["prompt"] if record else None}
                    repair_id = f"{root_id}-{mode}-repair-{intent_id}"
                    repair = records_by_id.get(repair_id)
                    if record:
                        primary_raw = raw_object(root, request_id)
                        normalized = generation.normalize_batch(primary_raw, intent_id=intent_id, source=mode, contract=contract, expanded=cleaned)
                        _, acceptance = generation._accepted_candidates(normalized, mode)
                        check(key + ":primary_minimum_acceptance:" + request_id,
                              record["validation"]["minimum_acceptance"] == acceptance)
                        check(key + ":one_repair_iff_original_below_minimum:" + request_id,
                              bool(repair) == (not acceptance["accepted"]))
                    if repair:
                        check(key + ":repair_refers_to_own_primary:" + repair_id, repair.get("repair_of") == request_id)
                        raw = raw_object(root, request_id)
                        spec = guided_repair(cleaned, contract, raw, mode) if mode == "contract_guided" else free_repair(cleaned, contract, raw)
                        expected[repair_id] = {"base": spec, "suffix": context["slot_suffixes"][index], "mode": mode,
                                               "intent_id": intent_id, "attempt": "repair", "saved": repair["prompt"]}
            observed_paths = {path.stem: path for path in (root / "raw/prompts").glob("*.txt")}
            check(key + ":no_extra_generation_requests", set(observed_paths) <= set(expected))
            for request_id, item in expected.items():
                label = key + ":" + request_id
                path = observed_paths.get(request_id)
                if path is None:
                    pending.append(label + ":prompt_missing")
                    continue
                expected_text = appended(item["base"]["prompt"], item["suffix"])
                actual_text = path.read_text(encoding="utf-8")
                actual_sha = sha256_bytes(actual_text.encode())
                check(label + ":own_builder_and_suffix_bytes", actual_text == expected_text)
                check(label + ":saved_spec", item["saved"] is not None and item["saved"]["prompt"] == actual_text and item["saved"]["prompt_sha256"] == actual_sha)
                state_path = root / "ledger/requests" / f"{request_id}.json"
                if state_path.is_file():
                    state = read(state_path)
                    meta = state["request_metadata"]
                    check(label + ":ledger_prompt_hash", state["prompt_sha256"] == actual_sha)
                    check(label + ":ledger_manifest", state["manifest_sha256"] == manifest["manifest_sha256"])
                    check(label + ":model", state["backend_identity"]["model"] == "Qwen/Qwen3-14B")
                    check(label + ":temperature_and_max_tokens", meta["temperature"] == 0.1 and meta["max_tokens"] == (4096 if item["mode"] == "intent_miner" else 8192))
                    check(label + ":isolated_call_identity", meta["arm"] == arm and meta["root_id"] == root_id and meta["intent_id"] == item["intent_id"] and meta["mode"] == item["mode"] and meta["attempt"] == item["attempt"])
                    if state["status"] == "completed":
                        raw_path = Path(state["raw_response_path"])
                        check(label + ":cached_response_hash", sha256_file(raw_path) == state["raw_response_sha256"])
                else:
                    pending.append(label + ":ledger_missing")
                if item["mode"] != "intent_miner":
                    check(label + ":no_empirical_packet_in_writer_suffix", all(marker not in item["suffix"] for marker in ('"solver_observations"', '"historical_qa"', '【有来源的反馈数据】')))
                rows.append({"arm": arm, "root_id": root_id, "request_id": request_id, "intent_id": item["intent_id"],
                             "mode": item["mode"], "attempt": item["attempt"], "live_own_contract": item["mode"] != "intent_miner",
                             "base_sha256": item["base"]["prompt_sha256"], "suffix_sha256": sha256_bytes(item["suffix"].encode()),
                             "actual_sha256": actual_sha, "rebuilt_sha256": sha256_bytes(expected_text.encode())})
            if has_report:
                check(key + ":completed_root_artifacts_present", not any(p.startswith(key + ":") for p in pending))
        check(root_id + ":same_cleaned_transcript", clean_by_arm[ARMS[0]] == clean_by_arm[ARMS[1]])
        check(root_id + ":same_miner_base", miner_bases[ARMS[0]] == miner_bases[ARMS[1]])
        check(root_id + ":same_fixed_contract_writer_bases", fixture_bases[ARMS[0]] == fixture_bases[ARMS[1]])
        pairs.append({"root_id": root_id, "cleaned_transcript_sha256": digest(clean_by_arm[ARMS[0]]),
                      "miner_base_sha256": miner_bases[ARMS[0]]["prompt_sha256"],
                      "cross_arm_actual_writer_prompt_equality_required": False,
                      "reason": "Live Writers use each arm's own newly generated intents; fixed-contract audit isolates suffix parity."})
    status = "FAIL" if errors else ("PASS_COMPLETE" if completed == 12 and not pending else ("PASS_OFFLINE" if completed == 0 else "PASS_OBSERVED_INCOMPLETE"))
    return {"version": "h4_r02_staged_prompt_parity_v1", "status": status, "completed_root_arms": completed,
            "expected_root_arms": 12, "checks": checks, "errors": errors, "pending": pending,
            "pairs": pairs, "fixed_contract_rows": fixed_rows, "live_prompt_rows": rows,
            "api_calls": 0, "memory_writes": 0, "live_contracts_replaced_by_fixture": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    result = audit(args.run.resolve())
    path = args.run / "audit" / ("prompt_parity_complete.json" if args.require_complete else "prompt_parity_preflight.json")
    atomic_json(path, result)
    print(json.dumps({k: result[k] for k in ("status", "completed_root_arms", "expected_root_arms", "errors")}, indent=2))
    return int(bool(result["errors"]) or (args.require_complete and result["status"] != "PASS_COMPLETE"))


if __name__ == "__main__":
    raise SystemExit(main())
