"""Compile a hash-bound local bank into isolated active-feedback artifacts."""

from __future__ import annotations

import argparse
import io
import socket
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from . import NO_FEEDBACK, READY, VERSION
from .projections import project_clusters
from .renderers import backend_record, feedback_envelope, fresh_view, history_envelope, render_role
from .sources import HASH_ONLY_PATHS, Sources, authority, canonical, file_sha, read, scope_filter, value_sha
from ..h4_round1_question_bank_feedback_compiler_v1.compiler import (
    _selection_rows, build_composition_pairs, build_fresh_negative_view,
    build_questioner_pools, make_cards, write_artifact,
)


@contextmanager
def offline_guard(output):
    audit = {"mode": "LOCAL_FILES_AND_PURE_RENDERERS_ONLY", "blocked_attempts": [],
             "api_calls": 0, "solver_calls": 0, "training_calls": 0, "memory_calls": 0,
             "credential_reads": 0, "production_entrypoints_started": 0}

    def blocked(*args, **kwargs):
        audit["blocked_attempts"].append("NETWORK_OR_PROCESS_SEND_BLOCKED")
        raise RuntimeError("offline_only_external_send_blocked")

    def hook(event, args):
        if event == "open":
            path = args[0]
            if isinstance(path, (str, bytes)):
                path = Path(path).resolve()
                if path.name in {"config.json", "config.toml", ".env", "auth.json", "credentials.json", "model_config.json"} and path not in HASH_ONLY_PATHS:
                    audit["blocked_attempts"].append("CREDENTIAL_OR_SERVICE_CONFIG_READ_BLOCKED")
                    raise RuntimeError("offline_service_config_read_forbidden")
                mode, flags = args[1], args[2]
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(flags & (1 | 2 | 64 | 512))
                if writing and not path.is_relative_to(output):
                    raise RuntimeError(f"offline_write_outside_output:{path}")
        if event in {"socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system", "os.fork", "os.posix_spawn"}:
            blocked()

    sys.addaudithook(hook)
    with patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked), patch.object(socket, "create_connection", blocked):
        yield audit


def stats(rows):
    return {"candidates": len(rows), "roots": len({r["root_id"] for r in rows}),
            "clusters": len({r["solver_prompt_fingerprint"] for r in rows}),
            "question_types": dict(sorted(Counter(r["question_type"] for r in rows).items())),
            "corpus": dict(sorted(Counter(r["corpus"] for r in rows).items())),
            "experiment_arms": dict(sorted(Counter(r["normalized_experiment_arm"] for r in rows).items()))}


def indexed(rows):
    return {r["solver_prompt_fingerprint"]: r for r in rows}


def dialogue_inventory(sources):
    inventory, dialogues = {}, {}
    for row in sources.get("dialogue_identity"):
        path = sources.verify(row["input_dialogue_path"], row["input_dialogue_file_sha256"])
        dialogue = read(path)
        if dialogue["root_id"] != row["root_id"] or dialogue["complete_dialogue_sha256"] != row["complete_dialogue_sha256"]:
            raise ValueError("dialogue_identity_mismatch")
        if row["source_prefix"] != "Parent+R01+R02-feedback" or not all(row["checks"].values()):
            raise ValueError("dialogue_scope_check_failed")
        root = row["root_id"]
        dialogues[root] = dialogue
        inventory[root] = {"root_id": root, "turns": {t["turn_ref"]: t for t in dialogue["turns"]},
                           "feedback_sha256": row["input_dialogue_file_sha256"],
                           "manifest_feedback_sha256": row["input_dialogue_file_sha256"],
                           "dialogue_content_sha256": row["complete_dialogue_sha256"]}
    return inventory, dialogues


def verify_later_sources(sources, scope, bank):
    corpus = {r["corpus_id"]: r for r in sources.get("corpus_scope")}
    source_candidates = {}
    for name in ("E1Q0", "E1Q1"):
        by_id = {}
        for spec in corpus[name]["source_files"]:
            sources.verify(spec["path"], spec["sha256"])
            path = Path(spec["path"])
            if path.name not in {"all_candidates_valid.jsonl", "candidates_valid_scoped.json"}:
                continue
            for row in read(path):
                by_id[row["candidate_id"]] = row
        source_candidates[name] = by_id
    final = {r["path"]: r for r in sources.get("FINAL_FILE_INVENTORY.json")}
    lineage = []
    for row in scope:
        if row["normalized_experiment_arm"] == "ROUND0_PARENT_PLUS_R01":
            continue
        path = row["source_path"]
        if path not in sources.identities:
            if path not in final:
                raise ValueError(f"later_lineage_not_in_final_inventory:{path}")
            sources.verify(path, final[path]["sha256"])
        cid = row["candidate_id"]
        if cid not in source_candidates[row["corpus_id"]]:
            raise ValueError(f"later_candidate_not_in_corpus_sources:{cid}")
        if row["normalized_experiment_arm"] != bank[cid]["normalized_experiment_arm"]:
            raise ValueError("later_arm_scope_mismatch")
        if row["corpus_id"] == "E1Q0":
            expected = {"control": "E0Q0_CONTROL_BLIND", "feedback": "E1Q0_FEEDBACK_BLIND"}.get(row["branch"])
            if expected != row["normalized_experiment_arm"]:
                raise ValueError("paired_branch_normalization_mismatch")
        lineage.append({"candidate_id": cid, "normalized_experiment_arm": row["normalized_experiment_arm"],
                        "source_path": path, "source_sha256": sources.identities[path],
                        "corpus_candidate_record_sha256": value_sha(source_candidates[row["corpus_id"]][cid])})
    return lineage


def compile_views(sources, output):
    emit = lambda name, value: write_artifact(output, name, value)
    protected = sources.get("FINAL_DO_NOT_OVERWRITE.json")
    # Hash opaque protected bytes without parsing any model configuration.
    protected_before = [{**r, "before_sha256": file_sha(r["path"])} for r in protected]
    emit("protected_before.json", protected_before)
    bank_rows, scope = sources.get("bank"), sources.get("scope")
    bank = {r["candidate_id"]: r for r in bank_rows}
    authoritative, originals, splits = authority(sources)
    excluded, kept = scope_filter(bank_rows, scope, authoritative, originals, splits)
    excluded_ids, allowed = {r["candidate_id"] for r in excluded}, {r["candidate_id"] for r in kept}
    later = verify_later_sources(sources, scope, bank)
    projection = project_clusters(clusters=sources.get("clusters"), members=sources.get("members"),
                                  origins=sources.get("origins"), eligibility=sources.get("eligibility"),
                                  target_weights=sources.get("target_weights"), composition_weights=sources.get("composition_weights"),
                                  excluded_ids=excluded_ids, candidates=bank)
    before_projection = project_clusters(clusters=sources.get("clusters"), members=sources.get("members"),
                                         origins=sources.get("origins"), eligibility=sources.get("eligibility"),
                                         target_weights=sources.get("target_weights"), composition_weights=sources.get("composition_weights"),
                                         excluded_ids=set(), candidates=bank)
    clusters, elig = indexed(projection["clusters"]), indexed(projection["eligibility"])
    target, comp = indexed(projection["target_weights"]), indexed(projection["composition_weights"])
    cooldown = indexed(sources.get("cooldown"))
    inventory, dialogues = dialogue_inventory(sources)
    pool, selections, reserves, routing_exclusions, by_root = _selection_rows(clusters, bank, elig, cooldown, inventory, target)
    targeted, bridge, parents, _, _ = build_questioner_pools(clusters, bank, elig, target, comp, inventory)
    pairs = build_composition_pairs(parents)
    cards, _, token_audit = make_cards([{**r, "cluster": clusters[r["solver_prompt_fingerprint"]]} for r in selections], bank, inventory)
    for row in cards:
        row["filter_version"] = VERSION
    for row in token_audit:
        row["token_method"] = "OFFLINE_REGEX_PROXY_NOT_MODEL_TOKENIZER"
    history = sources.get("saturation")
    fresh_backend = build_fresh_negative_view({}, history)
    historical_fresh = sources.get("live13_fresh")
    manifest_sha = file_sha(sources.manifest_path)
    records = {c["canonical_representative_candidate_id"]: backend_record(bank[c["canonical_representative_candidate_id"]], c, manifest_sha)
               for c in clusters.values() if c["active_observation_status"] == "RETAINED_INDEPENDENT_OBSERVATION"}
    global_target = [r for r in projection["target_weights"] if r["eligible"] and r["raw_weight"] > 0]
    global_comp = [r for r in projection["composition_weights"] if r["eligible"] and r["raw_weight"] > 0]
    active_ids = {r["canonical_representative_candidate_id"] for r in projection["eligibility"]
                  if any(r[k] for k in ("questioner_targeted_evolution_eligible", "composition_parent_eligible", "questioner_bridge_generation_eligible"))}
    active_records = [records[cid] for cid in sorted(active_ids) if cid in records]
    root_states, no_feedback = [], []
    roots = sorted({r["root_id"] for r in kept})
    for root in roots:
        global_sources = [r for r in global_target if r["root_id"] == root]
        state = {"root_id": root, "filter_version": VERSION,
                 "global_targeted_eligible": len(global_sources),
                 "global_targeted_status": "ELIGIBLE_FEEDBACK" if global_sources else NO_FEEDBACK,
                 "r03_branch_status": "IN_BOUND_REFERENCE_SCOPE" if root in inventory else "OUTSIDE_BOUND_R03_REFERENCE_SCOPE",
                 "expansion_source_count": len(by_root.get(root, [])),
                 "targeted_source_count": sum(r["root_id"] == root for r in targeted),
                 "composition_parent_count": sum(r["root_id"] == root for r in parents),
                 "composition_pair_count": sum(r["root_id"] == root for r in pairs)}
        if not global_sources:
            state["reason"] = "NO_RETAINED_SOURCE_PASSES_EXISTING_TARGETED_ELIGIBILITY"
            no_feedback.append({"root_id": root, "role": "global_targeted", "status": NO_FEEDBACK, "reason": state["reason"]})
        root_states.append(state)

    rendered, role_envelopes = [], []
    def render_one(role, root, chosen, key, reason=None):
        envelope = feedback_envelope(root, chosen, reason or "NO_RETAINED_SOURCE_PASSES_EXISTING_ROLE_RULES")
        if role == "fresh":
            envelope = history_envelope(root)
        role_envelopes.append({"selection_key": key, "role": role, **envelope})
        msg = render_role(role, envelope, dialogues[root], allowed, history=fresh_backend,
                          schema=sources.get(f"schema_{role}") if role != "expansion" else None,
                          strategy_schema=sources.get("strategy_schema") if role == "targeted" else None)
        msg["selection_key"] = key
        rendered.append(msg)
        return msg

    for root in sorted(inventory):
        primary = [records[r["canonical_representative_candidate_id"]] for r in selections if r["root_id"] == root]
        render_one("expansion", root, primary, f"expansion:{root}")
        render_one("fresh", root, [], f"fresh:{root}", "FRESH_USES_HISTORY_ONLY_NO_ACTIVE_SOURCE_BINDING")
        target_rows = [r for r in targeted if r["root_id"] == root]
        for row in target_rows:
            cid = row["canonical_representative_candidate_id"]
            render_one("targeted", root, [records[cid]], f"targeted:{cid}")
        if not target_rows:
            render_one("targeted", root, [], f"targeted:{root}")
        root_pairs = [r for r in pairs if r["root_id"] == root]
        for row in root_pairs:
            render_one("composition", root, [records[row["left_candidate_id"]], records[row["right_candidate_id"]]],
                       f"composition:{root}:{row['left_cluster_id']}:{row['right_cluster_id']}")
        if not root_pairs:
            render_one("composition", root, [], f"composition:{root}")

    # Real retained ROUND0 examples exercise full feedback independently of the 8-root R03 branch.
    round0_examples = []
    for qtype in ("single_choice", "multi_select"):
        row = next(r for r in global_target if bank[r["canonical_representative_candidate_id"]]["normalized_experiment_arm"] == "ROUND0_PARENT_PLUS_R01"
                   and bank[r["canonical_representative_candidate_id"]]["candidate"]["question_type"] == qtype)
        cid, root = row["canonical_representative_candidate_id"], row["root_id"]
        path = authoritative[cid]["source_snapshot"]["dialogue_path"]
        dialogue = read(path)
        if "turns" not in dialogue:
            dialogue = {"root_id": root, "turns": dialogue["dialogue"]["turns"]}
        dialogue["root_id"] = root
        message = render_role("targeted", feedback_envelope(root, [records[cid]]), dialogue, allowed,
                              schema=sources.get("schema_targeted"), strategy_schema=sources.get("strategy_schema"))
        round0_examples.append({"example_scope": "REAL_ROUND0_REFERENCE_SNAPSHOT_NOT_NEW_R03_QA",
                               "candidate_id": cid, "historical_qa_complete": bank[cid]["candidate"], "rendered": message})

    absence_examples = []
    for state in no_feedback:
        root = state["root_id"]
        identity = next(r for r in authoritative.values() if r["root_id"] == root)
        dialogue = read(identity["source_snapshot"]["dialogue_path"])
        if "turns" not in dialogue:
            dialogue = {"root_id": root, "turns": dialogue["dialogue"]["turns"]}
        dialogue["root_id"] = root
        envelope = feedback_envelope(root, [], state["reason"])
        for role in ("expansion", "targeted", "composition", "fresh"):
            message = render_role(role, envelope, dialogue, allowed, history=fresh_backend,
                                  schema=sources.get(f"schema_{role}") if role != "expansion" else None,
                                  strategy_schema=sources.get("strategy_schema") if role == "targeted" else None)
            absence_examples.append({"example_scope": "REAL_ROOT_WITH_NO_TARGETED_FEEDBACK_REFERENCE_SNAPSHOT",
                                     "composition_note": "Absence-contract demonstration; separate composition eligibility is not overridden.",
                                     "rendered": message})

    preserved_features = {name: [r for r in sources.get(name) if r["candidate_id"] in allowed]
                          for name in ("round0_features", "round0_routes", "round0_transitions")}
    current_cards = sources.get("live13_cards")
    current_122711 = next(r for r in current_cards if r["root_id"] == "cc-train-episode-122711")
    current_selection = {r["root_id"]: r["source_candidate_id"] for r in current_cards}
    new_selection = {r["root_id"]: r["source_candidate_id"] for r in cards}
    dev = sources.get("dev_global")
    dev_audit = {"source": sources.manifest["inputs"]["dev_global"], "historical_aggregate": dev,
                 "source_scope": "INDEPENDENT_DEV_AGGREGATE", "model_visible_in_this_version": False,
                 "affects_new_candidate_weights": False, "existing_global_policy_modified": False,
                 "reason": "The reused Bank compiler and shared renderer do not consume this historical aggregate; no new consumption channel added.",
                 "claim_limit": "ROUND0 Train ordering active contributions filtered; not all ordering or evaluation signals removed"}

    original_clusters = indexed(sources.get("clusters"))
    checks = {
        "authoritative_round0_count": len(authoritative) == 2294,
        "excluded_569": len(excluded) == 569,
        "retained_round0_1725": len(set(authoritative) & allowed) == 1725,
        "retained_bank_2493": len(kept) == 2493,
        "scope_disjoint_complete": not excluded_ids & allowed and excluded_ids | allowed == set(bank),
        "later_ordering_all_191_retained": sum(r["question_type"] == "ordering" for r in kept) == 191
            and all(r["candidate_id"] in allowed for r in scope if r["normalized_experiment_arm"] != "ROUND0_PARENT_PLUS_R01"),
        "cluster_members_no_excluded": all(not set(c["member_candidate_ids"]) & excluded_ids for c in clusters.values()),
        "observation_origin_no_excluded": all(c["active_observation_candidate_id"] not in excluded_ids for c in clusters.values()),
        "active_record_sources_allowed": {r["source_candidate_id"] for r in active_records} <= allowed,
        "pair_sources_allowed": all(r["left_candidate_id"] in allowed and r["right_candidate_id"] in allowed for r in pairs),
        "fresh_history_preserved": fresh_backend == historical_fresh,
        "fresh_coverage_count_nonzero": sum(r["candidate_count"] for r in history) == 3062,
        "fresh_visible_allowlist": all(set(c) == {"normalized_stem_signature", "prompt_cluster_count", "candidate_count"}
                                       for root in roots for c in fresh_view(root, fresh_backend)["historical_coverage"]),
        "current_122711_e1q1_outside_exclusion": current_122711["source_arm"] == "E1Q1_FEEDBACK_GUIDED" and current_122711["source_candidate_id"] not in excluded_ids,
        "full_bank_readonly": file_sha(sources.path("bank")) == sources.manifest["inputs"]["bank"]["sha256"],
        "retained_individual_cluster_metrics_unchanged": all(
            c["metrics"] == original_clusters[fp]["metrics"] for fp, c in clusters.items()),
        "all_root_count_aggregates_rebuilt": len(projection["root_aggregates"]) == 70 and all(r["status"] == "COMPLETE" for r in projection["root_aggregates"]),
    }
    # Generic ordering stems recur across snapshots. Identity and complete QA
    # provenance distinguish retained later sources from excluded old sources.
    rendered_text = canonical(rendered + round0_examples + absence_examples)
    leaked_ids = sorted(cid for cid in excluded_ids if cid in rendered_text)
    shared_stems = sorted(cid for cid in excluded_ids if originals[cid]["question"] in rendered_text)
    def check_visible_qa(value):
        if isinstance(value, dict):
            if "source_candidate_id" in value and "historical_qa" in value:
                cid = value["source_candidate_id"]
                if cid not in allowed:
                    return False
                qa = value["historical_qa"]
                if qa != records[cid]["historical_qa"]:
                    return False
            if "historical_qas" in value:
                for qa in value["historical_qas"]:
                    cid = qa["source_candidate_id"]
                    if cid not in allowed or qa != {"root_id": records[cid]["root_id"],
                            "source_cluster_id": records[cid]["source_cluster_id"], "source_candidate_id": cid,
                            **records[cid]["historical_qa"]}:
                        return False
            return all(check_visible_qa(child) for child in value.values())
        if isinstance(value, list):
            return all(check_visible_qa(child) for child in value)
        return True
    checks["rendered_excluded_id_absent"] = not leaked_ids
    checks["rendered_full_qa_identity_and_content_bound"] = all(check_visible_qa(row["feedback_view"]) for row in rendered)
    role_counts = {}
    for name, rows in {"expansion_pool": pool, "targeted_pool": targeted, "composition_parent_pool": parents,
                       "bridge_pool": bridge, "expansion_cards": cards, "composition_pairs": pairs}.items():
        ids = {row.get("source_candidate_id") or row.get("canonical_representative_candidate_id")
               for row in rows if row.get("source_candidate_id") or row.get("canonical_representative_candidate_id")}
        if name == "composition_pairs":
            ids = {row[key] for row in rows for key in ("left_candidate_id", "right_candidate_id")}
        role_counts[name] = {"rows": len(rows), "root_coverage": len({r["root_id"] for r in rows}),
                             "independent_source_candidates": len(ids),
                             "source_question_types": dict(Counter(bank[cid]["candidate"]["question_type"] for cid in ids)),
                             "source_experiment_arms": dict(Counter(bank[cid]["normalized_experiment_arm"] for cid in ids))}
    candidate_eligibility = []
    for row in excluded + kept:
        cid, fp = row["candidate_id"], row["solver_prompt_fingerprint"]
        representative = fp in clusters and clusters[fp]["canonical_representative_candidate_id"] == cid
        candidate_eligibility.append({"candidate_id": cid, "root_id": row["root_id"], "solver_prompt_fingerprint": fp,
                                      "filter_version": VERSION, "scope_allowed": cid in allowed,
                                      "independent_active_observation": representative and clusters[fp]["active_observation_candidate_id"] == cid,
                                      "targeted_eligible": representative and elig[fp]["questioner_targeted_evolution_eligible"],
                                      "composition_eligible": representative and elig[fp]["composition_parent_eligible"],
                                      "bridge_eligible": representative and elig[fp]["questioner_bridge_generation_eligible"],
                                      "alias_sampling_weight": None if representative else 0.0,
                                      "reason": "ROUND0_ORDERING_SCOPE_EXCLUDED" if cid in excluded_ids else "INDEPENDENT_REPRESENTATIVE_EXISTING_RULES" if representative else "ALIAS_NO_INDEPENDENT_ACTIVE_OBSERVATION"})
    grouped_counts = []
    for root in roots:
        for arm in sorted({r["normalized_experiment_arm"] for r in excluded + kept if r["root_id"] == root}):
            before = [r for r in excluded + kept if r["root_id"] == root and r["normalized_experiment_arm"] == arm]
            after = [r for r in before if r["candidate_id"] in allowed]
            grouped_counts.append({"root_id": root, "normalized_experiment_arm": arm, "before": stats(before), "after": stats(after)})
    counts = {"round0_before": stats([r for r in excluded + kept if r["candidate_id"] in authoritative]),
              "bank_before": stats(excluded + kept), "excluded": stats(excluded), "bank_retained_scope": stats(kept),
              "round0_retained_scope": stats([r for r in kept if r["candidate_id"] in authoritative]),
              "retained_clusters": len(clusters), "global_targeted_positive_eligible": len(global_target),
              "global_targeted_root_coverage": len({r["root_id"] for r in global_target}),
              "global_composition_positive_eligible": len(global_comp),
              "global_composition_root_coverage": len({r["root_id"] for r in global_comp}),
              "any_active_operation_eligible_representatives": len(active_records),
              "any_active_operation_root_coverage": len({r["root_id"] for r in active_records}),
              "r03_expansion_eligible_pool": len(pool), "r03_targeted_pool": len(targeted), "r03_bridge_pool": len(bridge),
              "r03_composition_parents": len(parents), "r03_composition_pairs": len(pairs),
              "r03_expansion_cards": len(cards), "r03_card_root_coverage": len(new_selection),
              "rendered_messages": len(rendered), "historical_candidates": len(bank),
              "historical_stem_signatures": len(history), "fresh_bounded_signatures": len(fresh_backend),
              "global_no_targeted_feedback_roots": len(no_feedback),
              "mixed_scope_clusters": sum(r["mixed_scope_cluster"] for r in projection["cluster_audit"])}
    weight_changes = {}
    for key in ("target_weights", "composition_weights"):
        old = indexed(sources.get(key))
        changed = [r["solver_prompt_fingerprint"] for r in projection[key]
                   if abs(r["raw_weight"] - old[r["solver_prompt_fingerprint"]]["raw_weight"]) > 1e-12]
        weight_changes[key] = {"retained_raw_weights_changed_by_active_family_projection": changed,
                               "changed_count": len(changed), "new_raw_weight_total": sum(r["raw_weight"] for r in projection[key]),
                               "normalization_rebuilt": ["global", "within_root"]}
    artifacts = {
        "policy.json": {"version": VERSION, "purpose": "ACTIVE_FEEDBACK_ONLY", "scope": {"corpus": "ROUND0_TRAIN", "source_round": "ROUND0_PARENT_PLUS_R01", "split": "train", "question_type": "ordering"},
                        "identity_basis": "authoritative Train candidate ID plus original pool/dialogue snapshot", "semantic_judgment": "NONE", "original_records_preserved": True,
                        "default_feedback_source_changed": False, "real_qa_integration": "NOT_CONFIRMED", "historical_gold_replaced": False,
                        "selection_rules": "existing Bank compiler functions after candidate/member scope mask", "active_family_counts": "remaining clusters; historical coverage stored separately"},
        "excluded_candidates.jsonl": excluded, "retained_candidate_scope.jsonl": kept,
        "authoritative_round0_candidate_scope.jsonl": list(authoritative.values()),
        "later_candidate_source_identity.jsonl": later,
        "active_question_bank_projection.jsonl": [bank[cid] for cid in sorted(allowed)],
        "candidate_active_eligibility.jsonl": candidate_eligibility,
        "active_shared_feedback_records.jsonl": active_records,
        "counts.json": counts, "root_feedback_states.jsonl": root_states,
        "role_selection_counts.json": role_counts,
        "before_after_root_arm_counts.jsonl": grouped_counts,
        "no_eligible_feedback_roots.jsonl": no_feedback,
        "active_expansion_seed_pool.jsonl": pool, "active_expansion_selection.jsonl": selections,
        "active_expansion_reserves.jsonl": reserves, "active_routing_exclusions.jsonl": routing_exclusions,
        "active_expansion_feedback_cards.jsonl": cards, "card_proxy_token_audit.jsonl": token_audit,
        "active_targeted_pool.jsonl": targeted, "active_bridge_pool.jsonl": bridge,
        "active_composition_parent_pool.jsonl": parents, "active_composition_pairs.jsonl": pairs,
        "fresh_historical_backend_view.jsonl": fresh_backend,
        "fresh_model_visible_views.jsonl": [fresh_view(root, fresh_backend) for root in roots],
        "history_preservation.json": {"bank": sources.manifest["inputs"]["bank"], "candidate_cluster_mapping": sources.manifest["inputs"]["members"],
                                      "stem_history": sources.manifest["inputs"]["saturation"], "mechanical_dedup_scope": "unchanged full Bank same root/train, including excluded ordering",
                                      "model_visible_fields": ["root_id", "normalized_stem_signature", "prompt_cluster_count", "candidate_count"],
                                      "family_multiplier_visibility": "BACKEND_ONLY", "semantic_family_status": "UNKNOWN_MECHANICAL_SIGNATURES_ONLY"},
        "role_feedback_envelopes.jsonl": role_envelopes, "rendered_role_messages.jsonl": rendered,
        "round0_nonordering_complete_examples.json": round0_examples,
        "explicit_no_feedback_role_examples.json": absence_examples,
        "before_root_aggregates.jsonl": before_projection["root_aggregates"],
        "derived_signal_rebuild_report.json": {"weights": weight_changes,
                                              "root_aggregate_status": "70_COMPLETE_FROM_INDEPENDENT_NUMERATOR_DENOMINATOR_SUMS",
                                              "unreconstructible_active_cluster_count": sum(c["active_observation_status"] != "RETAINED_INDEPENDENT_OBSERVATION" for c in clusters.values()),
                                              "unknown_and_not_forwarded": ["Per-member alias observation splits when canonical source is excluded and independent records are absent", "Historical root capability/gap narratives without reliable independent aggregation", "Semantic validity and Gold correctness"],
                                              "individual_metrics_and_lifecycle": "retained original observations and labels unchanged",
                                              "m0": "existing per-candidate transition history retained backend-only; no new M0 model-visible aggregate",
                                              "selection_rules_reused": True},
        "dev_global_source_and_usage.json": dev_audit,
        "current_card_comparison.json": {"live13_sources": current_selection, "new_sources": new_selection,
                                         "selection_identity_equal": current_selection == new_selection,
                                         "current_122711": current_122711, "current_frozen_inputs_changed": False},
        "offline_acceptance.json": {"checks": checks, "passed": all(checks.values()), "leaked_candidate_ids": leaked_ids,
                                    "generic_stem_collision_count": len(shared_stems),
                                    "generic_stem_collision_candidate_ids": shared_stems,
                                    "stem_collision_interpretation": "Generic wording is not candidate identity; complete QA and observation provenance checked independently."},
    }
    artifacts.update({f"active_{key}.jsonl": value for key, value in projection.items()})
    artifacts.update({f"retained_{key}_backend_history.jsonl": value for key, value in preserved_features.items()})
    for name, value in artifacts.items():
        emit(name, value)
    if not all(checks.values()):
        raise ValueError(f"offline_acceptance_failed:{[k for k,v in checks.items() if not v]}")
    return counts, protected_before


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("output_must_be_new_directory")
    output.mkdir(parents=True)
    with offline_guard(output) as call_audit:
        sources = Sources(args.source_manifest)
        for name, module in sorted(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("current_method.restart_n5_spell") and path and Path(path).suffix == ".py":
                sources.identities[str(Path(path).resolve())] = file_sha(path)
        counts, protected = compile_views(sources, output)
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py", top_level_dir=str(Path(__file__).parents[3]))
        stream = io.StringIO()
        with patch.object(tempfile, "tempdir", str(output)):
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        write_artifact(output, "test_results.json", {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                                                    "passed": result.wasSuccessful(), "output": stream.getvalue()})
        if not result.wasSuccessful():
            raise ValueError("offline_tests_failed")
        unchanged = sources.unchanged()
        protected_after = [{**r, "after_sha256": file_sha(r["path"]),
                            "unchanged_during_this_task": file_sha(r["path"]) == r["before_sha256"],
                            "matches_handoff_sha": file_sha(r["path"]) == r["sha256"]} for r in protected]
        write_artifact(output, "protected_after.json", protected_after)
        write_artifact(output, "source_identity_audit.json", unchanged)
        if not all(r["unchanged"] for r in unchanged) or not all(r["unchanged_during_this_task"] for r in protected_after):
            raise ValueError("source_or_protected_file_changed_during_export")
        code = [{"path": str(Path(module.__file__).resolve()), "sha256": file_sha(module.__file__)}
                for name, module in sorted(sys.modules.items()) if name.startswith("current_method.restart_n5_spell")
                and getattr(module, "__file__", None) and Path(module.__file__).suffix == ".py"]
        own_code = [{"path": str(path), "sha256": file_sha(path)} for path in sorted(Path(__file__).parent.glob("*.py"))]
        write_artifact(output, "implementation_identity.json", {"version": VERSION, "imported_local_code": code, "versioned_package_files": own_code,
                                                               "source_manifest": str(sources.manifest_path), "source_manifest_sha256": file_sha(sources.manifest_path)})
        write_artifact(output, "zero_external_calls.json", call_audit)
        write_artifact(output, "final_report.json", {"status": READY, "counts": counts, "offline_tests_passed": result.testsRun,
                                                     "protected_files_unchanged": len(protected_after), "external_calls": 0,
                                                     "current_entrypoint_switched": False, "new_r03_qa_send_integration": "NOT_CONFIRMED",
                                                     "next_step": "After the new R03 snapshot is frozen, explicitly bind this manifest to the offline loader; separately authorize any production QA sender."})
        inventory = [{"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}
                     for path in sorted(output.iterdir()) if path.is_file()]
        write_artifact(output, "artifact_inventory.json", inventory)
    print(canonical({"status": READY, "output": str(output), "counts": counts}))


if __name__ == "__main__":
    main()
