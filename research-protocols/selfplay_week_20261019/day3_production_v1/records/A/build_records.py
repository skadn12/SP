#!/usr/bin/env python3
"""Mechanically normalize existing A artifacts; no new authoring or semantic QA."""
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LABELS = list("ABCDEF")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


LIBRARY = BASE / "design/PUBLIC_DESIGN_LIBRARY.json"
ADMISSION = BASE / "inventory/PRIORITY_SOURCE_ADMISSION.json"
FULL_REVIEW = BASE / "inventory/PRIORITY_FULL_SOURCE_REVIEW.json"
admission = read(ADMISSION)
full_review = read(FULL_REVIEW)
records = []
provenance = []
slot_status = []


def criterion(identifier, refs, rationale, result="PASS"):
    return {"criterion_id": identifier, "result": result,
            "evidence_ref_ids": list(dict.fromkeys(refs)), "rationale": rationale}


for root_num in ["103866", "104814"]:
    root = "cc-train-episode-" + root_num
    author_dir = BASE / "candidates/A" / root
    plan_path = author_dir / "PLAN.json"
    packet_path = BASE / "packets/A" / (root + ".json")
    source_path = BASE / "sources" / (root + ".json")
    source = read(source_path)
    source_by_ref = {t["turn_ref"]: (i, t) for i, t in enumerate(source["turns"])}
    plan = read(plan_path)
    source_review_id = "DAY3-PRIORITY-SOURCE-ADMISSION-" + root
    source_admission = next(s for s in admission["sources"] if s["root_id"] == root)
    semantic_review = next(s for s in full_review["roots"] if s["root_id"] == root)
    assert source_admission["source_binding"]["sha256"] == sha(source_path)
    assert semantic_review["source_sha256"] == sha(source_path)
    frozen_at = plan.get("frozen_before_drafts_at_utc", plan.get("created_at_utc"))
    recon_path = BASE / ("qa/A_" + root_num + "/RECONSTRUCTION.json")
    review_path = BASE / ("qa/A_103866/CONTENT_REVIEW.json" if root_num == "103866"
                          else "qa/A_104814/POST_CLARIFICATION_REVIEW.json")
    reconstruction = read(recon_path)
    review = read(review_path)
    blind_items = reconstruction.get("items", reconstruction.get("reconstructions"))
    review_items = review.get("candidate_results", review.get("results"))
    for blind in blind_items:
        candidate_id = blind["candidate_id"]
        qtype = "ordering" if "-ordering-" in candidate_id else "multi_select"
        number = candidate_id.split("-" + qtype + "-")[1][:2]
        slot = next(s for s in plan["slot_plans"]
                    if s["question_type"] == qtype and s["local_slot"] == int(number))
        if root_num == "103866":
            stem = qtype + "-" + number
            private_path = author_dir / "private" / (stem + ".json")
            initial_public_path = author_dir / "public" / (stem + ".json")
            final_public_path = initial_public_path
            clarification = None
            clarification_path = None
        else:
            stem = qtype + "_" + number
            private_path = author_dir / (stem + ".private.json")
            initial_public_path = author_dir / (stem + ".public.json")
            final_public_path = author_dir / "clarified_public" / (stem + ".clarified.public.json")
            clarification_path = author_dir / "clarification_records" / (stem + ".json")
            clarification = read(clarification_path)
        private = read(private_path)
        public = read(final_public_path)
        reviewed = next(r for r in review_items if r["candidate_id"] == candidate_id)
        design = slot["design_id"]
        assert public["candidate_id"] == private["candidate_id"] == candidate_id
        if root_num == "103866":
            assert reviewed["original_public_binding"]["sha256"] == sha(final_public_path)
        else:
            assert reviewed["final_public_sha256"] == sha(final_public_path)
            assert reviewed["private_sha256_unchanged"] == sha(private_path)
            assert reviewed["initial_public_sha256_unchanged"] == sha(initial_public_path)

        refs = []
        ref_by_key = {}

        def add_ref(evidence, role):
            turn_ref = evidence["turn_ref"]
            index, turn = source_by_ref[turn_ref]
            quote = evidence["quote"]
            start = turn["text"].find(quote)
            assert start >= 0, (candidate_id, turn_ref, quote)
            assert evidence.get("speaker", turn["speaker"]) == turn["speaker"]
            if "global_turn_index_1_based" in evidence:
                assert evidence["global_turn_index_1_based"] == index + 1
            if "position" in evidence:
                assert evidence["position"] == index
            key = (turn_ref, quote, role)
            if key not in ref_by_key:
                ref_id = f"{root_num}-{qtype}-{number}-E{len(refs)+1:03d}"
                # Parent has no scene_ref; preserve its existing session identifier.
                scene_id = turn["scene_ref"] or turn["session_ref"]
                ref_by_key[key] = ref_id
                refs.append({"ref_id": ref_id, "scene_id": scene_id,
                             "turn_id": turn_ref,
                             "source_position": {"global_turn_index": index,
                                                 "span_start": start,
                                                 "span_end": start + len(quote)},
                             "quote": quote, "semantic_role": role})
            return ref_by_key[key]

        relation_keys = []
        selected_ids = []
        all_mentions = []
        revision_ids = []
        ordering_pairs = None
        if qtype == "multi_select":
            option_source = blind.get("option_verdicts", blind.get("option_reconstruction"))
            options = []
            for label in LABELS:
                item = option_source[label]
                original_verdict = item["verdict"]
                verdict = "REFUTED" if original_verdict.startswith("REFUTED_") else original_verdict
                assert verdict in ("SUPPORTED", "REFUTED")
                role = "SUPPORT" if verdict == "SUPPORTED" else "REFUTE"
                ev_ids = [add_ref(e, role) for e in item["evidence"]]
                relation_key = f"{private.get('fact_family', private.get('fact_family_id'))}|{label}|{public['options_or_items'][label]}"
                relation_keys.append(relation_key)
                author_option = private.get("evidence", private.get("evidence_per_label"))[label]
                options.append({"option_id": label, "relation_key": relation_key,
                                "source_verdict": verdict,
                                "support_ref_ids": ev_ids if verdict == "SUPPORTED" else [],
                                "refute_ref_ids": ev_ids if verdict == "REFUTED" else [],
                                "distractor_mechanism": author_option.get("distractor_mechanism") if verdict == "REFUTED" else None,
                                "truth_rationale": "Independent verdict=" + original_verdict + "; " + item.get("reason", item.get("rationale"))})
            mention_container = blind.get("related_semantic_mentions", blind.get("all_related_mentions", blind.get("all_reasonable_mentions")))
            mention_groups = mention_container if isinstance(mention_container, dict) else {"all": mention_container}
            for group, mentions in mention_groups.items():
                for mention in mentions:
                    included = mention.get("semantic_match", mention.get("same_semantic_report"))
                    ref_id = add_ref(mention["evidence"], "OCCURRENCE" if included else "EXCLUDED_OCCURRENCE")
                    all_mentions.append({"group": group, "ref_id": ref_id,
                                         "included_for_semantic_report": included,
                                         "reason": mention["reason"]})
            if design == "MC_FINAL_STATE_REVISION":
                if root_num == "103866":
                    for item in blind["revision_chain"].values():
                        if isinstance(item, dict) and "turn_ref" in item:
                            revision_ids.append(add_ref(item, "REVISION"))
                else:
                    for label in ["A", "B", "C", "F"]:
                        for e in option_source[label]["evidence"]:
                            revision_ids.append(add_ref(e, "REVISION"))
                    for mention in mention_container["Swiss_to_Indian"]:
                        if mention["same_semantic_report"]:
                            revision_ids.append(add_ref(mention["evidence"], "REVISION"))
                revision_ids = list(dict.fromkeys(revision_ids))
            gold = blind.get("blind_gold", blind.get("reconstructed_gold"))
            assert gold == private["gold"]
            assert set(gold) == {o["option_id"] for o in options if o["source_verdict"] == "SUPPORTED"}
            assert len(gold) == slot["target_gold_count"]
            question_payload = {"target_gold_count_inherited": slot["target_gold_count"],
                                "options": options, "gold_option_ids": gold,
                                "question_cutoff": private.get("cutoff", public["question"]),
                                "selection_semantics": "SELECT_SOURCE_SUPPORTED_STATEMENTS_WITH_EXPLICITLY_REFUTED_DISTRACTORS",
                                "gold_count_verified": "PASS"}
        else:
            ordering_items = []
            for label in LABELS:
                if root_num == "103866":
                    item = blind["items"][label]
                    selected = item["selected_evidence"]
                    mentions = item["all_reasonable_mentions"]
                    semantic_key = item["semantic_key"]
                else:
                    selected = blind["selected_first_reports"][label]
                    mentions = blind["all_reasonable_mentions"][label]
                    semantic_key = private["items"][label]["event_semantic_key"]
                occurrences = []
                selected_id = add_ref(selected, "OCCURRENCE")
                selected_ids.append(selected_id)
                for mention in mentions:
                    included = mention.get("semantic_match", mention.get("same_semantic_report"))
                    ref_id = add_ref(mention["evidence"], "OCCURRENCE" if included else "EXCLUDED_OCCURRENCE")
                    occurrences.append({"ref_id": ref_id, "included_for_event_key": included,
                                        "exclusion_reason": None if included else mention["reason"],
                                        "reviewer_verified": True})
                    all_mentions.append({"item_id": label, "ref_id": ref_id,
                                         "included_for_event_key": included,
                                         "reason": mention["reason"]})
                assert any(o["ref_id"] == selected_id and o["included_for_event_key"] for o in occurrences)
                included_count = sum(o["included_for_event_key"] for o in occurrences)
                relation_keys.append(semantic_key)
                ordering_items.append({"item_id": label, "event_semantic_key": semantic_key,
                                       "atomicity": "PASS", "mention_policy": "FIRST_IN_DECLARED_SCOPE",
                                       "requested_ordinal": None, "all_occurrences": occurrences,
                                       "selected_ref_id": selected_id,
                                       "repeat_status": "REPEATED_SAME_FACT" if included_count > 1 else "SINGLE",
                                       "selection_rationale": "Existing independent reconstruction identifies this as the earliest matching report under the predeclared full-source first-report policy; differing actor, modality or speech act is excluded with the independent review reason. See MECHANICAL_PROVENANCE.json for verbatim mention reasons and all 15 pairs."})
            ordering_pairs = blind.get("all15pair_comparisons", blind.get("all_15_pairs"))
            assert len(ordering_pairs) == 15
            gold = blind.get("blind_gold", blind.get("reconstructed_gold"))
            assert gold == private["gold"]
            profile = blind.get("profile", blind.get("scene_profile"))
            pair_count = blind.get("within_scene_pairs", blind.get("same_scene_pair_count"))
            question_payload = {"gold_definition": "SOURCE_APPEARANCE_ORDER",
                                "scope_predeclared": plan.get("ordering_rule", plan.get("global_ordering_rule")) + " Public policy: " + public["question"],
                                "scene_event_counts": profile,
                                "within_scene_pair_count": pair_count, "items": ordering_items,
                                "gold_item_ids": gold, "unique_order_verified": "PASS",
                                "all_reasonable_mentions_searched": "PASS"}
            positions = {r["ref_id"]: r["source_position"]["global_turn_index"] for r in refs}
            expected_gold = sorted(LABELS, key=lambda label: positions[ordering_items[LABELS.index(label)]["selected_ref_id"]])
            assert gold == expected_gold
            selected_scenes = Counter(next(r["scene_id"] for r in refs if r["ref_id"] == rid) for rid in selected_ids)
            assert sorted(selected_scenes.values(), reverse=True) == profile
            assert sum(n * (n - 1) // 2 for n in selected_scenes.values()) == pair_count

        all_ref_ids = [r["ref_id"] for r in refs]
        target_refs = selected_ids or list(dict.fromkeys(
            ref_id for option in question_payload["options"]
            for ref_id in option["support_ref_ids"] + option["refute_ref_ids"]))
        if root_num == "103866":
            target_claim = blind["target_assessment"]
            target_reason = json.dumps(target_claim, ensure_ascii=False)
            shortcut_reason = blind["shortcut_assessment"]["notes"]
            shortcut_flags = []
        else:
            target_claim = private["target_realization"]
            target_reason = blind["target_reason"]
            shortcut_reason = blind["shortcut_reason"]
            shortcut_flags = blind["shortcut_flags"]
        if design == "MC_BINDING":
            criteria = [criterion(k, target_refs, "Existing independent target PASS mapped to " + k + ": " +
                                 (target_claim.get(k, target_reason) if root_num == "103866" else target_reason))
                        for k in ["B1", "B2", "B3"]]
            focal = ["A", "B", "C", "D"] if root_num == "103866" else ["A", "B"]
        elif design == "MC_FINAL_STATE_REVISION":
            criteria = [criterion(k, revision_ids if k != "S2" else target_refs,
                                 "Existing independent target PASS mapped to " + k + ": " +
                                 (target_claim[k] if root_num == "103866" else target_reason))
                        for k in ["S1", "S2", "S3"]]
            focal = ["A", "B", "C"] if root_num == "103866" else ["A", "C", "D", "F"]
        else:
            detail = {
                "O1": "Independent reconstruction preserves actual source scene histogram " + str(profile) + "; no boundary changes.",
                "O2": f"Independent reconstruction lists all 15 uniquely comparable pairs, including {pair_count} same-scene pairs; verbatim pair records are in MECHANICAL_PROVENANCE.json.",
                "O3": "Existing independent content/target PASS covers six atomic events, the complete reasonable-mention inventory and first-report selection policy.",
                "O4": "Independent Gold equals strict ascending selected source positions; original event dates do not determine this order."
            }
            criteria = [criterion(k, target_refs, detail[k] + " Independent target assessment: " + target_reason)
                        for k in ["O1", "O2", "O3", "O4"]]
            focal = []
        tags = ["EXPLICIT_REVISION_CHAIN"] if design == "MC_FINAL_STATE_REVISION" else []
        lexical = private["lexical_shortcuts"]
        direct_quote = lexical.get("direct_quotes", lexical.get("direct_quote", lexical.get("direct_source_phrases", False)))
        name_locator = lexical.get("proper_names", lexical.get("proper_name_locator", lexical.get("proper_name_or_id_locator", lexical.get("proper_names_or_unique_objects_present", False))))
        # This field asks about an ordering cue; numeric localization is separately disclosed.
        numeric_order_cue = lexical.get("numeric_or_date_order_cue", False)
        generation = [{"attempt_id": candidate_id + ":INITIAL", "kind": "INITIAL",
                       "request_id": None, "started_at": private["created_at_utc"],
                       "source_review_id": source_review_id, "requested_output_token_cap": 3072,
                       "actual_cap_support": "UNKNOWN", "actual_output_tokens": None,
                       "request_input_sha256": sha(packet_path),
                       "raw_response_path": str(initial_public_path),
                       "candidate_content_sha256": sha(initial_public_path)}]
        if clarification:
            generation.append({"attempt_id": candidate_id + ":CONTENT_CLARIFICATION", "kind": "CONTENT_CLARIFICATION",
                               "request_id": None, "started_at": clarification["created_at_utc"],
                               "source_review_id": source_review_id, "requested_output_token_cap": 3072,
                               "actual_cap_support": "UNKNOWN", "actual_output_tokens": None,
                               "request_input_sha256": sha(packet_path),
                               "raw_response_path": str(final_public_path),
                               "candidate_content_sha256": sha(final_public_path)})
        turn_ids = list(dict.fromkeys(r["turn_id"] for r in refs))
        scene_ids = list(dict.fromkeys(r["scene_id"] for r in refs))
        qa_time = review.get("completed_at_utc", review.get("reviewed_at_utc"))
        record = {"candidate_id": candidate_id, "inherited_slot_id": slot["slot_id"],
                  "arm": "A", "stage": 1, "root_id": root, "question_type": qtype,
                  "design_assignment_frozen_at": frozen_at, "public_library_sha256": sha(LIBRARY),
                  "source_binding": {"source_path": str(source_path), "source_sha256": sha(source_path),
                                     "source_version": source["schema_version"],
                                     "scope_description": f"Complete {len(source['turns'])}-turn cumulative source, including parent/R01 and all six R02 scenes; authored questions target admitted R02 facts. Prior history remains visible for semantics and first-report checks.",
                                     "source_scope_sha256": sha(source_path),
                                     "train_admission_record": str(ADMISSION), "suffix_policy_id": None,
                                     "full_source_visible_to_qa": True},
                  "source_review": {"review_id": source_review_id,
                                    "reviewed_at": admission["created_at_utc"],
                                    "reviewer": full_review["reviewer"], "status": "PASS",
                                    "source_integrity": "PASS", "entities_and_state_consistency": "PASS",
                                    "available_relation_material": ["MC_BINDING", "MC_FINAL_STATE_REVISION"],
                                    "available_ordering_material": ["ORDER_3_2_1", "ORDER_2_2_2", "ORDER_4_2"],
                                    "shortfall_codes": ["SHORTFALL_SOURCE_RELATION_MC_NEGATION_CONDITIONAL_JOINT"],
                                    "rationale": "Timestamp is the existing structural/material admission timestamp, not an invented completion time for the separate untimestamped semantic review. Full prior history read=true in admission. Semantic scoped PASS and exclusions are bound in " + str(FULL_REVIEW) + "; that record was bound in the author packet/PLAN before drafting. No unqualified consistency or candidate acceptance claim: " + json.dumps(semantic_review, ensure_ascii=False)},
                  "budget": {"initial_allowed": 1, "content_clarification_allowed": 1,
                             "initial_used": 1, "content_clarification_used": 1 if clarification else 0,
                             "requested_output_token_cap": 3072, "actual_cap_support": "UNKNOWN",
                             "actual_cap_evidence": None, "solver_calls": 0},
                  "generation_history": generation,
                  "fact_family": {"family_id": private.get("fact_family", private.get("fact_family_id")),
                                  "atomic_relation_keys": relation_keys, "related_candidate_ids": [],
                                  "reuse_class": "OVERLAPPING_FACTS_DIFFERENT_RELATION",
                                  "duplicate_review": "UNKNOWN"},
                  "evidence_span": {"turn_ids": turn_ids, "scene_ids": scene_ids,
                                    "single_or_multi_turn": "MULTI_TURN" if len(turn_ids) > 1 else "SINGLE_TURN",
                                    "same_or_cross_scene": "CROSS_SCENE" if len(scene_ids) > 1 else "SAME_SCENE",
                                    "revision_chain_ref_ids": revision_ids, "evidence_refs": refs},
                  "lexical_shortcuts": {"direct_quote": direct_quote,
                                        "proper_name_or_id_locator": name_locator,
                                        "numeric_or_date_order_cue": numeric_order_cue,
                                        "option_lengths": [len(public["options_or_items"][label]) for label in LABELS],
                                        "correctness_correlated_style": "UNKNOWN",
                                        "scene_only_sufficient": False if qtype == "ordering" else None,
                                        "checks": [criterion("EXISTING_INDEPENDENT_SHORTCUT_REVIEW", target_refs,
                                                             shortcut_reason + " Flags: " + json.dumps(shortcut_flags))],
                                        "rationale": "Existing independent shortcut PASS is retained with disclosed lexical locators. Per-feature author flags are carried over, and lengths are recomputed from the exact final English public strings; separate correctness/style statistical classification was not performed, hence UNKNOWN. Numeric strings may locate content without deciding source order. " + shortcut_reason},
                  "target_realization": {"primary_relation_id": design if qtype == "multi_select" else None,
                                         "ordering_profile_id": design if qtype == "ordering" else None,
                                         "evidence_structure_tags": tags, "criteria": criteria,
                                         "focal_option_ids": focal, "premise_groups": [],
                                         "minimal_sufficient_ref_sets": [],
                                         "single_turn_sufficiency_sweep": {"full_scope_searched": False,
                                                                           "status": "NOT_APPLICABLE",
                                                                           "sufficient_single_turn_ids": [],
                                                                           "rationale": "No E_CROSS_TURN_NECESSARY claim. Full-source semantic mention review was performed independently, but no separate single-turn-necessity sweep or premise ablation is claimed."},
                                         "premise_ablation": [], "target_implemented": "YES",
                                         "target_implementation_rationale": "Mechanical mapping of the existing independent target PASS for the bound public content; no new semantic QA signature by the formatter. " + target_reason},
                  qtype: question_payload,
                  "qa_review": {"review_id": str(review_path) + "#" + candidate_id,
                                "reviewer": review["reviewer"], "reviewed_at": qa_time,
                                "phase": "AFTER_CONTENT_CLARIFICATION" if clarification else "INITIAL",
                                "source_review_id": source_review_id, "source_hash_matches": "PASS",
                                "all_option_or_pair_truth_checked": "PASS", "relation_target_checked": "PASS",
                                "shortcut_checked": "PASS", "gold_independently_reconstructed": "PASS",
                                "status": "PASS", "criterion_results": copy.deepcopy(criteria),
                                "rationale": "Existing independent content, target and shortcut PASS for exact public hash " + sha(final_public_path) + ". Review: " + str(review_path) + " sha256=" + sha(review_path) + "; blind reconstruction: " + str(recon_path) + " sha256=" + sha(recon_path) + ". This formatter does not replace that independent signature. Verbatim all-mention reasons and all15 ordering pairs are retained in MECHANICAL_PROVENANCE.json. Cross-pool family review is still pending."},
                  "disposition": {"state": "HOLD", "reason_codes": ["CROSS_POOL_FACT_FAMILY_DUPLICATE_REVIEW_PENDING"],
                                  "content_clarification_request": str(BASE / "qa/A_104814/CLARIFICATION_REQUEST.json") if clarification else None,
                                  "accepted_content_sha256": None, "feedback_binding_location": None}}
        records.append(record)
        provenance.append({"candidate_id": candidate_id, "record_file": candidate_id + ".json",
                           "original_public": binding(initial_public_path), "final_public": binding(final_public_path),
                           "private_author_artifact": binding(private_path), "plan": binding(plan_path),
                           "public_packet": binding(packet_path), "independent_reconstruction": binding(recon_path),
                           "independent_review": binding(review_path),
                           "clarification_record": binding(clarification_path) if clarification_path else None,
                           "all_independent_semantic_mentions": all_mentions,
                           "all_15_independent_ordering_pairs": ordering_pairs,
                           "independent_reconstruction_entry": copy.deepcopy(blind),
                           "existing_review_entry": copy.deepcopy(reviewed),
                           "mechanical_field_normalizations": [{
                               "field": "generation_history/0/raw_response_path",
                               "old_value": str(private_path),
                               "new_value": str(initial_public_path),
                               "reason": "Parent requested consistent actual-public-artifact semantics across A/B: raw_response_path must hash to candidate_content_sha256. This changes provenance encoding only; private author artifact remains separately bound, and no provider API trace is asserted."
                           }]})
        slot_status.append({"inherited_slot_id": slot["slot_id"], "candidate_id": candidate_id,
                            "state": "HOLD", "question_record": candidate_id + ".json",
                            "independent_content_QA": "PASS", "duplicate_review": "UNKNOWN"})
    short_slot = next(s for s in plan["slot_plans"] if s["question_type"] == "multi_select" and s["local_slot"] == 3)
    short_path = author_dir / ("shortfalls/multi_select-03.json" if root_num == "103866" else "multi_select_03.SHORTFALL.json")
    short = read(short_path)
    slot_status.append({"inherited_slot_id": short_slot["slot_id"], "candidate_id": None,
                        "state": "SHORTFALL", "question_record": None,
                        "question_type": "multi_select", "target_design": short_slot["design_id"],
                        "target_gold_count_inherited": short_slot["target_gold_count"],
                        "source_shortfall_artifact": binding(short_path),
                        "reason": short_slot.get("reason", short_slot.get("material")),
                        "counts_as_candidate_or_accepted": False,
                        "independent_shortfall_result": review.get("shortfall_result", review.get("shortfall"))})

# Local links derive solely from shared independently cited source turns. This is
# overlap disclosure, not the still pending semantic cross-pool duplicate review.
for record in records:
    cited = set(record["evidence_span"]["turn_ids"])
    record["fact_family"]["related_candidate_ids"] = [other["candidate_id"] for other in records
        if other["candidate_id"] != record["candidate_id"] and other["root_id"] == record["root_id"]
        and cited.intersection(other["evidence_span"]["turn_ids"])]
    write(OUT / (record["candidate_id"] + ".json"), record)

now = datetime.now(timezone.utc).isoformat()
write(OUT / "MECHANICAL_PROVENANCE.json", {
    "schema_version": "day3_A_mechanical_normalization_provenance_v1", "created_at_utc": now,
    "formatter": "/root/day3_eval_boundary", "scope": "10 existing actual questions only; no new authoring or semantic QA.",
    "record_schema": {"path": str(LIBRARY), "sha256": sha(LIBRARY), "pointer": "/schemas/candidate_record"},
    "source_admission": binding(ADMISSION), "full_source_semantic_review": binding(FULL_REVIEW),
    "raw_trace_limit": "No raw provider API response trace is available. generation_history.raw_response_path points to the actual preserved original public artifact for INITIAL and final public artifact for clarification, so each file hashes to candidate_content_sha256; these are interactive author artifacts, not provider response envelopes. Initial private artifacts remain separately bound. The initial draft of these normalized records used private artifact paths, and each per-candidate mechanical_field_normalizations entry records the old and corrected path. request_id, actual usage, and actual cap evidence remain null/UNKNOWN. started_at uses the preserved artifact creation timestamp, not an asserted provider request-start timestamp. request_input_sha256 binds the exact per-root public packet for both attempts; clarification additionally has its independently requested text and hash in the bound clarification record.",
    "language_deviation": "103866 is original English with zero clarifications; 104814 has original Chinese and one independently requested English translation per question. Original public/private files are preserved. This was a post-initial correction, not a preregistered language policy or equal realized compute.",
    "indices": "Every evidence source_position.global_turn_index is zero-based, rederived by exact turn_ref from the bound source; source span_start/end are Python character offsets. Original independent records retain their declared zero/one-based conventions verbatim in each entry.",
    "parent_scene_representation": "When the source scene_ref is null for a parent turn, scene_id preserves that turn's actual session_ref string. This is provenance only and does not resegment any R02 scene or count parent sessions toward ordering quotas.",
    "fact_family_encoding": "Existing author family_id values are preserved. Local related_candidate_ids are a conservative mechanical shared-cited-turn link, not an exhaustive semantic family graph. reuse_class=OVERLAPPING_FACTS_DIFFERENT_RELATION provisionally discloses source-fact reuse with relation tasks; it is not a signed cross-pool classification or a novelty claim. All duplicate_review=UNKNOWN and all dispositions HOLD pending independent cross-pool review. The 104814 work ordering family can have no same-root shared-citation link while still coming from historically exposed Train source.",
    "semantic_signature_limit": "Target criterion mapping B1-B3/S1-S3/O1-O4 inherits the bound independent target/content decisions and evidence. It adds no new semantic reviewer. 103866 reviewer is /root/day3_eval_boundary, 104814 reviewer is /root/day3_public_design. No E_CROSS_TURN_NECESSARY claims or premise-ablation claims.",
    "disposition_limit": "Content QA PASS is separate from global admission. All 10 records HOLD, accepted_content_sha256 null, feedback_binding_location null, solver_calls0. Two shortfalls are only in SLOT_STATUS and have no question record.",
    "candidates": provenance})
write(OUT / "SLOT_STATUS.json", {
    "schema_version": "day3_A_slot_status_v1", "created_at_utc": now,
    "counts": {"planned_slots": 12, "actual_candidate_records": 10,
               "content_QA_PASS": 10, "disposition_HOLD": 10,
               "SHORTFALL": 2, "accepted": 0},
    "slots": slot_status})
print(json.dumps({"records_written": len(records), "shortfalls": 2, "directory": str(OUT)}))
