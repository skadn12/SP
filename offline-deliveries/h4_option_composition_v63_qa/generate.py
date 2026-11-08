"""v63 soft option-composition probe over the unchanged complete root context.

prepare/verify are offline. generate handles two multi-select slots per run.
Ordering remains available in the library but is inactive in this probe.
Scenes remain provisional throughout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path

import collection_source
import prompt_rules
import prompt_layout
import slot_focus
import display_export
import reference_compat
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
QA = DELIVERIES / "h4_feedback_training_pool_v15_qa"
sys.path.insert(0, str(QA))
import qa as v15

legacy, base = v15.legacy, v15.base
ordering_gate = legacy.load("v52_local_ordering_gate", HERE / "ordering_gate_v45.py")
read, write, require = base.read, base.write, base.require
VERSION = "h4_option_composition_v63_generation_v1"
EXPANSION_VERSION = "h4_candidate_collection_v51"
MODEL, TARGET = v15.MODEL, v15.TARGET
TYPES = ("multi_select", "ordering")
GENERATION_TYPES = ("multi_select",)
GENERATION_OUTPUT_CAP = 4096
MIXED_COLLECTION_CONTRACT = {
    "version": "v63_option_composition_soft_hints_v1",
    "generation_types": list(GENERATION_TYPES), "generation_output_cap": GENERATION_OUTPUT_CAP,
    "question_rules_changed": False, "ordering_independent_review_restored": False,
    "ordering_library_supported": True, "ordering_review_active": False,
    "ordering_reviewer_rules_changed": False, "gold_anchor_rules_changed": False,
    "display_permutation_rules_changed": False,
    "full_source_feedback_and_priors_preserved": True,
    "automatic_replay": False, "original_runs_unchanged": True,
}

COMPOSITION_CONTRACT = {
    "version": "v63_soft_preferred_support_counts_v1",
    "preferred_support_range": [2, 3, 4, 5, 6],
    "preference_is_soft_diagnostic_only": True,
    "deviation_never_rejects_repairs_abstains_or_replaces": True,
    "no_extra_http_or_planner": True,
    "composition_instructions_only_no_new_admission_gate": True,
    "distinct_facts_and_no_true_false_twins_guidance": True,
    "same_scope_contradiction_and_unknown_not_false_guidance": True,
}


def generation_budget(root_count, slots):
    require(root_count == 1 and slots == 2, "multiselect_probe_exact_two_slots_required")
    return {"candidate_slots": 2, "initial_generation_requests": 1,
            "mechanical_repair_requests_max": 1, "ordering_review_requests_max": 0,
            "http_attempts_max": 2, "solver_full_calls_max": 36}

PROMPT_ARMS = ("old", "current")
COMPARISON_CONTRACT = {
    "version": "v60_multiselect_probe_unchanged_question_rules_v1",
    "experiment": "multiselect_source_aware_soft_focus_on_complete_root_context",
    "old": "legacy_optional_api_only_not_an_authorized_live_comparison_arm",
    "current": "v60_unchanged_multiselect_prompt_gold_and_permutation",
    "shared_adaptations": ["full_provisional_root_source", "per_candidate_soft_scene_topic_focus",
                           "exact_six_statement_set_dedup_only"],
    "initial_context": "frozen_prior_only_for_both_question_types",
    "repair_context": "frozen_prior_plus_same_question_type_accepted_drafts_only",
    "reviewer": "inactive_in_multiselect_probe_ordering_library_retained",
    "schema_fields_changed": False,
    "new_admission_gates": False,
    "new_semantic_admission_gates": False,
    "new_mechanical_checks": [],
    "inherited_mechanical_checks": ["fixed_public_ordering_stem"],
    "individual_answer_cardinality": [2, 3, 4, 5, 6],
    "batch_distribution_policy": "per_slot_soft_preferred_support_hints_diagnostic_only",
    "preferred_answer_cardinalities": [2, 3, 4, 5, 6], "fixed_cardinality_quota": False,
    "slot_focus_contract": slot_focus.CONTRACT,
    "evidence_output_order": "instruction_and_diagnostic_only_not_an_admission_gate",
    "condition_preservation": True,
    "contradiction_mix": "2_to_6_supports_any_remaining_options_require_explicit_contradiction",
    "prior_set_reuse": "explicit_public_used_sets_and_silent_exact_set_check_no_new_gate",
    "comparison_fields": "retained_for_compatible_single_run_identity_only",
}

INTERVAL = 35
DEFAULT_PRIOR = (DELIVERIES / "h4_root_merged_context_v17/runs/"
                 "merged_latest_02/training_pool/TRAINING_POOL.jsonl")

MERGED_INSTRUCTION = """
FULL CUMULATIVE ROOT CONTEXT:
All source_turns belong to one root: the complete formal baseline followed by
retained provisional A and B dialogue. The supplied dialogue is the evidence for this
QA probe; its provisional suffix has NOT been admitted to training. Do not claim
that the source scenes are approved, externally factual, or semantically certified.
Use the actual supplied words, without importing source cards, plans or reviews.
Use slot_focus_by_candidate_id for the requested candidate's soft scene and topic
preference. Begin the evidence search in its preferred_scene_refs and seek a
coherent combination of facts relevant to its focus. A focus is a topic cue,
never evidence or an answer. The complete source remains available for every slot;
use other turns when needed. Keep each question coherent; abstain rather than
inventing a fact to fill a slot. Following the focus is not an admission gate.
The complete source remains available, without truncation.
Scene storage order does not prove real-world cross-scene event chronology.
Ordering questions retain the explicit conversation-order scope: when the
statements APPEAR IN THE SUPPLIED DIALOGUE. Do not silently replace it with
real-world chronological inference. Preserve narrow coherent question scope.
Prior question inventory contains no current Solver scores, answers or Gold.
Supported paraphrases and partial overlap are allowed as training augmentation.
Avoid an exact copy of an entire six-statement set.
USED WHOLE-QUESTION SETS:
used_six_statement_sets is a compact index of already-used public statements,
grouped as complete six-statement sets. Each entry lists zero-based prior_indices
into previous_candidates_dedup_only: read the six facts at its first index; the
remaining indices name exact duplicate sets. Old stems and candidate IDs are omitted
from this index-based projection; all original records remain frozen on disk.
No statement is omitted or truncated.
It contains no answers or judgments.
For each requested candidate, first choose evidence from the full source using
its soft focus; do not copy a previous question as a draft. Before returning it,
compare its six proposition/item texts with every indexed set across both types
and with the other candidate in this response. Compare the complete set without
regard to display labels or statement order, ignoring case and extra whitespace.
If all six texts repeat a prior set, select a different evidenced combination
before emitting JSON. Renaming a topic, question, label or candidate ID does not
make the same six statements new. Individual overlap and supported paraphrases
remain allowed; this instruction adds no semantic novelty requirement. If no
coherent supported set is available, use the existing abstention field.
"""


def checked_sources(path, requested_roots=None):
    row = collection_source.checked_source(path, base, requested_roots=requested_roots)
    return read(path.parent / "COLLECTION_BINDINGS.json"), [row], []


def checked_expansion(history_path):
    return history_path.parent / "COLLECTION_BINDINGS.json", history_path.parent / "MANIFEST.json"


def _prior_pool_rows(path, root):
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        obj = read(path)
        rows = obj if isinstance(obj, list) else obj.get("questions", obj.get("candidates", []))
    result = []
    for original in rows:
        question = original.get("original_selected_question", original)
        if question.get("root_id") != root:
            continue
        values = question.get("options") or question.get("items")
        require(isinstance(values, dict) and all(isinstance(t, str) for t in values.values()),
                "prior_pool_statement_text_missing:" + str(path))
        result.append({"candidate_id": question["candidate_id"], "root_id": root,
                       "question_type": question["question_type"], "question": question["question"],
                       "facts": list(values.values()), "hard_dedup": True,
                       "history_status": "EXPLICIT_PRIOR_POOL_EXACT_SET_ONLY",
                       "source_path": str(path), "source_sha256": base.digest(path.read_bytes())})
    return result


def prompt_prior(rows):
    result = legacy.prompt_prior(rows)
    for row in result:
        row["dedup_action"] = ("avoid_exact_six_statement_set_only_paraphrases_allowed"
                               if row["dedup_action"] == "avoid_already_selected" else "warning_only")
        # Duplicate exclusion is defined by the six statements, not the old
        # question stem. Keep every statement/action, omit the redundant
        # stem and candidate ID in the HTTP projection; the full frozen prior remains on disk.
        row.pop("question", None)
        row.pop("candidate_id", None)
    return result


def used_statement_sets(prior):
    """Compact the public hard-dedup inventory; never include evidence or Gold."""
    grouped = {}
    for prior_index, row in enumerate(prior):
        facts = row.get('facts')
        kind = row.get('question_type')
        if (row.get('dedup_action') != 'avoid_exact_six_statement_set_only_paraphrases_allowed'
                or kind not in TYPES or not isinstance(facts, list) or len(facts) != 6
                or not all(isinstance(text, str) and text.strip() for text in facts)):
            continue
        key = (kind, tuple(sorted(' '.join(text.split()).casefold() for text in facts)))
        if key not in grouped:
            grouped[key] = {'question_type': kind, 'prior_indices': []}
        grouped[key]['prior_indices'].append(prior_index)
    return list(grouped.values())


def adapted_common():
    common = legacy.COMMON.replace(
        "It includes P, R01, and recovered R02 A/B turns.",
        "It includes the full formal baseline and retained provisional dialogue in one root.")
    common = common.replace(
        "Do not create another copy of\nthe same fact set by changing wording, negation or display labels.",
        "Supported paraphrases are allowed; avoid exact six-statement copies.")
    common = common.replace("Prioritize new R02 developments", "Use each candidate's soft slot focus when supplied")
    require("Do not create another copy" not in common, "legacy_paraphrase_ban_not_removed")
    return common


def generation_type_instructions(qtype, *, composition=False):
    """Reorder the displayed existing schema without changing accepted fields."""
    require(qtype in TYPES, "generation_question_type_invalid")
    if qtype == "multi_select":
        old = "{proposition,relation,turn_ref,speaker,quote,rationale}."
        new = "{turn_ref,speaker,quote,proposition,relation,rationale}."
        instructions = legacy.MULTI
        legacy_count = ("The number of supports may be 2,3,4,5 or6, including all six when warranted. "
                        "Do not force a count.")
        require(instructions.count(legacy_count) == 1, "legacy_multi_count_instruction_changed")
        sampling = ("Use the private per-candidate composition preference in the current task tail. "
                    "That soft preference never changes the valid 2–6 range or evidence-derived labels. "
                    if composition else
                    "For this collection, preferentially seek 2, 3 or 4 supports by finding explicitly "
                    "incompatible options first. This is a soft sampling preference, not a quota or validity gate; "
                    "5 or 6 supports remain valid. ")
        instructions = instructions.replace(legacy_count,
            "Each question may have 2, 3, 4, 5 or 6 supports, including all six. Any remaining "
            "options must be explicitly contradicts in the same actor/event/time scope. "
            + sampling + "Never relabel a true option or omit a supported Gold label to obtain variety.")
    else:
        old = "items,gold,item_evidence,order_reason."
        new = "item_evidence,items,gold,order_reason."
        instructions = legacy.ORDERING
        legacy_stem = ("The stem must explicitly contain 'APPEAR IN THE SUPPLIED DIALOGUE', "
                       "with a narrow coherent topic.")
        require(instructions.count(legacy_stem) == 1, "legacy_ordering_stem_instruction_changed")
        instructions = instructions.replace(legacy_stem,
            "Use the exact fixed public stem specified in the current task tail, with only its "
            "optional exact scene scope. No topic summary, ordered event list or answer clues may "
            "appear in the stem. Keep the coherent topic in the topic field.")
    require(instructions.count(old) == 1, "evidence_first_legacy_schema_listing_changed")
    return instructions.replace(old, new, 1)


def make_payload(run, root, qtype, context):
    require(qtype in TYPES, "generation_question_type_invalid")
    manifest = read(run / "MANIFEST.json")
    arm = manifest["prompt_arm"]
    require(arm in PROMPT_ARMS, "generation_prompt_arm_invalid")
    context = focus_context(manifest, context)
    composition = bool(manifest.get("preferred_support_count_by_candidate_id")) and qtype == "multi_select"
    type_instructions = (generation_type_instructions(qtype, composition=composition) if arm == "current"
                         else (legacy.MULTI if qtype == "multi_select" else legacy.ORDERING))
    instructions = (adapted_common() + (MERGED_INSTRUCTION if arm == "current" else "")
                    + type_instructions)
    payload, receipt = legacy.payload_with_feedback(run, root, instructions, context,
        min(GENERATION_OUTPUT_CAP, 1024 + 2048 * len(context["candidate_ids"])))
    return (prompt_layout.append_task_tail(payload, receipt, qtype, composition=composition)
            if arm == "current" else (payload, receipt))


def focus_context(manifest, context):
    """Keep each pending ID's frozen preference, including during repairs."""
    result = {**context, "priority_new_scene_refs": [],
            "used_six_statement_sets": used_statement_sets(context.get('previous_candidates_dedup_only', [])),
            "slot_focus_by_candidate_id": slot_focus.subset(
                manifest.get("slot_focus_by_candidate_id", {}), context["candidate_ids"])}
    preferences = manifest.get("preferred_support_count_by_candidate_id", {})
    if preferences:
        result["preferred_support_count_by_candidate_id"] = {
            qid: preferences[qid] for qid in context["candidate_ids"] if qid in preferences}
    else:
        result.pop("preferred_support_count_by_candidate_id", None)
    return result


def initial_context(run, root, qtype):
    """Build each type's first request from frozen source and prior inventory."""
    manifest = read(run / "MANIFEST.json")
    require(qtype in TYPES and root in manifest["roots"], "initial_context_identity_invalid")
    rows = [row for row in manifest["source_rows"] if row["root_id"] == root]
    require(len(rows) == 1, "initial_context_source_row_missing_or_ambiguous")
    source = read(Path(rows[0]["source_path"]))
    prior = read(run / "prior" / (root + ".json"))
    slots = manifest["candidate_slots_per_type_per_root"]
    ids = slot_focus.candidate_ids(root, manifest['comparison_id'], qtype, slots)
    return focus_context(manifest, {"root_id": root, "context_mode": "full_provisional_candidate_root",
        "candidate_ids": ids, "source_turns": source["turns"],
        "previous_candidates_dedup_only": prompt_prior(prior),
        "new_scene_evidence_required": False, "paraphrase_augmentation_allowed": True})


def verify_prompt_request(run, payload, proof):
    """Replay the exact frozen prompt layout, including legacy optional API."""
    manifest = read(run / "MANIFEST.json")
    arm = manifest.get("prompt_arm")
    require(arm in PROMPT_ARMS, "generation_prompt_arm_invalid")
    role = proof.get("role")
    require(role in ("question_generator", "ordering_reviewer"), "prompt_request_role_invalid")
    kind = "ordering_review" if role == "ordering_reviewer" else proof.get("question_type")
    require(kind in prompt_layout.KINDS, "prompt_request_kind_invalid")
    require(role == "question_generator" and kind in GENERATION_TYPES,
            "multiselect_probe_request_type_invalid")
    root = proof["root_id"]
    if role == "question_generator":
        actual_context = json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split("\nINPUT_JSON\n", 1)[1])[0]
        require(actual_context == focus_context(manifest, actual_context), "actual_slot_focus_changed")
    suffix = legacy.feedback.build_feedback_suffix(run / "feedback", root, "writer", TARGET)
    args = (payload, proof["feedback_receipt"], suffix, root, TARGET,
            base.digest((run / "feedback/source_lock.json").read_bytes()))
    if role == "question_generator" and arm == "old":
        result = prompt_layout.verify_legacy_request(*args)
    else:
        result = prompt_layout.verify_request(*args, expected_kind=kind,
            composition=bool(manifest.get("preferred_support_count_by_candidate_id")))
    require(proof.get("payload_sha256") == base.digest(payload), "frozen_payload_changed")
    return result


def make_review_payload(run, root, candidate, source):
    original = ordering_gate.build_review_prompt(candidate, source)
    prefix, separator, data = original.partition("\nDATA:\n")
    require(bool(separator), "ordering_review_data_marker_missing")
    content = prefix + separator + data
    payload = {"model": MODEL, "temperature": .25, "max_tokens": 4096, "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}]}
    payload, receipt = legacy.feedback.append_feedback_to_request(payload, run / "feedback", root, "writer", TARGET)
    return prompt_layout.append_task_tail(payload, receipt, "ordering_review")


def _dependency_paths():
    profile, factory, annotate, capacity, dev = base.transport()
    paths = {Path(__file__).resolve(), Path(legacy.__file__), Path(base.__file__),
             Path(v15.__file__), Path(v15.dedup.__file__), Path(legacy.feedback.__file__),
             Path(legacy.multi.__file__), Path(legacy.ordering_gate.__file__), legacy.ordering_gate.PROMPTS_PATH}
    paths.update(QA.glob("*.py"))
    paths.update(HERE.glob("*.py"))
    for obj in (profile, factory, annotate, capacity, dev):
        value = inspect.getsourcefile(obj) or inspect.getfile(obj)
        if value:
            paths.add(Path(value))
    paths.update(path for path in base.TOKENIZER.rglob("*") if path.is_file())
    return profile(), paths


def prepare(run, merge_manifest, *, roots=None, slots=2, prior_pools=(),
            prompt_arm="current", comparison_id=None, slot_focus_map=None,
            preferred_support_counts=None):
    require(not run.exists(), "generation_run_already_exists")
    require(prompt_arm in PROMPT_ARMS, "generation_prompt_arm_invalid")
    comparison_id = run.name if comparison_id is None else comparison_id
    require(isinstance(comparison_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", comparison_id),
            "comparison_id_invalid")
    require(type(slots) is int and 1 <= slots <= 8, "slots_outside_1_8")
    merge, sources, excluded = checked_sources(merge_manifest, roots)
    focus_map = deepcopy(slot_focus_map) if slot_focus_map is not None else {}
    focus_ids = [qid for row in sources for kind in GENERATION_TYPES
                 for qid in slot_focus.candidate_ids(row['root_id'], comparison_id, kind, slots)]
    preferences = {}
    if preferred_support_counts is not None:
        require(prompt_arm == "current", "composition_requires_current_prompt_arm")
        require(isinstance(preferred_support_counts, (list, tuple))
                and len(preferred_support_counts) == slots
                and all(type(n) is int and 2 <= n <= 6 for n in preferred_support_counts),
                "preferred_support_counts_require_one_2_to_6_integer_per_slot")
        preferences = {qid: count for row in sources for qid, count in zip(
            slot_focus.candidate_ids(row['root_id'], comparison_id, 'multi_select', slots),
            preferred_support_counts)}
    slot_focus.validate_map(focus_map, focus_ids, [turn for row in sources for turn in row['source']['turns']])
    feedback_lock = read(legacy.FEEDBACK / "source_lock.json")
    require(feedback_lock.get("status") == "EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY", "feedback_round_not_frozen")
    require(feedback_lock.get("merge_current_generation_feedback") is False, "current_feedback_not_allowed")
    require(feedback_lock["model_identity_source"]["sha256"] == v15.evaluator.TRAINED_AUDIT_SHA256,
            "feedback_and_solver_checkpoint_differ")
    provider, dependencies = _dependency_paths()
    audit_path, upstream_path = checked_expansion(merge_manifest)
    frozen = set(dependencies) | {merge_manifest, audit_path, upstream_path}
    frozen.update(collection_source.audit_dependencies(merge_manifest.parent))
    frozen.update(Path(binding["path"]).resolve() for binding in merge.get("files", []))
    for path in sorted(legacy.FEEDBACK.rglob("*")):
        if path.is_file():
            target = run / "feedback" / path.relative_to(legacy.FEEDBACK)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            frozen.update((path, target))
    prior_paths = [path.resolve() for path in prior_pools]
    if DEFAULT_PRIOR.resolve() not in prior_paths:
        prior_paths.insert(0, DEFAULT_PRIOR.resolve())
    require(all(path.is_file() for path in prior_paths), "explicit_prior_pool_missing")
    source_rows = []
    for row in sources:
        root = row["root_id"]
        target = run / "sources" / (root + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        write(target, row["source"])
        prior, paths = v15.prior_candidates(root)
        for path in prior_paths:
            prior += _prior_pool_rows(path, root)
        write(run / "prior" / (root + ".json"), prior)
        legacy.feedback.build_feedback_suffix(run / "feedback", root, "writer", TARGET)
        frozen.update([row["path"], target, run / "prior" / (root + ".json"), *paths, *prior_paths])
        source_rows.append({"root_id": root, "source_path": str(target.resolve()),
            "source_sha256": base.digest(target.read_bytes()), "merge_source_path": str(row["path"]),
            "candidate_history_sha256": row["sha256"], "candidate_source_training_pool_eligible": False,
            "retained_candidate_scenes": row["retained_candidate_scenes"],
            "new_admitted_scenes": row["new_admitted_scenes"], "turn_count": len(row["source"]["turns"]),
            "merge_record": row["merge_record"], "qa_status": "READY_PROVISIONAL_CANDIDATE_HISTORY"})
    write(run / "provider_profile.json", provider)
    frozen.add(run / "provider_profile.json")
    write(run / "UPSTREAM_ARTIFACT_AUDIT.json", sources[0]["audit"])
    frozen.add(run / "UPSTREAM_ARTIFACT_AUDIT.json")
    write(run / "COMPARISON_CONFIG.json", {"prompt_arm": prompt_arm, "comparison_id": comparison_id,
          "comparison_contract": COMPARISON_CONTRACT})
    frozen.add(run / "COMPARISON_CONFIG.json")
    write(run / "SLOT_FOCUS.json", {"contract": slot_focus.CONTRACT, "slot_focus_by_candidate_id": focus_map})
    frozen.add(run / "SLOT_FOCUS.json")
    write(run / "COMPOSITION_PREFERENCES.json", {"contract": COMPOSITION_CONTRACT,
          "preferred_support_count_by_candidate_id": preferences})
    frozen.add(run / "COMPOSITION_PREFERENCES.json")
    manifest = {"version": VERSION, "prepared_at_utc": v15.now(),
        "mixed_collection_contract": MIXED_COLLECTION_CONTRACT,
        "prompt_arm": prompt_arm, "comparison_id": comparison_id,
        "comparison_contract": COMPARISON_CONTRACT,
        "slot_focus_contract": slot_focus.CONTRACT, "slot_focus_by_candidate_id": focus_map,
        "composition_contract": COMPOSITION_CONTRACT,
        "preferred_support_count_by_candidate_id": preferences,
        "generation_task_tail_enabled": prompt_arm == "current",
        "merged_instruction_enabled": prompt_arm == "current",
        "adapted_common_sha256": base.digest(adapted_common().encode()),
        "merge_manifest_path": str(merge_manifest),
        "candidate_history_path": str(merge_manifest), "source_training_pool_eligible": False,
        "source_semantic_certified": False, "upstream_scene_promotion": False,
        "training_allowed": False, "source_admission_status": "PROVISIONAL_CANDIDATE_HISTORY",
        "upstream_admitted_scenes": 0, "merge_manifest_sha256": base.digest(merge_manifest.read_bytes()),
        "expansion_final_audit_path": str(audit_path), "expansion_final_audit_sha256": base.digest(audit_path.read_bytes()),
        "expansion_manifest_path": str(upstream_path), "expansion_manifest_sha256": base.digest(upstream_path.read_bytes()),
        "roots": [row["root_id"] for row in source_rows], "source_rows": source_rows,
        "excluded_merge_rows": excluded, "source_mode": "one_full_provisional_candidate_context_per_root",
        "branch_separated_question_pools": False, "generator_model": MODEL, "reviewer_model": None,
        "solver_model": TARGET, "question_types": list(GENERATION_TYPES), "candidate_slots_per_type_per_root": slots,
        "budget": generation_budget(len(sources), slots), "minimum_seconds_after_call_completion": INTERVAL,
        "automatic_transport_retries": 0, "mechanical_repair_attempts_max": 1,
        "semantic_multi_review": False, "ordering_independent_qwen_review": False,
        "ordering_review_status": "INACTIVE_MULTISELECT_ONLY_PROBE",
        "ordering_anchor_contract": "unique_exact_nonoverlapping_clause_spans_v1",
        "ordering_scope": "conversation_order_not_real_world_event_chronology",
        "paraphrases_allowed": True, "semantic_novelty_gate": False,
        "dedup_hard_scope": "exact_entire_six_statement_sets_only",
        "new_scene_evidence_required": False, "new_scene_evidence_priority": False, "codex_question_edits": False,
        "feedback_round": "previous_training_round_only", "current_solver_outputs_in_prompts": False,
        "full_policy_version": v15.POLICY, "selection_rule": "complete_valid_full_set and 0 < full_partial_mean < 1",
        "full_slots": {"multi_select": "6 random permutations x 3 full rollouts"},
        "ground_calls": 0, "training_calls": 0, "training_started": False,
        "merged_instruction": MERGED_INSTRUCTION,
        "prompt_repair_contract": prompt_rules.CONTRACT, "prompt_layout_contract": prompt_layout.CONTRACT,
        "files": base.bindings(frozen)}
    write(run / "MANIFEST.json", manifest)
    verify(run)
    return {key: manifest[key] for key in ("version", "roots", "source_rows", "excluded_merge_rows", "budget")}


def verify(run):
    manifest = read(run / "MANIFEST.json")
    require(manifest.get("mixed_collection_contract") == MIXED_COLLECTION_CONTRACT
            and manifest.get("question_types") == list(GENERATION_TYPES)
            and manifest.get("ordering_independent_qwen_review") is False
            and manifest.get("ordering_review_status") == "INACTIVE_MULTISELECT_ONLY_PROBE"
            and manifest.get("reviewer_model") is None
            and manifest.get("full_slots") == {"multi_select": "6 random permutations x 3 full rollouts"},
            "multiselect_probe_contract_changed")
    require(manifest["version"] == VERSION and manifest["merged_instruction"] == MERGED_INSTRUCTION,
            "generation_version_or_prompt_changed")
    require(manifest.get("prompt_arm") in PROMPT_ARMS
            and manifest.get("comparison_contract") == COMPARISON_CONTRACT
            and isinstance(manifest.get("comparison_id"), str)
            and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", manifest["comparison_id"])
            and manifest.get("generation_task_tail_enabled") is (manifest["prompt_arm"] == "current")
            and manifest.get("merged_instruction_enabled") is (manifest["prompt_arm"] == "current")
            and manifest.get("adapted_common_sha256") == base.digest(adapted_common().encode()),
            "generation_comparison_contract_changed")
    require(manifest.get("prompt_repair_contract") == prompt_rules.CONTRACT,
            "generation_prompt_repair_contract_changed")
    require(manifest.get("prompt_layout_contract") == prompt_layout.CONTRACT,
            "generation_prompt_layout_contract_changed")
    require(manifest.get("slot_focus_contract") == slot_focus.CONTRACT
            and manifest.get("new_scene_evidence_priority") is False, "generation_slot_focus_contract_changed")
    require(manifest["paraphrases_allowed"] is True and manifest["semantic_multi_review"] is False,
            "question_policy_changed")
    require(manifest["ground_calls"] == manifest["training_calls"] == 0, "unrequested_stage")
    require(manifest["source_training_pool_eligible"] is False
            and manifest["source_semantic_certified"] is False
            and manifest["upstream_scene_promotion"] is False
            and manifest["training_allowed"] is False
            and manifest["source_admission_status"] == "PROVISIONAL_CANDIDATE_HISTORY"
            and manifest["upstream_admitted_scenes"] == 0, "candidate_status_promoted")
    require(manifest["budget"] == generation_budget(len(manifest["roots"]), manifest["candidate_slots_per_type_per_root"]), "budget_changed")
    base.verify_bindings(manifest["files"])
    composition_path = (run / "COMPOSITION_PREFERENCES.json").resolve()
    require(manifest.get("composition_contract") == COMPOSITION_CONTRACT
            and any(Path(entry["path"]).resolve() == composition_path for entry in manifest["files"])
            and read(composition_path) == {"contract": COMPOSITION_CONTRACT,
                "preferred_support_count_by_candidate_id": manifest.get("preferred_support_count_by_candidate_id")},
            "generation_composition_preference_binding_changed")
    focus_path = (run / "SLOT_FOCUS.json").resolve()
    require(any(Path(entry["path"]).resolve() == focus_path for entry in manifest["files"])
            and read(focus_path) == {"contract": slot_focus.CONTRACT,
                "slot_focus_by_candidate_id": manifest["slot_focus_by_candidate_id"]},
            "generation_slot_focus_binding_changed")
    config_path = (run / "COMPARISON_CONFIG.json").resolve()
    require(any(Path(entry["path"]).resolve() == config_path for entry in manifest["files"]),
            "generation_comparison_config_not_bound")
    require(read(config_path) == {key: manifest[key]
            for key in ("prompt_arm", "comparison_id", "comparison_contract")},
            "generation_comparison_config_changed")
    _, sources, _ = checked_sources(Path(manifest["candidate_history_path"]), manifest["roots"])
    require(len(sources) == len(manifest["source_rows"]), "candidate_source_count_changed")
    for source, row in zip(sources, manifest["source_rows"]):
        local = Path(row["source_path"])
        require(read(local) == source["source"] and base.digest(local.read_bytes()) == row["source_sha256"],
                "candidate_local_source_changed")
    focus_ids = [qid for root in manifest['roots'] for kind in GENERATION_TYPES
                 for qid in slot_focus.candidate_ids(root, manifest['comparison_id'], kind,
                     manifest['candidate_slots_per_type_per_root'])]
    slot_focus.validate_map(manifest['slot_focus_by_candidate_id'], focus_ids,
                           [turn for row in sources for turn in row['source']['turns']])
    preferences = manifest["preferred_support_count_by_candidate_id"]
    require(isinstance(preferences, dict) and (not preferences or (
        manifest["prompt_arm"] == "current" and set(preferences) == set(focus_ids)
        and all(type(n) is int and 2 <= n <= 6 for n in preferences.values()))),
        "generation_composition_preferences_invalid")
    return manifest


def current_question_shape_errors(draft, qtype, source):
    """Check only emitted relation counts and stem syntax; never change claims/Gold."""
    if qtype == 'multi_select':
        options = draft.get('options')
        if (isinstance(options, list) and len(options) == 6
                and all(isinstance(option, dict) and option.get('relation') in
                        ('supports', 'contradicts') for option in options)):
            count = sum(option['relation'] == 'supports' for option in options)
            if not 2 <= count <= 6:
                return ['current_multi_requires_2_to_6_raw_supports:' + str(count)]
        return []  # Existing schema validation handles malformed/unknown relations.
    stem = draft.get('question')
    if stem == prompt_rules.ORDERING_STEM:
        return []
    prefix = prompt_rules.ORDERING_STEM_PREFIX + ' within the '
    if not isinstance(stem, str) or not stem.startswith(prefix):
        return ['current_ordering_fixed_public_stem_required']
    match = re.fullmatch(r'(.+?) (scene|scenes)\.', stem[len(prefix):])
    if match:
        scenes = match.group(1).split(' and ')
        available = {turn['turn_ref'].rsplit(':', 1)[0] for turn in source['turns']}
        if (len(set(scenes)) == len(scenes) and all(scene in available for scene in scenes)
                and match.group(2) == ('scene' if len(scenes) == 1 else 'scenes')):
            return []
    return ['current_ordering_fixed_public_stem_or_exact_scene_scope_required']


def validate_draft(raw, ids, root, qtype, source, source_path, *, prompt_arm='current'):
    """Account slots independently; malformed siblings never discard valid questions."""
    require(prompt_arm in PROMPT_ARMS, 'generation_prompt_arm_invalid')
    accepted, rejected, abstentions = [], [], []
    try:
        obj = json.loads(raw, object_pairs_hook=legacy._json_unique_pairs)
        base.require(isinstance(obj, dict) and obj.get('root_id') == root
            and isinstance(obj.get('questions'), list) and isinstance(obj.get('abstentions', []), list),
            'draft_schema_invalid')
    except (ValueError, TypeError) as error:
        return [], [{'candidate_id': qid, 'errors': [str(error)]} for qid in ids], [], str(error)
    rows, omitted = obj['questions'], obj.get('abstentions', [])
    unknown = [row for row in rows + omitted
               if not isinstance(row, dict) or row.get('candidate_id') not in ids]
    turns = {t['turn_ref']: t for t in source['turns']}
    for qid in ids:
        matches = [r for r in rows if isinstance(r, dict) and r.get('candidate_id') == qid]
        absent = [r for r in omitted if isinstance(r, dict) and r.get('candidate_id') == qid]
        if (matches and absent) or len(matches) > 1:
            rejected.append({'candidate_id': qid, 'errors': ['conflicting_or_duplicate_slot'], 'raw_slot_rows': matches + absent})
            continue
        if not matches:
            if absent:
                abstentions.append({**absent[0], 'format_diagnostics':
                    {'duplicate_abstention_count': len(absent) - 1, 'raw_abstention_rows': absent}})
            else:
                rejected.append({'candidate_id': qid, 'errors': ['candidate_slot_missing']})
            continue
        draft = matches[0]
        try:
            if qtype == 'multi_select':
                normalized, changes = reference_compat.normalize_question(draft, root, source)
                errors, candidate = legacy.multi.validate_candidate(normalized, qid, turns, root, source_path, base.digest(source_path.read_bytes()))
                if candidate is not None and changes:
                    candidate['raw_model_candidate'] = deepcopy(draft)
                    candidate['reference_normalizations'] = changes
            else:
                errors, candidate = ordering_gate.validate_generated_ordering(draft, qid, root, source, source_path)
            if prompt_arm == 'current':
                errors += current_question_shape_errors(draft, qtype, source)
        except (ValueError, TypeError, KeyError) as error:
            errors, candidate = ['slot_validation_error:' + str(error)], None
        if errors:
            rejected.append({'candidate_id': qid, 'errors': errors, 'draft': draft})
        else:
            candidate['generation_format_diagnostics'] = {
                'unknown_or_unassigned_rows': deepcopy(unknown),
                'extra_top_level_fields': sorted(set(obj) - {'root_id', 'questions', 'abstentions'}),
                'slot_independently_accounted': True}
            accepted.append(candidate)
    return accepted, rejected, abstentions, None


def _generate_locked(run):
    manifest = verify(run)
    require(not (run / "STARTED.json").exists(), "generation_already_started_no_replay")
    from qa_transport import Caller
    write(run / "STARTED.json", {"started_at_utc": v15.now(), "http_max": manifest["budget"]["http_attempts_max"]})
    caller = Caller(run)
    call = caller.call

    ready, mechanical, outcomes = [], [], []
    for source_row in manifest["source_rows"]:
        root, current = source_row["root_id"], []
        source_path = Path(source_row["source_path"])
        source, prior = read(source_path), read(run / "prior" / (root + ".json"))
        for qtype in GENERATION_TYPES:
            code = "multi" if qtype == "multi_select" else "order"
            slots = manifest["candidate_slots_per_type_per_root"]
            context = initial_context(run, root, qtype)
            ids = context["candidate_ids"]
            accepted, rejected, abstentions = [], [], []
            pending = ids[:]
            for attempt in (1, 2):
                payload, receipt = make_payload(run, root, qtype, context)
                raw, directory, metadata = call(f"{root.split('-')[-1]}-merged-{code}-generate-{attempt}", payload,
                    {"root_id": root, "source_mode": "full_provisional_candidate_root", "role": "question_generator",
                     "question_type": qtype, "candidate_ids": pending,
                     "source_sha256": source_row["source_sha256"], "feedback_receipt": receipt})
                if raw is None:
                    rejected = [{"candidate_id": qid, "errors": [read(directory / "ERROR.json")["error_type"]]}
                                for qid in pending]
                    break
                valid, rejected, abstained, envelope_error = validate_draft(
                    raw, pending, root, qtype, source, source_path, prompt_arm=manifest['prompt_arm'])
                evidence_key = "option_evidence" if qtype == "multi_select" else "item_evidence"
                for question in valid:
                    refs = [entry.get("turn_ref", "") for entry in question[evidence_key].values() if isinstance(entry, dict)]
                    question.update(generator_model=MODEL, generation_request_path=str(directory / "REQUEST.json"),
                        generation_response_path=str(directory / "RESPONSE.json"), raw_question_content_preserved=True,
                        expansion_route="merged_root", expansion_branch="MERGED",
                        source_context_version=VERSION, merged_source_manifest_path=manifest["merge_manifest_path"],
                        prompt_arm=manifest["prompt_arm"], comparison_id=manifest["comparison_id"],
                        source_training_pool_eligible=False, source_semantic_certified=False,
                        upstream_scene_promotion=False, training_allowed=False,
                        source_admission_status="PROVISIONAL_CANDIDATE_HISTORY", upstream_admitted_scenes=0,
                        new_scene_reference_diagnostic=any(ref.startswith("CANDIDATE:") for ref in refs),
                        evidence_routes_diagnostic=sorted({ref.split(":")[1] for ref in refs
                            if ref.startswith(("CANDIDATE:", "R02:")) and len(ref.split(":")) > 1}))
                accepted += valid
                abstentions += abstained
                write(directory / "VALIDATION.json", {"accepted_ids": [q["candidate_id"] for q in valid],
                    "rejected": rejected, "abstentions": abstained, "envelope_error": envelope_error,
                    "semantic_review": False, "new_scene_evidence_check": "diagnostic_only"})
                if not rejected:
                    break
                pending = [row["candidate_id"] for row in rejected]
                repair = {"instruction": "Repair only listed mechanical/source-binding errors; preserve accepted slots.",
                          "failures": legacy.repair_failures(rejected)}
                if envelope_error:
                    repair["envelope_error"] = envelope_error
                context = focus_context(manifest, {**context, "candidate_ids": pending, "mechanical_repair": repair,
                    "previous_candidates_dedup_only": prompt_prior(prior + legacy.compact_questions(accepted))})
            accepted, duplicates = v15.dedup.deduplicate(accepted, prior, current)
            rejected += duplicates
            current += accepted
            mechanical += accepted
            outcome = {"root_id": root, "question_type": qtype, "candidate_slots": slots,
                "mechanical_pass": len(accepted), "mechanical_rejected": rejected, "abstentions": abstentions,
                "review_rejected": [], "actual_generation_requests": attempt}
            reviewed = []
            if qtype == "ordering":
                for candidate in accepted:
                    payload, receipt = make_review_payload(run, root, candidate, source)
                    raw, directory, metadata = call(f"{root.split('-')[-1]}-merged-review-{candidate['candidate_id'].split('-')[-1]}",
                        payload, {"root_id": root, "role": "ordering_reviewer", "candidate_id": candidate["candidate_id"],
                        "feedback_receipt": receipt, "independent_request_without_prior_verdicts": True,
                        "source_sha256": source_row["source_sha256"]})
                    if raw is None:
                        outcome["review_rejected"].append({"candidate_id": candidate["candidate_id"], "reasons": ["review_request_failed"]})
                        continue
                    checked = ordering_gate.review_response(raw, [candidate], source,
                        directory / "RESPONSE.json", directory / "REQUEST.json", metadata)
                    write(directory / "REVIEW.json", checked)
                    reviewed += checked["accepted"]
                    outcome["review_rejected"] += checked["rejected"]
            else:
                reviewed = accepted
            ready += reviewed
            outcome["ready_for_solver"] = len(reviewed)
            outcomes.append(outcome)
            write(run / "outcomes" / f"{root}.{qtype}.json", outcome)
            print(json.dumps({"event": "merged_root_type_done", "root": root,
                              "question_type": qtype, "ready_for_solver": len(reviewed)}), flush=True)
    verify(run)
    for directory in sorted((run / "requests").glob("*")):
        payload, proof = read(directory / "REQUEST.json"), read(directory / "PROOF.json")
        verify_prompt_request(run, payload, proof)
    write(run / "MECHANICAL_POOL.json", {"questions": mechanical})
    write(run / "SOLVER_POOL.json", {"version": VERSION, "candidate_count": len(ready), "questions": ready,
        "provenance": {"requested_model": MODEL, "returned_model": MODEL, "codex_question_edits": False,
            "multi_select_semantic_review": False, "ordering_review_owner": None,
            "ordering_review_active": False,
            "source_mode": "one_full_provisional_candidate_context_per_root",
            "source_context_version": VERSION, "feedback_target_solver": TARGET,
            "prompt_arm": manifest["prompt_arm"], "comparison_id": manifest["comparison_id"],
            "comparison_contract": COMPARISON_CONTRACT},
        "ground_calls": 0, "training_started": False, "source_training_pool_eligible": False,
        "upstream_scene_promotion": False})
    preferred = manifest["preferred_support_count_by_candidate_id"]
    observed = {q["candidate_id"]: len(q["gold"]) for q in mechanical}
    report = {"version": VERSION, "http_attempts": len(list((run / "requests").glob("*/HTTP_ATTEMPT.json"))),
        "requests_prepared": caller.counter, "candidate_slots": manifest["budget"]["candidate_slots"],
        "mechanical_pass": len(mechanical), "ready_for_solver": len(ready), "roots": outcomes,
        "mechanical_answer_distribution": display_export.multi_answer_distribution(mechanical),
        "ready_answer_distribution": display_export.multi_answer_distribution(ready),
        "composition_diagnostics": {"diagnostic_only": True,
            "deviation_changes_admission_or_retry": False,
            "slots": [{"candidate_id": qid, "preferred_support_count": count,
                "actual_support_count": observed.get(qid),
                "matches_preference": (observed[qid] == count if qid in observed else None)}
                for qid, count in preferred.items()]},
        "feedback_in_every_request": True, "semantic_novelty_gate": False, "paraphrases_allowed": True,
        "prompt_arm": manifest["prompt_arm"], "comparison_id": manifest["comparison_id"],
        "comparison_contract": COMPARISON_CONTRACT,
        "no_current_solver_feedback_in_generation": True, "ground_calls": 0, "training_calls": 0}
    write(run / "REPORT.json", report)
    return report


def generate(run):
    from qa_transport import shared_lock
    with shared_lock():
        return _generate_locked(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "generate"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--candidate-history", "--merge-manifest", dest="merge_manifest", type=Path)
    parser.add_argument("--roots", nargs="+")
    parser.add_argument("--slots", type=int, default=2, help="candidate slots per type/root; not a selected quota")
    parser.add_argument("--prior-pool", action="append", type=Path, default=[])
    parser.add_argument("--prompt-arm", choices=PROMPT_ARMS, default="current")
    parser.add_argument("--comparison-id", help="probe ID used in candidate IDs; compatibility option name")
    parser.add_argument("--preferred-support-counts", nargs="+", type=int,
                        help="private soft 2–6 preference per slot; evidence determines actual Gold")
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.is_relative_to(HERE / "runs"), "run_must_be_in_v52_qa_runs")
    if args.action == "prepare":
        require(args.merge_manifest is not None, "prepare_requires_candidate_history")
        result = prepare(run, args.merge_manifest.resolve(), roots=args.roots, slots=args.slots,
                         prior_pools=args.prior_pool, prompt_arm=args.prompt_arm, comparison_id=args.comparison_id,
                         preferred_support_counts=args.preferred_support_counts)
    elif args.action == "verify":
        manifest = verify(run)
        result = {"status": "PASS", "roots": manifest["roots"], "budget": manifest["budget"]}
    else:
        result = generate(run)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
