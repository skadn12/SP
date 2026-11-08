#!/usr/bin/env python3
"""Read-only metadata/hash audit. Does not invoke Solver or open feedback content."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

P = Path(__file__).resolve().parents[1]
W = P.parent
REPO = W.parents[1]
FIRST = ["cc-train-episode-103866", "cc-train-episode-104814"]
CARD = P / "packets/B/ORDERING_FEEDBACK_BOUNDED.json"
inputs = {}
checks = []


def digest(path, mode="metadata_or_binding"):
    path = Path(path)
    data = path.read_bytes()
    value = hashlib.sha256(data).hexdigest()
    inputs[str(path)] = {
        "path": str(path), "sha256": value, "bytes": len(data),
        "observed_mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc).isoformat(),
        "read_mode": mode,
    }
    return value


def load(path):
    path = Path(path)
    if path == CARD:
        raise RuntimeError("Feedback-card content is outside this audit")
    digest(path)
    return json.loads(path.read_text())


def binding(path, mode="hash_only"):
    path = Path(path)
    return {"path": str(path), "sha256": digest(path, mode),
            "bytes": path.stat().st_size}


def verify(rec):
    path = Path(rec["path"])
    if not path.is_file():
        return {"path": str(path), "error": "MISSING"}
    actual = digest(path, "opaque_hash_only" if path == CARD else "hash_only")
    if actual != rec["sha256"]:
        return {"path": str(path), "error": "SHA256_MISMATCH",
                "expected": rec["sha256"], "actual": actual}
    if "bytes" in rec and rec["bytes"] != path.stat().st_size:
        return {"path": str(path), "error": "BYTE_COUNT_MISMATCH"}
    return None


def verify_many(records):
    return [error for rec in records if (error := verify(rec)) is not None]


def add(name, ok=None, **details):
    checks.append({"check": name,
                   "status": "UNKNOWN" if ok is None else ("PASS" if ok else "FAIL"),
                   **details})


def at(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def first_time(obj, keys):
    return next((obj[k] for k in keys if obj.get(k)), None)


def walk(obj, trail=""):
    if isinstance(obj, dict):
        yield trail, obj
        for key, value in obj.items():
            yield from walk(value, trail + "/" + key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from walk(value, trail + "/" + str(i))


freeze = load(P / "EXECUTION_FREEZE.json")
amendment = load(P / "DAY3_SCOPE_AMENDMENT.json")
slot_plan = load(W / "day1/SLOT_PLAN.json")
selection = load(W / "day1/evaluation/ROOT_SELECTION.json")
admission = load(P / "inventory/PRIORITY_SOURCE_ADMISSION.json")
inventory_freeze = load(P / "inventory/SOURCE_INVENTORY_FREEZE.json")
language = load(P / "LANGUAGE_DEVIATION_CORRECTION_V2.json")
wave1 = selection["wave1"]["root_ids"]
stage1 = [s for s in slot_plan["slots"] if s["stage"] == 1]
slots = freeze["slots"]
invariant_fields = ["slot_id", "candidate_slot_id", "stage", "root_id",
                    "question_type", "local_slot", "target_gold_count",
                    "options_or_items", "arm"]
project = lambda seq: [{k: s[k] for k in invariant_fields} for s in seq]
arm_counts = {}
for arm in ["A", "B"]:
    arm_slots = [s for s in slots if s["arm"] == arm]
    mc = [s for s in arm_slots if s["question_type"] == "multi_select"]
    arm_counts[arm] = {
        "slots": len(arm_slots), "multi_select": len(mc),
        "ordering": sum(s["question_type"] == "ordering" for s in arm_slots),
        "MC_gold_quota": dict(sorted(Counter(str(s["target_gold_count"])
                                             for s in mc).items())),
    }
add("inherited_exact_wave1_roots_slots_and_gold_quotas",
    len(wave1) == 20 and freeze["roots"] == wave1 and len(slots) == 240
    and len({s["candidate_slot_id"] for s in slots}) == 240
    and project(slots) == project(stage1)
    and all(c == {"slots": 120, "multi_select": 60, "ordering": 60,
                  "MC_gold_quota": {"2": 15, "3": 15, "4": 15, "5": 15}}
            for c in arm_counts.values()),
    wave1_roots=wave1, counts=arm_counts,
    retained_fields=invariant_fields,
    limitation="Planned slots/Gold quotas are not accepted-question counts.")

frozen_bindings = [
    {"path": str(P / "DAY3_SCOPE_AMENDMENT.json"),
     "sha256": freeze["amendment_sha256"]},
    {"path": str(P / "design/PUBLIC_DESIGN_LIBRARY.json"),
     "sha256": freeze["public_design_sha256"]},
    {"path": str(P / "inventory/PRIORITY_FULL_SOURCE_REVIEW.json"),
     "sha256": freeze["source_review_sha256"]},
    {"path": str(W / "day1/SLOT_PLAN.json"),
     "sha256": freeze["inherited_slots_sha256"]},
]
errors = verify_many(frozen_bindings)
add("execution_freeze_bound_artifacts_unchanged", not errors,
    errors=errors, bindings=frozen_bindings)
add("scope_amendment_narrows_target_to_ordering_cross_scene",
    amendment["new_stage1_B_requested_targeted_slots"] == 60
    and amendment["new_stage1_B_common_only_slots"] == 60
    and "cross-scene" in amendment["targeted_eligibility"]
    and "not a reproduced weakness" in amendment["targeted_eligibility"],
    targeted_slots=60, common_only_slots=60,
    interpretation=amendment["targeted_eligibility"],
    MC_exposure_limit=amendment["MC_exposure_limit"])

source_records = inventory_freeze["sources"]
errors = verify_many(source_records)
priority_errors = verify_many([inventory_freeze["inventory"],
                              inventory_freeze["priority_admission"]])
packaged_errors = verify_many(
    x["packaged_source_binding"] for x in admission["sources"])
first_hashes = {r: digest(P / "sources" / (r + ".json"), "hash_only")
                for r in FIRST}
add("first_two_roots_and_all20_source_bindings_unchanged",
    freeze["first_batch_roots"] == wave1[:2] == FIRST
    and len(source_records) == 20 and not errors and not priority_errors
    and not packaged_errors,
    first_batch_roots=FIRST, first_two_source_sha256=first_hashes,
    all20_source_mismatches=errors,
    inventory_mismatches=priority_errors,
    packaged_priority_source_mismatches=packaged_errors)

groups = {}
for arm in ["A", "B"]:
    for root in FIRST:
        directory = P / "candidates" / arm / root
        read_path = directory / "READ_MANIFEST.json"
        plan_path = directory / "PLAN.json"
        read = load(read_path)
        plan = load(plan_path)
        if arm == "A" and root == FIRST[1]:
            private_paths = sorted(directory.glob("*.private.json"))
        else:
            private_paths = sorted((directory / "private").glob("*.json"))
        private = [(x, load(x)) for x in private_paths]
        groups[(arm, root)] = {
            "directory": directory, "read_path": read_path, "read": read,
            "plan_path": plan_path, "plan": plan, "private": private,
        }

chronology = []
read_binding_errors = []
for (arm, root), g in groups.items():
    read, plan = g["read"], g["plan"]
    read_time = first_time(read, ["actual_read_completed_at_utc",
                                 "completed_full_read_at_utc", "created_at_utc"])
    plan_time = first_time(plan, ["frozen_before_drafts_at_utc", "created_at_utc"])
    initial_times = [d["created_at_utc"] for _, d in g["private"]]
    read_bindings = read.get("actual_reads", read.get(
        "actual_read_files", read.get("read_bindings", [])))
    errors = verify_many(read_bindings)
    read_binding_errors.extend(errors)
    plan_read_hash_ok = plan["read_manifest_sha256"] == digest(g["read_path"])
    private_plan_hash_ok = all(d["plan_sha256"] == digest(g["plan_path"])
                               for _, d in g["private"])
    private_read_hash_ok = all(
        d.get("read_manifest_sha256", digest(g["read_path"])) == digest(g["read_path"])
        for _, d in g["private"])
    source_hash_ok = all(
        d.get("source_sha256", d.get("source_binding", {}).get("sha256"))
        == first_hashes[root] for _, d in g["private"])
    times_ok = (at(admission["created_at_utc"]) < at(freeze["created_at_utc"])
                <= at(amendment["created_at_utc"]) < at(read_time) < at(plan_time)
                < min(map(at, initial_times)))
    chronology.append({
        "arm": arm, "root_id": root, "source_admitted_at": admission["created_at_utc"],
        "freeze_at": freeze["created_at_utc"], "amendment_at": amendment["created_at_utc"],
        "read_at": read_time, "plan_at": plan_time,
        "first_initial_at": min(initial_times), "last_initial_at": max(initial_times),
        "initial_private_count": len(initial_times), "recorded_chronology_ok": times_ok,
        "plan_binds_current_read_manifest": plan_read_hash_ok,
        "all_initials_bind_current_plan": private_plan_hash_ok,
        "all_available_initial_read_hashes_match": private_read_hash_ok,
        "all_initials_bind_same_complete_source": source_hash_ok,
        "read_binding_mismatches": errors,
    })
add("recorded_source_admission_freeze_read_plan_initial_order",
    all(c["recorded_chronology_ok"] and c["initial_private_count"] == 5
        and c["plan_binds_current_read_manifest"]
        and c["all_initials_bind_current_plan"]
        and c["all_available_initial_read_hashes_match"]
        and c["all_initials_bind_same_complete_source"]
        and not c["read_binding_mismatches"] for c in chronology),
    groups=chronology, candidate_count_at_freeze=freeze["candidate_count_at_freeze"],
    source_review_existence="Full-source review has no own timestamp; its hash is already bound by the earlier execution freeze and author read manifests.")
add("externally_authenticated_creation_and_read_times", None,
    reason="Recorded timestamps and current filesystem mtimes are not signed provider logs. They establish recorded chronology, not externally authenticated creation/read times.")

A_reads = []
for root in FIRST:
    g = groups[("A", root)]
    read = g["read"]
    records = read.get("actual_reads", read.get("actual_read_files", []))
    paths = [r["path"] for r in records]
    forbidden = [x for x in paths if "/packets/B/" in x
                 or "/candidates/B/" in x or "FEEDBACK" in Path(x).name.upper()
                 or "memory" in Path(x).name.lower()]
    A_reads.append({"root_id": root, "feedback_exposed": read["feedback_exposed"],
                    "listed_actual_read_paths": paths, "forbidden_paths": forbidden,
                    "binding_errors": verify_many(records)})
add("A_actual_read_manifests_exclude_feedback",
    all(x["feedback_exposed"] is False and not x["forbidden_paths"]
        and not x["binding_errors"] for x in A_reads),
    manifests=A_reads,
    scope="This checks author declarations, listed actual inputs and their bindings.")
add("A_technical_feedback_blinding", None,
    reason="Authors shared the filesystem and context cannot be retrospectively proved absent; no technical sandbox blinding or independently blinded MC causal control is claimed.")

B_reads = []
lineages = []
for root in FIRST:
    g = groups[("B", root)]
    card_bindings = [r for r in g["read"]["read_bindings"]
                     if Path(r["path"]) == CARD]
    B_reads.append({"root_id": root,
                    "feedback_exposed": g["read"]["feedback_exposed"],
                    "bounded_card_bindings": card_bindings,
                    "card_binding_errors": verify_many(card_bindings)})
    for path, private in g["private"]:
        if path.stem.startswith("ordering"):
            meta = private["ordering"]["feedback_lineage"]
            used_for = meta.get("used_for", "")
            narrow = ("scene boundaries" in used_for
                      and "Within-scene profile is public exploration" in used_for
                      and "no diagnosed within-scene cause" in used_for)
            lineages.append({"path": str(path), "metadata": meta,
                             "scope_explicitly_narrow": narrow,
                             "read_manifest_matches": Path(meta["bounded_read_manifest"])
                             == g["read_path"]})
add("B_bounded_read_to_plan_to_initial_and_narrow_ordering_lineage",
    all(x["feedback_exposed"] is True and len(x["bounded_card_bindings"]) == 1
        and not x["card_binding_errors"] for x in B_reads)
    and len(lineages) == 6
    and all(x["scope_explicitly_narrow"] and x["read_manifest_matches"]
            for x in lineages)
    and all(c["recorded_chronology_ok"] for c in chronology if c["arm"] == "B"),
    bounded_reads=B_reads, ordering_lineages=lineages,
    bounded_card_content_opened=False,
    limitation="Lineage is a documented operational target, not evidence that Solver has this failure on the new items.")

original_records = []
completion = load(P / "candidates/A" / FIRST[0] / "AUTHOR_COMPLETION.json")
original_records.extend(completion["records"])
draft = load(P / "candidates/A" / FIRST[1] / "DRAFT_MANIFEST.json")
original_records.extend(draft["files"])
for root in FIRST:
    for private_path, _ in groups[("B", root)]["private"]:
        review = load(P / "qa/B" / root / private_path.stem / "INITIAL_REVIEW.json")
        original_records.extend([review["public_binding"], review["author_private_binding"]])
errors = verify_many(original_records)
original_public_private = [
    x for x in original_records
    if ("/private/" in x["path"] or "/public/" in x["path"]
        or x["path"].endswith((".private.json", ".public.json")))
]
add("original_public_and_private_candidates_unchanged_against_earlier_bindings",
    len(original_public_private) == 40 and not errors,
    expected_original_public_private_files=40,
    checked_original_public_private_files=len(original_public_private),
    all_bound_artifacts_checked=len(original_records), errors=errors,
    records=original_public_private,
    limitation="Checks equality to preserved author-completion/draft/QA bindings; these are not externally timestamped seals.")

clarifications = []
for (arm, root), g in groups.items():
    for path in sorted((g["directory"] / "clarification_records").glob("*.json")):
        if path.name == "MANIFEST.json":
            continue
        d = load(path)
        if arm == "B":
            recs = [
                {"path": d["request_review_path"], "sha256": d["request_review_sha256"]},
                {"path": d["initial_path"], "sha256": d["initial_sha256"]},
                {"path": d["clarified_path"], "sha256": d["clarified_sha256"]},
                {"path": d["unchanged_private_path"], "sha256": d["unchanged_private_sha256"]},
            ]
            number = d["clarification_number"]
            post = P / "qa/B" / root / path.stem / "AFTER_CLARIFICATION_REVIEW.json"
        else:
            recs = [{"path": d["request_path"], "sha256": d["request_sha256"]},
                    d["original_public"], d["clarified_public"], d["private_original"]]
            number = d["wording_clarifications_used"]
            post = P / "qa/A_104814/POST_CLARIFICATION_REVIEW.json"
        request = load(recs[0]["path"])
        post_data = load(post) if post.exists() else {}
        post_time = first_time(post_data, ["created_at_utc", "reviewed_at_utc"])
        clarifications.append({
            "record": str(path), "arm": arm, "root_id": root,
            "candidate_id": d["candidate_id"], "single_clarification": number == 1,
            "request_at": request["created_at_utc"],
            "clarified_at": d["created_at_utc"],
            "request_precedes_clarification": at(request["created_at_utc"]) < at(d["created_at_utc"]),
            "post_review": str(post), "post_review_at": post_time,
            "post_review_after_clarification": bool(post_time and at(post_time) > at(d["created_at_utc"])),
            "binding_errors": verify_many(recs),
            "actual_cap": d.get("actual_cap"),
            "actual_usage": d.get("actual_usage", d.get("actual_usage_tokens")),
        })
counts = Counter((x["arm"], x["root_id"]) for x in clarifications)
add("exactly14_single_review_requested_clarifications",
    len(clarifications) == 14
    and len({x["candidate_id"] for x in clarifications}) == 14
    and counts == Counter({("A", FIRST[1]): 5, ("B", FIRST[0]): 5, ("B", FIRST[1]): 4})
    and all(x["single_clarification"] and x["request_precedes_clarification"]
            and x["post_review_after_clarification"] and not x["binding_errors"]
            for x in clarifications),
    count=len(clarifications),
    by_group={a + "/" + r: n for (a, r), n in sorted(counts.items())},
    records=clarifications,
    scope="Hash/private preservation and existing post-QA records verified; no new semantic retranslation judgment performed.")

usage = []
for (arm, root), g in groups.items():
    for path, private in g["private"]:
        budget = private.get("budget", private.get("generation", {}))
        usage.append({
            "path": str(path), "requested": budget.get(
                "requested_output_tokens_per_question", budget.get("requested_output_tokens")),
            "actual_cap": budget.get("actual_cap"),
            "actual_usage": budget.get("actual_usage", budget.get("actual_usage_tokens")),
        })
add("actual_provider_cap_and_usage_are_unknown_not_equal_compute",
    len(usage) == 20
    and all(x["requested"] == 3072 and x["actual_cap"] is None
            and x["actual_usage"] is None for x in usage)
    and all(x["actual_cap"] is None and x["actual_usage"] is None
            for x in clarifications)
    and language["process_pilot_not_equal_realized_compute_causal_comparison"] is True,
    recorded_status="UNKNOWN", initial_records=usage,
    claim_limit="Requested 3072 is not measured or enforced provider compute.")

latest_initial = max(at(d["created_at_utc"])
                     for g in groups.values() for _, d in g["private"])
latest_clarification = max(at(x["clarified_at"]) for x in clarifications)
add("language_deviation_explicitly_corrected_after_initials",
    language["language_was_not_explicitly_frozen_before_initial_drafts"] is True
    and language["wording_clarifications"] == 14
    and at(language["created_at_utc"]) > latest_initial
    and at(language["created_at_utc"]) > latest_clarification,
    authoritative_correction=binding(P / "LANGUAGE_DEVIATION_CORRECTION_V2.json"),
    created_at_utc=language["created_at_utc"],
    recorded_initial_languages=language["actual_initial_languages"],
    initial_language_independently_reclassified=False,
    interpretation="English was not originally frozen. A103866 English5; A104814 Chinese5; both B Chinese10. Later English alignment is a disclosed protocol deviation/correction, not preregistration.")

day2_path = W / "day2_pilot_v1/FINAL_MANIFEST.json"
day2 = load(day2_path)
day2_errors = verify_many(day2["files"])
add("day2_sealed_manifest_files_unchanged",
    day2["status"] == "SEALED_D2_PILOT" and day2["file_count"] == 946
    and len(day2["files"]) == 946 and not day2_errors,
    manifest=binding(day2_path), file_count=len(day2["files"]),
    mismatches=day2_errors, content_access="Only FINAL_MANIFEST parsed; listed files read as opaque bytes for hashing.",
    limitation="All files match the currently preserved manifest; the manifest has no independent earlier self-hash or external signature.")

qa_source_records = []
qa_source_errors = []
for path in sorted((P / "qa").rglob("*.json")):
    # Existing QA metadata is allowed; no new semantic judgments are made.
    d = load(path)
    for trail, value in walk(d):
        if "path" not in value or "sha256" not in value:
            continue
        bound = Path(str(value["path"]))
        if bound.parent == P / "sources":
            error = verify(value)
            item = {"qa_record": str(path), "field": trail, "source": str(bound),
                    "recorded_sha256": value["sha256"]}
            qa_source_records.append(item)
            if error:
                qa_source_errors.append(item | error)
add("existing_QA_source_hash_bindings_consistent",
    bool(qa_source_records) and not qa_source_errors,
    source_binding_count=len(qa_source_records),
    unique_source_paths=sorted({x["source"] for x in qa_source_records}),
    mismatches=qa_source_errors,
    scope="Checks explicit path+sha256 source bindings in existing QA records only, not semantic QA conclusions.")

source = load(P / "sources" / (FIRST[0] + ".json"))
scene_counts = Counter(t.get("scene_ref") for t in source["turns"])
actual = {"parent": scene_counts[None],
          "R01": sum(n for k, n in scene_counts.items() if k and k.startswith("R01:")),
          "R02": sum(n for k, n in scene_counts.items() if k and k.startswith("R02:")),
          "total": len(source["turns"])}
content_review_path = P / "qa/A_103866/CONTENT_REVIEW.json"
content_review = load(content_review_path)
note = content_review.get("provenance_note", {})
add("A103866_subgroup_count_metadata_correct_in_original", False,
    recorded={"parent": 70, "R01": 45, "R02": 82, "total": 197},
    actual=actual,
    interpretation="Preserved original read manifest has a subgroup-count typo. Total197, full0:100/100:197 declared ranges and source hash remain correct; this does not establish incomplete reading.",
    additive_correction_present=bool(note),
    correction_evidence=binding(content_review_path),
    correction_field="/provenance_note", correction=note)

published_record_dirs = sorted(
    (str(d.relative_to(P)), d.name) for arm in ["A", "B"]
    for d in (P / "candidates" / arm).iterdir() if d.is_dir())
add("only_frozen_first_batch_has_authored_candidate_directories",
    len(published_record_dirs) == 4 and all(r in FIRST for _, r in published_record_dirs),
    directories=[x for x, _ in published_record_dirs],
    scope="24 frozen first-batch slots, 20 initial questions and4 material shortfalls; remaining18 roots are not yet full production.")
add("new_Solver_calls_and_training_or_feedback_benefit", None,
    solver_calls_by_this_audit=0,
    evidence="No Solver invocation in this script. This audit does not assess accuracy, training benefit, or feedback effectiveness.")
add("external_independent_blind_semantic_validation", None,
    reason="Auditor previously authored source inventory and performed B semantic QA/record normalization. This process check must not be described as external independent blind confirmation of those semantic judgments.")

out = {
    "schema_version": "day3_process_lineage_check_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "auditor": "/root/day3_source_inventory",
    "status": "PROCESS_LINEAGE_VERIFIED_WITH_DISCLOSED_METADATA_DEVIATION_AND_UNKNOWN_LIMITS",
    "scope": "Read-only file-hash, recorded-timestamp, author-read-manifest and existing-QA lineage audit; no feedback-card content or Solver access.",
    "solver_calls": 0,
    "feedback_card_content_opened": False,
    "summary": dict(Counter(x["status"] for x in checks)),
    "checks": checks,
    "remaining_gaps": [
        {"id": "MANIFEST_SUBGROUP_COUNT_TYPO", "status": "DISCLOSED_ADDENDUM_EXISTS",
         "detail": "A103866 original70/45 must not be presented as correct;67/48 correction is already embedded in CONTENT_REVIEW.provenance_note. Preserve originals."},
        {"id": "LANGUAGE_NOT_ORIGINALLY_FROZEN", "status": "DISCLOSED_CORRECTED_AFTER_DRAFTS",
         "detail": "14 allowed single wording clarifications aligned language post hoc. This batch cannot support a claim of preregistered common initial language."},
        {"id": "REALIZED_COMPUTE", "status": "UNKNOWN",
         "detail": "Interactive provider revision, enforced cap, request IDs and actual token usage are not available; do not claim equal realized compute."},
        {"id": "TECHNICAL_BLINDING", "status": "UNKNOWN",
         "detail": "Shared filesystem/input declarations provide process separation, not provable technical blindness; B MC author was feedback exposed."},
        {"id": "TIMESTAMP_AUTHENTICITY", "status": "UNKNOWN",
         "detail": "Recorded chronology and hashes are internally consistent but lack externally authenticated timestamps."},
        {"id": "CROSS_POOL_FACT_FAMILY_REVIEW", "status": "UNKNOWN",
         "detail": "This audit does not perform cross-pool semantic/fact-family deduplication; preserve candidate HOLD until the dedicated review is recorded."},
        {"id": "SEMANTIC_REPLICATION_AND_BENEFIT", "status": "UNKNOWN",
         "detail": "Existing QA is lineage evidence. This auditor's own prior B QA is not independent external replication, and neither it nor this audit proves feedback/training benefit."},
        {"id": "FULL_STAGE1_PRODUCTION", "status": "NOT_COMPLETE",
         "detail": "Only the fixed first2 roots have initial drafts.240 planned slots do not mean240 produced/admitted questions."},
    ],
    "concrete_followups": [
        "In the next packet version, explicitly freeze common English before any initial draft; keep this first-batch language deviation visible.",
        "Generate author read-manifest turn counts from source scene metadata automatically; retain an additive correction for the current original.",
        "Complete the dedicated cross-pool fact-family/duplicate review before final candidate admission and Solver evaluation.",
        "Continue only the original remaining18 wave1 roots after their scoped full-source admission, retaining all unavailable relation slots as shortfalls.",
        "For compute-controlled comparison, use an endpoint that records model revision, request IDs, actual caps and usage; otherwise retain UNKNOWN.",
        "If external semantic replication is required, use a reviewer who did not author these records or this batch's prior QA, with reconstruction frozen before private disclosure.",
        "Evaluate feedback benefit only under the frozen A/B evaluation design; do not turn lineage compliance into a model-performance conclusion.",
    ],
    "input_file_count": len(inputs),
    "input_bindings": sorted(inputs.values(), key=lambda x: x["path"]),
    "mutation_policy": "Only this new audit script and its new output are written; original candidates, read manifests, QA and Day2 artifacts remain untouched.",
}
output = P / "audit/PROCESS_LINEAGE_CHECK.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(output), "summary": out["summary"],
                  "failed": [x["check"] for x in checks if x["status"] == "FAIL"],
                  "input_file_count": len(inputs)}, ensure_ascii=False))
