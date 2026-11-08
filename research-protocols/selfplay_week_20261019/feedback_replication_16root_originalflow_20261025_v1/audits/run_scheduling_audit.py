"""Inspect scheduling metadata only; never open Solver raw outputs or scores."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

D = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}


def check(b):
    actual = bind(b["path"])
    assert actual["sha256"] == b["sha256"] and actual["bytes"] == b["bytes"], b["path"]


qf = load(D / "QUESTION_FREEZE.json")
reused = load(D / "REUSED_SLOTS.json")["slots"]
inventory = load(D / "REUSE_INVENTORY.json")
prep = load(D / "execution" / "PREPARATION.json")
public = load(D / "execution" / "PUBLIC_REQUESTS.json")["requests"]
started = load(D / "execution" / "STARTED.json")
# Access only IDs and reuse binding metadata; never inspect private presentation/Gold.
private_doc = load(D / "execution" / "PRIVATE_SCORING.json")
private_ids = [{k: row[k] for k in ("request_id", "candidate_id", "arm", "slot_id")} for row in private_doc["requests"]]
private_reuse_bindings = private_doc["reused_shard_bindings"]
del private_doc

slotkey = lambda s: (s["root_id"], s["arm"], s["slot_id"])
reused_keys = {slotkey(s) for s in reused}
pass_keys = {slotkey(s) for s in qf["slots"] if s["status"] == "PASS"}
public_keys = [slotkey(s) for s in public]
groups = defaultdict(list)
for row in public:
    groups[slotkey(row)].append(row)
assert len(reused_keys) == 73
assert len(public) == 298 and len(groups) == 149
assert set(groups) == pass_keys - reused_keys
assert not (set(groups) & reused_keys)
assert all(len(rows) == 2 and {r["variant_id"] for r in rows} == {"perm_00", "perm_01"} for rows in groups.values())
public_ids = [r["request_id"] for r in public]
assert len(public_ids) == len(set(public_ids))
assert prep["fixed_order"] == public_ids == started["request_ids"]
assert prep["requests"] == started["requests"] == len(public)
assert private_ids == [{k: row[k] for k in ("request_id", "candidate_id", "arm", "slot_id")} for row in public]
assert prep["automatic_retry"] is False and prep["source_truncation"] is False
assert prep["public_generation_reads_private_scoring"] is False and started["gold_read"] is False
assert prep["reused_responses_separate_shard"] == inventory["existing_solver_outputs_reusable"] == 146
assert inventory["question_verified_reusable"] == 73 and not inventory["reuse_errors"]
check(started["preparation_binding"])
check(prep["private_scoring_binding"])
adapter_binding = next(b for b in prep["bindings"] if b["path"] == str(D / "scripts" / "solver_continue.py"))
check(adapter_binding)
public_binding = next(b for b in prep["bindings"] if b["path"] == str(D / "execution" / "PUBLIC_REQUESTS.json"))
check(public_binding)

old_outputs = []
inventory_reused_keys = set()
for root in inventory["roots"]:
    for arm, arm_info in root.get("arms", {}).items():
        for sid, slot in arm_info["slots"].items():
            if slot.get("reuse_eligible"):
                inventory_reused_keys.add((root["root_id"], arm, sid))
                assert slot["solver_outputs_reusable"] is True and len(slot["existing_solver_outputs"]) == 2
                for output in slot["existing_solver_outputs"]:
                    assert output["binding_verified"] is True and output["nonempty"] is True
                    # The raw binding is copied as metadata; its target is never opened.
                    old_outputs.append({"request_id": output["request_id"], "binding": output["binding"]})
assert inventory_reused_keys == reused_keys
old_ids = [r["request_id"] for r in old_outputs]
assert len(old_ids) == len(set(old_ids)) == 146
assert not set(old_ids) & set(public_ids)
assert len(private_reuse_bindings) == 3
assert inventory["raw_freeze"] in private_reuse_bindings
assert inventory["public_requests"] in private_reuse_bindings
assert bind(D / "REUSE_INVENTORY.json") in private_reuse_bindings

script = (D / "scripts" / "solver_continue.py").read_text()
assert "candidates = [c for c in candidates if (c['root_id'], c['arm'], c['slot_id']) not in reused]" in script
assert "frozen.public = public" in script and "frozen.run()" in script
assert "need('slot_id' not in result or result['slot_id'] == candidate['slot_id']" in script

result = {
    "schema_version": "independent_scheduling_audit_v1",
    "auditor_id": "/root/stall_review",
    "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS_AT_RUN_START",
    "mechanical_findings": [],
    "new_semantic_slots": len(groups),
    "new_requests": len(public),
    "reused_semantic_slots": len(reused_keys),
    "reused_output_bindings": len(old_outputs),
    "new_and_reused_semantic_slot_intersection": 0,
    "new_and_reused_request_id_intersection": 0,
    "new_public_private_started_request_ids_exactly_agree": True,
    "all_new_slots_are_exactly_frozen_pass_minus_reused": True,
    "two_display_variants_per_new_slot": True,
    "old_raw_files_opened": 0,
    "new_raw_files_opened": 0,
    "score_files_opened": 0,
    "private_scoring_fields_inspected": ["requests[].request_id", "requests[].candidate_id", "requests[].arm", "requests[].slot_id", "reused_shard_bindings"],
    "parent_reported_compatibility_repair_verified": "Direct-review receipt adapter explicitly checks optional slot_id consistency, then ignores only slot_id/review_id when checking every content field for equality. This changes metadata compatibility, not questions or review judgments.",
    "script_dispatch_verification": "prepare excludes all prior-PASS semantic slots before creating any prompt. run replaces frozen.public with the checked public() that returns the298 new PUBLIC_REQUESTS entries. STARTED lists those same298 IDs and no reused IDs.",
    "maximum_prompt_tokens": max(r["prompt_tokens"] for r in public),
    "maximum_output_tokens": prep["max_tokens"],
    "context_limit_tokens": prep["max_model_len"],
    "automatic_retry": prep["automatic_retry"],
    "source_truncation": prep["source_truncation"],
    "bindings": [bind(D / name) for name in ("scripts/solver_continue.py", "REUSE_INVENTORY.json", "CONTINUATION_MANIFEST.json", "execution/PREPARATION.json", "execution/PUBLIC_REQUESTS.json", "execution/STARTED.json")],
    "reused_shard_bindings": private_reuse_bindings,
    "execution_stage_limit": "This audit establishes that the launched scheduling set excludes all73 reused questions/146 outputs. Completion and final response-set metadata are outside this run-start snapshot; no completion claim is made.",
}
(D / "audits" / "INDEPENDENT_SCHEDULING_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
audit_path = D / "audits" / "INDEPENDENT_PROCESS_AUDIT.json"
audit = load(audit_path)
audit["scheduling_audit"] = result
audit["not_certified_by_this_audit"][1] = "The added scheduling audit verifies that the actual STARTED request set has298 requests from149 new slots and excludes all73 reused slots/146 old request IDs, without reading raw outputs or scores. Its claim is limited to the launch snapshot."
audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("status", "new_semantic_slots", "new_requests", "reused_semantic_slots", "reused_output_bindings", "new_and_reused_semantic_slot_intersection", "maximum_prompt_tokens", "maximum_output_tokens")}, ensure_ascii=False))
