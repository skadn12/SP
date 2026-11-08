"""Create the prospective day-3 execution freeze once; never overwrite it."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from collections import Counter

D = Path(__file__).resolve().parent
W = D.parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
now = datetime.now(timezone.utc).isoformat()
slots = json.loads((W/"day1/SLOT_PLAN.json").read_text())["slots"]
slots = [s for s in slots if s["stage"] == 1]
assert len(slots) == 240
roots = sorted({s["root_id"] for s in slots})
first = roots[:2]
review = D/"inventory/PRIORITY_FULL_SOURCE_REVIEW.json"
assert [r["root_id"] for r in json.loads(review.read_text())["roots"]] == first
library = D/"design/PUBLIC_DESIGN_LIBRARY.json"
mc = ["MC_BINDING", "MC_FINAL_STATE_REVISION", "MC_NEGATION_CONDITIONAL_JOINT"]
order = ["ORDER_3_2_1", "ORDER_2_2_2", "ORDER_4_2"]
assigned = []
for s in slots:
    row = dict(s)
    source = D/"sources"/(s["root_id"]+".json")
    row.update({
        "design_id": (mc if s["question_type"]=="multi_select" else order)[s["local_slot"]-1],
        "source_binding": {"path": str(source), "sha256": sha(source)},
        "source_admission_state": "PASS_SCOPED_WITH_FACT_EXCLUSIONS" if s["root_id"] in first else "PENDING_FULL_HISTORY_SEMANTIC_REVIEW",
        "generation_mode": "narrow_ordering_feedback" if s["arm"]=="B" and s["question_type"]=="ordering" else "free_diverse",
        "target_cause_id": "D2-ORDER-FIXED-CITATIONS" if s["arm"]=="B" and s["question_type"]=="ordering" else None,
        "question_generated": False,
    })
    assigned.append(row)
amendment = {
    "id": "DAY3_PROSPECTIVE_SCOPE_AMENDMENT_V1", "created_at_utc": now,
    "status": "FROZEN_BEFORE_ANY_DAY3_CANDIDATE",
    "original_protocol_preserved": True, "original_stage1_B_targeted": 84,
    "new_stage1_B_requested_targeted_slots": 60,
    "new_stage1_B_common_only_slots": 60,
    "justification": "Day2 admits one narrow cross-scene ordering operational relation; no formal MC target or within-scene diagnosis was admitted.",
    "interpretation": "New arm-level contrast: public diverse generation versus the same public design plus restricted ordering feedback. This is not the original 70 percent treatment and not a causal estimate of diagnosis for MC.",
    "targeted_eligibility": "Only ordering cross-scene relations may instantiate the card. New within-scene profiles are public structural exploration, not a reproduced weakness.",
    "maximum_targeted_coverage": "60/120 planned B slots (50 percent); accepted lineage can be lower. No MC shortfall may be relabelled targeted.",
    "MC_exposure_limit": "B author has feedback exposure even when MC instructions are common-only; MC is not independently blinded or isolated from arm-level exposure.",
    "primary_day3_outcome": "Content and target realization coverage, independent source reconstruction; no Solver score-based selection.",
    "first_batch_rule": "First two complete root_id strings from inherited wave1, not chosen using scores.",
    "budget": {"initial_per_slot": 1, "wording_clarification_per_slot": 1, "requested_output_tokens_per_item": 3072,
               "actual_provider_cap": None, "actual_output_tokens": None,
               "execution": "Interactive Codex authors, shared filesystem; requested per-item ceiling is not an enforceable API setting. No equal realized compute claim."},
    "scope": "Complete provided source is available for reconstruction. Target facts are R02; ordering uses first semantic report in the complete source unless an explicitly frozen speech act is stated. Never silently restrict mention search to R02.",
    "repair": "Only one wording clarification; changed facts, labels' semantic anchors or Gold cannot be silently repaired into the inherited slot.",
    "completion_rule": "Day3 is not complete until actual production coverage, factual-family deduplication and common evaluation content/request freeze are finished or explicit stop criteria adjudicated.",
}
dump(D/"DAY3_SCOPE_AMENDMENT.json", amendment)
dump(D/"EXECUTION_FREEZE.json", {
    "id": "DAY3_STAGE1_EXECUTION_FREEZE_V1", "created_at_utc": now,
    "candidate_count_at_freeze": 0, "solver_calls_at_freeze": 0,
    "amendment_sha256": sha(D/"DAY3_SCOPE_AMENDMENT.json"),
    "public_design_sha256": sha(library), "source_review_sha256": sha(review),
    "inherited_slots_sha256": sha(W/"day1/SLOT_PLAN.json"),
    "roots": roots, "first_batch_roots": first,
    "planned_counts": {"A":120,"B":120,"first_batch_slots":24},
    "gold_targets_each_arm": dict(Counter(str(s["target_gold_count"]) for s in slots if s["arm"]=="A" and s["question_type"]=="multi_select")),
    "source_pending_is_generation_blocking": True, "slots": assigned,
    "minimum_accepted_coverage_each_arm": {"substantive_MC_relations":2, "ordering_profiles":2},
    "above_minimum_is_not_full_production_complete": True,
    "source_material_shortfall_is_not_a_question": True,
})
for arm in ["A","B"]:
    for root in first:
        own = [s for s in assigned if s["arm"]==arm and s["root_id"]==root]
        # A gets no amendment, treatment information, feedback card, or B assignments.
        public = {
            "packet_id": f"DAY3-{arm}-{root}", "created_at_utc": now, "arm":arm,"root_id":root,
            "public_design": {"path":str(library),"sha256":sha(library)},
            "source_binding":own[0]["source_binding"],
            "source_admission":{"path":str(review),"sha256":sha(review)},
            "public_structure_slots":[{
                k:s[k] for k in ["slot_id","candidate_slot_id","local_slot","question_type","target_gold_count","options_or_items","design_id"]
            } for s in own],
            "public_contract": {
                "source_scope":"Read full source. Facts target R02. Search all history for earlier/equivalent mentions; first semantic report in full source is the ordering rule. Explicitly word any restricted speech act before authoring.",
                "minimum_relations":"At least two substantive MC relations and two ordering profiles per arm are a coverage goal, not permission to substitute a slot.",
                "insufficient_source":"Return SHORTFALL, do not fabricate a conditional rule or call plain status lookup a revision.",
                "budget":{"initial_per_slot":1,"wording_clarification_per_slot":1,"requested_output_tokens_per_question":3072,"actual_cap":None,"actual_usage":None},
                "records":"Write READ_MANIFEST then PLAN before any DRAFT. Each authored slot: one public JSON with candidate_id,root_id,question_type,question,options_or_items (A-F mapping). Separate private JSON with gold, evidence per label with exact quote/turn, intended design and all semantic mentions. Shortfall has its own record and no public question.",
                "review_status":"Author must not mark QA PASS; independent reconstruction still required.",
                "model_identity":"Interactive Codex; provider model revision, usage and actual cap unavailable.",
                "no_solver":"No Solver calls, prior scores, external answer search or own candidate repair.",
            },
        }
        dump(D/"packets"/arm/(root+".json"), public)
cards = json.loads((W/"day2_feedback_readiness_v1/OPERATIONAL_CARDS.json").read_text())
c = next(c for c in cards["cards"] if c["card_id"]=="D2-ORDER-FIXED-CITATIONS")
keys = ["card_id","status","admission","operational_claim","competing_explanations","limitations","targeted_design","generation_review_requirements"]
compact = {k:c[k] for k in keys}
compact["bounded_case"] = c["cases"][0]
compact["derived_from"] = {"path":str(W/"day2_feedback_readiness_v1/OPERATIONAL_CARDS.json"),"sha256":sha(W/"day2_feedback_readiness_v1/OPERATIONAL_CARDS.json"),"selection":"Prospectively selected first case; other cases not included, no claim of complete card read."}
compact["day3_use"] = "Cross-scene ordering only. Within-scene structure is a public design addition. No MC diagnosed cause is authorized."
dump(D/"packets/B/ORDERING_FEEDBACK_BOUNDED.json", compact)
print(json.dumps({"freeze":str(D/"EXECUTION_FREEZE.json"),"sha256":sha(D/"EXECUTION_FREEZE.json"),"first_batch_roots":first,"planned_slots":len(assigned)}))
