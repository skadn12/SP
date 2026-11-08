"""Read-only v116 delivery audit using an independent selection calculation.

No imports from the screen controller or delivery builder, no model calls, and
no interpretation of dialogue semantics. Writes one exclusive audit receipt.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BATCH = Path(__file__).resolve().parents[1]
CHECK_NAMES = {
    "identity_coreference", "gender_relationship_titles", "fact_state_time",
    "unsupported_reconciliation",
}
STATUS_PRIORITY = {"PASS": 0, "WARNING": 1, "HOLD": 2}
CLASSIFICATION_STATUS = {
    "HARD_CONTRADICTION": "HOLD", "UNRESOLVED_IDENTITY_OR_TITLE": "HOLD",
    "UNRESOLVED_CORE_EVENT": "HOLD", "UNRESOLVED_STATE_TIME": "WARNING",
    "ATTRIBUTED_UNVERIFIED_REPORT": "WARNING", "UNDERSPECIFIED_ATTRIBUTE": "WARNING",
    "COMPATIBLE_TIME_CHANGE": "PASS", "COMPATIBLE_NONEXCLUSIVE": "PASS",
}


class Audit:
    def __init__(self):
        self.errors = []
        self.bindings = {}
        self.counts = {}

    def require(self, condition, message):
        if not condition:
            raise ValueError(message)

    def read_bytes(self, path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        self.require(str(path) not in self.bindings or self.bindings[str(path)] == actual,
                     f"artifact changed during independent audit: {path}")
        self.bindings[str(path)] = actual
        if expected is not None:
            self.require(actual == expected, f"SHA256 mismatch: {path}")
        return data

    def object_pairs(self, pairs):
        result = {}
        for key, value in pairs:
            self.require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def read(self, path, expected=None):
        value = json.loads(self.read_bytes(path, expected), object_pairs_hook=self.object_pairs)
        self.require(isinstance(value, dict), f"JSON object required: {path}")
        return value

    def stage(self, name, operation):
        try:
            return operation()
        except Exception as error:
            self.errors.append(f"{name}: {type(error).__name__}: {error}")
            return None

    def timestamp(self, value):
        self.require(isinstance(value, str), "timestamp must be a string")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        self.require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
                     "timestamp must carry UTC timezone")
        return parsed

    def string(self, value):
        self.require(isinstance(value, str) and bool(value.strip()), "nonempty string required")

    def citation(self, citation, turns):
        self.require(isinstance(citation, dict) and set(citation) == {"turn_ref", "speaker", "quote"},
                     "invalid citation fields")
        self.require(citation["turn_ref"] in turns, "citation refers to missing source turn")
        source = turns[citation["turn_ref"]]
        self.require(citation["speaker"] == source["speaker"], "citation speaker mismatch")
        self.string(citation["quote"])
        self.require(citation["quote"] in source["text"], "citation is not an exact source substring")

    def review(self, plan, plan_sha, candidate, path):
        review = self.read(path)
        expected_fields = {
            "schema_version", "policy_version", "root_id", "baseline_path", "baseline_sha256",
            "plan_sha256", "policy_sha256", "reviewer", "started_at_utc", "finished_at_utc",
            "full_source_read", "covered_turn_refs", "checks", "findings",
            "continuation_constraints", "decision",
        }
        self.require(set(review) == expected_fields, "review schema field mismatch")
        self.require(review["schema_version"] == "codex_baseline_review_v1"
                     and review["policy_version"] == "v116", "review version mismatch")
        for field in ("root_id", "baseline_path", "baseline_sha256"):
            self.require(review[field] == candidate[field], f"review candidate binding mismatch: {field}")
        self.require(review["plan_sha256"] == plan_sha and review["policy_sha256"] == plan["policy_sha256"],
                     "review plan/policy binding mismatch")
        source = self.read(candidate["baseline_path"], candidate["baseline_sha256"])
        self.require(source["root_id"] == candidate["root_id"], "baseline root mismatch")
        rows = source["turns"]
        refs = [row["turn_ref"] for row in rows]
        self.require(len(refs) == len(set(refs)) and len(refs) == candidate["baseline_turn_count"],
                     "baseline refs/count invalid")
        self.require(sorted({row["speaker"] for row in rows}) == candidate["baseline_speakers"]
                     and len(candidate["baseline_speakers"]) == 2, "baseline speakers invalid")
        scenes = {row["scene_ref"] for row in rows if isinstance(row.get("scene_ref"), str)
                  and row["scene_ref"].startswith("R01:")}
        self.require(scenes == {"R01:S01", "R01:S02", "R01:S03"}, "baseline R01 structure invalid")
        self.require(review["full_source_read"] is True and review["covered_turn_refs"] == refs,
                     "source coverage must match all turns in original order")
        self.string(review["reviewer"])
        self.require(review["reviewer"] == review["reviewer"].strip(), "reviewer identity whitespace")
        self.require(self.timestamp(plan["created_at_utc"]) <= self.timestamp(review["started_at_utc"])
                     <= self.timestamp(review["finished_at_utc"]) <= datetime.now(timezone.utc),
                     "review timestamps not in permitted order")
        checks = review["checks"]
        self.require(isinstance(checks, dict) and set(checks) == CHECK_NAMES, "four complete checks required")
        for check in checks.values():
            self.require(set(check) == {"complete", "status", "reason"} and check["complete"] is True,
                         "incomplete check schema")
            self.require(check["status"] in STATUS_PRIORITY, "invalid check status")
            self.string(check["reason"])
        category_findings = {key: ["PASS"] for key in CHECK_NAMES}
        self.require(isinstance(review["findings"], list), "findings list required")
        seen_ids = set()
        citation_count = 0
        temporal_count = 0
        for finding in review["findings"]:
            self.require(set(finding) == {"id", "category", "classification", "disposition", "reason",
                                          "citations", "temporal_evidence"}, "finding schema mismatch")
            self.string(finding["id"])
            self.require(finding["id"] not in seen_ids, "duplicate finding ID")
            seen_ids.add(finding["id"])
            self.require(finding["category"] in CHECK_NAMES, "invalid finding category")
            self.require(finding["classification"] in CLASSIFICATION_STATUS, "invalid finding classification")
            self.require(finding["disposition"] == CLASSIFICATION_STATUS[finding["classification"]],
                         "classification disposition mismatch")
            self.string(finding["reason"])
            citations = finding["citations"]
            temporal = finding["temporal_evidence"]
            self.require(isinstance(citations, list) and bool(citations), "finding lacks source citations")
            self.require(isinstance(temporal, list), "temporal evidence list required")
            for citation in citations + temporal:
                self.citation(citation, dict(zip(refs, rows)))
            citation_count += len(citations)
            temporal_count += len(temporal)
            if finding["disposition"] == "HOLD":
                self.require(len({citation["turn_ref"] for citation in citations}) >= 2,
                             "HOLD lacks two distinct cited turns")
            if finding["classification"] == "COMPATIBLE_TIME_CHANGE":
                self.require(bool(temporal), "time-change finding lacks temporal evidence")
            category_findings[finding["category"]].append(finding["disposition"])
        for key, statuses in category_findings.items():
            self.require(checks[key]["status"] == max(statuses, key=STATUS_PRIORITY.get),
                         "finding to check aggregation mismatch")
        self.require(review["decision"] == max((check["status"] for check in checks.values()),
                                              key=STATUS_PRIORITY.get), "check to review aggregation mismatch")
        constraints = review["continuation_constraints"]
        self.require(isinstance(constraints, list), "constraints must be string list")
        for constraint in constraints:
            self.string(constraint)
        self.require(review["decision"] == "PASS" or bool(constraints), "non-PASS missing continuation constraints")
        self.counts["validated_findings"] = self.counts.get("validated_findings", 0) + len(review["findings"])
        self.counts["validated_citations"] = self.counts.get("validated_citations", 0) + citation_count
        self.counts["validated_temporal_citations"] = self.counts.get("validated_temporal_citations", 0) + temporal_count
        return review

    def run(self):
        plan_path = BATCH / "SCREEN_PLAN.json"
        plan = self.read(plan_path)
        plan_sha = self.bindings[str(plan_path)]
        self.require(plan["schema_version"] == "codex_baseline_screen_plan_v1", "plan schema mismatch")
        self.require(plan["candidate_denominator"] == 8 and plan["reviews_per_candidate"] == 2
                     and plan["selection_max"] == 4, "fixed denominators changed")
        self.require(plan["status"] == "FROZEN_BEFORE_THIS_BATCH_FULL_TEXT_SEMANTIC_READS", "plan not frozen")
        order = plan["candidate_order"]
        self.require(len(order) == 8 and len(set(order)) == 8
                     and [candidate["root_id"] for candidate in plan["candidates"]] == order,
                     "invalid candidate identities/order")
        self.require(order == sorted(order, key=lambda root: int(root.rsplit("-", 1)[-1])), "candidate order changed")
        self.require(not set(order).intersection(plan["excluded_previous_full_validation_roots"]), "excluded candidate reused")
        self.read_bytes(plan["policy_path"], plan["policy_sha256"])
        for row in plan["bindings"]:
            self.read_bytes(row["path"], row["sha256"])
        reviews = {}
        for candidate in plan["candidates"]:
            rid = candidate["root_id"]
            reviews[rid] = []
            for role in ("primary", "secondary"):
                path = BATCH / "reviews" / ("root" + rid.rsplit("-", 1)[-1]) / (role + ".json")
                review = self.stage(f"review {rid}/{role}", lambda c=candidate, p=path: self.review(plan, plan_sha, c, p))
                if review is not None:
                    reviews[rid].append((path, review))
            self.require(len(reviews[rid]) == 2, f"missing/invalid review for {rid}")
            self.require(len({review["reviewer"] for _, review in reviews[rid]}) == 2, "duplicate reviewer identity")
        self.counts.update(candidate_denominator=8, validated_reviews=16)
        actual_reviews = set((BATCH / "reviews").glob("root*/primary.json")) | set((BATCH / "reviews").glob("root*/secondary.json"))
        self.require(actual_reviews == {path for rows in reviews.values() for path, _ in rows}, "extra or missing final reviews")

        results = self.read(BATCH / "SCREEN_RESULTS.json")
        selected = self.read(BATCH / "SELECTED_ROOTS.json")
        self.require(results["schema_version"] == "codex_baseline_screen_results_v1"
                     and selected["schema_version"] == "codex_baseline_selected_roots_v1", "output schema mismatch")
        for output in (results, selected):
            self.require(output["batch_id"] == plan["batch_id"] and output["plan_path"] == str(plan_path)
                         and output["plan_sha256"] == plan_sha and output["policy_path"] == plan["policy_path"]
                         and output["policy_sha256"] == plan["policy_sha256"], "output input bindings mismatch")
            self.require(output["candidate_denominator"] == 8 and output["reviews_per_candidate"] == 2
                         and output["validated_review_count"] == 16 and output["selection_max"] == 4,
                         "output denominator mismatch")
            for field in ("generation_started", "solver_started", "training_started", "pool_merge_started",
                          "semantic_correctness_certified_by_program", "external_semantic_certification"):
                self.require(output[field] is False, f"unsupported activity/certification: {field}")
            self.require(self.timestamp(output["created_at_utc"]) >= max(
                self.timestamp(review["finished_at_utc"]) for rows in reviews.values() for _, review in rows),
                "aggregation predates completed reviews")
            self.require(self.timestamp(output["created_at_utc"]) <= datetime.now(timezone.utc),
                         "aggregation timestamp is in the future")
        self.require(results["created_at_utc"] == selected["created_at_utc"], "output timestamp mismatch")
        self.require([row["root_id"] for row in results["roots"]] == order, "results drop/reorder candidates")
        independently_selected = []
        independently_eligible = []
        for candidate, row in zip(plan["candidates"], results["roots"]):
            rid = candidate["root_id"]
            review_rows = sorted(reviews[rid], key=lambda pair: pair[1]["reviewer"])
            statuses = [review["decision"] for _, review in review_rows]
            decision = "HOLD" if "HOLD" in statuses else "WARNING" if "WARNING" in statuses else "PASS"
            eligible = decision != "HOLD"
            take = eligible and len(independently_selected) < 4
            constraints = []
            for _, review in review_rows:
                for constraint in review["continuation_constraints"]:
                    if constraint not in constraints:
                        constraints.append(constraint)
            self.require(all(row[key] == candidate[key] for key in ("root_id", "baseline_path", "baseline_sha256")),
                         "root entry source mismatch")
            self.require(row["decision"] == decision and row["eligible"] is eligible and row["selected"] is take,
                         f"root admission mismatch: {rid}")
            self.require(row["continuation_constraints"] == constraints, f"lost/changed continuation constraints: {rid}")
            expected_review_refs = [{"path": str(path), "sha256": self.bindings[str(path)],
                                     "reviewer": review["reviewer"], "decision": review["decision"],
                                     "continuation_constraints": review["continuation_constraints"]}
                                    for path, review in review_rows]
            self.require(row["reviews"] == expected_review_refs, "root dual-review binding mismatch")
            if eligible:
                independently_eligible.append(rid)
            if take:
                independently_selected.append(rid)
        self.require([row["root_id"] for row in selected["roots"]] == independently_selected,
                     "selected roots differ from independent fixed-order calculation")
        self.require(selected["roots"] == [row for row in results["roots"] if row["selected"]],
                     "selected root details differ from full result entries")
        self.require(results["selected_root_count"] == selected["selected_root_count"] == len(independently_selected)
                     and results["eligible_root_count"] == len(independently_eligible), "summary count mismatch")
        self.require(selected["screen_results_path"] == str(BATCH / "SCREEN_RESULTS.json")
                     and selected["screen_results_sha256"] == self.bindings[str(BATCH / "SCREEN_RESULTS.json")],
                     "selection result binding mismatch")
        self.counts.update(eligible_roots=len(independently_eligible), selected_roots=len(independently_selected),
                           held_roots=8-len(independently_eligible))

        comparison = self.read(BATCH / "REVIEW_COMPARISON.json")
        self.require([row["root_id"] for row in comparison["roots"]] == order, "comparison candidate order mismatch")
        for row in comparison["roots"]:
            statuses = [review["decision"] for _, review in reviews[row["root_id"]]]
            self.require(row["decisions"] == statuses and row["different_overall_decisions"] == (len(set(statuses)) > 1)
                         and row["admission_disagreement"] == (len({status == "HOLD" for status in statuses}) > 1),
                         "reader comparison mismatch")
        tools = self.read(BATCH / "TOOLING_FREEZE.json")
        self.require(tools["status"] == "FROZEN_BEFORE_AGGREGATION" and len(tools["files"]) == 3, "tooling freeze incomplete")
        self.require(self.timestamp(plan["created_at_utc"]) <= self.timestamp(tools["created_at_utc"])
                     <= self.timestamp(results["created_at_utc"]), "tooling freeze timing mismatch")
        for tool in tools["files"]:
            self.read_bytes(tool["path"], tool["sha256"])
            self.read_bytes(tool["snapshot"]["path"], tool["snapshot"]["sha256"])
            self.require(tool["sha256"] == tool["snapshot"]["sha256"], "tool snapshot differs")
        self.counts["verified_tool_snapshots"] = len(tools["files"])
        handoff = self.read(BATCH / "NEXT_BATCH_HANDOFF.json")
        self.require(self.timestamp(results["created_at_utc"]) <= self.timestamp(comparison["created_at_utc"])
                     <= self.timestamp(handoff["created_at_utc"]) <= datetime.now(timezone.utc),
                     "comparison/handoff timestamps out of order")
        self.require(handoff["status"] == "BASELINE_GATE_COMPLETED_ROOT_SELECTION_FIXED_GENERATION_NOT_STARTED"
                     and handoff["selection_authority"] == "SELECTED_ROOTS.json"
                     and handoff["generation_root_replacement_allowed"] is False, "invalid handoff authority/status")
        expected_handoff_files = {str(BATCH / name) for name in (
            "SCREEN_PLAN.json", "BASELINE_POLICY.md", "SCREEN_RESULTS.json", "SELECTED_ROOTS.json",
            "REVIEW_COMPARISON.json", "TOOLING_FREEZE.json")}
        self.require({row["path"] for row in handoff["screen_bindings"]} == expected_handoff_files
                     and len(handoff["screen_bindings"]) == 6, "handoff missing screen artifact bindings")
        for row in handoff["screen_bindings"]:
            self.read_bytes(row["path"], row["sha256"])
        expected_rules = {
            "sequence": ["R02:A:S01", "R02:A:S02", "R02:A:S03", "R02:B:S01", "R02:B:S02", "R02:B:S03"],
            "turns_per_scene": {"min": 12, "max": 16}, "independent_full_source_ready_before_next_scene": True,
            "author_attempts_per_scene": 1, "semantic_repairs": 0, "source_replacements_after_reservation": 0,
            "bind_all_selected_root_continuation_constraints": True, "new_speaking_characters": False,
            "question_gold_targets": [2, 3, 4, 5], "gold6_control_per_complete_root": 1,
            "ordering_per_complete_root": 2, "source_first_asset_and_public_question_reviews": True,
            "lossless_evidence_export": True, "solver_protocol": "v15_full_only_random_mcq6x3_order12x1_v1",
            "models": ["M0", "M_clean_policy"], "solver_automatic_retries": False,
            "selection": "complete-valid 0 < mean_partial < 1; exclude Gold6 control",
        }
        self.require(handoff["required_rules"] == expected_rules, "handoff changes inherited generation/QA rules")
        self.require(handoff["existing_training_exposure"] is True and handoff["unseen_holdout_claimed"] is False,
                     "handoff exposure disclosure mismatch")
        for field in ("generation_started", "solver_started", "training_started", "pool_merge_started"):
            self.require(handoff[field] is False, f"handoff activity flag mismatch: {field}")

        protection = self.read(BATCH / "HISTORICAL_PROTECTION.json")
        history = protection["files"]
        self.require(len({row["path"] for row in history}) == len(history), "duplicate historical protection entry")
        history_errors = []
        for row in history:
            try:
                actual = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
                if actual != row["sha256"]:
                    history_errors.append(row["path"])
            except OSError:
                history_errors.append(row["path"])
        self.require(not history_errors, f"historical files changed/missing: {history_errors}")
        self.counts["verified_historical_files"] = len(history)
        history_audit = self.read(BATCH / "HISTORICAL_AUDIT.json")
        self.require(self.timestamp(handoff["created_at_utc"]) <= self.timestamp(history_audit["created_at_utc"])
                     <= datetime.now(timezone.utc), "historical audit timing mismatch")
        self.require(history_audit["status"] == "PASS" and history_audit["errors"] == []
                     and history_audit["verified_file_count"] == len(history), "historical audit disagrees")
        self.read_bytes(history_audit["protection_binding"]["path"], history_audit["protection_binding"]["sha256"])
        self.read_bytes(__file__)

    def save(self):
        receipt = {
            "schema_version": "codex_baseline_delivery_audit_v1", "reviewer": "/root/v113_14304_recheck",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if not self.errors else "FAIL", "errors": self.errors,
            "scope": "Independent mechanical review/selection/constraint/provenance checks; not semantic certification.",
            "controller_imported": False, "finalize_screen_used_for_recomputation": False,
            "counts": self.counts,
            "input_bindings": [{"path": path, "sha256": digest} for path, digest in sorted(self.bindings.items())],
        }
        with (BATCH / "DELIVERY_AUDIT.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({"status": receipt["status"], "errors": receipt["errors"], "counts": self.counts}, ensure_ascii=False))
        return not self.errors


if __name__ == "__main__":
    audit = Audit()
    audit.stage("delivery", audit.run)
    raise SystemExit(0 if audit.save() else 1)
