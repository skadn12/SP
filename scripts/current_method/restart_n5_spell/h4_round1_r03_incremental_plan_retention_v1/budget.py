"""Fail-closed, persistent reservations. No model or Memory transport here."""
from __future__ import annotations

import fcntl
import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path

ROLES = ("planner", "supplement", "gate", "realizer", "adherence", "closure")
READINESS_FIELDS = {"role_capacity", "readiness_version", "cumulative_length_measure", "solver_training_limits"}


def validate_budget(value):
    required = {"total_requests", "role_requests", "max_input_tokens",
                "max_output_tokens", "context_window_tokens",
                "max_generated_tokens", "max_cumulative_tokens",
                "scene_output_reserve_tokens"}
    if not isinstance(value, dict) or set(value) not in (required, required | READINESS_FIELDS):
        raise ValueError("complete_explicit_budget_required")
    for name in required - {"role_requests", "max_output_tokens"}:
        if type(value[name]) is not int or value[name] <= 0:
            raise ValueError("positive_budget_required:" + name)
    for name in ("role_requests", "max_output_tokens"):
        if set(value[name]) != set(ROLES):
            raise ValueError("explicit_all_role_budgets_required:" + name)
        for role, count in value[name].items():
            if type(count) is not int or count < (1 if name == "max_output_tokens" else 0):
                raise ValueError("invalid_role_budget:" + role)
    if value["max_output_tokens"]["planner"] != 8192 or value["max_output_tokens"]["supplement"] != 8192:
        raise ValueError("planner_output_limit_must_be_8192")
    if value["role_requests"]["planner"] > 1 or value["role_requests"]["supplement"] > 1:
        raise ValueError("one_initial_and_one_supplement_maximum")
    if any(value["max_input_tokens"] + n > value["context_window_tokens"]
           for n in value["max_output_tokens"].values()):
        raise ValueError("context_output_reserve_exceeds_window")
    if "role_capacity" in value:
        from .capacity import API_ROLES, VERSION
        if value["readiness_version"] != VERSION or set(value["role_capacity"]) != set(API_ROLES):
            raise ValueError("explicit_role_capacity_version_required")
        if value["cumulative_length_measure"] != "local_tokenizer_canonical_dialogue_json":
            raise ValueError("explicit_cumulative_length_measure_required")
        for role, spec in value["role_capacity"].items():
            if set(spec) != {"model", "context_window_tokens", "max_input_tokens", "thinking_reserve_tokens", "safety_margin_tokens", "capacity_source"}:
                raise ValueError("complete_role_capacity_required:" + role)
            for name in ("context_window_tokens", "max_input_tokens", "thinking_reserve_tokens", "safety_margin_tokens"):
                if type(spec[name]) is not int or spec[name] < (1 if name in {"context_window_tokens", "max_input_tokens"} else 0):
                    raise ValueError("invalid_role_capacity:" + role)
            if not spec["model"] or not spec["capacity_source"] or spec["thinking_reserve_tokens"] != 0:
                raise ValueError("fixed_non_thinking_role_identity_required")
            if spec["max_input_tokens"] + value["max_output_tokens"][role] + spec["thinking_reserve_tokens"] + spec["safety_margin_tokens"] > spec["context_window_tokens"]:
                raise ValueError("role_context_output_reserve_exceeded:" + role)
    return json.loads(json.dumps(value))


class BudgetLedger:
    """One durable ledger per production root/round or explicit experiment ID.

    Identity excludes branch/run directory: changing an attempt directory does
    not reset the supplement opportunity. Entries are committed before I/O.
    Restart after an ambiguous request never replays that call.
    """
    def __init__(self, registry, *, root_id, round_index, budget, execution_scope=None):
        self.budget = validate_budget(budget)
        identity = {"root_id": root_id, "round_index": round_index}
        key_identity = identity
        if execution_scope is not None:
            required = {"mode", "experiment_id", "grant_sha256", "source_run",
                        "production_ledger_path", "production_ledger_sha256"}
            if (not isinstance(execution_scope, dict) or set(execution_scope) != required
                    or execution_scope.get("mode") != "extra_experiment"
                    or any(not isinstance(execution_scope.get(name), str) or not execution_scope[name].strip()
                           for name in required)
                    or not re.fullmatch(r"[A-Za-z0-9_-]+", execution_scope["experiment_id"])
                    or any(not re.fullmatch(r"[0-9a-f]{64}", execution_scope[name])
                           for name in ("grant_sha256", "production_ledger_sha256"))):
                raise ValueError("complete_explicit_experiment_scope_required")
            identity["execution_scope"] = json.loads(json.dumps(execution_scope))
            # Different bindings must collide with the original authorization
            # ledger and be rejected, not receive another opportunity.
            key_identity = {"experiment_id": execution_scope["experiment_id"]}
        registry = Path(registry)
        registry.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps(key_identity, sort_keys=True).encode()).hexdigest()
        self.path = registry / (key + ".jsonl")
        self.identity = identity
        with self._locked() as rows:
            if not rows:
                self._append({"kind": "identity", **identity, "budget": self.budget})
            elif rows[0] != {"kind": "identity", **identity, "budget": self.budget}:
                raise ValueError("persistent_budget_identity_mismatch")

    @contextmanager
    def _locked(self):
        # Lock a stable inode; append-only journal survives process restarts.
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            rows = [json.loads(line) for line in handle if line.strip()]
            yield rows

    def _append(self, row):
        import os
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _active(rows):
        released = {r["reservation_id"] for r in rows if r["kind"] == "release"}
        return [r for r in rows if r["kind"] == "reserve" and r["reservation_id"] not in released]

    def reserve(self, reservation_id, requests, *, cumulative_tokens, added_scene_count=0):
        """Reserve every required check before generation; never truncate input."""
        if any(role not in ROLES or type(n) is not int or n < 0 for role, n in requests.items()):
            raise ValueError("invalid_reservation")
        with self._locked() as rows:
            if any(r.get("reservation_id") == reservation_id for r in rows):
                raise ValueError("reservation_already_exists_no_replay")
            active = self._active(rows)
            counts = {role: sum(r["requests"].get(role, 0) for r in active) for role in ROLES}
            allowed = all(counts[r] + requests.get(r, 0) <= self.budget["role_requests"][r] for r in ROLES)
            allowed &= sum(counts.values()) + sum(requests.values()) <= self.budget["total_requests"]
            output = sum((counts[r] + requests.get(r, 0)) * self.budget["max_output_tokens"][r] for r in ROLES)
            allowed &= output <= self.budget["max_generated_tokens"]
            allowed &= cumulative_tokens + added_scene_count * self.budget["scene_output_reserve_tokens"] <= self.budget["max_cumulative_tokens"]
            if not allowed:
                self._append({"kind": "deferred", "reservation_id": reservation_id, "status": "DEFERRED_BUDGET"})
                return False
            self._append({"kind": "reserve", "reservation_id": reservation_id, "requests": requests})
            return True

    def begin(self, reservation_id, role, *, input_tokens):
        with self._locked() as rows:
            reservation = next((r for r in self._active(rows) if r["reservation_id"] == reservation_id), None)
            if reservation is None:
                raise ValueError("request_not_reserved")
            attempts = [r for r in rows if r["kind"] == "attempt"]
            used = sum(r["reservation_id"] == reservation_id and r["role"] == role for r in attempts)
            if used >= reservation["requests"].get(role, 0):
                raise ValueError("reserved_role_exhausted")
            if role in {"planner", "supplement"} and any(r["role"] == role for r in attempts):
                raise ValueError("root_round_" + role + "_already_consumed")
            from .capacity import role_capacity
            limit = role_capacity(self.budget, role)["max_input_tokens"]
            if type(input_tokens) is not int or input_tokens < 0 or input_tokens > limit:
                raise ValueError("input_budget_exceeded_no_truncation")
            call_id = len(attempts) + 1
            self._append({"kind": "attempt", "call_id": call_id,
                          "reservation_id": reservation_id, "role": role,
                          "input_tokens": input_tokens, "state": "ATTEMPTED_NO_AUTORETRY"})
            return call_id

    def finish(self, call_id, *, status):
        with self._locked():
            self._append({"kind": "result", "call_id": call_id, "status": status})

    def claim_execution(self, identity):
        """Fail closed on an outer second attempt, even after an interrupted run."""
        with self._locked() as rows:
            if any(r["kind"] == "execution" for r in rows):
                raise ValueError("root_round_execution_already_claimed_no_outer_attempt")
            self._append({"kind": "execution", "identity": identity})

    def claim_resume(self, identity, *, source_ledger_sha256, source_ledger_header):
        """One conservative downstream resume, never a new planning opportunity."""
        with self._locked() as rows:
            if not rows or rows[0] != source_ledger_header:
                raise ValueError("resume_ledger_identity_mismatch")
            if hashlib.sha256(self.path.read_bytes()).hexdigest() != source_ledger_sha256:
                raise ValueError("resume_ledger_changed_or_already_claimed")
            if any(row["kind"] == "resume_execution" for row in rows):
                raise ValueError("resume_already_claimed_no_outer_attempt")
            executions = [row for row in rows if row["kind"] == "execution"]
            attempts = [row for row in rows if row["kind"] == "attempt"]
            if (len(executions) != 1 or executions[0]["identity"] != identity["identity"]
                    or len(attempts) != 1 or attempts[0]["role"] != "supplement"):
                raise ValueError("resume_requires_one_completed_supplement_no_downstream_attempts")
            results = [row for row in rows if row["kind"] == "result"
                       and row.get("call_id") == attempts[0]["call_id"]]
            if len(results) != 1 or results[0]["status"] != "RESPONSE_RECEIVED":
                raise ValueError("resume_supplement_response_not_confirmed")
            reservations = self._active(rows)
            if ({row["reservation_id"] for row in reservations} != {"closure", "supplement"}
                    or any(row["requests"] != {row["reservation_id"]: 1} for row in reservations)):
                raise ValueError("resume_requires_original_closure_and_supplement_reservations")
            self._append({"kind": "resume_execution", "identity": identity,
                          "source_ledger_sha256": source_ledger_sha256,
                          "state": "CLAIMED_NO_AUTORESTART_NO_NEW_SUPPLEMENT"})

    def claim_gate_result_resume(self, identity, *, source_ledger_sha256, source_ledger_header):
        """Claim one explicit Gate-result reuse without restoring spent requests."""
        with self._locked() as rows:
            if not rows or rows[0] != source_ledger_header:
                raise ValueError("gate_reuse_ledger_identity_mismatch")
            if hashlib.sha256(self.path.read_bytes()).hexdigest() != source_ledger_sha256:
                raise ValueError("gate_reuse_ledger_changed_or_already_claimed")
            if self.identity.get("execution_scope", {}).get("mode") != "extra_experiment":
                raise ValueError("gate_reuse_requires_explicit_experiment_scope")
            if any(row["kind"] == "gate_result_resume" for row in rows):
                raise ValueError("gate_reuse_already_claimed_no_outer_attempt")
            executions = [row for row in rows if row["kind"] == "execution"]
            resumes = [row for row in rows if row["kind"] == "resume_execution"]
            if (not isinstance(identity, dict) or "identity" not in identity
                    or len(executions) != 1 or executions[0]["identity"] != identity["identity"]
                    or len(resumes) != 1 or resumes[0]["identity"].get("identity") != identity["identity"]):
                raise ValueError("gate_reuse_requires_original_execution_and_resume")
            attempts = [row for row in rows if row["kind"] == "attempt"]
            counts = {role: sum(row["role"] == role for row in attempts) for role in ROLES}
            if len(attempts) != 9 or counts != {"planner": 0, "supplement": 1, "gate": 8,
                                              "realizer": 0, "adherence": 0, "closure": 0}:
                raise ValueError("gate_reuse_requires_one_supplement_eight_gates_no_downstream_attempts")
            if len({row["call_id"] for row in attempts}) != len(attempts):
                raise ValueError("gate_reuse_request_identity_collision")
            for attempt in attempts:
                results = [row for row in rows if row["kind"] == "result"
                           and row.get("call_id") == attempt["call_id"]]
                if len(results) != 1 or results[0]["status"] != "RESPONSE_RECEIVED":
                    raise ValueError("gate_reuse_all_source_responses_must_be_confirmed")
            reservations = self._active(rows)
            fixed = [row for row in reservations if row["reservation_id"] in {"closure", "supplement"}]
            slots = [row for row in reservations if row["reservation_id"] not in {"closure", "supplement"}]
            if (len(fixed) != 2 or {row["reservation_id"] for row in fixed} != {"closure", "supplement"}
                    or any(row["requests"] != {row["reservation_id"]: 1} for row in fixed)
                    or len(slots) != 4
                    or len({row["reservation_id"] for row in slots}) != len(slots)
                    or any(row["requests"] != {"gate": 2, "realizer": 1, "adherence": 1} for row in slots)):
                raise ValueError("gate_reuse_requires_existing_reserved_generation_slots")
            if (any(sum(row["reservation_id"] == slot["reservation_id"] and row["role"] == "gate"
                        for row in attempts) != 2 for slot in slots)
                    or next(row for row in attempts if row["role"] == "supplement")["reservation_id"] != "supplement"):
                raise ValueError("gate_reuse_source_reservations_mismatch")
            slot_ids = [row["reservation_id"] for row in slots]
            self._append({"kind": "gate_result_resume", "identity": identity,
                          "source_ledger_sha256": source_ledger_sha256,
                          "generation_reservation_ids": slot_ids,
                          "state": "CLAIMED_NO_AUTORESTART_NO_NEW_PLANNER_OR_GATE"})
            return slot_ids

    def check_reused_generation_slot(self, reservation_id, *, cumulative_tokens, added_scene_count):
        """Read-only pre-generation check of an existing unused paired reservation."""
        if (type(cumulative_tokens) is not int or cumulative_tokens < 0
                or type(added_scene_count) is not int or not 0 <= added_scene_count <= 2):
            raise ValueError("invalid_reused_generation_capacity")
        with self._locked() as rows:
            claims = [row for row in rows if row["kind"] == "gate_result_resume"]
            if len(claims) != 1 or reservation_id not in claims[0]["generation_reservation_ids"]:
                raise ValueError("gate_reuse_generation_slot_not_claimed")
            active = self._active(rows)
            reservation = next((row for row in active if row["reservation_id"] == reservation_id), None)
            if reservation is None or reservation["requests"] != {"gate": 2, "realizer": 1, "adherence": 1}:
                raise ValueError("gate_reuse_generation_slot_not_reserved")
            attempts = [row for row in rows if row["kind"] == "attempt"]
            if any(row["reservation_id"] == reservation_id and row["role"] in {"realizer", "adherence"}
                   for row in attempts):
                raise ValueError("gate_reuse_generation_slot_already_attempted_no_replay")
            counts = {role: sum(row["requests"].get(role, 0) for row in active) for role in ROLES}
            used = {role: sum(row["role"] == role for row in attempts) for role in ROLES}
            allowed = all(counts[role] <= self.budget["role_requests"][role] for role in ROLES)
            allowed &= sum(counts.values()) <= self.budget["total_requests"]
            allowed &= all(used[role] + 1 <= self.budget["role_requests"][role]
                           for role in ("realizer", "adherence"))
            allowed &= sum(used.values()) + 2 <= self.budget["total_requests"]
            allowed &= sum(counts[role] * self.budget["max_output_tokens"][role] for role in ROLES) <= self.budget["max_generated_tokens"]
            allowed &= cumulative_tokens + added_scene_count * self.budget["scene_output_reserve_tokens"] <= self.budget["max_cumulative_tokens"]
            return allowed

    def snapshot(self):
        with self._locked() as rows:
            return rows

    def derived_scene_safety_limit(self):
        b = self.budget
        return min(b["role_requests"]["gate"],
                   b["role_requests"]["realizer"] * 2,
                   b["role_requests"]["adherence"] * 2,
                   b["max_generated_tokens"] // b["scene_output_reserve_tokens"],
                   max(0, (b["total_requests"] - b["role_requests"]["planner"] - b["role_requests"]["supplement"] - b["role_requests"]["closure"]) // 4) * 2)
