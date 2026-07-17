from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .backends import QuestionerBackend, QuestionerRequest, RealizerBackend, RealizerRequest, SolverBackend, SolverRequest
from .core import (atomic_json, canonical_json, classify_reward_group, grounding_decision,
                   context_preflight, local_reward, normalize_option, option_audit, parse_boxed_mcq_answer,
                   sha256_bytes, sha256_file)
from .ledger import RequestLedger
from .prompts import (RESPONDER_PROMPT, SOLVER_NOTHINK_RESPONDER_PROMPT,
                      build_questioner_prompt, build_realizer_prompt, prompt_template_hashes)
from .validation import (assess_candidate_shortfall, evidence_audit, parse_questioner_batches, validate_dialogue,
                         materialize_questioner_item, validate_memory, validate_qa, validate_realizer_identity_contract,
                         validate_realizer_output_schema, derive_local_character_ledger,
                         validate_event_arc_advancement, validate_hook_transition,
                         validate_durable_candidate_indices, validate_new_hook_descriptions,
                         validate_questioner_smoke_batch)
from .validation import DEFAULT_DURABLE_PREDICATES


class SimulatedCrash(RuntimeError):
    pass


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_unique_jsonl(path: Path, row: dict[str, Any], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        seen = {json.loads(line).get(key) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if row.get(key) not in seen:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class CanaryOrchestrator:
    def __init__(self, *, run_root: Path, manifest: dict[str, Any], config: dict[str, Any],
                 data: dict[str, Any], tokenizer: Any, realizer: RealizerBackend,
                 questioner: QuestionerBackend, solver: SolverBackend, backend_mode: str = "mock"):
        self.run_root, self.manifest, self.config, self.data = run_root, manifest, config, data
        self.tokenizer, self.realizer, self.questioner, self.solver = tokenizer, realizer, questioner, solver
        manifest_path = run_root / "config/run_manifest.json"
        self.manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.ledger = RequestLedger(run_root, self.manifest_sha, backend_mode)
        self.prompt_hashes = prompt_template_hashes()

    @staticmethod
    def _backend_identity(backend: Any) -> dict[str, Any]:
        identity: dict[str, Any] = {"backend_class":type(backend).__name__,
            "model_identifier":str(getattr(backend, "model_identifier", ""))}
        for key in ("max_model_len", "temperature", "top_p", "max_tokens", "tokenizer_path"):
            if hasattr(backend, key):
                identity[key] = str(getattr(backend, key)) if key == "tokenizer_path" else getattr(backend, key)
        client = getattr(backend, "client", None)
        config = getattr(client, "config", None)
        if config is not None:
            identity["challenger_config"] = {key:getattr(config, key) for key in (
                "base_url", "model", "timeout_seconds", "realizer_max_tokens",
                "questioner_max_tokens", "temperature")}
        return identity

    def _tokens(self, turns: Sequence[dict[str, Any]]) -> int:
        text = "\n".join(f"[{x.get('turn_id','')}] {x['speaker']}: {x['text']}" for x in turns)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _accepted_sessions(self) -> list[dict[str, Any]]:
        paths = sorted((self.run_root / "accepted").glob("session_*.json"))
        if not paths:
            return []
        index_path = self.run_root / "accepted/index.json"
        if not index_path.exists():
            raise RuntimeError("accepted-session integrity index is missing; refusing resume")
        index = _read_json(index_path)
        if (index.get("run_id") != self.manifest.get("run_id") or
                index.get("run_manifest_sha256") != self.manifest_sha):
            raise RuntimeError("accepted-session source identity drift; refusing resume")
        expected = {path.name:sha256_file(path) for path in paths}
        if index.get("session_sha256") != expected:
            raise RuntimeError("accepted-session hash drift; refusing resume")
        return [_read_json(path) for path in paths]

    @staticmethod
    def _active_hooks(sessions: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
        active: dict[str, dict[str, str]] = {}
        for session in sessions:
            for hook_id in session.get("resolved_hook_ids") or []:
                active.pop(str(hook_id), None)
            for hook in session.get("new_unresolved_hooks") or []:
                if isinstance(hook, dict) and hook.get("hook_id"):
                    active[str(hook["hook_id"])] = {"hook_id":str(hook["hook_id"]),
                                                     "description":str(hook.get("description") or "")}
        return [active[key] for key in sorted(active)]

    def _session_location(self, index: int) -> str:
        outline = self.data["outline"] if self is not None else {}
        scene_plan = outline.get("scene_plan") or []
        if scene_plan:
            scene = scene_plan[min(index - 1, len(scene_plan) - 1)]
            planned = scene.get("location") or scene.get("session_location")
            if planned:
                return str(planned)
        locations = outline.get("allowed_locations") or []
        if not locations:
            raise ValueError("outline has no deterministic session location")
        return str(locations[0])

    def _event_arc_state(self, sessions: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        counts = Counter(event_id for session in sessions
                         for event_id in session.get("advanced_event_arc_ids") or [])
        for arc in self.data["outline"].get("planned_event_arcs") or []:
            event_id = str(arc.get("event_id"))
            stages = list(arc.get("stages") or [arc.get("summary") or event_id])
            completed = min(int(counts.get(event_id, 0)), len(stages))
            state[event_id] = {"completed_stage_count":completed,
                "required_stage_count":len(stages), "closed":completed >= len(stages),
                "next_stage":None if completed >= len(stages) else stages[completed],
                "summary":str(arc.get("summary") or "")}
        return state

    def _assigned_narrative_beats(self, index: int,
                                  accepted: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        scene_plan = self.data["outline"].get("scene_plan") or []
        scene = scene_plan[min(index - 1, len(scene_plan) - 1)] if scene_plan else {}
        beats = []
        for event_id, state in self._event_arc_state(accepted).items():
            if not state["closed"]:
                beats.append({"event_id":event_id, "event_summary":state["summary"],
                    "event_stage_index":state["completed_stage_count"],
                    "event_stage":state["next_stage"], "scene_id":scene.get("scene_id"),
                    "scene_purpose":scene.get("purpose")})
        return beats

    def _session_prompt(self, index: int, accepted: list[dict[str, Any]], hooks: list[dict[str, str]]) -> str:
        outline = self.data["outline"] if self is not None else {}
        beat = self._assigned_narrative_beats(index, accepted)
        if not beat:
            raise RuntimeError("no assigned event arc remains before stopping conditions are met")
        allowed = list(outline.get("source_parent", {}).get("core_speakers") or []) + [
            x["name"] for x in outline.get("allowed_new_characters") or []
            if int(str(x.get("introduction_scene") or "S999").lstrip("S") or 999) <= index]
        advanced = {event_id for session in accepted for event_id in session.get("advanced_event_arc_ids") or []}
        active_plans = [x for x in outline.get("planned_event_arcs") or [] if x.get("event_id") not in advanced]
        seen_characters = sorted({turn.get("speaker") for session in accepted
                                  for turn in session.get("dialogue_turns") or [] if turn.get("speaker")})
        continuity_summary = {"accepted_session_count":len(accepted),
            "advanced_event_arc_ids":sorted(advanced), "active_hook_ids":[x["hook_id"] for x in hooks],
            "all_seen_character_names":seen_characters,
            "prior_session_locations":[x.get("session_location") for x in accepted if x.get("session_location")]} 
        outline_text = canonical_json(outline).casefold()
        durable_subjects = list(allowed)
        durable_subjects.extend(phrase for phrase in ("application", "priorities brief", "brief", "report", "plan")
                                if phrase in outline_text)
        return build_realizer_prompt(session_index=index, narrative_beats=beat,
            recent_sessions=accepted[-2:], continuity_summary=continuity_summary,
            active_plans=active_plans, pending_decisions=hooks, superseded_information=[],
            character_knowledge={"all_seen_character_names":seen_characters},
            visibility_boundaries=outline.get("visibility_constraints") or [], unresolved_hooks=hooks,
            allowed_people=allowed, session_location=self._session_location(index),
            allowed_event_arc_ids=[x["event_id"] for x in beat],
            forbidden_conflicts=outline.get("forbidden_conflicts") or [],
            parent_anchors=outline.get("parent_anchors") or [],
            allowed_durable_subjects=sorted(set(durable_subjects)),
            durable_predicate_whitelist=sorted(DEFAULT_DURABLE_PREDICATES))

    def realize(self, crash_before_session: int | None = None, request_limit: int | None = None) -> list[dict[str, Any]]:
        accepted = self._accepted_sessions()
        hooks = self._active_hooks(accepted)
        child_turn_count = sum(len(x["dialogue_turns"]) for x in accepted)
        hook_count = sum(len(x.get("new_unresolved_hooks") or []) for x in accepted)
        if accepted and accepted[-1].get("event_arcs_closed"):
            existing_child = [turn for session in accepted for turn in session["dialogue_turns"]]
            if self._tokens(existing_child) >= 2000:
                return accepted
        requests_this_invocation = 0
        for index in range(len(accepted) + 1, int(self.config["max_sessions"]) + 1):
            if crash_before_session == index:
                raise SimulatedCrash(f"simulated crash before session {index}")
            assigned_beats = self._assigned_narrative_beats(index, accepted)
            prompt = self._session_prompt(index, accepted, hooks)
            accepted_this_session = False
            for attempt in range(1, self.config["max_session_attempts_per_session"] + 1):
                request_id = f"realizer-s{index:02d}-a{attempt:02d}"
                request = RealizerRequest(request_id, index, attempt, prompt, _hash_text(prompt))
                prompt_path = self.run_root / f"raw/prompts/{request_id}.json"
                if prompt_path.exists():
                    if _read_json(prompt_path).get("prompt_sha256") != request.prompt_sha256:
                        raise RuntimeError(f"prompt drift for {request_id}")
                else:
                    atomic_json(prompt_path, {"request_id": request_id, "session_index": index,
                                              "attempt": attempt, "prompt_sha256": request.prompt_sha256,
                                              "prompt": prompt})
                response = self.ledger.invoke(request_id=request_id, intent="realize_session", attempt=attempt,
                    prompt_sha256=request.prompt_sha256, call=lambda r=request: self.realizer.generate_session(r),
                    budget_class="realizer", budget_limit=self.config["max_realizer_generations"],
                    request_metadata={"session_index":index, "attempt":attempt,
                        "session_location":self._session_location(index)},
                    backend_identity=self._backend_identity(self.realizer),
                    total_budget_limit=self.config.get("max_total_paid_api_generations") if self.ledger.backend_mode == "live" else None)
                # request_limit is a limit on physical calls made by this process.  A
                # completed response recovered from the ledger is intentionally free:
                # otherwise resume can consume its whole allowance replaying an already
                # rejected attempt and never reach the first unfinished attempt.
                if not response.metadata.get("reused", False):
                    requests_this_invocation += 1
                raw_record = {"request_id": request_id, "session_index": index, "attempt": attempt,
                              "raw_response": response.raw_text}
                atomic_json(self.run_root / f"raw/session_{index:02d}_attempt_{attempt:02d}.json", raw_record)
                errors: list[str] = []
                try:
                    parsed = json.loads(response.raw_text)
                except json.JSONDecodeError:
                    parsed, errors = {}, ["realizer_json_invalid"]
                if not isinstance(parsed, dict):
                    parsed, errors = {}, ["realizer_schema_invalid"]
                turns = parsed.get("dialogue_turns") or []
                outline = self.data["outline"]
                errors.extend(validate_realizer_output_schema(parsed))
                if not parsed.get("advanced_event_arc_ids"):
                    errors.append("event_arc_not_advanced")
                allowed_hook_ids = {x.get("hook_id") for x in hooks if isinstance(x, dict)}
                continued = parsed.get("continued_unresolved_hook_ids") or []
                resolved_hook_ids = parsed.get("resolved_hook_ids") or []
                errors.extend(validate_realizer_identity_contract(parsed, outline, index, hooks))
                arc_state_before = self._event_arc_state(accepted)
                assigned_arc_ids = {str(x["event_id"]) for x in assigned_beats}
                closed_arc_ids = {event_id for event_id, state in arc_state_before.items() if state["closed"]}
                beat_support = {str(x["event_id"]):f"{x.get('event_summary','')} {x.get('event_stage','')} {x.get('scene_purpose','')}"
                                for x in assigned_beats}
                errors.extend(validate_event_arc_advancement(
                    parsed.get("advanced_event_arc_ids") or [], assigned_arc_ids,
                    closed_arc_ids, turns, beat_support))
                new_hooks = parsed.get("new_unresolved_hooks") or []
                if any(not isinstance(x, dict) or set(x) != {"description"} or not str(x["description"]).strip()
                       for x in new_hooks):
                    errors.append("bad_new_unresolved_hook_schema")
                errors.extend(validate_new_hook_descriptions(new_hooks, hooks))
                errors.extend(validate_durable_candidate_indices(
                    parsed.get("durable_assertion_candidates") or [], len(turns)))
                if index == 1 and not parsed.get("used_parent_anchor_ids"):
                    errors.append("session_1_missing_anchor")
                if index > 1 and not (set(continued) | set(resolved_hook_ids)).intersection(allowed_hook_ids):
                    errors.append("unresolved_hook_not_continued_or_resolved")
                prospective_arc_state = self._event_arc_state([*accepted,
                    {"advanced_event_arc_ids":parsed.get("advanced_event_arc_ids") or []}])
                all_arcs_closed_after = bool(prospective_arc_state) and all(
                    state["closed"] for state in prospective_arc_state.values())
                prospective_turns = [turn for session in accepted for turn in session["dialogue_turns"]] + turns
                prospective_tokens = self._tokens(prospective_turns)
                can_stop_after = prospective_tokens >= 2000 and all_arcs_closed_after
                transition = validate_hook_transition(set(allowed_hook_ids), set(continued),
                    set(resolved_hook_ids), [f"NEW{position}" for position in range(len(new_hooks))],
                    can_stop_after=can_stop_after)
                errors.extend(transition["errors"])
                if all_arcs_closed_after and prospective_tokens < 2000:
                    errors.append("event_arcs_closed_before_min_tokens")
                if errors:
                    self.ledger.record_decision(request_id, "reject:" + ",".join(sorted(set(errors))))
                    _append_unique_jsonl(self.run_root / "rejects.jsonl",
                        {"request_id": request_id, "session_index": index, "attempt": attempt,
                         "reasons": sorted(set(errors)), "final_decision": "reject"}, "request_id")
                    if request_limit is not None and requests_this_invocation >= request_limit:
                        return accepted
                    continue
                assigned = []
                for turn in turns:
                    child_turn_count += 1
                    assigned.append({**turn, "turn_id": f"C{child_turn_count:04d}",
                                     "session_id": f"session_{index:02d}"})
                assigned_hooks = []
                for hook in new_hooks:
                    hook_count += 1
                    assigned_hooks.append({"hook_id":f"H{hook_count:04d}",
                                           "description":hook["description"]})
                parsed["dialogue_turns"] = assigned
                parsed["new_unresolved_hooks"] = assigned_hooks
                parsed["session_location"] = self._session_location(index)
                parsed.update(derive_local_character_ledger(parsed, self.data["outline"], accepted))
                arc_state_after = self._event_arc_state([*accepted, parsed])
                parsed["event_arc_state_after"] = arc_state_after
                parsed["event_arcs_closed"] = bool(arc_state_after) and all(
                    state["closed"] for state in arc_state_after.values())
                self.ledger.record_decision(request_id, "accept_session")
                parsed.update({"session_id": f"session_{index:02d}", "session_index": index,
                               "accepted_attempt": attempt, "request_id": request_id,
                               "final_decision": "accept"})
                accepted_path = self.run_root / f"accepted/session_{index:02d}.json"
                atomic_json(accepted_path, parsed)
                accepted_index_path = self.run_root / "accepted/index.json"
                accepted_index = (_read_json(accepted_index_path) if accepted_index_path.exists()
                                  else {"schema_version":"accepted_session_integrity_v2",
                                        "run_id":self.manifest["run_id"],
                                        "run_manifest_sha256":self.manifest_sha,
                                        "session_sha256":{}})
                accepted_index["session_sha256"][accepted_path.name] = sha256_file(accepted_path)
                atomic_json(accepted_index_path, accepted_index)
                accepted.append(parsed)
                hooks = self._active_hooks(accepted)
                atomic_json(self.run_root / "continuity_state.json",
                    {"accepted_session_count": len(accepted), "active_plans": [],
                     "pending_decisions": hooks, "superseded_information": [],
                     "character_knowledge": {"all_seen_character_names": parsed["all_seen_character_names"]},
                     "visibility_boundaries": []})
                atomic_json(self.run_root / "unresolved_hooks.json", hooks)
                _append_unique_jsonl(self.run_root / "realization_ledger.jsonl",
                    {"request_id": request_id, "session_index": index, "attempt": attempt,
                     "final_decision": "accept", "turn_count": len(assigned)}, "request_id")
                accepted_this_session = True
                child = [t for s in accepted for t in s["dialogue_turns"]]
                if parsed.get("event_arcs_closed") and self._tokens(child) >= 2000:
                    return accepted
                if request_limit is not None and requests_this_invocation >= request_limit:
                    return accepted
                break
            if not accepted_this_session:
                raise RuntimeError(f"session {index} exhausted attempts")
        return accepted

    def freeze(self, sessions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        parent_before = copy.deepcopy(self.data["parent"]["dialogue"]["turns"])
        child = [turn for session in sessions for turn in session["dialogue_turns"]]
        parent_after = copy.deepcopy(self.data["parent"]["dialogue"]["turns"])
        parent_unchanged = canonical_json(parent_before) == canonical_json(parent_after)
        child_tokens = self._tokens(child)
        errors = validate_dialogue(parent_before, child, child_tokens, sessions)
        if not parent_unchanged:
            errors.append("parent_modified")
        frozen_child = {"schema_version": "frozen_child_dialogue_v1", "turns": child,
                        "session_count": len(sessions), "child_tokens": child_tokens}
        frozen_all = {"schema_version": "frozen_parent_plus_child_v1",
                      "parent_turns": parent_after, "child_turns": child}
        atomic_json(self.run_root / "frozen_child_dialogue.json", frozen_child)
        atomic_json(self.run_root / "frozen_parent_plus_child.json", frozen_all)
        parent_tokens = self._tokens(parent_after)
        atomic_json(self.run_root / "token_length_report.json",
            {"parent_tokens": parent_tokens, "child_tokens": child_tokens,
             "child_token_delta": child_tokens, "tokenizer_path": self.manifest["tokenizer_path"]})
        report = {"decision": "pass" if not errors else "reject", "errors": errors,
                  "parent_turn_count_before": len(parent_before), "parent_turn_count_after": len(parent_after),
                  "parent_unchanged": parent_unchanged, "child_token_delta": child_tokens,
                  "no_truncation": True}
        atomic_json(self.run_root / "dialogue_validation_report.json", report)
        if errors:
            raise RuntimeError(f"dialogue freeze failed: {errors}")
        memory = self._durable_memory(sessions)
        atomic_json(self.run_root / "durable_memory.json", memory)
        return child, memory

    def _durable_memory(self, sessions: list[dict[str, Any]]) -> dict[str, Any]:
        """Materialize Realizer candidates against accepted local turn IDs; no extra model call."""
        child = [turn for session in sessions for turn in session.get("dialogue_turns") or []]
        frozen = {x["turn_id"]: x["text"] for x in child}
        candidates: list[dict[str, Any]] = []
        counter = 0
        for session in sessions:
            turns = session.get("dialogue_turns") or []
            for source in session.get("durable_assertion_candidates") or []:
                counter += 1
                indices = source.get("supporting_turn_indices") or []
                support_turns = [turns[index] for index in indices
                                 if isinstance(index, int) and 0 <= index < len(turns)]
                value = str(source.get("object_or_state") or "")
                spans = [{"turn_id":turn["turn_id"], "exact_span":turn.get("text", "")}
                         for turn in support_turns]
                candidates.append({"assertion_id":f"mem{counter:04d}",
                    "subject":source.get("subject"), "predicate":source.get("predicate"),
                    "object_or_state":source.get("object_or_state"), "polarity":"positive",
                    "temporal_scope":session.get("session_id"),
                    "supporting_turn_ids":[turn["turn_id"] for turn in support_turns],
                    "supporting_text_spans":spans, "status":"active",
                    "supersedes_assertion_ids":[],
                    "extraction_source":"realizer_candidate_local_materialization",
                    "source_session_id":session.get("session_id")})
        results, accepted = [], []
        outline = self.data["outline"] if self is not None else {}
        known_subjects = set(outline.get("source_parent", {}).get("core_speakers") or []) | {
            str(x.get("name")) for x in outline.get("allowed_new_characters") or [] if x.get("name")}
        outline_text = canonical_json(outline).casefold()
        known_subjects.update(phrase for phrase in ("application", "priorities brief", "brief", "report", "plan")
                              if phrase in outline_text)
        for candidate in candidates:
            conflicts = [prior for prior in accepted if prior.get("status") == "active"
                and normalize_option(prior.get("subject", "")) == normalize_option(candidate.get("subject", ""))
                and normalize_option(prior.get("predicate", "")) == normalize_option(candidate.get("predicate", ""))
                and prior.get("object_or_state") != candidate.get("object_or_state")]
            candidate["supersedes_assertion_ids"] = [x["assertion_id"] for x in conflicts]
            errors = validate_memory(candidate, frozen, accepted, known_subjects=known_subjects,
                                     predicate_whitelist=set(DEFAULT_DURABLE_PREDICATES))
            row = {**candidate, "validation_status": "accepted" if not errors else "rejected",
                   "reject_reason": ";".join(errors)}
            results.append(row)
            if not errors:
                if candidate["supersedes_assertion_ids"]:
                    for prior in accepted:
                        if prior["assertion_id"] in candidate["supersedes_assertion_ids"]:
                            prior["status"] = "superseded"
                accepted.append(row)
        return {"assertions": results, "accepted_count": sum(x["validation_status"] == "accepted" for x in results),
                "rejected_count": sum(x["validation_status"] == "rejected" for x in results)}

    def question(self, child: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        parent = self.data["parent"]["dialogue"]["turns"]
        frozen_dialogue = parent + child
        valid_turn_ids = tuple(x["turn_id"] for x in frozen_dialogue)
        frozen_turns = {x["turn_id"]: x["text"] for x in frozen_dialogue}
        turn_session_ids = {x["turn_id"]: str(x.get("session_id") or x["turn_id"].split(":", 1)[0])
                            for x in frozen_dialogue}
        valid_session_ids = tuple(sorted(set(turn_session_ids.values())))
        schedules = [("A","B","C"), ("D","A","B"), ("C","D","A"), ("B","C","D")]
        parsed_batches, accepted, rejected = [], [], []
        seen_questions: set[str] = set()
        seen_qa_ids: set[str] = set()
        seen_option_sets: set[tuple[str, ...]] = set()
        seen_reasoning_types: set[str] = set()
        for batch_index, letters in enumerate(schedules):
            prompt = build_questioner_prompt(frozen_dialogue=frozen_dialogue, valid_turn_ids=valid_turn_ids,
                valid_session_ids=valid_session_ids, assigned_gold_letters=letters,
                recent_frontier_history=[], batch_index=batch_index)
            request_id = f"questioner-b{batch_index:02d}"
            request = QuestionerRequest(request_id, batch_index, letters, prompt, _hash_text(prompt),
                                        valid_turn_ids, valid_session_ids)
            prompt_path = self.run_root / f"raw/prompts/{request_id}.json"
            if prompt_path.exists():
                if _read_json(prompt_path).get("prompt_sha256") != request.prompt_sha256:
                    raise RuntimeError(f"questioner prompt drift for {request_id}")
            else:
                atomic_json(prompt_path, {"request_id":request_id, "batch_id":batch_index,
                    "prompt_sha256":request.prompt_sha256, "assigned_gold_letters":list(letters),
                    "prompt":prompt})
            response = self.ledger.invoke(request_id=request_id, intent="questioner_batch", attempt=1,
                prompt_sha256=request.prompt_sha256, call=lambda r=request: self.questioner.generate_batch(r),
                budget_class="questioner", budget_limit=self.config["max_questioner_generations"],
                request_metadata={"batch_id":batch_index, "assigned_gold_letters":list(letters),
                    "valid_turn_ids_sha256":_hash_text(canonical_json(list(valid_turn_ids))),
                    "valid_session_ids_sha256":_hash_text(canonical_json(list(valid_session_ids)))},
                backend_identity=self._backend_identity(self.questioner),
                total_budget_limit=self.config.get("max_total_paid_api_generations") if self.ledger.backend_mode == "live" else None)
            atomic_json(self.run_root / f"raw/questioner_batch_{batch_index:02d}.json",
                        {"request_id": request_id, "raw_response": response.raw_text})
            try:
                batch = json.loads(response.raw_text)
            except json.JSONDecodeError:
                batch = None
            parsed_batches.append(batch)
            if not isinstance(batch, list) or len(batch) != 3:
                self.ledger.record_decision(request_id, "questioner_batch_count_invalid")
                rejected.append({"batch_index": batch_index, "reason": "questioner_batch_count_invalid"})
                continue
            batch_result = validate_questioner_smoke_batch(batch, list(letters), set(valid_turn_ids),
                set(valid_session_ids), frozen_turns, turn_session_ids)
            self.ledger.record_decision(request_id, batch_result["batch_decision"])
            for item_result in batch_result["items"]:
                item_index = item_result["item_index"]
                qa = item_result.get("materialized_qa")
                errors = list(item_result["reject_reasons"])
                key = normalize_option(qa.get("question", "")) if qa else ""
                qa_id = str(qa.get("qa_id", "")) if qa else ""
                option_set = tuple(sorted(normalize_option(x) for x in qa.get("options", {}).values())) if qa else ()
                reasoning_type = normalize_option(qa.get("reasoning_type", "")) if qa else ""
                if key in seen_questions or qa_id in seen_qa_ids or option_set in seen_option_sets or reasoning_type in seen_reasoning_types:
                    errors.append("cross_batch_duplicate")
                if errors:
                    rejected.append({"batch_index": batch_index, "item_index": item_index,
                                     "qa_id": qa_id or item_result.get("qa_id"),
                                     "reason": sorted(set(errors))})
                else:
                    seen_questions.add(key); seen_qa_ids.add(qa_id)
                    seen_option_sets.add(option_set); seen_reasoning_types.add(reasoning_type)
                    accepted.append({**qa, "questioner_batch_index": batch_index})
        batching = parse_questioner_batches(parsed_batches)
        admission = assess_candidate_shortfall(batching["generation_counts"], len(accepted),
            minimum_solver_eligible_candidates=int(self.config["minimum_solver_eligible_candidates"]))
        summary = {**batching, **admission, "physical_questioner_calls": 4,
                   "schema_accepted_count": len(accepted),
                   "questioner_schema_accepted_count":len(accepted), "item_rejects": rejected,
                   "questioner_candidate_shortfall":admission["candidate_shortfall"]}
        atomic_json(self.run_root / "questioner_summary.json", summary)
        atomic_json(self.run_root / "qa_candidates_accepted.json", accepted)
        atomic_json(self.run_root / "qa_candidates_rejected.json", rejected)
        return accepted, summary

    def question_smoke(self, frozen_dialogue: list[dict[str, Any]],
                       valid_turn_ids: tuple[str, ...], valid_session_ids: tuple[str, ...],
                       assigned_letters: tuple[str, str, str]) -> dict[str, Any]:
        """Execute exactly one live Questioner batch; never schedules filtering or another batch."""
        prompt = build_questioner_prompt(frozen_dialogue=frozen_dialogue,
            valid_turn_ids=valid_turn_ids, valid_session_ids=valid_session_ids,
            assigned_gold_letters=assigned_letters, recent_frontier_history=[], batch_index=0)
        request_id = "questioner-b00"
        request = QuestionerRequest(request_id, 0, assigned_letters, prompt, _hash_text(prompt),
                                    valid_turn_ids, valid_session_ids)
        prompt_path = self.run_root / "raw/prompts/questioner-b00.json"
        if prompt_path.exists():
            if _read_json(prompt_path).get("prompt_sha256") != request.prompt_sha256:
                raise RuntimeError("questioner smoke prompt drift")
        else:
            atomic_json(prompt_path, {"request_id":request_id, "batch_id":0,
                "prompt_sha256":request.prompt_sha256, "assigned_gold_letters":list(assigned_letters),
                "prompt":prompt})
        response = self.ledger.invoke(request_id=request_id, intent="questioner_batch", attempt=1,
            prompt_sha256=request.prompt_sha256, call=lambda: self.questioner.generate_batch(request),
            budget_class="questioner", budget_limit=self.config["max_questioner_generations"],
            request_metadata={"batch_id":0, "assigned_gold_letters":list(assigned_letters),
                "valid_turn_ids_sha256":_hash_text(canonical_json(list(valid_turn_ids))),
                "valid_session_ids_sha256":_hash_text(canonical_json(list(valid_session_ids)))},
            backend_identity=self._backend_identity(self.questioner),
            total_budget_limit=self.config.get("max_total_paid_api_generations") if self.ledger.backend_mode == "live" else None)
        atomic_json(self.run_root / "raw/questioner_batch_00.json",
                    {"request_id":request_id, "raw_response":response.raw_text})
        try:
            parsed = json.loads(response.raw_text)
            parse_status = "json_parsed"
            result = validate_questioner_smoke_batch(parsed, list(assigned_letters),
                set(valid_turn_ids), set(valid_session_ids),
                {x["turn_id"]: x["text"] for x in frozen_dialogue},
                {x["turn_id"]:str(x.get("session_id") or x["turn_id"].split(":", 1)[0])
                 for x in frozen_dialogue})
        except json.JSONDecodeError:
            parse_status = "json_invalid"
            result = {"json_top_level_array":False, "raw_object_count":0,
                "batch_decision":"questioner_json_invalid", "questioner_candidate_shortfall":True,
                "items":[]}
        result.update({"parse_status":parse_status, "request_id":request_id,
            "physical_questioner_calls":1, "questioner_smoke_only":True,
            "not_full_canary_context":True})
        atomic_json(self.run_root / "reports/questioner_materialized_items.json",
            [x.get("materialized_qa") for x in result.get("items", [])])
        self.ledger.record_decision(request_id, result["batch_decision"])
        atomic_json(self.run_root / "reports/live_questioner_smoke.json", result)
        return result

    def _rollouts(self, qa: dict[str, Any], mode: str, order_id: str, requested: int,
                  options: dict[str, str], slot_start: int = 0) -> dict[str, Any]:
        parent = self.data["parent"]["dialogue"]["turns"]
        child_path = self.run_root / "frozen_child_dialogue.json"
        child = _read_json(child_path)["turns"] if child_path.exists() else []
        evidence_ids = set(qa.get("evidence_turn_ids") or [])
        if mode == "no_context":
            context_turns = []
        elif mode == "evidence_only":
            context_turns = [x for x in parent + child if x.get("turn_id") in evidence_ids]
        elif mode == "evidence_masked":
            context_turns = [x for x in parent + child if x.get("turn_id") not in evidence_ids]
        else:
            context_turns = parent + child
        qa_text = qa["question"] + "\n" + "\n".join(f"{k}. {v}" for k,v in options.items())
        context_text = "\n".join(f"[{x.get('turn_id','')}] {x['speaker']}: {x['text']}" for x in context_turns)
        user_text = (f"Context:\n{context_text}\n\n" if context_text else "") + qa_text
        messages = [{"role":"system", "content":SOLVER_NOTHINK_RESPONDER_PROMPT},
                    {"role":"user", "content":user_text}]
        solver_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        thinking_disabled_marker = solver_prompt.endswith("<think>\n\n</think>\n\n")
        if not thinking_disabled_marker or solver_prompt.count("<think>") != solver_prompt.count("</think>"):
            raise RuntimeError("solver chat template did not render the expected pre-closed no-thinking block")
        rendered_prompt_tokens = len(self.tokenizer.encode(solver_prompt, add_special_tokens=False))
        reserved = self.config["reserved_generation_tokens"]
        preflight = {"rendered_prompt_tokens":rendered_prompt_tokens,
            "reserved_generation_tokens":reserved,
            "total_tokens":rendered_prompt_tokens + reserved,
            "solver_max_model_len":self.manifest["solver_max_model_len"],
            "decision":"pass" if rendered_prompt_tokens + reserved <= self.manifest["solver_max_model_len"] else "reject_context_overflow",
            "truncated":False}
        if preflight["decision"] != "pass":
            return {"requested_valid_rollouts": requested, "obtained_valid_rollouts": 0,
                "raw_generations": 0, "parse_invalid_generations": 0, "retry_generations": 0,
                "exhausted_target_slots": requested, "verified_rewards": [], "rollouts": [],
                "context_preflight": preflight}
        records, rewards = [], []
        invalid = retries = exhausted = 0
        for slot in range(slot_start, slot_start + requested):
            valid = False
            for retry in range(self.config["max_parse_retries_per_target"] + 1):
                seed = int(hashlib.sha256(f"{qa['qa_id']}:{mode}:{order_id}:{slot}:{retry}".encode()).hexdigest()[:8], 16)
                prompt_material = canonical_json({"qa":qa["qa_id"],"mode":mode,"order":order_id,
                                                   "slot":slot,"options":options})
                request_id = f"solver-{qa['qa_id']}-{mode}-{order_id}-s{slot:02d}-r{retry}"
                request = SolverRequest(request_id, qa["qa_id"], mode, order_id, slot, retry, seed,
                                        options, qa["canonical_answer_text"], _hash_text(solver_prompt), solver_prompt)
                prompt_path = self.run_root / f"raw/prompts/{request_id}.json"
                if prompt_path.exists():
                    if _read_json(prompt_path).get("prompt_sha256") != request.prompt_sha256:
                        raise RuntimeError(f"solver prompt drift for {request_id}")
                else:
                    atomic_json(prompt_path, {"request_id":request_id, "prompt_sha256":request.prompt_sha256,
                        "enable_thinking":False, "add_generation_prompt":True,
                        "thinking_disabled_marker":"preclosed_empty_think_block",
                        "contains_unclosed_think_tag":False, "prompt":solver_prompt})
                response = self.ledger.invoke(request_id=request_id, intent=f"solver_{mode}", attempt=retry + 1,
                    prompt_sha256=request.prompt_sha256, call=lambda r=request: self.solver.generate(r),
                    budget_class="local_solver", budget_limit=self.config.get("max_local_solver_generations"),
                    request_metadata={"qa_id":qa["qa_id"], "evaluation_mode":mode,
                        "permutation_id":order_id, "target_slot":slot, "retry_index":retry,
                        "seed":seed, "options_sha256":_hash_text(canonical_json(options))},
                    backend_identity=self._backend_identity(self.solver))
                parsed = parse_boxed_mcq_answer(response.raw_text, options)
                reward = local_reward(parsed, qa["canonical_answer_text"])
                row = {"request_id": request_id, "raw_response": response.raw_text, **parsed,
                       "local_reward": reward, "sampling_seed": seed, "target_valid_slot": slot,
                       "parse_retry_index": retry, "evaluation_mode": mode, "order_id": order_id}
                self.ledger.record_decision(request_id,
                    "parse_invalid" if reward is None else f"valid_reward_{reward}")
                records.append(row)
                if reward is not None:
                    rewards.append(reward); valid = True; break
                invalid += 1
                if retry < self.config["max_parse_retries_per_target"]:
                    retries += 1
            if not valid:
                exhausted += 1
        return {"requested_valid_rollouts": requested, "obtained_valid_rollouts": len(rewards),
            "raw_generations": len(records), "parse_invalid_generations": invalid,
            "retry_generations": retries, "exhausted_target_slots": exhausted,
            "verified_rewards": rewards, "rollouts": records, "context_preflight": preflight}

    @staticmethod
    def _combine(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        keys = ("requested_valid_rollouts","obtained_valid_rollouts","raw_generations",
                "parse_invalid_generations","retry_generations","exhausted_target_slots")
        return {**{k:left[k]+right[k] for k in keys},
                "verified_rewards": left["verified_rewards"]+right["verified_rewards"],
                "rollouts": left["rollouts"]+right["rollouts"],
                "context_preflight": left["context_preflight"]}

    @staticmethod
    def _permuted(options: dict[str, str], shift: int) -> dict[str, str]:
        letters = "ABCD"; values = [options[x] for x in letters]
        return {letter: values[(i - shift) % 4] for i, letter in enumerate(letters)}

    def filter(self, candidates: list[dict[str, Any]], shortfall: bool) -> list[dict[str, Any]]:
        if shortfall:
            atomic_json(self.run_root / "filter_results.json", [])
            return []
        results = []
        for qa in candidates:
            result: dict[str, Any] = {"qa_id": qa["qa_id"], "difficulty_level": qa["difficulty_level"]}
            initial = self._rollouts(qa, "no_context", "original", 3, qa["options"])
            ground = grounding_decision(initial["verified_rewards"])
            no_context = initial
            if ground["needs_extension"]:
                extra = self._rollouts(qa, "no_context", "original", 2, qa["options"], 3)
                no_context = self._combine(initial, extra)
                ground = grounding_decision(initial["verified_rewards"], extra["verified_rewards"])
            result.update({"no_context": no_context, "grounding_decision": ground["decision"]})
            if ground["decision"] != "pass_grounding":
                result["final_decision"] = ground["decision"]; results.append(result); continue
            full = self._rollouts(qa, "full_context", "original", 5, qa["options"])
            frontier = classify_reward_group(full["verified_rewards"])
            result.update({"full_context": full, "verified_rewards": full["verified_rewards"],
                           "verified_reward_mean": frontier["mean"], "verified_reward_std": frontier["std"],
                           "frontier_decision": frontier["decision"]})
            if frontier["decision"] != "retain_frontier":
                result["final_decision"] = frontier["decision"]; results.append(result); continue
            if qa["difficulty_level"] in ("L2", "L3"):
                evidence_only = self._rollouts(qa, "evidence_only", "original", 3, qa["options"])
                masked = self._rollouts(qa, "evidence_masked", "original", 3, qa["options"])
                evidence = evidence_audit(qa, evidence_only, masked)
                result.update({"evidence_only":evidence_only, "evidence_masked":masked,
                               "evidence_audit":evidence})
                if evidence["decision"] == "downgrade_l1":
                    result["difficulty_level"] = "L1"
                elif evidence["decision"] == "bad_evidence_relevance":
                    result["final_decision"] = "bad_evidence_relevance"; results.append(result); continue
                elif evidence["decision"] == "evidence_audit_inconclusive":
                    result["final_decision"] = "evidence_audit_inconclusive"; results.append(result); continue
            orders = [{"order_id":"original", "valid_rewards":full["verified_rewards"],
                       "gold_letter":qa["reference_answer_letter"], "context_hash":"frozen"}]
            permutation_records = []
            for shift, order_id in ((1,"perm1"),(2,"perm2")):
                options = self._permuted(qa["options"], shift)
                rollout = self._rollouts(qa, "option_permutation", order_id, 5, options)
                gold = next(k for k,v in options.items() if normalize_option(v) == normalize_option(qa["canonical_answer_text"]))
                orders.append({"order_id":order_id,"valid_rewards":rollout["verified_rewards"],
                               "gold_letter":gold,"context_hash":"frozen"})
                permutation_records.append({"order_id":order_id,"options":options,"rollout":rollout})
            audit = option_audit(orders)
            result.update({"option_orders":permutation_records, "option_audit":audit,
                           "final_decision":"bad_option_bias" if audit["decision"] == "bad_option_bias" else "retain_frontier"})
            results.append(result)
            atomic_json(self.run_root / f"filter/{qa['qa_id']}.json", result)
        atomic_json(self.run_root / "filter_results.json", results)
        return results

    def report(self, sessions: list[dict[str, Any]], memory: dict[str, Any], q_summary: dict[str, Any],
               filter_results: list[dict[str, Any]]) -> dict[str, Any]:
        decisions = Counter(x.get("final_decision") for x in filter_results)
        option_decisions = Counter((x.get("option_audit") or {}).get("decision") for x in filter_results)
        evidence_status = Counter((x.get("evidence_audit") or {}).get("status") for x in filter_results)
        accounting = Counter()
        for result in filter_results:
            for stage in ("no_context","full_context","evidence_only","evidence_masked"):
                if stage in result:
                    for key in ("requested_valid_rollouts","obtained_valid_rollouts","raw_generations",
                                "parse_invalid_generations","retry_generations","exhausted_target_slots"):
                        accounting[f"{stage}_{key}"] += result[stage][key]
            for order in result.get("option_orders") or []:
                for key in ("requested_valid_rollouts","obtained_valid_rollouts","raw_generations",
                            "parse_invalid_generations","retry_generations","exhausted_target_slots"):
                    accounting[f"permutation_{key}"] += order["rollout"][key]
        ledger = self.ledger.summary()
        report = {"run_id":self.manifest["run_id"], "backend_mode":"mock",
            "session_count":len(sessions), "session_retry_count":sum(x["accepted_attempt"]-1 for x in sessions),
            "child_token_delta":_read_json(self.run_root/"token_length_report.json")["child_token_delta"],
            "parent_unchanged":_read_json(self.run_root/"dialogue_validation_report.json")["parent_unchanged"],
            "questioner_physical_calls":q_summary["physical_questioner_calls"],
            "questioner_batch_counts":q_summary["generation_counts"],
            "questioner_raw_candidate_count":q_summary["raw_candidate_count"],
            "questioner_schema_accepted_count":q_summary["schema_accepted_count"],
            "questioner_solver_eligible_count":q_summary["questioner_solver_eligible_count"],
            "minimum_solver_eligible_candidates":q_summary["minimum_solver_eligible_candidates"],
            "raw_candidate_count_shortfall":q_summary["raw_candidate_count_shortfall"],
            "accepted_candidate_shortfall":q_summary["accepted_candidate_shortfall"],
            "questioner_candidate_shortfall":q_summary["questioner_candidate_shortfall"],
            "durable_memory_accepted":memory["accepted_count"], "durable_memory_rejected":memory["rejected_count"],
            "decisions":dict(decisions), "option_audit_decisions":dict(option_decisions),
            "evidence_audit_status":dict(evidence_status), "valid_rollout_accounting":dict(accounting),
            **ledger, "local_solver_generation_count":sum(1 for x in self.ledger.completed() if x["intent"].startswith("solver_")),
            "verifier_api_generation_count":0, "vllm_generation_count":0, "training_runs":0,
            "completed":not q_summary["candidate_shortfall"]}
        cumulative = 0
        first_min = None
        first_closed = None
        for session in sessions:
            cumulative += self._tokens(session["dialogue_turns"])
            if first_min is None and cumulative >= 2000:
                first_min = session["session_index"]
            if first_closed is None and session.get("event_arcs_closed"):
                first_closed = session["session_index"]
        report.update({"first_session_reaching_min_tokens":first_min,
            "first_session_with_all_arcs_closed":first_closed,
            "stop_decision_session":sessions[-1]["session_index"] if sessions else None,
            "soft_upper_overshoot_tokens":max(0, report["child_token_delta"] - 4000)})
        atomic_json(self.run_root / "reports/mock_e2e_report.json", report)
        return report

    def run(self, crash_before_session: int | None = None) -> dict[str, Any]:
        sessions = self.realize(crash_before_session)
        child, memory = self.freeze(sessions)
        candidates, q_summary = self.question(child)
        results = self.filter(candidates, q_summary["candidate_shortfall"])
        return self.report(sessions, memory, q_summary, results)
