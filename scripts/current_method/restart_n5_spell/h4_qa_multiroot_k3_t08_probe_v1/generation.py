"""Fresh root-specific intent, contract, and QA candidate generation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import (
    build_prompt as build_guided_prompt,
    build_repair_prompt as build_guided_repair,
)
from ..h4_qa_mining_hybrid_generation_v1.free_exploration_generator import (
    build_prompt as build_free_prompt,
    build_repair_prompt as build_free_repair,
)
from ..h4_qa_mining_probe_v1.intent_contract import canonicalize_intents
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_qa_mining_probe_v1.reasoning_contract import (
    build_reasoning_contract,
    contract_has_three_disjoint_groups,
)
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict

from .permutation import audit_gold_rebinding, position_variants
from .validator import validate_batch


GENERATOR_TEMPERATURE = 0.1
GENERATOR_VERSION = "h4_qa_multiroot_k3_t08_fresh_generation_v1"

MINIMUM_COUNTS = {
    "single_choice": 2,
    "multi_select": 1,
    "ordering": 1,
}


def _append_feedback_context(
        prompt_spec: Mapping[str, Any], suffix: str | None,
) -> dict[str, Any]:
    """Append an isolated curriculum section while preserving prompt hashes."""
    if not suffix:
        return dict(prompt_spec)
    prompt = "\n".join((str(prompt_spec["prompt"]), str(suffix)))
    return {
        **dict(prompt_spec),
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode()),
    }


def _call_json(*, client: OpenAICompatibleChallengerClient, ledger: RequestLedger,
               run_root: Path, request_id: str, prompt: str, max_tokens: int,
               intent: str, temperature: float | None = GENERATOR_TEMPERATURE,
               ) -> tuple[Any, str, str | None]:
    prompt_sha = sha256_bytes(prompt.encode())
    atomic_text(run_root / "raw" / "prompts" / f"{request_id}.txt", prompt)
    try:
        response = ledger.invoke(
            request_id=request_id,
            intent=intent,
            attempt=1,
            prompt_sha256=prompt_sha,
            call=lambda: client.complete(prompt=prompt, max_tokens=max_tokens, json_object=True),
            budget_class="fresh_qa_generation",
            budget_limit=10_000,
            total_budget_limit=10_000,
            request_metadata={
                "probe_version": GENERATOR_VERSION,
                "request_id": request_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            backend_identity={
                "transport": "openai_compatible_challenger",
                "model": client.config.model,
                "base_url": client.config.base_url,
                "temperature": temperature,
            },
        )
    except Exception as error:
        return None, "", f"{type(error).__name__}:{error}"
    raw_text = response.raw_text
    atomic_text(run_root / "raw" / "responses" / f"{request_id}.json.txt", raw_text)
    try:
        return load_json_object_strict(raw_text), raw_text, None
    except (json.JSONDecodeError, ValueError) as error:
        return None, raw_text, f"{type(error).__name__}:{error}"


def _prefix_candidate(row: Mapping[str, Any], *, root_id: str,
                      generation_source: str) -> dict[str, Any]:
    item = dict(row)
    old_id = str(item.get("candidate_id") or "")
    item["source_candidate_id"] = old_id
    item["candidate_id"] = f"{root_id}:{old_id}"
    item["root_id"] = root_id
    item["candidate_source"] = "fresh_generated"
    item["generation_source"] = generation_source
    return item


def _accepted_candidates(normalized: Mapping[str, Any], source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the isolated probe's minimum-count rule after structural checks.

    The upstream hybrid normalizer uses exact quotas and therefore quarantines
    an entire batch when a model emits an extra row or one malformed row.  This
    probe intentionally treats those quotas as minima: structurally valid rows
    are retained, invalid rows are reported, and a repair is requested only if
    a type falls below its minimum.
    """
    rows = [dict(row) for row in normalized.get("candidates") or []
            if isinstance(row, Mapping)]
    valid_rows = [row for row in rows if row.get("schema_valid")]
    counts: dict[str, int] = {}
    for row in valid_rows:
        qtype = str(row.get("question_type") or "")
        counts[qtype] = counts.get(qtype, 0) + 1
    missing = {
        qtype: minimum - counts.get(qtype, 0)
        for qtype, minimum in MINIMUM_COUNTS.items()
        if counts.get(qtype, 0) < minimum
    }
    audit = {
        "source": source,
        "minimum_counts": dict(MINIMUM_COUNTS),
        "valid_counts": dict(sorted(counts.items())),
        "valid_candidate_count": len(valid_rows),
        "invalid_candidate_count": len(rows) - len(valid_rows),
        "missing_minimums": missing,
        "accepted": not missing,
        "policy": "minimum_per_question_type_after_structural_validation",
    }
    return valid_rows, audit


def generate_root_intents(*, expanded: Mapping[str, Any], root_dir: Path,
                     root_id: str, api_config: Path, ledger: RequestLedger,
                     client: OpenAICompatibleChallengerClient | None = None,
                     feedback_context: Mapping[str, Any] | None = None,
                     hooks: Any | None = None) -> dict[str, Any]:
    """Run the original shared Miner phase once and return normally.

    No candidate from a historical run is read.  The optional client is only a
    test seam; live callers pass the configured Challenger client.
    """
    if client is None:
        config = load_challenger_config(api_config)
        client = OpenAICompatibleChallengerClient(replace(
            config, temperature=GENERATOR_TEMPERATURE, enable_thinking=False,
            timeout_seconds=max(180, int(config.timeout_seconds))))
    intent_spec = build_intent_prompt(expanded)
    intent_suffix = (
        str(feedback_context.get("intent_suffix") or "")
        if isinstance(feedback_context, Mapping) else ""
    )
    intent_spec = _append_feedback_context(intent_spec, intent_suffix)
    hook_context = {"expanded": expanded, "root_id": root_id, "root_dir": root_dir}
    if hooks is not None:
        intent_spec = hooks.miner_prompt(intent_spec, **hook_context)
    raw_intents, intent_text, intent_error = _call_json(
        client=client, ledger=ledger, run_root=root_dir,
        request_id=f"{root_id}-intent-miner", prompt=intent_spec["prompt"],
        max_tokens=4096, intent="h4_k3_intent_miner",
    )
    atomic_text(root_dir / "raw" / "responses" / f"{root_id}-intent-miner.json.txt", intent_text)
    projected_intents = (hooks.project_miner_response(raw_intents, **hook_context)
                         if hooks is not None else raw_intents)
    intent_result = canonicalize_intents(projected_intents, expanded_dialogue=expanded)
    if intent_error:
        intent_result["errors"] = sorted(set(list(intent_result.get("errors") or []) + [intent_error]))
        intent_result["valid"] = False
    atomic_json(root_dir / "intents.json", {
        "prompt": intent_spec,
        "validation": intent_result,
    })

    contracts: list[dict[str, Any]] = []
    for intent in intent_result.get("intents") or []:
        if not intent.get("valid"):
            continue
        contract = build_reasoning_contract(intent)
        contract["contract_valid"] = contract_has_three_disjoint_groups(contract)
        if contract["contract_valid"]:
            contracts.append(contract)
    atomic_json(root_dir / "reasoning_contracts.json", contracts)
    if hooks is not None:
        hooks.after_intents(raw_intents, intent_result, contracts, **hook_context)

    return {"intent_result": intent_result, "contracts": contracts}


def execute_writer_batch(*, expanded: Mapping[str, Any], contract: Mapping[str, Any],
                         root_dir: Path, root_id: str, ledger: RequestLedger,
                         client: OpenAICompatibleChallengerClient,
                         source: str = "contract_guided", intent_guidance: bool = True,
                         slot_suffix: str = "", hooks: Any | None = None,
                         after_call: Any | None = None,
                         ordering_policy: Mapping[str, Any] | None = None,
                         ordering_context: Mapping[str, Any] | None = None,
                         writer_max_tokens: int = 8192,
                         writer_temperature: float | None = GENERATOR_TEMPERATURE,
                         admission_policy: str | None = None,
                         ) -> dict[str, Any]:
    """Run one Writer batch, including at most one versioned repair.

    Both the standard Guided+Free route and intent ablation use this stage.
    ``after_call`` runs immediately after _call_json, even if that helper
    caught an exception, so callers can stop on uncertain transport state.
    Explicit Writer service budgets apply to both primary and repair; callers
    that omit them retain the historical request parameters.
    """
    if source not in ("contract_guided", "free_exploration"):
        raise ValueError(f"unknown Writer source: {source}")
    if admission_policy not in (None, "legacy", "partial_v1", "partial_grounded_v2"):
        raise ValueError("UNKNOWN_ADMISSION_POLICY")
    partial = admission_policy in ("partial_v1", "partial_grounded_v2")
    grounded = admission_policy == "partial_grounded_v2"
    if partial and source != "contract_guided":
        raise ValueError("PARTIAL_V1_REQUIRES_CONTRACT_GUIDED")
    if grounded != ((ordering_policy or {}).get("version") == "ordering_grounded_options_v2"):
        raise ValueError("GROUNDED_ORDERING_ADMISSION_POLICY_MISMATCH")
    if not intent_guidance:
        if source != "contract_guided" or slot_suffix or hooks is not None:
            raise ValueError("unguided contract Writer requires no suffix or hooks")
        # This projection also removes the normalizer's contract alignment
        # dependency. Candidate-authored variable/evidence requirements stay.
        contract = {"intent_id": str(contract.get("intent_id") or ""),
                    "required_evidence_groups": {}}
    intent_id = str(contract["intent_id"])
    if ordering_policy is not None:
        from ..h4_r02_ordering_temporal_contract_v1.contract import (
            policy_config, load_runtime_context,
        )
        ordering_policy = policy_config(ordering_policy)
        context = dict(ordering_context or {})
        if any(not isinstance(context.get(key), str) or not context[key]
               for key in ("run_id", "arm_id")):
            raise ValueError("ORDERING_TRUSTED_RUN_ARM_REQUIRED")
        if (root_dir / "ordering_runtime_context.json").exists():
            existing = load_runtime_context(root_dir)
            if (existing.get("ordering_policy") != ordering_policy or
                    existing.get("run_id") != context["run_id"] or
                    existing.get("arm_id") != context["arm_id"] or
                    existing.get("root_id") != root_id):
                raise ValueError("ORDERING_RUNTIME_IDENTITY_DRIFT")
    elif (root_dir / "ordering_runtime_context.json").exists():
        raise ValueError("ORDERING_EXPLICIT_POLICY_REQUIRED_FOR_VERSIONED_RUN")
    ordering_args = ({"ordering_policy": ordering_policy}
                     if ordering_policy is not None else {})

    def ordering_identity(stage, rid, text):
        if ordering_policy is None:
            if (root_dir / "ordering_runtime_context.json").exists():
                raise ValueError("ORDERING_EXPLICIT_POLICY_REQUIRED_FOR_VERSIONED_RUN")
            return {}
        from ..h4_r02_ordering_temporal_contract_v1.contract import (
            policy_config, register_runtime_context, load_runtime_context,
        )
        policy = policy_config(ordering_policy)
        context = dict(ordering_context or {})
        if any(not isinstance(context.get(key), str) or not context[key]
               for key in ("run_id", "arm_id")):
            raise ValueError("ORDERING_TRUSTED_RUN_ARM_REQUIRED")
        context.update(root_id=root_id)
        identity = {key: context[key] for key in ("run_id", "arm_id", "root_id")}
        path = root_dir / "ordering_runtime_context.json"
        registry = load_runtime_context(root_dir) if path.exists() else {
            **identity, "ordering_policy": policy, "requests": {}}
        if (registry.get("ordering_policy") != policy or
                any(registry.get(key) != value for key, value in identity.items())):
            raise ValueError("ORDERING_RUNTIME_IDENTITY_DRIFT")
        request = {"intent_id": intent_id, "source": source, "stage": stage,
                   "raw_sha256": sha256_bytes(text.encode())}
        requests = dict(registry.get("requests") or {})
        if rid in requests and requests[rid] != request:
            raise ValueError("ORDERING_REQUEST_IDENTITY_DRIFT")
        requests[rid] = request
        register_runtime_context(root_dir, policy, {**identity, "requests": requests})
        return {"ordering_policy": policy, "ordering_context": {
            **identity, **request, "request_id": rid, "root_dir": root_dir}}

    hook_context = {"expanded": expanded, "root_id": root_id, "root_dir": root_dir}
    all_candidates: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    is_guided = source == "contract_guided"
    # Compile once after mining, by identity. Repair reuses these bytes.
    hook_suffix = (hooks.writer_suffix(contract, source, **hook_context)
                   if hooks is not None else "")
    prompt_spec = (
        (build_guided_prompt(expanded, contract, **ordering_args) if intent_guidance else
         build_guided_prompt(expanded, contract, intent_guidance=False, **ordering_args))
        if is_guided else build_free_prompt(expanded, contract, **ordering_args)
    )
    prompt_spec = _append_feedback_context(prompt_spec, slot_suffix)
    prompt_spec = _append_feedback_context(prompt_spec, hook_suffix)
    request_id = f"{root_id}-{source}-generator-{intent_id}"
    if hooks is not None:
        hooks.writer_request(prompt_spec, contract, source, "primary", request_id,
                             **hook_context)
    raw, raw_text, parse_error = _call_json(
        client=client, ledger=ledger, run_root=root_dir,
        request_id=request_id, prompt=prompt_spec["prompt"],
        max_tokens=writer_max_tokens, intent="h4_k3_candidate_generator",
        temperature=writer_temperature,
    )
    if after_call is not None:
        after_call(stage="primary", request_id=request_id, raw=raw,
                   raw_text=raw_text, parse_error=parse_error)
    identity_args = ordering_identity("primary", request_id, raw_text)
    if partial:
        from ..h4_writer_partial_memory_v1.admission import normalize_stage, admit_stage, repair_required
        normalized = normalize_stage(raw, intent_id=intent_id, source=source,
            contract=contract, expanded=expanded, root_id=root_id, stage="primary",
            request_id=request_id, parse_error=parse_error,
            admission_policy=admission_policy, **identity_args)
        accepted_rows, acceptance = admit_stage(normalized, admission_policy=admission_policy)
    else:
        normalized = normalize_batch(raw, intent_id=intent_id, source=source,
            contract=contract, expanded=expanded, **identity_args)
        if parse_error:
            normalized["errors"] = sorted(set(list(normalized.get("errors") or []) + [parse_error]))
            normalized["valid"] = False
        accepted_rows, acceptance = _accepted_candidates(normalized, source)
    normalized = {**normalized, "minimum_acceptance": acceptance}
    record: dict[str, Any] = {
        "intent_id": intent_id,
        "generation_source": source,
        "request_id": request_id,
        "prompt": prompt_spec,
        "validation": normalized,
    }
    generation_records.append(record)
    needs_repair = (repair_required(raw, normalized, acceptance, expanded=expanded,
                                   admission_policy=admission_policy)
                    if partial else not acceptance["accepted"])
    if needs_repair:
        primary_abstentions = list(normalized.get("effective_abstentions") or []) if grounded else []
        repair_args = {**ordering_args, **({"primary_abstentions": primary_abstentions} if grounded else {})}
        repair_spec = (
            (build_guided_repair(expanded, contract, raw, source, **repair_args) if intent_guidance else
             build_guided_repair(expanded, contract, raw, source, intent_guidance=False, **repair_args))
            if is_guided else build_free_repair(expanded, contract, raw, **repair_args)
        )
        repair_spec = _append_feedback_context(repair_spec, slot_suffix)
        repair_spec = _append_feedback_context(repair_spec, hook_suffix)
        repair_id = f"{root_id}-{source}-repair-{intent_id}"
        if hooks is not None:
            hooks.writer_request(repair_spec, contract, source, "repair", repair_id,
                                 **hook_context)
        repaired_raw, repaired_text, repair_error = _call_json(
            client=client, ledger=ledger, run_root=root_dir,
            request_id=repair_id, prompt=repair_spec["prompt"],
            max_tokens=writer_max_tokens, intent="h4_k3_candidate_schema_repair",
            temperature=writer_temperature,
        )
        if after_call is not None:
            after_call(stage="repair", request_id=repair_id, raw=repaired_raw,
                       raw_text=repaired_text, parse_error=repair_error)
        identity_args = ordering_identity("repair", repair_id, repaired_text)
        if partial:
            repaired = normalize_stage(repaired_raw, intent_id=intent_id, source=source,
                contract=contract, expanded=expanded, root_id=root_id, stage="repair",
                request_id=repair_id, parse_error=repair_error, admission_policy=admission_policy,
                inherited_abstentions=primary_abstentions, **identity_args)
            repaired_rows, repaired_acceptance = admit_stage(repaired, retained=accepted_rows,
                                                            admission_policy=admission_policy)
        else:
            repaired = normalize_batch(repaired_raw, intent_id=intent_id, source=source,
                contract=contract, expanded=expanded, **identity_args)
            if repair_error:
                repaired["errors"] = sorted(set(list(repaired.get("errors") or []) + [repair_error]))
                repaired["valid"] = False
            repaired_rows, repaired_acceptance = _accepted_candidates(repaired, source)
        repaired = {**repaired, "minimum_acceptance": repaired_acceptance}
        generation_records.append({
            "intent_id": intent_id,
            "generation_source": source,
            "request_id": repair_id,
            "repair_of": request_id,
            "prompt": repair_spec,
            "validation": repaired,
            "repair_raw_sha256": sha256_bytes(repaired_text.encode()),
        })
        normalized = repaired
        accepted_rows = repaired_rows
    if partial:
        all_candidates.extend(accepted_rows)
    elif accepted_rows and (normalized.get("minimum_acceptance") or {}).get("accepted"):
        all_candidates.extend(
            _prefix_candidate(row, root_id=root_id, generation_source=source)
            for row in accepted_rows
        )
    result = {"generation_records": generation_records, "candidates": all_candidates}
    if partial:
        result.update(admission_policy=admission_policy, repair_requested=needs_repair,
                      admission_summary=normalized.get("minimum_acceptance"))
    return result


def finalize_writer_candidates(*, root_dir: Path, root_id: str,
                               candidates: list[dict[str, Any]],
                               generation_records: list[dict[str, Any]],
                               expanded: Mapping[str, Any],
                               ordering_policy: Mapping[str, Any] | None = None,
                               ordering_context: Mapping[str, Any] | None = None,
                               admission_policy: str | None = None) -> dict[str, Any]:
    """Apply the original final validator and Gold rebinding per root pool."""
    all_candidates = candidates
    if admission_policy not in (None, "legacy", "partial_v1", "partial_grounded_v2"):
        raise ValueError("UNKNOWN_ADMISSION_POLICY")
    partial = admission_policy in ("partial_v1", "partial_grounded_v2")
    if partial:
        from ..h4_writer_partial_memory_v1.sources import evidence_catalog
        valid_refs = set(evidence_catalog(expanded))
    else:
        valid_refs = {str(row.get("turn_ref")) for row in expanded.get("turns") or []}
    atomic_json(root_dir / "candidate_generation.json", generation_records)
    atomic_json(root_dir / "candidate_pool.json", all_candidates)
    from ..h4_r02_ordering_temporal_contract_v1.contract import (
        verify_candidate, resolve_runtime_context,
    )
    context = {**dict(ordering_context or {}), "root_dir": root_dir, "root_id": root_id}
    versioned = ordering_policy is not None or (root_dir / "ordering_runtime_context.json").exists()
    policy, context = resolve_runtime_context(root_dir, ordering_policy, context)
    if (admission_policy == "partial_grounded_v2") != (policy.get("version") == "ordering_grounded_options_v2"):
        raise ValueError("GROUNDED_ORDERING_ADMISSION_POLICY_MISMATCH")
    if partial:
        from ..h4_writer_partial_memory_v1.admission import verify_retained_candidate
        integrity_checker = verify_retained_candidate
    else:
        integrity_checker = verify_candidate
    integrity_args = {"admission_policy": admission_policy} if partial else {}
    integrity = [integrity_checker(candidate, expanded=expanded, policy=policy,
        context=context, **integrity_args) for candidate in all_candidates]
    if versioned:
        atomic_json(root_dir / "ordering_integrity_audit.json", integrity)
    validity = validate_batch(all_candidates, valid_turn_refs=valid_refs)
    atomic_json(root_dir / "validity_audit.json", validity)
    valid_candidates = [
        candidate for candidate, audit in zip(all_candidates, validity["rows"])
        if audit["decision"] == "pass"
    ]
    atomic_json(root_dir / "candidates_valid.json", valid_candidates)

    permutation_rows: list[dict[str, Any]] = []
    permutation_errors: list[str] = []
    for candidate in valid_candidates:
        try:
            variants = position_variants(candidate)
            audit = audit_gold_rebinding(candidate, variants)
            permutation_rows.append({
                "candidate_id": candidate["candidate_id"],
                "question_type": candidate["question_type"],
                "variants": variants,
                "gold_rebinding_audit": audit,
            })
            if not audit.get("pass"):
                permutation_errors.extend(str(x) for x in audit.get("errors") or [])
        except Exception as error:
            permutation_errors.append(f"{candidate.get('candidate_id')}:{type(error).__name__}:{error}")
    atomic_json(root_dir / "permutations.json", {
        "candidate_count": len(permutation_rows),
        "gold_rebinding_errors": len(permutation_errors),
        "errors": permutation_errors,
        "rows": permutation_rows,
    })
    return {
        "status": "ready" if valid_candidates and not permutation_errors else "incomplete",
        "root_id": root_id,
        "candidate_count": len(all_candidates),
        "valid_candidate_count": len(valid_candidates),
        "generation_source": "fresh_generated",
        "candidate_generation_calls": len(generation_records),
        "gold_rebinding_errors": len(permutation_errors),
        "permutation_errors": permutation_errors,
    }


def generate_root_qa(*, expanded: Mapping[str, Any], root_dir: Path,
                     root_id: str, api_config: Path, ledger: RequestLedger,
                     client: OpenAICompatibleChallengerClient | None = None,
                     feedback_context: Mapping[str, Any] | None = None,
                     hooks: Any | None = None,
                     ordering_policy: Mapping[str, Any] | None = None,
                     ordering_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the fixed H4 mining route once for one frozen root.

    No candidate from a historical run is read.  The optional client is only a
    test seam; live callers pass the configured Challenger client.
    """
    if (ordering_policy or {}).get("version") == "ordering_grounded_options_v2":
        raise ValueError("GROUNDED_ORDERING_REQUIRES_EXPLICIT_PARTIAL_WRITER_ENTRY")
    if client is None:
        config = load_challenger_config(api_config)
        client = OpenAICompatibleChallengerClient(replace(
            config, temperature=GENERATOR_TEMPERATURE, enable_thinking=False,
            timeout_seconds=max(180, int(config.timeout_seconds))))
    mined = generate_root_intents(
        expanded=expanded, root_dir=root_dir, root_id=root_id,
        api_config=api_config, ledger=ledger, client=client,
        feedback_context=feedback_context, hooks=hooks,
    )
    intent_result, contracts = mined["intent_result"], mined["contracts"]
    all_candidates: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    for contract_index, contract in enumerate(contracts):
        slot_suffixes = (
            feedback_context.get("slot_suffixes")
            if isinstance(feedback_context, Mapping) else None
        )
        slot_suffix = ""
        if isinstance(slot_suffixes, list) and contract_index < len(slot_suffixes):
            slot_suffix = str(slot_suffixes[contract_index] or "")
        for source in ("contract_guided", "free_exploration"):
            batch = execute_writer_batch(
                expanded=expanded, contract=contract, root_dir=root_dir,
                root_id=root_id, ledger=ledger, client=client, source=source,
                slot_suffix=slot_suffix, hooks=hooks,
                **({"ordering_policy": ordering_policy, "ordering_context": ordering_context}
                   if ordering_policy is not None else {}),
            )
            generation_records.extend(batch["generation_records"])
            all_candidates.extend(batch["candidates"])
    summary = finalize_writer_candidates(
        root_dir=root_dir, root_id=root_id, candidates=all_candidates,
        generation_records=generation_records, expanded=expanded,
        ordering_policy=ordering_policy, ordering_context=ordering_context,
    )
    return {
        **summary,
        "intent_count": len(intent_result.get("intents") or []),
        "valid_contract_count": len(contracts),
        "intent_errors": list(intent_result.get("errors") or []),
    }
