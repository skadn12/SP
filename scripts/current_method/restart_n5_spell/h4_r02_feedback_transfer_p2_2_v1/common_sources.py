"""Deterministic, read-only P1 source qualification and observation projection.

This module never runs the old feedback compiler, ranks candidates, cleans a
dialogue, or invokes a model. Eligibility precedes every active projection.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from ..h4_round0_b0_k3_difficulty_v1.contract import normalize_prediction

COMMON_BASE_VERSION = "h4_r02_feedback_common_base_v1"
OBSERVATION_VERSION = "h4_r02_feedback_observation_v1"
P1 = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_feedback_ablation_p0_p1_v1_live_1")
BASELINE_LABEL = "P1_LEGACY_FEEDBACK_TRANSFER_BASELINE"
ACTIVE = "ACTIVE_GUIDANCE_ELIGIBLE"
HISTORY = "AMBIGUOUS_HISTORY_ONLY"
EXCLUDED = "EXCLUDED_ACTIVE_GUIDANCE"
FIRST_ROOTS = tuple(f"cc-train-episode-{x}" for x in (198480, 122711, 197635))
SOURCE_POLICY = {
    f"cc-train-episode-{root}": {
        "source_candidate_id": f"cc-train-episode-{root}:{candidate}",
        "eligibility": status,
        "expected_review_status": "REVIEWED_NO_CLEAR_GOLD_ERROR" if status == ACTIVE else "REVIEWED_WITH_UNRESOLVED_AMBIGUITY",
    }
    for root, candidate, status in (
        (122711, "I002-F-Q06", ACTIVE), (136202, "I003-F-Q06", ACTIVE),
        (197635, "I002-C-Q03", HISTORY), (198480, "I003-C-Q03", ACTIVE),
        (254852, "I003-C-Q01", HISTORY), (94289, "I001-C-Q03", ACTIVE),
    )
}
MODELS = ("M0", "M_clean_policy", "M_inclusive_policy")
CONTEXTS = ("full_context", "no_context")
FIXED_USE_POLICY = """Historical observations are measurements on one historical question, with position variants and repeated rollouts; they are not independent new questions or model ability diagnoses.
1. If full-context performance is better than no-context performance, explore distinctions supported by current evidence that jointly distinguish people, events, states, or constraints. Do not infer a mastered ability.
2. If both full and no context perform well, explore distinctions that depend more on the specific current dialogue. Do not infer cheating or leakage.
3. When full-context outputs differ from historical Gold, use the actual predictions only for an eligible source to look for a verifiable confusion. Do not automatically infer a missing ability or require greater difficulty.
4. Disable score-based guidance for ineligible sources. Historical options include distractors and Gold is only a historical annotation. Facts and answers in new questions must be supported by the complete current frozen transcript.
5. Do not invent score thresholds, difficulty levels, strategy enums, curricula, or model ability fields. There is no mandatory R02 citation. Source eligibility is a fixed experimental choice, not independent human Gold certification."""


class SourceDrift(ValueError):
    """The actual frozen source does not match its recorded identity."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def jsonl(path: Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_frozen(path: Path, value: Any = None, *, raw: bytes | None = None) -> dict:
    path = Path(path)
    payload = raw if raw is not None else (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise SourceDrift(f"frozen_output_conflict:{path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}


class SourceReader:
    """Hash exact files and select exact records; never guess a result directory."""
    def __init__(self):
        self.audit: dict[str, dict] = {}
        self.selected: dict[tuple, Any] = {}

    def verify(self, source: dict) -> dict:
        path = Path(source["path"])
        if not path.is_file():
            raise SourceDrift(f"missing_source:{path}")
        key = str(path.resolve())
        if key not in self.audit:
            self.audit[key] = {"path": key, "sha256": file_sha(path)}
        if source.get("sha256") and self.audit[key]["sha256"] != source["sha256"]:
            raise SourceDrift(f"source_sha_mismatch:{path}")
        return {**self.audit[key], **({"selector": deepcopy(source["selector"])} if "selector" in source else {})}

    def record(self, source: dict) -> Any:
        checked = self.verify(source)
        path = Path(checked["path"])
        selector = checked.get("selector")
        key = (str(path), canonical(selector))
        if key in self.selected:
            return self.selected[key]
        if path.suffix == ".jsonl":
            if not selector:
                raise SourceDrift(f"jsonl_selector_missing:{path}")
            # Limit retained data to the six already-selected candidates.
            values = []
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    if selector.get("candidate_id") and selector["candidate_id"] not in line:
                        continue
                    row = json.loads(line)
                    if all(row.get(k) == v for k, v in selector.items()):
                        values.append(row)
            if len(values) != 1:
                raise SourceDrift(f"selector_cardinality:{path}:{len(values)}")
            value = values[0]
        else:
            value = load(path)
            if selector and (not isinstance(value, dict) or not all(value.get(k) == v for k, v in selector.items())):
                raise SourceDrift(f"json_selector_mismatch:{path}")
        self.selected[key] = value
        return value


def verify_eligibility(root_id: str, packet: dict, review: dict, selected: dict, expected: dict) -> str:
    cid = expected["source_candidate_id"]
    if packet.get("root_id") != root_id or packet.get("source_candidate_id") != cid:
        raise SourceDrift(f"seed_identity_mismatch:{root_id}")
    if selected.get("root_id") != root_id or selected.get("candidate_id") != cid or selected.get("selected") is not True:
        raise SourceDrift(f"selected_seed_mismatch:{root_id}")
    if review.get("candidate_id") != cid or review.get("review_status") != expected["expected_review_status"]:
        raise SourceDrift(f"review_status_mismatch:{root_id}")
    if review.get("confirmed_source_defect") is not False or review.get("historical_gold_changed") is not False:
        raise SourceDrift(f"review_identity_mismatch:{root_id}")
    if review != selected.get("manual_review"):
        raise SourceDrift(f"review_record_disagreement:{root_id}")
    if packet["historical_qa"].get("question_type") not in ("single_choice", "multi_select"):
        raise SourceDrift(f"selected_source_type_ineligible:{root_id}")
    return expected["eligibility"]


def recompute_observations(result: dict, question_type: str, gold: list, *, root_id: str, candidate_id: str) -> dict:
    """Rebind displayed predictions by the original normalization convention.

    Unknown predictions remain null; observed zero counts are distinguished from
    missing correctness values. The current frozen sources have complete scores.
    """
    requests = result.get("requests")
    if not isinstance(requests, list) or result.get("request_count") != len(requests):
        raise SourceDrift("solver_request_count_mismatch")
    if any(result.get(k) != v for k, v in (("root_id", root_id), ("candidate_id", candidate_id), ("question_type", question_type))):
        raise SourceDrift("solver_result_identity_mismatch")
    request_ids = [row.get("request_id") for row in requests]
    if any(not isinstance(x, str) or not x for x in request_ids) or len(set(request_ids)) != len(request_ids):
        raise SourceDrift("duplicate_or_missing_request_id")
    output = {}
    for context in CONTEXTS:
        rows = [row for row in requests if row.get("context_mode") == context]
        predictions = Counter()
        known_correctness = []
        scored = 0
        unknown_scored = 0
        for row in rows:
            if any(row.get(k) != v for k, v in (("root_id", root_id), ("candidate_id", candidate_id), ("question_type", question_type), ("canonical_gold", gold))):
                raise SourceDrift("solver_request_identity_or_gold_mismatch")
            displayed = row.get("display_prediction")
            mapping = row.get("display_to_canonical")
            saved = row.get("canonical_prediction")
            if displayed is None:
                if saved is not None:
                    raise SourceDrift("canonical_prediction_without_display_prediction")
                mapped = None
            else:
                if not isinstance(displayed, list) or not isinstance(mapping, dict) or len(set(mapping.values())) != len(mapping):
                    raise SourceDrift("invalid_display_to_canonical_mapping")
                if any(value not in mapping for value in displayed):
                    raise SourceDrift("display_prediction_outside_mapping")
                mapped = normalize_prediction(question_type, (mapping[value] for value in displayed))
                if mapped != saved:
                    raise SourceDrift("canonical_prediction_remap_mismatch")
            predictions[canonical(mapped)] += 1
            if row.get("scored") is True:
                scored += 1
                if row.get("correct") is None:
                    unknown_scored += 1
                elif not isinstance(row["correct"], bool):
                    raise SourceDrift("invalid_correctness_type")
                else:
                    if mapped is None or not mapped:
                        raise SourceDrift("scored_without_prediction")
                    actual_correct = mapped == normalize_prediction(question_type, gold)
                    if row["correct"] != actual_correct:
                        raise SourceDrift("historical_correctness_recompute_mismatch")
                    known_correctness.append(actual_correct)
        output[context] = {
            "request_count": len(rows),
            "scored_count": scored,
            "correct_count": sum(known_correctness) if known_correctness else None,
            "incorrect_count": sum(not x for x in known_correctness) if known_correctness else None,
            "unknown_unscored_count": len(rows) - scored + unknown_scored,
            "unscored_or_unknown_count": len(rows) - scored,
            "unknown_correctness_count": unknown_scored,
            "canonical_prediction_counts": [{"prediction": json.loads(key), "count": count} for key, count in sorted(predictions.items())],
        }
    if sum(row["request_count"] for row in output.values()) != len(requests):
        raise SourceDrift("unexpected_solver_context_mode")
    return output


def history_projection(row: dict, *, status: str) -> dict:
    """An explicit whitelist prevents inactive Gold/options/count leakage."""
    candidate = row.get("candidate") or {}
    source_id = row.get("candidate_id") or row.get("source_candidate_id")
    result = {
        "root_id": row.get("root_id"), "source_candidate_id": source_id,
        "question_type": row.get("question_type") or candidate.get("question_type"),
        "stem": row.get("stem") or row.get("question") or candidate.get("question"),
        "candidate_sha256": row.get("candidate_sha256"),
        "solver_prompt_fingerprint": row.get("solver_prompt_fingerprint"),
        "source_quality_status": status,
        "active_feedback_eligible": False,
    }
    return result


def model_projection(payload: dict) -> dict:
    """Project only public observations, independently blocking inactive cards."""
    fields = ("observation_version", "root_id", "feedback_id", "source_candidate_id", "source_question_type", "source_question", "options", "items", "historical_gold", "gold_version", "source_quality_status", "review_status", "source_quality_caveat", "source_evidence_refs", "source_cluster_id", "source_cluster_status", "solver_observations", "provenance")
    cards = []
    root_id = payload["root_id"]
    for card in payload.get("observation_cards", []):
        if card.get("root_id") != root_id or card.get("source_quality_status") != ACTIVE:
            raise SourceDrift("inactive_or_cross_root_card_in_active_projection")
        projected = {key: deepcopy(card[key]) for key in fields if key in card}
        if "/root/" in canonical(projected) or "raw_result_source" in canonical(projected):
            raise SourceDrift("engineering_provenance_in_model_projection")
        cards.append(projected)
    return {"root_id": root_id, "feedback_status": "ACTIVE_FEEDBACK" if cards else "NO_ACTIVE_FEEDBACK", "observation_cards": cards}


def _check_model_identity(model: str, identity: dict, reader: SourceReader) -> dict:
    source = reader.record(identity["model_identity_source"])
    if identity.get("model_name") != model:
        raise SourceDrift("model_identity_name_mismatch")
    if model == "M0":
        if source.get("solver_checkpoint") != identity.get("checkpoint") or object_sha(source["checkpoint_identity"]) != identity.get("checkpoint_identity_sha256"):
            raise SourceDrift("base_model_identity_mismatch")
        return {"model_name": model, "checkpoint_name": Path(identity["checkpoint"]).name, "checkpoint_identity_sha256": identity["checkpoint_identity_sha256"]}
    for key in ("model_name", "inference_model_path", "model_fingerprint_sha256", "solver_model_identifier"):
        if identity.get(key) != source.get(key):
            raise SourceDrift(f"trained_model_identity_mismatch:{model}:{key}")
    return {"model_name": model, "checkpoint_name": Path(identity["inference_model_path"]).name, "model_fingerprint_sha256": identity["model_fingerprint_sha256"]}


def _validate_root(root_id: str, expected: dict, pairing: dict, reviews: list, review_document: str, reader: SourceReader, p1: Path) -> tuple[dict, dict, dict | None, dict]:
    packet_path = p1 / "inputs/feedback_on" / root_id / "feedback_packet.json"
    packet_source = pairing["arms"]["feedback_on"]["packet"]
    if Path(packet_source["path"]).resolve() != packet_path.resolve():
        raise SourceDrift("packet_path_not_frozen_p1_input")
    packet = reader.record(packet_source)
    cid = expected["source_candidate_id"]
    matching_reviews = [row for row in reviews if row.get("candidate_id") == cid]
    if len(matching_reviews) != 1:
        raise SourceDrift("manual_review_not_unique")
    review = matching_reviews[0]
    selected = reader.record(packet["sources"]["seed_manifest"])
    status = verify_eligibility(root_id, packet, review, selected, expected)
    if f"## {cid}\n" not in review_document or review["judgment"] not in review_document or f"Status: {review['review_status']}" not in review_document:
        raise SourceDrift("review_document_disagreement")
    if pairing["arms"]["feedback_on"]["seed_candidate_id"] != cid:
        raise SourceDrift("pairing_seed_mismatch")
    transcript_path = p1 / "inputs/shared" / root_id / "expanded_dialogue_clean.json"
    transcript_source = pairing["transcript"]["cleaned_transcript_file"]
    if Path(transcript_source["path"]).resolve() != transcript_path.resolve():
        raise SourceDrift("transcript_path_not_frozen_p1_input")
    dialogue = reader.record(transcript_source)
    refs = [row["turn_ref"] for row in dialogue["turns"]]
    if dialogue.get("root_id") != root_id or len(refs) != len(set(refs)):
        raise SourceDrift("frozen_dialogue_identity_mismatch")
    if not set(review["evidence_refs"]) <= set(refs) or not set(packet["historical_qa"]["evidence_refs"]) <= set(refs):
        raise SourceDrift("historical_source_ref_not_in_current_transcript")
    historical_row = reader.record(packet["sources"]["historical_qa"])
    candidate = historical_row["candidate"]
    if historical_row.get("root_id") != root_id or historical_row.get("candidate_id") != cid or candidate.get("candidate_id") != cid:
        raise SourceDrift("historical_candidate_identity_mismatch")
    if {key: candidate[key] for key in packet["historical_qa"]} != packet["historical_qa"] or object_sha(candidate) != selected["candidate_sha256"]:
        raise SourceDrift("historical_candidate_content_mismatch")
    pool = reader.record(packet["sources"]["historical_candidate_pool"])
    pool_rows = pool if isinstance(pool, list) else pool.get("candidates", [])
    if sum(row == candidate for row in pool_rows) != 1:
        raise SourceDrift("historical_candidate_pool_mismatch")
    for source in packet["sources"].values():
        reader.verify(source)
    review_source = reader.verify({"path": str(p1 / "inputs/seed_manual_review.jsonl")})
    review_source.update({"selector": {"candidate_id": cid}, "record_sha256": object_sha(review)})
    provenance = {
        "common_base_version": COMMON_BASE_VERSION, "root_id": root_id,
        "source_candidate_id": cid, "packet": reader.verify(packet_source),
        "transcript": reader.verify(transcript_source), "historical_sources": deepcopy(packet["sources"]),
        "review_source": review_source,
        "review_document": reader.verify({"path": str(p1 / "diagnostics/feedback_seed_content_review.md")}),
        "actual_review": review, "historical_candidate_sha256": object_sha(candidate),
        "solver_sources": {},
    }
    entry = {"root_id": root_id, "source_candidate_id": cid, "eligibility": status,
             "status": "PASS", "review_status": review["review_status"], "review_source": review_source,
             "packet_source": reader.verify(packet_source), "historical_candidate_sha256": object_sha(candidate),
             "remaining_risks": review["remaining_risks"], "independent_gold_certification": False,
             "live_scope": root_id in FIRST_ROOTS}
    # Unreliable source results are preserved by reference only. The gate occurs
    # before active-card statistics or packet projection, including fallback.
    if status != ACTIVE:
        provenance["inactive_solver_sources_not_aggregated"] = {}
        for model in MODELS:
            copy_source = reader.verify({"path": str(p1 / "inputs/feedback_on" / root_id / "historical_solver_requests" / f"{model}.json")})
            original_source = reader.verify(packet["solver_observations"][model]["raw_result_source"])
            copied = reader.record(copy_source)
            if copied != reader.record(original_source) or copied.get("candidate_id") != cid or copied.get("root_id") != root_id:
                raise SourceDrift(f"inactive_historical_request_identity_disagreement:{model}")
            provenance["inactive_solver_sources_not_aggregated"][model] = {"historical_copy": copy_source, "raw_result_source": original_source}
        return entry, provenance, None, candidate
    observations = {}
    request_keys = []
    for model in MODELS:
        old = packet["solver_observations"][model]
        copy_source = reader.verify({"path": str(p1 / "inputs/feedback_on" / root_id / "historical_solver_requests" / f"{model}.json")})
        copied = reader.record(copy_source)
        original = reader.record(old["raw_result_source"])
        if copied != original:
            raise SourceDrift(f"historical_request_copy_disagreement:{model}")
        for field in ("root_id", "candidate_id", "candidate_pool_sha256", "dialogue_sha256"):
            if copied.get(field) != historical_row.get(field):
                raise SourceDrift(f"historical_request_candidate_disagreement:{field}")
        computed = recompute_observations(copied, candidate["question_type"], candidate["gold"], root_id=root_id, candidate_id=cid)
        for context, counts in computed.items():
            expected_counts = old["by_context"][context]
            for key in ("request_count", "scored_count", "correct_count", "incorrect_count", "unscored_or_unknown_count", "unknown_correctness_count", "canonical_prediction_counts"):
                if counts[key] != expected_counts[key]:
                    raise SourceDrift(f"packet_observation_mismatch:{model}:{context}:{key}")
        observations[model] = {"model_identity": _check_model_identity(model, old["model_identity"], reader), "by_context": computed}
        provenance["solver_sources"][model] = {"historical_copy": copy_source, "raw_result_source": reader.verify(old["raw_result_source"]),
                                               "model_identity": deepcopy(old["model_identity"]), "recomputed_request_count": len(copied["requests"])}
        request_keys.append({(row["request_id"], row["context_mode"], row["variant_id"], row["slot_id"], row["seed"]) for row in copied["requests"]})
    if any(keys != request_keys[0] for keys in request_keys[1:]):
        raise SourceDrift("historical_model_request_set_mismatch")
    provided_gold_version = packet.get("gold_version") or packet["historical_qa"].get("gold_version")
    gold_version = {"id": provided_gold_version or "derived-sha256:" + object_sha({"candidate": candidate, "gold": candidate["gold"]}),
                    "derived": provided_gold_version is None,
                    "derivation": "canonical JSON of exact historical candidate content and historical Gold; SHA-256" if provided_gold_version is None else None}
    cluster_id = packet.get("source_cluster_id", packet["historical_qa"].get("source_cluster_id"))
    card = {
        "observation_version": OBSERVATION_VERSION, "root_id": root_id,
        "feedback_id": f"FB-{root_id.rsplit('-', 1)[-1]}-{cid.split(':', 1)[1]}",
        "source_candidate_id": cid, "source_question_type": candidate["question_type"],
        "source_question": candidate["question"], "options": candidate["options"], "items": candidate["items"],
        "historical_gold": candidate["gold"], "gold_version": gold_version,
        "source_quality_status": status, "review_status": review["review_status"],
        "remaining_risks": deepcopy(review["remaining_risks"]),
        "source_quality_caveat": "Fixed experimental source eligibility; not independent human Gold certification. Historical paraphrase, scope and citation risks remain recorded in the source review.",
        "source_evidence_refs": candidate["evidence_refs"], "source_cluster_id": cluster_id,
        "source_cluster_status": "EXPLICIT_FROZEN_SOURCE" if cluster_id is not None else "NOT_PROVIDED_IN_FROZEN_SOURCE",
        "solver_observations": observations,
        "provenance": {"packet_sha256": provenance["packet"]["sha256"], "historical_candidate_sha256": object_sha(candidate),
                       "review_record_sha256": object_sha(review), "details": "separate provenance sidecar"},
    }
    return entry, provenance, card, candidate


def prepare_sources(run: Path, *, p1: Path = P1) -> dict:
    """Freeze all-six eligibility plus exact inputs for the first three roots.

    Returns {status, manifest, inputs, path_audit}. SOURCE_DRIFT is recorded for
    each affected root, with empty cards and no substitution; other offline work
    continues. Caller must block affected live requests.
    """
    run, p1 = Path(run).resolve(), Path(p1).resolve()
    if run == p1 or p1 in run.parents:
        raise ValueError("P1_is_read_only")
    reader = SourceReader()
    pairing_path = p1 / "pairing_manifest.jsonl"
    reader.verify({"path": str(pairing_path)})
    pairings = jsonl(pairing_path)
    if len(pairings) != 6 or {row["root_id"] for row in pairings} != set(SOURCE_POLICY):
        raise SourceDrift("pairing_root_scope_mismatch")
    pairings = {row["root_id"]: row for row in pairings}
    reviews = jsonl(p1 / "inputs/seed_manual_review.jsonl")
    review_document = (p1 / "diagnostics/feedback_seed_content_review.md").read_text()
    ordering_source = reader.verify({"path": str(p1 / "inputs/history/ordering_records.jsonl")})
    ordering_rows = jsonl(Path(ordering_source["path"]))
    selection_rows = jsonl(p1 / "feedback_selection_audit.jsonl")
    all_entries, inputs = [], {}
    for root_id, expected in SOURCE_POLICY.items():
        try:
            entry, provenance, card, candidate = _validate_root(root_id, expected, pairings[root_id], reviews, review_document, reader, p1)
        except (SourceDrift, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
            entry = {"root_id": root_id, **expected, "status": "SOURCE_DRIFT", "eligibility": "SOURCE_DRIFT", "error": str(error), "live_scope": root_id in FIRST_ROOTS}
            provenance, card, candidate = {"root_id": root_id, "status": "SOURCE_DRIFT", "error": str(error)}, None, None
        all_entries.append(entry)
        if root_id not in FIRST_ROOTS:
            continue
        directory = run / "inputs" / root_id
        artifacts = {}
        if entry["status"] == "PASS":
            transcript_source = provenance["transcript"]
            artifacts["transcript"] = write_frozen(directory / "expanded_dialogue_clean.json", raw=Path(transcript_source["path"]).read_bytes())
            if artifacts["transcript"]["sha256"] != transcript_source["sha256"]:
                raise SourceDrift("copied_transcript_bytes_changed")
        cards = {"common_base_version": COMMON_BASE_VERSION, "root_id": root_id,
                 "feedback_status": "ACTIVE_FEEDBACK" if card is not None else "NO_ACTIVE_FEEDBACK",
                 "source_validation_status": entry["status"], "observation_cards": [card] if card is not None else []}
        if entry["status"] == "PASS":
            model_projection(cards)
        historical = [history_projection(row, status=EXCLUDED) for row in ordering_rows if row["root_id"] == root_id]
        # Keep explicitly defective identities and the fixed ambiguous seed.
        for row in selection_rows:
            if row["root_id"] != root_id or row["question_type"] == "ordering":
                continue
            old_review = row.get("manual_review") or {}
            if old_review.get("confirmed_source_defect"):
                source_row = reader.record({**row["historical_sources"]["difficulty_manifest"], "selector": {"candidate_id": row["candidate_id"]}})
                historical.append(history_projection({**source_row, "candidate_sha256": row["candidate_sha256"]}, status=EXCLUDED))
        if entry["status"] == "PASS" and entry["eligibility"] == HISTORY:
            historical.append(history_projection({"root_id": root_id, "candidate_id": candidate["candidate_id"], "candidate": candidate,
                                                  "candidate_sha256": object_sha(candidate)}, status=HISTORY))
        history_payload = {"common_base_version": COMMON_BASE_VERSION, "root_id": root_id,
                           "policy": "Identity/stem/fingerprint tracking only; no new dedup admission gate; no Gold, options or performance guidance.",
                           "records": sorted(historical, key=lambda row: row["source_candidate_id"])}
        provenance["history_ordering_source"] = ordering_source
        artifacts["observation_cards"] = write_frozen(directory / "observation_cards.json", cards)
        artifacts["history_only"] = write_frozen(directory / "history_only.json", history_payload)
        artifacts["provenance"] = write_frozen(directory / "provenance.json", provenance)
        inputs[root_id] = {"status": entry["status"], "eligibility": entry["eligibility"], "feedback_status": cards["feedback_status"], "artifacts": artifacts}
    manifest = {
        "common_base_version": COMMON_BASE_VERSION, "baseline_label": BASELINE_LABEL,
        "status": "PASS" if all(row["status"] == "PASS" for row in all_entries) else "SOURCE_DRIFT",
        "selection_policy": "Only P1 final six seeds; no compiler, rank, seed replacement, source expansion, Gold repair or Memory writes.",
        "roots": all_entries, "first_live_root_order": list(FIRST_ROOTS),
        "ordering_and_confirmed_defects": EXCLUDED,
        "source_quality_policy": "Eligibility is a fixed experimental choice and is not independent Gold certification.",
        "aggregation_unit": "One historical source question per eligible root; model/context kept separate; variants and rollouts are repeated observations.",
        "fixed_use_policy": FIXED_USE_POLICY,
        "inputs": inputs,
    }
    artifact = write_frozen(run / "common_base/source_eligibility_manifest.json", manifest)
    path_audit = sorted(reader.audit.values(), key=lambda row: row["path"])
    write_frozen(run / "common_base/source_input_path_audit.json", {"sources": path_audit})
    write_frozen(run / "common_base/observation_policy.json", {"common_base_version": COMMON_BASE_VERSION, "observation_version": OBSERVATION_VERSION,
                 "fixed_use_policy": FIXED_USE_POLICY, "missing_values": "Unknown predictions/counts stay null; measured zero counts remain zero.",
                 "canonical_prediction_mapping": "Original normalize_prediction applied after exact display_to_canonical mapping; verify frozen canonical prediction and correctness.",
                 "ineligible_sources": "Filtered before card construction, statistical aggregation and every model projection."})
    return {"status": manifest["status"], "manifest": artifact, "inputs": inputs, "path_audit": path_audit}
