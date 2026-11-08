"""Offline E1Q1 multilabel rerouting and provisional bank construction.

The stage only reads frozen artifacts.  It does not call a model and never
rewrites a source manifest.  Semantic similarity is deliberately represented
as a review label, while exact identity uses independently auditable axes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from . import (
    DEFAULT_REROUTE_RUN,
    E1Q0_RUN,
    E1Q0_SOLVER_RUN,
    E1Q1_COMPILER_RUN,
    E1Q1_GENERATION_RUN,
    E1Q1_SOLVER_RUN,
    ROUND0_DIFFICULTY_RUN,
    REROUTE_VERSION,
)
from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.contract import (
    QUESTION_TYPES,
    shape_errors,
)


UNKNOWN = "UNKNOWN"
MODELS = ("M_clean_policy", "M_inclusive_policy")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_sidecar(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing_source:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing_source_sidecar:{sidecar}")
    observed = sha256_file(path)
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if observed != expected:
        raise RuntimeError(f"source_sidecar_mismatch:{path}")
    return observed


def write_artifact(path: Path, value: Any) -> str:
    if path.suffix == ".jsonl":
        text = "".join(canonical(row) + "\n" for row in value)
    elif isinstance(value, str):
        text = value if value.endswith("\n") else value + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def norm_text(value: Any) -> str:
    value = "" if value is None else str(value)
    value = value.casefold().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def tokens(value: Any) -> set[str]:
    return set(norm_text(value).split())


def ref_suffix(ref: Any) -> str:
    value = str(ref or "")
    parts = value.split(":")
    markers = {"R01", "R02", "P01", "P02", "P03", "P04", "P05"}
    for index, part in enumerate(parts):
        if part in markers:
            return ":".join(parts[index:])
    return value


def option_texts(candidate: dict[str, Any]) -> dict[str, str]:
    options = candidate.get("options") or {}
    items = candidate.get("items") or {}
    values: dict[str, str] = {}
    if isinstance(options, dict):
        values.update({str(key): str(value) for key, value in options.items()})
    if isinstance(items, dict):
        values.update({str(key): str(value) for key, value in items.items()})
    return values


def canonical_gold(candidate: dict[str, Any]) -> list[str]:
    values = option_texts(candidate)
    gold = candidate.get("gold")
    if gold is None:
        gold = [candidate.get("correct_option")] if candidate.get("correct_option") else []
    if not isinstance(gold, list):
        gold = [gold]
    # Gold is represented by option/item text so a display-letter permutation
    # cannot create a false negative exact-surface comparison.
    normalized = [norm_text(values.get(str(value), value)) for value in gold]
    if str(candidate.get("question_type") or "") in {"multi_select", "multiple_choice"}:
        return sorted(normalized)
    return normalized


def surface_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    values = option_texts(candidate)
    return {
        "question_type": str(candidate.get("question_type") or ""),
        "question": norm_text(candidate.get("question")),
        "option_or_item_multiset": sorted(norm_text(value) for value in values.values()),
        "gold": canonical_gold(candidate),
    }


def surface_signature(candidate: dict[str, Any]) -> str:
    return sha256_value(surface_payload(candidate))


def evidence_signature(candidate: dict[str, Any]) -> str:
    groups = candidate.get("evidence_groups") or {}
    return sha256_value({
        "evidence_refs": sorted(norm_text(ref) for ref in (candidate.get("evidence_refs") or [])),
        "evidence_groups": {
            str(key): sorted(norm_text(ref) for ref in (values or []))
            for key, values in sorted(groups.items())
        } if isinstance(groups, dict) else {},
    })


def stem_signature(candidate: dict[str, Any]) -> str:
    return sha256_value({"question_type": candidate.get("question_type"), "stem": norm_text(candidate.get("question"))})


def lexical_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_stem = norm_text(left.get("question"))
    right_stem = norm_text(right.get("question"))
    left_tokens, right_tokens = tokens(left_stem), tokens(right_stem)
    union = left_tokens | right_tokens
    return {
        "stem_sequence_ratio": round(SequenceMatcher(None, left_stem, right_stem).ratio(), 6),
        "stem_token_jaccard": round(len(left_tokens & right_tokens) / len(union), 6) if union else 1.0,
        "surface_equal": float(surface_signature(left) == surface_signature(right)),
        "evidence_equal": float(evidence_signature(left) == evidence_signature(right)),
    }


def is_lexical_near(metrics: dict[str, float]) -> bool:
    return bool(metrics["stem_sequence_ratio"] >= 0.90 or metrics["stem_token_jaccard"] >= 0.80)


def gate_fingerprint(row: dict[str, Any], solver_contract_sha256: str | None = None) -> str:
    audit = []
    for item in row.get("prompt_audit") or []:
        audit.append(
            {
                "context_mode": item.get("context_mode"),
                "variant_id": item.get("variant_id"),
                "prompt_sha256": item.get("prompt_sha256"),
                "prompt_token_count": item.get("prompt_token_count"),
            }
        )
    payload = {
        "question_type": row.get("question_type"),
        "max_model_len": row.get("max_model_len"),
        "output_reserve": row.get("output_reserve"),
        "position_count": row.get("position_count"),
        "solver_contract_sha256": solver_contract_sha256,
        "prompt_audit": sorted(audit, key=lambda item: (str(item.get("context_mode")), str(item.get("variant_id")))),
    }
    return sha256_value(payload)


def dialogue_refs(dialogue: dict[str, Any]) -> set[str]:
    return {str(turn.get("turn_ref")) for turn in (dialogue.get("turns") or []) if turn.get("turn_ref")}


def candidate_structure(candidate: dict[str, Any], valid_refs: set[str]) -> dict[str, Any]:
    errors = list(candidate.get("schema_errors") or [])
    values = option_texts(candidate)
    if not candidate.get("question"):
        errors.append("missing_question")
    if len(values) < 2:
        errors.append("insufficient_options_or_items")
    gold = candidate.get("gold")
    if not isinstance(gold, list) or not gold:
        errors.append("invalid_gold")
    elif any(str(value) not in values for value in gold):
        errors.append("gold_reference_missing")
    # R03 v1.1 candidates use the explicit discriminator and the same
    # deterministic shape contract as the output schema.  Older rerouted
    # records with legacy question types remain supported, but recognized
    # v1.1 types are never accepted through the old loose length check.
    qtype = str(candidate.get("question_type") or "")
    if qtype in QUESTION_TYPES:
        contract_gold = gold
        if qtype == "single_choice" and not isinstance(contract_gold, list):
            contract_gold = [candidate.get("correct_option")]
        errors.extend(f"qa_contract:{error}" for error in shape_errors(qtype, values, contract_gold))
    refs = [str(ref) for ref in (candidate.get("evidence_refs") or [])]
    invalid_refs = [ref for ref in refs if ref not in valid_refs and ref_suffix(ref) not in {ref_suffix(v) for v in valid_refs}]
    if invalid_refs:
        errors.append("invalid_evidence_refs")
    return {
        "structural_status": "PASS" if not errors and candidate.get("schema_valid", True) else "STRUCTURAL_QUARANTINE",
        "structural_errors": sorted(set(errors)),
        "evidence_refs_valid": not invalid_refs,
        "invalid_evidence_refs": invalid_refs,
    }


def root_split(round0: list[dict[str, Any]]) -> dict[str, str]:
    by_root: dict[str, set[str]] = defaultdict(set)
    for row in round0:
        by_root[str((row.get("candidate") or {}).get("root_id") or row.get("root_id"))].add(str(row.get("split")))
    result = {}
    for root, splits in by_root.items():
        if len(splits) != 1:
            raise RuntimeError(f"root_has_multiple_round0_splits:{root}:{sorted(splits)}")
        result[root] = next(iter(splits))
    return result


def load_e1q1() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    dialogues: dict[str, dict[str, Any]] = {}
    files: list[Path] = []
    for root_dir in sorted(E1Q1_GENERATION_RUN.glob("cc-train-episode-*")):
        cp = root_dir / "candidates_valid_scoped.json"
        dp = root_dir / "dialogue_source.json"
        if not cp.is_file() or not dp.is_file():
            continue
        assert_sidecar(cp)
        assert_sidecar(dp)
        candidates = load_json(cp)
        dialogue = load_json(dp)
        root_id = str(dialogue.get("root_id") or root_dir.name)
        rows.extend(candidates)
        dialogues[root_id] = dialogue
        files.extend([cp, dp])
    return rows, dialogues, files


def source_seed(packet: dict[str, Any]) -> dict[str, Any]:
    source = packet.get("source_options_or_items") or {}
    candidate = {
        "candidate_id": packet.get("source_candidate_id"),
        "root_id": packet.get("root_id"),
        "question": packet.get("source_question"),
        "question_type": packet.get("source_question_type"),
        "options": source if packet.get("source_question_type") != "ordering" else {},
        "items": source if packet.get("source_question_type") == "ordering" else {},
        "gold": packet.get("source_canonical_gold") or [],
        "evidence_refs": packet.get("source_evidence_refs") or [],
    }
    return candidate


def arm_label(branch: str | None, corpus: str) -> str:
    if corpus == "E1Q1":
        return "E1Q1_FEEDBACK_GUIDED"
    if corpus == "E1Q0" and branch == "control":
        return "E0Q0_CONTROL_BLIND"
    if corpus == "E1Q0" and branch == "feedback":
        return "E1Q0_FEEDBACK_BLIND"
    return "ROUND0_PARENT_PLUS_R01"


def reroute(run: Path = DEFAULT_REROUTE_RUN) -> dict[str, Any]:
    run = run.resolve()
    if run.exists():
        if any(run.iterdir()):
            raise RuntimeError(f"output_already_exists:{run}")
        # A previous interrupted preflight may have left an empty directory.
        # Reusing that empty staging directory is deterministic and removes no
        # user artifact.
        run.rmdir()
    run.mkdir(parents=True, exist_ok=False)

    round0_path = ROUND0_DIFFICULTY_RUN / "difficulty_manifest.jsonl"
    e1q0_path = E1Q0_RUN / "all_candidates_valid.jsonl"
    e1q0_dialogue_path = E1Q0_RUN / "paired_dialogue_manifest.jsonl"
    packet_path = E1Q1_COMPILER_RUN / "source_qa_packets.jsonl"
    scope_report_path = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_question_bank_bootstrap_v1_live_1/final_scope_freeze_report.json")
    for source in (round0_path, e1q0_path, e1q0_dialogue_path, packet_path, scope_report_path):
        assert_sidecar(source)

    round0_rows_all = load_jsonl(round0_path)
    # Dev/Test remain scope-audit metadata only. Their candidate正文 is not
    # read into active family or duplicate calculations.
    round0_rows = [row for row in round0_rows_all if str(row.get("split")) == "train"]
    e1q0_rows = load_jsonl(e1q0_path)
    e1q0_dialogue_manifest = {str(row["root_id"]): row for row in load_jsonl(e1q0_dialogue_path)}
    packets = {str(row["root_id"]): row for row in load_jsonl(packet_path)}
    e1q1_rows, e1q1_dialogues, e1q1_files = load_e1q1()
    splits = root_split(round0_rows_all)
    if any(root not in splits for root in e1q0_dialogue_manifest):
        raise RuntimeError("e1q0_root_missing_round0_split")
    if any(str(row.get("root_id")) not in splits for row in e1q1_rows):
        raise RuntimeError("e1q1_root_missing_round0_split")

    # Prompt gates and per-model manifests are independent coverage evidence.
    e1q1_gate_path = E1Q1_SOLVER_RUN / "program_gate_results.jsonl"
    e1q0_gate_path = E1Q0_SOLVER_RUN / "program_gate_results.jsonl"
    assert_sidecar(e1q1_gate_path)
    assert_sidecar(e1q0_gate_path)
    e1q1_gates = {(str(row.get("branch")), str(row.get("candidate_id"))): row for row in load_jsonl(e1q1_gate_path)}
    e1q0_gates = {(str(row.get("branch")), str(row.get("candidate_id"))): row for row in load_jsonl(e1q0_gate_path)}
    model_coverage: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in MODELS}
    for model in MODELS:
        path = E1Q1_SOLVER_RUN / model / "feedback" / "difficulty_manifest.jsonl"
        assert_sidecar(path)
        for row in load_jsonl(path):
            model_coverage[model][str(row["candidate_id"])] = row
    e1q1_covered = {cid for model in MODELS for cid in model_coverage[model]}
    e1q0_model_coverage: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in MODELS}
    for model in MODELS:
        for branch in ("control", "feedback"):
            path = E1Q0_SOLVER_RUN / model / branch / "difficulty_manifest.jsonl"
            assert_sidecar(path)
            for row in load_jsonl(path):
                e1q0_model_coverage[model][str(row["candidate_id"])] = row

    # Rebuild the frozen Solver prompts locally for every E1Q1 candidate. This
    # is tokenization and hashing only: no backend/model request is made. It
    # lets missing candidates participate in exact prompt-cluster routing
    # without guessing from evidence annotations.
    from transformers import AutoTokenizer
    from ..h4_round0_b0_k3_difficulty_v1.gate import gate_candidate
    from ..h4_round1_qa_feedback_solver_eval_v1 import BASE_MODEL as SOLVER_TOKENIZER_PATH
    tokenizer = AutoTokenizer.from_pretrained(str(SOLVER_TOKENIZER_PATH), local_files_only=True)
    rebuilt_e1q1_gates: dict[str, dict[str, Any]] = {}
    prompt_rebuild_mismatches: list[str] = []
    for candidate in e1q1_rows:
        cid = str(candidate["candidate_id"])
        root = str(candidate.get("root_id"))
        gate = gate_candidate(candidate, e1q1_dialogues[root], tokenizer, root_id=root)
        if gate.get("gate_status") != "PROGRAM_PASS":
            raise RuntimeError(f"e1q1_offline_prompt_rebuild_failed:{cid}:{gate.get('program_errors')}")
        rebuilt_e1q1_gates[cid] = gate
        recorded = e1q1_gates.get(("feedback", cid))
        if recorded and canonical(gate.get("prompt_audit")) != canonical(recorded.get("prompt_audit")):
            prompt_rebuild_mismatches.append(cid)
    if prompt_rebuild_mismatches:
        raise RuntimeError(f"e1q1_prompt_rebuild_mismatch:{prompt_rebuild_mismatches[:10]}")
    e1q1_contract_sha = next(iter(next(iter(model_coverage.values())).values()))["solver_contract_sha256"]
    e1q0_contract_sha = next(iter(next(iter(e1q0_model_coverage.values())).values()))["solver_contract_sha256"]

    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for outer in round0_rows:
        candidate = dict(outer.get("candidate") or {})
        cid = str(outer["candidate_id"])
        record = {
            "candidate_id": cid,
            "corpus_id": "ROUND0",
            "candidate": candidate,
            "root_id": str(outer.get("root_id") or candidate.get("root_id")),
            "normalized_experiment_arm": arm_label(None, "ROUND0"),
            "split": str(outer.get("split")),
            "split_source": "round0_manifest",
            "context_signature": str(outer.get("dialogue_sha256") or ""),
            "dialogue_path": outer.get("dialogue_path"),
            "solver_prompt_fingerprint": None,
            "prompt_fingerprint_status": "NOT_RECORDED_FOR_ROUND0_SOURCE",
            "lineage_confidence": "EXPLICIT" if candidate.get("source_candidate_id") else UNKNOWN,
            "lineage_source_candidate_ids": [str(candidate["source_candidate_id"])] if candidate.get("source_candidate_id") else [],
            "lineage_parent_candidate_ids": [],
            "lineage_family_id": sha256_value(["ROUND0", candidate.get("root_id"), candidate.get("source_candidate_id")]),
            "feedback_target_hit": None,
            "feedback_contract_hit": None,
            "r02_touch": any(ref_suffix(ref).startswith("R02:") for ref in candidate.get("evidence_refs") or []),
            "r02_ref_aligned": None,
            "off_target": False,
            "structural_status": "PASS" if candidate.get("schema_valid", True) and not candidate.get("schema_errors") else "STRUCTURAL_QUARANTINE",
            "structural_errors": list(candidate.get("schema_errors") or []),
            "evidence_refs_valid": True,
            "invalid_evidence_refs": [],
            "solver_coverage": None,
        }
        rows.append(record); by_id[cid] = record; source_records[cid] = candidate

    for candidate in e1q0_rows:
        cid = str(candidate["candidate_id"])
        branch = str(candidate.get("branch") or "")
        gate = e1q0_gates.get((branch, cid))
        context = e1q0_dialogue_manifest.get(str(candidate.get("root_id")), {})
        context_sha = context.get(f"{branch}_sha256")
        record = {
            "candidate_id": cid,
            "corpus_id": "E1Q0",
            "candidate": candidate,
            "root_id": str(candidate.get("root_id")),
            "normalized_experiment_arm": arm_label(branch, "E1Q0"),
            "split": splits[str(candidate.get("root_id"))],
            "split_source": "root_membership",
            "context_signature": str(context_sha or (gate or {}).get("dialogue_sha256") or ""),
            "dialogue_path": context.get(f"{branch}_path"),
            "solver_prompt_fingerprint": gate_fingerprint(gate, (e1q0_model_coverage[MODELS[0]].get(cid) or {}).get("solver_contract_sha256")) if gate else None,
            "prompt_fingerprint_status": "RECORDED_GATE" if gate else "NOT_EVALUATED",
            "lineage_confidence": "EXPLICIT" if candidate.get("source_candidate_id") else UNKNOWN,
            "lineage_source_candidate_ids": [str(candidate["source_candidate_id"])] if candidate.get("source_candidate_id") else [],
            "lineage_parent_candidate_ids": [],
            "lineage_family_id": sha256_value(["E1Q0", candidate.get("root_id"), branch, candidate.get("source_candidate_id")]),
            "feedback_target_hit": None,
            "feedback_contract_hit": None,
            "r02_touch": any(ref_suffix(ref).startswith("R02:") for ref in candidate.get("evidence_refs") or []),
            "r02_ref_aligned": None,
            "off_target": False,
            "structural_status": "PASS" if candidate.get("schema_valid", True) and not candidate.get("schema_errors") else "STRUCTURAL_QUARANTINE",
            "structural_errors": list(candidate.get("schema_errors") or []),
            "evidence_refs_valid": True,
            "invalid_evidence_refs": [],
            "solver_coverage": {
                "clean": "COMPLETE" if cid in e1q0_model_coverage["M_clean_policy"] else "MISSING",
                "inclusive": "COMPLETE" if cid in e1q0_model_coverage["M_inclusive_policy"] else "MISSING",
                "candidate_complete_both_models": all(cid in e1q0_model_coverage[model] for model in MODELS),
                "solver_eval_status": "CANDIDATE_COMPLETE" if all(cid in e1q0_model_coverage[model] for model in MODELS) else "INCOMPLETE",
                "cluster_representative_candidate_id": None,
            },
        }
        rows.append(record); by_id[cid] = record; source_records[cid] = candidate

    e1q1_records: list[dict[str, Any]] = []
    for candidate in e1q1_rows:
        cid = str(candidate["candidate_id"])
        root = str(candidate.get("root_id"))
        dialogue = e1q1_dialogues[root]
        recorded_gate = e1q1_gates.get(("feedback", cid))
        gate = rebuilt_e1q1_gates[cid]
        refs = [str(ref) for ref in (candidate.get("evidence_refs") or [])]
        valid_refs = dialogue_refs(dialogue)
        valid_suffixes = {ref_suffix(ref) for ref in valid_refs}
        r02_refs = {ref_suffix(ref) for ref in refs if ref_suffix(ref).startswith("R02:")}
        packet_r02 = {str(ref) for ref in (packets.get(root, {}).get("r02_adherence", {}).get("r02_turn_refs") or [])}
        r02_touch = bool(r02_refs)
        # Alignment means the refs exist in this root's complete dialogue.
        # Packet adherence is retained as source metadata, but is not used to
        # equate a syntactic R02 touch with semantic target adherence.
        r02_aligned = bool(r02_refs) and r02_refs.issubset(valid_suffixes)
        slot = str(candidate.get("slot_type") or "")
        structure = candidate_structure(candidate, valid_refs)
        if slot in {"TARGETED_EVOLUTION", "R02_COVERAGE_QA"}:
            contract_hit = r02_aligned
            off_target = not r02_aligned
        elif slot == "CONTRASTIVE_QA":
            contract_hit = structure["evidence_refs_valid"]
            off_target = not contract_hit
        else:
            contract_hit = False
            off_target = True
        record = {
            "candidate_id": cid,
            "corpus_id": "E1Q1",
            "candidate": candidate,
            "root_id": root,
            "normalized_experiment_arm": arm_label("feedback", "E1Q1"),
            "split": splits[root],
            "split_source": "root_membership",
            "context_signature": str(dialogue.get("complete_dialogue_sha256") or ""),
            "dialogue_path": str((E1Q1_GENERATION_RUN / root / "dialogue_source.json").resolve()),
            "solver_prompt_fingerprint": gate_fingerprint(gate, e1q1_contract_sha),
            "prompt_fingerprint_status": "RECORDED_GATE" if recorded_gate else "REBUILT_FROZEN_CONTRACT",
            "lineage_confidence": "EXPLICIT" if candidate.get("source_candidate_id") else UNKNOWN,
            "lineage_source_candidate_ids": [str(candidate["source_candidate_id"])] if candidate.get("source_candidate_id") else [],
            "lineage_parent_candidate_ids": [],
            "lineage_family_id": sha256_value(["E1Q1", root, candidate.get("source_candidate_id"), slot, candidate.get("intent_id")]),
            "feedback_target_hit": UNKNOWN,
            "feedback_contract_hit": contract_hit,
            "r02_touch": r02_touch,
            "r02_ref_aligned": r02_aligned,
            "off_target": off_target,
            "question_type": candidate.get("question_type"),
            "slot_type": candidate.get("slot_type"),
            **structure,
            "solver_coverage": {
                "clean": "COMPLETE" if cid in model_coverage["M_clean_policy"] else "MISSING",
                "inclusive": "COMPLETE" if cid in model_coverage["M_inclusive_policy"] else "MISSING",
                "candidate_complete_both_models": all(cid in model_coverage[model] for model in MODELS),
                "solver_eval_status": "CANDIDATE_COMPLETE" if all(cid in model_coverage[model] for model in MODELS) else "MISSING_BOTH",
                "cluster_representative_candidate_id": None,
            },
        }
        rows.append(record); by_id[cid] = record; source_records[cid] = candidate; e1q1_records.append(record)

    # The four-axis signatures are attached after all source records exist.
    for record in rows:
        candidate = record["candidate"]
        record["surface_signature"] = surface_signature(candidate)
        record["normalized_stem_signature"] = stem_signature(candidate)
        record["evidence_signature"] = evidence_signature(candidate)
        record["evidence_ref_count"] = len(candidate.get("evidence_refs") or [])
        record["semantic_family_status"] = UNKNOWN

    surface_index: dict[str, list[str]] = defaultdict(list)
    provenance_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for record in rows:
        surface_index[record["surface_signature"]].append(record["candidate_id"])
        provenance_index[(record["surface_signature"], record["context_signature"], record["evidence_signature"])].append(record["candidate_id"])
    for record in rows:
        cid = record["candidate_id"]
        same_surface = sorted(value for value in surface_index[record["surface_signature"]] if value != cid)
        same_provenance = sorted(value for value in provenance_index[(record["surface_signature"], record["context_signature"], record["evidence_signature"])] if value != cid)
        record["surface_match_ids"] = same_surface
        record["context_equal_surface_match_ids"] = sorted(other for other in same_surface if by_id[other]["context_signature"] == record["context_signature"])
        record["evidence_equal_surface_match_ids"] = sorted(other for other in same_surface if by_id[other]["evidence_signature"] == record["evidence_signature"])
        record["exact_surface_qa_duplicate"] = bool(same_surface)
        record["exact_same_provenance"] = bool(same_provenance)
        record["exact_same_provenance_match_ids"] = same_provenance
        record["evidence_annotation_variant"] = bool(same_surface) and any(
            by_id[other]["context_signature"] == record["context_signature"] and by_id[other]["evidence_signature"] != record["evidence_signature"]
            for other in same_surface
        )
        record["context_variant"] = bool(same_surface) and any(
            by_id[other]["context_signature"] != record["context_signature"] for other in same_surface
        )
        prompt_matches = [other for other in same_provenance if record.get("solver_prompt_fingerprint") and by_id[other].get("solver_prompt_fingerprint")]
        record["solver_prompt_equal_for_provenance_matches"] = [other for other in prompt_matches if by_id[other]["solver_prompt_fingerprint"] == record["solver_prompt_fingerprint"]]
        record["solver_prompt_cluster_status"] = (
            "SAME_PROMPT_CONFIRMED" if record.get("solver_prompt_fingerprint") and any(by_id[other].get("solver_prompt_fingerprint") == record["solver_prompt_fingerprint"] for other in same_surface)
            else "PROMPT_VARIANT_UNRESOLVED" if record["context_equal_surface_match_ids"] and record["evidence_equal_surface_match_ids"] != record["context_equal_surface_match_ids"]
            else "NO_SURFACE_MATCH"
        )
        record["true_composite_near_duplicate"] = UNKNOWN
        record["duplicate_labels"] = {
            "exact_surface_equal": record["exact_surface_qa_duplicate"],
            "context_equal": bool(record["context_equal_surface_match_ids"]),
            "evidence_equal": bool(record["evidence_equal_surface_match_ids"]),
            "evidence_annotation_variant": record["evidence_annotation_variant"],
            "context_variant": record["context_variant"],
            "lexical_similarity_candidate": False,
            "semantic_family_status": UNKNOWN,
        }
        record["feedback_labels"] = {
            "r02_touch": record["r02_touch"], "r02_ref_aligned": record["r02_ref_aligned"],
            "feedback_contract_hit": record["feedback_contract_hit"], "feedback_target_hit": record["feedback_target_hit"],
            "off_target": record["off_target"],
        }
        record["lineage"] = {
            "source_candidate_id": (record["lineage_source_candidate_ids"] or [None])[0],
            "intent_id": record["candidate"].get("intent_id"), "slot_type": record["candidate"].get("slot_type"),
            "generation_source": record["candidate"].get("generation_source"),
            "lineage_confidence": record["lineage_confidence"],
            "parent_candidate_ids": record["lineage_parent_candidate_ids"],
        }

    # Pair-level audit is intentionally limited to E1Q1's historical E1Q0
    # comparison, E1Q1 internal pairs, and each root's compiler seed.
    duplicate_pairs: list[dict[str, Any]] = []
    e1q0_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        if record["corpus_id"] == "E1Q0":
            e1q0_by_root[record["root_id"]].append(record)
    for record in e1q1_records:
        for other in e1q0_by_root.get(record["root_id"], []):
            metrics = lexical_metrics(record["candidate"], other["candidate"])
            if metrics["surface_equal"] or is_lexical_near(metrics):
                duplicate_pairs.append({
                    "left_candidate_id": record["candidate_id"], "right_candidate_id": other["candidate_id"],
                    "relation_scope": "E1Q1_VS_SAME_ROOT_E1Q0", "metrics": metrics,
                    "review_labels": ["exact_surface_equal"] if metrics["surface_equal"] else ["historical_corpus_near_duplicate"],
                    "semantic_claim": UNKNOWN,
                })
    for left, right in itertools.combinations(e1q1_records, 2):
        metrics = lexical_metrics(left["candidate"], right["candidate"])
        if metrics["surface_equal"] or is_lexical_near(metrics):
            labels = ["exact_surface_equal"] if metrics["surface_equal"] else ["lexical_similarity_candidate"]
            if metrics["surface_equal"] and left["context_signature"] == right["context_signature"] and left["evidence_signature"] != right["evidence_signature"]:
                labels.append("evidence_shift_variant")
            duplicate_pairs.append({
                "left_candidate_id": left["candidate_id"], "right_candidate_id": right["candidate_id"],
                "relation_scope": "E1Q1_INTERNAL", "metrics": metrics, "review_labels": labels,
                "semantic_claim": UNKNOWN,
            })
    for record in e1q1_records:
        packet = packets.get(record["root_id"])
        if not packet:
            continue
        seed = source_seed(packet)
        metrics = lexical_metrics(record["candidate"], seed)
        if metrics["surface_equal"] or is_lexical_near(metrics):
            labels = ["source_seed_near_duplicate"] if not metrics["surface_equal"] else ["exact_surface_equal", "source_seed_near_duplicate"]
            duplicate_pairs.append({
                "left_candidate_id": record["candidate_id"], "right_candidate_id": seed["candidate_id"],
                "relation_scope": "E1Q1_VS_COMPILER_SOURCE_SEED", "metrics": metrics,
                "review_labels": labels, "semantic_claim": UNKNOWN,
            })
    for pair in duplicate_pairs:
        left = by_id.get(pair["left_candidate_id"])
        right = by_id.get(pair["right_candidate_id"])
        if left and right and left["surface_signature"] == right["surface_signature"] and left["evidence_signature"] != right["evidence_signature"]:
            pair["review_labels"].append("evidence_shift_variant")

    # Family manifest is a deterministic lexical/provenance index, not a
    # semantic family adjudication.
    family_rows: list[dict[str, Any]] = []
    for record in rows:
        candidate = record["candidate"]
        family_rows.append({
            "candidate_id": record["candidate_id"], "corpus_id": record["corpus_id"],
            "root_id": record["root_id"], "normalized_experiment_arm": record["normalized_experiment_arm"],
            "split": record["split"], "bank_activation": "ACTIVE" if record["split"] == "train" and record["structural_status"] != "STRUCTURAL_QUARANTINE" else "EVAL_LOCKED",
            "normalized_stem_signature": record["normalized_stem_signature"], "surface_signature": record["surface_signature"],
            "context_signature": record["context_signature"], "evidence_signature": record["evidence_signature"],
            "solver_prompt_fingerprint": record["solver_prompt_fingerprint"],
            "prompt_fingerprint_status": record["prompt_fingerprint_status"],
            "lineage_family_id": record["lineage_family_id"], "lineage_confidence": record["lineage_confidence"],
            "lineage_source_candidate_ids": record["lineage_source_candidate_ids"],
            "semantic_family_status": UNKNOWN, "exact_surface_qa_duplicate": record["exact_surface_qa_duplicate"],
            "exact_same_provenance": record["exact_same_provenance"],
            "evidence_annotation_variant": record["evidence_annotation_variant"], "context_variant": record["context_variant"],
            "lexical_similarity_candidate": any(
                pair["left_candidate_id"] == record["candidate_id"] and "lexical_similarity_candidate" in pair["review_labels"]
                for pair in duplicate_pairs
            ),
            "source_seed_near_duplicate": any(
                pair["left_candidate_id"] == record["candidate_id"] and "source_seed_near_duplicate" in pair["review_labels"]
                for pair in duplicate_pairs
            ),
            "historical_corpus_near_duplicate": any(
                pair["left_candidate_id"] == record["candidate_id"] and "historical_corpus_near_duplicate" in pair["review_labels"]
                for pair in duplicate_pairs
            ),
            "evidence_shift_variant": any(
                pair["left_candidate_id"] == record["candidate_id"] and "evidence_shift_variant" in pair["review_labels"]
                for pair in duplicate_pairs
            ),
            "true_composite_near_duplicate": UNKNOWN,
            "feedback_target_hit": record["feedback_target_hit"], "feedback_contract_hit": record["feedback_contract_hit"],
            "r02_touch": record["r02_touch"], "r02_ref_aligned": record["r02_ref_aligned"],
            "off_target": record["off_target"], "structural_status": record["structural_status"],
            "question_type": candidate.get("question_type"), "slot_type": candidate.get("slot_type"),
            "solver_coverage": record.get("solver_coverage"),
            "duplicate_labels": record["duplicate_labels"],
            "feedback_labels": record["feedback_labels"],
        })

    active_records = [record for record in rows if record["split"] == "train" and record["structural_status"] != "STRUCTURAL_QUARANTINE"]
    provisional = []
    for record in active_records:
        provisional.append({
            "candidate_id": record["candidate_id"], "corpus_id": record["corpus_id"],
            "normalized_experiment_arm": record["normalized_experiment_arm"], "split": record["split"],
            "bank_entry_status": "PROVISIONAL", "candidate": record["candidate"],
            "surface_signature": record["surface_signature"], "context_signature": record["context_signature"],
            "evidence_signature": record["evidence_signature"], "lineage_confidence": record["lineage_confidence"],
            "semantic_family_status": UNKNOWN, "solver_coverage": record.get("solver_coverage"),
            "duplicate_labels": record.get("duplicate_labels"), "feedback_labels": record.get("feedback_labels"),
        })

    # Existing E1Q1 coverage is exact candidate-id coverage. Prompt hashes for
    # missing candidates are rebuilt locally with the frozen contract.
    e1q1_coverage_rows = []
    for record in e1q1_records:
        cid = record["candidate_id"]
        models_present = [model for model in MODELS if cid in model_coverage[model]]
        e1q1_coverage_rows.append({
            "candidate_id": cid, "models_present": models_present,
            "complete_both_models": len(models_present) == len(MODELS),
            "gate_prompt_recorded": record["prompt_fingerprint_status"] == "RECORDED_GATE",
            "candidate_pool_sha256_by_model": {model: model_coverage[model][cid].get("candidate_pool_sha256") for model in models_present},
        })
    complete_ids = {row["candidate_id"] for row in e1q1_coverage_rows if row["complete_both_models"]}
    missing_both_ids = {row["candidate_id"] for row in e1q1_coverage_rows if not row["models_present"]}
    one_model_ids = {row["candidate_id"] for row in e1q1_coverage_rows if 0 < len(row["models_present"]) < len(MODELS)}
    existing_solver_audit = {
        "version": REROUTE_VERSION,
        "candidate_count": len(e1q1_records), "complete_both_model_count": len(complete_ids),
        "missing_both_model_count": len(missing_both_ids), "partial_model_count": len(one_model_ids),
        "complete_candidate_ids": sorted(complete_ids), "missing_both_candidate_ids": sorted(missing_both_ids),
        "partial_candidate_ids": sorted(one_model_ids), "model_counts": {model: len(model_coverage[model]) for model in MODELS},
        "gate_candidate_count": len(e1q1_gates), "gate_id_set_matches_complete_union": set(e1q1_gates) >= {("feedback", cid) for cid in complete_ids},
        "offline_prompt_rebuild_count": len(rebuilt_e1q1_gates),
        "recorded_gate_rebuild_match_count": len(e1q1_gates),
        "recorded_gate_rebuild_mismatch_count": len(prompt_rebuild_mismatches),
        "rerun_count": 0, "online_calls": 0,
        "coverage_rows": e1q1_coverage_rows,
    }

    # Exact Solver-input groups share one future request. Evidence annotation
    # changes only split a cluster when they actually change the frozen prompt.
    # Existing E1Q0 results may be representatives because both evaluations
    # use the same trained models and B0 Solver contract.
    e1q1_prompt_groups: dict[str, list[str]] = defaultdict(list)
    for record in e1q1_records:
        e1q1_prompt_groups[str(record["solver_prompt_fingerprint"] or "UNAVAILABLE")].append(record["candidate_id"])
    e1q0_complete_ids = set(e1q0_model_coverage[MODELS[0]]) & set(e1q0_model_coverage[MODELS[1]])
    e1q0_prompt_groups: dict[str, list[str]] = defaultdict(list)
    for record in rows:
        if record["corpus_id"] == "E1Q0" and record["candidate_id"] in e1q0_complete_ids and record.get("solver_prompt_fingerprint"):
            e1q0_prompt_groups[str(record["solver_prompt_fingerprint"])].append(record["candidate_id"])
    missing_manifest: list[dict[str, Any]] = []
    missing_decisions: list[dict[str, Any]] = []
    for record in sorted(e1q1_records, key=lambda item: item["candidate_id"]):
        cid = record["candidate_id"]
        if record["structural_status"] == "STRUCTURAL_QUARANTINE" or cid not in missing_both_ids:
            continue
        fingerprint = str(record["solver_prompt_fingerprint"] or "UNAVAILABLE")
        group = sorted(e1q1_prompt_groups[fingerprint])
        complete_in_group = sorted((set(group) & complete_ids) | set(e1q0_prompt_groups.get(fingerprint, [])))
        is_representative = cid == group[0]
        if complete_in_group:
            decision = "SKIP_EXACT_SOLVER_PROMPT_CLUSTER_COVERED"
        elif len(group) > 1 and not is_representative:
            decision = "SKIP_EXACT_SOLVER_PROMPT_CLUSTER_REPRESENTED"
        else:
            decision = "REQUEST_MISSING_BOTH_MODELS"
            missing_manifest.append({
                "candidate_id": cid, "root_id": record["root_id"], "question_type": record["candidate"].get("question_type"),
                "normalized_experiment_arm": record["normalized_experiment_arm"], "split": record["split"],
                "surface_signature": record["surface_signature"], "context_signature": record["context_signature"],
                "evidence_signature": record["evidence_signature"], "solver_prompt_fingerprint": record["solver_prompt_fingerprint"],
                "prompt_fingerprint_status": "REBUILT_FROZEN_CONTRACT_WILL_BE_VERIFIED_BY_SOLVER",
                "request_reason": "MISSING_BOTH_MODELS" if len(group) == 1 else "EXACT_SOLVER_PROMPT_CLUSTER_REPRESENTATIVE",
                "cluster_candidate_ids": group, "external_complete_representative_ids": sorted(e1q0_prompt_groups.get(fingerprint, [])),
                "semantic_family_status": UNKNOWN,
            })
        missing_decisions.append({"candidate_id": cid, "decision": decision, "cluster_candidate_ids": group, "complete_in_cluster": complete_in_group})

    for record in e1q1_records:
        fingerprint = str(record["solver_prompt_fingerprint"] or "UNAVAILABLE")
        group = sorted(e1q1_prompt_groups[fingerprint])
        complete_in_cluster = sorted((set(group) & complete_ids) | set(e1q0_prompt_groups.get(fingerprint, [])))
        representative = complete_in_cluster[0] if complete_in_cluster else (group[0] if len(group) > 1 else None)
        record["solver_coverage"]["cluster_representative_candidate_id"] = representative
        record["solver_coverage"]["solver_eval_status"] = (
            "CANDIDATE_COMPLETE" if record["candidate_id"] in complete_ids
            else "CLUSTER_REPRESENTATIVE_AVAILABLE" if representative and representative != record["candidate_id"]
            else "MISSING_BOTH"
        )
        record["duplicate_labels"].update({
            "lexical_similarity_candidate": any(
                pair["left_candidate_id"] == record["candidate_id"] and "lexical_similarity_candidate" in pair["review_labels"]
                for pair in duplicate_pairs
            ),
        })

    arms = Counter(record["normalized_experiment_arm"] for record in rows)
    split_counts = Counter(record["split"] for record in rows)
    active_split_counts = Counter(record["split"] for record in active_records)
    arm_audit = {
        "version": REROUTE_VERSION,
        "normalization": {
            "control": "E0Q0_CONTROL_BLIND", "feedback": "E1Q0_FEEDBACK_BLIND", "E1Q1_FULL_QA_FEEDBACK_COMPOSITE": "E1Q1_FEEDBACK_GUIDED",
            "ROUND0_PARENT_PLUS_R01": "ROUND0_PARENT_PLUS_R01",
        },
        "counts": dict(sorted(arms.items())), "unmapped_branch_or_arm_values": [],
        "expected_counts": {"E0Q0_CONTROL_BLIND": 276, "E1Q0_FEEDBACK_BLIND": 254, "E1Q1_FEEDBACK_GUIDED": 238},
        "counts_match_expected": all(arms[key] == value for key, value in {"E0Q0_CONTROL_BLIND": 276, "E1Q0_FEEDBACK_BLIND": 254, "E1Q1_FEEDBACK_GUIDED": 238}.items()),
        "source_scope_report_sha256": sha256_file(scope_report_path),
    }
    split_guard = {
        "version": REROUTE_VERSION, "policy": "ROUND0_DEV_TEST_AUDIT_LOCKED_TRAIN_ACTIVE_ONLY",
        "all_entry_counts": {"dev": sum(1 for row in round0_rows_all if str(row.get("split")) == "dev"), "test": sum(1 for row in round0_rows_all if str(row.get("split")) == "test"), "train": len(active_records)}, "active_entry_counts": dict(sorted(active_split_counts.items())),
        "round0_dev_locked_count": sum(1 for row in round0_rows_all if str(row.get("split")) == "dev"),
        "round0_test_locked_count": sum(1 for row in round0_rows_all if str(row.get("split")) == "test"),
        "dev_test_active_entries": sum(1 for row in active_records if row["split"] in {"dev", "test"}),
        "guard_pass": sum(1 for row in active_records if row["split"] in {"dev", "test"}) == 0,
        "e1q0_e1q1_train_root_membership_only": True,
    }
    scope_v2_rows = [
        {
            "scope_version": REROUTE_VERSION + "_scope_v2",
            "corpus_id": "ROUND0_ALL",
            "candidate_count": len(round0_rows_all),
            "split_counts": dict(sorted(Counter(str(row.get("split")) for row in round0_rows_all).items())),
            "active_candidate_count": len(round0_rows),
            "active_policy": "train_only",
            "source_manifest": str(round0_path.resolve()),
            "source_manifest_sha256": sha256_file(round0_path),
        },
        {
            "scope_version": REROUTE_VERSION + "_scope_v2",
            "corpus_id": "E1Q0",
            "candidate_count": len(e1q0_rows),
            "arm_counts": {"E0Q0_CONTROL_BLIND": sum(str(row.get("branch")) == "control" for row in e1q0_rows), "E1Q0_FEEDBACK_BLIND": sum(str(row.get("branch")) == "feedback" for row in e1q0_rows)},
            "active_candidate_count": len(e1q0_rows),
            "active_policy": "train_root_membership",
            "source_manifest": str(e1q0_path.resolve()),
            "source_manifest_sha256": sha256_file(e1q0_path),
        },
        {
            "scope_version": REROUTE_VERSION + "_scope_v2",
            "corpus_id": "E1Q1",
            "candidate_count": len(e1q1_rows),
            "arm_counts": {"E1Q1_FEEDBACK_GUIDED": len(e1q1_rows)},
            "active_candidate_count": len(e1q1_rows),
            "active_policy": "train_root_membership",
            "missing_generation_roots": ["cc-train-episode-96462"],
            "source_manifest": str(packet_path.resolve()),
            "source_manifest_sha256": sha256_file(packet_path),
        },
    ]

    feedback_summary = {
        "version": REROUTE_VERSION,
        "e1q1_total": len(e1q1_records),
        "by_slot_type": dict(sorted(Counter(str(record["candidate"].get("slot_type") or "UNKNOWN") for record in e1q1_records).items())),
        "r02_touch_count": sum(bool(record["r02_touch"]) for record in e1q1_records),
        "r02_ref_aligned_count": sum(bool(record["r02_ref_aligned"]) for record in e1q1_records),
        "feedback_contract_hit_counts": dict(sorted(Counter(str(record["feedback_contract_hit"]) for record in e1q1_records).items())),
        "feedback_target_hit_counts": dict(sorted(Counter(str(record["feedback_target_hit"]) for record in e1q1_records).items())),
        "off_target_count": sum(bool(record["off_target"]) for record in e1q1_records),
        "structural_quarantine_count": sum(record["structural_status"] == "STRUCTURAL_QUARANTINE" for record in e1q1_records),
        "semantic_target_audit": "NOT_PERFORMED_OFFLINE_REVIEW_LABEL_ONLY",
    }
    family_summary = {
        "version": REROUTE_VERSION, "candidate_count": len(family_rows), "active_count": len(provisional),
        "corpus_counts": dict(sorted(Counter(record["corpus_id"] for record in family_rows).items())),
        "arm_counts": dict(sorted(Counter(record["normalized_experiment_arm"] for record in family_rows).items())),
        "split_counts": dict(sorted(Counter(record["split"] for record in family_rows).items())),
        "active_split_counts": dict(sorted(active_split_counts.items())),
        "exact_surface_duplicate_count": sum(record["exact_surface_qa_duplicate"] for record in family_rows),
        "exact_provenance_duplicate_count": sum(record["exact_same_provenance"] for record in family_rows),
        "semantic_family_status_counts": {UNKNOWN: len(family_rows)},
        "lineage_confidence_counts": dict(sorted(Counter(record["lineage_confidence"] for record in family_rows).items())),
        "duplicate_pair_count": len(duplicate_pairs),
    }
    final_report = {
        "version": REROUTE_VERSION, "status": "E1Q1_MULTILABEL_REROUTING_COMPLETE_WITH_REVIEW_LABELS",
        "next_status": "READY_FOR_MISSING_E1Q1_SOLVER_EVALUATION",
        "statuses": ["QUESTION_BANK_ARM_AND_SPLIT_GUARDS_PASS", "E1Q1_MULTILABEL_REROUTING_COMPLETE_WITH_REVIEW_LABELS", "PROVISIONAL_QUESTION_BANK_ENTRIES_READY", "READY_FOR_MISSING_E1Q1_SOLVER_EVALUATION"],
        "offline_only": True, "online_calls": 0, "source_manifests_rewritten": False,
        "arm_label_normalization_pass": arm_audit["counts_match_expected"],
        "split_usage_guard_pass": split_guard["guard_pass"],
        "structural_quarantine_count": feedback_summary["structural_quarantine_count"],
        "e1q1_candidate_count": len(e1q1_records), "e1q1_existing_complete_both_model_count": len(complete_ids),
        "missing_solver_request_count": len(missing_manifest), "missing_solver_decision_count": len(missing_decisions),
        "missing_generation_roots": ["cc-train-episode-96462"],
        "semantic_labels": "LEXICAL_AND_REVIEW_ONLY_NO_CONFIRMED_SEMANTIC_PARAPHRASE_OR_FAMILY",
    }
    report_md = "\n".join([
        f"# {REROUTE_VERSION}", "", "Offline multilabel rerouting completed.", "",
        f"- Arm normalization pass: `{arm_audit['counts_match_expected']}`",
        f"- Split guard pass: `{split_guard['guard_pass']}` (dev/test active entries: {split_guard['dev_test_active_entries']})",
        f"- E1Q1 candidates: {len(e1q1_records)}; complete two-model coverage: {len(complete_ids)}",
        f"- Missing solver representatives: {len(missing_manifest)}",
        f"- Structural quarantine: {feedback_summary['structural_quarantine_count']}",
        "- Semantic status: review labels only; no semantic paraphrase/family claim.", "",
        "Next stage: missing E1Q1 solver evaluation. No model was called in this stage.", "",
    ])

    write_artifact(run / "arm_label_normalization_audit.json", arm_audit)
    write_artifact(run / "split_usage_guard.json", split_guard)
    write_artifact(run / "scope_v2_manifest.jsonl", scope_v2_rows)
    write_artifact(run / "e1q1_multilabel_routing.jsonl", [
        {key: value for key, value in record.items() if key != "candidate"} for record in e1q1_records
    ])
    write_artifact(run / "duplicate_match_manifest.jsonl", duplicate_pairs)
    write_artifact(run / "duplicate_definition_audit.json", {
        "version": REROUTE_VERSION,
        "axes": ["surface_signature", "context_signature", "evidence_signature", "solver_prompt_fingerprint"],
        "exact_surface_qa_duplicate": "surface signature equal; display labels canonicalized to option/item text",
        "exact_same_provenance": "surface + context + evidence signatures equal",
        "solver_prompt_fingerprint": "hash of actual gate prompt hashes and frozen prompt parameters; null when unevaluated",
        "semantic_policy": "lexical/review labels only; true_composite_near_duplicate remains UNKNOWN",
        "lineage_policy": "explicit source IDs are retained; missing parent IDs remain UNKNOWN and are never fabricated",
    })
    write_artifact(run / "feedback_target_routing_summary.json", feedback_summary)
    write_artifact(run / "question_family_manifest.jsonl", family_rows)
    write_artifact(run / "question_family_summary.json", family_summary)
    write_artifact(run / "provisional_question_bank_entries.jsonl", provisional)
    write_artifact(run / "existing_solver_coverage_audit.json", existing_solver_audit)
    write_artifact(run / "missing_solver_evaluation_manifest.jsonl", missing_manifest)
    write_artifact(run / "missing_solver_request_plan.json", {
        "version": REROUTE_VERSION, "online_calls": 0, "request_count": len(missing_manifest),
        "candidate_ids": [row["candidate_id"] for row in missing_manifest],
        "decision_counts": dict(sorted(Counter(row["decision"] for row in missing_decisions).items())),
        "decisions": missing_decisions,
        "coverage_policy": "candidate_id_exact_and_exact_solver_prompt_cluster_representative",
        "prompt_fingerprint_policy": "rebuild_with_frozen_contract_offline_then_verify_against_actual_solver_gate",
    })
    write_artifact(run / "final_multilabel_rerouting_report.json", final_report)
    write_artifact(run / "final_multilabel_rerouting_report.md", report_md)
    write_artifact(run / "run_identity.json", {
        "version": REROUTE_VERSION, "output_dir": str(run), "offline_only": True,
        "source_scope_report": str(scope_report_path.resolve()), "source_scope_report_sha256": sha256_file(scope_report_path),
        "source_files": sorted({str(path.resolve()) for path in [round0_path, e1q0_path, e1q0_dialogue_path, packet_path, e1q1_gate_path, e1q0_gate_path] + e1q1_files}),
    })
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_REROUTE_RUN)
    args = parser.parse_args()
    print(json.dumps(reroute(args.run), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
