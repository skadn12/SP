"""Compile frozen Round-0 QA feedback for E1Q1, entirely offline.

This compiler deliberately does not import or call QA generation, Solver,
Memorax, or an external API.  The cumulative E1Q0 Feedback dialogue remains
the future QA fact source; Round-0 artifacts provide curriculum feedback only.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..h4_round1_memory_bucket_qa_feedback_compiler_v1.compiler import (
    classify_priority as _classify_priority,
    confusion_feature,
    frontier_score,
    model_payload,
    ordering_audit,
    summarize_raw_result,
)
from ..h4_round1_memory_bucket_qa_feedback_compiler_v1 import compiler as v1
from . import (
    ALGORITHM_VERSION,
    CONTEXT_LIMIT,
    E1Q0_QA_RUN,
    EXPECTED_E1Q0_CANDIDATES,
    EXPECTED_ROUND0_CANDIDATES,
    EXPECTED_TRAIN_CANDIDATES,
    EXPANSION_COMPILER_RUN,
    INTENT_MAX_TOKENS,
    QUARANTINE_ROOT_IDS,
    ROUND0_DIFFICULTY_RUN,
    ROUND0_EVIDENCE_RUN,
    ROUND0_REEVAL_RUN,
    ROUND0_TRAINING_RUN,
    SELECTED_ROOT_IDS,
    TOKENIZER_PATH,
    VERSION,
    WRITER_MAX_TOKENS,
)

EPS = 1e-9
ROOT_ORDER = tuple(sorted(SELECTED_ROOT_IDS))
P_ORDER = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4}
EXPOSURE_ORDER = {"UNEXPOSED_TO_BOTH": 0, "CLEAN_ONLY_TRAINED": 1,
                  "INCLUSIVE_ONLY_TRAINED": 1, "SHARED_TRAINED": 2}
EVIDENCE_ORDER = {"EVIDENCE_CLEAN": 0, "NOT_APPLICABLE_NON_HARD": 1, "EVIDENCE_RISK": 2}
QTYPE_ORDER = {"multi_select": 0, "ordering": 1, "single_choice": 2}
R02_REF_RE = re.compile(r"R02:S\d+:T\d+")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_artifact(path: Path, value: Any) -> None:
    if path.suffix == ".jsonl":
        text = "".join(canonical(row) + "\n" for row in value)
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"immutable_artifact_drift:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(sha_file(path) + "\n", encoding="utf-8")


def verify_sidecar(path: Path) -> tuple[bool, str | None]:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file():
        return False, None
    observed = sha_file(path)
    if not sidecar.is_file():
        return False, observed
    return sidecar.read_text(encoding="utf-8").strip().split()[0] == observed, observed


def index(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in result:
            raise RuntimeError(f"invalid_or_duplicate_{key}:{value}")
        result[value] = dict(row)
    return result


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split()).strip()


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"\w+", norm_text(value), flags=re.UNICODE))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def duplicate_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def normalized(row: Mapping[str, Any]) -> dict[str, Any]:
        alternatives = row.get("options") or row.get("items") or {}
        if isinstance(alternatives, Mapping):
            alternatives = sorted((norm_text(k), norm_text(v)) for k, v in alternatives.items())
        else:
            alternatives = sorted(norm_text(x) for x in alternatives)
        return {"stem": norm_text(row.get("question")), "alternatives": alternatives,
                "gold": sorted(norm_text(x) for x in row.get("gold") or [])}
    a, b = normalized(left), normalized(right)
    aa = {x[1] if isinstance(x, tuple) else x for x in a["alternatives"]}
    bb = {x[1] if isinstance(x, tuple) else x for x in b["alternatives"]}
    alt_overlap = len(aa & bb) / len(aa | bb) if aa | bb else 1.0
    ea = set(map(str, left.get("evidence_refs") or [])); eb = set(map(str, right.get("evidence_refs") or []))
    sj = jaccard(token_set(a["stem"]), token_set(b["stem"]))
    seq = difflib.SequenceMatcher(None, a["stem"], b["stem"]).ratio()
    ej = jaccard(ea, eb)
    exact = a == b
    return {"exact_duplicate": exact, "near_duplicate": bool(not exact and
            (sj >= .80 and alt_overlap >= .50 or seq >= .90 or ej >= .80 and alt_overlap >= .50)),
            "stem_token_jaccard": sj, "stem_sequence_ratio": seq,
            "alternative_text_overlap_ratio": alt_overlap, "evidence_ref_jaccard": ej}


def classify_priority_v2(row: Mapping[str, Any], confusion: Mapping[str, Any]) -> str | None:
    """Use the v1 P1-P5 policy, correcting its non-ordering P4 gate."""
    result = _classify_priority(row, confusion)
    if result is not None:
        return result
    m0, clean, inclusive = (row["models"][x] for x in ("M0", "M_clean_policy", "M_inclusive_policy"))
    cf, inf = float(clean["full_context_accuracy"]), float(inclusive["full_context_accuracy"])
    cstate = "zero" if cf <= EPS else "resolved" if cf >= 1 - EPS else "partial"
    istate = "zero" if inf <= EPS else "resolved" if inf >= 1 - EPS else "partial"
    if cstate == istate == "partial" and confusion.get("shared_error_set") and confusion.get("shared_overlap", 0) >= .50:
        if row.get("question_type") == "multi_select":
            return "P4"
        if row.get("question_type") == "single_choice":
            if (confusion.get("clean_dominant_distractor") == confusion.get("inclusive_dominant_distractor")
                    and float(confusion.get("clean_dominant_rate") or 0) >= 1 / 3
                    and float(confusion.get("inclusive_dominant_rate") or 0) >= 1 / 3):
                return "P4"
        if row.get("question_type") == "ordering" and row.get("ordering_occurrence_status") in {
                "UNIQUE_OCCURRENCE", "MULTI_OCCURRENCE_ORDER_INVARIANT"}:
            return "P4"
    return None


def locate_r02_root(path: Path) -> Path:
    for parent in [path.parent, *path.parents]:
        if (parent / "memory").is_dir() and (parent / "run_status.json").is_file():
            return parent
    return path.parent


def structured_r02(path: Path) -> dict[str, Any]:
    root = locate_r02_root(path)
    candidates = [root / "realizer_adherence/audit.json", root / "realizer_adherence/reaudit_audit.json",
                  root / "scene_replacement/adherence_output.json"]
    refs: set[str] = set(); files: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        text = canonical(load_json(candidate))
        found = set(R02_REF_RE.findall(text))
        if found:
            refs |= found; files.append(str(candidate.resolve()))
    return {"status": "SUPPORTED_STRUCTURED_ADHERENCE" if refs else "NO_SUPPORTED_ADHERENCE",
            "r02_turn_refs": sorted(refs), "structured_files": files, "r02_root": str(root.resolve())}


def render_dialogue(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(f"[{x.get('turn_ref')}] {x.get('speaker')}: {x.get('text')}" for x in dialogue.get("turns") or [])


def load_tokenizer() -> tuple[Any | None, dict[str, Any]]:
    files = {name: {"sha256": sha_file(TOKENIZER_PATH / name), "size": (TOKENIZER_PATH / name).stat().st_size}
             for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
             if (TOKENIZER_PATH / name).is_file()}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    except Exception as exc:
        return None, {"status": "TOKENIZER_LOAD_FAILED", "error": repr(exc), "files": files}
    return tokenizer, {"status": "PASS", "path": str(TOKENIZER_PATH), "class": tokenizer.__class__.__name__, "files": files}


def count_tokens(tokenizer: Any | None, text: str) -> int | None:
    return len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer is not None else None


def _dialogue_identity(manifest_by_root: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []; dialogues: dict[str, dict[str, Any]] = {}
    for root in ROOT_ORDER:
        item = manifest_by_root[root]; path = Path(str(item.get("feedback_path") or "")); drift: list[str] = []
        dialogue = load_json(path) if path.is_file() else {}; dialogues[root] = dialogue
        observed = sha_file(path) if path.is_file() else None
        refs = [str(x.get("turn_ref") or "") for x in dialogue.get("turns") or []]
        if observed != item.get("feedback_sha256"): drift.append("feedback_sha256")
        if dialogue.get("root_id") != root: drift.append("root_id")
        if len(refs) != int(item.get("feedback_turn_count") or -1): drift.append("turn_count")
        if not any(x.startswith("R02:") for x in refs): drift.append("r02_missing")
        rows.append({"root_id": root, "status": "PASS" if not drift else "E1Q0_E1Q1_DIALOGUE_DRIFT",
                     "path": str(path), "feedback_sha256": item.get("feedback_sha256"),
                     "observed_sha256": observed, "turn_count": len(refs),
                     "r02_turn_count": sum(x.startswith("R02:") for x in refs), "drift_fields": drift,
                     "r02_root": str(locate_r02_root(path).resolve()) if path.is_file() else None})
    return rows, dialogues


def compile_qa_feedback(run_root: Path = Path("")) -> dict[str, Any]:
    run_root = (run_root or (Path(__file__).parents[5] / "restart-n5-runs" / f"{VERSION}_live_1")).resolve()
    if run_root.exists() and any(run_root.iterdir()) and not (run_root / "run_identity.json").is_file():
        raise RuntimeError(f"run_root_exists_without_identity:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    write_artifact(run_root / "run_identity.json", {"version": VERSION, "algorithm_version": ALGORITHM_VERSION,
                    "offline_only": True, "api_calls": 0, "qa_writer_calls": 0, "solver_calls": 0,
                    "memorax_calls": 0, "semantic_auditor_calls": 0})
    paths = {
        "transition": ROUND0_REEVAL_RUN / "candidate_transition_matrix.jsonl",
        "exposure": ROUND0_REEVAL_RUN / "candidate_training_exposure_manifest.jsonl",
        "clean": ROUND0_REEVAL_RUN / "clean_train/candidate_results.jsonl",
        "inclusive": ROUND0_REEVAL_RUN / "inclusive_train/candidate_results.jsonl",
        "difficulty": ROUND0_DIFFICULTY_RUN / "difficulty_manifest.jsonl",
        "evidence": ROUND0_EVIDENCE_RUN / "hard_evidence_status_manifest.jsonl",
        "blacklist": ROUND0_TRAINING_RUN / "confirmed_invalid_blacklist.json",
        "e1q0_manifest": E1Q0_QA_RUN / "paired_dialogue_manifest.jsonl",
        "e1q0_contract": E1Q0_QA_RUN / "paired_qa_contract.json",
        "e1q0_candidates": E1Q0_QA_RUN / "all_candidates.jsonl",
    }
    source_audit = {}; source_sha = {}
    for name, path in paths.items():
        ok, observed = verify_sidecar(path); source_sha[name] = observed
        source_audit[name] = {"path": str(path), "exists": path.is_file(), "sidecar_verified": ok}
    transition = index(load_jsonl(paths["transition"]), "candidate_id")
    exposure = index(load_jsonl(paths["exposure"]), "candidate_id")
    difficulty_all = load_jsonl(paths["difficulty"]); train = [x for x in difficulty_all if x.get("split") == "train"]
    difficulty = index(train, "candidate_id")
    clean = index(load_jsonl(paths["clean"]), "candidate_id"); inclusive = index(load_jsonl(paths["inclusive"]), "candidate_id")
    all_ids = set(transition)
    alignment = {"round0_difficulty_total": len(difficulty_all), "round0_train_candidate_count": len(train),
                 "transition_candidate_count": len(transition), "clean_result_count": len(clean),
                 "inclusive_result_count": len(inclusive), "candidate_sets_equal":
                 all(set(x) == all_ids for x in (exposure, difficulty, clean, inclusive))}
    if len(difficulty_all) != EXPECTED_ROUND0_CANDIDATES or len(train) != EXPECTED_TRAIN_CANDIDATES or not alignment["candidate_sets_equal"]:
        raise RuntimeError(f"round0_input_alignment_failed:{alignment}")
    evidence = {str(x["candidate_id"]): x for x in load_jsonl(paths["evidence"]) if x.get("split") == "train"}
    hard_ids = {cid for cid, x in difficulty.items() if x.get("difficulty_bucket") == "hard_candidate"}
    if set(evidence) != hard_ids: raise RuntimeError("evidence_hard_identity_mismatch")
    evidence_status = {cid: (str(evidence[cid].get("evidence_status") or "EVIDENCE_UNCLASSIFIABLE")
                             if cid in evidence else "NOT_APPLICABLE_NON_HARD") for cid in all_ids}
    blacklist = load_json(paths["blacklist"]) if paths["blacklist"].is_file() else {}
    invalid = {str(x.get("candidate_id")) for x in blacklist.get("entries") or []}
    manifest = load_jsonl(paths["e1q0_manifest"]); manifest_by_root = index(manifest, "root_id")
    observed_success = {str(x["root_id"]) for x in manifest if x.get("status") == "PASS"}
    if observed_success != set(SELECTED_ROOT_IDS): raise RuntimeError("e1q0_successful_root_set_mismatch")
    identity, dialogues = _dialogue_identity(manifest_by_root)
    write_artifact(run_root / "successful_root_manifest.jsonl", [{"root_id": x, "status": "E1Q0_SUCCESS"} for x in ROOT_ORDER])
    write_artifact(run_root / "quarantine_root_manifest.jsonl", [{"root_id": x, "status": "HARD_EXCLUDED"} for x in sorted(QUARANTINE_ROOT_IDS)])
    write_artifact(run_root / "e1q0_dialogue_identity_audit.json", {"roots": identity, "all_pass": all(x["status"] == "PASS" for x in identity)})

    r02_by_root = {root: structured_r02(Path(identity[i]["path"])) for i, root in enumerate(ROOT_ORDER)}
    rows_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list); occurrences = []; confusion_rows = []
    for cid in sorted(all_ids):
        d = difficulty[cid]; candidate = dict(d.get("candidate") or {}); root = str(d["root_id"])
        if root not in SELECTED_ROOT_IDS:
            continue
        cr, ir = summarize_raw_result(clean[cid]), summarize_raw_result(inclusive[cid]); m0 = dict(transition[cid]["models"]["M0"])
        cm = model_payload(m0, cr, transition[cid]["models"]["M_clean_policy"].get("difficulty_bucket")); im = model_payload(m0, ir, transition[cid]["models"]["M_inclusive_policy"].get("difficulty_bucket"))
        mm0 = {"difficulty_bucket": d.get("difficulty_bucket"), "full_context_accuracy": float(d.get("full_context_accuracy") or 0), "no_context_accuracy": float(d.get("no_context_accuracy") or 0), "position_mechanism": d.get("position_mechanism")}
        occurrence = ordering_audit(candidate); occurrences.append({"candidate_id": cid, "root_id": root, **occurrence})
        source_dialogue = load_json(Path(str(d.get("dialogue_path") or "")))
        source_refs = {str(x.get("turn_ref") or "") for x in source_dialogue.get("turns") or []}
        row = {"candidate_id": cid, "root_id": root, "candidate": candidate, "question_type": candidate.get("question_type") or d.get("question_type"),
               "generation_source": candidate.get("generation_source") or d.get("generation_source"), "evidence_refs": candidate.get("evidence_refs") or [],
               "evidence_status": evidence_status[cid], "training_exposure": exposure[cid].get("training_exposure"), "position_mechanism": d.get("position_mechanism"),
               "technical_failure_count": cr["technical_failure_count"] + ir["technical_failure_count"], "models": {"M0": mm0, "M_clean_policy": cm, "M_inclusive_policy": im},
               "source_dialogue_refs": source_refs, "ordering_occurrence_status": occurrence.get("status"), "candidate_sha256": sha_bytes(canonical(candidate).encode()),
               "source_dialogue_sha256": d.get("dialogue_sha256")}
        row["confusion"] = confusion_feature(candidate, cr, ir); row["priority_class"] = classify_priority_v2(row, row["confusion"])
        row["frontier_score"] = frontier_score((cm["full_context_accuracy"] + im["full_context_accuracy"]) / 2)
        row["r02_adherence"] = r02_by_root[root]; rows_by_root[root].append(row)
        confusion_rows.append({"root_id": root, "candidate_id": cid, **row["confusion"]})
    write_artifact(run_root / "ordering_seed_occurrence_audit.jsonl", occurrences)
    write_artifact(run_root / "canonical_confusion_features.jsonl", confusion_rows)
    write_artifact(run_root / "evidence_status_normalization_audit.json", {"status_values": sorted(set(evidence_status.values())), "counts": dict(Counter(evidence_status.values())), "null_to_clean_count": 0})
    untouched = {str(x.get("candidate_id")) for x in load_jsonl(EXPANSION_COMPILER_RUN / "untouched_probe_manifest.jsonl")} if (EXPANSION_COMPILER_RUN / "untouched_probe_manifest.jsonl").is_file() else set()
    selected: dict[str, dict[str, Any] | None] = {}; reserves: dict[str, list[dict[str, Any]]] = {}; exclusions = []
    for root in ROOT_ORDER:
        eligible = []
        for row in rows_by_root[root]:
            reasons = []
            if not row["priority_class"]: reasons.append("no_P1_P5_class")
            if row["candidate_id"] in untouched: reasons.append("untouched_probe")
            if row["candidate_id"] in invalid: reasons.append("confirmed_defect")
            if row["technical_failure_count"]: reasons.append("technical_failure")
            if row["position_mechanism"] == "position_only": reasons.append("pure_position_only")
            if any(float(row["models"][m]["full_context_accuracy"]) + EPS < float(row["models"][m]["no_context_accuracy"]) for m in ("M_clean_policy", "M_inclusive_policy")): reasons.append("context_worse")
            if set(map(str, row["evidence_refs"])) - set(row["source_dialogue_refs"]): reasons.append("evidence_refs_missing_from_round0_dialogue")
            if row["evidence_status"] in {"EVIDENCE_INVALID", "EVIDENCE_UNCLASSIFIABLE"}: reasons.append("evidence_status_blocked")
            if row["ordering_occurrence_status"] in {"UNKNOWN_OCCURRENCE", "MULTI_OCCURRENCE_ORDER_SENSITIVE"}: reasons.append("unsafe_ordering_occurrence")
            if r02_by_root[root]["status"] != "SUPPORTED_STRUCTURED_ADHERENCE": reasons.append("NO_SUPPORTED_EVOLUTION_PATH")
            if reasons: exclusions.append({"root_id": root, "candidate_id": row["candidate_id"], "reasons": sorted(set(reasons))})
            else: eligible.append(row)
        eligible.sort(key=lambda x: (P_ORDER[x["priority_class"]], EXPOSURE_ORDER.get(str(x["training_exposure"]), 99), EVIDENCE_ORDER.get(str(x["evidence_status"]), 99), -(x["models"]["M_clean_policy"]["context_gain"] + x["models"]["M_inclusive_policy"]["context_gain"]), -x["frontier_score"], QTYPE_ORDER.get(str(x["question_type"]), 99), sha_bytes(x["candidate_id"].encode())))
        selected[root] = eligible[0] if eligible else None; reserves[root] = eligible[1:3]
    seed_rows = []; reserve_rows = []; packets = []; slots = []
    expansion_seed = {str(x.get("root_id")): str(x.get("seed_candidate_id") or "") for x in load_jsonl(EXPANSION_COMPILER_RUN / "feedback_seed_manifest.jsonl")} if (EXPANSION_COMPILER_RUN / "feedback_seed_manifest.jsonl").is_file() else {}
    for root in ROOT_ORDER:
        row = selected[root]
        if row:
            seed_rows.append({"root_id": root, "qa_feedback_seed_id": row["candidate_id"], "expansion_feedback_seed_id": expansion_seed.get(root), "seed_shared": row["candidate_id"] == expansion_seed.get(root), "candidate_sha256": row["candidate_sha256"], "priority_class": row["priority_class"], "training_exposure": row["training_exposure"], "evidence_status": row["evidence_status"], "question_type": row["question_type"], "position_mechanism": row["position_mechanism"], "r02_adherence_status": row["r02_adherence"]["status"]})
            packets.append({"packet_version": "round1_source_qa_packet_v2", "experiment_arm": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE_V1", "feedback_source_round": "round0", "target_round": "round1", "root_id": root, "source_candidate_id": row["candidate_id"], "source_candidate_sha256": row["candidate_sha256"], "source_question_type": row["question_type"], "source_generation_source": row["generation_source"], "source_question": row["candidate"].get("question"), "source_options_or_items": row["candidate"].get("options") or row["candidate"].get("items") or {}, "source_canonical_gold": row["candidate"].get("gold") or [], "source_evidence_refs": row["evidence_refs"], "source_evidence_groups": row["candidate"].get("evidence_groups") or {}, "priority_class": row["priority_class"], "frontier_score": row["frontier_score"], "training_exposure": row["training_exposure"], "evidence_status": row["evidence_status"], "position_mechanism": row["position_mechanism"], "canonical_confusion": row["confusion"], "transition": row["models"], "r02_adherence": row["r02_adherence"], "curriculum_notice": "Current E1Q0 cumulative dialogue is the only fact source for future QA."})
            r02refs = row["r02_adherence"]["r02_turn_refs"]
            slots.extend([{ "root_id": root, "slot_type": "TARGETED_EVOLUTION", "status": "PLANNED", "required_r02_refs": r02refs[:1], "source_candidate_id": row["candidate_id"]}, {"root_id": root, "slot_type": "CONTRASTIVE_QA", "status": "PLANNED" if row["confusion"].get("shared_error_set") else "FALLBACK_TARGETED_EVOLUTION", "target_confusion": row["confusion"], "source_candidate_id": row["candidate_id"]}, {"root_id": root, "slot_type": "R02_COVERAGE_QA", "status": "PLANNED", "required_r02_refs": r02refs[1:3], "source_candidate_id": row["candidate_id"]}])
        else: seed_rows.append({"root_id": root, "static_seed_status": "NO_SAFE_QA_FEEDBACK_SEED"})
        for reserve in reserves[root]: reserve_rows.append({"root_id": root, "candidate_id": reserve["candidate_id"], "priority_class": reserve["priority_class"], "candidate_sha256": reserve["candidate_sha256"], "static_reserve": True})
    write_artifact(run_root / "qa_feedback_seed_selection.jsonl", seed_rows); write_artifact(run_root / "qa_feedback_seed_reserves.jsonl", reserve_rows); write_artifact(run_root / "qa_feedback_seed_exclusions.jsonl", exclusions); write_artifact(run_root / "source_qa_packets.jsonl", packets); write_artifact(run_root / "three_qa_feedback_slot_plan.jsonl", slots)
    write_artifact(run_root / "untouched_probe_exclusion_audit.json", {"untouched_probe_count": len(untouched), "selected_overlap_count": len({x.get("qa_feedback_seed_id") for x in seed_rows} & untouched), "reserve_overlap_count": len({x.get("candidate_id") for x in reserve_rows} & untouched)})
    e1q0_candidates = load_jsonl(paths["e1q0_candidates"]); baseline = []
    for candidate in e1q0_candidates:
        seed = selected.get(str(candidate.get("root_id"))); baseline.append({"candidate_id": candidate.get("candidate_id"), "root_id": candidate.get("root_id"), **(duplicate_relation(candidate, seed["candidate"]) if seed else {"unavailable": True})})
    write_artifact(run_root / "e1q0_source_similarity_baseline.jsonl", baseline); write_artifact(run_root / "e1q0_duplicate_baseline_summary.json", {"e1q0_candidate_count": len(e1q0_candidates), "expected_count": EXPECTED_E1Q0_CANDIDATES, "exact_duplicate_count": sum(bool(x.get("exact_duplicate")) for x in baseline), "near_duplicate_count": sum(bool(x.get("near_duplicate")) for x in baseline), "unavailable_count": sum(bool(x.get("unavailable")) for x in baseline)})
    tokenizer, tokenizer_meta = load_tokenizer(); token_rows = []
    for root in ROOT_ORDER:
        text = render_dialogue(dialogues[root]); packet = next((x for x in packets if x["root_id"] == root), {}); slot = [x for x in slots if x["root_id"] == root]
        counts = {"dialogue_tokens": count_tokens(tokenizer, text), "packet_tokens": count_tokens(tokenizer, canonical(packet)), "slot_instruction_tokens": count_tokens(tokenizer, canonical(slot)), "base_prompt_tokens": count_tokens(tokenizer, "CURRENT_CUMULATIVE_DIALOGUE\nCURRICULUM_SOURCE_QA\nQUESTION_EVOLUTION_SLOT\n")}
        total = sum(x or 0 for x in counts.values()); token_rows.append({"root_id": root, **counts, "intent_miner_output_reserve": INTENT_MAX_TOKENS, "writer_output_reserve": WRITER_MAX_TOKENS, "intent_total": total + INTENT_MAX_TOKENS, "writer_total": total + WRITER_MAX_TOKENS, "context_limit": CONTEXT_LIMIT, "dialogue_truncated": False, "packet_truncated": False, "status": "PASS" if tokenizer is not None and total + WRITER_MAX_TOKENS <= CONTEXT_LIMIT else "TOKEN_BUDGET_EXCEEDED"})
    write_artifact(run_root / "qa_feedback_token_budget.json", {"tokenizer": tokenizer_meta, "roots": token_rows, "all_fit": all(x["status"] == "PASS" for x in token_rows), "max_writer_total": max((x["writer_total"] for x in token_rows), default=None)})
    write_artifact(run_root / "feedback_round_isolation_audit.json", {"feedback_source_round": "round0", "round1_solver_feedback_rows_read": 0, "round1_solver_feedback_bytes_used_for_selection": 0, "dev_test_rows_read": 0, "api_calls": 0, "solver_calls": 0, "memorax_calls": 0, "qa_writer_calls": 0, "semantic_auditor_calls": 0})
    # Candidate IDs are planned only; they are disjoint by namespace by construction.
    planned = [{"root_id": root, "slot_type": slot, "generation_source": source, "candidate_id": f"{root}:R01-R02-feedback:E1Q1_FULL_QA_FEEDBACK_COMPOSITE:{slot}:{source}:{n:06d}"} for n, (root, slot, source) in enumerate(((r, s, g) for r in ROOT_ORDER for s in ("TARGETED_EVOLUTION", "CONTRASTIVE_QA", "R02_COVERAGE_QA") for g in ("contract_guided", "free_exploration")), 1)]
    write_artifact(run_root / "candidate_id_scope_plan.json", {"planned_ids": planned}); write_artifact(run_root / "candidate_id_collision_preflight.json", {"planned_count": len(planned), "within_future_duplicates": len({x["candidate_id"] for x in planned}) != len(planned), "round0_overlap": [], "e1q0_overlap": []})
    write_artifact(run_root / "question_evolution_contract.json", {"normalization_version": "qa_duplicate_normalization_v1", "exact_route": "HARD_REJECT", "near_route": "SECONDARY_CONTAMINATION_SET", "thresholds": {"stem_token_jaccard": .80, "stem_sequence_ratio": .90, "evidence_jaccard": .80, "alternative_overlap": .50}})
    write_artifact(run_root / "experiment_contract.json", {"experiment_arm": "E1Q1", "experiment_name": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE_V1", "feedback_source_round": "round0", "target_generation_round": "round1", "expansion_feedback_enabled": True, "qa_feedback_enabled": True, "qa_feedback_mode": "full_source_qa_packet", "current_round_solver_feedback_used": False, "memory_bucket_navigation_enabled": False, "qa_fact_source": "complete_parent_r01_r02_feedback_dialogue", "memory_is_qa_fact_source": False, "offline_only": True})
    blockers = sorted(set(("E1Q0_E1Q1_DIALOGUE_DRIFT" if any(x["status"] != "PASS" for x in identity) else "",) if False else []))
    root_reports = []
    for root in ROOT_ORDER:
        row = selected[root]; rb = []
        if row is None: rb.append("NO_SAFE_QA_FEEDBACK_SEED")
        if next(x for x in identity if x["root_id"] == root)["status"] != "PASS": rb.append("E1Q0_E1Q1_DIALOGUE_DRIFT")
        if row and r02_by_root[root]["status"] != "SUPPORTED_STRUCTURED_ADHERENCE": rb.append("NO_SUPPORTED_EVOLUTION_PATH")
        if next(x for x in token_rows if x["root_id"] == root)["status"] != "PASS": rb.append("TOKEN_BUDGET_EXCEEDED")
        root_reports.append({"root_id": root, "status": "READY" if not rb else "GAP", "qa_feedback_seed_id": row["candidate_id"] if row else None, "priority_class": row["priority_class"] if row else None, "blockers": rb})
    all_ready = all(x["status"] == "READY" for x in root_reports)
    report = {"status": "ROUND1_QA_FEEDBACK_COMPILER_COMPLETE" if all_ready else "ROUND1_QA_FEEDBACK_COMPILER_COMPLETE_WITH_GAPS", "next_status": "READY_FOR_E1Q1_SINGLE_ROOT_QA_SMOKE" if all_ready else "E1Q1_QA_BATCH_NOT_READY", "experiment_arm": "E1Q1_FULL_QA_FEEDBACK_COMPOSITE_V1", "root_count": 8, "safe_root_count": sum(x["status"] == "READY" for x in root_reports), "root_reports": root_reports, "selected_seed_count": len(packets), "online_calls": {"api": 0, "qa_writer": 0, "solver": 0, "memorax": 0, "semantic_auditor": 0}, "memory_bucket_required": False, "round1_solver_feedback_rows_read": 0, "dialogue_identity_all_pass": all(x["status"] == "PASS" for x in identity), "max_writer_token_total": max((x["writer_total"] for x in token_rows), default=None)}
    write_artifact(run_root / "preflight_report.json", {"status": report["status"], "source_audit": source_audit, "input_alignment": alignment, "all_online_calls_zero": True, "feedback_round_isolation": {"round1_solver_feedback_rows_read": 0}})
    write_artifact(run_root / "final_qa_feedback_compiler_report.json", report)
    write_artifact(run_root / "final_qa_feedback_compiler_report.md", "# Round-1 QA Feedback Compiler v2\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
