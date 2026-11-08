"""Compile and load the compatible post-training feedback for the next round.

The compiler is deliberately offline and source locked.  It reads the selected
previous training/reevaluation artifacts and candidate pools, then writes a
new versioned delivery.  It never discovers a latest run, calls a model, or
changes an old feedback delivery.

V4 describes the current contracts: six-option multi-select (Gold cardinality
2..6) and six-item ordering by conversation appearance.  ``full``/``no`` and
exact agreement are historical diagnostics only; no ground or stability gate
is expressed here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


VERSION = "h4_v91_root_specific_history_feedback"
PROJECT = Path("/root/autodl-tmp/gxy-selfplay")
HERE = Path(__file__).resolve().parent
SOURCE_REGISTRY_PATH = HERE / "SOURCE_REGISTRY.json"
SOURCE_REGISTRY_SHA256 = "2e217792d368f8a54d772be274246926d6bc0e28afd22ac4d0685b12a1e1e39a"
SOURCE_REGISTRY = json.loads(SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
ROOT_SPECS = {row["root_id"]: row for row in SOURCE_REGISTRY["roots"]}
ROOTS = tuple(ROOT_SPECS)
EVALUATION_RUN = "h4_round0_trained_solver_train_reeval_v1_live_1"
QUESTION_RUN = "h4_unexpanded122_qa_generation_only_v1"
FORMAL_TRAINING_RUN = "h4_round0_b0_formal_training_v1_live_2"
SOURCE_RUN = PROJECT / "restart-n5-runs" / EVALUATION_RUN
QUESTION_ROOT = PROJECT / "restart-n5-runs" / QUESTION_RUN / "shards"
OUTPUT_DIR = HERE / "compiled_history_v91"

ROOT_SOURCE = {root: (Path(row["first_round_feedback"]["candidate_pool_path"]),
                      row["first_round_feedback"]["candidate_pool_sha256"])
               for root, row in ROOT_SPECS.items()}
EXPECTED_SHA256 = {
    "candidate_transition_matrix.jsonl": "853118ab1f484b70c440dd486bb39b5be48dabefb7e567a998bc7e49e04aafae",
    "candidate_training_exposure_manifest.jsonl": "1fcf461f27366792bb4140091af90736450233a93e4e8e0af7a04e3ca7205b0f",
    "model_identity_report.json": "add007d4e4f937f2ea1b7e124ccac3c53e5eda5b8bbed1beab3dd4b0855cadd0",
    "solver_contract.json": "bf951f3c35a9e6ce7a3009157eef20850966f2a249bfaff5cd2c80118be0a99e",
    "final_formal_training_report.json": "51681e65b86b9a0b0a8705487abdc1a4a382c5e4e90003a0e2de6e4493de3bdd",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, expected: str | None = None) -> dict[str, str]:
    actual = sha256_file(path)
    if expected is not None:
        require(actual == expected, f"source_hash_mismatch:{path.name}")
    return {"path": str(path.resolve()), "sha256": actual}


def _source_paths() -> dict[str, Path]:
    return {
        "candidate_transition_matrix": SOURCE_RUN / "candidate_transition_matrix.jsonl",
        "candidate_training_exposure": SOURCE_RUN / "candidate_training_exposure_manifest.jsonl",
        "model_identity": SOURCE_RUN / "model_identity_report.json",
        "run_identity": SOURCE_RUN / "run_identity.json",
        "solver_contract": SOURCE_RUN / "solver_contract.json",
        "formal_training_checkpoint": PROJECT / "restart-n5-runs" / FORMAL_TRAINING_RUN / "final_formal_training_report.json",
    }


def build_source_lock() -> dict[str, Any]:
    """Return the immutable source declaration used by compile_feedback.

    The exact SHA values for run_identity/model reports are read from the
    official files at compile time and frozen in the emitted lock.  A future
    rerun against changed bytes therefore fails unless a new V4 is explicitly
    created; no mtime or ``latest`` lookup is possible.
    """
    require(sha256_file(SOURCE_REGISTRY_PATH) == SOURCE_REGISTRY_SHA256, "source_registry_changed")
    paths = _source_paths()
    for key, path in paths.items():
        require(path.is_file(), f"missing_source:{key}")
    run_identity = _json(paths["run_identity"])
    require(run_identity.get("run_identity_version", "").startswith("h4_round0_trained_solver_train_reeval_v1"),
            "wrong_evaluation_run_identity")
    require(run_identity.get("scope") == "round0_train_only_parent_plus_r01", "wrong_evaluation_scope")
    formal_report = _json(paths["formal_training_checkpoint"])
    require(formal_report.get("status") == "FORMAL_TRAINING_COMPLETE", "formal_training_not_complete")
    require(formal_report.get("external_api_calls") == 0 and formal_report.get("solver_calls") == 0,
            "formal_training_external_calls_present")
    model_identity = _json(paths["model_identity"])
    require(set(model_identity.get("models", {})) >= {"M_clean_policy", "M_inclusive_policy"},
            "trained_model_identity_incomplete")
    artifacts = {
        key: _artifact(path, EXPECTED_SHA256.get(path.name))
        for key, path in paths.items()
    }
    for root, (path, expected) in ROOT_SOURCE.items():
        artifacts.setdefault("candidate_pools", {})[root] = _artifact(path, expected)
    transcripts = {}
    for root, row in ROOT_SPECS.items():
        frozen = row["first_round_feedback"]
        require(frozen["has_root_specific_feedback"] is True, "root_feedback_unavailable")
        require(frozen["evaluation_run_id"] == EVALUATION_RUN, "root_feedback_evaluation_mismatch")
        for field, expected_field, key in (
                ("candidate_transition_matrix_path", "candidate_transition_matrix_sha256", "candidate_transition_matrix"),
                ("candidate_training_exposure_path", "candidate_training_exposure_sha256", "candidate_training_exposure")):
            require(_artifact(Path(frozen[field]), frozen[expected_field]) == artifacts[key], "root_feedback_evaluation_source_mismatch")
        transcripts[root] = _artifact(Path(frozen["transcript_path"]), frozen["transcript_sha256"])
        require(_json(Path(frozen["transcript_path"]))["root_id"] == root, "transcript_root_mismatch")
    return {
        "source_registry": _artifact(SOURCE_REGISTRY_PATH, SOURCE_REGISTRY_SHA256),
        "historical_transcripts": transcripts,
        "version": VERSION + "_source_lock_v1",
        "status": "EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY",
        "automatic_latest_run_selection": False,
        "merge_current_generation_feedback": False,
        "question_run_id": QUESTION_RUN,
        "evaluation_run_id": EVALUATION_RUN,
        "formal_training_run_id": FORMAL_TRAINING_RUN,
        "model_identity_source": artifacts["model_identity"],
        "solver_checkpoint_source": artifacts["solver_contract"],
        "formal_training_checkpoint_source": artifacts["formal_training_checkpoint"],
        "evaluation_sources": {k: artifacts[k] for k in ("candidate_transition_matrix", "candidate_training_exposure")},
        "candidate_pools": artifacts["candidate_pools"],
        "root_order": list(ROOTS),
        "model_ids": ["M0", "M_clean_policy", "M_inclusive_policy"],
        "note": "All metrics are historical exact-Gold agreement diagnostics from the selected reevaluation. They are neither current partial scores nor semantic Gold certification.",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _question_payload(question: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(question.get(key)) for key in (
        "question_type", "question", "options", "items", "gold", "correct_option",
        "evidence_refs", "evidence_groups", "intent_id", "root_id") if key in question}


def _model_history(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "full_context": {
            "correct_count": model.get("full_context_correct_count"),
            "total_count": model.get("full_context_total_count"),
            "accuracy": model.get("full_context_accuracy"),
        },
        "no_context": {
            "correct_count": model.get("no_context_correct_count"),
            "total_count": model.get("no_context_total_count"),
            "accuracy": model.get("no_context_accuracy"),
        },
        "context_gain": model.get("context_gain"),
        "difficulty_bucket": model.get("difficulty_bucket"),
        "position_mechanism": model.get("position_mechanism"),
        "canonical_flip_rate": model.get("canonical_flip_rate"),
        "parse_failure_count": model.get("parse_failure_count", 0),
        "technical_failure_count": model.get("technical_failure_count", 0),
    }


def _candidate_history(row: Mapping[str, Any], question: Mapping[str, Any], exposure: Mapping[str, Any]) -> dict[str, Any]:
    require(row["candidate_id"] == exposure["candidate_id"] == question["candidate_id"], "exposure_candidate_mismatch")
    require(row["root_id"] == exposure["root_id"] == question["root_id"], "history_input_cross_root")
    return {
        "candidate_id": row["candidate_id"],
        "root_id": row["root_id"],
        "question_type": row.get("question_type"),
        "generation_source": row.get("generation_source"),
        "primary_transition": row.get("primary_transition"),
        "training_exposure": row.get("training_exposure"),
        "exposure_manifest_value": exposure.get("training_exposure"),
        "pair_id": row.get("pair_id"),
        "pair_role": row.get("pair_role"),
        "historical_question": _question_payload(question),
        "models": {name: _model_history(value) for name, value in row.get("models", {}).items()},
        "flags": copy.deepcopy(row.get("flags", {})),
        "claims_status": "historical_generated_question_and_Gold_are_unverified; metrics_are_exact_agreement_only",
    }


def _aggregate(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_exposure = Counter(str(row.get("training_exposure")) for row in candidates)
    by_type = Counter(str(row.get("question_type")) for row in candidates)
    model_aggregate: dict[str, Any] = {}
    for model in ("M0", "M_clean_policy", "M_inclusive_policy"):
        rows = [row for row in candidates if model in row["models"]]
        def sums(mode: str) -> dict[str, Any]:
            good = sum((row["models"][model][mode]["correct_count"] or 0) for row in rows)
            total = sum((row["models"][model][mode]["total_count"] or 0) for row in rows)
            return {"correct_count": good, "total_count": total, "accuracy": good / total if total else None}
        model_aggregate[model] = {
            "candidate_count": len(rows), "full_context": sums("full_context"), "no_context": sums("no_context"),
            "primary_transition_counts": dict(Counter(row.get("primary_transition") for row in rows)),
        }
    type_model: dict[str, Any] = {}
    for qtype in sorted(by_type):
        type_model[qtype] = {}
        subset = [row for row in candidates if row.get("question_type") == qtype]
        for model in ("M0", "M_clean_policy", "M_inclusive_policy"):
            rows = [row for row in subset if model in row["models"]]
            def one(mode: str) -> dict[str, Any]:
                correct = sum(row["models"][model][mode]["correct_count"] or 0 for row in rows)
                total = sum(row["models"][model][mode]["total_count"] or 0 for row in rows)
                return {"correct_count": correct, "total_count": total, "accuracy": correct / total if total else None}
            type_model[qtype][model] = {"candidate_count": len(rows), "full_context": one("full_context"), "no_context": one("no_context")}
    return {"candidate_count": len(candidates), "by_training_exposure": dict(by_exposure), "by_question_type": dict(by_type), "by_model": model_aggregate, "by_question_type_and_model": type_model}


def _guidance(aggregate: Mapping[str, Any]) -> list[str]:
    return [
        "Use prior-round statistics as diagnostic guidance only; do not copy historical questions, options or Gold.",
        "Current multi-select permits exactly six displayed options and Gold cardinality 2 through 6; preserve evidence mapping when options are randomly permuted.",
        "Current ordering uses six statements ordered by conversation appearance (conversation_order), with Gold remapped after every reproducible random display permutation.",
        "Do not add a ground/no-context gate, full-greater-than-no gate, exact-success gate, context-gain gate, or permutation-stability rejection. Complete the configured full evaluation before applying 0 < S_full < 1.",
        "Keep Qwen as the generator. Historical exact agreement is not semantic certification; unresolved wording, evidence, or Gold uncertainty must remain unresolved rather than being silently repaired by feedback.",
    ]


def _suffix(root: str, aggregate: Mapping[str, Any], candidates: list[Mapping[str, Any]], role: str, model: str) -> str:
    # The rows are compact enough to make the per-candidate training signal
    # inspectable; the candidate question itself remains historical context.
    rows = [{"candidate_id": row["candidate_id"], "question_type": row["question_type"],
             "training_exposure": row["training_exposure"], "primary_transition": row["primary_transition"],
             "models": {k: row["models"][k] for k in (model,) if k in row["models"]}}
            for row in candidates]
    payload = {"root_id": root, "role": role, "target_solver_model": model,
               "aggregate": aggregate, "candidate_history": rows,
               "guidance": _guidance(aggregate),
               "scope": "selected previous training round only; no current-generation feedback"}
    return "\n[PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]\n" + canonical(payload) + "\n[/PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]\n"


def compile_feedback(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Compile the registry-bound roots without network/model calls.

    Existing output is never overwritten.  A changed source hash requires an
    explicit new output directory/version.
    """
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "compiled_output_exists_no_overwrite")
    lock = build_source_lock()
    matrix_path = Path(lock["evaluation_sources"]["candidate_transition_matrix"]["path"])
    exposure_path = Path(lock["evaluation_sources"]["candidate_training_exposure"]["path"])
    matrix = _read_jsonl(matrix_path)
    exposure = {row["candidate_id"]: row for row in _read_jsonl(exposure_path)}
    by_root = defaultdict(list)
    for row in matrix:
        if row.get("root_id") in ROOTS:
            require(row["candidate_id"] in exposure, "missing_exposure_candidate")
            by_root[row["root_id"]].append(row)
    require(set(by_root) == set(ROOTS), "registry_root_scope_incomplete")
    records = []
    output_dir.mkdir(parents=True)
    (output_dir / "source_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for root in ROOTS:
        pool_path = Path(lock["candidate_pools"][root]["path"])
        pool_rows = _json(pool_path)
        require(all(row.get("root_id") == root for row in pool_rows), "candidate_pool_cross_root")
        pool = {row["candidate_id"]: row for row in pool_rows}
        require(len(pool) == len(pool_rows), "duplicate_pool_candidate_id")
        rows = sorted(by_root[root], key=lambda x: x["candidate_id"])
        require(set(pool) >= {row["candidate_id"] for row in rows}, f"candidate_pool_scope_mismatch:{root}")
        candidates = [_candidate_history(row, pool[row["candidate_id"]], exposure[row["candidate_id"]]) for row in rows]
        aggregate = _aggregate(candidates)
        root_dir = output_dir / root
        root_dir.mkdir()
        (root_dir / "candidate_history.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (root_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        suffixes = {}
        for role in ("writer", "miner", "planner"):
            suffixes[role] = {}
            for model in ("M0", "M_clean_policy", "M_inclusive_policy"):
                suffix = _suffix(root, aggregate, candidates, role, model)
                suffixes[role][model] = suffix
                (root_dir / f"{role}__{model}_feedback.txt").write_text(suffix, encoding="utf-8")
        records.append({"root_id": root, "candidate_count": len(candidates), "aggregate": aggregate,
                        "candidate_history_sha256": sha256_file(root_dir / "candidate_history.json"),
                        "aggregate_sha256": sha256_file(root_dir / "aggregate.json"),
                        "suffix_sha256": {role: {model: sha256_bytes(value.encode()) for model, value in values.items()}
                                         for role, values in suffixes.items()}})
    manifest = {"version": VERSION, "status": "COMPILED_OFFLINE_NO_EXTERNAL_CALLS", "external_calls": 0,
                "root_order": list(ROOTS), "source_lock": "source_lock.json", "records": records,
                "selection_contract": {"ground": "removed", "full": "complete configured evaluation; 0 < S_full < 1",
                                       "context_gain": "diagnostic_only", "permutation_stability": "diagnostic_only",
                                       "multi_select_gold_cardinality": [2, 3, 4, 5, 6],
                                       "ordering": "six_items_conversation_order"}}
    (output_dir / "COMPILED_FEEDBACK.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_feedback_context(compiled_dir: Path, root_id: str) -> dict[str, Any]:
    compiled_dir = Path(compiled_dir).resolve()
    manifest = _json(compiled_dir / "COMPILED_FEEDBACK.json")
    require(root_id in manifest["root_order"], "unknown_feedback_root")
    root_dir = compiled_dir / root_id
    aggregate = _json(root_dir / "aggregate.json")
    history = _json(root_dir / "candidate_history.json")
    return {"version": manifest["version"], "root_id": root_id, "aggregate": aggregate,
            "candidate_history": history, "source_lock": _json(compiled_dir / "source_lock.json")}


def build_feedback_suffix(compiled_dir: Path, root_id: str, role: str = "writer", target_solver_model: str = "M0") -> str:
    require(role in {"writer", "miner", "planner"}, "unknown_feedback_role")
    require(target_solver_model in {"M0", "M_clean_policy", "M_inclusive_policy"}, "unknown_solver_model")
    path = Path(compiled_dir).resolve() / root_id / f"{role}__{target_solver_model}_feedback.txt"
    suffix = path.read_text(encoding="utf-8")
    require(f'"target_solver_model":"{target_solver_model}"' in suffix, "feedback_model_binding_missing")
    return suffix


def append_feedback_to_request(payload: Any, compiled_dir: Path, root_id: str,
                               role: str = "writer", target_solver_model: str = "M0") -> tuple[Any, dict[str, Any]]:
    """Append the frozen suffix and return an auditable receipt.

    Supports a chat payload (``messages``), a single ``prompt``/``user_prompt``
    field, or a raw list of messages.  It never mutates the caller's object.
    """
    suffix = build_feedback_suffix(compiled_dir, root_id, role, target_solver_model)
    result = copy.deepcopy(payload)
    if isinstance(result, list):
        messages = result
    elif isinstance(result, dict) and isinstance(result.get("messages"), list):
        messages = result["messages"]
    else:
        messages = None
    if messages is not None:
        users = [message for message in messages if message.get("role") == "user"]
        require(users, "request_has_no_user_message")
        content = users[-1].get("content", "")
        require(isinstance(content, str), "user_content_must_be_text")
        users[-1]["content"] = content + suffix
        final_text = users[-1]["content"]
    elif isinstance(result, dict) and isinstance(result.get("prompt"), str):
        result["prompt"] += suffix
        final_text = result["prompt"]
    elif isinstance(result, dict) and isinstance(result.get("user_prompt"), str):
        result["user_prompt"] += suffix
        final_text = result["user_prompt"]
    else:
        raise ValueError("unsupported_request_shape")
    require(suffix in final_text, "feedback_suffix_not_in_request")
    receipt = {"version": VERSION, "root_id": root_id, "role": role,
               "target_solver_model": target_solver_model, "suffix_sha256": sha256_bytes(suffix.encode()),
               "request_user_text_sha256": sha256_bytes(final_text.encode()),
               "source_lock_sha256": sha256_file(Path(compiled_dir).resolve() / "source_lock.json"),
               "feedback_appended": True}
    return result, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--compile", action="store_true", help="compile the frozen registry-root delivery")
    args = parser.parse_args()
    if not args.compile:
        parser.error("pass --compile to create the offline delivery")
    print(json.dumps(compile_feedback(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
