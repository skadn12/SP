"""Generate fresh QA candidates without running Solver or difficulty admission.

This is an isolated continuation route for an existing expansion batch. It
reads only completed, hash-verified expansion outputs from --source-run and
writes all QA artifacts below a separate --output-run. The Challenger API is
used for intent/QA generation; no local Solver is loaded or called.

The reused generator still performs its minimum structural validation and
records the raw generation responses. Semantic validity, Solver difficulty,
position sensitivity, and training admission are deliberately not decided by
this route.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_qa
from ..h4_qa_multiroot_k3_t08_probe_v1.root_inventory import clean_named_view, load_expanded
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from .execute import _existing_ready, _result_path, _select_shards


PROBE_VERSION = "h4_qa_generation_only_v1"
SOURCE_EXECUTOR_VERSION = "h4_batch30_memory_event_k3_executor_v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_object(path: Path) -> dict[str, Any]:
    value = _load(path)
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "root"


def _root_dir(output_run: Path, shard: str, root_id: str) -> Path:
    return output_run / "shards" / _safe_name(shard) / "roots" / _safe_name(root_id)


def _load_feedback_context(entry: Mapping[str, Any], root_id: str) -> dict[str, Any]:
    """Read the exact frozen text additions, without changing QA policy."""
    path = Path(str(entry["path"]))
    content = path.read_bytes()
    if sha256_bytes(content) != entry["sha256"]:
        raise RuntimeError(f"feedback_context_hash_drift:{root_id}:{path}")
    context = json.loads(content)
    if (not isinstance(context, dict)
            or set(context) != {"intent_suffix", "slot_suffixes"}
            or not isinstance(context["intent_suffix"], str)
            or not isinstance(context["slot_suffixes"], list)
            or len(context["slot_suffixes"]) != 3
            or not all(isinstance(value, str) for value in context["slot_suffixes"])):
        raise ValueError(f"feedback_context_schema_invalid:{root_id}:{path}")
    return context


def _freeze_feedback_context(path: Path, root_ids: list[str]) -> dict[str, Any]:
    """Freeze selected roots from {roots: {root_id: {path, sha256}}}."""
    path = path.resolve()
    content = path.read_bytes()
    source = json.loads(content)
    if (not isinstance(source, dict) or set(source) != {"roots"}
            or not isinstance(source["roots"], dict)):
        raise ValueError(f"feedback_context_manifest_schema_invalid:{path}")
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("feedback_context_duplicate_root_scope")
    selected: dict[str, Any] = {}
    for root_id in root_ids:
        entry = source["roots"].get(root_id)
        if entry is None:
            raise ValueError(f"feedback_context_root_missing:{root_id}")
        if (not isinstance(entry, dict) or set(entry) != {"path", "sha256"}
                or not isinstance(entry["path"], str) or not entry["path"]
                or not isinstance(entry["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None):
            raise ValueError(f"feedback_context_root_entry_invalid:{root_id}")
        context_path = Path(entry["path"])
        if not context_path.is_absolute():
            context_path = path.parent / context_path
        frozen = {"path": str(context_path.resolve()), "sha256": entry["sha256"]}
        _load_feedback_context(frozen, root_id)
        selected[root_id] = frozen
    return {
        "manifest_path": str(path),
        "manifest_sha256": sha256_bytes(content),
        "roots": selected,
    }


def _verify_feedback_context(manifest: Mapping[str, Any], requested: Path | None) -> dict[str, Any] | None:
    frozen = manifest.get("feedback_context")
    if frozen is None:
        if requested is not None:
            raise RuntimeError("feedback_context_requires_inventory_freeze")
        return None
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if sha256_bytes(canonical_json(payload).encode("utf-8")) != manifest.get("manifest_sha256"):
        raise RuntimeError("feedback_run_manifest_hash_drift")
    path = Path(str(frozen["manifest_path"]))
    if requested is not None and requested.resolve() != path:
        raise RuntimeError("feedback_context_manifest_path_mismatch")
    if sha256_file(path) != frozen["manifest_sha256"]:
        raise RuntimeError("feedback_context_manifest_hash_drift")
    actual = _freeze_feedback_context(path, [str(row["root_id"]) for row in manifest["roots"]])
    if actual != frozen:
        raise RuntimeError("feedback_context_frozen_mapping_drift")
    return frozen


def _source_inventory(plan: Mapping[str, Any], shard_names: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_shards = _select_shards(plan, shard_names)
    ready: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for shard in selected_shards:
        shard_name = str(shard.get("shard") or "")
        for root_plan in shard.get("roots") or []:
            root_id = str(root_plan.get("root_id") or "")
            result_path = _result_path(shard, root_id)
            result = _existing_ready(
                result_path,
                root_id=root_id,
                input_sha256=str(root_plan.get("input_sha256") or ""),
                parent_dialogue_sha256=str(root_plan.get("parent_dialogue_sha256") or "") or None,
            )
            if result is None:
                missing.append({
                    "shard": shard_name,
                    "root_id": root_id,
                    "expansion_result_path": str(result_path),
                    "reason": "no_hash_verified_expansion_ready_result",
                })
                continue
            source = Path(str(result["selected_source"])).resolve()
            ready.append({
                "shard": shard_name,
                "root_id": root_id,
                "source_path": str(source),
                "source_sha256": sha256_file(source),
                "expansion_result_path": str(result_path),
                "successful_attempt_root": str(result.get("successful_attempt_root") or ""),
                "selected_source_sha256": str(result.get("selected_source_sha256") or ""),
            })
    return ready, missing


def _manifest(source_run: Path, output_run: Path, plan: Mapping[str, Any],
              ready: list[dict[str, Any]], missing: list[dict[str, Any]],
              shard_names: list[str], feedback_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "probe_version": PROBE_VERSION,
        "source_executor_version": SOURCE_EXECUTOR_VERSION,
        "source_run": str(source_run),
        "source_batch_plan_sha256": sha256_file(source_run / "batch_plan.json"),
        "output_run": str(output_run),
        "selected_shards": shard_names or sorted({str(row["shard"]) for row in ready}),
        "roots": ready,
        "missing_roots": missing,
        "api_contract": {
            "challenger_generation_enabled": True,
            "solver_enabled": False,
            "solver_calls": 0,
        },
        "screening_contract": {
            "program_structural_validation": "enabled_inside_reused_generator",
            "semantic_model_admission": False,
            "difficulty_filter": False,
            "position_solver_audit": False,
            "training_ready": False,
        },
        "lineage_contract": {
            "source_expansions_read_only": True,
            "historical_qa_imported": False,
            "output_namespace_isolated": True,
        },
    }
    if feedback_context is not None:
        payload["feedback_context"] = dict(feedback_context)
    return {**payload, "manifest_sha256": sha256_bytes(canonical_json(payload).encode("utf-8"))}


def inventory_phase(args: argparse.Namespace) -> dict[str, Any]:
    source_run = args.source_run.resolve()
    output_run = args.output_run.resolve()
    plan = _load_object(source_run / "batch_plan.json")
    if plan.get("offline_only") is not True:
        raise ValueError("source_batch_plan_offline_contract_missing")
    ready, missing = _source_inventory(plan, list(args.shard))
    if missing and not args.allow_partial:
        raise RuntimeError(
            "incomplete_expansion; rerun with --allow-partial to inventory only ready roots"
        )
    if not ready:
        raise RuntimeError("no_hash_verified_expansion_ready_roots")
    feedback_path = getattr(args, "feedback_context_manifest", None)
    feedback_context = (
        _freeze_feedback_context(feedback_path, [str(row["root_id"]) for row in ready])
        if feedback_path is not None else None
    )
    manifest = _manifest(source_run, output_run, plan, ready, missing, list(args.shard), feedback_context)
    if output_run.exists():
        existing_path = output_run / "manifest.json"
        if not existing_path.is_file():
            raise RuntimeError(f"output_run_exists_without_manifest:{output_run}")
        existing = _load_object(existing_path)
        if existing.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise RuntimeError("output_manifest_mismatch; use a new output-run")
    else:
        output_run.mkdir(parents=True)
        atomic_json(output_run / "manifest.json", manifest)
    atomic_json(output_run / "source_inventory.json", {
        "probe_version": PROBE_VERSION,
        "source_run": str(source_run),
        "ready_count": len(ready),
        "missing_count": len(missing),
        "ready": ready,
        "missing": missing,
        "api_calls": 0,
        "solver_calls": 0,
    })
    atomic_text(output_run / "summary.md", (
        "# H4 QA generation-only run\n\n"
        "Inventory completed without Challenger API or Solver calls.\n"
    ))
    return {
        "status": "inventory_ready",
        "probe_version": PROBE_VERSION,
        "output_run": str(output_run),
        "ready_count": len(ready),
        "missing_count": len(missing),
        "solver_calls": 0,
    }


def _client(args: argparse.Namespace) -> OpenAICompatibleChallengerClient:
    config = load_challenger_config(args.api_config)
    return OpenAICompatibleChallengerClient(replace(
        config,
        temperature=0.1,
        enable_thinking=False,
        timeout_seconds=max(180, int(config.timeout_seconds)),
    ))


def _prepare_root(root_entry: Mapping[str, Any], output_run: Path) -> tuple[Path, dict[str, Any]]:
    root_id = str(root_entry["root_id"])
    root_dir = _root_dir(output_run, str(root_entry["shard"]), root_id)
    root_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(str(root_entry["source_path"]))
    if sha256_file(source_path) != str(root_entry["source_sha256"]):
        raise RuntimeError(f"source_hash_drift:{root_id}")
    source = load_expanded(source_path)
    if str(source.get("root_id") or "") != root_id:
        raise RuntimeError(f"root_id_drift:{root_id}")
    cleaned, mapping = clean_named_view(source)
    atomic_json(root_dir / "source_metadata.json", {
        "probe_version": PROBE_VERSION,
        "root_id": root_id,
        "source_path": str(source_path),
        "source_sha256": str(root_entry["source_sha256"]),
        "speaker_name_mapping": mapping,
        "turn_count": len(cleaned.get("turns") or []),
        "source_is_frozen": True,
    })
    atomic_json(root_dir / "expanded_dialogue_clean.json", cleaned)
    return root_dir, cleaned


def generate_phase(args: argparse.Namespace) -> dict[str, Any]:
    output_run = args.output_run.resolve()
    manifest = _load_object(output_run / "manifest.json")
    if manifest.get("probe_version") != PROBE_VERSION:
        raise ValueError("generation_manifest_version_mismatch")
    if not args.live:
        raise RuntimeError("generate_requires_live_to_call_qa_writer")
    feedback_context = _verify_feedback_context(
        manifest, getattr(args, "feedback_context_manifest", None),
    )
    client = _client(args)
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for entry in manifest.get("roots") or []:
        root_id = str(entry["root_id"])
        feedback_kwargs = (
            {"feedback_context": _load_feedback_context(feedback_context["roots"][root_id], root_id)}
            if feedback_context is not None else {}
        )
        root_dir, expanded = _prepare_root(entry, output_run)
        report_path = root_dir / "qa_generation_report.json"
        if report_path.is_file() and not args.force:
            report = _load_object(report_path)
            report["status"] = "reused_existing_generation"
            reports.append(report)
            continue
        ledger = RequestLedger(root_dir, str(manifest["manifest_sha256"]), "live")
        try:
            report = generate_root_qa(
                expanded=expanded,
                root_dir=root_dir,
                root_id=root_id,
                api_config=args.api_config,
                ledger=ledger,
                client=client,
                **feedback_kwargs,
            )
            report["probe_version"] = PROBE_VERSION
            report["solver_calls"] = 0
            report["difficulty_filter_applied"] = False
            report["semantic_admission_applied"] = False
            atomic_json(report_path, report)
            reports.append(report)
        except Exception as error:
            errors.append({"root_id": root_id, "error": f"{type(error).__name__}:{error}"})
    summary = {
        "probe_version": PROBE_VERSION,
        "source_run": manifest["source_run"],
        "output_run": str(output_run),
        "status": "completed" if not errors else "completed_with_errors",
        "root_count": len(reports),
        "candidate_count": sum(int(row.get("candidate_count") or 0) for row in reports),
        "valid_candidate_count": sum(int(row.get("valid_candidate_count") or 0) for row in reports),
        "challenger_generation_calls": sum(int(row.get("candidate_generation_calls") or 0) for row in reports),
        "solver_calls": 0,
        "difficulty_filter_applied": False,
        "semantic_admission_applied": False,
        "root_reports": reports,
        "errors": errors,
    }
    atomic_json(output_run / "generation_only_report.json", summary)
    atomic_text(output_run / "summary.md", "# H4 QA generation-only run\n\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("inventory", "generate", "all"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--feedback-context-manifest", type=Path,
                        help="Frozen roots-to-context JSON (path and sha256); adds prompt guidance only.")
    parser.add_argument("--shard", action="append", default=[])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase in {"inventory", "all"}:
        print(json.dumps(inventory_phase(args), ensure_ascii=False, indent=2), flush=True)
    if args.phase in {"generate", "all"}:
        print(json.dumps(generate_phase(args), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
