"""Runtime adapter for the isolated feedback R02 smoke.

The adapter is intentionally separate from the historical continuation
runner. It supplies the two missing contracts: a frozen abstract feedback card
is appended only to the Planner prompt, and the R02 Add intent uses the frozen
`:R02-feedback` session. Memorax receives only the generated R02 dialogue
delta; feedback provenance stays in local reports and ledgers. The default CLI
mode is offline; live execution is explicit and guarded by the completed
offline preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from . import VERSION
from .preflight import (
    CONFIG_PATH,
    DEFAULT_OUTPUT,
    RECON_ROOT,
    ROOT_ID,
    TOKENIZER_PATH,
    _load,
    _sha,
)


FEEDBACK_PROMPT_VERSION = VERSION + "_planner_card_v1"
MEMORY_WRITE_CONTRACT_VERSION = VERSION + "_dialogue_only_memory_v2"
FEEDBACK_SESSION_SUFFIX = ":memorax:R02-feedback"
ALLOWED_CARD_FIELDS = (
    "feedback_version", "target",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def write_artifact_sidecars(root: Path) -> dict[str, Any]:
    """Hash every completed run artifact without changing its contents."""
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not path.name.endswith(".sha256")
        and path.name != "artifact_sha256_manifest.json"
    )
    entries = []
    for path in files:
        digest = _sha(path)
        path.with_name(path.name + ".sha256").write_text(
            digest + "\n", encoding="utf-8")
        entries.append({"path": str(path.relative_to(root)), "sha256": digest})
    manifest = {
        "manifest_version": VERSION + "_artifact_sha256_v1",
        "root": str(root.resolve()),
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path = root / "artifact_sha256_manifest.json"
    manifest_path.write_text(_json(manifest) + "\n", encoding="utf-8")
    manifest_path.with_name(manifest_path.name + ".sha256").write_text(
        _sha(manifest_path) + "\n", encoding="utf-8")
    return manifest


def _card_payload(card: Mapping[str, Any]) -> dict[str, Any]:
    target = card.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("feedback_card_target_missing")
    allowed_target = {
        "anchor_refs", "constraints", "desired_event_operator",
        "frontier_status", "gap_type", "source_candidate_sha256",
    }
    unexpected = sorted(set(target) - allowed_target)
    if unexpected:
        raise ValueError("feedback_card_unexpected_fields:" + ",".join(unexpected))
    return {
        "feedback_version": str(card.get("feedback_version") or ""),
        "target": {key: target[key] for key in sorted(target)},
    }


def inject_feedback_card(
    planner_prompt: Mapping[str, Any], card: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a new prompt envelope containing only the abstract card."""
    base = str(planner_prompt.get("prompt") or "")
    if not base:
        raise ValueError("planner_prompt_empty")
    payload = _card_payload(card)
    card_text = _json(payload)
    text = (
        base
        + "\n\nFROZEN ROUND-1 FEEDBACK CARD (QUALITY TARGET ONLY)\n"
        + "Use this abstract target to shape a genuinely new continuation. "
          "It is not factual evidence; facts remain grounded in the canonical "
          "Parent + R01 input and the retrieved Memory. Do not copy this card "
          "into dialogue, do not reveal its control metadata, and never reveal "
          "Gold, options, answer labels, or Solver output.\n"
        + card_text
        + "\nEND FROZEN ROUND-1 FEEDBACK CARD"
    )
    result = dict(planner_prompt)
    result["prompt"] = text
    result["prompt_version"] = FEEDBACK_PROMPT_VERSION
    result["feedback_card_exposed"] = True
    result["feedback_card_sha256"] = hashlib.sha256(
        _json(payload).encode("utf-8")).hexdigest()
    return result


def override_feedback_session(
    intent: Mapping[str, Any], *, root_id: str,
) -> dict[str, Any]:
    """Override the R02 session and strip feedback-only payload metadata."""
    if int(intent.get("round_index") or 0) != 2:
        raise ValueError("feedback_session_override_requires_round_2")
    expected_root = str(intent.get("root_id") or "")
    if expected_root != root_id:
        raise ValueError("feedback_session_root_mismatch")
    payload = intent.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("feedback_intent_payload_missing")
    payload_copy = dict(payload)
    session_id = f"{root_id}{FEEDBACK_SESSION_SUFFIX}"
    payload_copy["session_id"] = session_id
    metadata = dict(payload_copy.get("metadata") or {})
    metadata["session_label"] = "R02-feedback"
    # Feedback is generation-time control input, not persisted Memory content
    # or Memorax metadata. Keep provenance in local reports/ledgers instead.
    for key in ("feedback_card_exposed", "feedback_card_sha256",
                "feedback_version", "feedback_card"):
        metadata.pop(key, None)
    payload_copy["metadata"] = metadata
    for key in ("feedback_card_exposed", "feedback_card_sha256",
                "feedback_version", "feedback_card"):
        payload_copy.pop(key, None)
    assert_dialogue_only_memorax_payload(payload_copy)
    canonical = json.dumps(payload_copy, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    payload_sha = hashlib.sha256(canonical).hexdigest()
    result = dict(intent)
    result["session_id"] = session_id
    result["payload"] = payload_copy
    result["payload_sha256"] = payload_sha
    result["request_id"] = (
        f"memorax-memory-event-{root_id}-r02-feedback-{payload_sha[:12]}")
    result["feedback_card_exposed"] = True
    return result


def assert_dialogue_only_memorax_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if feedback control data reaches the Memory Add payload."""
    forbidden = {"feedback_card_exposed", "feedback_card_sha256",
                 "feedback_version", "feedback_card"}
    top_hits = sorted(forbidden.intersection(payload))
    metadata = payload.get("metadata")
    metadata_hits = sorted(forbidden.intersection(metadata or {})) \
        if isinstance(metadata, Mapping) else []
    if top_hits or metadata_hits:
        raise ValueError(
            "feedback_control_in_memorax_payload:"
            + ",".join(top_hits + metadata_hits))
    for field in ("messages", "window_context"):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"memorax_{field}_missing_or_invalid")


def patch_existing_runtime(card: Mapping[str, Any]) -> dict[str, Any]:
    """Install narrow monkey patches and return an auditable patch record."""
    from ..simple_append_memorax_v1_memory_event_continuation_probe_v1 import workflow

    original_prompt = workflow.build_planner_prompt
    original_intent = workflow.build_add_intent

    def build_prompt_with_feedback(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return inject_feedback_card(original_prompt(*args, **kwargs), card)

    def build_intent_with_feedback(*args: Any, **kwargs: Any) -> dict[str, Any]:
        intent = original_intent(*args, **kwargs)
        if int(kwargs.get("round_index", 0) or 0) == 2:
            return override_feedback_session(
                intent, root_id=str(kwargs.get("root_id") or ""))
        return intent

    workflow.build_planner_prompt = build_prompt_with_feedback
    workflow.build_add_intent = build_intent_with_feedback
    return {
        "adapter_version": MEMORY_WRITE_CONTRACT_VERSION,
        "planner_patch": "workflow.build_planner_prompt -> inject_feedback_card",
        "add_patch": "workflow.build_add_intent round_index=2 -> override_feedback_session",
        "feedback_prompt_version": FEEDBACK_PROMPT_VERSION,
        "feedback_session_suffix": FEEDBACK_SESSION_SUFFIX,
        "original_runtime_module": str(Path(workflow.__file__).resolve()),
        "original_runtime_sha256": _sha(Path(workflow.__file__).resolve()),
        "card_exposed_only_to_planner": True,
        "r02_messages_window_context_only": True,
        "memorax_payload_excludes_feedback_metadata": True,
    }


def stage_recovery_source(preflight_root: Path, *, output_root: Path) -> Path:
    """Adapt reconstruction artifacts to the old runner's read-only shape.

    The staged directory is kept beside the run output so any runtime report
    can continue to resolve its source paths after the process exits.
    """
    source = preflight_root / "feedback_reconstruction"
    root = output_root.parent / (output_root.name + ".r01_recovery_source")
    if root.exists():
        raise RuntimeError(f"staged_recovery_source_exists:{root}")
    memory = root / "memory"
    memory.mkdir(parents=True)
    for name in ("r01_child_add_intent.json", "r01_child_add_completion.json"):
        shutil.copyfile(source / name, memory / name.replace("r01_", ""))
    return root


def live_smoke(
    *, preflight_root: Path, output_root: Path,
    config_path: Path = CONFIG_PATH, tokenizer_path: Path = TOKENIZER_PATH,
    credentials_path: Path = Path("/root/.config/memorax/credentials.env"),
    allow_insecure_http: bool = True,
) -> dict[str, Any]:
    """Run one explicitly requested feedback R02 workflow.

    This function is intentionally not called by the module's default CLI.
    """
    report = _load(preflight_root / "preflight_report.json")
    if report.get("status") != "PASS":
        raise RuntimeError("r02_feedback_offline_preflight_required")
    plan = _load(preflight_root / "r02_feedback_generation_plan.json")
    if plan.get("runner_adapter_required") is not True:
        raise RuntimeError("feedback_runner_adapter_contract_missing")
    if output_root.exists():
        raise RuntimeError(f"live_output_already_exists:{output_root}")
    card_rows = [
        row for row in (json.loads(line) for line in
                        (Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
                              "h4_round1_feedback_compiler_v1_live_1/"
                              "root_feedback_cards.jsonl")
                        .read_text(encoding="utf-8").splitlines()) if line.strip())
        if row.get("root_id") == ROOT_ID
    ]
    if len(card_rows) != 1:
        raise RuntimeError("feedback_card_not_unique_for_live")
    card = card_rows[0]
    patch_record = patch_existing_runtime(card)
    from ..simple_append_memorax_v1_memory_event_continuation_probe_v1 import live
    # The R02 preflight directory is the launch authority; the already
    # completed R01 reconstruction is a separate immutable source.
    staged_source = stage_recovery_source(RECON_ROOT,
                                          output_root=output_root)
    source_manifest = _load(preflight_root / "source_manifest.json")
    input_path = Path(str((source_manifest.get("dialogue") or {}).get("path") or ""))
    if not input_path.is_file():
        raise RuntimeError("feedback_dialogue_source_manifest_missing")
    runtime_report = live.preflight(
        input_path=input_path,
        run_root=output_root, config_path=config_path,
        tokenizer_path=tokenizer_path, credentials_path=credentials_path,
        allow_insecure_http=allow_insecure_http,
        reuse_existing_namespace=True, source_run_root=staged_source,
        memorax_user_namespace="gxy-selfplay:round1-paired-feedback:2338d0e1c349",
        memory_only_fast=True, event_bucket_only=True, round_index=2,
    )
    if runtime_report.get("decision") != "pass":
        raise RuntimeError("existing_runtime_preflight_failed")
    runtime_report["feedback_adapter"] = patch_record
    runtime_report["feedback_card_sha256"] = hashlib.sha256(
        _json(_card_payload(card)).encode("utf-8")).hexdigest()
    result = live.execute_live(
        report=runtime_report,
        input_path=Path(runtime_report["input_path"]), run_root=output_root,
        config_path=config_path, tokenizer_path=tokenizer_path,
        credentials_path=credentials_path,
        allow_insecure_http=allow_insecure_http,
        reuse_existing_namespace=True,
        memorax_user_namespace="gxy-selfplay:round1-paired-feedback:2338d0e1c349",
        memory_only_fast=True, event_bucket_only=True, round_index=2,
    )
    result["artifact_sha256_manifest"] = write_artifact_sidecars(output_root)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-root", type=Path,
                        default=DEFAULT_OUTPUT.parent /
                        "h4_round1_feedback_r02_single_root_v1_live_3")
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_OUTPUT.parent /
                        "h4_round1_feedback_r02_single_root_v1_live_4")
    parser.add_argument("--live", action="store_true",
                        help="run the guarded live generation and Add")
    args = parser.parse_args(argv)
    if not args.live:
        report = _load(args.preflight_root / "preflight_report.json")
        print(json.dumps({"status": report["status"], "decision": report["decision"], "live_started": False}, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2
    result = live_smoke(preflight_root=args.preflight_root,
                        output_root=args.output_root)
    print(json.dumps({"status": result.get("status"), "run_root": str(args.output_root.resolve()), "live_started": True}, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
