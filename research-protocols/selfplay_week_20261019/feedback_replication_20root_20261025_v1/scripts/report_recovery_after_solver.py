#!/usr/bin/env python3
"""Rebuild the report after the late Solver recovery.

The first report/build was written before the recovered raw freeze existed and
therefore froze a zero-return snapshot.  This engineering-only utility keeps
that exact snapshot under ``engineering_archive`` and asks the frozen observer
to read the current verified artifacts.  It does not call a model, alter scores
or content, or create a closeout audit.  A fresh report is written only after
the old report files have been copied and hash-recorded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(SCRIPTS))
import finalize_report as frozen  # noqa: E402
import finalize_report_delivery as delivery  # noqa: E402

ARCHIVE = BASE / "engineering_archive" / "report_before_solver_recovery"
REPORT_FILES = ("FINAL_REPORT.json", "FINAL_REPORT_ZH.md", "FINAL_REPORT_BUILD.json")
DISPOSITION = BASE / "REPORT_RECOVERY_DISPOSITION.json"
INVALID_CLOSEOUT_ARCHIVE = BASE / "engineering_archive" / "invalid_pre_recovery_closeout"
STALE_CLOSEOUT_NAMES = ("INDEPENDENT_CLOSEOUT_AUDIT.json", "INDEPENDENT_CLOSEOUT_AUDIT_RESPONSE.json",
                        "INDEPENDENT_CLOSEOUT_AUDIT_METADATA.json", "INDEPENDENT_CLOSEOUT_AUDIT_CALL.json",
                        "FINAL_ARTIFACT_MANIFEST.json")


def bind(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def dump_json(path: Path, value: object) -> None:
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError(f"IMMUTABLE_OUTPUT_DRIFT:{path}")
    if not path.exists():
        path.write_bytes(raw)


def closeout_failure_record() -> dict:
    root = BASE / "cli_calls" / "q20_closeout_frozen_report_01"
    paths = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            paths.append(bind(path))
    failure = root / "SEMANTIC_FAILURE.json"
    observed = json.loads(failure.read_text(encoding="utf-8")) if failure.is_file() else None
    invalid_audit = BASE / "INDEPENDENT_CLOSEOUT_AUDIT.json"
    return {
        "schema_version": "fgr20_closeout_audit_failure_disposition_v1",
        "status": "NO_VALID_INDEPENDENT_CLOSEOUT_AUDIT",
        "call_id": "q20_closeout_frozen_report_01",
        "semantic_attempt_consumed": bool(isinstance(observed, dict) and observed.get("semantic_attempt_consumed") is True),
        "recovery_classification": observed.get("recovery_classification") if isinstance(observed, dict) else None,
        "helper_error": observed.get("error") if isinstance(observed, dict) else None,
        "valid_audit_written": False,
        "audit_path": "INDEPENDENT_CLOSEOUT_AUDIT.json",
        "audit_file_present": False,
        "evidence_bindings": paths,
        "disposition": "The sole authorized closeout semantic attempt was consumed and rejected fail-closed by codex_calls.py after a top-level Codex error event. Its LAST_MESSAGE is preserved as raw evidence and is not promoted to an independent audit verdict; no replacement semantic call is authorized.",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def archive_stale_closeout_files() -> list[dict]:
    """Preserve old hash-bound closeout outputs without treating them as valid."""
    INVALID_CLOSEOUT_ARCHIVE.mkdir(parents=True, exist_ok=True)
    archived = []
    for name in STALE_CLOSEOUT_NAMES:
        source = BASE / name
        if not source.is_file():
            continue
        target = INVALID_CLOSEOUT_ARCHIVE / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"ARCHIVE_BINDING_DRIFT:{target}")
        if not target.exists():
            shutil.copy2(source, target)
        archived.append({"original": bind(source), "archived": bind(target)})
        source.unlink()
    return archived


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert frozen.bind(BASE / "scripts" / "finalize_report.py")["bytes"] > 0
        assert closeout_failure_record()["valid_audit_written"] is False
        print(json.dumps({"status": "PASS_ENGINEERING_ONLY", "model_calls": 0, "files_written": False}))
        return 0
    if not args.write:
        parser.error("choose --self-check or --write")

    # The old outputs are immutable evidence of the pre-recovery snapshot.
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    old_bindings = []
    for name in REPORT_FILES:
        source = BASE / name
        if not source.is_file():
            continue
        target = ARCHIVE / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"ARCHIVE_BINDING_DRIFT:{target}")
        if not target.exists():
            shutil.copy2(source, target)
        old_bindings.append({"original": bind(source), "archived": bind(target)})
    stale_closeout_bindings = archive_stale_closeout_files()

    # Freeze the new source/analysis snapshot before rebuilding the report.
    if not all((BASE / name).is_file() and (BASE / name).stat().st_size for name in
               ("REPLICATION_COMPARISON.json", "REPLICATION_TABLES_ZH.md", "REPLICATION_ROOT_DIFFERENCES.csv")):
        raise RuntimeError("REPLICATION_OUTPUTS_MISSING")
    receipt_path = BASE / "ORIGINAL_INTEGRITY_AFTER.json"
    fresh_receipt = frozen.actual_original_receipt(BASE)
    if receipt_path.is_file():
        prior = frozen.read_json(receipt_path)
        if frozen.integrity_receipt_identity(prior) != frozen.integrity_receipt_identity(fresh_receipt):
            raise RuntimeError("ORIGINAL_INTEGRITY_CHANGED_AFTER_RECEIPT_FREEZE")
    else:
        frozen.publish_immutable(receipt_path, fresh_receipt)
    receipt_binding = frozen.bind(receipt_path)

    # Reuse the report wrapper's factual, nonsemantic disclosures and renderer.
    facts = delivery.factual_disclosure(strict=True)
    # The old raw disclosure predates the recovered RAW_RESPONSE_FREEZE.  Keep
    # any differing prior bytes in the recovery archive before republishing the
    # same named disclosure with the current observed freeze state.
    for disclosure_path, value in ((delivery.DISCLOSURE, facts),):
        if disclosure_path.exists():
            try:
                frozen.publish_immutable(disclosure_path, value)
            except RuntimeError as exc:
                if not str(exc).startswith("IMMUTABLE_OUTPUT_DRIFT:"):
                    raise
                target = ARCHIVE / disclosure_path.name
                if target.exists() and target.read_bytes() != disclosure_path.read_bytes():
                    raise RuntimeError(f"ARCHIVE_BINDING_DRIFT:{target}")
                if not target.exists():
                    shutil.copy2(disclosure_path, target)
                disclosure_path.unlink()
    facts_receipt = frozen.publish_immutable(delivery.DISCLOSURE, facts)
    facts = {**facts, "disclosure_receipt_binding": facts_receipt}
    raw_facts = delivery.unfrozen_raw_disclosure()
    try:
        raw_receipt = frozen.publish_immutable(delivery.RAW_DISCLOSURE, raw_facts)
    except RuntimeError as exc:
        if not str(exc).startswith("IMMUTABLE_OUTPUT_DRIFT:"):
            raise
        target = ARCHIVE / delivery.RAW_DISCLOSURE.name
        if target.exists() and target.read_bytes() != delivery.RAW_DISCLOSURE.read_bytes():
            raise RuntimeError(f"ARCHIVE_BINDING_DRIFT:{target}")
        if not target.exists():
            shutil.copy2(delivery.RAW_DISCLOSURE, target)
        delivery.RAW_DISCLOSURE.unlink()
        raw_receipt = frozen.publish_immutable(delivery.RAW_DISCLOSURE, raw_facts)
    raw_facts = {**raw_facts, "disclosure_receipt_binding": raw_receipt}
    delivery.install_disclosure(facts, raw_facts)

    report, markdown, diagnostics = frozen.observe(BASE, receipt_binding)
    failure = closeout_failure_record()
    report.setdefault("engineering_disclosures", {})["closeout_audit_failure_disposition"] = failure
    report.setdefault("observed_issues", []).append("NO_VALID_INDEPENDENT_CLOSEOUT_AUDIT_SEMANTIC_ATTEMPT_CONSUMED")
    report["closeout_audit_state_at_report_freeze"] = "FAILED_SEMANTIC_ATTEMPT_NO_VALID_AUDIT"
    # Re-render from the updated report and append a plainly visible evidence block.
    markdown = frozen.render_markdown(report, BASE, report.get("failures_and_shortfalls", {}),
                                      report.get("usage", {}), report.get("observed_issues", []),
                                      fresh_receipt)
    marker = "## 收口审计失败处置"
    markdown += ("\n\n" + marker + "\n\n"
                 "唯一授权的 independent closeout semantic call 已被 `codex_calls.py` fail-closed；"
                 "其原始事件和 LAST_MESSAGE 仅作为证据保存，未生成有效 `INDEPENDENT_CLOSEOUT_AUDIT.json`，"
                 "因此本报告保持 PARTIAL，不能生成声称通过的独立审计。\n")

    # Remove only the stale derived names after their exact copies are archived.
    for name in ("FINAL_REPORT.json", "FINAL_REPORT_ZH.md", "FINAL_REPORT_BUILD.json"):
        path = BASE / name
        if path.exists():
            path.unlink()
    build = {"schema_version": "fgr20_report_build_recovery_v1",
             "assembler_binding": frozen.bind(Path(frozen.__file__)),
             "recovery_script_binding": frozen.bind(Path(__file__)),
             "original_integrity_receipt": receipt_binding,
             "report": report, "markdown": markdown, "diagnostics": diagnostics,
             "model_calls": 0,
             "purpose": "Rebuilt after recovered 98-record Solver freeze; prior report/build archived and bound."}
    frozen.publish_immutable(BASE / "FINAL_REPORT_BUILD.json", build)
    report_binding = frozen.publish_immutable(BASE / "FINAL_REPORT.json", report)
    markdown_binding = frozen.publish_text_immutable(BASE / "FINAL_REPORT_ZH.md", markdown)
    disposition = {
        "schema_version": "fgr20_report_recovery_disposition_v1",
        "status": "RECOVERED_REPORT_REBUILT_PARTIAL",
        "reason": "The pre-recovery report froze before late Solver recovery and reported zero returned responses.",
        "prior_report_bindings": old_bindings,
        "archived_directory": str(ARCHIVE.resolve()),
        "new_report_binding": report_binding,
        "new_markdown_binding": markdown_binding,
        "new_build_binding": frozen.bind(BASE / "FINAL_REPORT_BUILD.json"),
        "solver_response_count_observed": report.get("counts", {}).get("new_responses_returned"),
        "closeout_failure": failure,
        "stale_closeout_bindings": stale_closeout_bindings,
        "model_calls": 0,
    }
    dump_json(DISPOSITION, disposition)
    print(json.dumps({"status": report.get("overall_status"), "counts": report.get("counts"),
                      "report": report_binding, "markdown": markdown_binding,
                      "disposition": frozen.bind(DISPOSITION)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
