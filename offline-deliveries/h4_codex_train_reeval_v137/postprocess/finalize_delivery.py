"""Seal completed v137 results and verify a plain, weight-free supplemental tar."""
from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


BATCH = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    archive = BATCH.with_suffix(".tar")
    checksum = Path(str(archive) + ".sha256")
    status_path = BATCH / "FINAL_STATUS.json"
    manifest_path = BATCH / "COMPLETE_MANIFEST.json"
    archive_check_path = BATCH.parent / (BATCH.name + ".ARCHIVE_VERIFICATION.json")
    for path in [archive, checksum, status_path, manifest_path, archive_check_path]:
        require(not path.exists(), "REFUSE_TO_OVERWRITE:" + str(path))
    completed = load(BATCH / "evaluation/RUN_COMPLETE.json")
    summary = load(BATCH / "evaluation/reports/SUMMARY.json")
    audit = load(BATCH / "evidence/INDEPENDENT_RESULTS_AUDIT.json")
    diagnosis = load(BATCH / "diagnostics/FIT_COMPARISON.json")
    require(completed.get("status") == "COMPLETE"
            and summary.get("status") == "COMPLETE" and audit.get("status") == "PASS",
            "RESULTS_NOT_COMPLETE_AND_AUDITED")
    require(summary.get("question_count") == 363 and summary.get("calls_per_model") == 5784,
            "RESULT_SCOPE_CHANGED")
    require(not summary.get("errors") and not audit.get("errors"), "RESULT_ERRORS")
    require(audit.get("failure_count") == 0 and audit.get("fixed_questions") == 363
            and audit.get("calls_per_model") == 5784
            and audit.get("planned_calls_all_models") == 11568, "AUDIT_SCOPE_OR_FAILURES")
    require(diagnosis.get("status") == "COMPLETE", "DIAGNOSTIC_NOT_COMPLETE")
    bindings = audit["audited_files"] + diagnosis["source_bindings"] + [diagnosis["markdown_output"]]
    verified = {}
    for row in bindings:
        path = Path(row["path"])
        if not path.is_absolute():
            path = BATCH / path
        path = path.resolve()
        require(path.is_file(), "AUDITED_FILE_MISSING:" + str(path))
        if path not in verified:
            verified[path] = sha(path)
        require(verified[path] == row["sha256"], "AUDITED_FILE_CHANGED:" + str(path))
        size = row.get("bytes", row.get("size_bytes"))
        require(size is None or size == path.stat().st_size, "AUDITED_SIZE_CHANGED:" + str(path))
    for relative in ["README_ZH.md", "evaluation/reports/REPORT_ZH.md", "diagnostics/FIT_DIAGNOSTIC_ZH.md"]:
        require((BATCH / relative).is_file(), "REQUIRED_REPORT_MISSING:" + relative)
    runtime = {}
    for side in ["clean", "new"]:
        run = BATCH / "evaluation/run" / side
        terminal = load(run / "COMPLETE.json")
        require(terminal.get("status") == "COMPLETE" and terminal.get("actual_call_count") == 5784,
                "INCOMPLETE_SIDE:" + side)
        require(len(list((run / "responses").glob("*.json"))) == 5784, "RESPONSE_COUNT:" + side)
        require(not (run / "FAILED.json").exists(), "FAILED_SIDE:" + side)
        runtime[side] = {k: terminal[k] for k in ["actual_call_count", "wall_seconds", "completed_at_utc"]}
    save(status_path, {
        "status": "COMPLETE", "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "363 actual training questions; M-clean versus round2 final; seen-question fit diagnosis",
        "question_count": 363, "training_root_count": 68, "calls_per_model": 5784,
        "total_model_calls": 11568, "new_training": False, "new_base_inference": False,
        "independent_results_audit": "PASS", "models": runtime,
        "supplement_to": "h4_codex_round2_lora_bundle_v136.tar",
        "weights_included": False,
        "primary_report": "diagnostics/FIT_DIAGNOSTIC_ZH.md",
        "numerical_report": "evaluation/reports/REPORT_ZH.md",
    })
    files = []
    for path in sorted(BATCH.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        require(not path.is_symlink() and path.resolve().is_relative_to(BATCH), "SYMLINK_OR_ESCAPE")
        require(path.suffix not in {".pt", ".pth", ".safetensors", ".bin"}, "UNEXPECTED_WEIGHTS:" + str(path))
        files.append({"path": path.relative_to(BATCH).as_posix(),
                      "bytes": path.stat().st_size, "sha256": sha(path)})
    save(manifest_path, {"status": "COMPLETE", "files": files, "file_count": len(files),
                         "self_excluded": True, "weights_included": False})
    expected = {BATCH.name + "/" + row["path"]: row for row in files}
    expected[BATCH.name + "/COMPLETE_MANIFEST.json"] = {
        "bytes": manifest_path.stat().st_size, "sha256": sha(manifest_path)}
    with tarfile.open(archive, "x", format=tarfile.PAX_FORMAT) as handle:
        for name in sorted(expected):
            handle.add(BATCH.parent / name, arcname=name, recursive=False)
    seen = set()
    with tarfile.open(archive, "r:") as handle:
        for member in handle:
            require(member.isfile() and member.name in expected and member.name not in seen,
                    "UNEXPECTED_ARCHIVE_MEMBER:" + member.name)
            row = expected[member.name]
            require(member.size == row["bytes"], "ARCHIVE_SIZE_MISMATCH:" + member.name)
            digest = hashlib.sha256()
            source = handle.extractfile(member)
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            require(digest.hexdigest() == row["sha256"], "ARCHIVE_HASH_MISMATCH:" + member.name)
            seen.add(member.name)
    require(seen == set(expected), "MISSING_ARCHIVE_MEMBERS")
    archive_sha = sha(archive)
    with checksum.open("x", encoding="utf-8") as handle:
        handle.write(archive_sha + "  " + archive.name + "\n")
    save(archive_check_path, {
        "status": "PASS", "archive": str(archive), "bytes": archive.stat().st_size,
        "sha256": archive_sha, "verified_regular_members": len(seen),
        "plain_tar": True, "weights_included": False, "verified_each_member_sha256": True,
        "manifest_sha256": sha(manifest_path),
    })
    print(json.dumps({"status": "COMPLETE", "archive": str(archive),
                      "bytes": archive.stat().st_size, "sha256": archive_sha}))


if __name__ == "__main__":
    main()
