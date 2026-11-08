"""Hash and make this completed baseline-screen delivery read-only."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {"path": str(path.resolve()), "sha256": sha(path)}


def save(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    receipt_path = HERE / "SEAL_RECEIPT.json"
    manifest_path = HERE / "COMPLETE_MANIFEST.json"
    assert not receipt_path.exists() and not manifest_path.exists()
    final = read(HERE / "FINAL_STATUS.json")
    assert final["status"] == "COMPLETE_BASELINE_SCREEN_NEXT_ROOT_SELECTION_FIXED"
    assert final["candidate_roots"] == 8 and final["review_count"] == 16
    for key in ("generation_started", "solver_started", "training_started", "pool_merge_started"):
        assert final[key] is False
    for binding in final["bindings"]:
        assert sha(binding["path"]) == binding["sha256"]
    audit = read(HERE / "DELIVERY_AUDIT.json")
    assert audit["status"] == "PASS" and audit["errors"] == []
    for binding in audit["input_bindings"]:
        assert sha(binding["path"]) == binding["sha256"]
    report_review_path = HERE / "evidence/final_report_review.json"
    report_review = read(report_review_path)
    assert report_review["status"] == "PASS" and report_review["errors"] == []
    for binding in report_review["input_bindings"]:
        assert sha(binding["path"]) == binding["sha256"]
    tests = read(HERE / "UNIT_TEST_RUN.json")
    assert tests["status"] == "PASS" and tests["exit_code"] == 0
    for binding in tests["bindings"] + [tests["log"]]:
        assert sha(binding["path"]) == binding["sha256"]
    protected = read(HERE / "HISTORICAL_PROTECTION.json")["files"]
    for binding in protected:
        assert sha(binding["path"]) == binding["sha256"]
    for binding in read(HERE / "TOOLING_FREEZE.json")["files"]:
        assert sha(binding["path"]) == binding["sha256"]
        assert sha(binding["snapshot"]["path"]) == binding["snapshot"]["sha256"]
    created = datetime.now(timezone.utc).isoformat()
    save(receipt_path, {
        "status": "VERIFIED_BASELINE_SCREEN_READY_FOR_READ_ONLY_SEAL",
        "created_at_utc": created,
        "candidate_roots": 8, "review_count": 16,
        "selected_root_ids": final["selected_root_ids"],
        "held_root_ids": final["held_root_ids"],
        "historical_protected_files_verified": len(protected),
        "final_status": bind(HERE / "FINAL_STATUS.json"),
        "final_report_review": bind(report_review_path),
        "scope": "Baseline screen and next batch selection only; no generation, Solver, training or pool merge.",
        "seal_mechanism": "Complete file SHA256 manifest plus file mode 0444 and directory mode 0555; not immutable storage.",
    })
    files = []
    directories = []
    for path in sorted(HERE.rglob("*")):
        assert not path.is_symlink(), path
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            directories.append(path)
        else:
            raise AssertionError(f"Unsupported artifact: {path}")
    save(manifest_path, {
        "status": "COMPLETE_BASELINE_SCREEN_NO_GENERATION",
        "created_at_utc": created,
        "covered_files_excluding_this_manifest": len(files),
        "self_hash_excluded": True,
        "files": [{**bind(path), "relative_path": str(path.relative_to(HERE)),
                   "size_bytes": path.stat().st_size} for path in files],
    })
    manifest = read(manifest_path)
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file()}
    assert actual == {r["relative_path"] for r in manifest["files"]} | {manifest_path.name}
    for binding in manifest["files"]:
        assert sha(binding["path"]) == binding["sha256"]
    for path in files + [manifest_path]:
        path.chmod(0o444)
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True) + [HERE]:
        path.chmod(0o555)
    assert all((p.stat().st_mode & 0o777) == 0o444 for p in files + [manifest_path])
    assert all((p.stat().st_mode & 0o777) == 0o555 for p in directories + [HERE])
    print(json.dumps({"status": "SEALED", "covered_files": len(files),
                      "manifest": bind(manifest_path),
                      "historical_protected_files_verified": len(protected)}))


if __name__ == "__main__":
    main()
