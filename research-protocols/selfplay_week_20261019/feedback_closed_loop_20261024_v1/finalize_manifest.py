"""Seal the completed local experiment without touching parent experiments."""
import datetime
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def main():
    status = json.loads((P / "FINAL_STATUS.json").read_text())
    assert status["status"] == "COMPLETE_NO_TRAINING" and status["training"] is False
    for binding in status["bindings"]:
        data = Path(binding["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == binding["sha256"]
        assert len(data) == binding["bytes"]
    review = json.loads((P / "FINAL_REPORT_REVIEW.json").read_text())
    assert review["status"] == "PASS"
    files = []
    for path in sorted(P.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name in ("FINAL_MANIFEST.json", "FINAL_MANIFEST.sha256"):
            continue
        data = path.read_bytes()
        files.append({"path": str(path.relative_to(P)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"schema_version": "feedback_closed_loop_manifest_v1",
        "status": "SEALED_COMPLETE_NO_TRAINING", "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_directory": str(P), "files": files, "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files), "excluded": ["__pycache__", "FINAL_MANIFEST.json", "FINAL_MANIFEST.sha256"],
        "training": False, "post_solver_candidate_revision": False}
    destination = P / "FINAL_MANIFEST.json"
    with destination.open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    with (P / "FINAL_MANIFEST.sha256").open("x") as stream:
        stream.write(digest + "  FINAL_MANIFEST.json\n")
    for record in files:
        data = (P / record["path"]).read_bytes()
        assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
    print(json.dumps({"status": manifest["status"], "files": len(files), "bytes": manifest["total_bytes"], "manifest_sha256": digest}))


if __name__ == "__main__":
    main()
