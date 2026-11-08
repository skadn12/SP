"""Verify the sealed closeout without model loading, network, or file writes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def digest(path):
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def verify_binding(binding):
    path = Path(binding["path"])
    size, sha = digest(path)
    assert size == binding["bytes"], f"byte mismatch: {path}"
    assert sha == binding["sha256"], f"sha256 mismatch: {path}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", required=True)
    args = parser.parse_args()
    manifest_path = HERE / "FINAL_MANIFEST.json"
    nav_path = HERE / "POSTWEEK_TRAIN_REPLICATION_LATEST.json"
    manifest = json.loads(manifest_path.read_text())
    nav = json.loads(nav_path.read_text())

    listed = set()
    for item in manifest["files"]:
        path = Path(item["path"]).resolve()
        assert path.is_relative_to(HERE), f"out of package: {path}"
        verify_binding(item)
        listed.add(path)

    excluded = {manifest_path.resolve(), nav_path.resolve()}
    actual = {path.resolve() for path in HERE.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts}
    assert actual - excluded == listed, "package file inventory differs from manifest"
    for key in ("decision", "report", "closeout", "final_scores",
                "preflight_audit", "independent_post_audit", "manifest",
                "provenance_clarification", "reproduction", "next_control_design_draft"):
        verify_binding(nav[key])
    assert nav_path.read_bytes() == (HERE.parent / nav_path.name).read_bytes(), \
        "discovery navigation differs from package navigation"

    # Recompute deterministic score aggregation without touching the sealed files.
    import adjudicate
    computed = adjudicate.compute()
    saved = json.loads((HERE / "FINAL_SCORES.json").read_text())
    assert computed["summary"] == saved["summary"]
    assert computed["requests"] == saved["requests"]
    assert computed["roots"] == saved["roots"]
    decision = json.loads((HERE / "DECISION.json").read_text())
    assert decision["new_roots_qualifying_both_mappings"] == 1
    assert decision["required_new_roots_and_families"] == 3
    assert decision["additional_calls_authorized"] is False
    print(json.dumps({"status": "PASS", "manifest_files": len(listed),
                      "scoring_recomputed": True, "requests": 12,
                      "qualifying_roots": 1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
