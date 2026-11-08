"""Seal the additive completed delivery; do not modify experimental inputs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

S = Path(__file__).resolve().parent
EXCLUDED = {"FINAL_MANIFEST.json", "FINAL_MANIFEST_CHECK.json"}


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def bind(p):
    data = p.read_bytes()
    return {"path": str(p), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    for filename in ["INDEPENDENT_DESIGN_CHECK.json", "INDEPENDENT_RESULTS_CHECK.json", "FINAL_REPORT_REVIEW.json"]:
        assert read(S / filename)["status"] == "PASS", filename
    assert read(S / "CLOSURE_CHECK.json")["integrity_pass"] is True
    assert read(S / "EXAMPLES_EXPORT_CHECK.json")["verbatim_text_present"] is True
    assert read(S / "FINAL_STATUS.json")["semantic_audit_status"] == "COMPLETE_FIXED_SUBSET"
    assert read(S / "semantic_audit/SUMMARY.json")["coverage_and_bindings_pass"] is True
    old = read(S / "OLD_P_FINAL_MANIFEST_CHECK.json")
    old_manifest = read(Path(old["manifest_binding"]["path"]))
    old_bindings = []

    def collect(obj):
        if isinstance(obj, dict):
            if {"path", "sha256", "bytes"} <= set(obj):
                old_bindings.append(obj)
            else:
                for value in obj.values():
                    collect(value)
        elif isinstance(obj, list):
            for value in obj:
                collect(value)

    collect(old_manifest)
    assert len(old_bindings) == 775
    for b in [old["manifest_binding"], *old_bindings]:
        assert bind(Path(b["path"])) == {k: b[k] for k in ["path", "sha256", "bytes"]}, b["path"]
    artifacts = [bind(p) for p in sorted(S.rglob("*")) if p.is_file() and str(p.relative_to(S)) not in EXCLUDED]
    packet = read(S / "semantic_audit/SEMANTIC_AUDIT_PACKET.json")
    sources = {c["source_binding"]["path"]: c["source_binding"] for c in packet["cases"]}
    u = S.parent / "thinking_switch_20261024_v1"
    u_files = ["PROTOCOL_ZH.md", "REPORT_ZH.md", "ENGINEERING_FOLLOWUP_DECISION.json", "SEMANTIC_AUDIT_DISPOSITION.json", "CLOSURE_CHECK.py", "CLOSURE_CHECK.json", "execution/COMPARISON.json", "execution/RAW_RESPONSE_FREEZE.json"]
    dependencies = list(sources.values()) + [old["manifest_binding"]] + [bind(u / f) for f in u_files]
    manifest = {"schema_version": "thinking_switch_completed_delivery_manifest_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "All files in this v2 delivery except the manifest and its check; four original sources and selected immutable reference-experiment records. Existing raw freeze/closure checks bind the remaining generation dependencies.", "artifacts": artifacts, "dependencies": dependencies, "excluded_relative_paths": sorted(EXCLUDED)}
    dump(S / "FINAL_MANIFEST.json", manifest)
    errors = []
    for b in artifacts + dependencies:
        try:
            got = bind(Path(b["path"]))
            if any(got[k] != b[k] for k in ["path", "sha256", "bytes"]):
                errors.append({"path": b["path"], "error": "BINDING_MISMATCH"})
        except Exception as exc:
            errors.append({"path": b["path"], "error": str(exc)})
    expected = {b["path"] for b in artifacts}
    actual = {str(p) for p in S.rglob("*") if p.is_file() and str(p.relative_to(S)) not in EXCLUDED}
    if expected != actual:
        errors.append({"error": "DELIVERY_FILE_SET_MISMATCH"})
    result = {"status": "PASS" if not errors else "FAIL", "manifest_binding": bind(S / "FINAL_MANIFEST.json"), "artifacts_checked": len(artifacts), "dependencies_checked": len(dependencies), "original_p_bindings_rechecked_now": len(old_bindings), "delivery_file_set_complete": expected == actual, "errors": errors, "scope": "Byte/hash and file-set integrity, not an answer-correctness or causal-benefit guarantee."}
    dump(S / "FINAL_MANIFEST_CHECK.json", result)
    assert not errors, errors
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
