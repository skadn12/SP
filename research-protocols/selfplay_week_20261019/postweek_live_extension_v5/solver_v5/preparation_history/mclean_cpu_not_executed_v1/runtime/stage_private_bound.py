"""Stage the exact pre-frozen private bytes; never parse Gold answer JSON."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import common as c

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--private-source", type=Path, required=True)
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--public-export-freeze", type=Path, required=True)
    args = p.parse_args()
    c.environment()
    run, private, source, export_path = (x.resolve() for x in
        (args.run, args.private, args.private_source, args.public_export_freeze))
    c.require(not private.exists(), "PRIVATE_SIDECAR_EXISTS_NO_OVERWRITE")
    c.require(not c.below(private, run) and not c.below(run, private), "PRIVATE_SIDECAR_MUST_BE_DISJOINT")
    manifest = c.verify(run, check_models=False)
    c.require(any(Path(b["path"]).resolve() == export_path and b["sha256"] == c.sha(export_path)
        for b in manifest["input_bindings"]), "PUBLIC_EXPORT_NOT_BOUND_TO_THIS_RUN")
    export = c.read(export_path)
    c.require(export["status"] == "EXPORTED_AND_FROZEN_BEFORE_SOLVER", "PUBLIC_EXPORT_NOT_FROZEN")
    c.verify_bindings(list(c.all_bindings(export)))
    expected = export["bindings"]["private_questions"]
    c.require(Path(expected["path"]).resolve() == source and c.bind(source) == expected,
        "PRIVATE_SOURCE_NOT_EXACTLY_FROZEN_EXPORT")
    private.mkdir(parents=True, mode=0o700)
    dest = private / "PRIVATE_QUESTIONS.json"
    shutil.copy2(source, dest)
    dest.chmod(0o600)
    copied = c.bind(dest)
    c.require(copied["sha256"] == expected["sha256"] and copied["bytes"] == expected["bytes"], "PRIVATE_COPY_DRIFT")
    c.save(private / "PRIVATE_BINDINGS.json", {"created_at_utc": c.now(),
        "public_manifest_binding": c.bind(run / "manifest.json"),
        "public_preflight_binding": c.bind(run / "PREFLIGHT_FREEZE.json"),
        "public_export_freeze_binding": c.bind(export_path),
        "original_private_binding": expected, "private_answer_binding": copied,
        "staging_code_binding": c.bind(__file__), "private_json_parsed_by_staging": False,
        "exact_frozen_private_source_required": True,
        "gold_access_scope": "Byte copy only; prior author/reviewer/exporter Gold exposure disclosed in export freeze."})
    (private / "PRIVATE_BINDINGS.json").chmod(0o600)
    print("PASS: exact frozen private bytes staged without parsing")

if __name__ == "__main__":
    main()
