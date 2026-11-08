"""Byte-copy Gold sidecar without parsing answers; public runtime stays disjoint."""
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
    args = p.parse_args()
    c.environment()
    run, private, source = args.run.resolve(), args.private.resolve(), args.private_source.resolve()
    c.require(not private.exists(), "PRIVATE_SIDECAR_EXISTS_NO_OVERWRITE")
    c.require(not c.below(private, run) and not c.below(run, private), "PRIVATE_SIDECAR_MUST_BE_DISJOINT")
    c.verify(run, check_models=False)
    private.mkdir(parents=True, mode=0o700)
    dest = private / "PRIVATE_QUESTIONS.json"
    shutil.copy2(source, dest)
    dest.chmod(0o600)
    c.save(private / "PRIVATE_BINDINGS.json", {"created_at_utc": c.now(),
        "public_manifest_binding": c.bind(run / "manifest.json"),
        "public_preflight_binding": c.bind(run / "PREFLIGHT_FREEZE.json"),
        "original_private_binding": c.bind(source), "private_answer_binding": c.bind(dest),
        "private_json_parsed_by_staging": False,
        "gold_access_scope": "Byte copy only; author/exporter prior Gold exposure disclosed in PUBLIC_EXPORT_FREEZE.json."})
    (private / "PRIVATE_BINDINGS.json").chmod(0o600)
    print("PASS: private bytes staged without parsing")

if __name__ == "__main__":
    main()
