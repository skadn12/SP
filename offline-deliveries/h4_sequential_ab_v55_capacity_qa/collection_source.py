"""Read-only bridge from audited provisional v52 A/B dialogue into QA evidence.

The source remains a candidate. This module checks artifact integrity and never
converts a scene's disposition or claims semantic/training admission.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

VERSION = "h4_candidate_collection_v51"
MODEL = "Qwen/Qwen3.6-35B-A3B"


def replay_audit(run):
    program = run.parent.parent / "candidate_runner.py"
    result = subprocess.run([sys.executable, str(program), "audit", "--run", str(run)],
        cwd=program.parent, text=True, capture_output=True, timeout=180, check=True)
    return json.loads(result.stdout)


def audit_dependencies(run):
    """Return local expansion audit code that must be frozen with this QA run."""
    home = Path(run).parent.parent
    if not (home / 'candidate_runner.py').is_file():
        raise ValueError('candidate_audit_program_missing')
    return sorted(home.glob('*.py'))


def checked_source(history_path, base, *, requested_roots=None):
    history_path = Path(history_path).resolve()
    run = history_path.parent
    require, read = base.require, base.read
    require(history_path.name == "CANDIDATE_HISTORY.json", "candidate_history_required")
    manifest = read(run / "MANIFEST.json")
    require(manifest.get("version") == VERSION and manifest.get("scope") == "fresh_collection",
            "candidate_collection_version_mismatch")
    require(manifest.get("model") == MODEL and manifest.get("training_pool_eligible") is False,
            "candidate_collection_policy_mismatch")
    root = manifest["root_id"]
    require(not requested_roots or set(requested_roots) == {root}, "requested_root_not_in_collection")
    audit = replay_audit(run)
    require(audit.get("status") == "ARTIFACT_REPLAY_VERIFIED"
            and audit.get("semantic_certified") is False
            and audit.get("training_pool_eligible") is False, "candidate_artifact_audit_failed")
    history = read(history_path)
    baseline = read(run / "FORMAL_BASELINE.json")["turns"]
    records = read(run / "CANDIDATE_RECORDS.json")
    turns = history["turns"]
    require(history.get("training_pool_eligible") is False
            and history.get("history_policy") == "formal_prefix_then_provisional_retained_dialogue"
            and history.get("formal_baseline_turns") == len(baseline), "candidate_history_policy_changed")
    require(turns[:len(baseline)] == baseline, "candidate_formal_prefix_changed")
    require(all(isinstance(t, dict) and all(isinstance(t.get(k), str) and t[k].strip()
        for k in ("turn_ref", "speaker", "text")) for t in turns), "candidate_turn_schema")
    require(len({t["turn_ref"] for t in turns}) == len(turns), "candidate_duplicate_turn_refs")
    suffix = turns[len(baseline):]
    require(all(t.get("origin") == "provisional_candidate" and t.get("training_pool_eligible") is False
            and t["turn_ref"].startswith("CANDIDATE:") for t in suffix), "candidate_suffix_status_changed")
    retained = [r for r in records if r["disposition"] == "retained_pending_review"]
    require(len(suffix) == sum(len(r["realization"]["turns"]) for r in retained), "candidate_suffix_count_changed")
    require([{key: turn[key] for key in ('speaker', 'text')} for turn in suffix]
            == [{key: turn[key] for key in ('speaker', 'text')}
                for record in retained for turn in record['realization']['turns']],
            "candidate_suffix_records_changed")
    source = {**deepcopy(history), "root_id": root,
        "source_mode": "formal_prefix_then_provisional_retained_dialogue",
        "source_history_sha256": base.digest(history_path.read_bytes()),
        "new_admitted_scenes": 0, "retained_candidate_scenes": len(retained),
        "candidate_source_run": str(run), "semantic_certified": False,
        "training_allowed": False, "source_admission_status": "PROVISIONAL_CANDIDATE_HISTORY",
        "upstream_admitted_scenes": 0,
        "candidate_scene_ids": [r["candidate_id"] for r in retained]}
    return {"root_id": root, "path": history_path, "source": source,
        "sha256": base.digest(history_path.read_bytes()), "new_admitted_scenes": 0,
        "retained_candidate_scenes": len(retained), "audit": audit,
        "merge_record": {"status": "ARTIFACT_REPLAY_VERIFIED", "semantic_certified": False,
                         "training_pool_eligible": False, "candidate_source_run": str(run)}}
