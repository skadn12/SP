"""Explicit export validation and future frozen-R03 binding, offline only."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from . import READY, VERSION
from .renderers import backend_record, render_role, validate_envelope
from .sources import canonical, file_sha, read, value_sha


BOUND = "QA_SNAPSHOT_BOUND_OFFLINE_ONLY"
ROLES = ("expansion", "fresh", "targeted", "composition")
FORBIDDEN_FILENAMES = {"config.json", "config.toml", "model_config.json", "auth.json", "credentials.json", ".env"}
REQUIRED_ARTIFACTS = (
    "implementation_identity.json", "final_report.json", "offline_acceptance.json",
    "retained_candidate_scope.jsonl", "excluded_candidates.jsonl",
    "active_question_bank_projection.jsonl", "active_clusters.jsonl",
    "active_shared_feedback_records.jsonl", "role_feedback_envelopes.jsonl",
    "active_expansion_seed_pool.jsonl", "active_expansion_selection.jsonl",
    "active_targeted_pool.jsonl", "active_composition_parent_pool.jsonl",
    "active_composition_pairs.jsonl", "fresh_historical_backend_view.jsonl",
)


def _safe_path(path):
    path = Path(path).resolve()
    if path.name in FORBIDDEN_FILENAMES:
        raise ValueError("credential_or_service_config_read_forbidden")
    return path


def _checked(path, expected):
    path = _safe_path(path)
    if not isinstance(expected, str) or len(expected) != 64 or file_sha(path) != expected:
        raise ValueError(f"export_or_source_sha_mismatch:{path}")
    return path


def _sidecar_sha(path):
    return Path(str(path) + ".sha256").read_text(encoding="utf-8").split()[0]


def _index(rows, key):
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate_export_identity:{key}")
    return indexed


def _source(loaded, name):
    spec = loaded["source_manifest"]["inputs"][name]
    path = _checked(spec["path"], spec["sha256"])
    return read(path)


def validate_export_data(artifacts, source_manifest_sha):
    """Validate linked identities after artifact file hashes were verified."""
    scope = _index(artifacts["retained_candidate_scope.jsonl"], "candidate_id")
    excluded = {row["candidate_id"] for row in artifacts["excluded_candidates.jsonl"]}
    bank = _index(artifacts["active_question_bank_projection.jsonl"], "candidate_id")
    clusters = _index(artifacts["active_clusters.jsonl"], "solver_prompt_fingerprint")
    records = _index(artifacts["active_shared_feedback_records.jsonl"], "source_candidate_id")
    if set(scope) != set(bank) or set(scope) & excluded:
        raise ValueError("export_retained_bank_scope_mismatch")
    for cid, item in scope.items():
        entry = bank[cid]
        qa = entry["candidate"]
        if item.get("filter_version") != VERSION or item.get("split") != "train" or entry.get("split") != "train" or item.get("active_feedback_scope_allowed") is not True:
            raise ValueError("export_non_train_or_disallowed_scope")
        if qa["root_id"] != item["root_id"] or value_sha(qa) != item["candidate_record_sha256"] or value_sha(entry) != item["bank_record_sha256"]:
            raise ValueError("export_candidate_record_sha_mismatch")
        if item.get("normalized_experiment_arm") == "ROUND0_PARENT_PLUS_R01" and qa["question_type"] == "ordering":
            raise ValueError("export_round0_ordering_reintroduced")
    for fp, cluster in clusters.items():
        ids = cluster["member_candidate_ids"]
        if not ids or len(ids) != len(set(ids)) or any(cid not in scope or scope[cid]["solver_prompt_fingerprint"] != fp for cid in ids):
            raise ValueError("export_cluster_members_outside_scope")
    for cid, record in records.items():
        fp = record["source_cluster_id"]
        if cid not in scope or fp not in clusters:
            raise ValueError("export_record_outside_retained_scope")
        cluster = clusters[fp]
        if cluster.get("active_observation_candidate_id") != cid or cluster.get("canonical_representative_candidate_id") != cid or cluster.get("active_observation_status") != "RETAINED_INDEPENDENT_OBSERVATION":
            raise ValueError("export_record_observation_origin_mismatch")
        expected = backend_record(bank[cid], cluster, source_manifest_sha)
        if value_sha(record) != value_sha(expected):
            raise ValueError("export_feedback_record_sha_mismatch")

    def check_pool(name):
        result = {}
        for item in artifacts[name]:
            cid = item["canonical_representative_candidate_id"]
            if cid not in records or records[cid]["root_id"] != item["root_id"] or records[cid]["source_cluster_id"] != item["solver_prompt_fingerprint"] or cid in result:
                raise ValueError(f"export_role_pool_membership_mismatch:{name}")
            if records[cid]["source_arm"] == "E0Q0_CONTROL_BLIND":
                raise ValueError("export_control_source_in_feedback_pool")
            result[cid] = item
        return result

    expansion = check_pool("active_expansion_seed_pool.jsonl")
    targeted = check_pool("active_targeted_pool.jsonl")
    parents = check_pool("active_composition_parent_pool.jsonl")
    selections = check_pool("active_expansion_selection.jsonl")
    if not set(selections) <= set(expansion):
        raise ValueError("export_expansion_selection_not_in_pool")
    envelopes = _index(artifacts["role_feedback_envelopes.jsonl"], "selection_key")
    root_roles = defaultdict(set)
    for envelope in envelopes.values():
        role, root = envelope["role"], envelope["root_id"]
        if role not in ROLES:
            raise ValueError("unsupported_export_role")
        root_roles[root].add(role)
        validate_envelope(envelope, set(scope))
        for record in envelope["records"]:
            if record != records.get(record["source_candidate_id"]):
                raise ValueError("export_envelope_record_sha_mismatch")
    if any(roles != set(ROLES) for roles in root_roles.values()) or not root_roles:
        raise ValueError("export_root_role_coverage_incomplete")
    expected = {}
    for root in root_roles:
        selected = [cid for cid, row in selections.items() if row["root_id"] == root]
        available = [cid for cid, row in expansion.items() if row["root_id"] == root]
        if len(selected) != min(1, len(available)):
            raise ValueError("forged_no_feedback_or_expansion_selection_cardinality")
        expected[f"expansion:{root}"] = ("expansion", root, selected)
        expected[f"fresh:{root}"] = ("fresh", root, [])
        target_ids = [cid for cid, row in targeted.items() if row["root_id"] == root]
        if target_ids:
            for cid in target_ids:
                expected[f"targeted:{cid}"] = ("targeted", root, [cid])
        else:
            expected[f"targeted:{root}"] = ("targeted", root, [])
        pairs = [row for row in artifacts["active_composition_pairs.jsonl"] if row["root_id"] == root]
        for row in pairs:
            left, right = row["left_candidate_id"], row["right_candidate_id"]
            if left == right or left not in parents or right not in parents or parents[left]["root_id"] != root or parents[right]["root_id"] != root or records[left]["source_cluster_id"] != row["left_cluster_id"] or records[right]["source_cluster_id"] != row["right_cluster_id"] or row["left_cluster_id"] == row["right_cluster_id"]:
                raise ValueError("export_composition_pair_membership_mismatch")
            key = f"composition:{root}:{row['left_cluster_id']}:{row['right_cluster_id']}"
            if key in expected:
                raise ValueError("duplicate_export_composition_pair")
            expected[key] = ("composition", root, [left, right])
        if not pairs:
            expected[f"composition:{root}"] = ("composition", root, [])
    if set(expected) != set(envelopes):
        raise ValueError("export_role_envelope_selection_coverage_mismatch")
    for key, (role, root, ids) in expected.items():
        envelope = envelopes[key]
        if envelope["role"] != role or envelope["root_id"] != root or [record["source_candidate_id"] for record in envelope["records"]] != ids:
            raise ValueError("forged_no_feedback_or_role_source_membership")
    all_role_roots = {row["root_id"] for name in ("active_expansion_seed_pool.jsonl", "active_targeted_pool.jsonl", "active_composition_parent_pool.jsonl", "active_composition_pairs.jsonl") for row in artifacts[name]}
    if not all_role_roots <= set(root_roles):
        raise ValueError("export_role_pool_root_without_envelope")
    return {"scope": scope, "bank": bank, "clusters": clusters, "records": records, "envelopes": envelopes, "roots": sorted(root_roles)}


def load_export(export_manifest):
    """Load a complete export directory or its SHA-sidecar-bound inventory."""
    manifest = Path(export_manifest).resolve()
    if manifest.suffix != ".json":
        manifest = manifest / "artifact_inventory.json"
    manifest = _checked(manifest, _sidecar_sha(manifest))
    inventory = read(manifest)
    artifacts = {}
    seen_paths = set()
    for row in inventory:
        path = _checked(row["path"], row["sha256"])
        if path.parent != manifest.parent or path in seen_paths:
            raise ValueError("export_inventory_path_scope_or_duplicate")
        seen_paths.add(path)
        if path.name in REQUIRED_ARTIFACTS:
            artifacts[path.name] = read(path)
    if set(REQUIRED_ARTIFACTS) - set(artifacts):
        raise ValueError("export_required_artifact_missing")
    if artifacts["final_report.json"].get("status") != READY or artifacts["offline_acceptance.json"].get("passed") is not True:
        raise ValueError("export_not_offline_ready")
    identity = artifacts["implementation_identity.json"]
    source_path = _checked(identity["source_manifest"], identity["source_manifest_sha256"])
    source_manifest = read(source_path)
    if source_manifest.get("filter_version") != VERSION or source_manifest.get("real_calls_authorized") is not False:
        raise ValueError("export_source_manifest_contract_mismatch")
    result = {"export_manifest": str(manifest), "export_manifest_sha256": file_sha(manifest),
              "source_manifest_path": str(source_path), "source_manifest_sha256": identity["source_manifest_sha256"],
              "source_manifest": source_manifest, "artifacts": artifacts}
    result.update(validate_export_data(artifacts, identity["source_manifest_sha256"]))
    splits = _index(_source(result, "round0_split"), "root_id")
    if any(splits.get(row["root_id"], {}).get("split") != "train" for row in result["scope"].values()):
        raise ValueError("export_authoritative_split_not_train")
    reference = _index(_source(result, "dialogue_identity"), "root_id")
    if set(reference) != set(result["roots"]):
        raise ValueError("export_reference_root_coverage_mismatch")
    result["reference_identity"] = reference
    result["validated_payload_sha256"] = value_sha({key: result[key] for key in (
        "artifacts", "scope", "bank", "clusters", "records", "envelopes", "roots", "reference_identity",
    )})
    return result


def bind_snapshot(loaded, r03_manifest):
    """Render exported role choices against an explicit, appended R03 snapshot."""
    if loaded.get("validated_payload_sha256") != value_sha({key: loaded[key] for key in (
        "artifacts", "scope", "bank", "clusters", "records", "envelopes", "roots", "reference_identity",
    )}):
        raise ValueError("loaded_export_mutated_after_validation")
    manifest = _safe_path(r03_manifest)
    rows = read(manifest)
    if not isinstance(rows, list):
        raise ValueError("r03_manifest_must_be_explicit_rows")
    identities = _index(rows, "root_id")
    if not identities or not set(identities) <= set(loaded["roots"]):
        raise ValueError("new_r03_root_outside_export_reference_scope")
    dialogues = {}
    for root, identity in identities.items():
        if identity.get("status") != "FROZEN_NEW_R03" or identity.get("source_prefix") != "Parent+R01+R02-feedback+R03":
            raise ValueError("new_frozen_r03_required_r02_rejected")
        reference = loaded["reference_identity"][root]
        if reference.get("source_prefix") != "Parent+R01+R02-feedback":
            raise ValueError("export_reference_not_frozen_feedback_prefix")
        old = read(_checked(reference["input_dialogue_path"], reference["input_dialogue_file_sha256"]))
        if old.get("root_id") != root or old.get("complete_dialogue_sha256") != reference["complete_dialogue_sha256"]:
            raise ValueError("export_reference_dialogue_identity_mismatch")
        dialogue = read(_checked(identity["input_dialogue_path"], identity["input_dialogue_file_sha256"]))
        if dialogue.get("root_id") != root or dialogue.get("complete_dialogue_sha256") != identity["complete_dialogue_sha256"]:
            raise ValueError("new_r03_dialogue_identity_mismatch")
        if identity["input_dialogue_file_sha256"] == reference["input_dialogue_file_sha256"] or identity["complete_dialogue_sha256"] == reference["complete_dialogue_sha256"]:
            raise ValueError("new_frozen_r03_required_r02_rejected")
        turns, previous = dialogue.get("turns"), old.get("turns")
        if not isinstance(turns, list) or not previous or len(turns) <= len(previous) or turns[:len(previous)] != previous:
            raise ValueError("r03_feedback_prefix_not_preserved")
        refs = [turn.get("turn_ref") for turn in turns]
        if any(not isinstance(ref, str) or not ref for ref in refs) or len(refs) != len(set(refs)):
            raise ValueError("r03_turn_identity_missing_or_duplicate")
        if any("R03" not in turn["turn_ref"].split(":") for turn in turns[len(previous):]):
            raise ValueError("new_r03_appended_round_not_r03")
        for envelope in loaded["envelopes"].values():
            if envelope["root_id"] != root:
                continue
            for record in envelope["records"]:
                cid = record["source_candidate_id"]
                entry = loaded["bank"][cid]
                arm = record["source_arm"]
                if arm not in {"ROUND0_PARENT_PLUS_R01", "E1Q0_FEEDBACK_BLIND", "E1Q1_FEEDBACK_GUIDED"}:
                    raise ValueError("new_r03_source_branch_not_feedback")
                if arm != "ROUND0_PARENT_PLUS_R01" and entry.get("context_signature") not in {reference["input_dialogue_file_sha256"], reference["complete_dialogue_sha256"]}:
                    raise ValueError("new_r03_source_snapshot_branch_mismatch")
                forms = set(refs) | {ref[len(root) + 1:] for ref in refs if ref.startswith(root + ":")}
                if any(ref not in forms and f"{root}:{ref}" not in forms for ref in record["historical_qa"]["evidence_refs"]):
                    raise ValueError("new_r03_source_evidence_missing")
        dialogues[root] = dialogue
    schemas = {role: _source(loaded, f"schema_{role}") for role in ("targeted", "composition", "fresh")}
    strategy = _source(loaded, "strategy_schema")
    rendered = []
    for key, envelope in loaded["envelopes"].items():
        root, role = envelope["root_id"], envelope["role"]
        if root not in identities:
            continue
        output = render_role(role, envelope, dialogues[root], set(loaded["scope"]),
                             history=loaded["artifacts"]["fresh_historical_backend_view.jsonl"],
                             schema=schemas.get(role), strategy_schema=strategy if role == "targeted" else None)
        rendered.append({**output, "selection_key": key, "qa_snapshot_status": BOUND,
                         "r03_input_dialogue_file_sha256": identities[root]["input_dialogue_file_sha256"]})
    return {"status": BOUND, "filter_version": VERSION, "offline_only": True,
            "api_call_authorized": False, "external_calls": 0, "production_qa_integration": "NOT_CONFIRMED",
            "export_manifest": loaded["export_manifest"], "export_manifest_sha256": loaded["export_manifest_sha256"],
            "source_manifest_sha256": loaded["source_manifest_sha256"],
            "r03_manifest": str(manifest), "r03_manifest_sha256": file_sha(manifest),
            "bound_root_count": len(identities), "rendered": rendered, "snapshot_identity": rows}


def main():
    sys.dont_write_bytecode = True
    from .run import offline_guard
    from ..h4_round1_question_bank_feedback_compiler_v1.compiler import write_artifact

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--r03-manifest")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("output_must_be_new_directory")
    output.mkdir(parents=True)
    with offline_guard(output) as audit:
        loaded = load_export(args.export_manifest)
        if args.validate_only:
            result = {"status": "EXPLICIT_EXPORT_LOAD_VALIDATED_OFFLINE", "export_manifest": loaded["export_manifest"],
                      "export_manifest_sha256": loaded["export_manifest_sha256"], "candidate_scope": len(loaded["scope"]),
                      "active_records": len(loaded["records"]), "role_envelopes": len(loaded["envelopes"]),
                      "api_call_authorized": False}
            try:
                bind_snapshot(loaded, loaded["source_manifest"]["inputs"]["dialogue_identity"]["path"])
            except ValueError as error:
                if str(error) != "new_frozen_r03_required_r02_rejected":
                    raise
                result["old_r02_manifest_rejected"] = True
            else:
                raise ValueError("old_r02_manifest_incorrectly_accepted")
            write_artifact(output, "export_load_validation.json", result)
        else:
            result = bind_snapshot(loaded, args.r03_manifest)
            write_artifact(output, "bound_role_messages.jsonl", result.pop("rendered"))
            write_artifact(output, "snapshot_binding_report.json", result)
        write_artifact(output, "zero_external_calls.json", audit)
    print(canonical({"status": result["status"], "output": str(output), "external_calls": 0}))


if __name__ == "__main__":
    main()
