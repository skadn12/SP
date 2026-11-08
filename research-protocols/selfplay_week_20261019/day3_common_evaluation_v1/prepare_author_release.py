#!/usr/bin/env python3
"""Prepare author-safe exclusions without opening any bound original input.

Reads only local protocol policy and frozen boundary artifacts.  The release
allowlist is exactly EXCLUSION_MANIFEST.json and EXCLUSION_INDEX.jsonl.
INTERNAL_RELEASE_VALIDATION.json and this program are internal, not release inputs.
"""
import argparse
import collections
import copy
import hashlib
import json
import pathlib
import re
import sys

EXPECTED_COUNT = 4629
EXPECTED_INPUT_COUNT = 671
FIELDS = [
    'exclusion_record_id', 'root_id', 'original_split', 'source_sha256',
    'canonical_fact_family_id', 'semantic_fact_fingerprints', 'related_family_ids',
    'evidence_turn_refs', 'scope', 'exposure_role', 'exclusion_manifest_sha256',
]
SEMANTIC_FIELDS = [k for k in FIELDS if k not in {'exposure_role', 'exclusion_manifest_sha256'}]
SCOPE_FIELDS = {
    'all_options_evidence_coverage', 'all_source_turns_excluded', 'comparison_unit',
    'evidence_mapping_status', 'mapping_limitation_interpretation', 'question_type',
    'semantic_near_duplicate_clearance', 'source_item_count', 'uncertainty_reasons',
    'unordered_atomic_event_set',
}
FACT_FIELDS = {
    'action', 'actor', 'canonical_proposition', 'condition', 'entities',
    'fingerprint_id', 'modality', 'object', 'polarity', 'projection_method',
    'semantic_normalization_status', 'time',
}
FORBIDDEN_KEYS = {
    'gold', 'answer', 'correct', 'correctness', 'score', 'prediction', 'question',
    'options', 'label', 'weakness', 'feedback', 'rationale', 'verdict', 'relation',
    'candidate_id', 'outcome', 'outcomes', 'predicted_answer', 'model_answer',
    'is_correct', 'error_card', 'feedback_card',
}
PATH_RE = re.compile(r"(?:/(?:root|home|tmp|mnt|workspace|autodl-tmp)/|[A-Za-z]:[\\/]|file://|\\\\[^\\]+\\|(?:^|[\s\"'])\.\.?[\\/])", re.I)
# These terms may occur in source propositions; metadata is audited separately.
PERFORMANCE_RE = re.compile(r'\b(?:gold|scores?|correctness|predictions?|outcomes?|feedback|failures?|failed)\b', re.I)
OUTCOME_CODED_RE = re.compile(r'(?:frontier|legacy_reuse|all_failed|repair_v|feedback_id|diagnostic_feedback|sealed_new_train_private)', re.I)
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
SAFE_MANIFEST_KEYS = {
    'schema_version', 'status', 'fields', 'record_count', 'input_binding_count',
    'index_payload_sha256_without_manifest_field', 'input_bindings', 'coverage',
    'digest_convention', 'record_metadata_policy', 'semantic_clearance',
    'semantic_pass_claimed', 'whole_root_exposure_status',
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def blob(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def pretty(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read_json(path):
    return json.loads(path.read_bytes())

def payload(rows):
    return [{k: v for k, v in row.items() if k != 'exclusion_manifest_sha256'} for row in rows]

def coarse_role(row):
    return ['training_exposure' if row['original_split'] == 'train' else 'historical_exposure']

def strings(obj, location=()):
    if isinstance(obj, dict):
        for key, val in obj.items():
            yield from strings(val, location + (key,))
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            yield from strings(val, location + (i,))
    elif isinstance(obj, str):
        yield location, obj

def audit_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            require(k.lower() not in FORBIDDEN_KEYS, 'Forbidden key in author record: ' + k)
            require(not re.search(r'(?:path|filename|directory|uri|url)$', k, re.I), 'Path-like key in author record: ' + k)
            audit_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            audit_keys(v)

def source_text_location(loc):
    return (len(loc) > 2 and loc[0] == 'semantic_fact_fingerprints'
            and loc[2] in {'actor', 'canonical_proposition', 'condition', 'entities', 'object', 'time'}) \
        or loc[:2] == ('scope', 'unordered_atomic_event_set')

def audit_rows(rows):
    source_word_hits = collections.Counter()
    for n, row in enumerate(rows, 1):
        require(set(row) == set(FIELDS), f'Index row {n} violates exact top-level whitelist')
        require(isinstance(row['scope'], dict) and set(row['scope']) == SCOPE_FIELDS, f'Unexpected scope schema at row {n}')
        require(isinstance(row['semantic_fact_fingerprints'], list), f'Unexpected fact schema at row {n}')
        for fact in row['semantic_fact_fingerprints']:
            require(isinstance(fact, dict) and set(fact) == FACT_FIELDS, f'Unexpected nested fact schema at row {n}')
        audit_keys(row)
        for loc, value in strings(row):
            require(not PATH_RE.search(value), f'Path value at row {n}, field {loc}')
            if PERFORMANCE_RE.search(value):
                require(source_text_location(loc), f'Performance term outside immutable source text at row {n}, field {loc}')
                source_word_hits['.'.join(str(p) for p in loc if not isinstance(p, int))] += 1
            require(not OUTCOME_CODED_RE.search(value), f'Outcome-coded value at row {n}, field {loc}')
    return dict(sorted(source_word_hits.items()))

def safe_inputs(raw_bindings):
    # Raw path order and raw roles never determine either IDs or public order.
    # Identical digest/size occurrences are represented separately using a local
    # multiplicity counter, then cryptographically named and sorted by opaque ID.
    multiplicities = collections.Counter()
    for binding in raw_bindings:
        require(set(binding) == {'path', 'sha256', 'bytes', 'roles'}, 'Unexpected raw binding schema')
        require(SHA_RE.fullmatch(binding['sha256']) is not None, 'Invalid input digest')
        require(isinstance(binding['bytes'], int) and not isinstance(binding['bytes'], bool) and binding['bytes'] >= 0, 'Invalid input byte count')
        multiplicities[(binding['sha256'], binding['bytes'])] += 1
    result = []
    for (digest, size), count in sorted(multiplicities.items()):
        for occurrence in range(count):
            input_id = 'INPUT-' + sha(blob(['author-release-input-v1', digest, size, occurrence]))
            result.append({'input_id': input_id, 'sha256': digest, 'bytes': size, 'role': 'sealed_input'})
    return sorted(result, key=lambda item: item['input_id'])

def validate_manifest(manifest, raw_manifest, rows):
    require(set(manifest) == SAFE_MANIFEST_KEYS, 'Unexpected safe manifest keys')
    require(manifest['fields'] == FIELDS, 'Safe manifest field list changed')
    require(manifest['record_count'] == len(rows) == EXPECTED_COUNT, 'Safe manifest count mismatch')
    require(manifest['input_binding_count'] == EXPECTED_INPUT_COUNT, 'Safe input count mismatch')
    require(manifest['semantic_clearance'] == 'PENDING' and manifest['semantic_pass_claimed'] is False, 'Semantic clearance must remain pending')
    expected_inputs = safe_inputs(raw_manifest['input_bindings'])
    require(manifest['input_bindings'] == expected_inputs, 'Safe bindings differ from path-independent projection')
    for b in manifest['input_bindings']:
        require(set(b) == {'input_id', 'sha256', 'bytes', 'role'}, 'Safe binding whitelist violated')
        require(re.fullmatch(r'INPUT-[0-9a-f]{64}', b['input_id']) is not None, 'Nonopaque input ID')
        require(b['role'] == 'sealed_input', 'Fine-grained input role leaked')
    for loc, value in strings(manifest):
        require(not PATH_RE.search(value), f'Manifest path leak at {loc}')
        require(not PERFORMANCE_RE.search(value), f'Manifest performance/feedback term at {loc}')
        require(not OUTCOME_CODED_RE.search(value), f'Manifest coded role at {loc}')
    raw_paths = [b['path'] for b in raw_manifest['input_bindings']]
    serialized = json.dumps(manifest, ensure_ascii=False)
    require(not any(p in serialized for p in raw_paths), 'Raw binding path found in manifest')
    require(manifest['index_payload_sha256_without_manifest_field'] == sha(blob(payload(rows))), 'Manifest payload digest mismatch')

def execute(base, check_only):
    boundary = base / 'boundary'
    release = base / 'author_release'
    policy_path = base / 'protocol' / 'EXPOSURE_AND_FAMILY_INTERPRETATION.json'
    policy = read_json(policy_path)
    require(policy['author_release_fields'] == FIELDS, 'Protocol author whitelist differs from locked release schema')
    # This snapshot enumerates and hashes local frozen boundary files only.
    # No path stored inside a boundary JSON object is ever opened.
    frozen_before = {str(p.relative_to(base)): sha(p.read_bytes()) for p in sorted(boundary.rglob('*')) if p.is_file()}
    manifest_bytes = (boundary / 'EXCLUSION_MANIFEST.json').read_bytes()
    index_bytes = (boundary / 'EXCLUSION_INDEX.jsonl').read_bytes()
    raw_manifest = json.loads(manifest_bytes)
    raw_rows = [json.loads(line) for line in index_bytes.splitlines() if line.strip()]
    raw_manifest_hash = sha(manifest_bytes)
    require(raw_manifest['fields'] == FIELDS, 'Raw manifest whitelist mismatch')
    require(raw_manifest['record_count'] == len(raw_rows) == EXPECTED_COUNT, 'Raw record count mismatch')
    require(len(raw_manifest['input_bindings']) == EXPECTED_INPUT_COUNT, 'Raw input binding count mismatch')
    violations = []
    for i, row in enumerate(raw_rows, 1):
        if set(row) != set(FIELDS):
            violations.append({'line': i, 'extra_keys': sorted(set(row) - set(FIELDS)), 'missing_keys': sorted(set(FIELDS) - set(row))})
    require(not violations, 'Raw index has non-whitelisted or missing fields; refusing release')
    require(all(r['exclusion_manifest_sha256'] == raw_manifest_hash for r in raw_rows), 'Raw manifest binding mismatch')
    require(raw_manifest['index_payload_sha256_without_manifest_field'] == sha(blob(payload(raw_rows))), 'Raw payload convention mismatch')
    raw_output = read_json(boundary / 'OUTPUT_BINDINGS.json')
    require(raw_output['manifest']['sha256'] == raw_manifest_hash, 'Raw manifest output binding mismatch')
    require(raw_output['index']['sha256'] == sha(index_bytes), 'Raw index output binding mismatch')
    rows = copy.deepcopy(raw_rows)
    for row in rows:
        row['exposure_role'] = coarse_role(row)
        row.pop('exclusion_manifest_sha256')
    safe_manifest = {
        'schema_version': 'author_safe_exclusion_manifest_v1',
        'status': 'SAFE_EXPORT_PREPARED_SEMANTIC_CLEARANCE_PENDING',
        'fields': FIELDS,
        'record_count': len(rows),
        'input_binding_count': len(raw_manifest['input_bindings']),
        'index_payload_sha256_without_manifest_field': sha(blob(rows)),
        'input_bindings': safe_inputs(raw_manifest['input_bindings']),
        'coverage': copy.deepcopy(raw_manifest['coverage']),
        'digest_convention': {
            'index_payload': 'SHA256 of UTF-8 JSON array of records after removing exclusion_manifest_sha256; sort_keys=true, ensure_ascii=false, separators=comma-colon',
            'manifest': 'SHA256 of complete standalone manifest file bytes, including final LF; referenced by every index row',
            'index_file': 'SHA256 of complete final JSONL file bytes; recorded only in internal validation',
            'cycle_avoidance': 'Manifest contains only index payload digest excluding the manifest field; final index file digest is absent from manifest',
        },
        'record_metadata_policy': {
            'exposure_role': 'training_exposure when original_split equals train; historical_exposure otherwise',
            'semantic_fields': 'All other record fields are copied exactly except exclusion_manifest_sha256 rebinding',
            'comparison_scope': 'Compare every relevant record in all five required family checks; coarse roles never waive a check',
            'input_id': 'Opaque domain-separated digest of input content digest, byte count and multiplicity; independent of original locations and roles',
        },
        'semantic_clearance': 'PENDING',
        'semantic_pass_claimed': False,
        'whole_root_exposure_status': 'REGRESSION_ONLY; root identity and source digest inequality do not establish distinct fact families',
    }
    safe_manifest_bytes = pretty(safe_manifest)
    safe_manifest_hash = sha(safe_manifest_bytes)
    for row in rows:
        row['exclusion_manifest_sha256'] = safe_manifest_hash
    safe_index_bytes = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows).encode('utf-8')
    source_word_hits = audit_rows(rows)
    validate_manifest(safe_manifest, raw_manifest, rows)
    semantic_equal = all(all(raw[k] == safe[k] for k in SEMANTIC_FIELDS) for raw, safe in zip(raw_rows, rows))
    require(semantic_equal, 'Immutable index semantic fields changed')
    require(len({r['exclusion_record_id'] for r in rows}) == EXPECTED_COUNT, 'Duplicate exclusion record IDs')
    require(all(r['exposure_role'] == coarse_role(r) for r in rows), 'Coarse role mismatch')
    safe_serialized = safe_manifest_bytes + safe_index_bytes
    require(not any(b['path'].encode('utf-8') in safe_serialized for b in raw_manifest['input_bindings']), 'Original input path leaked')
    if check_only:
        require((release / 'EXCLUSION_MANIFEST.json').read_bytes() == safe_manifest_bytes, 'On-disk safe manifest differs from reproducible build')
        require((release / 'EXCLUSION_INDEX.jsonl').read_bytes() == safe_index_bytes, 'On-disk safe index differs from reproducible build')
    else:
        release.mkdir(exist_ok=True)
        (release / 'EXCLUSION_MANIFEST.json').write_bytes(safe_manifest_bytes)
        (release / 'EXCLUSION_INDEX.jsonl').write_bytes(safe_index_bytes)
    frozen_after = {str(p.relative_to(base)): sha(p.read_bytes()) for p in sorted(boundary.rglob('*')) if p.is_file()}
    require(frozen_before == frozen_after, 'Frozen boundary files changed')
    # Re-read both written payloads and validate their actual byte bindings.
    disk_manifest_bytes = (release / 'EXCLUSION_MANIFEST.json').read_bytes()
    disk_index_bytes = (release / 'EXCLUSION_INDEX.jsonl').read_bytes()
    disk_manifest = json.loads(disk_manifest_bytes)
    disk_rows = [json.loads(line) for line in disk_index_bytes.splitlines() if line.strip()]
    require(disk_manifest_bytes == safe_manifest_bytes and disk_index_bytes == safe_index_bytes, 'Output write/read mismatch')
    require(all(r['exclusion_manifest_sha256'] == sha(disk_manifest_bytes) for r in disk_rows), 'Written index binding mismatch')
    audit_rows(disk_rows)
    validate_manifest(disk_manifest, raw_manifest, disk_rows)
    validation = {
        'schema_version': 'internal_author_release_validation_v1',
        'status': 'PASS_EXPORT_SANITIZATION_AND_INTEGRITY_ONLY',
        'dispatch_status': 'NOT_DISPATCHED',
        'author_release_allowlist': ['EXCLUSION_INDEX.jsonl', 'EXCLUSION_MANIFEST.json'],
        'internal_not_for_authors': ['INTERNAL_RELEASE_VALIDATION.json', 'RELEASE_NOTES_ZH.md'],
        'record_count': len(disk_rows),
        'input_binding_count': len(disk_manifest['input_bindings']),
        'raw_index_whitelist_violations': violations,
        'raw_index_nonwhitelist_policy': 'Reject release on any missing, extra or forbidden nested field; no silent drops',
        'immutable_semantic_fields': SEMANTIC_FIELDS,
        'immutable_semantic_fields_equal_all_rows': semantic_equal,
        'exposure_role_normalization': 'original_split=train -> training_exposure; all others -> historical_exposure',
        'exposure_role_changes': sum(a['exposure_role'] != b['exposure_role'] for a, b in zip(raw_rows, rows)),
        'safe_role_counts': dict(sorted(collections.Counter(r['exposure_role'][0] for r in rows).items())),
        'original_role_mapping_retained_only_in_frozen_boundary': True,
        'all_five_semantic_family_checks_still_required': True,
        'semantic_clearance': 'PENDING',
        'semantic_pass_claimed': False,
        'internal_path_leaks': 0,
        'forbidden_key_leaks': 0,
        'performance_or_selection_metadata_leaks': 0,
        'input_id_order_and_role_independent': True,
        'immutable_source_text_term_occurrences': source_word_hits,
        'source_text_term_interpretation': 'Source-grounded ordinary-language words are preserved in immutable source facts/events; word occurrences are not evaluation metadata. No source originals were opened for this release.',
        'input_binding_targets_opened': 0,
        'original_input_content_hashes_reverified': False,
        'original_input_reverification_reason': 'Forbidden in this task: only sealed local boundary metadata was read; binding targets were not opened.',
        'raw_bindings_unchanged': True,
        'frozen_boundary_file_hashes_before': frozen_before,
        'frozen_boundary_file_hashes_after': frozen_after,
        'raw_output_bindings_sha256': frozen_before['boundary/OUTPUT_BINDINGS.json'],
        'raw_input_bindings_sha256': frozen_before['boundary/INPUT_BINDINGS.json'],
        'raw_manifest_sha256': raw_manifest_hash,
        'raw_index_sha256': sha(index_bytes),
        'raw_payload_sha256_without_manifest_field': raw_manifest['index_payload_sha256_without_manifest_field'],
        'outputs': {
            'EXCLUSION_MANIFEST.json': {'sha256': sha(disk_manifest_bytes), 'bytes': len(disk_manifest_bytes)},
            'EXCLUSION_INDEX.jsonl': {'sha256': sha(disk_index_bytes), 'bytes': len(disk_index_bytes)},
        },
        'safe_payload_sha256_without_manifest_field': disk_manifest['index_payload_sha256_without_manifest_field'],
        'prepare_script_sha256': sha(pathlib.Path(__file__).read_bytes()),
        'reproducibility': 'python prepare_author_release.py; verify without writes via python prepare_author_release.py --check-only',
        'new_solver_calls': 0,
        'new_training_runs': 0,
    }
    if check_only:
        prior = read_json(release / 'INTERNAL_RELEASE_VALIDATION.json')
        require(prior == validation, 'Internal validation differs from deterministic verification')
    else:
        (release / 'INTERNAL_RELEASE_VALIDATION.json').write_bytes(pretty(validation))
    print(json.dumps({
        'status': validation['status'], 'check_only': check_only,
        'records': len(disk_rows), 'inputs': len(disk_manifest['input_bindings']),
        'semantic_clearance': 'PENDING', 'raw_bindings_unchanged': True,
        'outputs': validation['outputs'],
        'internal_validation_sha256': sha((release / 'INTERNAL_RELEASE_VALIDATION.json').read_bytes()),
    }, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='reconstruct in memory and compare; no writes')
    args = parser.parse_args()
    try:
        execute(pathlib.Path(__file__).resolve().parent, args.check_only)
    except Exception as exc:
        print(json.dumps({'status': 'REJECTED_AUTHOR_RELEASE', 'reason': str(exc), 'semantic_clearance': 'PENDING'}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
