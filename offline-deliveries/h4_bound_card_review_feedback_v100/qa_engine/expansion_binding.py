"""Recheck exact materialized v98 source lineage; this makes no READY judgment."""
from pathlib import Path
import json

import batch_scope as scope

VERSION = 'v98_materialized_expansion_provenance_v1'


def validate(root, source_binding, provenance_path):
    scope.need(root in scope.selected_roots(), 'unselected_materialized_root')
    run = scope.upstream_run()
    path = Path(provenance_path).resolve()
    record = json.loads(path.read_text())
    scope.need(record.get('version') == VERSION and record.get('root_id') == root
               and Path(record.get('run', '')).resolve() == run.resolve(),
               'materialized_provenance_identity')
    scope.need(record.get('output_source') == record.get('source') == source_binding
               and scope.bind(source_binding['path']) == source_binding,
               'materialized_source_binding_changed')
    upstream_source = run / 'sources' / (root + '.json')
    output = run / 'roots' / root / 'B/cumulative_dialogue.json'
    result_path = output.parent / 'RESULT.json'
    result = json.loads(result_path.read_text())
    before, after = json.loads(upstream_source.read_text()), json.loads(output.read_text())
    source = Path(source_binding['path'])
    scope.need(record.get('input_source') == scope.bind(upstream_source)
               and record.get('accepted_output') == scope.bind(output)
               and source.read_bytes() == output.read_bytes(),
               'materialized_source_not_exact_accepted_output')
    scope.need(before['root_id'] == after['root_id'] == root
               and after['turns'][:len(before['turns'])] == before['turns'],
               'materialized_prefix_changed')
    suffix = after['turns'][len(before['turns']):]
    scope.need(result['accepted_scenes'] == record.get('accepted_scene_count') == 1
               and len(suffix) == result['added_turns'] == record.get('new_turn_count')
               and record.get('new_turn_refs') == [t['turn_ref'] for t in suffix]
               and len(suffix) > 0 and all(t['turn_ref'].startswith('R02:B:S01:') for t in suffix)
               and record.get('historical_prefix_preserved') is True
               and record.get('source_consistency_decision') is None
               and record.get('question_answer_certification') is False
               and record.get('training_allowed') is False
               and record.get('historical_content_fallback') is False,
               'materialized_accepted_suffix_contract')
    scope.need(record.get('structural_audit') == scope.bind(run / 'FINAL_AUDIT.json'),
               'materialized_structural_audit_changed')
    bindings = record.get('bound_files')
    scope.need(isinstance(bindings, list) and bool(bindings), 'materialized_bindings_required')
    by_path = {b['path']: b for b in bindings}
    scope.need(len(by_path) == len(bindings), 'materialized_duplicate_binding')
    required = [upstream_source, output, result_path, source, run / 'MANIFEST.json',
                run / 'REPORT.json', run / 'FINAL_AUDIT.json',
                run / 'SOURCE_RESERVATIONS.json',
                run / 'UNIQUE_EVENT_ALLOCATION.json']
    manifest = json.loads((run / 'MANIFEST.json').read_text())
    required.extend(Path(binding['path']) for binding in manifest['files'])
    required.extend(output.parent.glob('*.json'))
    allocation = json.loads((run / 'UNIQUE_EVENT_ALLOCATION.json').read_text())
    source_ids = {item['source_id'] for item in allocation['assignments'][root]}
    for directory in (run / 'requests').iterdir():
        proof_path = directory / 'PROOF.json'
        if proof_path.is_file():
            proof = json.loads(proof_path.read_text())
            if proof.get('root_id') == root or proof.get('source_id') in source_ids:
                required.extend(directory.glob('*.json'))
    scope.need({str(p.resolve()) for p in required} <= set(by_path),
               'materialized_binding_coverage')
    for binding in bindings:
        scope.need(scope.bind(binding['path']) == binding,
                   'materialized_bound_file_changed:' + binding['path'])
    return record
