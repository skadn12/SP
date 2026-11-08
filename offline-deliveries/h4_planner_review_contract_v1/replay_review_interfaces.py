"""Replay only frozen review interfaces; no generated outputs, API or truth labels."""
from pathlib import Path
import sys
import hashlib
import json
from copy import deepcopy
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
WORK = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-planner-review-contract-v1')
sys.path.insert(0, str(WORK))
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import scoped_gate, scoped_outline

SOURCE = BASE.parent / 'h4_planner_codex_feedback_v1/prepared_run'
ROOTS = ['cc-train-episode-259192', 'cc-train-episode-96462',
         'cc-train-episode-112812', 'cc-train-episode-218519']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def replay():
    results = []
    for root in ROOTS:
        src = SOURCE / 'expansion/roots' / root
        for suffix in ('', '_supplement'):
            context = read(src / f'semantic_gate/context{suffix}.json')
            catalog = context['catalog']
            validation_path = src / ('planner/supplement_validation.json' if suffix else 'planner/validation.json')
            plan = read(validation_path)['compiled_plan']
            original_path = src / f'semantic_gate/output{suffix}.json'
            original = read(original_path)
            before = deepcopy(original)
            old = read(src / f'semantic_gate/validation{suffix}.json')
            result = scoped_gate.validate_gate(original, plan=plan, catalog=catalog,
                expected_root_id=root, expected_catalog_sha256=catalog['sha256'])
            assert original == before
            assert result['valid']
            target = BASE / 'offline_replay' / root / ('supplement' if suffix else 'primary')
            write(target / 'gate_interface_audit.json', result)
            prompt = scoped_outline.outline_prompt(plan=plan, catalog=catalog, gate_validation=result)
            scoped_outline.validate_dispatch(prompt)
            body = json.loads(prompt['prompt'].split('\nFROZEN INPUT\n', 1)[1])
            assert body['evidence_catalog'] == catalog
            for prior in body['prior_gate_semantic_judgments_not_facts']:
                assert 'reason' not in prior and 'historical_support_disputed' not in prior
            write(target / 'outline_request_preview.json', prompt)
            aliases = [x for x in result['reference_resolution']['bindings']
                       if x['match_kind'] == 'EXACT_TRANSCRIPT_TURN_REF']
            results.append({'root_id': root, 'stage': 'gate_supplement' if suffix else 'gate',
                'old_quarantined': old['quarantined_candidate_refs'],
                'new_quarantined': result['quarantined_candidate_refs'],
                'old_eligible': old['passed_candidate_refs'],
                'new_eligible': result['passed_candidate_refs'],
                'alias_binding_count': len(aliases), 'alias_bindings': aliases,
                'raw_unchanged': True, 'original_response': str(original_path),
                'original_response_sha256': sha(original_path),
                'gate_reason_forwarded_to_outline': False,
                'catalog_identity_and_all_source_text_preserved': True,
                'semantic_correctness': 'NOT_REEVALUATED_INTERFACE_REPLAY_ONLY'})
    fixed = next(x for x in results if x['root_id'].endswith('-218519') and x['stage'] == 'gate_supplement')
    assert len(fixed['old_quarantined']) == 3 and fixed['new_quarantined'] == []
    assert fixed['alias_binding_count'] == 6
    # Preserve previous stage's request and code snapshots rather than rehashing
    # them in place after edits. Only compare their already frozen records.
    freezes = []
    for path in [SOURCE/'freeze_manifest.json', SOURCE.parent/'inputs/manifest.json',
                 BASE.parent/'h4_planner_method_v2/prepared_run/freeze_manifest.json']:
        files = read(path)['files']
        changed = [x['path'] for x in files if not Path(x['path']).is_file() or sha(Path(x['path'])) != x['sha256']]
        assert not changed, changed
        freezes.append({'manifest':str(path),'count':len(files),'changed':changed})
    report = {'status':'INTERFACE_REPLAY_PASS', 'experimental_api_calls':0,
              'scope':'Eight frozen Gate responses and corresponding source catalogs; no model generation or semantic certification.',
              'new_outline_outputs':0, 'new_qualified_scene_count':'NOT_EVALUATED',
              'new_qa_count':0, 'results':results, 'historical_freezes':freezes,
              'limitations':['Alias acceptance does not demonstrate evidence support.',
                             'Old Outline judgments are not migrated to the new scope contract.',
                             '218519 may still lack three compatible useful scenes; no semantic verdict was overridden.']}
    write(BASE/'reports/INTERFACE_REPLAY.json', report)
    print(json.dumps({'status':report['status'],'gate_responses':len(results),
                      '218519_alias_bindings':fixed['alias_binding_count'],
                      '218519_old_quarantine_count':len(fixed['old_quarantined']),
                      '218519_new_quarantine_count':len(fixed['new_quarantined']),
                      'experimental_api_calls':0}, ensure_ascii=False))


if __name__ == '__main__':
    with patch('socket.socket.connect', side_effect=AssertionError('offline only')), \
            patch('socket.socket.connect_ex', side_effect=AssertionError('offline only')):
        replay()
