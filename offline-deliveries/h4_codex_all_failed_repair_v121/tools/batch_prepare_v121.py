"""Mechanical staging for the fixed v121 cohort; no semantic judgments.

Source and prefix decisions are supplied by the assigned actual readers. This
helper only binds their records, preserves history, and computes denominators.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil

B = Path(__file__).resolve().parents[1]
SCENES = [f'R02:{r}:S{i:02d}' for r in 'AB' for i in (1, 2, 3)]


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
def need(ok, message):
    if not ok: raise ValueError(message)
def check(row):
    need(bind(row['path']) == {k: row[k] for k in ('path', 'sha256')}, 'BOUND_FILE_CHANGED:' + row['path'])
def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')
def scene_dir(unit, root, scene):
    return Path(unit) / 'roots' / ('root' + root.rsplit('-', 1)[1]) / 'scenes' / scene.removeprefix('R02:').replace(':', '_')


def finalized(scope):
    result = []
    for row in scope['roots']:
        unit = Path(row['unit_path'])
        fp = unit / 'source_repair/FINALIZE_RECEIPT.json'
        need(fp.is_file(), 'SOURCE_NOT_FINALIZED:' + row['root_id'])
        final = read(fp)
        need(final['status'] in ('SOURCE_READY', 'SOURCE_HOLD'), 'SOURCE_TERMINAL_REQUIRED')
        check(final['source_output_binding'])
        output = read(final['source_output_binding']['path'])
        need(output['root_id'] == row['root_id'], 'SOURCE_OUTPUT_ROOT')
        check(output['baseline_binding'])
        expected_base = unit / ('inputs/BASELINE.json' if final['status'] == 'SOURCE_READY' else 'source_repair/CANDIDATE_BASELINE.json')
        need(output['baseline_binding'] == bind(expected_base), 'SOURCE_OUTPUT_BASELINE_PATH_OR_HASH')
        need(output['review_bindings'] == final['review_bindings'] and len(final['review_bindings']) == 2,
             'SOURCE_FINAL_REVIEW_BINDINGS')
        for binding in final['review_bindings']: check(binding)
        result.append((row, final, fp))
    return result


def prepare_prefix():
    need(not (B / 'PREFIX_INPUTS.json').exists(), 'PREFIX_ALREADY_PREPARED')
    scope = read(B / 'REPAIR_SCOPE.json')
    rows = []
    for row, final, fp in finalized(scope):
        if final['status'] != 'SOURCE_READY': continue
        unit, old = Path(row['unit_path']), Path(row['old_unit_path'])
        base, cp = unit / 'inputs/BASELINE.json', unit / 'inputs/CONTINUATION_CONSTRAINTS.json'
        retained = row['proposed_retained_scene_refs']
        need(retained == SCENES[:len(retained)], 'NONCONTIGUOUS_RETAINED_PREFIX')
        candidate = scene_dir(old, row['root_id'], retained[-1]) / 'CUMULATIVE_SOURCE.json' if retained else base
        if retained:
            need(base.read_bytes() == (old / 'inputs/BASELINE.json').read_bytes(), 'EDITED_SOURCE_CANNOT_IMPORT_OLD_PREFIX')
        if row['a_plan_mode'] == 'INHERIT_A':
            selection = old / 'selection.json'
            policy = {'root_id': row['root_id'], 'plan_mode': 'INHERIT_A',
                      'historical_selection_binding': bind(selection), 'a_continuations': read(selection)['a_continuations']}
        else:
            policy = {'root_id': row['root_id'], 'plan_mode': 'NEW_SELECTION', 'a_continuations': [],
                      'selection_rule': 'Choose three source-evidence-bound continuation plans during actual frozen matching; align numbered R01 scenes where all three exist.'}
        save(unit / 'A_PLAN_POLICY.json', policy)
        save(unit / 'CONSTRAINTS_PATH_ADAPTATION.json', {
            'actual_executor': '/root', 'created_at_utc': now(),
            'original_binding': bind(old / 'inputs/CONTINUATION_CONSTRAINTS.json'),
            'derived_binding': bind(cp), 'adaptation_kind': 'NEW_DUAL_REVIEW_SOURCE_GATE_WITH_EXPLICIT_HISTORICAL_UNIT_LOCATOR',
            'continuation_constraints_verbatim': False, 'source_finalize_binding': bind(fp)})
        inputs = [{'label': label, **bind(path)} for label, path in [
            ('baseline', base), ('constraints', cp), ('selection', unit / 'A_PLAN_POLICY.json'), ('candidate_prefix', candidate)]]
        if any(':B:' in s for s in retained):
            reservations = read(old / 'SOURCE_RESERVATIONS.json')['allocation']
            for scene in retained:
                if ':B:' not in scene: continue
                matches = [r for r in reservations if r['root_id'] == row['root_id'] and r['scene_ref'] == scene]
                need(len(matches) == 1, 'MISSING_OLD_NATIVE_ALLOCATION')
                item = matches[0]
                check({'path': item['source_item_path'], 'sha256': item['source_item_sha256']})
                inputs.append({'label': 'retained_native_' + scene, **bind(item['source_item_path'])})
        rows.append({'root_id': row['root_id'], 'retained_scene_refs': retained,
                     'proposed_retry_start_scene': row['proposed_retry_start'],
                     'candidate_prefix_binding': bind(candidate), 'inputs': inputs})
    save(B / 'PREFIX_INPUTS.json', {'schema_version': 'v121_prefix_inputs_v1', 'created_at_utc': now(),
        'role_assignments_binding': bind(B / 'ROLE_ASSIGNMENTS.json'), 'repair_scope_binding': bind(B / 'REPAIR_SCOPE.json'),
        'source_protocol_binding': bind(B / 'SOURCE_PROTOCOL_FREEZE.json'), 'roots': rows,
        'source_first_review': True, 'allowed_inputs_only': True})
    paths = [B / 'ROLE_ASSIGNMENTS.json', B / 'PREFIX_INPUTS.json', B / 'tools/prefix_review_v2.py',
             B / 'tools/batch_prepare_v121.py', B / 'REPAIR_SCOPE.json']
    for row in rows: paths += [Path(r['path']) for r in row['inputs']]
    paths = list(dict.fromkeys(paths))
    save(B / 'PREFIX_PROTOCOL_FREEZE.json', {'status': 'FROZEN_BEFORE_PREFIX_READS', 'created_at_utc': now(),
        'actual_executor': '/root', 'bindings': [bind(p) for p in paths],
        'scope': 'Assigned complete source-prefix judgment only; no generation release.',
        'semantic_criteria': 'Existing content contract, complete approved source, all current continuation constraints, inherited or explicitly new A-plan policy, and retained complete native events. Prior source judgments means historical prefix/scene judgments; baseline gates remain declared constraints.'})
    print(json.dumps({'status': 'PREFIX_INPUTS_FROZEN', 'source_ready': len(rows), 'source_hold': 24 - len(rows)}))


def import_prefix():
    need(not (B / 'MANIFEST.json').exists(), 'MANIFEST_ALREADY_EXISTS')
    scope, prefix = read(B / 'REPAIR_SCOPE.json'), read(B / 'PREFIX_INPUTS.json')
    for binding in read(B / 'PREFIX_PROTOCOL_FREEZE.json')['bindings']: check(binding)
    roles = {r['root_id']: r for r in read(B / 'ROLE_ASSIGNMENTS.json')['roots']}
    candidates = {r['root_id']: r for r in prefix['roots']}
    rows = []
    for row, final, fp in finalized(scope):
        root, unit, old = row['root_id'], Path(row['unit_path']), Path(row['old_unit_path'])
        out = {'root_id': root, 'source_ready': final['status'] == 'SOURCE_READY', 'source_status': final['source_status'],
               'source_finalize_binding': bind(fp), 'prefix_reuse_approved': False,
               'retry_start': row['proposed_retry_start'], 'retained_scene_refs': row['proposed_retained_scene_refs'],
               'remaining_scene_refs': row['proposed_new_scene_refs'], 'a_plan_mode': row['a_plan_mode'],
               'new_b_scene_refs': [s for s in row['proposed_new_scene_refs'] if ':B:' in s], 'unit_path': str(unit)}
        if not out['source_ready']:
            rows.append(out); continue
        candidate = candidates[root]
        auditp = B / 'prefix_audits' / unit.name / 'PREFIX_SOURCE_REVIEW.json'
        audit = read(auditp)
        need(audit['actual_executor'] == roles[root]['prefix_reviewer'] != roles[root]['author'], 'PREFIX_REVIEW_ROLE')
        need(audit['candidate_prefix_binding'] == candidate['candidate_prefix_binding'], 'PREFIX_REVIEW_INPUT')
        need(audit['retained_scene_refs'] == row['proposed_retained_scene_refs'], 'PREFIX_REVIEW_SCOPE')
        for item in audit['input_bindings']: check(item)
        for key in ['read_context_binding', 'input_render_binding', 'raw_judgment_binding', 'candidate_prefix_binding']: check(audit[key])
        need(audit['judgment'] == read(audit['raw_judgment_binding']['path']), 'PREFIX_RAW_JUDGMENT_MISMATCH')
        receiptp = auditp.parent / 'SAVE_RECEIPT.json'; receipt = read(receiptp)
        need(receipt['actual_executor'] == roles[root]['prefix_reviewer'] and receipt['output_binding'] == bind(auditp), 'PREFIX_SAVE_RECEIPT')
        out['prefix_review_binding'] = bind(auditp)
        out['prefix_reuse_approved'] = audit['content_reuse_approved']
        need(out['prefix_reuse_approved'] == (audit['judgment']['decision'] == 'PASS'), 'PREFIX_DECISION')
        if not out['prefix_reuse_approved']:
            rows.append(out); continue
        imported = []
        previous = read(unit / 'inputs/BASELINE.json')['turns']
        for scene in row['proposed_retained_scene_refs']:
            historical, target = scene_dir(old, root, scene), scene_dir(unit, root, scene)
            draft, source, review = [read(historical / n) for n in ['EXPANSION_DRAFT.json', 'CUMULATIVE_SOURCE.json', 'SOURCE_REVIEW.json']]
            need(source['turns'] == previous + draft['turns'], 'IMPORT_NOT_APPEND_ONLY')
            need(review['decision'] == 'READY' and review['source_sha256'] == bind(historical / 'CUMULATIVE_SOURCE.json')['sha256'], 'HISTORICAL_READY_REQUIRED')
            target.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            for name in ['EXPANSION_DRAFT.json', 'CUMULATIVE_SOURCE.json', 'SOURCE_REVIEW.json']:
                need(not (target / name).exists(), 'IMPORT_ALREADY_EXISTS')
                shutil.copyfile(historical / name, target / name)
                artifacts[name] = {'historical_binding': bind(historical / name), 'imported_binding': bind(target / name)}
            imported.append({'scene_ref': scene, 'verbatim_artifacts': artifacts})
            previous = source['turns']
        need(previous == read(candidate['candidate_prefix_binding']['path'])['turns'], 'IMPORT_CANDIDATE_DIFFERS')
        ip = unit / 'PREFIX_IMPORT.json'
        save(ip, {'schema_version': 'historical_prefix_import_v1', 'created_at_utc': now(), 'actual_executor': '/root',
            'root_id': root, 'retained_scene_refs': row['proposed_retained_scene_refs'],
            'role_assignments_binding': bind(B / 'ROLE_ASSIGNMENTS.json'), 'prefix_review_binding': bind(auditp),
            'prefix_read_receipt_binding': bind(receiptp), 'candidate_prefix_binding': candidate['candidate_prefix_binding'],
            'scenes': imported, 'historical_identities_and_timestamps_preserved': True,
            'no_new_event_consumption_for_retained_prefix': True})
        out.update({'baseline_binding': bind(unit / 'inputs/BASELINE.json'),
                    'constraints_binding': bind(unit / 'inputs/CONTINUATION_CONSTRAINTS.json'),
                    'prefix_import_binding': bind(ip), 'candidate_prefix_binding': candidate['candidate_prefix_binding']})
        rows.append(out)
    ready = sum(r['source_ready'] for r in rows); passed = sum(r['prefix_reuse_approved'] for r in rows)
    save(B / 'MANIFEST.json', {'schema_version': 'v121_all_failed_repair_manifest', 'batch_id': B.name,
        'created_at_utc': now(), 'actual_executor': '/root', 'fixed_candidates': 24, 'source_ready': ready,
        'source_hold': 24 - ready, 'prefix_pass': passed, 'prefix_hold': ready - passed, 'roots': rows,
        'new_generation_started': False, 'training_pool_merge': False, 'original_batches_immutable': True,
        'source_repair_has_explicit_synthetic_provenance': True,
        'new_scene_budget': sum(len(r['remaining_scene_refs']) for r in rows if r['prefix_reuse_approved']),
        'new_b_event_budget': sum(len(r['new_b_scene_refs']) for r in rows if r['prefix_reuse_approved'])})
    print(json.dumps({'status': 'PREFIX_IMPORT_FINISHED', 'source_ready': ready, 'prefix_pass': passed, 'prefix_hold': ready - passed}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare-prefix', 'import-prefix'])
    args = parser.parse_args()
    if args.action == 'prepare-prefix': prepare_prefix()
    else: import_prefix()
