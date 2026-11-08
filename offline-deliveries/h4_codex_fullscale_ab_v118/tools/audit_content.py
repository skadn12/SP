"""Recompute source/procedure/export integrity; semantic truth remains reviewed."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import json
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
P = B.parent.parent
sys.path.insert(0, str(P / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_sequential_ab import SCENES, inspect_root, load_config, load_baseline, scene_folder
from codex_scene_payload import build_draft
from codex_sequential_questions import validate_final_review
from codex_evidence_export import export_root_questions
from status import population


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def verify(row): assert sha(row['path']) == row['sha256'], row['path']
def recursive_bindings(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str): verify(value)
        for v in value.values(): recursive_bindings(v)
    elif isinstance(value, list):
        for v in value: recursive_bindings(v)


def main(wave, output):
    started = datetime.now(timezone.utc).isoformat()
    bindings = [bind(__file__), bind(B / 'MASTER_FREEZE.json'), bind(B / 'MANIFEST.json'), bind(B / 'ROLE_ASSIGNMENTS.json')]
    errors = []
    frozen = read(B / 'MASTER_FREEZE.json')
    for item in frozen['files']: verify(item)
    historical = read(B / 'evidence/HISTORICAL_PROTECTION.json')['files']
    for item in historical: verify(item)
    manifest = read(B / 'MANIFEST.json')
    states = {r['root_id']: r for r in population()['roots']}
    roles = {r['root_id']: r for r in read(B / 'ROLE_ASSIGNMENTS.json')['roots']}
    target = [r for r in manifest['roots'] if r['action'] in ['SCREEN', 'INHERIT_SCREEN'] and (wave == 'all' or r['wave'] == wave)]
    assert target and all(states[r['root_id']]['terminal'] for r in target), 'WAIT_FOR_FIXED_WAVE_TERMINAL_STATES'
    all_reservations = [a for p in (B / 'units').glob('*/SOURCE_RESERVATIONS.json') for a in read(p)['allocation']]
    assert len(all_reservations) == len({a['source_id'] for a in all_reservations}), 'CROSS_UNIT_DUPLICATE_EVENT'
    consumed = set(read(B / 'evidence/SOURCE_POOL_FREEZE.json')['consumed_source_ids'])
    assert not {a['source_id'] for a in all_reservations} & consumed
    totals = Counter(); gold_counts = Counter(); root_results = []
    for row in target:
        rid = row['root_id']; unit = Path(row['unit_path']); role = roles[rid]
        result = {'root_id': rid, 'status': states[rid]['status'], 'cohort': row['cohort']}
        try:
            source = read(row['baseline_path']); base_turns = source['turns']
            cp = unit / 'inputs/CONTINUATION_CONSTRAINTS.json'; constraints = read(cp)
            assert constraints['root_id'] == rid and constraints['baseline_binding'] == row['baseline_binding']
            actual_reviewers = []
            decisions = []
            for review_binding in constraints['review_bindings']:
                verify(review_binding); review = read(review_binding['path'])
                actual_reviewers.append(review['reviewer']); decisions.append(review['decision'])
                assert review['full_source_read'] is True and review['covered_turn_refs'] == [t['turn_ref'] for t in base_turns]
                assert review['baseline_sha256'] == row['baseline_sha256']
            assert len(set(actual_reviewers)) == 2 and sorted(actual_reviewers) == sorted(role['baseline_reviewers'])
            rank = {'PASS': 0, 'WARNING': 1, 'HOLD': 2}
            assert constraints['decision'] == max(decisions, key=rank.get)
            assert constraints['continuation_constraints']
            if constraints['decision'] == 'HOLD':
                assert not (unit / 'RUN_CONFIG.json').exists()
                totals['baseline_hold_roots'] += 1
            elif (unit / 'MATCHING_HOLD.json').exists():
                totals['matching_hold_roots'] += 1
            else:
                inputs = read(unit / 'EXECUTION_INPUTS_FROZEN.json')
                for item in inputs['files']: verify(item)
                cfg = load_config(unit); state = inspect_root(unit, cfg, rid, record_reviews=False)
                selected = read(unit / 'selection.json')
                assert selected['selector'] == role['author'] and selected['constraints_binding'] == bind(cp)
                assert selected['baseline_turn_refs_read'] == [t['turn_ref'] for t in base_turns]
                pool = {e['dialogue_id']: e for e in read(unit / 'inputs/EVENT_POOL.json')['events']}
                reserved = read(unit / 'SOURCE_RESERVATIONS.json')['allocation']
                assert len(reserved) == 3
                for a in reserved:
                    event = read(a['source_item_path']); verify({'path': a['source_item_path'], 'sha256': a['source_item_sha256']})
                    assert pool[a['source_id'].removeprefix('multiwoz22:train:')] == event
                    assert a['native_turn_ids_read'] == [t['turn_id'] for t in event['turns']]
                previous = Path(row['baseline_path']); previous_review_end = None
                for s in state['scenes']:
                    scene = s['scene_ref']; folder = scene_folder(unit, rid, scene)
                    draft_path = folder / 'EXPANSION_DRAFT.json'
                    if not draft_path.exists(): continue
                    totals['attempted_scenes'] += 1
                    raw = read(folder / 'AUTHOR_PAYLOAD.json'); receipt = read(folder / 'AUTHOR_RECEIPT.json') if (folder / 'AUTHOR_RECEIPT.json').exists() else None
                    prebind = read(folder / 'PAYLOAD_SERIALIZATION_BINDING.json')
                    assert prebind['saved_before_draft'] is True
                    assert (folder / 'AUTHOR_PAYLOAD.json').stat().st_mtime_ns <= (folder / 'PAYLOAD_SERIALIZATION_BINDING.json').stat().st_mtime_ns <= draft_path.stat().st_mtime_ns
                    source_id = next((a['source_id'] for a in reserved if a['scene_ref'] == scene), None)
                    expected = build_draft(rid, scene, role['author'], previous, raw['turns'], source_id=source_id, **{k: raw[k] for k in ['state_notes', 'event_mapping'] if k in raw})
                    assert read(draft_path) == expected
                    if receipt is not None:
                        assert receipt['author'] == receipt['executor'] == role['author']
                        assert receipt['constraints_binding'] == bind(cp)
                        assert receipt['full_source_read'] is True and receipt['all_continuation_constraints_read'] is True
                        assert receipt['turn_refs_read'] == [t['turn_ref'] for t in read(previous)['turns']]
                        if previous_review_end: assert receipt['started_at_utc'] >= previous_review_end
                    if (folder / 'SOURCE_REVIEW.json').exists():
                        review = read(folder / 'SOURCE_REVIEW.json')
                        assert review['reviewer'] == role['reviewer'] != role['author']
                        assert review['all_continuation_constraints_read'] is True
                        judged = review['baseline_constraint_review']['constraint_results']
                        assert [j['constraint'] for j in judged] == constraints['continuation_constraints']
                        assert review['full_source_read'] is True
                        if s['status'] == 'READY':
                            totals['ready_scenes'] += 1
                            assert all(j['status'] != 'FAIL' for j in judged)
                            assert review['decision'] == 'READY'
                        if ':B:' in scene:
                            assert review['native_event_full_read'] is True and receipt['native_event_full_read'] is True
                        previous_review_end = review['finished_at_utc']
                    previous = folder / 'CUMULATIVE_SOURCE.json'
                root_folder = unit / 'roots' / ('root' + rid.rsplit('-', 1)[1])
                if state['full_expansion_complete']:
                    totals['complete_expansion_roots'] += 1
                    assert (root_folder / 'CUMULATIVE_SOURCE.json').read_bytes() == (scene_folder(unit, rid, SCENES[-1]) / 'CUMULATIVE_SOURCE.json').read_bytes()
                    if (root_folder / 'FINAL_CROSSCHECK.json').exists():
                        _, qs, final_review, outcomes = validate_final_review(root_folder, cfg)
                        assert final_review['reviewer'] == role['reviewer'] != read(root_folder / 'QA_AUTHOR_RECEIPT.json')['author']
                        assert final_review['final_public_questions_actually_reread'] is True
                        assert final_review['private_answers_read_before_save'] is False
                        assert (root_folder / 'ORDERING_ANCHOR_PLAN.json').stat().st_mtime_ns <= (root_folder / 'QA_DRAFT.json').stat().st_mtime_ns
                        cross = read(root_folder / 'FINAL_CROSSCHECK.json')
                        gold = {q['candidate_id']: q for q in read(root_folder / 'PRIVATE_GOLD.json')['questions']}
                        exported = export_root_questions(root_folder)
                        assert {q['candidate_id'] for q in exported} == {r['candidate_id'] for r in cross['results'] if r['status'] == 'CONTENT_PASS'}
                        for q in exported:
                            assert q['gold'] == gold[q['candidate_id']]['gold'] == outcomes[q['candidate_id']]['source_derived_answer']
                            assert not outcomes[q['candidate_id']]['source_issues']
                            if q['question_type'] == 'multi_select': gold_counts[str(len(q['gold']))] += 1
                            totals['content_pass_questions'] += 1
                            totals['control_questions' if q['role'] == 'control6' else q['question_type']] += 1
                        result['content_pass'] = len(exported)
                        result['excluded'] = sum(r['status'] != 'CONTENT_PASS' for r in cross['results'])
                else:
                    assert state['root_stopped']
                    totals['stopped_expansion_roots'] += 1
            for p in unit.rglob('*.json'):
                recursive_bindings(read(p)); bindings.append(bind(p))
        except Exception as exc:
            errors.append({'root_id': rid, 'error': str(exc), 'type': type(exc).__name__})
        root_results.append(result)
    data = {'status': 'PASS' if not errors else 'FAIL', 'wave': wave, 'started_at_utc': started,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'root_denominator': len(target),
        'historical_files_verified_unchanged': len(historical), 'totals': dict(totals), 'multi_gold_counts': dict(gold_counts),
        'roots': root_results, 'input_bindings': bindings, 'errors': errors,
        'scope': 'Mechanical source, declaration, prefix, timing, no-overwrite, selection, crosscheck and lossless-export integrity; no external semantic certification.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in data.items() if k not in ['input_bindings', 'roots']}, ensure_ascii=False))
    assert not errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', choices=['validation', 'production', 'all'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); main(args.wave, args.output.resolve())
