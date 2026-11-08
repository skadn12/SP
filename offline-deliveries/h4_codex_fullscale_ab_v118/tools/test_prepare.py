"""Exercise real unit configuration, reservation and overwrite checks on fake inputs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location('prepare_v118', Path(__file__).with_name('prepare.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

with tempfile.TemporaryDirectory() as temp:
    b = Path(temp); m.B = b
    rid = 'cc-train-episode-999'; unit = b / 'units/root999'
    author = '/root/test_author'; reviewer = '/root/test_reviewer'
    turns = [{'turn_ref': f'R01:S{i:02d}:T001', 'scene_ref': f'R01:S{i:02d}',
              'speaker': 'A', 'text': f'Existing unresolved topic {i}.'} for i in range(1, 4)]
    baseline = unit / 'inputs/BASELINE.json'; m.save(baseline, {'root_id': rid, 'turns': turns})
    dummy = b / 'baseline_review.json'; m.save(dummy, {'decision': 'PASS'})
    cp = unit / 'inputs/CONTINUATION_CONSTRAINTS.json'
    m.save(cp, {'root_id': rid, 'decision': 'PASS', 'continuation_constraints': ['Keep stated identities.'],
                'baseline_binding': m.bind(baseline), 'review_bindings': [m.bind(dummy)]})
    m.save(b / 'ROLE_ASSIGNMENTS.json', {'roots': [{'root_id': rid, 'author': author, 'reviewer': reviewer}]})
    row = {'root_id': rid, 'unit_path': str(unit), 'baseline_path': str(baseline),
           'baseline_sha256': m.sha(baseline), 'baseline_binding': m.bind(baseline), 'cohort': 'standard'}
    m.master = lambda: {'roots': [row]}
    m.save(b / 'MANIFEST.json', {'roots': [row]})
    m.save(b / 'MASTER_FREEZE.json', {'test_fixture': True})
    events = [{'dialogue_id': f'TEST{i}.json', 'services': ['restaurant'],
               'turns': [{'turn_id': '0', 'speaker': 'USER', 'utterance': f'Please query {i}.'}]} for i in range(1, 4)]
    m.save(unit / 'inputs/EVENT_POOL.json', {'events': events})
    m.save(b / 'evidence/SOURCE_POOL_FREEZE.json', {'consumed_source_ids': []})
    (b / 'evidence/CONTENT_CONTRACT.md').write_text('Fixture contract')
    try:
        m.match_read(rid, reviewer)
        raise RuntimeError('Wrong author accepted')
    except AssertionError as exc:
        assert str(exc) == 'ACTUAL_ASSIGNED_AUTHOR_REQUIRED'
    with contextlib.redirect_stdout(io.StringIO()): m.match_read(rid, author)
    raw = {'read_attestation': {'full_source_read': True, 'all_continuation_constraints_read': True,
                               'all_candidate_events_fully_read': True},
           'a_continuations': [{'scene_ref': f'R02:A:S{i:02d}', 'baseline_scene_ref': f'R01:S{i:02d}',
                                'thread': f'topic {i}', 'evidence': [{'turn_ref': t['turn_ref'], 'speaker': 'A', 'quote': t['text']}]}
                               for i, t in enumerate(turns, 1)],
           'allocations': [{'scene_ref': f'R02:B:S{i:02d}', 'source_id': f'multiwoz22:train:TEST{i}.json',
                            'compatibility_reason': 'New query by established speakers.', 'main_purpose': f'query {i}'} for i in range(1, 4)]}
    bad = json.loads(json.dumps(raw)); bad['allocations'][0]['source_id'] = 'multiwoz22:train:FOREIGN.json'
    m.save(b / 'bad.json', bad)
    try:
        m.match_save(rid, author, b / 'bad.json')
        raise RuntimeError('Foreign partition accepted')
    except AssertionError: pass
    assert not (unit / 'selection.json').exists()
    m.save(b / 'good.json', raw)
    with contextlib.redirect_stdout(io.StringIO()): m.match_save(rid, author, b / 'good.json')
    assert (unit / 'EXECUTION_INPUTS_FROZEN.json').exists()
    assert len(m.read(unit / 'SOURCE_RESERVATIONS.json')['allocation']) == 3
    selected_hash = m.sha(unit / 'selection.json')
    try:
        m.match_save(rid, author, b / 'good.json')
        raise RuntimeError('Duplicate write accepted')
    except FileExistsError: pass
    assert m.sha(unit / 'selection.json') == selected_hash
    print(json.dumps({'status': 'PASS', 'checks': ['wrong_executor_rejected',
          'outside_partition_rejected_before_reservation', 'real_controller_freeze_and_no_overwrite']}))
