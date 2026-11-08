"""Immutable v118 baseline read receipts, semantic judgments and two-reader gate."""
import argparse
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKS = ('identity_coreference', 'gender_relationship_titles', 'fact_state_time', 'unsupported_reconciliation')
RANK = {'PASS': 0, 'WARNING': 1, 'HOLD': 2}
CLASSES = {'HARD_CONTRADICTION': 'HOLD', 'UNRESOLVED_IDENTITY_OR_TITLE': 'HOLD', 'UNRESOLVED_CORE_EVENT': 'HOLD',
           'UNRESOLVED_STATE_TIME': 'WARNING', 'ATTRIBUTED_UNVERIFIED_REPORT': 'WARNING', 'UNDERSPECIFIED_ATTRIBUTE': 'WARNING',
           'COMPATIBLE_TIME_CHANGE': 'PASS', 'COMPATIBLE_NONEXCLUSIVE': 'PASS'}
JUDGMENT = {'full_source_read', 'covered_turn_refs', 'checks', 'findings', 'continuation_constraints', 'decision'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def bound(item):
    require(Path(item['path']).is_absolute() and binding(item['path'])['sha256'] == item['sha256'], 'binding mismatch: ' + item['path'])


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, 'timestamp must be UTC')
    return result


def save(path, value, raw=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value if raw else (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def load(root):
    freeze_path = ROOT / 'MASTER_FREEZE.json'
    freeze = read(freeze_path)
    require(freeze['status'] == 'FROZEN_BEFORE_NEW_BASELINE_FULL_TEXT_READS', 'master is not frozen')
    frozen = {item['path']: item for item in freeze['files']}
    required = [ROOT / 'MANIFEST.json', ROOT / 'ROLE_ASSIGNMENTS.json', ROOT / 'evidence/BASELINE_POLICY.md', Path(__file__).resolve()]
    require(all(str(path) in frozen for path in required), 'freeze must bind manifest, roles, baseline policy and helper')
    for item in freeze['files']:
        bound(item)
    require(timestamp(freeze['created_at_utc']) <= datetime.now(timezone.utc), 'future freeze timestamp')
    manifest = read(required[0])
    require(len(manifest['roots']) == 68 and len({r['root_id'] for r in manifest['roots']}) == 68, 'fixed 68-root universe required')
    root = root if root.startswith('cc-train-episode-') else 'cc-train-episode-' + root
    require(re.fullmatch(r'cc-train-episode-\d+', root) is not None, 'invalid root identity')
    rows = [r for r in manifest['roots'] if r['root_id'] == root]
    require(len(rows) == 1, 'root outside frozen universe')
    row = rows[0]
    expected = {'path': row['baseline_path'], 'sha256': row['baseline_sha256']}
    bound(expected)
    require(row['baseline_binding'] == expected, 'baseline binding disagrees with manifest fields')
    if row.get('upstream_binding'):
        bound(row['upstream_binding'])
        require(row['upstream_binding']['sha256'] == row['baseline_sha256'], 'baseline differs from upstream')
    if row.get('upstream_path'):
        bound({'path': row['upstream_path'], 'sha256': row.get('upstream_sha256', row['baseline_sha256'])})
        require(binding(row['upstream_path'])['sha256'] == row['baseline_sha256'], 'baseline differs from upstream')
    unit = Path(row.get('unit_path', ROOT / 'units' / ('root' + root.rsplit('-', 1)[1]))).resolve()
    require(unit == ROOT / 'units' / ('root' + root.rsplit('-', 1)[1]), 'unexpected unit path')
    return row, unit, [binding(freeze_path), *[binding(path) for path in required], expected], freeze['created_at_utc']


def source(row):
    value = read(row['baseline_path'])
    require(value['root_id'] == row['root_id'], 'baseline root mismatch')
    turns = value['turns']
    require(bool(turns) and all(isinstance(t.get('turn_ref'), str) and isinstance(t.get('speaker'), str) and isinstance(t.get('text'), str) and t['text'].strip() for t in turns), 'invalid baseline turns')
    require(len({t['turn_ref'] for t in turns}) == len(turns), 'duplicate turn references')
    return value, {t['turn_ref']: t for t in turns}


def reader_dir(row, unit, executor):
    readers = row['baseline_reviewers']
    require(len(readers) == 2 and len(set(readers)) == 2 and executor in readers, 'executor is not an assigned baseline reader')
    assignments = read(ROOT / 'ROLE_ASSIGNMENTS.json')['roots']
    assignment = next(r for r in assignments if r['root_id'] == row['root_id'])
    require(assignment['baseline_reviewers'] == readers, 'role assignment mismatch')
    return unit / 'baseline_reviews' / ('reader' + str(readers.index(executor) + 1))


def validate(judgment, turns):
    require(judgment['full_source_read'] is True and judgment['covered_turn_refs'] == list(turns), 'complete ordered source coverage required')
    require(set(judgment['checks']) == set(CHECKS), 'four checks required')
    statuses = {key: [] for key in CHECKS}
    ids = set()
    def cite(c):
        require(set(c) == {'turn_ref', 'speaker', 'quote'} and c['turn_ref'] in turns, 'invalid citation fields/reference')
        turn = turns[c['turn_ref']]
        require(c['speaker'] == turn['speaker'] and isinstance(c['quote'], str) and c['quote'].strip() and c['quote'] in turn['text'], 'citation speaker/quote mismatch')
    for finding in judgment['findings']:
        require(set(finding) == {'id', 'category', 'classification', 'disposition', 'reason', 'citations', 'temporal_evidence'}, 'invalid finding fields')
        require(isinstance(finding['id'], str) and finding['id'].strip() and finding['id'] not in ids, 'invalid/duplicate finding ID')
        ids.add(finding['id'])
        category, classification = finding['category'], finding['classification']
        require(category in CHECKS and classification in CLASSES and finding['disposition'] == CLASSES[classification], 'invalid finding classification/disposition')
        require(isinstance(finding['reason'], str) and finding['reason'].strip() and isinstance(finding['citations'], list) and finding['citations'], 'reason and citations required')
        for c in finding['citations']:
            cite(c)
        require(finding['disposition'] != 'HOLD' or len({c['turn_ref'] for c in finding['citations']}) >= 2, 'HOLD needs two distinct source turns')
        require(isinstance(finding['temporal_evidence'], list), 'temporal evidence must be a list')
        for c in finding['temporal_evidence']:
            cite(c)
        require(classification != 'COMPATIBLE_TIME_CHANGE' or finding['temporal_evidence'], 'time change needs explicit temporal evidence')
        statuses[category].append(finding['disposition'])
    for category, check in judgment['checks'].items():
        require(set(check) == {'complete', 'status', 'reason'} and check['complete'] is True and isinstance(check['reason'], str) and check['reason'].strip(), 'incomplete check')
        require(check['status'] == max(statuses[category], key=RANK.get, default='PASS'), 'check does not aggregate findings')
    require(judgment['decision'] == max((c['status'] for c in judgment['checks'].values()), key=RANK.get), 'decision does not aggregate checks')
    constraints = judgment['continuation_constraints']
    require(isinstance(constraints, list) and all(isinstance(x, str) and x.strip() for x in constraints), 'constraints must be nonempty strings')
    require(bool(constraints), 'every review needs genuine source-derived continuation constraints, including PASS')


def run(args):
    row, unit, inputs, frozen_at = load(args.root)
    started = now()
    if args.command == 'read':
        require(row['action'] == 'SCREEN', 'only newly SCREEN roots need a new read')
        folder = reader_dir(row, unit, args.executor)
        value, turns = source(row)
        render = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        save(folder / 'BASELINE_RENDER.txt', render.encode(), raw=True)
        save(folder / 'READ_CONTEXT.json', {'root_id': row['root_id'], 'reviewer': args.executor, 'started_at_utc': started, 'input_bindings': inputs, 'render_binding': binding(folder / 'BASELINE_RENDER.txt'), 'covered_turn_refs': list(turns)})
        print(render, end='')
        return
    if args.command == 'save':
        require(row['action'] == 'SCREEN', 'new judgment only allowed for SCREEN')
        folder = reader_dir(row, unit, args.executor)
        context = read(folder / 'READ_CONTEXT.json')
        require(context['root_id'] == row['root_id'] and context['reviewer'] == args.executor and context['input_bindings'] == inputs, 'read context identity/input mismatch')
        bound(context['render_binding'])
        raw = Path(args.judgment).read_bytes()
        submission = folder / 'submissions' / (uuid.uuid4().hex + '.json')
        save(submission, raw, raw=True)
        try:
            judgment = read(submission)
            require(set(judgment) == JUDGMENT, 'judgment fields must be exactly: ' + ', '.join(sorted(JUDGMENT)))
            _, turns = source(row)
            require(context['covered_turn_refs'] == list(turns), 'read context coverage changed')
            validate(judgment, turns)
            finished = now()
            require(timestamp(frozen_at) <= timestamp(context['started_at_utc']) <= timestamp(finished), 'invalid actual review interval')
            review = {**judgment, 'schema_version': 'codex_baseline_review_v1', 'policy_version': 'v118', 'root_id': row['root_id'], 'reviewer': args.executor, 'baseline_path': row['baseline_path'], 'baseline_sha256': row['baseline_sha256'], 'started_at_utc': context['started_at_utc'], 'finished_at_utc': finished, 'input_bindings': inputs, 'read_context_binding': binding(folder / 'READ_CONTEXT.json'), 'raw_judgment_binding': binding(submission)}
            require(load(args.root)[2] == inputs, 'inputs changed during save')
            save(folder / 'REVIEW.json', review)
        except Exception as error:
            save(submission.with_suffix('.failure.json'), {'failed_at_utc': now(), 'reviewer': args.executor, 'error': str(error), 'raw_judgment_binding': binding(submission)})
            raise
        print(json.dumps({'review': binding(folder / 'REVIEW.json'), 'decision': judgment['decision']}))
        return
    require(row['action'] in {'SCREEN', 'INHERIT_SCREEN', 'HISTORICAL_HOLD'}, 'REUSE roots keep historical completion records')
    _, turns = source(row)
    if row['action'] == 'SCREEN':
        bindings = [binding(reader_dir(row, unit, executor) / 'REVIEW.json') for executor in row['baseline_reviewers']]
    else:
        bindings = row['inherited_reviews']
        authority = ROOT.parent / 'h4_codex_baseline_screen_v116/SCREEN_RESULTS.json'
        if row.get('inherited_screen_binding'):
            bound(row['inherited_screen_binding'])
            require(row['inherited_screen_binding']['path'] == str(authority), 'unexpected inherited screen authority')
        prior = next(r for r in read(authority)['roots'] if r['root_id'] == row['root_id'])
        require(prior['baseline_sha256'] == row['baseline_sha256'] and len(bindings) == 2, 'inherited baseline mismatch')
        require({(b['path'], b['sha256']) for b in bindings} == {(b['path'], b['sha256']) for b in prior['reviews']}, 'inheritance differs from v116 authority')
        inputs = [*inputs, binding(authority)]
    reviews = []
    for item in bindings:
        bound(item)
        review = read(item['path'])
        require(review['root_id'] == row['root_id'] and review['baseline_sha256'] == row['baseline_sha256'], 'review baseline mismatch')
        validate(review, turns)
        if row['action'] == 'SCREEN':
            require(review['schema_version'] == 'codex_baseline_review_v1' and review['policy_version'] == 'v118' and review['input_bindings'] == inputs and review['reviewer'] in row['baseline_reviewers'], 'review identity/version/input mismatch')
            bound(review['raw_judgment_binding'])
            bound(review['read_context_binding'])
            require(read(review['raw_judgment_binding']['path']) == {k: review[k] for k in JUDGMENT}, 'review differs from original judgment')
            context = read(review['read_context_binding']['path'])
            require(context['reviewer'] == review['reviewer'] and context['root_id'] == row['root_id'] and context['input_bindings'] == inputs and context['covered_turn_refs'] == list(turns) and context['started_at_utc'] == review['started_at_utc'], 'review differs from read context')
            require(timestamp(frozen_at) <= timestamp(review['started_at_utc']), 'review predates current freeze')
            bound(context['render_binding'])
        else:
            require(review['policy_version'] == 'v116', 'invalid inherited policy')
        require(timestamp(review['started_at_utc']) <= timestamp(review['finished_at_utc']) <= datetime.now(timezone.utc), 'invalid review time')
        reviews.append(review)
    require(len(reviews) == 2 and len({r['reviewer'] for r in reviews}) == 2, 'two distinct readers required')
    decision = max((r['decision'] for r in reviews), key=RANK.get)
    require(row['action'] != 'HISTORICAL_HOLD' or decision == 'HOLD', 'historical HOLD cannot be rescued')
    require(row['action'] != 'INHERIT_SCREEN' or decision != 'HOLD', 'inherited eligible screen became HOLD')
    output = {'root_id': row['root_id'], 'decision': decision, 'continuation_constraints': list(dict.fromkeys(c for r in reviews for c in r['continuation_constraints'])), 'baseline_binding': binding(row['baseline_path']), 'review_bindings': bindings, 'input_bindings': inputs, 'executor': args.executor, 'created_at_utc': now(), 'semantic_correctness_certified_by_program': False}
    save(unit / 'inputs/CONTINUATION_CONSTRAINTS.json', output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['read', 'save', 'gate'])
    parser.add_argument('--executor', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--judgment')
    arguments = parser.parse_args()
    require(arguments.command != 'save' or arguments.judgment, 'save requires --judgment')
    run(arguments)
