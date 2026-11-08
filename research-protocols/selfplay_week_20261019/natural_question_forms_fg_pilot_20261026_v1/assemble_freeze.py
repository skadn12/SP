#!/usr/bin/env python3
"""Assemble final 40 slots and reconcile frozen independent keys, before Solver.

Only runs with an explicit --spec. Does not create review keys, fix questions,
replace rejected slots, or start preparation/inference. Input spec keys:
  candidate_files: [final author-list paths]
  initial_candidate_files: [original draft-list paths] (defaults to final)
  review_rounds: [{packet, key, freeze, blind_map}, ...] in time order
  revision_ledger: optional path to {records:[{candidate_id,revision_count:1,
    initial_candidate_sha256,final_candidate_sha256,trigger_review_binding}]}
  extra_bindings: optional paths (e.g. plans/plan freezes, draft/revision freezes)
Every key is Gold/arm blind and its separately frozen hash is retained. A final
question must have a key matching its exact final hash. At most one tracked
revision is allowed; unresolved content stays HOLD/FAIL in the 40-slot ledger.
"""
from __future__ import annotations
import argparse
import collections
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner

read, binding, fingerprint = runner.read, runner.bind, runner.orig.fingerprint
CHECKS = ('source_support', 'gold_truth', 'option_distinctness',
          'distractor_relation', 'natural_concrete_task', 'source_dependence',
          'r03_necessity', 'operation_alignment')
PUBLIC = ('root_id', 'question_type', 'task_operation', 'question', 'options', 'source_binding')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (HERE / p).resolve()


def save(filename, value):
    p = HERE / filename
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def candidate_list(files):
    rows = []
    for name in files:
        value = read(path(name))
        rows.extend(value['candidates'] if isinstance(value, dict) else value)
    need(len(rows) == 40 and len({c['candidate_id'] for c in rows}) == 40,
         'Exactly all 40 planned candidate slots, including rejected slots, required.')
    expected = {f'CL1-{a}-{r.split("-")[-1]}-{s}'
                for r in runner.ROOTS for a in runner.ARMS for s in runner.SLOTS}
    need({c['candidate_id'] for c in rows} == expected, 'Missing or unplanned candidate slot.')
    return {c['candidate_id']: c for c in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    args = parser.parse_args()
    names = ('CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json',
             'MECHANICAL_FINAL.json', 'CONTENT_REVIEW_FINAL_AUDIT.json', 'CONTENT_FREEZE.json')
    need(not any((HERE / n).exists() for n in names), 'Final artifacts already exist; no overwrite.')
    need(not (HERE / 'execution').exists(), 'Cannot freeze/rewrite content after preparation.')
    spec_path = path(args.spec)
    spec = read(spec_path)
    runner.validate_protocol(read(HERE / 'PROTOCOL_FREEZE.json'))
    inputs = [binding(spec_path)]
    for b in read(HERE / 'AUTHORING_INPUT_FREEZE.json')['bindings']:
        need(binding(b['path']) == b, 'Authoring input changed: ' + b['path'])
        inputs.append(b)
    sources = {r['root_id']: r for r in read(HERE / 'SOURCE_SELECTION_FINAL.json')['roots']}
    need(set(sources) == set(runner.ROOTS), 'Original four roots required.')
    source_turns = {}
    for root, item in sources.items():
        b = {k: item[k] for k in ('path', 'sha256', 'bytes')}
        need(binding(b['path']) == b, 'Source bytes/hash changed: ' + root)
        source_turns[root] = {t['turn_ref']: t for t in read(b['path'])['turns']}
        inputs.append(b)
    final = candidate_list(spec['candidate_files'])
    initial = candidate_list(spec.get('initial_candidate_files', spec['candidate_files']))
    for name in spec['candidate_files'] + spec.get('initial_candidate_files', []):
        inputs.append(binding(path(name)))
    gate_path = HERE.parents[2] / 'scripts/current_method/production_question_gate.py'
    sys.path.insert(0, str(gate_path.parent))
    gs = importlib.util.spec_from_file_location('natural_form_production_gate', gate_path)
    gate = importlib.util.module_from_spec(gs)
    gs.loader.exec_module(gate)
    inputs.append(binding(gate_path))
    for c in list(initial.values()) + list(final.values()):
        cid = c['candidate_id']
        root, slot, arm = c['root_id'], c['slot_id'], c['arm']
        need(cid == f'CL1-{arm}-{root.split("-")[-1]}-{slot}' and
             root in runner.ROOTS and slot in runner.SLOTS and arm in runner.ARMS,
             'Candidate identity drift: ' + cid)
        need(c['question_type'] == 'multi_select' and
             c['task_operation'] == runner.SLOTS[slot][0] and
             list(c['options']) == list('ABCDEF') and
             len(c['gold']) == runner.SLOTS[slot][1] and
             len(set(c['gold'])) == len(c['gold']) and set(c['gold']) <= set('ABCDEF'),
             'Slot operation/options/Gold-quota drift: ' + cid)
        need(c.get('author_id') and c['source_binding'] ==
             {k: sources[root][k] for k in ('path', 'sha256', 'bytes')},
             'Missing author or source binding drift: ' + cid)
    changed = {cid for cid in final if final[cid] != initial[cid]}
    history, keys = collections.defaultdict(list), {}
    for index, round_spec in enumerate(spec['review_rounds']):
        files = {name: path(round_spec[name]) for name in ('packet', 'key', 'freeze', 'blind_map')}
        packet, key, freeze, mapping = (read(files[n]) for n in ('packet', 'key', 'freeze', 'blind_map'))
        key_binding = binding(files['key'])
        need(freeze['binding'] == key_binding and freeze.get('frozen_before_author_gold') is True,
             'Independent key hash/freeze mismatch: ' + str(files['key']))
        need(key.get('gold_and_arm_blind') is True and key.get('frozen_before_author_gold') is True
             and key.get('reviewer_is_human') is False and key.get('reviewer_id'),
             'Explicit independent AI blind-key metadata required.')
        questions = {q['review_id']: q for q in packet['questions']}
        rows = key['reviews']
        need(len(questions) == len(packet['questions']) and len(rows) == len(questions) and
             len({r['review_id'] for r in rows}) == len(rows) and
             {r['review_id'] for r in rows} == set(questions), 'Exact blind review coverage required.')
        keys[key_binding['path']] = key_binding
        for original in rows:
            bid = original['review_id']
            mapped, q = mapping[bid], questions[bid]
            cid, chash = mapped['candidate_id'], mapped['candidate_sha256']
            need(cid in final, 'Unplanned blind mapped candidate.')
            versions = [c for c in (initial[cid], final[cid]) if fingerprint(c) == chash]
            need(versions, 'Blind key maps to neither retained initial nor final candidate: ' + cid)
            c = versions[0]
            need(all(q[k] == c[k] for k in PUBLIC), 'Blind public question differs: ' + cid)
            need(not ({'gold', 'arm', 'author_id', 'item_analysis', 'feedback_target'} & set(q)),
                 'Private information in blind public question: ' + cid)
            need(mapped.get('author_id', c['author_id']) == c['author_id'] and
                 key['reviewer_id'] != c['author_id'], 'Reviewer must differ from author: ' + cid)
            row = copy.deepcopy(original)
            row.update(candidate_id=cid, candidate_sha256=chash, source_binding=c['source_binding'],
                       author_id=c['author_id'], reviewer_id=key['reviewer_id'],
                       reviewer_is_human=False, gold_blind=True,
                       independent_key_binding=key_binding, review_round=index,
                       original_independent_decision=original.get('decision'))
            errors = []
            independent = row.get('independent_gold')
            if not (isinstance(independent, list) and all(isinstance(x, str) for x in independent)
                    and len(independent) == len(set(independent)) and set(independent) <= set('ABCDEF')):
                errors.append('Malformed independent Gold.')
                independent = []
            row['author_gold_agrees'] = set(independent) == set(c['gold'])
            if not row['author_gold_agrees']:
                errors.append('Frozen independent key disagrees with author Gold.')
            checks = row.get('checks', {})
            for k in CHECKS:
                check = checks.get(k, {})
                if not (check.get('decision') in ('PASS', 'HOLD', 'FAIL') and
                        isinstance(check.get('rationale'), str) and check['rationale'].strip()):
                    errors.append('Missing explicit review check: ' + k)
                elif check['decision'] != 'PASS':
                    errors.append('Unresolved check: ' + k)
            if row.get('decision') not in ('PASS', 'HOLD', 'FAIL'):
                errors.append('Unknown overall decision.')
            options = row.get('options', {})
            if set(options) != set('ABCDEF'):
                errors.append('Independent per-option coverage mismatch.')
            for label, option in options.items():
                if not isinstance(option.get('selected'), bool) or option.get('selected') != (label in independent):
                    errors.append('Independent option selected/Gold inconsistency: ' + label)
                if not option.get('rationale') or not option.get('evidence'):
                    errors.append('Independent option needs rationale/evidence: ' + label)
                for ev in option.get('evidence', []):
                    turn = source_turns[c['root_id']].get(ev.get('turn_ref'))
                    if (not turn or ev.get('speaker') != turn['speaker'] or
                        not ev.get('quote') or ev['quote'] not in turn['text']):
                        errors.append('Independent literal evidence mismatch: ' + label)
            if c['slot_id'] == 'MC6' and 'distractor_relation' in checks:
                checks['six_gold_control'] = copy.deepcopy(checks['distractor_relation'])
            row['reconciliation_errors'] = errors
            if errors and row.get('decision') != 'FAIL':
                row['decision'] = 'HOLD'
            history[cid].append(row)
        inputs.extend(binding(p) for p in files.values())
    ledger_path = spec.get('revision_ledger')
    ledger = read(path(ledger_path)) if ledger_path else []
    records = ledger.get('records', []) if isinstance(ledger, dict) else ledger
    need(len(records) == len(changed) and {r['candidate_id'] for r in records} == changed,
         'Every actual changed slot requires exactly one revision ledger record.')
    for r in records:
        cid = r['candidate_id']
        need(r['revision_count'] == 1 and r['initial_candidate_sha256'] == fingerprint(initial[cid])
             and r['final_candidate_sha256'] == fingerprint(final[cid]), 'Revision count/hash drift: ' + cid)
        trigger = r['trigger_review_binding']
        need(keys.get(trigger['path']) == trigger, 'Revision trigger is not a retained frozen blind key: ' + cid)
        need(any(h['candidate_sha256'] == fingerprint(initial[cid]) and
                 h['independent_key_binding'] == trigger and
                 (h['original_independent_decision'] != 'PASS' or h.get('issues'))
                 for h in history[cid]), 'Revision has no initial independent review issue: ' + cid)
        for keyname in ('candidate_id', 'root_id', 'arm', 'slot_id', 'question_type', 'task_operation', 'author_id', 'source_binding'):
            need(initial[cid][keyname] == final[cid][keyname], 'Revision changed immutable slot identity: ' + cid)
    if ledger_path:
        inputs.append(binding(path(ledger_path)))
    reviews, mechanics, validations = [], [], []
    for cid, c in sorted(final.items()):
        eligible = [h for h in history[cid] if h['candidate_sha256'] == fingerprint(c)]
        need(eligible, 'No frozen independent key matches final candidate: ' + cid)
        row = copy.deepcopy(eligible[-1])
        try:
            audit = gate.audit_candidate(c, HERE / 'SOURCE_SELECTION_FINAL.json')
        except Exception as exc:
            audit = {'candidate_id': cid, 'candidate_sha256': fingerprint(c),
                     'mechanical_pass': False, 'mechanical_errors': [str(exc)]}
        if not audit['mechanical_pass'] and row['decision'] != 'FAIL':
            row['decision'] = 'HOLD'
            row['mechanical_hold'] = audit['mechanical_errors']
        if audit['mechanical_pass']:
            validation = gate.validate_review(c, row, HERE / 'SOURCE_SELECTION_FINAL.json')
            if validation['review_errors'] and row['decision'] != 'FAIL':
                row['decision'] = 'HOLD'
        else:
            validation = {'candidate_id': cid, 'admitted': False,
                          'review_errors': [], 'mechanical_pass': False}
        row['revision_count'] = int(cid in changed)
        reviews.append(row)
        mechanics.append(audit)
        validations.append(validation)
    by_id = {r['candidate_id']: r for r in reviews}
    matched = [(root, slot) for root in runner.ROOTS for slot in runner.SLOTS
               if all(by_id[f'CL1-{a}-{root.split("-")[-1]}-{slot}']['decision'] == 'PASS'
                      for a in runner.ARMS)]
    counts = dict(collections.Counter(r['decision'] for r in reviews))
    audit = {'records': validations, 'decisions': counts, 'planned_questions': 40,
             'individual_pass': counts.get('PASS', 0), 'matched_fg_slots': len(matched),
             'admitted_questions': 2 * len(matched), 'planned_solver_requests': 8 * len(matched),
             'revised_candidates': sorted(changed), 'no_replacement_sampling': True,
             'remaining_content_revision_budget': 0, 'new_solver_calls_so_far': 0}
    for name in ('PROTOCOL_FREEZE.json', 'SOURCE_SELECTION_FINAL.json', 'AUTHORING_INPUT_FREEZE.json',
                 'ERROR_MATCHING_RULES_ZH.md', 'RUNNER_CPU_VALIDATION.json', 'runner.py', 'assemble_freeze.py'):
        inputs.append(binding(HERE / name))
    inputs.extend(binding(path(p)) for p in spec.get('extra_bindings', []))
    unique = {}
    for b in inputs:
        need(binding(b['path']) == b, 'Input changed during assembly: ' + b['path'])
        need(b['path'] not in unique or unique[b['path']] == b, 'Contradictory retained hash.')
        unique[b['path']] = b
    save('CANDIDATES_PRIVATE.json', [final[cid] for cid in sorted(final)])
    save('CONTENT_REVIEW_AI.json', {'reviews': reviews,
         'basis': 'Independent keys frozen before author Gold comparison; all drafts retained; one revision maximum.',
         'blind_review_history': dict(history), 'input_spec_binding': binding(spec_path)})
    save('MECHANICAL_FINAL.json', {'records': mechanics,
         'passed': sum(a['mechanical_pass'] for a in mechanics), 'total': 40, 'semantic_proof': False})
    save('CONTENT_REVIEW_FINAL_AUDIT.json', audit)
    inputs = list(unique.values()) + [binding(HERE / n) for n in names if n != 'CONTENT_FREEZE.json']
    save('CONTENT_FREEZE.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
         'bindings': inputs, 'frozen_before_solver': True, 'decisions': counts,
         'planned_questions': 40, 'matched_fg_slots': len(matched),
         'admitted_questions': 2 * len(matched), 'no_replacement_sampling': True,
         'remaining_content_revision_budget': 0, 'fresh_plans_allowed': True})
    print(json.dumps(audit | {'records': 'See CONTENT_REVIEW_FINAL_AUDIT.json'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
