"""Reuse frozen v15 QA contracts for one additional, paced question batch."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
D = HERE.parent
QA = D / 'h4_feedback_training_pool_v15_qa'
sys.path.insert(0, str(QA))
import qa

VERSION = 'h4_r02_existing_source_yield_v16'
EXPANSION = D / 'h4_feedback_training_pool_v15_expansion/runs/end_to_end_5roots_01'
PARENTS = [QA / 'runs/end_to_end_5roots_01', QA / 'runs/end_to_end_5roots_01_429_retry']
EXISTING_POOL = D / 'h4_feedback_training_pool_v15/runs/end_to_end_5roots_01_final_training_pool'
SELECTED = [D / 'h4_r02_yield_pilot_v12/runs/contract_smoke_2roots_01_qa/SELECTED_QUESTIONS.json',
            D / 'h4_r02_question_contract_v14/runs/frozen_v12_2roots_01/SELECTED_QUESTIONS.json',
            *[p / 'SELECTED_QUESTIONS.json' for p in PARENTS]]
INTERVAL = 35
INSTRUCTION = '''
ADDITIONAL-QUESTION PASS ON UNCHANGED SOURCE:
The earlier batch has already asked questions on this exact conversation.
Use previous_candidates_dedup_only as a coverage inventory, never as evidence.
Find remaining, substantively different facts or relations supported by the
source. Do not recycle the same tested fact by paraphrasing, reversing truth,
swapping labels, or replacing only one or two statements in a previous question.
Prefer an untested aspect of a coherent event, including supported distinctions
of actor, object, condition, time or plan versus outcome when the source has them.
Do not invent these distinctions and do not force cross-event combinations.
For ordering, seek a different supported set of actions/utterances, not another
permutation of the same six items. The response schema and Gold rules stay the
same. Use abstentions if the source does not support a sufficiently distinct
question. There are no current Solver scores or answers in this inventory.
'''

read, write, require = qa.base.read, qa.base.write, qa.require


def historical_prior(root):
    rows, files = ORIGINAL_PRIOR(root)
    selected, paths = qa.dedup.selected_history(D, root, HERE, SELECTED)
    rows += selected
    files += paths
    for parent in PARENTS:
        for branch in qa.ROUTES:
            path = parent / branch / 'generation/MECHANICAL_POOL.json'
            if not path.exists():
                continue
            candidates = [q for q in read(path)['questions'] if q['root_id'] == root]
            rows += [{**row, 'hard_dedup': False,
                      'history_status': 'PREVIOUS_GENERATED_CONTENT_WARNING_ONLY',
                      'source_path': str(path), 'source_sha256': qa.base.digest(path.read_bytes())}
                     for row in qa.legacy.compact_questions(candidates)]
            files.append(path)
    return rows, files


ORIGINAL_PRIOR = qa.prior_candidates


def make_payload(generation, root, qtype, context):
    instructions = qa.legacy.COMMON.replace(
        'It includes P, R01, and recovered R02 A/B turns.',
        "It includes P, R01, and only this independent R02 branch's accepted turns.")
    return qa.legacy.payload_with_feedback(generation, root,
        instructions + INSTRUCTION + (qa.legacy.MULTI if qtype == 'multi_select' else qa.legacy.ORDERING),
        context, min(8192, 1024 + 2048 * len(context['candidate_ids'])))


def prepare(run):
    require(read(EXISTING_POOL / 'REPORT.json')['total_unique'] == 25, 'existing_pool_count_changed')
    require(read(EXPANSION / 'FINAL_AUDIT.json')['status'] == 'PASS', 'expansion_not_audited')
    for parent in PARENTS:
        require(read(parent / 'AUDIT.json')['status'] == 'PASS', 'parent_qa_not_audited')
    qa.prior_candidates = historical_prior
    try:
        qa.prepare(EXPANSION, run, slots=4)
    finally:
        qa.prior_candidates = ORIGINAL_PRIOR
    active = {}
    for branch in qa.ROUTES:
        manifest = read(run / branch / 'generation/MANIFEST.json')
        active[branch] = [r['root_id'] for r in manifest['source_rows'] if r['accepted_scenes']]
        for row in manifest['source_rows']:
            prior = read(run / branch / 'generation/prior' / (row['root_id'] + '.json'))
            expected = {q['candidate_id'] for p in SELECTED for q in read(p)['questions']
                        if q['root_id'] == row['root_id']}
            actual = {q['candidate_id'] for q in prior if q.get('hard_dedup')}
            require(expected <= actual, 'existing_selected_missing_from_prior')
    spec = {'version': VERSION, 'expansion_run': str(EXPANSION),
            'source_reuse_only': True, 'new_dialogue_turns': 0, 'new_dataset_events': 0,
            'existing_pool': str(EXISTING_POOL), 'existing_selected_exports': list(map(str, SELECTED)),
            'existing_selected_questions': 25, 'parent_question_runs': list(map(str, PARENTS)),
            'active_roots_by_branch': active, 'active_branches': sum(map(len, active.values())),
            'additional_slots_per_active_branch_type': 4, 'active_candidate_slots': 40,
            'active_http_max': 40, 'active_solver_full_max': 600,
            'minimum_seconds_after_call_completion': INTERVAL, 'automatic_transport_retries': 0,
            'generator_model': qa.MODEL, 'reviewer_model': qa.MODEL, 'solver_model': qa.TARGET,
            'policy_version': qa.POLICY, 'feedback_round': 'previous_training_round_only',
            'current_solver_outputs_in_prompts': False, 'novelty_instruction': INSTRUCTION,
            'semantic_novelty_certified': False, 'training_started': False,
            'prepared_at_utc': qa.now()}
    require(spec['active_branches'] == 5, 'source_inventory_not_the_expected_five_branches')
    write(run / 'SUPPLEMENT_SPEC.json', spec)
    extra = qa.base.bindings([Path(__file__), HERE / 'POLICY.md', run / 'SUPPLEMENT_SPEC.json',
        EXISTING_POOL / 'REPORT.json', EXISTING_POOL / 'MANIFEST.json',
        EXPANSION / 'FINAL_AUDIT.json', *[p / 'AUDIT.json' for p in PARENTS]])
    for path in [run / 'MANIFEST.json', *[run / b / 'generation/MANIFEST.json' for b in qa.ROUTES]]:
        manifest = read(path)
        manifest['supplement_version'] = VERSION
        manifest['supplement_spec_path'] = str(run / 'SUPPLEMENT_SPEC.json')
        manifest['files'] += extra
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    verify(run)
    return spec


def verify(run):
    spec = read(run / 'SUPPLEMENT_SPEC.json')
    require(spec['version'] == VERSION and spec['novelty_instruction'] == INSTRUCTION,
            'supplement_version_or_prompt_changed')
    require(spec['active_candidate_slots'] == 40 and spec['active_http_max'] == 40,
            'supplement_budget_changed')
    qa.base.verify_bindings(read(run / 'MANIFEST.json')['files'])
    for branch in qa.ROUTES:
        manifest = read(run / branch / 'generation/MANIFEST.json')
        qa.base.verify_bindings(manifest['files'])
        for row in manifest['source_rows']:
            original = EXPANSION / 'roots' / row['root_id'] / branch / 'cumulative_dialogue.json'
            require(qa.base.digest(original.read_bytes()) == row['source_sha256'], 'reused_source_changed')
    return spec


def generate(run):
    spec = verify(run)
    require(not (run / 'SUPPLEMENT_STARTED.json').exists(), 'supplement_started_no_automatic_replay')
    write(run / 'SUPPLEMENT_STARTED.json', {'started_at_utc': qa.now(), 'active_http_max': 40})
    original_caller = qa.legacy.Caller
    timings, last_end = [], None

    class PacedCaller(original_caller):
        def call(self, label, payload, proof):
            nonlocal last_end
            require(len(timings) < spec['active_http_max'], 'supplement_global_http_budget_exhausted')
            while last_end is not None and time.monotonic() - last_end < INTERVAL:
                time.sleep(min(INTERVAL, INTERVAL - (time.monotonic() - last_end)))
            start, began = time.monotonic(), qa.now()
            result = super().call(label, payload, proof)
            ended = time.monotonic()
            row = {'call_sequence': len(timings) + 1, 'started_at_utc': began,
                   'finished_at_utc': qa.now(), 'request_directory': str(result[1]),
                   'interval_since_previous_call_completion_seconds': None if last_end is None else start - last_end,
                   'elapsed_seconds': ended - start,
                   'http_attempted': (result[1] / 'HTTP_ATTEMPT.json').exists(),
                   'payload_sha256': qa.base.digest(payload)}
            write(result[1] / 'TIMING.json', row)
            timings.append(row)
            last_end = ended
            return result

    qa.legacy.Caller = PacedCaller
    qa.make_generation_payload = make_payload
    for branch in qa.ROUTES:
        qa.generate_route(run, branch)
    write(run / 'GENERATION_COMPLETE.json', {'version': VERSION, 'requests': len(timings),
          'timings': timings, 'finished_at_utc': qa.now(), 'new_dialogue_turns': 0})
    return {'generation_complete': True, 'requests': len(timings)}


def preflight(run):
    verify(run)
    result = [qa.solver_preflight(run, branch) for branch in qa.ROUTES]
    require(sum(r['request_count'] for r in result) <= 600, 'solver_budget_exceeded')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'generate', 'solver-preflight'))
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.is_relative_to(HERE / 'runs'), 'supplement_run_must_be_in_version_directory')
    result = {'prepare': prepare, 'verify': verify, 'generate': generate,
              'solver-preflight': preflight}[args.action](run)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
