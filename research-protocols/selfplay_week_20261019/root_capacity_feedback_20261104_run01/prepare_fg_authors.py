#!/usr/bin/env python3
"""Create isolated, answer-free author contexts for the frozen F/G comparison."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
TEN_ROOTS = PROJECT / 'research-protocols/selfplay_week_20261019/base_qa_v2_2_1_ten_root_20261103/roots'
SELECTION = json.loads((HERE / 'FG_ROOT_SELECTION.json').read_text())
HISTORY = json.loads((HERE / 'preparation/HISTORICAL_COVERAGE.json').read_text())
HISTORY_BY_ROOT = {x['root_id']: x['records'] for x in HISTORY['coverage']}

CURRENT_FILES = {
    'cc-train-episode-109164': [
        HERE / 'replication/cc-train-episode-109164/round_01/ALL_CANDIDATES_PRIVATE.json',
        HERE / 'replication/cc-train-episode-109164/round_02/ALL_CANDIDATES_PRIVATE.json',
    ],
    'cc-train-episode-156136': [
        HERE / 'replication/cc-train-episode-156136/round_01/ALL_CANDIDATES_PRIVATE.json',
        HERE / 'replication/cc-train-episode-156136/round_02/canonical_hash_review/ALL_CANDIDATES_PRIVATE.json',
    ],
    'cc-train-episode-179186': [
        HERE / f'capacity/179186/round_{i:02d}/ALL_CANDIDATES_PRIVATE.json'
        for i in range(1, 5)
    ],
}


def candidate_summary(row):
    refs = list(row.get('evidence_refs', []))
    for ev in row.get('evidence', []):
        ref = ev.get('turn_ref')
        if ref and ref not in refs:
            refs.append(ref)
    return {
        'candidate_id': row.get('candidate_id'),
        'question_type': row.get('question_type'),
        'question_summary': row.get('question'),
        'natural_information_need': row.get('natural_information_need'),
        'cognitive_task': row.get('cognitive_task') or row.get('task_operation'),
        'evidence_refs': refs,
    }


def current_summaries(root_id):
    rows = []
    for path in CURRENT_FILES.get(root_id, []):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get('candidates', [])
        rows.extend(candidate_summary(row) for row in data if row.get('root_id') == root_id)
    return rows


def old_summaries(root_id):
    if root_id in HISTORY_BY_ROOT:
        return [candidate_summary(row) for row in HISTORY_BY_ROOT[root_id]], 'deduplicated historical coverage index'
    old = json.loads((TEN_ROOTS / root_id / 'HISTORICAL_CANDIDATE.json').read_text())
    refs = []
    for value in old.get('item_analysis', {}).values():
        for ev in value.get('evidence', []):
            ref = ev.get('turn_ref')
            if ref and ref not in refs:
                refs.append(ref)
    return [{
        'candidate_id': old.get('candidate_id'),
        'question_type': old.get('question_type'),
        'question_summary': old.get('question'),
        'natural_information_need': None,
        'cognitive_task': old.get('task_operation'),
        'evidence_refs': refs,
    }], 'one ten-root candidate only; historical exposure is not exhaustively deduplicated'


written = []
for root in SELECTION['roots']:
    root_id = root['root_id']
    src_path = Path(root['source_path'])
    source = json.loads(src_path.read_text())
    old, old_coverage_note = old_summaries(root_id)
    recent = current_summaries(root_id)
    for arm in ('F', 'G'):
        context_dir = HERE / 'comparison' / root_id / arm / 'author_context'
        author_dir = HERE / 'comparison' / root_id / arm / 'author'
        context_dir.mkdir(parents=True, exist_ok=False)
        author_dir.mkdir(parents=True, exist_ok=False)
        input_data = {
            'phase': 'fg_feedback_control_comparison',
            'root_id': root_id,
            'arm': arm,
            'source_path': str(src_path),
            'source_sha256': root['sha256'],
            'source_turn_count': root['turn_count'],
            'source_status': source.get('source_status'),
            'warnings': source.get('warnings', []),
            'author_scope': root['author_scope'],
            'candidate_limit': 1,
            'allowed_question_types': ['multi_select', 'ordering', 'partial_order'],
            'type_quotas': False,
            'difficulty_hard_gate': False,
            'historical_coverage_note': old_coverage_note,
            'prior_task_coverage': old + recent,
            'information_policy': 'Task summaries and evidence refs only. No options, Gold, blind answers, quality outcomes, Solver outputs, or scores.',
            'constraints': [
                'First draft only; no revision after review.',
                'Use the complete frozen source and preserve origin/warnings; cite exact source turns.',
                'No forced candidate, question form, answer cardinality, or difficulty.',
                'Do not reuse or paraphrase prior tasks; explain genuinely different information need and cognitive operation.',
                'Do not call APIs or models, run Solver, train, or extend source material.',
                'This is research use only; do not assert training admission.',
            ],
        }
        if arm == 'F':
            input_data['feedback_card_path'] = str(HERE / 'FEEDBACK_CARD_01.json')
            input_data['feedback_policy'] = 'Use the card as a low-confidence search cue, not a quota or a requirement. Do not copy its observed items.'
        write_path = context_dir / 'AUTHOR_INPUT.json'
        with write_path.open('x') as f:
            json.dump(input_data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        written.append(str(write_path))

index = {
    'schema_version': 'fg_author_context_index_v1',
    'selection_path': str(HERE / 'FG_ROOT_SELECTION.json'),
    'feedback_card_path': str(HERE / 'FEEDBACK_CARD_01.json'),
    'contexts': written,
    'arm_isolation': 'Each root/arm has a separate author and context directory. G contexts do not contain a feedback card path or Solver output.',
}
with (HERE / 'comparison/AUTHOR_CONTEXT_INDEX.json').open('x') as f:
    json.dump(index, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'author_contexts': len(written), 'roots': len(SELECTION['roots']), 'arms': ['F', 'G']}))
