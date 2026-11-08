"""Bounded derivation of three reviewed v81 failures; no new writer calls."""
import argparse
from copy import deepcopy
from pathlib import Path
import re
import shutil

from common import HERE, aware, bind, need, now, read, require_bindings, sha, write
import ordering_review

PREVIOUS = HERE.parent / 'h4_bound_scene_ordering_mixed_batch_v81'
VERSION = 'v82_actor_reference_minfix_ordering_v1'


def config(home=HERE):
    value = read(Path(home) / 'CONFIG.json')
    need(value['version'] == VERSION and value['question_types'] == ['ordering']
         and value['max_candidates'] == 3 and value['max_solver_calls_per_model'] == 36
         and value['max_generation_http'] == 0 and value['training_allowed'] is False
         and value['transport_retries'] == value['semantic_retries'] == 0
         and value['source_mode'] == 'complete_frozen_sources', 'derivation_policy_changed')
    require_bindings(value['upstream'], [PREVIOUS / 'ordering/POOL.json',
                     PREVIOUS / 'ordering/PUBLIC_QUESTIONS.json', PREVIOUS / 'ordering/CONTENT_REVIEW.json',
                     PREVIOUS / 'evidence/RUN_COMPLETE.json'])
    return value


def projected_rows(home=HERE):
    home = Path(home).resolve(); value = config(home)
    plan = read(home / 'REPLACEMENT_PLAN.json')
    originals = read(PREVIOUS / 'ordering/POOL.json')['questions']
    old_review = {q['candidate_id']: q for q in read(PREVIOUS / 'ordering/CONTENT_REVIEW.json')['questions']}
    need(len(originals) == len(plan['questions']) == 3, 'exact_three_originals_required')
    planned = {q['parent_candidate_id']: q for q in plan['questions']}
    need(set(planned) == {q['candidate_id'] for q in originals}, 'derivation_coverage')
    slots = {s['root_id']: s for s in value['ordering_slots']}
    rows = []; changes = []
    for original in originals:
        p = planned[original['candidate_id']]; review = old_review[original['candidate_id']]
        need(review['status'] == 'EXCLUDED' and review['reasons']
             and all(r.endswith(':actor_ok_failed') for r in review['reasons']), 'only_actor_failures_may_be_derived')
        row = deepcopy(original)
        need(p['candidate_id'] == original['candidate_id'].replace('v81_bound_scene_mixed_01', 'v82_actor_reference_minfix_01'), 'new_identity_required')
        row['candidate_id'] = p['candidate_id']
        row['source_path'] = str(home / 'ordering' / slots[row['root_id']]['slot_id'] / 'SOURCE.json')
        need(set(p['edits']) == {r.split(':')[0] for r in review['reasons']}, 'exact_failed_item_coverage')
        for label, change in p['edits'].items():
            need(row['items'][label] == change['before'], 'original_item_changed')
            transformed = change['before']
            allowed = ({' him ': ' Liam ', ' he ': ' Liam ', ' her ': " Elena's "}
                       if row['root_id'] == 'cc-train-episode-194259' else
                       {' his ': " Ethan's ", ' he ': ' Ethan ', ' himself': ''})
            for edit in change['substitutions']:
                need(edit['old'] in allowed and edit['new'] == allowed[edit['old']], 'only_known_actor_reference_replacements_allowed')
                need(edit['old'] in transformed and edit['old'] != edit['new'], 'replacement_not_applicable')
                transformed = transformed.replace(edit['old'], edit['new'])
            need(transformed == change['after'] and transformed != change['before'], 'replacement_replay_failed')
            need(not re.search(r'\b(he|him|his|himself|she|her|hers|herself)\b', transformed, re.I), 'gendered_actor_reference_remains')
            row['items'][label] = transformed
            changes.append({'parent_candidate_id': original['candidate_id'], 'candidate_id': row['candidate_id'], 'label': label, **change})
        need(row['gold'] == original['gold'] == original['original_gold'] == review['computed_order'], 'original_gold_must_be_retained')
        rows.append(row)
    need(len(changes) == 12, 'expected_twelve_actor_item_edits')
    return rows, changes


def derive(home=HERE):
    home = Path(home).resolve(); value = config(home)
    need(not (home / 'ordering/GENERATION.COMPLETE.json').exists(), 'derive_no_replay')
    rows, changes = projected_rows(home)
    paths = []
    for slot in value['ordering_slots']:
        source = home / 'ordering' / slot['slot_id'] / 'SOURCE.json'
        source.parent.mkdir(parents=True, exist_ok=True)
        need(not source.exists(), 'source_already_exists')
        shutil.copyfile(slot['source_path'], source)
        need(sha(source) == slot['source_sha256'], 'source_copy_changed')
        paths.append(source)
    public = [{k:q[k] for k in (*ordering_review.IDENTITY_FIELDS, 'items')} for q in rows]
    write(home / 'ordering/PUBLIC_QUESTIONS.json', {'questions': public})
    write(home / 'ordering/POOL.json', {'questions': rows,
          'provenance': {'requested_model': 'Qwen/Qwen3.6-35B-A3B',
                         'model_role': 'original_v81_writer_only', 'derivation': VERSION},
          'training_allowed': False})
    write(home / 'DERIVATION.json', {'version': VERSION, 'created_at_utc': now(),
          'method': 'explicit_actor_reference_text_edits', 'parent_pool': bind(PREVIOUS / 'ordering/POOL.json'),
          'replacement_plan': bind(home / 'REPLACEMENT_PLAN.json'), 'changes': changes,
          'candidate_count': len(rows), 'item_edit_count': len(changes), 'generation_http_calls': 0,
          'gold_edits': [], 'evidence_edits': [], 'source_byte_edits': [], 'stem_edits': [],
          'excluded_wrong_gold_parent': 'cc-train-episode-194259-v81_bound_scene_mixed_01-ordering-01'})
    paths += [home / 'ordering/PUBLIC_QUESTIONS.json', home / 'ordering/POOL.json']
    write(home / 'ordering/GENERATION.COMPLETE.json', {'version': VERSION, 'status': 'GENERATED',
          'generation_method': 'derived_existing_drafts_no_model_generation', 'completed_at_utc': now(),
          'candidate_count': len(rows), 'http_attempts': 0, 'training_allowed': False,
          'files': [bind(p) for p in paths]})
    return {'derived_candidates': len(rows), 'changed_items': len(changes), 'generation_http_calls': 0}


def verify_derivation(home=HERE):
    home = Path(home).resolve(); rows, changes = projected_rows(home)
    need(read(home / 'ordering/POOL.json')['questions'] == rows, 'private_derivation_changed')
    need(read(home / 'DERIVATION.json')['changes'] == changes, 'edit_receipt_changed')
    expected = [{k:q[k] for k in (*ordering_review.IDENTITY_FIELDS, 'items')} for q in rows]
    need(read(home / 'ordering/PUBLIC_QUESTIONS.json')['questions'] == expected, 'public_derivation_changed')
    generation = read(home / 'ordering/GENERATION.COMPLETE.json')
    need(generation['candidate_count'] == 3 and generation['http_attempts'] == 0, 'derived_count_changed')
    require_bindings(generation['files'], [home / 'ordering/POOL.json', home / 'ordering/PUBLIC_QUESTIONS.json'])


def freeze(home=HERE):
    home = Path(home).resolve(); config(home)
    # Validate saved independent source judgments before replaying private rows.
    ordering_review.validate_sources(home); verify_derivation(home); ordering_review.verify_review(home)
    need(not list((home / 'solver').glob('*/STARTED.json')), 'freeze_before_solver_required')
    questions = read(home / 'ordering/EVALUATION_POOL.json')['questions']
    write(home / 'EVALUATION_POOL.json', {'questions': questions,
          'provenance': {'requested_model': 'Qwen/Qwen3.6-35B-A3B', 'model_role': 'original_v81_writer_only', 'derivation': VERSION},
          'training_allowed': False, 'formal_training_admission': False})
    paths = [*home.glob('*.py'), *home.glob('*.md')]
    paths += [home / p for p in ('CONFIG.json', 'REPLACEMENT_PLAN.json', 'DERIVATION.json',
        'CHARACTER_REGISTRY.json', 'EVALUATION_POOL.json', 'ordering/CONTENT_REVIEW.COMPLETE.json')]
    write(home / 'BATCH_REVIEW.COMPLETE.json', {'version': VERSION, 'status': 'FROZEN',
          'completed_at_utc': now(), 'before_any_solver': True, 'solver_scores_used': False,
          'training_allowed': False, 'candidate_count': len(questions), 'files': [bind(p) for p in sorted(set(paths))]})
    return {'frozen_candidates': len(questions)}


def verify_frozen(home=HERE):
    home = Path(home).resolve(); config(home)
    ordering_review.validate_sources(home)
    frozen = read(home / 'BATCH_REVIEW.COMPLETE.json')
    require_bindings(frozen['files'], [*home.glob('*.py'), *home.glob('*.md'),
        *[home / p for p in ('CONFIG.json', 'REPLACEMENT_PLAN.json', 'DERIVATION.json',
           'CHARACTER_REGISTRY.json', 'EVALUATION_POOL.json', 'ordering/CONTENT_REVIEW.COMPLETE.json')]])
    need(frozen['status'] == 'FROZEN' and frozen['before_any_solver'] is True
         and frozen['training_allowed'] is False and frozen['solver_scores_used'] is False
         and aware(frozen['completed_at_utc']) <= aware(now()), 'invalid_freeze')
    verify_derivation(home); ordering_review.verify_review(home)
    questions = read(home / 'ordering/EVALUATION_POOL.json')['questions']
    need(read(home / 'EVALUATION_POOL.json')['questions'] == questions and len(questions) == frozen['candidate_count'] <= 3, 'frozen_pool_changed')
    return frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['derive','finish-review','freeze','prepare-evaluation','live-trained','live-base','report','verify'])
    args = parser.parse_args()
    if args.action == 'derive': result = derive()
    elif args.action == 'finish-review': result = ordering_review.finish_review(HERE)
    elif args.action == 'freeze': result = freeze()
    elif args.action == 'verify': result = verify_frozen()
    else:
        import mixed_evaluation
        if args.action == 'prepare-evaluation': result = mixed_evaluation.prepare(HERE)
        elif args.action.startswith('live-'):
            audit = read(HERE / 'evidence/INPUT_AUDIT.json')
            need(audit['status'] == 'PASS', 'independent_input_audit_required')
            result = mixed_evaluation.live(HERE, args.action[5:])
        else: result = mixed_evaluation.report(HERE)
    print({k:v for k,v in result.items() if k not in ['files','questions','responses','by_type']})
