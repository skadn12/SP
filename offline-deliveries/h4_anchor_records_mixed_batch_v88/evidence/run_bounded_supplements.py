"""Execute the frozen engine's single reviewed supplement, never repair drafts."""
from pathlib import Path
import json
import sys

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from common import bind, load_multi, now, write
import batch


def main():
    batch.verify_plan(HOME)
    assets = load_multi('asset_generation')
    decisions = []
    runs = []
    for slot in batch.config(HOME)['multiselect_config']['slots']:
        run = HOME / 'multiselect/assets' / slot['slot_id']
        reviewed = assets.reviewed_round(run, 'initial')
        eligible = reviewed['parents_valid'] and len(reviewed['approved_fact_ids']) < 4
        reason = ('Immutable parent failed source review; no supplement permitted.'
                  if not reviewed['parents_valid'] else
                  'At least four distinct false facts already approved; no supplement permitted.'
                  if len(reviewed['approved_fact_ids']) >= 4 else
                  'Valid parent and fewer than four approved false facts; use the sole bounded supplement.')
        row = {'slot_id': slot['slot_id'], 'root_id': slot['root_id'],
               'parents_valid': reviewed['parents_valid'],
               'initial_approved_fact_ids': reviewed['approved_fact_ids'],
               'supplement_permitted': eligible, 'reason': reason,
               'initial_source_judgments': bind(run / 'rounds/initial/SOURCE_JUDGMENTS.json'),
               'initial_private_crosscheck_receipt': bind(run / 'rounds/initial/REVIEW_RECEIPT.json')}
        if eligible:
            targets, feedback, paths = assets.supplement_spec(run)
            row['target_fact_ids'] = targets
            row['content_feedback'] = feedback
            row['feedback_bindings'] = [bind(p) for p in paths]
            runs.append(run)
        decisions.append(row)
    write(HOME / 'evidence/BOUNDED_SUPPLEMENT_DECISIONS.json', {
        'version': 'v88_frozen_rule_bounded_supplement_decisions_v1',
        'created_at_utc': now(), 'current_solver_scores_read': False,
        'solver_scores_used': False, 'production_rules_changed': False,
        'mechanical_repairs': 0, 'max_supplements_per_family': 1,
        'decisions': decisions})
    outputs = []
    for run in runs:
        prepared = assets.prepare_supplement(run)
        print(json.dumps({'event': 'supplement_prepared', **prepared}), flush=True)
        result = assets.generate_round(run, 'supplement')
        print(json.dumps({'event': 'supplement_generated', **result}), flush=True)
        outputs.append(result)
    write(HOME / 'evidence/BOUNDED_SUPPLEMENT_EXECUTION.json', {
        'completed_at_utc': now(), 'decisions': bind(HOME / 'evidence/BOUNDED_SUPPLEMENT_DECISIONS.json'),
        'results': outputs, 'automatic_followup': False, 'training_allowed': False})


if __name__ == '__main__':
    main()
