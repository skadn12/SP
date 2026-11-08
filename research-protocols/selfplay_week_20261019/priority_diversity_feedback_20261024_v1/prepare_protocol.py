"""Freeze a bounded Train-only diversity / feedback-consumption batch."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
PACKAGE = WEEK.parents[1] / 'offline-deliveries/h4_codex_train_pool_v125'


def binding(path):
    p = Path(path).resolve(strict=True)
    b = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    inventory_path = WEEK / 'postweek_production_batch_v2/SOURCE_SELECTION_INVENTORY.json'
    inventory = json.loads(inventory_path.read_text())
    for key in ['source_manifest_binding', 'authoritative_split_binding', 'root_boundary_binding']:
        assert binding(inventory[key]['path']) == inventory[key], key
    manifest_path = Path(inventory['source_manifest_binding']['path'])
    manifest = {r['root_id']: r for r in json.loads(manifest_path.read_text())}
    split_path = Path(inventory['authoritative_split_binding']['path'])
    splits = {r['root_id']: r for r in map(json.loads, split_path.read_text().splitlines())}
    excluded = set(inventory['excluded_root_ids']) | set(inventory['recommended_root_ids'])
    chosen = sorted(r for r in manifest if r not in excluded)[:4]
    assert chosen == ['cc-train-episode-218519', 'cc-train-episode-222937',
                      'cc-train-episode-225575', 'cc-train-episode-229686']
    cards_path = WEEK / 'postweek_production_feedback_audit_v1/EXPLORATORY_WEAKNESS_CARDS.json'
    cards = json.loads(cards_path.read_text())['cards']
    card_ids = {c['dimension']: c['card_id'] for c in cards}
    mc_ops = ['MC_QUANTITATIVE_RELATION', 'MC_CONSTRAINT_CONJUNCTION',
              'MC_EVIDENCE_SELECTION', 'MC_CAUSAL_SUPPORT']
    mc_dims = ['QUANTITY_OR_TIME', 'ENTITY_BINDING', 'SOURCE_STATE', 'SOURCE_STATE']
    roots = []
    for index, root in enumerate(chosen):
        original = manifest[root]
        assert original['split'] == splits[root]['split'] == 'train'
        source_path = PACKAGE / original['packaged_source']
        b = binding(source_path)
        assert b['sha256'] == original['source_binding']['sha256']
        src = json.loads(source_path.read_text())
        assert src['root_id'] == root
        plan = []
        for k, gold in enumerate(range(2, 6)):
            op_index = (k + index) % 4
            plan.append({'slot': f'MC{gold}', 'question_type': 'multi_select',
                         'gold_count': gold, 'task_operation': mc_ops[op_index],
                         'feedback_card_id': card_ids[mc_dims[op_index]],
                         'feedback_dimension': mc_dims[op_index]})
        plan.append({'slot': 'MC6', 'question_type': 'multi_select', 'gold_count': 6,
                     'task_operation': 'MC_CONSTRAINT_CONJUNCTION',
                     'feedback_card_id': card_ids['ENTITY_BINDING'],
                     'feedback_dimension': 'ENTITY_BINDING', 'all_select_control': True})
        for slot, op in [('ORDER1', 'ORDER_EXPLICIT_TIME'),
                         ('ORDER2', 'ORDER_EXPLICIT_PROCEDURE'),
                         ('ORDER3', 'ORDER_DISCOURSE_RECONSTRUCTION')]:
            plan.append({'slot': slot, 'question_type': 'ordering', 'task_operation': op,
                         'feedback_card_id': card_ids['ORDER_RELATION'],
                         'feedback_dimension': 'ORDER_RELATION'})
        roots.append({'root_id': root, **b, 'split': 'train',
                      'historically_exposed_train_source': True,
                      'new_unseen_root_claim': False, 'turns': len(src['turns']),
                      'author_group': 'a' if index < 2 else 'b', 'slot_plan': plan})
    save('SELECTION_FREEZE.json', {'schema_version': 'priority_selection_v1',
         'created_at_utc': datetime.now(timezone.utc).isoformat(),
         'selection_rule': 'First four full root IDs lexicographically from verified v125 Train manifest, excluding first two production batches and nine quality roots; no solver-score selection or substitution.',
         'planned_questions': 32, 'roots': roots, 'excluded_root_ids': sorted(excluded),
         'bindings': [binding(inventory_path), binding(manifest_path), binding(split_path)]})
    save('FEEDBACK_INPUT_FREEZE.json', {'schema_version': 'priority_feedback_input_v1',
         'source_cards_binding': binding(cards_path), 'cards': cards,
         'scope': 'Exploratory audited observations from current B1; generation instructions, not confirmed internal causes.',
         'formal_weakness_admission': False, 'causal_feedback_gain_claim': False,
         'old_cards_remain_unchanged': True})
    save('PROTOCOL_FREEZE.json', {'schema_version': 'priority_diversity_feedback_protocol_v1',
         'created_at_utc': datetime.now(timezone.utc).isoformat(),
         'goals': ['Verify seven source-dependent question operations and gold cardinalities 2/3/4/5/6',
                   'Consume current audited weakness cards and trace predictions into new tasks, audited rollouts, and revised exploratory cards'],
         'selection_binding': binding(HERE / 'SELECTION_FREEZE.json'),
         'feedback_binding': binding(HERE / 'FEEDBACK_INPUT_FREEZE.json'),
         'planned_slots': 32, 'maximum_drafts_per_slot': 1, 'maximum_pre_solver_revision_rounds': 1,
         'material_shortfall_policy': 'Retain unfilled slot; no invented relation or operation/root substitution.',
         'review': 'Independent nonauthor AI reads full source and reconstructs every answer before Solver; shared workspace and Gold visible are disclosed.',
         'admission': 'Source truth, unique ordering, distinct options, no source-independent shortcut; never filter by Solver scores.',
         'whole_question_duplicate_policy': 'Exclude only whole-question equivalents; annotate shared facts without excluding them.',
         'display_policy': 'Two deterministic Gold-independent distinct permutations with bijective mapping; source order is never the public canonical label order.',
         'solver_model': '/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model',
         'required_rope_theta': 1000000, 'conditions': ['FULL', 'QUESTION_ONLY'],
         'maximum_solver_requests': 128, 'automatic_retries': 0,
         'temperature': 0.8, 'top_p': 0.9, 'max_output_tokens': 3072,
         'max_context_tokens': 16384, 'native_thinking': False, 'input_truncation': False,
         'output': 'Per-item source references, brief checkable reasons, answer_reason, and final <box> answer; retain raw even when invalid.',
         'semantic_followup': 'Audit every FULL response; separate final-answer errors, citation errors, reasoning errors and output inconsistency. Publish versioned exploratory feedback only after review.',
         'reporting': ['planned/submitted/admitted counts and initial/final decisions', 'coverage by operation and Gold',
                       'FULL vs QUESTION_ONLY with paired-display dependency disclosed', 'predicted-error realization by card and root',
                       'author consumption and independent task realization', 'next-generation handoff consuming revised cards'],
         'formal_weakness_status': 'No promotion in this batch: existing prospective cross-root controlled-probe admission remains required.',
         'no_new_training': True, 'no_dev_test_use': True, 'new_dialogue_turns_planned': 0,
         'limits': 'Not a randomized feedback/no-feedback comparison; cannot establish causal training benefit or generalization to unseen roots.'})
    print(json.dumps({'roots': chosen, 'planned_slots': 32, 'feedback_cards': len(cards)}))


if __name__ == '__main__':
    main()
