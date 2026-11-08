#!/usr/bin/env python3
"""Freeze a finite, score-free source-family screening run before authoring."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
WEEK = OUT.parent
PREP = WEEK / 'day3_common_evaluation_v1'


def bind(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def write(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError('Refusing to rewrite an executed input: ' + str(path))
    path.write_bytes(data)


def main():
    lock_path = WEEK / 'day3_production_v1/evaluation/EVAL_SOURCE_FAMILY_LOCK.json'
    lock = json.loads(lock_path.read_text())
    queue_path = WEEK / 'day3_eval_semantic_v1/COMMON_EVAL_WORK_QUEUE.json'
    queue = json.loads(queue_path.read_text())
    original = {x['slot_id']: x for x in lock['itemized_slot_family_ledger']}
    sources = {x['root_id']: x for x in lock['source_bindings']}
    dev = lock['root_scope']['dev_root_ids']
    groups = [('dev_first', dev[:8]), ('dev_rest', dev[8:]),
              ('test', lock['root_scope']['test_root_ids'])]
    index = PREP / 'author_release/EXCLUSION_INDEX.jsonl'
    manifest = PREP / 'author_release/EXCLUSION_MANIFEST.json'
    policy = {
        'schema_version': 'finite_common_eval_family_screening_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PROSPECTIVE_FINITE_SCREENING_FREEZE_BEFORE_THIS_RUN',
        'inheritance': bind(lock_path),
        'source_material_queue': bind(queue_path),
        'exclusion_index': bind(index), 'exclusion_manifest': bind(manifest),
        'scope': 'All60 material-supported slots only, preserving original186-slot identity and sources.',
        'inventory_scope': {
            'versioned_new_work_limit': True,
            'max_semantically_distinct_plans_per_slot': 3,
            'rule': 'Read complete source and semantic inventory. Within the original locked relation/scope, inventory up to3 coherent minimal source-fact plans, prioritized by source-order evidence refs, then normalized source-fact tuples. Coherent plans require genuine dependency for relation claims, and six distinct atomic events for ordering. Preserve all inspected plans and pruning reasons; never count wording, Gold-subset or label changes as new families.',
            'selection': 'Compute original canonical SHA256 key for every inventoried plan. Only after all five checks PASS select the lexicographically smallest eligible key within this explicitly finite inventory.',
            'scope_limit': 'This new finite work limit is disclosed prospectively. It is not a claim that all natural-language questions were enumerated or that no other source family exists.',
        },
        'early_exclusion': 'A concrete proven historical necessary-fact overlap may reject a plan before other checks. Preserve all five status fields; unperformed checks remain NOT_EVALUATED_AFTER_DECISIVE_EXCLUSION. A source-clause/turn or lexical overlap alone is not proof of necessary-fact equivalence.',
        'uncertainty': 'Relevant unresolved support/scope/semantic overlap stays HOLD. Unrelated old-format gaps are not a blanket global block. Retrieval alone is not semantic PASS.',
        'ordering_later_rule': 'A later-report plan must declare an explicit report index within its explicit source scope; prefer the second genuine semantic report. Never turn a question/intent into completed-event report.',
        'generation_gate': 'No public questions or final Gold until all five semantic checks PASS, including global cross-shard eval-fact overlap. No Solver or training in this screening run.',
        'global_selection': 'Review slots in original DEV then TEST order. Within-eval conflicts with earlier chosen fact families require explicit resolution before downstream generation.',
        'zero_eligible_rule': 'Report NO_GO_FOR_NEW_FAMILY_EVAL_UNDER_CURRENT_FREEZE, not source impossibility or completed evaluation. Do not swap roots/sources, add suffixes or recast historical questions as new.',
        'read_boundary': 'Fresh reviewers receive this policy, their source-only packet, bound public content/exposure contracts and only the two approved no-score exclusion files. No current statuses, B feedback, private Gold, scores, new Train candidate content or raw boundary files.',
        'shared_filesystem_limit': 'Workflow provenance only; technical access isolation is not claimed.',
        'new_solver_calls': 0, 'new_training_runs': 0,
    }
    old_policy = OUT / 'SCREENING_FREEZE.json'
    if old_policy.exists():
        policy['created_at_utc'] = json.loads(old_policy.read_text())['created_at_utc']
    write(old_policy, policy)
    packets = []
    for group, roots in groups:
        entries = []
        for root in roots:
            split = sources[root]['original_split']
            binding = sources[root]['source_binding']
            source_path = Path(binding['path'])
            assert bind(source_path)['sha256'] == binding['sha256']
            review_path = PREP / 'source_admission' / split / (root + '.json')
            review = json.loads(review_path.read_text())
            slots = [row for row in queue['priority_slots'] if row['root_id'] == root]
            if not slots:
                continue
            entries.append({
                'root_id': root, 'original_split': split,
                'historical_role': 'REGRESSION_ONLY',
                'source_binding': binding,
                'complete_source': json.loads(source_path.read_text()),
                'source_semantic_inventory': review['semantic_inventory'],
                'source_ambiguity_notes': review['cross_reference_and_revision_notes_zh'],
                'supported_slots': [dict(slot_id=r['slot_id'],
                                         question_type=r['question_type'],
                                         design_target=original[r['slot_id']]['design_target'],
                                         material_witnesses=r['material_witnesses'],
                                         material_rationale=r['rationale_zh']) for r in slots],
            })
        packet_path = OUT / 'inputs' / (group + '.json')
        write(packet_path, {'schema_version': 'score_free_source_family_inventory_packet_v1',
                            'group': group, 'policy': bind(old_policy), 'roots': entries,
                            'forbidden_inputs': ['Gold', 'model outcomes', 'B feedback', 'new Train question contents'],
                            'limits': 'Material support is not family eligibility; all five checks still required.'})
        packets.append(dict(group=group, roots=len(entries),
                            slots=sum(len(x['supported_slots']) for x in entries),
                            packet=bind(packet_path)))
    assert sum(x['slots'] for x in packets) == 60
    write(OUT / 'INPUT_MANIFEST.json', {'policy': bind(old_policy), 'groups': packets,
                                       'source_lock': bind(lock_path), 'exclusion_index': bind(index),
                                       'exclusion_manifest': bind(manifest),
                                       'source_prefix_edited': False, 'new_solver_calls': 0,
                                       'new_training_runs': 0})
    print(json.dumps({'status': 'INPUTS_FROZEN', 'groups': packets}))


if __name__ == '__main__':
    main()
